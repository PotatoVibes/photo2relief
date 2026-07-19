"""FastAPI app: static frontend + /api routes."""

from __future__ import annotations

import io
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image

from app import depth, heightmap, jobs, meshing
from app.config import MODEL_REGISTRY, resolve_default_model
from app.schemas import (
    ErrorResponse,
    ExportJobResponse,
    HealthResponse,
    ImageInfo,
    JobStatusResponse,
    ModelInfo,
    ModelsResponse,
    ReliefParams,
    SessionCreateResponse,
    StatusResponse,
)
from app.sessions import (
    SOURCE_IMAGE_NAME,
    InvalidImageError,
    SessionNotFoundError,
    create_session,
    read_meta,
    read_params,
    session_dir,
    write_params,
)

logger = logging.getLogger("photo2relief")

PREVIEW_MAX_PX = 1024  # SPEC §4: preview PNG capped at ≤ 1024 px
PREVIEW_MESH_MAX_PX = 256  # SPEC §4: 3D preview grid hard-capped at 256 (long side)


def _error(status_code: int, error: str, detail: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=ErrorResponse(error=error, detail=detail).model_dump(),
    )


def _not_found(session_id: str) -> JSONResponse:
    return _error(404, "not_found", f"No session {session_id}")


@asynccontextmanager
async def _lifespan(app: FastAPI):
    # SPEC §5.1: log the chosen device and model loudly at startup.
    cuda_available, torch_version = _torch_info()
    device = depth.select_device()
    logger.warning(
        "photo2relief starting: device=%s cuda_available=%s torch=%s default_model=%s "
        "(models lazy-load on first inference)",
        device,
        cuda_available,
        torch_version,
        resolve_default_model(device),
    )
    yield


app = FastAPI(title="Photo2Relief", lifespan=_lifespan)

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


def _torch_info() -> tuple[bool, str | None]:
    try:
        import torch
    except ImportError:
        return False, None
    return torch.cuda.is_available(), torch.__version__


@app.get("/api/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    cuda_available, torch_version = _torch_info()
    return HealthResponse(
        status="ok",
        device=depth.select_device(),
        cuda_available=cuda_available,
        torch_version=torch_version,
        model_loaded=depth.active_model_id() is not None,
        active_model=depth.active_model_id(),
    )


@app.post(
    "/api/sessions",
    response_model=SessionCreateResponse,
    responses={400: {"model": ErrorResponse}},
)
async def create_session_endpoint(image: UploadFile):
    data = await image.read()
    try:
        info = create_session(data, image.filename or "upload")
    except InvalidImageError as exc:
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(error="invalid_image", detail=str(exc)).model_dump(),
        )

    # Run depth inference for the default model in the background; client polls /status.
    # Persist params.json with that same model_id up front -- ReliefParams()'s own
    # default assumes CUDA, which would silently disagree with a CPU session's actual
    # model and make every preview request 409 until the user explicitly picks one.
    model_id = resolve_default_model(depth.select_device())
    write_params(info.session_id, ReliefParams(depth_model=model_id))
    depth.start_inference_job(info.session_id, model_id)

    return SessionCreateResponse(
        session_id=info.session_id,
        image=ImageInfo(w=info.width, h=info.height),
        status="processing",
    )


@app.get(
    "/api/sessions/{session_id}/status",
    response_model=StatusResponse,
    responses={404: {"model": ErrorResponse}},
)
async def session_status(session_id: str):
    try:
        meta = read_meta(session_id)
    except SessionNotFoundError:
        return _not_found(session_id)
    return StatusResponse(
        status=meta.get("status", "created"),
        model_id=meta.get("model_id"),
        device=meta.get("device"),
        elapsed_s=meta.get("elapsed_s"),
        error=meta.get("error"),
        width=meta.get("width"),
        height=meta.get("height"),
        original_filename=meta.get("original_filename"),
    )


@app.get(
    "/api/sessions/{session_id}/params",
    response_model=ReliefParams,
    responses={404: {"model": ErrorResponse}},
)
async def get_params(session_id: str):
    try:
        return read_params(session_id)
    except SessionNotFoundError:
        return _not_found(session_id)


@app.put(
    "/api/sessions/{session_id}/params",
    response_model=ReliefParams,
    responses={404: {"model": ErrorResponse}},
)
async def put_params(session_id: str, params: ReliefParams):
    try:
        meta = read_meta(session_id)
    except SessionNotFoundError:
        return _not_found(session_id)

    write_params(session_id, params)
    # Changing the depth model requires (cached) re-inference; everything else in
    # ReliefParams is a cheap heightmap re-run and needs no backend work here.
    if params.depth_model != meta.get("model_id"):
        depth.start_inference_job(session_id, params.depth_model)
    return params


def _encode_png_l(render: np.ndarray) -> bytes:
    img = Image.fromarray((np.clip(render, 0.0, 1.0) * 255).astype(np.uint8), mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@app.get(
    "/api/sessions/{session_id}/preview/heightmap",
    responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
)
async def preview_heightmap(session_id: str, grayscale: bool = False):
    try:
        params = read_params(session_id)
    except SessionNotFoundError:
        return _not_found(session_id)

    depth_path = depth.session_depth_path(session_id, params.depth_model)
    if not depth_path.exists():
        return JSONResponse(
            status_code=409,
            content=ErrorResponse(
                error="depth_not_ready",
                detail=(
                    f"Depth for model '{params.depth_model}' isn't ready yet "
                    "for this session; poll /status."
                ),
            ).model_dump(),
        )

    raw_depth = np.load(depth_path)
    with Image.open(session_dir(session_id) / SOURCE_IMAGE_NAME) as img:
        luma = heightmap.luma_at_shape(img.convert("RGB"), raw_depth.shape)

    h = heightmap.compute_heightmap(raw_depth, luma, params, target_long_side=PREVIEW_MAX_PX)
    render = (
        h if grayscale else heightmap.hillshade(h, params.total_width_mm, params.relief_height_mm)
    )
    return Response(content=_encode_png_l(render), media_type="image/png")


@app.get("/api/models", response_model=ModelsResponse)
async def list_models() -> ModelsResponse:
    """Registry contents for the UI's depth-model dropdown (SPEC §5.1.1)."""
    default_model = resolve_default_model(depth.select_device())
    return ModelsResponse(
        models=[
            ModelInfo(model_id=s.model_id, role=s.role, license=s.license, available=s.available)
            for s in MODEL_REGISTRY.values()
        ],
        default_model=default_model,
        default_params=ReliefParams(depth_model=default_model),
    )


@app.post(
    "/api/sessions/{session_id}/preview/mesh",
    responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
)
async def preview_mesh(session_id: str, params: ReliefParams):
    """Reduced-res relief as binary glTF for the three.js viewport (SPEC §4: grid long
    side ≤ 256, includes base, target < 2 s CPU). Params come in the body -- unlike the
    2D preview, this doesn't read params.json, so a slider drag can preview un-saved
    values. No decimation and no watertight gate here: the preview never ships to CAM,
    and skipping validation keeps the refresh snappy."""
    try:
        read_meta(session_id)
    except SessionNotFoundError:
        return _not_found(session_id)

    depth_path = depth.session_depth_path(session_id, params.depth_model)
    if not depth_path.exists():
        return _error(
            409,
            "depth_not_ready",
            f"Depth for model '{params.depth_model}' isn't ready yet for this session; "
            "poll /status.",
        )

    raw_depth = np.load(depth_path)
    with Image.open(session_dir(session_id) / SOURCE_IMAGE_NAME) as img:
        luma = heightmap.luma_at_shape(img.convert("RGB"), raw_depth.shape)

    h = heightmap.compute_heightmap(raw_depth, luma, params, target_long_side=PREVIEW_MESH_MAX_PX)
    mesh = meshing.build_relief_mesh(h, params)
    return Response(content=meshing.export_bytes(mesh, "glb"), media_type="model/gltf-binary")


@app.post(
    "/api/sessions/{session_id}/export",
    response_model=ExportJobResponse,
    responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
)
async def export_session(session_id: str, params: ReliefParams):
    try:
        read_meta(session_id)
    except SessionNotFoundError:
        return _not_found(session_id)

    depth_path = depth.session_depth_path(session_id, params.depth_model)
    if not depth_path.exists():
        return _error(
            409,
            "depth_not_ready",
            f"Depth for model '{params.depth_model}' isn't ready yet for this session; "
            "poll /status.",
        )

    write_params(session_id, params)
    job_id = jobs.start_export_job(session_id, params)
    return ExportJobResponse(job_id=job_id)


@app.get(
    "/api/jobs/{job_id}",
    response_model=JobStatusResponse,
    responses={404: {"model": ErrorResponse}},
)
async def job_status(job_id: str):
    job = jobs.get_job(job_id)
    if job is None:
        return _error(404, "not_found", f"No job {job_id}")
    return JobStatusResponse(
        status=job.status,
        error=job.error,
        download_url=f"/api/jobs/{job_id}/download" if job.status == "ready" else None,
        triangles=job.triangles,
        file_bytes=job.file_bytes,
        width_mm=job.width_mm,
        height_mm=job.height_mm,
    )


@app.get(
    "/api/jobs/{job_id}/download",
    responses={404: {"model": ErrorResponse}},
)
async def job_download(job_id: str):
    job = jobs.get_job(job_id)
    if job is None or job.status != "ready" or job.file_path is None:
        return _error(404, "not_found", f"No completed job {job_id}")
    media_type = "application/sla" if job.filename.endswith(".stl") else "model/obj"
    return FileResponse(path=job.file_path, media_type=media_type, filename=job.filename)


if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
