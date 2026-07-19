"""FastAPI app: static frontend + /api routes."""

from __future__ import annotations

import io
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image

from app import depth, heightmap
from app.config import resolve_default_model
from app.schemas import (
    ErrorResponse,
    HealthResponse,
    ImageInfo,
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


def _not_found(session_id: str) -> JSONResponse:
    return JSONResponse(
        status_code=404,
        content=ErrorResponse(error="not_found", detail=f"No session {session_id}").model_dump(),
    )


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


def _luma_at_shape(source_rgb: Image.Image, shape: tuple[int, int]) -> np.ndarray:
    """Grayscale luminance of the source image, resampled to (height, width) = shape."""
    luma_full = np.asarray(source_rgb.convert("L"), dtype=np.float32) / 255.0
    height, width = shape
    if luma_full.shape == (height, width):
        return luma_full
    upscaling = width * height > luma_full.shape[1] * luma_full.shape[0]
    interp = cv2.INTER_CUBIC if upscaling else cv2.INTER_AREA
    return cv2.resize(luma_full, (width, height), interpolation=interp).astype(np.float32)


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
        luma = _luma_at_shape(img.convert("RGB"), raw_depth.shape)

    h = heightmap.compute_heightmap(raw_depth, luma, params, target_long_side=PREVIEW_MAX_PX)
    render = (
        h if grayscale else heightmap.hillshade(h, params.model_width_mm, params.relief_height_mm)
    )
    return Response(content=_encode_png_l(render), media_type="image/png")


if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
