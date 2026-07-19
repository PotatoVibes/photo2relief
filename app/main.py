"""FastAPI app: static frontend + /api routes."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app import depth
from app.config import resolve_default_model
from app.schemas import (
    ErrorResponse,
    HealthResponse,
    ImageInfo,
    SessionCreateResponse,
    StatusResponse,
)
from app.sessions import (
    InvalidImageError,
    SessionNotFoundError,
    create_session,
    read_meta,
)

logger = logging.getLogger("photo2relief")


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
    model_id = resolve_default_model(depth.select_device())
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
        return JSONResponse(
            status_code=404,
            content=ErrorResponse(
                error="not_found", detail=f"No session {session_id}"
            ).model_dump(),
        )
    return StatusResponse(
        status=meta.get("status", "created"),
        model_id=meta.get("model_id"),
        device=meta.get("device"),
        elapsed_s=meta.get("elapsed_s"),
        error=meta.get("error"),
    )


if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
