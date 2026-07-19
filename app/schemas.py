"""Pydantic models for API payloads. ReliefParams (the tuning-parameter surface) lands in M3."""

from __future__ import annotations

from pydantic import BaseModel


class ImageInfo(BaseModel):
    w: int
    h: int


class SessionCreateResponse(BaseModel):
    session_id: str
    image: ImageInfo
    status: str


class HealthResponse(BaseModel):
    status: str
    device: str
    cuda_available: bool
    torch_version: str | None
    model_loaded: bool
    active_model: str | None


class StatusResponse(BaseModel):
    status: str  # "created" | "processing" | "ready" | "error"
    model_id: str | None
    device: str | None
    elapsed_s: float | None
    error: str | None


class ErrorResponse(BaseModel):
    error: str
    detail: str
