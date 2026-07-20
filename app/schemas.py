"""Pydantic models for API payloads: ReliefParams (the tuning-parameter surface, SPEC §3)
plus session/health/status/error contracts.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.config import MODEL_REGISTRY, resolve_default_model


def _default_depth_model() -> str:
    return resolve_default_model("cuda")


class ReliefParams(BaseModel):
    """The full tuning-parameter surface (SPEC §3). Persisted per session as params.json,
    sent whole on every preview/export call. Defaults re-tuned in M6 against three real
    photos (object/pet/object): edge-preserving smoothing + a mild subject-forward gamma
    give clean, forgiving reliefs across varied scenes, not just portraits."""

    # Size
    model_width_mm: float = Field(150.0, ge=10, le=1000)
    relief_height_mm: float = Field(8.0, ge=0.5, le=100)
    base_thickness_mm: float = Field(3.0, ge=0.5, le=50)
    border_frame_mm: float = Field(0.0, ge=0, le=50)

    # Depth shaping
    depth_model: str = Field(default_factory=_default_depth_model)
    invert_depth: bool = False
    gamma: float = Field(1.2, ge=0.2, le=5.0)
    depth_floor: float = Field(0.0, ge=0.0, le=1.0)
    depth_ceiling: float = Field(1.0, ge=0.0, le=1.0)
    flatten_background: bool = False
    background_threshold: float = Field(0.15, ge=0.0, le=1.0)

    # Surface
    smoothing: float = Field(2.5, ge=0.0, le=10.0)
    edge_preserve: bool = True
    detail_blend: float = Field(0.12, ge=0.0, le=1.0)
    edge_taper_mm: float = Field(0.0, ge=0, le=50)

    # Export
    resolution: Literal[512, 1024, 2048] = 1024
    decimate_ratio: float = Field(0.0, ge=0.0, le=0.95)
    output_format: Literal["stl", "obj"] = "stl"

    def model_post_init(self, __context: object) -> None:
        if self.depth_model not in MODEL_REGISTRY:
            raise ValueError(f"Unknown depth_model '{self.depth_model}'.")
        if self.depth_ceiling <= self.depth_floor:
            raise ValueError("depth_ceiling must be greater than depth_floor.")

    @property
    def total_width_mm(self) -> float:
        """Physical X of the finished part: model_width_mm is the IMAGE CONTENT width;
        a border frame adds to it on both sides. Everything that maps the (possibly
        frame-padded) heightmap grid to millimeters must use this, so the frame comes
        out exactly border_frame_mm wide and the content exactly model_width_mm."""
        return self.model_width_mm + 2 * self.border_frame_mm


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
    # Last measured inference duration for this (model, device) -- the UI's depth ETA.
    # None until a first real inference has ever been timed.
    eta_s: float | None = None
    # Session identity, so a reloaded page can resume without re-uploading.
    width: int | None = None
    height: int | None = None
    original_filename: str | None = None


class ErrorResponse(BaseModel):
    error: str
    detail: str


class ExportJobResponse(BaseModel):
    job_id: str


class JobStatusResponse(BaseModel):
    status: str  # "processing" | "ready" | "error"
    error: str | None
    download_url: str | None
    # Coarse stage-based progress while processing (SPEC §4's `progress?`).
    progress: float = 0.0  # 0..1
    stage: str | None = None
    # Populated once ready -- feeds the UI's export summary line (SPEC §5.4).
    triangles: int | None = None
    file_bytes: int | None = None
    width_mm: float | None = None
    height_mm: float | None = None


class ModelInfo(BaseModel):
    model_id: str
    role: str
    license: str
    available: bool


class ModelsResponse(BaseModel):
    models: list[ModelInfo]
    default_model: str  # the id auto-selected for this device
    default_params: ReliefParams  # canonical defaults for this device (UI Reset button)
