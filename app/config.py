"""Env-driven settings. All knobs are read here, once, with the P2R_ prefix."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


class Settings:
    port: int = _env_int("P2R_PORT", 8090)
    host: str = os.environ.get("P2R_HOST", "0.0.0.0")

    data_dir: Path = Path(os.environ.get("P2R_DATA_DIR", "data"))
    sessions_dir: Path = Path(os.environ.get("P2R_SESSIONS_DIR", str(data_dir / "sessions")))
    models_dir: Path = Path(os.environ.get("P2R_MODELS_DIR", str(data_dir / "models")))
    cache_dir: Path = Path(os.environ.get("P2R_CACHE_DIR", str(data_dir / "cache")))

    device_override: str = os.environ.get("P2R_DEVICE", "auto")  # "auto" | "cpu" | "cuda"

    max_infer_px: int = _env_int("P2R_MAX_INFER_PX", 1024)
    max_upload_bytes: int = _env_int("P2R_MAX_UPLOAD_MB", 40) * 1024 * 1024
    max_upload_megapixels: float = float(os.environ.get("P2R_MAX_UPLOAD_MEGAPIXELS", "30"))

    default_model_id: str = os.environ.get("P2R_DEFAULT_MODEL_ID", "da3mono-large")

    def __init__(self) -> None:
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # HF_HOME should point at the mounted models volume so weights persist offline.
        os.environ.setdefault("HF_HOME", str(self.models_dir))


settings = Settings()


# --- Model registry (SPEC §5.1.1) -------------------------------------------------

# Internal depth convention after adapter normalization is always "1.0 = closest to
# camera". Each backend declares its NATIVE convention so the adapter knows how to flip:
#   "disparity" → native larger = closer (DA2 relative models); no flip needed.
#   "depth"     → native larger = farther (DA3MONO direct depth); flip before normalize.


@dataclass(frozen=True)
class ModelSpec:
    model_id: str  # registry key / ReliefParams.depth_model value
    weights: str  # Hugging Face repo id
    backend: str  # "da2" (transformers) | "da3" (depth_anything_3)
    convention: str  # native output convention: "disparity" | "depth"
    license: str  # as listed on the HF page at build time
    role: str  # short UI hint (geometric vs artistic vs fast/CPU)
    available: bool  # False = adapter not wired yet (selecting it errors clearly)


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "da3mono-large": ModelSpec(
        model_id="da3mono-large",
        weights="depth-anything/DA3MONO-LARGE",
        backend="da3",
        convention="depth",
        license="CC-BY-NC (verify HF page)",
        role="Geometric fidelity (direct depth). Best true-to-life proportions.",
        available=False,  # M2.5: depth_anything_3 pins numpy<2 + open3d/xformers — deferred
    ),
    "da2-large": ModelSpec(
        model_id="da2-large",
        weights="depth-anything/Depth-Anything-V2-Large-hf",
        backend="da2",
        convention="disparity",
        license="CC-BY-NC-4.0",
        role="Artistic bas-relief look (exaggerated near-field).",
        available=True,
    ),
    "da2-small": ModelSpec(
        model_id="da2-small",
        weights="depth-anything/Depth-Anything-V2-Small-hf",
        backend="da2",
        convention="disparity",
        license="Apache-2.0",
        role="Fast, CPU-safe fallback (only Apache-licensed option).",
        available=True,
    ),
}


def resolve_default_model(device: str) -> str:
    """The model auto-selected when the user hasn't chosen one.

    SPEC default is ``da3mono-large``, but the DA3 backend is deferred to M2.5, so on
    GPU we fall back to the best *available* model (``da2-large``); on CPU, ``da2-small``.
    """
    if device != "cuda":
        return "da2-small"
    spec = MODEL_REGISTRY.get(settings.default_model_id)
    if spec is not None and spec.available:
        return settings.default_model_id
    return "da2-large"
