"""Monocular depth inference: backend adapters, lazy model singleton, and caching.

The expensive step (running the model) keys on ``(image_hash, model_id, MAX_INFER_PX)``.
Everything downstream (heightmap shaping) is cheap numpy and re-runs freely. Adapters
normalize every backend to one internal convention — **1.0 = closest to camera** — so
nothing downstream needs to know which model produced the depth.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
from PIL import Image

from app.config import MODEL_REGISTRY, ModelSpec, settings
from app.sessions import SOURCE_IMAGE_NAME, read_meta, session_dir, write_status


class BackendUnavailableError(RuntimeError):
    """Raised when a registry model's backend is not wired (e.g. DA3 before M2.5)."""


class DepthInferenceError(RuntimeError):
    """Raised when inference fails (bad session, model load failure, etc.)."""


# --- device selection -------------------------------------------------------------


def select_device() -> str:
    """Resolve the active device from P2R_DEVICE + torch CUDA availability."""
    try:
        import torch
    except ImportError:
        return "cpu"

    if settings.device_override == "cpu":
        return "cpu"
    cuda = torch.cuda.is_available()
    if settings.device_override == "cuda":
        if not cuda:
            raise DepthInferenceError("P2R_DEVICE=cuda but no CUDA device is available.")
        return "cuda"
    return "cuda" if cuda else "cpu"


# --- convention normalization (pure, unit-tested) ---------------------------------


def normalize_to_closest(raw: np.ndarray, convention: str) -> np.ndarray:
    """Map a backend's native output to the internal convention: [0,1], 1.0 = closest.

    ``convention``:
      * ``"disparity"`` — native larger = closer (DA2). No flip.
      * ``"depth"``     — native larger = farther (DA3MONO). Flip so larger = closer.
    """
    a = raw.astype(np.float32)
    if convention == "depth":
        a = -a
    elif convention != "disparity":
        raise ValueError(f"Unknown depth convention: {convention!r}")
    lo = float(np.min(a))
    hi = float(np.max(a))
    if hi - lo < 1e-8:
        return np.zeros_like(a, dtype=np.float32)
    return ((a - lo) / (hi - lo)).astype(np.float32)


# --- backend protocol + adapters --------------------------------------------------


@runtime_checkable
class DepthBackend(Protocol):
    spec: ModelSpec

    def load(self, device: str) -> None: ...

    def infer(self, image: Image.Image) -> np.ndarray:
        """Return float32 HxW in [0,1], 1.0 = closest to camera, at image resolution."""
        ...

    def unload(self) -> None: ...


class TransformersDA2Backend:
    """Depth-Anything-V2 via the HF ``transformers`` depth-estimation pipeline."""

    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec
        self._pipe = None
        self._device = "cpu"

    def load(self, device: str) -> None:
        import torch
        from transformers import pipeline

        self._device = device
        pipe_device = 0 if device == "cuda" else -1
        self._pipe = pipeline(
            task="depth-estimation",
            model=self.spec.weights,
            device=pipe_device,
            dtype=torch.float32,
        )

    def infer(self, image: Image.Image) -> np.ndarray:
        if self._pipe is None:
            raise DepthInferenceError(f"Backend {self.spec.model_id} not loaded.")
        import torch

        with torch.inference_mode():
            out = self._pipe(image)
        pred = out["predicted_depth"]  # torch tensor, model resolution
        if pred.ndim == 3:
            pred = pred.unsqueeze(1)  # (N,H,W) -> (N,1,H,W)
        elif pred.ndim == 2:
            pred = pred.unsqueeze(0).unsqueeze(0)
        resized = torch.nn.functional.interpolate(
            pred.float(),
            size=(image.height, image.width),
            mode="bicubic",
            align_corners=False,
        )
        native = resized.squeeze().detach().cpu().numpy().astype(np.float32)
        return normalize_to_closest(native, self.spec.convention)

    def unload(self) -> None:
        self._pipe = None


DA3_WORKER_DIR = Path(__file__).resolve().parent.parent / "da3worker"
DA3_WORKER_TIMEOUT_S = 600  # generous: first-ever run may download ~1.3 GB of weights


def _da3_worker_python() -> Path:
    """Path to the isolated worker venv's interpreter (Windows or POSIX layout)."""
    win = DA3_WORKER_DIR / ".venv" / "Scripts" / "python.exe"
    posix = DA3_WORKER_DIR / ".venv" / "bin" / "python"
    return win if os.name == "nt" else posix


class DA3Backend:
    """depth_anything_3 via subprocess isolation (M2.5).

    The ``depth_anything_3`` package pins ``numpy<2`` and pulls open3d/xformers/pycolmap,
    which conflict with this project's numpy-2 stack — so it lives in its own venv
    (``da3worker/``) and we shell out per inference. Raw depth comes back via a temp
    ``.npy``; convention normalization happens here in the main app. Cost is ~15 s per
    call on the dev box (venv import + model load dominate), paid once per (image,
    model) thanks to the depth cache. Nothing stays resident in this process, so
    unload() is a no-op and idle VRAM is zero.
    """

    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec
        self._device = "cpu"

    def load(self, device: str) -> None:
        py = _da3_worker_python()
        if not py.exists():
            extra = "cu128" if device == "cuda" else "cpu"
            raise BackendUnavailableError(
                f"Model '{self.spec.model_id}' needs the DA3 worker venv, which is missing. "
                f"Create it with: cd da3worker && uv sync --extra {extra}"
            )
        self._device = device

    def infer(self, image: Image.Image) -> np.ndarray:
        with tempfile.TemporaryDirectory(prefix="p2r_da3_") as tmp:
            in_png = Path(tmp) / "in.png"
            out_npy = Path(tmp) / "out.npy"
            image.save(in_png, format="PNG")
            cmd = [
                str(_da3_worker_python()),
                str(DA3_WORKER_DIR / "worker.py"),
                str(in_png),
                str(out_npy),
                "--device",
                self._device,
                "--model",
                self.spec.weights,
            ]
            try:
                proc = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=DA3_WORKER_TIMEOUT_S
                )
            except subprocess.TimeoutExpired as exc:
                raise DepthInferenceError(
                    f"DA3 worker timed out after {DA3_WORKER_TIMEOUT_S}s."
                ) from exc
            if proc.returncode != 0 or not out_npy.exists():
                detail = (proc.stderr or proc.stdout or "no output").strip()[-2000:]
                raise DepthInferenceError(f"DA3 worker failed: {detail}")
            raw = np.load(out_npy)
        return normalize_to_closest(raw, self.spec.convention)

    def unload(self) -> None:
        pass  # nothing resident — the worker process exits after each call


def _make_backend(spec: ModelSpec) -> DepthBackend:
    if spec.backend == "da2":
        return TransformersDA2Backend(spec)
    if spec.backend == "da3":
        return DA3Backend(spec)
    raise DepthInferenceError(f"Unknown backend '{spec.backend}' for model {spec.model_id}.")


# --- lazy singleton: one model resident at a time ---------------------------------

_lock = threading.RLock()
_active_backend: DepthBackend | None = None
_active_device = "cpu"
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="depth")


def active_model_id() -> str | None:
    return _active_backend.spec.model_id if _active_backend is not None else None


def _ensure_backend(model_id: str) -> DepthBackend:
    """Load ``model_id``, unloading any other model first (one resident at a time)."""
    global _active_backend, _active_device
    with _lock:
        spec = MODEL_REGISTRY.get(model_id)
        if spec is None:
            raise DepthInferenceError(f"Unknown model_id '{model_id}'.")
        if not spec.available:
            _make_backend(spec).load("cpu")  # raises BackendUnavailableError with guidance
            # Backstop: never fall through and load a model marked unavailable, even if
            # its adapter's load() doesn't raise.
            raise BackendUnavailableError(f"Model '{model_id}' is marked unavailable.")

        device = select_device()
        if _active_backend is not None and _active_backend.spec.model_id == model_id:
            return _active_backend

        if _active_backend is not None:
            _active_backend.unload()
            _active_backend = None
            _free_cuda()

        backend = _make_backend(spec)
        backend.load(device)
        _active_backend = backend
        _active_device = device
        return backend


def _free_cuda() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


# --- caching + inference ----------------------------------------------------------


def _cache_path(image_hash: str, model_id: str) -> Path:
    d = settings.cache_dir / image_hash
    d.mkdir(parents=True, exist_ok=True)
    return d / f"raw_depth_{model_id}_{settings.max_infer_px}.npy"


def session_depth_path(session_id: str, model_id: str) -> Path:
    return session_dir(session_id) / f"raw_depth_{model_id}.npy"


def _downscale_for_inference(image: Image.Image) -> Image.Image:
    long_side = max(image.width, image.height)
    if long_side <= settings.max_infer_px:
        return image
    scale = settings.max_infer_px / long_side
    new_size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(new_size, Image.LANCZOS)


def run_inference(session_id: str, model_id: str) -> np.ndarray:
    """Produce (or reuse cached) normalized depth for a session + model.

    Cache key is ``(image_hash, model_id, MAX_INFER_PX)`` via the content-addressed
    cache dir; the session dir also keeps ``raw_depth_{model_id}.npy`` (SPEC §5.1).
    Never re-runs the model on a cache hit.
    """
    meta = read_meta(session_id)
    image_hash = meta["image_hash"]
    sess_path = session_depth_path(session_id, model_id)
    cache_path = _cache_path(image_hash, model_id)

    if sess_path.exists():
        return np.load(sess_path)
    if cache_path.exists():
        shutil.copyfile(cache_path, sess_path)
        return np.load(sess_path)

    src = session_dir(session_id) / SOURCE_IMAGE_NAME
    if not src.exists():
        raise DepthInferenceError(f"Session {session_id} has no source image.")

    with Image.open(src) as img:
        img = img.convert("RGB")
        infer_img = _downscale_for_inference(img)
        with _lock:
            backend = _ensure_backend(model_id)
            depth = backend.infer(infer_img)

    depth = np.ascontiguousarray(depth.astype(np.float32))
    np.save(cache_path, depth)
    np.save(sess_path, depth)
    return depth


# --- background jobs + status -----------------------------------------------------


def start_inference_job(session_id: str, model_id: str) -> None:
    """Submit inference to the single-worker executor and track status in meta.json."""
    write_status(session_id, status="processing", model_id=model_id, device=select_device())
    _executor.submit(_run_job, session_id, model_id)


def _run_job(session_id: str, model_id: str) -> None:
    started = time.monotonic()
    try:
        run_inference(session_id, model_id)
        elapsed = round(time.monotonic() - started, 3)
        write_status(
            session_id,
            status="ready",
            model_id=model_id,
            device=_active_device,
            elapsed_s=elapsed,
            error=None,
        )
    except Exception as exc:  # noqa: BLE001 - surface any failure via status
        # Never call anything that can raise in this handler (select_device() raises on
        # a cuda-override misconfig) — a failure here would leave the session stuck at
        # "processing" forever. _active_device is always a plain string.
        elapsed = round(time.monotonic() - started, 3)
        write_status(
            session_id,
            status="error",
            model_id=model_id,
            device=_active_device,
            elapsed_s=elapsed,
            error=str(exc),
        )
