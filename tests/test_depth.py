"""Unit tests for depth convention normalization, registry, caching, and status.

These never run a real model (SPEC §7). The real-inference smoke test lives in
test_depth_smoke.py behind @pytest.mark.slow.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from app import depth
from app.config import MODEL_REGISTRY, resolve_default_model


def _fake_torch(*, cuda: bool, mps: bool) -> types.ModuleType:
    """A minimal stand-in torch exposing just what select_device() touches."""
    mod = types.ModuleType("torch")
    mod.cuda = types.SimpleNamespace(is_available=lambda: cuda)
    mod.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: mps))
    return mod


def _select_with(monkeypatch, *, override: str, cuda: bool, mps: bool) -> str:
    monkeypatch.setattr(depth.settings, "device_override", override)
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda=cuda, mps=mps))
    return depth.select_device()


def test_select_device_auto_precedence_cuda_over_mps_over_cpu(monkeypatch) -> None:
    # cuda wins when present, regardless of mps.
    assert _select_with(monkeypatch, override="auto", cuda=True, mps=True) == "cuda"
    # mps is preferred over cpu when there's no cuda.
    assert _select_with(monkeypatch, override="auto", cuda=False, mps=True) == "mps"
    # cpu is the floor.
    assert _select_with(monkeypatch, override="auto", cuda=False, mps=False) == "cpu"


def test_select_device_forced_mps_requires_availability(monkeypatch) -> None:
    assert _select_with(monkeypatch, override="mps", cuda=False, mps=True) == "mps"
    monkeypatch.setattr(depth.settings, "device_override", "mps")
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda=False, mps=False))
    with pytest.raises(depth.DepthInferenceError, match="mps"):
        depth.select_device()


def test_select_device_forced_cpu_and_cuda(monkeypatch) -> None:
    # Forced cpu never touches an accelerator.
    assert _select_with(monkeypatch, override="cpu", cuda=True, mps=True) == "cpu"
    # Forced cuda requires cuda.
    assert _select_with(monkeypatch, override="cuda", cuda=True, mps=False) == "cuda"
    monkeypatch.setattr(depth.settings, "device_override", "cuda")
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda=False, mps=True))
    with pytest.raises(depth.DepthInferenceError, match="cuda"):
        depth.select_device()


def test_select_device_no_torch_is_cpu(monkeypatch) -> None:
    monkeypatch.setattr(depth.settings, "device_override", "auto")
    # Simulate torch not being importable.
    monkeypatch.setitem(sys.modules, "torch", None)
    assert depth.select_device() == "cpu"


def test_mps_available_tolerates_old_torch(monkeypatch) -> None:
    # A torch build without torch.backends.mps must not raise.
    bare = types.ModuleType("torch")
    assert depth._mps_available(bare) is False


def test_registry_has_three_models() -> None:
    assert set(MODEL_REGISTRY) == {"da3mono-large", "da2-large", "da2-small"}
    assert MODEL_REGISTRY["da2-small"].license == "Apache-2.0"
    assert MODEL_REGISTRY["da2-small"].available is True
    assert MODEL_REGISTRY["da2-large"].available is True
    # M2.5: DA3 served via the isolated da3worker venv.
    assert MODEL_REGISTRY["da3mono-large"].available is True
    assert MODEL_REGISTRY["da3mono-large"].convention == "depth"


def test_resolve_default_model() -> None:
    assert resolve_default_model("cpu") == "da2-small"
    # M2.5: the SPEC default is back in effect on GPU.
    assert resolve_default_model("cuda") == "da3mono-large"


def test_normalize_disparity_no_flip() -> None:
    # Disparity: larger = closer already. Max input stays at 1.0 (closest).
    raw = np.array([[0.0, 5.0], [10.0, 2.0]], dtype=np.float32)
    out = depth.normalize_to_closest(raw, "disparity")
    assert out.dtype == np.float32
    assert out.min() == 0.0 and out.max() == 1.0
    assert out[1, 0] == 1.0  # the largest disparity is closest


def test_normalize_depth_flips() -> None:
    # Direct depth: larger = farther. After flip, the *smallest* raw depth is closest (1.0).
    raw = np.array([[1.0, 100.0], [50.0, 2.0]], dtype=np.float32)
    out = depth.normalize_to_closest(raw, "depth")
    assert out[0, 0] == 1.0  # nearest object (smallest depth) maps to 1.0
    assert out[0, 1] == 0.0  # farthest maps to 0.0


def test_normalize_constant_input() -> None:
    raw = np.full((4, 4), 7.0, dtype=np.float32)
    out = depth.normalize_to_closest(raw, "disparity")
    assert np.all(out == 0.0)


def test_normalize_rejects_unknown_convention() -> None:
    with pytest.raises(ValueError):
        depth.normalize_to_closest(np.zeros((2, 2), np.float32), "bogus")


def test_da3_missing_worker_venv_errors_clearly(monkeypatch, tmp_path) -> None:
    """Selecting DA3 without the worker venv must raise setup guidance, not hang."""
    monkeypatch.setattr(depth, "_da3_worker_python", lambda: tmp_path / "nope" / "python")
    backend = depth.DA3Backend(MODEL_REGISTRY["da3mono-large"])
    with pytest.raises(depth.BackendUnavailableError, match="uv sync"):
        backend.load("cuda")


def _fake_worker_run(depth_array: np.ndarray, returncode: int = 0):
    """Build a subprocess.run stand-in that writes depth_array to the worker's out path."""

    def fake_run(cmd, capture_output, text, timeout):  # noqa: ANN001, ARG001
        out_npy = cmd[3]
        if returncode == 0:
            np.save(out_npy, depth_array)

        class _Proc:
            pass

        p = _Proc()
        p.returncode = returncode
        p.stderr = "boom from worker" if returncode else ""
        p.stdout = ""
        return p

    return fake_run


def test_da3_backend_normalizes_worker_output(monkeypatch, tmp_path) -> None:
    """DA3Backend must flip the worker's raw direct depth (larger = farther) so the
    smallest raw value maps to 1.0 = closest."""
    from PIL import Image

    py = tmp_path / "python"
    py.touch()
    monkeypatch.setattr(depth, "_da3_worker_python", lambda: py)

    raw = np.array([[1.0, 3.0], [2.0, 5.0]], dtype=np.float32)  # direct depth
    monkeypatch.setattr(depth.subprocess, "run", _fake_worker_run(raw))

    backend = depth.DA3Backend(MODEL_REGISTRY["da3mono-large"])
    backend.load("cuda")
    out = backend.infer(Image.new("RGB", (2, 2), (0, 0, 0)))
    assert out[0, 0] == 1.0  # nearest (smallest raw depth)
    assert out[1, 1] == 0.0  # farthest (largest raw depth)


def test_da3_backend_surfaces_worker_failure(monkeypatch, tmp_path) -> None:
    from PIL import Image

    py = tmp_path / "python"
    py.touch()
    monkeypatch.setattr(depth, "_da3_worker_python", lambda: py)
    monkeypatch.setattr(depth.subprocess, "run", _fake_worker_run(np.zeros((2, 2)), 1))

    backend = depth.DA3Backend(MODEL_REGISTRY["da3mono-large"])
    backend.load("cpu")
    with pytest.raises(depth.DepthInferenceError, match="boom from worker"):
        backend.infer(Image.new("RGB", (2, 2), (0, 0, 0)))


def test_da3_backend_times_out(monkeypatch, tmp_path) -> None:
    import subprocess as sp

    from PIL import Image

    py = tmp_path / "python"
    py.touch()
    monkeypatch.setattr(depth, "_da3_worker_python", lambda: py)

    def hang(cmd, capture_output, text, timeout):  # noqa: ANN001, ARG001
        raise sp.TimeoutExpired(cmd, timeout)

    monkeypatch.setattr(depth.subprocess, "run", hang)
    backend = depth.DA3Backend(MODEL_REGISTRY["da3mono-large"])
    backend.load("cpu")
    with pytest.raises(depth.DepthInferenceError, match="timed out"):
        backend.infer(Image.new("RGB", (2, 2), (0, 0, 0)))


def test_unknown_model_id_errors() -> None:
    with pytest.raises(depth.DepthInferenceError):
        depth._ensure_backend("does-not-exist")


def test_downscale_caps_long_side() -> None:
    from PIL import Image

    from app.config import settings

    big = Image.new("RGB", (settings.max_infer_px * 3, settings.max_infer_px), (0, 0, 0))
    out = depth._downscale_for_inference(big)
    assert max(out.size) == settings.max_infer_px
    small = Image.new("RGB", (100, 50), (0, 0, 0))
    assert depth._downscale_for_inference(small).size == (100, 50)


def test_run_inference_reuses_cache(monkeypatch, tmp_path) -> None:
    """A cached session .npy means the model is never invoked on a repeat call."""
    from app import sessions
    from app.config import settings
    from tests.conftest import make_solid_image_bytes

    monkeypatch.setattr(settings, "sessions_dir", tmp_path / "sessions")
    monkeypatch.setattr(settings, "cache_dir", tmp_path / "cache")
    settings.sessions_dir.mkdir(parents=True)
    settings.cache_dir.mkdir(parents=True)

    info = sessions.create_session(make_solid_image_bytes(size=(32, 32)), "x.png")

    fake = np.linspace(0, 1, 32 * 32, dtype=np.float32).reshape(32, 32)
    calls = {"n": 0}

    def fake_ensure(model_id):  # noqa: ANN001
        calls["n"] += 1

        class _B:
            def infer(self, image):  # noqa: ANN001
                return fake

        return _B()

    monkeypatch.setattr(depth, "_ensure_backend", fake_ensure)

    d1 = depth.run_inference(info.session_id, "da2-small")
    d2 = depth.run_inference(info.session_id, "da2-small")
    assert calls["n"] == 1  # second call hit the cache, no backend load
    assert np.array_equal(d1, d2)
    assert depth.session_depth_path(info.session_id, "da2-small").exists()

    # M2 acceptance: RE-UPLOADING the same file (new session, same content hash) must
    # reuse the shared cache — the model is never invoked again.
    info2 = sessions.create_session(make_solid_image_bytes(size=(32, 32)), "again.png")
    assert info2.session_id != info.session_id
    d3 = depth.run_inference(info2.session_id, "da2-small")
    assert calls["n"] == 1  # still one — cross-session cache hit
    assert np.array_equal(d1, d3)
    assert depth.session_depth_path(info2.session_id, "da2-small").exists()


def test_failed_job_writes_error_status(monkeypatch, tmp_path) -> None:
    """A failing inference must land the session at status=error, never stuck at
    processing — even if device probing is broken (regression: the error handler
    used to call select_device(), which itself can raise)."""
    from app import sessions
    from app.config import settings
    from tests.conftest import make_solid_image_bytes

    monkeypatch.setattr(settings, "sessions_dir", tmp_path / "sessions")
    settings.sessions_dir.mkdir(parents=True)

    info = sessions.create_session(make_solid_image_bytes(size=(16, 16)), "x.png")

    def boom(session_id, model_id):  # noqa: ANN001
        raise depth.DepthInferenceError("model exploded")

    monkeypatch.setattr(depth, "run_inference", boom)

    def raising_select_device():
        raise depth.DepthInferenceError("device probe broken too")

    monkeypatch.setattr(depth, "select_device", raising_select_device)

    depth._run_job(info.session_id, "da2-small")

    meta = sessions.read_meta(info.session_id)
    assert meta["status"] == "error"
    assert "model exploded" in meta["error"]
