"""Unit tests for depth convention normalization, registry, caching, and status.

These never run a real model (SPEC §7). The real-inference smoke test lives in
test_depth_smoke.py behind @pytest.mark.slow.
"""

from __future__ import annotations

import numpy as np
import pytest

from app import depth
from app.config import MODEL_REGISTRY, resolve_default_model


def test_registry_has_three_models() -> None:
    assert set(MODEL_REGISTRY) == {"da3mono-large", "da2-large", "da2-small"}
    assert MODEL_REGISTRY["da2-small"].license == "Apache-2.0"
    assert MODEL_REGISTRY["da2-small"].available is True
    assert MODEL_REGISTRY["da2-large"].available is True
    # DA3 deferred to M2.5.
    assert MODEL_REGISTRY["da3mono-large"].available is False


def test_resolve_default_model() -> None:
    assert resolve_default_model("cpu") == "da2-small"
    # da3mono-large unavailable in M2 → GPU default falls back to da2-large.
    assert resolve_default_model("cuda") == "da2-large"


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


def test_da3_backend_reports_unavailable() -> None:
    with pytest.raises(depth.BackendUnavailableError):
        depth._ensure_backend("da3mono-large")


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
