"""Real-inference smoke test (SPEC §7): one model, marked slow, deselected by default.

Run with:  uv run pytest -m slow
"""

from __future__ import annotations

import numpy as np
import pytest

from app import depth, sessions
from app.config import settings
from tests.conftest import make_solid_image_bytes


@pytest.mark.slow
def test_da2_small_real_inference(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(settings, "sessions_dir", tmp_path / "sessions")
    monkeypatch.setattr(settings, "cache_dir", tmp_path / "cache")
    settings.sessions_dir.mkdir(parents=True)
    settings.cache_dir.mkdir(parents=True)

    info = sessions.create_session(make_solid_image_bytes(size=(256, 384)), "smoke.png")
    d = depth.run_inference(info.session_id, "da2-small")

    assert d.dtype == np.float32
    assert d.shape == (384, 256)  # (H, W) at source resolution
    assert np.isfinite(d).all()
    assert 0.0 <= float(d.min()) and float(d.max()) <= 1.0
    # A cache hit on repeat: file exists and second call returns identical array.
    d2 = depth.run_inference(info.session_id, "da2-small")
    assert np.array_equal(d, d2)


@pytest.mark.slow
def test_da3mono_real_inference_subject_nearer(monkeypatch, tmp_path) -> None:
    """Real subprocess round-trip through the da3worker venv: the sphere subject must
    come out nearer (higher) than the background after convention normalization."""
    from tests.conftest import make_sphere_image_bytes, region_means

    monkeypatch.setattr(settings, "sessions_dir", tmp_path / "sessions")
    monkeypatch.setattr(settings, "cache_dir", tmp_path / "cache")
    settings.sessions_dir.mkdir(parents=True)
    settings.cache_dir.mkdir(parents=True)

    info = sessions.create_session(make_sphere_image_bytes(size=(192, 256)), "sphere.png")
    d = depth.run_inference(info.session_id, "da3mono-large")

    assert d.dtype == np.float32
    assert d.shape == (256, 192)
    assert np.isfinite(d).all()
    center, corners = region_means(d)
    assert center > corners + 0.2, f"subject not nearer: center={center} corners={corners}"
