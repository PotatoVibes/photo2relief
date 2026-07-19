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
