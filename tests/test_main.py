from __future__ import annotations

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import depth
from app.main import app

client = TestClient(app)


def _create_session(image_bytes: bytes, filename: str = "portrait.jpg") -> str:
    res = client.post(
        "/api/sessions",
        files={"image": (filename, image_bytes, "image/jpeg")},
    )
    return res.json()["session_id"]


def _write_fake_depth(session_id: str, model_id: str, shape: tuple[int, int]) -> None:
    """Fast tests never touch a real depth model -- fabricate a plausible raw depth map
    directly at the session's cached path, exactly like a finished inference job would."""
    from app.sessions import session_dir

    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]].astype(np.float32)
    r = np.sqrt((xx - shape[1] / 2) ** 2 + (yy - shape[0] / 2) ** 2) / (max(shape) / 2)
    raw = np.clip(1.0 - r, 0.0, 1.0)
    np.save(session_dir(session_id) / f"raw_depth_{model_id}.npy", raw)


@pytest.fixture(autouse=True)
def _stub_inference(monkeypatch):
    """Fast API tests must not kick off real model inference.

    Mirrors the one real side effect callers depend on: the real start_inference_job
    writes meta.model_id synchronously (before backgrounding the actual work), which is
    how a later params PUT knows whether the depth_model actually changed.
    """
    from app.sessions import write_status

    calls = []

    def _fake_start(sid: str, mid: str) -> None:
        calls.append((sid, mid))
        write_status(sid, status="processing", model_id=mid, device="cpu")

    monkeypatch.setattr(depth, "start_inference_job", _fake_start)
    return calls


def test_health() -> None:
    res = client.get("/api/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["device"] in {"cpu", "cuda"}
    assert isinstance(body["cuda_available"], bool)


def test_index_page_loads() -> None:
    res = client.get("/")
    assert res.status_code == 200
    assert "Photo2Relief" in res.text


def test_create_session_from_jpeg(portrait_jpeg_bytes: bytes, _stub_inference) -> None:
    res = client.post(
        "/api/sessions",
        files={"image": ("portrait.jpg", portrait_jpeg_bytes, "image/jpeg")},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "processing"
    assert body["image"] == {"w": 400, "h": 600}
    assert body["session_id"]
    # Inference was scheduled for a valid registry model.
    assert len(_stub_inference) == 1
    from app.config import MODEL_REGISTRY

    assert _stub_inference[0][1] in MODEL_REGISTRY


def test_create_session_from_png(gradient_png_bytes: bytes) -> None:
    res = client.post(
        "/api/sessions",
        files={"image": ("gradient.png", gradient_png_bytes, "image/png")},
    )
    assert res.status_code == 200
    assert res.json()["image"] == {"w": 64, "h": 64}


def test_create_session_rejects_bad_data() -> None:
    res = client.post(
        "/api/sessions",
        files={"image": ("not_an_image.txt", b"hello world", "text/plain")},
    )
    assert res.status_code == 400
    assert res.json()["error"] == "invalid_image"


def test_create_session_rejects_oversized_megapixels() -> None:
    img = Image.new("RGB", (6000, 6000), color=(10, 10, 10))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    res = client.post(
        "/api/sessions",
        files={"image": ("huge.png", buf.getvalue(), "image/png")},
    )
    assert res.status_code == 400
    assert "MP" in res.json()["detail"]


def test_session_dir_contains_normalized_image(portrait_jpeg_bytes: bytes) -> None:
    from app.sessions import META_FILE_NAME, SOURCE_IMAGE_NAME, session_dir

    res = client.post(
        "/api/sessions",
        files={"image": ("portrait.jpg", portrait_jpeg_bytes, "image/jpeg")},
    )
    session_id = res.json()["session_id"]
    out_dir = session_dir(session_id)
    assert (out_dir / SOURCE_IMAGE_NAME).exists()
    assert (out_dir / META_FILE_NAME).exists()


def test_status_endpoint_reports_meta(gradient_png_bytes: bytes) -> None:
    res = client.post(
        "/api/sessions",
        files={"image": ("gradient.png", gradient_png_bytes, "image/png")},
    )
    session_id = res.json()["session_id"]
    status = client.get(f"/api/sessions/{session_id}/status")
    assert status.status_code == 200
    # Inference is stubbed, so status stays at its created default.
    assert status.json()["status"] in {"created", "processing", "ready"}


def test_status_endpoint_404_for_unknown_session() -> None:
    res = client.get("/api/sessions/deadbeef/status")
    assert res.status_code == 404
    assert res.json()["error"] == "not_found"


# --- params -------------------------------------------------------------------------


def test_get_params_returns_defaults_persisted_at_creation(portrait_jpeg_bytes: bytes) -> None:
    session_id = _create_session(portrait_jpeg_bytes)
    res = client.get(f"/api/sessions/{session_id}/params")
    assert res.status_code == 200
    body = res.json()
    assert body["model_width_mm"] == 150.0
    assert body["relief_height_mm"] == 8.0
    from app.config import MODEL_REGISTRY

    assert body["depth_model"] in MODEL_REGISTRY


def test_get_params_404_for_unknown_session() -> None:
    res = client.get("/api/sessions/deadbeef/params")
    assert res.status_code == 404
    assert res.json()["error"] == "not_found"


def test_put_params_round_trips(portrait_jpeg_bytes: bytes) -> None:
    session_id = _create_session(portrait_jpeg_bytes)
    res = client.put(
        f"/api/sessions/{session_id}/params",
        json={"model_width_mm": 200.0, "gamma": 2.0, "invert_depth": True},
    )
    assert res.status_code == 200
    assert res.json()["model_width_mm"] == 200.0
    assert res.json()["gamma"] == 2.0

    reread = client.get(f"/api/sessions/{session_id}/params")
    assert reread.json()["model_width_mm"] == 200.0
    assert reread.json()["invert_depth"] is True


def test_put_params_404_for_unknown_session() -> None:
    res = client.put("/api/sessions/deadbeef/params", json={"gamma": 1.0})
    assert res.status_code == 404


def test_put_params_rejects_out_of_range_values(portrait_jpeg_bytes: bytes) -> None:
    session_id = _create_session(portrait_jpeg_bytes)
    res = client.put(f"/api/sessions/{session_id}/params", json={"gamma": 99.0})
    assert res.status_code == 422


def test_put_params_rejects_bad_depth_window(portrait_jpeg_bytes: bytes) -> None:
    session_id = _create_session(portrait_jpeg_bytes)
    res = client.put(
        f"/api/sessions/{session_id}/params",
        json={"depth_floor": 0.8, "depth_ceiling": 0.2},
    )
    assert res.status_code == 422


def test_put_params_switching_model_triggers_new_inference_job(
    portrait_jpeg_bytes: bytes, _stub_inference
) -> None:
    session_id = _create_session(portrait_jpeg_bytes)
    _stub_inference.clear()  # drop the creation-time call, only care about the switch
    other_model = "da2-small"
    res = client.put(f"/api/sessions/{session_id}/params", json={"depth_model": other_model})
    assert res.status_code == 200
    assert (session_id, other_model) in _stub_inference


def test_put_params_same_model_does_not_retrigger_inference(
    portrait_jpeg_bytes: bytes, _stub_inference
) -> None:
    session_id = _create_session(portrait_jpeg_bytes)
    from app.config import MODEL_REGISTRY

    current_model = next(iter(MODEL_REGISTRY))
    client.put(f"/api/sessions/{session_id}/params", json={"depth_model": current_model})
    _stub_inference.clear()
    client.put(f"/api/sessions/{session_id}/params", json={"gamma": 1.5})
    assert _stub_inference == []


# --- preview/heightmap -----------------------------------------------------------------


def test_preview_heightmap_404_for_unknown_session() -> None:
    res = client.get("/api/sessions/deadbeef/preview/heightmap")
    assert res.status_code == 404


def test_preview_heightmap_409_when_depth_not_ready(portrait_jpeg_bytes: bytes) -> None:
    session_id = _create_session(portrait_jpeg_bytes)
    res = client.get(f"/api/sessions/{session_id}/preview/heightmap")
    assert res.status_code == 409
    assert res.json()["error"] == "depth_not_ready"


def test_preview_heightmap_returns_png_once_depth_is_ready(portrait_jpeg_bytes: bytes) -> None:
    session_id = _create_session(portrait_jpeg_bytes)
    params = client.get(f"/api/sessions/{session_id}/params").json()
    _write_fake_depth(session_id, params["depth_model"], shape=(96, 64))

    res = client.get(f"/api/sessions/{session_id}/preview/heightmap")
    assert res.status_code == 200
    assert res.headers["content-type"] == "image/png"
    img = Image.open(io.BytesIO(res.content))
    assert img.mode == "L"
    assert max(img.size) == 1024  # PREVIEW_MAX_PX, long side


def test_preview_heightmap_grayscale_differs_from_hillshade(portrait_jpeg_bytes: bytes) -> None:
    session_id = _create_session(portrait_jpeg_bytes)
    params = client.get(f"/api/sessions/{session_id}/params").json()
    _write_fake_depth(session_id, params["depth_model"], shape=(96, 64))

    shaded = client.get(f"/api/sessions/{session_id}/preview/heightmap").content
    gray = client.get(f"/api/sessions/{session_id}/preview/heightmap?grayscale=true").content
    assert shaded != gray


def test_preview_heightmap_completes_quickly(portrait_jpeg_bytes: bytes) -> None:
    import time

    session_id = _create_session(portrait_jpeg_bytes)
    params = client.get(f"/api/sessions/{session_id}/params").json()
    _write_fake_depth(session_id, params["depth_model"], shape=(300, 200))

    started = time.monotonic()
    res = client.get(f"/api/sessions/{session_id}/preview/heightmap")
    elapsed = time.monotonic() - started
    assert res.status_code == 200
    assert elapsed < 0.3  # SPEC M3 accept: previews return in < 300 ms
