from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import depth
from app.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def _stub_inference(monkeypatch):
    """Fast API tests must not kick off real model inference."""
    calls = []
    monkeypatch.setattr(depth, "start_inference_job", lambda sid, mid: calls.append((sid, mid)))
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
