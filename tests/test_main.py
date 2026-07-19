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


def test_session_dir_rejects_non_canonical_ids() -> None:
    """A hostile/malformed id must never be joined into a filesystem path.

    session_dir is the single choke point for turning an id into a path, so validating
    there protects every current and future endpoint. A bad id raises SessionNotFoundError
    (endpoints turn that into a 404) rather than returning a path that could escape.
    """
    from app.sessions import SessionNotFoundError, session_dir

    hostile = [
        "..",
        "../../etc/passwd",
        "..\\..\\windows",
        "abc",  # too short
        "g" * 32,  # right length, not hex
        "A" * 32,  # uppercase (uuid4().hex is lowercase)
        "",
    ]
    for bad in hostile:
        with pytest.raises(SessionNotFoundError):
            session_dir(bad)


def test_endpoints_404_on_traversal_style_id() -> None:
    # A non-hex id reaching a handler is rejected as "no such session", not a 500 or a
    # file read outside the sessions dir.
    for path in ("status", "params", "source"):
        res = client.get(f"/api/sessions/{'z' * 32}/{path}")
        assert res.status_code == 404, path
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


# --- models ------------------------------------------------------------------------------


def test_models_endpoint_lists_registry() -> None:
    from app.config import MODEL_REGISTRY

    res = client.get("/api/models")
    assert res.status_code == 200
    body = res.json()
    assert {m["model_id"] for m in body["models"]} == set(MODEL_REGISTRY)
    assert body["default_model"] in MODEL_REGISTRY
    for m in body["models"]:
        assert m["role"] and m["license"]
    # Canonical defaults for the UI's Reset button, keyed to this device's model.
    assert body["default_params"]["depth_model"] == body["default_model"]
    assert body["default_params"]["model_width_mm"] == 150.0


def test_status_endpoint_carries_session_identity_for_resume(portrait_jpeg_bytes: bytes) -> None:
    session_id = _create_session(portrait_jpeg_bytes)
    body = client.get(f"/api/sessions/{session_id}/status").json()
    assert body["width"] == 400
    assert body["height"] == 600
    assert body["original_filename"] == "portrait.jpg"


def test_session_source_serves_the_normalized_image(portrait_jpeg_bytes: bytes) -> None:
    session_id = _create_session(portrait_jpeg_bytes)
    res = client.get(f"/api/sessions/{session_id}/source")
    assert res.status_code == 200
    assert res.headers["content-type"] == "image/png"
    img = Image.open(io.BytesIO(res.content))
    assert img.size == (400, 600)


def test_session_source_404_for_unknown_session() -> None:
    assert client.get("/api/sessions/deadbeef/source").status_code == 404


def test_status_eta_comes_from_recorded_inference_times(
    portrait_jpeg_bytes: bytes, tmp_path, monkeypatch
) -> None:
    """The depth ETA is the last *measured* duration for (model, device) -- no
    hardcoded guesses. Unmeasured combinations report eta_s = None."""
    from app.config import settings

    monkeypatch.setattr(settings, "cache_dir", tmp_path)
    session_id = _create_session(portrait_jpeg_bytes)
    model_id = client.get(f"/api/sessions/{session_id}/status").json()["model_id"]

    assert client.get(f"/api/sessions/{session_id}/status").json()["eta_s"] is None

    depth.record_inference_time(model_id, "cpu", 17.5)  # stub writes device="cpu"
    assert client.get(f"/api/sessions/{session_id}/status").json()["eta_s"] == 17.5


# --- preview/mesh ------------------------------------------------------------------------


def test_preview_mesh_404_for_unknown_session() -> None:
    res = client.post("/api/sessions/deadbeef/preview/mesh", json={})
    assert res.status_code == 404


def test_preview_mesh_409_when_depth_not_ready(portrait_jpeg_bytes: bytes) -> None:
    session_id = _create_session(portrait_jpeg_bytes)
    res = client.post(f"/api/sessions/{session_id}/preview/mesh", json={})
    assert res.status_code == 409
    assert res.json()["error"] == "depth_not_ready"


def test_preview_mesh_returns_glb_capped_at_256(portrait_jpeg_bytes: bytes) -> None:
    session_id = _create_session(portrait_jpeg_bytes)
    params = client.get(f"/api/sessions/{session_id}/params").json()
    _write_fake_depth(session_id, params["depth_model"], shape=(600, 400))

    res = client.post(f"/api/sessions/{session_id}/preview/mesh", json=params)
    assert res.status_code == 200
    assert res.headers["content-type"] == "model/gltf-binary"
    assert res.content.startswith(b"glTF")

    import trimesh

    scene = trimesh.load(io.BytesIO(res.content), file_type="glb")
    mesh = scene.to_geometry()
    # Grid long side hard-capped at 256 vertices: top grid 256x171 + perimeter ring + center.
    n_perim = 2 * (256 + 171) - 4
    assert len(mesh.vertices) == 256 * 171 + n_perim + 1
    # Physical size survives the glb round trip (mm).
    assert mesh.bounds[1][0] == pytest.approx(params["model_width_mm"], abs=0.01)


def test_preview_mesh_uses_body_params_not_saved_ones(portrait_jpeg_bytes: bytes) -> None:
    """The 3D preview must reflect the params in the request body (a mid-drag slider
    value), not whatever params.json last persisted."""
    session_id = _create_session(portrait_jpeg_bytes)
    params = client.get(f"/api/sessions/{session_id}/params").json()
    _write_fake_depth(session_id, params["depth_model"], shape=(96, 64))

    params["model_width_mm"] = 321.0
    res = client.post(f"/api/sessions/{session_id}/preview/mesh", json=params)

    import trimesh

    mesh = trimesh.load(io.BytesIO(res.content), file_type="glb").to_geometry()
    assert mesh.bounds[1][0] == pytest.approx(321.0, abs=0.01)
    # ...and the saved params were NOT clobbered by a preview call.
    assert client.get(f"/api/sessions/{session_id}/params").json()["model_width_mm"] == 150.0


# --- export / jobs -----------------------------------------------------------------------


def _poll_job(job_id: str, timeout_s: float = 10.0) -> dict:
    import time

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        body = client.get(f"/api/jobs/{job_id}").json()
        if body["status"] != "processing":
            return body
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish within {timeout_s}s")


def test_export_404_for_unknown_session() -> None:
    res = client.post("/api/sessions/deadbeef/export", json={})
    assert res.status_code == 404


def test_export_409_when_depth_not_ready(portrait_jpeg_bytes: bytes) -> None:
    session_id = _create_session(portrait_jpeg_bytes)
    res = client.post(f"/api/sessions/{session_id}/export", json={})
    assert res.status_code == 409
    assert res.json()["error"] == "depth_not_ready"


def test_export_full_flow_produces_downloadable_stl(portrait_jpeg_bytes: bytes) -> None:
    session_id = _create_session(portrait_jpeg_bytes)
    params = client.get(f"/api/sessions/{session_id}/params").json()
    _write_fake_depth(session_id, params["depth_model"], shape=(96, 64))
    params["resolution"] = 512

    res = client.post(f"/api/sessions/{session_id}/export", json=params)
    assert res.status_code == 200
    job_id = res.json()["job_id"]

    status = _poll_job(job_id)
    assert status["status"] == "ready"
    assert status["download_url"] == f"/api/jobs/{job_id}/download"
    # Stage-based progress reaches 1.0 on completion (SPEC §4 `progress?`).
    assert status["progress"] == 1.0
    assert status["stage"] == "done"
    # Summary fields for the UI's export summary line (SPEC §5.4).
    assert status["triangles"] > 0
    assert status["file_bytes"] > 0
    assert status["width_mm"] == params["model_width_mm"]
    assert status["height_mm"] == pytest.approx(params["model_width_mm"] * 96 / 64, abs=0.5)

    download = client.get(status["download_url"])
    assert download.status_code == 200
    assert download.headers["content-type"] == "application/sla"
    assert not download.content.startswith(b"solid")  # binary STL, not ASCII

    import trimesh

    mesh = trimesh.load(io.BytesIO(download.content), file_type="stl")
    assert mesh.is_watertight


def test_export_obj_format_downloads_as_text(portrait_jpeg_bytes: bytes) -> None:
    session_id = _create_session(portrait_jpeg_bytes)
    params = client.get(f"/api/sessions/{session_id}/params").json()
    _write_fake_depth(session_id, params["depth_model"], shape=(96, 64))
    params["resolution"] = 512
    params["output_format"] = "obj"

    res = client.post(f"/api/sessions/{session_id}/export", json=params)
    job_id = res.json()["job_id"]
    status = _poll_job(job_id)
    assert status["status"] == "ready"

    download = client.get(status["download_url"])
    assert download.headers["content-type"] == "model/obj"
    assert download.content.startswith(b"#")


def test_job_status_404_for_unknown_job() -> None:
    res = client.get("/api/jobs/deadbeef")
    assert res.status_code == 404


def test_job_download_404_for_unknown_job() -> None:
    res = client.get("/api/jobs/deadbeef/download")
    assert res.status_code == 404


def test_job_download_404_while_still_processing() -> None:
    from app import jobs as jobs_module

    job = jobs_module.ExportJob(job_id="still-processing", session_id="unused")
    jobs_module._jobs[job.job_id] = job  # bypass the executor: deterministic "processing" state

    res = client.get(f"/api/jobs/{job.job_id}/download")
    assert res.status_code == 404
