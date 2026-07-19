"""In-memory export job tracking (SPEC Sec4 /api/jobs/*).

No persistence: an app restart clears jobs along with sessions, matching the
project's "no database" non-goal (SPEC Sec1.2) -- a job is just a background task plus
a status dict, same pattern as depth.py's inference jobs.
"""

from __future__ import annotations

import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from app import depth, heightmap, meshing
from app.schemas import ReliefParams
from app.sessions import SOURCE_IMAGE_NAME, read_meta, session_dir

_UNSAFE_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9._ -]")


@dataclass
class ExportJob:
    job_id: str
    session_id: str
    status: str = "processing"  # "processing" | "ready" | "error"
    error: str | None = None
    file_path: Path | None = None
    filename: str | None = None
    elapsed_s: float | None = None
    # Export summary (SPEC §5.4): triangle count, file size, physical dims.
    triangles: int | None = None
    file_bytes: int | None = None
    width_mm: float | None = None
    height_mm: float | None = None
    # Coarse stage-based progress (SPEC §4's `progress?`). A true ETA isn't honest --
    # the big steps are single opaque library calls -- but stage boundaries are.
    progress: float = 0.0  # 0..1
    stage: str | None = None


_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="export")
_lock = threading.RLock()
_jobs: dict[str, ExportJob] = {}


def get_job(job_id: str) -> ExportJob | None:
    with _lock:
        return _jobs.get(job_id)


def start_export_job(session_id: str, params: ReliefParams) -> str:
    job_id = uuid.uuid4().hex
    job = ExportJob(job_id=job_id, session_id=session_id)
    with _lock:
        _jobs[job_id] = job
    _executor.submit(_run_export, job, params)
    return job_id


def _fmt_dim_mm(mm: float) -> str:
    rounded = round(mm, 1)
    return str(int(rounded)) if rounded == int(rounded) else str(rounded)


def _export_stem(original_filename: str) -> str:
    """Base the export filename on the uploaded photo's name, not the session id."""
    stem = Path(original_filename).stem.strip()
    cleaned = _UNSAFE_FILENAME_CHARS.sub("_", stem)
    return cleaned or "photo2relief"


def _set_progress(job: ExportJob, progress: float, stage: str) -> None:
    with _lock:
        job.progress = progress
        job.stage = stage


def _run_export(job: ExportJob, params: ReliefParams) -> None:
    started = time.monotonic()
    try:
        # Weights are rough shares of wall time at resolution=1024 on the dev box;
        # watertight validation dominates, serialization of ~100 MB comes second.
        _set_progress(job, 0.05, "loading depth")
        raw_depth = depth.run_inference(job.session_id, params.depth_model)
        with Image.open(session_dir(job.session_id) / SOURCE_IMAGE_NAME) as img:
            luma = heightmap.luma_at_shape(img.convert("RGB"), raw_depth.shape)

        _set_progress(job, 0.15, "shaping heightmap")
        h = heightmap.compute_heightmap(raw_depth, luma, params, target_long_side=params.resolution)
        _set_progress(job, 0.3, "building mesh")
        mesh = meshing.build_export_mesh(
            h, params, on_stage=lambda frac, stage: _set_progress(job, frac, stage)
        )
        _set_progress(job, 0.8, f"writing {params.output_format.upper()}")
        data = meshing.export_bytes(mesh, params.output_format)

        # Filename encodes the stock size (matches the UI's "Stock" readout): full
        # width/height including the border frame, plus thickness (base + relief).
        model_height_mm = params.total_width_mm * h.shape[0] / h.shape[1]
        thickness_mm = params.base_thickness_mm + params.relief_height_mm
        stem = _export_stem(read_meta(job.session_id).get("original_filename") or "")
        filename = (
            f"{stem}_"
            f"{_fmt_dim_mm(params.total_width_mm)}x{_fmt_dim_mm(model_height_mm)}"
            f"x{_fmt_dim_mm(thickness_mm)}mm."
            f"{params.output_format}"
        )
        out_path = session_dir(job.session_id) / filename
        out_path.write_bytes(data)

        with _lock:
            # Endpoint readers don't take this lock, so publish status *last*: they must
            # never observe "ready" while file_path/filename are still None.
            job.file_path = out_path
            job.filename = filename
            job.triangles = len(mesh.faces)
            job.file_bytes = len(data)
            job.width_mm = params.total_width_mm
            job.height_mm = round(model_height_mm, 2)
            job.elapsed_s = round(time.monotonic() - started, 3)
            job.progress = 1.0
            job.stage = "done"
            job.status = "ready"
    except Exception as exc:  # noqa: BLE001 - surface any failure via job status
        with _lock:
            job.error = str(exc)
            job.elapsed_s = round(time.monotonic() - started, 3)
            job.status = "error"
