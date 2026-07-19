"""In-memory export job tracking (SPEC Sec4 /api/jobs/*).

No persistence: an app restart clears jobs along with sessions, matching the
project's "no database" non-goal (SPEC Sec1.2) -- a job is just a background task plus
a status dict, same pattern as depth.py's inference jobs.
"""

from __future__ import annotations

import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from app import depth, heightmap, meshing
from app.schemas import ReliefParams
from app.sessions import SOURCE_IMAGE_NAME, session_dir


@dataclass
class ExportJob:
    job_id: str
    session_id: str
    status: str = "processing"  # "processing" | "ready" | "error"
    error: str | None = None
    file_path: Path | None = None
    filename: str | None = None
    elapsed_s: float | None = None


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


def _run_export(job: ExportJob, params: ReliefParams) -> None:
    started = time.monotonic()
    try:
        raw_depth = depth.run_inference(job.session_id, params.depth_model)
        with Image.open(session_dir(job.session_id) / SOURCE_IMAGE_NAME) as img:
            luma = heightmap.luma_at_shape(img.convert("RGB"), raw_depth.shape)

        h = heightmap.compute_heightmap(raw_depth, luma, params, target_long_side=params.resolution)
        mesh = meshing.build_export_mesh(h, params)
        data = meshing.export_bytes(mesh, params.output_format)

        model_height_mm = params.model_width_mm * h.shape[0] / h.shape[1]
        filename = (
            f"photo2relief_{job.session_id}_"
            f"{_fmt_dim_mm(params.model_width_mm)}x{_fmt_dim_mm(model_height_mm)}mm."
            f"{params.output_format}"
        )
        out_path = session_dir(job.session_id) / filename
        out_path.write_bytes(data)

        with _lock:
            job.status = "ready"
            job.file_path = out_path
            job.filename = filename
            job.elapsed_s = round(time.monotonic() - started, 3)
    except Exception as exc:  # noqa: BLE001 - surface any failure via job status
        with _lock:
            job.status = "error"
            job.error = str(exc)
            job.elapsed_s = round(time.monotonic() - started, 3)
