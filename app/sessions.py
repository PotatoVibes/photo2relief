"""Session directory management: one directory per upload under P2R_SESSIONS_DIR.

No database — each session is `data/sessions/{session_id}/` holding the EXIF-corrected
source image plus (from M2 onward) cached depth maps and params.json.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import pillow_heif
from PIL import Image, ImageOps

from app.config import settings
from app.schemas import ReliefParams

pillow_heif.register_heif_opener()

ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP", "HEIF"}
SOURCE_IMAGE_NAME = "source.png"
META_FILE_NAME = "meta.json"
PARAMS_FILE_NAME = "params.json"

# Session ids are uuid4().hex — exactly 32 lowercase hex chars. Everything else is
# rejected before it can touch the filesystem, so a hand-crafted id (e.g. "..") can
# never be joined into a path that escapes the sessions dir. Defense-in-depth: the
# FastAPI path param can't contain "/" either, but validating here is cheap and makes
# the guarantee explicit rather than relying on framework routing behavior.
_SESSION_ID_RE = re.compile(r"\A[0-9a-f]{32}\Z")


_meta_lock = Lock()


class InvalidImageError(ValueError):
    """Raised when an upload fails type/size/decoding validation."""


class SessionNotFoundError(KeyError):
    """Raised when a session id has no directory on disk."""


@dataclass
class SessionInfo:
    session_id: str
    width: int
    height: int
    image_hash: str


def session_dir(session_id: str) -> Path:
    # Reject any id that isn't a canonical session id *before* it reaches a path join,
    # so a malformed/hostile id can never escape sessions_dir. A bad id maps to no real
    # session, so SessionNotFoundError is the right signal (endpoints already 404 it).
    if not _SESSION_ID_RE.match(session_id):
        raise SessionNotFoundError(session_id)
    return settings.sessions_dir / session_id


def meta_path(session_id: str) -> Path:
    return session_dir(session_id) / META_FILE_NAME


def read_meta(session_id: str) -> dict:
    path = meta_path(session_id)
    if not path.exists():
        raise SessionNotFoundError(session_id)
    return json.loads(path.read_text())


def write_status(session_id: str, **fields: object) -> dict:
    """Merge status fields (status, model_id, device, elapsed_s, error) into meta.json."""
    with _meta_lock:
        meta = read_meta(session_id)
        meta.update(fields)
        meta_path(session_id).write_text(json.dumps(meta, indent=2))
        return meta


def params_path(session_id: str) -> Path:
    return session_dir(session_id) / PARAMS_FILE_NAME


def read_params(session_id: str) -> ReliefParams:
    """Load persisted params, or the schema defaults if none have been saved yet."""
    if not session_dir(session_id).exists():
        raise SessionNotFoundError(session_id)
    path = params_path(session_id)
    if not path.exists():
        return ReliefParams()
    return ReliefParams.model_validate_json(path.read_text())


def write_params(session_id: str, params: ReliefParams) -> None:
    if not session_dir(session_id).exists():
        raise SessionNotFoundError(session_id)
    params_path(session_id).write_text(params.model_dump_json(indent=2))


def create_session(image_bytes: bytes, filename: str) -> SessionInfo:
    """Validate an uploaded image, EXIF-correct it, and persist it as a new session."""
    if len(image_bytes) > settings.max_upload_bytes:
        raise InvalidImageError(
            f"Image exceeds the {settings.max_upload_bytes // (1024 * 1024)} MB upload limit."
        )

    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            if img.format not in ALLOWED_FORMATS:
                raise InvalidImageError(
                    f"Unsupported image format '{img.format}'. Use JPEG, PNG, WebP, or HEIC."
                )

            megapixels = (img.width * img.height) / 1_000_000
            mp_limit = settings.max_upload_megapixels
            if megapixels > mp_limit:
                raise InvalidImageError(
                    f"Image is {megapixels:.1f} MP; the limit is {mp_limit:.0f} MP."
                )

            corrected = ImageOps.exif_transpose(img)
            if corrected is None:
                corrected = img
            corrected = corrected.convert("RGB")
    except InvalidImageError:
        raise
    except Exception as exc:  # noqa: BLE001 - any decode failure is a validation error
        raise InvalidImageError(f"Could not decode image: {exc}") from exc

    session_id = uuid.uuid4().hex
    out_dir = session_dir(session_id)
    out_dir.mkdir(parents=True, exist_ok=False)

    # Persist the normalized PNG, then hash those exact bytes. The hash is the depth
    # cache key, so re-uploading the same photo (a new session) reuses cached depth.
    source_path = out_dir / SOURCE_IMAGE_NAME
    corrected.save(source_path, format="PNG")
    image_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()

    info = SessionInfo(
        session_id=session_id,
        width=corrected.width,
        height=corrected.height,
        image_hash=image_hash,
    )
    _write_meta(out_dir, info, original_filename=filename)
    return info


def _write_meta(out_dir: Path, info: SessionInfo, *, original_filename: str) -> None:
    meta = {
        "session_id": info.session_id,
        "original_filename": original_filename,
        "width": info.width,
        "height": info.height,
        "image_hash": info.image_hash,
        "status": "created",
        "model_id": None,
        "device": None,
        "elapsed_s": None,
        "error": None,
    }
    (out_dir / META_FILE_NAME).write_text(json.dumps(meta, indent=2))
