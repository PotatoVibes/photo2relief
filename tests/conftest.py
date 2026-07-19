from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session", autouse=True)
def _ensure_fixtures() -> None:
    """Generate the small synthetic fixtures on first run; they're git-ignored binaries."""
    FIXTURES_DIR.mkdir(exist_ok=True)

    gradient_path = FIXTURES_DIR / "gradient.png"
    if not gradient_path.exists():
        size = 64
        yy, xx = np.mgrid[0:size, 0:size]
        radial = 1.0 - np.sqrt((xx - size / 2) ** 2 + (yy - size / 2) ** 2) / (size / 2)
        radial = np.clip(radial, 0, 1)
        img = Image.fromarray((radial * 255).astype("uint8"), mode="L")
        img.save(gradient_path)

    portrait_path = FIXTURES_DIR / "portrait.jpg"
    if not portrait_path.exists():
        img = Image.new("RGB", (400, 600), color=(120, 140, 160))
        img.save(portrait_path, format="JPEG")


@pytest.fixture
def gradient_png_bytes() -> bytes:
    with open(FIXTURES_DIR / "gradient.png", "rb") as f:
        return f.read()


@pytest.fixture
def portrait_jpeg_bytes() -> bytes:
    with open(FIXTURES_DIR / "portrait.jpg", "rb") as f:
        return f.read()


def make_solid_image_bytes(fmt: str = "PNG", size: tuple[int, int] = (100, 100)) -> bytes:
    img = Image.new("RGB", size, color=(200, 50, 50))
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def make_sphere_image_bytes(size: tuple[int, int] = (256, 320)) -> bytes:
    """A shaded sphere on a flat background — a synthetic 'portrait' whose subject any
    depth model must read as nearer than the corners. Used to verify conventions."""
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx = (xx - w * 0.5) / (min(w, h) * 0.34)
    dy = (yy - h * 0.52) / (min(w, h) * 0.34)
    r2 = dx * dx + dy * dy
    inside = r2 < 1.0
    dz = np.sqrt(np.clip(1.0 - r2, 0.0, 1.0))
    shade = np.clip(-0.4 * dx - 0.5 * dy + 0.75 * dz, 0.0, 1.0)
    img = np.full((h, w), 0.35, dtype=np.float32)
    img[inside] = 0.25 + 0.7 * shade[inside]
    rgb = (np.stack([img, img * 0.95, img * 0.88], axis=-1).clip(0, 1) * 255).astype("uint8")
    buf = io.BytesIO()
    Image.fromarray(rgb, "RGB").save(buf, format="PNG")
    return buf.getvalue()


def region_means(depth: np.ndarray) -> tuple[float, float]:
    """(center mean, corner mean) of a depth map — for subject-vs-background checks."""
    h, w = depth.shape
    center = float(depth[int(h * 0.35) : int(h * 0.70), int(w * 0.35) : int(w * 0.65)].mean())
    corners = float(
        np.mean(
            [
                depth[: h // 6, : w // 6].mean(),
                depth[: h // 6, -w // 6 :].mean(),
                depth[-h // 6 :, : w // 6].mean(),
                depth[-h // 6 :, -w // 6 :].mean(),
            ]
        )
    )
    return center, corners
