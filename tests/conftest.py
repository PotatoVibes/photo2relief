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
