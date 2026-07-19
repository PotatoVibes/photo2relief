"""Regenerate the help page's example renders (static/help/img/) from a session.

Usage:  uv run python scripts/gen_help_examples.py <session_id>

The session must have raw depth cached for BOTH da3mono-large and da2-large
(upload a photo, then switch models once in the UI). Level-type params render as
grayscale heightmaps (hillshade shows slopes, not levels, and would hide what they
do); surface-texture params render as hillshade. Output images are gitignored --
they derive from whatever personal photo lives in the chosen session.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

from app import heightmap
from app.config import settings
from app.schemas import ReliefParams

OUT = Path("static/help/img")
LONG_SIDE = 400
DA3 = "da3mono-large"
DA2 = "da2-large"


def render(session: Path, params: ReliefParams, model_id: str, mode: str) -> np.ndarray:
    raw = np.load(session / f"raw_depth_{model_id}.npy")
    with Image.open(session / "source.png") as img:
        luma = heightmap.luma_at_shape(img.convert("RGB"), raw.shape)
    h = heightmap.compute_heightmap(raw, luma, params, target_long_side=LONG_SIDE)
    if mode == "gray":
        return h
    return heightmap.hillshade(h, params.model_width_mm, params.relief_height_mm)


def p(**kw: object) -> ReliefParams:
    kw.setdefault("depth_model", DA3)
    return ReliefParams(**kw)  # type: ignore[arg-type]


CASES: list[tuple[str, ReliefParams, str, str]] = [
    ("baseline_gray", p(), DA3, "gray"),
    ("baseline_shade", p(), DA3, "shade"),
    ("model_da2_gray", p(depth_model=DA2), DA2, "gray"),
    ("invert_on", p(invert_depth=True), DA3, "gray"),
    ("gamma_05", p(gamma=0.5), DA3, "gray"),
    ("gamma_25", p(gamma=2.5), DA3, "gray"),
    ("floor_035", p(depth_floor=0.35), DA3, "gray"),
    ("ceiling_07", p(depth_ceiling=0.7), DA3, "gray"),
    ("flatten_030", p(flatten_background=True, background_threshold=0.3), DA3, "gray"),
    ("flatten_050", p(flatten_background=True, background_threshold=0.5), DA3, "gray"),
    (
        "flatten_da2",
        p(depth_model=DA2, depth_floor=0.36, flatten_background=True, background_threshold=0.06),
        DA2,
        "gray",
    ),
    ("smoothing_0", p(smoothing=0.0), DA3, "shade"),
    ("smoothing_5", p(smoothing=5.0), DA3, "shade"),
    ("gaussian_3", p(smoothing=3.0, edge_preserve=False), DA3, "shade"),
    ("bilateral_3", p(smoothing=3.0, edge_preserve=True), DA3, "shade"),
    ("detail_0", p(detail_blend=0.0), DA3, "shade"),
    ("detail_07", p(detail_blend=0.7), DA3, "shade"),
    ("taper_10", p(edge_taper_mm=10.0), DA3, "gray"),
    ("frame_10", p(border_frame_mm=10.0), DA3, "gray"),
]


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    session = settings.sessions_dir / sys.argv[1]
    for model in (DA3, DA2):
        if not (session / f"raw_depth_{model}.npy").exists():
            print(f"error: {session} has no cached depth for {model}")
            return 1
    OUT.mkdir(parents=True, exist_ok=True)
    for name, params, model, mode in CASES:
        shade = render(session, params, model, mode)
        img = Image.fromarray((np.clip(shade, 0, 1) * 255).astype(np.uint8), mode="L")
        img.save(OUT / f"{name}.png", optimize=True)
        print(f"{name}.png  {img.size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
