"""Pure heightmap transforms: raw_depth (+ luminance) + ReliefParams -> heightmap in [0,1].

No I/O, no globals (CLAUDE.md hard rule) -- every function takes arrays/scalars and
returns arrays, so the whole pipeline is trivially unit-testable and never touches a
depth model. Pipeline order matches SPEC Sec2.3 / Sec5.2 exactly:

    percentile clip+normalize -> invert -> gamma -> background flatten -> smoothing ->
    luminance detail blend -> resize to target grid -> edge taper -> border frame -> clamp
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

from app.schemas import ReliefParams

DETAIL_HIGHPASS_SIGMA_PX = 8.0
DETAIL_SCALE = 0.15  # detail_blend=1.0 contributes ~15% of relief range (SPEC Sec5.2)


def percentile_clip_normalize(depth: np.ndarray, floor: float, ceiling: float) -> np.ndarray:
    """Clip to [floor, ceiling] then contrast-stretch back to [0, 1].

    Kills far-background noise below `floor` and clamps near-field blowouts above
    `ceiling`, then re-stretches the surviving window to fill [0, 1].
    """
    if ceiling <= floor:
        raise ValueError("ceiling must be greater than floor")
    clipped = np.clip(depth, floor, ceiling)
    return ((clipped - floor) / (ceiling - floor)).astype(np.float32)


def invert(h: np.ndarray) -> np.ndarray:
    return (1.0 - h).astype(np.float32)


def apply_gamma(h: np.ndarray, gamma: float) -> np.ndarray:
    """gamma > 1 compresses the background (values near 0 shrink further, h**gamma < h)
    while steepening the slope near 1 -- widening foreground contrast. Net effect:
    background flattens, subject relief is emphasized.
    """
    return np.power(np.clip(h, 0.0, 1.0), gamma).astype(np.float32)


def flatten_background(h: np.ndarray, threshold: float) -> np.ndarray:
    out = h.copy()
    out[out < threshold] = 0.0
    return out


def smooth(h: np.ndarray, sigma: float, edge_preserve: bool) -> np.ndarray:
    if sigma <= 0:
        return h
    if edge_preserve:
        src = np.ascontiguousarray(h, dtype=np.float32)
        return cv2.bilateralFilter(src, d=-1, sigmaColor=0.15, sigmaSpace=float(sigma) * 2.0)
    return gaussian_filter(h, sigma=sigma).astype(np.float32)


def detail_blend(h: np.ndarray, luma: np.ndarray, amount: float) -> np.ndarray:
    """Add a luminance high-pass (hair, fabric, fine texture depth models smooth over)."""
    if amount <= 0:
        return h
    highpass = luma - gaussian_filter(luma, sigma=DETAIL_HIGHPASS_SIGMA_PX)
    return (h + amount * DETAIL_SCALE * highpass).astype(np.float32)


def resize_to_long_side(h: np.ndarray, target_long_side: int) -> np.ndarray:
    """Resize preserving aspect ratio; the long side becomes exactly target_long_side."""
    height, width = h.shape
    long_side = max(height, width)
    if long_side == target_long_side:
        return h.astype(np.float32)
    scale = target_long_side / long_side
    new_w = max(1, round(width * scale))
    new_h = max(1, round(height * scale))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    return cv2.resize(h, (new_w, new_h), interpolation=interp).astype(np.float32)


def edge_taper(h: np.ndarray, taper_mm: float, model_width_mm: float) -> np.ndarray:
    """Linearly ramp height to 0 over `taper_mm` at the outer border of the image content."""
    if taper_mm <= 0:
        return h
    height, width = h.shape
    px_per_mm = width / model_width_mm
    taper_px = max(1, round(taper_mm * px_per_mm))
    taper_px = min(taper_px, (min(height, width) + 1) // 2)
    row_dist = np.minimum(np.arange(height), height - 1 - np.arange(height))
    col_dist = np.minimum(np.arange(width), width - 1 - np.arange(width))
    dist = np.minimum(row_dist[:, None], col_dist[None, :])
    ramp = np.clip(dist / taper_px, 0.0, 1.0)
    return (h * ramp).astype(np.float32)


def border_frame(h: np.ndarray, frame_mm: float, model_width_mm: float) -> np.ndarray:
    """Pad with a flat, zero-height margin around the image (mounting/clamping flange).

    Uses the same model_width_mm->px scale as the image content, since the frame is
    physically outside the modeled subject, not a rescale of it.
    """
    if frame_mm <= 0:
        return h
    width = h.shape[1]
    px_per_mm = width / model_width_mm
    frame_px = round(frame_mm * px_per_mm)
    if frame_px <= 0:
        return h
    height = h.shape[0]
    out = np.zeros((height + 2 * frame_px, width + 2 * frame_px), dtype=np.float32)
    out[frame_px : frame_px + height, frame_px : frame_px + width] = h
    return out


def compute_heightmap(
    raw_depth: np.ndarray,
    luma: np.ndarray,
    params: ReliefParams,
    target_long_side: int,
) -> np.ndarray:
    """Full pipeline: raw_depth (+ luma, same shape, both at working/inference resolution)
    + params -> heightmap in [0, 1] at the requested grid resolution.

    Pure and cheap -- re-runs freely on every param change (SPEC Sec2.3); never touches
    the depth model.
    """
    if raw_depth.shape != luma.shape:
        raise ValueError(
            f"raw_depth {raw_depth.shape} and luma {luma.shape} must share a shape "
            "(resample luma to the depth's working resolution before calling)."
        )
    h = percentile_clip_normalize(raw_depth, params.depth_floor, params.depth_ceiling)
    if params.invert_depth:
        h = invert(h)
    h = apply_gamma(h, params.gamma)
    if params.flatten_background:
        h = flatten_background(h, params.background_threshold)
    h = smooth(h, params.smoothing, params.edge_preserve)
    h = detail_blend(h, luma, params.detail_blend)
    h = resize_to_long_side(h, target_long_side)
    h = edge_taper(h, params.edge_taper_mm, params.model_width_mm)
    h = border_frame(h, params.border_frame_mm, params.model_width_mm)
    return np.clip(h, 0.0, 1.0).astype(np.float32)


def hillshade(
    h: np.ndarray,
    model_width_mm: float,
    relief_height_mm: float,
    azimuth_deg: float = 315.0,
    altitude_deg: float = 45.0,
) -> np.ndarray:
    """Simple Lambertian hillshade of the heightmap surface -- reads far better than raw
    grayscale for judging relief (SPEC M3). Returns float32 [0, 1].
    """
    px_size_mm = model_width_mm / h.shape[1]
    z = h.astype(np.float32) * relief_height_mm
    gy, gx = np.gradient(z, px_size_mm)
    normal = np.stack([-gx, -gy, np.ones_like(z, dtype=np.float32)], axis=-1)
    normal = normal / np.linalg.norm(normal, axis=-1, keepdims=True)
    az = np.radians(azimuth_deg)
    alt = np.radians(altitude_deg)
    light = np.array(
        [np.cos(alt) * np.cos(az), np.cos(alt) * np.sin(az), np.sin(alt)],
        dtype=np.float32,
    )
    shade = np.clip(normal @ light, 0.0, 1.0)
    return shade.astype(np.float32)
