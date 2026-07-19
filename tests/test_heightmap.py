from __future__ import annotations

import numpy as np
import pytest

from app import heightmap as hm
from app.schemas import ReliefParams

# --- percentile_clip_normalize ------------------------------------------------------


def test_percentile_clip_normalize_stretches_window() -> None:
    depth = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
    out = hm.percentile_clip_normalize(depth, floor=0.25, ceiling=0.75)
    assert out.min() == pytest.approx(0.0)
    assert out.max() == pytest.approx(1.0)
    assert out[2] == pytest.approx(0.5)  # midpoint unchanged after re-stretch


def test_percentile_clip_normalize_identity_for_full_range() -> None:
    depth = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    out = hm.percentile_clip_normalize(depth, floor=0.0, ceiling=1.0)
    np.testing.assert_allclose(out, depth)


def test_percentile_clip_normalize_rejects_bad_window() -> None:
    with pytest.raises(ValueError):
        hm.percentile_clip_normalize(np.zeros((2, 2), dtype=np.float32), floor=0.5, ceiling=0.5)


# --- invert ---------------------------------------------------------------------------


def test_invert_flips_near_far() -> None:
    h = np.array([0.0, 0.3, 1.0], dtype=np.float32)
    out = hm.invert(h)
    np.testing.assert_allclose(out, [1.0, 0.7, 0.0], atol=1e-6)


# --- apply_gamma ------------------------------------------------------------------------


def test_gamma_one_is_identity() -> None:
    h = np.array([0.0, 0.3, 0.7, 1.0], dtype=np.float32)
    np.testing.assert_allclose(hm.apply_gamma(h, 1.0), h)


def test_gamma_greater_than_one_compresses_background_emphasizes_foreground() -> None:
    h = np.array([0.5, 0.9, 1.0], dtype=np.float32)
    out = hm.apply_gamma(h, 2.0)
    # Background midtone shrinks toward 0.
    assert out[0] < h[0]
    # But the *foreground* delta (0.9 -> 1.0) widens relative to the input delta.
    input_delta = h[2] - h[1]
    output_delta = out[2] - out[1]
    assert output_delta > input_delta


# --- flatten_background ------------------------------------------------------------------


def test_flatten_background_zeroes_below_threshold_only() -> None:
    h = np.array([0.05, 0.1, 0.2, 0.5], dtype=np.float32)
    out = hm.flatten_background(h, threshold=0.15)
    np.testing.assert_allclose(out, [0.0, 0.0, 0.2, 0.5])


# --- smooth -----------------------------------------------------------------------------


def test_smooth_zero_sigma_is_noop() -> None:
    h = np.random.default_rng(0).random((16, 16)).astype(np.float32)
    out = hm.smooth(h, sigma=0.0, edge_preserve=False)
    assert out is h


def test_smooth_gaussian_reduces_variance() -> None:
    rng = np.random.default_rng(1)
    h = rng.random((32, 32)).astype(np.float32)
    out = hm.smooth(h, sigma=2.0, edge_preserve=False)
    assert out.var() < h.var()
    assert out.shape == h.shape


def test_smooth_bilateral_preserves_step_edge_better_than_gaussian() -> None:
    h = np.zeros((40, 40), dtype=np.float32)
    h[:, 20:] = 1.0
    gauss = hm.smooth(h.copy(), sigma=3.0, edge_preserve=False)
    bilateral = hm.smooth(h.copy(), sigma=3.0, edge_preserve=True)
    # Both blur, but bilateral keeps the step steeper (fewer smeared midtone pixels).
    mid_col = 20
    gauss_gradient = abs(float(gauss[20, mid_col + 2]) - float(gauss[20, mid_col - 2]))
    bilateral_gradient = abs(float(bilateral[20, mid_col + 2]) - float(bilateral[20, mid_col - 2]))
    assert bilateral_gradient >= gauss_gradient


# --- detail_blend -----------------------------------------------------------------------


def test_detail_blend_zero_amount_is_noop() -> None:
    h = np.full((10, 10), 0.5, dtype=np.float32)
    luma = np.random.default_rng(2).random((10, 10)).astype(np.float32)
    out = hm.detail_blend(h, luma, amount=0.0)
    assert out is h


def test_detail_blend_adds_high_frequency_content() -> None:
    size = 32
    h = np.full((size, size), 0.5, dtype=np.float32)
    luma = np.zeros((size, size), dtype=np.float32)
    luma[:, size // 2 :] = 1.0  # sharp edge -> strong high-pass response
    out = hm.detail_blend(h, luma, amount=1.0)
    assert not np.allclose(out, 0.5)
    # Effect stays small relative to the full relief range (~15% target, SPEC Sec5.2).
    assert np.abs(out - 0.5).max() < 0.2


# --- resize_to_long_side ------------------------------------------------------------------


def test_resize_to_long_side_noop_when_already_target() -> None:
    h = np.zeros((10, 20), dtype=np.float32)
    out = hm.resize_to_long_side(h, target_long_side=20)
    assert out.shape == (10, 20)


def test_resize_to_long_side_downscales_preserving_aspect() -> None:
    h = np.zeros((256, 512), dtype=np.float32)
    out = hm.resize_to_long_side(h, target_long_side=128)
    assert out.shape == (64, 128)


def test_resize_to_long_side_upscales_preserving_aspect() -> None:
    h = np.zeros((30, 60), dtype=np.float32)
    out = hm.resize_to_long_side(h, target_long_side=120)
    assert out.shape == (60, 120)


# --- edge_taper (geometry) ----------------------------------------------------------------


def test_edge_taper_zero_mm_is_noop() -> None:
    h = np.ones((10, 10), dtype=np.float32)
    out = hm.edge_taper(h, taper_mm=0.0, model_width_mm=100.0)
    assert out is h


def test_edge_taper_zeroes_the_outer_edge_and_preserves_center() -> None:
    size = 100
    h = np.ones((size, size), dtype=np.float32)
    # 100 px wide == 100 mm -> 1 px/mm; a 10 mm taper is a 10 px ramp.
    out = hm.edge_taper(h, taper_mm=10.0, model_width_mm=100.0)
    assert out[0, 0] == pytest.approx(0.0)
    assert out[size // 2, size // 2] == pytest.approx(1.0)
    # Monotonic ramp along a row moving inward from the left edge.
    row = out[size // 2, : size // 2]
    assert np.all(np.diff(row) >= -1e-6)


# --- border_frame (geometry) --------------------------------------------------------------


def test_border_frame_zero_mm_is_noop() -> None:
    h = np.ones((10, 10), dtype=np.float32)
    out = hm.border_frame(h, frame_mm=0.0, model_width_mm=100.0)
    assert out is h


def test_border_frame_subpixel_mm_rounds_to_noop() -> None:
    # frame_mm > 0 but too small to round to a whole pixel at this scale.
    h = np.ones((10, 10), dtype=np.float32)
    out = hm.border_frame(h, frame_mm=0.01, model_width_mm=1000.0)
    assert out.shape == h.shape
    np.testing.assert_array_equal(out, h)


def test_border_frame_pads_with_flat_full_height_margin() -> None:
    # The frame is a machinable reference at full stock height (base + relief), NOT a
    # zero-height flange -- its physical height must track relief_height_mm (owner
    # decision 2026-07-19; see Decisions log).
    size = 100
    h = np.full((size, size), 0.25, dtype=np.float32)
    # 1 px/mm, 5 mm frame -> 5 px pad on every side.
    out = hm.border_frame(h, frame_mm=5.0, model_width_mm=100.0)
    assert out.shape == (110, 110)
    assert np.all(out[:5, :] == 1.0)
    assert np.all(out[-5:, :] == 1.0)
    assert np.all(out[:, :5] == 1.0)
    assert np.all(out[:, -5:] == 1.0)
    assert np.all(out[5:105, 5:105] == 0.25)


# --- compute_heightmap (end-to-end orchestration) ------------------------------------------


def _sample_raw_depth(size: int = 64) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    r = np.sqrt((xx - size / 2) ** 2 + (yy - size / 2) ** 2) / (size / 2)
    return np.clip(1.0 - r, 0.0, 1.0)


def test_compute_heightmap_shape_and_range() -> None:
    raw = _sample_raw_depth(64)
    luma = np.random.default_rng(3).random((64, 64)).astype(np.float32)
    params = ReliefParams()
    out = hm.compute_heightmap(raw, luma, params, target_long_side=128)
    assert out.shape == (128, 128)
    assert out.dtype == np.float32
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_compute_heightmap_rejects_mismatched_shapes() -> None:
    raw = _sample_raw_depth(64)
    luma = np.zeros((32, 32), dtype=np.float32)
    with pytest.raises(ValueError):
        hm.compute_heightmap(raw, luma, ReliefParams(), target_long_side=64)


def test_compute_heightmap_applies_invert_and_flatten_and_frame() -> None:
    raw = _sample_raw_depth(64)
    luma = np.zeros((64, 64), dtype=np.float32)
    params = ReliefParams(
        invert_depth=True,
        flatten_background=True,
        background_threshold=0.1,
        border_frame_mm=10.0,
        model_width_mm=64.0,
        edge_taper_mm=0.0,
        smoothing=0.0,
        detail_blend=0.0,
    )
    out = hm.compute_heightmap(raw, luma, params, target_long_side=64)
    # border_frame_mm=10 @ 1 px/mm on a 64px grid -> 10 px pad each side, at full height.
    assert out.shape == (84, 84)
    assert np.all(out[:10, :] == 1.0)


# --- hillshade ------------------------------------------------------------------------------


def test_hillshade_flat_surface_is_uniform() -> None:
    h = np.full((20, 20), 0.5, dtype=np.float32)
    shade = hm.hillshade(h, model_width_mm=50.0, relief_height_mm=8.0)
    assert shade.shape == (20, 20)
    assert shade.min() >= 0.0
    assert shade.max() <= 1.0
    np.testing.assert_allclose(shade, shade[0, 0], atol=1e-5)


def test_hillshade_responds_to_slope() -> None:
    # A uniformly tilted plane has one normal everywhere (uniform shade, by design --
    # see test_hillshade_flat_surface_is_uniform for that case). A dome varies its
    # slope across the surface, so a real depth-map-like bump should shade unevenly.
    h = _sample_raw_depth(30)
    shade = hm.hillshade(h, model_width_mm=30.0, relief_height_mm=10.0)
    assert not np.allclose(shade, shade[0, 0])
