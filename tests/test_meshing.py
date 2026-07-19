"""Golden-file tests for app.meshing: pure numpy/trimesh, never touches a depth model
(CLAUDE.md hard rule -- heightmap.py/meshing.py stay pure and model-free).
"""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from app import meshing
from app.schemas import ReliefParams


def _synthetic_heightmap(h: int, w: int) -> np.ndarray:
    """Deterministic, smoothly-varying heightmap in [0, 1] -- a stand-in for a real
    depth-derived heightmap, sized however a test needs."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    return (np.sin(xx / w * np.pi) * np.cos(yy / h * np.pi) * 0.5 + 0.5).astype(np.float32)


def _params(**overrides: object) -> ReliefParams:
    defaults = dict(
        model_width_mm=150.0,
        relief_height_mm=8.0,
        base_thickness_mm=3.0,
        depth_model="da2-small",
    )
    defaults.update(overrides)
    return ReliefParams(**defaults)


# --- watertightness at export resolutions -------------------------------------------


@pytest.mark.parametrize("side", [512, 1024, 2048])
def test_watertight_at_export_resolutions(side: int) -> None:
    hm = _synthetic_heightmap(side, side)
    mesh = meshing.build_relief_mesh(hm, _params(resolution=side))
    assert mesh.is_watertight
    assert mesh.volume > 0  # positive signed volume confirms outward-consistent winding


# --- bounding box --------------------------------------------------------------------


def test_bbox_matches_params_within_tolerance() -> None:
    h, w = 40, 64
    hm = _synthetic_heightmap(h, w)
    params = _params(model_width_mm=200.0, relief_height_mm=12.0, base_thickness_mm=4.0)
    mesh = meshing.build_relief_mesh(hm, params)

    expected_height_mm = params.model_width_mm * h / w
    expected_z_max = params.base_thickness_mm + params.relief_height_mm  # heightmap peaks at 1.0
    bounds = mesh.bounds

    assert bounds[0] == pytest.approx([0.0, 0.0, 0.0], abs=1e-6)
    assert bounds[1][0] == pytest.approx(params.model_width_mm, abs=0.01)
    assert bounds[1][1] == pytest.approx(expected_height_mm, abs=0.01)
    assert bounds[1][2] == pytest.approx(expected_z_max, abs=0.01)


def test_border_frame_adds_to_physical_size_at_exact_mm() -> None:
    """model_width_mm is the IMAGE CONTENT width; a border frame extends the part.
    Round trip heightmap.border_frame -> mesh: the part must be exactly
    content + 2*frame wide, and the frame surface must sit at full stock height.
    """
    from app import heightmap as hm

    content = np.zeros((80, 100), dtype=np.float32)  # 100 px content = 100 mm -> 1 px/mm
    params = _params(model_width_mm=100.0, border_frame_mm=10.0)
    padded = hm.border_frame(content, frame_mm=10.0, model_width_mm=100.0)
    mesh = meshing.build_relief_mesh(padded, params)

    assert mesh.is_watertight
    assert mesh.bounds[1][0] == pytest.approx(120.0, abs=0.01)  # 100 + 2*10
    # Frame pixels (heightmap = 1.0) sit at base + relief = full stock height.
    assert mesh.bounds[1][2] == pytest.approx(
        params.base_thickness_mm + params.relief_height_mm, abs=0.01
    )


def test_flat_zero_heightmap_bbox_is_just_the_base_slab() -> None:
    hm = np.zeros((20, 30), dtype=np.float32)
    params = _params(model_width_mm=90.0, relief_height_mm=5.0, base_thickness_mm=2.5)
    mesh = meshing.build_relief_mesh(hm, params)
    assert mesh.is_watertight
    assert mesh.bounds[1][2] == pytest.approx(2.5, abs=0.01)  # base_thickness only, no relief


# --- Y-orientation ---------------------------------------------------------------------


def test_row_zero_is_image_top_and_maps_to_max_y() -> None:
    """SPEC Sec5.3: image row 0 is top; mesh +Y must be image top so the relief isn't
    mirrored. Use an asymmetric heightmap (tall at row 0 only) to prove both the height
    and the Y placement land on the correct row.
    """
    h, w = 6, 8
    hm = np.zeros((h, w), dtype=np.float32)
    hm[0, :] = 1.0  # only the image's top row is raised

    params = _params(model_width_mm=120.0, relief_height_mm=10.0, base_thickness_mm=2.0)
    mesh = meshing.build_relief_mesh(hm, params)
    assert mesh.is_watertight

    top_vertices = mesh.vertices[: h * w].reshape(h, w, 3)
    raised_row = 0 if top_vertices[0, 0, 2] > top_vertices[-1, 0, 2] else h - 1

    # The raised row must be the one at maximum Y (top of the physical model).
    assert top_vertices[raised_row, 0, 1] == pytest.approx(top_vertices[:, :, 1].max())
    assert raised_row == 0
    np.testing.assert_allclose(top_vertices[0, :, 2], 12.0)  # base(2) + relief(10) * 1.0
    np.testing.assert_allclose(top_vertices[-1, :, 2], 2.0)  # base only


# --- decimation -------------------------------------------------------------------------


def test_decimate_ratio_zero_is_a_no_op() -> None:
    hm = _synthetic_heightmap(30, 30)
    mesh = meshing.build_relief_mesh(hm, _params())
    same = meshing.decimate_mesh(mesh, 0.0)
    assert same is mesh


def test_decimate_reduces_triangle_count_and_stays_watertight() -> None:
    hm = _synthetic_heightmap(80, 80)
    mesh = meshing.build_relief_mesh(hm, _params())
    original_tris = len(mesh.faces)

    decimated = meshing.decimate_mesh(mesh, 0.5)
    assert decimated.is_watertight
    assert len(decimated.faces) < original_tris


def test_build_export_mesh_without_decimation_skips_that_stage() -> None:
    hm = _synthetic_heightmap(40, 40)
    stages: list[str] = []
    mesh = meshing.build_export_mesh(hm, _params(), on_stage=lambda frac, name: stages.append(name))
    assert mesh.is_watertight
    assert stages == ["validating watertight"]  # no decimate stages at ratio 0


def test_build_export_mesh_reports_stage_progress() -> None:
    hm = _synthetic_heightmap(40, 40)
    stages: list[tuple[float, str]] = []
    meshing.build_export_mesh(
        hm, _params(decimate_ratio=0.5), on_stage=lambda frac, name: stages.append((frac, name))
    )
    names = [name for _, name in stages]
    assert names == ["validating watertight", "decimating", "re-validating watertight"]
    fracs = [frac for frac, _ in stages]
    assert fracs == sorted(fracs)  # progress is monotonic


def test_build_export_mesh_applies_decimation() -> None:
    hm = _synthetic_heightmap(80, 80)
    params = _params(decimate_ratio=0.6)
    mesh = meshing.build_export_mesh(hm, params)
    assert mesh.is_watertight
    # A fresh, non-decimated build for comparison.
    full = meshing.build_relief_mesh(hm, _params())
    assert len(mesh.faces) < len(full.faces)


# --- export bytes -----------------------------------------------------------------------


def test_export_bytes_stl_is_binary_and_reloadable() -> None:
    import io

    hm = _synthetic_heightmap(20, 24)
    mesh = meshing.build_relief_mesh(hm, _params())
    data = meshing.export_bytes(mesh, "stl")
    assert isinstance(data, bytes)
    # Binary STL: an 80-byte header + a uint32 triangle count, not the ASCII "solid" keyword.
    assert not data.startswith(b"solid")

    reloaded = trimesh.load(io.BytesIO(data), file_type="stl")
    assert reloaded.is_watertight
    assert len(reloaded.faces) == len(mesh.faces)


def test_export_bytes_obj_is_text() -> None:
    hm = _synthetic_heightmap(20, 24)
    mesh = meshing.build_relief_mesh(hm, _params())
    data = meshing.export_bytes(mesh, "obj")
    assert isinstance(data, bytes)
    assert data.startswith(b"#")  # trimesh's OBJ export leads with a comment header


# --- error paths ------------------------------------------------------------------------


def test_build_relief_mesh_rejects_degenerate_heightmap() -> None:
    with pytest.raises(ValueError):
        meshing.build_relief_mesh(np.zeros((1, 10), dtype=np.float32), _params())


def test_not_watertight_error_is_a_runtime_error() -> None:
    assert issubclass(meshing.NotWatertightError, RuntimeError)


def _open_triangle() -> trimesh.Trimesh:
    # A single triangle: a valid Trimesh, but never watertight (all edges are boundary).
    return trimesh.Trimesh(
        vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]], faces=[[0, 1, 2]], process=False
    )


def test_build_export_mesh_raises_when_assembly_not_watertight(monkeypatch) -> None:
    monkeypatch.setattr(meshing, "build_relief_mesh", lambda hm, params: _open_triangle())
    with pytest.raises(meshing.NotWatertightError, match="Assembled"):
        meshing.build_export_mesh(_synthetic_heightmap(4, 4), _params())


def test_build_export_mesh_raises_when_decimation_breaks_watertightness(monkeypatch) -> None:
    monkeypatch.setattr(meshing, "decimate_mesh", lambda mesh, ratio: _open_triangle())
    with pytest.raises(meshing.NotWatertightError, match="after decimation"):
        meshing.build_export_mesh(_synthetic_heightmap(4, 4), _params(decimate_ratio=0.5))
