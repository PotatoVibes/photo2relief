"""Heightmap -> watertight relief mesh (grid triangulation) + STL/OBJ export.

Rationale for grid triangulation over marching cubes (SPEC Sec2.1): the input is a
heightmap, so a structured grid mesh is simpler, faster, produces cleaner topology for
CAM, and trivially guarantees manifoldness once the base/walls are stitched.

Geometry (SPEC Sec5.3): the top surface is an (H x W) vertex grid at
z = base_thickness_mm + h * relief_height_mm. The outer ring of that grid (its
perimeter) is walled straight down to a flat base at z=0, and the base is closed with
a fan triangulation from its centroid -- this needs only O(H+W) extra triangles rather
than a full second grid, while still sharing exact edges with the walls (required for
watertightness: every edge must border exactly two faces).

Winding is chosen analytically, not repaired after the fact: for any triangle, the
z-component of (v1-v0) x (v2-v0) depends only on the x,y coordinates of its vertices,
never on z (a general identity of the 3D cross product) -- so each face's outward
orientation is fully determined by its (x, y) projection alone and can be gotten right
at construction time, no matter how steep the relief. This matters in practice:
trimesh's ``fix_normals()`` repair walks the face-adjacency graph and was measured at
~1 min for a 512x512 grid (~0.5M faces) here -- far too slow to run per export at
1024/2048. Building correct winding up front avoids that pass entirely.
"""

from __future__ import annotations

import numpy as np
import trimesh

from app.schemas import ReliefParams

DEFAULT_DECIMATE_AGGRESSIVENESS = 7.0  # fast_simplification default; balances speed/quality


class NotWatertightError(RuntimeError):
    """Raised when an assembled mesh fails trimesh's watertight check.

    CLAUDE.md hard rule: never ship a leaky mesh -- fail loudly instead.
    """


def _perimeter_indices(h: int, w: int) -> np.ndarray:
    """Cyclic list of grid linear indices (row-major, idx = i*w + j) tracing the
    rectangle boundary exactly once, without repeating a corner."""
    if h < 2 or w < 2:
        raise ValueError(f"heightmap must be at least 2x2 to mesh, got {h}x{w}")

    def idx(i: int, j: int) -> int:
        return i * w + j

    top = [idx(0, j) for j in range(w)]
    right = [idx(i, w - 1) for i in range(1, h)]
    bottom = [idx(h - 1, j) for j in range(w - 2, -1, -1)]
    left = [idx(i, 0) for i in range(h - 2, 0, -1)]
    return np.array(top + right + bottom + left, dtype=np.int64)


def build_relief_mesh(heightmap: np.ndarray, params: ReliefParams) -> trimesh.Trimesh:
    """Assemble a watertight solid: displaced top surface + flat base + side walls.

    Units: millimeters. X spans `model_width_mm`; Y is derived from the heightmap's
    aspect ratio. +Y = image top (row 0), per SPEC Sec5.3 -- row 0 maps to the maximum
    Y coordinate so the relief isn't mirrored vertically.
    """
    h, w = heightmap.shape
    # total_width_mm, not model_width_mm: the heightmap may carry a border-frame pad,
    # and the frame adds to the part's physical size rather than squeezing the image.
    total_width_mm = params.total_width_mm
    model_height_mm = total_width_mm * h / w

    xs = np.linspace(0.0, total_width_mm, w, dtype=np.float64)
    ys = np.linspace(model_height_mm, 0.0, h, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(xs, ys)
    z_top = params.base_thickness_mm + heightmap.astype(np.float64) * params.relief_height_mm

    top_vertices = np.stack([grid_x, grid_y, z_top], axis=-1).reshape(-1, 3)

    i0, j0 = np.meshgrid(np.arange(h - 1), np.arange(w - 1), indexing="ij")

    def idx(ii: np.ndarray, jj: np.ndarray) -> np.ndarray:
        return ii * w + jj

    v00 = idx(i0, j0).ravel()
    v01 = idx(i0, j0 + 1).ravel()
    v10 = idx(i0 + 1, j0).ravel()
    v11 = idx(i0 + 1, j0 + 1).ravel()
    top_faces = np.concatenate(
        [np.stack([v00, v10, v01], axis=1), np.stack([v10, v11, v01], axis=1)], axis=0
    )

    perimeter = _perimeter_indices(h, w)
    n_top = top_vertices.shape[0]
    n_perim = perimeter.shape[0]

    bottom_perim_vertices = top_vertices[perimeter].copy()
    bottom_perim_vertices[:, 2] = 0.0
    bottom_start = n_top
    center_idx = n_top + n_perim
    center = np.array([total_width_mm / 2.0, model_height_mm / 2.0, 0.0])

    local = np.arange(n_perim)
    top_a = perimeter
    top_b = perimeter[(local + 1) % n_perim]
    bot_a = bottom_start + local
    bot_b = bottom_start + (local + 1) % n_perim

    # Winding chosen analytically (see module docstring) for outward normals:
    # walls point horizontally away from center, the base cap points -Z.
    wall_faces = np.concatenate(
        [np.stack([top_a, top_b, bot_a], axis=1), np.stack([bot_a, top_b, bot_b], axis=1)],
        axis=0,
    )
    cap_faces = np.stack([np.full(n_perim, center_idx), bot_a, bot_b], axis=1)

    vertices = np.concatenate([top_vertices, bottom_perim_vertices, center[None, :]], axis=0)
    faces = np.concatenate([top_faces, wall_faces, cap_faces], axis=0)

    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def decimate_mesh(mesh: trimesh.Trimesh, ratio: float) -> trimesh.Trimesh:
    """Decimate the whole assembled solid (not just the top surface) and re-validate
    watertightness -- SPEC Sec5.3 offers either approach; whole-solid is simpler and
    keeps walls/base consistent with a resampled top surface without re-stitching.
    """
    if ratio <= 0:
        return mesh
    import fast_simplification

    points, faces = fast_simplification.simplify(
        mesh.vertices, mesh.faces, target_reduction=ratio, agg=DEFAULT_DECIMATE_AGGRESSIVENESS
    )
    # Edge-collapse simplification preserves surviving faces' original (already-outward)
    # winding, so no fix_normals() pass is needed here either.
    return trimesh.Trimesh(vertices=points, faces=faces, process=True)


def build_export_mesh(heightmap: np.ndarray, params: ReliefParams) -> trimesh.Trimesh:
    """Full meshing pipeline for export: build, validate, optionally decimate + re-validate."""
    mesh = build_relief_mesh(heightmap, params)
    if not mesh.is_watertight:
        raise NotWatertightError("Assembled relief mesh failed the watertight check.")
    mesh = decimate_mesh(mesh, params.decimate_ratio)
    if not mesh.is_watertight:
        raise NotWatertightError("Mesh failed the watertight check after decimation.")
    return mesh


def export_bytes(mesh: trimesh.Trimesh, output_format: str) -> bytes:
    """Serialize to binary STL, OBJ, or glb (three.js preview). Units are millimeters
    (STL itself is unitless -- the README documents selecting millimeters on Fusion
    import)."""
    data = mesh.export(file_type=output_format)
    return data if isinstance(data, bytes) else data.encode("utf-8")
