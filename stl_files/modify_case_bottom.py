"""Resize screw bores and the four narrow underside slots in Case_Bottom.stl."""

from pathlib import Path

import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "Case_Bottom.stl"
OUTPUT = HERE / "Case_Bottom_modified.stl"
HOLE_RADIUS = 3.0
SLOT_WIDTH = 5.5


def curved_components(mesh):
    adjacency = mesh.face_adjacency
    similarity = np.einsum(
        "ij,ij->i",
        mesh.face_normals[adjacency[:, 0]],
        mesh.face_normals[adjacency[:, 1]],
    )
    adjacency = adjacency[similarity > 0.92]
    graph = coo_matrix(
        (
            np.ones(2 * len(adjacency)),
            (
                np.r_[adjacency[:, 0], adjacency[:, 1]],
                np.r_[adjacency[:, 1], adjacency[:, 0]],
            ),
        ),
        shape=(len(mesh.faces), len(mesh.faces)),
    ).tocsr()
    count, labels = connected_components(graph, directed=False)

    result = []
    for label in range(count):
        face_ids = np.flatnonzero(labels == label)
        if len(face_ids) != 48:
            continue
        normals = mesh.face_normals[face_ids]
        values, vectors = np.linalg.eigh(np.cov(normals.T))
        axis = vectors[:, 0]
        vertex_ids = np.unique(mesh.faces[face_ids])
        points = mesh.vertices[vertex_ids]
        center = points.mean(axis=0)
        relative = points - center
        radial = relative - np.outer(relative @ axis, axis)
        radius = float(np.median(np.linalg.norm(radial, axis=1)))
        if 1.8 <= radius <= 2.8 and values[1] > 0.1:
            result.append((vertex_ids, center, axis, radial, radius))
    return result


mesh = trimesh.load(SOURCE, force="mesh")
original_bounds = mesh.bounds.copy()
holes = curved_components(mesh)
if len(holes) != 24:
    raise RuntimeError(f"Expected 24 screw bores, found {len(holes)}")

# Expand the existing faceted bore walls radially while keeping their axes fixed.
vertices = mesh.vertices.copy()
for vertex_ids, center, axis, radial, radius in holes:
    valid = np.linalg.norm(radial, axis=1) > 1e-8
    vertices[vertex_ids[valid]] = (
        center
        + np.outer((vertices[vertex_ids[valid]] - center) @ axis, axis)
        + radial[valid] * (HOLE_RADIUS / radius)
    )
mesh.vertices = vertices

# The underside has four long slots with 4.5 mm openings in the sectioned mesh.
# Move their two side walls outward by 0.5 mm to reach the requested width.
slot_centers_x = (-18.7, -9.2, 9.2, 18.7)
vertices = mesh.vertices.copy()
for center_x in slot_centers_x:
    dx = vertices[:, 0] - center_x
    side = (np.abs(dx) > 2.0) & (np.abs(dx) < 2.6)
    in_slot_depth = (vertices[:, 1] > -39.0) & (vertices[:, 1] < -31.0)
    along_slot = (vertices[:, 2] > -60.5) & (vertices[:, 2] < -35.0)
    mask = side & in_slot_depth & along_slot
    vertices[mask, 0] += np.sign(dx[mask]) * ((SLOT_WIDTH - 4.5) / 2.0)
mesh.vertices = vertices

mesh.remove_unreferenced_vertices()
mesh.fix_normals()
if not mesh.is_watertight:
    raise RuntimeError("Modified mesh is not watertight")
if not np.allclose(mesh.bounds, original_bounds, atol=1.0):
    raise RuntimeError("Unexpected change to the outer envelope")

mesh.export(OUTPUT)
print(f"Saved {OUTPUT}")
print(f"Screw bores: {len(holes)} at 6.0 mm diameter")
print(f"Underside slots: {len(slot_centers_x)} at 5.5 mm width")
print(f"Watertight: {mesh.is_watertight}; bounds: {mesh.extents}")
