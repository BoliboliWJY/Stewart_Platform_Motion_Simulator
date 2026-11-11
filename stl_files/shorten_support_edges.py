"""Shorten the six 28.45 mm support edges in Case_Bottom by 3 mm."""

from pathlib import Path

import numpy as np
import trimesh


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "Case_Bottom.stl"
OUTPUT = HERE / "Case_Bottom_supports_shortened_3mm.stl"
SHORTEN_BY = 3.0
EDGE_TOP_Y = -33.7678
EDGE_BOTTOM_Y = -62.2178
EDGE_XZ = [
    (41.01, 49.78),
    (-41.01, 49.78),
    (-63.62, 10.63),
    (63.62, 10.63),
    (22.61, -60.41),
    (-22.61, -60.41),
]


mesh = trimesh.load(SOURCE, force="mesh")
vertices = mesh.vertices.copy()
original_length = EDGE_TOP_Y - EDGE_BOTTOM_Y
new_length = original_length - SHORTEN_BY
if new_length <= 0:
    raise ValueError("Shortening amount must be less than the edge length")
scale = new_length / original_length
edge_vertex_counts = []

for x, z in EDGE_XZ:
    distance = np.hypot(vertices[:, 0] - x, vertices[:, 2] - z)
    selected = (
        (distance < 0.06)
        & (vertices[:, 1] >= EDGE_BOTTOM_Y - 0.02)
        & (vertices[:, 1] <= EDGE_TOP_Y + 0.02)
    )
    if not selected.any():
        raise RuntimeError(f"No mesh vertices found for support edge at {(x, z)}")

    vertices[selected, 1] = EDGE_BOTTOM_Y + (
        vertices[selected, 1] - EDGE_BOTTOM_Y
    ) * scale
    edge_vertex_counts.append(int(selected.sum()))

mesh.vertices = vertices
mesh.fix_normals()
if not mesh.is_watertight:
    raise RuntimeError("Modified mesh is not watertight")

mesh.export(OUTPUT)
print(f"Saved {OUTPUT}")
print(f"Support edges shortened: {len(EDGE_XZ)} by {SHORTEN_BY:.1f} mm each")
print(f"Length: {original_length:.3f} mm -> {new_length:.3f} mm")
print(f"Vertices adjusted per edge: {edge_vertex_counts}")
print(f"Watertight: {mesh.is_watertight}; bounds: {mesh.extents}")
