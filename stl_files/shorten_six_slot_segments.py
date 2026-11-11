"""Shorten both long boundary edges of each of the six bottom slots by 3 mm."""

from pathlib import Path

import numpy as np
import trimesh


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "Case_Bottom.stl"
OUTPUT = HERE / "Case_Bottom_six_slots_segments_shortened_3mm.stl"
TARGET_LENGTH = 28.4688
LENGTH_TOLERANCE = 0.08
SHORTEN_BY = 3.0


mesh = trimesh.load(SOURCE, force="mesh")
edges = mesh.edges_unique
lengths = mesh.edges_unique_length
selected = np.flatnonzero(np.abs(lengths - TARGET_LENGTH) <= LENGTH_TOLERANCE)
if len(selected) != 12:
    raise RuntimeError(f"Expected 12 slot boundary edges, found {len(selected)}")

vertices = mesh.vertices.copy()
updates = {}
for edge_id in selected:
    i, j = edges[edge_id]
    a, b = vertices[i], vertices[j]
    # These slot boundaries lie in the y=-30.36 mm plane. The larger XZ radius
    # is the outer end; move it inward along the measured edge by 3 mm.
    center = (a + b) / 2.0
    if np.linalg.norm(a[[0, 2]]) > np.linalg.norm(b[[0, 2]]):
        outer_id, inner_id = i, j
    else:
        outer_id, inner_id = j, i
    outer, inner = vertices[outer_id], vertices[inner_id]
    direction = (inner - outer) / lengths[edge_id]
    target = outer + SHORTEN_BY * direction
    previous = updates.get(outer_id)
    if previous is not None and not np.allclose(previous, target, atol=0.08):
        raise RuntimeError("Selected slot edges share a vertex with conflicting moves")
    updates[outer_id] = target

for vertex_id, target in updates.items():
    vertices[vertex_id] = target

mesh.vertices = vertices
mesh.fix_normals()
if not mesh.is_watertight:
    raise RuntimeError("Modified mesh is not watertight")

measured_after = np.linalg.norm(
    mesh.vertices[edges[selected, 0]] - mesh.vertices[edges[selected, 1]], axis=1
)
if not np.allclose(measured_after, lengths[selected] - SHORTEN_BY, atol=1e-6):
    raise RuntimeError("The selected slot boundary edges did not shorten as expected")

mesh.export(OUTPUT)
check = trimesh.load(OUTPUT, force="mesh")

print(f"Saved {OUTPUT}")
print(f"Slot boundary edges shortened: {len(selected)} across six slots")
print(f"Measured lengths: {lengths[selected].min():.3f}-{lengths[selected].max():.3f} mm")
print(f"After edit: {measured_after.min():.3f}-{measured_after.max():.3f} mm")
print(f"Watertight: {check.is_watertight}; bounds: {check.extents}")
