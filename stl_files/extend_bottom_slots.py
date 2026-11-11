"""Extend the six radial bottom slots of Case_Bottom by 3 mm inward."""

from pathlib import Path

import numpy as np
import trimesh


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "Case_Bottom.stl"
OUTPUT = HERE / "Case_Bottom_slots_inward_3mm.stl"
SECTION_Y = -21.0
SLOT_DEPTH = 10.0
EXTENSION = 3.0


mesh = trimesh.load(SOURCE, force="mesh")
section = mesh.section(
    plane_origin=[0.0, SECTION_Y, 0.0],
    plane_normal=[0.0, 1.0, 0.0],
)
if section is None:
    raise RuntimeError("Could not section the source mesh at the slot level")

# At this plane, the six bottom-slot profiles are the 19-point paths. The
# central bore and outer shell have much longer paths.
contours = [path[:, [0, 2]] for path in section.discrete if len(path) == 19]
if len(contours) != 6:
    raise RuntimeError(f"Expected six slot contours, found {len(contours)}")


def slot_span_at(points, radial, tangent, station):
    projected = np.column_stack((points @ radial, points @ tangent))
    intersections = []
    for first, second in zip(projected, np.roll(projected, -1, axis=0)):
        r0, r1 = first[0], second[0]
        if (r0 <= station < r1) or (r1 <= station < r0):
            ratio = (station - r0) / (r1 - r0)
            intersections.append(first[1] + ratio * (second[1] - first[1]))
    if len(intersections) < 2:
        raise RuntimeError("Could not measure the slot width at its inner end")
    low, high = min(intersections), max(intersections)
    return (low + high) / 2.0, high - low


cutters = []
for points in contours:
    center = points.mean(axis=0)
    radial = center / np.linalg.norm(center)
    tangent = np.array([-radial[1], radial[0]])
    radial_position = points @ radial
    tangent_position = points @ tangent

    inner_end = float(radial_position.min())
    tangent_center, tangent_width = slot_span_at(
        points, radial, tangent, inner_end + 0.5
    )
    tangent_width += 0.15
    cutter_center_xz = radial * (inner_end - EXTENSION / 2.0) + tangent * tangent_center

    cutter = trimesh.creation.box(
        extents=[EXTENSION + 0.2, SLOT_DEPTH, tangent_width]
    )
    radial_3d = np.array([radial[0], 0.0, radial[1]])
    tangent_3d = np.array([tangent[0], 0.0, tangent[1]])
    transform = np.eye(4)
    transform[:3, :3] = np.column_stack(
        (radial_3d, np.array([0.0, 1.0, 0.0]), tangent_3d)
    )
    transform[:3, 3] = [cutter_center_xz[0], SECTION_Y, cutter_center_xz[1]]
    cutter.apply_transform(transform)
    cutters.append(cutter)

result = trimesh.boolean.difference([mesh, *cutters], engine="manifold")
if result is None or len(result.faces) == 0:
    raise RuntimeError("Slot extension boolean produced an empty mesh")
result.remove_unreferenced_vertices()
result.fix_normals()
if not result.is_watertight:
    raise RuntimeError("Extended mesh is not watertight")
if not np.allclose(result.bounds, mesh.bounds, atol=0.1):
    raise RuntimeError("Slot extension unexpectedly changed the outer envelope")

result.export(OUTPUT)
print(f"Saved {OUTPUT}")
print(
    f"Extended slots: {len(cutters)}; inward extension: {EXTENSION:.1f} mm; "
    "original slot widths retained"
)
print(f"Watertight: {result.is_watertight}; bounds: {result.extents}")
