"""Check the rigid-kinematic alias construction, not an AMASS experiment.

This intentionally omits SMPL pose blend shapes, regressed surface landmarks,
rendering, detectors and S-JEPA preprocessing. Those require separate gates.
"""
from pathlib import Path
import json
import numpy as np


def rotation(axis, theta):
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(theta) * cross + (1-np.cos(theta)) * (cross @ cross)


def forward(local, offsets, parents):
    world, positions = [], []
    for i, parent in enumerate(parents):
        if parent == -1:
            world.append(local[i])
            positions.append(offsets[i])
        else:
            world.append(world[parent] @ local[i])
            positions.append(positions[parent] + world[parent] @ offsets[i])
    return np.array(world), np.array(positions)


rng = np.random.default_rng(9407)
parents = [-1, 0, 1, 2, 2]
errors, moved = [], []
for _ in range(100):
    offsets = rng.normal(size=(5, 3)) * .25
    local = np.stack([rotation(rng.normal(size=3), rng.uniform(-1, 1)) for _ in parents])
    joint, child = 1, 2
    q = rotation(offsets[child], rng.uniform(.2, 1.0))
    changed = local.copy()
    changed[joint] = local[joint] @ q
    changed[child] = q.T @ local[child]
    w0, p0 = forward(local, offsets, parents)
    w1, p1 = forward(changed, offsets, parents)
    errors.append(float(np.max(np.abs(p1-p0))))
    # A single material point rigidly attached off-axis to joint 1.
    radial = np.cross(offsets[child], rng.normal(size=3))
    radial *= .08 / np.linalg.norm(radial)
    moved.append(float(np.linalg.norm(w1[joint] @ radial - w0[joint] @ radial)))

result = {
    "kind": "analytical rigid-kinematic construction sanity check",
    "not_an_amass_or_model_result": True,
    "seed": 9407,
    "random_configurations": 100,
    "maximum_absolute_joint_position_change_m": max(errors),
    "median_off_axis_material_point_displacement_m": float(np.median(moved)),
    "minimum_off_axis_material_point_displacement_m": min(moved),
    "limits": ["Not SMPL mesh validation", "Not equality of detector or S-JEPA input tensors", "Not an optical-flow recovery result", "No forecast or clinical claim"]
}
assert max(errors) < 1e-12
assert min(moved) > 0
Path(__file__).with_name("joint-preserving-rotation-check.json").write_text(json.dumps(result, indent=2)+"\n")
print(json.dumps(result, indent=2))
