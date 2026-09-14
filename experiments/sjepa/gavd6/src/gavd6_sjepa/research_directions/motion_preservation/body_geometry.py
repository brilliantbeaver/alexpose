"""Whole-body SMPL-H geometry, angle interventions and observable descriptors."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .motion_data import MotionParameters


JOINT_NAMES = ("pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
               "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
               "neck", "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
               "left_elbow", "right_elbow", "left_wrist", "right_wrist")
PARENTS = np.array([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19])
# Proper rotation: [x,y,z]_AMASS -> [x,z,-y]. Both mesh and joints use it once.
AMASS_TO_Y_UP = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float32)
EVENT_FAMILIES = ("arm_leg_timing", "foot_clearance", "trunk_pelvis_timing")


@dataclass
class BodySequence:
    joints: np.ndarray
    vertices: np.ndarray
    faces: np.ndarray
    fps: float
    timestamps: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)
    coordinate_system: str = "y_up"


class SMPLHBody:
    """Licensed AMASS SMPL-H + DMPL body model, frozen during all experiments.

    ``body_model_root`` contains ``smplh/{gender}/model.npz`` (or is the smplh
    directory itself). ``dmpl_root`` contains ``{gender}/model.npz`` and defaults
    to the sibling ``dmpls`` folder. Missing assets raise an error.
    """
    def __init__(self, body_model_root, dmpl_root=None, device="cpu", batch_size=128):
        self.root = Path(body_model_root).expanduser()
        self.body_root = self.root / "smplh" if (self.root / "smplh").is_dir() else self.root
        self.dmpl_root = Path(dmpl_root).expanduser() if dmpl_root else (
            self.root / "dmpls" if self.body_root != self.root else self.root.parent / "dmpls")
        self.device, self.batch_size, self.models = device, int(batch_size), {}

    def _model(self, gender):
        import inspect
        import torch
        from human_body_prior.body_model.body_model import BodyModel
        if gender not in self.models:
            body_path, dmpl_path = self.body_root / gender / "model.npz", self.dmpl_root / gender / "model.npz"
            for path in (body_path, dmpl_path):
                if not path.is_file():
                    raise FileNotFoundError(f"Licensed AMASS body-model asset required: {path}")
            if not {"bm_fname", "dmpl_fname"} <= set(inspect.signature(BodyModel).parameters):
                raise RuntimeError(
                    "This study requires the official AMASS human_body_prior BodyModel "
                    "with bm_fname/dmpl_fname and dynamic shape support. The older PyPI "
                    "API does not apply DMPL coefficients. Install the GitHub version: "
                    "python -m pip install --upgrade git+https://github.com/nghorbani/human_body_prior.git"
                )
            model = BodyModel(bm_fname=str(body_path), num_betas=16,
                              dmpl_fname=str(dmpl_path), num_dmpls=8).to(self.device).eval()
            if (model.model_type != "smplh" or model.shapedirs.shape[-1] != 16
                    or not getattr(model, "use_dmpl", False) or model.dmpldirs.shape[-1] != 8):
                raise ValueError("Expected an SMPL-H model with 16 shape and 8 active DMPL components")
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            self.models[gender] = model
        return self.models[gender]

    def forward(self, motion: MotionParameters) -> BodySequence:
        import torch
        model = self._model(motion.gender)
        joints, vertices = [], []
        for start in range(0, len(motion.poses), self.batch_size):
            stop = min(start + self.batch_size, len(motion.poses))
            tensor = lambda a: torch.as_tensor(a, dtype=torch.float32, device=self.device)
            pose = tensor(motion.poses[start:stop])
            with torch.inference_mode():
                output = model(root_orient=pose[:, :3], pose_body=pose[:, 3:66],
                               pose_hand=pose[:, 66:156], trans=tensor(motion.trans[start:stop]),
                               betas=tensor(np.repeat(motion.betas[None], stop - start, axis=0)),
                               dmpls=tensor(motion.dmpls[start:stop]))
            joints.append(output.Jtr[:, :22].detach().cpu().numpy())
            vertices.append(output.v.detach().cpu().numpy())
        faces = model.f.detach().cpu().numpy() if torch.is_tensor(model.f) else np.asarray(model.f)
        transform = lambda a: (np.concatenate(a) @ AMASS_TO_Y_UP.T).astype(np.float32)
        joints, vertices = transform(joints), transform(vertices)
        if not np.isfinite(joints).all() or not np.isfinite(vertices).all():
            raise ValueError("SMPL-H returned nonfinite geometry; this motion cannot supply reference truth")
        return BodySequence(joints, vertices, faces.astype(np.int32), motion.fps,
                            motion.timestamps.copy(), {**motion.metadata, "geometry": "SMPL-H+DMPL",
                                                       "world_units": "meters", "world_up": "+Y"})


def _envelope(frames: int, center=0.5, half_width=0.38):
    distance = np.abs(np.linspace(0, 1, frames) - center) / half_width
    return np.where(distance < 1, 0.5 + 0.5 * np.cos(np.pi * np.minimum(distance, 1)), 0)


def edit_motion(motion: MotionParameters, family: str, amplitude: float = 1.0,
                phase_shift_s: float = 0.18) -> MotionParameters:
    """Change local angles, retaining root trajectory and all body parameters.

    Timing edits interpolate rotations shifted in time inside this same window.
    Foot clearance applies a smooth hip/knee/ankle flexion combination. Its actual
    world-space effect must pass the descriptor and geometry screens; the name
    does not guarantee that every source pose yields a clearance increase.
    """
    if family not in EVENT_FAMILIES:
        raise ValueError(f"Unknown event family: {family}")
    poses = motion.poses.copy().reshape(len(motion.poses), 52, 3)
    envelope = _envelope(len(poses))
    if family == "foot_clearance":
        # Local sagittal flexion, with counter-rotation at the hip and ankle.
        indices, coefficients = (1, 4, 7), (-0.35, 0.70, -0.35)
        for joint, angle in zip(indices, coefficients):
            added = np.zeros((len(poses), 3))
            added[:, 0] = amplitude * angle * envelope
            poses[:, joint] = (Rotation.from_rotvec(poses[:, joint]) * Rotation.from_rotvec(added)).as_rotvec()
        affected = [4, 7, 10]
    else:
        indices = (13, 14, 16, 17, 18, 19, 20, 21) if family == "arm_leg_timing" else (3, 6, 9)
        affected = [16, 17, 18, 19, 20, 21] if family == "arm_leg_timing" else [6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
        times = motion.timestamps
        query = np.clip(times - phase_shift_s, times[0], times[-1])
        for joint in indices:
            original = Rotation.from_rotvec(poses[:, joint])
            shifted = Slerp(times, original)(query)
            difference = (original.inv() * shifted).as_rotvec()
            poses[:, joint] = (original * Rotation.from_rotvec(difference * (amplitude * envelope[:, None]))).as_rotvec()
    metadata = {**motion.metadata, "event_family": family, "event_amplitude_setting": float(amplitude),
                "phase_shift_s": float(phase_shift_s), "event_joints": affected,
                "event_frame_support": np.flatnonzero(envelope > 0).tolist()}
    return replace(motion, poses=poses.reshape(-1, 156), metadata=metadata)


def _timing_index(first, second):
    first = first - np.mean(first)
    velocity = np.gradient(second)
    velocity -= np.mean(velocity)
    denominator = np.sqrt(np.mean(first**2) * np.mean(velocity**2))
    return float(np.mean(first * velocity) / max(denominator, 1e-8))


def descriptor(joints, family: str) -> float:
    """Prespecified scalar observable; use paired edited-minus-clean differences.

    Timing indices are signed, normalized position/velocity couplings, not clinical
    diagnoses or estimated lag in seconds. Degenerate/weak event deltas are
    screened before fitting. Foot clearance is measured in world meters.
    """
    x = np.asarray(joints)
    if x.ndim != 3 or x.shape[1:] != (22, 3) or len(x) < 3:
        raise ValueError("Descriptors require [T>=3,22,3] whole-body positions")
    if family == "foot_clearance":
        n = max(1, len(x) // 8)
        height = x[:, 10, 1]
        return float(np.max(height[n:-n]) - np.mean(np.r_[height[:n], height[-n:]]))
    if family == "arm_leg_timing":
        lateral = x[:, 2] - x[:, 1]
        lateral[:, 1] = 0
        lateral /= np.maximum(np.linalg.norm(lateral, axis=1, keepdims=True), 1e-8)
        forward = np.cross(lateral, [0, 1, 0])
        arm = np.sum((x[:, 20] - x[:, 16]) * forward, axis=1)
        leg = np.sum((x[:, 8] - x[:, 2]) * forward, axis=1)
        return _timing_index(arm, leg)
    if family == "trunk_pelvis_timing":
        shoulder, hip = x[:, 17] - x[:, 16], x[:, 2] - x[:, 1]
        upper = np.unwrap(np.arctan2(shoulder[:, 2], shoulder[:, 0]))
        lower = np.unwrap(np.arctan2(hip[:, 2], hip[:, 0]))
        return _timing_index(upper, lower)
    raise ValueError(f"Unknown descriptor family: {family}")


def estimate_bone_lengths(raw, observed=None):
    """Median lengths from permitted observations; never takes hidden truth."""
    x = np.asarray(raw)
    mask = np.isfinite(x).all(axis=-1) if observed is None else np.asarray(observed, bool) & np.isfinite(x).all(axis=-1)
    lengths = np.zeros(22, dtype=np.float32)
    for j in range(1, 22):
        valid = mask[:, j] & mask[:, PARENTS[j]]
        values = np.linalg.norm(x[valid, j] - x[valid, PARENTS[j]], axis=-1)
        values = values[values > 1e-6]
        if not len(values):
            raise ValueError(f"No observed length for {JOINT_NAMES[j]}; supply a training-only template")
        lengths[j] = np.median(values)
    return lengths


def project_bone_lengths(joints, bone_lengths):
    """Keep the predicted root and directions, enforcing externally chosen lengths."""
    x = np.asarray(joints, dtype=np.float32)
    lengths = np.asarray(bone_lengths)
    if x.shape[1:] != (22, 3) or lengths.shape != (22,):
        raise ValueError("Expected [T,22,3] positions and 22 bone lengths")
    result = x.copy()
    for j in range(1, 22):
        direction = x[:, j] - x[:, PARENTS[j]]
        norm = np.linalg.norm(direction, axis=-1, keepdims=True)
        zero = norm[:, 0] < 1e-8
        direction /= np.maximum(norm, 1e-8)
        direction[zero] = [0, -1, 0]
        result[:, j] = result[:, PARENTS[j]] + lengths[j] * direction
    return result


def _demo_body(motion: MotionParameters) -> BodySequence:
    """Small articulated tube mesh, explicitly not AMASS or a learned prior."""
    rest = np.array([[0,0,0], [.09,-.06,0], [-.09,-.06,0], [0,.12,0], [0,-.40,0], [0,-.40,0],
                     [0,.12,0], [0,-.39,0], [0,-.39,0], [0,.14,0], [0,-.03,.15], [0,-.03,.15],
                     [0,.13,0], [.06,.06,0], [-.06,.06,0], [0,.14,0], [.13,0,0], [-.13,0,0],
                     [.25,0,0], [-.25,0,0], [.24,0,0], [-.24,0,0]], dtype=np.float32)
    rotations = Rotation.from_rotvec(motion.poses[:, :66].reshape(-1, 3)).as_matrix().reshape(-1, 22, 3, 3)
    positions, globals_ = np.zeros((len(rotations),22,3)), np.empty_like(rotations)
    positions[:,0], globals_[:,0] = motion.trans, rotations[:,0]
    for j in range(1,22):
        p = PARENTS[j]
        positions[:,j] = positions[:,p] + np.einsum("tij,j->ti", globals_[:,p],rest[j])
        globals_[:,j] = globals_[:,p] @ rotations[:,j]
    vertices, faces = [], []
    sides = 8
    for j in range(1,22):
        direction = rest[j] / np.linalg.norm(rest[j])
        helper = np.array([0,0,1]) if abs(direction[2]) < .8 else np.array([1,0,0])
        first = np.cross(direction,helper); first /= np.linalg.norm(first)
        second = np.cross(direction,first)
        theta = np.arange(sides) * 2 * np.pi / sides
        radial = .035 * (np.cos(theta)[:,None]*first + np.sin(theta)[:,None]*second)
        base = len(vertices) * sides * 2
        local = np.concatenate([radial,rest[j]+radial])
        world = positions[:,PARENTS[j],None] + np.einsum("tij,vj->tvi",globals_[:,PARENTS[j]],local)
        vertices.append(world)
        for k in range(sides):
            a,b = base+k,base+(k+1)%sides
            faces.extend([[a,b,a+sides],[b,b+sides,a+sides]])
    return BodySequence(positions.astype(np.float32),np.concatenate(vertices,axis=1).astype(np.float32),
                        np.asarray(faces,np.int32),motion.fps,motion.timestamps.copy(),
                        {**motion.metadata,"evidence_origin":"procedural_demo","geometry":"articulated_tubes"})


def demo_motion(frames=32, fps=20, family="arm_leg_timing"):
    """Return clean and angle-edited DEMO mesh sequences without licensed assets."""
    t = np.arange(frames) / fps
    poses = np.zeros((frames,52,3),np.float32)
    phase = 2*np.pi*1.2*t
    poses[:,1,0],poses[:,2,0] = .35*np.sin(phase),-.35*np.sin(phase)
    poses[:,4,0],poses[:,5,0] = .35+.3*np.sin(phase),.35-.3*np.sin(phase)
    poses[:,16,0],poses[:,17,0] = -.6*np.sin(phase),.6*np.sin(phase)
    poses[:,16,2],poses[:,17,2] = -1.1,1.1
    poses[:,18,1],poses[:,19,1] = .3+.15*np.sin(phase),-.3+.15*np.sin(phase)
    poses[:,0,1],poses[:,3,1] = .1*np.sin(phase),.13*np.sin(phase+.6)
    trans = np.column_stack([np.zeros(frames),np.full(frames,.92),.15*t]).astype(np.float32)
    parameters = MotionParameters(poses.reshape(frames,156),trans,np.zeros(16,np.float32),
                                  np.zeros((frames,8),np.float32),"neutral",fps,t,
                                  {"evidence_origin":"procedural_demo","person_id":"demo_person"})
    return _demo_body(parameters), _demo_body(edit_motion(parameters,family))
