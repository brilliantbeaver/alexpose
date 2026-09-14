"""Frozen author-model bridges and image-motion measurements for Proposal 01.

The wrappers load local, explicitly supplied weights. Nothing is downloaded by
these functions. MoMask calls the author's RVQVAE and HumanML3D routines; RAFT
outputs are pixel displacements, never probabilities of a movement being real.

Author interfaces inspected in September 2026:
https://github.com/EricGuo5513/momask-codes
https://github.com/princeton-vl/SEA-RAFT
https://pytorch.org/vision/stable/models/optical_flow.html
"""

from __future__ import annotations

import ast
import builtins
from contextlib import contextmanager
from dataclasses import dataclass, field
import importlib
import json
from pathlib import Path
import sys
import threading
from types import ModuleType, SimpleNamespace
from typing import Any, Iterator
import warnings

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class PriorResult:
    joints: np.ndarray
    frame_indices: np.ndarray
    input_features: np.ndarray | None = None
    reconstruction_features: np.ndarray | None = None
    bridge_roundtrip_joints: np.ndarray | None = None
    bridge_roundtrip_error_m: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FlowResult:
    """Arrays use [frame pair, image y, image x, (dx,dy)]."""

    forward: np.ndarray
    backward: np.ndarray | None = None
    uncertainty: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


_IMPORT_LOCK = threading.RLock()


class _AuthorRepository:
    """Keep legacy author imports (``utils``, ``models``) out of the notebook.

    A short import context saves and restores every conflicting module. Loaded
    author modules are retained privately between calls, including lazy imports.
    Use the wrappers from the notebook's main thread, not concurrent importers.
    """

    def __init__(self, search_path: str | Path):
        self.path = Path(search_path).expanduser().resolve()
        if not self.path.is_dir():
            raise FileNotFoundError(f"Author repository directory not found: {self.path}")
        self.roots = {
            p.stem if p.is_file() else p.name
            for p in self.path.iterdir()
            if (p.is_file() and p.suffix == ".py") or
            (p.is_dir() and not p.name.startswith("."))
        }
        self.modules: dict[str, ModuleType] = {}

    def _owns(self, name: str) -> bool:
        return name.split(".", 1)[0] in self.roots

    @contextmanager
    def activate(self) -> Iterator[None]:
        with _IMPORT_LOCK:
            saved = {k: v for k, v in sys.modules.copy().items() if self._owns(k)}
            old_path = list(sys.path)
            for name in saved:
                sys.modules.pop(name, None)
            sys.modules.update(self.modules)
            sys.path.insert(0, str(self.path))
            try:
                yield
            finally:
                self.modules = {
                    k: v for k, v in sys.modules.copy().items() if self._owns(k)
                }
                for name in self.modules:
                    sys.modules.pop(name, None)
                sys.modules.update(saved)
                sys.path[:] = old_path


class _LegacyNumpy:
    """Compatibility only inside the old motion module, without patching NumPy."""

    float = float

    def __getattr__(self, name: str) -> Any:
        return getattr(np, name)


def _load_legacy_numpy_module(name: str) -> ModuleType:
    """Load one old source module with a local ``np.float`` compatibility view."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.find_spec(name)
    if spec is None or spec.loader is None:
        raise ImportError(name)
    module = importlib.util.module_from_spec(spec)
    normal_import = builtins.__import__

    def local_import(import_name, globals=None, locals=None, fromlist=(), level=0):
        if import_name == "numpy" and level == 0:
            return _LegacyNumpy()
        return normal_import(import_name, globals, locals, fromlist, level)

    module.__dict__["__builtins__"] = {**vars(builtins), "__import__": local_import}
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _options(path: Path) -> SimpleNamespace:
    result: dict[str, Any] = {}
    for line in path.read_text().splitlines():
        if ": " not in line:
            continue
        key, value = line.strip().split(": ", 1)
        try:
            result[key] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            result[key] = value
    result.update(dim_pose=263, joints_num=22, is_train=False)
    return SimpleNamespace(**result)


def _world_to_y_up(joints: np.ndarray, coordinate_system: str) -> tuple[np.ndarray, np.ndarray]:
    if coordinate_system in {"y_up", "world_y_up"}:
        rotation = np.eye(3)
    elif coordinate_system == "amass_z_up":
        # Proper rotation, not a reflection: [x,y,z] -> [x,z,-y].
        rotation = np.array([[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]])
    else:
        raise ValueError("coordinate_system must be y_up, world_y_up, or amass_z_up")
    return joints @ rotation.T, rotation


@dataclass
class HumanMLBridgeState:
    """Parameters derived only from the supplied observed clip, not its truth."""

    axis_rotation: np.ndarray
    root_xz: np.ndarray
    floor_height: float
    initial_heading: np.ndarray
    root_scale: float
    source_bone_lengths: np.ndarray
    parents: np.ndarray


class HumanML3DBridge:
    """Author 22-joint/263-feature conversion with an explicit inverse.

    HumanML3D retargets bone lengths before feature extraction. The inverse
    restores observed first-frame bone lengths and world coordinates. That
    retargeting can remove length noise, so ``roundtrip`` is itself a baseline.
    Its error must not be attributed to the learned prior.

    Extraction is offline: the author's smoothed heading, floor, velocities and
    contacts can use the whole supplied clip. Pass only an observed prefix when
    using this bridge for a forecasting experiment.
    """

    def __init__(self, repo_dir: str | Path, target_skeleton_path: str | Path | None = None):
        self.repository = _AuthorRepository(repo_dir)
        with self.repository.activate():
            _load_legacy_numpy_module("common.quaternion")
            self.motion = importlib.import_module("utils.motion_process")
            self.params = importlib.import_module("utils.paramUtil")
            self.skeleton_class = importlib.import_module("common.skeleton").Skeleton
        self.motion.np = _LegacyNumpy()
        self.raw_offsets = torch.as_tensor(self.params.t2m_raw_offsets, dtype=torch.float32)
        self.chains = self.params.t2m_kinematic_chain
        self.face = [2, 1, 17, 16]
        self.parents = np.asarray(self._skeleton().parents(), dtype=int)
        target_path = Path(target_skeleton_path).expanduser() if target_skeleton_path else (
            self.repository.path / "example_data" / "000612.npy"
        )
        if not target_path.is_file():
            raise FileNotFoundError(
                f"HumanML3D target skeleton missing: {target_path}. Supply a canonical "
                "HumanML3D new_joints .npy or 263-channel example, not raw AMASS."
            )
        target = np.load(target_path, allow_pickle=False)
        if target.ndim == 2 and target.shape[-1] == 263:
            target = self.motion.recover_from_ric(torch.as_tensor(target).float(), 22).numpy()
        if target.ndim == 3:
            target = target[0]
        if target.shape != (22, 3) or not np.isfinite(target).all():
            raise ValueError("Target skeleton must be [22,3], [T,22,3], or [T,263]")
        self.target_offsets = self._skeleton().get_offsets_joints(torch.as_tensor(target).float())
        self.target_path = str(target_path.resolve())

    def _skeleton(self):
        return self.skeleton_class(self.raw_offsets, self.chains, "cpu")

    def encode(self, joints: np.ndarray, coordinate_system: str = "world_y_up") -> tuple[np.ndarray, HumanMLBridgeState]:
        raw = np.asarray(joints, dtype=np.float64)
        if raw.ndim != 3 or raw.shape[1:] != (22, 3) or len(raw) < 3:
            raise ValueError("MoMask needs at least 3 frames of [T,22,3] joints at 20 Hz")
        if not np.isfinite(raw).all():
            raise ValueError("Impute missing observed joints before the prior; preserve the original missingness mask")
        positions, axes = _world_to_y_up(raw, coordinate_system)
        source_offsets = self._skeleton().get_offsets_joints(torch.as_tensor(positions[0]).float()).numpy()
        source_lengths = np.linalg.norm(source_offsets, axis=-1)
        if np.any(source_lengths[1:] < 1e-6):
            raise ValueError("Degenerate first-frame bone in observed skeleton")
        target_offsets = self.target_offsets.numpy()
        scale = float((np.abs(target_offsets[5]).max() + np.abs(target_offsets[8]).max()) /
                      (np.abs(source_offsets[5]).max() + np.abs(source_offsets[8]).max()))
        # Use the author's exact retargeting and feature extraction routines.
        m = self.motion
        m.n_raw_offsets, m.kinematic_chain = self.raw_offsets, self.chains
        m.l_idx1, m.l_idx2, m.face_joint_indx = 5, 8, self.face
        with self.repository.activate():
            canonical = m.uniform_skeleton(positions.copy(), self.target_offsets)
            floor = float(canonical[..., 1].min())
            canonical[..., 1] -= floor
            root_xz = canonical[0, 0] * np.array([1., 0., 1.])
            canonical -= root_xz
            across = (canonical[0, 2] - canonical[0, 1] +
                      canonical[0, 17] - canonical[0, 16])
            forward = np.cross(np.array([0., 1., 0.]), across)
            if np.linalg.norm(forward) < 1e-8:
                raise ValueError("Initial hip/shoulder configuration has no identifiable heading")
            forward /= np.linalg.norm(forward)
            heading = m.qbetween_np(forward[None], np.array([[0., 0., 1.]]))[0]
            heading_all = np.broadcast_to(heading, canonical.shape[:-1] + (4,)).copy()
            canonical = m.qrot_np(heading_all, canonical)
            features = m.extract_features(canonical.copy(), 0.002, self.raw_offsets,
                                          self.chains, self.face, [8, 11], [7, 10])
        if features.shape != (len(raw) - 1, 263) or not np.isfinite(features).all():
            raise ValueError("HumanML3D conversion failed; inspect degenerate or extreme input poses")
        state = HumanMLBridgeState(axes, root_xz, floor, heading, scale, source_lengths, self.parents)
        return features.astype(np.float32), state

    def decode(self, features: np.ndarray, state: HumanMLBridgeState) -> np.ndarray:
        with self.repository.activate():
            canonical = self.motion.recover_from_ric(torch.as_tensor(features).float(), 22).numpy()
            inverse = self.motion.qinv_np(state.initial_heading[None])[0]
            inverse_all = np.broadcast_to(inverse, canonical.shape[:-1] + (4,)).copy()
            target_world = self.motion.qrot_np(inverse_all, canonical)
        target_world += state.root_xz
        target_world[..., 1] += state.floor_height
        # Inverse bone retargeting preserves decoded directions and observed
        # source lengths. A rigid scalar alone cannot invert differing shapes.
        restored = np.zeros_like(target_world)
        restored[:, 0] = target_world[:, 0] / state.root_scale
        for joint in range(1, 22):
            parent = state.parents[joint]
            vector = target_world[:, joint] - target_world[:, parent]
            length = np.linalg.norm(vector, axis=-1, keepdims=True)
            if np.any(length < 1e-8):
                raise ValueError("Decoded prior has a degenerate bone; cannot invert retargeting")
            restored[:, joint] = restored[:, parent] + vector / length * state.source_bone_lengths[joint]
        return (restored @ state.axis_rotation).astype(np.float32)

    def roundtrip(self, joints: np.ndarray, coordinate_system: str = "world_y_up") -> PriorResult:
        features, state = self.encode(joints, coordinate_system)
        recovered = self.decode(features, state)
        error = float(np.linalg.norm(recovered - np.asarray(joints)[:-1], axis=-1).mean())
        return PriorResult(recovered, np.arange(len(features)), features, features,
                           recovered, error, {"model": "humanml3d_conversion_only", "pretrained": False})


class MoMaskPrior:
    """Frozen author RVQ-VAE reconstruction, not text-to-motion generation."""

    def __init__(self, repo_dir: str | Path, checkpoint_dir: str | Path,
                 target_skeleton_path: str | Path | None = None, device: str = "cuda",
                 checkpoint_name: str = "net_best_fid.tar"):
        root = Path(checkpoint_dir).expanduser()
        checkpoint = root / "model" / checkpoint_name
        if not checkpoint.is_file():
            raise FileNotFoundError(f"MoMask RVQ checkpoint not found: {checkpoint}")
        self.device = torch.device(device)
        self.bridge = HumanML3DBridge(repo_dir, target_skeleton_path)
        self.repository = self.bridge.repository
        opt = _options(root / "opt.txt")
        opt.device = self.device
        if getattr(opt, "dataset_name", "t2m") != "t2m":
            raise ValueError("Use the HumanML3D t2m checkpoint, not the KIT model")
        self.temporal_multiple = int(opt.stride_t ** opt.down_t)
        self.mean = np.load(root / "meta" / "mean.npy", allow_pickle=False).astype(np.float32)
        self.std = np.load(root / "meta" / "std.npy", allow_pickle=False).astype(np.float32)
        if (self.mean.shape != (263,) or self.std.shape != (263,) or np.any(self.std <= 0)
                or not np.isfinite(self.mean).all() or not np.isfinite(self.std).all()):
            raise ValueError("MoMask normalization must contain 263 finite, positive standard deviations")
        with self.repository.activate():
            model_class = importlib.import_module("models.vq.model").RVQVAE
            self.model = model_class(opt, 263, opt.nb_code, opt.code_dim, opt.output_emb_width,
                                    opt.down_t, opt.stride_t, opt.width, opt.depth,
                                    opt.dilation_growth_rate, opt.vq_act, opt.vq_norm)
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            key = "vq_model" if "vq_model" in saved else "net"
            self.model.load_state_dict(saved[key], strict=True)
        self.model.to(self.device).eval().requires_grad_(False)
        self.checkpoint = str(checkpoint.resolve())

    def reconstruct(self, joints: np.ndarray, coordinate_system: str = "world_y_up") -> PriorResult:
        features, bridge_state = self.bridge.encode(joints, coordinate_system)
        roundtrip = self.bridge.decode(features, bridge_state)
        normalized = (features - self.mean) / self.std
        padding = (-len(normalized)) % self.temporal_multiple
        if padding:
            normalized = np.pad(normalized, ((0, padding), (0, 0)), mode="edge")
        with self.repository.activate(), torch.inference_mode():
            tensor = torch.from_numpy(normalized[None]).to(self.device)
            indices, _ = self.model.encode(tensor)
            decoded = self.model.forward_decoder(indices)
            if decoded.ndim != 3 or decoded.shape[-1] != 263:
                raise ValueError(f"Unexpected author RVQ decoder shape: {tuple(decoded.shape)}")
            reconstructed = decoded[0, :len(features)].float().cpu().numpy() * self.std + self.mean
        output = self.bridge.decode(reconstructed, bridge_state)
        error = float(np.linalg.norm(roundtrip - np.asarray(joints)[:-1], axis=-1).mean())
        return PriorResult(output, np.arange(len(features)), features, reconstructed, roundtrip, error, {
            "model": "momask_rvqvae", "pretrained": True, "checkpoint": self.checkpoint,
            "coordinate_system": coordinate_system, "fps": 20, "context": "whole_supplied_clip",
            "target_skeleton": self.bridge.target_path, "padded_feature_frames": padding,
            "last_input_frame": "used for final velocity, not returned as reconstructed pose",
        })


def load_external_prior(path: str | Path) -> PriorResult:
    """Load an independently run second prior, for example an MDM restoration.

    Required NPZ fields: joints [K,22,3], frame_indices [K], metadata_json (a
    JSON string recording model, checkpoint, coordinate_system and context).
    This is an interchange format, not an implementation of MDM reconstruction.
    Choose the restoration procedure and strength on development data separately.
    """
    with np.load(path, allow_pickle=False) as data:
        joints = data["joints"].astype(np.float32)
        frames = np.asarray(data["frame_indices"])
        metadata = json.loads(str(data["metadata_json"].item()))
    if (frames.ndim != 1 or not np.issubdtype(frames.dtype, np.number)
            or not np.isfinite(frames).all() or np.any(frames != np.floor(frames))):
        raise ValueError("External prior frame indices must be a one-dimensional array of integers")
    frames = frames.astype(int)
    if joints.shape != (len(frames), 22, 3) or not np.isfinite(joints).all():
        raise ValueError("External prior needs finite joints [K,22,3] and aligned frame indices")
    if np.any(np.diff(frames) <= 0) or np.any(frames < 0):
        raise ValueError("External prior frame indices must be strictly increasing and nonnegative")
    required = {"model", "checkpoint", "coordinate_system", "context"}
    if not required.issubset(metadata):
        raise ValueError(f"External prior metadata is missing {sorted(required - metadata.keys())}")
    metadata["integration"] = "external_cached_predictions"
    return PriorResult(joints, frames, metadata=metadata)


class OpticalFlowEstimator:
    """Frozen SEA-RAFT or torchvision RAFT with explicitly local weights.

    SEA-RAFT accepts the author's ``model.safetensors`` and a matching JSON
    architecture config, normally ``config/eval/spring-M.json``. Torchvision
    accepts its ``raft_small_C_T_V2-01064c6d.pth`` state dictionary by default.
    ``scale`` is a power of two applied to inference images, as in SEA-RAFT's
    demo. Returned vectors and uncertainty scales are restored to input pixels.
    """

    def __init__(self, backend: str, checkpoint_path: str | Path, *,
                 repo_dir: str | Path | None = None, config_path: str | Path | None = None,
                 device: str = "cuda", iterations: int | None = None,
                 scale: float = 0, raft_size: str = "small", pair_batch_size: int = 2):
        checkpoint = Path(checkpoint_path).expanduser()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Local optical-flow checkpoint not found: {checkpoint}")
        self.device, self.backend = torch.device(device), backend
        self.scale, self.batch_size = float(scale), int(pair_batch_size)
        if self.batch_size < 1:
            raise ValueError("pair_batch_size must be positive")
        self.repository = None
        if backend == "sea_raft":
            if repo_dir is None or config_path is None:
                raise ValueError("SEA-RAFT requires its repository and matching config_path")
            self.repository = _AuthorRepository(Path(repo_dir) / "core")
            config = json.loads(Path(config_path).expanduser().read_text())
            self.args = SimpleNamespace(**config.get("args", config))
            self.iterations = int(iterations if iterations is not None else self.args.iters)
            with self.repository.activate():
                raft_module = importlib.import_module("raft")
                extractor = importlib.import_module("extractor")
                # The author constructor would download an ImageNet ResNet that
                # is immediately overwritten by the full flow checkpoint.
                initializer = extractor.ResNetFPN._init_weights
                extractor.ResNetFPN._init_weights = lambda instance, args: None
                try:
                    self.model = raft_module.RAFT(self.args)
                finally:
                    extractor.ResNetFPN._init_weights = initializer
        elif backend == "torchvision_raft":
            from torchvision.models.optical_flow import raft_large, raft_small
            if raft_size not in {"small", "large"}:
                raise ValueError("raft_size must be small or large")
            self.model = (raft_small if raft_size == "small" else raft_large)(weights=None)
            self.iterations = int(iterations if iterations is not None else 12)
        else:
            raise ValueError("backend must be sea_raft or torchvision_raft")
        if checkpoint.suffix == ".safetensors":
            from safetensors.torch import load_file
            weights = load_file(str(checkpoint), device="cpu")
        else:
            weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
        weights = weights.get("state_dict", weights)
        weights = {k.removeprefix("module."): v for k, v in weights.items()}
        self.model.load_state_dict(weights, strict=True)
        self.model.to(self.device).eval().requires_grad_(False)
        self.checkpoint = str(checkpoint.resolve())

    @contextmanager
    def _context(self):
        if self.repository is None:
            yield
        else:
            with self.repository.activate():
                yield

    def _pairs(self, first: torch.Tensor, second: torch.Tensor) -> tuple[np.ndarray, np.ndarray | None]:
        height, width = first.shape[-2:]
        target_h = max(128, int(round(height * 2 ** self.scale)))
        target_w = max(128, int(round(width * 2 ** self.scale)))
        if self.backend == "torchvision_raft":
            target_h, target_w = ((target_h + 7) // 8) * 8, ((target_w + 7) // 8) * 8
        first = F.interpolate(first.to(self.device).float(), (target_h, target_w), mode="bilinear", align_corners=False)
        second = F.interpolate(second.to(self.device).float(), (target_h, target_w), mode="bilinear", align_corners=False)
        with self._context(), torch.inference_mode():
            if self.backend == "sea_raft":
                result = self.model(first, second, iters=self.iterations, test_mode=True)
                flow = result["flow"][-1]
                info = result["info"][-1]
                logits, scales = info[:, :2], info[:, 2:]
                if self.args.use_var:
                    large = scales[:, 0].clamp(0, self.args.var_max)
                    small = scales[:, 1].clamp(self.args.var_min, 0)
                    log_scales = torch.stack((large, small), dim=1)
                else:
                    log_scales = torch.zeros_like(scales)
                # Mixture expected Laplace scale, not calibrated correctness.
                uncertainty = (logits.softmax(1) * log_scales.exp()).sum(1, keepdim=True)
            else:
                flow = self.model(first / 127.5 - 1, second / 127.5 - 1,
                                  num_flow_updates=self.iterations)[-1]
                uncertainty = None
            flow = F.interpolate(flow, (height, width), mode="bilinear", align_corners=False)
            flow[:, 0] *= width / target_w
            flow[:, 1] *= height / target_h
            if uncertainty is not None:
                uncertainty = F.interpolate(uncertainty, (height, width), mode="bilinear", align_corners=False)
                # Retain two coordinate scales if resizing was anisotropic.
                uncertainty = torch.cat((uncertainty * width / target_w,
                                         uncertainty * height / target_h), dim=1)
                uncertainty_array = uncertainty.permute(0, 2, 3, 1).cpu().numpy()
            else:
                uncertainty_array = None
        return flow.permute(0, 2, 3, 1).cpu().numpy(), uncertainty_array

    def estimate(self, rgb: np.ndarray, *, bidirectional: bool = True) -> FlowResult:
        images = np.asarray(rgb)
        if images.ndim != 4 or images.shape[-1] != 3 or len(images) < 2:
            raise ValueError("RGB frames must have shape [T,H,W,3], T >= 2")
        if images.dtype != np.uint8:
            raise ValueError("Pass uint8 RGB [0,255], not normalized or BGR images")
        tensor = torch.from_numpy(np.ascontiguousarray(images)).permute(0, 3, 1, 2)
        forward, backward, uncertainty = [], [], []
        for start in range(0, len(images) - 1, self.batch_size):
            end = min(start + self.batch_size, len(images) - 1)
            flow, diagnostic = self._pairs(tensor[start:end], tensor[start + 1:end + 1])
            forward.append(flow)
            if diagnostic is not None:
                uncertainty.append(diagnostic)
            if bidirectional:
                reverse, _ = self._pairs(tensor[start + 1:end + 1], tensor[start:end])
                backward.append(reverse)
        return FlowResult(np.concatenate(forward), np.concatenate(backward) if backward else None,
                          np.concatenate(uncertainty) if uncertainty else None, {
                              "backend": self.backend, "checkpoint": self.checkpoint,
                              "units": "input_image_pixels", "frame_pairs": "t_to_t_plus_1",
                              "uncertainty": "mixture_expected_laplace_scale_not_probability" if uncertainty else "unavailable",
                              "inference_scale_power_two": self.scale,
                          })


def bilinear_sample(field: np.ndarray, points_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sample an HWC field, returning NaN outside its pixel-center domain."""
    values = np.asarray(field)
    if values.ndim == 2:
        values = values[..., None]
    points = np.asarray(points_xy, dtype=float)
    height, width = values.shape[:2]
    valid = np.isfinite(points).all(-1)
    valid &= (points[..., 0] >= 0) & (points[..., 0] <= width - 1)
    valid &= (points[..., 1] >= 0) & (points[..., 1] <= height - 1)
    safe = np.where(valid[..., None], points, 0)
    x0, y0 = np.floor(safe[..., 0]).astype(int), np.floor(safe[..., 1]).astype(int)
    x1, y1 = np.minimum(x0 + 1, width - 1), np.minimum(y0 + 1, height - 1)
    dx, dy = (safe[..., 0] - x0)[..., None], (safe[..., 1] - y0)[..., None]
    out = ((1 - dy) * ((1 - dx) * values[y0, x0] + dx * values[y0, x1]) +
           dy * ((1 - dx) * values[y1, x0] + dx * values[y1, x1]))
    valid &= np.isfinite(out).all(-1)
    return np.where(valid[..., None], out, np.nan), valid


def _homography(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate((points, np.ones(points.shape[:-1] + (1,))), axis=-1)
    projected = homogeneous @ np.asarray(matrix).T
    denominator = projected[..., 2:3]
    return np.divide(projected[..., :2], denominator,
                     out=np.full_like(projected[..., :2], np.nan), where=np.abs(denominator) > 1e-12)


def restore_flow_endpoints(points_crop: np.ndarray, flow_crop: np.ndarray,
                           crop_to_world_t: np.ndarray, crop_to_world_next: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map source and transported target through their own inverse crop maps.

    These 3x3 matrices map cropped pixel coordinates to full-image pixels.
    This handles moving or resized crops; multiplying a flow vector by the
    first crop's scale alone loses the second crop's camera/translation change.
    """
    start = _homography(np.asarray(points_crop), crop_to_world_t)
    end = _homography(np.asarray(points_crop) + np.asarray(flow_crop), crop_to_world_next)
    return start, end


def joint_transport_diagnostics(flow: FlowResult, projected: np.ndarray, *,
                                rgb: np.ndarray | None = None,
                                valid: np.ndarray | None = None,
                                neighborhood_radius: int = 3,
                                minimum_support: float = 0.5,
                                crop_to_full: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Compare joint displacements with robust nearby image transport.

    ``projected`` is [T,J,2] in the *input flow image* coordinates. Optional
    ``crop_to_full[T,3,3]`` maps each frame into the same full-image pixel frame.
    ``valid`` is observed track availability [T,J], never synthetic truth.

    Scalar arrays have [T-1,J] shape. Low texture, flow disagreement and clothing
    remain diagnostics for later calibration. Anatomical joint centers need
    not move exactly like nearby visible material; no hard semantic verdict is
    inferred here. Unavailable diagnostics remain NaN, not artificial certainty.
    """
    forward = np.asarray(flow.forward)
    points = np.asarray(projected, dtype=float)
    pairs, height, width, channels = forward.shape
    if channels != 2 or points.ndim != 3 or points.shape[0] != pairs + 1 or points.shape[-1] != 2:
        raise ValueError("Flow [T-1,H,W,2] and projected joints [T,J,2] must align")
    if not 0 < minimum_support <= 1 or neighborhood_radius < 0:
        raise ValueError("Use radius >= 0 and 0 < minimum_support <= 1")
    if valid is None:
        valid = np.isfinite(points).all(-1)
    valid = np.asarray(valid, dtype=bool)
    if valid.shape != points.shape[:2]:
        raise ValueError("valid must have shape [T,J]")
    valid = valid & np.isfinite(points).all(-1)
    valid &= (points[..., 0] >= 0) & (points[..., 0] <= width - 1)
    valid &= (points[..., 1] >= 0) & (points[..., 1] <= height - 1)
    if rgb is not None and np.asarray(rgb).shape != (pairs + 1, height, width, 3):
        raise ValueError("RGB and flow image dimensions must match")
    if crop_to_full is not None and np.shape(crop_to_full) != (pairs + 1, 3, 3):
        raise ValueError("crop_to_full must have shape [T,3,3]")
    axis = np.arange(-neighborhood_radius, neighborhood_radius + 1)
    offsets = np.stack(np.meshgrid(axis, axis), axis=-1).reshape(-1, 2)
    shape = (pairs, points.shape[1])
    names = ("transport_error_px", "forward_backward_error_px", "photometric_error",
             "flow_dispersion_px", "flow_uncertainty_px", "texture_std", "support_fraction")
    output = {key: np.full(shape, np.nan, dtype=np.float32) for key in names}
    output["transport_residual_xy"] = np.full(shape + (2,), np.nan, np.float32)
    output["local_flow_xy"] = np.full(shape + (2,), np.nan, np.float32)
    output["evidence_valid"] = np.zeros(shape, dtype=bool)
    for t in range(pairs):
        neighborhoods = points[t, :, None] + offsets[None]
        local_flow, support = bilinear_sample(forward[t], neighborhoods)
        target = neighborhoods + local_flow
        # Out-of-frame transported endpoints are missing measurements too.
        _, target_support = bilinear_sample(forward[t], target)
        support &= target_support
        support &= (valid[t] & valid[t + 1])[:, None]
        if crop_to_full is not None:
            source_full, target_full = restore_flow_endpoints(neighborhoods, local_flow,
                                                            crop_to_full[t], crop_to_full[t + 1])
            displacement = target_full - source_full
            observed_displacement = (_homography(points[t + 1], crop_to_full[t + 1]) -
                                     _homography(points[t], crop_to_full[t]))
        else:
            displacement = local_flow
            observed_displacement = points[t + 1] - points[t]
        displacement = np.where(support[..., None], displacement, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            center = np.nanmedian(displacement, axis=1)
            residual = observed_displacement - center
            output["local_flow_xy"][t] = center
            output["transport_residual_xy"][t] = residual
            output["transport_error_px"][t] = np.linalg.norm(residual, axis=-1)
            output["flow_dispersion_px"][t] = np.nanmedian(np.linalg.norm(displacement - center[:, None], axis=-1), axis=1)
            output["support_fraction"][t] = support.mean(axis=1)
            output["evidence_valid"][t] = (support.mean(axis=1) >= minimum_support) & np.isfinite(residual).all(-1)
            if flow.backward is not None:
                reverse, reverse_valid = bilinear_sample(flow.backward[t], target)
                roundtrip = target + reverse
                if crop_to_full is not None:
                    cycle = _homography(roundtrip, crop_to_full[t]) - _homography(neighborhoods, crop_to_full[t])
                else:
                    cycle = local_flow + reverse
                error = np.linalg.norm(cycle, axis=-1)
                output["forward_backward_error_px"][t] = np.nanmedian(np.where(support & reverse_valid, error, np.nan), axis=1)
            if flow.uncertainty is not None:
                unc, unc_valid = bilinear_sample(flow.uncertainty[t], neighborhoods)
                # A homography makes uncertainty anisotropic and location-
                # dependent. Do not mislabel crop scales as full-image pixels.
                if crop_to_full is None:
                    output["flow_uncertainty_px"][t] = np.nanmedian(np.where((support & unc_valid)[..., None], unc, np.nan), axis=(1, 2))
            if rgb is not None:
                first, first_ok = bilinear_sample(np.asarray(rgb[t], dtype=float) / 255, neighborhoods)
                second, second_ok = bilinear_sample(np.asarray(rgb[t + 1], dtype=float) / 255, target)
                photo = np.abs(first - second).mean(-1)
                output["photometric_error"][t] = np.nanmedian(np.where(support & first_ok & second_ok, photo, np.nan), axis=1)
                intensity = first.mean(-1)
                output["texture_std"][t] = np.nanstd(np.where(support & first_ok, intensity, np.nan), axis=1)
    return output
