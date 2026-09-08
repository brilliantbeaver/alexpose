"""Adapter for the reviewed V-JEPA 2.1 Hub builder; never downloads weights."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from gavd6_sjepa.shared_infrastructure.artifact_io_operations import sha256_file

from .fi_contracts import FRAME, VJEPA_COMMIT, check_run, read_json
from .fi_token_regions import context_indices
from .fi_video_pose import CropGeometry


def deterministic_torch(seed=0):
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


class FrozenVJEPAAdapter:
    def __init__(self, encoder, processor, device="cpu", frame=FRAME):
        self.frame, self.device = frame, torch.device(device)
        self.encoder = (
            encoder.to(device=self.device, dtype=torch.float32)
            .eval()
            .requires_grad_(False)
        )
        self.processor = processor
        projection = self.encoder.patch_embed.proj
        if (
            tuple(projection.kernel_size)
            != (frame.tubelet_size, frame.patch_size, frame.patch_size)
            or tuple(projection.stride) != tuple(projection.kernel_size)
            or any(projection.padding)
        ):
            raise ValueError(
                "Teacher patch embedding crosses the frozen temporal boundary"
            )

    @classmethod
    def from_run(cls, root, device="cuda"):
        check_run(root)
        teacher = read_json(Path(root) / "config/teacher-contract.json")
        if teacher["synthetic"]:
            raise ValueError("Synthetic run cannot load the scientific teacher")
        repo = Path(teacher["repository_path"])
        commit = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
        dirty = subprocess.check_output(
            [
                "git",
                "-C",
                str(repo),
                "status",
                "--porcelain",
                "--untracked-files=normal",
            ],
            text=True,
        ).strip()
        if commit != VJEPA_COMMIT or dirty:
            raise ValueError(
                "Teacher checkout must be clean and at the reviewed commit"
            )
        checkpoint = Path(teacher["checkpoint_path"])
        if sha256_file(checkpoint) != teacher["checkpoint_sha256"]:
            raise ValueError("Teacher checkpoint checksum mismatch")
        deterministic_torch()
        # Upstream's reviewed Hub URL points to localhost. Build without downloads
        # and load the documented ema_encoder key with strict=True instead.
        processor = torch.hub.load(
            str(repo), "vjepa2_preprocessor", source="local", crop_size=384
        )
        encoder, predictor = torch.hub.load(
            str(repo), teacher["builder"], source="local", pretrained=False
        )
        del predictor
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state = payload[teacher["checkpoint_key"]]
        cleaned = {
            key.replace("module.", "").replace("backbone.", ""): value
            for key, value in state.items()
        }
        if len(cleaned) != len(state):
            raise ValueError("Checkpoint key normalization collision")
        encoder.load_state_dict(cleaned, strict=True)
        del payload, state, cleaned
        if (
            encoder.embed_dim != 768
            or encoder.num_frames != 64
            or len(encoder.blocks) != 12
        ):
            raise ValueError("Unexpected V-JEPA 2.1 base architecture")
        if encoder.out_layers is not None or encoder.return_hierarchical:
            raise ValueError("Expected only final normalized encoder layer")
        return cls(encoder, processor, device)

    def preprocess(self, video):
        if (
            video.dtype != np.uint8
            or video.ndim != 4
            or len(video) != self.frame.frames_per_clip
            or video.shape[-1] != 3
        ):
            raise ValueError("Teacher requires uint8 [64,H,W,3] RGB")
        result = self.processor(list(video))
        if not isinstance(result, list) or len(result) != 1:
            raise ValueError("Pinned preprocessor must return exactly one crop")
        tensor = result[0]
        expected = (
            3,
            self.frame.frames_per_clip,
            self.frame.resolution,
            self.frame.resolution,
        )
        if tuple(tensor.shape) != expected:
            raise ValueError(
                f"Unexpected preprocessor layout {tensor.shape}, expected {expected}"
            )
        return tensor.unsqueeze(0).to(device=self.device, dtype=torch.float32)

    def verify_geometry(self, video):
        official = self.preprocess(video).cpu()[0]
        pixels = CropGeometry(*video.shape[1:3], self.frame.resolution).pixels(video)
        mirrored = torch.from_numpy(pixels.copy()).permute(3, 0, 1, 2).float() / 255
        mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None, None]
        std = torch.tensor([0.229, 0.224, 0.225])[:, None, None, None]
        error = float((official - (mirrored - mean) / std).abs().max())
        if error > 1e-6:
            raise ValueError(
                f"Pinned preprocessor differs from box geometry mirror: {error}"
            )
        return error

    @torch.inference_mode()
    def _encode(self, video, indices=None):
        tensor = self.preprocess(video)
        masks = (
            None
            if indices is None
            else torch.as_tensor(indices, device=self.device, dtype=torch.long)[None]
        )
        expected = (
            self.frame.frames_per_clip // self.frame.tubelet_size * self.frame.grid**2
            if indices is None
            else len(indices)
        )
        seen = []
        # Verify that future token removal occurred before the first attention block.
        hook = self.encoder.blocks[0].register_forward_pre_hook(
            lambda module, args: seen.append(args[0].shape[1])
        )
        try:
            result = self.encoder(tensor, masks=masks)
        finally:
            hook.remove()
        if (
            seen != [expected]
            or not isinstance(result, torch.Tensor)
            or result.shape[0:2] != (1, expected)
        ):
            raise ValueError("Teacher token selection/output contract violated")
        # Official training applies non-affine target layer_norm to encoder outputs.
        # This is distinct from norms_block[-1]'s learned affine transformation.
        result = F.layer_norm(result, (result.shape[-1],), eps=1e-5)
        if not torch.isfinite(result).all():
            raise ValueError("Teacher produced non-finite tokens")
        return result[0].cpu().numpy()

    def encode_past_context(self, video, indices=None):
        allowed = context_indices(self.frame)
        if indices is not None and not np.array_equal(indices, allowed):
            raise ValueError("Context mask must keep precisely all past tubelets")
        return self._encode(video, allowed)

    def encode_full_target(self, video):
        return self._encode(video)
