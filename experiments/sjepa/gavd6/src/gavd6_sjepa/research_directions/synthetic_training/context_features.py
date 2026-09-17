"""Frozen video context features and explicit simple/image comparators.

No encoder is fitted on deployment data. V-JEPA loads the downloaded encoder
from the author's local Hub interface, without downloading a second checkpoint
or using its predictor as a simulator of pose-student updates.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ContextSpec:
    """One explicitly selected frozen representation.

    V-JEPA 2.1 base/large use ``ema_encoder``; V-JEPA 2 usually uses
    ``target_encoder``. Match builder, resolution, and checkpoint key to the
    actual downloaded model. The image comparator is ImageNet ResNet50 V2.
    """

    kind: str = "simple"
    repository: str = ""
    checkpoint: str = ""
    builder: str = "vjepa2_1_vit_base_384"
    checkpoint_key: str = "ema_encoder"
    image_size: int = 384
    frames: int = 64


def sample_clip(clip: np.ndarray, frames: int, shuffle_seed: int | None = None) -> np.ndarray:
    """Select evenly spaced frames, then optionally shuffle that exact set.

    Short clips repeat frames. The temporal ablation changes order, not which
    frames are observed, the native image resolution, or the number of frames.
    """
    clip = np.asarray(clip)
    if (clip.dtype != np.uint8 or clip.ndim != 4 or clip.shape[-1] != 3
            or len(clip) < 1 or min(clip.shape[1:3]) < 2):
        raise ValueError("Expected a nonempty RGB uint8 clip [T,H,W,3]")
    if frames < 2:
        raise ValueError("Context features require at least two selected frames")
    indices = np.rint(np.linspace(0, len(clip) - 1, frames)).astype(int)
    if shuffle_seed is not None:
        indices = indices[np.random.default_rng(shuffle_seed).permutation(frames)]
    return np.ascontiguousarray(clip[indices])


def simple_context_features(clip: np.ndarray) -> np.ndarray:
    """Sixteen image statistics: RGB mean/std, edges, change mean/std, log size.

    Absolute adjacent-frame differences measure image change, not optical flow
    or semantic movement. Fixed-size spatial sampling limits work on large RGB
    frames; native dimensions remain two explicit metadata features.
    """
    from PIL import Image

    height, width = clip.shape[1:3]
    sampled = np.stack([np.asarray(Image.fromarray(frame).resize((64, 64), Image.Resampling.BILINEAR))
                        for frame in clip]).astype(np.float64) / 255.0
    gray = sampled @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    change = np.abs(np.diff(sampled, axis=0))
    features = np.concatenate([
        sampled.mean(axis=(0, 1, 2)), sampled.std(axis=(0, 1, 2)),
        [np.abs(np.diff(gray, axis=2)).mean(), np.abs(np.diff(gray, axis=1)).mean()],
        change.mean(axis=(0, 1, 2)), change.std(axis=(0, 1, 2)),
        np.log1p([width, height]),
    ])
    return features.astype(np.float32)


def _clean_encoder_state(state: dict) -> dict:
    """Remove the released DDP/backbone prefixes without altering inner names."""
    cleaned = {}
    for key, value in state.items():
        while key.startswith("module.") or key.startswith("backbone."):
            key = key.split(".", 1)[1]
        if key in cleaned:
            raise ValueError(f"Encoder checkpoint has colliding normalized key {key}")
        cleaned[key] = value
    return cleaned


class ContextEncoder:
    """Return one frozen feature vector per video clip."""

    def __init__(self, spec: ContextSpec, device: str = "cpu", *, model=None,
                 processor=None, author_repository=None):
        self.spec = spec
        self.device = torch.device(device)
        self.model = model
        self.processor = processor
        self.author_repository = author_repository
        if model is not None:
            model.to(self.device).eval().requires_grad_(False)

    @torch.inference_mode()
    def encode(self, clip: np.ndarray, shuffle_seed: int | None = None) -> np.ndarray:
        """Encode RGB context without pose labels or model adaptation outcomes."""
        clip = sample_clip(clip, self.spec.frames, shuffle_seed)
        if self.spec.kind == "none":
            return np.empty(0, dtype=np.float32)
        if self.spec.kind == "simple":
            return simple_context_features(clip)
        if self.spec.kind == "image":
            tensors = torch.from_numpy(clip).permute(0, 3, 1, 2)
            # The official V2 transform includes resize/crop and normalization.
            tensors = self.processor(tensors)
            features = []
            for batch in tensors.split(16):
                features.append(self.model(batch.to(self.device)))
            vector = torch.cat(features).mean(dim=0)
        elif self.spec.kind == "vjepa":
            with self.author_repository.activate():
                crops = self.processor(list(clip))
                if not isinstance(crops, list) or len(crops) != 1:
                    raise ValueError("V-JEPA preprocessor must return one deterministic crop")
                tensor = crops[0]
                expected = (3, self.spec.frames, self.spec.image_size, self.spec.image_size)
                if tuple(tensor.shape) != expected:
                    raise ValueError(f"V-JEPA preprocessing returned {tuple(tensor.shape)}, expected {expected}")
                precision = (torch.autocast("cuda", dtype=torch.bfloat16)
                             if self.device.type == "cuda" else nullcontext())
                with precision:
                    tokens = self.model(tensor.unsqueeze(0).to(self.device, dtype=torch.float32))
            if not isinstance(tokens, torch.Tensor) or tokens.ndim != 3 or tokens.shape[0] != 1:
                raise ValueError("Select a V-JEPA encoder returning final [batch,tokens,features]")
            vector = F.layer_norm(tokens.float(), (tokens.shape[-1],)).mean(dim=(0, 1))
        else:
            raise ValueError(f"Unknown context feature kind: {self.spec.kind}")
        if not torch.isfinite(vector).all():
            raise FloatingPointError("The frozen encoder produced non-finite context features")
        return F.normalize(vector, dim=0).cpu().numpy().astype(np.float32)


def load_context_encoder(spec: ContextSpec, device: str = "cuda") -> ContextEncoder:
    """Load local pretrained weights only; there is no demonstration fallback."""
    if spec.kind in {"none", "simple"}:
        return ContextEncoder(spec, device)
    if spec.kind not in {"image", "vjepa"}:
        raise ValueError("context kind must be none, simple, image, or vjepa")
    checkpoint = Path(spec.checkpoint).expanduser()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Frozen context checkpoint not found: {checkpoint}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if spec.kind == "image":
        from torchvision.models import ResNet50_Weights, resnet50

        model = resnet50(weights=None)
        model.load_state_dict(payload, strict=True)
        model.fc = torch.nn.Identity()
        return ContextEncoder(spec, device, model=model,
                              processor=ResNet50_Weights.IMAGENET1K_V2.transforms())

    from ..motion_preservation.pretrained_models import _AuthorRepository

    repository = _AuthorRepository(spec.repository)
    with repository.activate():
        model, predictor = torch.hub.load(str(repository.path), spec.builder,
                                         source="local", pretrained=False)
        del predictor
        processor = torch.hub.load(str(repository.path), "vjepa2_preprocessor",
                                   source="local", crop_size=spec.image_size)
    if spec.checkpoint_key not in payload:
        raise KeyError(f"Checkpoint lacks {spec.checkpoint_key!r}; available keys: {list(payload)}")
    model.load_state_dict(_clean_encoder_state(payload[spec.checkpoint_key]), strict=True)
    if getattr(model, "num_frames", spec.frames) != spec.frames:
        raise ValueError("Context frame count does not match the released V-JEPA builder")
    if getattr(model, "out_layers", None) is not None or getattr(model, "return_hierarchical", False):
        raise ValueError("Select a V-JEPA encoder exposing only its final token representation")
    return ContextEncoder(spec, device, model=model, processor=processor, author_repository=repository)
