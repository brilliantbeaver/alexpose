"""Trainable MMPose students with a shared twelve-landmark interface.

The released COCO heads and their author losses/codecs are retained. Only the
prediction head is updated; frozen feature extractors remain in evaluation
mode, including batch-normalization statistics. Inputs are RGB uint8 images and
fixed person boxes. Outputs and labels are in the original image coordinates.

See the official ``mmpose.apis.init_model``, ``TopdownPoseEstimator`` and
``TopdownAffine`` implementations. Optional OpenMMLab imports occur at loading,
so analysis and selector fitting do not need an installed pose-training stack.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
import torch


BODY_JOINTS = (
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
    "right_knee", "left_ankle", "right_ankle",
)
COCO_BODY_INDICES = np.arange(5, 17)


@dataclass(frozen=True)
class StudentSpec:
    """One released student and optional previously adapted prediction head.

    ``config`` is a local MMPose 1.x Python configuration and ``checkpoint`` is
    its matching local COCO checkpoint. ``head_checkpoint`` denotes a genuine
    source adaptation history, not an independently pretrained architecture.
    """

    student_id: str
    family: str
    config: str
    checkpoint: str
    head_checkpoint: str | None = None


def _images_and_boxes(images, boxes=None) -> tuple[list[np.ndarray], np.ndarray]:
    """Validate RGB frames and retain externally fixed xyxy crop geometry."""
    images = list(images)
    for image in images:
        if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
                or image.ndim != 3 or image.shape[-1] != 3
                or min(image.shape[:2]) < 2):
            raise ValueError("images must contain uint8 [H,W,3] RGB frames")
    if boxes is None:
        boxes = [[0, 0, image.shape[1], image.shape[0]] for image in images]
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    if boxes.shape != (len(images), 4) or not np.isfinite(boxes).all():
        raise ValueError("boxes must be finite [N,4] xyxy in original pixels")
    if np.any(boxes[:, 2:] <= boxes[:, :2]):
        raise ValueError("Every person box must have positive width and height")
    return images, boxes


def coco_training_labels(keypoints, visible) -> tuple[np.ndarray, np.ndarray]:
    """Expand body labels to COCO17, with zero weight for unsupervised joints.

    Missing labels may be NaN only where ``visible`` is false. They are replaced
    with zero before affine transforms and codec encoding, avoiding NaN losses
    even in codecs that multiply an already computed target by its weight.
    """
    keypoints = np.asarray(keypoints, dtype=np.float32)
    visible = np.asarray(visible, dtype=bool)
    if (keypoints.ndim != 3 or keypoints.shape[1:] != (12, 2)
            or visible.shape != keypoints.shape[:2]):
        raise ValueError("Expected keypoints [N,12,2] and visibility [N,12]")
    if not np.isfinite(keypoints[visible]).all():
        raise ValueError("Visible training landmarks must have finite coordinates")
    expanded = np.zeros((len(keypoints), 17, 2), dtype=np.float32)
    weights = np.zeros((len(keypoints), 17), dtype=np.float32)
    expanded[:, COCO_BODY_INDICES] = np.where(visible[..., None], keypoints, 0)
    weights[:, COCO_BODY_INDICES] = visible
    return expanded, weights


def _adaptation_pipelines(config) -> tuple[list[dict], list[dict]]:
    """Use exactly the released deterministic crop geometry for both paths.

    Training adds the author's target codec, without introducing an unshared
    random augmentation or secretly changing the study's selected lesson.
    """
    prediction = deepcopy(list(config.test_dataloader.dataset.pipeline))
    names = [str(item["type"]).split(".")[-1] for item in prediction]
    expected = ["LoadImage", "GetBBoxCenterScale", "TopdownAffine", "PackPoseInputs"]
    if names != expected:
        raise ValueError(
            "Use a standard COCO top-down config with LoadImage, "
            "GetBBoxCenterScale, TopdownAffine, PackPoseInputs; "
            f"received {names}"
        )
    codec = deepcopy(config.model.head.decoder)
    loss = config.model.head.get("loss", {}) if isinstance(config.model.head, dict) else None
    if loss is not None and not loss.get("use_target_weight", False):
        raise ValueError("The released head loss must honor visibility/target weights")
    training = deepcopy(prediction[:-1]) + [dict(type="GenerateTarget", encoder=codec), deepcopy(prediction[-1])]
    return prediction, training


class MMPoseEstimator:
    """Thin supervised adapter over an actual author pose estimator.

    Construct with :func:`load_estimator` in experiments. Injected pipelines in
    the constructor also allow small tests of gradient and branch isolation
    without replacing the scientific estimator with a demonstration model.
    """

    def __init__(self, model, spec: StudentSpec, prediction_pipeline: Callable,
                 training_pipeline: Callable, collate: Callable):
        self.model = model
        self.spec = spec
        self.prediction_pipeline = prediction_pipeline
        self.training_pipeline = training_pipeline
        self.collate = collate
        if not hasattr(model, "head") or model.head is None:
            raise ValueError("The student must expose a trainable prediction head")
        model.requires_grad_(False)
        model.head.requires_grad_(True)
        model.eval()

    def _batch(self, images, boxes, keypoints=None, visible=None) -> dict:
        """Pack BGR author inputs, leaving RGB conversion to its preprocessor."""
        images, boxes = _images_and_boxes(images, boxes)
        if not images:
            raise ValueError("A training or inference batch cannot be empty")
        training = keypoints is not None
        if training:
            keypoints, visible = coco_training_labels(keypoints, visible)
            if len(keypoints) != len(images):
                raise ValueError("Training images and landmark labels differ in length")
        pipeline = self.training_pipeline if training else self.prediction_pipeline
        records = []
        for index, (image, box) in enumerate(zip(images, boxes)):
            record = deepcopy(self.model.dataset_meta)
            record.update(
                img=np.ascontiguousarray(image[..., ::-1]),
                bbox=box[None].copy(), bbox_score=np.ones(1, dtype=np.float32),
                id=index, img_id=index,
            )
            if training:
                record.update(keypoints=keypoints[index:index + 1].copy(),
                              keypoints_visible=visible[index:index + 1].copy())
            packed = pipeline(record)
            if packed is None:
                raise RuntimeError("The pose pipeline discarded a study frame")
            records.append(packed)
        return self.collate(records)

    @torch.inference_mode()
    def predict(self, images, *, boxes=None, batch_size: int = 32) -> np.ndarray:
        """Return original-pixel landmarks, retaining failed joints as NaNs.

        Missing predictions must receive the study's fixed reference-error
        penalty, rather than dropping the corresponding evaluation frames.
        Training still rejects non-finite losses or gradients.
        """
        images, boxes = _images_and_boxes(images, boxes)
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.model.eval()
        predictions = []
        for start in range(0, len(images), batch_size):
            batch = self._batch(images[start:start + batch_size], boxes[start:start + batch_size])
            samples = self.model.test_step(batch)
            if len(samples) != len(batch["data_samples"]):
                raise RuntimeError("The student changed the number of person predictions")
            for sample in samples:
                coordinates = sample.pred_instances.keypoints
                if isinstance(coordinates, torch.Tensor):
                    coordinates = coordinates.detach().cpu().numpy()
                coordinates = np.asarray(coordinates, dtype=np.float32)
                if coordinates.shape != (1, 17, 2):
                    raise RuntimeError(f"Expected one COCO17 person, got {coordinates.shape}")
                predictions.append(coordinates[0, COCO_BODY_INDICES])
        result = np.asarray(predictions, dtype=np.float32).reshape(-1, 12, 2)
        result[~np.isfinite(result).all(axis=-1)] = np.nan
        return result

    def make_optimizer(self, lr: float, weight_decay: float = 0.0):
        """Create a fresh AdamW state for a probe or an independent lesson branch."""
        if lr <= 0 or weight_decay < 0:
            raise ValueError("lr must be positive and weight_decay non-negative")
        return torch.optim.AdamW(self.model.head.parameters(), lr=lr, weight_decay=weight_decay)

    def train_batch(self, images, keypoints, visible, optimizer, *, boxes=None) -> float:
        """Apply one head-only supervised update using the released head's loss.

        The caller assembles the replay/synthetic mixture. Every image occupies
        one batch slot, so mixture and replay-only arms can use equal budgets.
        """
        if not np.asarray(visible, dtype=bool).any():
            raise ValueError("A supervised batch needs at least one visible landmark")
        self.model.eval()
        self.model.head.train()
        batch = self._batch(images, boxes, keypoints, visible)
        batch = self.model.data_preprocessor(batch, training=True)
        optimizer.zero_grad(set_to_none=True)
        losses = self.model(**batch, mode="loss")
        terms = []
        for name, value in losses.items():
            if "loss" in name:
                terms.extend(value if isinstance(value, (tuple, list)) else [value])
        if not terms:
            raise RuntimeError("The author head returned no supervised loss")
        loss = sum(term.mean() for term in terms)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite supervised pose loss")
        loss.backward()
        for parameter in self.model.head.parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise FloatingPointError("Non-finite pose-head gradient")
        optimizer.step()
        return float(loss.detach().cpu())

    def head_state(self) -> dict[str, torch.Tensor]:
        """Clone weights and head buffers so later updates cannot mutate a branch."""
        # Keep PyTorch's module-version metadata. Author checkpoint hooks use it
        # to distinguish current heads from older state-dict layouts.
        state = self.model.head.state_dict()
        for name, value in state.items():
            state[name] = value.detach().cpu().clone()
        return state

    def load_head_state(self, state: Mapping[str, torch.Tensor]) -> None:
        """Restore a branch point; create a new optimizer separately afterwards."""
        self.model.head.load_state_dict(state, strict=True)
        self.model.zero_grad(set_to_none=True)
        self.model.eval()

    def save_head(self, path: str | Path) -> None:
        """Save only mutable state; the released full checkpoint remains unchanged."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.head_state(), path)

    def load_head(self, path: str | Path) -> None:
        """Load a head file created by :meth:`save_head`."""
        self.load_head_state(torch.load(Path(path), map_location="cpu", weights_only=True))


def load_estimator(spec: StudentSpec, device: str = "cuda") -> MMPoseEstimator:
    """Load a real local COCO checkpoint, failing if any weights are unmatched."""
    for name, path in (("config", spec.config), ("checkpoint", spec.checkpoint)):
        if not Path(path).expanduser().is_file():
            raise FileNotFoundError(f"Student {spec.student_id} {name} does not exist: {path}")
    try:
        from mmengine.dataset import Compose, pseudo_collate
        from mmengine.runner import load_checkpoint
        from mmpose.apis import init_model
    except ImportError as error:
        raise ImportError(
            "Pose adaptation requires the separate MMPose environment described "
            "in slurm/synthetic-training/README.md (MMPose, MMCV, MMDetection, MMPreTrain)."
        ) from error
    # Building without a checkpoint suppresses any separate backbone download.
    # Strict loading then prevents accidental experiments with missing head keys.
    model = init_model(str(Path(spec.config).expanduser()), checkpoint=None, device=device)
    payload = load_checkpoint(model, str(Path(spec.checkpoint).expanduser()),
                              map_location="cpu", strict=True)
    if "dataset_meta" in payload.get("meta", {}):
        model.dataset_meta = payload["meta"]["dataset_meta"]
    names = model.dataset_meta.get("keypoint_id2name", {})
    if (int(model.dataset_meta.get("num_keypoints", 0)) != 17
            or tuple(names.get(int(i), names.get(str(i))) for i in COCO_BODY_INDICES) != BODY_JOINTS):
        raise ValueError("This adapter requires the released COCO17 landmark convention")
    prediction, training = _adaptation_pipelines(model.cfg)
    estimator = MMPoseEstimator(model, spec, Compose(prediction), Compose(training), pseudo_collate)
    if spec.head_checkpoint:
        estimator.load_head(Path(spec.head_checkpoint).expanduser())
    return estimator
