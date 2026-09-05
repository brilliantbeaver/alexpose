"""pretrain_fold_local.py -- fold-local, seeded, faithful copy of the canonical
nb04 5-stage curriculum, for the INDUCTIVE (unseen-source-video) redesign.

`pretrain(allowed_video_ids, seed, out_path)` runs the ENTIRE curriculum
(Stage-0 normal 300ep @1e-3; +parkinsons/+stroke/+myopathic/+cerebralpalsy 75ep
each @3e-4) on ONLY the sequences whose source video is in `allowed_video_ids`,
seeded by `seed`. One model / vicreg_projector / target_center / EMA-target is
kept alive across all 5 stages; optimizer / scheduler / batch-RNG reset per stage
(batch RNG default_rng(seed + 1000*stage); mask seed seed + 100000*stage + step).

FAITHFULNESS. Model classes, losses, masking, geometric_view, balanced replay,
train_stage, and the fingerprint recipe are copied from _nb04_recipe_dump.txt
(the canonical trainer) and e3b_reflection_augmented_retrain.py (the validated
self-contained Stage-0 copy). The ONLY changes vs canonical: (a) restrict records
to allowed_video_ids; (b) coverage-drop is applied to EVERY condition (verified to
reproduce the canonical 626 cohort); (c) census equality asserts relaxed (fold
subsets are smaller); (d) seed threaded into torch/np/batch-RNG/mask-seed;
(e) per-fold dataset_fingerprint recomputed (recorded, not asserted).

RECIPE-FIDELITY GATE. `dataset_fingerprint` is a pure function of the cohort
bytes + config + curriculum + completed-stage summaries + hyperparameters (NOT the
model weights). So `python pretrain_fold_local.py --gate` recomputes the full
5-stage fingerprint chain for ALL 93 videos, seed 42, WITHOUT training, and
asserts it equals 7d13841a (and the content/validity/mapping sub-hashes match).
This must pass before spending 15x curriculum compute.

Source video is the independent unit; folder labels are dataset annotations, not
diagnoses; no clinical claim. Unseen source video != unseen individual.
"""
import os
import sys
import copy
import math
import json
import time
import hashlib
import argparse
import warnings
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _probe_common import (  # noqa: E402
    ROOT, ART, CONDITIONS, MASK_KEYPOINTS, pose_records_from_cache, prepare_sequence,
)

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------ canonical recommended profile
MODE = "real"
RUN_PROFILE = "recommended"
FRAMES, SEGMENT_LENGTH = 64, 4
SEGMENTS = FRAMES // SEGMENT_LENGTH
EMBED_DIM, ENCODER_DEPTH, PREDICTOR_DEPTH, HEADS = 96, 4, 2, 4
NORMAL_EPOCHS = 300
FINETUNE_EPOCHS = 75
SAMPLES_PER_CONDITION = 4
EMA_START = 0.999
MASK_FRACTION = 0.60
VICREG_WEIGHT, GROUP_WEIGHT, GROUP_MARGIN = 0.05, 0.25, 1.0
NORMAL_LR, FINETUNE_LR = 0.001, 0.0003
MIN_COVERAGE = 0.50

CONFIG = {
    "frames": FRAMES, "joints": 33, "coordinate_dim": 3, "segment_length": SEGMENT_LENGTH,
    "embed_dim": EMBED_DIM, "encoder_depth": ENCODER_DEPTH,
    "predictor_depth": PREDICTOR_DEPTH, "heads": HEADS,
}

CURRICULUM = [
    {"stage": 0, "name": "normal_only", "add": "normal", "conditions": CONDITIONS[:1]},
    {"stage": 1, "name": "add_parkinsons", "add": "parkinsons", "conditions": CONDITIONS[:2]},
    {"stage": 2, "name": "add_stroke", "add": "stroke", "conditions": CONDITIONS[:3]},
    {"stage": 3, "name": "add_myopathic", "add": "myopathic", "conditions": CONDITIONS[:4]},
    {"stage": 4, "name": "add_cerebralpalsy", "add": "cerebralpalsy", "conditions": CONDITIONS[:5]},
]
assert [s["add"] for s in CURRICULUM] == CONDITIONS

# mapping file (enters the fingerprint) -- PROJECT_ROOT is the alexpose repo root
PROJECT_ROOT = ROOT.parents[2]
MAPPING_RELATIVE_PATH = Path("experiments/multiple-sclerosis/mapping-data/ms-pd-mapping.md")
MAPPING_PATH = PROJECT_ROOT / MAPPING_RELATIVE_PATH
if not MAPPING_PATH.is_file():
    raise FileNotFoundError(f"Required mapping file is missing: {MAPPING_PATH}")
MAPPING_SHA256 = hashlib.sha256(MAPPING_PATH.read_bytes()).hexdigest()

# canonical ground truth (verified against sjepa_curriculum_final.pt) for the gate
CANONICAL_FINGERPRINT = "7d13841aceac9eda843d43ca8434193e294d2fa10a48b6c6d21f6413a6e457e2"
CANONICAL_CONTENT_SHA = "df64c99fee66b66b55a2f6ba96a1df2c3dd72aea3d771e340e09733adc0bbb92"
CANONICAL_VALIDITY_SHA = "b67f5cc6f553ffc6f28e035175977148cb5f9e1bcbf0f63746ad9b58bd8691e1"
CANONICAL_MAPPING_SHA = "e7a0899ff7a919d2f18c4d5db018de56b9516947a53048b3d01bbb8619fc649d"


# ================================================================== model (training classes, from recipe)
class SkeletonPatchEncoder(nn.Module):
    def __init__(self, frames=64, joints=33, coordinate_dim=3, segment_length=4,
                 embed_dim=64, depth=2, heads=4, dropout=0.0):
        super().__init__()
        if frames % segment_length:
            raise ValueError("frames must be divisible by segment_length")
        self.frames, self.joints, self.coordinate_dim = frames, joints, coordinate_dim
        self.segment_length, self.embed_dim = segment_length, embed_dim
        self.segments = frames // segment_length
        self.patch_embed = nn.Linear(segment_length * coordinate_dim, embed_dim)
        self.time_pos = nn.Parameter(torch.randn(self.segments, embed_dim) * 0.02)
        self.joint_pos = nn.Parameter(torch.randn(joints, embed_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=heads, dim_feedforward=embed_dim * 4,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

    def patchify(self, x):
        batch, frames, joints, channels = x.shape
        expected = (self.frames, self.joints, self.coordinate_dim)
        if (frames, joints, channels) != expected:
            raise ValueError(f"Expected [B, {expected}], received {x.shape}")
        patches = x.reshape(batch, self.segments, self.segment_length, joints, channels)
        patches = patches.permute(0, 1, 3, 2, 4).contiguous()
        return patches.flatten(3)

    def positioned_tokens(self, x):
        tokens = self.patch_embed(self.patchify(x))
        return tokens + self.time_pos[None, :, None, :] + self.joint_pos[None, None, :, :]

    def forward(self, x, keep_mask=None):
        tokens = self.positioned_tokens(x)
        batch = len(tokens)
        flat = tokens.reshape(batch, self.segments * self.joints, self.embed_dim)
        if keep_mask is not None:
            keep_mask = keep_mask.reshape(batch, -1)
            kept_per_sample = keep_mask.sum(dim=1)
            if not torch.equal(kept_per_sample, kept_per_sample[:1].expand_as(kept_per_sample)):
                raise ValueError("Each sample must keep the same number of tokens")
            flat = flat[keep_mask].reshape(batch, int(kept_per_sample[0]), self.embed_dim)
        return self.norm(self.blocks(flat))


class SkeletonPredictor(nn.Module):
    def __init__(self, segments, joints, encoder_dim=64, predictor_dim=64, depth=2, heads=4, dropout=0.0):
        super().__init__()
        self.segments, self.joints = segments, joints
        self.encoder_to_predictor = nn.Linear(encoder_dim, predictor_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_dim))
        nn.init.normal_(self.mask_token, std=0.02)
        self.time_pos = nn.Parameter(torch.randn(segments, predictor_dim) * 0.02)
        self.joint_pos = nn.Parameter(torch.randn(joints, predictor_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=predictor_dim, nhead=heads, dim_feedforward=predictor_dim * 4,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(predictor_dim)
        self.output = nn.Linear(predictor_dim, encoder_dim)

    def forward(self, visible_features, target_mask):
        batch = len(visible_features)
        target_mask = target_mask.reshape(batch, self.segments * self.joints)
        visible_mask = ~target_mask
        visible = self.encoder_to_predictor(visible_features)
        full = self.mask_token.expand(batch, self.segments * self.joints, -1).clone()
        full[visible_mask] = visible.reshape(-1, visible.shape[-1])
        positions = (self.time_pos[:, None, :] + self.joint_pos[None, :, :]).reshape(
            1, self.segments * self.joints, -1)
        full = full + positions
        predicted = self.output(self.norm(self.blocks(full)))
        return predicted[target_mask].reshape(batch, -1, predicted.shape[-1])


class SJEPAGait(nn.Module):
    def __init__(self, frames=64, joints=33, coordinate_dim=3, segment_length=4,
                 embed_dim=64, encoder_depth=2, predictor_depth=2, heads=4):
        super().__init__()
        self.view_encoder = SkeletonPatchEncoder(
            frames, joints, coordinate_dim, segment_length, embed_dim, encoder_depth, heads)
        self.target_encoder = copy.deepcopy(self.view_encoder)
        for parameter in self.target_encoder.parameters():
            parameter.requires_grad_(False)
        self.predictor = SkeletonPredictor(
            self.view_encoder.segments, joints, embed_dim, embed_dim, predictor_depth, heads)
        self.register_buffer("target_center", torch.zeros(embed_dim))

    def forward(self, view, target, target_mask):
        visible_features = self.view_encoder(view, keep_mask=~target_mask)
        predicted = self.predictor(visible_features, target_mask)
        with torch.no_grad():
            all_targets = self.target_encoder(target)
            flat_mask = target_mask.reshape(len(target), -1)
            selected = all_targets[flat_mask].reshape(len(target), -1, all_targets.shape[-1])
        return predicted, selected

    @torch.no_grad()
    def update_target(self, momentum):
        for tp, vp in zip(self.target_encoder.parameters(), self.view_encoder.parameters()):
            tp.mul_(momentum).add_(vp, alpha=1.0 - momentum)

    @torch.no_grad()
    def update_center(self, targets, beta=0.9):
        self.target_center.mul_(beta).add_(targets.mean(dim=(0, 1)), alpha=1.0 - beta)


class VICRegProjector(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dimension, dimension), nn.GELU(),
                                 nn.Linear(dimension, dimension))

    def forward(self, x):
        return self.net(x)


# ================================================================== losses / masking / view (from recipe)
def sjepa_cross_entropy(predicted, targets, center, predictor_temperature=0.10, target_temperature=0.06):
    target_prob = torch.softmax((targets - center[None, None, :]) / target_temperature, dim=-1).detach()
    prediction_log_prob = torch.log_softmax(predicted / predictor_temperature, dim=-1)
    return -(target_prob * prediction_log_prob).sum(dim=-1).mean()


def cosine_ema(step, total_steps, start=0.996, end=1.0):
    progress = min(max(step / max(total_steps - 1, 1), 0.0), 1.0)
    return end - (end - start) * (math.cos(math.pi * progress) + 1.0) / 2.0


def uniform_neurologic_mask(valid_patch, mask_fraction=0.60, seed=None):
    valid_patch = np.asarray(valid_patch, dtype=bool)
    if valid_patch.ndim != 3 or valid_patch.shape[2] != 33:
        raise ValueError(f"Expected [B, S, 33], received {valid_patch.shape}")
    if not 0.0 < mask_fraction < 1.0:
        raise ValueError("mask_fraction must be between 0 and 1")
    rng = np.random.default_rng(seed)
    eligible_joint = np.zeros(33, dtype=bool)
    eligible_joint[MASK_KEYPOINTS] = True
    eligible = valid_patch & eligible_joint[None, None, :]
    counts = eligible.reshape(len(eligible), -1).sum(axis=1)
    if np.any(counts < 2):
        raise ValueError("Each sample needs at least two valid eligible tokens")
    n_mask = max(1, int(np.floor(counts.min() * mask_fraction)))
    n_mask = min(n_mask, int(counts.min()) - 1)
    mask = np.zeros_like(eligible)
    for batch_index in range(len(mask)):
        candidates = np.flatnonzero(eligible[batch_index].reshape(-1))
        chosen = rng.choice(candidates, size=n_mask, replace=False)
        mask[batch_index].reshape(-1)[chosen] = True
    forbidden = sorted(set(range(33)).difference(MASK_KEYPOINTS))
    assert not mask[:, :, forbidden].any()
    assert mask.reshape(len(mask), -1).any(axis=1).all()
    assert (~mask).reshape(len(mask), -1).any(axis=1).all()
    return mask


def mask_forbidden_count(mask):
    forbidden = sorted(set(range(33)) - set(MASK_KEYPOINTS))
    return int(np.asarray(mask, dtype=bool)[:, :, forbidden].sum())


def authorized_pool(tokens, valid_patch):
    batch, segments, _, dimension = tokens.shape
    selected = tokens[:, :, MASK_KEYPOINTS].reshape(batch, -1, dimension)
    weights = valid_patch[:, :, MASK_KEYPOINTS].reshape(batch, -1).to(tokens.dtype)
    denominator = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
    return (selected * weights.unsqueeze(-1)).sum(dim=1) / denominator


def off_diagonal(matrix):
    n, m = matrix.shape
    if n != m:
        raise ValueError("Covariance matrix must be square")
    return matrix.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def vicreg_terms(first, second, gamma=1.0, eps=1e-4):
    invariance = F.mse_loss(first, second)
    first_std = torch.sqrt(first.var(dim=0, unbiased=False) + eps)
    second_std = torch.sqrt(second.var(dim=0, unbiased=False) + eps)
    variance = 0.5 * (F.relu(gamma - first_std).mean() + F.relu(gamma - second_std).mean())
    first_centered = first - first.mean(dim=0)
    second_centered = second - second.mean(dim=0)
    denominator = max(len(first) - 1, 1)
    first_cov = first_centered.T @ first_centered / denominator
    second_cov = second_centered.T @ second_centered / denominator
    covariance = (off_diagonal(first_cov).square().sum()
                  + off_diagonal(second_cov).square().sum()) / (2.0 * first.shape[1])
    weighted = 25.0 * invariance + 25.0 * variance + covariance
    return weighted, invariance, variance, covariance


def condition_group_terms(representations, condition_ids, margin=1.0):
    unique = torch.unique(condition_ids)
    zero = representations.sum() * 0.0
    if len(unique) < 2:
        return zero, zero, torch.tensor(float("nan"), device=representations.device)
    normalized = F.normalize(representations, dim=1)
    centroids = torch.stack([
        F.normalize(normalized[condition_ids == value].mean(dim=0), dim=0) for value in unique])
    compactness = torch.stack([
        (normalized[condition_ids == value] - centroids[index]).square().sum(dim=1).mean()
        for index, value in enumerate(unique)]).mean()
    pairwise = (centroids[:, None] - centroids[None, :]).square().sum(dim=-1).clamp_min(1e-12).sqrt()
    upper = torch.triu(torch.ones_like(pairwise, dtype=torch.bool), diagonal=1)
    distances = pairwise[upper]
    separation = F.relu(margin - distances).square().mean()
    return compactness, separation, distances.min()


def geometric_view(x, max_degrees=8.0, translate=0.03, flip_probability=0.0):
    """Rotation about vertical + translation. Flip OFF (laterality can matter)."""
    view = x.clone()
    present = view.abs().sum(dim=-1) > 1e-8
    batch = len(view)
    angles = (torch.rand(batch, device=x.device) * 2.0 - 1.0) * math.radians(max_degrees)
    cosine, sine = torch.cos(angles), torch.sin(angles)
    original_x = view[..., 0].clone()
    original_z = view[..., 2].clone()
    view[..., 0] = cosine[:, None, None] * original_x + sine[:, None, None] * original_z
    view[..., 2] = -sine[:, None, None] * original_x + cosine[:, None, None] * original_z
    offsets = (torch.rand(batch, 1, 1, 2, device=x.device) * 2.0 - 1.0) * translate
    view[..., :2] += offsets
    if flip_probability > 0:  # unused in canonical recipe (kept for structural fidelity)
        raise NotImplementedError("flip is off in the canonical laterality recipe")
    return view.masked_fill(~present[..., None], 0.0)


def balanced_epoch_batches(condition_data, active_conditions, per_condition, rng):
    lengths = {c: len(condition_data[c]["xyz"]) for c in active_conditions}
    if min(lengths.values()) < 1:
        raise ValueError(f"Empty curriculum condition: {lengths}")
    steps = max(1, int(np.ceil(max(lengths.values()) / per_condition)))
    required = steps * per_condition
    orders = {}
    for condition in active_conditions:
        pieces = []
        while sum(len(piece) for piece in pieces) < required:
            pieces.append(rng.permutation(lengths[condition]))
        orders[condition] = np.concatenate(pieces)[:required]
    for step in range(steps):
        xyz_parts, valid_parts, label_parts = [], [], []
        for label, condition in enumerate(active_conditions):
            take = orders[condition][step * per_condition:(step + 1) * per_condition]
            xyz_parts.append(condition_data[condition]["xyz"][take])
            valid_parts.append(condition_data[condition]["valid"][take])
            label_parts.extend([label] * per_condition)
        permutation = rng.permutation(len(label_parts))
        yield (np.concatenate(xyz_parts)[permutation],
               np.concatenate(valid_parts)[permutation],
               np.asarray(label_parts, dtype=np.int64)[permutation])


# ================================================================== data assembly (coverage-drop-all)
def assemble_condition_data(allowed_video_ids, min_coverage=MIN_COVERAGE):
    """Return {condition: {records, xyz, valid}} restricted to allowed_video_ids,
    coverage-drop applied to EVERY condition (reproduces the canonical cohort rule).
    records aligned with xyz[i]/valid[i]; records carry sequence_id/video_id."""
    allowed = set(allowed_video_ids)
    condition_data = {}
    for condition in CONDITIONS:
        recs, xyzs, valids = [], [], []
        for r in pose_records_from_cache(conditions=[condition]):
            if r["video_id"] not in allowed:
                continue
            xyz, valid = prepare_sequence(r["raw"], FRAMES)
            if float(valid[:, MASK_KEYPOINTS].mean()) < min_coverage:
                continue
            recs.append({"sequence_id": r["sequence_id"], "video_id": r["video_id"],
                         "condition": condition, "cohort": "canonical"})
            xyzs.append(xyz)
            valids.append(valid)
        if recs:
            condition_data[condition] = {
                "records": recs,
                "xyz": np.stack(xyzs),
                "valid": np.stack(valids),
            }
    return condition_data


# ================================================================== fingerprint recipe (from recipe dump)
def _stage_cohort_and_hashes(condition_data, active):
    """Reproduce checkpoint_for_stage's cohort iteration + content/validity hashes."""
    content_hasher = hashlib.sha256()
    validity_hasher = hashlib.sha256()
    sequence_ids, video_ids, cohorts = [], [], []
    for condition in active:
        payload = condition_data[condition]
        for record, array, validity in sorted(
            zip(payload["records"], payload["xyz"], payload["valid"]),
            key=lambda item: item[0]["sequence_id"],
        ):
            sid = record["sequence_id"]
            content_hasher.update(sid.encode("utf-8"))
            content_hasher.update(np.ascontiguousarray(array).tobytes())
            validity_hasher.update(sid.encode("utf-8"))
            validity_hasher.update(np.ascontiguousarray(validity, dtype=np.bool_).tobytes())
            sequence_ids.append(sid)
            video_ids.append(record["video_id"])
            cohorts.append(record.get("cohort", "canonical"))
    return (sequence_ids, video_ids, cohorts,
            content_hasher.hexdigest(), validity_hasher.hexdigest())


def _stage_optimizer_updates(condition_data, active, epochs):
    steps_per_epoch = max(1, int(np.ceil(
        max(len(condition_data[c]["xyz"]) for c in active) / SAMPLES_PER_CONDITION)))
    return epochs * steps_per_epoch


def _completed_entry(stage, epochs, lr, optimizer_updates):
    return {
        "stage": stage["stage"], "name": stage["name"], "added_condition": stage["add"],
        "conditions": stage["conditions"], "epochs": epochs,
        "optimizer_updates": optimizer_updates, "learning_rate": lr,
    }


def _fingerprint_payload(stage, active, completed_stages, parent_fingerprint,
                         sequence_ids, content_sha, validity_sha, seed):
    return {
        "mode": MODE,
        "stage": stage["stage"],
        "stage_name": stage["name"],
        "conditions_seen": active,
        "curriculum": CURRICULUM,
        "completed_stages": completed_stages,
        "parent_fingerprint": parent_fingerprint,
        "sequence_ids": sorted(sequence_ids),
        "preprocessed_content_sha256": content_sha,
        "validity_mask_sha256": validity_sha,
        "mapping_path": str(MAPPING_RELATIVE_PATH),
        "mapping_sha256": MAPPING_SHA256,
        "mask_keypoints": MASK_KEYPOINTS,
        "mask_fraction": MASK_FRACTION,
        "model_config": CONFIG,
        "run_profile": RUN_PROFILE,
        "normal_epochs": NORMAL_EPOCHS,
        "finetune_epochs": FINETUNE_EPOCHS,
        "samples_per_condition": SAMPLES_PER_CONDITION,
        "ema_start": EMA_START,
        "vicreg_weight": VICREG_WEIGHT,
        "group_weight": GROUP_WEIGHT,
        "group_margin": GROUP_MARGIN,
        "seed": seed,
    }


def _fingerprint(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def fingerprint_chain(condition_data, seed):
    """Compute the full 5-stage fingerprint chain WITHOUT training (weight-free).
    Returns list of per-stage dicts. Final stage's `fingerprint` is the checkpoint's
    dataset_fingerprint."""
    completed_stages, parent, out = [], None, []
    for stage in CURRICULUM:
        active = stage["conditions"]
        epochs = NORMAL_EPOCHS if stage["stage"] == 0 else FINETUNE_EPOCHS
        lr = NORMAL_LR if stage["stage"] == 0 else FINETUNE_LR
        upd = _stage_optimizer_updates(condition_data, active, epochs)
        completed_stages = completed_stages + [_completed_entry(stage, epochs, lr, upd)]
        seq, vids, cohorts, content_sha, validity_sha = _stage_cohort_and_hashes(condition_data, active)
        payload = _fingerprint_payload(stage, active, completed_stages, parent,
                                       seq, content_sha, validity_sha, seed)
        fp = _fingerprint(payload)
        out.append({"stage": stage["stage"], "fingerprint": fp, "parent": parent,
                    "content_sha": content_sha, "validity_sha": validity_sha,
                    "n_sequences": len(seq)})
        parent = fp
    return out


# ================================================================== training
def _resolve_device():
    return torch.device("cuda" if torch.cuda.is_available()
                        else "mps" if torch.backends.mps.is_available() else "cpu")


@torch.no_grad()
def _target_authorized_embeddings(model, arrays, validity, device, batch_size=16):
    model.target_encoder.eval()
    vectors = []
    for start in range(0, len(arrays), batch_size):
        batch = torch.tensor(arrays[start:start + batch_size], dtype=torch.float32, device=device)
        valid = torch.tensor(validity[start:start + batch_size], dtype=torch.bool, device=device)
        valid_patch = valid.reshape(len(batch), SEGMENTS, SEGMENT_LENGTH, 33).all(dim=2)
        tokens = model.target_encoder(batch).reshape(len(batch), SEGMENTS, 33, EMBED_DIM)
        vectors.append(authorized_pool(tokens, valid_patch).cpu())
    return torch.cat(vectors)


def train_stage(model, vicreg_projector, condition_data, stage, epochs, learning_rate,
                seed, device, verbose=True):
    """Continue the same model; only optimizer/schedule/batch-RNG restart. Faithful
    to _nb04_recipe_dump.txt train_stage, with `seed` threaded in place of 42."""
    active = stage["conditions"]
    trainable = [
        *[p for p in model.view_encoder.parameters() if p.requires_grad],
        *[p for p in model.predictor.parameters() if p.requires_grad],
        *list(vicreg_projector.parameters()),
    ]
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.05)
    steps_per_epoch = max(1, int(np.ceil(
        max(len(condition_data[c]["xyz"]) for c in active) / SAMPLES_PER_CONDITION)))
    total_steps = epochs * steps_per_epoch
    warmup_steps = max(1, min(steps_per_epoch, total_steps // 10))

    def lr_factor(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps - 1, 1)
        return 0.5 + 0.5 * (1.0 + math.cos(math.pi * progress)) / 2.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_factor)
    rng = np.random.default_rng(seed + 1000 * stage["stage"])
    global_step = 0
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        vicreg_projector.train()
        epoch_losses = []
        for xyz_np, valid_np, labels_np in balanced_epoch_batches(
            condition_data, active, SAMPLES_PER_CONDITION, rng
        ):
            coordinates = torch.tensor(xyz_np, dtype=torch.float32, device=device)
            valid = torch.tensor(valid_np, dtype=torch.bool, device=device)
            labels_tensor = torch.tensor(labels_np, dtype=torch.long, device=device)
            valid_patch = valid.reshape(len(valid), SEGMENTS, SEGMENT_LENGTH, 33).all(dim=2)
            eligible_counts = valid_patch[:, :, MASK_KEYPOINTS].reshape(len(valid), -1).sum(dim=1)
            if int(eligible_counts.min()) < 2:
                raise ValueError("Every sampled sequence needs at least two valid authorized patches")
            mask_np = uniform_neurologic_mask(
                valid_patch.cpu().numpy(), mask_fraction=MASK_FRACTION,
                seed=seed + 100000 * stage["stage"] + global_step)
            if mask_forbidden_count(mask_np) != 0:
                raise AssertionError("A forbidden keypoint entered the target mask")
            target_mask = torch.tensor(mask_np, dtype=torch.bool, device=device)
            view_a = geometric_view(coordinates, max_degrees=8.0, translate=0.03, flip_probability=0.0)
            view_b = geometric_view(coordinates, max_degrees=8.0, translate=0.03, flip_probability=0.0)

            prediction, target = model(view_a, coordinates, target_mask)
            jepa_loss = sjepa_cross_entropy(prediction, target, model.target_center)

            tokens_a = model.view_encoder(view_a).reshape(len(view_a), SEGMENTS, 33, EMBED_DIM)
            tokens_b = model.view_encoder(view_b).reshape(len(view_b), SEGMENTS, 33, EMBED_DIM)
            pooled_a = authorized_pool(tokens_a, valid_patch)
            pooled_b = authorized_pool(tokens_b, valid_patch)
            vicreg_loss, _, _, _ = vicreg_terms(vicreg_projector(pooled_a), vicreg_projector(pooled_b))
            compactness, separation, _ = condition_group_terms(pooled_a, labels_tensor, margin=GROUP_MARGIN)
            group_loss = compactness + separation
            total_loss = jepa_loss + VICREG_WEIGHT * vicreg_loss + GROUP_WEIGHT * group_loss
            if not torch.isfinite(total_loss):
                raise FloatingPointError(f"Non-finite loss in {stage['name']} step {global_step}")

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            assert all(p.grad is None for p in model.target_encoder.parameters())
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()
            scheduler.step()
            model.update_target(cosine_ema(global_step, total_steps, start=EMA_START, end=1.0))
            model.update_center(target, beta=0.9)
            epoch_losses.append(float(jepa_loss.detach().cpu()))
            global_step += 1
        if verbose and ((epoch + 1) % 25 == 0 or epoch == 0 or epoch == epochs - 1):
            print(f"  stage {stage['stage']} {stage['name']} epoch {epoch + 1:03d}/{epochs}  "
                  f"JEPA {np.mean(epoch_losses):.4f}  lr {optimizer.param_groups[0]['lr']:.2e}  "
                  f"elapsed {time.time() - t0:.0f}s", flush=True)
    return total_steps


def build_stage_checkpoint(model, vicreg_projector, condition_data, stage,
                           completed_stages, parent_fingerprint, seed, fold_id,
                           allowed_video_ids):
    active = stage["conditions"]
    seq, vids, cohorts, content_sha, validity_sha = _stage_cohort_and_hashes(condition_data, active)
    payload = _fingerprint_payload(stage, active, completed_stages, parent_fingerprint,
                                   seq, content_sha, validity_sha, seed)
    fingerprint = _fingerprint(payload)
    return {
        "model_state": model.state_dict(),
        "vicreg_projector_state": vicreg_projector.state_dict(),
        "config": CONFIG,
        "mode": MODE,
        "mask_keypoints": MASK_KEYPOINTS,
        "mask_source": str(MAPPING_RELATIVE_PATH),
        "mask_source_sha256": MAPPING_SHA256,
        "dataset_fingerprint": fingerprint,
        "parent_fingerprint": parent_fingerprint,
        "training_stage": stage["name"],
        "stage_index": stage["stage"],
        "conditions_seen": active,
        "completed_stages": completed_stages,
        "curriculum": CURRICULUM,
        "curriculum_complete": stage["stage"] == len(CURRICULUM) - 1,
        "label_aware_after_stage0": True,
        "sequence_ids": seq,
        "video_ids": vids,
        "cohorts": cohorts,
        "fingerprint_payload": payload,
        # inductive-context provenance:
        "seed": seed,
        "fold_id": fold_id,
        "allowed_video_ids": sorted(allowed_video_ids),
        "inductive": True,
    }


def pretrain(allowed_video_ids, seed, out_path, fold_id=None, verbose=True):
    """Run the full 5-stage curriculum on allowed_video_ids only, seeded, and save
    the final checkpoint to out_path. Returns the final checkpoint dict (sans state)."""
    out_path = Path(out_path)
    device = _resolve_device()
    condition_data = assemble_condition_data(allowed_video_ids)
    missing = [c for c in CONDITIONS if c not in condition_data]
    if missing:
        raise ValueError(f"fold {fold_id}: conditions absent from train cohort: {missing}")
    sizes = {c: len(condition_data[c]["xyz"]) for c in CONDITIONS}
    if verbose:
        print(f"[fold {fold_id} seed {seed}] device={device} cohort sizes={sizes} "
              f"total={sum(sizes.values())} videos={len(set(allowed_video_ids))}", flush=True)

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = SJEPAGait(**CONFIG).to(device)
    vicreg_projector = VICRegProjector(EMBED_DIM).to(device)
    assert not any(p.requires_grad for p in model.target_encoder.parameters())

    completed_stages, parent_fingerprint = [], None
    t0 = time.time()
    final_ckpt = None
    for stage in CURRICULUM:
        epochs = NORMAL_EPOCHS if stage["stage"] == 0 else FINETUNE_EPOCHS
        lr = NORMAL_LR if stage["stage"] == 0 else FINETUNE_LR
        updates = train_stage(model, vicreg_projector, condition_data, stage, epochs, lr,
                              seed, device, verbose=verbose)
        completed_stages = completed_stages + [_completed_entry(stage, epochs, lr, updates)]
        ckpt = build_stage_checkpoint(model, vicreg_projector, condition_data, stage,
                                      completed_stages, parent_fingerprint, seed, fold_id,
                                      allowed_video_ids)
        parent_fingerprint = ckpt["dataset_fingerprint"]
        final_ckpt = ckpt
        if verbose:
            print(f"[fold {fold_id} seed {seed}] stage {stage['stage']} done "
                  f"fp={ckpt['dataset_fingerprint'][:12]} ({time.time() - t0:.0f}s)", flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(final_ckpt, out_path)
    if verbose:
        print(f"[fold {fold_id} seed {seed}] wrote {out_path}  "
              f"final fp={final_ckpt['dataset_fingerprint'][:12]}  "
              f"wall={time.time() - t0:.0f}s", flush=True)
    return {k: v for k, v in final_ckpt.items() if k not in ("model_state", "vicreg_projector_state")}


# ================================================================== recipe-fidelity gate
def run_gate():
    """Recompute the 5-stage fingerprint chain for ALL 93 videos, seed 42, WITHOUT
    training, and assert it reproduces the canonical 7d13841a (and sub-hashes)."""
    all_videos = sorted({r["video_id"] for r in pose_records_from_cache()})
    condition_data = assemble_condition_data(all_videos)
    sizes = {c: len(condition_data[c]["xyz"]) for c in CONDITIONS}
    total = sum(sizes.values())
    print(f"[gate] full cohort sizes={sizes} total={total} videos={len(all_videos)}")
    assert total == 626, f"gate cohort total {total} != 626"

    chain = fingerprint_chain(condition_data, seed=42)
    final = chain[-1]
    print("[gate] per-stage fingerprints:")
    for c in chain:
        print(f"    stage {c['stage']}: n={c['n_sequences']:4d} fp={c['fingerprint'][:12]} "
              f"parent={(c['parent'] or 'none')[:12]}")

    ok = True
    def check(name, got, want):
        nonlocal ok
        status = "OK " if got == want else "MISMATCH"
        if got != want:
            ok = False
        print(f"[gate] {status} {name}: {got[:16]}... {'==' if got == want else '!='} {want[:16]}...")

    check("mapping_sha256", MAPPING_SHA256, CANONICAL_MAPPING_SHA)
    check("preprocessed_content_sha256", final["content_sha"], CANONICAL_CONTENT_SHA)
    check("validity_mask_sha256", final["validity_sha"], CANONICAL_VALIDITY_SHA)
    check("dataset_fingerprint", final["fingerprint"], CANONICAL_FINGERPRINT)

    if ok:
        print("\n[gate] PASS -- fold-local recipe reproduces canonical 7d13841a. "
              "Cohort bytes + recipe are byte-identical; safe to run the 15 curriculum trainings.")
    else:
        print("\n[gate] FAIL -- recipe/cohort diverges from canonical. STOP and reconcile "
              "before spending 15x compute.")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", action="store_true",
                    help="recompute the fingerprint chain (no training) and assert 7d13841a")
    args = ap.parse_args()
    if args.gate:
        sys.exit(0 if run_gate() else 1)
    ap.error("nothing to do; use --gate, or import pretrain() from the harness")
