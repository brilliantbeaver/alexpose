"""Matched normal-only joint-context prediction with explicit information access.

Uniform masks cover the original 8 x 16 grid. Own, contemporaneous-patch and
full branches re-encode permitted measurements with the same frozen encoder.
Coordinate and contextual-latent targets have separate heads and optimizers.
These are completion experiments, not forecasts or clinical causal models.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from .model import JOINT_INDICES, ModelConfig, SJEPA, token_validity, uniform_mask
from .training import representation_diagnostics, seed_everything

READOUT_JOINTS = (13, 14, 15, 16, 25, 26, 27, 28, 29, 30, 31, 32)
BRANCHES = ("own", "patch", "full")
TARGET_TYPES = ("coordinate", "latent")


@dataclass(frozen=True)
class ContextConfig:
    pretrain_updates: int = 512
    predictor_updates: int = 512
    batch_size: int = 16
    validation_interval: int = 64
    validation_batches: int = 4
    frames: int = 32
    patch_frames: int = 4
    dim: int = 64
    depth: int = 2
    heads: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 0.05
    ema: float = 0.996
    vicreg_weight: float = 0.1
    mask_ratio: float = 0.5
    min_own_context: int = 2
    device: str = "cpu"

    def __post_init__(self):
        if (self.frames, self.patch_frames) != (32, 4):
            raise ValueError("Protocol fixes 32 frames, four-frame patches, and 128 tokens")
        if self.pretrain_updates < 0 or self.predictor_updates < 1:
            raise ValueError("Need pretrain_updates >= 0 and predictor_updates >= 1")
        if self.batch_size < 2 or min(self.validation_interval, self.validation_batches) < 1:
            raise ValueError("Need batch_size >= 2 and positive validation settings")
        if self.min_own_context != 2:
            raise ValueError("Protocol requires two valid visible own-joint patches")
        if not 0 < self.mask_ratio < 1 or not 0 <= self.ema < 1:
            raise ValueError("Invalid masking fraction or EMA momentum")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("Invalid optimizer settings")
        self.model_config()

    def model_config(self):
        return ModelConfig(frames=self.frames, patch_frames=self.patch_frames,
                           dim=self.dim, depth=self.depth, heads=self.heads,
                           mask_ratio=self.mask_ratio, vicreg_weight=self.vicreg_weight)


def _array(x, cfg):
    array = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
    array = np.asarray(array, dtype=np.float32)
    if array.ndim != 4 or tuple(array.shape[1:]) != (32, 16, 2) or len(array) == 0:
        raise ValueError("Expected nonempty [windows,32,16,2] in JOINT_INDICES order")
    if np.isinf(array).any():
        raise ValueError("Infinite coordinates are corrupt input; use NaN for missingness")
    valid = np.isfinite(array).reshape(len(array), 8, 4, 16, 2).all(axis=(2, 4))
    if (valid.sum(axis=(1, 2)) == 0).any():
        raise ValueError("An entirely unobserved window cannot be encoded")
    return array


def context_access(mask, valid, target_indices, branch):
    """Allowed observed keys for each [B] target, before contextualization."""
    if branch not in BRANCHES or mask.shape != valid.shape or mask.dtype != torch.bool:
        raise ValueError("Invalid branch, validity, or mask")
    if target_indices.shape != (len(mask),) or ((target_indices < 0) | (target_indices >= 128)).any():
        raise ValueError("One target grid index in [0,128) per example is required")
    slots = torch.arange(128, device=mask.device)[None]
    own = slots.remainder(16) == target_indices[:, None].remainder(16)
    same_patch = slots.div(16, rounding_mode="floor") == target_indices[:, None].div(16, rounding_mode="floor")
    restriction = own if branch == "own" else (own | same_patch if branch == "patch" else torch.ones_like(own))
    return (~mask) & valid & restriction


@torch.no_grad()
def _restricted_encode(encoder, x, allowed):
    """Gather allowed keys before layer one; padding remains in every layer.

    Empty rows get an unscored zero placeholder. Tokenization is independent
    per joint patch. No globally contextualized features enter restricted arms.
    """
    counts = allowed.sum(dim=1)
    width = max(1, int(counts.max().item()))
    order = torch.argsort((~allowed).to(torch.int32), dim=1, stable=True)[:, :width]
    padding = torch.arange(width, device=x.device)[None] >= counts[:, None]
    tokens = encoder.tokenize(x).gather(1, order[..., None].expand(-1, -1, encoder.cfg.dim))
    tokens = tokens.masked_fill(padding[..., None], 0.)
    safe_padding = padding.clone()
    safe_padding[counts == 0, 0] = False
    memory = encoder.transformer(tokens, src_key_padding_mask=safe_padding)
    return memory.masked_fill(padding[..., None], 0.), safe_padding, counts


class QueryHead(nn.Module):
    """Cross-attention queries contain positions, never target measurements."""
    def __init__(self, cfg, output_dim):
        super().__init__()
        self.queries = nn.Embedding(128, cfg.dim)
        nn.init.normal_(self.queries.weight, std=.02)
        self.attention = nn.MultiheadAttention(cfg.dim, cfg.heads, dropout=0., batch_first=True)
        self.norm = nn.LayerNorm(cfg.dim)
        self.output = nn.Sequential(nn.Linear(cfg.dim, cfg.dim * 2), nn.GELU(), nn.Linear(cfg.dim * 2, output_dim))

    def forward(self, memory, padding, indices):
        query = self.queries(indices)
        attended, _ = self.attention(query, memory, memory, key_padding_mask=padding, need_weights=False)
        return self.output(self.norm(query + attended))


class ContextBundle(nn.Module):
    """Frozen pretrained networks and six independently fitted matched heads."""
    def __init__(self, backbone, config, *, seed=42, provenance=None):
        super().__init__()
        expected = config.model_config()
        for field in ("frames", "patch_frames", "joints", "channels", "dim", "depth", "heads"):
            if getattr(backbone.cfg, field) != getattr(expected, field):
                raise ValueError(f"Backbone configuration differs at {field}")
        self.config = config
        self.backbone = copy.deepcopy(backbone).requires_grad_(False).eval()
        self.heads = nn.ModuleDict()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 3000)
            for kind, width in (("coordinate", 8), ("latent", config.dim)):
                initial = QueryHead(config, width)
                for branch in BRANCHES:
                    self.heads[f"{kind}_{branch}"] = copy.deepcopy(initial)
        self.provenance = dict(provenance or {})
        self.history = {}
        self.to(config.device)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    @torch.no_grad()
    def targets(self, x):
        patches = x.reshape(len(x), 8, 4, 16, 2).permute(0, 1, 3, 2, 4).reshape(len(x), 128, 8)
        teacher = self.backbone.target_encoder(x)
        log_teacher = F.log_softmax((teacher - self.backbone.target_center) /
                                    self.backbone.cfg.target_temperature, dim=-1)
        return {"coordinate": patches, "latent": log_teacher}

    def predict_joint(self, x, mask, joint_slots, *, targets=None, full_memory=None):
        """Predict one joint per example at eight positions; filter without redraw.

        Loss grids [B,8] contain NaN outside common eligible requested targets.
        The training joint is sampled independently of mask and observations.
        """
        if joint_slots.shape != (len(x),) or ((joint_slots < 0) | (joint_slots >= 16)).any():
            raise ValueError("One whitelist joint slot per example is required")
        if mask.shape != (len(x), 128) or mask.dtype != torch.bool:
            raise ValueError("Expected boolean uniform grid mask [B,128]")
        valid = token_validity(x, self.backbone.cfg)
        indices = torch.arange(8, device=x.device)[None] * 16 + joint_slots[:, None]
        requested = mask.gather(1, indices)
        target_valid = valid.gather(1, indices)
        own_allowed = context_access(mask, valid, indices[:, 0], "own")
        own_counts = own_allowed.sum(1)
        eligible = requested & target_valid & (own_counts[:, None] >= self.config.min_own_context)
        rows, patches = eligible.nonzero(as_tuple=True)
        observed = (~mask) & valid
        patch_counts = observed.reshape(len(x), 8, 16).sum(-1) + own_counts[:, None] - observed.gather(1, indices).long()
        losses = {kind: {branch: x.new_full((len(x), 8), torch.nan) for branch in BRANCHES} for kind in TARGET_TYPES}
        result = dict(losses=losses, eligible=eligible, requested=requested, target_valid=target_valid,
                      own_visible_count=own_counts, full_visible_count=observed.sum(1),
                      patch_visible_count=patch_counts, predictions={})
        if len(rows) == 0:
            return result
        if targets is None:
            targets = self.targets(x)
        own_memory = _restricted_encode(self.backbone.view_encoder, x, own_allowed)
        if full_memory is None:
            full_memory = _restricted_encode(self.backbone.view_encoder, x, observed)
        flat_target = indices[rows, patches]
        patch_allowed = context_access(mask[rows], valid[rows], flat_target, "patch")
        patch_memory = _restricted_encode(self.backbone.view_encoder, x[rows], patch_allowed)
        for kind in TARGET_TYPES:
            expected = targets[kind][rows, flat_target].detach()
            for branch, memory in (("own", own_memory), ("patch", patch_memory), ("full", full_memory)):
                if branch == "patch":
                    prediction = self.heads[f"{kind}_{branch}"](memory[0], memory[1], flat_target[:, None])[:, 0]
                else:
                    prediction = self.heads[f"{kind}_{branch}"](memory[0], memory[1], indices)[rows, patches]
                result["predictions"][f"{kind}_{branch}"] = prediction
                if kind == "coordinate":
                    error = (prediction - expected).square().mean(-1)
                else:
                    log_prediction = F.log_softmax(prediction / self.backbone.cfg.student_temperature, dim=-1)
                    error = (expected.exp() * (expected - log_prediction)).sum(-1).clamp_min(0.)
                losses[kind][branch][rows, patches] = error
        return result


def _groups(values, n):
    if values is None:
        return [np.arange(n)]
    values = np.asarray(values).astype(str)
    if values.shape != (n,):
        raise ValueError("One source ID per input window is required")
    return [np.flatnonzero(values == value) for value in np.unique(values)]


def _sample_rows(rng, groups, count):
    return np.asarray([rng.choice(groups[int(rng.integers(len(groups)))]) for _ in range(count)], dtype=int)


def _matched_mean(grid, eligible):
    counts = eligible.sum(1)
    # Clamp before division. nanmean on an empty row can produce NaN gradients
    # even when the corresponding row is subsequently filtered out.
    per_window = torch.where(eligible, grid, torch.zeros_like(grid)).sum(1) / counts.clamp_min(1)
    return per_window[counts > 0].mean()


def _model_loss_summary(model, x, rows, seed, cfg):
    model.eval()
    records, embeddings = [], []
    with torch.no_grad():
        rng = torch.Generator().manual_seed(seed)
        for batch_rows in rows:
            batch = torch.as_tensor(x[batch_rows], device=cfg.device)
            loss = model.loss(batch, generator=rng)
            records.append({key: float(loss[key]) for key in ("loss", "prediction", "vicreg", "projection_std", "embedding_std")})
            embeddings.append(model.embed(batch).cpu().numpy())
    summary = {"validation_" + key: float(np.mean([row[key] for row in records])) for key in records[0]}
    summary.update({"validation_" + key: value for key, value in representation_diagnostics(np.concatenate(embeddings)).items()})
    with torch.no_grad():
        numerator = sum((p - q).square().sum().item() for p, q in zip(model.view_encoder.parameters(), model.target_encoder.parameters(), strict=True))
        denominator = sum(p.square().sum().item() for p in model.view_encoder.parameters())
    summary["teacher_relative_drift"] = float(np.sqrt(numerator / max(denominator, 1e-12)))
    return summary


def _head_summary(bundle, x, rows, seed):
    bundle.eval()
    rng = torch.Generator().manual_seed(seed)
    values = {f"{kind}_{branch}": [] for kind in TARGET_TYPES for branch in BRANCHES}
    eligible, requested = 0, 0
    with torch.no_grad():
        for batch_rows in rows:
            batch = torch.as_tensor(x[batch_rows], device=bundle.config.device)
            mask = uniform_mask(len(batch), 128, bundle.config.mask_ratio, generator=rng, device=batch.device)
            joint = torch.randint(16, (len(batch),), generator=rng).to(batch.device)
            result = bundle.predict_joint(batch, mask, joint)
            eligible += int(result["eligible"].sum())
            requested += int(result["requested"].sum())
            if result["eligible"].any():
                for kind in TARGET_TYPES:
                    for branch in BRANCHES:
                        values[f"{kind}_{branch}"].append(float(_matched_mean(result["losses"][kind][branch], result["eligible"])))
    return {**{"validation_" + key: float(np.mean(v)) if v else None for key, v in values.items()},
            "validation_eligible_targets": eligible, "validation_requested_targets": requested}


def fit_context_models(x_train, x_validation, *, config=None, seed=42,
                       pretrained_model=None, train_source_ids=None,
                       validation_source_ids=None, train_labels=None,
                       validation_labels=None, provenance=None):
    """Fit fixed budgets on supplied normal train arrays, with validation read-only.

    Source IDs and labels, when supplied, are enforced. Otherwise these remain
    caller assertions, explicitly recorded in provenance. A provided backbone
    is copied and skips pretraining. Validation never changes weights or center.
    Sampling is source-balanced when source IDs are provided.
    """
    cfg = config or ContextConfig()
    train, validation = _array(x_train, cfg), _array(x_validation, cfg)
    if min(len(train), len(validation)) < 2:
        raise ValueError("Training and validation each need at least two windows")
    if (train_source_ids is None) != (validation_source_ids is None):
        raise ValueError("Provide both source ID arrays or neither")
    if train_source_ids is not None and set(map(str, train_source_ids)) & set(map(str, validation_source_ids)):
        raise ValueError("Normal training and validation source IDs overlap")
    for labels, values in ((train_labels, train), (validation_labels, validation)):
        if labels is not None and (len(labels) != len(values) or set(labels) != {"normal"}):
            raise ValueError("Only normal labels may enter training or schedule validation")
    train_groups, val_groups = _groups(train_source_ids, len(train)), _groups(validation_source_ids, len(validation))
    seed_everything(seed)
    rng = np.random.default_rng(seed)
    validation_rng = np.random.default_rng(seed + 4000)
    val_rows = [_sample_rows(validation_rng, val_groups, cfg.batch_size) for _ in range(cfg.validation_batches)]
    started = time.perf_counter()
    history = {"pretrain": [], "predictors": [], "seed": seed, "config": asdict(cfg)}
    model = (copy.deepcopy(pretrained_model) if pretrained_model is not None else SJEPA(cfg.model_config())).to(cfg.device)
    if pretrained_model is None:
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        mask_rng = torch.Generator().manual_seed(seed + 1000)
        position_rng = torch.Generator().manual_seed(seed + 2000)
        history["pretrain"].append({"update": 0, **_model_loss_summary(model, validation, val_rows, seed + 5000, cfg)})
        recent = []
        for update in range(1, cfg.pretrain_updates + 1):
            chosen = _sample_rows(rng, train_groups, cfg.batch_size)
            if len(np.unique(chosen)) == 1:
                chosen[-1] = int(rng.choice(np.flatnonzero(np.arange(len(train)) != chosen[0])))
            model.train()
            optimizer.zero_grad(set_to_none=True)
            losses = model.loss(torch.as_tensor(train[chosen], device=cfg.device), generator=mask_rng,
                                positional_generator=position_rng)
            if not torch.isfinite(losses["loss"]):
                raise FloatingPointError("Nonfinite encoder loss")
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1., error_if_nonfinite=True)
            optimizer.step()
            model.update_target(cfg.ema, losses["target_mean"])
            recent.append({key: float(losses[key].detach()) for key in ("loss", "prediction", "vicreg", "projection_std", "embedding_std")})
            if update % cfg.validation_interval == 0 or update == cfg.pretrain_updates:
                history["pretrain"].append({"update": update,
                    **{"train_" + key: float(np.mean([v[key] for v in recent])) for key in recent[0]},
                    **_model_loss_summary(model, validation, val_rows, seed + 5000, cfg)})
                recent = []
    provenance = {**(provenance or {}), "seed": seed,
                  "normal_only_labels_checked": train_labels is not None and validation_labels is not None,
                  "source_disjointness_checked": train_source_ids is not None,
                  "pretraining": "provided_checkpoint" if pretrained_model is not None else "fixed_update_normal_training",
                  "train_array_sha256": hashlib.sha256(train.tobytes()).hexdigest(),
                  "validation_array_sha256": hashlib.sha256(validation.tobytes()).hexdigest(),
                  "joint_indices": list(JOINT_INDICES), "minimum_own_context": cfg.min_own_context,
                  "target_metrics": {"coordinate": "mean squared normalized XY error", "latent": "KL(teacher || prediction)"},
                  "head_objectives": "six separate objectives and optimizers; frozen shared encoders",
                  "context_reference": "conditional on identical supplied pelvis/torso normalization"}
    identity = hashlib.sha256()
    for name, value in model.state_dict().items():
        identity.update(name.encode())
        identity.update(value.detach().cpu().contiguous().numpy().tobytes())
    provenance["frozen_backbone_sha256"] = identity.hexdigest()
    provenance["context_implementation_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if train_source_ids is not None:
        provenance["train_source_ids"] = sorted(set(map(str, train_source_ids)))
        provenance["validation_source_ids"] = sorted(set(map(str, validation_source_ids)))
    bundle = ContextBundle(model, cfg, seed=seed, provenance=provenance)
    optimizers = {name: torch.optim.AdamW(head.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
                  for name, head in bundle.heads.items()}
    history["predictors"].append({"update": 0, **_head_summary(bundle, validation, val_rows, seed + 6000)})
    head_rng = torch.Generator().manual_seed(seed + 7000)
    # Separate from pretraining draws, so pretrained/untrained controls and
    # different pretraining budgets receive identical head-fitting batches.
    head_sample_rng = np.random.default_rng(seed + 8000)
    recent = []
    for update in range(1, cfg.predictor_updates + 1):
        chosen = _sample_rows(head_sample_rng, train_groups, cfg.batch_size)
        batch = torch.as_tensor(train[chosen], device=cfg.device)
        mask = uniform_mask(len(batch), 128, cfg.mask_ratio, generator=head_rng, device=batch.device)
        joints = torch.randint(16, (len(batch),), generator=head_rng).to(batch.device)
        bundle.train()
        result = bundle.predict_joint(batch, mask, joints)
        if not result["eligible"].any():
            raise ValueError("Training draw has no common eligible target; mask redraw is forbidden")
        record = {"eligible_targets": int(result["eligible"].sum()), "requested_targets": int(result["requested"].sum())}
        for kind in TARGET_TYPES:
            for branch in BRANCHES:
                name = f"{kind}_{branch}"
                loss = _matched_mean(result["losses"][kind][branch], result["eligible"])
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite {name} prediction loss")
                optimizers[name].zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(bundle.heads[name].parameters(), 1., error_if_nonfinite=True)
                optimizers[name].step()
                record[name] = float(loss.detach())
        recent.append(record)
        if update % cfg.validation_interval == 0 or update == cfg.predictor_updates:
            history["predictors"].append({"update": update,
                **{"train_" + key: float(np.mean([row[key] for row in recent])) for key in recent[0]},
                **_head_summary(bundle, validation, val_rows, seed + 6000)})
            recent = []
    history["elapsed_seconds"] = float(time.perf_counter() - started)
    history["pretrain_updates_completed"] = cfg.pretrain_updates if pretrained_model is None else 0
    history["predictor_updates_completed_per_head"] = cfg.predictor_updates
    bundle.history = history
    bundle.eval()
    return bundle, history


def make_mask_bank(config, mask_seed=17, mask_repeats=16):
    """Same bank for every window, independent of pose, validity and batching."""
    if mask_repeats < 1:
        raise ValueError("mask_repeats must be positive")
    return uniform_mask(mask_repeats, 128, config.mask_ratio,
                        generator=torch.Generator().manual_seed(mask_seed))


@torch.no_grad()
def score_context_models(bundle, x, *, mask_seed=17, mask_repeats=16,
                         batch_size=16, joints=READOUT_JOINTS):
    """Paired long table includes invalid requested targets with NaN losses.

    `valid` is identical across all branches and target types. attrs counts a
    requested target once, not twice for its two target types. Window indices
    refer to the supplied array. Distinct physical support is counted downstream.
    """
    array = _array(x, bundle.config)
    joints = tuple(joints)
    if batch_size < 1 or not joints or len(set(joints)) != len(joints) or set(joints) - set(JOINT_INDICES):
        raise ValueError("Use positive batches and distinct whitelisted readout joints")
    bundle.eval()
    device = next(bundle.parameters()).device
    bank = make_mask_bank(bundle.config, mask_seed, mask_repeats).to(device)
    tables = []
    counts = {"requested_targets": 0, "valid_targets": 0, "target_unobserved": 0, "insufficient_own_context": 0}
    for start in range(0, len(array), batch_size):
        batch = torch.as_tensor(array[start:start + batch_size], device=device)
        targets = bundle.targets(batch)
        valid = token_validity(batch, bundle.backbone.cfg)
        for mask_id, one_mask in enumerate(bank):
            mask = one_mask.expand(len(batch), -1)
            full_memory = _restricted_encode(bundle.backbone.view_encoder, batch, (~mask) & valid)
            for joint in joints:
                slots = torch.full((len(batch),), JOINT_INDICES.index(joint), dtype=torch.long, device=device)
                result = bundle.predict_joint(batch, mask, slots, targets=targets, full_memory=full_memory)
                rows, patches = result["requested"].nonzero(as_tuple=True)
                ok = result["eligible"][rows, patches].cpu().numpy()
                observed = result["target_valid"][rows, patches].cpu().numpy()
                own_count = result["own_visible_count"][rows].cpu().numpy()
                reason = np.where(~observed, "target_unobserved", np.where(own_count < 2, "insufficient_own_context", ""))
                counts["requested_targets"] += len(rows)
                counts["valid_targets"] += int(ok.sum())
                counts["target_unobserved"] += int((~observed).sum())
                counts["insufficient_own_context"] += int((observed & (own_count < 2)).sum())
                for kind in TARGET_TYPES:
                    tables.append(pd.DataFrame({
                        "window_index": rows.cpu().numpy().astype(np.int32) + start,
                        "joint": np.full(len(rows), joint, np.int16),
                        "patch_index": patches.cpu().numpy().astype(np.int16),
                        "mask_id": np.full(len(rows), mask_id, np.int16),
                        "target_type": pd.Categorical([kind] * len(rows), categories=TARGET_TYPES),
                        **{"loss_" + branch: result["losses"][kind][branch][rows, patches].cpu().numpy().astype(np.float32) for branch in BRANCHES},
                        "valid": ok, "target_valid": observed,
                        "own_visible_count": own_count.astype(np.int16),
                        "patch_visible_count": result["patch_visible_count"][rows, patches].cpu().numpy().astype(np.int16),
                        "full_visible_count": result["full_visible_count"][rows].cpu().numpy().astype(np.int16),
                        "exclusion_reason": pd.Categorical(reason, categories=("", "target_unobserved", "insufficient_own_context")),
                    }))
    table = pd.concat(tables, ignore_index=True)
    table.attrs.update(counts)
    table.attrs.update(mask_seed=mask_seed, mask_repeats=mask_repeats, joints=list(joints), config=asdict(bundle.config))
    return table


def save_context_bundle(bundle, path, *, provenance=None, overwrite=False):
    """Save all heads, teacher/center, configuration, history and input identity."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(format_version=1, config=asdict(bundle.config),
                   backbone_config=asdict(bundle.backbone.cfg), state_dict=bundle.state_dict(),
                   provenance={**bundle.provenance, **(provenance or {})}, history=bundle.history)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return path


def load_context_bundle(path, *, expected_provenance=None, device="cpu"):
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("format_version") != 1:
        raise ValueError("Unsupported context-bundle format")
    if any(payload["provenance"].get(key) != value for key, value in (expected_provenance or {}).items()):
        raise ValueError("Context-bundle provenance differs from the requested experiment")
    config = ContextConfig(**{**payload["config"], "device": device})
    backbone = SJEPA(ModelConfig(**payload["backbone_config"]))
    bundle = ContextBundle(backbone, config, provenance=payload["provenance"])
    bundle.load_state_dict(payload["state_dict"])
    bundle.history = payload["history"]
    return bundle.eval()
