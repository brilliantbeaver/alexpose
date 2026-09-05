"""E3(b) -- Can training INDUCE reflection-equivariance? Stage-0 normal-only retrain, +/- reflection aug.

E1: the frozen curriculum encoder is not reflection-equivariant (free-readout mirror slope ~= -0.70) and
    the signed axis is not decodable above the untrained floor. Crucially, the canonical S-JEPA recipe
    trains with rotation (+/-8deg) and translation augmentation but NO reflection (geometric_view is called
    with flip_probability=0.0 for both views; the recipe comment: "Flip defaults to off because laterality
    can matter for stroke"). So reflection symmetry was never in the SSL training signal.

E3(a): a frame-averaging wrapper makes the encoder EXACTLY equivariant with zero retraining (architecture).
E3(b) asks the softer, complementary question: if we ADD reflection augmentation to SSL pretraining, does
    approximate equivariance EMERGE (Benton et al. 2020, augmentation -> approximate equivariance)?

Design -- a clean one-switch ablation. We reproduce the canonical Stage-0 normal-only run EXACTLY (same
270 normal sequences -- verified identical sequence_ids to sjepa_normal.pt -- same recipe, seed 42, MPS,
300 epochs, same mask / EMA / VICReg / group loss), and toggle only reflection augmentation:
    ARM OFF (FLIP_PROB=0.0): identical to canonical Stage-0 (validation: probe ~= sjepa_normal.pt).
    ARM ON  (FLIP_PROB=0.5): with prob p per sample, the ENTIRE sample (coordinates AND validity) is
        replaced by its anatomical mirror M before view augmentation, masking, and the JEPA target.
        Because view and target are mirrored CONSISTENTLY, the JEPA task stays self-consistent and the
        data marginal becomes reflection-symmetric -> equivariance pressure (not invariance).

Both arms use identical code / seed / hardware; the ONLY difference is the mirror. Each arm's target
encoder is saved and later probed on E1's instrument (free lane A + frame-averaged Phi + mirror slope).
This is an explicit PROOF-OF-CONCEPT PROTOTYPE: transductive, single seed, normal-only Stage 0.

Run:  FLIP_PROB=0.0 SJEPA_NORMAL_EPOCHS=300 python e3b_reflection_augmented_retrain.py
      FLIP_PROB=0.5 SJEPA_NORMAL_EPOCHS=300 python e3b_reflection_augmented_retrain.py
Source video is the independent unit; folder labels are dataset annotations, not diagnoses.
"""
import os, sys, math, json, time, hashlib, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _probe_common import (  # noqa: E402
    ART, MASK_KEYPOINTS, FULL_MIRROR_PAIRS, pose_records_from_cache,
    prepare_sequence, anatomical_mirror,
)

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------ config (canonical recommended profile)
FRAMES, SEGMENT_LENGTH = 64, 4
EMBED_DIM, ENCODER_DEPTH, PREDICTOR_DEPTH, HEADS = 96, 4, 2, 4
NORMAL_EPOCHS = int(os.getenv("SJEPA_NORMAL_EPOCHS", "300"))
SAMPLES_PER_CONDITION = 4
EMA_START = 0.999
MASK_FRACTION = 0.60
VICREG_WEIGHT, GROUP_WEIGHT, GROUP_MARGIN = 0.05, 0.25, 1.0
NORMAL_LR = 0.001
SEED = 42
FLIP_PROB = float(os.getenv("FLIP_PROB", "0.0"))
SEGMENTS = FRAMES // SEGMENT_LENGTH
CONFIG = {"frames": FRAMES, "joints": 33, "coordinate_dim": 3, "segment_length": SEGMENT_LENGTH,
          "embed_dim": EMBED_DIM, "encoder_depth": ENCODER_DEPTH, "predictor_depth": PREDICTOR_DEPTH,
          "heads": HEADS}
TAG = f"flip{FLIP_PROB:.2f}".replace(".", "p")
OUT_CKPT = ART / f"sjepa_normal_e3b_{TAG}.pt"

device = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available() else "cpu")
torch.manual_seed(SEED)
np.random.seed(SEED)

# ------------------------------------------------------------------ full training model (from nb04 recipe)
import copy


class SkeletonPatchEncoder(nn.Module):
    def __init__(self, frames=64, joints=33, coordinate_dim=3, segment_length=4,
                 embed_dim=64, depth=2, heads=4, dropout=0.0):
        super().__init__()
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
            kept = keep_mask.sum(dim=1)
            if not torch.equal(kept, kept[:1].expand_as(kept)):
                raise ValueError("Each sample must keep the same number of tokens")
            flat = flat[keep_mask].reshape(batch, int(kept[0]), self.embed_dim)
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
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)
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
        self.net = nn.Sequential(nn.Linear(dimension, dimension), nn.GELU(), nn.Linear(dimension, dimension))

    def forward(self, x):
        return self.net(x)


# ------------------------------------------------------------------ losses / masking (from recipe)
def sjepa_cross_entropy(predicted, targets, center, predictor_temperature=0.10, target_temperature=0.06):
    target_prob = torch.softmax((targets - center[None, None, :]) / target_temperature, dim=-1).detach()
    prediction_log_prob = torch.log_softmax(predicted / predictor_temperature, dim=-1)
    return -(target_prob * prediction_log_prob).sum(dim=-1).mean()


def cosine_ema(step, total_steps, start=0.996, end=1.0):
    progress = min(max(step / max(total_steps - 1, 1), 0.0), 1.0)
    return end - (end - start) * (math.cos(math.pi * progress) + 1.0) / 2.0


def uniform_neurologic_mask(valid_patch, mask_fraction=0.60, seed=None):
    valid_patch = np.asarray(valid_patch, dtype=bool)
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
    for b in range(len(mask)):
        candidates = np.flatnonzero(eligible[b].reshape(-1))
        chosen = rng.choice(candidates, size=n_mask, replace=False)
        mask[b].reshape(-1)[chosen] = True
    forbidden = sorted(set(range(33)).difference(MASK_KEYPOINTS))
    assert not mask[:, :, forbidden].any()
    return mask


def authorized_pool(tokens, valid_patch):
    batch, segments, _, dim = tokens.shape
    selected = tokens[:, :, MASK_KEYPOINTS].reshape(batch, -1, dim)
    weights = valid_patch[:, :, MASK_KEYPOINTS].reshape(batch, -1).to(tokens.dtype)
    denom = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
    return (selected * weights.unsqueeze(-1)).sum(dim=1) / denom


def off_diagonal(matrix):
    n, m = matrix.shape
    return matrix.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def vicreg_terms(first, second, gamma=1.0, eps=1e-4):
    invariance = F.mse_loss(first, second)
    first_std = torch.sqrt(first.var(dim=0, unbiased=False) + eps)
    second_std = torch.sqrt(second.var(dim=0, unbiased=False) + eps)
    variance = 0.5 * (F.relu(gamma - first_std).mean() + F.relu(gamma - second_std).mean())
    fc = first - first.mean(dim=0)
    sc = second - second.mean(dim=0)
    denom = max(len(first) - 1, 1)
    first_cov = fc.T @ fc / denom
    second_cov = sc.T @ sc / denom
    covariance = (off_diagonal(first_cov).square().sum()
                  + off_diagonal(second_cov).square().sum()) / (2.0 * first.shape[1])
    return 25.0 * invariance + 25.0 * variance + covariance, invariance, variance, covariance


def condition_group_terms(reps, condition_ids, margin=1.0):
    unique = torch.unique(condition_ids)
    zero = reps.sum() * 0.0
    if len(unique) < 2:
        return zero, zero
    normalized = F.normalize(reps, dim=1)
    centroids = torch.stack([F.normalize(normalized[condition_ids == v].mean(dim=0), dim=0) for v in unique])
    compactness = torch.stack([
        (normalized[condition_ids == v] - centroids[i]).square().sum(dim=1).mean()
        for i, v in enumerate(unique)]).mean()
    pairwise = (centroids[:, None] - centroids[None, :]).square().sum(dim=-1).clamp_min(1e-12).sqrt()
    upper = torch.triu(torch.ones_like(pairwise, dtype=torch.bool), diagonal=1)
    separation = F.relu(margin - pairwise[upper]).square().mean()
    return compactness, separation


def geometric_view(x, max_degrees=8.0, translate=0.03):
    """Rotation about vertical + translation (canonical view aug; reflection handled at sample level)."""
    view = x.clone()
    present = view.abs().sum(dim=-1) > 1e-8
    batch = len(view)
    angles = (torch.rand(batch, device=x.device) * 2.0 - 1.0) * math.radians(max_degrees)
    cosine, sine = torch.cos(angles), torch.sin(angles)
    ox, oz = view[..., 0].clone(), view[..., 2].clone()
    view[..., 0] = cosine[:, None, None] * ox + sine[:, None, None] * oz
    view[..., 2] = -sine[:, None, None] * ox + cosine[:, None, None] * oz
    offsets = (torch.rand(batch, 1, 1, 2, device=x.device) * 2.0 - 1.0) * translate
    view[..., :2] += offsets
    return view.masked_fill(~present[..., None], 0.0)


# ------------------------------------------------------------------ data (canonical 270 normal, + mirror)
SIGMA = np.arange(33)
for a, b in FULL_MIRROR_PAIRS:
    SIGMA[a], SIGMA[b] = b, a

print(f"[e3b] FLIP_PROB={FLIP_PROB} epochs={NORMAL_EPOCHS} device={device} tag={TAG}")
recs = [r for r in pose_records_from_cache() if r["condition"] == "normal"]
prep = [prepare_sequence(r["raw"], FRAMES) for r in recs]
xyz = np.stack([p[0] for p in prep]).astype(np.float32)
valid = np.stack([p[1] for p in prep])
cov = valid[:, :, MASK_KEYPOINTS].mean(axis=(1, 2))
keep = np.where(cov >= 0.50)[0]
xyz, valid = xyz[keep], valid[keep]
ids = [recs[i]["sequence_id"] for i in keep]
# exact mirror (matches probe operator M): anatomical_mirror on raw, then prepare
prep_m = [prepare_sequence(anatomical_mirror(recs[i]["raw"]), FRAMES) for i in keep]
xyz_mir = np.stack([p[0] for p in prep_m]).astype(np.float32)
valid_mir = np.stack([p[1] for p in prep_m])
canon = torch.load(ART / "sjepa_normal.pt", map_location="cpu", weights_only=False)
assert sorted(ids) == sorted(map(str, canon["sequence_ids"])), "normal cohort != canonical sjepa_normal.pt"
print(f"[e3b] normal sequences: {len(ids)} (verified identical to sjepa_normal.pt)")

model = SJEPAGait(**CONFIG).to(device)
vicreg_projector = VICRegProjector(EMBED_DIM).to(device)
assert not any(p.requires_grad for p in model.target_encoder.parameters())


def balanced_epoch_batches(rng):
    n = len(xyz)
    per = SAMPLES_PER_CONDITION
    steps = max(1, int(np.ceil(n / per)))
    required = steps * per
    order = []
    while len(order) < required:
        order.append(rng.permutation(n))
    order = np.concatenate(order)[:required]
    for s in range(steps):
        take = order[s * per:(s + 1) * per]
        yield take


def train():
    trainable = [*[p for p in model.view_encoder.parameters() if p.requires_grad],
                 *[p for p in model.predictor.parameters() if p.requires_grad],
                 *list(vicreg_projector.parameters())]
    optimizer = torch.optim.AdamW(trainable, lr=NORMAL_LR, betas=(0.9, 0.95), weight_decay=0.05)
    steps_per_epoch = max(1, int(np.ceil(len(xyz) / SAMPLES_PER_CONDITION)))
    total_steps = NORMAL_EPOCHS * steps_per_epoch
    warmup = max(1, min(steps_per_epoch, total_steps // 10))

    def lr_factor(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(total_steps - warmup - 1, 1)
        return 0.5 + 0.5 * (1.0 + math.cos(math.pi * progress)) / 2.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_factor)
    rng = np.random.default_rng(42 + 0)             # stage 0
    flip_rng = np.random.default_rng(20240517)      # separate stream; unused when FLIP_PROB==0
    history = []
    global_step = 0
    t0 = time.time()
    for epoch in range(NORMAL_EPOCHS):
        model.train(); vicreg_projector.train()
        rows = []
        for take in balanced_epoch_batches(rng):
            xb, vb = xyz[take].copy(), valid[take].copy()
            if FLIP_PROB > 0:
                flip = flip_rng.random(len(take)) < FLIP_PROB
                if flip.any():
                    idx = np.where(flip)[0]
                    xb[idx] = xyz_mir[take][idx]
                    vb[idx] = valid_mir[take][idx]
            coordinates = torch.tensor(xb, dtype=torch.float32, device=device)
            valid_t = torch.tensor(vb, dtype=torch.bool, device=device)
            labels_t = torch.zeros(len(take), dtype=torch.long, device=device)  # normal-only
            valid_patch = valid_t.reshape(len(valid_t), SEGMENTS, SEGMENT_LENGTH, 33).all(dim=2)
            if int(valid_patch[:, :, MASK_KEYPOINTS].reshape(len(valid_t), -1).sum(dim=1).min()) < 2:
                raise ValueError("sequence needs >=2 valid authorized patches")
            mask_np = uniform_neurologic_mask(valid_patch.cpu().numpy(), MASK_FRACTION,
                                              seed=42 + global_step)
            target_mask = torch.tensor(mask_np, dtype=torch.bool, device=device)
            view_a = geometric_view(coordinates)
            view_b = geometric_view(coordinates)

            prediction, target = model(view_a, coordinates, target_mask)
            jepa_loss = sjepa_cross_entropy(prediction, target, model.target_center)

            tokens_a = model.view_encoder(view_a).reshape(len(view_a), SEGMENTS, 33, EMBED_DIM)
            tokens_b = model.view_encoder(view_b).reshape(len(view_b), SEGMENTS, 33, EMBED_DIM)
            pa = authorized_pool(tokens_a, valid_patch)
            pb = authorized_pool(tokens_b, valid_patch)
            vic_loss, inv, var, cov_ = vicreg_terms(vicreg_projector(pa), vicreg_projector(pb))
            comp, sep = condition_group_terms(pa, labels_t, margin=GROUP_MARGIN)
            group_loss = comp + sep
            total = jepa_loss + VICREG_WEIGHT * vic_loss + GROUP_WEIGHT * group_loss
            if not torch.isfinite(total):
                raise FloatingPointError(f"non-finite loss at step {global_step}")

            optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step(); scheduler.step()
            model.update_target(cosine_ema(global_step, total_steps, start=EMA_START, end=1.0))
            model.update_center(target, beta=0.9)
            rows.append(float(jepa_loss.detach().cpu()))
            global_step += 1
        if (epoch + 1) % 20 == 0 or epoch == 0 or epoch == NORMAL_EPOCHS - 1:
            print(f"[e3b {TAG}] epoch {epoch+1:03d}/{NORMAL_EPOCHS}  JEPA {np.mean(rows):.4f}  "
                  f"lr {optimizer.param_groups[0]['lr']:.2e}  elapsed {time.time()-t0:.0f}s", flush=True)
        history.append({"epoch": epoch + 1, "jepa": float(np.mean(rows))})
    return history, total_steps, time.time() - t0


history, total_steps, wall = train()
ckpt = {
    "model_state": model.state_dict(),
    "vicreg_projector_state": vicreg_projector.state_dict(),
    "config": CONFIG,
    "mode": "real",
    "mask_keypoints": MASK_KEYPOINTS,
    "training_stage": "normal_only_e3b",
    "flip_prob": FLIP_PROB,
    "sequence_ids": ids,
    "seed": SEED,
    "normal_epochs": NORMAL_EPOCHS,
    "total_steps": total_steps,
    "wall_seconds": wall,
    "parent_reference": "sjepa_normal.pt",
    "history": history,
    "note": ("E3(b) Stage-0 normal-only retrain with sample-level consistent reflection augmentation "
             f"(FLIP_PROB={FLIP_PROB}). Identical to canonical Stage-0 except the mirror toggle. Transductive, "
             "single-seed prototype. Source video is the independent unit; folder labels are dataset "
             "annotations, not diagnoses."),
}
torch.save(ckpt, OUT_CKPT)
print(f"[e3b {TAG}] wrote {OUT_CKPT}  ({wall:.0f}s, final JEPA {history[-1]['jepa']:.4f})")
