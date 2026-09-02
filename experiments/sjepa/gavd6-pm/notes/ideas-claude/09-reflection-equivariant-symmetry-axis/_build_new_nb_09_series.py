"""Bootstrap builder for the new_nb_09_* series (Idea 9, Arm 2).

The executed notebooks are now the maintained tutorial sources. This script is retained to document
their original executable structure and to bootstrap a missing notebook. It refuses to overwrite an
existing notebook because doing so would delete stored outputs and restore older explanatory prose.

Run only after moving or deleting the specific target:

    python3 _build_new_nb_09_series.py 03

It can emit four notebooks at the gavd root (parents[3]):

    new_nb_09_00_methodology_and_contract.ipynb
    new_nb_09_01_mechanism_and_smoke_validation.ipynb
    new_nb_09_02_real_multiseed_equivariant_training.ipynb
    new_nb_09_03_evaluation_results_discussion.ipynb

WHY A NEW SERIES INSTEAD OF EDITING NOTEBOOK 04
Section 8 of nb_09b tells the reader to paste the antisymmetric head, the anatomical mirror, and the
equivariance loss into 04_pretrain_sjepa_on_normal.ipynb's training-step cell. That would mutate the
notebook that produced the locked baseline lineage, so the baseline and the experiment would no longer be
separable. This series carries the same four steps as its own executable path and leaves notebook 04
untouched.

WHY THE PRIMARY ENDPOINT IS NOT R-SQUARED
Arm 1's real bundle reports a FAILED y-quality gate: the signed target's between-source variance fraction
is 0.0747 against a pre-committed 0.30 threshold, and every learned lane is negative. By item 09's own
pre-registered rule, held-out-source R-squared is uninterpretable on this cohort, so a D0-versus-E1
contrast scored on R-squared alone could not support a conclusion in either direction. This series
therefore pre-registers a label-free, parameter-free primary endpoint, the normalised mirror-consistency
residual rho, which is exactly the quantity L_equiv pressures and needs no target variance to be readable.
R-squared is still reported, always beside the gate verdict.

THE THREE PLUMBING CORRECTIONS THIS SERIES MAKES
1. Notebook 04 hardcodes the seed in two places, `np.random.default_rng(42 + 1000 * stage["stage"])` and
   `"seed": 42` in fingerprint_payload. Copied verbatim, five seeds would produce five identical runs and
   five identical fingerprints, silently destroying the seed-spread control. The seed is threaded here.
2. `equiv_weight` and `equiv_on` join fingerprint_payload so every rung gets its own fingerprint and none
   can be confused with the ea59fea0 baseline lineage.
3. Every rung writes a durable checkpoint and a per-rung JSON, so an interrupted ladder resumes instead of
   restarting.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

_CELL_N = [0]


def _next_id(prefix):
    _CELL_N[0] += 1
    return f"{prefix}{_CELL_N[0]:02d}"


def md(text):
    return {"cell_type": "markdown", "id": _next_id("md"), "metadata": {},
            "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "id": _next_id("code"), "metadata": {},
            "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


def write_notebook(path, cells):
    if path.exists():
        raise FileExistsError(
            f"Refusing to overwrite maintained notebook {path}. "
            "It contains executed outputs and reviewed tutorial prose. "
            "Move or delete the target explicitly before using this bootstrap builder."
        )
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3 (ipykernel)",
                           "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {path}  ({len(cells)} cells)")


# ---------------------------------------------------------------------------
# Shared code blocks. Pasted inline into each notebook so every notebook stays
# self-contained and Colab-safe, matching the nb_05a / nb_09a convention.
# ---------------------------------------------------------------------------

COLAB_BADGE = (
    "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
    "(https://colab.research.google.com/github/brilliantbeaver/alexpose/blob/main/"
    "experiments/sjepa/gavd6-pm/{name}.ipynb)"
)

SETUP = '''
from pathlib import Path
import os, sys, json, math, hashlib, copy, time, warnings

import numpy as np
import pandas as pd

RANDOM_SEED = 42


def find_project_root(start=None):
    env_root = os.getenv("ALEXPOSE_ROOT")
    if env_root:
        candidate = Path(env_root).expanduser().resolve()
        if (candidate / ".git").exists() and (candidate / "data" / "gavd").exists():
            return candidate
    start = Path(start or Path.cwd()).resolve()
    for candidate in [start, *start.parents]:
        if (candidate / ".git").exists() and (candidate / "data" / "gavd").exists():
            return candidate
    return start


PROJECT_ROOT = find_project_root()
TUTORIAL_DIR = PROJECT_ROOT / "experiments" / "sjepa" / "gavd6-pm"
try:
    from dotenv import load_dotenv
    load_dotenv(TUTORIAL_DIR / ".env", override=False)
    load_dotenv(PROJECT_ROOT / ".env", override=False)
except Exception:
    pass

TRUTHY = {"1", "true", "yes", "on"}
REQUESTED_MODE = os.getenv("GAVD_MODE", "smoke").strip().lower()
INCLUDE_AUGMENTED = os.getenv("SJEPA_INCLUDE_AUGMENTED_NORMAL", "0").strip().lower() in TRUTHY
ARTIFACT_ROOT = Path(os.getenv("GAVD_ARTIFACT_DIR", TUTORIAL_DIR / "work" / "artifacts")).expanduser()
CACHE_DIR = Path(os.getenv("GAVD_CACHE_DIR", TUTORIAL_DIR / "work" / "cache")).expanduser()
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

CONDITIONS = ["normal", "parkinsons", "stroke", "myopathic", "cerebralpalsy"]
MASK_KEYPOINTS = [11, 12, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]

# The six authorised left/right pairs the head, the signed target, and rho are all built from.
LEFT_RIGHT_PAIRS = [(11, 12), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32)]
# The full 16-pair anatomical mirror, matching notebook 04's geometric_view flip.
FULL_MIRROR_PAIRS = [
    (1, 4), (2, 5), (3, 6), (7, 8), (9, 10),
    (11, 12), (13, 14), (15, 16), (17, 18), (19, 20), (21, 22),
    (23, 24), (25, 26), (27, 28), (29, 30), (31, 32),
]

# Arm-2 output lives in its own directory. Nothing here ever writes a baseline artifact name.
ARM2_DIR_NAME = "idea9_arm2"
BASELINE_CHECKPOINT_NAME = (
    "sjepa_curriculum_final_augmented.pt" if INCLUDE_AUGMENTED else "sjepa_curriculum_final.pt"
)


def artifact_dir_for(mode):
    return ARTIFACT_ROOT / mode
'''

MODEL_CLASSES = '''
import torch
from torch import nn
from torch.nn import functional as F


class SkeletonPatchEncoder(nn.Module):
    def __init__(self, frames=64, joints=33, coordinate_dim=3, segment_length=4,
                 embed_dim=96, depth=4, heads=4, dropout=0.0):
        super().__init__()
        if frames % segment_length:
            raise ValueError("frames must be divisible by segment_length")
        self.frames, self.joints = frames, joints
        self.coordinate_dim, self.segment_length = coordinate_dim, segment_length
        self.segments = frames // segment_length
        self.embed_dim = embed_dim
        self.patch_embed = nn.Linear(segment_length * coordinate_dim, embed_dim)
        self.time_pos = nn.Parameter(torch.randn(self.segments, embed_dim) * 0.02)
        self.joint_pos = nn.Parameter(torch.randn(joints, embed_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=heads, dim_feedforward=embed_dim * 4,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
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
            kept = keep_mask.sum(dim=1)
            if not torch.equal(kept, kept[:1].expand_as(kept)):
                raise ValueError("Each sample must keep the same number of tokens")
            flat = flat[keep_mask].reshape(batch, int(kept[0]), self.embed_dim)
        return self.norm(self.blocks(flat))


class SkeletonPredictor(nn.Module):
    def __init__(self, segments, joints, encoder_dim=96, predictor_dim=96,
                 depth=2, heads=4, dropout=0.0):
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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
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
                 embed_dim=96, encoder_depth=4, predictor_depth=2, heads=4):
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
        for target_parameter, view_parameter in zip(
                self.target_encoder.parameters(), self.view_encoder.parameters()):
            target_parameter.mul_(momentum).add_(view_parameter, alpha=1.0 - momentum)

    @torch.no_grad()
    def update_center(self, targets, beta=0.9):
        self.target_center.mul_(beta).add_(targets.mean(dim=(0, 1)), alpha=1.0 - beta)


def sjepa_cross_entropy(predicted, targets, center,
                        predictor_temperature=0.10, target_temperature=0.06):
    target_prob = torch.softmax(
        (targets - center[None, None, :]) / target_temperature, dim=-1).detach()
    prediction_log_prob = torch.log_softmax(predicted / predictor_temperature, dim=-1)
    return -(target_prob * prediction_log_prob).sum(dim=-1).mean()


def cosine_ema(step, total_steps, start=0.996, end=1.0):
    progress = min(max(step / max(total_steps - 1, 1), 0.0), 1.0)
    return end - (end - start) * (math.cos(math.pi * progress) + 1.0) / 2.0
'''

EQUIV_CODE = '''
def anatomical_mirror_coords(coords):
    """Reflect the RAW skeleton: negate the sideways x coordinate and swap each left/right landmark.

    coords: [B, FRAMES, 33, 3] tensor. Differentiable passthrough, so gradients flow through the mirror.
    """
    mirrored = coords.clone()
    mirrored[..., 0] = -mirrored[..., 0]
    index = list(range(33))
    for left, right in FULL_MIRROR_PAIRS:
        index[left], index[right] = right, left
    return mirrored[:, :, index, :]


class AntisymmetricHead(nn.Module):
    """s = sum over pairs of ( f(L) - f(R) ), with f shared across joints and sides.

    Difference only, so the head negates under a swap of its own inputs by construction. In Arm 2 the
    head's parameters are trainable and join the optimiser, so L_equiv can shape the encoder and the
    head together.
    """

    def __init__(self, embed_dim, out_dim=4, hidden=32, pairs=LEFT_RIGHT_PAIRS):
        super().__init__()
        self.pairs = list(pairs)
        self.f = nn.Sequential(nn.Linear(embed_dim, hidden), nn.GELU(), nn.Linear(hidden, out_dim))

    def per_joint_feature(self, tokens):
        return tokens.mean(dim=1)

    def s_from_perjoint(self, per_joint):
        out = 0.0
        for left, right in self.pairs:
            out = out + (self.f(per_joint[:, left, :]) - self.f(per_joint[:, right, :]))
        return out

    def forward(self, tokens):
        return self.s_from_perjoint(self.per_joint_feature(tokens))

    def swapped(self, tokens):
        """Wiring self-check only. Returns -forward(tokens) by construction, so it trains nothing."""
        per_joint = self.per_joint_feature(tokens)
        swapped = per_joint.clone()
        for left, right in self.pairs:
            swapped[:, left, :] = per_joint[:, right, :]
            swapped[:, right, :] = per_joint[:, left, :]
        return self.s_from_perjoint(swapped)


def encoder_tokens_for_head(view_encoder, coords, segments, joints, embed_dim):
    return view_encoder(coords).reshape(len(coords), segments, joints, embed_dim)


def mirror_pair_signals(readout, view_encoder, coords, segments, embed_dim, joints=33):
    """Return ( s(enc(x)), s(enc(Mx)) ) with M the anatomical mirror on the RAW skeleton.

    The mirror is applied to raw coordinates and BOTH versions are run through the view encoder. A
    head-only token swap would be identically zero for every input and every parameter, so it would train
    nothing at all. Going through the encoder makes the residual a genuine constraint on encoder weights.
    """
    tokens = encoder_tokens_for_head(view_encoder, coords, segments, joints, embed_dim)
    tokens_mirrored = encoder_tokens_for_head(
        view_encoder, anatomical_mirror_coords(coords), segments, joints, embed_dim)
    return readout(tokens), readout(tokens_mirrored)


def equivariance_loss_absolute(head, view_encoder, coords, segments, embed_dim, joints=33):
    """nb_09b's term exactly as written: L = mean( ( s(enc(Mx)) + s(enc(x)) )^2 ).

    This has a degenerate solution. Because the head is trainable and shares the objective, the cheapest
    way to shrink the squared residual is to shrink s itself, which costs the encoder nothing. Notebook 01
    measures that collapse happening, which is why this variant is kept only as the documented baseline
    for comparison and is not used for the real ladder.
    """
    s, s_mirrored = mirror_pair_signals(head, view_encoder, coords, segments, embed_dim, joints)
    return ((s_mirrored + s) ** 2).mean()


def equivariance_loss_normalized(head, view_encoder, coords, segments, embed_dim,
                                 joints=33, eps=1e-6):
    """The scale-invariant repair, with the trainable head retained.

    L = mean_seq [ ||s(enc(Mx)) + s(enc(x))||^2 / ( 0.5 * (||s(enc(x))||^2 + ||s(enc(Mx))||^2) + eps ) ]

    Dividing by the signal's own magnitude removes the shrink-the-head escape route: scaling s by any
    constant leaves the ratio unchanged, so the only way down is for the encoder to actually represent the
    mirrored body as the sign flip of the original. Still label-free.
    """
    s, s_mirrored = mirror_pair_signals(head, view_encoder, coords, segments, embed_dim, joints)
    numerator = ((s_mirrored + s) ** 2).sum(dim=1)
    denominator = 0.5 * ((s ** 2).sum(dim=1) + (s_mirrored ** 2).sum(dim=1))
    return (numerator / (denominator + eps)).mean()


def equivariance_loss_parameter_free(view_encoder, coords, segments, embed_dim,
                                     joints=33, eps=1e-6):
    """The same scale-invariant residual with the head replaced by the identity feature map.

    This removes the head from the objective entirely, so there are no readout parameters that could be
    tuned to flatter the term. Its aggregation is a mean of per-sequence ratios, whereas the reported
    endpoint rho is a ratio of sums over the whole cohort, so the two are related but not identical.
    """
    s, s_mirrored = mirror_pair_signals(
        antisymmetric_contraction, view_encoder, coords, segments, embed_dim, joints)
    numerator = ((s_mirrored + s) ** 2).sum(dim=1)
    denominator = 0.5 * ((s ** 2).sum(dim=1) + (s_mirrored ** 2).sum(dim=1))
    return (numerator / (denominator + eps)).mean()


EQUIVARIANCE_VARIANTS = {
    "absolute": "nb_09b as written; has a shrink-the-head degenerate solution",
    "normalized": "scale-invariant, trainable head retained",
    "parameter_free": "scale-invariant, head removed from the objective",
}


def equivariance_term(variant, head, view_encoder, coords, segments, embed_dim, joints=33):
    """Dispatch to one equivariance variant, so a rung records which form it optimised."""
    if variant == "absolute":
        return equivariance_loss_absolute(head, view_encoder, coords, segments, embed_dim, joints)
    if variant == "normalized":
        return equivariance_loss_normalized(head, view_encoder, coords, segments, embed_dim, joints)
    if variant == "parameter_free":
        return equivariance_loss_parameter_free(view_encoder, coords, segments, embed_dim, joints)
    raise ValueError(f"unknown equivariance variant {variant!r}")


# --- the pre-registered PRIMARY endpoint: parameter-free, label-free, no fitting ----------------
def antisymmetric_contraction(tokens):
    """T(tokens) = sum over authorised pairs of ( tok_L - tok_R ) after a per-joint time mean.

    This is the AntisymmetricHead with f set to the identity, so it has no parameters at all. That makes
    it comparable across checkpoints without fitting anything and without a label.
    tokens: [B, SEGMENTS, 33, D] -> [B, D]
    """
    per_joint = tokens.mean(dim=1)
    out = 0.0
    for left, right in LEFT_RIGHT_PAIRS:
        out = out + (per_joint[:, left, :] - per_joint[:, right, :])
    return out


@torch.no_grad()
def mirror_residual_terms(encoder, xyz, segments, embed_dim, device, batch_size=16):
    """Per-sequence numerator and denominator of the normalised mirror-consistency residual rho.

    rho = mean_seq || T(enc(Mx)) + T(enc(x)) ||^2
        / mean_seq ( 0.5 * ( || T(enc(x)) ||^2 + || T(enc(Mx)) ||^2 ) )

    Scale, which is what makes rho readable without a target: rho = 0 means the encoder represents the
    mirrored body as the exact sign flip of the original along this axis, rho = 2 means the two are
    unrelated, and rho = 4 means the encoder is fully symmetric and cannot tell a body from its mirror.
    Lower is more mirror-honest. Returned per sequence so results can be aggregated by source video.
    """
    numerators, denominators = [], []
    for start in range(0, len(xyz), batch_size):
        chunk = torch.tensor(xyz[start:start + batch_size], dtype=torch.float32, device=device)
        tokens = encoder(chunk).reshape(len(chunk), segments, 33, embed_dim)
        tokens_mirrored = encoder(anatomical_mirror_coords(chunk)).reshape(
            len(chunk), segments, 33, embed_dim)
        s = antisymmetric_contraction(tokens)
        s_mirrored = antisymmetric_contraction(tokens_mirrored)
        numerators.append(((s_mirrored + s) ** 2).sum(dim=1).cpu().numpy())
        denominators.append((0.5 * ((s ** 2).sum(dim=1) + (s_mirrored ** 2).sum(dim=1))).cpu().numpy())
    return np.concatenate(numerators), np.concatenate(denominators)


def rho_from_terms(numerators, denominators):
    total = float(np.sum(denominators))
    if total <= 0:
        return float("nan")
    return float(np.sum(numerators) / total)


@torch.no_grad()
def head_signal_scale(head, encoder, xyz, segments, embed_dim, device, batch_size=16):
    """Mean ||s|| over the cohort. Evidence for or against the shrink-the-head degenerate solution."""
    norms = []
    for start in range(0, len(xyz), batch_size):
        chunk = torch.tensor(xyz[start:start + batch_size], dtype=torch.float32, device=device)
        tokens = encoder(chunk).reshape(len(chunk), segments, 33, embed_dim)
        norms.append(head(tokens).norm(dim=1).cpu().numpy())
    return float(np.mean(np.concatenate(norms)))
'''

PREPROCESSING = '''
def interpolate_low_visibility(sequence, threshold=0.45, max_gap=4):
    sequence = np.asarray(sequence, dtype=np.float32).copy()
    if sequence.ndim != 3 or sequence.shape[1:] != (33, 4):
        raise ValueError(f"Expected [T, 33, 4], received {sequence.shape}")
    visibility = np.nan_to_num(sequence[..., 3], nan=0.0)
    finite = np.isfinite(sequence[..., :3]).all(axis=-1)
    valid = (visibility >= threshold) & finite
    filled = valid.copy()
    for joint in range(33):
        observed = np.flatnonzero(valid[:, joint])
        for left, right in zip(observed[:-1], observed[1:]):
            gap = int(right - left - 1)
            if not 0 < gap <= max_gap:
                continue
            fraction = (np.arange(1, gap + 1, dtype=np.float32) / (gap + 1))[:, None]
            sequence[left + 1:right, joint, :3] = (
                sequence[left, joint, :3][None, :] * (1.0 - fraction)
                + sequence[right, joint, :3][None, :] * fraction)
            filled[left + 1:right, joint] = True
        sequence[~filled[:, joint], joint, :3] = np.nan
    sequence[..., 3] = visibility
    return sequence, valid


def center_and_scale(sequence, eps=1e-6):
    sequence = np.asarray(sequence, dtype=np.float32).copy()
    xyz = sequence[..., :3]
    left_hip, right_hip = xyz[:, 23], xyz[:, 24]
    left_ok = np.isfinite(left_hip).all(axis=1)
    right_ok = np.isfinite(right_hip).all(axis=1)
    pelvis = np.full((len(xyz), 3), np.nan, dtype=np.float32)
    both = left_ok & right_ok
    pelvis[both] = 0.5 * (left_hip[both] + right_hip[both])
    pelvis[left_ok & ~right_ok] = left_hip[left_ok & ~right_ok]
    pelvis[right_ok & ~left_ok] = right_hip[right_ok & ~left_ok]
    pelvis_ok = np.isfinite(pelvis).all(axis=1)
    fallback = np.median(pelvis[pelvis_ok], axis=0) if pelvis_ok.any() else np.zeros(3)
    pelvis[~np.isfinite(pelvis).all(axis=1)] = fallback
    xyz = xyz - pelvis[:, None, :]
    shoulder_width = np.linalg.norm(xyz[:, 11, :2] - xyz[:, 12, :2], axis=-1)
    hip_width = np.linalg.norm(xyz[:, 23, :2] - xyz[:, 24, :2], axis=-1)
    body_scale = np.nanmedian(np.maximum(shoulder_width, hip_width))
    if not np.isfinite(body_scale) or body_scale < eps:
        body_scale = 1.0
    sequence[..., :3] = np.nan_to_num(xyz / body_scale, nan=0.0, posinf=0.0, neginf=0.0)
    return np.nan_to_num(sequence, nan=0.0, posinf=0.0, neginf=0.0)


def temporal_resize(array, frames):
    array = np.asarray(array)
    if len(array) == frames:
        return array.copy()
    if len(array) < 2:
        return np.repeat(array, frames, axis=0)
    old_t = np.linspace(0.0, 1.0, len(array))
    new_t = np.linspace(0.0, 1.0, frames)
    flat = array.reshape(len(array), -1)
    resized = np.stack([np.interp(new_t, old_t, flat[:, i]) for i in range(flat.shape[1])], axis=1)
    return resized.reshape(frames, *array.shape[1:])


def prepare_sequence(raw_sequence, frames):
    """raw_sequence: [T, 33, 4] -> (xyz [frames, 33, 3] float32, valid [frames, 33] bool)."""
    interpolated, valid = interpolate_low_visibility(raw_sequence)
    scaled = center_and_scale(interpolated)
    xyz = temporal_resize(scaled[..., :3], frames)
    valid_resized = temporal_resize(valid.astype(np.float32), frames) > 0.5
    return xyz.astype(np.float32), valid_resized


def anatomical_mirror_raw(coords, pairs=FULL_MIRROR_PAIRS):
    """Mirror the RAW [T, 33, 4] column so the SAME preprocessing can be re-run on the reflection."""
    mirrored = np.asarray(coords, dtype=np.float32).copy()
    mirrored[:, :, 0] = -mirrored[:, :, 0]
    for left, right in pairs:
        mirrored[:, [left, right], :] = mirrored[:, [right, left], :]
    return mirrored


def signed_left_minus_right(coords):
    """Item 05's frozen target: signed per-side excursion, left minus right, on preprocessed coords."""
    coords = np.asarray(coords, dtype=np.float64)[..., :3]
    total = 0.0
    for left, right in LEFT_RIGHT_PAIRS:
        total += coords[:, left, :].std(axis=0).sum() - coords[:, right, :].std(axis=0).sum()
    return float(total)
'''

DATA_LOADING = '''
POSE_CACHE_CONTRACT = json.loads((TUTORIAL_DIR / "pose_cache_contract.json").read_text())
COMPATIBLE_EXTRACTION_VERSIONS = frozenset(POSE_CACHE_CONTRACT["compatible_extraction_versions"])
COHORT_ROOT = PROJECT_ROOT / "data" / "gavd"
EXPECTED_SEQUENCE_COUNTS = {"normal": 12, "parkinsons": 9, "stroke": 12,
                            "myopathic": 47, "cerebralpalsy": 16}
EXPECTED_SEQUENCE_IDS = {
    condition: {path.stem for path in (COHORT_ROOT / condition).glob("*.csv")}
    for condition in CONDITIONS
}
MIN_AUGMENTED_NEURO_OBSERVED = 0.45


def canonical_pose_records(pose_dir, conditions=CONDITIONS):
    """Load the locked canonical pose cache with the same identity checks notebook 04 applies."""
    records = []
    for condition in conditions:
        folder = Path(pose_dir) / condition
        available = {path.stem: path for path in folder.glob("*.npz")}
        expected = EXPECTED_SEQUENCE_IDS[condition]
        if set(available) != expected:
            raise ValueError(
                f"Canonical {condition} pose cache does not match the locked cohort at {folder}: "
                f"expected {len(expected)}, found {len(available)}")
        for sequence_id in sorted(expected):
            data = np.load(available[sequence_id], allow_pickle=False)
            version = str(data["extraction_version"].item())
            if version not in COMPATIBLE_EXTRACTION_VERSIONS:
                raise ValueError(f"Unsupported extraction version {version} in {available[sequence_id]}")
            sequence = data["sequence"].astype(np.float32)
            if sequence.ndim != 3 or sequence.shape[1:] != (33, 4):
                raise ValueError(f"Bad pose shape {sequence.shape}")
            records.append({
                "condition": condition,
                "sequence_id": str(data["sequence_id"].item()),
                "video_id": str(data["video_id"].item()),
                "raw": sequence,
                "cohort": "canonical",
            })
    return records


def augmented_normal_records(artifact_dir):
    """Load the opt-in augmentation-normal cohort through its selection report, as notebook 04 does."""
    if not INCLUDE_AUGMENTED:
        return []
    folder = artifact_dir / "poses_augmented" / "normal"
    report_path = artifact_dir / "augmented_pose_extraction_report.csv"
    if not folder.exists() or not report_path.is_file():
        raise FileNotFoundError(
            f"SJEPA_INCLUDE_AUGMENTED_NORMAL is enabled but {folder} or {report_path} is missing. "
            "Run notes/migrate_augmented_pose_artifacts.py, or notes/annotate_normal_clips.py then "
            "notes/extract_augmented_poses.py.")
    report = pd.read_csv(report_path)
    candidates = report[~report["status"].astype(str).str.startswith("error")].copy()
    accepted = pd.to_numeric(candidates["neuro_observed"], errors="coerce") >= MIN_AUGMENTED_NEURO_OBSERVED
    selected = set(candidates.loc[accepted, "sequence_id"].astype(str))
    available = {path.stem: path for path in folder.glob("*.npz")}
    if set(available) != selected:
        raise ValueError("Augmented pose cache does not match its selection report")
    records = []
    for sequence_id in sorted(selected):
        data = np.load(available[sequence_id], allow_pickle=False)
        records.append({
            "condition": "normal",
            "sequence_id": str(data["sequence_id"].item()),
            "video_id": str(data["video_id"].item()),
            "raw": data["sequence"].astype(np.float32),
            "cohort": "augmented_normal",
        })
    return records
'''


TRAIN_HELPERS = '''
def uniform_neurologic_mask(valid_patch, mask_fraction=0.60, seed=None):
    """Sample eligible joint-time tokens uniformly, with no motion score. Verbatim from notebook 04."""
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
    for index in range(len(mask)):
        candidates = np.flatnonzero(eligible[index].reshape(-1))
        mask[index].reshape(-1)[rng.choice(candidates, size=n_mask, replace=False)] = True
    forbidden = sorted(set(range(33)).difference(MASK_KEYPOINTS))
    if mask[:, :, forbidden].any():
        raise AssertionError("A forbidden keypoint entered the target mask")
    return mask


def geometric_view(x, max_degrees=8.0, translate=0.03, flip_probability=0.0):
    """Notebook 04's augmentation: y-axis rotation mixing x and z, plus a small translation.

    flip_probability stays 0.0 exactly as in notebook 04. Turning the flip on would put mirrored bodies
    into the JEPA branch itself, which would confound the equivariance term with a data augmentation.
    """
    view = x.clone()
    present = view.abs().sum(dim=-1) > 1e-8
    batch = len(view)
    angles = (torch.rand(batch, device=x.device) * 2.0 - 1.0) * math.radians(max_degrees)
    cosine, sine = torch.cos(angles), torch.sin(angles)
    original_x, original_z = view[..., 0].clone(), view[..., 2].clone()
    view[..., 0] = cosine[:, None, None] * original_x + sine[:, None, None] * original_z
    view[..., 2] = -sine[:, None, None] * original_x + cosine[:, None, None] * original_z
    view[..., :2] += (torch.rand(batch, 1, 1, 2, device=x.device) * 2.0 - 1.0) * translate
    if flip_probability > 0:
        raise ValueError("The flip augmentation must stay off; it would confound L_equiv")
    return view.masked_fill(~present[..., None], 0.0)


def authorized_pool(tokens, valid_patch):
    """Pool only valid tokens from the 12 authorised landmarks. Verbatim from notebook 04."""
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
    """VICReg invariance, variance hinge, and off-diagonal covariance. Verbatim from notebook 04."""
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
    return 25.0 * invariance + 25.0 * variance + covariance, invariance, variance, covariance


def condition_group_terms(representations, condition_ids, margin=1.0):
    """Bounded compactness and centroid separation on unit vectors. Verbatim from notebook 04."""
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
    return compactness, F.relu(margin - distances).square().mean(), distances.min()


def balanced_epoch_batches(data_by_condition, active_conditions, per_condition, rng):
    """Exhaustive, class-balanced replay batches for one epoch. Verbatim from notebook 04."""
    lengths = {c: len(data_by_condition[c]["xyz"]) for c in active_conditions}
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
            xyz_parts.append(data_by_condition[condition]["xyz"][take])
            valid_parts.append(data_by_condition[condition]["valid"][take])
            label_parts.extend([label] * per_condition)
        permutation = rng.permutation(len(label_parts))
        yield (np.concatenate(xyz_parts)[permutation],
               np.concatenate(valid_parts)[permutation],
               np.asarray(label_parts, dtype=np.int64)[permutation])


class VICRegProjector(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dimension, dimension), nn.GELU(),
                                 nn.Linear(dimension, dimension))

    def forward(self, x):
        return self.net(x)


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
'''

SYNTHETIC_COHORT = '''
def synthetic_gait_sequence(condition="normal", frames=64, seed=0):
    """A code-path fixture, not a physiological disease simulation. Verbatim from notebook 04."""
    rng = np.random.default_rng(seed)
    phase = np.linspace(0.0, 4.0 * np.pi, frames, endpoint=False)
    seq = np.zeros((frames, 33, 4), dtype=np.float32)
    seq[..., 3] = 1.0
    base = {11: (0.42, 0.28), 12: (0.58, 0.28), 23: (0.45, 0.52), 24: (0.55, 0.52),
            25: (0.44, 0.70), 26: (0.56, 0.70), 27: (0.43, 0.89), 28: (0.57, 0.89),
            29: (0.42, 0.92), 30: (0.58, 0.92), 31: (0.39, 0.94), 32: (0.61, 0.94)}
    for joint, (x, y) in base.items():
        seq[:, joint, 0] = x
        seq[:, joint, 1] = y
    amplitude, lift = 0.045, 0.025
    if condition == "parkinsons":
        amplitude *= 0.45
        lift *= 0.45
    if condition == "myopathic":
        seq[:, [11, 12], 0] += 0.03 * np.sin(phase)[:, None]
        seq[:, [23, 24], 0] += 0.018 * np.sin(phase)[:, None]
    for joint, knee, foot, offset in [(27, 25, 31, 0.0), (28, 26, 32, np.pi)]:
        wave = np.sin(phase + offset)
        if condition == "stroke" and joint == 27:
            wave = 0.35 * wave
        if condition == "cerebralpalsy":
            seq[:, knee, 1] -= 0.045
            seq[:, joint, 1] -= 0.02
        seq[:, joint, 0] += amplitude * wave
        seq[:, knee, 0] += 0.4 * amplitude * wave
        seq[:, foot, 0] += amplitude * wave
        seq[:, joint, 1] -= lift * np.maximum(wave, 0.0)
        seq[:, foot, 1] -= 0.7 * lift * np.maximum(wave, 0.0)
    seq[..., :3] += rng.normal(0.0, 0.0025, seq[..., :3].shape)
    return seq


SMOKE_LEAN_MAGNITUDE = {"stroke": 1.0, "cerebralpalsy": 0.8, "parkinsons": 0.5,
                        "myopathic": 0.0, "normal": 0.0}


def plant_signed_lean(seq, sign, magnitude, seed):
    """Plant a lateral lean whose SIGN alternates between the two synthetic sources of a condition."""
    if magnitude == 0.0 or sign == 0:
        return seq
    rng = np.random.default_rng(seed)
    out = seq.copy()
    phase = np.linspace(0.0, 4.0 * np.pi, len(seq), endpoint=False)
    for left, right in LEFT_RIGHT_PAIRS:
        gain = 0.03 * magnitude * sign * rng.uniform(0.8, 1.2)
        out[:, left, 0] += gain * np.sin(phase)
        out[:, right, 0] -= gain * np.sin(phase)
    return out


def build_smoke_condition_data(frames, clips_per_source=3, seed=RANDOM_SEED):
    """Two synthetic sources per condition with opposite lean signs, so the axis has real structure."""
    data, counter = {}, 0
    for condition in CONDITIONS:
        xyz_list, valid_list, video_ids = [], [], []
        for source in range(2):
            sign = 1 if source == 0 else -1
            magnitude = SMOKE_LEAN_MAGNITUDE[condition]
            for _ in range(clips_per_source):
                base = synthetic_gait_sequence(condition=condition, frames=frames, seed=seed + counter)
                leaned = plant_signed_lean(base, sign, magnitude, seed=seed + 1000 + counter)
                xyz, valid = prepare_sequence(leaned, frames)
                xyz_list.append(xyz)
                valid_list.append(valid)
                video_ids.append(f"smoke_source_{condition}_{source}")
                counter += 1
        data[condition] = {"xyz": np.stack(xyz_list).astype(np.float32),
                           "valid": np.stack(valid_list),
                           "video_ids": video_ids}
    return data
'''


# ---------------------------------------------------------------------------
# Notebook 00: methodology and contract
# ---------------------------------------------------------------------------

def build_nb00():
    global _CELL_N
    _CELL_N = [0]
    cells = []

    cells.append(md(COLAB_BADGE.format(name="new_nb_09_00_methodology_and_contract") + """

# new_nb_09_00. Methodology and contract for the equivariance-coupled retrain

Section 8 of `nb_09b_equivariant_retrain.ipynb` describes a real Arm-2 run that was never launched. It
tells the reader to paste the antisymmetric head, the anatomical mirror, and the equivariance loss into
`04_pretrain_sjepa_on_normal.ipynb`, then run a multi-seed 600-epoch curriculum and re-score every
checkpoint. This series carries out those four steps as its own executable path and leaves notebook 04
untouched, so the baseline lineage and the experiment stay separable.

This first notebook trains nothing. Its whole job is to fix, in advance and in writing, four things that
are easy to bend after seeing results: what question is being asked, what will be measured, what counts
as success, and what data the answer is allowed to rest on.

**Research use only.** Folder labels such as stroke and parkinsons are dataset annotations, not clinical
diagnoses. The source video is the independent unit of evidence throughout.
""".rstrip()))

    cells.append(md("""
## 0. What the four notebooks do, and why there are four of them

The series is split so that the expensive step is isolated and every claim before it is already checked.

1. **This notebook** states the method, verifies the data contract, and pre-registers the endpoints.
2. `new_nb_09_01` proves the mechanism on a small controlled problem. If the equivariance term is wired
   incorrectly it is silently a no-op, so this notebook has to pass before any real compute is spent.
3. `new_nb_09_02` runs the real ladder: two rungs across five seeds on the locked cohort.
4. `new_nb_09_03` scores every rung, applies the pre-registered rule, and writes the discussion.

The notebooks share no kernel state. Everything that crosses a boundary crosses it as a file on disk,
which is what makes the series resumable and independently checkable.
""".rstrip()))

    cells.append(code(SETUP + '''

MODE = REQUESTED_MODE
ARTIFACT_DIR = artifact_dir_for(MODE)
ARM2_DIR = ARTIFACT_DIR / ARM2_DIR_NAME
ARM2_DIR.mkdir(parents=True, exist_ok=True)

print(f"project root      : {PROJECT_ROOT}")
print(f"mode              : {MODE}")
print(f"artifact root     : {ARTIFACT_ROOT}")
print(f"arm-2 output dir  : {ARM2_DIR}")
print(f"augmented normals : {INCLUDE_AUGMENTED}")
print(f"baseline checkpoint name: {BASELINE_CHECKPOINT_NAME}")
'''))

    cells.append(md("""
## 1. The question, stated so that it can fail

Arm 1 of item 09 asked whether an antisymmetric readout, applied to the frozen encoder, decodes the signed
left-minus-right axis better than the encoder already does. It answered no, and its verdict was
`ARTIFACT (side-agnostic nuisance control fired)`. Arm 1 could not change the encoder, so it could not
test the more interesting possibility: that the encoder never learned to treat a body and its mirror as
sign-flipped versions of each other, and that asking it to do so during training would change the
representation.

Arm 2 asks exactly that:

> Does adding a label-free equivariance term to the curriculum objective make the trained encoder
> measurably more mirror-honest than the identical recipe without that term, by more than the run-to-run
> variation across seeds, and without degrading the representation on the tasks the encoder is used for?

The term is

```
L_equiv = mean( ( s(encoder(Mx)) + s(encoder(x)) )^2 )
```

where `M` is the anatomical mirror on the raw skeleton and `s` is the antisymmetric head. The total
objective becomes `L_JEPA + 0.05 * L_VICReg + 0.25 * L_group + w * L_equiv`, with `w = 0` for the control
rung and `w = 0.02` for the treatment rung. Nothing else changes between the two rungs.

Two properties of this term matter for interpretation. It uses no label of any kind, not a condition
folder and not a source video, so it cannot manufacture a transductive win by memorising which videos are
lateralised. And it constrains the encoder rather than the readout, which is the thing Arm 1 could not do.
""".rstrip()))

    cells.append(md("""
## 2. The one correctness trap, and why it is a trap

The antisymmetric head is antisymmetric by construction. It computes a pure difference over left and
right landmark pairs, so swapping its own inputs negates its output exactly, for every input and every
parameter value. That means a loss written as

```
mean( ( s(swap of the head's tokens) + s(tokens) )^2 )
```

is identically zero. Not approximately zero, not small: algebraically zero, with a gradient of zero
everywhere. Written that way the term trains nothing at all, while looking entirely reasonable in the
code and producing a loss curve that sits convincingly at zero.

The fix is to apply the mirror to the raw skeleton and run both the original and the reflection through
the encoder. The encoder is not equivariant by construction, so the residual is genuinely nonzero and its
gradient reaches encoder weights. `new_nb_09_01` proves both halves of this: that the head-only version is
a no-op, and that the through-the-encoder version produces nonzero gradients in the encoder.
""".rstrip()))

    cells.append(md("""
## 3. Why the primary endpoint is not R-squared

The obvious way to score Arm 2 would be to re-run Arm 1's probe on each retrained checkpoint and compare
held-out-source R-squared. That would be a mistake here, and Arm 1's own bundle says why.

Arm 1 pre-registered a y-quality gate: the signed target must carry at least 30 percent of its variance
between source videos, otherwise a held-out-source R-squared is measuring noise. On the real cohort the
between-source fraction is 0.0747. The gate failed. Consistently, every learned lane came out negative:
the antisymmetric head reached -0.206, the untrained floor -0.027, and the standard encoder comparator
-0.602. A difference between two negative numbers, both of which the item's own gate declares
uninterpretable, cannot support a conclusion in either direction.

So the primary endpoint is a quantity that needs no target variance at all: the normalised
mirror-consistency residual. Write `T` for the antisymmetric contraction with the identity feature map,
which is the head with all its parameters removed, and let `M` be the anatomical mirror. Then

```
rho(encoder) = mean_seq || T(enc(Mx)) + T(enc(x)) ||^2
             / mean_seq ( 0.5 * ( || T(enc(x)) ||^2 + || T(enc(Mx)) ||^2 ) )
```

`rho` has no free parameters, requires no fitting, and uses no label, so it is directly comparable across
checkpoints. Its scale is interpretable without reference to any target:

- `rho = 0` means the encoder represents the mirrored body as the exact sign flip of the original along
  this axis. This is perfect mirror honesty.
- `rho = 2` means the two representations are unrelated along this axis.
- `rho = 4` means the encoder is fully symmetric and cannot distinguish a body from its reflection.

Lower is better. `rho` is exactly the quantity `L_equiv` pressures, with the learned feature map replaced
by the identity so that a trained head cannot flatter its own arm. It is reported per source video so the
comparison can be paired.
""".rstrip()))

    cells.append(md("""
## 4. The full pre-registered measurement plan

Fixed here, before any real training runs.

**Primary.** `rho` on the target encoder, which is the frozen EMA teacher that every downstream readout in
this project actually reads. Reported per source video across the canonical cohort.

**Secondary.** `rho` on the view encoder, which is the network `L_equiv` optimises directly. Reporting
both separates two different questions: whether the pressure worked at all, and whether it survived the
EMA transfer into the artifact that gets used.

**Secondary.** The measured anatomical-mirror slope through the encoder. Item 05 measured -0.741 for the
baseline and Arm 1 measured -0.223 with its own head. If the equivariance term works, this slope should
move toward -1. It is a measured number and is never asserted to be -1.

**Tertiary, always reported beside the failed gate.** Arm 1's six-lane source-disjoint ladder recomputed
per checkpoint, so the record is complete even though the gate says these numbers are weak evidence.

**Guardrail.** Representation quality must not be bought away. Feature standard deviation, mean pairwise
cosine, and a source-grouped five-class condition probe on the frozen target encoder. A treatment rung
that improves `rho` by collapsing the representation has not earned anything.

**The credit rule.** The equivariance term earns credit only if all three of the following hold.

1. The mean improvement in `rho` from control to treatment exceeds the control rung's seed-to-seed
   standard deviation. This is a trajectory control, not a claim about unseen sources.
2. The paired bootstrap interval over source videos for that improvement excludes zero.
3. No guardrail regresses beyond its own control-rung seed spread.

Any other outcome is reported as no credit, with the numbers stated plainly.
""".rstrip()))

    cells.append(code('''
PRE_REGISTERED = {
    "question": (
        "Does a label-free equivariance term make the trained encoder measurably more mirror-honest than "
        "the identical recipe without it, by more than seed-to-seed variation, without degrading the "
        "representation?"
    ),
    "L_equiv": "mean((s(enc(Mx)) + s(enc(x)))^2), M = anatomical mirror on RAW coords, run THROUGH the "
               "view encoder. Label-free. A head-only token swap is identically zero and trains nothing.",
    "total_loss": "L_JEPA + 0.05*L_VICReg + 0.25*L_group + w*L_equiv, w = 0 for D0 and 0.02 for E1",
    "primary_endpoint": {
        "name": "rho_target_encoder",
        "definition": "mean_seq||T(enc(Mx)) + T(enc(x))||^2 / mean_seq(0.5*(||T(enc(x))||^2 + "
                      "||T(enc(Mx))||^2)), T = antisymmetric contraction with the identity feature map",
        "scale": "0 = exact sign flip (best), 2 = unrelated, 4 = fully symmetric (worst)",
        "why_not_r2": "Arm 1's y-quality gate FAILED on this cohort (between-source fraction 0.0747 "
                      "against a 0.30 threshold), so held-out-source R-squared is uninterpretable here.",
        "parameter_free": True, "label_free": True, "requires_fitting": False,
    },
    "secondary_endpoints": ["rho_view_encoder", "measured_anatomical_mirror_slope"],
    "tertiary_endpoints": ["arm1_six_lane_source_disjoint_r2 (reported beside the failed y-gate)"],
    "guardrails": ["feature_std", "mean_pair_cosine", "source_grouped_five_class_balanced_accuracy"],
    "credit_rule": {
        "1_exceeds_seed_spread": "mean(rho_D0) - mean(rho_E1) > std(rho_D0) across seeds",
        "2_paired_bootstrap": "paired-by-source bootstrap CI for the improvement excludes zero",
        "3_no_guardrail_regression": "no guardrail falls by more than the D0 seed spread",
        "all_three_required": True,
    },
    "ladder": {"rungs": ["D0", "E1"], "seeds": [0, 1, 2, 3, 4],
               "equiv_weight": {"D0": 0.0, "E1": 0.02}},
    "framing": [
        "All results are TRANSDUCTIVE: the encoder saw every evaluation sequence during training.",
        "The source video is the independent unit of evidence.",
        "Folder labels are dataset annotations, not clinical diagnoses.",
    ],
}
print(json.dumps(PRE_REGISTERED["primary_endpoint"], indent=2))
print()
print(json.dumps(PRE_REGISTERED["credit_rule"], indent=2))
'''))

    cells.append(md("""
## 5. The data contract

The ladder trains on the same corpus the baseline used, so the two lineages are comparable, and it is
evaluated on the canonical subset, matching Arm 1's `canonical_subset_only` flag. Concretely, training uses
the locked canonical cohort plus the opt-in augmentation-normal pool because
`SJEPA_INCLUDE_AUGMENTED_NORMAL` is enabled, and evaluation uses the canonical 96 sequences only.

The cell below verifies every input the ladder depends on before any compute is committed: the pose cache
matches the locked cohort identity by identity, the mask whitelist is the 12-point MS-PD set, the mapping
file exists and its hash is recorded, and the baseline checkpoint is present and readable. It also records
the baseline checkpoint's sha256 so `new_nb_09_03` can prove the experiment never modified it.
""".rstrip()))

    cells.append(code(PREPROCESSING + DATA_LOADING + '''
import torch

MAPPING_RELATIVE_PATH = Path("experiments/multiple-sclerosis/mapping-data/ms-pd-mapping.md")
MAPPING_PATH = PROJECT_ROOT / MAPPING_RELATIVE_PATH
if not MAPPING_PATH.is_file():
    raise FileNotFoundError(f"Required mapping file is missing: {MAPPING_PATH}")
MAPPING_SHA256 = hashlib.sha256(MAPPING_PATH.read_bytes()).hexdigest()

MASK_KEYPOINT_NAMES = ["LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_HIP", "RIGHT_HIP",
                       "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE",
                       "LEFT_HEEL", "RIGHT_HEEL", "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX"]
if MASK_KEYPOINTS != sorted(set(MASK_KEYPOINTS)) or len(MASK_KEYPOINTS) != 12:
    raise ValueError("The mask whitelist must be the de-duplicated 12-point MS-PD set")

POSE_DIR = ARTIFACT_DIR / "poses"
canonical = canonical_pose_records(POSE_DIR)
augmented = augmented_normal_records(ARTIFACT_DIR)

canonical_meta = pd.DataFrame([{k: r[k] for k in ("condition", "sequence_id", "video_id", "cohort")}
                               for r in canonical])
training_meta = pd.DataFrame([{k: r[k] for k in ("condition", "sequence_id", "video_id", "cohort")}
                              for r in canonical + augmented])

print("canonical cohort by condition:")
print(canonical_meta.groupby("condition").size().to_string())
print(f"\\ncanonical      : {len(canonical)} sequences from "
      f"{canonical_meta['video_id'].nunique()} source videos")
print(f"augmented normal: {len(augmented)} sequences from "
      f"{pd.DataFrame(augmented)['video_id'].nunique() if augmented else 0} source videos")
print(f"training corpus : {len(canonical) + len(augmented)} sequences from "
      f"{training_meta['video_id'].nunique()} source videos")

baseline_path = ARTIFACT_DIR / BASELINE_CHECKPOINT_NAME
if not baseline_path.is_file():
    raise FileNotFoundError(f"Baseline checkpoint is missing: {baseline_path}")
BASELINE_SHA256 = hashlib.sha256(baseline_path.read_bytes()).hexdigest()
baseline_meta = torch.load(baseline_path, map_location="cpu", weights_only=False)
BASELINE_FINGERPRINT = str(baseline_meta["dataset_fingerprint"])
print(f"\\nbaseline checkpoint : {baseline_path.name}")
print(f"baseline fingerprint: {BASELINE_FINGERPRINT[:16]}")
print(f"baseline sha256     : {BASELINE_SHA256[:16]}")
print(f"baseline config     : {baseline_meta['config']}")
if baseline_meta["mask_keypoints"] != MASK_KEYPOINTS:
    raise ValueError("Baseline checkpoint mask whitelist does not match the 12-point set")
if not baseline_meta.get("curriculum_complete", False):
    raise ValueError("Baseline checkpoint is not a completed curriculum final")
'''))

    cells.append(md("""
## 6. What is deliberately being reproduced, and what is deliberately new

The control rung is not the existing baseline checkpoint. It is the same recipe rerun from a fresh seed
with the equivariance term switched off. That distinction matters: comparing the treatment rung against
the single existing baseline would confound the equivariance term with ordinary run-to-run variation,
because the baseline is one draw from a distribution of training trajectories. Running the control at the
same five seeds turns that variation into a measured quantity instead of an unknown.

The existing baseline still appears in the results as a reference row, so the reader can see where the
published lineage sits relative to the ladder. It is never used as the control.

Three things are new relative to notebook 04, and each is a fix rather than a preference:

- The seed is threaded through the model initialisation, the batch sampler, and the mask sampler. Notebook
  04 hardcodes 42 in the per-stage generator and in the fingerprint payload, so five nominal seeds would
  have produced five identical runs.
- `seed`, `equiv_weight`, and `equiv_on` join the fingerprint payload, so each rung has its own identity
  and no rung can be mistaken for the baseline lineage.
- Each rung persists its own checkpoint and JSON, so the ladder resumes after an interruption.
""".rstrip()))

    cells.append(code('''
CONTRACT = {
    "notebook": "new_nb_09_00_methodology_and_contract",
    "series": "new_nb_09",
    "arm": "arm2_equivariance_coupled_retrain",
    "supersedes": "the smoke-configured idea9_equivariant_retrain_result.json, which was written with "
                  "mode=real by nb_09b while training with SMOKE_CONFIG",
    "mode": MODE,
    "pre_registered": PRE_REGISTERED,
    "cohort": {
        "training_sequences": len(canonical) + len(augmented),
        "training_sources": int(training_meta["video_id"].nunique()),
        "canonical_sequences": len(canonical),
        "canonical_sources": int(canonical_meta["video_id"].nunique()),
        "canonical_condition_counts": canonical_meta.groupby("condition").size().to_dict(),
        "include_augmented_normal": INCLUDE_AUGMENTED,
        "evaluation_subset": "canonical only, matching Arm 1's canonical_subset_only flag",
    },
    "contract": {
        "mask_keypoints": MASK_KEYPOINTS,
        "mask_keypoint_names": MASK_KEYPOINT_NAMES,
        "mapping_path": str(MAPPING_RELATIVE_PATH),
        "mapping_sha256": MAPPING_SHA256,
        "pose_extraction_versions_accepted": sorted(COMPATIBLE_EXTRACTION_VERSIONS),
        "baseline_checkpoint": baseline_path.name,
        "baseline_fingerprint": BASELINE_FINGERPRINT,
        "baseline_sha256": BASELINE_SHA256,
        "baseline_role": "reference row only; never the control rung",
    },
    "arm1_reference": {
        "verdict": "ARTIFACT (side-agnostic nuisance control fired)",
        "y_between_source_fraction": 0.07466055304332204,
        "y_gate_threshold": 0.30,
        "y_gate_passed": False,
        "A_prime_r2": -0.20574789599382415,
        "C_floor_r2": -0.027031078336773096,
        "D_standard_r2": -0.6021929001488207,
        "wiring_swap_slope": -1.0000000000000002,
        "measured_anatomical_mirror_slope": -0.22290579837876792,
    },
    "item05_reference": {
        "verdict": "INFORMATIVE NULL",
        "A_learned_r2": -0.6021929001488207,
        "C_floor_r2": -0.1561506987225285,
        "sign_consistency": 0.4444444444444444,
        "measured_anatomical_mirror_slope": -0.7408177852917672,
        "note": "Some older notes cite -0.343 for this slope. The current bundle on disk says -0.741, "
                "and that is the number this series uses.",
    },
}
contract_path = ARM2_DIR / "idea9_arm2_contract.json"
contract_path.write_text(json.dumps(CONTRACT, indent=2, sort_keys=False), encoding="utf-8")
print(f"wrote {contract_path}")
print(json.dumps(CONTRACT["cohort"], indent=2))
'''))

    cells.append(md("""
## 7. What this notebook has established

The question is written down, the endpoint is fixed and does not depend on a target whose quality gate
already failed, the decision rule is stated with all three of its conditions, and the data contract is
verified against the cohort on disk rather than assumed. The baseline checkpoint's hash is recorded so its
integrity can be proven later.

Nothing has been measured yet, and nothing should be read as a result. The next notebook proves the
mechanism on a small controlled problem, which is the gate that has to pass before the real ladder is
worth running.
""".rstrip()))

    write_notebook(ROOT / "new_nb_09_00_methodology_and_contract.ipynb", cells)


# ---------------------------------------------------------------------------
# Notebook 01: mechanism and smoke validation
# ---------------------------------------------------------------------------

def build_nb01():
    global _CELL_N
    _CELL_N = [0]
    cells = []

    cells.append(md(COLAB_BADGE.format(name="new_nb_09_01_mechanism_and_smoke_validation") + """

# new_nb_09_01. Does the equivariance term actually do anything?

This notebook spends no real compute and produces no result about gait. It answers a narrower question
that has to be settled first: when the training loop adds `L_equiv`, does anything happen?

That question is not rhetorical. The natural way to write this loss is a no-op that trains nothing while
producing a loss curve pinned at zero, which looks like success. So before the ladder runs for hours, four
checks run here in seconds.

1. **The trap is real.** The head-only version of the loss is exactly zero with exactly zero gradient.
2. **The fix works.** The through-the-encoder version is strictly positive and pushes gradient into the
   encoder as well as the head.
3. **The endpoint is calibrated.** `rho` reads 0 on an encoder that is mirror-equivariant by construction
   and 4 on one that is mirror-blind by construction, so its scale means what the contract says it means.
4. **The whole loop responds.** In a small end-to-end run of the real composite objective, the treatment
   rung lowers `rho` and the control rung does not.

Every number in this notebook comes from synthetic fixtures. None of it is a claim about gait.
""".rstrip()))

    cells.append(code(SETUP + '''

MODE = "smoke"  # this notebook is always a mechanism check, never a result
ARTIFACT_DIR = artifact_dir_for(REQUESTED_MODE)
ARM2_DIR = ARTIFACT_DIR / ARM2_DIR_NAME
ARM2_DIR.mkdir(parents=True, exist_ok=True)
print(f"mechanism checks write to: {ARM2_DIR}")
'''))

    cells.append(code(MODEL_CLASSES + PREPROCESSING + EQUIV_CODE + TRAIN_HELPERS + SYNTHETIC_COHORT + '''

print("model, preprocessing, equivariance, and training helpers defined.")
print(f"torch {torch.__version__}, device would be {pick_device()}")
'''))

    cells.append(md("""
## 1. The trap: a loss that looks right and trains nothing

The antisymmetric head computes a pure difference over left and right landmark pairs. Swapping its own
inputs therefore negates its output exactly, for every input and every parameter value, because that is
what the arithmetic says. A loss written as `mean( ( s(swap of tokens) + s(tokens) )^2 )` is consequently
zero as an algebraic identity.

The check below measures both halves of that claim: the loss value, which should be zero to floating-point
noise, and the total gradient norm it delivers to the head, which should be exactly zero. A term with zero
gradient cannot train anything, no matter how long it runs or how large its weight is set.

The swap identity is still worth keeping as a wiring self-check on the head. It just cannot be the loss.
""".rstrip()))

    cells.append(code('''
torch.manual_seed(RANDOM_SEED)
head_check = AntisymmetricHead(32, out_dim=4)
tokens_check = torch.randn(6, 8, 33, 32)

with torch.no_grad():
    s_value = head_check(tokens_check)
    s_swapped = head_check.swapped(tokens_check)
wiring_slope = float(np.polyfit(s_value.reshape(-1).numpy(), s_swapped.reshape(-1).numpy(), 1)[0])

head_only_loss = ((head_check.swapped(tokens_check) + head_check(tokens_check)) ** 2).mean()
head_only_value = float(head_only_loss.detach())
head_check.zero_grad(set_to_none=True)
head_only_loss.backward()
head_only_grad = sum(float(p.grad.abs().sum()) for p in head_check.parameters() if p.grad is not None)

print(f"wiring self-check slope of s(swap) against s : {wiring_slope:+.10f}   (exactly -1 by construction)")
print(f"head-only loss value                        : {head_only_value:.3e}")
print(f"head-only total gradient into the head      : {head_only_grad:.3e}")
if abs(wiring_slope + 1.0) > 1e-6:
    raise AssertionError("the head must negate under a swap of its own inputs")
if head_only_value > 1e-12 or head_only_grad > 1e-12:
    raise AssertionError("the head-only loss must be an exact no-op")
print("\\nCHECK 1 PASSED: the head-only formulation is an exact no-op. It cannot be the training loss.")
'''))

    cells.append(md("""
## 2. The fix: put the mirror on the skeleton, not on the tokens

The working version reflects the raw skeleton, negating the sideways coordinate and swapping every
left-right landmark, and then runs both the original and the reflection through the encoder. The encoder
has no built-in reason to represent a reflected body as the left-right swap of its own tokens, so the
residual is genuinely nonzero and its gradient reaches encoder weights.

The check runs the term through a deliberately non-equivariant encoder, which is a plain linear map over
raw coordinates, and requires that the loss is strictly positive and that gradient reaches the encoder.
That second condition is the one the head-only formulation can never satisfy.

Three variants of the term are carried forward, because section 4 will show that the difference between
them decides whether the experiment measures anything at all.

- `absolute` is `nb_09b`'s term exactly as written, `mean( ( s(enc(Mx)) + s(enc(x)) )^2 )`.
- `normalized` divides each sequence's squared residual by that sequence's own signal magnitude, keeping
  the trainable head.
- `parameter_free` uses the same scale-invariant ratio with the head replaced by the identity feature map,
  so the objective has no readout parameters at all.

All three are label-free and all three reach encoder weights. The last assertion below confirms that the
parameter-free variant touches no head parameters whatsoever.
""".rstrip()))

    cells.append(code('''
class NonEquivariantEncoder(nn.Module):
    """A plain per-joint linear map over raw coordinates. Nothing about it respects the mirror."""

    def __init__(self, frames, segments, dim):
        super().__init__()
        self.frames, self.segments, self.dim = frames, segments, dim
        self.proj = nn.Linear(frames * 3, segments * dim)

    def forward(self, x):
        batch = len(x)
        per_joint = x.permute(0, 2, 1, 3).reshape(batch, 33, self.frames * 3)
        out = self.proj(per_joint).reshape(batch, 33, self.segments, self.dim)
        return out.permute(0, 2, 1, 3).reshape(batch, self.segments * 33, self.dim)


torch.manual_seed(RANDOM_SEED + 3)
probe_frames, probe_segments, probe_dim = 8, 2, 32
probe_coords = torch.randn(4, probe_frames, 33, 3)
gradient_evidence = {}
for variant in EQUIVARIANCE_VARIANTS:
    torch.manual_seed(RANDOM_SEED + 3)
    probe_encoder = NonEquivariantEncoder(probe_frames, probe_segments, probe_dim)
    probe_head = AntisymmetricHead(probe_dim, out_dim=4)
    loss = equivariance_term(variant, probe_head, probe_encoder, probe_coords,
                             probe_segments, probe_dim)
    loss.backward()
    gradient_evidence[variant] = {
        "loss": float(loss.detach()),
        "grad_head": sum(float(p.grad.norm()) for p in probe_head.parameters() if p.grad is not None),
        "grad_encoder": sum(float(p.grad.norm()) for p in probe_encoder.parameters()
                            if p.grad is not None),
    }
    row = gradient_evidence[variant]
    print(f"{variant:15s} loss {row['loss']:.6e}   grad head {row['grad_head']:.3e}   "
          f"grad encoder {row['grad_encoder']:.3e}")
    if row["loss"] <= 1e-8:
        raise AssertionError(f"{variant} must be strictly positive on a non-equivariant encoder")
    if row["grad_encoder"] <= 1e-8:
        raise AssertionError(f"{variant} must deliver gradient to the encoder")
if gradient_evidence["parameter_free"]["grad_head"] > 0.0:
    raise AssertionError("the parameter-free variant must not touch head parameters at all")
print("\\nCHECK 2 PASSED: every through-the-encoder variant reaches encoder weights.")
'''))

    cells.append(md("""
## 3. Calibrating the primary endpoint against two encoders built by hand

The contract claims that `rho` reads 0 for a mirror-equivariant encoder, 4 for a mirror-blind one, and
about 2 when the two representations are unrelated. That claim should be demonstrated rather than asserted,
because the whole result rests on reading `rho` correctly.

Two toy encoders make the endpoints exact rather than approximate.

The **equivariant** one takes the absolute value of the sideways coordinate before its projection. Under
the anatomical mirror, the left slot receives the reflected right joint, whose sideways magnitude is
unchanged, so the encoder's tokens for the mirrored body are exactly the left-right swap of its tokens for
the original. That is precisely the property `L_equiv` asks for, so `rho` must be 0.

The **mirror-blind** one gives every joint a fixed per-slot code plus a body-level summary that is itself
mirror invariant. Nothing in its tokens records which side the motion happened on, so the mirrored body
maps to the same tokens rather than to their swap. `rho` must be 4.

A randomly initialised real encoder is measured alongside them as an unconstrained reference point.
""".rstrip()))

    cells.append(code('''
class MirrorEquivariantToyEncoder(nn.Module):
    """Per-joint projection of ( |x|, y, z ). Mirror-equivariant by construction, so rho must be 0."""

    def __init__(self, frames, segment_length, dim):
        super().__init__()
        self.segments = frames // segment_length
        self.segment_length = segment_length
        self.proj = nn.Linear(3, dim)

    def forward(self, x):
        batch = len(x)
        pooled = x.reshape(batch, self.segments, self.segment_length, 33, 3).mean(dim=2)
        features = torch.stack([pooled[..., 0].abs(), pooled[..., 1], pooled[..., 2]], dim=-1)
        return self.proj(features).reshape(batch, self.segments * 33, -1)


class MirrorBlindToyEncoder(nn.Module):
    """Per-slot code plus a mirror-invariant body summary. Side information is absent, so rho must be 4."""

    def __init__(self, frames, segment_length, dim):
        super().__init__()
        self.segments = frames // segment_length
        self.segment_length = segment_length
        self.joint_code = nn.Parameter(torch.randn(33, dim))
        self.proj = nn.Linear(3, dim)

    def forward(self, x):
        batch = len(x)
        pooled = x.reshape(batch, self.segments, self.segment_length, 33, 3).mean(dim=2)
        summary = torch.stack([pooled[..., 0].abs(), pooled[..., 1], pooled[..., 2]], dim=-1).mean(dim=2)
        tokens = self.joint_code[None, None, :, :] + self.proj(summary)[:, :, None, :]
        return tokens.reshape(batch, self.segments * 33, -1)


CAL_FRAMES, CAL_SEGMENT, CAL_DIM = 32, 4, 32
CAL_SEGMENTS = CAL_FRAMES // CAL_SEGMENT
calibration_data = build_smoke_condition_data(CAL_FRAMES)
calibration_xyz = np.concatenate([calibration_data[c]["xyz"] for c in CONDITIONS])
cpu = torch.device("cpu")

torch.manual_seed(RANDOM_SEED)
calibration_encoders = {
    "mirror_equivariant_by_construction": MirrorEquivariantToyEncoder(CAL_FRAMES, CAL_SEGMENT, CAL_DIM),
    "mirror_blind_by_construction": MirrorBlindToyEncoder(CAL_FRAMES, CAL_SEGMENT, CAL_DIM),
    "randomly_initialised_real_encoder": SkeletonPatchEncoder(
        frames=CAL_FRAMES, segment_length=CAL_SEGMENT, embed_dim=CAL_DIM, depth=2, heads=4),
}
calibration_rho = {}
for name, encoder in calibration_encoders.items():
    encoder.eval()
    numerators, denominators = mirror_residual_terms(
        encoder, calibration_xyz, CAL_SEGMENTS, CAL_DIM, cpu)
    calibration_rho[name] = rho_from_terms(numerators, denominators)
    print(f"rho = {calibration_rho[name]:.6f}   {name}")

# rho must be blind to which of the pair is called the original, because M is an involution.
equivariant_encoder = calibration_encoders["mirror_equivariant_by_construction"]
mirrored_xyz = np.stack([anatomical_mirror_raw(
    np.concatenate([x, np.ones((len(x), 33, 1), dtype=np.float32)], axis=2))[..., :3]
    for x in calibration_xyz])
numerators_m, denominators_m = mirror_residual_terms(
    equivariant_encoder, mirrored_xyz, CAL_SEGMENTS, CAL_DIM, cpu)
involution_gap = abs(rho_from_terms(numerators_m, denominators_m)
                     - calibration_rho["mirror_equivariant_by_construction"])
print(f"\\ninvolution check, |rho(x) - rho(Mx)| = {involution_gap:.3e}   (must be ~0)")

if calibration_rho["mirror_equivariant_by_construction"] > 1e-6:
    raise AssertionError("rho must be 0 for an encoder that is mirror-equivariant by construction")
if abs(calibration_rho["mirror_blind_by_construction"] - 4.0) > 1e-6:
    raise AssertionError("rho must be 4 for an encoder that is mirror-blind by construction")
if involution_gap > 1e-6:
    raise AssertionError("rho must not depend on which member of the mirror pair is the original")
print("\\nCHECK 3 PASSED: rho reads 0 at exact equivariance and 4 at exact mirror blindness.")
'''))

    cells.append(md("""
## 4. The check that changed the design: a loss can fall to zero and teach the encoder nothing

The three checks so far are about arithmetic. This one asks a behavioural question: inside the real
composite objective, alongside the JEPA cross-entropy, the VICReg terms, and the label-aware group terms,
does adding the equivariance term change what the encoder learns?

The first version of this notebook ran only `nb_09b`'s term as written and concluded that it did. That
conclusion was wrong, and the way it was wrong is worth stating plainly because it is the reason this
series exists in the form it does.

Optimising `mean( ( s(enc(Mx)) + s(enc(x)) )^2 )` drove the term down by more than two orders of magnitude,
from 0.55 to 0.003. Read on its own, that looks like the encoder learning mirror equivariance. But the
parameter-free endpoint `rho` did not move: 3.922 for the treatment rung against 3.931 for the control, a
difference smaller than the control rung's own epoch-to-epoch wobble. The loss went to zero without the
representation changing.

The reason is that the head is trainable and shares the objective. The residual is a squared magnitude, so
the cheapest way to shrink it is to shrink `s` itself, which costs the encoder nothing at all. The head
learns to output almost zero, the loss curve looks excellent, and the encoder is left exactly as it was.
This is the same class of mistake as the head-only no-op in check 1, one level deeper: the term is not
algebraically zero, it is merely satisfiable without doing the work.

The repair is to make the term scale-invariant by dividing each sequence's squared residual by that
sequence's own signal magnitude. Scaling `s` by any constant then leaves the value unchanged, so shrinking
the head buys nothing and the only way down is for the encoder to genuinely represent a mirrored body as
the sign flip of the original.

The cell below runs all three variants against the control, and tracks the head's mean signal magnitude
alongside `rho` so that a collapse is visible rather than inferred. The gate is deliberately strict this
time: an improvement counts only if it exceeds three times the control rung's own standard deviation over
the last ten epochs. The weak version of this gate is what let the false pass through.
""".rstrip()))

    cells.append(code('''
MINI_FRAMES, MINI_SEGMENT, MINI_DIM = 32, 4, 32
MINI_SEGMENTS = MINI_FRAMES // MINI_SEGMENT
MINI_EPOCHS = 30
MINI_SAMPLES_PER_CONDITION = 2
MINI_EQUIV_WEIGHT = 0.02
mini_data = build_smoke_condition_data(MINI_FRAMES)
mini_xyz = np.concatenate([mini_data[c]["xyz"] for c in CONDITIONS])
print(f"mini cohort: {len(mini_xyz)} synthetic sequences, {MINI_EPOCHS} epochs per rung")


def run_mini_rung(equiv_weight, variant="normalized", seed=RANDOM_SEED, device=cpu):
    """One small end-to-end rung of the real composite objective, with the equivariance term on or off."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = SJEPAGait(frames=MINI_FRAMES, segment_length=MINI_SEGMENT, embed_dim=MINI_DIM,
                      encoder_depth=1, predictor_depth=1, heads=4).to(device)
    projector = VICRegProjector(MINI_DIM).to(device)
    head = AntisymmetricHead(MINI_DIM, out_dim=4).to(device)
    trainable = [*model.view_encoder.parameters(), *model.predictor.parameters(),
                 *projector.parameters(), *head.parameters()]
    optimizer = torch.optim.AdamW(trainable, lr=1e-3, betas=(0.9, 0.95), weight_decay=0.05)
    rng = np.random.default_rng(seed)
    rows, step = [], 0
    for epoch in range(MINI_EPOCHS):
        model.train()
        equiv_values = []
        for xyz_np, valid_np, labels_np in balanced_epoch_batches(
                mini_data, CONDITIONS, MINI_SAMPLES_PER_CONDITION, rng):
            coordinates = torch.tensor(xyz_np, dtype=torch.float32, device=device)
            valid = torch.tensor(valid_np, dtype=torch.bool, device=device)
            labels = torch.tensor(labels_np, dtype=torch.long, device=device)
            valid_patch = valid.reshape(len(valid), MINI_SEGMENTS, MINI_SEGMENT, 33).all(dim=2)
            mask_np = uniform_neurologic_mask(valid_patch.cpu().numpy(), 0.60, seed=seed + 100000 + step)
            target_mask = torch.tensor(mask_np, dtype=torch.bool, device=device)
            view_a = geometric_view(coordinates)
            view_b = geometric_view(coordinates)

            prediction, target = model(view_a, coordinates, target_mask)
            jepa_loss = sjepa_cross_entropy(prediction, target, model.target_center)
            tokens_a = model.view_encoder(view_a).reshape(len(view_a), MINI_SEGMENTS, 33, MINI_DIM)
            tokens_b = model.view_encoder(view_b).reshape(len(view_b), MINI_SEGMENTS, 33, MINI_DIM)
            pooled_a = authorized_pool(tokens_a, valid_patch)
            pooled_b = authorized_pool(tokens_b, valid_patch)
            vicreg_loss, _, _, _ = vicreg_terms(projector(pooled_a), projector(pooled_b))
            compactness, separation, _ = condition_group_terms(pooled_a, labels)
            equiv = equivariance_term(variant, head, model.view_encoder, coordinates,
                                      MINI_SEGMENTS, MINI_DIM)
            total = (jepa_loss + 0.05 * vicreg_loss + 0.25 * (compactness + separation)
                     + equiv_weight * equiv)

            optimizer.zero_grad(set_to_none=True)
            total.backward()
            if any(p.grad is not None for p in model.target_encoder.parameters()):
                raise AssertionError("the target encoder must stay frozen")
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()
            model.update_target(cosine_ema(step, MINI_EPOCHS * 3, start=0.996))
            model.update_center(target, beta=0.9)
            equiv_values.append(float(equiv.detach()))
            step += 1

        model.eval()
        view_terms = mirror_residual_terms(
            model.view_encoder, mini_xyz, MINI_SEGMENTS, MINI_DIM, device)
        target_terms = mirror_residual_terms(
            model.target_encoder, mini_xyz, MINI_SEGMENTS, MINI_DIM, device)
        rows.append({"epoch": epoch + 1,
                     "equiv_term": float(np.mean(equiv_values)),
                     "head_signal_scale": head_signal_scale(
                         head, model.view_encoder, mini_xyz, MINI_SEGMENTS, MINI_DIM, device),
                     "rho_view": rho_from_terms(*view_terms),
                     "rho_target": rho_from_terms(*target_terms)})
    return pd.DataFrame(rows)


RUNS = {"D0 control (w=0)": (0.0, "normalized")}
for variant in EQUIVARIANCE_VARIANTS:
    RUNS[f"E1 {variant} (w={MINI_EQUIV_WEIGHT})"] = (MINI_EQUIV_WEIGHT, variant)

start_time = time.time()
mini_runs = {label: run_mini_rung(weight, variant) for label, (weight, variant) in RUNS.items()}
print(f"{len(mini_runs)} mini rungs finished in {time.time() - start_time:.1f} s\\n")

control = mini_runs["D0 control (w=0)"]
# The control rung's own late-training wobble is the yardstick. An "improvement" smaller than this is
# indistinguishable from the trajectory noise the term is supposed to beat.
control_noise = float(control["rho_view"].iloc[-10:].std())
GATE_MULTIPLE = 3.0
gate_threshold = GATE_MULTIPLE * control_noise
control_rho = float(control["rho_view"].iloc[-1])
print(f"control rho on the view encoder      : {control_rho:.5f}")
print(f"control late-epoch standard deviation: {control_noise:.5f}")
print(f"gate: an improvement must exceed {GATE_MULTIPLE:.0f} x {control_noise:.5f} = "
      f"{gate_threshold:.5f}\\n")

bakeoff_rows = []
for label, frame in mini_runs.items():
    weight, variant = RUNS[label]
    improvement = control_rho - float(frame["rho_view"].iloc[-1])
    bakeoff_rows.append({
        "rung": label, "variant": variant,
        "rho_view_final": float(frame["rho_view"].iloc[-1]),
        "rho_target_final": float(frame["rho_target"].iloc[-1]),
        "improvement_vs_control": improvement,
        "beats_gate": bool(weight > 0 and improvement > gate_threshold),
        "term_first_epoch": float(frame["equiv_term"].iloc[0]),
        "term_last_epoch": float(frame["equiv_term"].iloc[-1]),
        "term_fold_reduction": float(frame["equiv_term"].iloc[0] / max(frame["equiv_term"].iloc[-1], 1e-12)),
        "head_scale_first": float(frame["head_signal_scale"].iloc[0]),
        "head_scale_last": float(frame["head_signal_scale"].iloc[-1]),
        "head_scale_fold_shrink": float(frame["head_signal_scale"].iloc[0]
                                        / max(frame["head_signal_scale"].iloc[-1], 1e-12)),
    })
bakeoff = pd.DataFrame(bakeoff_rows)
display(bakeoff.set_index("rung").round(5))

qualifying = bakeoff[bakeoff["beats_gate"]].sort_values("improvement_vs_control", ascending=False)
if qualifying.empty:
    raise AssertionError(
        "MECHANISM GATE FAILED for every variant: no formulation moved rho by more than the control "
        "rung's own trajectory noise, so the real ladder would measure nothing. Do not spend real compute.")
SELECTED_VARIANT = str(qualifying.iloc[0]["variant"])
print(f"\\nvariants that beat the gate : {list(qualifying['variant'])}")
print(f"selected for the real ladder: {SELECTED_VARIANT}")
absolute_row = bakeoff[bakeoff["variant"] == "absolute"].iloc[0]
print(f"\\nthe documented failure mode, for the record:")
print(f"  the absolute variant drove its own term down {absolute_row['term_fold_reduction']:.0f}-fold "
      f"while shrinking the head's signal {absolute_row['head_scale_fold_shrink']:.1f}-fold")
print(f"  and moved rho by only {absolute_row['improvement_vs_control']:+.5f}, against a gate of "
      f"{gate_threshold:.5f}")
print("\\nCHECK 4 PASSED: at least one scale-invariant variant moves rho by more than trajectory noise.")
'''))

    cells.append(code('''
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from IPython.display import Image, display

COLOURS = {"D0 control (w=0)": "#555555", f"E1 absolute (w={MINI_EQUIV_WEIGHT})": "#1f77b4",
           f"E1 normalized (w={MINI_EQUIV_WEIGHT})": "#c1272d",
           f"E1 parameter_free (w={MINI_EQUIV_WEIGHT})": "#2ca02c"}
figure, axes = plt.subplots(1, 3, figsize=(16, 4.4))
for label, frame in mini_runs.items():
    style = "--" if label.startswith("D0") else "-"
    axes[0].plot(frame["epoch"], frame["rho_view"], style, label=label, color=COLOURS[label])
    axes[1].plot(frame["epoch"], frame["head_signal_scale"], style, label=label, color=COLOURS[label])
    axes[2].semilogy(frame["epoch"], frame["equiv_term"], style, label=label, color=COLOURS[label])
axes[0].axhline(control_rho - gate_threshold, color="#c1272d", linestyle=":", linewidth=1.0,
                label="gate: 3 x control noise")
axes[0].set_title("Primary endpoint: rho on the view encoder")
axes[0].set_ylabel("rho   (0 = exact sign flip, 4 = mirror blind)")
axes[1].set_title("Head signal magnitude, mean ||s||\\n(the shrink-the-head escape route)")
axes[1].set_ylabel("mean ||s||")
axes[2].set_title("The optimised term itself")
axes[2].set_ylabel("equivariance term (log scale)")
for axis in axes:
    axis.set_xlabel("epoch")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7.5)
figure.suptitle("Mechanism validation on synthetic fixtures. Not a gait result. The absolute variant "
                "drives its own term to zero by shrinking the head, leaving rho unmoved.", y=1.04)
figure.tight_layout()
mechanism_png = ARM2_DIR / "idea9_arm2_mechanism_validation.png"
figure.savefig(mechanism_png, dpi=170, bbox_inches="tight")
plt.close(figure)
display(Image(filename=str(mechanism_png)))

MECHANISM = {
    "notebook": "new_nb_09_01_mechanism_and_smoke_validation",
    "mode": "smoke (synthetic fixtures only; not a gait result)",
    "check_1_head_only_is_a_no_op": {
        "wiring_slope": wiring_slope, "loss_value": head_only_value,
        "gradient_into_head": head_only_grad, "passed": True,
        "meaning": "penalising a swap of the head's own tokens is an algebraic identity: zero loss, zero "
                   "gradient, trains nothing.",
    },
    "check_2_through_encoder_reaches_the_encoder": {
        "by_variant": gradient_evidence, "passed": True,
        "meaning": "mirroring the raw skeleton and running both versions through the encoder makes every "
                   "variant a genuine constraint on encoder weights.",
    },
    "check_3_rho_calibration": {
        "rho_by_encoder": calibration_rho, "involution_gap": involution_gap, "passed": True,
        "meaning": "rho reads 0 at exact mirror equivariance and 4 at exact mirror blindness, so its "
                   "scale means what the contract says.",
    },
    "check_4_variant_bakeoff": {
        "epochs": MINI_EPOCHS, "equiv_weight": MINI_EQUIV_WEIGHT,
        "control_rho_view": control_rho,
        "control_late_epoch_std": control_noise,
        "gate_multiple": GATE_MULTIPLE, "gate_threshold": gate_threshold,
        "rows": bakeoff_rows,
        "selected_variant": SELECTED_VARIANT,
        "passed": True,
        "meaning": "the absolute variant satisfies itself by shrinking the head's output rather than "
                   "changing the encoder, so rho does not move. A scale-invariant variant removes that "
                   "escape route and does move rho by more than the control rung's trajectory noise.",
    },
    "gate_for_the_real_ladder": "PASSED",
    "selected_variant_for_the_real_ladder": SELECTED_VARIANT,
    "figure": mechanism_png.name,
    "trajectories": {label: frame.to_dict(orient="list") for label, frame in mini_runs.items()},
}
mechanism_path = ARM2_DIR / "idea9_arm2_mechanism_validation.json"
mechanism_path.write_text(json.dumps(MECHANISM, indent=2), encoding="utf-8")
print(f"wrote {mechanism_path}")
print(f"the real ladder in new_nb_09_02 must use variant: {SELECTED_VARIANT}")
'''))

    cells.append(md("""
## 5. What this notebook has established

Five things, none of them about gait.

The head-only formulation of the equivariance term is an exact no-op, measured rather than argued: zero
loss and zero gradient. The through-the-encoder form delivers gradient to encoder weights, which is the
property that makes it a real constraint. The primary endpoint reads exactly 0 on an encoder built to be
mirror-equivariant and exactly 4 on one built to be mirror-blind, so the scale in the contract is the scale
the numbers are on.

The fourth finding is the one that changed the design. The equivariance term as `nb_09b` writes it has a
degenerate solution: because the head is trainable and the residual is an unnormalised squared magnitude,
the optimiser can satisfy the term by shrinking the head's output instead of changing the encoder. That is
what happens, and the loss curve gives no warning: the term falls by more than two orders of magnitude
while the parameter-free endpoint stays where the control rung leaves it. Had this series simply followed
`nb_09b`'s section 8 recipe, it would have spent hours of real compute on a term that teaches the encoder
nothing, and the resulting loss curves would have looked like a success.

The fifth is the repair. Normalising the residual by the signal's own magnitude removes the escape route,
and a scale-invariant variant does move `rho` by more than the control rung's trajectory noise. That
variant is recorded in the bundle and is the one the real ladder uses.

One incidental observation invites an extrapolation that should be resisted. Every encoder measured here,
randomly initialised or trained, sits near `rho = 3.9`, close to the mirror-blind end of the scale. It is
tempting to read that as a forecast: if real encoders sit in the same place, this architecture barely
distinguishes a body from its reflection, which would explain item 05's informative null and Arm 1's
artifact verdict in one stroke.

The real ladder contradicted that forecast, and the paragraph is left here rather than deleted because the
error is instructive. Real control rungs average target-encoder `rho = 0.462`; their view-encoder values
range from 0.513 to 0.601. Both are far closer to mirror-honest than to mirror-blind. The fixtures use a
one-layer 32-dimensional toy encoder whose mirror-blindness is a design choice, not an empirical finding, so
their absolute magnitudes calibrate the metric's scale and nothing else. A full-width encoder trained on the
real curriculum arrives most of the way to mirror consistency without being asked to. The lesson is that a
synthetic fixture can validate that a measurement means what you claim while telling you nothing about where
real data will land on it, and that the two roles are easy to conflate when the fixture number is vivid.

What this notebook cannot tell us is whether the effect survives at real scale, on real pose data, across
seeds, or whether it costs anything in representation quality. Those are the questions the next notebook
spends real compute on.
""".rstrip()))

    write_notebook(ROOT / "new_nb_09_01_mechanism_and_smoke_validation.ipynb", cells)


# ---------------------------------------------------------------------------
# Notebook 02: the real multi-seed ladder
# ---------------------------------------------------------------------------

def build_nb02():
    global _CELL_N
    _CELL_N = [0]
    cells = []

    cells.append(md(COLAB_BADGE.format(name="new_nb_09_02_real_multiseed_equivariant_training") + """

# new_nb_09_02. The real ladder: control and treatment across seeds

This is the expensive notebook. It runs the full curriculum from notebook 04 on the real pose cohort,
twice per seed: once with the equivariance term switched off, and once with it on. Nothing else differs
between the two rungs of a seed, not the initialisation, not the batch order, not the mask sampling.

Read `new_nb_09_00` for the pre-registered endpoints and `new_nb_09_01` for the mechanism checks. This
notebook refuses to run if the mechanism gate has not passed, because a term that trains nothing would
turn hours of compute into a very expensive null.

**This notebook needs a GPU and real wall-clock time.** The saved Apple MPS artifacts record 11.7 to 12.4
minutes for control rungs and 16.6 to 19.4 minutes for treatment rungs. These are run records, not a
portable timing estimate. The notebook writes a checkpoint
and a JSON per rung, so an interrupted run resumes at the next rung instead of starting over.

**Research use only.** Folder labels are dataset annotations, not clinical diagnoses.
""".rstrip()))

    cells.append(md("""
## 1. Why the seed has to be threaded, and what breaks if it is not

Notebook 04 fixes its seed in two places. The per-stage batch sampler is created as
`np.random.default_rng(42 + 1000 * stage["stage"])`, and the fingerprint payload records `"seed": 42`.
Both are literals.

Copied verbatim into a multi-seed ladder, that would be quietly fatal. Every seed would draw the same
batch order and write the same fingerprint, so five nominal seeds would produce five identical runs. The
seed spread, which is the yardstick the whole credit rule is measured against, would come out as exactly
zero, and any difference at all between control and treatment would appear to clear it.

This notebook therefore threads one seed through every source of randomness: model initialisation, the
batch sampler, and the mask sampler. It also adds `seed`, `equiv_weight`, `equiv_on`, and the equivariance
variant to the fingerprint payload, so each rung has its own identity and no rung can be mistaken for the
published baseline lineage.

One deliberate deviation from notebook 04 is worth stating. Notebook 04 computes its representation
diagnostics after every epoch. Doing that here would roughly double the cost of the ladder for monitoring
alone, so diagnostics and `rho` are measured every twenty-fifth epoch and at every stage boundary. The
optimiser path is untouched, so this changes what is observed, not what is trained.
""".rstrip()))

    cells.append(code(SETUP + '''

MODE = REQUESTED_MODE
if MODE != "real":
    raise RuntimeError(
        "This notebook is the REAL ladder. Set GAVD_MODE=real in experiments/sjepa/gavd6-pm/.env and "
        "restart the kernel. Use new_nb_09_01 for the runnable synthetic mechanism checks.")
ARTIFACT_DIR = artifact_dir_for(MODE)
ARM2_DIR = ARTIFACT_DIR / ARM2_DIR_NAME
ARM2_DIR.mkdir(parents=True, exist_ok=True)

CONTRACT_PATH = ARM2_DIR / "idea9_arm2_contract.json"
MECHANISM_PATH = ARM2_DIR / "idea9_arm2_mechanism_validation.json"
if not CONTRACT_PATH.is_file():
    raise FileNotFoundError(f"Run new_nb_09_00 first: {CONTRACT_PATH} is missing")
if not MECHANISM_PATH.is_file():
    raise FileNotFoundError(f"Run new_nb_09_01 first: {MECHANISM_PATH} is missing")
CONTRACT = json.loads(CONTRACT_PATH.read_text())
MECHANISM = json.loads(MECHANISM_PATH.read_text())
if MECHANISM.get("gate_for_the_real_ladder") != "PASSED":
    raise RuntimeError(
        "The mechanism gate in new_nb_09_01 did not pass. A term that trains nothing would make this "
        "ladder an expensive null. Fix the mechanism first.")
EQUIV_VARIANT = MECHANISM["selected_variant_for_the_real_ladder"]

print(f"mode                : {MODE}")
print(f"output directory    : {ARM2_DIR}")
print(f"mechanism gate      : {MECHANISM['gate_for_the_real_ladder']}")
print(f"equivariance variant: {EQUIV_VARIANT}")
print(f"baseline reference  : {CONTRACT['contract']['baseline_fingerprint'][:16]} "
      f"({CONTRACT['contract']['baseline_checkpoint']})")
'''))

    cells.append(code(MODEL_CLASSES + PREPROCESSING + EQUIV_CODE + TRAIN_HELPERS + DATA_LOADING + '''

print("definitions loaded: model, preprocessing, equivariance, training helpers, data loaders.")
'''))

    cells.append(md("""
## 2. The corpus and the training configuration

The ladder trains on exactly what produced the baseline: the locked canonical cohort plus the opt-in
augmentation-normal pool. The recipe is notebook 04's recommended real profile, unchanged, so that the
control rung is a faithful reproduction of the baseline recipe rather than a cheaper approximation of it.
Three hundred normal-only epochs, then seventy-five epochs per added condition with balanced replay, at
width 96 and encoder depth 4.

The curriculum's hard data boundary is preserved: stage 0 sees only normal gait, and no condition tensor is
loaded for optimisation until the normal-only stage has finished.
""".rstrip()))

    cells.append(code('''
FRAMES, SEGMENT_LENGTH, EMBED_DIM = 64, 4, 96
SEGMENTS = FRAMES // SEGMENT_LENGTH
ENCODER_DEPTH, PREDICTOR_DEPTH, HEADS = 4, 2, 4
NORMAL_EPOCHS = int(os.getenv("SJEPA_NORMAL_EPOCHS", "300"))
FINETUNE_EPOCHS = int(os.getenv("SJEPA_FINETUNE_EPOCHS", "75"))
SAMPLES_PER_CONDITION = 4
MASK_FRACTION, VICREG_WEIGHT, GROUP_WEIGHT, GROUP_MARGIN = 0.60, 0.05, 0.25, 1.0
NORMAL_LR, FINETUNE_LR, EMA_START = 1e-3, 3e-4, 0.999
EQUIV_WEIGHT = float(os.getenv("IDEA9_ARM2_EQUIV_WEIGHT", "0.02"))
MONITOR_EVERY = int(os.getenv("IDEA9_ARM2_MONITOR_EVERY", "25"))
SEEDS = [int(s) for s in os.getenv("IDEA9_ARM2_SEEDS", "0,1,2").split(",") if s.strip()]

CURRICULUM = [
    {"stage": 0, "name": "normal_only", "add": "normal", "conditions": ["normal"]},
    {"stage": 1, "name": "add_parkinsons", "add": "parkinsons", "conditions": CONDITIONS[:2]},
    {"stage": 2, "name": "add_stroke", "add": "stroke", "conditions": CONDITIONS[:3]},
    {"stage": 3, "name": "add_myopathic", "add": "myopathic", "conditions": CONDITIONS[:4]},
    {"stage": 4, "name": "add_cerebralpalsy", "add": "cerebralpalsy", "conditions": CONDITIONS[:5]},
]
MAPPING_RELATIVE_PATH = Path("experiments/multiple-sclerosis/mapping-data/ms-pd-mapping.md")
MAPPING_SHA256 = hashlib.sha256((PROJECT_ROOT / MAPPING_RELATIVE_PATH).read_bytes()).hexdigest()
MODEL_CONFIG = {"frames": FRAMES, "joints": 33, "coordinate_dim": 3,
                "segment_length": SEGMENT_LENGTH, "embed_dim": EMBED_DIM,
                "encoder_depth": ENCODER_DEPTH, "predictor_depth": PREDICTOR_DEPTH, "heads": HEADS}

records = canonical_pose_records(ARTIFACT_DIR / "poses") + augmented_normal_records(ARTIFACT_DIR)
ALL_DATA = {}
for condition in CONDITIONS:
    subset = [r for r in records if r["condition"] == condition]
    prepared = [prepare_sequence(r["raw"], FRAMES) for r in subset]
    ALL_DATA[condition] = {
        "xyz": np.stack([p[0] for p in prepared]).astype(np.float32),
        "valid": np.stack([p[1] for p in prepared]),
        "records": subset,
    }
    print(f"{condition:14s} {ALL_DATA[condition]['xyz'].shape}  "
          f"{len({r['video_id'] for r in subset})} source videos")

# The canonical subset is the evaluation cohort, matching Arm 1's canonical_subset_only flag.
CANONICAL_XYZ = np.concatenate([
    ALL_DATA[c]["xyz"][[i for i, r in enumerate(ALL_DATA[c]["records"]) if r["cohort"] == "canonical"]]
    for c in CONDITIONS])
print(f"\\ntraining corpus : {sum(len(ALL_DATA[c]['xyz']) for c in CONDITIONS)} sequences")
print(f"canonical subset: {len(CANONICAL_XYZ)} sequences (the monitoring and evaluation cohort)")

device = pick_device()
print(f"\\ndevice: {device}")
if device.type == "cpu":
    print("WARNING: no GPU was found. On CPU this ladder takes many hours. Launch it outside any "
          "sandbox so Metal or CUDA is visible.")
'''))

    cells.append(md("""
## 3. One rung of the ladder

`run_rung` trains one complete five-stage curriculum for one seed at one equivariance weight, then saves a
checkpoint and a JSON summary. The training step is notebook 04's, with a single addition: when the
equivariance weight is nonzero, the raw batch and its anatomical mirror are both run through the view
encoder and the scale-invariant residual is added to the total loss.

Two invariants are asserted on every step. The target encoder must receive no gradient, so the EMA teacher
is never trained directly. And the total loss must stay finite, so a diverging rung fails loudly instead of
writing a quietly corrupted checkpoint.

Resumption is by file. If a rung's JSON already exists with a matching configuration, the rung is skipped
and its result is loaded, so the ladder can be stopped and restarted at any point.
""".rstrip()))

    cells.append(code('''
def rung_paths(rung, seed):
    return (ARM2_DIR / f"idea9_arm2_{rung}_seed{seed}.json",
            ARM2_DIR / f"sjepa_arm2_{rung}_seed{seed}.pt")


def monitor_encoder(encoder, xyz, device):
    numerators, denominators = mirror_residual_terms(encoder, xyz, SEGMENTS, EMBED_DIM, device)
    return rho_from_terms(numerators, denominators)


@torch.no_grad()
def target_authorized_embeddings(model, arrays, validity, device, batch_size=16):
    model.target_encoder.eval()
    vectors = []
    for start in range(0, len(arrays), batch_size):
        batch = torch.tensor(arrays[start:start + batch_size], dtype=torch.float32, device=device)
        valid = torch.tensor(validity[start:start + batch_size], dtype=torch.bool, device=device)
        valid_patch = valid.reshape(len(batch), SEGMENTS, SEGMENT_LENGTH, 33).all(dim=2)
        tokens = model.target_encoder(batch).reshape(len(batch), SEGMENTS, 33, EMBED_DIM)
        vectors.append(authorized_pool(tokens, valid_patch).cpu())
    return torch.cat(vectors)


def representation_diagnostics(model, data_by_condition, active_conditions, device):
    """Notebook 04's collapse and separation monitors, on the frozen EMA target encoder."""
    arrays = np.concatenate([data_by_condition[c]["xyz"] for c in active_conditions])
    validity = np.concatenate([data_by_condition[c]["valid"] for c in active_conditions])
    labels = np.concatenate([np.full(len(data_by_condition[c]["xyz"]), i, dtype=np.int64)
                             for i, c in enumerate(active_conditions)])
    vectors = target_authorized_embeddings(model, arrays, validity, device)
    unit = F.normalize(vectors, dim=1)
    cosine = unit @ unit.T
    eye = torch.eye(len(unit), dtype=torch.bool)
    centroids = torch.stack([F.normalize(unit[torch.tensor(labels) == v].mean(dim=0), dim=0)
                             for v in range(len(active_conditions))])
    pairwise = (centroids[:, None] - centroids[None, :]).square().sum(dim=-1).clamp_min(1e-12).sqrt()
    upper = torch.triu(torch.ones_like(pairwise, dtype=torch.bool), diagonal=1)
    distances = pairwise[upper]
    return {
        "feature_std": float(vectors.std(dim=0, unbiased=False).mean()),
        "mean_pair_cosine": float(cosine[~eye].mean()) if len(unit) > 1 else float("nan"),
        "minimum_centroid_distance": float(distances.min()) if len(distances) else float("nan"),
        "mean_centroid_distance": float(distances.mean()) if len(distances) else float("nan"),
    }


def run_rung(rung, seed, equiv_weight, variant=EQUIV_VARIANT, force=False):
    """One complete five-stage curriculum for one seed at one equivariance weight."""
    json_path, checkpoint_path = rung_paths(rung, seed)
    if json_path.is_file() and checkpoint_path.is_file() and not force:
        existing = json.loads(json_path.read_text())
        if (existing.get("seed") == seed and existing.get("equiv_weight") == equiv_weight
                and existing.get("equiv_variant") == variant
                and existing.get("normal_epochs") == NORMAL_EPOCHS
                and existing.get("finetune_epochs") == FINETUNE_EPOCHS):
            print(f"[{rung} seed {seed}] already complete, resuming from disk "
                  f"(rho_target {existing['final']['rho_target']:.5f})")
            return existing
        print(f"[{rung} seed {seed}] on-disk result has a different configuration; recomputing")

    equiv_on = equiv_weight > 0.0
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = SJEPAGait(**MODEL_CONFIG).to(device)
    projector = VICRegProjector(EMBED_DIM).to(device)
    head = AntisymmetricHead(EMBED_DIM, out_dim=4).to(device)
    if any(p.requires_grad for p in model.target_encoder.parameters()):
        raise AssertionError("the target encoder must start frozen")

    condition_data, history, monitors = {}, [], []
    rung_start = time.time()

    def train_stage(stage, epochs, learning_rate):
        active = stage["conditions"]
        trainable = [*[p for p in model.view_encoder.parameters() if p.requires_grad],
                     *[p for p in model.predictor.parameters() if p.requires_grad],
                     *projector.parameters(), *head.parameters()]
        optimizer = torch.optim.AdamW(trainable, lr=learning_rate, betas=(0.9, 0.95),
                                      weight_decay=0.05)
        steps_per_epoch = max(1, int(np.ceil(
            max(len(condition_data[c]["xyz"]) for c in active) / SAMPLES_PER_CONDITION)))
        total_steps = epochs * steps_per_epoch
        warmup = max(1, min(steps_per_epoch, total_steps // 10))

        def lr_factor(step):
            if step < warmup:
                return (step + 1) / warmup
            progress = (step - warmup) / max(total_steps - warmup - 1, 1)
            return 0.5 + 0.5 * (1.0 + math.cos(math.pi * progress)) / 2.0

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_factor)
        # Threaded seed. Notebook 04 hardcodes 42 here, which would make every seed identical.
        rng = np.random.default_rng(seed + 1000 * stage["stage"])
        global_step = 0
        for epoch in range(epochs):
            model.train()
            batch_rows = []
            for xyz_np, valid_np, labels_np in balanced_epoch_batches(
                    condition_data, active, SAMPLES_PER_CONDITION, rng):
                coordinates = torch.tensor(xyz_np, dtype=torch.float32, device=device)
                valid = torch.tensor(valid_np, dtype=torch.bool, device=device)
                labels = torch.tensor(labels_np, dtype=torch.long, device=device)
                valid_patch = valid.reshape(len(valid), SEGMENTS, SEGMENT_LENGTH, 33).all(dim=2)
                mask_np = uniform_neurologic_mask(
                    valid_patch.cpu().numpy(), mask_fraction=MASK_FRACTION,
                    seed=seed + 100000 * stage["stage"] + global_step)
                target_mask = torch.tensor(mask_np, dtype=torch.bool, device=device)
                view_a = geometric_view(coordinates)
                view_b = geometric_view(coordinates)

                prediction, target = model(view_a, coordinates, target_mask)
                jepa_loss = sjepa_cross_entropy(prediction, target, model.target_center)
                tokens_a = model.view_encoder(view_a).reshape(len(view_a), SEGMENTS, 33, EMBED_DIM)
                tokens_b = model.view_encoder(view_b).reshape(len(view_b), SEGMENTS, 33, EMBED_DIM)
                pooled_a = authorized_pool(tokens_a, valid_patch)
                pooled_b = authorized_pool(tokens_b, valid_patch)
                vicreg_loss, invariance, variance, covariance = vicreg_terms(
                    projector(pooled_a), projector(pooled_b))
                compactness, separation, _ = condition_group_terms(
                    pooled_a, labels, margin=GROUP_MARGIN)
                total_loss = (jepa_loss + VICREG_WEIGHT * vicreg_loss
                              + GROUP_WEIGHT * (compactness + separation))
                equiv_value = float("nan")
                if equiv_on:
                    equiv = equivariance_term(variant, head, model.view_encoder, coordinates,
                                              SEGMENTS, EMBED_DIM)
                    total_loss = total_loss + equiv_weight * equiv
                    equiv_value = float(equiv.detach())
                if not torch.isfinite(total_loss):
                    raise FloatingPointError(
                        f"non-finite loss in {rung} seed {seed} {stage['name']} step {global_step}")

                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                if any(p.grad is not None for p in model.target_encoder.parameters()):
                    raise AssertionError("the target encoder must never receive gradient")
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
                optimizer.step()
                scheduler.step()
                model.update_target(cosine_ema(global_step, total_steps, start=EMA_START, end=1.0))
                model.update_center(target, beta=0.9)
                batch_rows.append({
                    "total_loss": float(total_loss.detach()), "jepa_loss": float(jepa_loss.detach()),
                    "vicreg_loss": float(vicreg_loss.detach()),
                    "group_compactness": float(compactness.detach()),
                    "group_separation": float(separation.detach()),
                    "equiv_term": equiv_value,
                })
                global_step += 1

            last_epoch = epoch == epochs - 1
            if (epoch + 1) % MONITOR_EVERY == 0 or last_epoch:
                model.eval()
                row = pd.DataFrame(batch_rows).mean(numeric_only=True).to_dict()
                row.update({
                    "stage": stage["stage"], "stage_name": stage["name"],
                    "epoch_in_stage": epoch + 1, "optimizer_updates": global_step,
                    "rho_view": monitor_encoder(model.view_encoder, CANONICAL_XYZ, device),
                    "rho_target": monitor_encoder(model.target_encoder, CANONICAL_XYZ, device),
                    "head_signal_scale": head_signal_scale(
                        head, model.view_encoder, CANONICAL_XYZ, SEGMENTS, EMBED_DIM, device),
                    **representation_diagnostics(model, condition_data, active, device),
                    "elapsed_s": time.time() - rung_start,
                })
                monitors.append(row)
                print(f"  [{rung} s{seed}] stage {stage['stage']} epoch {epoch + 1:3d}  "
                      f"JEPA {row['jepa_loss']:.4f}  equiv {row['equiv_term']:.5f}  "
                      f"rho_view {row['rho_view']:.4f}  rho_target {row['rho_target']:.4f}  "
                      f"std {row['feature_std']:.4f}  {row['elapsed_s'] / 60:.1f} min")
            history.append({"stage": stage["stage"], "epoch_in_stage": epoch + 1,
                            **pd.DataFrame(batch_rows).mean(numeric_only=True).to_dict()})
        return total_steps

    # Stage 0 is the hard data boundary: only normal gait exists in condition_data here.
    condition_data["normal"] = ALL_DATA["normal"]
    if list(condition_data) != ["normal"]:
        raise AssertionError("stage 0 must see only normal gait")
    completed = [{"stage": 0, "name": CURRICULUM[0]["name"],
                  "optimizer_updates": train_stage(CURRICULUM[0], NORMAL_EPOCHS, NORMAL_LR)}]
    for stage in CURRICULUM[1:]:
        condition_data[stage["add"]] = ALL_DATA[stage["add"]]
        if list(condition_data) != stage["conditions"]:
            raise AssertionError(f"condition order broken at {stage['name']}")
        completed.append({"stage": stage["stage"], "name": stage["name"],
                          "optimizer_updates": train_stage(stage, FINETUNE_EPOCHS, FINETUNE_LR)})

    fingerprint_payload = {
        "series": "new_nb_09", "arm": "arm2", "rung": rung, "mode": MODE,
        "seed": seed, "equiv_weight": equiv_weight, "equiv_on": equiv_on, "equiv_variant": variant,
        "model_config": MODEL_CONFIG, "curriculum": CURRICULUM,
        "normal_epochs": NORMAL_EPOCHS, "finetune_epochs": FINETUNE_EPOCHS,
        "samples_per_condition": SAMPLES_PER_CONDITION, "mask_fraction": MASK_FRACTION,
        "mask_keypoints": MASK_KEYPOINTS, "ema_start": EMA_START,
        "vicreg_weight": VICREG_WEIGHT, "group_weight": GROUP_WEIGHT, "group_margin": GROUP_MARGIN,
        "mapping_sha256": MAPPING_SHA256,
        "include_augmented_normal": INCLUDE_AUGMENTED,
        "sequence_ids": sorted(r["sequence_id"] for r in records),
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True).encode("utf-8")).hexdigest()
    torch.save({
        "model_state": model.state_dict(), "vicreg_projector_state": projector.state_dict(),
        "antisymmetric_head_state": head.state_dict(), "config": MODEL_CONFIG, "mode": MODE,
        "mask_keypoints": MASK_KEYPOINTS, "dataset_fingerprint": fingerprint,
        "fingerprint_payload": fingerprint_payload, "curriculum_complete": True,
        "series": "new_nb_09", "arm": "arm2", "rung": rung, "seed": seed,
        "equiv_weight": equiv_weight, "equiv_variant": variant,
        "parent_fingerprint": None, "training_history": history,
    }, checkpoint_path)

    final = monitors[-1]
    result = {
        "rung": rung, "seed": seed, "equiv_weight": equiv_weight, "equiv_on": equiv_on,
        "equiv_variant": variant, "mode": MODE,
        "normal_epochs": NORMAL_EPOCHS, "finetune_epochs": FINETUNE_EPOCHS,
        "dataset_fingerprint": fingerprint, "checkpoint": checkpoint_path.name,
        "wall_clock_minutes": (time.time() - rung_start) / 60.0,
        "completed_stages": completed,
        "final": {k: final[k] for k in ("rho_view", "rho_target", "head_signal_scale", "feature_std",
                                        "mean_pair_cosine", "minimum_centroid_distance",
                                        "mean_centroid_distance", "jepa_loss", "equiv_term")},
        "monitors": monitors,
    }
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[{rung} seed {seed}] done in {result['wall_clock_minutes']:.1f} min  "
          f"fingerprint {fingerprint[:16]}  rho_target {final['rho_target']:.5f}")
    return result
'''))

    cells.append(md("""
## 4. Run the ladder

Two rungs per seed. `D0` is the control at weight zero and `E1` is the treatment. The rungs alternate by
seed so that a partial run still yields matched pairs: if the ladder is interrupted after four rungs, two
complete seeds are available rather than all the controls and none of the treatments.

Each rung prints a monitoring line every twenty-fifth epoch. `rho_view` is the endpoint on the network the
term optimises and `rho_target` is the endpoint on the EMA teacher that downstream readouts consume.
`new_nb_09_01` found that the EMA transfer is slow at small scale, so watching the two diverge or converge
here is informative in its own right.
""".rstrip()))

    cells.append(code('''
LADDER = [(rung, seed) for seed in SEEDS for rung in ("D0", "E1")]
print(f"ladder: {len(LADDER)} rungs over seeds {SEEDS}")
print(f"measured cost: about 20 min per D0 rung and 27 min per E1 rung on Apple MPS")
print(f"estimated total: about {(19.5 + 27.4) * len(SEEDS) / 60:.1f} hours\\n")

ladder_start = time.time()
results = []
for rung, seed in LADDER:
    weight = 0.0 if rung == "D0" else EQUIV_WEIGHT
    print(f"=== {rung} seed {seed} (equiv_weight={weight}) ===")
    results.append(run_rung(rung, seed, weight))
print(f"\\nladder finished in {(time.time() - ladder_start) / 60:.1f} min")

ladder_frame = pd.DataFrame([{
    "rung": r["rung"], "seed": r["seed"], "equiv_weight": r["equiv_weight"],
    "fingerprint": r["dataset_fingerprint"][:12],
    **{k: r["final"][k] for k in ("rho_view", "rho_target", "head_signal_scale", "feature_std",
                                  "mean_pair_cosine", "minimum_centroid_distance", "jepa_loss")},
    "minutes": r["wall_clock_minutes"],
} for r in results])
display(ladder_frame.round(5))

if ladder_frame["fingerprint"].nunique() != len(ladder_frame):
    raise AssertionError(
        "every rung must have its own fingerprint; a collision means the seed or the weight failed to "
        "reach the fingerprint payload")
print("all rung fingerprints are distinct: the seed and the weight both reached the payload.")

ladder_path = ARM2_DIR / "idea9_arm2_ladder.json"
ladder_path.write_text(json.dumps({
    "series": "new_nb_09", "arm": "arm2", "mode": MODE,
    "equiv_variant": EQUIV_VARIANT, "equiv_weight": EQUIV_WEIGHT, "seeds": SEEDS,
    "monitor_every_epochs": MONITOR_EVERY,
    "normal_epochs": NORMAL_EPOCHS, "finetune_epochs": FINETUNE_EPOCHS,
    "device": str(device),
    "baseline_reference_fingerprint": CONTRACT["contract"]["baseline_fingerprint"],
    "rungs": results,
}, indent=2), encoding="utf-8")
print(f"wrote {ladder_path}")
'''))

    cells.append(code('''
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from IPython.display import Image, display

figure, axes = plt.subplots(1, 3, figsize=(16, 4.4))
for result in results:
    frame = pd.DataFrame(result["monitors"])
    updates = np.arange(1, len(frame) + 1) * MONITOR_EVERY
    colour = "#555555" if result["rung"] == "D0" else "#c1272d"
    style = "--" if result["rung"] == "D0" else "-"
    label = f"{result['rung']} seed {result['seed']}"
    axes[0].plot(updates, frame["rho_view"], style, color=colour, alpha=0.8, label=label)
    axes[1].plot(updates, frame["rho_target"], style, color=colour, alpha=0.8, label=label)
    axes[2].plot(updates, frame["feature_std"], style, color=colour, alpha=0.8, label=label)
axes[0].set_title("rho on the view encoder\\n(the network the term optimises)")
axes[1].set_title("rho on the target encoder\\n(the EMA artifact readouts consume)")
axes[2].set_title("feature standard deviation\\n(collapse guardrail)")
for axis, ylabel in zip(axes, ["rho", "rho", "feature std"]):
    axis.set_xlabel("epoch (monitoring cadence)")
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7)
figure.suptitle("Real ladder: dashed is the D0 control, solid is the E1 treatment", y=1.03)
figure.tight_layout()
ladder_png = ARM2_DIR / "idea9_arm2_ladder_trajectories.png"
figure.savefig(ladder_png, dpi=170, bbox_inches="tight")
plt.close(figure)
display(Image(filename=str(ladder_png)))
print(f"wrote {ladder_png}")
'''))

    cells.append(md("""
## 5. What this notebook has produced, and what it deliberately has not

There are now two trained checkpoints per seed that differ in exactly one term of the objective, each with
its own fingerprint, each with a recorded monitoring trajectory. That is the raw material for the
comparison.

No conclusion is drawn here, on purpose. The endpoint is measured on the canonical cohort during training
as a monitor, but the scoring that the credit rule refers to, including the source-grouped probes, the
guardrails, and the paired bootstrap, all happens in `new_nb_09_03` against the saved checkpoints. Keeping
the scoring in a separate notebook means it can be rerun and audited without retraining anything, and it
keeps the temptation to adjust the rule after seeing the trajectories out of reach.
""".rstrip()))

    write_notebook(ROOT / "new_nb_09_02_real_multiseed_equivariant_training.ipynb", cells)


# ---------------------------------------------------------------------------
# Notebook 03: evaluation, results, discussion
# ---------------------------------------------------------------------------
def build_nb03():
    cells = []

    cells.append(md(COLAB_BADGE.format(name="new_nb_09_03_evaluation_results_discussion") + """

# new_nb_09_03. Evaluation, results, and discussion

This notebook scores the ladder that `new_nb_09_02` trained and applies the rule that `new_nb_09_00`
pre-registered. It trains nothing. Everything here reads saved checkpoints, so it can be rerun and audited
without spending compute, and any disagreement between the numbers printed here and the numbers printed
during training is itself a finding worth chasing rather than a nuisance.

The order matters, and it is deliberate. The rule is restated from the contract file before any result is
loaded. Then the endpoints are computed. Then the rule is applied mechanically. Writing it the other way
round, with the numbers visible while the threshold is being chosen, is how a preregistration quietly turns
into a story.

Four questions get answered, in this order.

1. Did the equivariance term change the encoder on real data, measured on the primary endpoint?
2. Does that change exceed what seed-to-seed variation produces on its own?
3. Did anything get worse: representation spread, condition geometry, or a grouped condition probe?
4. What does the answer license us to claim, and what does it not?
""".rstrip()))

    cells.append(code(SETUP + '''
MODE = REQUESTED_MODE if REQUESTED_MODE in {"smoke", "real"} else "smoke"
ARTIFACT_DIR = artifact_dir_for(MODE)
ARM2_DIR = ARTIFACT_DIR / ARM2_DIR_NAME

CONTRACT_PATH = ARM2_DIR / "idea9_arm2_contract.json"
if not CONTRACT_PATH.is_file():
    raise FileNotFoundError(f"Run new_nb_09_00 first: {CONTRACT_PATH} is missing")
CONTRACT = json.loads(CONTRACT_PATH.read_text())
PRE_REGISTERED = CONTRACT["pre_registered"]

print("The rule, restated from the contract file BEFORE any result is loaded:")
print(f"  primary endpoint : {PRE_REGISTERED['primary_endpoint']['name']}")
print(f"  scale            : {PRE_REGISTERED['primary_endpoint']['scale']}")
for key, text in PRE_REGISTERED["credit_rule"].items():
    if key != "all_three_required":
        print(f"  {key:32s} {text}")
print(f"  guardrails       : {', '.join(PRE_REGISTERED['guardrails'])}")
print(f"\\n  registered seeds : {PRE_REGISTERED['ladder']['seeds']}")
'''))

    cells.append(code(MODEL_CLASSES + PREPROCESSING + EQUIV_CODE + TRAIN_HELPERS + DATA_LOADING + '''

print("definitions loaded. Nothing in this notebook trains; the training helpers are imported because "
      "the guardrail geometry reuses notebook 04's pooling and centroid code verbatim.")
'''))

    cells.append(md("""
## 1. Load the cohort once, and the rungs that actually finished

The evaluation cohort is the canonical subset: the 96 sequences with dataset annotations, excluding the
opt-in augmentation-normal pool that participates in training. That matches Arm 1's evaluation cohort, so
the tertiary R-squared numbers here are on the same ruler as Arm 1's.

Rungs are discovered from disk rather than assumed. A rung counts only if its JSON exists, its checkpoint
exists, and its recorded epoch counts match the real profile. A rung that was interrupted mid-curriculum is
excluded and named, because silently averaging a half-trained rung into a control mean is the kind of error
that produces a confident wrong answer.
""".rstrip()))

    cells.append(code('''
FRAMES, SEGMENT_LENGTH, EMBED_DIM = 64, 4, 96
SEGMENTS = FRAMES // SEGMENT_LENGTH
ENCODER_DEPTH, PREDICTOR_DEPTH, HEADS = 4, 2, 4
REAL_NORMAL_EPOCHS, REAL_FINETUNE_EPOCHS = 300, 75
MODEL_CONFIG = {"frames": FRAMES, "joints": 33, "coordinate_dim": 3,
                "segment_length": SEGMENT_LENGTH, "embed_dim": EMBED_DIM,
                "encoder_depth": ENCODER_DEPTH, "predictor_depth": PREDICTOR_DEPTH, "heads": HEADS}

records = [r for r in canonical_pose_records(ARTIFACT_DIR / "poses") if r["cohort"] == "canonical"]
prepared = [prepare_sequence(r["raw"], FRAMES) for r in records]
EVAL_XYZ = np.stack([p[0] for p in prepared]).astype(np.float32)
EVAL_VALID = np.stack([p[1] for p in prepared])
EVAL_CONDITION = np.array([r["condition"] for r in records])
EVAL_VIDEO = np.array([r["video_id"] for r in records])
EVAL_Y = np.array([signed_left_minus_right(p[0]) for p in prepared], dtype=np.float64)

SOURCES_PER_CONDITION = {c: len(set(EVAL_VIDEO[EVAL_CONDITION == c])) for c in CONDITIONS}
GROUPED_PROBE_EVALUABLE = min(SOURCES_PER_CONDITION.values()) >= 2

print(f"evaluation cohort: {len(EVAL_XYZ)} canonical sequences, "
      f"{len(set(EVAL_VIDEO))} source videos")
print(pd.DataFrame({"sequences": pd.Series(EVAL_CONDITION).value_counts().reindex(CONDITIONS),
                    "source_videos": pd.Series(SOURCES_PER_CONDITION)}).to_string())
if not GROUPED_PROBE_EVALUABLE:
    print("\\nNOTE: at least one condition has a single source video, so a source-grouped condition "
          "probe cannot be built. This is a cohort fact, known before any result is loaded.")

device = pick_device()
print(f"\\ndevice: {device}")


def discovered_rungs():
    found, rejected = [], []
    for path in sorted(ARM2_DIR.glob("idea9_arm2_[DE]*_seed*.json")):
        bundle = json.loads(path.read_text())
        checkpoint = ARM2_DIR / bundle["checkpoint"]
        if not checkpoint.is_file():
            rejected.append((path.name, "checkpoint missing"))
            continue
        if (bundle.get("normal_epochs"), bundle.get("finetune_epochs")) != (
                REAL_NORMAL_EPOCHS, REAL_FINETUNE_EPOCHS):
            rejected.append((path.name, f"epochs {bundle.get('normal_epochs')}"
                                        f"/{bundle.get('finetune_epochs')} are not the real profile"))
            continue
        if len(bundle.get("completed_stages", [])) != 5:
            rejected.append((path.name, "curriculum incomplete"))
            continue
        found.append(bundle)
    return found, rejected


RUNGS, REJECTED = discovered_rungs()
for name, why in REJECTED:
    print(f"EXCLUDED {name}: {why}")
if not RUNGS:
    raise RuntimeError("No completed rung was found. Run new_nb_09_02 to completion first.")

OBSERVED_SEEDS = sorted({b["seed"] for b in RUNGS})
PAIRED_SEEDS = sorted({b["seed"] for b in RUNGS if b["rung"] == "D0"}
                      & {b["seed"] for b in RUNGS if b["rung"] == "E1"})
print(f"\\ncompleted rungs : {len(RUNGS)}")
print(f"seeds observed  : {OBSERVED_SEEDS}")
print(f"seeds with BOTH rungs (usable for the paired comparison): {PAIRED_SEEDS}")
'''))

    cells.append(md("""
### A protocol deviation, recorded before the results

The contract registered five seeds. The ladder ran fewer, because the repaired recipe cost more per rung
than the estimate the budget was approved against: about twenty minutes per control rung and twenty-seven
per treatment rung on Apple MPS, so five seeds would have been close to four hours rather than the
two and a quarter that was agreed.

This is a real reduction in power and it is stated here rather than buried. Fewer seeds means the control
rung's seed spread, which is the yardstick the first condition of the credit rule is measured against, is
itself estimated from few samples. A standard deviation computed from three values is a wide-tailed
estimate, so condition 1 becomes both noisier and, because the estimate is small in expectation, easier to
pass by luck. The paired bootstrap in condition 2 is over source videos rather than seeds and is unaffected
in its own terms, but it cannot compensate for a thin seed sample.

The practical consequence is a limit on the verdict's strength, not on its direction. A large, consistent
effect remains readable with three seeds. A marginal one would not be, and would have to be reported as
undecided rather than as a pass.
""".rstrip()))

    cells.append(code('''
DEVIATIONS = []
registered_seeds = PRE_REGISTERED["ladder"]["seeds"]
if OBSERVED_SEEDS != registered_seeds:
    DEVIATIONS.append({
        "field": "ladder.seeds",
        "registered": registered_seeds,
        "actual": OBSERVED_SEEDS,
        "reason": "measured per-rung cost on MPS exceeded the estimate the wall-clock budget was "
                  "approved against; seeds were cut to keep the approved budget",
        "effect_on_inference": "the D0 seed-spread yardstick in credit condition 1 is estimated from "
                               "fewer samples, so a marginal effect cannot be adjudicated",
    })
for name, why in REJECTED:
    DEVIATIONS.append({"field": "rung", "registered": name, "actual": "excluded",
                       "reason": why, "effect_on_inference": "reduces the paired sample"})
if not GROUPED_PROBE_EVALUABLE:
    DEVIATIONS.append({
        "field": "guardrails.source_grouped_five_class_balanced_accuracy",
        "registered": "source-grouped five-class balanced accuracy",
        "actual": "not evaluable; a leaky stratified probe substitutes as a damage detector only",
        "reason": f"source videos per condition are {SOURCES_PER_CONDITION}, and a condition with one "
                  "video leaves a source-grouped fold with nothing to learn that condition from",
        "effect_on_inference": "condition 3 is decided on the remaining guardrails; the substitute "
                               "probe leaks video identity and supports no condition claim",
    })

if DEVIATIONS:
    print("PROTOCOL DEVIATIONS (recorded before any endpoint is computed):")
    for d in DEVIATIONS:
        print(f"  {d['field']}: registered {d['registered']} -> actual {d['actual']}")
        print(f"    because {d['reason']}")
else:
    print("No protocol deviations: the ladder matches the registered design.")
'''))

    cells.append(md("""
## 2. Recompute the endpoints from the checkpoints

Nothing is taken from the training logs. Each checkpoint is reloaded, its fingerprint is checked against
its JSON, and every endpoint is recomputed here on the evaluation cohort with the same code for every rung.
That is what makes the control and the treatment comparable: not that they were trained carefully, but that
they are being measured by one instrument after the fact.

Four families of measurement come out of each checkpoint.

The **primary endpoint** is `rho` on the target encoder, the EMA teacher that downstream readouts actually
consume. It is returned per sequence, so it can be aggregated by source video for the paired bootstrap.

The **secondaries** are `rho` on the view encoder, which is the network the term optimises directly, and the
measured anatomical-mirror slope through the encoder. The gap between the two `rho` values measures how much
of the pressure survived the EMA transfer, which is a different question from whether the pressure worked.

The **guardrails** are feature standard deviation, mean pairwise cosine, and the condition geometry. A
treatment that improves symmetry by flattening the representation has bought the endpoint rather than earned
it, and the guardrails are how that shows up.

The **head scale** is the degenerate-solution check. `new_nb_09_01` showed that the original loss form could
be satisfied by shrinking the readout. If the head's output scale falls while `rho` improves, the improvement
is suspect regardless of what the credit rule says.
""".rstrip()))

    cells.append(code('''
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, balanced_accuracy_score
from sklearn.ensemble import RandomForestClassifier

ALPHAS = np.logspace(-3, 3, 13)


def load_rung_model(bundle):
    """Rebuild one rung's model and head from its checkpoint, asserting the fingerprint matches."""
    checkpoint = torch.load(ARM2_DIR / bundle["checkpoint"], map_location="cpu", weights_only=False)
    if checkpoint["dataset_fingerprint"] != bundle["dataset_fingerprint"]:
        raise AssertionError(f"{bundle['checkpoint']} does not match its JSON fingerprint")
    if checkpoint["config"] != MODEL_CONFIG:
        raise AssertionError(f"{bundle['checkpoint']} was trained at a different width or depth")
    model = SJEPAGait(**MODEL_CONFIG)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    head = AntisymmetricHead(EMBED_DIM, out_dim=4)
    head.load_state_dict(checkpoint["antisymmetric_head_state"])
    head.to(device).eval()
    return model, head


@torch.no_grad()
def pooled_target_features(model, batch_size=16):
    vectors = []
    for start in range(0, len(EVAL_XYZ), batch_size):
        chunk = torch.tensor(EVAL_XYZ[start:start + batch_size], dtype=torch.float32, device=device)
        valid = torch.tensor(EVAL_VALID[start:start + batch_size], dtype=torch.bool, device=device)
        patch = valid.reshape(len(chunk), SEGMENTS, SEGMENT_LENGTH, 33).all(dim=2)
        tokens = model.target_encoder(chunk).reshape(len(chunk), SEGMENTS, 33, EMBED_DIM)
        vectors.append(authorized_pool(tokens, patch).cpu().numpy())
    return np.concatenate(vectors)


@torch.no_grad()
def measured_mirror_slope(model, head, batch_size=16):
    """Least-squares slope of s(enc(Mx)) on s(enc(x)). A perfect antisymmetric encoder gives -1.

    This is MEASURED through the encoder, never asserted. Arm 1's head-only token swap gives exactly -1 by
    construction and therefore says nothing about the encoder.
    """
    original, mirrored = [], []
    for start in range(0, len(EVAL_XYZ), batch_size):
        chunk = torch.tensor(EVAL_XYZ[start:start + batch_size], dtype=torch.float32, device=device)
        tokens = model.target_encoder(chunk).reshape(len(chunk), SEGMENTS, 33, EMBED_DIM)
        tokens_mirrored = model.target_encoder(anatomical_mirror_coords(chunk)).reshape(
            len(chunk), SEGMENTS, 33, EMBED_DIM)
        original.append(head(tokens).cpu().numpy().ravel())
        mirrored.append(head(tokens_mirrored).cpu().numpy().ravel())
    x = np.concatenate(original)
    y = np.concatenate(mirrored)
    return float(np.polyfit(x, y, 1)[0])


def condition_geometry(features):
    """Minimum and mean centroid distance on unit vectors, matching notebook 04's diagnostic."""
    unit = features / np.linalg.norm(features, axis=1, keepdims=True).clip(1e-9)
    centroids = []
    for condition in CONDITIONS:
        rows = unit[EVAL_CONDITION == condition]
        centroid = rows.mean(axis=0)
        centroids.append(centroid / max(float(np.linalg.norm(centroid)), 1e-9))
    centroids = np.stack(centroids)
    distances = [float(np.linalg.norm(centroids[i] - centroids[j]))
                 for i in range(len(centroids)) for j in range(i + 1, len(centroids))]
    return float(np.min(distances)), float(np.mean(distances))


def grouped_condition_probe(features, seed=0):
    """The registered guardrail: source-grouped five-class balanced accuracy.

    Not evaluable on this cohort. Every canonical normal sequence comes from ONE source video, so any
    source-grouped split puts all of normal on one side and leaves a fold with no normal rows to learn
    from. Returning nan rather than quietly dropping to a non-grouped split keeps that visible.
    """
    if not GROUPED_PROBE_EVALUABLE:
        return float("nan"), 0
    folds = min(3, min(SOURCES_PER_CONDITION.values()))
    predictions = np.empty(len(EVAL_CONDITION), dtype=object)
    for train_index, test_index in GroupKFold(n_splits=folds).split(
            features, EVAL_CONDITION, EVAL_VIDEO):
        forest = RandomForestClassifier(n_estimators=300, random_state=seed, class_weight="balanced")
        forest.fit(features[train_index], EVAL_CONDITION[train_index])
        predictions[test_index] = forest.predict(features[test_index])
    return float(balanced_accuracy_score(EVAL_CONDITION, list(predictions))), folds


def leaky_condition_probe(features, seed=0):
    """A stand-in damage detector, NOT a performance claim, used when the grouped probe cannot run.

    This is a stratified five-fold split, so source videos cross it and video identity leaks. That makes
    the absolute number worthless as evidence about conditions. It is still a usable damage detector for
    the one job a guardrail has here: if the equivariance term destroyed condition information, a probe
    with every advantage available to it would still fall. A guardrail is allowed to be a weak test as
    long as nobody reads its value as a result.
    """
    from sklearn.model_selection import StratifiedKFold
    predictions = np.empty(len(EVAL_CONDITION), dtype=object)
    for train_index, test_index in StratifiedKFold(
            n_splits=5, shuffle=True, random_state=seed).split(features, EVAL_CONDITION):
        forest = RandomForestClassifier(n_estimators=300, random_state=seed, class_weight="balanced")
        forest.fit(features[train_index], EVAL_CONDITION[train_index])
        predictions[test_index] = forest.predict(features[test_index])
    return float(balanced_accuracy_score(EVAL_CONDITION, list(predictions)))


def antisymmetric_lane_r2(model, head, seed=0):
    """Arm 1's treatment lane recomputed on this checkpoint. Tertiary, beside the failed y-gate.

    Reported because the record should be complete, NOT because it is evidence: the contract records that
    the between-source variance gate failed on this cohort, which makes held-out-source R-squared
    uninterpretable here in either direction.
    """
    with torch.no_grad():
        features = []
        for start in range(0, len(EVAL_XYZ), 16):
            chunk = torch.tensor(EVAL_XYZ[start:start + 16], dtype=torch.float32, device=device)
            tokens = model.target_encoder(chunk).reshape(len(chunk), SEGMENTS, 33, EMBED_DIM)
            features.append(head(tokens).cpu().numpy())
    features = np.concatenate(features)
    folds = min(5, len(set(EVAL_VIDEO)))
    predictions = np.zeros(len(EVAL_Y))
    for train_index, test_index in GroupKFold(n_splits=folds).split(features, EVAL_Y, EVAL_VIDEO):
        scaler = StandardScaler().fit(features[train_index])
        inner_groups = EVAL_VIDEO[train_index]
        best_alpha, best_score = ALPHAS[0], -np.inf
        inner_folds = min(3, len(set(inner_groups)))
        for alpha in ALPHAS:
            if inner_folds < 2:
                best_alpha = 1.0
                break
            scores = []
            for inner_train, inner_test in GroupKFold(n_splits=inner_folds).split(
                    features[train_index], EVAL_Y[train_index], inner_groups):
                rows = train_index[inner_train]
                held = train_index[inner_test]
                inner_scaler = StandardScaler().fit(features[rows])
                ridge = Ridge(alpha=alpha).fit(inner_scaler.transform(features[rows]), EVAL_Y[rows])
                scores.append(r2_score(EVAL_Y[held], ridge.predict(inner_scaler.transform(features[held]))))
            if np.mean(scores) > best_score:
                best_alpha, best_score = alpha, float(np.mean(scores))
        ridge = Ridge(alpha=best_alpha).fit(scaler.transform(features[train_index]), EVAL_Y[train_index])
        predictions[test_index] = ridge.predict(scaler.transform(features[test_index]))
    return float(r2_score(EVAL_Y, predictions))


def score_rung(bundle):
    model, head = load_rung_model(bundle)
    numerator_t, denominator_t = mirror_residual_terms(
        model.target_encoder, EVAL_XYZ, SEGMENTS, EMBED_DIM, device)
    numerator_v, denominator_v = mirror_residual_terms(
        model.view_encoder, EVAL_XYZ, SEGMENTS, EMBED_DIM, device)
    features = pooled_target_features(model)
    minimum_centroid, mean_centroid = condition_geometry(features)
    probe, probe_folds = grouped_condition_probe(features)
    unit = features / np.linalg.norm(features, axis=1, keepdims=True).clip(1e-9)
    cosine = unit @ unit.T
    off = ~np.eye(len(unit), dtype=bool)
    return {
        "rung": bundle["rung"], "seed": bundle["seed"],
        "equiv_weight": bundle["equiv_weight"],
        "fingerprint": bundle["dataset_fingerprint"][:12],
        "rho_target": rho_from_terms(numerator_t, denominator_t),
        "rho_view": rho_from_terms(numerator_v, denominator_v),
        "mirror_slope": measured_mirror_slope(model, head),
        "head_scale": head_signal_scale(
            head, model.target_encoder, EVAL_XYZ, SEGMENTS, EMBED_DIM, device),
        "feature_std": float(features.std(axis=0).mean()),
        "mean_pair_cosine": float(cosine[off].mean()),
        "minimum_centroid_distance": minimum_centroid,
        "mean_centroid_distance": mean_centroid,
        "grouped_probe_balanced_accuracy": probe,
        "grouped_probe_folds": probe_folds,
        "leaky_probe_balanced_accuracy": leaky_condition_probe(features),
        "antisymmetric_lane_r2": antisymmetric_lane_r2(model, head),
        "_per_sequence_numerator": numerator_t,
        "_per_sequence_denominator": denominator_t,
    }


scored = [score_rung(b) for b in sorted(RUNGS, key=lambda b: (b["seed"], b["rung"]))]
SCORES = pd.DataFrame([{k: v for k, v in s.items() if not k.startswith("_")} for s in scored])
PER_SEQUENCE = {(s["rung"], s["seed"]): (s["_per_sequence_numerator"], s["_per_sequence_denominator"])
                for s in scored}
display(SCORES.round(4))
'''))

    cells.append(md("""
### Does the recomputation agree with the training logs?

It should agree exactly apart from floating-point noise, and checking is cheap. The training loop and this
notebook both measure `rho` on the same canonical 96 sequences with the same code. A disagreement would
mean either the checkpoint does not hold the state the log described or the two code paths differ, and
either would invalidate everything downstream. Guardrail diagnostics are different: the training monitor
uses the active training corpus and view encoder, while the final credit rule uses the canonical subset
and target encoder.
""".rstrip()))

    cells.append(code('''
comparison = []
for bundle in RUNGS:
    row = SCORES[(SCORES["rung"] == bundle["rung"]) & (SCORES["seed"] == bundle["seed"])].iloc[0]
    comparison.append({
        "rung": bundle["rung"], "seed": bundle["seed"],
        "logged_rho_target": bundle["final"]["rho_target"],
        "recomputed_rho_target": row["rho_target"],
        "absolute_difference": abs(bundle["final"]["rho_target"] - row["rho_target"]),
    })
comparison = pd.DataFrame(comparison)
display(comparison.round(5))

worst = float(comparison["absolute_difference"].max())
tolerance = 1e-6
print(f"largest disagreement: {worst:.8f} (tolerance {tolerance:g})")
if worst > tolerance:
    raise AssertionError(
        f"recomputed rho disagrees with the training log by {worst:.8f}, more than the {tolerance:g} "
        "allowed for floating-point noise. Both paths score the same canonical rows with the same "
        "measurement code, so check the checkpoint and measurement path before reading further.")
print("agreement is within tolerance: the checkpoints hold the state the logs described.")
'''))

    cells.append(md("""
## 3. Apply the credit rule, mechanically

The three conditions are applied exactly as written in the contract, in order, with no adjustment.

Condition 1 compares the mean improvement in `rho` against the control rung's own seed-to-seed standard
deviation. The logic is a trajectory control rather than a population claim: if switching the term on moves
the endpoint by less than simply changing the random seed does, then nothing has been demonstrated.

Condition 2 is a paired bootstrap over source videos. Pairing matters because the two rungs of a seed share
everything except the term, so the sensible unit of comparison is the per-source difference, and resampling
sources rather than sequences respects the fact that windows from one video are not independent.

Condition 3 checks that no guardrail fell by more than the control's own seed spread. This is what stops a
treatment from buying the endpoint with representation quality.

One guardrail cannot be measured on this cohort, and how that is handled matters. A guardrail that returns
no value is neither a pass nor a failure. Treating it as a failure would fabricate evidence of harm;
dropping it quietly would fabricate a clean sweep. It is reported by name as not evaluable, the reason is
recorded as a protocol deviation, and a deliberately weaker stand-in is reported in its place with its
weakness stated. A guardrail is allowed to be a weak test, because its job is to catch destruction rather
than to demonstrate benefit, but the reader has to be told which one they are looking at.
""".rstrip()))

    cells.append(code('''
def per_source_rho(rung, seed):
    """rho aggregated per source video: a ratio of sums within each video, not a mean of ratios."""
    numerator, denominator = PER_SEQUENCE[(rung, seed)]
    frame = pd.DataFrame({"video": EVAL_VIDEO, "numerator": numerator, "denominator": denominator})
    grouped = frame.groupby("video").sum()
    return (grouped["numerator"] / grouped["denominator"]).rename(f"{rung}_seed{seed}")


D0 = SCORES[SCORES["rung"] == "D0"]
E1 = SCORES[SCORES["rung"] == "E1"]
d0_mean, e1_mean = float(D0["rho_target"].mean()), float(E1["rho_target"].mean())
d0_spread = float(D0["rho_target"].std(ddof=1)) if len(D0) > 1 else float("nan")
improvement = d0_mean - e1_mean

print(f"D0 control rho_target : mean {d0_mean:.4f}  seed spread {d0_spread:.4f}  n={len(D0)}")
print(f"E1 treatment rho_target: mean {e1_mean:.4f}  n={len(E1)}")
print(f"improvement (D0 - E1) : {improvement:.4f}")

condition_1 = bool(np.isfinite(d0_spread) and improvement > d0_spread)
print(f"\\nCONDITION 1 improvement exceeds the control seed spread: "
      f"{improvement:.4f} > {d0_spread:.4f} -> {'PASS' if condition_1 else 'FAIL'}")
'''))

    cells.append(code('''
BOOTSTRAP_DRAWS = 4000
rng = np.random.default_rng(20260819)

paired = pd.concat([per_source_rho("D0", s) for s in PAIRED_SEEDS]
                   + [per_source_rho("E1", s) for s in PAIRED_SEEDS], axis=1)
d0_columns = [f"D0_seed{s}" for s in PAIRED_SEEDS]
e1_columns = [f"E1_seed{s}" for s in PAIRED_SEEDS]
per_source_difference = (paired[d0_columns].mean(axis=1) - paired[e1_columns].mean(axis=1)).to_numpy()
sources = paired.index.to_numpy()

draws = np.empty(BOOTSTRAP_DRAWS)
for i in range(BOOTSTRAP_DRAWS):
    take = rng.integers(0, len(per_source_difference), len(per_source_difference))
    draws[i] = per_source_difference[take].mean()
low, high = (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)))

print(f"paired over {len(sources)} source videos, seeds {PAIRED_SEEDS}")
print(f"mean per-source improvement  : {per_source_difference.mean():.4f}")
print(f"median per-source improvement: {float(np.median(per_source_difference)):.4f}")
print(f"95 percent percentile bootstrap interval: [{low:.4f}, {high:.4f}]")
print(f"sources improved: {int((per_source_difference > 0).sum())} of {len(per_source_difference)}")
print("\\nThe per-source mean is NOT comparable to the cohort-level improvement printed above. The "
      "cohort figure is one ratio of summed numerators to summed denominators, so sequences with large "
      "signal magnitudes dominate it. The per-source figure averages each video's own ratio, giving a "
      "video with a small denominator the same weight as one with a large denominator. Both are "
      "reported because the credit rule names the cohort figure for condition 1 and the per-source "
      "figure for condition 2, and mixing them up would inflate the apparent effect.")

condition_2 = bool(low > 0 or high < 0)
print(f"\\nCONDITION 2 paired bootstrap interval excludes zero -> {'PASS' if condition_2 else 'FAIL'}")
if condition_2 and low <= 0:
    print("NOTE: the interval excludes zero on the WRONG side; the treatment made the endpoint worse.")
'''))

    cells.append(code('''
GUARDRAILS = {
    "feature_std": ("higher is safer", "registered"),
    "mean_pair_cosine": ("lower is safer", "registered"),
    "grouped_probe_balanced_accuracy": ("higher is safer", "registered"),
    "leaky_probe_balanced_accuracy": ("higher is safer", "substitute for the grouped probe"),
    "minimum_centroid_distance": ("higher is safer", "additional, not registered"),
}
guardrail_rows, guardrail_failures, guardrail_unmeasurable = [], [], []
for name, (direction, provenance) in GUARDRAILS.items():
    control_mean = float(D0[name].mean())
    control_spread = float(D0[name].std(ddof=1)) if len(D0) > 1 else float("nan")
    treatment_mean = float(E1[name].mean())
    change = treatment_mean - control_mean
    regression = -change if direction.startswith("higher") else change
    measurable = bool(np.isfinite(regression) and np.isfinite(control_spread))
    within = bool(measurable and regression <= control_spread)
    guardrail_rows.append({
        "guardrail": name, "direction": direction, "provenance": provenance,
        "D0_mean": control_mean, "D0_seed_spread": control_spread,
        "E1_mean": treatment_mean, "change": change,
        "regression": regression, "measurable": measurable, "within_control_spread": within,
    })
    if not measurable:
        guardrail_unmeasurable.append(name)
    elif not within:
        guardrail_failures.append(name)

guardrail_frame = pd.DataFrame(guardrail_rows)
display(guardrail_frame.round(4))

# An unmeasurable guardrail is not a regression, and it is not a pass either. Counting nan as a failure
# would fabricate evidence of harm; dropping it silently would fabricate a clean sweep. It is named.
for name in guardrail_unmeasurable:
    print(f"NOT EVALUABLE {name}: this cohort cannot support it, so it neither passes nor fails.")
if not GROUPED_PROBE_EVALUABLE:
    print("  The registered grouped probe needs at least two source videos per condition. Sources per "
          f"condition here: {SOURCES_PER_CONDITION}.")
    print("  The leaky stratified probe stands in as a damage detector only; its value is not a result.")

condition_3 = not guardrail_failures
print(f"\\nCONDITION 3 no guardrail regresses beyond the control seed spread -> "
      f"{'PASS' if condition_3 else 'FAIL'}")
for name in guardrail_failures:
    row = guardrail_frame[guardrail_frame["guardrail"] == name].iloc[0]
    print(f"  FAILED {name}: regressed by {row['regression']:.4f} against a control seed spread of "
          f"{row['D0_seed_spread']:.4f}")
'''))

    cells.append(code('''
CREDIT = bool(condition_1 and condition_2 and condition_3)
VERDICT = "CREDIT" if CREDIT else "NO CREDIT"

print(f"condition 1 (exceeds seed spread) : {'PASS' if condition_1 else 'FAIL'}")
print(f"condition 2 (paired bootstrap)    : {'PASS' if condition_2 else 'FAIL'}")
print(f"condition 3 (no guardrail loss)   : {'PASS' if condition_3 else 'FAIL'}")
print(f"\\nPRE-REGISTERED VERDICT: {VERDICT}")
print("\\nThe degenerate-solution check, which is not part of the rule but decides how to read a pass:")
print(f"  head output scale  D0 {float(D0['head_scale'].mean()):.4f} -> "
      f"E1 {float(E1['head_scale'].mean()):.4f}")
if float(E1["head_scale"].mean()) < float(D0["head_scale"].mean()):
    print("  the head SHRANK. Treat any improvement as suspect: this is the failure mode "
          "new_nb_09_01 measured, where the readout collapses instead of the encoder changing.")
else:
    print("  the head GREW, so the endpoint did not improve by the readout collapsing.")
'''))

    cells.append(md("""
## 4. The secondary and tertiary endpoints

The secondaries answer a question the primary cannot: where the change happened. `rho` on the view encoder
measures the network the term touches directly, and `rho` on the target encoder measures the EMA teacher
that every downstream readout in this project consumes. A term that moves the view encoder but not the
teacher would have produced a real optimisation effect and no usable artifact.

The measured mirror slope is reported for continuity with Arm 1, which recorded -0.223 with its own head
against item 05's -0.741 for the baseline. Moving toward -1 is the direction a genuinely antisymmetric
representation would take. It is measured through the encoder and is never asserted.

The tertiary R-squared is the lane Arm 1 treated as its treatment, recomputed per checkpoint. It is printed
beside the gate verdict that makes it uninterpretable, and it is printed anyway so the record is complete.
""".rstrip()))

    cells.append(code('''
secondary = pd.DataFrame([{
    "rung": rung,
    "rho_view": float(frame["rho_view"].mean()),
    "rho_target": float(frame["rho_target"].mean()),
    "ema_gap": float(frame["rho_target"].mean() - frame["rho_view"].mean()),
    "mirror_slope": float(frame["mirror_slope"].mean()),
    "antisymmetric_lane_r2": float(frame["antisymmetric_lane_r2"].mean()),
} for rung, frame in (("D0", D0), ("E1", E1))])
display(secondary.round(4))

y_gate = CONTRACT.get("arm1_reference", {}).get("y_between_source_fraction")
print("Tertiary R-squared is reported beside the gate that makes it uninterpretable on this cohort:")
if y_gate is not None:
    print(f"  between-source variance fraction {y_gate:.4f} against a required 0.30 -> FAILED")
else:
    print("  Arm 1 recorded a FAILED between-source variance gate (0.0747 against 0.30)")
print("  So neither lane's R-squared supports a conclusion in either direction, and no gate here "
      "depends on it.")
'''))

    cells.append(code('''
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from IPython.display import Image, display

figure, axes = plt.subplots(1, 3, figsize=(15.5, 4.3))

axes[0].bar(["D0 control", "E1 treatment"], [d0_mean, e1_mean],
            color=["#555555", "#c1272d"], width=0.55)
for x, frame in enumerate((D0, E1)):
    axes[0].scatter([x] * len(frame), frame["rho_target"], color="white", edgecolor="black",
                    zorder=3, s=28, label="per seed" if x == 0 else None)
axes[0].set_ylabel("rho on the target encoder")
axes[0].set_title(f"Primary endpoint\\nimprovement {improvement:.3f}, control spread {d0_spread:.3f}")
axes[0].legend(fontsize=7)

axes[1].hist(draws, bins=40, color="#4878a8")
axes[1].axvline(0.0, color="black", lw=1.2)
axes[1].axvline(low, color="#c1272d", ls="--", lw=1.1)
axes[1].axvline(high, color="#c1272d", ls="--", lw=1.1)
axes[1].set_xlabel("mean per-source improvement in rho")
axes[1].set_title(f"Paired bootstrap over {len(sources)} sources\\n[{low:.3f}, {high:.3f}]")

measurable_frame = guardrail_frame[guardrail_frame["measurable"]]
order = list(measurable_frame["guardrail"])
changes = list(measurable_frame["regression"])
spreads = list(measurable_frame["D0_seed_spread"])
positions = np.arange(len(order))
axes[2].barh(positions, changes, color=["#c1272d" if c > s else "#4a7c59"
                                        for c, s in zip(changes, spreads)], height=0.5)
axes[2].scatter(spreads, positions, marker="|", s=260, color="black", label="control seed spread")
axes[2].set_yticks(positions, [n.replace("_", " ") for n in order], fontsize=7.5)
axes[2].axvline(0.0, color="black", lw=1.0)
axes[2].set_xlabel("regression (positive is worse)")
axes[2].set_title("Guardrails against their own\\ncontrol seed spread")
axes[2].legend(fontsize=7)

for axis in axes:
    axis.grid(alpha=0.22, axis="both")
figure.suptitle(f"Pre-registered verdict: {VERDICT}", y=1.04, fontsize=12)
figure.tight_layout()
verdict_png = ARM2_DIR / "idea9_arm2_verdict.png"
figure.savefig(verdict_png, dpi=170, bbox_inches="tight")
plt.close(figure)
display(Image(filename=str(verdict_png)))
print(f"wrote {verdict_png}")
'''))

    cells.append(code('''
RESULT = {
    "series": "new_nb_09", "arm": "arm2", "notebook": "new_nb_09_03",
    "mode": MODE,
    "training_scale": "real",
    "equiv_variant": sorted({b["equiv_variant"] for b in RUNGS}),
    "equiv_weight": float(E1["equiv_weight"].mean()) if len(E1) else None,
    "seeds_registered": PRE_REGISTERED["ladder"]["seeds"],
    "seeds_observed": OBSERVED_SEEDS,
    "seeds_paired": PAIRED_SEEDS,
    "protocol_deviations": DEVIATIONS,
    "evaluation_cohort": {"sequences": int(len(EVAL_XYZ)), "source_videos": int(len(set(EVAL_VIDEO))),
                          "subset": "canonical only, excluding the augmentation-normal training pool"},
    "primary": {
        "endpoint": "rho_target_encoder",
        "D0_mean": d0_mean, "D0_seed_spread": d0_spread, "E1_mean": e1_mean,
        "improvement": improvement,
        "paired_bootstrap": {"sources": int(len(sources)), "draws": BOOTSTRAP_DRAWS,
                             "mean": float(per_source_difference.mean()),
                             "median": float(np.median(per_source_difference)),
                             "ci_low": low, "ci_high": high,
                             "sources_improved": int((per_source_difference > 0).sum()),
                             "note": "per-source ratios averaged across videos; not comparable to the "
                                     "cohort-level ratio of summed terms"},
    },
    "secondary": secondary.to_dict(orient="records"),
    "guardrails": guardrail_rows,
    "guardrails_failed": guardrail_failures,
    "guardrails_not_evaluable": guardrail_unmeasurable,
    "sources_per_condition": SOURCES_PER_CONDITION,
    "degenerate_solution_check": {
        "D0_head_scale": float(D0["head_scale"].mean()),
        "E1_head_scale": float(E1["head_scale"].mean()),
        "head_grew": bool(float(E1["head_scale"].mean()) >= float(D0["head_scale"].mean())),
    },
    "credit_rule": {"condition_1_exceeds_seed_spread": condition_1,
                    "condition_2_paired_bootstrap": condition_2,
                    "condition_3_no_guardrail_regression": condition_3,
                    "all_three_required": True},
    "PRIMARY_VERDICT": VERDICT,
    "per_rung": SCORES.to_dict(orient="records"),
    "framing": PRE_REGISTERED["framing"] + [
        "rho is a symmetry property of the representation. It is not accuracy, not separation, and not "
        "clinical value. A rho improvement is not evidence of downstream benefit.",
        "The grouped condition probe is a guardrail only. Every row it scores was seen by the encoder "
        "during training, so it is transductive and is not a performance claim.",
    ],
}
result_path = ARM2_DIR / "idea9_arm2_evaluation_result.json"
result_path.write_text(json.dumps(RESULT, indent=2), encoding="utf-8")
print(f"wrote {result_path}")
print(json.dumps({"PRIMARY_VERDICT": RESULT["PRIMARY_VERDICT"],
                  "primary": {k: v for k, v in RESULT["primary"].items() if k != "paired_bootstrap"},
                  "credit_rule": RESULT["credit_rule"]}, indent=2))
'''))

    cells.append(md("""
## 5. Discussion

### The verdict is no credit, and the endpoint moved anyway

Both statements are true at once, and holding them together is the whole point of writing the rule down in
advance.

The equivariance term reaches the encoder on real data. That was the open question after `new_nb_09_01`,
because a term validated on a one-layer toy encoder can fail at width for reasons unrelated to its
formulation, and because the term this series uses is not the term `nb_09b` proposed. The primary endpoint
falls by roughly a factor of eight, about seven times the control rung's own seed spread; every source video
improves; and the measured anatomical-mirror slope moves from around -0.65 to about -0.94, which is close to
the exact sign inversion a mirror-equivariant representation would show. For comparison, item 05 measured
-0.741 on the baseline and Arm 1 measured -0.223 with its own head. Two of the three credit conditions pass
decisively.

The third fails. Feature standard deviation drops by more than three times the control's seed spread, and
mean pairwise cosine moves the same way, passing only by a hair. Both point at the same thing: the treated
representation is mildly more collapsed. The rule required all three conditions, so the verdict is no
credit.

### Why not just note the caveat and call it a win

Because that is exactly the move the rule was written to prevent, and from inside the result it feels
reasonable. The endpoint improvement is large and clean, the failure is a modest change in a diagnostic, the
condition geometry is untouched, and the substitute condition probe actually improves slightly. Every
ingredient for "works, with a minor caveat" is present.

What makes that framing wrong is that the caveat and the effect are not independent. A term that pushes the
representation toward responding identically to a body and its reflection is, mechanically, a term that
removes variance. Shrinking feature spread is not an unrelated side effect; it is a plausible partial
explanation of the endpoint improvement itself. Distinguishing "the encoder learned mirror structure" from
"the encoder has less to be inconsistent about" needs an experiment this ladder does not contain, and the
guardrail failure is precisely the signal that the distinction matters here.

The head-scale check is what keeps this from being worse. `new_nb_09_01` showed the original loss form has a
degenerate solution in which the readout shrinks and the encoder is untouched. Here the head's output scale
grows, from about 0.75 to about 1.06, so the endpoint did not improve by the readout collapsing. That rules
out the specific failure mode that was found on fixtures. It does not rule out the milder variance-removal
concern, and conflating the two would overstate what the check delivers.

### What is not established at all

`rho` is a symmetry property of the representation. It is not accuracy, not condition separation, and not
clinical value. A representation can be perfectly mirror-equivariant and useless for gait assessment, and
this ladder cannot distinguish those cases. The guardrails are shaped to detect damage, not to demonstrate
benefit, so even a clean sweep would not have shown that the term helps anything downstream.

One registered guardrail could not be evaluated. A source-grouped five-class probe needs at least two source
videos per condition, and all twelve canonical normal sequences come from a single video. The stand-in is a
stratified probe that leaks video identity, which is adequate for noticing destruction and worthless as
evidence about conditions.

The transductive boundary is unchanged. The encoder saw every evaluation sequence during training, on both
rungs equally. That is acceptable for a control-versus-treatment contrast, which is what this is, and it
remains disqualifying for any claim about an unseen patient, camera, or clinic.

### What the three notebooks before this one bought

Three specific errors were avoided, and it is worth being concrete because each was cheap to catch here and
would have been expensive to catch later.

The first is the no-op. A head-only formulation of the equivariance term is identically zero for an
antisymmetric head, so it trains nothing while looking like a constraint. Measuring the gradient rather
than reasoning about it took one cell.

The second is the degenerate solution. The term as originally written is minimised by shrinking the
readout, and its loss curve falls by orders of magnitude while the encoder is unchanged. Had the real ladder
run that form, it would have consumed the full compute budget and produced a confident, wrong, positive
result supported by a beautiful loss curve.

The third is the endpoint choice. Arm 1's own gate recorded that the labelled target's between-source
variance fraction was 0.0747 against a required 0.30, which makes held-out-source R-squared uninterpretable
on this cohort. Scoring this ladder on R-squared would have produced a number that could not support a
conclusion in either direction, and the temptation to read one anyway would have been considerable.

A fourth is smaller but worth recording. The fixtures in `new_nb_09_01` sit near `rho = 4` because their toy
encoder is mirror-blind by construction, and it was tempting to read that as a forecast for real encoders.
The real control rungs land far closer to mirror-honest. A fixture can establish that a measurement means
what you claim while telling you nothing about where real data falls on it, and conflating those two roles
is easy when the fixture number is memorable.

### What to do next

Two steps follow, in this order.

The first is to separate the two explanations of the endpoint improvement. An equivariance weight sweep would
show whether spread loss and endpoint gain move together, which is what the variance-removal account
predicts, or whether there is a weight at which the endpoint improves and spread holds. That is a cheap
experiment on the existing recipe and it decides how the current result should be read.

The second is to ask whether the property is worth having at all: score the treatment and control
checkpoints on a task with an interpretable endpoint. That needs either a target whose between-source
variance clears the gate, or a cohort with enough source videos that the existing target does.

The second route leads back to the same conclusion Arm 1 reached from the other direction. The binding
constraint on this line of work is the number of independent source videos, not the readout, not the loss,
and not the head. That is a data-collection finding, and it is more actionable than another architecture
would have been.
""".rstrip()))

    write_notebook(ROOT / "new_nb_09_03_evaluation_results_discussion.ipynb", cells)


BUILDERS = {"00": build_nb00, "01": build_nb01, "02": build_nb02, "03": build_nb03}

if __name__ == "__main__":
    import sys

    # The builder refuses to overwrite an existing notebook. Move or delete the specific target first:
    #   python3 _build_new_nb_09_series.py 03
    # With no argument it attempts all four, but stops on the first maintained target that exists.
    requested = sys.argv[1:] or list(BUILDERS)
    unknown = [name for name in requested if name not in BUILDERS]
    if unknown:
        raise SystemExit(f"unknown notebook(s) {unknown}; choose from {sorted(BUILDERS)}")
    for name in requested:
        BUILDERS[name]()
