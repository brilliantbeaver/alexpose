"""Builder for nb_09b_equivariant_retrain.ipynb (Idea 9, Arm 2).

Run:  python3 _build_nb_09b.py
Emits the notebook at the gavd root (parents[3]) beside nb_09a/nb_09c. Committed alongside the notebook.

Arm 2 is the ONLY arm that changes the encoder. It adds a LABEL-FREE equivariance term to the training
loss and retrains the curriculum, then re-scores the retrained checkpoint through the SAME Arm-1
instrument. The equivariance term is

    L_equiv = mean( ( s(encoder(Mx)) + s(encoder(x)) )^2 )

where s is the antisymmetric head from nb_09a and M is the ANATOMICAL mirror on the RAW skeleton (negate
the sideways coordinate, swap left/right landmarks). BOTH the original and the mirrored skeleton are run
through the VIEW encoder before the head sees them. This is the load-bearing correctness point: penalizing
s(swap of the head's own tokens) + s(tokens) is identically zero for every input and parameter (it is
algebra about the head, which is antisymmetric by construction, so it trains nothing and has zero
gradient). Going THROUGH the encoder makes the term a genuine constraint on the ENCODER, which is not
equivariant by construction. It is zero only when the encoder represents the mirrored body as the sign-
flip of the original along the head's axis. The term uses NO source/condition label; a label-supervised
axis on about seven lateralized source videos would be a vacuous transductive win, so none is added.

The experiment is an ABLATION LADDER across seeds:
    D  = the existing baseline checkpoint (scored by nb_09a; referenced here, not retrained).
    D0 = the baseline recipe reproduced, flip OFF, EQUIV_WEIGHT = 0 (controls for lineage/trajectory).
    E1 = D0 + L_equiv (EQUIV_WEIGHT > 0).
Run D0 and E1 across several seeds; credit the L_equiv effect only if E1 - D0 exceeds D0's seed-to-seed
spread. This is a TRAJECTORY control, not a source-variation claim.

DEFAULT MODE IS SMOKE. The smoke path is fully runnable and self-contained: it reuses nb_09a's synthetic
cohort (with a planted, sign-flipping lateral lean) for every condition, so the ladder, the loss plumbing,
the collapse monitors, the distinct-fingerprint checkpoints, and the re-scoring all exercise end to end in
seconds, WITHOUT the real MS-PD mapping file, pose cache, or the 600-epoch cost. The real multi-seed
600-epoch run is deferred: section 8 documents exactly how to launch it against notebook 04's real cohort.
Every number this notebook produces is an illustrative plumbing check, NOT a baseline-lineage result.
"""
import json
from pathlib import Path

NB_PATH = Path(__file__).resolve().parents[3] / "nb_09b_equivariant_retrain.ipynb"

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


CELLS = []

# ------------------------------------------------------------------ Title
CELLS.append(md(r"""
# Notebook 09b: Equivariance-coupled retrain (Idea 9, Arm 2)

This notebook reifies **Arm 2** of research idea 9. Arm 1 (`nb_09a`) never touches the encoder; it asks
whether an antisymmetry-constrained readout beats a binding bar on the FROZEN baseline features, and the
expected honest answer is an informative null (Idea 05 already showed the frozen encoder does not expose
the signed axis). Arm 2 is the only arm that can change that: it adds a **label-free** equivariance term
to the training loss and retrains, then re-scores the retrained encoder through the exact Arm-1
instrument.

**The one new loss term.** With the antisymmetric head `s` from nb_09a and the anatomical mirror `M` on the
RAW skeleton,

`L_equiv = mean( ( s(encoder(Mx)) + s(encoder(x)) )^2 )`.

Both the original and the anatomically mirrored skeleton are run through the VIEW encoder before the head
reads them. Why through the encoder and not at the head input? Because the head is antisymmetric under a
swap of its OWN inputs by construction, so a head-only `s(swap) + s` is identically zero: it trains nothing
and has exactly zero gradient. Running the mirror THROUGH the encoder makes `L_equiv` a real constraint on
the ENCODER (which is not equivariant by construction): it is zero only when the encoder represents the
mirrored body as the sign-flip of the original along the head's axis. It uses NO label at all. We
deliberately do NOT add a label-supervised axis term: with about seven lateralized source videos, a
supervised axis would be a vacuous transductive win.

**This form of the term is defective, and section 8 says what replaced it.** Because `s` is trainable and
the residual is absolute, the head can shrink its own output to drive `L_equiv` down while the encoder stays
exactly as mirror-blind as it started. Read this notebook as the scaffold that demonstrates the ladder
machinery and, unintentionally, the failure mode; the real run uses a scale-invariant residual and an
endpoint no trainable weight can influence.

**The experiment is an ablation ladder across seeds.** `D` is the existing baseline (scored in nb_09a).
`D0` reproduces the recipe with flip off and `EQUIV_WEIGHT = 0`. `E1` is `D0 + L_equiv`. We run `D0` and
`E1` across several seeds and credit the equivariance effect only if `E1 - D0` clears `D0`'s seed-to-seed
spread. That is a trajectory control, not a source-variation claim.

**Everything here is synthetic and smoke scale, whatever `GAVD_MODE` says.** The ladder always reuses
nb_09a's synthetic cohort and a tiny model, so it runs in seconds. `GAVD_MODE` selects only the directory
the bundle is written to, which is why the bundle is labelled by `data_source` and `training_scale` instead.
No number in this notebook is evidence about the real encoder. Section 8 records where the real multi-seed
run actually happened and what it found. All results are transductive; the source video is the independent
unit; folder labels are dataset annotations, not diagnoses.
"""))

# ------------------------------------------------------------------ 0. Environment + mode
CELLS.append(md(r"""
## 0. Environment, mode, and the run knobs

We resolve the project root as notebooks 04-06 and nb_05a do, decide the mode, and read the Arm-2 knobs
from the environment (all with safe defaults): `IDEA9_EQUIV_WEIGHT` (weight on `L_equiv`, default 0.02,
below the main terms), `IDEA9_SEEDS` (comma list of seeds for the D0/E1 ladder, default `0,1,2`), and the
smoke training-size knobs. Real mode here is a documented, deferred launch (section 8): it prints the
recipe and stops, so nobody accidentally kicks off a 600-epoch run inside this scaffold.
"""))
CELLS.append(code(r"""
from pathlib import Path
import os, math, json, hashlib, copy, warnings

import numpy as np
import pandas as pd

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


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

ARTIFACT_ROOT = Path(os.getenv("GAVD_ARTIFACT_DIR", TUTORIAL_DIR / "work" / "artifacts")).expanduser()
REQUESTED_MODE = os.getenv("GAVD_MODE", "smoke").strip().lower()
if REQUESTED_MODE not in {"smoke", "real"}:
    raise ValueError("GAVD_MODE must be smoke or real")
MODE = REQUESTED_MODE

CONDITIONS = ["normal", "parkinsons", "stroke", "myopathic", "cerebralpalsy"]
CURRICULUM = [
    {"stage": 0, "name": "normal_only", "add": "normal", "conditions": ["normal"]},
    {"stage": 1, "name": "add_parkinsons", "add": "parkinsons", "conditions": CONDITIONS[:2]},
    {"stage": 2, "name": "add_stroke", "add": "stroke", "conditions": CONDITIONS[:3]},
    {"stage": 3, "name": "add_myopathic", "add": "myopathic", "conditions": CONDITIONS[:4]},
    {"stage": 4, "name": "add_cerebralpalsy", "add": "cerebralpalsy", "conditions": CONDITIONS[:5]},
]
MASK_KEYPOINTS = [11, 12, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
LEFT_RIGHT_PAIRS = [(11, 12), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32)]

# ---- Arm-2 knobs (env-driven, safe defaults) ----
EQUIV_WEIGHT = float(os.getenv("IDEA9_EQUIV_WEIGHT", "0.02"))
SEEDS = [int(s) for s in os.getenv("IDEA9_SEEDS", "0,1,2").split(",") if s.strip() != ""]
HEAD_OUT_DIM = int(os.getenv("IDEA9_HEAD_OUT_DIM", "4"))
VICREG_WEIGHT = float(os.getenv("SJEPA_VICREG_WEIGHT", "0.05"))
GROUP_WEIGHT = float(os.getenv("SJEPA_GROUP_WEIGHT", "0.25"))
GROUP_MARGIN = float(os.getenv("SJEPA_GROUP_MARGIN", "1.0"))
MASK_FRACTION = float(os.getenv("SJEPA_MASK_FRACTION", "0.60"))

OUT_DIR = ARTIFACT_ROOT / MODE
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"PROJECT_ROOT : {PROJECT_ROOT}")
print(f"mode         : {MODE}")
print(f"EQUIV_WEIGHT : {EQUIV_WEIGHT}")
print(f"SEEDS        : {SEEDS}")
print(f"total loss   : jepa + {VICREG_WEIGHT}*vicreg + {GROUP_WEIGHT}*group + {EQUIV_WEIGHT}*L_equiv")
if MODE == "smoke":
    print("SMOKE MODE: synthetic cohort, tiny model, few epochs. Numbers are plumbing checks only.")
"""))

# ------------------------------------------------------------------ 1. Torch + model classes (verbatim nb04)
CELLS.append(md(r"""
## 1. Model classes and loss functions (reused verbatim from notebook 04)

`SkeletonPatchEncoder`, `SkeletonPredictor`, `SJEPAGait` (with `update_target`, `update_center`),
`sjepa_cross_entropy`, `cosine_ema`, and `geometric_view` are pasted exactly as notebook 04 defines them,
so the retrained checkpoint is drop-in loadable by the nb_09a instrument and the loss matches the trained
lineage. `geometric_view` keeps `flip_probability = 0.0` (left/right identity matters for stroke).
"""))
CELLS.append(code(r"""
try:
    import torch
    from torch import nn
    from torch.nn import functional as F
    HAVE_TORCH = True
except Exception as exc:  # pragma: no cover
    HAVE_TORCH = False
    raise RuntimeError(f"Arm 2 requires PyTorch: {exc}")


class SkeletonPatchEncoder(nn.Module):
    def __init__(self, frames=64, joints=33, coordinate_dim=3, segment_length=4,
                 embed_dim=96, depth=4, heads=4, dropout=0.0):
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
        if (frames, joints, channels) != (self.frames, self.joints, self.coordinate_dim):
            raise ValueError(f"bad shape {x.shape}")
        patches = x.reshape(batch, self.segments, self.segment_length, joints, channels)
        return patches.permute(0, 1, 3, 2, 4).contiguous().flatten(3)

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
    def __init__(self, segments, joints, encoder_dim=96, predictor_dim=96, depth=2, heads=4, dropout=0.0):
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
        for tp, vp in zip(self.target_encoder.parameters(), self.view_encoder.parameters()):
            tp.mul_(momentum).add_(vp, alpha=1.0 - momentum)

    @torch.no_grad()
    def update_center(self, targets, beta=0.9):
        self.target_center.mul_(beta).add_(targets.mean(dim=(0, 1)), alpha=1.0 - beta)


def sjepa_cross_entropy(predicted, targets, center, predictor_temperature=0.10, target_temperature=0.06):
    target_prob = torch.softmax((targets - center[None, None, :]) / target_temperature, dim=-1).detach()
    prediction_log_prob = torch.log_softmax(predicted / predictor_temperature, dim=-1)
    return -(target_prob * prediction_log_prob).sum(dim=-1).mean()


def cosine_ema(step, total_steps, start=0.996, end=1.0):
    progress = min(max(step / max(total_steps - 1, 1), 0.0), 1.0)
    return end - (end - start) * (math.cos(math.pi * progress) + 1.0) / 2.0


FULL_LR_PAIRS = [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16),
                 (17, 18), (19, 20), (21, 22), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32)]


def geometric_view(x, max_degrees=8.0, translate=0.03, flip_probability=0.0):
    '''Verbatim from nb04: y-axis rotation (mixes x,z) + small translation; flip OFF by default.'''
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
    if flip_probability > 0:
        flip = torch.rand(batch, device=x.device) < flip_probability
        for bi in torch.where(flip)[0].tolist():
            view[bi, ..., 0] *= -1.0
            original = view[bi].clone()
            for left, right in FULL_LR_PAIRS:
                view[bi, :, left] = original[:, right]
                view[bi, :, right] = original[:, left]
    view = view.masked_fill(~present[..., None], 0.0)
    return view


print("model + loss + augmentation defined (verbatim from notebook 04).")
"""))

# ------------------------------------------------------------------ 2. The antisymmetric head + L_equiv
CELLS.append(md(r"""
## 2. The antisymmetric head and the label-free equivariance loss

Same head as nb_09a: `s = sum_k ( f(L_k) - f(R_k) )` with a shared `f = Linear(D,32)->GELU->Linear(32,m)`,
difference only (no `l + r`). Here `f`'s parameters ARE trainable and join the optimizer, so `L_equiv` can
shape the encoder AND the head together.

**The loss must go THROUGH the encoder, not around it.** This is the correctness point that a first draft
got wrong. The head is antisymmetric under a swap of its OWN inputs by construction, so penalizing
`s(swap of tokens) + s(tokens)` is identically zero for every input and every parameter: it is algebra
about the head, it trains nothing, and its gradient is exactly zero. The encoder, by contrast, is NOT
built to be equivariant. So the equivariance term reflects the RAW skeleton anatomically (negate the
sideways coordinate, swap left/right landmarks), runs BOTH the original and the mirrored skeleton through
the VIEW ENCODER, and penalizes

`L_equiv = mean( ( s(encoder(Mx)) + s(encoder(x)) )^2 )`,

where `M` is the anatomical mirror on raw coordinates. This is zero only when the encoder has learned to
represent the mirrored body as the sign-flip of the original along the head's axis, which is a real,
non-trivial constraint on the encoder's weights. It is still label-free.

Two guardrails proven inline: (a) the head negates under a swap of its OWN input by construction (slope -1
to tolerance) - this remains a valid wiring self-check of the head, but it is explicitly NOT the training
loss; (b) on a deliberately NON-equivariant encoder, `L_equiv` is strictly positive and produces nonzero
gradients into BOTH the encoder and the head (we assert nonzero gradient norms, which the old
around-the-encoder loss could never satisfy).
"""))
CELLS.append(code(r"""
FULL_LR_PAIRS_MIRROR = FULL_LR_PAIRS  # 16-pair anatomical mirror, same as nb04's geometric_view flip


def anatomical_mirror_coords(coords):
    '''Reflect the RAW skeleton: negate the sideways (x) coordinate and swap each left/right landmark.
    coords: [B, FRAMES, 33, 3] tensor. Returns the mirrored tensor (differentiable passthrough).'''
    m = coords.clone()
    m[..., 0] = -m[..., 0]
    idx = list(range(33))
    for li, ri in FULL_LR_PAIRS_MIRROR:
        idx[li], idx[ri] = ri, li
    return m[:, :, idx, :]


class AntisymmetricHead(nn.Module):
    '''s = sum_k ( f(L_k) - f(R_k) ); f shared across joints/sides; difference only. Trainable in Arm 2.'''
    def __init__(self, embed_dim, out_dim=4, hidden=32, pairs=LEFT_RIGHT_PAIRS):
        super().__init__()
        self.pairs = list(pairs)
        self.f = nn.Sequential(nn.Linear(embed_dim, hidden), nn.GELU(), nn.Linear(hidden, out_dim))

    def per_joint_feature(self, tokens):
        # tokens: [B, SEGMENTS, 33, D] -> per-joint time-mean [B, 33, D]
        return tokens.mean(dim=1)

    def s_from_perjoint(self, pj):
        out = 0.0
        for li, ri in self.pairs:
            out = out + (self.f(pj[:, li, :]) - self.f(pj[:, ri, :]))
        return out

    def forward(self, tokens):
        return self.s_from_perjoint(self.per_joint_feature(tokens))

    def swapped(self, tokens):
        '''Wiring self-check ONLY (swap the head's own inputs). By construction returns -forward(tokens);
        it is NOT the training loss, because it trains nothing.'''
        pj = self.per_joint_feature(tokens)
        pj_sw = pj.clone()
        for li, ri in self.pairs:
            pj_sw[:, li, :] = pj[:, ri, :]
            pj_sw[:, ri, :] = pj[:, li, :]
        return self.s_from_perjoint(pj_sw)


def encoder_tokens_for_head(view_encoder, coords, segments, joints, embed_dim):
    '''Run raw coords through the VIEW encoder and reshape to [B, SEGMENTS, 33, D] for the head.'''
    return view_encoder(coords).reshape(len(coords), segments, joints, embed_dim)


def equivariance_loss(head, view_encoder, coords, segments, embed_dim, joints=33):
    '''Label-free equivariance term THROUGH the encoder:
        L_equiv = mean( ( s(encoder(Mx)) + s(encoder(x)) )^2 ),  M = anatomical mirror on raw coords.
    Zero only when the encoder represents the mirrored body as the sign-flip of the original along the
    head's axis. This is a real constraint on the ENCODER (the encoder is not equivariant by construction).'''
    tok = encoder_tokens_for_head(view_encoder, coords, segments, joints, embed_dim)
    tok_m = encoder_tokens_for_head(view_encoder, anatomical_mirror_coords(coords), segments, joints, embed_dim)
    s = head(tok)
    s_m = head(tok_m)
    return ((s_m + s) ** 2).mean()


# ---- guardrail (a): head negates under a swap of its OWN input, by construction (wiring self-check) ----
torch.manual_seed(RANDOM_SEED)
_head = AntisymmetricHead(32, out_dim=HEAD_OUT_DIM)
_tok = torch.randn(6, 8, 33, 32)
with torch.no_grad():
    _s, _ssw = _head(_tok), _head.swapped(_tok)
_slope = float(np.polyfit(_s.reshape(-1).numpy(), _ssw.reshape(-1).numpy(), 1)[0])
assert abs(_slope + 1.0) < 1e-4, f"head must negate under its own-input swap; slope={_slope}"
print(f"guardrail (a): head own-input swap slope = {_slope:+.6f}  (exact -1 by construction; NOT the loss)")

# ---- guardrail (b): the REAL loss is nonzero with nonzero grads into a deliberately NON-equivariant encoder ----
# A random Linear over the raw coords is not equivariant, so L_equiv through it must be > 0 with real grads.
torch.manual_seed(RANDOM_SEED + 3)
_fake_frames, _fake_seg, _fake_dim = 8, 2, 32
class _NonEquivEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(_fake_frames * 3, _fake_seg * _fake_dim)
    def forward(self, x):  # x: [B, F, 33, 3] -> [B, SEG*33, D]
        b = len(x)
        per_joint = x.permute(0, 2, 1, 3).reshape(b, 33, _fake_frames * 3)
        out = self.proj(per_joint).reshape(b, 33, _fake_seg, _fake_dim)
        return out.permute(0, 2, 1, 3).reshape(b, _fake_seg * 33, _fake_dim)
_enc = _NonEquivEncoder()
_head_b = AntisymmetricHead(_fake_dim, out_dim=HEAD_OUT_DIM)
_coords = torch.randn(4, _fake_frames, 33, 3)
_le = equivariance_loss(_head_b, _enc, _coords, _fake_seg, _fake_dim)
_le.backward()
_gh = sum(float(p.grad.norm()) for p in _head_b.parameters() if p.grad is not None)
_ge = sum(float(p.grad.norm()) for p in _enc.parameters() if p.grad is not None)
assert float(_le) > 1e-8, f"L_equiv must be > 0 on a non-equivariant encoder; got {float(_le):.3e} (no-op!)"
assert _gh > 1e-8 and _ge > 1e-8, f"L_equiv must push head AND encoder; head|grad|={_gh:.3e} enc|grad|={_ge:.3e}"
print(f"guardrail (b): L_equiv={float(_le):.4e} (> 0)  head|grad|={_gh:.3e}  encoder|grad|={_ge:.3e}  (real, not a no-op)")
"""))

# ------------------------------------------------------------------ 3. Smoke cohort (from nb_09a/05a)
CELLS.append(md(r"""
## 3. Smoke cohort with a planted, sign-flipping lateral lean (reused from nb_09a)

For the runnable smoke ladder we reuse nb_09a's synthetic cohort: a hand-authored gait fixture per
condition, with a planted lateral lean whose SIGN alternates between the two synthetic sources of each
condition, and whose magnitude is largest for the "lateralized" folders and zero for myopathic/normal.
This gives the equivariance loss and the readout a genuine signed structure to organize around, so the
plumbing is exercised. It is NOT physiology. Real mode does not use this; it defers to section 8.
"""))
CELLS.append(code(r"""
def synthetic_gait_sequence(condition="normal", frames=64, seed=0):
    rng = np.random.default_rng(seed)
    phase = np.linspace(0.0, 4.0 * np.pi, frames, endpoint=False)
    seq = np.zeros((frames, 33, 4), dtype=np.float32)
    seq[..., 3] = 1.0
    base = {11: (0.42, 0.28), 12: (0.58, 0.28), 23: (0.45, 0.52), 24: (0.55, 0.52),
            25: (0.44, 0.70), 26: (0.56, 0.70), 27: (0.43, 0.89), 28: (0.57, 0.89),
            29: (0.42, 0.92), 30: (0.58, 0.92), 31: (0.39, 0.94), 32: (0.61, 0.94)}
    for joint, (x, y_) in base.items():
        seq[:, joint, 0] = x
        seq[:, joint, 1] = y_
    amplitude, lift = 0.045, 0.025
    if condition == "parkinsons":
        amplitude *= 0.45; lift *= 0.45
    if condition == "myopathic":
        seq[:, [11, 12], 0] += 0.03 * np.sin(phase)[:, None]
        seq[:, [23, 24], 0] += 0.018 * np.sin(phase)[:, None]
    for joint, knee, foot, offset in [(27, 25, 31, 0.0), (28, 26, 32, np.pi)]:
        wave = np.sin(phase + offset)
        if condition == "stroke" and joint == 27:
            wave = 0.35 * wave
        if condition == "cerebralpalsy":
            seq[:, knee, 1] -= 0.045; seq[:, joint, 1] -= 0.02
        seq[:, joint, 0] += amplitude * wave
        seq[:, knee, 0] += 0.4 * amplitude * wave
        seq[:, foot, 0] += amplitude * wave
        seq[:, joint, 1] -= lift * np.maximum(wave, 0.0)
        seq[:, foot, 1] -= 0.7 * lift * np.maximum(wave, 0.0)
    seq[..., :3] += rng.normal(0.0, 0.0025, seq[..., :3].shape)
    return seq


SMOKE_LEAN_MAGNITUDE = {"stroke": 1.0, "cerebralpalsy": 0.8, "parkinsons": 0.5, "myopathic": 0.0, "normal": 0.0}


def plant_signed_lean(seq, sign, magnitude, seed):
    if magnitude == 0.0 or sign == 0:
        return seq
    rng = np.random.default_rng(seed)
    out = seq.copy()
    phase = np.linspace(0.0, 4.0 * np.pi, len(seq), endpoint=False)
    for li, ri in LEFT_RIGHT_PAIRS:
        gain = 0.03 * magnitude * sign * rng.uniform(0.8, 1.2)
        out[:, li, 0] += gain * np.sin(phase)
        out[:, ri, 0] -= gain * np.sin(phase)
    return out


def center_and_scale_min(seq):
    xyz = np.asarray(seq, dtype=np.float32)[..., :3].copy()
    pelvis = 0.5 * (xyz[:, 23] + xyz[:, 24])
    xyz = xyz - pelvis[:, None, :]
    sh = np.linalg.norm(xyz[:, 11, :2] - xyz[:, 12, :2], axis=-1)
    hp = np.linalg.norm(xyz[:, 23, :2] - xyz[:, 24, :2], axis=-1)
    scale = np.nanmedian(np.maximum(sh, hp))
    if not np.isfinite(scale) or scale < 1e-6:
        scale = 1.0
    return np.nan_to_num(xyz / scale).astype(np.float32)


FRAMES = int(os.getenv("SJEPA_FRAMES", "32"))
SEGMENT_LENGTH = 4
SEGMENTS = FRAMES // SEGMENT_LENGTH


def build_smoke_condition_data(clips_per_source=3, frames=FRAMES):
    data, counter = {}, 0
    for condition in CONDITIONS:
        xyz_list, valid_list, vids = [], [], []
        for source in range(2):
            sign = 1 if source == 0 else -1
            mag = SMOKE_LEAN_MAGNITUDE[condition]
            for clip in range(clips_per_source):
                base = synthetic_gait_sequence(condition=condition, frames=frames, seed=RANDOM_SEED + counter)
                seq = plant_signed_lean(base, sign, mag, seed=RANDOM_SEED + 1000 + counter)
                xyz = center_and_scale_min(seq)
                xyz_list.append(xyz)
                valid_list.append(np.ones((frames, 33), dtype=bool))
                vids.append(f"smoke_source_{condition}_{source}")
                counter += 1
        data[condition] = {"xyz": np.stack(xyz_list).astype(np.float32),
                           "valid": np.stack(valid_list),
                           "video_ids": vids,
                           "records": [{"sequence_id": f"{condition}_{i}", "video_id": v}
                                       for i, v in enumerate(vids)]}
    return data


condition_data = build_smoke_condition_data()
for c in CONDITIONS:
    print(f"{c:14s}: xyz {condition_data[c]['xyz'].shape}  sources {len(set(condition_data[c]['video_ids']))}")
"""))

# ------------------------------------------------------------------ 4. Training helpers (adapted from nb04 cell 17)
CELLS.append(md(r"""
## 4. Training helpers (adapted from notebook 04's stage loop)

`authorized_pool`, `vicreg_terms`, `condition_group_terms`, `balanced_epoch_batches`, and the per-step
block are adapted from notebook 04. The ONE change is the extra term: after the JEPA/VICReg/group losses,
we compute `L_equiv` by running the raw batch AND its anatomical mirror through the VIEW encoder and
penalizing `s(enc(Mx)) + s(enc(x))`, then add `EQUIV_WEIGHT * L_equiv` to `total_loss` (when `equiv_on` is
True). The antisymmetric head's parameters join the AdamW `trainable` set; the target encoder stays frozen
(asserted every step). A simplified uniform mask over the 12 authorized joints is used so the smoke path is
self-contained.
"""))
CELLS.append(code(r"""
def authorized_pool(tokens, valid_patch):
    batch, segments, _, dim = tokens.shape
    selected = tokens[:, :, MASK_KEYPOINTS].reshape(batch, -1, dim)
    weights = valid_patch[:, :, MASK_KEYPOINTS].reshape(batch, -1).to(tokens.dtype)
    denom = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
    return (selected * weights.unsqueeze(-1)).sum(dim=1) / denom


def off_diagonal(m):
    n, _ = m.shape
    return m.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def vicreg_terms(a, b, gamma=1.0, eps=1e-4):
    invariance = F.mse_loss(a, b)
    a_std = torch.sqrt(a.var(dim=0, unbiased=False) + eps)
    b_std = torch.sqrt(b.var(dim=0, unbiased=False) + eps)
    variance = 0.5 * (F.relu(gamma - a_std).mean() + F.relu(gamma - b_std).mean())
    ac, bc = a - a.mean(dim=0), b - b.mean(dim=0)
    denom = max(len(a) - 1, 1)
    cov = (off_diagonal(ac.T @ ac / denom).square().sum()
           + off_diagonal(bc.T @ bc / denom).square().sum()) / (2.0 * a.shape[1])
    return 25.0 * invariance + 25.0 * variance + cov


def condition_group_terms(reps, cond_ids, margin=1.0):
    unique = torch.unique(cond_ids)
    zero = reps.sum() * 0.0
    if len(unique) < 2:
        return zero, zero
    norm = F.normalize(reps, dim=1)
    centroids = torch.stack([F.normalize(norm[cond_ids == v].mean(dim=0), dim=0) for v in unique])
    compact = torch.stack([(norm[cond_ids == v] - centroids[i]).square().sum(dim=1).mean()
                           for i, v in enumerate(unique)]).mean()
    pw = (centroids[:, None] - centroids[None, :]).square().sum(dim=-1).clamp_min(1e-12).sqrt()
    upper = torch.triu(torch.ones_like(pw, dtype=torch.bool), diagonal=1)
    sep = F.relu(margin - pw[upper]).square().mean()
    return compact, sep


def balanced_epoch_batches(data, active, per_condition, rng):
    lengths = {c: len(data[c]["xyz"]) for c in active}
    steps = max(1, int(np.ceil(max(lengths.values()) / per_condition)))
    required = steps * per_condition
    orders = {}
    for c in active:
        pieces = []
        while sum(len(p) for p in pieces) < required:
            pieces.append(rng.permutation(lengths[c]))
        orders[c] = np.concatenate(pieces)[:required]
    for step in range(steps):
        xs, vs, ls = [], [], []
        for label, c in enumerate(active):
            take = orders[c][step * per_condition:(step + 1) * per_condition]
            xs.append(data[c]["xyz"][take]); vs.append(data[c]["valid"][take]); ls.extend([label] * per_condition)
        perm = rng.permutation(len(ls))
        yield (np.concatenate(xs)[perm], np.concatenate(vs)[perm], np.asarray(ls, dtype=np.int64)[perm])


def uniform_authorized_mask(valid_patch, frac, seed):
    eligible_joint = np.zeros(33, dtype=bool); eligible_joint[MASK_KEYPOINTS] = True
    eligible = valid_patch & eligible_joint[None, None, :]
    counts = eligible.reshape(len(eligible), -1).sum(axis=1)
    n_mask = max(1, min(int(np.floor(counts.min() * frac)), int(counts.min()) - 1))
    rng = np.random.default_rng(seed)
    mask = np.zeros_like(eligible)
    for i in range(len(mask)):
        cand = np.flatnonzero(eligible[i].reshape(-1))
        mask[i].reshape(-1)[rng.choice(cand, size=n_mask, replace=False)] = True
    return mask


print("training helpers ready.")
"""))

# ------------------------------------------------------------------ 5. One run of the curriculum
CELLS.append(md(r"""
## 5. One full curriculum run (returns a distinct-fingerprint checkpoint)

`run_curriculum(seed, equiv_on)` builds a fresh tiny `SJEPAGait` + head, runs the five-stage loop, adds
`EQUIV_WEIGHT * L_equiv` when `equiv_on`, monitors collapse (feature std, mean pairwise cosine), asserts
the target encoder never receives gradients, and returns the model plus a checkpoint dict whose fingerprint
payload includes `equiv_weight` and `equiv_on`, so every run gets its OWN fingerprint, never the baseline's.
"""))
CELLS.append(code(r"""
SMOKE_EPOCHS_STAGE0 = int(os.getenv("IDEA9_SMOKE_EPOCHS0", "2"))
SMOKE_EPOCHS_FT = int(os.getenv("IDEA9_SMOKE_EPOCHS_FT", "1"))
SAMPLES_PER_CONDITION = int(os.getenv("IDEA9_SAMPLES_PER_CONDITION", "2"))
SMOKE_CONFIG = {"frames": FRAMES, "joints": 33, "coordinate_dim": 3, "segment_length": SEGMENT_LENGTH,
                "embed_dim": 32, "encoder_depth": 1, "predictor_depth": 1, "heads": 4}
EMBED_DIM = SMOKE_CONFIG["embed_dim"]

# Two labels that are easy to confuse, and were confused in an earlier version of this bundle.
# MODE is only the artifact directory this notebook writes into, taken from GAVD_MODE. It says
# nothing about either the data or the training scale here: section 3 always builds a SYNTHETIC
# cohort, and this notebook always trains at smoke scale. A bundle labelled by MODE alone can
# therefore read "real" while carrying two-epoch numbers from planted toy sequences.
DATA_SOURCE = "synthetic"
REAL_SCALE = {"embed_dim": 96, "encoder_depth": 4, "predictor_depth": 2}
TRAINING_SCALE = ("real" if all(SMOKE_CONFIG[k] == v for k, v in REAL_SCALE.items())
                  and SMOKE_EPOCHS_STAGE0 >= 300 else "smoke")
assert TRAINING_SCALE == "smoke", (
    "nb_09b is the smoke-scale plumbing notebook. A real-scale run belongs in "
    "new_nb_09_02_real_multiseed_equivariant_training.ipynb, which checkpoints per rung.")
print(f"artifact mode: {MODE} | data: {DATA_SOURCE} | training scale: {TRAINING_SCALE} "
      "(the bundle is labelled by the last two, not the first)")
device = torch.device("cpu")


def collapse_monitor(model, data, active):
    arrays = np.concatenate([data[c]["xyz"] for c in active])
    valid = np.concatenate([data[c]["valid"] for c in active])
    model.target_encoder.eval()
    with torch.no_grad():
        x = torch.tensor(arrays, dtype=torch.float32, device=device)
        vp = torch.tensor(valid, dtype=torch.bool, device=device).reshape(
            len(x), SEGMENTS, SEGMENT_LENGTH, 33).all(dim=2)
        tok = model.target_encoder(x).reshape(len(x), SEGMENTS, 33, EMBED_DIM)
        pooled = authorized_pool(tok, vp)
        std = float(pooled.std(dim=0, unbiased=False).mean())
        unit = F.normalize(pooled, dim=1)
        cos = unit @ unit.T
        eye = torch.eye(len(unit), dtype=torch.bool)
        mpc = float(cos[~eye].mean()) if len(unit) > 1 else float("nan")
    return {"feature_std": std, "mean_pair_cosine": mpc}


def run_curriculum(seed, equiv_on):
    torch.manual_seed(seed); np.random.seed(seed)
    model = SJEPAGait(**SMOKE_CONFIG).to(device)
    head = AntisymmetricHead(EMBED_DIM, out_dim=HEAD_OUT_DIM).to(device)

    class VProj(nn.Module):
        def __init__(self, d):
            super().__init__(); self.net = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        def forward(self, x): return self.net(x)
    vproj = VProj(EMBED_DIM).to(device)

    assert not any(p.requires_grad for p in model.target_encoder.parameters())
    history, global_step_all = [], 0
    l_equiv_last = float("nan")
    for stage in CURRICULUM:
        active = stage["conditions"]
        epochs = SMOKE_EPOCHS_STAGE0 if stage["stage"] == 0 else SMOKE_EPOCHS_FT
        trainable = [*[p for p in model.view_encoder.parameters() if p.requires_grad],
                     *[p for p in model.predictor.parameters() if p.requires_grad],
                     *list(vproj.parameters()), *list(head.parameters())]
        opt = torch.optim.AdamW(trainable, lr=3e-4, betas=(0.9, 0.95), weight_decay=0.05)
        steps_per_epoch = max(1, int(np.ceil(max(len(condition_data[c]["xyz"]) for c in active) / SAMPLES_PER_CONDITION)))
        total_steps = epochs * steps_per_epoch
        rng = np.random.default_rng(seed + 1000 * stage["stage"])
        for epoch in range(epochs):
            model.train(); vproj.train(); head.train()
            for xyz_np, valid_np, labels_np in balanced_epoch_batches(condition_data, active, SAMPLES_PER_CONDITION, rng):
                coords = torch.tensor(xyz_np, dtype=torch.float32, device=device)
                valid = torch.tensor(valid_np, dtype=torch.bool, device=device)
                labels = torch.tensor(labels_np, dtype=torch.long, device=device)
                vp = valid.reshape(len(valid), SEGMENTS, SEGMENT_LENGTH, 33).all(dim=2)
                mask_np = uniform_authorized_mask(vp.cpu().numpy(), MASK_FRACTION, seed=seed + 100000 * stage["stage"] + global_step_all)
                target_mask = torch.tensor(mask_np, dtype=torch.bool, device=device)
                view_a = geometric_view(coords, flip_probability=0.0)
                view_b = geometric_view(coords, flip_probability=0.0)
                prediction, target = model(view_a, coords, target_mask)
                jepa_loss = sjepa_cross_entropy(prediction, target, model.target_center)
                tok_a = model.view_encoder(view_a).reshape(len(view_a), SEGMENTS, 33, EMBED_DIM)
                tok_b = model.view_encoder(view_b).reshape(len(view_b), SEGMENTS, 33, EMBED_DIM)
                vic = vicreg_terms(vproj(authorized_pool(tok_a, vp)), vproj(authorized_pool(tok_b, vp)))
                compact, sep = condition_group_terms(authorized_pool(tok_a, vp), labels, margin=GROUP_MARGIN)
                total_loss = jepa_loss + VICREG_WEIGHT * vic + GROUP_WEIGHT * (compact + sep)
                if equiv_on:
                    # L_equiv runs the RAW skeleton and its anatomical mirror through the VIEW encoder,
                    # then penalizes s(enc(Mx)) + s(enc(x)). This shapes the ENCODER (the head alone is
                    # already antisymmetric, so a head-only swap would be an identically-zero no-op).
                    l_equiv = equivariance_loss(head, model.view_encoder, coords, SEGMENTS, EMBED_DIM)
                    total_loss = total_loss + EQUIV_WEIGHT * l_equiv
                    l_equiv_last = float(l_equiv.detach())
                if not torch.isfinite(total_loss):
                    raise FloatingPointError(f"non-finite loss seed={seed} stage={stage['stage']} step={global_step_all}")
                opt.zero_grad(set_to_none=True)
                total_loss.backward()
                assert all(p.grad is None for p in model.target_encoder.parameters())
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
                opt.step()
                model.update_target(cosine_ema(global_step_all, max(total_steps, 1), start=0.996, end=1.0))
                model.update_center(target, beta=0.9)
                global_step_all += 1
        mon = collapse_monitor(model, condition_data, active)
        history.append({"seed": seed, "equiv_on": equiv_on, "stage": stage["stage"],
                        "feature_std": mon["feature_std"], "mean_pair_cosine": mon["mean_pair_cosine"],
                        "l_equiv_last": l_equiv_last})
    model.eval()
    payload = {"mode": MODE, "seed": seed, "equiv_on": equiv_on, "equiv_weight": EQUIV_WEIGHT,
               "config": SMOKE_CONFIG, "curriculum": [s["name"] for s in CURRICULUM],
               "vicreg_weight": VICREG_WEIGHT, "group_weight": GROUP_WEIGHT}
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    checkpoint = {"model_state": model.state_dict(), "head_state": head.state_dict(),
                  "config": SMOKE_CONFIG, "mode": MODE, "mask_keypoints": MASK_KEYPOINTS,
                  "curriculum_complete": True, "conditions_seen": CONDITIONS,
                  "dataset_fingerprint": fingerprint, "fingerprint_payload": payload,
                  "equiv_on": equiv_on, "equiv_weight": EQUIV_WEIGHT if equiv_on else 0.0}
    return model, head, checkpoint, history


# Unconditional: the run is seconds at this scale, and gating it on GAVD_MODE used to leave the
# notebook with two silent sections whenever it was pointed at the real artifact directory.
_m, _h, _ck, _hist = run_curriculum(seed=SEEDS[0], equiv_on=True)
print(f"sanity run OK. fingerprint={_ck['dataset_fingerprint'][:12]} (its own lineage, never the baseline's)")
print(pd.DataFrame(_hist)[["stage", "feature_std", "mean_pair_cosine", "l_equiv_last"]].to_string(index=False))
"""))

# ------------------------------------------------------------------ 6. Re-score through the Arm-1 instrument
CELLS.append(md(r"""
## 6. Re-score each checkpoint through the Arm-1 instrument

We evaluate every retrained checkpoint with the SAME antisymmetric-head readout and the SAME
source-disjoint probe nb_09a uses, so D0 and E1 are judged on one ruler. The signed target `y` is the
raw-coordinate `signed_left_minus_right`; the feature is the antisymmetric head `s` on the retrained TARGET
encoder tokens; the probe is GroupKFold on `video_id` with an inner alpha choice on training sources only.
We also record the retrained encoder's measured anatomical-mirror slope (through the encoder; not the exact
wiring -1). We report per-rung held-out R-squared across seeds.
"""))
CELLS.append(code(r"""
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
ALPHAS = np.logspace(-3, 3, 13)


def signed_left_minus_right(coords):
    coords = np.asarray(coords, dtype=np.float64)[..., :3]
    total = 0.0
    for li, ri in LEFT_RIGHT_PAIRS:
        total += coords[:, li, :].std(axis=0).sum() - coords[:, ri, :].std(axis=0).sum()
    return float(total)


def anatomical_mirror_xyz(coords):
    m = np.asarray(coords, dtype=np.float32).copy()
    m[:, :, 0] = -m[:, :, 0]
    for li, ri in FULL_LR_PAIRS:
        m[:, [li, ri], :] = m[:, [ri, li], :]
    return m


def eval_pool():
    xyz_list, y_list, groups = [], [], []
    for c in CONDITIONS:
        for i, xyz in enumerate(condition_data[c]["xyz"]):
            xyz_list.append(xyz); y_list.append(signed_left_minus_right(xyz))
            groups.append(condition_data[c]["video_ids"][i])
    return np.stack(xyz_list).astype(np.float32), np.array(y_list, dtype=np.float64), np.array(groups)


EVAL_XYZ, EVAL_Y, EVAL_GROUPS = eval_pool()


def antisym_features(model, head, xyz):
    model.target_encoder.eval(); head.eval()
    feats = []
    with torch.no_grad():
        for i in range(0, len(xyz), 16):
            x = torch.tensor(xyz[i:i + 16], dtype=torch.float32, device=device)
            tok = model.target_encoder(x).reshape(len(x), SEGMENTS, 33, EMBED_DIM)
            feats.append(head(tok).cpu().numpy())
    return np.concatenate(feats)


def source_disjoint_r2(X, y, groups):
    n_splits = max(2, min(5, len(np.unique(groups))))
    gkf = GroupKFold(n_splits=n_splits)
    preds = np.full(len(y), np.nan)
    for tr, te in gkf.split(X, y, groups):
        ig = groups[tr]; best_a = ALPHAS[0]; best = -np.inf
        if len(np.unique(ig)) >= 2:
            inner = GroupKFold(n_splits=min(3, len(np.unique(ig))))
            for a in ALPHAS:
                sc = []
                for itr, iva in inner.split(X[tr], y[tr], ig):
                    scaler = StandardScaler().fit(X[tr][itr])
                    mdl = Ridge(alpha=a).fit(scaler.transform(X[tr][itr]), y[tr][itr])
                    sc.append(r2_score(y[tr][iva], mdl.predict(scaler.transform(X[tr][iva]))))
                if np.mean(sc) > best:
                    best, best_a = np.mean(sc), a
        scaler = StandardScaler().fit(X[tr])
        mdl = Ridge(alpha=best_a).fit(scaler.transform(X[tr]), y[tr])
        preds[te] = mdl.predict(scaler.transform(X[te]))
    ok = ~np.isnan(preds)
    return float(r2_score(y[ok], preds[ok]))


def measured_mirror_slope(model, head, xyz):
    Xo = antisym_features(model, head, xyz)
    Xm = antisym_features(model, head, np.stack([anatomical_mirror_xyz(x) for x in xyz]))
    sc = StandardScaler().fit(Xo)
    probe = Ridge(alpha=1.0).fit(sc.transform(Xo), EVAL_Y)
    do, dm = probe.predict(sc.transform(Xo)), probe.predict(sc.transform(Xm))
    return float(np.polyfit(do, dm, 1)[0])


def score_checkpoint(model, head):
    X = antisym_features(model, head, EVAL_XYZ)
    return {"r2": source_disjoint_r2(X, EVAL_Y, EVAL_GROUPS),
            "mirror_slope": measured_mirror_slope(model, head, EVAL_XYZ)}


_sc = score_checkpoint(_m, _h)
print(f"re-score of the sanity run: R2={_sc['r2']:+.3f}  measured mirror slope={_sc['mirror_slope']:+.3f}")
"""))

# ------------------------------------------------------------------ 7. The ablation ladder across seeds
CELLS.append(md(r"""
## 7. Run the ablation ladder D0 vs E1 across seeds and decide

We run D0 (`equiv_on=False`) and E1 (`equiv_on=True`) for each seed, score every rung through section 6,
and compare `mean(E1) - mean(D0)` against D0's seed-to-seed standard deviation. The pre-registered credit
rule: the equivariance term earns credit only if `mean(E1) - mean(D0)` EXCEEDS `std(D0)` (the trajectory
control) AND E1's mean clears its own floor by the same 0.05 margin the primary gate uses. In smoke this is
a plumbing demonstration, not evidence about the real encoder.
"""))
CELLS.append(code(r"""
FLOOR_MARGIN = 0.05
ladder_rows = []
all_history = []
for seed in SEEDS:
    for equiv_on in (False, True):
        model, head, ck, hist = run_curriculum(seed=seed, equiv_on=equiv_on)
        sc = score_checkpoint(model, head)
        all_history.extend(hist)
        ladder_rows.append({"rung": "E1" if equiv_on else "D0", "seed": seed,
                            "r2": sc["r2"], "mirror_slope": sc["mirror_slope"],
                            "fingerprint": ck["dataset_fingerprint"][:12],
                            "final_feature_std": hist[-1]["feature_std"],
                            "final_mean_pair_cosine": hist[-1]["mean_pair_cosine"]})
        print(f"seed {seed} {'E1' if equiv_on else 'D0'}: R2={sc['r2']:+.3f} "
              f"mirror={sc['mirror_slope']:+.3f} fp={ck['dataset_fingerprint'][:8]}")

ladder = pd.DataFrame(ladder_rows)
d0_r2 = ladder[ladder["rung"] == "D0"]["r2"]
e1_r2 = ladder[ladder["rung"] == "E1"]["r2"]
d0_mean, d0_std = float(d0_r2.mean()), float(d0_r2.std(ddof=0))
e1_mean = float(e1_r2.mean())
effect = e1_mean - d0_mean
exceeds_spread = bool(effect > max(d0_std, 1e-9))
beats_floor = bool((e1_mean - d0_mean) >= FLOOR_MARGIN)
equiv_credited = bool(exceeds_spread and beats_floor)
print("\n" + ladder.to_string(index=False))
print(f"\nD0 mean R2 = {d0_mean:+.3f} (std {d0_std:.3f})   E1 mean R2 = {e1_mean:+.3f}")
print(f"effect E1-D0 = {effect:+.3f}   exceeds D0 spread: {exceeds_spread}   beats floor 0.05: {beats_floor}")
print(f"L_equiv earns credit: {equiv_credited}")
# Printed unconditionally. This warning used to be gated on GAVD_MODE, so it stayed hidden in exactly
# the configuration where a reader is most likely to mistake these numbers for real ones.
print(f"\n{DATA_SOURCE.upper()} DATA AT {TRAINING_SCALE.upper()} SCALE: this is a plumbing demonstration of "
      "the ladder, NOT evidence about the real encoder. The real ladder is new_nb_09_02.")
"""))

# ------------------------------------------------------------------ 8. The real run happened elsewhere + persist
CELLS.append(md(r"""
## 8. The real run happened elsewhere, and why this recipe was not followed

This section used to hold a recipe for a deferred real run: paste section 2's head, mirror, and
equivariance loss into `04_pretrain_sjepa_on_normal.ipynb`, add the head to the trainable list, run D0 and
E1 rungs across seeds, and re-score each checkpoint through `nb_09a`. That run has since been carried out,
but **not** by following those steps, and the reason is a defect in section 2's loss rather than a change
of plan.

**Why the recipe above must not be pasted into notebook 04 as written.** The head `s` in section 2 is
trainable, and the loss is an absolute squared residual, so the objective has a degenerate solution: the
head can shrink its own output toward zero and drive the term down without the encoder changing at all.
Measured on synthetic fixtures, that is exactly what it does. The term falls about 184-fold while the
head's output scale shrinks about 4.8-fold, and a separately measured, parameter-free mirror residual stays
pinned at its mirror-blind value, improving by 0.010 against a control. Anyone watching the loss curve
would have recorded a success. The repair is to normalize the residual by the signal's own magnitude, so
that shrinking the head scales numerator and denominator together and buys nothing.

**Where the real run lives.** A separate four-notebook series carries it out, leaving notebook 04 untouched
so that the baseline lineage and the experiment stay separable. Run them in this order:

1. `new_nb_09_00_methodology_and_contract` fixes the endpoints, guardrails, and credit rule in advance, and
   verifies the pose cache, mask whitelist, mapping hash, and baseline checkpoint. It trains nothing.
2. `new_nb_09_01_mechanism_and_smoke_validation` proves the term reaches the encoder, calibrates the
   endpoint against fixtures with known answers, and bakes off the loss variants. It is what exposed the
   degenerate solution described above, and it selects the variant the real run uses.
3. `new_nb_09_02_real_multiseed_equivariant_training` runs the real D0-versus-E1 ladder on the locked
   cohort with the full 11,400-update curriculum per rung, checkpointing per rung so it can resume.
4. `new_nb_09_03_evaluation_results_discussion` recomputes every endpoint and guardrail and applies the
   credit rule.

**What that run found.** The next cell reads the verdict out of the series' own bundle rather than
restating it, so this notebook cannot drift from it. In summary: the endpoint moved decisively and the term
still earned no credit, because a guardrail regressed and variance removal is a competing explanation for
the endpoint gain rather than an unrelated cost.

**What this notebook remains good for.** It is the plumbing scaffold: it demonstrates the ladder shape, the
per-rung fingerprinting, the collapse monitors, and the credit-rule arithmetic on a synthetic cohort at
smoke scale. Its ladder numbers are not evidence about the real encoder, and the bundle below is labelled
by data source and training scale so they cannot be read as if they were.
"""))
CELLS.append(code(r"""
# Kept verbatim as the plan of record, with the reason it must not be followed attached to it. Deleting it
# would hide the mistake; presenting it without the status field would invite someone to repeat it.
SUPERSEDED_RECIPE = {
    "status": "SUPERSEDED, DO NOT FOLLOW AS WRITTEN. Step 1 installs the absolute equivariance term below, "
              "which a trainable head can satisfy by shrinking its own output instead of changing the "
              "encoder. Use the scale-invariant term and the executable path in EXECUTED_PATH instead.",
    "step_1": "Paste AntisymmetricHead + anatomical_mirror_coords + equivariance_loss into nb04 training-step cell; add head params to trainable; add EQUIV_WEIGHT*equivariance_loss(head, model.view_encoder, coords, SEGMENTS, EMBED_DIM) to total_loss behind equiv_on. Loss runs raw coords AND their anatomical mirror THROUGH the encoder; a head-only token swap is a no-op.",
    "step_2": "Add equiv_weight and equiv_on to fingerprint_payload (each checkpoint gets its own fingerprint; never the baseline's).",
    "step_3_D0": "GAVD_MODE=real SJEPA_RUN_PROFILE=recommended IDEA9_EQUIV_WEIGHT=0 across seeds 0,1,2[,3,4].",
    "step_3_E1": "GAVD_MODE=real SJEPA_RUN_PROFILE=recommended IDEA9_EQUIV_WEIGHT=0.02 across the same seeds.",
    "step_4": "Re-score every checkpoint through nb_09a (SJEPA_INSPECT_CHECKPOINT=<file>); apply the section-7 credit rule (E1-D0 must exceed D0 seed spread AND clear the 0.05 floor).",
}
EXECUTED_PATH = {
    "notebooks_in_order": [
        "new_nb_09_00_methodology_and_contract",
        "new_nb_09_01_mechanism_and_smoke_validation",
        "new_nb_09_02_real_multiseed_equivariant_training",
        "new_nb_09_03_evaluation_results_discussion",
    ],
    "why_not_this_scaffold": "section 2's absolute term is satisfiable by shrinking the trainable head, so "
                            "the real run uses a scale-invariant form; notebook 04 was left untouched so "
                            "the baseline lineage and the experiment stay separable",
    "endpoint": "rho, a parameter-free normalized mirror residual read with an identity head, so no "
                "trainable weight can influence it (0 = mirror equivariant, 4 = mirror blind)",
}


def real_run_outcome():
    # Read the superseding series' verdict instead of restating it, so this bundle cannot drift from it.
    path = OUT_DIR / "idea9_arm2" / "idea9_arm2_evaluation_result.json"
    if not path.is_file():
        return {"status": "not present in this artifact root", "expected_bundle": str(path)}
    result = json.loads(path.read_text(encoding="utf-8"))
    primary = result["primary"]
    return {
        "status": "completed",
        "bundle": str(path.relative_to(OUT_DIR)),
        "verdict": result["PRIMARY_VERDICT"],
        "seeds_per_rung": len({row["seed"] for row in result["per_rung"]}),
        "rho_D0_mean": primary["D0_mean"],
        "rho_E1_mean": primary["E1_mean"],
        "improvement": primary["improvement"],
        "D0_seed_spread": primary["D0_seed_spread"],
        "credit_rule": result["credit_rule"],
        "failed_guardrails": [row["guardrail"] for row in result["guardrails"]
                              if row["measurable"] and not row["within_control_spread"]],
        "unmeasurable_guardrails": [row["guardrail"] for row in result["guardrails"] if not row["measurable"]],
        "protocol_deviations": [entry["field"] for entry in result.get("protocol_deviations", [])],
    }


REAL_RUN = real_run_outcome()
bundle = {
    "notebook": "nb_09b_equivariant_retrain",
    "arm": "arm2_equivariance_coupled_retrain",
    "role": "smoke-scale plumbing scaffold; superseded for results by the new_nb_09 series",
    "mode": TRAINING_SCALE,
    "data_source": DATA_SOURCE,
    "training_scale": TRAINING_SCALE,
    "artifact_mode": MODE,
    "equiv_weight": EQUIV_WEIGHT,
    "seeds": SEEDS,
    "total_loss": f"jepa + {VICREG_WEIGHT}*vicreg + {GROUP_WEIGHT}*group + {EQUIV_WEIGHT}*L_equiv",
    "L_equiv": "mean((s(encoder(Mx)) + s(encoder(x)))^2), M = anatomical mirror on raw coords, run THROUGH the view encoder, label-free (no L_axis). A head-only token swap would be identically zero (no-op). DEGENERATE: a trainable s can shrink its output to satisfy this without changing the encoder; the real run uses a scale-invariant form instead.",
    "ladder": ladder_rows,
    "collapse_history": all_history,
    "decision": {"d0_mean_r2": d0_mean, "d0_std_r2": d0_std, "e1_mean_r2": e1_mean,
                 "effect_E1_minus_D0": effect, "exceeds_D0_spread": exceeds_spread,
                 "beats_floor_0.05": beats_floor, "equiv_credited": equiv_credited},
    "superseded_recipe": SUPERSEDED_RECIPE,
    "executed_path": EXECUTED_PATH,
    "real_run": REAL_RUN,
    "deferred": REAL_RUN["status"] != "completed",
    "notes": "These numbers are plumbing checks on a SYNTHETIC cohort at SMOKE scale, not a result: "
             "`data_source` and `training_scale` say so, `artifact_mode` is only the directory written to, "
             "and `mode` repeats the training scale for readers of older bundles that used it for the "
             "cache. The real multi-seed result is `real_run` above, produced by new_nb_09_02 and "
             "adjudicated by new_nb_09_03 with a repaired scale-invariant term. The token-swap slope -1 is "
             "exact by construction; the anatomical-mirror slope is measured through the encoder. All "
             "results transductive; source video is the independent unit; folder labels are dataset "
             "annotations, not diagnoses.",
}
bundle_path = OUT_DIR / "idea9_equivariant_retrain_result.json"
bundle_path.write_text(json.dumps(bundle, indent=2))
print(f"wrote {bundle_path}")
print(json.dumps({k: bundle[k] for k in ("mode", "data_source", "artifact_mode", "seeds", "decision")}, indent=2))
print("\nthe real run this scaffold points at:")
print(json.dumps(REAL_RUN, indent=2))
"""))

nb = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3 (gavd3-sjepa)", "language": "python", "name": "gavd3-sjepa"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
NB_PATH.write_text(json.dumps(nb, indent=1))
print(f"wrote {NB_PATH}  ({len(CELLS)} cells)")
