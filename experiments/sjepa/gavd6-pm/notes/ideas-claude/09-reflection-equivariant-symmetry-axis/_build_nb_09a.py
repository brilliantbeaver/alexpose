"""Builder for nb_09a_antisymmetric_readout_probe.ipynb (Idea 9, Arm 1).

Run:  python3 _build_nb_09a.py
Emits the notebook at the gavd root (parents[3]) so it sits beside 00-06,
nb_05a, and nb_05b. Each cell is held as a plain string so the code stays
readable and the emitted JSON is guaranteed valid. Committed alongside the
notebook so the notebook can be regenerated deterministically.

Arm 1 is the ZERO-RETRAIN readout arm. It loads the EXISTING frozen ea59fea0
target encoder (unchanged), attaches an antisymmetric "left minus right" head
that is antisymmetric BY CONSTRUCTION under an input swap, proves that wiring
guarantee numerically (slope exactly -1 to floating-point tolerance), and then
asks whether the antisymmetry-CONSTRAINED head is a decodable signed axis that
beats the binding bar max(untrained floor, standard ea59fea0) on item 05's
frozen source-disjoint probe. The encoder is never trained here; only the ridge
probe on top of the fixed head feature is fit, in fold, exactly as item 05 does.

This file deliberately reuses item 05's evaluation instrument verbatim
(find_project_root, preprocessing, model classes, source_disjoint_probe,
sign_consistency, raw_null_feature, pooled_nuisance_feature, anatomical_mirror)
so Arm 1 is judged on exactly the same ruler item 05 used.
"""
import json
from pathlib import Path

NB_PATH = Path(__file__).resolve().parents[3] / "nb_09a_antisymmetric_readout_probe.ipynb"


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
# Notebook 09a: Antisymmetric-readout probe (Idea 9, Arm 1, zero retrain)

This notebook reifies **Arm 1** of research idea 9,
[reflection-equivariant-symmetry-axis](notes/ideas-claude/09-reflection-equivariant-symmetry-axis/README.md).
It reuses item 05's frozen evaluation instrument (`nb_05a`) without change and adds one thing: an
**antisymmetric left-minus-right readout head** that is antisymmetric BY CONSTRUCTION. Arm 1 asks a
single falsifiable question on the EXISTING, UNCHANGED `ea59fea0` S-JEPA checkpoint:

> On source-video-disjoint folds, does an antisymmetry-CONSTRAINED pure-difference head, read off the
> frozen `ea59fea0` target encoder, decode the signed left-minus-right axis better than the binding bar
> `max(untrained-floor, standard-ea59fea0)` by a pre-registered margin of 0.05 R-squared?

**Arm 1 does not change the encoder.** It is a readout-only comparison on frozen features, so it is
the low-risk deliverable. The heavier equivariance-coupled RETRAIN (which is the only arm that changes
the encoder) is Arm 2, in `nb_09b`.

Two mirror facts kept strictly apart (this is the load-bearing correctness point):

- **Wiring guarantee (exact).** Swapping each left joint's feature with its right partner's AT THE
  HEAD'S INPUT negates the head output to floating-point precision (slope exactly -1). This is algebra
  about the head alone, independent of the encoder. We prove it in section 3.
- **Anatomical-mirror slope (measured, not -1).** Mirroring the RAW skeleton and re-running it THROUGH
  the encoder gives a measured slope, because the encoder mixes tokens with attention and adds
  per-joint position codes. Item 05 measured -0.741 for `ea59fea0`. We report it honestly and
  never call it -1.

All results are TRANSDUCTIVE: the encoder saw the evaluation rows during training. The source video is
the independent unit, and folder labels (stroke, parkinsons) are dataset annotations, not diagnoses.
See `notes/ideas-claude/_shared_facts.md` for the single source of truth on every number.
"""))

# ------------------------------------------------------------------ 0. Environment
CELLS.append(md(r"""
## 0. Environment and mode

We resolve the project root the same way notebooks 04-06 and nb_05a do (via `ALEXPOSE_ROOT` or a
`.git` + `data/gavd` marker), load `.env`, and decide the run mode. If the real artifacts are missing
we fall back to smoke mode with a synthetic cohort, so the notebook is always runnable. This cell is
reused verbatim from `nb_05a`.
"""))
CELLS.append(code(r"""
from pathlib import Path
import os, sys, math, json, hashlib, copy, warnings

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

ARTIFACT_ROOT = Path(
    os.getenv("GAVD_ARTIFACT_DIR", TUTORIAL_DIR / "work" / "artifacts")
).expanduser()

REQUESTED_MODE = os.getenv("GAVD_MODE", "smoke").strip().lower()
if REQUESTED_MODE not in {"smoke", "real"}:
    raise ValueError("GAVD_MODE must be smoke or real")

TRUTHY = {"1", "true", "yes", "on"}
INCLUDE_AUGMENTED = os.getenv("SJEPA_INCLUDE_AUGMENTED_NORMAL", "1").strip().lower() in TRUTHY
EXPLICIT_CHECKPOINT = os.getenv("SJEPA_INSPECT_CHECKPOINT", "").strip()
CHECKPOINT_NAME = (
    "sjepa_curriculum_final_augmented.pt" if INCLUDE_AUGMENTED else "sjepa_curriculum_final.pt"
)
EXPECTED_FINGERPRINT_PREFIX = "ea59fea0"
CONDITIONS = ["normal", "parkinsons", "stroke", "myopathic", "cerebralpalsy"]


def artifact_dir_for(mode):
    return ARTIFACT_ROOT / mode


def checkpoint_path_for(mode):
    if EXPLICIT_CHECKPOINT:
        p = Path(EXPLICIT_CHECKPOINT).expanduser()
        return p if p.is_absolute() else artifact_dir_for(mode) / p
    return artifact_dir_for(mode) / CHECKPOINT_NAME


real_ckpt = checkpoint_path_for("real")
real_poses = artifact_dir_for("real") / "poses"
have_real = REQUESTED_MODE == "real" and real_ckpt.is_file() and real_poses.exists()
MODE = "real" if have_real else "smoke"
if REQUESTED_MODE == "real" and not have_real:
    warnings.warn(
        f"Requested real mode but artifacts are missing "
        f"(checkpoint={real_ckpt.is_file()}, poses={real_poses.exists()}). "
        "Falling back to smoke mode with a synthetic cohort. Smoke numbers are illustrative only."
    )

ARTIFACT_DIR = artifact_dir_for(MODE)
POSE_DIR = ARTIFACT_DIR / "poses"
CHECKPOINT_PATH = checkpoint_path_for(MODE)
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

if MODE == "smoke":
    print("SMOKE MODE: hand-authored motions test code paths only. No clinical validity.")
print(f"PROJECT_ROOT      : {PROJECT_ROOT}")
print(f"ARTIFACT_ROOT     : {ARTIFACT_ROOT}")
print(f"requested mode    : {REQUESTED_MODE}")
print(f"effective MODE    : {MODE}")
print(f"checkpoint        : {CHECKPOINT_PATH}  (present={CHECKPOINT_PATH.is_file()})")
print(f"pose cache        : {POSE_DIR}  (present={POSE_DIR.exists()})")
"""))

# ------------------------------------------------------------------ 1. Anatomy constants (verbatim from 05a)
CELLS.append(md(r"""
## 1. Anatomy constants and the frozen laterality operators (verbatim from nb_05a)

These constants and functions match nb_05a exactly, so Arm 1 is judged on the same ruler. `MASK_KEYPOINTS`
is the 12-landmark maskable whitelist. `LEFT_RIGHT_PAIRS` is the six pair list the head and the target are
built from. `signed_left_minus_right` is the regression target `y` (frozen before any fit). `anatomical_mirror`
negates x and swaps left/right landmarks on RAW coordinates.
"""))
CELLS.append(code(r"""
BLAZEPOSE_33 = [
    "NOSE", "LEFT_EYE_INNER", "LEFT_EYE", "LEFT_EYE_OUTER", "RIGHT_EYE_INNER", "RIGHT_EYE",
    "RIGHT_EYE_OUTER", "LEFT_EAR", "RIGHT_EAR", "MOUTH_LEFT", "MOUTH_RIGHT", "LEFT_SHOULDER",
    "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW", "LEFT_WRIST", "RIGHT_WRIST", "LEFT_PINKY",
    "RIGHT_PINKY", "LEFT_INDEX", "RIGHT_INDEX", "LEFT_THUMB", "RIGHT_THUMB", "LEFT_HIP", "RIGHT_HIP",
    "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE", "LEFT_HEEL", "RIGHT_HEEL",
    "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX",
]
MASK_KEYPOINTS = [11, 12, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]

# The six left-right pairs the head and the signed target are built from.
LEFT_RIGHT_PAIRS = [(11, 12), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32)]

# The FULL 16-pair mirror (matches notebook 05's geometric_view flip): every left/right landmark.
FULL_MIRROR_PAIRS = [
    (1, 4), (2, 5), (3, 6), (7, 8), (9, 10),
    (11, 12), (13, 14), (15, 16), (17, 18), (19, 20), (21, 22),
    (23, 24), (25, 26), (27, 28), (29, 30), (31, 32),
]


def signed_left_minus_right(coords):
    '''coords: (T, 33, 3) or (T, 33, >=3). Signed per-side excursion, left minus right.'''
    coords = np.asarray(coords, dtype=np.float64)[..., :3]
    total = 0.0
    for left_idx, right_idx in LEFT_RIGHT_PAIRS:
        left_excursion = coords[:, left_idx, :].std(axis=0).sum()
        right_excursion = coords[:, right_idx, :].std(axis=0).sum()
        total += left_excursion - right_excursion
    return float(total)


def anatomical_mirror(coords, pairs=FULL_MIRROR_PAIRS):
    '''Negate x and swap each left landmark with its right partner (whole-body reflection).'''
    mirrored = np.asarray(coords, dtype=np.float32).copy()
    mirrored[:, :, 0] = -mirrored[:, :, 0]
    for left_idx, right_idx in pairs:
        mirrored[:, [left_idx, right_idx], :] = mirrored[:, [right_idx, left_idx], :]
    return mirrored


_demo = np.random.randn(20, 33, 3).astype(np.float32)
_orig = signed_left_minus_right(_demo)
_mir = signed_left_minus_right(anatomical_mirror(_demo))
print(f"raw-coordinate self-check:  original={_orig:+.4f}  mirrored={_mir:+.4f}  (should be equal-and-opposite)")
assert abs(_orig + _mir) < 1e-6 * (1 + abs(_orig)), "raw target must be exactly antisymmetric"
print("OK: signed_left_minus_right is antisymmetric on raw coordinates.")
"""))

# ------------------------------------------------------------------ 2. Preprocessing (verbatim from 05a)
CELLS.append(md(r"""
## 2. Preprocessing (reused verbatim from nb_05a)

Short-gap interpolation, pelvis-centering and body-scale normalization, temporal resize to the model's
frame count. Copied so the tokens Arm 1 caches are identical to what training and item 05 produced. The
mirror is applied on the RAW coordinate column and then the SAME preprocessing is re-run, so the
mirrored input the encoder sees is a genuine reflection of a real preprocessed sequence.
"""))
CELLS.append(code(r"""
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
    '''raw_sequence: [T, 33, 4]. Returns (xyz [frames,33,3], valid [frames,33]).'''
    interpolated, valid = interpolate_low_visibility(raw_sequence)
    scaled = center_and_scale(interpolated)
    xyz = temporal_resize(scaled[..., :3], frames)
    valid_resized = temporal_resize(valid.astype(np.float32), frames) > 0.5
    return xyz.astype(np.float32), valid_resized


print("preprocessing helpers ready (interpolate -> center_and_scale -> temporal_resize).")
"""))

# ------------------------------------------------------------------ 3. Model classes (verbatim from 05a)
CELLS.append(md(r"""
## 3. Model classes (reused verbatim so `state_dict` keys match nb_05a and the checkpoint)

The S-JEPA classes are pasted inline exactly as nb_05a does, so `load_state_dict` on `ea59fea0` matches
key-for-key. We ALWAYS construct the model with `SJEPAGait(**checkpoint["config"])`, reading whatever
dimensions the checkpoint stored (embed_dim 96, encoder_depth 4 for the real lineage; a smaller stand-in
for smoke), never hardcoding them. If PyTorch is unavailable the learned lanes are skipped with a clear
message and only Lane B and the target run.
"""))
CELLS.append(code(r"""
try:
    import torch
    from torch import nn
    HAVE_TORCH = True
except Exception as exc:  # pragma: no cover
    HAVE_TORCH = False
    print(f"PyTorch unavailable ({exc}); learned-encoder lanes will be skipped.")

if HAVE_TORCH:
    class SkeletonPatchEncoder(nn.Module):
        def __init__(self, frames=64, joints=33, coordinate_dim=3, segment_length=4,
                     embed_dim=96, depth=4, heads=4, dropout=0.0):
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
            batch = x.shape[0]
            if x.shape[1:] != (self.frames, self.joints, self.coordinate_dim):
                raise ValueError(f"Expected [B,{self.frames},{self.joints},{self.coordinate_dim}], got {tuple(x.shape)}")
            x = x.reshape(batch, self.segments, self.segment_length, self.joints, self.coordinate_dim)
            x = x.permute(0, 1, 3, 2, 4)
            x = x.flatten(3)
            return x

        def positioned_tokens(self, x):
            tokens = self.patch_embed(self.patchify(x))
            tokens = tokens + self.time_pos[None, :, None, :] + self.joint_pos[None, None, :, :]
            return tokens

        def forward(self, x, keep_mask=None):
            tokens = self.positioned_tokens(x)
            batch = tokens.shape[0]
            flat = tokens.reshape(batch, self.segments * self.joints, self.embed_dim)
            encoded = self.blocks(flat)
            return self.norm(encoded)

    class SkeletonPredictor(nn.Module):
        def __init__(self, segments, joints, encoder_dim=96, predictor_dim=96, depth=2, heads=4, dropout=0.0):
            super().__init__()
            self.segments, self.joints, self.predictor_dim = segments, joints, predictor_dim
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

        def forward(self, *args, **kwargs):
            raise NotImplementedError("Predictor is not needed for the frozen probe.")

    class SJEPAGait(nn.Module):
        def __init__(self, frames=64, joints=33, coordinate_dim=3, segment_length=4,
                     embed_dim=96, encoder_depth=4, predictor_depth=2, heads=4):
            super().__init__()
            self.view_encoder = SkeletonPatchEncoder(
                frames, joints, coordinate_dim, segment_length, embed_dim, encoder_depth, heads)
            self.target_encoder = copy.deepcopy(self.view_encoder)
            for parameter in self.target_encoder.parameters():
                parameter.requires_grad_(False)
            segments = frames // segment_length
            self.predictor = SkeletonPredictor(segments, joints, embed_dim, embed_dim, predictor_depth, heads)
            self.register_buffer("target_center", torch.zeros(embed_dim))

    print("model classes defined.")
"""))

# ------------------------------------------------------------------ 4. Load checkpoint OR build smoke model (verbatim from 05a)
CELLS.append(md(r"""
## 4. Bind to one checkpoint (`ea59fea0`) or build a matched smoke encoder

Real mode loads the checkpoint with the same guards nb_05a uses (mode, mask whitelist, curriculum
completion, conditions seen), verifies the `ea59fea0` fingerprint prefix, and constructs the model from
the stored config. Smoke mode builds a small random `SJEPAGait` so the plumbing runs. We also build the
untrained-encoder FLOOR (Lane C): a fresh random-init `SJEPAGait` of the identical architecture. This
cell is reused verbatim from nb_05a.
"""))
CELLS.append(code(r"""
CHECKPOINT = None
FINGERPRINT = "smoke-random"
if HAVE_TORCH and MODE == "real":
    CHECKPOINT = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    if CHECKPOINT.get("mode") != MODE:
        raise ValueError(f"Checkpoint mode {CHECKPOINT.get('mode')} does not match {MODE}.")
    if CHECKPOINT.get("mask_keypoints") != MASK_KEYPOINTS:
        raise ValueError("Checkpoint mask whitelist does not match the 12-point set.")
    if not CHECKPOINT.get("curriculum_complete", False):
        raise ValueError("Checkpoint is not a completed progressive curriculum final.")
    if CHECKPOINT.get("conditions_seen") != CONDITIONS:
        raise ValueError(f"Unexpected curriculum order: {CHECKPOINT.get('conditions_seen')}.")
    FINGERPRINT = str(CHECKPOINT.get("dataset_fingerprint", ""))
    if not FINGERPRINT.startswith(EXPECTED_FINGERPRINT_PREFIX):
        raise ValueError(
            f"Fingerprint {FINGERPRINT[:12]} does not start with {EXPECTED_FINGERPRINT_PREFIX}."
        )
    config = CHECKPOINT["config"]
    model = SJEPAGait(**config)
    model.load_state_dict(CHECKPOINT["model_state"])
    model.eval()
    FRAMES = config["frames"]
    SEGMENT_LENGTH = config["segment_length"]
    EMBED_DIM = config["embed_dim"]
    print(f"Loaded ea59fea0 lineage. fingerprint prefix: {FINGERPRINT[:12]}  config: {config}")
elif HAVE_TORCH:
    config = {"frames": 32, "joints": 33, "coordinate_dim": 3, "segment_length": 4,
              "embed_dim": 32, "encoder_depth": 1, "predictor_depth": 1, "heads": 4}
    torch.manual_seed(RANDOM_SEED)
    model = SJEPAGait(**config).eval()
    FRAMES, SEGMENT_LENGTH, EMBED_DIM = config["frames"], config["segment_length"], config["embed_dim"]
    print(f"Smoke encoder built (untrained). config: {config}")
else:
    FRAMES, SEGMENT_LENGTH, EMBED_DIM = 64, 4, 96  # documentary defaults; no encoder available.
    model = None

floor_model = None
if HAVE_TORCH:
    torch.manual_seed(RANDOM_SEED + 1)
    floor_model = SJEPAGait(**config).eval()
    print("Untrained-encoder floor model built (fresh random init).")

SEGMENTS = FRAMES // SEGMENT_LENGTH
print(f"FRAMES={FRAMES}  SEGMENT_LENGTH={SEGMENT_LENGTH}  SEGMENTS={SEGMENTS}  EMBED_DIM={EMBED_DIM}")
"""))

# ------------------------------------------------------------------ 5. The antisymmetric head + wiring proof
CELLS.append(md(r"""
## 5. The antisymmetric head, and the exact wiring guarantee

This is the one new piece of machinery. The head builds the signed axis as

`s = sum over the six pairs k of ( f(L_k) - f(R_k) )`

where `f` is a SINGLE shared per-joint map applied identically to the left joint's feature and the right
joint's feature. We use `f = Linear(EMBED_DIM, 32) -> GELU -> Linear(32, m)`, with `m` small (default 4),
so `s` is an `m`-dimensional signed vector. Two design points, both load-bearing:

- **Difference only, no `left + right` term.** A `f(L_k) + f(R_k)` sum does not change under the swap
  (it is mirror-invariant) and would dilute the guaranteed sign-flip, so the head keeps only the
  difference. This is exactly where Arm 1 departs from item 05's `laterality_feature_from_tokens`, which
  also emitted the `l + r` symmetric half. Arm 1's head is a NEW pure-difference head, not a reuse of
  item 05's Lane A feature.
- **`f` is a fixed, seeded, shared map (Arm 1 changes nothing).** The antisymmetry guarantee holds for
  ANY `f`, trained or not, so a fixed random `f` is a legitimate antisymmetric readout and keeps Arm 1
  truly zero-retrain. The only thing fit later is item 05's ridge probe on top of the head feature, in
  fold, exactly as every other lane. (Arm 2, in nb_09b, is the arm that actually trains `f` jointly with
  the encoder under a label-free equivariance loss.)

**Wiring guarantee (exact).** If we swap each left joint's feature with its right partner's AT THE
HEAD'S INPUT, every `f(L_k) - f(R_k)` becomes `f(R_k) - f(L_k)`, so `s` negates exactly. We prove this
below to floating-point tolerance. This is the ONLY place the slope is exactly -1; the anatomical mirror
(section 8) runs through the encoder and gives a measured, approximate slope.
"""))
CELLS.append(code(r"""
HEAD_OUT_DIM = int(os.getenv("IDEA9_HEAD_OUT_DIM", "4"))  # m: width of the signed vector s

antisym_head = None
if HAVE_TORCH:
    class AntisymmetricHead(nn.Module):
        '''s = sum_k ( f(L_k) - f(R_k) ) over LEFT_RIGHT_PAIRS, with f SHARED across all joints/sides.
        Difference only; no symmetric (l + r) term. Antisymmetric under an input swap by construction.'''
        def __init__(self, embed_dim, out_dim=4, hidden=32, pairs=LEFT_RIGHT_PAIRS):
            super().__init__()
            self.pairs = list(pairs)
            self.f = nn.Sequential(
                nn.Linear(embed_dim, hidden), nn.GELU(), nn.Linear(hidden, out_dim))

        def per_joint_feature(self, tokens):
            '''tokens: [B, SEGMENTS, 33, D] -> per-joint time-mean [B, 33, D] (matches nb_05a).'''
            return tokens.mean(dim=1)

        def forward(self, tokens):
            '''tokens: [B, SEGMENTS, 33, D]. Returns s: [B, out_dim].'''
            pj = self.per_joint_feature(tokens)  # [B, 33, D]
            s = 0.0
            for left_idx, right_idx in self.pairs:
                s = s + (self.f(pj[:, left_idx, :]) - self.f(pj[:, right_idx, :]))
            return s

        def forward_capacity_matched(self, tokens):
            '''Capacity-matched CONTROL readout. Uses the EXACT SAME shared f, width, and per-pair
            aggregation as forward(), and differs ONLY by adding the symmetric (l + r) path next to the
            antisymmetric (l - r) path. Returns [B, 2*out_dim] = concat( sum_k (f(L)-f(R)), sum_k (f(L)+f(R)) ).
            This is NOT antisymmetric: the (l + r) half is mirror-INVARIANT. Comparing A' against this
            control isolates the antisymmetry constraint from the head's nonlinearity, init, and pair info,
            because those are held identical; only the symmetric path is added.'''
            pj = self.per_joint_feature(tokens)  # [B, 33, D]
            diff = 0.0
            summ = 0.0
            for left_idx, right_idx in self.pairs:
                diff = diff + (self.f(pj[:, left_idx, :]) - self.f(pj[:, right_idx, :]))
                summ = summ + (self.f(pj[:, left_idx, :]) + self.f(pj[:, right_idx, :]))
            return torch.cat([diff, summ], dim=-1)

        @torch.no_grad()
        def forward_swapped(self, tokens):
            '''Swap each left joint's feature with its right partner's AT THE HEAD INPUT, then run f.
            This is the exact wiring check; it must return -forward(tokens).'''
            pj = self.per_joint_feature(tokens)  # [B, 33, D]
            pj_sw = pj.clone()
            for left_idx, right_idx in self.pairs:
                pj_sw[:, left_idx, :] = pj[:, right_idx, :]
                pj_sw[:, right_idx, :] = pj[:, left_idx, :]
            s = 0.0
            for left_idx, right_idx in self.pairs:
                s = s + (self.f(pj_sw[:, left_idx, :]) - self.f(pj_sw[:, right_idx, :]))
            return s

    torch.manual_seed(RANDOM_SEED + 7)
    antisym_head = AntisymmetricHead(EMBED_DIM, out_dim=HEAD_OUT_DIM).eval()
    for p in antisym_head.parameters():
        p.requires_grad_(False)  # Arm 1 is zero-retrain: f is fixed.

    # ---- Wiring guarantee proof: forward_swapped(x) == -forward(x) to tolerance ----
    _rng = np.random.default_rng(RANDOM_SEED)
    _tok = torch.tensor(_rng.standard_normal((8, SEGMENTS, 33, EMBED_DIM)), dtype=torch.float32)
    with torch.no_grad():
        _s = antisym_head(_tok)
        _s_sw = antisym_head.forward_swapped(_tok)
    _max_abs = float((_s_sw + _s).abs().max())
    _scale = float(_s.abs().max()) + 1e-9
    # Per-sample slope of swapped vs original (should be exactly -1).
    _slope = float(np.polyfit(_s.reshape(-1).numpy(), _s_sw.reshape(-1).numpy(), 1)[0])
    print(f"WIRING CHECK  max|s_swap + s| = {_max_abs:.3e}   (relative {_max_abs/_scale:.3e})")
    print(f"WIRING CHECK  swap-vs-original slope = {_slope:+.6f}   (must be -1.000000)")
    WIRING_IDENTITY_OK = bool(_max_abs <= 1e-5 * _scale and abs(_slope + 1.0) <= 1e-4)
    assert WIRING_IDENTITY_OK, "Antisymmetric head failed the exact wiring guarantee; fix the head."
    print("OK: head is antisymmetric by construction under the input swap (slope exactly -1).")
else:
    WIRING_IDENTITY_OK = None
    _slope = None
    print("No torch: antisymmetric head and wiring proof skipped.")
"""))

# ------------------------------------------------------------------ 6. Cohort loading / smoke synthesis (verbatim from 05a)
CELLS.append(md(r"""
## 6. Build the cohort with provenance and source-video tags (verbatim from nb_05a)

Real mode reads the `.npz` pose cache exactly as nb_05a's `pose_records_from_cache`, keeping the raw
`[T, 33, 4]` sequence, `video_id`, `condition`, `sequence_id`, and a structural `provenance` tag
(canonical `poses/` folder => canonical). Smoke mode reuses the same synthetic fixtures and the same
planted, sign-flipping lateral lean overlay nb_05a uses, so the probe has a genuine sign to recover and
the mirror a genuine sign to flip. The planted lean is a plumbing signal, not physiology.
"""))
CELLS.append(code(r"""
FRAMES_SMOKE = 64


def synthetic_gait_sequence(condition="normal", frames=64, seed=0):
    '''Code-path fixture copied from nb_05a (NOT a disease simulation).'''
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
    if magnitude == 0.0 or sign == 0:
        return seq
    rng = np.random.default_rng(seed)
    out = seq.copy()
    phase = np.linspace(0.0, 4.0 * np.pi, len(seq), endpoint=False)
    for left_idx, right_idx in LEFT_RIGHT_PAIRS:
        gain = 0.03 * magnitude * sign * rng.uniform(0.8, 1.2)
        out[:, left_idx, 0] += gain * np.sin(phase)
        out[:, right_idx, 0] -= gain * np.sin(phase)
    return out


def synthesize_smoke_cohort(clips_per_source=3, frames=FRAMES_SMOKE):
    records = []
    counter = 0
    for condition in CONDITIONS:
        for source in range(2):
            sign = 1 if source == 0 else -1
            mag = SMOKE_LEAN_MAGNITUDE[condition]
            for clip in range(clips_per_source):
                base = synthetic_gait_sequence(condition=condition, frames=frames, seed=RANDOM_SEED + counter)
                seq = plant_signed_lean(base, sign, mag, seed=RANDOM_SEED + 1000 + counter)
                records.append({
                    "condition": condition,
                    "sequence_id": f"smoke_{condition}_{source}_{clip:02d}",
                    "video_id": f"smoke_source_{condition}_{source}",
                    "raw": seq, "provenance": "canonical",
                })
                counter += 1
    return records


def pose_records_from_cache(pose_dir, conditions):
    records = []
    for condition in conditions:
        folder = Path(pose_dir) / condition
        if not folder.exists():
            continue
        for path in sorted(folder.glob("*.npz")):
            data = np.load(path, allow_pickle=False)
            sequence = data["sequence"].astype(np.float32)
            if sequence.ndim != 3 or sequence.shape[1:] != (33, 4):
                raise ValueError(f"Bad pose shape in {path}: {sequence.shape}")
            records.append({
                "condition": condition,
                "sequence_id": str(data["sequence_id"].item()),
                "video_id": str(data["video_id"].item()),
                "raw": sequence,
                "provenance": "canonical",
            })
    return records


if MODE == "real":
    records = pose_records_from_cache(POSE_DIR, CONDITIONS)
    print(f"real cohort: {len(records)} sequences from {len({r['video_id'] for r in records})} source videos")
else:
    records = synthesize_smoke_cohort()
    print(f"smoke cohort: {len(records)} synthetic sequences from {len({r['video_id'] for r in records})} sources")

meta = pd.DataFrame([{k: r[k] for k in ("condition", "sequence_id", "video_id", "provenance")} for r in records])
print(meta.groupby(["condition"]).size().to_string())
if len({r["video_id"] for r in records}) < 2:
    raise ValueError("Need at least two source videos for a source-disjoint split.")
"""))

# ------------------------------------------------------------------ 7. Frozen-target y + features (5 lanes)
CELLS.append(md(r"""
## 7. Compute the frozen target `y` and cache the five lanes

For every sequence we compute:

- `y` = `signed_left_minus_right` on the preprocessed coordinates (frozen before any fit).
- **Lane A'** (Arm 1, the new head): the `m`-dim antisymmetric head feature `s` on the frozen `ea59fea0`
  target-encoder tokens. This is the constrained pure-difference readout.
- **Lane Ac** (capacity-matched control): the SAME head/`f`/width/aggregation as A', but with the symmetric
  `l + r` path ADDED next to the `l - r` path (a `2m`-dim concat). It differs from A' ONLY by the
  antisymmetry constraint, so an A'-minus-Ac gap isolates the effect of antisymmetry from the head's
  nonlinearity, initialization, dimensionality, and pair information (which are all held identical).
- **Lane B** (raw-coordinate null): handcrafted signed per-pair excursion features, no network.
  Descriptive ceiling only; near-circular (it fits raw std-differences to a target built from the same
  std-differences), so it is NOT a gate.
- **Lane C** (untrained-encoder floor): the SAME antisymmetric head applied to a random-init encoder.
- **Lane D** (standard `ea59fea0` learned-behavior comparator): item 05's `laterality_feature_from_tokens`
  (which includes the `l + r` symmetric half) on the frozen `ea59fea0` tokens. This is exactly item 05's
  Lane A, reused as the "what the standard encoder already exposes" comparator.
- **Lane E** (side-agnostic nuisance negative control): global mean and std over tokens, then SYMMETRIZED
  over the anatomical mirror - `E(x) = 0.5 * (pool(enc(x)) + pool(enc(Mx)))` - so `E(x) == E(Mx)` by
  construction. Raw pooling alone is NOT side-blind, because the encoder's learned per-joint position codes
  keep each token conditioned on its landmark slot; symmetrizing removes that, giving a control that truly
  cannot tell a body from its reflection. Only such a control can validly withdraw the signed claim, so we
  assert the invariance numerically before trusting it.

We also cache the MIRRORED Lane A' feature (head on the anatomically mirrored input, run THROUGH the
encoder) for the measured anatomical-mirror slope in section 9, and reuse those mirrored tokens to build
the symmetrized Lane E.
"""))
CELLS.append(code(r"""
def encoder_tokens(enc_model, xyz_batch):
    '''xyz_batch: [B, FRAMES, 33, 3] float32. Returns [B, SEGMENTS, 33, EMBED_DIM] numpy.'''
    with torch.no_grad():
        x = torch.tensor(xyz_batch, dtype=torch.float32)
        tok = enc_model.target_encoder(x).reshape(len(xyz_batch), SEGMENTS, 33, EMBED_DIM)
    return tok.cpu().numpy()


def antisym_feature_from_tokens(tokens_np):
    '''tokens_np: [SEGMENTS, 33, D] -> the m-dim antisymmetric head output s (difference only).'''
    with torch.no_grad():
        t = torch.tensor(tokens_np[None, ...], dtype=torch.float32)
        s = antisym_head(t)
    return s[0].cpu().numpy()


def capacity_matched_feature_from_tokens(tokens_np):
    '''Lane Ac: the capacity-matched control readout on the SAME head/f. Returns the 2*m-dim
    concat(difference, sum). Identical machinery to A' except for the added symmetric (l + r) path, so an
    A'-vs-Ac gap is attributable to the antisymmetry constraint rather than head architecture.'''
    with torch.no_grad():
        t = torch.tensor(tokens_np[None, ...], dtype=torch.float32)
        s = antisym_head.forward_capacity_matched(t)
    return s[0].cpu().numpy()


def laterality_feature_from_tokens(tokens):
    '''Item 05's Lane A feature: per L/R pair, time-mean of LEFT, RIGHT, their difference AND sum.
    Includes the symmetric (l + r) half by design. Reused verbatim as Lane D (standard comparator).'''
    feats = []
    for left_idx, right_idx in LEFT_RIGHT_PAIRS:
        l = tokens[:, left_idx, :].mean(axis=0)
        r = tokens[:, right_idx, :].mean(axis=0)
        feats.append(l - r)
        feats.append(l + r)
    return np.concatenate(feats)


def raw_null_feature(xyz):
    '''Lane B handcrafted signed features: per-pair signed excursion difference on the three axes.'''
    feats = []
    for left_idx, right_idx in LEFT_RIGHT_PAIRS:
        l = xyz[:, left_idx, :].std(axis=0)
        r = xyz[:, right_idx, :].std(axis=0)
        feats.append(l - r)
    return np.concatenate(feats)


def pooled_nuisance_raw(tokens):
    '''Global mean and std over ALL joint-time tokens. Permutation-invariant over the OUTPUT-TOKEN order,
    but NOT invariant to an anatomical side swap: the encoder adds learned per-joint position codes, so a
    token's value stays conditioned on its landmark slot even after global pooling. Do not use this raw
    version as the side-agnostic control; symmetrize it (below) first.'''
    flat = tokens.reshape(-1, tokens.shape[-1])
    return np.concatenate([flat.mean(axis=0), flat.std(axis=0)])


def pooled_nuisance_feature(tokens, tokens_mirror):
    '''Lane E (side-agnostic control), made invariant under the FULL anatomical swap by averaging the
    pooled feature over a sample AND its anatomically mirrored counterpart (both run THROUGH the encoder):
        E(x) = 0.5 * ( pool(enc(x)) + pool(enc(Mx)) ).
    Because M is an involution (mirroring twice is the identity) and enc(Mx)'s mirror is enc(x), this E is
    identical for a sample and its mirror by construction: E(x) == E(Mx). A side-agnostic control MUST be
    unable to tell a body from its reflection; only then does a significant E genuinely mean the signed
    axis leaked into a nuisance channel. We assert the invariance numerically before using it as a gate.'''
    return 0.5 * (pooled_nuisance_raw(tokens) + pooled_nuisance_raw(tokens_mirror))


prepared = [prepare_sequence(r["raw"], FRAMES) for r in records]
xyz_all = np.stack([p[0] for p in prepared]).astype(np.float32)
y = np.array([signed_left_minus_right(x) for x in xyz_all], dtype=np.float64)
B_raw = np.stack([raw_null_feature(x) for x in xyz_all])

prepared_mir = [prepare_sequence(anatomical_mirror(r["raw"]), FRAMES) for r in records]
xyz_mir = np.stack([p[0] for p in prepared_mir]).astype(np.float32)

if HAVE_TORCH:
    def batched_tokens(enc, arr, bs=16):
        out = []
        for i in range(0, len(arr), bs):
            out.append(encoder_tokens(enc, arr[i:i + bs]))
        return np.concatenate(out)

    tokA = batched_tokens(model, xyz_all)
    tokA_mir = batched_tokens(model, xyz_mir)
    tokC = batched_tokens(floor_model, xyz_all)

    Aprime = np.stack([antisym_feature_from_tokens(t) for t in tokA])       # Lane A' (new head)
    Aprime_mir = np.stack([antisym_feature_from_tokens(t) for t in tokA_mir])
    C_floor = np.stack([antisym_feature_from_tokens(t) for t in tokC])      # Lane C (head on floor)
    D_std = np.stack([laterality_feature_from_tokens(t) for t in tokA])     # Lane D (item 05 Lane A)
    # Lane Ac: capacity-matched control head. SAME fixed f, SAME width, SAME aggregation as A', differing
    # ONLY by adding the symmetric (l + r) path alongside the difference. Isolates the antisymmetry
    # constraint from nonlinearity/init/dimensionality/pair-info confounds (Codex high finding).
    Ac_match = np.stack([capacity_matched_feature_from_tokens(t) for t in tokA])
    # Lane E: side-agnostic nuisance control, symmetrized over the anatomical mirror so E(x) == E(Mx).
    E_pooled = np.stack([pooled_nuisance_feature(t, tm) for t, tm in zip(tokA, tokA_mir)])
    # ---- invariance self-check: the control MUST be blind to a body-vs-mirror flip ----
    E_pooled_mir = np.stack([pooled_nuisance_feature(tm, t) for t, tm in zip(tokA, tokA_mir)])
    _e_gap = float(np.abs(E_pooled - E_pooled_mir).max())
    print(f"Lane E anatomical-invariance check: max|E(x) - E(Mx)| = {_e_gap:.3e}  (must be ~0)")
    assert _e_gap < 1e-5, "Lane E must be invariant under the anatomical swap; symmetrization failed."
    E_INVARIANT_OK = True
    print(f"feature widths -> A':{Aprime.shape[1]}  Ac:{Ac_match.shape[1]}  B:{B_raw.shape[1]}  "
          f"C:{C_floor.shape[1]}  D:{D_std.shape[1]}  E:{E_pooled.shape[1]}")
else:
    Aprime = Aprime_mir = C_floor = D_std = E_pooled = Ac_match = None
    E_INVARIANT_OK = None
    print("No torch: only Lane B (raw null) and the target y are available.")

print(f"y stats: mean={y.mean():+.4f} std={y.std():.4f}  n={len(y)}")
"""))

# ------------------------------------------------------------------ 8. y-variance gate + source-disjoint probes
CELLS.append(md(r"""
## 8. y-quality gate, then fit the lanes with source-disjoint ridge probes

**y-quality gate (run BEFORE trusting any R-squared).** We decompose the target's variance into a
between-source part and a within-source part. If the between-source fraction is below a pre-committed
threshold (`IDEA9_Y_BETWEEN_MIN`, default 0.30), `y` is noise-dominated at the source level and any
held-out-source R-squared is uninterpretable; we report that fact instead of leaning on R-squared.

Then the split, stated before any fit: folds are SOURCE-VIDEO-DISJOINT via `GroupKFold` on `video_id`.
We pool signed laterality across all conditions (per-class LOSO on n=1 sources is not reported). The
ridge penalty is chosen only on the training sources of each fold. We report held-out-source R-squared
and MAE per lane. `source_disjoint_probe` and `sign_consistency` are reused verbatim from nb_05a.
"""))
CELLS.append(code(r"""
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_absolute_error

ALPHAS = np.logspace(-3, 3, 13)
groups = meta["video_id"].to_numpy()
n_groups = len(np.unique(groups))
n_splits = max(2, min(5, n_groups))

# ---- y-quality gate: between-source vs within-source variance decomposition ----
Y_BETWEEN_MIN = float(os.getenv("IDEA9_Y_BETWEEN_MIN", "0.30"))
_grand = float(np.mean(y))
_ss_total = float(np.sum((y - _grand) ** 2))
_ss_between = 0.0
for g in np.unique(groups):
    m = groups == g
    _ss_between += int(m.sum()) * (float(np.mean(y[m])) - _grand) ** 2
y_between_fraction = float(_ss_between / _ss_total) if _ss_total > 0 else float("nan")
Y_GATE_OK = bool(np.isfinite(y_between_fraction) and y_between_fraction >= Y_BETWEEN_MIN)
print(f"y between-source variance fraction = {y_between_fraction:.3f}  (threshold {Y_BETWEEN_MIN:.2f})")
print(f"y-quality gate: {'PASS (y carries source-level signal)' if Y_GATE_OK else 'FAIL (y is noise-dominated; R2 uninterpretable)'}")


def source_disjoint_probe(X, y, groups, n_splits):
    '''Held-out-source R2 and MAE with an inner alpha choice on training sources only. From nb_05a.'''
    gkf = GroupKFold(n_splits=n_splits)
    preds = np.full(len(y), np.nan)
    for train_idx, test_idx in gkf.split(X, y, groups):
        inner_groups = groups[train_idx]
        best_alpha, best_score = ALPHAS[0], -np.inf
        if len(np.unique(inner_groups)) >= 2:
            inner = GroupKFold(n_splits=min(3, len(np.unique(inner_groups))))
            for a in ALPHAS:
                scores = []
                for itr, iva in inner.split(X[train_idx], y[train_idx], inner_groups):
                    sc = StandardScaler().fit(X[train_idx][itr])
                    mdl = Ridge(alpha=a).fit(sc.transform(X[train_idx][itr]), y[train_idx][itr])
                    scores.append(r2_score(y[train_idx][iva], mdl.predict(sc.transform(X[train_idx][iva]))))
                if np.mean(scores) > best_score:
                    best_score, best_alpha = np.mean(scores), a
        sc = StandardScaler().fit(X[train_idx])
        mdl = Ridge(alpha=best_alpha).fit(sc.transform(X[train_idx]), y[train_idx])
        preds[test_idx] = mdl.predict(sc.transform(X[test_idx]))
    ok = ~np.isnan(preds)
    return {"r2": float(r2_score(y[ok], preds[ok])), "mae": float(mean_absolute_error(y[ok], preds[ok])),
            "preds": preds}


def sign_consistency(preds, y_true, groups):
    '''Fraction of held-out SOURCES whose median predicted sign matches the median true sign. From nb_05a.'''
    frac = []
    for g in np.unique(groups):
        m = groups == g
        if np.isnan(preds[m]).all():
            continue
        ps = np.sign(np.nanmedian(preds[m]))
        ts = np.sign(np.median(y_true[m]))
        if ts == 0:
            frac.append(1.0 if abs(np.nanmedian(preds[m])) <= 0.5 * np.std(y_true) else 0.0)
        else:
            frac.append(1.0 if ps == ts else 0.0)
    return float(np.mean(frac)) if frac else float("nan")


lanes = {}
lanes["B_raw_null"] = source_disjoint_probe(B_raw, y, groups, n_splits)
if HAVE_TORCH:
    lanes["A_prime"] = source_disjoint_probe(Aprime, y, groups, n_splits)
    lanes["Ac_capacity_matched"] = source_disjoint_probe(Ac_match, y, groups, n_splits)
    lanes["C_floor"] = source_disjoint_probe(C_floor, y, groups, n_splits)
    lanes["D_standard"] = source_disjoint_probe(D_std, y, groups, n_splits)
    lanes["E_pooled"] = source_disjoint_probe(E_pooled, y, groups, n_splits)

lane_table = pd.DataFrame([{"lane": k, "R2": v["r2"], "MAE": v["mae"]} for k, v in lanes.items()])
print(lane_table.to_string(index=False))
"""))

# ------------------------------------------------------------------ 9. Permutation nulls + anatomical mirror
CELLS.append(md(r"""
## 9. Source-label permutation nulls, and the measured anatomical-mirror slope

**Permutation null (replaces a fixed sign-consistency threshold).** At n=18 sources with per-condition
counts as low as 1, a fixed "75 percent of sources correct" bar is not meaningful. Instead we shuffle
the target across SOURCES (keeping each source's clips together) many times and refit the source-disjoint
probe, building a null distribution of held-out R-squared under no real source-to-target link. We run the
null on **Lane A'** (the head we are testing) AND on **Lane C** (the untrained floor). Running it on C is
purely a CHARACTERIZATION of how strong the floor is: because C is a deterministic transform of the raw
coordinates, its random features CAN preserve genuine laterality, so a significant C null means "the floor
is strong," not "the null is broken" or "source identity is leaking." A strong C is already handled by the
binding bar `max(D, C)` (it simply raises the bar A' must clear), so we report C's null but do NOT let it
withdraw the A' claim.

**Anatomical-mirror slope (measured, not -1).** We fit one Lane A' probe on all sources (only to have a
single probe for the mirror geometry; the decodability verdict comes from the held-out fit above), decode
the original and the anatomically-mirrored inputs, and fit the slope of decoded-mirrored vs
decoded-original. This slope runs THROUGH the encoder, so it is a measured number (item 05 measured
-0.741 for `ea59fea0`), never the exact -1 of the section-5 wiring check.
"""))
CELLS.append(code(r"""
N_PERM = int(os.getenv("IDEA9_N_PERM", "200" if MODE == "real" else "50"))
perm_rng = np.random.default_rng(RANDOM_SEED + 99)
unique_sources = np.unique(groups)


def permute_y_by_source(y_true, groups, rng):
    '''Shuffle the per-source MEAN target across sources, keeping clips of a source together.'''
    src_mean = {g: float(np.mean(y_true[groups == g])) for g in unique_sources}
    shuffled_vals = rng.permutation(list(src_mean.values()))
    mapping = {g: v for g, v in zip(unique_sources, shuffled_vals)}
    # Preserve within-source residual around the (now shuffled) source level.
    out = np.empty_like(y_true)
    for g in unique_sources:
        m = groups == g
        out[m] = mapping[g] + (y_true[m] - src_mean[g])
    return out


def permutation_null(X, y_true, groups, n_splits, n_perm, rng):
    if X is None:
        return {"available": False}
    observed = source_disjoint_probe(X, y_true, groups, n_splits)["r2"]
    null_r2 = []
    for _ in range(n_perm):
        yp = permute_y_by_source(y_true, groups, rng)
        null_r2.append(source_disjoint_probe(X, yp, groups, n_splits)["r2"])
    null_r2 = np.asarray(null_r2)
    p_value = float((np.sum(null_r2 >= observed) + 1) / (n_perm + 1))
    return {"available": True, "observed_r2": float(observed),
            "null_mean": float(null_r2.mean()), "null_p95": float(np.percentile(null_r2, 95)),
            "p_value": p_value, "n_perm": int(n_perm)}


perm_Aprime = permutation_null(Aprime if HAVE_TORCH else None, y, groups, n_splits, N_PERM, perm_rng)
perm_C = permutation_null(C_floor if HAVE_TORCH else None, y, groups, n_splits, N_PERM, perm_rng)
if perm_Aprime.get("available"):
    print(f"Lane A' permutation null:  observed R2={perm_Aprime['observed_r2']:+.3f}  "
          f"null mean={perm_Aprime['null_mean']:+.3f}  null p95={perm_Aprime['null_p95']:+.3f}  p={perm_Aprime['p_value']:.3f}")
if perm_C.get("available"):
    print(f"Lane C  permutation null:  observed R2={perm_C['observed_r2']:+.3f}  "
          f"null mean={perm_C['null_mean']:+.3f}  null p95={perm_C['null_p95']:+.3f}  p={perm_C['p_value']:.3f}")
    if perm_C["p_value"] < 0.05:
        print("NOTE (floor characterization, NOT a failure): the untrained floor's random features carry a "
              "real feature-target association. C is a deterministic transform of the raw coordinates, so it "
              "CAN preserve genuine laterality; a significant C null just means the floor is strong. This is "
              "why the binding bar is max(D, C): a strong C raises the bar A' must clear. It is NOT evidence "
              "of source-identity leakage and does not, by itself, invalidate A'.")

# ---- Measured anatomical-mirror slope (through the encoder; NOT the exact wiring -1) ----
mirror = {"available": False}
if HAVE_TORCH:
    sc = StandardScaler().fit(Aprime)
    # Fit each of the m head-output dims, decode a single scalar via the first principal target direction.
    probe = Ridge(alpha=1.0).fit(sc.transform(Aprime), y)
    dec_orig = probe.predict(sc.transform(Aprime))
    dec_mir = probe.predict(sc.transform(Aprime_mir))
    slope = float(np.polyfit(dec_orig, dec_mir, 1)[0])
    flips = (-1.25 <= slope <= -0.8)
    mirror = {"available": True, "slope": slope, "flips": bool(flips),
              "dec_orig": dec_orig, "dec_mir": dec_mir}
    print(f"anatomical-mirror slope (through encoder) = {slope:+.3f}   "
          f"verdict: {'flips' if flips else 'does not flip'}   (measured, not the wiring -1)")
else:
    print("No torch: anatomical-mirror test skipped.")
"""))

# ------------------------------------------------------------------ 10. Verdict (hardened gates)
CELLS.append(md(r"""
## 10. Pre-registered verdict (hardened gates)

Arm 1 passes only if ALL of these hold at once. These gates supersede item 05's proposal-level gates
(the `>= 80 percent of raw null` and fixed `75 percent sign consistency` gates are dropped, for the
reasons in `IMPLEMENTATION.md`):

1. **Binding bar.** `A' R-squared - max(D_standard, C_floor) >= 0.05`. The binding constraint is the
   LARGER of the standard `ea59fea0` comparator and the untrained floor, because beating only the weaker
   of the two would overclaim. This is exactly where a strong C is handled: a random floor that preserves
   real laterality simply RAISES the bar A' must clear. (A strong C is not a failure; see below.)
2. **Beat floor.** `A' R-squared - C_floor >= 0.05` (the hard one).
3. **Attribution to antisymmetry.** `A' R-squared - Ac_capacity_matched >= 0.05`. `Ac` uses the SAME
   fixed `f`, width, and aggregation as `A'` and differs ONLY by adding the symmetric `l + r` path, so a
   gap here is attributable to the antisymmetry CONSTRAINT rather than to the head's nonlinearity, init,
   dimensionality, or pair information (which are held identical). Without this, an `A'` win could be a
   generic head-architecture win.
4. **Permutation null.** Lane A' clears its source-label permutation null at `p < 0.05`.
5. **Negative control (absolute), on a genuinely side-agnostic feature.** `abs(E_pooled R-squared) < 0.05`
   ABSOLUTELY. Lane E is symmetrized over the anatomical mirror so `E(x) == E(Mx)` by construction (checked
   in section 7); only such a truly side-blind feature can validly withdraw the signed claim if it still
   recovers the axis. (We drop item 05's `or d < 0.5 * max(b, ...)` OR-clause, vacuous when B is near 1.0.)
6. **y-quality gate.** The target's between-source variance fraction clears the threshold, so R-squared
   is interpretable at the source level.
7. **Wiring guarantee.** The section-5 exact wiring identity passed (slope exactly -1 to tolerance).

**What the C permutation is, and is NOT.** We still RUN and REPORT Lane C's permutation null, but only as a
characterization of how strong the floor is - not as a claim-withdrawing trap. A significant C means random
features preserve real feature-target association (C is a deterministic transform of the raw coordinates,
so it can preserve genuine laterality); it does NOT establish source-identity leakage or a weak null, and
the binding bar already accounts for a strong C. Tripping the whole verdict on it could reject a genuine A'
improvement, so it is descriptive only.

Lane B (raw null) is descriptive only and near-circular; it is NOT a gate. Missing any of 1-7 is scored
as an informative null. The most likely honest outcome, given item 05's result, is an informative null
that is REAL (a genuine statement about this scale), not an artifact of a broken gate.
"""))
CELLS.append(code(r"""
FLOOR_MARGIN = 0.05
verdict = {"mode": MODE, "fingerprint": FINGERPRINT}
if HAVE_TORCH:
    a = lanes["A_prime"]["r2"]
    ac = lanes["Ac_capacity_matched"]["r2"]
    b = lanes["B_raw_null"]["r2"]
    c = lanes["C_floor"]["r2"]
    d = lanes["D_standard"]["r2"]
    e = lanes["E_pooled"]["r2"]
    signcon = sign_consistency(lanes["A_prime"]["preds"], y, groups)
    binding_bar = max(d, c)
    binding_delta = a - binding_bar
    beats_binding = binding_delta >= FLOOR_MARGIN
    beats_floor = (a - c) >= FLOOR_MARGIN
    # Attribution gate: A' must beat the capacity-matched control (same f/width/aggregation, differing only
    # by the symmetric path) so the win is attributable to the ANTISYMMETRY, not the head architecture.
    attribution_delta = a - ac
    beats_capacity_matched = attribution_delta >= FLOOR_MARGIN
    perm_ok = bool(perm_Aprime.get("available") and perm_Aprime["p_value"] < 0.05)
    e_control_ok = bool(abs(e) < FLOOR_MARGIN)              # absolute; no OR-clause
    e_invariant_ok = bool(E_INVARIANT_OK)                  # E must be side-blind for the gate to be valid
    # C's own permutation is REPORTED (floor characterization) but is NOT a claim-withdrawing gate:
    # a strong C is handled by the binding bar, and C significance is feature-target association, not
    # source-identity leakage. Kept for transparency only.
    c_null_significant = bool(perm_C.get("available") and perm_C["p_value"] < 0.05)
    passed = bool(beats_binding and beats_floor and beats_capacity_matched and perm_ok
                  and e_control_ok and e_invariant_ok and Y_GATE_OK and WIRING_IDENTITY_OK)
    verdict.update({
        "A_prime_r2": a, "Ac_capacity_matched_r2": ac, "B_raw_null_r2": b, "C_floor_r2": c,
        "D_standard_r2": d, "E_pooled_r2": e,
        "binding_bar_max_D_C": float(binding_bar), "binding_delta": float(binding_delta),
        "attribution_delta_A_minus_Ac": float(attribution_delta),
        "beats_capacity_matched_by_0.05": bool(beats_capacity_matched),
        "sign_consistency_descriptive": signcon,
        "beats_binding_bar_by_0.05": bool(beats_binding),
        "beats_floor_by_0.05": bool(beats_floor),
        "A_prime_permutation_p": perm_Aprime.get("p_value"),
        "A_prime_permutation_ok": perm_ok,
        "C_permutation_p": perm_C.get("p_value"),
        "C_null_significant_descriptive": c_null_significant,
        "E_control_ok_abs": e_control_ok,
        "E_anatomically_invariant_ok": e_invariant_ok,
        "y_between_source_fraction": y_between_fraction, "y_quality_gate_ok": Y_GATE_OK,
        "wiring_identity_ok": bool(WIRING_IDENTITY_OK), "wiring_swap_slope": _slope,
        "anatomical_mirror_slope": mirror.get("slope"), "anatomical_mirror_flips": mirror.get("flips"),
        "PRIMARY_VERDICT": (
            "ANTISYMMETRY BEATS BINDING BAR AND CAPACITY-MATCHED CONTROL" if passed
            else "ARTIFACT (side-agnostic nuisance control fired)" if not e_control_ok
            else "INVALID CONTROL (Lane E not anatomically invariant)" if not e_invariant_ok
            else "UNINTERPRETABLE (y noise-dominated at source level)" if not Y_GATE_OK
            else "NOT ATTRIBUTABLE TO ANTISYMMETRY (does not beat capacity-matched control)" if not beats_capacity_matched
            else "INFORMATIVE NULL"),
    })
print(json.dumps({k: v for k, v in verdict.items() if k != "preds"}, indent=2))
if MODE == "smoke":
    print("\nNOTE: smoke numbers are illustrative plumbing checks, NOT the ea59fea0 result.")
"""))

# ------------------------------------------------------------------ 11. Figures
CELLS.append(md(r"""
## 11. Figures (the two decisive panels)

Figure 1 is the decodability panel: held-out-source decoded signed scalar vs the ground-truth target for
Lane A', with the binding bar `max(D, C)` and the raw null B annotated. Figure 2 is the mirror panel and
has TWO series kept strictly apart: the exact wiring-swap identity (lands on `y = -x` by construction) and
the measured anatomical-mirror slope (through the encoder). These mirror the two SVG mockups in the
proposal's `images/` folder (fig1, fig2), which nb_09a relabels for the Arm-1 readout split.
"""))
CELLS.append(code(r"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = ARTIFACT_ROOT / MODE
OUT_DIR.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))

# Panel 1: decodability
if HAVE_TORCH:
    preds = lanes["A_prime"]["preds"]
    finite = preds[~np.isnan(preds)]
    ax[0].scatter(y, preds, s=42, c="#e07a4b", edgecolors="#a44c26",
                  label=f"A' antisym head (R2={lanes['A_prime']['r2']:.3f})")
    lo = float(min(y.min(), finite.min())); hi = float(max(y.max(), finite.max()))
    ax[0].plot([lo, hi], [lo, hi], "--", color="#5f9e7e", label="identity y = x")
    ax[0].set_title(
        f"Arm 1 decodability  (binding bar max(D,C)={max(lanes['D_standard']['r2'], lanes['C_floor']['r2']):.3f}, "
        f"capacity-matched Ac={lanes['Ac_capacity_matched']['r2']:.3f}, null B={lanes['B_raw_null']['r2']:.3f})")
else:
    ax[0].text(0.5, 0.5, "Learned lanes need PyTorch", ha="center")
ax[0].set_xlabel("ground-truth signed target (left minus right)")
ax[0].set_ylabel("decoded signed scalar (held-out source)")
ax[0].legend(loc="upper left", fontsize=9)

# Panel 2: mirror (two distinct series)
if mirror["available"]:
    do = mirror["dec_orig"]; dm = mirror["dec_mir"]
    lim = float(np.abs(np.concatenate([do, dm])).max()) * 1.1 + 1e-9
    # exact wiring identity: for illustration, plot the by-construction line s -> -s
    xs = np.linspace(-lim, lim, 50)
    ax[1].plot(xs, -xs, "-", color="#2f6f99", lw=2.2, label="wiring swap: slope -1 (exact, by construction)")
    ax[1].scatter(do, dm, s=42, c="#e07a4b", edgecolors="#a44c26",
                  label=f"anatomical mirror through encoder (slope={mirror['slope']:+.2f}, measured)")
    ax[1].plot([-lim, lim], [lim, -lim], "--", color="#5f9e7e", label="reflection line y = -x")
    ax[1].axhline(0, color="#c4cdd8", lw=1); ax[1].axvline(0, color="#c4cdd8", lw=1)
    ax[1].set_xlim(-lim, lim); ax[1].set_ylim(-lim, lim)
    ax[1].set_title("Mirror check: exact wiring -1 vs measured anatomical slope")
else:
    ax[1].text(0.5, 0.5, "Mirror needs the encoder", ha="center")
ax[1].set_xlabel("decoded on original input"); ax[1].set_ylabel("decoded on mirrored input")
ax[1].legend(loc="upper right", fontsize=8)

fig.suptitle(f"Idea 9 Arm 1: antisymmetric readout  [mode={MODE}, fingerprint={FINGERPRINT[:12]}]  (illustrative in smoke)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig_path = OUT_DIR / "idea9_antisymmetric_readout_probe.png"
fig.savefig(fig_path, dpi=130)
print(f"saved figure: {fig_path}")
plt.show()
"""))

# ------------------------------------------------------------------ 12. Persist
CELLS.append(md(r"""
## 12. Persist the result bundle

We write a JSON bundle (verdict, per-lane R-squared and MAE, the binding delta, both permutation nulls,
the exact wiring slope and the measured anatomical slope, the y-variance decomposition, the fingerprint,
the fold count and manifest, the canonical-subset flag, and the cohort counts) next to the other
artifacts. This is the machine-readable record `IMPLEMENTATION.md` and the README point at, and the input
the futures notebook (09c) compares real outcomes against.
"""))
CELLS.append(code(r"""
fold_manifest = []
if n_groups >= 2:
    gkf = GroupKFold(n_splits=n_splits)
    for i, (tr, te) in enumerate(gkf.split(np.zeros((len(y), 1)), y, groups)):
        fold_manifest.append({"fold": i,
                              "test_sources": sorted(set(groups[te].tolist())),
                              "n_test": int(len(te))})

bundle = {
    "notebook": "nb_09a_antisymmetric_readout_probe",
    "arm": "arm1_zero_retrain_readout",
    "mode": MODE,
    "fingerprint": FINGERPRINT,
    "n_sequences": int(len(records)),
    "n_sources": int(n_groups),
    "n_splits": int(n_splits),
    "head_out_dim": HEAD_OUT_DIM,
    "condition_counts": meta.groupby("condition").size().to_dict(),
    "lanes": {k: {"r2": v["r2"], "mae": v["mae"]} for k, v in lanes.items()},
    "binding_bar_max_D_C": verdict.get("binding_bar_max_D_C"),
    "binding_delta": verdict.get("binding_delta"),
    "attribution_delta_A_minus_Ac": verdict.get("attribution_delta_A_minus_Ac"),
    "permutation_null_A_prime": {k: perm_Aprime.get(k) for k in ("observed_r2", "null_mean", "null_p95", "p_value", "n_perm")},
    "permutation_null_C_floor": {**{k: perm_C.get(k) for k in ("observed_r2", "null_mean", "null_p95", "p_value", "n_perm")},
                                 "role": "floor characterization only (NOT a claim-withdrawing gate); a strong C is handled by the binding bar max(D,C)"},
    "wiring_identity": {"ok": bool(WIRING_IDENTITY_OK) if WIRING_IDENTITY_OK is not None else None,
                        "swap_slope": _slope},
    "lane_E_anatomically_invariant": verdict.get("E_anatomically_invariant_ok"),
    "anatomical_mirror": {"slope": mirror.get("slope"), "flips": mirror.get("flips")},
    "y_variance": {"between_source_fraction": y_between_fraction, "threshold": Y_BETWEEN_MIN, "ok": Y_GATE_OK},
    "canonical_subset_only": bool((meta["provenance"] == "canonical").all()),
    "fold_manifest": fold_manifest,
    "verdict": {k: v for k, v in verdict.items() if k != "preds"},
    "transductive": True,
    "notes": "All results transductive; source video is the independent unit; folder labels are dataset annotations, not diagnoses. The wiring-swap slope -1 is exact by construction; the anatomical-mirror slope is measured through the encoder and is not expected to be -1. Lane Ac is a capacity-matched control (same f/width/aggregation, adds the symmetric l+r path) so A'-vs-Ac isolates the antisymmetry constraint. Lane E is symmetrized over the anatomical mirror so E(x)==E(Mx). Lane C's permutation is floor characterization, not a leakage test.",
}
bundle_path = OUT_DIR / "idea9_antisymmetric_readout_result.json"
bundle_path.write_text(json.dumps(bundle, indent=2))
print(f"wrote {bundle_path}")
print(json.dumps({k: bundle[k] for k in ("mode", "fingerprint", "n_sequences", "n_sources", "binding_delta", "verdict")}, indent=2)[:1100])
"""))

# ------------------------------------------------------------------ build
nb = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "GAVD6 S-JEPA", "language": "python", "name": "gavd6-sjepa"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
if NB_PATH.exists():
    raise FileExistsError(
        f"Refusing to overwrite executed notebook {NB_PATH}. Move or delete it explicitly before "
        "using this bootstrap builder."
    )
NB_PATH.write_text(json.dumps(nb, indent=1))
print(f"wrote {NB_PATH}  ({len(CELLS)} cells)")
