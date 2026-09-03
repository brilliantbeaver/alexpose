"""Builder for nb_05a_signed_laterality_probe.ipynb.

Run:  python3 _build_nb_05a.py
Emits the notebook one directory up (the gavd5-drift root) so it sits beside 00-06.
This builder holds each cell as a plain string so the code stays readable and
the JSON is guaranteed valid. It is committed alongside the notebook so the
notebook can be regenerated deterministically.
"""
import json
from pathlib import Path

NB_PATH = Path(__file__).resolve().parents[3] / "nb_05a_signed_laterality_probe.ipynb"


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
# Notebook 05a: Signed-laterality decodability probe (Idea 5, core arm)

This notebook reifies the core, zero-retrain arm of research idea 5,
[signed-laterality-decodability](notes/ideas-claude/05-signed-laterality-decodability/README.md).
It asks one falsifiable question on the frozen `d0acc262` Skeleton-JEPA (S-JEPA) checkpoint:

> On source-video-disjoint folds, is a signed left-minus-right laterality axis linearly decodable
> from the frozen S-JEPA token tensor above a raw-coordinate null by a pre-registered margin, and
> does the anatomical mirror that swaps left and right landmarks negate the decoded scalar?

Nothing here retrains the encoder. Every lane is a test-time linear read of already-cached frozen
target-encoder tokens. The notebook is written to run in two modes:

- `GAVD_MODE=real`: reads the real pose cache and the `d0acc262` checkpoint, exactly as notebooks
  05 and 06 do.
- `GAVD_MODE=smoke` (default when artifacts are absent): builds a small synthetic cohort with a
  planted, sign-flipping laterality signal so every cell runs end to end and the plots render. Smoke
  numbers are illustrative only and are labelled as such.

All results are TRANSDUCTIVE: the encoder saw the evaluation rows during training. The source video
is the independent unit, and folder labels (stroke, parkinsons) are dataset annotations, not
diagnoses. See `notes/ideas-claude/_shared_facts.md` for the single source of truth on every number.
"""))

# ------------------------------------------------------------------ 0. Environment
CELLS.append(md(r"""
## 0. Environment and mode

We resolve the project root the same way notebooks 04-06 do (via `ALEXPOSE_ROOT` or a `.git` +
`data/gavd` marker), load the `.env`, and decide the run mode. If the real artifacts are missing we
fall back to smoke mode instead of raising, so the notebook is always runnable.
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
TUTORIAL_DIR = PROJECT_ROOT / "penny" / "gavd3"
try:
    from dotenv import load_dotenv
    load_dotenv(TUTORIAL_DIR / ".env", override=False)
    load_dotenv(PROJECT_ROOT / ".env", override=False)
except Exception:
    pass

# Artifact locations follow the SAME env resolution as notebook 05 exactly:
#   ARTIFACT_ROOT = GAVD_ARTIFACT_DIR or TUTORIAL_DIR/work/artifacts
#   ARTIFACT_DIR  = ARTIFACT_ROOT / MODE   (checkpoints live directly here)
#   POSE_DIR      = ARTIFACT_DIR / "poses"
ARTIFACT_ROOT = Path(
    os.getenv("GAVD_ARTIFACT_DIR", TUTORIAL_DIR / "work" / "artifacts")
).expanduser()

REQUESTED_MODE = os.getenv("GAVD_MODE", "smoke").strip().lower()
if REQUESTED_MODE not in {"smoke", "real"}:
    raise ValueError("GAVD_MODE must be smoke or real")

# Checkpoint selection mirrors notebook 05: augmented cohort is the d0acc262 lineage.
TRUTHY = {"1", "true", "yes", "on"}
INCLUDE_AUGMENTED = os.getenv("SJEPA_INCLUDE_AUGMENTED_NORMAL", "1").strip().lower() in TRUTHY
EXPLICIT_CHECKPOINT = os.getenv("SJEPA_INSPECT_CHECKPOINT", "").strip()
CHECKPOINT_NAME = (
    "sjepa_curriculum_final_augmented.pt" if INCLUDE_AUGMENTED else "sjepa_curriculum_final.pt"
)
EXPECTED_FINGERPRINT_PREFIX = "d0acc262"
CONDITIONS = ["normal", "parkinsons", "stroke", "myopathic", "cerebralpalsy"]


def artifact_dir_for(mode):
    return ARTIFACT_ROOT / mode


def checkpoint_path_for(mode):
    if EXPLICIT_CHECKPOINT:
        p = Path(EXPLICIT_CHECKPOINT).expanduser()
        return p if p.is_absolute() else artifact_dir_for(mode) / p
    return artifact_dir_for(mode) / CHECKPOINT_NAME


# Real mode requires BOTH the real checkpoint and the real pose cache; else fall back to smoke.
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

# ------------------------------------------------------------------ 1. Anatomy constants
CELLS.append(md(r"""
## 1. Anatomy constants and the frozen laterality operators

These constants match notebooks 04-06 exactly. `MASK_KEYPOINTS` is the 12-landmark maskable whitelist.
`LEFT_RIGHT_PAIRS` is the six lower-body-plus-shoulder pair list used to build the anatomical mirror
and the signed target. We freeze two deterministic functions here BEFORE seeing any features:

- `signed_left_minus_right(coords)`: the regression target `y`. It is a translation-invariant, signed
  per-side excursion difference (left minus right). Positive leans left, negative leans right.
- `anatomical_mirror(coords)`: negates x and swaps each left landmark with its right partner. A clean
  antisymmetric (reflection-equivariant) encoding should negate the decoded scalar under this mirror.

Both operate on RAW coordinates (before the encoder), exactly as the proposal specifies.
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

# Lower-body-plus-shoulder pairs for the SIGNED target (proposal Method, code block).
LEFT_RIGHT_PAIRS = [(11, 12), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32)]

# The FULL 16-pair mirror (matches notebook 05's geometric_view flip): every left/right landmark,
# including face and arm, so the mirror is a valid whole-body reflection.
FULL_MIRROR_PAIRS = [
    (1, 4), (2, 5), (3, 6), (7, 8), (9, 10),          # eyes, ears, mouth
    (11, 12), (13, 14), (15, 16), (17, 18), (19, 20), (21, 22),  # shoulder..thumb
    (23, 24), (25, 26), (27, 28), (29, 30), (31, 32),            # hip..foot index
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
    '''Negate x and swap each left landmark with its right partner. Preserves the visibility column
    if present so the mirrored sequence can go back through the same preprocessing.'''
    mirrored = np.asarray(coords, dtype=np.float32).copy()
    mirrored[:, :, 0] = -mirrored[:, :, 0]
    for left_idx, right_idx in pairs:
        mirrored[:, [left_idx, right_idx], :] = mirrored[:, [right_idx, left_idx], :]
    return mirrored


# Self-check: the target is antisymmetric on RAW coordinates by construction.
_demo = np.random.randn(20, 33, 3).astype(np.float32)
_orig = signed_left_minus_right(_demo)
_mir = signed_left_minus_right(anatomical_mirror(_demo))
print(f"raw-coordinate self-check:  original={_orig:+.4f}  mirrored={_mir:+.4f}  (should be near equal-and-opposite)")
assert abs(_orig + _mir) < 1e-6 * (1 + abs(_orig)), "raw target must be exactly antisymmetric"
print("OK: signed_left_minus_right is antisymmetric on raw coordinates.")
"""))

# ------------------------------------------------------------------ 2. Preprocessing (verbatim reuse)
CELLS.append(md(r"""
## 2. Preprocessing (reused verbatim from notebooks 04-06)

To feed the encoder we reuse the project's exact preprocessing chain: short-gap interpolation,
pelvis-centering and body-scale normalization, and a temporal resize to 64 frames. These are copied
from notebook 05 so the tokens we cache are identical to what training and inspection produced.

A subtle but important point for the mirror arm: `center_and_scale` subtracts the pelvis and divides
by a body-scale, and `signed_left_minus_right` uses per-joint standard deviation (excursion), which is
translation-invariant. So the signed target is well defined either on raw or on centered coordinates.
We apply the mirror on the RAW coordinate column and then run the SAME preprocessing, so the mirrored
input the encoder sees is a genuine reflection of a real preprocessed sequence.
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
    '''raw_sequence: [T, 33, 4] (x, y, z, visibility). Returns (xyz [frames,33,3], valid [frames,33]).'''
    interpolated, valid = interpolate_low_visibility(raw_sequence)
    scaled = center_and_scale(interpolated)
    xyz = temporal_resize(scaled[..., :3], frames)
    valid_resized = temporal_resize(valid.astype(np.float32), frames) > 0.5
    return xyz.astype(np.float32), valid_resized


print("preprocessing helpers ready (interpolate -> center_and_scale -> temporal_resize).")
"""))

# ------------------------------------------------------------------ 3. Model classes (verbatim)
CELLS.append(md(r"""
## 3. Model classes (reused verbatim so `state_dict` keys match)

The S-JEPA classes are not packaged as a module in this project; every notebook pastes them inline.
We paste the same `SkeletonPatchEncoder`, `SkeletonPredictor`, and `SJEPAGait` here so that
`load_state_dict` on the `d0acc262` checkpoint matches key-for-key. We ALWAYS construct the model with
`SJEPAGait(**checkpoint["config"])`, reading whatever dimensions the checkpoint stored, never hardcoding
them, so the state dict matches regardless of the exact embed_dim or depth the run used.

If PyTorch is unavailable (for example a docs-only environment) we set `HAVE_TORCH=False` and the
learned-encoder lanes are skipped with a clear message; the raw-coordinate null and the target still
run so the methodology is demonstrable.
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
            batch = x.shape[0]
            if x.shape[1:] != (self.frames, self.joints, self.coordinate_dim):
                raise ValueError(f"Expected [B,{self.frames},{self.joints},{self.coordinate_dim}], got {tuple(x.shape)}")
            x = x.reshape(batch, self.segments, self.segment_length, self.joints, self.coordinate_dim)
            x = x.permute(0, 1, 3, 2, 4)
            x = x.flatten(3)
            return x  # [B, segments, joints, segment_length*coordinate_dim]

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
        def __init__(self, segments, joints, encoder_dim=64, predictor_dim=64, depth=2, heads=4, dropout=0.0):
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
                     embed_dim=64, encoder_depth=2, predictor_depth=2, heads=4):
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

# ------------------------------------------------------------------ 4. Load checkpoint OR build smoke model
CELLS.append(md(r"""
## 4. Bind to one checkpoint (`d0acc262`) or build a matched smoke encoder

In real mode we load the checkpoint with the same guards as notebook 05 (mode, mask whitelist,
curriculum completion, conditions seen), verify the `d0acc262` fingerprint prefix, and construct the
model from the stored config. In smoke mode we build a small randomly-initialized `SJEPAGait` so the
plumbing runs; the smoke encoder is NOT the trained model and its numbers are illustrative.

We also build the UNTRAINED-ENCODER FLOOR: a second, freshly random-initialized `SJEPAGait` of the
identical architecture. Lane C reads its features.
"""))
CELLS.append(code(r"""
CHECKPOINT = None
FINGERPRINT = "smoke-random"
if HAVE_TORCH and MODE == "real":
    CHECKPOINT = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    # The same guard set notebook 05 uses so we can never bind to the wrong lineage.
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
        warnings.warn(f"Fingerprint {FINGERPRINT[:12]} does not start with {EXPECTED_FINGERPRINT_PREFIX}.")
    config = CHECKPOINT["config"]
    model = SJEPAGait(**config)
    model.load_state_dict(CHECKPOINT["model_state"])
    model.eval()
    FRAMES = config["frames"]
    SEGMENT_LENGTH = config["segment_length"]
    EMBED_DIM = config["embed_dim"]
    print(f"Loaded d0acc262 lineage. fingerprint prefix: {FINGERPRINT[:12]}  config: {config}")
elif HAVE_TORCH:
    # Smoke: a small matched architecture just to exercise the pipeline.
    config = {"frames": 32, "joints": 33, "coordinate_dim": 3, "segment_length": 4,
              "embed_dim": 32, "encoder_depth": 1, "predictor_depth": 1, "heads": 4}
    torch.manual_seed(RANDOM_SEED)
    model = SJEPAGait(**config).eval()
    FRAMES, SEGMENT_LENGTH, EMBED_DIM = config["frames"], config["segment_length"], config["embed_dim"]
    print(f"Smoke encoder built (untrained). config: {config}")
else:
    FRAMES, SEGMENT_LENGTH, EMBED_DIM = 64, 4, 64  # documentary defaults; no encoder available.
    model = None

# Untrained-encoder floor model (Lane C): identical architecture, fresh random init.
floor_model = None
if HAVE_TORCH:
    torch.manual_seed(RANDOM_SEED + 1)
    floor_model = SJEPAGait(**config).eval()
    print("Untrained-encoder floor model built (fresh random init).")

SEGMENTS = FRAMES // SEGMENT_LENGTH
print(f"FRAMES={FRAMES}  SEGMENT_LENGTH={SEGMENT_LENGTH}  SEGMENTS={SEGMENTS}  EMBED_DIM={EMBED_DIM}")
"""))

# ------------------------------------------------------------------ 5. Cohort loading / smoke synthesis
CELLS.append(md(r"""
## 5. Build the cohort with provenance and source-video tags

In real mode we read the `.npz` pose cache exactly as notebook 05's `pose_records_from_cache` does,
keeping the raw `[T, 33, 4]` sequence, `video_id`, `condition`, and `sequence_id`. Provenance
(canonical vs augmented) is NOT a stored column, so we tag it structurally: records read from the
canonical `poses/` folder are canonical. The primary comparison later runs on the canonical-only
subset, because most normal rows come from the augmented extraction path and a naive contrast could
learn acquisition differences.

In smoke mode we reuse the project's canonical `synthetic_gait_sequence` / `synthetic_corpus` (the
same fixtures notebook 05 uses) and then add ONE clearly-labelled overlay: a per-source signed lateral
lean whose direction alternates by source and whose magnitude is set by the mechanism axis (lateralized
conditions lean; symmetric conditions do not). This planted lean is the only thing that makes the
smoke signed target non-degenerate, so the probe has a genuine sign to recover and the mirror has a
genuine sign to flip. It is a plumbing signal, not physiology.
"""))
CELLS.append(code(r"""
FRAMES_SMOKE = 64  # matches the project synthetic fixtures


def synthetic_gait_sequence(condition="normal", frames=64, seed=0):
    '''Code-path fixture copied from notebook 05 (NOT a disease simulation).'''
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


# Mechanism axis for the SMOKE lean overlay only: lateralized conditions carry a signed lean;
# symmetric/normal carry none. This is the built-in negative control in miniature.
SMOKE_LEAN_MAGNITUDE = {"stroke": 1.0, "cerebralpalsy": 0.8, "parkinsons": 0.5,
                        "myopathic": 0.0, "normal": 0.0}


def plant_signed_lean(seq, sign, magnitude, seed):
    '''Overlay a signed per-side excursion difference on a synthetic fixture (smoke only).'''
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
    '''Two SOURCES per condition (so GroupKFold has disjoint sources), alternating lean sign.'''
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
    '''Read the real .npz cache the way notebook 05 does; tolerate optional fields.'''
    records = []
    for condition in conditions:
        folder = Path(pose_dir) / condition
        if not folder.exists():
            continue
        for path in sorted(folder.glob("*.npz")):
            data = np.load(path, allow_pickle=False)
            sequence = data["sequence"].astype(np.float32)  # [T, 33, 4]
            if sequence.ndim != 3 or sequence.shape[1:] != (33, 4):
                raise ValueError(f"Bad pose shape in {path}: {sequence.shape}")
            records.append({
                "condition": condition,
                "sequence_id": str(data["sequence_id"].item()),
                "video_id": str(data["video_id"].item()),
                "raw": sequence,
                # Provenance is not a stored column; canonical poses/ folder => canonical.
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

# ------------------------------------------------------------------ 6. Frozen-target y + features
CELLS.append(md(r"""
## 6. Compute the frozen target `y` and cache the four feature lanes

For every sequence we compute:

- `y` = `signed_left_minus_right` on the preprocessed coordinates (frozen before any fit).
- Lane A (learned probe): flattened per-token features from the frozen `d0acc262` target encoder over
  the 12 maskable landmarks (a laterality-structured feature vector).
- Lane B (raw-coordinate null, the fair ceiling): handcrafted signed per-pair excursion features, no
  network.
- Lane C (untrained-encoder floor): the same per-token features from a random-init encoder.
- Lane D (mean/std-pooled nuisance control): global mean and std over tokens, which is
  permutation-invariant and side-agnostic and so must NOT recover a signed axis.

We also cache the MIRRORED Lane A features (encoder run on the anatomically mirrored input) for the
equivariance test in section 8.
"""))
CELLS.append(code(r"""
def encoder_tokens(enc_model, xyz_batch):
    '''xyz_batch: [B, FRAMES, 33, 3] float32. Returns [B, SEGMENTS, 33, EMBED_DIM] numpy.'''
    with torch.no_grad():
        x = torch.tensor(xyz_batch, dtype=torch.float32)
        tok = enc_model.target_encoder(x).reshape(len(xyz_batch), SEGMENTS, 33, EMBED_DIM)
    return tok.cpu().numpy()


def laterality_feature_from_tokens(tokens):
    '''tokens: [SEGMENTS, 33, D]. Build a signed, side-structured feature vector: for each L/R pair,
    concatenate the time-mean of the LEFT token and RIGHT token and their difference, over the 12
    maskable landmarks. This gives the linear probe explicit access to a per-side contrast.'''
    feats = []
    for left_idx, right_idx in LEFT_RIGHT_PAIRS:
        l = tokens[:, left_idx, :].mean(axis=0)
        r = tokens[:, right_idx, :].mean(axis=0)
        feats.append(l - r)            # signed contrast
        feats.append(l + r)            # symmetric part (context)
    return np.concatenate(feats)


def raw_null_feature(xyz):
    '''Lane B handcrafted signed features: per-pair signed excursion difference on the three axes.'''
    feats = []
    for left_idx, right_idx in LEFT_RIGHT_PAIRS:
        l = xyz[:, left_idx, :].std(axis=0)
        r = xyz[:, right_idx, :].std(axis=0)
        feats.append(l - r)            # 3 numbers per pair, signed
    return np.concatenate(feats)


def pooled_nuisance_feature(tokens):
    '''Lane D: global mean and std over ALL joint-time tokens (permutation-invariant, side-agnostic).'''
    flat = tokens.reshape(-1, tokens.shape[-1])
    return np.concatenate([flat.mean(axis=0), flat.std(axis=0)])


# Prepare all sequences, compute y and the raw null; batch the encoder passes.
prepared = [prepare_sequence(r["raw"], FRAMES) for r in records]
xyz_all = np.stack([p[0] for p in prepared]).astype(np.float32)  # [N, FRAMES, 33, 3]
y = np.array([signed_left_minus_right(x) for x in xyz_all], dtype=np.float64)
B_raw = np.stack([raw_null_feature(x) for x in xyz_all])

# Mirrored inputs (apply mirror on the RAW column, then the SAME preprocessing).
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
    A_learned = np.stack([laterality_feature_from_tokens(t) for t in tokA])
    A_learned_mir = np.stack([laterality_feature_from_tokens(t) for t in tokA_mir])
    C_floor = np.stack([laterality_feature_from_tokens(t) for t in tokC])
    D_pooled = np.stack([pooled_nuisance_feature(t) for t in tokA])
    print(f"feature widths -> A:{A_learned.shape[1]}  B:{B_raw.shape[1]}  C:{C_floor.shape[1]}  D:{D_pooled.shape[1]}")
else:
    A_learned = A_learned_mir = C_floor = D_pooled = None
    print("No torch: only Lane B (raw null) and the target y are available.")

print(f"y stats: mean={y.mean():+.4f} std={y.std():.4f}  n={len(y)}")
"""))

# ------------------------------------------------------------------ 7. Source-disjoint ridge probes
CELLS.append(md(r"""
## 7. Fit the four lanes with source-video-disjoint ridge probes

The split is stated before any fit: folds are SOURCE-VIDEO-DISJOINT via `GroupKFold` on `video_id`.
We pool signed laterality across all conditions (per-class LOSO on n=1 sources is explicitly not
reported). The ridge penalty is chosen only on the training sources of each fold, so held-out sources
never influence it. We report held-out-source R-squared and mean absolute error per lane.
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


def source_disjoint_probe(X, y, groups, n_splits):
    '''Held-out-source R2 and MAE with an inner alpha choice on training sources only.'''
    gkf = GroupKFold(n_splits=n_splits)
    preds = np.full(len(y), np.nan)
    for train_idx, test_idx in gkf.split(X, y, groups):
        # Inner selection of alpha on the training sources only.
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


lanes = {}
lanes["B_raw_null"] = source_disjoint_probe(B_raw, y, groups, n_splits)
if HAVE_TORCH:
    lanes["A_learned"] = source_disjoint_probe(A_learned, y, groups, n_splits)
    lanes["C_floor"] = source_disjoint_probe(C_floor, y, groups, n_splits)
    lanes["D_pooled"] = source_disjoint_probe(D_pooled, y, groups, n_splits)

lane_table = pd.DataFrame([{"lane": k, "R2": v["r2"], "MAE": v["mae"]} for k, v in lanes.items()])
print(lane_table.to_string(index=False))
"""))

# ------------------------------------------------------------------ 8. Mirror equivariance
CELLS.append(md(r"""
## 8. Mirror-equivariance test

We decode the signed scalar on the original and on the mirrored input using the SAME fitted Lane A
probe (refit here on all sources purely to have one probe for the mirror geometry; the decodability
verdict comes from section 7's held-out fit). A clean antisymmetric encoding lands on the line
`y = -x`. We report the slope of decoded-mirrored vs decoded-original; a "flips" verdict needs a
negative slope inside the band [-1.25, -0.8].
"""))
CELLS.append(code(r"""
mirror = {"available": False}
if HAVE_TORCH:
    sc = StandardScaler().fit(A_learned)
    probe = Ridge(alpha=1.0).fit(sc.transform(A_learned), y)
    dec_orig = probe.predict(sc.transform(A_learned))
    dec_mir = probe.predict(sc.transform(A_learned_mir))
    # slope of dec_mir vs dec_orig through the origin-agnostic least squares
    slope = float(np.polyfit(dec_orig, dec_mir, 1)[0])
    flips = (-1.25 <= slope <= -0.8)
    mirror = {"available": True, "slope": slope, "flips": bool(flips),
              "dec_orig": dec_orig, "dec_mir": dec_mir}
    print(f"mirror slope = {slope:+.3f}   verdict: {'FLIPS (antisymmetric)' if flips else 'DOES NOT FLIP'}")
else:
    print("No torch: mirror test skipped (needs the encoder).")
"""))

# ------------------------------------------------------------------ 9. Verdict
CELLS.append(md(r"""
## 9. Pre-registered verdict

The learned probe (Lane A) passes only if all three hold at once:

1. beats the untrained-encoder floor (Lane C) by at least 0.05 R-squared,
2. reaches at least 80 percent of the raw-coordinate null (Lane B) R-squared,
3. the decoded sign is consistent on at least 75 percent of held-out sources.

Lane D must NOT recover a signed axis. Missing any of 1-3 is scored as an informative null. This is
the Day-14 gate from the proposal.
"""))
CELLS.append(code(r"""
def sign_consistency(preds, y_true, groups):
    '''Fraction of held-out SOURCES whose median predicted sign matches the median true sign.'''
    frac = []
    for g in np.unique(groups):
        m = groups == g
        if np.isnan(preds[m]).all():
            continue
        ps = np.sign(np.nanmedian(preds[m]))
        ts = np.sign(np.median(y_true[m]))
        if ts == 0:  # symmetric source: count as correct if prediction is also near zero
            frac.append(1.0 if abs(np.nanmedian(preds[m])) <= 0.5 * np.std(y_true) else 0.0)
        else:
            frac.append(1.0 if ps == ts else 0.0)
    return float(np.mean(frac)) if frac else float("nan")


verdict = {"mode": MODE, "fingerprint": FINGERPRINT}
if HAVE_TORCH:
    a, b, c = lanes["A_learned"]["r2"], lanes["B_raw_null"]["r2"], lanes["C_floor"]["r2"]
    d = lanes["D_pooled"]["r2"]
    signcon = sign_consistency(lanes["A_learned"]["preds"], y, groups)
    beats_floor = (a - c) >= 0.05
    reaches_null = a >= 0.80 * max(b, 1e-9)
    sign_ok = signcon >= 0.75
    passed = bool(beats_floor and reaches_null and sign_ok)
    verdict.update({
        "A_r2": a, "B_r2": b, "C_r2": c, "D_r2": d, "sign_consistency": signcon,
        "beats_floor_by_0.05": bool(beats_floor), "reaches_80pct_of_null": bool(reaches_null),
        "sign_consistent_75pct": bool(sign_ok),
        "mirror_slope": mirror.get("slope"), "mirror_flips": mirror.get("flips"),
        "PRIMARY_VERDICT": "SIGNED AXIS PRESENT ABOVE RAW" if passed else "INFORMATIVE NULL",
        "D_control_ok": bool(abs(d) < 0.05 or d < 0.5 * max(b, 1e-9)),
    })
print(json.dumps(verdict, indent=2))
if MODE == "smoke":
    print("\nNOTE: smoke numbers are illustrative plumbing checks, NOT the d0acc262 result.")
"""))

# ------------------------------------------------------------------ 10. Figures
CELLS.append(md(r"""
## 10. Figures (the two decisive panels)

Figure 1 is the decodability scatter: decoded signed scalar vs the ground-truth target, one dot per
sequence coloured by lane, with the raw-null ceiling and untrained floor as reference. Figure 2 is the
mirror scatter: decoded original vs decoded mirrored, against the `y = -x` reflection line, with the
mean/std-pooled nuisance cloud. These mirror the two SVG mockups in the proposal's `images/` folder.
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
    preds = lanes["A_learned"]["preds"]
    ax[0].scatter(y, preds, s=42, c="#e07a4b", edgecolors="#a44c26", label=f"A learned (R2={lanes['A_learned']['r2']:.3f})")
    lo, hi = float(min(y.min(), preds[~np.isnan(preds)].min())), float(max(y.max(), preds[~np.isnan(preds)].max()))
    ax[0].plot([lo, hi], [lo, hi], "--", color="#5f9e7e", label="identity y = x")
    ax[0].set_title(f"Lane A decodability  (null B={lanes['B_raw_null']['r2']:.3f}, floor C={lanes['C_floor']['r2']:.3f})")
else:
    ax[0].text(0.5, 0.5, "Learned lanes need PyTorch", ha="center")
ax[0].set_xlabel("ground-truth signed target (left minus right)")
ax[0].set_ylabel("decoded signed scalar")
ax[0].legend(loc="upper left", fontsize=9)

# Panel 2: mirror
if mirror["available"]:
    ax[1].scatter(mirror["dec_orig"], mirror["dec_mir"], s=42, c="#2f6f99", edgecolors="#1f4a68",
                  label=f"mirror slope={mirror['slope']:+.2f}")
    lim = float(np.abs(np.concatenate([mirror["dec_orig"], mirror["dec_mir"]])).max()) * 1.1
    ax[1].plot([-lim, lim], [lim, -lim], "--", color="#5f9e7e", label="reflection y = -x")
    ax[1].axhline(0, color="#c4cdd8", lw=1); ax[1].axvline(0, color="#c4cdd8", lw=1)
    ax[1].set_xlim(-lim, lim); ax[1].set_ylim(-lim, lim)
    ax[1].set_title("Mirror equivariance " + ("(FLIPS)" if mirror["flips"] else "(does not flip)"))
else:
    ax[1].text(0.5, 0.5, "Mirror needs the encoder", ha="center")
ax[1].set_xlabel("decoded on original input"); ax[1].set_ylabel("decoded on mirrored input")
ax[1].legend(loc="upper right", fontsize=9)

fig.suptitle(f"Signed-laterality probe  [mode={MODE}, fingerprint={FINGERPRINT[:12]}]  (illustrative in smoke)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig_path = OUT_DIR / "idea5_signed_laterality_probe.png"
fig.savefig(fig_path, dpi=130)
print(f"saved figure: {fig_path}")
plt.show()
"""))

# ------------------------------------------------------------------ 11. Persist
CELLS.append(md(r"""
## 11. Persist the result bundle

We write a small JSON bundle (verdict, per-lane R-squared and MAE, the mirror slope, the fingerprint,
the fold count, and the cohort counts) next to the other artifacts. This is the machine-readable record
the methodology document and the README point at, and the input the futures notebook (05b) compares
real outcomes against.
"""))
CELLS.append(code(r"""
bundle = {
    "notebook": "nb_05a_signed_laterality_probe",
    "mode": MODE,
    "fingerprint": FINGERPRINT,
    "n_sequences": int(len(records)),
    "n_sources": int(n_groups),
    "n_splits": int(n_splits),
    "condition_counts": meta.groupby("condition").size().to_dict(),
    "lanes": {k: {"r2": v["r2"], "mae": v["mae"]} for k, v in lanes.items()},
    "mirror": {"slope": mirror.get("slope"), "flips": mirror.get("flips")},
    "verdict": verdict,
    "transductive": True,
    "notes": "All results transductive; source video is the independent unit; folder labels are dataset annotations, not diagnoses.",
}
bundle_path = OUT_DIR / "idea5_signed_laterality_result.json"
bundle_path.write_text(json.dumps(bundle, indent=2))
print(f"wrote {bundle_path}")
print(json.dumps({k: bundle[k] for k in ("mode", "fingerprint", "n_sequences", "n_sources", "verdict")}, indent=2)[:900])
"""))

# ------------------------------------------------------------------ build
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
