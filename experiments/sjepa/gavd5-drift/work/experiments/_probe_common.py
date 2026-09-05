"""Shared, side-effect-free machinery for the NeurIPS laterality probes (E1/E2/E3).

Every definition here is copied VERBATIM from the validated
`e1_laterality_hardened.py` (which in turn mirrors
`notes/ideas-claude/05-signed-laterality-decodability/_build_nb_05a.py`). Importing this
module runs NO experiment code -- it only defines constants, preprocessing, model classes,
feature builders, and probe engines. E2/E3 import from here so all three experiments share
one bit-for-bit-identical pipeline; E1 stays standalone (already validated) as the anchor.

All results downstream are TRANSDUCTIVE; the source video is the independent unit; folder
labels are dataset annotations, not diagnoses.
"""
from pathlib import Path
import copy
import numpy as np
import torch
from torch import nn
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_absolute_error

# ----------------------------------------------------------------- paths / constants
ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "work" / "artifacts" / "real"
POSE_DIR = ART / "poses"
CKPT_PATH = ART / "sjepa_curriculum_final.pt"
EXPECTED_FP = "7d13841a"
CONDITIONS = ["normal", "parkinsons", "stroke", "myopathic", "cerebralpalsy"]

MASK_KEYPOINTS = [11, 12, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
LEFT_RIGHT_PAIRS = [(11, 12), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32)]
FULL_MIRROR_PAIRS = [
    (1, 4), (2, 5), (3, 6), (7, 8), (9, 10),
    (11, 12), (13, 14), (15, 16), (17, 18), (19, 20), (21, 22),
    (23, 24), (25, 26), (27, 28), (29, 30), (31, 32),
]
ALPHAS = np.logspace(-3, 3, 13)


# ----------------------------------------------------------------- target + operator
def signed_left_minus_right(coords):
    coords = np.asarray(coords, dtype=np.float64)[..., :3]
    total = 0.0
    for left_idx, right_idx in LEFT_RIGHT_PAIRS:
        total += coords[:, left_idx, :].std(axis=0).sum() - coords[:, right_idx, :].std(axis=0).sum()
    return float(total)


def anatomical_mirror(coords, pairs=FULL_MIRROR_PAIRS):
    mirrored = np.asarray(coords, dtype=np.float32).copy()
    mirrored[:, :, 0] = -mirrored[:, :, 0]
    for left_idx, right_idx in pairs:
        mirrored[:, [left_idx, right_idx], :] = mirrored[:, [right_idx, left_idx], :]
    return mirrored


# ----------------------------------------------------------------- preprocessing
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
    interpolated, valid = interpolate_low_visibility(raw_sequence)
    scaled = center_and_scale(interpolated)
    xyz = temporal_resize(scaled[..., :3], frames)
    valid_resized = temporal_resize(valid.astype(np.float32), frames) > 0.5
    return xyz.astype(np.float32), valid_resized


# ----------------------------------------------------------------- model classes
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


# ----------------------------------------------------------------- checkpoint / cohort loaders
def load_models(seed=42):
    """Return (model, floor_model, config, fingerprint, train_ids). model = canonical 7d13841a;
    floor_model = untrained same-config encoder (seed+1, matching e1)."""
    ck = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    assert ck.get("mode") == "real"
    assert ck.get("mask_keypoints") == MASK_KEYPOINTS
    assert ck.get("curriculum_complete", False)
    assert ck.get("conditions_seen") == CONDITIONS
    fp = str(ck["dataset_fingerprint"])
    assert fp.startswith(EXPECTED_FP), f"fingerprint {fp[:12]} != {EXPECTED_FP}"
    config = ck["config"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = SJEPAGait(**config)
    model.load_state_dict(ck["model_state"])
    model.eval()
    torch.manual_seed(seed + 1)
    floor_model = SJEPAGait(**config).eval()
    train_ids = set(map(str, ck["sequence_ids"]))
    return model, floor_model, config, fp, train_ids


def pose_records_from_cache(pose_dir=POSE_DIR, conditions=CONDITIONS):
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
            })
    return records


# ----------------------------------------------------------------- feature builders
def encoder_tokens(enc_model, xyz_batch, segments, embed_dim):
    with torch.no_grad():
        x = torch.tensor(xyz_batch, dtype=torch.float32)
        tok = enc_model.target_encoder(x).reshape(len(xyz_batch), segments, 33, embed_dim)
    return tok.cpu().numpy()


def batched_tokens(enc, arr, segments, embed_dim, bs=16):
    out = []
    for i in range(0, len(arr), bs):
        out.append(encoder_tokens(enc, arr[i:i + bs], segments, embed_dim))
    return np.concatenate(out)


def laterality_feature_from_tokens(tokens):
    feats = []
    for left_idx, right_idx in LEFT_RIGHT_PAIRS:
        l = tokens[:, left_idx, :].mean(axis=0)
        r = tokens[:, right_idx, :].mean(axis=0)
        feats.append(l - r)
        feats.append(l + r)
    return np.concatenate(feats)


def raw_null_feature(xyz):
    feats = []
    for left_idx, right_idx in LEFT_RIGHT_PAIRS:
        feats.append(xyz[:, left_idx, :].std(axis=0) - xyz[:, right_idx, :].std(axis=0))
    return np.concatenate(feats)


def pooled_nuisance_feature(tokens):
    flat = tokens.reshape(-1, tokens.shape[-1])
    return np.concatenate([flat.mean(axis=0), flat.std(axis=0)])


def missingness_feature(valid):
    frac = valid.mean(axis=0)
    feats = []
    for left_idx, right_idx in LEFT_RIGHT_PAIRS:
        feats.append([frac[left_idx] - frac[right_idx], frac[left_idx] + frac[right_idx]])
    return np.asarray(feats, dtype=np.float64).ravel()


# ----------------------------------------------------------------- probe engines
def source_disjoint_probe(X, y, groups, n_splits, solver="auto"):
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
                    mdl = Ridge(alpha=a, solver=solver).fit(sc.transform(X[train_idx][itr]), y[train_idx][itr])
                    scores.append(r2_score(y[train_idx][iva], mdl.predict(sc.transform(X[train_idx][iva]))))
                if np.mean(scores) > best_score:
                    best_score, best_alpha = np.mean(scores), a
        sc = StandardScaler().fit(X[train_idx])
        mdl = Ridge(alpha=best_alpha, solver=solver).fit(sc.transform(X[train_idx]), y[train_idx])
        preds[test_idx] = mdl.predict(sc.transform(X[test_idx]))
    ok = ~np.isnan(preds)
    return {"r2": float(r2_score(y[ok], preds[ok])), "mae": float(mean_absolute_error(y[ok], preds[ok])),
            "preds": preds}


def shuffled_group_kfold(groups, n_splits, seed):
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(uniq)
    folds = np.array_split(perm, n_splits)
    for fold_groups in folds:
        test_mask = np.isin(groups, fold_groups)
        yield np.flatnonzero(~test_mask), np.flatnonzero(test_mask)


def repeated_probe(X, y, groups, n_splits, seeds, solver="svd"):
    r2s, signs = [], []
    for seed in seeds:
        preds = np.full(len(y), np.nan)
        for train_idx, test_idx in shuffled_group_kfold(groups, n_splits, seed):
            inner_groups = groups[train_idx]
            best_alpha, best_score = ALPHAS[0], -np.inf
            if len(np.unique(inner_groups)) >= 2:
                for a in ALPHAS:
                    scores = []
                    for itr, iva in shuffled_group_kfold(inner_groups, min(3, len(np.unique(inner_groups))), seed):
                        sc = StandardScaler().fit(X[train_idx][itr])
                        mdl = Ridge(alpha=a, solver=solver).fit(sc.transform(X[train_idx][itr]), y[train_idx][itr])
                        scores.append(r2_score(y[train_idx][iva], mdl.predict(sc.transform(X[train_idx][iva]))))
                    if np.mean(scores) > best_score:
                        best_score, best_alpha = np.mean(scores), a
            sc = StandardScaler().fit(X[train_idx])
            mdl = Ridge(alpha=best_alpha, solver=solver).fit(sc.transform(X[train_idx]), y[train_idx])
            preds[test_idx] = mdl.predict(sc.transform(X[test_idx]))
        ok = ~np.isnan(preds)
        r2s.append(float(r2_score(y[ok], preds[ok])))
        signs.append(sign_consistency(preds, y, groups))
    return np.array(r2s), np.array(signs)


def sign_consistency(preds, y_true, groups):
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


def ci95(arr):
    arr = np.asarray(arr, dtype=float)
    n = len(arr)
    mean = float(arr.mean())
    sd = float(arr.std(ddof=1)) if n > 1 else 0.0
    tcrit = {5: 2.776, 10: 2.262}.get(n, 2.262)
    half = tcrit * sd / np.sqrt(n) if n > 1 else 0.0
    return {"mean": mean, "sd": sd, "ci95_lo": mean - half, "ci95_hi": mean + half,
            "min": float(arr.min()), "max": float(arr.max()), "n": n}
