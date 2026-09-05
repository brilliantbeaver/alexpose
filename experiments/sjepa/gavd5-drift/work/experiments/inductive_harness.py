"""inductive_harness.py -- driver for the inductive (unseen-source-video) redesign.

Trains the 15 fold-local curriculum encoders enc[f,s] (f in 0..4 outer folds,
s in {42,43,44} SSL seeds), each on the TRAIN source videos of fold f only, via
pretrain_fold_local.pretrain(). Then builds a per-(f,s) FEATURE CACHE by embedding
ALL 626 cohort sequences (clean pass + anatomical-mirror pass) with enc[f,s] and a
per-seed untrained floor encoder (seed s+1). The probe (inductive_probe.py) scores
only the HELD-OUT test rows of each fold -> genuinely out-of-sample.

Feature cache per (f,s) holds every matrix E1/E2/E3 need, in ONE fixed cohort order
(manifest "cohort" order): A/A_mir (learned laterality feat on x / Mx), C/C_mir
(untrained floor), D (side-blind pooled nuisance), Ap/Ap_mir + Cp/Cp_mir (frame-
averaged encoder E'(x)=1/2(E(x)+sigma.E(Mx))), and the encoder-invariant y, B (raw
ceiling), E_miss (missingness control), video_ids (groups), seq_ids. The floor and
B/E/y are recomputed per cache so each npz is self-contained.

Fold encoders carry per-fold dataset_fingerprints (recorded, NOT asserted == 7d13841a;
that hard bind stays on the transductive scripts). Source video is the independent
unit; folder labels are dataset annotations, not diagnoses; unseen video != unseen
individual.
"""
import sys
import time
import argparse
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _probe_common import (  # noqa: E402
    ART, CONDITIONS, MASK_KEYPOINTS, LEFT_RIGHT_PAIRS, FULL_MIRROR_PAIRS,
    SJEPAGait, pose_records_from_cache, prepare_sequence, anatomical_mirror,
    signed_left_minus_right, raw_null_feature, missingness_feature,
    batched_tokens, laterality_feature_from_tokens, pooled_nuisance_feature,
)
from inductive_split import load_folds  # noqa: E402
from pretrain_fold_local import pretrain  # noqa: E402

SEEDS = [42, 43, 44]
FOLDS = [0, 1, 2, 3, 4]

INDUCTIVE_DIR = ART / "inductive"
ENC_DIR = INDUCTIVE_DIR / "encoders"
FEAT_DIR = INDUCTIVE_DIR / "features"
COHORT_CACHE = INDUCTIVE_DIR / "cohort_inputs.npz"

# joint permutation sigma (L/R swap) + diff/sum column masks (verbatim from e3a)
SIGMA = np.arange(33)
for _a, _b in FULL_MIRROR_PAIRS:
    SIGMA[_a], SIGMA[_b] = _b, _a


def enc_path(f, s):
    return ENC_DIR / f"enc_f{f}_s{s}.pt"


def feat_path(f, s):
    return FEAT_DIR / f"feat_f{f}_s{s}.npz"


# ----------------------------------------------------------------- shared cohort inputs (encoder-invariant)
def build_cohort_inputs(verbose=True):
    """Prepare xyz / mirror-xyz / y / B / E_miss for all 626 sequences, in manifest
    cohort order. Cached to COHORT_CACHE. Reused by every (f,s) feature build."""
    manifest = load_folds()
    order = manifest["cohort"]  # [{sequence_id, video_id, condition}], canonical 626 order
    by_id = {r["sequence_id"]: r for r in pose_records_from_cache()}

    xyz_list, xyz_mir_list, y_list, B_list, E_list = [], [], [], [], []
    seq_ids, video_ids, conditions = [], [], []
    for entry in order:
        sid = entry["sequence_id"]
        rec = by_id[sid]
        xyz, valid = prepare_sequence(rec["raw"], 64)
        xyz_mir, _ = prepare_sequence(anatomical_mirror(rec["raw"]), 64)
        xyz_list.append(xyz.astype(np.float32))
        xyz_mir_list.append(xyz_mir.astype(np.float32))
        y_list.append(signed_left_minus_right(xyz))
        B_list.append(raw_null_feature(xyz))
        E_list.append(missingness_feature(valid))
        seq_ids.append(sid)
        video_ids.append(entry["video_id"])
        conditions.append(entry["condition"])

    assert len(seq_ids) == 626, f"cohort inputs {len(seq_ids)} != 626"
    INDUCTIVE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        COHORT_CACHE,
        xyz=np.stack(xyz_list), xyz_mir=np.stack(xyz_mir_list),
        y=np.asarray(y_list, dtype=np.float64),
        B=np.stack(B_list), E_miss=np.stack(E_list),
        seq_ids=np.asarray(seq_ids), video_ids=np.asarray(video_ids),
        conditions=np.asarray(conditions),
    )
    if verbose:
        print(f"[cohort] wrote {COHORT_CACHE}  ({len(seq_ids)} seq / "
              f"{len(set(video_ids))} videos, manifest order)")
    return COHORT_CACHE


def load_cohort_inputs():
    if not COHORT_CACHE.exists():
        build_cohort_inputs()
    return np.load(COHORT_CACHE, allow_pickle=False)


# ----------------------------------------------------------------- training (15 fold-local curricula)
def _encoder_is_valid(path):
    if not path.exists():
        return False
    try:
        ck = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return False
    return bool(ck.get("curriculum_complete")) and ck.get("conditions_seen") == CONDITIONS


def train_all(verbose=True):
    manifest = load_folds()
    ENC_DIR.mkdir(parents=True, exist_ok=True)
    order = [(s, f) for s in SEEDS for f in FOLDS]  # seed-major: seed 42 all folds first
    t0 = time.time()
    for i, (s, f) in enumerate(order):
        out = enc_path(f, s)
        if _encoder_is_valid(out):
            if verbose:
                print(f"[train {i+1}/15] enc_f{f}_s{s} exists -> skip", flush=True)
            continue
        train_vids = manifest["folds"][f]["train_video_ids"]
        if verbose:
            print(f"[train {i+1}/15] enc_f{f}_s{s}: {len(train_vids)} train videos "
                  f"(elapsed {time.time()-t0:.0f}s)", flush=True)
        pretrain(train_vids, seed=s, out_path=out, fold_id=f, verbose=verbose)
    if verbose:
        print(f"[train] all 15 fold-local curricula present ({time.time()-t0:.0f}s total)", flush=True)


# ----------------------------------------------------------------- fold encoder loader (relaxed fingerprint)
def load_fold_encoder(path, floor_seed):
    """Load a fold-local curriculum encoder. Asserts the structural invariants that
    remain true for fold encoders (mode/mask_keypoints/curriculum_complete/conditions
    _seen) but RECORDS the per-fold dataset_fingerprint instead of asserting 7d13841a."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    assert ck.get("mode") == "real", f"{path}: mode != real"
    assert ck.get("mask_keypoints") == MASK_KEYPOINTS, f"{path}: mask_keypoints drift"
    assert ck.get("curriculum_complete"), f"{path}: curriculum incomplete"
    assert ck.get("conditions_seen") == CONDITIONS, f"{path}: conditions_seen drift"
    config = ck["config"]
    model = SJEPAGait(**config)
    model.load_state_dict(ck["model_state"])
    model.eval()
    torch.manual_seed(floor_seed)
    floor = SJEPAGait(**config).eval()
    return model, floor, config, str(ck["dataset_fingerprint"])


# ----------------------------------------------------------------- feature derivation
def _frame_averaged_tokens(tok_x, tok_mx):
    """E'(x) = 1/2 ( E(x) + sigma . E(Mx) ).  tok_* : [N, segments, 33, D]."""
    return 0.5 * (tok_x + tok_mx[:, :, SIGMA, :])


def _lat_stack(tokens):
    return np.stack([laterality_feature_from_tokens(t) for t in tokens])


def build_features(f, s, verbose=True):
    """Embed all 626 with enc[f,s] + floor(seed s+1); derive & cache every lane matrix."""
    coh = load_cohort_inputs()
    xyz, xyz_mir = coh["xyz"], coh["xyz_mir"]
    model, floor, config, fp = load_fold_encoder(enc_path(f, s), floor_seed=s + 1)
    segments, embed = config["frames"] // config["segment_length"], config["embed_dim"]

    # four token tensors -> everything else is derived
    Ex = batched_tokens(model, xyz, segments, embed)         # E(x)
    Emx = batched_tokens(model, xyz_mir, segments, embed)     # E(Mx)
    Fx = batched_tokens(floor, xyz, segments, embed)          # floor(x)
    Fmx = batched_tokens(floor, xyz_mir, segments, embed)     # floor(Mx)

    A = _lat_stack(Ex)
    A_mir = _lat_stack(Emx)
    C = _lat_stack(Fx)
    C_mir = _lat_stack(Fmx)
    D = np.stack([pooled_nuisance_feature(t) for t in Ex])
    Ap = _lat_stack(_frame_averaged_tokens(Ex, Emx))          # A'(x)
    Ap_mir = _lat_stack(_frame_averaged_tokens(Emx, Ex))      # A'(Mx)
    Cp = _lat_stack(_frame_averaged_tokens(Fx, Fmx))
    Cp_mir = _lat_stack(_frame_averaged_tokens(Fmx, Fx))

    # token-level equivariance error sigma(E'(x)) == E'(Mx) (exact, weight/partition-free)
    Ep_x = _frame_averaged_tokens(Ex, Emx)
    Ep_mx = _frame_averaged_tokens(Emx, Ex)
    token_equiv_err = float(np.abs(Ep_x[:, :, SIGMA, :] - Ep_mx).max())

    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        feat_path(f, s),
        A=A.astype(np.float32), A_mir=A_mir.astype(np.float32),
        C=C.astype(np.float32), C_mir=C_mir.astype(np.float32),
        D=D.astype(np.float32),
        Ap=Ap.astype(np.float32), Ap_mir=Ap_mir.astype(np.float32),
        Cp=Cp.astype(np.float32), Cp_mir=Cp_mir.astype(np.float32),
        B=coh["B"].astype(np.float32), E_miss=coh["E_miss"].astype(np.float32),
        y=coh["y"], video_ids=coh["video_ids"], seq_ids=coh["seq_ids"],
        embed_dim=np.int64(embed), fold=np.int64(f), seed=np.int64(s),
        fingerprint=np.asarray(fp), token_equiv_err=np.float64(token_equiv_err),
    )
    if verbose:
        print(f"[feat] enc_f{f}_s{s} fp={fp[:12]} token_equiv_err={token_equiv_err:.2e} "
              f"-> {feat_path(f, s).name}", flush=True)


def build_all_features(verbose=True):
    build_cohort_inputs(verbose=verbose)
    for s in SEEDS:
        for f in FOLDS:
            if feat_path(f, s).exists():
                if verbose:
                    print(f"[feat] feat_f{f}_s{s} exists -> skip", flush=True)
                continue
            if not _encoder_is_valid(enc_path(f, s)):
                print(f"[feat] MISSING encoder enc_f{f}_s{s} -> cannot build features", flush=True)
                continue
            build_features(f, s, verbose=verbose)


def load_features(f, s):
    return np.load(feat_path(f, s), allow_pickle=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true", help="train the 15 fold-local curricula (cache/skip)")
    ap.add_argument("--features", action="store_true", help="build the 15 feature caches (needs encoders)")
    ap.add_argument("--cohort", action="store_true", help="(re)build shared cohort inputs cache only")
    args = ap.parse_args()
    did = False
    if args.cohort:
        build_cohort_inputs(); did = True
    if args.train:
        train_all(); did = True
    if args.features:
        build_all_features(); did = True
    if not did:
        ap.error("choose --train and/or --features (or --cohort)")
