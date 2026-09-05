"""E3(b) probe + merge -- does reflection augmentation INDUCE approximate equivariance?

Loads three Stage-0 normal-only encoders that differ by ONE switch:
  canonical : work/artifacts/real/sjepa_normal.pt              (original nb04 Stage-0, flip-off)
  arm_off   : work/artifacts/real/sjepa_normal_e3b_flip0p00.pt (our faithful trainer, FLIP_PROB=0.0)
  arm_on    : work/artifacts/real/sjepa_normal_e3b_flip0p50.pt (our faithful trainer, FLIP_PROB=0.5)

canonical vs arm_off is a trainer-fidelity cross-check; arm_off vs arm_on is the clean ablation
(identical code / seed / hardware, only the mirror toggled).

Each encoder is probed on E1's instrument. PRIMARY metric = the FREE-readout mirror slope: E1/E2
found the curriculum encoder gives slope ~= -0.70 (reflection equivariance did NOT emerge). The
question here is whether adding reflection augmentation to SSL pretraining moves the free-readout
slope toward -1 (Benton et al. 2020: augmentation -> approximate equivariance) and whether the
signed axis clears the untrained floor.

Eval cohort = the exact 270 normal sequences these encoders were trained on (fully TRANSDUCTIVE,
identical for all three; source video is the independent unit; folder labels are dataset
annotations, not diagnoses). A 626 extrapolation view is added as robustness (normal-only encoder
applied to unseen conditions).

Merges into work/artifacts/real/idea9_equivariant_encoder_result.json under the "retrain" key.
"""
import sys, json, warnings
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _probe_common import (  # noqa: E402
    ART, LEFT_RIGHT_PAIRS, MASK_KEYPOINTS, SJEPAGait, load_models,
    pose_records_from_cache, prepare_sequence, anatomical_mirror, signed_left_minus_right,
    raw_null_feature, batched_tokens, laterality_feature_from_tokens,
    source_disjoint_probe, repeated_probe, sign_consistency, ci95,
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
N_SEEDS = 10
OUT_JSON = ART / "idea9_equivariant_encoder_result.json"

CANON = ART / "sjepa_normal.pt"
ARM_OFF = ART / "sjepa_normal_e3b_flip0p00.pt"
ARM_ON = ART / "sjepa_normal_e3b_flip0p50.pt"


def load_stage0_encoder(path):
    """Instantiate an inference SJEPAGait and load a Stage-0 checkpoint's model_state."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    m = SJEPAGait(**cfg)
    m.load_state_dict(ck["model_state"])
    m.eval()
    return m, cfg, ck


# floor: untrained encoder, seed+1 (matches E1/E2/E3a convention)
_, floor_model, _, _, _ = load_models(seed=SEED)

enc_canon, CFG, ck_canon = load_stage0_encoder(CANON)
enc_off, _, ck_off = load_stage0_encoder(ARM_OFF)
enc_on, _, ck_on = load_stage0_encoder(ARM_ON)
FRAMES, SEGMENT_LENGTH, EMBED_DIM = CFG["frames"], CFG["segment_length"], CFG["embed_dim"]
SEGMENTS = FRAMES // SEGMENT_LENGTH
print(f"config={CFG}")
print(f"canonical flip={ck_canon.get('flip_prob','n/a')} seqs={len(ck_canon['sequence_ids'])}")
print(f"arm_off   flip={ck_off.get('flip_prob')} seqs={len(ck_off['sequence_ids'])} "
      f"finalJEPA={ck_off['history'][-1]['jepa']:.4f}")
print(f"arm_on    flip={ck_on.get('flip_prob')} seqs={len(ck_on['sequence_ids'])} "
      f"finalJEPA={ck_on['history'][-1]['jepa']:.4f}")

ENCODERS = {"canonical": enc_canon, "arm_off": enc_off, "arm_on": enc_on}


def lat(enc, xyz):
    tok = batched_tokens(enc, xyz, SEGMENTS, EMBED_DIM)
    return np.stack([laterality_feature_from_tokens(t) for t in tok])


def linear_mirror_slope(X_orig, X_mir, y, alpha=1.0):
    sc = StandardScaler().fit(X_orig)
    probe = Ridge(alpha=alpha).fit(sc.transform(X_orig), y)
    return float(np.polyfit(probe.predict(sc.transform(X_orig)),
                            probe.predict(sc.transform(X_mir)), 1)[0])


def build_cohort(records):
    prep = [prepare_sequence(r["raw"], FRAMES) for r in records]
    xyz = np.stack([p[0] for p in prep]).astype(np.float32)
    prep_m = [prepare_sequence(anatomical_mirror(r["raw"]), FRAMES) for r in records]
    xyz_m = np.stack([p[0] for p in prep_m]).astype(np.float32)
    y = np.array([signed_left_minus_right(x) for x in xyz], dtype=np.float64)
    B = np.stack([raw_null_feature(x) for x in xyz])
    groups = np.array([r["video_id"] for r in records])
    return xyz, xyz_m, y, B, groups


def probe_encoder(enc, xyz, xyz_m, y, groups, n_splits, do_ci):
    A = lat(enc, xyz)
    A_mir = lat(enc, xyz_m)
    Phi = A - A_mir                              # frame-averaged antisymmetric readout (E2 construction)
    out = {}
    a_res = source_disjoint_probe(A, y, groups, n_splits, solver="svd")
    out["A_free_r2"] = a_res["r2"]
    out["A_free_sign_consistency"] = sign_consistency(a_res["preds"], y, groups)
    out["A_free_mirror_slope"] = linear_mirror_slope(A, A_mir, y)      # PRIMARY: emergent equivariance
    phi_res = source_disjoint_probe(Phi, y, groups, n_splits, solver="svd")
    out["Phi_r2"] = phi_res["r2"]
    out["Phi_mirror_slope"] = linear_mirror_slope(Phi, -Phi, y)        # exact -1 by construction
    if do_ci:
        r2s, signs = repeated_probe(A, y, groups, n_splits, list(range(N_SEEDS)), solver="svd")
        out["A_free_r2_ci95"] = ci95(r2s)
        out["A_free_sign_ci95"] = ci95(signs)
        r2s_phi, _ = repeated_probe(Phi, y, groups, n_splits, list(range(N_SEEDS)), solver="svd")
        out["Phi_r2_ci95"] = ci95(r2s_phi)
    return out


def run_cohort(records, label, do_ci):
    xyz, xyz_m, y, B, groups = build_cohort(records)
    n_groups = len(np.unique(groups))
    n_splits = max(2, min(5, n_groups))
    res = {"label": label, "n_sequences": len(records), "n_sources": int(n_groups),
           "n_splits": int(n_splits), "encoders": {}}
    for name, enc in ENCODERS.items():
        res["encoders"][name] = probe_encoder(enc, xyz, xyz_m, y, groups, n_splits, do_ci)
    # untrained floor
    floor = probe_encoder(floor_model, xyz, xyz_m, y, groups, n_splits, do_ci)
    res["encoders"]["floor_untrained"] = floor
    # raw ceiling
    b_res = source_disjoint_probe(B, y, groups, n_splits, solver="svd")
    res["B_raw_r2"] = b_res["r2"]
    return res


# ---- PRIMARY: 270 normal (transductive, matched to training)
records_all = pose_records_from_cache()
normal_recs = [r for r in records_all if r["condition"] == "normal"]
prep = [prepare_sequence(r["raw"], FRAMES) for r in normal_recs]
valid = np.stack([p[1] for p in prep])
cov = valid[:, :, MASK_KEYPOINTS].mean(axis=(1, 2))
keep = np.where(cov >= 0.50)[0]
normal_270 = [normal_recs[i] for i in keep]
train_ids = set(map(str, ck_off["sequence_ids"]))
assert {r["sequence_id"] for r in normal_270} == train_ids, "270 normal cohort != trainer cohort"
print(f"\n=== E3(b) PRIMARY: {len(normal_270)} normal (transductive) ===")
primary = run_cohort(normal_270, "270_normal_transductive", do_ci=True)
for name in ("canonical", "arm_off", "arm_on", "floor_untrained"):
    e = primary["encoders"][name]
    ci = e.get("A_free_r2_ci95", {})
    print(f"  {name:16s} A_free r2={e['A_free_r2']:+.3f} "
          f"[{ci.get('ci95_lo', float('nan')):+.3f},{ci.get('ci95_hi', float('nan')):+.3f}]  "
          f"free_slope={e['A_free_mirror_slope']:+.3f}  Phi r2={e['Phi_r2']:+.3f}")
print(f"  B_raw ceiling r2={primary['B_raw_r2']:.4f}")

# ---- ROBUSTNESS: 626 (normal-only encoder extrapolated to unseen conditions)
_, _, _, FP, TRAIN_IDS_626 = load_models(seed=SEED)
records_626 = [r for r in records_all if r["sequence_id"] in TRAIN_IDS_626]
print(f"\n=== E3(b) ROBUSTNESS: {len(records_626)} (extrapolation) ===")
robust = run_cohort(records_626, "626_extrapolation", do_ci=False)
for name in ("canonical", "arm_off", "arm_on", "floor_untrained"):
    e = robust["encoders"][name]
    print(f"  {name:16s} A_free r2={e['A_free_r2']:+.3f}  free_slope={e['A_free_mirror_slope']:+.3f}  "
          f"Phi r2={e['Phi_r2']:+.3f}")

retrain_bundle = {
    "construction": ("Stage-0 normal-only retrain with sample-level consistent reflection augmentation; "
                     "arm_off (FLIP_PROB=0.0) vs arm_on (FLIP_PROB=0.5), identical trainer/seed/hardware; "
                     "canonical = original nb04 Stage-0 (sjepa_normal.pt) cross-check"),
    "checkpoints": {
        "canonical": "sjepa_normal.pt",
        "arm_off": "sjepa_normal_e3b_flip0p00.pt",
        "arm_on": "sjepa_normal_e3b_flip0p50.pt",
    },
    "training": {
        "arm_off": {"flip_prob": ck_off.get("flip_prob"), "epochs": ck_off.get("normal_epochs"),
                    "wall_seconds": ck_off.get("wall_seconds"), "final_jepa": ck_off["history"][-1]["jepa"]},
        "arm_on": {"flip_prob": ck_on.get("flip_prob"), "epochs": ck_on.get("normal_epochs"),
                   "wall_seconds": ck_on.get("wall_seconds"), "final_jepa": ck_on["history"][-1]["jepa"]},
    },
    "primary_cohort": primary,
    "robustness_cohort": robust,
    "transductive": True,
    "notes": ("PRIMARY metric = A_free_mirror_slope (free-readout emergent equivariance). E1/E2 curriculum "
              "encoder ~= -0.70. arm_off vs arm_on isolates reflection augmentation (Benton et al. 2020). "
              "Phi_r2 is the frame-averaged antisymmetric readout (E2 construction; slope = -1 exactly). "
              "Eval = 270 normal transductive; 626 = normal-only encoder extrapolated to unseen conditions. "
              "Single-seed proof-of-concept prototype. Source video is the independent unit; folder labels "
              "are dataset annotations, not diagnoses."),
}

bundle = json.loads(OUT_JSON.read_text()) if OUT_JSON.exists() else {}
bundle["retrain"] = retrain_bundle
OUT_JSON.write_text(json.dumps(bundle, indent=2))
print(f"\nmerged retrain results into {OUT_JSON}")
