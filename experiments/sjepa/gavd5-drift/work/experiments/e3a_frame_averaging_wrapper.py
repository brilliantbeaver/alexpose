"""E3(a) -- Reflection-equivariant ENCODER by frame averaging (exact, zero-retrain).

E1: the frozen encoder does not emerge reflection-equivariant (mirror slope ~= -0.70).
E2: imposing frame averaging on the scalar READOUT feature A gives exact antisymmetry
    (Phi = A(x) - A(Mx), slope = -1.000) and recovers latent decodability (~0.27 vs A_free ~0.20).

E3(a) lifts the same group action (order-2 G = {I, M}, M = anatomical_mirror, M^2 = I) from the
readout to the ENCODER itself, via Puny-style frame averaging with the L/R token permutation sigma:

    E'(x)[t, j] = 1/2 ( E(x)[t, j] + E(Mx)[t, sigma(j)] )

sigma swaps every left/right joint partner (FULL_MIRROR_PAIRS). This E' is EXACTLY reflection-
equivariant at the token level: E'(Mx) = sigma . E'(x)  (verified to machine precision), with NO
retraining. It is a stronger structural statement than E2 -- the whole latent field transforms
correctly under reflection, not just one scalar readout.

Consequences we verify numerically:
  (1) token equivariance          sigma(E'(x)) == E'(Mx)                       (machine precision)
  (2) the laterality feature on E' splits EXACTLY into an antisymmetric block
      (the per-pair L-R "diff" channels: A'_diff(Mx) = -A'_diff(x)) and a symmetric block
      (the L+R "sum" channels: A'_sum(Mx) = +A'_sum(x))                        (machine precision)
  (3) probing the antisymmetric block A'_diff reproduces E2's Phi_learned r2 (~0.27) and a linear
      readout on it has mirror slope = -1.000 for ANY weights -- i.e. E2's Phi IS (up to a factor
      of 1/2) the antisymmetric projection of the frame-averaged encoder. E3(a) unifies E2 and E3.
  (4) a FREE ridge on the full A' (both blocks) does NOT give slope -1: encoder equivariance alone
      does not force an antisymmetric decoder -- the readout must still select the antisymmetric
      part. This is the honest boundary of the zero-retrain fix and motivates E3(b) (train it in).

PRIMARY cohort = 626 modeled (transductive, cohort-matched). Source video is the independent unit;
folder labels are dataset annotations, not diagnoses.
"""
import sys, json, warnings
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _probe_common import (  # noqa: E402
    ART, LEFT_RIGHT_PAIRS, FULL_MIRROR_PAIRS, load_models, pose_records_from_cache,
    prepare_sequence, anatomical_mirror, signed_left_minus_right, raw_null_feature,
    batched_tokens, laterality_feature_from_tokens, source_disjoint_probe,
    repeated_probe, sign_consistency, ci95,
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
N_SEEDS = 10
OUT_JSON = ART / "idea9_equivariant_encoder_result.json"

model, floor_model, CONFIG, FP, TRAIN_IDS = load_models(seed=SEED)
FRAMES, SEGMENT_LENGTH, EMBED_DIM = CONFIG["frames"], CONFIG["segment_length"], CONFIG["embed_dim"]
SEGMENTS = FRAMES // SEGMENT_LENGTH
print(f"checkpoint {FP[:12]} config={CONFIG} train_ids={len(TRAIN_IDS)}")

records_642 = pose_records_from_cache()
records_626 = [r for r in records_642 if r["sequence_id"] in TRAIN_IDS]
print(f"642 availability: {len(records_642)} seq / {len({r['video_id'] for r in records_642})} sources")
print(f"626 modeled     : {len(records_626)} seq / {len({r['video_id'] for r in records_626})} sources")

# joint permutation sigma (L/R swap) as an index array over 33 joints
SIGMA = np.arange(33)
for a, b in FULL_MIRROR_PAIRS:
    SIGMA[a], SIGMA[b] = b, a

# diff/sum column masks inside laterality_feature_from_tokens layout
# per pair: [ (L-R): EMBED_DIM ][ (L+R): EMBED_DIM ]; 6 pairs -> 12*EMBED_DIM
DIFF_MASK = np.zeros(len(LEFT_RIGHT_PAIRS) * 2 * EMBED_DIM, dtype=bool)
for p in range(len(LEFT_RIGHT_PAIRS)):
    DIFF_MASK[p * 2 * EMBED_DIM: p * 2 * EMBED_DIM + EMBED_DIM] = True
SUM_MASK = ~DIFF_MASK


def frame_averaged_tokens(enc, xyz, xyz_mir):
    """E'(x) = 1/2 ( E(x) + sigma . E(Mx) ). Returns (B, SEGMENTS, 33, D)."""
    tok = batched_tokens(enc, xyz, SEGMENTS, EMBED_DIM)              # E(x)
    tok_m = batched_tokens(enc, xyz_mir, SEGMENTS, EMBED_DIM)        # E(Mx)
    tok_m_swap = tok_m[:, :, SIGMA, :]                              # sigma . E(Mx)
    return 0.5 * (tok + tok_m_swap)


def lat_from_prime(prime_tokens):
    return np.stack([laterality_feature_from_tokens(t) for t in prime_tokens])


def linear_mirror_slope(X_orig, X_mir, y, alpha=1.0):
    sc = StandardScaler().fit(X_orig)
    probe = Ridge(alpha=alpha).fit(sc.transform(X_orig), y)
    return float(np.polyfit(probe.predict(sc.transform(X_orig)),
                            probe.predict(sc.transform(X_mir)), 1)[0])


def featurize(records):
    prep = [prepare_sequence(r["raw"], FRAMES) for r in records]
    xyz = np.stack([p[0] for p in prep]).astype(np.float32)
    prep_m = [prepare_sequence(anatomical_mirror(r["raw"]), FRAMES) for r in records]
    xyz_m = np.stack([p[0] for p in prep_m]).astype(np.float32)
    y = np.array([signed_left_minus_right(x) for x in xyz], dtype=np.float64)
    B = np.stack([raw_null_feature(x) for x in xyz])
    groups = np.array([r["video_id"] for r in records])

    # frame-averaged encoder on x and on Mx (learned + untrained floor)
    Ap = lat_from_prime(frame_averaged_tokens(model, xyz, xyz_m))         # A'(x)
    Ap_mir = lat_from_prime(frame_averaged_tokens(model, xyz_m, xyz))     # A'(Mx)
    Cp = lat_from_prime(frame_averaged_tokens(floor_model, xyz, xyz_m))   # untrained
    Cp_mir = lat_from_prime(frame_averaged_tokens(floor_model, xyz_m, xyz))
    return dict(y=y, B=B, groups=groups, xyz=xyz, xyz_m=xyz_m,
                Ap=Ap, Ap_mir=Ap_mir, Cp=Cp, Cp_mir=Cp_mir)


def check_equivariance(records, n=8):
    """Verify sigma(E'(x)) == E'(Mx) to machine precision on a sample."""
    sub = records[:n]
    prep = [prepare_sequence(r["raw"], FRAMES) for r in sub]
    xyz = np.stack([p[0] for p in prep]).astype(np.float32)
    prep_m = [prepare_sequence(anatomical_mirror(r["raw"]), FRAMES) for r in sub]
    xyz_m = np.stack([p[0] for p in prep_m]).astype(np.float32)
    Ep_x = frame_averaged_tokens(model, xyz, xyz_m)      # E'(x)
    Ep_Mx = frame_averaged_tokens(model, xyz_m, xyz)     # E'(Mx)
    sigma_Ep_x = Ep_x[:, :, SIGMA, :]                    # sigma . E'(x)
    return float(np.abs(sigma_Ep_x - Ep_Mx).max())


def run_cohort(records, label, do_ci=False):
    f = featurize(records)
    y, groups = f["y"], f["groups"]
    n_groups = len(np.unique(groups))
    n_splits = max(2, min(5, n_groups))

    # structural exactness of the laterality feature on the frame-averaged encoder
    diff_antisym = float(np.abs(f["Ap_mir"][:, DIFF_MASK] + f["Ap"][:, DIFF_MASK]).max())
    sum_sym = float(np.abs(f["Ap_mir"][:, SUM_MASK] - f["Ap"][:, SUM_MASK]).max())

    Ap_diff = f["Ap"][:, DIFF_MASK]         # exactly antisymmetric block (learned)
    Ap_diff_mir = f["Ap_mir"][:, DIFF_MASK]
    Cp_diff = f["Cp"][:, DIFF_MASK]         # untrained floor, antisymmetric block
    Cp_diff_mir = f["Cp_mir"][:, DIFF_MASK]

    lanes = {}
    for key, X in [("Aprime_free", f["Ap"]),          # free ridge on full frame-averaged feature
                   ("Aprime_diff_learned", Ap_diff),  # antisymmetric block, learned encoder
                   ("Aprime_diff_floor", Cp_diff),     # antisymmetric block, untrained encoder
                   ("B_raw", f["B"])]:
        r = source_disjoint_probe(X, y, groups, n_splits)
        r["sign_consistency"] = sign_consistency(r["preds"], y, groups)
        lanes[key] = {k: v for k, v in r.items() if k != "preds"}

    slopes = {
        "Aprime_free": linear_mirror_slope(f["Ap"], f["Ap_mir"], y),
        "Aprime_diff_learned": linear_mirror_slope(Ap_diff, Ap_diff_mir, y),
        "Aprime_diff_floor": linear_mirror_slope(Cp_diff, Cp_diff_mir, y),
    }

    out = {
        "label": label, "n_sequences": len(records), "n_sources": int(n_groups),
        "n_splits": int(n_splits),
        "exactness": {
            "token_equivariance_max_abs_err": check_equivariance(records),
            "diff_block_antisymmetry_max_abs_err": diff_antisym,
            "sum_block_symmetry_max_abs_err": sum_sym,
        },
        "lanes": lanes, "mirror_slopes": slopes,
    }
    if do_ci:
        ci = {}
        for key, X in [("Aprime_diff_learned", Ap_diff), ("Aprime_diff_floor", Cp_diff)]:
            r2s, signs = repeated_probe(X, y, groups, n_splits, list(range(N_SEEDS)), solver="svd")
            ci[key] = ci95(r2s)
            if key == "Aprime_diff_learned":
                ci["Aprime_diff_learned_sign_consistency"] = ci95(signs)
        out["repeated_cv_ci95"] = ci
    return out


print("\n=== E3(a) PRIMARY: 626 modeled ===")
primary = run_cohort(records_626, "626_modeled", do_ci=True)
print(json.dumps({k: primary[k] for k in ("exactness", "lanes", "mirror_slopes")}, indent=2))
print("repeated-CV CI:", json.dumps(primary["repeated_cv_ci95"], indent=2))

print("\n=== E3(a) ROBUSTNESS: 642 availability ===")
robust = run_cohort(records_642, "642_availability", do_ci=False)
print(json.dumps({k: robust[k] for k in ("exactness", "lanes", "mirror_slopes")}, indent=2))

bundle = {
    "notebook": "e3a_frame_averaging_wrapper",
    "mode": "real",
    "fingerprint": FP,
    "checkpoint": "sjepa_curriculum_final.pt",
    "construction": ("frame-averaged encoder E'(x)=1/2(E(x)+sigma.E(Mx)) over G={I,anatomical_mirror}; "
                     "sigma = L/R joint permutation; exact token-level reflection equivariance, zero retrain"),
    "primary_cohort": primary,
    "robustness_cohort": robust,
    "transductive": True,
    "notes": ("E3(a): lifts the E2 group action from the readout to the encoder. E'(Mx)=sigma.E'(x) to "
              "machine precision (no retraining). The laterality feature on E' splits EXACTLY into an "
              "antisymmetric block (L-R channels; slope -1.000 for any linear readout) and a symmetric "
              "block (L+R). Probing the antisymmetric block reproduces E2's Phi_learned -- E2's Phi is the "
              "antisymmetric projection of the frame-averaged encoder. A free ridge on the full feature does "
              "NOT give slope -1 (encoder equivariance alone does not force an antisymmetric decoder), which "
              "motivates E3(b) training reflection into the encoder. PRIMARY = 626 transductive cohort. "
              "Source video is the independent unit; folder labels are dataset annotations, not diagnoses."),
}
OUT_JSON.write_text(json.dumps(bundle, indent=2))
print(f"\nwrote {OUT_JSON}")
