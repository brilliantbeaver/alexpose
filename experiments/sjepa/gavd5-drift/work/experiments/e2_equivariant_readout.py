"""E2 -- Reflection-equivariant readout head, BY CONSTRUCTION (NeurIPS laterality paper).

E1 established the informative null: the frozen S-JEPA encoder does NOT encode the signed
left-right axis as a decodable, sign-flipping quantity (learned lane ~= untrained floor;
anatomical-mirror slope ~= -0.70, not -1.0). E2 asks the constructive question:

    If we IMPOSE reflection-equivariance on the readout, do we recover exact antisymmetry,
    and does the axis become decodable?

Construction (frame averaging over the order-2 group G = {I, M}, M = anatomical_mirror, an
involution M^2 = I; Puny et al. 2021). For laterality feature A(x) = lat_feat(E(x)):
    Phi(x) = A(x) - A(Mx)   (antisymmetric coordinate)   Phi(Mx) = -Phi(x)  EXACTLY
    Psi(x) = A(x) + A(Mx)   (symmetric coordinate)       Psi(Mx) = +Psi(x)  EXACTLY
A LINEAR readout w.Phi is therefore reflection-antisymmetric by construction: pred(Mx) =
w.Phi(Mx) = -w.Phi(x) = -pred(x)  =>  mirror slope = -1.000 to machine precision, for ANY w.
The target y = signed_left_minus_right is itself exactly antisymmetric (y(Mx) = -y(x)), so a
symmetric readout on Psi cannot track its sign -- a clean falsification control.

Lanes (all source-disjoint GroupKFold on video_id, inner alpha selection, SVD ridge for the
repeated CV to match E1):
  A_free      : free ridge on A (the emergent readout; reproduces E1)          slope ~ -0.70
  Phi_learned : ridge on A - A_mir, learned encoder (the constructive fix)     slope  = -1.000
  Phi_floor   : ridge on C - C_mir, UNTRAINED encoder (learning vs geometry)   slope  = -1.000
  Psi_learned : ridge on A + A_mir, symmetric control                          slope  = +1.000
  B_raw       : raw-coordinate ceiling                                          ~ 1.0
Plus a learnable head comparison on the frozen tokens:
  eq_mlp      : s(x) = m(A) - m(A_mir), shared MLP m -> exactly antisymmetric for ANY m
  free_mlp    : m'([A ; A_mir]) -> unconstrained MLP, >= capacity, NOT antisymmetric
showing the antisymmetry (and any gain) comes from the CONSTRAINT, not capacity.

PRIMARY cohort = 626 sequences the encoder trained on (fully transductive; cohort-matched to
the checkpoint). Source video is the independent unit; folder labels are dataset annotations,
not diagnoses.
"""
import sys, json, warnings
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _probe_common import (  # noqa: E402
    ART, CONDITIONS, load_models, pose_records_from_cache, prepare_sequence,
    anatomical_mirror, signed_left_minus_right, raw_null_feature, batched_tokens,
    laterality_feature_from_tokens, source_disjoint_probe, repeated_probe,
    sign_consistency, ci95,
)

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
N_SEEDS = 10
OUT_JSON = ART / "idea9_equivariant_readout_result.json"

model, floor_model, CONFIG, FP, TRAIN_IDS = load_models(seed=SEED)
FRAMES, SEGMENT_LENGTH, EMBED_DIM = CONFIG["frames"], CONFIG["segment_length"], CONFIG["embed_dim"]
SEGMENTS = FRAMES // SEGMENT_LENGTH
print(f"checkpoint {FP[:12]} config={CONFIG}  train_ids={len(TRAIN_IDS)}")

records_642 = pose_records_from_cache()
records_626 = [r for r in records_642 if r["sequence_id"] in TRAIN_IDS]
print(f"642 availability: {len(records_642)} seq / {len({r['video_id'] for r in records_642})} sources")
print(f"626 modeled     : {len(records_626)} seq / {len({r['video_id'] for r in records_626})} sources")


# ----------------------------------------------------------------- featurization
def tok_lat(enc, xyz):
    toks = batched_tokens(enc, xyz, SEGMENTS, EMBED_DIM)
    return np.stack([laterality_feature_from_tokens(t) for t in toks])


def featurize(records):
    prep = [prepare_sequence(r["raw"], FRAMES) for r in records]
    xyz = np.stack([p[0] for p in prep]).astype(np.float32)
    prep_mir = [prepare_sequence(anatomical_mirror(r["raw"]), FRAMES) for r in records]
    xyz_mir = np.stack([p[0] for p in prep_mir]).astype(np.float32)
    y = np.array([signed_left_minus_right(x) for x in xyz], dtype=np.float64)
    B = np.stack([raw_null_feature(x) for x in xyz])
    A = tok_lat(model, xyz)            # A(x)
    A_mir = tok_lat(model, xyz_mir)    # A(Mx)
    C = tok_lat(floor_model, xyz)      # C(x)   untrained encoder
    C_mir = tok_lat(floor_model, xyz_mir)
    groups = np.array([r["video_id"] for r in records])
    return dict(y=y, B=B, A=A, A_mir=A_mir, C=C, C_mir=C_mir, groups=groups)


# ----------------------------------------------------------------- linear mirror-slope helper
def linear_mirror_slope(X_orig, X_mir, y, alpha=1.0):
    """Fit ridge on X_orig->y, then slope of pred(mirrored input) vs pred(original input)."""
    from sklearn.linear_model import Ridge
    sc = StandardScaler().fit(X_orig)
    probe = Ridge(alpha=alpha).fit(sc.transform(X_orig), y)
    dec_orig = probe.predict(sc.transform(X_orig))
    dec_mir = probe.predict(sc.transform(X_mir))
    return float(np.polyfit(dec_orig, dec_mir, 1)[0])


# ----------------------------------------------------------------- learnable heads (torch, grouped CV)
class SharedMLP(nn.Module):
    def __init__(self, d_in, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_head(Xtr_parts, ytr, Xte_parts, mode, hidden=64, epochs=400, lr=1e-3, wd=1e-4, seed=0):
    """mode='eq': s = m(A) - m(A_mir), shared m (exact antisymmetry).
       mode='free': m'([A ; A_mir]) unconstrained.
    Xtr_parts/Xte_parts = (A, A_mir) standardized arrays."""
    torch.manual_seed(seed)
    A_tr, Am_tr = (torch.tensor(v, dtype=torch.float32) for v in Xtr_parts)
    A_te, Am_te = (torch.tensor(v, dtype=torch.float32) for v in Xte_parts)
    yt = torch.tensor(ytr, dtype=torch.float32)
    if mode == "eq":
        net = SharedMLP(A_tr.shape[1], hidden)

        def fwd(A, Am):
            return net(A) - net(Am)
    else:
        net = SharedMLP(A_tr.shape[1] * 2, hidden)

        def fwd(A, Am):
            return net(torch.cat([A, Am], dim=1))
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.MSELoss()
    net.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = lossf(fwd(A_tr, Am_tr), yt)
        loss.backward()
        opt.step()
    net.eval()
    with torch.no_grad():
        pred_te = fwd(A_te, Am_te).cpu().numpy()
        # antisymmetry probe: prediction on mirrored input swaps (A, Am)
        pred_te_mir = fwd(Am_te, A_te).cpu().numpy()
    return pred_te, pred_te_mir


def head_cv(f, mode, n_splits, hidden=64, epochs=400, seed=0):
    from _probe_common import shuffled_group_kfold
    from sklearn.metrics import r2_score, mean_absolute_error
    y, groups = f["y"], f["groups"]
    A, Am = f["A"], f["A_mir"]
    preds = np.full(len(y), np.nan)
    preds_mir = np.full(len(y), np.nan)
    for tr, te in shuffled_group_kfold(groups, n_splits, seed):
        scA = StandardScaler().fit(np.vstack([A[tr], Am[tr]]))  # same transform for A and A_mir
        Xtr = (scA.transform(A[tr]), scA.transform(Am[tr]))
        Xte = (scA.transform(A[te]), scA.transform(Am[te]))
        p, pm = train_head(Xtr, y[tr], Xte, mode, hidden=hidden, epochs=epochs, seed=seed)
        preds[te] = p
        preds_mir[te] = pm
    ok = ~np.isnan(preds)
    slope = float(np.polyfit(preds[ok], preds_mir[ok], 1)[0])
    return {
        "r2": float(r2_score(y[ok], preds[ok])),
        "mae": float(mean_absolute_error(y[ok], preds[ok])),
        "mirror_slope": slope,
        "sign_consistency": sign_consistency(preds, y, groups),
    }


# ----------------------------------------------------------------- run one cohort
def run_cohort(records, label, do_ci=False, do_mlp=False):
    f = featurize(records)
    y, groups = f["y"], f["groups"]
    n_groups = len(np.unique(groups))
    n_splits = max(2, min(5, n_groups))

    Phi = f["A"] - f["A_mir"]
    Phi_mir = f["A_mir"] - f["A"]          # feature of the mirrored input = -Phi (exact)
    Psi = f["A"] + f["A_mir"]
    Psi_mir = f["A_mir"] + f["A"]          # = +Psi
    Phi_floor = f["C"] - f["C_mir"]

    # exactness assertions (machine precision)
    assert np.allclose(Phi_mir, -Phi, atol=1e-5), "Phi not exactly antisymmetric"
    assert np.allclose(Psi_mir, Psi, atol=1e-5), "Psi not exactly symmetric"

    lanes = {}
    for key, X in [("A_free", f["A"]), ("Phi_learned", Phi), ("Phi_floor", Phi_floor),
                   ("Psi_learned", Psi), ("B_raw", f["B"])]:
        r = source_disjoint_probe(X, y, groups, n_splits)
        r["sign_consistency"] = sign_consistency(r["preds"], y, groups)
        lanes[key] = {k: v for k, v in r.items() if k != "preds"}

    slopes = {
        "A_free": linear_mirror_slope(f["A"], f["A_mir"], y),
        "Phi_learned": linear_mirror_slope(Phi, Phi_mir, y),
        "Phi_floor": linear_mirror_slope(Phi_floor, -Phi_floor, y),
        "Psi_learned": linear_mirror_slope(Psi, Psi_mir, y),
    }

    out = {
        "label": label, "n_sequences": len(records), "n_sources": int(n_groups),
        "n_splits": int(n_splits), "lanes": lanes, "mirror_slopes": slopes,
    }

    if do_ci:
        seeds = list(range(N_SEEDS))
        ci = {}
        for key, X in [("A_free", f["A"]), ("Phi_learned", Phi), ("Phi_floor", Phi_floor)]:
            r2s, signs = repeated_probe(X, y, groups, n_splits, seeds, solver="svd")
            ci[key] = ci95(r2s)
            if key == "Phi_learned":
                ci["Phi_learned_sign_consistency"] = ci95(signs)
        out["repeated_cv_ci95"] = ci

    if do_mlp:
        out["learnable_heads"] = {
            "eq_mlp": head_cv(f, "eq", n_splits, hidden=64, epochs=400, seed=SEED),
            "free_mlp": head_cv(f, "free", n_splits, hidden=64, epochs=400, seed=SEED),
        }
    return out


print("\n=== PRIMARY: 626 modeled cohort ===")
primary = run_cohort(records_626, "626_modeled", do_ci=True, do_mlp=True)
print(json.dumps({k: primary[k] for k in ("lanes", "mirror_slopes")}, indent=2))
print("repeated-CV CI:", json.dumps(primary["repeated_cv_ci95"], indent=2))
print("learnable heads:", json.dumps(primary["learnable_heads"], indent=2))

print("\n=== ROBUSTNESS: 642 availability superset ===")
robust = run_cohort(records_642, "642_availability", do_ci=False, do_mlp=False)
print(json.dumps({k: robust[k] for k in ("lanes", "mirror_slopes")}, indent=2))

# sanity: A_free on 642 should reproduce the canonical/E1 robustness A (0.2409)
a642 = robust["lanes"]["A_free"]["r2"]
print(f"\n[validate] A_free(642) = {a642:.6f}  (canonical E1 robustness A = 0.240901)")

bundle = {
    "notebook": "e2_equivariant_readout",
    "mode": "real",
    "fingerprint": FP,
    "checkpoint": "sjepa_curriculum_final.pt",
    "construction": "frame_averaging over G={I, anatomical_mirror}; Phi=A-A_mir (antisym), Psi=A+A_mir (sym)",
    "primary_cohort": primary,
    "robustness_cohort": robust,
    "transductive": True,
    "notes": ("Reflection-equivariant readout by construction. Phi (frame-averaged antisymmetric) "
              "linear readout has mirror slope = -1.000 to machine precision for ANY weights; Psi "
              "(symmetric) = +1.000 and cannot predict the antisymmetric target. Phi_floor uses the "
              "UNTRAINED encoder to separate a learning benefit from an architecture/frame-averaging "
              "benefit. eq_mlp (shared m, s=m(A)-m(A_mir)) is exactly antisymmetric for any m; free_mlp "
              "is a >=capacity unconstrained control. PRIMARY = 626 transductive cohort. Source video is "
              "the independent unit; folder labels are dataset annotations, not diagnoses."),
}
OUT_JSON.write_text(json.dumps(bundle, indent=2))
print(f"\nwrote {OUT_JSON}")
