"""inductive_probe.py -- out-of-sample (unseen-source-video) scoring of E1/E2/E3.

Consumes the 15 per-(fold,seed) feature caches built by inductive_harness.py and the
source-disjoint manifest from inductive_split.py. For every lane it fits the ridge
read-out on the TRAIN rows of each outer fold (alpha via an INNER source-disjoint CV,
grid _probe_common.ALPHAS -- identical inner loop to source_disjoint_probe) and predicts
the HELD-OUT test rows using enc[f,s], which never saw those source videos. Test-fold
predictions are pooled across the 5 folds -> POOLED OOF R^2 per seed -> mean +/- SD
across the 3 SSL seeds (primary). Per-fold OOF R^2 gives the across-fold spread.

Encoder-invariant lanes (B raw ceiling, E missingness) depend only on the split, so they
are computed once. The mirror slope is fit per fold on train, decoded on test x AND test
Mx, then pooled (out-of-sample analog of e2/e3a linear_mirror_slope). The analytic
guarantees are RE-ASSERTED, not re-estimated: E2 Phi slope = -1.000 and E3.1 token-
equivariance error = 0 hold for ANY weights on ANY partition.

Emits (work/artifacts/real/):
  idea5_inductive_result.json                    (E1 laterality)
  idea9_equivariant_readout_inductive_result.json (E2 frame-averaged readout)
  idea9_equivariant_encoder_inductive_result.json (E3 frame-averaged encoder)

Source video is the independent unit; folder labels are dataset annotations, not
diagnoses; unseen source video != unseen individual; no clinical claim is made.
"""
import sys
import json
import argparse
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_absolute_error

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _probe_common import ART, ALPHAS, LEFT_RIGHT_PAIRS, sign_consistency  # noqa: E402
from inductive_split import load_folds  # noqa: E402
from inductive_harness import SEEDS, FOLDS, feat_path  # noqa: E402

SOLVER = "svd"          # deterministic ridge (matches repeated_probe engine)
MIRROR_ALPHA = 1.0      # matches transductive linear_mirror_slope convention
E1_OUT = ART / "idea5_inductive_result.json"
E2_OUT = ART / "idea9_equivariant_readout_inductive_result.json"
E3_OUT = ART / "idea9_equivariant_encoder_inductive_result.json"

_FCACHE = {}


def feat(f, s):
    key = (f, s)
    if key not in _FCACHE:
        p = feat_path(f, s)
        if not p.exists():
            raise FileNotFoundError(
                f"missing feature cache {p.name}; run "
                f"`python inductive_harness.py --train --features` first")
        d = np.load(p, allow_pickle=False)
        _FCACHE[key] = {k: d[k] for k in d.files}
    return _FCACHE[key]


# ----------------------------------------------------------------- cohort alignment
def cohort_alignment():
    """Row index + train/test global row arrays per fold, validated identical across all
    15 caches (encoder-invariant seq order / target / groups)."""
    manifest = load_folds()
    base = feat(FOLDS[0], SEEDS[0])
    seq_ids = [str(s) for s in base["seq_ids"]]
    idx = {sid: i for i, sid in enumerate(seq_ids)}
    y = base["y"].astype(np.float64)
    groups = np.asarray([str(g) for g in base["video_ids"]])
    for s in SEEDS:
        for f in FOLDS:
            d = feat(f, s)
            assert [str(x) for x in d["seq_ids"]] == seq_ids, f"seq order drift in feat_f{f}_s{s}"
            assert np.allclose(d["y"].astype(np.float64), y), f"target drift in feat_f{f}_s{s}"
    fold_rows = []
    for fold in manifest["folds"]:
        tr = np.array([idx[str(s)] for s in fold["train_sequence_ids"]], dtype=int)
        te = np.array([idx[str(s)] for s in fold["test_sequence_ids"]], dtype=int)
        assert not (set(tr) & set(te)), f"fold {fold['fold']}: train/test row overlap"
        fold_rows.append((tr, te))
    all_te = np.concatenate([te for _, te in fold_rows])
    assert len(all_te) == len(seq_ids) == 626 and len(set(all_te)) == 626, \
        "test folds are not a clean partition of the 626 cohort"
    return idx, y, groups, fold_rows


# ----------------------------------------------------------------- nested pooled OOF
def _select_alpha(X, y, groups):
    """Inner source-disjoint CV over TRAIN rows -- identical to source_disjoint_probe."""
    best_alpha, best_score = ALPHAS[0], -np.inf
    uniq = np.unique(groups)
    if len(uniq) >= 2:
        inner = GroupKFold(n_splits=min(3, len(uniq)))
        for a in ALPHAS:
            scores = []
            for itr, iva in inner.split(X, y, groups):
                sc = StandardScaler().fit(X[itr])
                mdl = Ridge(alpha=a, solver=SOLVER).fit(sc.transform(X[itr]), y[itr])
                scores.append(r2_score(y[iva], mdl.predict(sc.transform(X[iva]))))
            m = float(np.mean(scores))
            if m > best_score:
                best_score, best_alpha = m, a
    return best_alpha


def pooled_oof(get_X, y, groups, fold_rows, alpha_select=True):
    """get_X(f) -> full [626, d] feature matrix from enc[f,*]. Fit on fold-f train rows,
    predict fold-f test rows; pool test predictions across folds (each row predicted once)."""
    preds = np.full(len(y), np.nan)
    per_fold_r2 = []
    for f in FOLDS:
        X = get_X(f)
        tr, te = fold_rows[f]
        a = _select_alpha(X[tr], y[tr], groups[tr]) if alpha_select else MIRROR_ALPHA
        sc = StandardScaler().fit(X[tr])
        mdl = Ridge(alpha=a, solver=SOLVER).fit(sc.transform(X[tr]), y[tr])
        preds[te] = mdl.predict(sc.transform(X[te]))
        per_fold_r2.append(float(r2_score(y[te], preds[te])))
    ok = ~np.isnan(preds)
    return {
        "r2": float(r2_score(y[ok], preds[ok])),
        "mae": float(mean_absolute_error(y[ok], preds[ok])),
        "sign_consistency": sign_consistency(preds, y, groups),
        "per_fold_r2": per_fold_r2,
        "preds": preds,
    }


def pooled_mirror_slope(get_XM, y, fold_rows):
    """get_XM(f) -> (X_orig[626,d], X_mir[626,d]). For each fold, fit Ridge(alpha=1.0) on
    fold-f TRAIN X_orig->y, decode that fold's TEST X_orig and TEST X_mir, and regress the
    two decoded vectors -> a PER-FOLD slope; return the mean across folds (out-of-sample
    analog of linear_mirror_slope).

    Why per-fold-then-mean, NOT concatenate-then-fit: each fold is a distinct linear map
    (its own StandardScaler + Ridge intercept). Within a fold an antisymmetric feature
    (X_mir = -X_orig) gives dec_m = const_f - dec_o, i.e. slope exactly -1; but const_f
    DIFFERS per fold, so concatenating dec_o/dec_m across folds before one polyfit mixes in
    the between-fold intercept shifts and pushes the pooled slope off -1 (empirically ~-0.99
    down to worse). The analytic guarantee "Phi mirror slope = -1.000 on any partition" is a
    per-linear-map statement; the per-fold slope recovers it to machine precision (~1e-8),
    and the mean is the correct OOS estimate for the non-analytic lanes (E1 A, etc.)."""
    slopes = []
    for f in FOLDS:
        Xo, Xm = get_XM(f)
        tr, te = fold_rows[f]
        sc = StandardScaler().fit(Xo[tr])
        probe = Ridge(alpha=MIRROR_ALPHA).fit(sc.transform(Xo[tr]), y[tr])
        dec_o = probe.predict(sc.transform(Xo[te]))
        dec_m = probe.predict(sc.transform(Xm[te]))
        slopes.append(float(np.polyfit(dec_o, dec_m, 1)[0]))
    return float(np.mean(slopes))


def agg(vals):
    a = np.asarray(vals, dtype=float)
    return {"mean": float(a.mean()), "sd": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
            "min": float(a.min()), "max": float(a.max()), "n": int(len(a)),
            "values": [float(v) for v in a]}


def fold_fingerprints():
    return {f"f{f}_s{s}": str(feat(f, s)["fingerprint"]) for s in SEEDS for f in FOLDS}


# ----------------------------------------------------------------- E1 laterality
def run_e1(y, groups, fold_rows):
    per_seed = {}
    for s in SEEDS:
        A = pooled_oof(lambda f: feat(f, s)["A"], y, groups, fold_rows)
        C = pooled_oof(lambda f: feat(f, s)["C"], y, groups, fold_rows)
        D = pooled_oof(lambda f: feat(f, s)["D"], y, groups, fold_rows)
        slope = pooled_mirror_slope(lambda f: (feat(f, s)["A"], feat(f, s)["A_mir"]), y, fold_rows)
        per_seed[s] = {
            "A_learned": {k: A[k] for k in ("r2", "mae", "sign_consistency", "per_fold_r2")},
            "C_floor": {k: C[k] for k in ("r2", "mae", "per_fold_r2")},
            "D_pooled": {k: D[k] for k in ("r2", "mae", "per_fold_r2")},
            "mirror_slope": slope,
        }
    # encoder-invariant lanes (identical across seeds) -> compute once
    B = pooled_oof(lambda f: feat(f, SEEDS[0])["B"], y, groups, fold_rows)
    E = pooled_oof(lambda f: feat(f, SEEDS[0])["E_miss"], y, groups, fold_rows)

    a = agg([per_seed[s]["A_learned"]["r2"] for s in SEEDS])
    c = agg([per_seed[s]["C_floor"]["r2"] for s in SEEDS])
    d = agg([per_seed[s]["D_pooled"]["r2"] for s in SEEDS])
    sign = agg([per_seed[s]["A_learned"]["sign_consistency"] for s in SEEDS])
    slope = agg([per_seed[s]["mirror_slope"] for s in SEEDS])
    b_r2, e_r2 = B["r2"], E["r2"]

    beats_floor = bool((a["mean"] - c["mean"]) >= 0.05)
    reaches_null = bool(a["mean"] >= 0.80 * max(b_r2, 1e-9))
    sign_ok = bool(sign["mean"] >= 0.75)
    passed = bool(beats_floor and reaches_null and sign_ok)
    return {
        "notebook": "inductive_probe/E1", "mode": "real", "transductive": False,
        "protocol": "K5xS3 source-disjoint fold-local curriculum; pooled OOF R2",
        "seeds": SEEDS, "n_folds": len(FOLDS),
        "pooled_oof_r2": {
            "A_learned": a, "C_floor": c, "D_pooled": d,
            "B_raw_null": {"r2": b_r2, "mae": B["mae"], "note": "encoder-invariant; single value"},
            "E_missingness": {"r2": e_r2, "mae": E["mae"], "note": "encoder-invariant; single value"},
        },
        "mirror_slope": {**slope, "flips": bool(-1.25 <= slope["mean"] <= -0.8)},
        "sign_consistency": sign,
        "beats_floor_by_0.05": beats_floor,
        "reaches_80pct_of_null": reaches_null,
        "sign_consistent_75pct": sign_ok,
        "missingness_control_ok": bool(e_r2 < a["mean"] - 0.05 or e_r2 < 0.05),
        "D_control_ok": bool(abs(d["mean"]) < 0.05 or d["mean"] < 0.5 * max(b_r2, 1e-9)),
        "PRIMARY_VERDICT": ("SIGNED AXIS PRESENT ABOVE RAW (OUT-OF-SAMPLE)" if passed
                            else "INDUCTIVE INFORMATIVE NULL"),
        "per_seed": {str(s): per_seed[s] for s in SEEDS},
        "fold_fingerprints": fold_fingerprints(),
        "notes": ("Out-of-sample across unseen SOURCE VIDEOS (K=5 source-disjoint, fold-local "
                  "5-stage curriculum, 3 SSL seeds). Primary metric = pooled OOF R2 (test rows "
                  "pooled across folds), mean+/-SD over seeds. B/E are input-only, encoder-"
                  "invariant. Source video is the independent unit; folder labels are dataset "
                  "annotations, not diagnoses; unseen video != unseen individual; no clinical claim."),
    }


# ----------------------------------------------------------------- E2 frame-averaged readout
def run_e2(y, groups, fold_rows):
    # exactness of Phi/Psi construction on every cache (machine precision, any partition)
    for s in SEEDS:
        for f in FOLDS:
            d = feat(f, s)
            Phi = d["A"] - d["A_mir"]
            Psi = d["A"] + d["A_mir"]
            assert np.allclose(d["A_mir"] - d["A"], -Phi, atol=1e-5), f"Phi not antisym f{f}s{s}"
            assert np.allclose(d["A_mir"] + d["A"], Psi, atol=1e-5), f"Psi not sym f{f}s{s}"

    def phi(f, s):
        d = feat(f, s); return d["A"] - d["A_mir"]

    def psi(f, s):
        d = feat(f, s); return d["A"] + d["A_mir"]

    def phi_floor(f, s):
        d = feat(f, s); return d["C"] - d["C_mir"]

    per_seed = {}
    for s in SEEDS:
        lanes = {
            "A_free": pooled_oof(lambda f: feat(f, s)["A"], y, groups, fold_rows),
            "Phi_learned": pooled_oof(lambda f: phi(f, s), y, groups, fold_rows),
            "Phi_floor": pooled_oof(lambda f: phi_floor(f, s), y, groups, fold_rows),
            "Psi_learned": pooled_oof(lambda f: psi(f, s), y, groups, fold_rows),
        }
        slopes = {
            "A_free": pooled_mirror_slope(lambda f: (feat(f, s)["A"], feat(f, s)["A_mir"]), y, fold_rows),
            "Phi_learned": pooled_mirror_slope(lambda f: (phi(f, s), -phi(f, s)), y, fold_rows),
            "Phi_floor": pooled_mirror_slope(lambda f: (phi_floor(f, s), -phi_floor(f, s)), y, fold_rows),
            "Psi_learned": pooled_mirror_slope(lambda f: (psi(f, s), psi(f, s)), y, fold_rows),
        }
        # ANALYTIC re-assert (partition/weight-independent): Phi slope = -1.000
        assert abs(slopes["Phi_learned"] + 1.0) < 1e-6, f"Phi_learned slope {slopes['Phi_learned']} != -1 (s{s})"
        assert abs(slopes["Phi_floor"] + 1.0) < 1e-6, f"Phi_floor slope {slopes['Phi_floor']} != -1 (s{s})"
        assert abs(slopes["Psi_learned"] - 1.0) < 1e-6, f"Psi_learned slope {slopes['Psi_learned']} != +1 (s{s})"
        per_seed[s] = {
            "lanes": {k: {kk: v[kk] for kk in ("r2", "mae", "sign_consistency", "per_fold_r2")}
                      for k, v in lanes.items()},
            "mirror_slopes": slopes,
        }
    B = pooled_oof(lambda f: feat(f, SEEDS[0])["B"], y, groups, fold_rows)
    agg_lanes = {k: agg([per_seed[s]["lanes"][k]["r2"] for s in SEEDS])
                 for k in ("A_free", "Phi_learned", "Phi_floor", "Psi_learned")}
    agg_lanes["B_raw"] = {"r2": B["r2"], "mae": B["mae"], "note": "encoder-invariant; single value"}
    return {
        "notebook": "inductive_probe/E2", "mode": "real", "transductive": False,
        "construction": "frame_averaging over G={I, anatomical_mirror}; Phi=A-A_mir (antisym), Psi=A+A_mir (sym)",
        "protocol": "K5xS3 source-disjoint fold-local curriculum; pooled OOF R2",
        "seeds": SEEDS, "n_folds": len(FOLDS),
        "pooled_oof_r2": agg_lanes,
        "mirror_slopes": {k: agg([per_seed[s]["mirror_slopes"][k] for s in SEEDS])
                          for k in ("A_free", "Phi_learned", "Phi_floor", "Psi_learned")},
        "analytic_guarantee": {
            "Phi_slope_equals_minus_one": True,
            "Psi_slope_equals_plus_one": True,
            "max_abs_dev_from_target": float(max(
                max(abs(per_seed[s]["mirror_slopes"]["Phi_learned"] + 1.0) for s in SEEDS),
                max(abs(per_seed[s]["mirror_slopes"]["Phi_floor"] + 1.0) for s in SEEDS),
                max(abs(per_seed[s]["mirror_slopes"]["Psi_learned"] - 1.0) for s in SEEDS))),
            "note": "Phi antisymmetry -> slope -1.000 for ANY weights on ANY partition (re-asserted OOS).",
        },
        "per_seed": {str(s): per_seed[s] for s in SEEDS},
        "fold_fingerprints": fold_fingerprints(),
        "notes": ("Reflection-equivariant readout by construction, scored out-of-sample. Phi's "
                  "mirror slope = -1.000 is analytic (holds on any partition); the inductive run "
                  "only refreshes decodability magnitudes. Phi_floor uses the untrained encoder to "
                  "separate a learning benefit from the geometry. Source video is the independent "
                  "unit; folder labels are dataset annotations, not diagnoses; no clinical claim."),
    }


# ----------------------------------------------------------------- E3 frame-averaged encoder
def run_e3(y, groups, fold_rows):
    embed = int(feat(FOLDS[0], SEEDS[0])["embed_dim"])
    diff_mask = np.zeros(len(LEFT_RIGHT_PAIRS) * 2 * embed, dtype=bool)
    for p in range(len(LEFT_RIGHT_PAIRS)):
        diff_mask[p * 2 * embed: p * 2 * embed + embed] = True
    sum_mask = ~diff_mask

    def ap_diff(f, s):
        return feat(f, s)["Ap"][:, diff_mask]

    def ap_diff_mir(f, s):
        return feat(f, s)["Ap_mir"][:, diff_mask]

    def cp_diff(f, s):
        return feat(f, s)["Cp"][:, diff_mask]

    def cp_diff_mir(f, s):
        return feat(f, s)["Cp_mir"][:, diff_mask]

    # exactness across every cache
    token_errs, diff_antisym, sum_sym = [], [], []
    for s in SEEDS:
        for f in FOLDS:
            d = feat(f, s)
            token_errs.append(float(d["token_equiv_err"]))
            diff_antisym.append(float(np.abs(d["Ap_mir"][:, diff_mask] + d["Ap"][:, diff_mask]).max()))
            sum_sym.append(float(np.abs(d["Ap_mir"][:, sum_mask] - d["Ap"][:, sum_mask]).max()))
    # ANALYTIC re-assert: sigma(E'(x)) == E'(Mx) and block (anti)symmetry, to machine precision
    assert max(token_errs) < 1e-5, f"token equivariance err {max(token_errs)} !~ 0"
    assert max(diff_antisym) < 1e-4, f"diff-block antisymmetry {max(diff_antisym)} !~ 0"
    assert max(sum_sym) < 1e-4, f"sum-block symmetry {max(sum_sym)} !~ 0"

    per_seed = {}
    for s in SEEDS:
        lanes = {
            "Aprime_free": pooled_oof(lambda f: feat(f, s)["Ap"], y, groups, fold_rows),
            "Aprime_diff_learned": pooled_oof(lambda f: ap_diff(f, s), y, groups, fold_rows),
            "Aprime_diff_floor": pooled_oof(lambda f: cp_diff(f, s), y, groups, fold_rows),
        }
        slopes = {
            "Aprime_free": pooled_mirror_slope(
                lambda f: (feat(f, s)["Ap"], feat(f, s)["Ap_mir"]), y, fold_rows),
            "Aprime_diff_learned": pooled_mirror_slope(
                lambda f: (ap_diff(f, s), ap_diff_mir(f, s)), y, fold_rows),
            "Aprime_diff_floor": pooled_mirror_slope(
                lambda f: (cp_diff(f, s), cp_diff_mir(f, s)), y, fold_rows),
        }
        assert abs(slopes["Aprime_diff_learned"] + 1.0) < 1e-6, \
            f"Aprime_diff_learned slope {slopes['Aprime_diff_learned']} != -1 (s{s})"
        assert abs(slopes["Aprime_diff_floor"] + 1.0) < 1e-6, \
            f"Aprime_diff_floor slope {slopes['Aprime_diff_floor']} != -1 (s{s})"
        per_seed[s] = {
            "lanes": {k: {kk: v[kk] for kk in ("r2", "mae", "sign_consistency", "per_fold_r2")}
                      for k, v in lanes.items()},
            "mirror_slopes": slopes,
        }
    B = pooled_oof(lambda f: feat(f, SEEDS[0])["B"], y, groups, fold_rows)
    agg_lanes = {k: agg([per_seed[s]["lanes"][k]["r2"] for s in SEEDS])
                 for k in ("Aprime_free", "Aprime_diff_learned", "Aprime_diff_floor")}
    agg_lanes["B_raw"] = {"r2": B["r2"], "mae": B["mae"], "note": "encoder-invariant; single value"}
    return {
        "notebook": "inductive_probe/E3", "mode": "real", "transductive": False,
        "construction": ("frame-averaged encoder E'(x)=1/2(E(x)+sigma.E(Mx)) over G={I,anatomical_mirror}; "
                         "sigma = L/R joint permutation; exact token-level reflection equivariance, zero retrain"),
        "protocol": "K5xS3 source-disjoint fold-local curriculum; pooled OOF R2",
        "seeds": SEEDS, "n_folds": len(FOLDS),
        "exactness": {
            "token_equivariance_max_abs_err": float(max(token_errs)),
            "diff_block_antisymmetry_max_abs_err": float(max(diff_antisym)),
            "sum_block_symmetry_max_abs_err": float(max(sum_sym)),
        },
        "pooled_oof_r2": agg_lanes,
        "mirror_slopes": {k: agg([per_seed[s]["mirror_slopes"][k] for s in SEEDS])
                          for k in ("Aprime_free", "Aprime_diff_learned", "Aprime_diff_floor")},
        "analytic_guarantee": {
            "token_equivariance_error_zero": True,
            "Aprime_diff_slope_equals_minus_one": True,
            "note": "E'(Mx)=sigma.E'(x) exactly (no retrain); antisymmetric block slope -1.000 on any partition.",
        },
        "per_seed": {str(s): per_seed[s] for s in SEEDS},
        "fold_fingerprints": fold_fingerprints(),
        "notes": ("Frame-averaged encoder lifts the E2 group action from readout to encoder; scored "
                  "out-of-sample. Token-level equivariance and block (anti)symmetry are exact on any "
                  "partition; the inductive run only refreshes decodability magnitudes. A free ridge on "
                  "the full A' does NOT give slope -1 (encoder equivariance alone does not force an "
                  "antisymmetric decoder). Source video is the independent unit; folder labels are "
                  "dataset annotations, not diagnoses; no clinical claim."),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify caches + print fold fingerprints only")
    args = ap.parse_args()

    idx, y, groups, fold_rows = cohort_alignment()
    print(f"[align] 626 cohort, {len(FOLDS)} folds x {len(SEEDS)} seeds; "
          f"test folds partition OK", flush=True)
    fps = fold_fingerprints()
    for k in sorted(fps):
        print(f"  {k}: {fps[k][:12]}", flush=True)
    if args.check:
        return

    e1 = run_e1(y, groups, fold_rows)
    E1_OUT.write_text(json.dumps(e1, indent=2))
    print(f"\n[E1] A={e1['pooled_oof_r2']['A_learned']['mean']:.4f}+/-{e1['pooled_oof_r2']['A_learned']['sd']:.4f}"
          f"  C={e1['pooled_oof_r2']['C_floor']['mean']:.4f}"
          f"  B={e1['pooled_oof_r2']['B_raw_null']['r2']:.4f}"
          f"  slope={e1['mirror_slope']['mean']:.3f}  sign={e1['sign_consistency']['mean']:.3f}"
          f"  -> {e1['PRIMARY_VERDICT']}", flush=True)
    print(f"       wrote {E1_OUT.name}", flush=True)

    e2 = run_e2(y, groups, fold_rows)
    E2_OUT.write_text(json.dumps(e2, indent=2))
    print(f"[E2] Phi_learned={e2['pooled_oof_r2']['Phi_learned']['mean']:.4f}  "
          f"A_free={e2['pooled_oof_r2']['A_free']['mean']:.4f}  "
          f"Phi slope=-1.000 re-asserted (max dev {e2['analytic_guarantee']['max_abs_dev_from_target']:.1e})", flush=True)
    print(f"       wrote {E2_OUT.name}", flush=True)

    e3 = run_e3(y, groups, fold_rows)
    E3_OUT.write_text(json.dumps(e3, indent=2))
    print(f"[E3] Aprime_diff_learned={e3['pooled_oof_r2']['Aprime_diff_learned']['mean']:.4f}  "
          f"token_equiv_err={e3['exactness']['token_equivariance_max_abs_err']:.1e}  "
          f"diff_antisym={e3['exactness']['diff_block_antisymmetry_max_abs_err']:.1e}", flush=True)
    print(f"       wrote {E3_OUT.name}", flush=True)


if __name__ == "__main__":
    main()
