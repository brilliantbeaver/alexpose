"""Phase 0: freeze provenance and reproduce E0 (the current baseline).

This script does the leakage-safe, reproducible groundwork that every later
comparison depends on. It is intentionally deterministic and self-documenting.

It performs:

1. Cache + manifest inventory and SHA-256 hashing (data provenance).
2. A locked, versioned grouped fold registry (generation ``g1``, source-grouped,
   StratifiedGroupKFold, seed 42), written to JSON so RF and every S-JEPA
   comparator share the identical folds.
3. The Random Forest baseline (E0-RF), scored per clip on those exact folds,
   with pooled out-of-fold (OOF) predictions saved.
4. A battery of shortcut controls on the same folds: static pose, pose
   mean+std, visibility-only, duration+acquisition, and body-proportion. These
   quantify how much of any score is nuisance rather than gait.
5. A run manifest recording git revision, hashes, environment, and the exact
   registry, ending with an atomic ``COMPLETED.json`` marker.

Everything is written under ``artifacts/eval/g1/`` and never overwrites an
existing completed run (fail-safe). Cached keypoints are treated as read-only.

Run:  python scripts_phase0_provenance.py
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

EXP_DIR = Path(__file__).resolve().parent.parent  # scripts/ -> experiment dir
sys.path.insert(0, str(EXP_DIR))

from sjepa.data import load_index, grouped_kfold, SequenceRecord  # noqa: E402
from sjepa.classical import build_feature_matrix, train_rf_and_predict  # noqa: E402
from sjepa.eval import evaluate, aggregate_folds  # noqa: E402

CACHE_DIR = EXP_DIR / "artifacts" / "keypoints"
MANIFEST = EXP_DIR / "video-data" / "manifest.csv"
OUT_DIR = EXP_DIR / "artifacts" / "eval" / "g1"
LABELS = ["normal", "ms", "pd"]
SEED = 42
N_SPLITS = 5


# ---------------------------------------------------------------------------
# Provenance helpers
# ---------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_rev() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=EXP_DIR, text=True
        ).strip()
    except Exception:
        return "unknown"


def _git_dirty() -> bool:
    """True if the tree has uncommitted changes (records provenance honesty)."""
    try:
        out = subprocess.check_output(["git", "status", "--porcelain"],
                                      cwd=EXP_DIR, text=True)
        return bool(out.strip())
    except Exception:
        return True


def cache_manifest() -> dict:
    """Hash every cached .npz plus the video manifest into one provenance record."""
    files = sorted(CACHE_DIR.glob("*.npz"))
    per_file = {p.name: sha256_file(p) for p in files}
    combined = hashlib.sha256(
        "".join(f"{k}:{v}" for k, v in sorted(per_file.items())).encode()
    ).hexdigest()
    return {
        "cache_version": "cache_v0",
        "n_files": len(files),
        "combined_sha256": combined,
        "manifest_sha256": sha256_file(MANIFEST) if MANIFEST.exists() else None,
        "per_file_sha256": per_file,
    }


# ---------------------------------------------------------------------------
# Fold registry (generation g1)
# ---------------------------------------------------------------------------
def build_registry(records) -> dict:
    """Lock the source-grouped folds to JSON so every branch shares them."""
    folds = []
    for k, (train_recs, test_recs) in enumerate(
        grouped_kfold(records, n_splits=N_SPLITS, seed=SEED)
    ):
        folds.append(
            {
                "fold": k,
                "train_clips": sorted(r.clip_name for r in train_recs),
                "test_clips": sorted(r.clip_name for r in test_recs),
                "train_sources": sorted({r.source_id for r in train_recs}),
                "test_sources": sorted({r.source_id for r in test_recs}),
                "test_labels": {r.clip_name: r.label for r in test_recs},
            }
        )
    return {
        "generation": "g1",
        "grouping": "source_id (provisional; NOT verified participant)",
        "splitter": "StratifiedGroupKFold",
        "n_splits": len(folds),
        "seed": SEED,
        "labels_order": LABELS,
        "folds": folds,
    }


def records_by_clip(records) -> dict:
    return {r.clip_name: r for r in records}


# ---------------------------------------------------------------------------
# Shortcut-control feature extractors (per clip, from the cache)
# ---------------------------------------------------------------------------
def _norm_xy(rec: SequenceRecord) -> np.ndarray:
    seq = rec.load_norm()            # (T, 33, 2/3)
    return seq[:, :, :2]             # (T, 33, 2)


def feat_mean_pose(rec):
    xy = _norm_xy(rec)
    return np.nan_to_num(xy.mean(axis=0).reshape(-1))          # 66


def feat_mean_std_pose(rec):
    xy = _norm_xy(rec)
    m = xy.mean(axis=0).reshape(-1)
    s = xy.std(axis=0).reshape(-1)
    return np.nan_to_num(np.concatenate([m, s]))              # 132


def feat_visibility(rec):
    raw = rec.load_raw()                                      # (T,33,3)
    vis = raw[:, :, 2]
    return np.nan_to_num(np.array([np.nanmean(vis), np.nanstd(vis),
                                   *np.nanmean(vis, axis=0)]))  # 2 + 33


def feat_duration_acq(rec):
    """Duration + a coarse acquisition proxy (n_frames only; fps is uniform in cache)."""
    raw = rec.load_raw()
    return np.array([raw.shape[0], float(raw.shape[0])])       # duration proxy


def feat_body_proportion(rec):
    """A few inter-joint distances from the median (raw) frame: anthropometry."""
    raw = rec.load_raw()                                      # (T,33,3)
    xy = raw[:, :, :2]
    med = np.nanmedian(xy, axis=0)                            # (33,2)
    def d(a, b):
        return float(np.linalg.norm(med[a] - med[b]))
    pairs = [(11, 12), (11, 23), (23, 24), (23, 25), (25, 27),
             (24, 26), (26, 28), (11, 13), (13, 15)]
    return np.nan_to_num(np.array([d(a, b) for a, b in pairs]))


CONTROLS = {
    "mean_pose": feat_mean_pose,
    "mean_std_pose": feat_mean_std_pose,
    "visibility_only": feat_visibility,
    "duration_acq": feat_duration_acq,
    "body_proportion": feat_body_proportion,
}


def run_control(name, featfn, registry, by_clip):
    """Score one control on the locked folds; return fold metrics + pooled OOF."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

    fold_metrics_lr, fold_metrics_rf = [], []
    oof_lr, oof_rf = [], []          # pooled OOF preds, so we can report the SAME
    for fold in registry["folds"]:   # pooled macro-F1 that R1 reports (AR-5 P1-B)
        tr = [by_clip[c] for c in fold["train_clips"]]
        te = [by_clip[c] for c in fold["test_clips"]]
        Xtr = np.stack([featfn(r) for r in tr])
        Xte = np.stack([featfn(r) for r in te])
        ytr = [r.label for r in tr]
        yte = [r.label for r in te]
        keep = Xtr.std(axis=0) > 0
        if keep.sum() == 0:
            keep = np.ones(Xtr.shape[1], dtype=bool)
        Xtr, Xte = Xtr[:, keep], Xte[:, keep]
        sc = StandardScaler().fit(Xtr)
        Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
        # logistic
        lr = LogisticRegression(max_iter=2000, class_weight="balanced")
        lr.fit(Xtr_s, ytr)
        lr_pred = lr.predict(Xte_s)
        fold_metrics_lr.append(evaluate(yte, lr_pred, LABELS))
        # random forest
        rf = RandomForestClassifier(n_estimators=100, max_depth=5, max_features="sqrt",
                                    class_weight="balanced", random_state=SEED, n_jobs=-1)
        rf.fit(Xtr_s, ytr)
        rf_pred = rf.predict(Xte_s)
        fold_metrics_rf.append(evaluate(yte, rf_pred, LABELS))
        for r, yt, pl, pr in zip(te, yte, lr_pred, rf_pred):
            oof_lr.append({"clip": r.clip_name, "true": yt, "pred": str(pl)})
            oof_rf.append({"clip": r.clip_name, "true": yt, "pred": str(pr)})
    return {
        # fold-mean (kept for continuity) PLUS pooled macro-F1 recomputed from OOF,
        # which is the apples-to-apples number to compare against R1's pooled S-JEPA/RF.
        "logreg": {**aggregate_folds(fold_metrics_lr),
                   "pooled_macro_f1": pooled_macro_f1(oof_lr).macro_f1},
        "rf": {**aggregate_folds(fold_metrics_rf),
               "pooled_macro_f1": pooled_macro_f1(oof_rf).macro_f1},
    }


# ---------------------------------------------------------------------------
# E0-RF baseline (the study's Random Forest, on the locked folds)
# ---------------------------------------------------------------------------
def run_rf_baseline(registry, by_clip):
    fold_metrics = []
    oof = []          # pooled out-of-fold predictions
    for fold in registry["folds"]:
        tr = [by_clip[c] for c in fold["train_clips"]]
        te = [by_clip[c] for c in fold["test_clips"]]
        Xtr, ytr, _, _ = build_feature_matrix(tr, fps=15)
        Xte, yte, te_src, _ = build_feature_matrix(te, fps=15)
        pred = train_rf_and_predict(Xtr, ytr, Xte, seed=SEED)
        fold_metrics.append(evaluate(yte, pred, LABELS))
        for c, s, yt, yp in zip(fold["test_clips"], te_src, yte, list(pred)):
            oof.append({"clip": c, "source_id": s, "fold": fold["fold"],
                        "true": yt, "pred": str(yp)})
    return aggregate_folds(fold_metrics), fold_metrics, oof


def pooled_macro_f1(oof):
    yt = [r["true"] for r in oof]
    yp = [r["pred"] for r in oof]
    m = evaluate(yt, yp, LABELS)
    return m


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    completed_marker = OUT_DIR / "COMPLETED.json"
    if completed_marker.exists():
        print(f"Refusing to overwrite completed run at {completed_marker}. "
              f"Delete it to re-run.")
        return

    print("== Phase 0: provenance + E0 ==")
    records = load_index(CACHE_DIR)
    by_clip = records_by_clip(records)
    print(f"loaded {len(records)} records")

    prov = cache_manifest()
    print(f"cache_v0 combined sha256 = {prov['combined_sha256'][:16]}...")

    registry = build_registry(records)
    (OUT_DIR / "fold_registry.json").write_text(json.dumps(registry, indent=2))
    print(f"locked {registry['n_splits']}-fold registry (generation g1)")

    # RF baseline
    rf_agg, rf_folds, rf_oof = run_rf_baseline(registry, by_clip)
    rf_pooled = pooled_macro_f1(rf_oof)
    (OUT_DIR / "E0_RF_oof.json").write_text(json.dumps(rf_oof, indent=2))
    print(f"E0-RF fold-mean macro-F1 = {rf_agg['macro_f1']['mean']:.3f} "
          f"+/- {rf_agg['macro_f1']['std']:.3f} | pooled macro-F1 = "
          f"{rf_pooled.macro_f1:.3f}")

    # Shortcut controls
    control_results = {}
    for name, fn in CONTROLS.items():
        control_results[name] = run_control(name, fn, registry, by_clip)
        best = max(control_results[name]["logreg"]["macro_f1"]["mean"],
                   control_results[name]["rf"]["macro_f1"]["mean"])
        print(f"control {name:16s}: best fold-mean macro-F1 = {best:.3f}")

    results = {
        "generation": "g1",
        "labels_order": LABELS,
        "E0_RF": {
            "fold_mean": rf_agg,
            "pooled_macro_f1": rf_pooled.macro_f1,
            "pooled_confusion": rf_pooled.confusion,
            "pooled_per_class": rf_pooled.per_class,
        },
        "shortcut_controls": control_results,
    }
    (OUT_DIR / "E0_results.json").write_text(json.dumps(results, indent=2))

    import sklearn
    manifest = {
        "run_id": "phase0_g1",
        "git_rev": git_rev(),
        "git_dirty": _git_dirty(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "seed": SEED,
        "n_splits": N_SPLITS,
        "data_provenance": prov,
        "registry_file": "fold_registry.json",
        "results_file": "E0_results.json",
        "oof_file": "E0_RF_oof.json",
        "notes": "Source-grouped (provisional), NOT participant-disjoint. cache_v0 read-only.",
    }
    (OUT_DIR / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    completed_marker.write_text(json.dumps(
        {"status": "ok", "generation": "g1",
         "cache_sha256": prov["combined_sha256"]}, indent=2))
    print(f"\nOK: Phase 0 artifacts written to {OUT_DIR}")


if __name__ == "__main__":
    main()
