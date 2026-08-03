"""R1_repaired32: the bounded, frozen, repaired S-JEPA baseline on cache_v0.

This runs the corrected pipeline end to end on the locked ``g1`` fold registry and
saves pooled out-of-fold (OOF) predictions plus a run manifest. It is deliberately
modest (Phase 4 of the plan): 32-frame windows, the small 3-layer/96-width
encoder, PredictorV2 target positions, per-example stochastic graph-time masks,
label-free source-uniform SSL, a frozen mean-pool linear probe. The Random Forest
runs on the same folds so the comparison is paired.

Everything is a development estimate on a previously inspected, source-grouped (NOT
participant-disjoint) collection. It is never a confirmatory or clinical result.

Usage:
  python scripts_r1_repaired.py --fold-limit 1 --total-updates 1000 --seed 42 \
      --output-dir artifacts/runs/r1_smoke
Flags let a bounded run fit the compute budget; a full run drops --fold-limit.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

EXP_DIR = Path(__file__).resolve().parent.parent  # scripts/ -> experiment dir
sys.path.insert(0, str(EXP_DIR))

from sjepa.config import get_config  # noqa: E402
from sjepa.data import load_index, SequenceWindowDataset, sliding_windows  # noqa: E402
from sjepa.masking_v2 import sample_target_mask  # noqa: E402
from sjepa.models import build_model, pick_device  # noqa: E402
from sjepa.train_v2 import train_sjepa_v2, save_checkpoint_v2  # noqa: E402
from sjepa.classical import build_feature_matrix, train_rf_and_predict  # noqa: E402
from sjepa.eval import evaluate  # noqa: E402

LABELS = ["normal", "ms", "pd"]
CACHE_DIR = EXP_DIR / "artifacts" / "keypoints"
REGISTRY = EXP_DIR / "artifacts" / "eval" / "g1" / "fold_registry.json"


def git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=EXP_DIR,
                                       text=True).strip()
    except Exception:
        return "unknown"


def git_dirty() -> bool:
    try:
        return bool(subprocess.check_output(["git", "status", "--porcelain"],
                                            cwd=EXP_DIR, text=True).strip())
    except Exception:
        return True


def embed_records(model, records, cfg, device, target_mask):
    """One embedding per clip: mean over its windows of the masked-target token pool."""
    tm = torch.from_numpy(target_mask).to(device)
    vecs, labels, srcs, clips = [], [], [], []
    model.target_encoder.eval()
    for r in records:
        seq = r.load_norm()
        wins = sliding_windows(seq, cfg.window_frames, cfg.window_stride)
        x = torch.from_numpy(wins).float().to(device)
        with torch.no_grad():
            e = model.embed(x, tm).mean(dim=0).cpu().numpy()
        vecs.append(e); labels.append(r.label); srcs.append(r.source_id)
        clips.append(r.clip_name)
    return np.stack(vecs), labels, srcs, clips


def run_fold(fold, by_clip, cfg, device, total_updates, mask_ratio, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    train_recs = [by_clip[c] for c in fold["train_clips"]]
    test_recs = [by_clip[c] for c in fold["test_clips"]]

    # --- paired Random Forest on identical folds ---
    Xtr, ytr, _, _ = build_feature_matrix(train_recs, fps=cfg.target_fps)
    Xte, yte, te_src, _ = build_feature_matrix(test_recs, fps=cfg.target_fps)
    rf_pred = train_rf_and_predict(Xtr, ytr, Xte, seed=seed)

    # --- repaired label-free S-JEPA SSL on ALL training sources ---
    ds = SequenceWindowDataset(train_recs, cfg.window_frames, cfg.window_stride)
    model = build_model(cfg, device=device, repaired=True)
    state = train_sjepa_v2(model, ds, cfg, total_updates=total_updates, device=device,
                           mask_ratio=mask_ratio, seed=seed)

    # frozen probe on a fixed target-token pool (the clinical joints, as a stable readout)
    tm = sample_target_mask(cfg.num_joints, cfg.num_time_tokens,
                            np.random.default_rng(0), target_ratio=0.6)
    Etr, ytr2, _, _ = embed_records(model, train_recs, cfg, device, tm)
    Ete, yte2, te_src2, te_clip2 = embed_records(model, test_recs, cfg, device, tm)
    sc = StandardScaler().fit(Etr)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(sc.transform(Etr), ytr2)
    proba = clf.predict_proba(sc.transform(Ete))
    classes = list(clf.classes_)
    sj_pred = [classes[i] for i in proba.argmax(1)]

    # OOF rows (one per test clip)
    oof = []
    for i, (clip, src, yt) in enumerate(zip(te_clip2, te_src2, yte2)):
        prob_vec = {c: float(proba[i, classes.index(c)]) for c in LABELS if c in classes}
        oof.append({"clip": clip, "source_id": src, "fold": fold["fold"],
                    "true": yt, "pred_sjepa": sj_pred[i], "proba": prob_vec})
    # attach RF preds aligned by clip
    rf_by_clip = {c: str(p) for c, p in zip(fold["test_clips"], rf_pred)}
    for row in oof:
        row["pred_rf"] = rf_by_clip.get(row["clip"], None)

    diag = {
        "final_loss": float(state.losses[-1]),
        "emb_std_final": float(state.emb_std[-1]),
        "eff_rank_final": float(state.eff_rank[-1]),
        "teacher_drift_final": float(state.teacher_drift[-1]),
        "ema_half_life_steps": float(state.ema_half_life_steps),
        "n_train_sources": len(fold["train_sources"]),
        "total_updates": state.total_updates,
    }
    return oof, diag, model, state


def pooled_metrics(oof, pred_key):
    yt = [r["true"] for r in oof]
    yp = [r[pred_key] for r in oof]
    return evaluate(yt, yp, LABELS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold-limit", type=int, default=None,
                    help="run only the first N folds (bounded run)")
    ap.add_argument("--total-updates", type=int, default=1000)
    ap.add_argument("--mask-ratio", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--output-dir", type=str, required=True)
    args = ap.parse_args()

    out = EXP_DIR / args.output_dir if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "COMPLETED.json").exists():
        print(f"Refusing to overwrite completed run at {out}. Delete COMPLETED.json to re-run.")
        return

    device = args.device or pick_device()
    cfg = get_config("laptop", smoke=False)
    registry = json.loads(REGISTRY.read_text())
    records = load_index(CACHE_DIR)
    by_clip = {r.clip_name: r for r in records}

    folds = registry["folds"]
    if args.fold_limit is not None:
        folds = folds[:args.fold_limit]

    print(f"R1_repaired32 | device={device} | updates={args.total_updates} | "
          f"mask_ratio={args.mask_ratio} | folds={len(folds)}/{len(registry['folds'])}")

    t0 = time.time()
    all_oof, diags = [], []
    for fold in folds:
        oof, diag, model, state = run_fold(
            fold, by_clip, cfg, device, args.total_updates, args.mask_ratio, args.seed)
        all_oof.extend(oof)
        diags.append({"fold": fold["fold"], **diag})
        m_sj = evaluate([r["true"] for r in oof], [r["pred_sjepa"] for r in oof], LABELS)
        m_rf = evaluate([r["true"] for r in oof], [r["pred_rf"] for r in oof], LABELS)
        print(f"  fold {fold['fold']}: S-JEPA f1 {m_sj.macro_f1:.3f} | RF f1 "
              f"{m_rf.macro_f1:.3f} | loss {diag['final_loss']:.3f} "
              f"eff_rank {diag['eff_rank_final']:.1f} | n_test {len(oof)}")
        save_checkpoint_v2(out / f"sjepa_fold{fold['fold']}.pt", model, cfg, train_state=state)

    dt = time.time() - t0
    sj = pooled_metrics(all_oof, "pred_sjepa")
    rf = pooled_metrics(all_oof, "pred_rf")

    # Finite-value gate (Codex AR-1 P2): a collapse diagnostic that silently went
    # NaN must NOT certify a successful run.
    bad = [d for d in diags
           if not (np.isfinite(d["eff_rank_final"]) and np.isfinite(d["emb_std_final"])
                   and np.isfinite(d["final_loss"]))]
    if bad:
        (out / "FAILED.json").write_text(json.dumps(
            {"status": "non-finite diagnostics", "folds": [d["fold"] for d in bad]}, indent=2))
        raise SystemExit(f"R1 aborted: non-finite diagnostics in folds "
                         f"{[d['fold'] for d in bad]}; no COMPLETED marker written.")

    results = {
        "run": "R1_repaired32",
        "generation": "g1",
        "note": "development estimate; source-grouped (NOT participant-disjoint); cache_v0.",
        "folds_run": [f["fold"] for f in folds],
        "n_folds_total": len(registry["folds"]),
        "total_updates": args.total_updates,
        "mask_ratio": args.mask_ratio,
        "seed": args.seed,
        "sjepa_pooled": {"macro_f1": sj.macro_f1, "accuracy": sj.accuracy,
                         "per_class": sj.per_class, "confusion": sj.confusion,
                         "pd_recall": sj.per_class["pd"]["recall"]},
        "rf_pooled": {"macro_f1": rf.macro_f1, "accuracy": rf.accuracy,
                      "per_class": rf.per_class, "confusion": rf.confusion,
                      "pd_recall": rf.per_class["pd"]["recall"]},
        "diagnostics": diags,
        "wall_seconds": dt,
    }
    (out / "oof.json").write_text(json.dumps(all_oof, indent=2))
    (out / "results.json").write_text(json.dumps(results, indent=2))
    import sklearn
    manifest = {
        "run_id": out.name,
        "git_rev": git_rev(),
        "git_dirty": git_dirty(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "device": device,
        "cache": "cache_v0",
        "registry": "g1/fold_registry.json",
        "args": vars(args),
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    (out / "COMPLETED.json").write_text(json.dumps(
        {"status": "ok", "sjepa_macro_f1": sj.macro_f1, "rf_macro_f1": rf.macro_f1,
         "folds_run": [f["fold"] for f in folds]}, indent=2))

    print(f"\n== R1 pooled (folds {[f['fold'] for f in folds]}) ==")
    print(f"S-JEPA: macro-F1 {sj.macro_f1:.3f} acc {sj.accuracy:.3f} PD-recall {sj.per_class['pd']['recall']:.3f}")
    print(f"RF    : macro-F1 {rf.macro_f1:.3f} acc {rf.accuracy:.3f} PD-recall {rf.per_class['pd']['recall']:.3f}")
    print(f"wall {dt:.1f}s -> {out}")


if __name__ == "__main__":
    main()
