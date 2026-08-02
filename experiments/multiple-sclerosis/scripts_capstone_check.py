"""End-to-end pipeline check on the real cached data.

Runs the whole story quickly: pretrain S-JEPA on normal, fine-tune with ms+pd and
VICReg, then compare the classical RF branch against a linear probe on frozen
S-JEPA embeddings, using leakage-safe grouped k-fold. This is the working
reference that notebook 06 mirrors. Keep epochs small here; the notebook uses the
full profile.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_DIR))

from sjepa.config import get_config
from sjepa.data import load_index, grouped_kfold
from sjepa.masking import AnatomicalMaskSampler
from sjepa.models import build_model
from sjepa.train import train_sjepa
from sjepa.classical import build_feature_matrix, train_rf_and_predict
from sjepa.eval import evaluate, aggregate_folds, silhouette


def embed_records(model, records, cfg, device):
    """One embedding vector per video: mean over its windows of the masked pool."""
    from sjepa.data import sliding_windows

    sampler = AnatomicalMaskSampler(cfg.num_joints, cfg.num_time_tokens)
    tm = torch.from_numpy(sampler.target_mask).to(device)
    vecs, labels, srcs = [], [], []
    for r in records:
        seq = r.load_norm()
        wins = sliding_windows(seq, cfg.window_frames, cfg.window_stride)
        x = torch.from_numpy(wins).float().to(device)
        with torch.no_grad():
            e = model.embed(x, tm).mean(dim=0).cpu().numpy()
        vecs.append(e); labels.append(r.label); srcs.append(r.source_id)
    return np.stack(vecs), labels, srcs


def main():
    cfg = get_config("laptop", smoke=False)
    cfg.pretrain_epochs = 10
    cfg.finetune_epochs = 8
    device = "cpu"
    labels_order = ["normal", "ms", "pd"]

    records = load_index(EXP_DIR / "artifacts" / "keypoints")
    print(f"loaded {len(records)} records")

    rf_metrics, sj_metrics, sils = [], [], []
    for fold, (train_recs, test_recs) in enumerate(grouped_kfold(records, n_splits=5, seed=42)):
        # --- classical RF branch ---
        Xtr, ytr, _, _ = build_feature_matrix(train_recs, fps=cfg.target_fps)
        Xte, yte, _, _ = build_feature_matrix(test_recs, fps=cfg.target_fps)
        rf_pred = train_rf_and_predict(Xtr, ytr, Xte, seed=cfg.seed)
        rf_m = evaluate(yte, rf_pred, labels_order)
        rf_metrics.append(rf_m)

        # --- S-JEPA branch: pretrain normal, finetune all, then linear probe ---
        from sjepa.data import SequenceWindowDataset
        normal_train = [r for r in train_recs if r.label == "normal"]
        model = build_model(cfg, device=device)
        if normal_train:
            ds_norm = SequenceWindowDataset(normal_train, cfg.window_frames, cfg.window_stride)
            train_sjepa(model, ds_norm, cfg, epochs=cfg.pretrain_epochs, device=device)
        ds_all = SequenceWindowDataset(train_recs, cfg.window_frames, cfg.window_stride)
        train_sjepa(model, ds_all, cfg, epochs=cfg.finetune_epochs,
                    use_vicreg=True, class_aware_vicreg=True, device=device)

        Etr, ytr2, _ = embed_records(model, train_recs, cfg, device)
        Ete, yte2, _ = embed_records(model, test_recs, cfg, device)
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler().fit(Etr)
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf.fit(sc.transform(Etr), ytr2)
        sj_pred = clf.predict(sc.transform(Ete))
        sj_m = evaluate(yte2, sj_pred, labels_order)
        sj_metrics.append(sj_m)

        allE = np.vstack([Etr, Ete]); allY = ytr2 + yte2
        sils.append(silhouette(allE, allY))

        print(f"fold {fold}: RF acc {rf_m.accuracy:.3f} f1 {rf_m.macro_f1:.3f} | "
              f"S-JEPA acc {sj_m.accuracy:.3f} f1 {sj_m.macro_f1:.3f} | test n={len(test_recs)}")

    print("\n=== grouped k-fold mean+/-std ===")
    print("RF     :", aggregate_folds(rf_metrics))
    print("S-JEPA :", aggregate_folds(sj_metrics))
    print("silhouette (S-JEPA embeddings):", np.nanmean(sils).round(3))
    print("\nOK: full capstone pipeline runs end to end on real data")


if __name__ == "__main__":
    main()
