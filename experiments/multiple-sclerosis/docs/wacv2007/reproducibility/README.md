# Anonymous metric reproduction packet

This packet reproduces the reported classification metrics from saved held-out predictions. It is self-contained and requires Python 3 with only its standard library. It does not include source videos, pose arrays, participant identities, trained weights, or a mapping back to original recording identifiers.

Run from this directory:

```sh
python3 verify_metrics.py
```

The verifier checks file hashes, 88 unique clips, 41 source groups, collection-label counts, all five source-grouped training/validation/test splits, and exactly one held-out prediction for each clip. It independently recomputes pooled and per-fold macro-F1, accuracy, and confusion matrices and compares them with the expected values. It also verifies the 12 changed Random Forest predictions between the two recorded executions.

## Files

| File | Contents |
|---|---|
| `heldout_predictions.csv` | 88 rows with anonymous clip/source IDs, collection label, held-out fold, all five primary predictions, and earlier Random Forest predictions. |
| `splits.json` | Training, validation, and test source/clip assignments for each of five folds. |
| `expected_metrics.json` | Expected clip-weighted and source-weighted pooled scores and confusion matrices; source-weighted results for each fold. |
| `training_diagnostics.csv` | Exact six-decimal values displayed in saved notebook 03, cell 18, for fold 0's 800-update training stage. |
| `provenance.json` | Immutable run timestamps, original registry/data/code fingerprints, active model settings, and training/readout details. |
| `verify_metrics.py` | Independent standard-library metric and split verifier. |
| `manifest.json` | SHA-256 checksums of the other packet files. |

`S001`–`S041` identify source recordings and `C001`–`C088` identify clips. They are pseudonyms, and recording IDs do not establish unique people. Fold indices are 0–4; manuscript figures may display the same folds as 1–5. Collection labels (`normal`, `ms`, `pd`) are retained as dataset categories and are not independent clinical diagnoses.

## Primary and earlier results

The primary execution is dated **2026-10-09T06:09:55.236580Z**. It gives source-weighted macro-F1 0.456624159096 for S-JEPA and 0.325184104104 for the corrected Random Forest, an observed difference of 0.131440054992. Mean-pose, visibility, and majority-label results are also included.

The earlier execution is dated **2026-09-21T01:13:05.448893Z**. Its Random Forest feature matrix duplicated right ankle range in the left ankle range field. Correcting that field changed 12 of 88 RF predictions. Earlier RF predictions and expected metrics are retained for transparent reconciliation; all other systems' held-out decisions were the same across the two recorded executions. The primary comparison consistently uses the corrected execution.

For source-weighted scoring, each clip receives weight equal to one divided by the number of clips from its source. Each source therefore contributes total weight one. Weighted confusion counts are pooled before computing F1 for each of the three labels, then those three F1 values are averaged. This is not majority voting by source, and pooled macro-F1 is not the average of fold macro-F1. Clip-weighted results give each clip weight one. A class with an undefined F1 receives zero.

## Training diagnostics and reproduction scope

The four diagnostic rows reproduce the saved **notebook03c18** output: mean loss, effective rank, embedding standard deviation, and teacher/student drift over the first and last 50 updates, plus the final update. These are training-only diagnostics, with different aggregation rules:

- Feature-prediction loss averages cross-entropy over the masked target tokens within each window, then over windows in the batch. It does not pool all teacher tokens into an embedding.
- Effective rank and embedding standard deviation use the **all-token teacher pool**: the updated teacher's features are averaged over all token positions to form one vector per window. Effective rank uses singular values of the centered batch; standard deviation averages the sample standard deviations across feature dimensions.
- Teacher/student drift uses model parameters, without feature pooling. After the optimizer step and before the teacher's moving-average update, it averages `1 − cosine similarity` equally across corresponding parameter tensors. The denominator is clamped to avoid division by zero; consequently, initially zero tensors can inflate this diagnostic.

Downstream classification instead averages a fixed selection of teacher output positions and then windows within a clip. The diagnostic table contains rounded saved output, not a full learning-curve history or a newly trained run. Its `pooling` field states the aggregation rule for each diagnostic.

This packet allows independent **metric reproduction**, not full model retraining. Reproducing pose extraction and training would additionally require the original data and preprocessing, the implementation, model states, and a compatible software environment. No public repository or data release is claimed by this local packet. Its verification concerns stored predictions and split structure; it cannot verify clinical labels, establish participant independence, or isolate the causal benefit of pretraining or temporal order.
