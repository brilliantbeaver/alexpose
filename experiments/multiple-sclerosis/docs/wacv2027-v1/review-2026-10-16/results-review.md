# Fresh review of numerical results and their interpretation

**Review date:** 16 October 2026. **Manuscript:** `docs/wacv2027-v1/wacv-v1-revision.tex`, SHA-256 `67a5ed3f42f3a01b40f458e8ee0e71d00f0f3516655390604cf4a796b86385b3`. Line references below apply to that file. Notebook cell positions are **one-based across both Markdown and code cells**; stable cell IDs are included where useful.

The current manuscript's reported classification scores, correct-clip counts, class percentages, split counts, and figure values agree with a fresh calculation from the retained prediction records and pose arrays. I found no material numerical error in the main manuscript. The important remaining consistency problem is in the companion notebooks: notebook 06 combines current October outputs with older September narrative, and notebook 03 retains a smaller mismatch in its diagnostic commentary. The paper correctly uses the corrected October RF result and labels the earlier result separately.

This review was read-only except for this report. It did not train an encoder, fit a classifier, change the manuscript or supplement, or rerun notebook training. Earlier reviews and their numerical approval were not used as primary evidence. I recomputed the classification results directly with exact rational weights, checked the underlying split lists and pose-array lengths, and then compared those fresh results to the manuscript and figure files.

## Findings requiring follow-up in the companion notebooks

**R1 — Notebook 06 still describes a different RF experiment from its saved code outputs.** This is a documentation consistency issue, not a reason to replace the paper's corrected scores.

- **Corresponding manuscript locations:** line 137 describes the corrected ankle features; lines 182 and 194–200 give the corrected and historical results; line 207 explains the change; line 217 gives the selected stages.
- **Conflicting notebook evidence:** `06_capstone_rf_vs_sjepa.ipynb`, cell 9 (`06-tutorial-systems`) still calls the duplicate-ankle error unresolved. Cell 15 (`06-tutorial-results`) calls RF 0.411 the “current extractor” and supplies the September RF fold scores. Cell 18 (`06-tutorial-ms`) says RF recognizes 5 of 29 MS clips; the corrected predictions recognize 4. Cell 12 (`06-tutorial-selection`) gives continued-stage validation scores 0.344 and 0.284 for folds 1 and 3 and a 216-second execution time; the October scores are 0.250462 and 0.332239, and its recorded time is 257.352 seconds.
- **Current notebook evidence:** cell 11 (`06_capstone_rf_vs_sjepa-08`, execution 5) identifies `capstone-20261009T060955236580Z` and logs the current training/selection execution. Cell 14 (`06_capstone_rf_vs_sjepa-10`, execution 6) reports RF source macro-F1 0.325184 and clip macro-F1 0.320940 from that run.
- **Raw artifact anchors:** the October and September `oof.json`, `results.json`, `provenance.json`, and corresponding feature-cache arrays identified below.
- **Action:** identify the older notebook commentary explicitly as historical, or synchronize its prose and tables with the October outputs. Preserve the older predictions and their provenance. The manuscript's present version choice is supported and should remain coherent across its tables and figures.

**R2 — Notebook 03's interpretation cell retains an earlier endpoint and date.** Its cell 18 (`03-tutorial-results`) labels the example September 20, 2026 and gives final effective rank 12.134796142578125. The currently saved executed cell 15 (`03_sjepa_model_and_pretrain_normal-09`, execution 6) prints 12.134888648986816, matching the primary Fold-0 initial checkpoint. Cell 19 (`03-tutorial-numerical-summary`, execution 8) also reports that current endpoint and exact first/last-50-update means, so the earlier sentence saying only final rank was printed numerically is now incomplete. Synchronize the commentary with the saved execution and acknowledge the numerical-summary cell. Both rank values round to 12.13; the main manuscript's line 215 is supported by the current outputs and requires no numerical change. The capstone endpoint checks and the limit on independently recovering full diagnostic histories are recorded below.

## Classification values independently verified

The primary run directory is:

```text
artifacts/runs/full-v1/9496e61b050f/laptop-1d09c8e1eea5/
  capstone-20261009T060955236580Z/
```

For each row of its `oof.json`, I used its recorded true label and each system's predicted label to construct a confusion matrix. Under source weighting, every clip from a source with `n` clips receives the exact fraction `1/n`. Class F1 is twice the diagonal count divided by the sum of its row and column; macro-F1 averages the three class scores. A separate calculation uses weight one per clip. This calculation does not import the project's scoring function or use the earlier verification JSON.

| System | Source macro-F1 | Clip macro-F1 | Correct clips | Raw clip accuracy |
|---|---:|---:|---:|---:|
| S-JEPA | 0.456624159 | 0.397033957 | 36/88 | 0.409091 |
| Corrected RF | 0.325184104 | 0.320940497 | 30/88 | 0.340909 |
| Mean pose | 0.451892876 | 0.409840252 | 37/88 | 0.420455 |
| Visibility | 0.317564628 | 0.318057047 | 29/88 | 0.329545 |
| Majority | 0.259398496 | 0.202125202 | 23/88 | 0.261364 |

**Manuscript checks:** abstract line 19; study claim line 32; metric definitions lines 169–171; Results lines 182–184; Table 3 lines 194–198; Discussion line 224; Conclusion line 239. These match the fresh results. The paper does not call macro-F1 accuracy. Its separate correct-clip column is appropriate: for example, S-JEPA's clip macro-F1 of 0.397 is different from its raw accuracy of 36/88, or 0.409.

The source-weighted S-JEPA–corrected-RF margin is 0.131440055; the corresponding clip-weighted margin is 0.076093460. The source-weighted margin over mean pose is 0.004731283, while mean pose has the higher clip-weighted score. The paper reports these relationships correctly. Its narrow statement that the fitted S-JEPA procedure exceeded the evaluated RF is supported; the results do not establish a general advantage over all handcrafted methods, which the paper explicitly avoids at line 231.

**Notebook anchors:** notebook 06 cells 11 and 14; notebook 06 cell 17 (`06_capstone_rf_vs_sjepa-11`) generates its confusion matrices from the same loaded results. **Implementation anchors:** `sjepa/full_experiment.py:67` defines source weights; lines 208–231 validate and pool OOF predictions.

### Fold comparisons and class errors

| Fold | S-JEPA source F1 | Corrected RF source F1 | Mean-pose source F1 |
|---|---:|---:|---:|
| 0 | 0.515263 | 0.315897 | 0.553613 |
| 1 | 0.362559 | 0.267191 | 0.499800 |
| 2 | 0.779598 | 0.323300 | 0.187037 |
| 3 | 0.315654 | 0.363604 | 0.509726 |
| 4 | 0.217267 | 0.256290 | 0.386487 |

Line 210 correctly states that S-JEPA exceeds corrected RF in three folds, mean pose exceeds S-JEPA in four, and S-JEPA ranges from 0.217 to 0.780. The manuscript retains the large Fold-2 result without portraying every fold as a win.

The raw S-JEPA confusion matrix, with true-label rows and predicted-label columns in Normal/MS/PD order, is:

```text
[[10, 6, 8],
 [ 8, 8,13],
 [10, 7,18]]
```

This supports all the raw counts in line 212: 10/24 Normal, 8/29 MS and 18/35 PD correctly classified, including 13 MS clips predicted as PD. The source-weighted recalls are 39.5833%, 42.8994% and 56.9048%; the corresponding corrected-RF recalls are 27.0833%, 23.0769% and 50.9524%. S-JEPA's source-weighted PD precision is 38.5787%. All values round to the percentages printed in line 212. They are source-weighted class recalls and precision, not counts of participants or raw clip percentages; the paragraph identifies the weighting and keeps the raw counts separate.

### Figure verification

All 15 fold scores and 27 source-weighted confusion cells in `wacv-v1-revision-assets/figure-values.json` match the fresh rational-weight calculation within `1e-14`. `build_figures.py:42` plots the five folds in registry order with small horizontal marker offsets. Lines 55–57 normalize **rows**, then apply the same `vmin=0, vmax=100` scale to all three confusion matrices. The F1 plot uses a 0–1 vertical range and does not connect folds as a time series. These choices match the caption at manuscript line 176.

The current rendered `results.png` shows the mathematically correct 43.8% for mean-pose Normal→Normal, derived from 7/16 = 43.75%. Its annotation formatting now removes floating-point representation noise before rounding (`build_figures.py:60–61`). All other cell annotations match the independently calculated percentages. The historical RF is excluded from the primary graphic and remains labeled in the table.

## Historical RF attribution independently verified

The historical run is the adjacent directory `capstone-20260921T011305448893Z`. Its `oof.json` gives RF source macro-F1 **0.411262910**, clip macro-F1 **0.397315740**, and **37/88** correct clips. Its source-weighted difference from S-JEPA is **0.045361249**. This supports Table 3's historical row at line 200 and the prose at line 207, including the statement that both clip F1 values round to 0.397 even though historical RF is slightly higher before rounding.

Fresh comparison of the two prediction files found **12 changed RF labels** and **no changed S-JEPA, mean-pose, visibility or majority labels**. Both configurations use seed 42 and the same registry and pose-input checksums. The paper appropriately does not treat the executions as independent training-seed or dataset replications.

I reconstructed the feature-cache keys from each `provenance.json` rather than taking a prior audit's mapping:

- Historical features: `artifacts/cache/capstone/features/95d1e7aa1ada202aab3a00520ce4b71c214176be987051a860637e177fa66523/`.
- Corrected features: `artifacts/cache/capstone/features/b2a10814f6ece4edb4eeb05cde8ec0be35a17cf5cd9244f9139716865a54af11/`.

All 88 RF rows differ in exactly `left_ankle_range`. Historical left/right ranges duplicate each other in 88/88 rows; corrected ranges do not duplicate in any row. All remaining RF and comparison-feature values are identical. All five corrected training partitions contain 15 varying RF fields out of the 82-field container, supporting line 137.

The content hash of the current non-test `sjepa` and `ambient` source files reproduces the October provenance hash. Reversing only the corrected ankle assignment **in memory** reproduces the September hash. This verifies the one-line implementation difference without editing source files or fitting another forest. The baseline score change is a consequence of the baseline correction, not an additional learning gain by S-JEPA; line 207 describes it correctly.

## Dataset, training size and split boundaries

I counted the current MP4 files, examined each retained pose array's length and shape, and checked the registry's clip/source lists. All rows in Table 1 (lines 78–82) match:

| Label | Raw clips | Usable clips | Sources | Padded clips | Windows |
|---|---:|---:|---:|---:|---:|
| Normal | 26 | 24 | 16 | 5 | 59 |
| MS | 30 | 29 | 13 | 0 | 171 |
| PD | 35 | 35 | 12 | 10 | 128 |
| Total | 91 | 88 | 41 | 15 | 358 |

Windows were counted from the actual `keypoints_norm` arrays using 32 frames and stride 16, with one window for a shorter retained clip. This also verifies lines 96 and 233. Source `tsOMPBS277Q` contributes 13 usable MS clips, supporting line 184.

Every entry in Table 2 (lines 157–161) matches `artifacts/eval/full-v1/fold_registry.json`: training always has 24 sources and 45–58 clips; validation has 8–9 sources; test has 8–9 sources. The additional development-source count of 32–33 at line 148 is correct. Within every fold, the three source sets are disjoint and together contain all 41 retained sources. The full test assignments contain every usable clip exactly once. Each of the 88 OOF rows agrees with its registry label, source, and test fold.

The splitting implementation uses a stratified five-fold split on one row per source, followed by the first split of a stratified four-fold splitter on the remaining sources (`sjepa/splits.py:86–119`). It does not split overlapping windows independently. The wording at lines 143–148 accurately describes one inner validation holdout per outer fold and distinguishes source separation from participant separation.

**Notebook anchors:** notebook 00 cell 9 (`00_overview_and_video_gallery-08`) supplies the raw collection counts; notebook 01 cells 14 and 16 (`01_pose_extraction_from_raw_video-12` and `-14`) report the exclusions and retained caches; notebook 06 cell 7 (`06_capstone_rf_vs_sjepa-06`) prints the registry partition table. Notebooks 02–05 load the same registry for their Fold-0 examples.

## Training, selection and representation claims

The primary run's `results.json` records 800 initial and 400 continuation updates. Each of its ten checkpoint payloads carries the correct fold-specific training-source and training-clip lists, stage-local update count, and seed 42. `sjepa/full_experiment.py:125–183` builds a fresh model for each fold, fits each probe on training embeddings, selects using validation scores, and only then computes S-JEPA test predictions. `train_checkpoint` invokes a fresh training loop without passing optimizer-resume state; `sjepa/train_v2.py:150–171` creates the optimizer, center and seeded streams. This supports lines 102 and 124, including the distinction from uninterrupted 1,200-update training and the absence of a validation-data refit.

The primary run's recorded validation scores are:

| Fold | Initial | Continued | Selected |
|---|---:|---:|---|
| 0 | 0.293979 | 0.276105 | Initial |
| 1 | 0.490618 | 0.250462 | Initial |
| 2 | 0.324930 | 0.257210 | Initial |
| 3 | 0.266008 | 0.332239 | Continued |
| 4 | 0.375309 | 0.559710 | Continued |

The selected stage is the larger recorded validation score in each fold, and the saved final minibatch loss is lower after continuation in all five. Lines 124 and 217 match these records. Notebook 04 cell 14 (`04_progressive_finetune_ms_pd_vicreg-11`) independently displays the Fold-0 validation choice; notebook 06 cell 11 logs the complete five-fold execution. These choices were checked against recorded validation values and code order, not by refitting probes.

Line 215's first/last-50-update means are present in notebook 03 cell 19 (`03-tutorial-numerical-summary`, execution 8): loss **3.858368 → 1.022220**. The cell's source explicitly computes those two means from its current training history. Its final rank is **12.134889**; the primary initial-stage endpoint range is **10.895738–12.134889**, matching the manuscript's rounded range. The notebook's Fold-0 initial checkpoint and the primary run's Fold-0 initial checkpoint have identical contents in all 351 serialized archive members after removing the archive-root directory names. This ties the notebook summary to the primary model payload.

The stage configurations also confirm the numerical architecture description at lines 100–102, batch size 32 at line 120, and 96-dimensional readout at line 130. The linear classifier's `C=1`, class balancing and training-only scaler agree with `sjepa/full_experiment.py:56–64`, supporting line 132. The old epoch/VICReg fields printed in notebook configuration summaries do not control this executed update-based training path, and the manuscript does not claim they do.

For the masking statistic at line 105, I generated 5,000 masks with the current sampler, seed 42, 33 landmarks, eight time blocks, and target ratio 0.6. The fresh hidden-token mean is **0.632043939**, matching 63.2%. This is a mask-sampling diagnostic, not a new trained-model experiment. Notebook 02 cells 10, 12 and 14 anchor tokenization, the 12 preferred landmarks and the stochastic sampler; its smaller 512-mask display is not the source of the paper's 5,000-mask estimate.

The manuscript consistently separates the trained representation's observed utility from an isolated benefit of training or temporal order (lines 19, 184, 215, 226 and 239). Notebook 05's training-only projections and silhouette outputs do not establish held-out disease separation; the main manuscript does not use them for that purpose. Its figures show the actual held-out classifiers rather than interpreting training-space clusters as diagnostic evidence.

## Limits of this review

The main numerical claims are supported by the retained artifacts, but this is an audit of an existing experiment, not a retraining replication. Per-update diagnostic histories are not stored in the capstone checkpoints, so the first/last-50 means can be checked against the notebook's saved numerical summary and matching checkpoint payload, not independently regenerated from a full retained loss vector. Validation-stage choices were checked against their saved scores; the held-out classification scores were independently recalculated from all prediction rows.

The checks establish source separation within the recorded folds. They cannot establish participant independence, verify clinical labels, rule out repeated or reposted footage, or prove that the collection was untouched during all prior development. These limitations are compatible with the manuscript's explicit source/participant distinction and its stated clinical and causal boundaries. They do not require changing the verified numerical results or presenting the study as a statistical null finding.
