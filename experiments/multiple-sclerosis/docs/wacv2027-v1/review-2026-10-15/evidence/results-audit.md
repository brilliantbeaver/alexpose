# Independent results audit

Reviewed 15 October 2026. This audit uses saved per-clip predictions, split registries, feature arrays, checkpoint tensors, run manifests, notebook outputs, and current source code as primary evidence. Previous manuscript audits were not used to generate the findings. All numerical checks are reproducible with [verify_results.py](verify_results.py); the complete machine-readable output is [results-verification.json](results-verification.json). The verification script passed without training a model, fitting a classifier, modifying predictions, or editing the original notebooks.

## The most important correction: the draft reports an obsolete RF implementation

The supplied `docs/wacv2027-v1/wacv-v1.pdf` reports S-JEPA 0.457, RF 0.411, and mean pose 0.452. Its RF value belongs to the September 21 experiment, which accidentally duplicated the right-ankle range in the left-ankle feature. The October 9 experiment contains the corrected left-ankle range, produces RF 0.325, and leaves S-JEPA's held-out predicted labels unchanged. The corrected experiment should provide the paper's primary comparison; the older result should remain traceable as a superseded implementation.

The two complete full-data runs are:

| Role | Retained run directory under `artifacts/runs/full-v1/9496e61b050f/laptop-1d09c8e1eea5/` | RF source macro-F1 | RF clip macro-F1 |
|---|---|---:|---:|
| Historical, duplicated ankle feature | `capstone-20260921T011305448893Z` | 0.411262910 | 0.397315740 |
| Corrected, recommended primary results | `capstone-20261009T060955236580Z` | 0.325184104 | 0.320940497 |

This attribution is stronger than an inference from timestamps:

1. The current content hash of every non-test Python file in `sjepa/` and `ambient/` is `b0bcda1d8fc911cf92b270d09a8ad6b3c60732fb9d45c9a73310dbd1e94e1cab`, exactly matching the October run's `provenance.json`.
2. Reversing only `ambient/classification/features.py:912` in memory from `left_ankle_stats` to `right_ankle_stats` yields `476a9f57e6863889c8c9b24d6f6af2b489456b35bcfcec40c9e9af3c56128779`, exactly matching the September run. No source file was changed by this check.
3. Reconstructing each feature-cache key from that run's provenance locates September cache `artifacts/cache/capstone/features/95d1e7aa1ada202aab3a00520ce4b71c214176be987051a860637e177fa66523/` and October cache `artifacts/cache/capstone/features/b2a10814f6ece4edb4eeb05cde8ec0be35a17cf5cd9244f9139716865a54af11/`. Across all 88 clips, the RF arrays differ in exactly one column, `left_ankle_range`. September's left and right ranges are identical in 88/88 clips; October's are distinct in 88/88. Every other RF feature, every visibility feature, and every mean-pose feature is unchanged.
4. The actual Git change at `d79441db188f7abad916243d50a83ac10aad6dc8` documents the same one-line correction. The exact code-hash reconstruction and stored arrays independently confirm that history.

There are still 15 nonconstant RF columns in every fold after correction: six angle means, three absolute left/right mean differences, and six angle ranges. In the historical implementation, two of those 15 varying columns held the same values. Consequently, the draft's description of 15 varying columns was numerically true but did not disclose that one intended measurement was absent. The baseline is a narrow joint-angle summary with 100 trees and maximum depth five; it is not an exhaustive or tuned classical gait-analysis system. RF summarizes lower-limb angles, while S-JEPA takes all 33 landmarks and visibility. Different classifier families and S-JEPA-only checkpoint selection also prevent attributing the comparison purely to feature learning.

The correction changes 12 of the 88 RF test predictions. It lowers the baseline score; it does not improve the learned model. This distinction belongs in the provenance note so that the larger revised margin is not presented as additional progress by S-JEPA.

## Verified primary results

The independent scoring code constructs the confusion counts directly from `oof.json`, then calculates each class's precision, recall and F1, and averages the three F1 scores. Each source receives total weight one, divided among its clips. All saved scalar metrics, class metrics, confusion cells and fold summaries match independent recalculation within `1e-12` for both complete runs. Every retained pose-cache checksum matches the registry, all 88 clips appear once, and every row's label, source and test fold agrees with the registry.

| System | Pooled source macro-F1 | Pooled clip macro-F1 | Fold mean ± population SD | Correct clips |
|---|---:|---:|---:|---:|
| Frozen S-JEPA + linear classifier | 0.456624 | 0.397034 | 0.438068 ± 0.195980 | 36/88 |
| Corrected angle-feature RF | 0.325184 | 0.320940 | 0.305256 ± 0.039217 | 30/88 |
| Mean pose + linear classifier | 0.451893 | 0.409840 | 0.427332 ± 0.132210 | 37/88 |
| Visibility + linear classifier | 0.317565 | 0.318057 | 0.284918 ± 0.104069 | 29/88 |
| Majority class | 0.259398 | 0.202125 | 0.159394 ± 0.021986 | 23/88 |

The observed S-JEPA–RF margin is **0.131440** under equal-source scoring and **0.076093** under equal-clip scoring. These are F1 differences, not percentage changes in accuracy. S-JEPA has the largest source-weighted pooled score among the five evaluated procedures; mean pose has the largest clip-weighted score. The source-weighted S-JEPA–mean-pose margin is 0.004731. F1 is calculated after pooling counts, so it is not interchangeable with averaging the fold scores or counting correct clips.

| Fold | S-JEPA | Corrected RF | Mean pose | Visibility | Majority |
|---|---:|---:|---:|---:|---:|
| 0 | 0.515263 | 0.315897 | 0.553613 | 0.235897 | 0.166667 |
| 1 | 0.362559 | 0.267191 | 0.499800 | 0.222222 | 0.181818 |
| 2 | 0.779598 | 0.323300 | 0.187037 | 0.272200 | 0.133333 |
| 3 | 0.315654 | 0.363604 | 0.509726 | 0.488406 | 0.181818 |
| 4 | 0.217267 | 0.256290 | 0.386487 | 0.205866 | 0.133333 |

S-JEPA exceeds the corrected RF in three of five folds and mean pose in one. The performance range is large. This can be reported plainly without organizing the paper around a significance test.

For Normal, MS and PD respectively, source-weighted class F1 is 0.466604, 0.443438 and 0.459831 for S-JEPA, compared with 0.343558, 0.223662 and 0.408332 for corrected RF. S-JEPA correctly labels 10/24 Normal, 8/29 MS and 18/35 PD clips; corrected RF labels 7/24, 4/29 and 19/35 correctly. The S-JEPA MS row contains 8 predictions of Normal, 8 of MS, and 13 of PD. Its raw MS recall is 8/29, while its source-weighted MS recall is 0.428994. These denominators must remain distinct.

Corrected RF source-weighted confusion counts, in Normal/MS/PD order, are:

```text
[[4.333333, 7.833333, 3.833333],
 [2.000000, 3.000000, 8.000000],
 [2.892857, 2.992857, 6.114286]]
```

Normalized within each true-label row, the corresponding percentages for the paper's confusion figure are:

```text
[[27.083333, 48.958333, 23.958333],
 [15.384615, 23.076923, 61.538462],
 [24.107143, 24.940476, 50.952381]]
```

## How small was the training set?

The collection contains 88 usable clips from 41 source videos, but each fitted encoder and classifier used only **24 training sources**. Across the five folds, those sources supplied 45–58 clips and 190–238 distinct windows. Additional validation sources were used to choose between two training stages; they were not used to fit the encoder, scaler, or classifier. The complete development procedure therefore draws on 32–33 sources per fold when training and validation are counted together.

| Fold | Training sources / clips / windows | Validation sources / clips | Test sources / clips |
|---|---|---|---|
| 0 | 24 / 51 / 207 | 8 / 19 | 9 / 18 |
| 1 | 24 / 58 / 226 | 9 / 17 | 8 / 13 |
| 2 | 24 / 54 / 238 | 9 / 17 | 8 / 17 |
| 3 | 24 / 53 / 224 | 9 / 21 | 8 / 14 |
| 4 | 24 / 45 / 190 | 9 / 17 | 8 / 26 |

The claim “trained using a small video collection” is supported. “A few-shot model,” “sample efficient,” or a quantified advantage from small-data training would need a learning curve or a matched data-budget comparison. Windows overlap, and batches repeatedly sample windows; 6,000 optimization updates across folds do not create additional independent videos. Source identifiers are recording identifiers, not verified people.

Primary locations: `artifacts/eval/full-v1/fold_registry.json`; notebook 06 zero-based cell 6, ID `06_capstone_rf_vs_sjepa-06`; `sjepa/full_experiment.py:91` and `sjepa/train_v2.py:68`.

## Learning evidence and its interpretation

The trained encoder produces varied 96-dimensional representations that support held-out label prediction with a separately fitted linear classifier. The supervised class labels are absent from the encoder's training loss. The linear classifier, class-stratified splits, and validation-based checkpoint choice do use labels, so the complete evaluation is not label-free.

The corrected run's Fold-0 log starts at feature-prediction loss 6.3014 and its saved endpoint is 1.097048759. Final initial-stage effective ranks are 10.895738–12.134889; continuation-stage ranks are 9.558105–11.638123. These measurements show that the learned features have not all become identical. They are training diagnostics, not measurements of disease-specific information. The retained capstone checkpoints store optimizer and random-number state, not complete per-update loss histories; notebook 06 retains rounded log samples every 80 updates in the first stage and every 40 in the continuation. Endpoint diagnostics are retained in each stage manifest and `results.json`.

| Fold | Initial loss | Continued loss | Initial rank | Continued rank | Initial validation F1 | Continued validation F1 | Selected |
|---|---:|---:|---:|---:|---:|---:|---|
| 0 | 1.097049 | 0.617744 | 12.134889 | 11.419020 | 0.293979 | 0.276105 | Initial |
| 1 | 1.130461 | 0.967042 | 11.721886 | 11.638123 | 0.490618 | 0.250462 | Initial |
| 2 | 0.986413 | 0.879909 | 12.113443 | 11.129522 | 0.324930 | 0.257210 | Initial |
| 3 | 1.083167 | 1.031109 | 10.895738 | 9.558105 | 0.266008 | 0.332239 | Continued |
| 4 | 0.898562 | 0.717982 | 11.156235 | 10.358450 | 0.375309 | 0.559710 | Continued |

Each fold trains for 800 updates and then 400 additional updates; continuation retains network weights but restarts the optimizer, running center, schedules, and seeded sampling streams. The initial checkpoint is selected in folds 0–2 and the continued checkpoint in folds 3–4. All ten checkpoints' embedded training-source lists match the registry. Selection does not fit the final classifier on validation clips. These facts support a clear workflow description rather than a claim about a single uninterrupted 1,200-update optimization or progressive class-aware fine-tuning.

Suggested central interpretation:

> Using 45–58 training clips from 24 source videos per fold, the trained S-JEPA encoder produced representations that supported three-class prediction with a simple linear classifier. The resulting system exceeded the corrected handcrafted joint-angle Random Forest on both source-weighted macro-F1 (0.457 versus 0.325) and clip-weighted macro-F1 (0.397 versus 0.321). This provides an encouraging demonstration of learned skeleton representations on a small video collection. Mean pose remained competitive, and an untrained-encoder comparison was not included, so the present results do not isolate how much of the performance came from self-supervised training or from temporal information.

This says what the representation supports while keeping the missing controls visible. A causal statement that “self-supervision improved classification,” a claim that temporal dynamics caused the gain, or a claim that anatomical masking improved classification is not established. An untrained encoder with identical readout and classifier would assess the first claim; a matched temporal-order intervention would address the second; a masking comparison would address the third. None is a completed experiment in the retained result families. The RF comparison concerns the evaluated complete procedures, not a universal advantage of learned features over handcrafted features.

## The two full-data runs are not two independent-seed replications

Both execution manifests report all five folds as `computed`, with different cache keys and environment fingerprints. September reports 215.981 seconds; October reports 257.352 seconds. Every corresponding model tensor differs numerically between dates, and all stored training/validation embedding arrays differ. For example, the largest absolute target-encoder tensor difference ranges from 0.000138 to 0.010933 across corresponding checkpoints. Two continuation validation scores change. All S-JEPA test hard labels nevertheless match, as do mean-pose, visibility and majority labels.

The different bytes therefore do not arise solely from checkpoint serialization, and the evidence does not support describing October as a simple copy of September's cached models. The retained evidence is consistent with fresh same-seed executions under changed operating-system versions. Both use seed 42 and identical source folds; they do not constitute independent-seed replication or additional test data. The exact reason for small numerical differences cannot be established from the manifests alone. Do not claim measured stability across seeds.

## Uncertainty suitable for a supplement

The verification script independently generated 10,000 paired, label-stratified source-bootstrap draws using NumPy PCG64 seed 42. It sorts source IDs, generates all Normal draws first, followed by MS and PD, and gives each drawn source total weight one shared across its clips. All systems use the same draws. The 2.5th and 97.5th percentiles use linear interpolation. The resulting corrected-run intervals are:

| Quantity | Observed value | Conditional 95% interval |
|---|---:|---:|
| S-JEPA source macro-F1 | 0.456624 | [0.331344, 0.580095] |
| Corrected RF source macro-F1 | 0.325184 | [0.207430, 0.444719] |
| Mean-pose source macro-F1 | 0.451893 | [0.324058, 0.572948] |
| S-JEPA minus corrected RF | 0.131440 | [−0.046030, 0.308832] |
| S-JEPA minus mean pose | 0.004731 | [−0.167046, 0.174485] |

The intervals describe variation under source resampling while keeping the fitted predictions fixed. They do not include retraining or model-selection variation, and they assume that source videos are independent units. Their modest numerical differences from older bootstrap tables reflect an explicitly specified new random stream. These intervals can remain in supplementary material while the main Results reports the observed comparison and fold variation. Removing significance-test language is reasonable; implying established superiority in a broader population would not be.

## Earlier results must remain separate

`artifacts/runs/r1_g1_1k_s42/oof.json` contains 47 clips from 35 sources in an earlier cache and split generation, with S-JEPA pooled clip macro-F1 0.438402 and RF 0.666573. Both are independently reproduced. Its RF predictions match `artifacts/eval/g1/E0_RF_oof.json` exactly. `artifacts/capstone_results.json` is an index into that historical family, not the full-data result in the current paper. Its displayed “chance (3 classes) = 1/3” is not a general chance level for macro-F1. The earlier experiment has different clips, split construction and update budgets; neither numerical improvement nor decline relative to it can be attributed solely to the present learning method. The incomplete `capstone-20260921T005254795528Z` directory has no complete OOF/result files and cannot supply a pooled comparison.

## Exact locations that need synchronized revision

| Location in supplied PDF / notebook | Required correction |
|---|---|
| Abstract, p. 1 | Change RF 0.411 to corrected 0.325; make the positive claim about trained frozen representations plus the fitted classifier. |
| Section 5.1, p. 5 | Describe the corrected 15 angle summaries; briefly identify the archived duplicate-ankle error and corrected evaluation. Preserve narrower landmark coverage and fixed RF settings. |
| Section 6.1 and Figure 3, pp. 5–6 | Use October endpoint losses and ranks; continued-rank range becomes 9.558–11.638. Fold-1 and Fold-3 continuation validation values are 0.250462 and 0.332239. |
| Section 6.2 and Table 3, p. 6 | Source gap becomes 0.131440 and clip gap 0.076093. RF correct count becomes 30/88. Replace RF row with 0.325 / 0.321 / 0.305 ± 0.039. All other systems' pooled values stay as shown above. |
| Section 6.3, p. 6 | RF fold-win statement becomes S-JEPA ahead in three folds. Fold-2 RF becomes 0.323300. Move detailed uncertainty procedure/intervals to a supplement if desired, using corrected predictions. |
| Figure 4(a,c), p. 7 | Redraw RF fold scores and RF row-normalized confusion percentages from the corrected run. S-JEPA and mean-pose predictions are unchanged. |
| Section 7.1 and Conclusion, pp. 7–8 | Replace all 0.046 RF gaps with 0.131; focus on useful trained representations on 24 training sources per fold, while retaining the missing untrained/time-order controls and collection-label limits. |
| Notebook 06 cell 10, ID `06_capstone_rf_vs_sjepa-08`, saved output | Correctly points to October, 257.4 seconds, and new validation results. Treat this execution output as current. |
| Notebook 06 cell 11, ID `06-tutorial-selection` | Still reports September Fold-1/Fold-3 continuation scores and 216 seconds. Needs October values. |
| Notebook 06 cell 13, ID `06_capstone_rf_vs_sjepa-10`, saved output | Correctly reports RF 0.325184 and October output path. |
| Notebook 06 cells 8, 14, 17–19 | Prose still treats duplicate-ankle extraction as unresolved or uses RF 0.411 / 5 correct MS clips. Current correct values are RF 0.325 / 4 correct MS clips; the old fold table must also be updated. All cell references use zero-based positions across both markdown and code cells. |

The supplied PDF is a coherent report of the historical September predictions except that it does not disclose the baseline error. The notebook now mixes October outputs with September narrative. Correcting the manuscript requires choosing one coherent experiment version across all text, tables, figures, diagnostics, and supplementary uncertainty—not selecting favorable numbers independently from each run.
