# Retained notebook evidence checked on 13 September 2026

This record supports the September progress report and slide revision. It distinguishes results retained in the seven top-level notebooks from results imported from older frozen experiment artifacts, and checks important claims against the present implementation. No training was rerun and no notebook, checkpoint, or experiment result was overwritten for this review. Notebook cells are identified by their zero-based position in the notebook JSON.

The current notebooks retain execution outputs but do not retain cell execution timestamps. Therefore, 13 September is the date of this evidence check, not an asserted execution date. A notebook's output is evidence of its retained run; it does not by itself prove that every part of the current implementation produced that output.

## 1. Evidence hierarchy and current execution status

| Notebook | Evidence retained | Scope and interpretation |
|---|---|---|
| 00: overview and video gallery | Counts and one video display per label executed. Cell 3 retains output despite a null execution count. | The raw collection contains 49 clips from 37 source video identifiers. Its roadmap text still describes superseded fixed masking, normal-only pretraining, and class-aware VICReg. |
| 01: pose extraction | Example extraction, cleaning, normalized animation, cache inventory, and shape assertions executed. | The retained cache contains 47 usable clips from 35 source identifiers. Existing cache files are reused; this output does not imply all 47 videos were freshly extracted in the retained run. |
| 02: masking and tokenization | Configuration, landmark table, mask-bank diagnostics, and illustrative mask animation executed. | Current stochastic masking is demonstrated mechanically. The checks establish sampler coverage, not a clinical benefit or a symmetry result. |
| 03: model and label-free pretraining | Real laptop-profile training on fold 0's training sources, 800 optimizer updates, retained loss/rank diagnostics and plots. | This is training evidence from a single fold, with no diagnosis label used in its SSL objective. The historical filename still says “pretrain_normal,” but the code uses all three labels' training sources. |
| 04: additional training and supervised probes | 400 additional updates from the notebook-03 weights; fold-0 probe scores 0.600 before and 0.600 afterward. | Only the frozen linear classification heads use training labels. The historical filename mentions progressive MS/PD VICReg, which the current training code does not run. |
| 05: representation visualization | Setup/display cells ran; embedding cell 7 failed with `FileNotFoundError` for `artifacts/sjepa_ssl.pt`. Cells 9, 11, 12, and 14 have no execution counts or outputs. | No current retained nuisance matrix, t-SNE, UMAP, or silhouette result. Both required checkpoints now exist on disk, so the retained failure does not mean the files are currently absent. |
| 06: capstone comparison | Imports completed five-fold R1/E0 metrics; executes a separate 500-update fold-0 demonstration; displays frozen confusion matrices; writes a scoreboard. | The five-fold headline is an imported frozen result, not training newly performed by the notebook. The 500-update example is a separate single-fold result. |

## 2. Dataset accounting and preserved provenance

| Label | Raw clips | Raw source IDs | Cached clips | Cached source IDs | Cached frames | 32-frame windows, stride 16 |
|---|---:|---:|---:|---:|---:|---:|
| Normal | 20 | 17 | 19 | 16 | 1,840 | 92 |
| MS | 12 | 12 | 11 | 11 | 2,918 | 167 |
| PD | 17 | 8 | 17 | 8 | 3,935 | 222 |
| Total | 49 | 37 | 47 | 35 | 8,693 | 481 |

Notebook 01's retained extraction output excludes `diCVwltkV5M.mp4` (normal) and `JUsbspdRRJA.mp4` (MS) for too few valid frames. Its one-video example returns `(155, 33, 3)`, reports 61% fully detected frames, and retains that shape after cleaning and normalization; normalized x ranges from −9.95 to 2.88. These are an example clip's properties, not dataset-wide detection statistics.

The current cache was independently scanned: every normalized sequence has shape `(T, 33, 3)` and finite values, and the total counts agree with notebook 01. The SHA-256 of each of the 47 cache files and the video manifest matches `artifacts/eval/g1/run_manifest.json`. This supports data continuity with the frozen comparison. The E0 manifest reports `git_dirty: true`; R1's older manifest does not record a dirty-tree field or all the software fields that the current runner would save.

Source IDs derive from video filenames after removing `_clip-NN`. They identify source videos, not verified people. Original manifest names include multiple numbered patients within some sources; existing project records also flag possible repeated people across MS source IDs. The 35 sources cannot be reported as 35 independent participants. Source grouping prevents portions of the same source video from crossing the split, but does not establish participant separation or independence of recording domains.

The raw video metadata was reread during this check. All 12 raw MS clips are 60 fps and 1080 × 1080. Normal clips have mixed rates and dimensions; PD clips also have mixed rates and dimensions, including two square 720 × 720 videos at 30 fps. Thus “all MS clips are square and 60 fps” is supported; “all square videos are MS” is false. Mean visibility, averaging each clip's mean equally, is approximately 0.8618 normal, 0.9196 MS, and 0.8698 PD. These associations motivate acquisition controls; they do not isolate a causal explanation for any classifier's predictions.

## 3. What “preserving shape” can accurately mean here

The current pipeline preserves joint indexing, temporal ordering within retained windows, and the documented array layout. It does not preserve every original measurement or the empirical sampling distribution unchanged.

1. `load_video_sequence` samples by integer frame stride `round(source_fps / target_fps)`. Source rates around 24 or 25 fps become approximately 12 or 12.5 fps when the requested target is 15, while the saved cache labels all sequences as 15 fps. Original timestamps are not saved. A 32-frame window therefore is not uniformly the same physical duration across all sources.
2. `clean_sequence` requires at least 30% fully detected frames, trims undetected ends, and interpolates every missing joint/channel value using `np.interp`. Despite prose saying “short gaps,” there is no maximum gap duration. Visibility is interpolated too, and no missing-value or interpolation mask is saved. The cache field named `keypoints` contains cleaned data, not untouched detector output.
3. `normalize_sequence` subtracts the pelvis midpoint and divides by torso length separately in every frame. This is a per-frame translation and uniform scale, so within-frame angles and length ratios survive for the same valid coordinates. Absolute position, torso scale, and root travel do not survive; time-varying scale can also alter motion trajectories. This is independent per clip and fits no statistic across train/test clips, but that separation does not restore discarded physical information.
4. `sliding_windows` uses 32 frames with stride 16. Three clips shorter than 32 frames are padded by repeating their final frame: normal `3FXUw98rrUY` (15 frames), normal `gp4H7Z2Vvn0_clip-01` (29), and PD `pFLC9C-xH8E_clip-01` (26). Longer clips can lose a trailing segment shorter than a stride; the implementation does not force a final window aligned to the last frame. These are 481 overlapping windows from 47 clips, not 481 independent walks.
5. Four adjacent frames from one joint are regrouped into a 12-value token: `(B,32,33,3) → (B,8,33,12) → (B,264,96)`. The regrouping preserves its input values and joint/time ordering; the learned linear projection is a representation change rather than an assertion that raw measurements remain recoverable.
6. The training loop samples windows with replacement using weight `1 / number_of_windows_in_source`. Each source has equal probability in expectation. This changes the training weighting from the natural clip/window frequency; it does not force exactly equal exposure in a finite run, and it does not increase the number of independent sources.
7. `random_view` clones the training window, applies one rotation angle (up to ±15°), one uniform scale (0.9–1.1), and one translation (up to ±0.1 normalized units per axis) consistently over the window. Half of windows are mirrored in expectation, with left/right landmark indices exchanged. Rotation/translation preserve within-frame distance and angle geometry; scaling preserves angles and ratios; reflection changes handedness and side labels. Visibility values move with their relabeled joints. No claim of exact 3-D geometry, preserved side-specific disease information, or guaranteed encoder invariance is justified by these augmentations.

The input is **2-D x/y plus detector visibility**, not three-dimensional position. `augment.py` still contains a stale docstring about keeping weak z as a channel; the stored arrays and active code use visibility. Reflection augmentation is not a measured bilateral gait-asymmetry test. No notebook estimates mirror agreement, phase-aligned left/right trajectories, a symmetry index as the primary endpoint, or a clinically validated asymmetry difference between diagnoses.

## 4. Exact split boundaries and statistical units

The frozen registry is `artifacts/eval/g1/fold_registry.json`: five-fold `StratifiedGroupKFold`, seed 42, grouping by source ID. Every cached clip appears once in a test partition; no fold has a source ID on both sides. These properties were recomputed from the registry and cache.

| Fold | Training clips | Test clips | Training sources | Test sources | Training windows | Test normal / MS / PD |
|---|---:|---:|---:|---:|---:|---|
| 0 | 37 | 10 | 29 | 6 | 395 | 3 / 2 / 5 |
| 1 | 37 | 10 | 28 | 7 | 376 | 4 / 3 / 3 |
| 2 | 38 | 9 | 28 | 7 | 426 | 4 / 2 / 3 |
| 3 | 38 | 9 | 28 | 7 | 354 | 4 / 2 / 3 |
| 4 | 38 | 9 | 27 | 8 | 373 | 4 / 2 / 3 |

Each encoder trains using only that fold's training-source windows, including all three label groups with their labels excluded from the self-supervised loss. The frozen target encoder reads each complete clip in windows; a fixed seed-0 set of target-token positions is mean-pooled within each window and the window vectors are averaged into one clip vector. The present fixed pool contains 161 of 264 token positions and covers all 33 joints over time; it is not only the twelve clinical joints despite a stale runner comment. Its choice does not inspect test labels.

The S-JEPA probe's standardizer and class-balanced logistic regression are fitted on training clip embeddings only. For the RF and shortcut models, varying feature columns are selected using training features, and the standardizer and classifier are fitted using training rows only. Test clips are transformed with those fitted parameters. This controls explicit training/test contamination at those steps, while source/participant identity and acquisition confounding remain unresolved.

Pooled out-of-fold scoring collects one held-out prediction for each of the 47 clips, then computes F1 separately for the three classes and averages the three values. Equal class weighting does not give equal source weighting: a source supplying five test clips contributes five predictions. The five folds have overlapping training sets and are not five independent experiments. The collection has already been inspected during development; fixed folds do not turn it into an untouched confirmatory test set.

## 5. Notebook-specific retained numbers

### Notebook 02: mask checks

The laptop model uses 32 frames, 33 joints, three channels, four frames per token, eight time blocks, 264 tokens, a three-layer 96-dimensional encoder with four attention heads, and a two-layer 96-dimensional predictor. The twelve clinically selected landmark indices are 11, 12, 23–32.

The six-example mask batch has shape `(6,264)` and contains six distinct masks; every row contains both visible context and hidden targets. Across the 512-mask bank with seed 0, the minimum fraction of masks in which a joint is visible **at least once over time** is 0.740234375, and the minimum fraction in which a joint is targeted **at least once** is 0.80859375. Mean targeted token fraction is 0.6327607126. These first two quantities can sum above one because one joint can be visible in one time block and targeted in another. They must not be called complementary token percentages.

The latest guarantee in `sample_target_mask` preserves one of the clinical landmark tokens when every clinical token would otherwise be hidden. The eligible set includes shoulders, so this is not a strict guarantee that a lower-body joint is visible.

### Notebook 03: 800-update training demonstration

The code and checkpoint stage metadata agree on fold 0 training only: 37 clips, 29 source IDs, and 395 windows. Retained output reports 0.40 million trainable parameters and MPS execution. The update budget is 800, regardless of the legacy “40 pretrain epochs, 30 finetune epochs” printed in the generic config summary.

| Logged update | Loss | Embedding standard deviation | Effective rank |
|---|---:|---:|---:|
| 0 | 6.0458 | 0.1485 | 7.9 |
| 200 | 2.0674 | 0.1815 | 7.4 |
| 400 | 1.3479 | 0.3536 | 8.8 |
| 600 | 1.3662 | 0.4783 | 10.5 |

The final cell compares the mean of the **first five** losses against the mean of the **last five**, reporting 4.990 → 1.112, with final batch effective rank 9.5. The reported EMA half-life is 346.2 updates, calculated at the middle schedule momentum rather than as a constant half-life throughout the run. `sjepa_ssl.pt` is present with training state step 800 and schedule horizon 800.

This supports that optimization progressed and the observed batch representation retained variation. It does not measure diagnosis accuracy, establish clinical symmetry learning, or prove that all forms of partial collapse are absent.

### Notebook 04: additional training from saved weights

The code loads notebook 03's weights, then calls `train_sjepa_v2` without `resume_state`. Therefore the 400-update stage starts a new optimizer, center, RNG stream, and learning-rate/EMA schedule. This is further training from a checkpoint's weights, not an exact resume of an uninterrupted 1,200-update run. The present continued checkpoint records step 400 and schedule horizon 400, which agrees with that behavior.

| Logged stage update | Loss | Embedding standard deviation | Effective rank |
|---|---:|---:|---:|
| 0 | 7.7224 | 0.4780 | 9.1 |
| 100 | 1.2868 | 0.5090 | 8.1 |
| 200 | 1.1934 | 0.6388 | 9.8 |
| 300 | 0.8718 | 0.6312 | 9.4 |

Final effective rank is 10.4. Separate training-only logistic heads yield held-out fold-0 macro-F1 of **0.600 before** and **0.600 after** the additional training. Equal rounded F1 does not prove unchanged individual predictions or statistical equivalence; no paired uncertainty analysis is retained. It establishes no observed change at the reported precision on this ten-clip comparison.

### Notebook 05: visualization remains unexecuted after a retained error

The first embedding operation failed before producing either representation matrix. The planned visibility matrix has 66 columns (per-joint means plus per-joint standard deviations); this differs from E0's 35-column visibility control. No values from that planned diagnostic can be substituted with E0's score or an older report's silhouette result. A future whole-dataset embedding scatter would include both training and held-out clips and remain a descriptive view, not an independent validation result.

### Notebook 06: imported five-fold evidence and live single-fold example

The frozen result uses 1,000 updates per fold and seed 42, saved under `artifacts/runs/r1_g1_1k_s42/`. Its 47 OOF rows were rescored during this review and reproduce all headline metrics and both confusion matrices exactly. Frozen RF predictions also match E0's saved predictions clip by clip.

| System | Pooled macro-F1 | Accuracy | Correct normal / 19 | Correct MS / 11 | Correct PD / 17 |
|---|---:|---:|---:|---:|---:|
| Frozen S-JEPA with linear probe | 0.4384024578 | 0.4468085106 | 11 | 6 | 4 |
| Paired gait-feature Random Forest | 0.6665728492 | 0.6595744681 | 13 | 8 | 10 |

The absolute F1 gap, RF minus S-JEPA, is 0.2281703915. This is a descriptive paired difference from this development collection, not a formal significance result. PD recall is 4/17 = 0.2352941176 for S-JEPA versus 10/17 = 0.5882352941 for RF.

Confusion matrices use rows = true class, columns = predicted class, in the order normal, MS, PD:

```text
                 S-JEPA                       RF
normal          11   3   5                  13   1   5
MS               0   6   5                   1   8   2
PD               5   8   4                   5   2  10
```

The dominant S-JEPA off-diagonal count is eight PD clips predicted as MS. RF's largest mistakes are five normal clips predicted as PD and five PD clips predicted as normal; the notebook's claim that RF shares PD→MS as its dominant error is incorrect.

The notebook also retains a **separate 500-update fold-0 demonstration**: RF macro-F1 0.915, S-JEPA 0.644, final batch effective rank 8.9. This is useful as an executed tutorial demonstration, not a replacement for the five-fold result. Frozen RF fold 0 has macro-F1 0.9153439153, agreeing with the live rounded value. Frozen S-JEPA fold 0 has 0.4666666667 at the older run's 1,000-update budget; different code provenance and training runs prevent interpreting the 500/1,000 comparison as a controlled training-budget ablation.

For transparency, scores recomputed from the frozen OOF rows are:

| Fold | Frozen S-JEPA macro-F1 | Frozen RF macro-F1 | Frozen final batch effective rank |
|---|---:|---:|---:|
| 0 | 0.466667 | 0.915344 | 8.758249 |
| 1 | 0.507937 | 0.623810 | 9.510392 |
| 2 | 0.396825 | 0.600000 | 8.235277 |
| 3 | 0.450794 | 0.642857 | 8.775340 |
| 4 | 0.361111 | 0.473016 | 7.710423 |

The five retained final losses range from 1.1355 to 1.5840, embedding standard deviations from 0.5655 to 0.6431, and teacher/student parameter cosine distances from 0.01407 to 0.01711. The rank is computed from a centered **32 × 96 batch** using the entropy of normalized singular values, not from all 47 clip vectors. Its mathematical rank ceiling is 31, not 96. Rank around eight or nine therefore cannot justify claims that the model has “unused dimensions” or that larger architectures cannot help.

## 6. Exact shortcut controls and selection caveat

The control table in `E0_results.json` contains both logistic and RF results. The capstone's “best” entry is chosen after comparing their pooled OOF scores; this is a descriptive maximum across two models and does not include an independent selection-validation split.

| Feature representation | Dimensions | Logistic pooled F1 | RF pooled F1 | Capstone maximum |
|---|---:|---:|---:|---:|
| Mean normalized x/y pose | 66 | 0.510720 | 0.656511 | 0.656511 |
| Mean and standard deviation of normalized x/y | 132 | 0.702745 | 0.624178 | 0.702745 |
| Visibility: global mean/std and per-joint means | 35 | 0.550538 | 0.635720 | 0.635720 |
| Cached frame count repeated in two columns | 2 | 0.353384 | 0.469499 | 0.469499 |
| Nine distances between joints in the median raw-pixel pose | 9 | 0.406165 | 0.439900 | 0.439900 |

The feature names `duration_acq` and `body_proportion` overstate what their active implementations measure. The former does not input source fps, frame resolution, or a verified physical duration; the latter uses raw-pixel distances, not dimensionless proportions. Mean/std pose discards temporal order but can retain meaningful posture and movement-amplitude information. Its high score does not prove it contains only nuisance variation. Visibility may reflect disease-related posture as well as acquisition and pose-detector behavior. The evidence supports that a temporal JEPA representation has not outperformed these summaries on the retained comparison, rather than a proven causal account of why each representation succeeds.

The strongest table entry is mean/std pose with logistic regression (0.702745), not the gait-feature RF (0.666573). The E0 RF fold-mean macro-F1 is 0.651005 with fold standard deviation 0.144888; these differ from its pooled macro-F1 of 0.666573 and must not be mixed on one unlabeled comparison axis.

## 7. Inherited claims that should be corrected in new deliverables

1. **“The exact dataset shape is preserved throughout.”** Specify topology and array-layout preservation, while stating the sampling, interpolation, normalization, padding, and weighting changes described above.
2. **“Leakage-safe” or “participant-disjoint,” without qualification.** Use “source-separated, with train-only fitting” and explain that participant identity and recording-domain overlap remain unresolved.
3. **“Pretraining on normal, then adding MS and PD with VICReg.”** This describes a superseded branch. Current notebooks 03/04 use all training labels' source videos, label-free SSL, and separately supervised linear heads.
4. **“The repaired score proves removing shortcuts caused the score decrease.”** Repairs changed multiple aspects simultaneously; the cache still contains acquisition confounding. No controlled repair-by-repair ablation establishes the cause of the historical decrease.
5. **“Current code produced the frozen fully repaired headline.”** The current implementation contains later review fixes. Frozen R1 checkpoints lack the saved schedule field present in the notebook-03/04 checkpoints; their manifest cannot establish execution with every present fix. AR5 specifically identified a per-example centering bug, now fixed to update once per batch. Keep the retained R1 evidence and present-code mechanism distinct unless a new complete run verifies their connection.
6. **“Effective rank proves no collapse or proves sufficient capacity.”** The observed batch diagnostics argue against an identical-output collapse in those batches. They establish neither clinical usefulness nor architecture sufficiency.
7. **“Additional training had no effect.”** The rounded fold-0 macro-F1 is unchanged; prediction changes and equivalence were not tested, and the optimizer/schedule restart limits the experiment's interpretation.
8. **“Representation clusters separated in notebook 05.”** No such output is retained for the current notebook. Older silhouette values cannot be carried forward as current findings.
9. **“RF is strongest” or “RF's main PD error is MS.”** Both contradict the current retained table and confusion matrix.
10. **“Chance macro-F1 is 1/3.”** Macro-F1 depends on the class distribution and prediction rule; `1/3` is not a universal or estimated null reference. A defined group-level permutation or dummy-classifier procedure would be required for a justified baseline. Remove the capstone's inherited reference from new result graphics.
11. **“A 95% confidence interval establishes the gap is real.”** `docs/07-0803-METHODOLOGY_ROADMAP.md` claims a bootstrap interval near 0.07–0.39, but the top-level notebooks and their saved result JSON contain no uncertainty analysis or its specification. Do not treat this interval as a current notebook result. A separately documented reanalysis would need its resampling unit, repetitions, pairing, and inferential limitations stated.
12. **“Initial null hypothesis was preregistered and rejected.”** The repository records a development decision process, but no untouched participant-level confirmatory test or prespecified formal hypothesis test is retained. An organizing null can be stated transparently as the current report's scientific question: the learned representation may add no useful held-out signal beyond simpler features. The retained results do not reject that null in favor of a JEPA gain.
13. **“The notebooks discover condition-specific asymmetry.”** These notebooks compare clip labels, demonstrate geometric modeling, and test controls. They do not estimate a condition-specific asymmetry endpoint or identify the physical cause of an error. Clinical motivation and proposed geometry experiments should be labeled separately from measured results.

## 8. Checks performed for this record

All seven notebook JSON files were read, including cell source, execution counts, text outputs, error outputs, and presence of image outputs. The cache, raw video metadata, fold registry, checkpoint metadata, saved OOF predictions, frozen result JSON, capstone scoreboard, provenance manifest, current data/token/mask/augmentation/training/model/classical code, and relevant previous reports/review conclusions were inspected. Pooled and per-fold F1/accuracy/confusion values were recomputed from saved OOF rows without training. Cache shape/finite checks, exact fold separation/coverage, present sampler-bank statistics, fixed readout dimensions, padding counts, and per-file provenance hashes were independently checked.

The review intentionally did not reconstruct notebook 05's missing retained plots, rerun training, recompute a new inferential confidence interval, or alter the original records. Those would be new analyses requiring a separate provenance record.
