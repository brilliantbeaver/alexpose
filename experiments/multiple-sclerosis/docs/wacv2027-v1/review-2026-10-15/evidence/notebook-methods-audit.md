# Notebook and implementation evidence audit

**Audit date:** 15 October 2026. **Scope:** all seven root notebooks, their saved outputs, and all modules in `sjepa/`. This audit uses the files present during review; it does not treat older explanatory prose as the record of the current execution. No training was rerun, and no notebook or implementation file was changed.

Cell references below are **one-based cell positions**, including Markdown cells. `exec` is the notebook's saved execution count. The notebooks contain 124 cells: 55 executed code cells and 69 Markdown cells. Code cells have consecutive execution counts within each notebook. Their cell metadata do not retain execution dates. A date written in narrative text is therefore not an execution timestamp.

All source cells and scientifically meaningful text outputs were reviewed. Saved PNG results, mask timelines, representative GIF frames, and representative frames from the three embedded gallery videos were inspected visually. SVG text was inspected directly. Notebook 04's very large displayed checkpoint dictionary contains model and optimizer tensor arrays; its metadata and training-state fields were inspected without treating the thousands of tensor values as interpretable experimental results. Readable exports and extracted images are under `../working/notebook-methods/`.

## Findings that should control the paper revision

1. **The current saved run and the surrounding notebook narrative differ materially.** Notebook 06, cell 11 (`exec=5`), names `capstone-20261009T060955236580Z`; cell 14 (`exec=6`) reports RF source-weighted macro-F1 **0.325184**, while cell 15's Markdown retains **0.411** from the earlier run. The current saved RF confusion image in cell 17 agrees with the newer result. The older RF value is not an arithmetic error in its own historical artifact, but it must not be combined with the later corrected implementation as though they were one experiment. The independent results audit traces the ankle-range correction and should govern the manuscript's choice of primary run.
2. **There is defensible positive evidence for representation learning, with a specific boundary.** Notebook 03's exact saved diagnostic summary shows decreasing feature-prediction loss and increasing variation between embeddings. The trained, frozen representation also supports the held-out classification result in notebook 06. These observations support a small-data representation-learning study. They do not isolate the contribution of pretraining from the architecture, establish that time order is essential, or identify disease-specific motion features; no untrained-encoder or temporal-order comparison is present.
3. **The current experiment is not normal-only pretraining followed by supervised disease adaptation.** All three dataset labels contribute training sources from the start; the labels do not enter the encoder loss. Notebook 04 continues the same label-free task. The class-aware VICReg code remains in legacy files but is not used by notebooks 03–06. The filenames and legacy configuration fields must not be used to describe the active method.
4. **The manuscript's mask-frequency description needs a denominator correction.** A roughly 77–78% frequency for shoulders/hips describes the fraction of their **joint–time token positions** that are masked, not the fraction of masks in which the joint is hidden at least once. A fresh 5,000-mask, seed-42 check gives the latter at about 98%. The two quantities answer different questions; detailed values are below.
5. **The low-data statement should name the actual training unit.** Every outer fold trains on **24 source recordings, contributing 45–58 usable clips and 190–238 windows**. Repeatedly drawing windows does not increase the number of recordings or people. The S-JEPA model is initialized afresh, but pose extraction uses a pretrained MediaPipe front end. There is no learning-curve experiment establishing sample efficiency.
6. **Several preprocessing descriptions are more exact than the code permits.** The loader takes every rounded-integer-stride frame; it does not resample timestamps to exactly 15 fps. With the current files, that rule gives approximately **11.88–15 retained frames/s**. Cleaning interpolates all remaining missing gaps without a maximum gap length. Neither “exactly 15 fps” nor “short gaps only” is an accurate description of this implementation.
7. **Anatomical masking is an explicit design choice, not a demonstrated clinical mechanism.** The implementation uses seven predefined joint groups, with a modest preference for groups containing the twelve specified shoulder/leg landmarks. The notebooks verify how masking behaves. They contain no matched masking ablation showing that the anatomical preference improves classification or preserves a particular clinical signal.

## Data, preprocessing, and sample sizes

The present raw inventory, saved output of notebook 00 cell 9 (`exec=5`), and verified registry agree:

| Dataset label | Raw clips | Retained clips | Source recordings retained | Cleaned pose frames |
|---|---:|---:|---:|---:|
| Normal | 26 | 24 | 16 | 1,427 |
| MS | 30 | 29 | 13 | 3,450 |
| PD | 35 | 35 | 12 | 2,766 |
| Total | 91 | 88 | 41 | 7,643 |

Notebook 01 cell 14 (`exec=8`) names the three exclusions: `JD1AGVpftps_P1_02` and `diCVwltkV5M_P1_01` from Normal, and `tsOMPBS277Q_P1` from MS. All 41 source IDs remain represented after exclusion. `artifacts/eval/full-v1/exclusions.json` correctly states that the saved generic “too few valid frames” message does not distinguish failure of the 30% detection criterion from failure of the eight-retained-frame criterion. Do not retrospectively assign a more specific reason to an individual exclusion.

An independent lightweight inventory check during this audit recomputed the raw/cache correspondence, cache checksums, and registry structural validation successfully. The registry is `artifacts/eval/full-v1/fold_registry.json`, SHA-256 `9496e61b050fc8b62ccbc59cb737376396205a3356b323733b9b8915d56600e1`.

Source IDs come from filenames after a recognized clip suffix is removed (`sjepa/data.py:189–200`). They identify a source recording, not a verified participant. The split validation checks that one source has a single label, that clip names are unique, and that sources do not cross partitions. It does not discover duplicate content uploaded under different IDs or repeated people across different recordings (`sjepa/splits.py:33–47`, `125–169`). “Unseen source recordings” is supported; “unseen patients” is not verified.

The input array has shape `(T,33,3)` with **pixel x, pixel y, and MediaPipe visibility**, not x/y/z. The visibility channel is retained after normalization. Missing detections become NaNs. Cleaning requires finite x coordinates for all 33 landmarks in at least 30% of sampled frames, trims ends with no detected landmark, and linearly interpolates each joint/channel over remaining gaps. There is no visibility-score threshold and no maximum allowed interpolation gap in this function (`sjepa/data.py:106–161`). Notebook 01 separately rejects cleaned sequences with fewer than eight frames.

Each frame is centered on the midpoint of the hips and divided by the distance from that midpoint to the shoulder midpoint. The denominator is floored at `1e-3`; visibility is unchanged (`sjepa/data.py:164–182`). This removes image-plane translation and rescales apparent torso size. It does not establish viewpoint invariance or remove detector-confidence, camera, or recording effects.

The loader's sampling stride is `max(1, round(source_fps/15))` (`sjepa/data.py:73–85`). A read-only metadata check of all 91 videos found source rates from about 23.756 to 59.940 fps, implying retained cadence from 11.878 to 15 fps under that rule. The cache stores the nominal value 15, not the per-video effective cadence. A 32-frame window is therefore approximately 2.1–2.7 seconds under this frame-count convention, rather than uniformly 2.1 seconds. This check uses video-level fps metadata; variable-rate per-frame timestamps were not recovered. Raw videos span nine resolutions and approximately 0.90–30.0 seconds each. These facts support describing heterogeneous acquisition, without attributing any model prediction to a particular acquisition cue.

Windowing uses 32 frames, stride 16, and repeats the last frame to pad a shorter clip (`sjepa/data.py:266–282`). Fifteen retained clips are shorter than 32 frames: five Normal and ten PD. The retained clips range from 14 to 450 pose frames and yield 358 windows in total. Long-clip remainder frames after the last complete stride-aligned window are not included. RF and the simple controls summarize full cleaned clips, whereas learned readouts average these fixed windows; “same pose cache” is accurate, while “identical temporal samples” would be too strong.

### Fold membership and actual window counts

These values were independently recomputed from the retained arrays and registry without training:

| Fold | Training clips / sources / windows | Validation clips / sources / windows | Test clips / sources / windows |
|---|---|---|---|
| 0 | 51 / 24 / 207 | 19 / 8 / 56 | 18 / 9 / 95 |
| 1 | 58 / 24 / 226 | 17 / 9 / 55 | 13 / 8 / 77 |
| 2 | 54 / 24 / 238 | 17 / 9 / 83 | 17 / 8 / 37 |
| 3 | 53 / 24 / 224 | 21 / 9 / 79 | 14 / 8 / 55 |
| 4 | 45 / 24 / 190 | 17 / 9 / 74 | 26 / 8 / 94 |

The splitter uses a five-fold `StratifiedKFold` on one row per source (seed 42). Within the development sources it takes the first split of a four-fold source-level splitter with seed `43 + fold`. This is five outer test folds with one validation holdout per fold, not a fully averaged nested four-fold search (`sjepa/splits.py:86–122`). Each clip has exactly one outer-test assignment. The principal source counts are approximately 60% training, 20% validation, and 20% test, but unequal clips per source make the clip proportions different.

## Active model and learning procedure

| Component | Verified implementation | Primary evidence |
|---|---|---|
| Input | Normalized x, normalized y, visibility; 33 landmarks | `config.py:47–53`; `data.py:164–182` |
| Tokens | Four adjacent frames of one landmark; 12 input numbers projected to 96; joint and time positions added; 264 tokens/window | `tokenizer.py:22–50`; notebook 02 cells 8–10 |
| Encoder | Three transformer layers, width 96, four attention heads, MLP ratio 2, dropout 0 | `config.py:55–62`; `models.py:34–91` |
| Predictor | Two layers, width 96; separate learned joint/time positions | `models.py:161–201` |
| Teacher | Initially copied from the view encoder; no loss-gradient updates; moving-average updates | `models.py:204–218`, `240–248`, `282–297` |
| Mask | One independently sampled joint/time mask per example and update, nominal target fraction 0.6 | `train_v2.py:205–220`; `masking_v2.py:108–189` |
| Transform | One rotation up to ±15°, scale 0.9–1.1, translation up to ±0.1, and optional horizontal reflection with left/right landmark swaps per window | `augment.py:40–89` |
| Objective | Cross-entropy between centered/sharpened teacher feature distributions and predictions at masked tokens; averaged per example | `losses.py:73–93` |
| Optimization | AdamW, learning rate 0.001, weight decay 0.05, betas 0.9/0.95, gradient norm clipping 1; 10% warmup then cosine schedule | `train_v2.py:150–155`, `189–225` |
| Training sampling | Each source receives equal total probability; sampling with replacement within the pooled windows | `train_v2.py:70–80`, `173–199` |
| Budget | 800 updates, followed by a candidate continuation of 400 updates on the same training sources | notebook 03 cell 15; notebook 04 cell 11; `full_experiment.py:132–179` |
| Continuation | Model/teacher weights retained; optimizer, loss center, schedules, and seeded random streams restarted | `train_v2.py:150–171`; `full_experiment.py:91–114`, `132–148` |
| Readout | Full window into frozen teacher; mean over a fixed seed-0 subset of output token positions; mean across windows | `full_experiment.py:17–53`; `models.py:299–312` |
| Probe | Training-only standardization and class-balanced logistic regression, C=1, max iterations 2,000, seed 42 | `full_experiment.py:56–64` |
| Selection | Higher validation source-weighted macro-F1; tie retains 800-update model; no refit on validation | `full_experiment.py:169–183` |

The view encoder computes all token slots with hidden slots excluded as attention keys; hidden output slots are replaced by a learned mask token before prediction (`models.py:113–134`). Thus “the view encoder uses visible context” describes the information flow more faithfully than claiming hidden coordinates are physically removed from every intermediate array.

The current downstream readout uses **161 of 264 token positions**, verified by calling `sample_target_mask(33,8,default_rng(0), target_ratio=0.6)`. Both legs are selected throughout all eight time blocks, but some face, arm, and shoulder positions are selected too. It is neither an all-token average nor pooling only the twelve clinical landmarks. The mask selects **outputs to average**; full input skeletons reach the teacher during feature extraction. Notebook 03's training diagnostics instead average all teacher tokens.

Condition labels are used to balance source splits, fit the probe, and evaluate/select its stage; the encoder loss itself is label-free. `train.py`, `masking.py`, and `VICRegLoss` are legacy paths, not evidence that their losses or fixed masks were used in the reported current run. The strings “pretrain 40ep, finetune 30ep” printed by `describe(cfg)` likewise reflect unused historical fields for this loop.

Equal-source window sampling does not make the supervised probe source-balanced. The probe and RF use class balancing over clips, without explicit inverse-source weighting in fitting. In fold 0, 13 of 23 MS training clips come from one source. Equal-source **evaluation** subsequently gives every source weight one, divided among its clips. The metric is computed from weighted clip predictions, not a majority vote per source and not an average of one F1 per source (`full_experiment.py:67–74`; `eval.py:39–61`).

## What the saved learning diagnostics show

Notebook 03 cell 19 (`exec=8`) contains exact numeric summaries, which are stronger evidence than the approximate graph readings retained in cell 18:

| Diagnostic | Mean of first 50 updates | Mean of last 50 updates | Final point |
|---|---:|---:|---:|
| Feature-prediction loss | 3.858368 | 1.022220 | 1.097049 |
| Effective rank | 8.132109 | 10.939291 | 12.134889 |
| Embedding standard deviation | 0.153868 | 0.525584 | 0.549869 |
| Teacher/student drift | 0.022306 | 0.019918 | 0.019944 |

The cell records 800 updates, 25,600 window draws across 24 sources, and observed source counts from 1,025 to 1,129 draws. The plot saved in cell 17 (`exec=7`) visually agrees: loss falls from roughly six toward one, the rank rises into roughly 10–12, and drift settles after an initial spike. These are training-batch diagnostics. The cross-entropy target changes as the teacher changes, so a decrease demonstrates increasing agreement with evolving targets, not a percentage reduction in classification error.

Effective rank is computed from normalized singular values of centered batch embeddings (`train_v2.py:83–101`). A centered batch of 32 has at most 31 nonzero directions even though feature width is 96. Combined rank and spread measurements argue against constant-output collapse on the sampled batches. They do not identify what physical or recording properties the embedding varies with.

The drift statistic averages `1 − cosine similarity` over parameter tensors and uses a denominator floor (`train_v2.py:259–267`). Initially zero tensors make the early spike unsuitable as evidence that weights suddenly moved by 31%. Neither effective rank nor drift should be presented as a clinical-performance score.

Notebook 03 cell 18 dates its account to September 20 and gives final rank 12.134796142578125; the currently saved numeric cells give 12.134888648986816. The difference has no practical bearing on an approximately 12.13 statement, but the historical date and claim that only one numeric point is available are stale. Cell 19 now explicitly preserves exact summary values. Per-update histories themselves are not saved in the checkpoint payload (`train_v2.py:315–330`).

For the October capstone, the retained `results.json` records final effective rank 10.90–12.13 after the first stage and 9.56–11.64 after continuation. All five continuations have lower final feature-prediction loss, but the validation classifier chooses continuation only in folds 3 and 4. Final-loss comparisons across restarted centers/schedules should remain descriptive rather than a claim that one objective value universally measures representation quality.

### Affirmative wording supported by this evidence

> Label-free training produced representations that retained variation between walking clips while improving prediction of hidden features. A simple classifier fitted to the frozen representations then achieved a higher held-out score than the implemented Random Forest based on manually designed joint-angle summaries. This result supports the feasibility of learning useful features from a small collection of source recordings. The present comparisons do not yet identify how much of that information depends on temporal order or on the anatomical masking preference.

The sentence about the RF score must use the same explicitly identified run as the results table. “Pretraining caused the improvement,” “learned disease biomarkers,” “proved sample efficiency,” and “anatomical masking improves clinical discrimination” exceed these experiments. The average-pose control remains relevant evidence even when RF is the main comparison.

## Anatomical masks: verified behavior and the denominator error

The seven predefined groups are head, mouth, left arm, right arm, trunk, left leg, and right leg (`masking_v2.py:43–51`). The twelve selected landmarks are shoulders 11/12 and the bilateral hips, knees, ankles, heels, and foot indices 23–32. A group receives the clinical preference if it contains **any** of these landmarks; this includes the arm groups through their shoulders. The region weight is 1.5 divided by its average joint-membership multiplicity, then normalized (`masking_v2.py:84–105`). It is not an observed 1.5-fold masking probability for each selected joint.

Whole groups are covered over contiguous time spans, each at most six of eight blocks in this configuration. Regions accumulate until the nominal budget is met or the guard stops sampling. Their union can cover one joint for the whole window. The sampler guarantees some context and at least one visible token from the selected shoulder/leg set somewhere in the window; it does not guarantee a visible hip, a visible leg, or any context in every time block (`masking_v2.py:135–165`).

A cheap independent check using `mask_bank_stats(33,8,n_masks=5000,seed=42)` produced:

| Landmark or group | Targeted token positions | Windows with any target for that joint | Windows fully masked for that joint |
|---|---:|---:|---:|
| Left shoulder | 78.330% | 98.560% | 25.180% |
| Right shoulder | 78.445% | 98.600% | 25.960% |
| Left hip | 76.778% | 97.820% | 23.720% |
| Right hip | 77.103% | 98.140% | 23.780% |
| Left knee | 61.833% | 91.380% | 10.940% |
| Right knee | 62.070% | 91.620% | 10.960% |
| Nose / head group | 59.058% | 94.200% | 5.460% |
| Mouth landmarks | 53.470% | 83.080% | 8.780% |

Mean fraction of all token positions targeted: **0.6320439394**. This overshoots the nominal 0.6 because regions are added as groups. The first and second numeric columns must not be interchanged. The sample's 77–78% shoulder/hip values approximately match a **token-frequency** claim; an “at least once” claim at those values is incorrect. Different seeds or a different number of draws can change the last digits, so copy the exact audit settings alongside the statistic.

Notebook 02 cell 14 (`exec=8`) uses a different bank: 512 masks, seed 0. It reports mean target fraction 0.63, minimum probability of a joint being visible somewhere in a window 0.74, and minimum probability of being targeted somewhere 0.81. The hip visibility table demonstrates temporal imbalance: central blocks can have roughly 7–13% hip context, compared with roughly 46–52% at the ends. Passing the “ever visible” check does not demonstrate balanced joint/time coverage.

Notebook 02 cell 18 (`exec=10`) creates eight **fresh mask draws on the same training window**. Its exact timeline shows all joints in both roles across the eight samples, but both complete legs hidden throughout sample 1. Across the 64 joint/time blocks in these eight samples, the left hip is visible in 17 and the right in 10. The saved GIF's repeated playback adds no new mask draws or training observations. The figure is a faithful visualization of the sampled masks, not evidence that anatomy-biased masking improves representation quality.

Some old comments say that motion-aware masking is “contraindicated” or that masking more active joints would hide the clinical evidence (`masking_v2.py:15–19`; notebook 02 cell 2). No experiment here establishes that assertion, and masking inputs while predicting their features can itself be a learning signal. In the paper, state the choice and its motivation cautiously: the experiment uses predefined anatomical regions without selecting regions by motion magnitude. Whether that choice improves performance remains untested.

## Train/validation/test isolation: what is verified

The shared helper rejects training records that differ from the registry's training clips or sources (`full_experiment.py:91–105`). Checkpoint context stores dataset and registry hashes, fold, training membership, configuration, and stage; loading checks equality before installing the weights (`splits.py:215–226`). These safeguards are good evidence that the reported code implements the intended separation.

Within `run_fold`, train and validation embeddings are obtained and separate probes are fitted before test feature extraction. Stage selection is fixed at lines 178–181; the selected teacher/probe is then applied to test data at line 183. RF fitting, feature-column selection, and scaling use training clips. Controls use training-only scalers and heads. Source-independent deterministic per-clip feature caching is safe across partitions because no learned dataset-level transformation is cached (`experiment_cache.py:112–148`). Every retained test clip must be present exactly once and with the correct fold/source/label before pooled metrics are written (`full_experiment.py:208–232`).

These checks support source-level separation **within each recorded evaluation**. They do not prove an untouched final test set after an extended development history, nor do they establish participant independence. Any method changes motivated by already viewed test scores should be described as further development on this collection. Running an identical seed again is not independent-seed replication; the independent results audit found changed floating-point weights between the two capstone runs despite unchanged selected S-JEPA test labels.

## Notebook-by-notebook record and discrepancies

| Notebook and cells inspected | Saved execution / evidence | Revision consequence |
|---|---|---|
| **00, cells 1–5** | Setup, placeholder GitHub/Colab URL, full-data paths; laptop/non-smoke | Placeholder links are not a usable public release. Avoid claiming public reproducibility from them alone. |
| **00, cells 6–7** | Pipeline SVG | The image still labels RF as “82 hand-made gait features.” That is a container size, not 82 varying measured gait descriptors. Correct if reused in the paper. |
| **00, cells 8–10** | 91 clips / 41 sources, label counts, within-source concentration | Counts agree with inventory; grouping is by recording. One raw MS source has 14 clips, of which 13 are retained. |
| **00, cells 11–12** | Three video previews and roadmap | Visual inspection confirms heterogeneous framing/backgrounds in examples. It does not verify diagnoses, population-level acquisition differences, or an actual shortcut. |
| **01, cells 1–8** | Extraction setup and example `(68,33,3)` raw sequence, about 54% fully detected | Confirms channels and whole-frame extraction. The example output does not characterize all clips. |
| **01, cells 9–12** | Example cleaned sequence `(48,33,3)` and unmasked normalized skeleton GIF | “Short gaps” is inaccurate: code has no gap-length restriction. The GIF illustrates extracted geometry, not pose accuracy against ground truth. |
| **01, cells 13–16** | Three named exclusions; 88 retained caches and class/frame counts | Use retained counts for evaluation; do not merge the earlier `g1` cache. |
| **02, cells 1–7** | Mask rationale, setup, same registry and fold table | Current mask is stochastic. Legacy fixed-mask rationale is not the active training method. |
| **02, cells 8–12** | 264 tokens; twelve-landmark clinical mapping table | These landmarks guide mask preference; the feature names in the mapping are not measured outputs of S-JEPA or all populated RF features. |
| **02, cells 13–18** | 512-mask statistics, repair SVG, eight-draw timeline/GIF | Quantify mask coverage with its correct denominator. The SVG's “~1.5x” is schematic region weighting, not observed per-joint frequency. |
| **03, cells 1–8** | Current method uses all three conditions and 51/24 training example | The `pretrain_normal` filename is historical. |
| **03, cells 9–15** | Architecture, masks, label-free objective; 800 updates and final rank | Ignore legacy epoch counts. No coordinate-reconstruction or condition-classification objective is used here. |
| **03, cells 16–20** | Diagnostics figure and exact summary; transition to probe experiment | Use exact cell 19 summaries. Cell 18's September date/old final rank and “only final rank printed” description are stale. |
| **04, cells 1–9** | Split and loaded original checkpoint dictionary | Labels have not entered the encoder objective. The dictionary is model state, not additional independent evidence. |
| **04, cells 10–14** | Extra 400 updates; validation scores 0.2939794319 vs 0.2761052994; selects original | The comparison is a restarted continuation recipe. It is not supervised fine-tuning or class-aware VICReg. |
| **04, cells 15–17** | Interpretation, additional per-class validation table, next-step discussion | Aggregate scores are direct outputs. The added class table is a separate described CPU check, not printed by cell 14; distinguish its provenance if used. These are validation results, not the paper's final test result. |
| **05, cells 1–10** | Training-only feature extraction of 51 clips from 24 sources | No validation/test embeddings enter these maps. Multiple dots can share a source. |
| **05, cells 11–16** | Separate scaled t-SNE and UMAP maps for original, continued, visibility | Current saved images mix labels. Narrative spatial descriptions were retained from older maps: e.g., the current original UMAP has a small lower-left group, not the stated upper group. Do not copy geometric descriptions without the matching image. |
| **05, cells 17–21** | Silhouette: original −0.004, continued −0.012, visibility −0.004 | Computed in standardized original features, not the two-dimensional plots. These are clip-weighted training-only grouping summaries, not proof of no class information. Cell 17's equation contains a malformed `frac` escape. |
| **06, cells 1–10** | Five-system design and active procedure | Much method prose is accurate, but the RF-bug limitation is stale relative the current corrected extractor and saved later run. |
| **06, cell 11** | October 9 run; all five folds computed; 257.4 seconds | Identifies current saved execution and budgets. Do not cite the Markdown's 216-second value as this run's timing. |
| **06, cells 12–14** | Selection narrative/table and current aggregate output | Cell 12 retains old continued validation scores for folds 1/3. Current outputs are approximately 0.250 and 0.332, not 0.344 and 0.284. Selected stages remain original/original/original/continued/continued. |
| **06, cells 15–20** | Older RF result table, fold interpretation, current confusion image, stale submission advice | Update the RF numbers and correction status consistently. The original older result should remain identified in the evidence history. The ICLR/PerFail discussion is not guidance for the requested WACV workshop. |

### Current saved validation selections, for avoiding mixed-run tables

The October run's retained `results.json` reports:

| Fold | First stage | Continued stage | Selected |
|---|---:|---:|---|
| 0 | 0.293979 | 0.276105 | First |
| 1 | 0.490618 | 0.250462 | First |
| 2 | 0.324930 | 0.257210 | First |
| 3 | 0.266008 | 0.332239 | Continued |
| 4 | 0.375309 | 0.559710 | Continued |

The current saved pooled source-weighted macro-F1 output is RF 0.325184, S-JEPA 0.456624, visibility 0.317565, mean pose 0.451893, and majority 0.259398. The corresponding clip-weighted scores are 0.320940, 0.397034, 0.318057, 0.409840, and 0.202125. Reporting the RF advantage is supported as a result of these defined procedures; it does not justify removing the close mean-pose result. That control identifies a remaining scientific question about what information the learned representation uses.

## Suggested plain-language method text

> We extracted 33 body landmarks from each video frame using MediaPipe and represented each landmark by its horizontal and vertical position and visibility score. Coordinates were centered at the hips and scaled by torso length. Frames were sampled at an approximately 15-fps target cadence using an integer frame stride. We removed clips with insufficient detected pose, trimmed empty ends, and interpolated remaining missing values. The resulting collection contained 88 clips from 41 source recordings. Because several clips could come from the same recording, we kept all clips and derived windows from a source in the same training, validation, or test partition.

> S-JEPA was trained from a fresh initialization in each fold using 45–58 clips from 24 training sources. Each 32-frame window was divided into groups of four consecutive frames for each landmark. The model learned to predict features of hidden joint–time regions from the visible skeleton context, with the complete training window supplying targets through a slowly updated encoder. Masks were sampled anew for each window, with a preference for predefined regions containing shoulder and leg landmarks. We compared 800 training updates with a further 400-update stage and selected between them using validation performance. The chosen encoder was then frozen, and a simple linear classifier fitted on training clips predicted the three dataset labels for held-out source recordings.

> The masking preference was specified from anatomical considerations, and the coverage checks confirmed that every joint could appear as either context or prediction target across repeated samples. These checks validate the behavior of the sampler. A comparison against alternative masking schemes is still needed to determine whether the anatomical preference improves classification.

The raw pose front end, code version, feature correction, and selected primary artifact should be specified elsewhere in the reproducibility materials. No newly calculated mask statistic or metadata inspection in this audit constitutes a new model-performance experiment.
