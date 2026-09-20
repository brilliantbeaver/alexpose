# Training, validation, and testing for the full video collection

Review date: September 20, 2026. Scope: notebooks 02–06, their generator, and
the shared code used to split data, load checkpoints, and evaluate models.

## Why this change was needed

The notebooks did not all describe or use the same dataset. Notebook 02 read
the full pose cache, while notebooks 03–06 still read the older `keypoints/`
cache and `g1` registry. Those older grouped splits were not a split of the
current `video-data-full/` collection.

There was also no separate validation partition in the teaching comparison.
Notebook 04 compared training regimes using held-out test scores. Using those
scores to choose a regime would turn the test set into tuning data. Notebook 05
embedded and projected every cached clip together, including held-out clips.
Those plots were labeled as descriptive, but inspecting them while choosing a
model could still feed test information back into development.

This review does not show that every earlier score suffered source overlap.
It shows that the old paths and workflow could not support the requested
full-data training, validation, and testing boundary.

| Notebook | Finding | Current behavior |
|---|---|---|
| 02 | Selected the first full-cache clip without assigning a partition | Loads the common registry; the mask example uses a training clip |
| 03 | Trained on legacy g1 fold 0; saved to a shared checkpoint filename | Uses full-data fold 0 training sources; saves partition provenance |
| 04 | Compared legacy checkpoints on test clips | Continues on training sources; fits heads on training clips; compares on validation |
| 05 | Fit t-SNE/UMAP and computed silhouette on all legacy clips | Uses training clips only, with partition metadata saved beside embeddings |
| 06 | Displayed legacy scores and ran a single live legacy fold | Runs all five full-data folds from fresh models, with validation selection and complete test predictions |

The legacy `g1` registry, results, and reproduction scripts remain historical
artifacts. They are not imported into the new evaluation. Notebook 01 and its
existing extraction outputs were preserved during this change.

## Data and exclusions

The raw collection has 91 MP4 clips from 41 filename-derived source videos.
The current pose cache has 88 usable clips from those same 41 sources:

| Condition | Raw clips | Cached clips | Excluded clips | Cached sources |
|---|---:|---:|---:|---:|
| Normal | 26 | 24 | 2 | 16 |
| MS | 30 | 29 | 1 | 13 |
| PD | 35 | 35 | 0 | 12 |
| Total | 91 | 88 | 3 | 41 |

The saved output in notebook 01 reports `skip (too few valid frames)` for:

- `Normal/JD1AGVpftps_P1_02.mp4`
- `Normal/diCVwltkV5M_P1_01.mp4`
- `MS/tsOMPBS277Q_P1.mp4`

The extraction code requires at least 30% fully detected frames before cleaning
and at least eight frames after cleaning. The saved message does not tell us
which of those conditions failed for each clip. We therefore do not invent a
more specific reason. The
[reviewed exclusions](../artifacts/eval/full-v1/exclusions.json) preserve that limit.
No complete source disappeared through these three exclusions.

The loader requires every raw clip to have either a valid cache or a reviewed
exclusion, never both. It rejects unexpected caches, duplicate clip names,
metadata disagreements, malformed full-data filenames, conflicting condition
labels within a cached source, non-finite arrays, wrong array shapes, short
sequences, and an unexpected cached frame-rate setting. A partially extracted
cache cannot silently become a smaller experiment.

The registry records filenames, labels, source IDs, frame counts, and SHA-256
hashes of all 88 NPZ files. It also records the raw filename inventory and the
exclusions. Its dataset fingerprint covers those records. It does not hash the
raw MP4 content or verify that a cached pose still matches a replaced MP4.
After replacing a video, rerun extraction and create a new dataset version.

## What counts as one group

The 11-character source ID is the grouping unit. For example:

| Clip filename | Source ID |
|---|---|
| `tsOMPBS277Q_P3.mp4` | `tsOMPBS277Q` |
| `tsOMPBS277Q_P5_02.mp4` | `tsOMPBS277Q` |
| `EHymg4AGMJs_P1_01.mp4` | `EHymg4AGMJs` |
| `EHymg4AGMJs_P2.mp4` | `EHymg4AGMJs` |
| `DfRhvdCiUJk.mp4` | `DfRhvdCiUJk` |

The `_P...` suffix distinguishes clips. It is not evidence of a different
participant. We remove the whole suffix and keep the recording together.
The parser also supports the older `_clip-NN` convention, but the full-data
inventory check accepts only the current full-data naming convention.

Source `tsOMPBS277Q` has 14 raw MS clips and 13 usable cached clips. Source
`pFLC9C-xH8E` has seven PD clips. Splitting clips or overlapping windows would
let material from such recordings reach both training and testing. Grouping
happens before window creation, so all their windows follow their source.

## The split algorithm

We use **five-fold cross-validation over sources, with one inner validation
holdout in each outer fold**. This is not a full nested cross-validation search.

1. Sort the unique source IDs. Require one condition label per source. Make a
   table with one row per source, regardless of its number of clips.
2. Apply scikit-learn `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
   to that table. Stratification distributes condition labels across folds.
   Each source belongs to exactly one outer test fold.
3. For outer fold `k`, set that test fold aside. On the remaining sources, apply
   `StratifiedKFold(n_splits=4, shuffle=True, random_state=43+k)`. Take only its
   first split: the smaller part is validation, and the rest is training.
4. Expand source assignments to all their clips. Require disjoint training,
   validation, and test sources; all three conditions in each partition; and
   one outer test assignment for every usable clip.
5. Save the assignments before fitting any model. No score is used to try
   different seeds, rebalance folds, or choose a more favorable split.

Applying `StratifiedKFold` to a deduplicated source table is a grouped,
source-stratified design. Applying it directly to clips would be wrong here.
`StratifiedGroupKFold` on clip rows is another established option, but its
stratification target is the distribution of clip labels. We choose to balance
source counts because the number of clips per source varies greatly. This
choice deliberately allows unequal clip counts between folds.

There are 12–16 usable sources per condition, so five outer folds leave at least
two sources per condition in every test fold. The code refuses an infeasible
fold count rather than silently forcing two folds or dropping a condition.
Five folds give more training sources per round than a two-fold split while
keeping the number of separately trained models practical. This is a study
design choice, not a proof that five is statistically optimal.

The inner holdout reduces computation compared with training every candidate
in all four inner folds. Its selection is noisier. For a wider hyperparameter
search, use a fully grouped inner cross-validation loop within each outer
development set, rebuilding the encoder inside every inner training split.
Do not train an encoder on all outer development sources before scoring its
inner validation sources.

### Exact frozen partitions

Counts below are **sources / clips**. Folds are numbered 0–4 in the code.
Each row covers all 41 sources and all 88 usable clips.

| Fold | Training | Validation | Testing | Test sources: Normal / MS / PD | Test clips: Normal / MS / PD |
|---|---:|---:|---:|---|---|
| 0 | 24 / 51 | 8 / 19 | 9 / 18 | 4 / 3 / 2 | 6 / 4 / 8 |
| 1 | 24 / 58 | 9 / 17 | 8 / 13 | 3 / 2 / 3 | 5 / 3 / 5 |
| 2 | 24 / 54 | 9 / 17 | 8 / 17 | 3 / 2 / 3 | 3 / 3 / 11 |
| 3 | 24 / 53 | 9 / 21 | 8 / 14 | 3 / 3 / 2 | 6 / 4 / 4 |
| 4 | 24 / 45 | 9 / 17 | 8 / 26 | 3 / 3 / 2 | 4 / 15 / 7 |

The large MS source is in fold 4's test set, which explains its 15 MS test clips.
We keep the recording intact even though the clip counts become uneven.
`split_summary()` prints condition counts for every training and validation
partition too. Exact membership is in
[`artifacts/eval/full-v1/fold_registry.json`](../artifacts/eval/full-v1/fold_registry.json).
The registry was created with scikit-learn 1.9.0. Stored assignments are loaded
as written rather than regenerated with whichever library version is installed.

## What can learn from each partition

Pose cleaning and pelvis/torso normalization operate within one clip. They do
not estimate a mean or scale across the collection, so they can be cached before
splitting. This does not make it safe to fit a dataset-wide scaler before splitting.

In each outer fold, the executable procedure is:

1. Build a fresh repaired S-JEPA model with seed 42. Train on training sources
   for 800 updates, using source-uniform window sampling and the label-free loss.
2. Freeze that encoder. Average its window embeddings into one vector per clip.
   The token readout uses mask seed 0 and target ratio 0.6. Fit a `StandardScaler`
   and class-balanced logistic regression on training embeddings only. The
   head uses C=1 and at most 2,000 iterations.
3. Score the head on validation clips using source-weighted macro-F1.
4. Continue the encoder on the same training sources for 400 more updates.
   Model and teacher weights carry forward; optimizer, schedule, and centering
   state restart. Fit a new training-only scaler and head, then score validation.
5. Choose the stage with the higher validation source-weighted macro-F1.
   Ties choose the original 800-update stage. Keep its encoder, scaler, and head.
   There is no refit on validation and no test-based early stopping.
6. Only after selection, compute test embeddings and predictions. Run the paired
   RF and fixed controls using exactly the same training and test clips.

The Random Forest uses 100 trees, maximum depth 5, square-root feature selection,
balanced class weights, and seed 42. Its varying-column filter and scaler fit
only training features. Visibility controls use each joint's mean and standard
deviation of visibility; mean-pose controls use mean normalized x/y positions.
Both use the same fixed linear-head recipe. We report both controls without
choosing the better one based on test results. The majority baseline predicts
the most common training clip label; ties follow Normal, MS, PD order.

Notebook 06 repeats this entire procedure inside all five outer folds. It does
not reuse fold 0's teaching checkpoint in the other folds. Smoke mode uses the
same partitions and selection procedure with a tiny model and 4+2 updates.

Notebook 05 fits its scalers, t-SNE, and UMAP on training clips only and computes
training-only silhouette scores. The plots are exploratory diagnostics. They
are not test scores. The exported embeddings include clip names, source IDs,
the registry checksum, and an explicit training partition label.

## Reporting without overstating the evidence

The output contains one out-of-fold prediction per usable clip for each system.
A prediction is out-of-fold when its source was excluded from the model's
training and validation data. A missing, duplicated, or wrongly attributed
prediction prevents publication of a complete five-fold summary.

Two pooled summaries answer different questions:

- **Clip-weighted:** each of the 88 clips has weight one. A 13-clip source has
  13 times the total weight of a one-clip source.
- **Source-weighted:** a clip from a source with `m` clips has weight `1/m`.
  Each of the 41 sources has total weight one. Weighted confusion counts feed
  accuracy, precision, recall, and F1. Macro averaging then weights the three
  conditions equally. This is still clip classification; predictions are not
  combined into one person-level or source-level diagnosis.

We also show the mean and population standard deviation (`ddof=0`) of the five
source-weighted fold scores. The pooled macro-F1 generally differs from the
mean fold macro-F1 because F1 is not a linear average. Fold SD is descriptive:
the five training sets overlap, so the five scores are not five independent
experiments. No confidence interval, significance claim, or winner selected by
test performance is produced by this change.

## Reproducing and checking the boundary

Run from `experiments/multiple-sclerosis` after creating the full cache:

```bash
# Validate inventory, hashes, exclusions, all partitions, and test coverage.
uv run python scripts/scripts_full_data.py

# Exercise all five folds with real cached inputs and a tiny training budget.
uv run python scripts/scripts_full_data.py --run --smoke --device cpu

# Run the configured 800 + 400 update experiment.
uv run python scripts/scripts_full_data.py --run

# Regression checks and notebook-generator consistency.
uv run --with pytest python -m pytest sjepa/tests -q
uv run python scripts/scripts_build_notebooks.py --check --only 02 03 04 05 06
```

Normal and smoke outputs live in different configuration-specific directories
under `artifacts/runs/full-v1/`. Every evaluation gets a fresh output directory.
Each checkpoint contains the dataset checksum, registry checksum, exact training
clips and sources, fold, model configuration, and training stage. Loading checks
that context before installing weights. Legacy or incompatible checkpoints fail.
Checksums detect changed content; they are not signatures proving who made it.

If inputs change, do not edit the old registry to keep an old checkpoint working.
Create a new directory, review its `exclusions.json` (use `{}` if none), and run
the audit with `--registry artifacts/eval/<new-version>/fold_registry.json`.
Point notebook calls to that new path and train again. The loader checks the
saved assignments against the current data without silently choosing new folds.

The regression suite covers group separation, a 13-clip source, stable source
assignments despite clip multiplicity or input order, infeasible splits,
conflicting labels, reviewed exclusions, changed cache bytes, tampered
registries, incompatible checkpoints, rejection of validation clips at the
training entry point, training-only scaling, source weights, and complete OOF
coverage. Tests should not be read as model-performance evidence.

### Verification performed on September 20, 2026

The full-cache audit passed: all 15 partitions contain the three conditions,
the three partitions in each fold have disjoint sources, and all 88 usable
clips appear in exactly one test fold. The complete package suite passed
38 tests, including 20 split and leakage regression cases. The five generated
notebooks match their source code.

All 39 Python code cells in notebooks 02–06 executed successfully in smoke mode,
including training, checkpoint loading, t-SNE, UMAP, and five-fold evaluation.
The check used a fresh variable namespace per notebook in one Python process,
CPU execution, and a noninteractive plotting backend. Real data and code were
linked into a temporary experiment directory so generated outputs stayed there.
This tested the code cells, not the Jupyter or Colab browser interface.

The command-line five-fold smoke run also completed, with 88 test predictions
per system. The environment used Python 3.12.12, NumPy 2.4.6, scikit-learn 1.9.0,
and PyTorch 2.13.0. The normal 800+400-update experiment was **not** run for this
change. No smoke score is presented as trained-model performance. The
[machine-readable verification record](../artifacts/reviews/2026-09-20-full-data-split-validation.json)
records the dataset and registry checksums, commands, counts, and these limits.

## Limits that grouping does not solve

Source IDs are filename-derived recording IDs. We do not have verified person
identities, so the same participant could appear under different IDs. Reuploads
or overlapping footage under different IDs can also escape this grouping.
Checking those relationships needs additional provenance review.

Grouping does not remove differences in camera angle, frame rate, resolution,
background, compression, or pose-detector confidence. It also does not establish
that labels are clinically verified. The fixed controls provide competing
explanations for good scores; they cannot prove the cause of a learned feature.

All 41 sources have already belonged to a development collection that people
have inspected. No new external test cohort was collected for this change.
Repeatedly changing settings after seeing outer test results would further bias
the estimates. The code enforces the fitting boundary for each execution; it
cannot undo earlier human decisions made after seeing data. These results must
be described as development estimates, not clinical validation.

## Statistical references

- [scikit-learn: StratifiedKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html)
  defines stratified folds and seeded shuffling. We apply it to the source table.
- [scikit-learn: cross-validation with grouped data](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data)
  explains why related observations must stay together and describes group-aware alternatives.
- [scikit-learn: nested versus non-nested cross-validation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html)
  explains the bias from selecting and evaluating settings on the same data.
  Our one-inner-holdout design keeps selection separate but does not average inner folds.
- [scikit-learn: avoiding data leakage](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage)
  explains training-only fitting of preprocessing and the use of pipelines.
