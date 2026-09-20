# S-JEPA for gait: normal, MS, and Parkinson's

These notebooks turn walking videos into skeleton sequences, learn motion features
with S-JEPA, and compare those features with a Random Forest. The three dataset
labels are normal gait, multiple sclerosis (MS), and Parkinson's disease (PD).
They are labels supplied with the collection, not diagnoses made by this project.

## The current dataset

Notebooks 02–06 use `video-data-full/` through the pose cache in
`artifacts/keypoints-full/`. As checked on September 20, 2026:

| Condition | Raw clips | Usable cached clips | Source videos in cache |
|---|---:|---:|---:|
| Normal | 26 | 24 | 16 |
| MS | 30 | 29 | 13 |
| PD | 35 | 35 | 12 |
| Total | 91 | 88 | 41 |

Notebook 01 skipped three clips at its pose-quality gate. Their filenames and the
limits of the saved rejection log are recorded in the
[split methodology](docs/11-full-data-splits.md#data-and-exclusions).
The full cache must be available before running notebooks 02–06. The older
`artifacts/keypoints/` cache and `g1` results are historical; they are not results
for the full dataset. See the [video download instructions](docs/10-video-data-full-lfs.md)
if the MP4 files are still Git LFS pointers.

## How we keep training and testing separate

Imagine cutting one walking video into ten clips. A random clip split might put
eight in training and two in testing. The model would then be tested on motion,
a person, and a background it had already seen. That can make its score look too good.

We split **source videos first**. For example, `tsOMPBS277Q_P3.mp4` and
`tsOMPBS277Q_P5_02.mp4` both belong to source `tsOMPBS277Q`. All clips from that
source, and every window cut from those clips, stay together within a round.
The suffix is not treated as a separate person.

We use five-fold cross-validation: divide the source videos into five groups,
then take turns testing on each group. We apply scikit-learn's `StratifiedKFold`
to one row per source, so each test group has a similar mix of condition labels.
This gives source-grouped folds without letting a source with many clips dominate
the allocation. A second split within the remaining sources creates validation
data. Its first fold is used as a holdout; we do not run a full inner CV search.
The [statistical method and exact seeds](docs/11-full-data-splits.md#the-split-algorithm)
are documented with links to the scikit-learn references.

Each round uses roughly 60% of sources for training, 20% for validation, and 20%
for testing. These percentages describe sources, not clips or frames. In round 0,
the actual counts are:

| Purpose | Sources | Clips | Allowed use |
|---|---:|---:|---|
| Training | 24 | 51 | Train S-JEPA, fit scalers and classifiers, inspect training plots |
| Validation | 8 | 19 | Choose the original or continued S-JEPA checkpoint |
| Testing | 9 | 18 | Score the chosen model after the choice is fixed |

S-JEPA never trains on validation or test motion, even without labels. Each outer
round starts with a new model. A source can be in training in one round and testing
in another; its test prediction always comes from a model that did not learn from
or choose settings using that source.

The shared [fold registry](artifacts/eval/full-v1/fold_registry.json) lists every
clip and source assignment. Loading it checks cache contents and metadata, reviewed
exclusions, class coverage, disjoint partitions, and complete test coverage.
Checkpoints carry the dataset, registry, fold, training membership, stage, and
configuration. An incompatible checkpoint raises an error before its weights load.

## What each notebook does

| Notebook | Role |
|---|---|
| `00_overview_and_video_gallery.ipynb` | Inspect the collection and its labels |
| `01_pose_extraction_from_raw_video.ipynb` | Extract, clean, and cache skeletons |
| `02_anatomical_mask_and_tokenization.ipynb` | Check all folds; demonstrate masks using a training clip |
| `03_sjepa_model_and_pretrain_normal.ipynb` | Train label-free S-JEPA on round 0's training sources |
| `04_progressive_finetune_ms_pd_vicreg.ipynb` | Compare original and continued training using validation sources |
| `05_representation_visualization.ipynb` | Plot training embeddings and visibility controls only |
| `06_capstone_rf_vs_sjepa.ipynb` | Train fresh models in all five rounds; save paired test predictions |

Some filenames retain earlier experiment names. Notebook 03 now trains on all
three conditions in its training partition. Notebook 04 uses label-free additional
training and supervised linear probes; it does not use class-aware VICReg.
The model uses stochastic masks so each joint can be visible context or a target.
That variation is across fresh mask draws; a joint may be hidden for an entire
individual window. Notebook 02 now replays the same motion with eight sampled
masks and reports per-joint and exact-time coverage. A static timeline shows all
mask bits without relying on GIF playback. From this experiment directory, run
`python scripts/scripts_mask_demo.py` to refresh all three historical GIF paths
and the timeline, or add `--check` to detect stale/overwritten output.
See [mask semantics and the hip audit](docs/12-mask-visualization.md).

## Run locally

From the repository root:

```bash
cd experiments/multiple-sclerosis
uv sync
uv run jupyter lab
```

Python 3.12 is required by this project's dependency configuration. Run notebook
01 if the full pose cache is missing, then run 02–06 in order. The root `.env` may
set `SJEPA_PROFILE=laptop` or `SJEPA_PROFILE=gpu`. The laptop profile uses 32-frame
windows and a smaller encoder; the GPU profile uses 64-frame windows and a larger
encoder. Set `SJEPA_SMOKE=1` for a tiny model and very short execution checks.

Audit the data and folds without training:

```bash
uv run python scripts/scripts_full_data.py
```

Run the same five-fold evaluation used by notebook 06:

```bash
# Execution check: all five folds, tiny model, 4 + 2 updates per fold.
uv run python scripts/scripts_full_data.py --run --smoke --device cpu

# Normal experiment: 800 + 400 updates per fold; may take substantially longer.
uv run python scripts/scripts_full_data.py --run
```

Outputs go into separate directories under `artifacts/runs/full-v1/`. Smoke
checkpoints and results are separate from normal runs. Each completed evaluation
saves `oof.json` (one test prediction per usable clip) and `results.json`.
Neither a short smoke run nor successful tests establish model quality.

The notebooks also contain Colab setup cells. Edit the repository URL to your
fork, obtain the full videos with Git LFS, and run extraction before training.

## How to read the results

Notebook 06 compares Random Forest, the validation-selected S-JEPA probe,
visibility and mean-pose controls, and a training-majority baseline. It reports
both clip-weighted and source-weighted scores. In the latter, a source with
13 clips gets the same total weight as a source with one clip. Macro-F1 balances
the three conditions. Fold standard deviation describes variation across rounds;
it is not a confidence interval.

Source IDs are not verified participant IDs. The same person or reposted footage
could appear under different IDs. Recording conditions may also predict labels.
The collection has been inspected during development, and there is no separate
external test cohort. These scores estimate performance on held-out source videos
within this collection; they do not establish diagnostic accuracy or clinical use.
See the [full audit and limitations](docs/11-full-data-splits.md).

## Checks and code locations

```bash
uv run --with pytest python -m pytest sjepa/tests -q
uv run python scripts/scripts_build_notebooks.py --check --only 02 03 04 05 06
```

`sjepa/splits.py` owns the registry and partition checks.
`sjepa/full_experiment.py` owns training, validation selection, and test scoring.
`scripts/notebook_content.py` and `scripts/notebook_full_data.py` generate notebook
content. Rebuild only the requested notebooks with `--only 02 03 04 05 06` to avoid
overwriting work in notebook 01. The detailed method lists what was actually
verified for this change.

## Research background

- [S-JEPA](https://sjepa.github.io): skeleton features learned by predicting hidden features.
- Pose extraction and classical gait features reuse the repository's `ambient` package.
- The [methodology document](docs/11-full-data-splits.md#statistical-references) links
  the statistical and leakage-prevention references used for this split.
