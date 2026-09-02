# GAVD3 S-JEPA gait tutorials

This folder is an executable course on learning motion features from gait video. It starts with a source video, extracts a 33-landmark skeleton, hides selected joint-time tokens, and asks a Skeleton Joint-Embedding Predictive Architecture, or S-JEPA, to predict their latent features. Training begins with normal gait and then continues through four cumulative condition stages.

The completed real run used 159 sequences from 35 source videos. It stayed numerically stable and did not totally collapse. It did not produce clean five-condition clusters. The classifier results are useful engineering diagnostics inside this known corpus. They are not estimates for a new patient, video, camera, or clinic.

![Seven notebook learning path](images/09_notebook_roadmap.svg)

## The question

A gait video contains the motion of interest, but it also contains camera angle, clothing, background, crop quality, and pose-detector failures. This project asks whether a compact latent representation can retain useful motion structure while making every data and evaluation choice visible.

The implementation follows the main learning graph in S-JEPA, but it is a paper-aligned reimplementation rather than official code. The main changes are:

- monocular MediaPipe landmarks replace calibrated 3D laboratory skeletons;
- prediction targets come from one fixed 12-landmark whitelist;
- targets are sampled uniformly, without a motion score;
- a normal-first five-stage curriculum replaces one pooled training run;
- VICReg resists collapse;
- a label-aware group term is active after Stage 0;
- every readout reports video overlap and prior encoder exposure.

![End-to-end method](docs/figures/pipeline.svg)

## Data layers

The project keeps two data layers separate.

|Layer|Sequences|Videos|Purpose|
|---|---:|---:|---|
|Canonical GAVD experiment|96|18|The fixed five-condition inspection and comparison cohort|
|Added normal candidates|64|17|Self-annotated windows from additional YouTube videos|
|Accepted added normal|63|17|Candidates with neurologic-landmark coverage at least 0.45|
|Stage 0 normal total|75|18|12 canonical plus 63 accepted added normal sequences|
|Full curriculum|159|35|75 normal plus 84 canonical non-normal sequences|

The canonical class counts are 12 normal, 9 Parkinson's, 12 stroke, 47 myopathic, and 16 cerebral palsy sequences. The added normal windows use self-annotated time spans and automatic MediaPipe bounding boxes. They are not canonical GAVD annotations and were not independently clinically verified. One of 64 candidate windows had neurologic-landmark coverage of 0.027 and was rejected. Notebook 04 now reads the extraction report as an explicit selection contract, so the accepted cohort does not depend on which pose files happen to be present.

This provenance difference matters. Most normal rows use the added extraction path, while every abnormal row uses the canonical path. A normal-versus-abnormal classifier could learn acquisition or extraction differences as well as gait differences.

![Cohort and curriculum](docs/figures/cohort_curriculum.svg)

## Model in plain language

Each sequence is resized to 64 frames. Four adjacent frames form one time patch, which gives 16 time positions. With 33 joints, the encoder receives 528 possible joint-time tokens.

The view encoder sees a partly hidden sequence. The target encoder sees the complete sequence. The predictor uses the visible view features to predict the hidden target features. The target encoder is not updated by backpropagation. It follows the view encoder through an exponential moving average, or EMA.

The loss has three jobs:

\[
L = L_{\mathrm{JEPA}} + 0.05L_{\mathrm{VICReg}} + 0.25L_{\mathrm{group}}.
\]

- JEPA trains latent prediction.
- VICReg keeps dimensions variable and reduces redundant covariance, which helps resist collapse.
- The group term encourages same-label compactness and a centroid margin after Stage 0.

The group term uses condition labels. Stages 1 through 4 are therefore label-informed representation fine-tuning, not purely self-supervised learning.

### What VICReg, `group`, and `std` mean in the training log

A line such as

```text
JEPA 0.4585  VICReg 12.8508  group 0.0005  std 0.4297
```

mixes losses with one diagnostic, so its short labels need care:

|Printed field|Exact meaning|How to read it|
|---|---|---|
|`JEPA`|Epoch-mean masked latent-prediction loss|Lower means the predictor better matches teacher features at hidden authorized tokens.|
|`VICReg`|Epoch-mean inner value `25 * invariance + 25 * variance + covariance`, before the outer weight 0.05|It aligns two views and resists constant or redundant projected features. It does not use condition labels.|
|`group`|Only the epoch-mean **centroid-separation penalty**|Near zero means batch condition centroids usually met, or nearly met, the margin. It is not the complete group loss.|
|`std`|Mean per-dimension standard deviation of unprojected EMA-teacher embeddings over the entire active corpus after the epoch|A value away from zero is evidence against total collapse. It is not a loss and has no target value of 1.|

VICReg receives two independently transformed views of each sequence. The trainable view encoder processes both, valid tokens from the 12 authorized landmark identities are pooled, and a separate projector maps each sequence to a feature vector. VICReg then applies three terms:

1. **Invariance:** mean squared error between the two projected vectors for the same sequence. It asks two views of one walk to agree.
2. **Variance:** for each projected feature dimension and each view, a hinge penalty `max(0, 1 - standard_deviation)`. It acts only when a dimension has less than the requested spread.
3. **Covariance:** squared off-diagonal covariance. It discourages multiple dimensions from carrying the same changing signal.

The separate group objective uses condition labels and the **unprojected** pooled student representation. It normalizes every sequence vector, averages vectors with the same condition label to form a centroid, and normalizes that centroid. For centroid distance `d` and margin 1.0, the separation penalty is:

\[
\left[\max(0,1-d)\right]^2.
\]

A distance of 1.2 contributes 0; 0.9 contributes 0.01; and 0.5 contributes 0.25. Because unit vectors have distances from 0 to 2, margin 1.0 is equivalent to requiring at least a 60-degree angle, or cosine similarity at most 0.5, between centroid directions. The reported value is averaged across condition pairs and balanced batches, so `group 0.0005` cannot be converted into one exact centroid distance.

The optimized group loss is actually `compactness + separation`: compactness pulls examples toward their own condition centroid, while separation pushes different centroids apart. With the default weights, optimization uses

```text
total = JEPA + 0.05 * VICReg + 0.25 * (compactness + separation)
```

Therefore the printed `group` field understates the complete group contribution. Neither a small group penalty nor nonzero `std` proves clean clinical clusters or unseen-video generalization; they are training-health signals only.

## The 12 landmark identities eligible for prediction masking

The whitelist is expanded and de-duplicated from `experiments/multiple-sclerosis/mapping-data/ms-pd-mapping.md`.

|Indices|Landmarks|
|---|---|
|11, 12|left and right shoulder|
|23, 24|left and right hip|
|25, 26|left and right knee|
|27, 28|left and right ankle|
|29, 30|left and right heel|
|31, 32|left and right foot index|

This is a project whitelist, not a validated neurologic biomarker. All 33 joints may provide context. Only the 12 listed joints may become hidden prediction targets.

The configured mask target is 0.60, but the code uses a batch-safe rule. It takes 60% of the smallest valid eligible-token count in the batch, rounds down, and masks that same count in every sample. It always leaves at least one eligible token visible. The realized mean eligible-token fraction was 0.551 at the end of Stage 0 and 0.423 at the end of Stage 4. The sampler never reads coordinate size, displacement, velocity, acceleration, or a learned motion score.

![Eligible masking region](images/03_neurologic_mask.svg)

## Training curriculum

One model continued through all five stages. Earlier groups remained available through condition-balanced replay.

|Stage|New group|Active sequences|Epochs|Final normal-anchor cosine|
|---:|---|---:|---:|---:|
|0|Normal|75|300|reference|
|1|Parkinson's|84|75|0.954|
|2|Stroke|96|75|0.839|
|3|Myopathic|143|75|0.707|
|4|Cerebral palsy|159|75|0.594|

The completed run used 600 curriculum epochs and 11,400 optimizer updates. Its final checkpoint is:

```text
cache/artifacts/real/sjepa_curriculum_final_augmented.pt
experiment fingerprint:
d0acc2628d134959d8b91e96d5112fc3bed560fe8feb9569e5b13b11a8b614d1
```

The fingerprint identifies the stored experiment and data payload. It is not the byte-level checksum of the checkpoint file.

## What the completed run found

### Training health

The final feature standard deviation was 0.414, so the representation did not shrink to one constant vector. The mean pairwise cosine similarity was 0.609. However, the normal-anchor cosine fell to 0.594. This is substantial drift, so the run does not show strong retention of the original normal representation.

![Training health](docs/figures/training_health.svg)

### Canonical five-condition geometry

Notebook 05 pooled each canonical sequence into a 384-dimensional vector. On the 96 canonical rows:

- cosine silhouette: 0.009;
- minimum centroid distance: 0.0367;
- mean centroid distance: 0.2921;
- mean within-condition distance: 0.1195;
- closest centroids: myopathic and cerebral palsy.

A silhouette near zero means that many samples sit near class boundaries. The minimum centroid distance is also smaller than the mean within-condition spread. These values do not support a claim of clean five-class clustering.

![Canonical representation geometry](docs/figures/representation_geometry.svg)

### Descriptive classifier readouts

|Readout|Accuracy|Balanced accuracy|Macro-F1|Main limitation|
|---|---:|---:|---:|---|
|All-96 stratified S-JEPA|0.793|0.889|0.821|All 16 test videos overlap training; all 29 test rows trained the encoder|
|All-96 missingness-only|0.448|0.466|0.429|Uses visibility only, with no gait coordinates|
|Exact exp5 S-JEPA|0.714|0.730|0.742|All 9 test videos overlap; all 21 test rows trained the encoder|
|Historical 82-feature exp5|0.762|not saved|0.728|Different pose and feature system; same video-confounded split|
|Lane C binary fold mean|0.849|0.874|0.826|Random Forest folds group videos, but the encoder saw all 159 rows|
|Lane C five-class fold mean|0.653|0.603|0.625|Two stratified video-group folds; encoder saw all 159 rows|

The binary Lane C intervals are percentile bootstrap ranges over only five fold scores. They are not population confidence intervals. The corrected five-class lane uses two stratified video-group folds because Parkinson's and cerebral palsy have only two videos each. Every training and test fold now contains all five labels, and macro-F1 always uses the same label list. Its pooled out-of-fold accuracy is 0.654 and pooled macro-F1 is 0.619. Two folds are still too few for a stable performance claim, and the encoder exposure remains complete.

![Readout results](docs/figures/readout_results.svg)

## What “all-96 stratified” means

The split starts with all 96 canonical sequence rows. It keeps about 70% of each class for Random Forest training and 30% for testing. The exact counts are:

|Condition|All|Train|Test|
|---|---:|---:|---:|
|Normal|12|8|4|
|Parkinson's|9|6|3|
|Stroke|12|9|3|
|Myopathic|47|33|14|
|Cerebral palsy|16|11|5|
|Total|96|67|29|

Stratification keeps rare groups on both sides and makes the class mix more stable. It does not make the rows independent. Sequences from the same source video can appear on both sides, and the encoder learned from all 96 rows before this classifier split was made. This lane asks whether a shallow classifier can recover labels from frozen features inside a known corpus. It cannot estimate unseen-video or unseen-patient performance.

![All-96 stratification tutorial](images/10_all96_stratification.svg)

## What a valid generalization test requires

The outer source-video split must happen first. Pose preprocessing rules, all five representation-learning stages, and the Random Forest must then be fitted using only the outer-training videos. The held-out videos can be opened only after the complete pipeline is frozen. Grouping only the Random Forest is not enough.

![Required fold-local evaluation](images/11_nested_evaluation.svg)

## Notebook order

|Notebook|Main output|
|---|---|
|`00_sjepa_from_first_principles.ipynb`|A small S-JEPA learning graph built from first principles|
|`01_gavd_manifest_and_youtube.ipynb`|A traceable manifest and one cached copy of each source video|
|`02_extract_and_watch_skeletons.ipynb`|Versioned 33-landmark pose sequences and alignment checks|
|`03_neurologic_keypoint_masking.ipynb`|The whitelist parser, uniform sampler, and forbidden-target assertions|
|`04_pretrain_sjepa_on_normal.ipynb`|The complete five-stage checkpoint lineage and training history|
|`05_inspect_latent_motion.ipynb`|Prediction, collapse, drift, retrieval, and condition-geometry audits|
|`06_capstone_health_condition_classifiers.ipynb`|Three leakage-aware readout lanes and missingness controls|

Each notebook repeats the code it needs. Later notebooks reject missing, incomplete, wrong-mode, or wrong-cohort artifacts instead of silently falling back.

## Local setup

Run from this folder:

```bash
uv sync
uv run python -m ipykernel install --user --name gavd5-sjepa --display-name "GAVD5 S-JEPA"
uv run jupyter lab
```

Copy `.env.example` to `.env`. The completed augmented real path uses:

```dotenv
GAVD_MODE=real
GAVD_CACHE_DIR=/absolute/path/to/gavd5-draft/cache
GAVD_ARTIFACT_DIR=/absolute/path/to/gavd5-draft/work/artifacts
SJEPA_INCLUDE_AUGMENTED_NORMAL=1
SJEPA_RUN_PROFILE=recommended
```

Leave `SJEPA_INSPECT_CHECKPOINT` and `SJEPA_CLASSIFIER_CHECKPOINT` unset unless you intend to select a specific artifact. With augmentation enabled, notebooks 05 and 06 select `sjepa_curriculum_final_augmented.pt`. Restart the kernel after changing `.env`.

The recommended profile uses 300 Stage 0 epochs, four 75-epoch continuation stages, a 0.999 starting target-encoder EMA, AdamW, gradient clipping, VICReg weight 0.05, group weight 0.25, and group margin 1.0. The quick profile only checks that the data and checkpoint path work.

All artifact-producing code resolves paths from the same settings: `GAVD_CACHE_DIR` holds runtime
cache such as the MediaPipe model, while `GAVD_ARTIFACT_DIR` is the sole root for generated
experiment artifacts. Real artifacts therefore live under `GAVD_ARTIFACT_DIR/real`; smoke artifacts
live under `GAVD_ARTIFACT_DIR/smoke` and have no clinical meaning. Restart the kernel after changing
these settings.

### Augmented-normal pose workflow and legacy migration

With `SJEPA_INCLUDE_AUGMENTED_NORMAL=1`, notebook 04 requires both of these files below the **same**
active real artifact root:

```text
$GAVD_ARTIFACT_DIR/real/poses_augmented/normal/*.npz
$GAVD_ARTIFACT_DIR/real/augmented_pose_extraction_report.csv
```

For a new extraction, first create the matching augmentation CSV/video pair with
`notes/annotate_normal_clips.py`, then run:

```bash
uv run python notes/extract_augmented_poses.py
```

The extractor loads `gavd5-draft/.env` and honours `GAVD_MODE`, `GAVD_CACHE_DIR`, and
`GAVD_ARTIFACT_DIR`. It runs only in `GAVD_MODE=real` and performs all input checks before writing a
report or pose archive. In particular, the augmentation videos must be the exact clips used to create
the CSVs; renaming a full YouTube video to a clip ID is invalid because its frame numbers and crops do
not match the CSV contract.

Older versions wrote valid augmentation artifacts under the hard-coded legacy root
`gavd5-draft/cache/artifacts/real`. If that completed legacy cohort is available, migrate it without
re-extracting poses:

```bash
# Inspection only: validates the report, all 63 eligible archives, and destination compatibility.
uv run python notes/migrate_augmented_pose_artifacts.py

# Copies only after the dry run succeeds. It never overwrites different files or deletes anything.
uv run python notes/migrate_augmented_pose_artifacts.py --apply
```

The migration utility requires `GAVD_MODE=real`, expects exactly 63 report-eligible archives by
default, validates every archive against notebook 04's input contract, and compares SHA-256 hashes
after copying. Use `--source-root`, `--destination-root`, or `--expected-count` only when deliberately
migrating a different known cohort. After a successful migration, restart the kernel and rerun notebook
04 from its first cell.

## Main saved artifacts

- `manifest.csv`, `source_video_census.csv`, and download audits
- `poses/<condition>/<sequence_id>.npz` and pose-coverage reports
- `augmented_pose_extraction_report.csv` and `poses_augmented/normal/*.npz`
- five `_augmented.pt` stage checkpoints and the final alias
- `curriculum_training_history_augmented.csv` and stage summary
- canonical and augmented frozen sequence embeddings
- geometry tables, confusion matrices, error tables, and missingness controls
- `classifier_contract.json`, which binds downstream results to the final fingerprint
- `lane_c_video_disjoint_metrics.csv`, which records representation exposure

## Papers, tutorial, figures, and slides

- [Staged paper](docs/staged_sjepa_gait.md)
- [Long tutorial](docs/staged_details.md)
- [S-JEPA methodology evolution tutorial](docs/staged_evolution.md)
- [Progressive training guide](docs/progressive_training.md)
- [S-JEPA class and tensor-flow guide](docs/tutorials/sjepa_model_internals.md)
- [Maintainer evolution notes](notes/10_sjepa_evolution_tutorial.md)
- [Research notes index](notes/README.md)
- [Documentation build guide](docs/README.md)
- [Presentation source and deck](slides/README.md)

## Authoritative sources

- Ranjan et al., GAVD, *IEEE Access* 2025, [DOI 10.1109/ACCESS.2025.3545787](https://doi.org/10.1109/ACCESS.2025.3545787)
- Abdelfattah and Alahi, S-JEPA, ECCV 2024, [DOI 10.1007/978-3-031-73411-3_21](https://doi.org/10.1007/978-3-031-73411-3_21)
- Assran et al., I-JEPA, CVPR 2023, [DOI 10.1109/CVPR52729.2023.01499](https://doi.org/10.1109/CVPR52729.2023.01499)
- Bardes, Ponce, and LeCun, VICReg, ICLR 2022, [OpenReview](https://openreview.net/forum?id=xm6YD62D1Ub)
- Grishchenko et al., BlazePose GHUM, 2022, [arXiv:2206.11678](https://arxiv.org/abs/2206.11678)
- Google AI Edge, [Pose Landmarker documentation](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker)
- Breiman, Random Forests, 2001, [DOI 10.1023/A:1010933404324](https://doi.org/10.1023/A:1010933404324)
- Roberts et al., grouped validation for structured data, 2017, [DOI 10.1111/ecog.02881](https://doi.org/10.1111/ecog.02881)
- Rousseeuw, silhouettes, 1987, [DOI 10.1016/0377-0427(87)90125-7](https://doi.org/10.1016/0377-0427(87)90125-7)

## Responsible use

Folder conditions and `gait_pat` values are dataset annotations, not diagnoses made by these notebooks. A low loss or high in-corpus classifier score does not prove clinical usefulness. Always report the checkpoint fingerprint, source-video support, class support, extraction provenance, representation exposure, missingness control, and confusion pattern. Treat every current classifier as a descriptive research tool, not a diagnostic system.
