# Gait dynamics as an observation layer for medical world models

This research restart asks whether past articulation supports useful future prediction, and whether opposite-leg timing contributes information beyond own-leg history in a shared body reference. It uses complete eligible GAVD annotation spans, a nominal 10 Hz sampling contract, normal-only fitted models, and nested source-grouped evaluation. The neurological motivation is explicit, but the study does not infer neural circuitry, diagnoses, or treatment effects from video poses.

The completed forecasting experiment is **`medwm-dynamics-audit-v1`**. Fresh extraction completed **46,783 real pose-model calls**. The evaluated cohort contains **2,815 windows from 456 sequences and 91 source videos**, including **29 normal sources**. Eight learned forecast variants and four reference/control predictions are retained, with 80 saved fold/leg/variant checkpoints. Some variants share the same fitted ridge coefficients and differ in shrinkage; 80 checkpoints are not 80 independent studies.

The primary bilateral-history forecast changes normal source-average error over the eight nominal future samples from 0.1 to 0.8 seconds by **33.5% relative reduction** compared with persistence. The paired difference is **0.04063**, with a conditional 95% source-bootstrap interval of **[0.02399, 0.05911]**. Positive differences favor the forecast. This is exploratory internal evidence after earlier GAVD outcomes were already known. The current-pose-only learned transition forecasts better than the history model, so a stronger claim that bilateral history is necessary is unsupported. Shrinkage did not improve the unshrunk bilateral model. The prespecified excess-error score has source AUROC 0.793, with interval [0.680, 0.896]. Read the complete paired comparisons before interpreting timing or clinical claims.

The [standalone draft02 manuscript](docs/draft02/manuscript.pdf) and its [evidence audit](docs/draft02/README.md) now explain the anomaly result alongside a post hoc low-motion control. Negating persistence error gives AUROC **0.806**, compared with **0.793** for excess error. The paired audit does not establish an incremental learned anomaly advantage. This is a positive forecasting and methodological finding, with an explicit next test: add information beyond movement amount, posture, and observation quality.

Start with the [next-iteration research instruction](../../notes/aaai-worldmodels/05-iterate-02-INSTR.md), [iteration02 evidence and reproduction guide](docs/iteration02/README.md), [current illustrated slides](slides/iteration02.html), and [vector slide PDF](slides/iteration02.pdf). The [draft01 report](docs/draft01/README.md), [research review](docs/restart/research_review.md), and [implementation review](docs/restart/implementation_review.md) preserve the earlier evidence. The [prior README](README.before_restart.md), earlier eight notebooks, and original run identities preserve the previous short-prefix JEPA study. Its scores are historical evidence, not results of this restart.

## What changed and why

The earlier study used the first 16 or 32 sampled frames of each sequence. Its temporal controls did not establish useful dynamics. The restart processes complete eligible annotated spans and predicts eight future observations from sixteen prefix observations. This nominal 0.8-second forecast horizon is comparable across source frame rates. The observed prefix spans 1.5 seconds between endpoints. The model targets body-relative 2D articulation, not global locomotion or physical speed.

Normal-only ridge predictors compare own distal-leg history, bilateral history, static bilateral posture, and a matched opposite-leg timing control. A window-specific hashed shift changes the alignment of the same selected coordinate/validity pairs while retaining the current anchor. This avoids the invalid control of one globally fixed feature permutation that a linear model could simply absorb. Persistence and damped velocity remain explicit references.

A prospective amendment adds a conservative blend of persistence and each learned residual. The blend strength and ridge regularization are selected using inner normal-source validation only. A blend strength of zero means persistence was chosen. Original unshrunk arms remain primary. Failed comparisons and fixed anomaly-score directions are preserved. The [protocol](docs/restart/protocol.md) records the design decisions before fitting, and the [model contract](docs/restart/model_contract.md) specifies the implemented mathematics and artifacts.

## Dataset coverage and generalizability

The complete local GAVD inventory has **1,874 sequences, 458,116 annotation rows, and 348 source IDs**. Upstream `source/` CSVs are provenance and are not counted again. The five requested labels contain 666 sequences, 140,641 rows, and 103 source IDs. Media checks retain 652 sequences from 98 sources. Preserving two documented source identity exclusions leaves 641 candidates from 96 sources. Pose quality and common target support produce the final cohort below. The repeated stroke entry in the task denotes one class.

| Dataset annotation | Windows | Sequences | Source videos |
| --- | ---: | ---: | ---: |
| cerebral palsy | 518 | 52 | 8 |
| myopathic | 845 | 168 | 27 |
| normal | 465 | 141 | 29 |
| parkinsons | 292 | 37 | 11 |
| stroke | 695 | 58 | 16 |

All **32 annotated normal source IDs** are already in the selected five-label scope. One normal source has a known video-reference mismatch. The seven other labels add 245 source IDs without cached media and include exercise, pregnancy, stylized, prosthetic, antalgic, inebriated and broadly abnormal walking. They can support a separately defined transfer study, but cannot increase normal-source support for the primary forecasting question. We audited them before outcomes instead of adding observations until a significance threshold was crossed. See the [complete data audit](docs/restart/data_audit.md).

**GAVD supports an exploratory heterogeneous-source benchmark; it does not by itself establish clinical generalizability.** Source IDs are grouping proxies, not verified participant IDs. A person can recur across uploads, and a source can contain several people. More overlapping windows do not create more independent patients. Annotation folders do not provide independently verified diagnoses. Confidence and person-box filtering select a subset of observed footage. External participant-disjoint and site-disjoint evaluation, verified identity and clinical labels, and prospective capture are needed for stronger claims.

The five source folds preserve every related sequence and window. Inner validation, feature scaling and model fitting use normal training sources only. Errors are averaged across observed coordinates within each leg, equally across legs, then over windows, sequences and sources. Source-bootstrap intervals condition on the fixed overlapping cross-validation fits. They are descriptive uncertainty estimates for this exploratory run, not confirmatory patient-population significance tests. Raw forecast error, extra bilateral information, and annotation association are separate endpoints. Pooled out-of-fold anomaly ranking assumes that scores from separately fitted fold models are comparable. The model cohort excludes three of the 2,818 prepared windows because they fail the predeclared common distal-target support gate.

## Media, pose and observation provenance

All originals remain in `data/videos-gavd`. Existing condition folders hold distinct source IDs and are reused in place. Any future download must use the shared Ambient downloader and `data/videos-gavd/sources/<source_id>.*`, with previews under `data/videos-gavd/previews/`. This restart needed no new download because all available requested-condition originals were already present. Missing/private videos remain recorded, and no bulk media is staged.

The adapter uses Ambient's `SequenceKeypointExtractor.extract_from_image`, verifies source-video hashes, and records annotations, model weights, source code, frame numbers, image dimensions, confidence and decoder times. GAVD frames are one-based; the conversion to OpenCV indexes occurs once. Nearest-native-frame sampling does not interpolate joints. A prefix-only torso scale prevents future data from changing the observed reference. Per-frame pelvis centering means the target is relative articulation. Both hips enter this reference, so the own-leg condition is not unilateral raw sensory access.

Confidence must be at least 0.5. Prefix and future each require at least 70% valid observations, plus shared distal-target support for both legs. Unresolved annotation geometry is invalidated. Full-image first-person detection and pelvis containment still cannot guarantee person identity. Full-duration visual QA is limited; its deterministic contact sheet and frame ledger are retained. Read the [observation contract](docs/restart/data_contract.md) for exact arrays, time conventions, exclusions and cache validation.

## Run from the repository root

Local prerequisites are the current `data/gavd` annotation tree, the authorized source videos under `data/videos-gavd`, and `data/models/pose_landmarker_lite.task`. The latter is the exact local pose-weight file pinned by this run. Use the experiment's Python 3.12 environment and shared editable Ambient package:

```bash
uv sync --project experiments/aaai-worldmodels --python 3.12
uv run --project experiments/aaai-worldmodels jupyter lab experiments/aaai-worldmodels
```

Run the following from `experiments/aaai-worldmodels` to reproduce the current research revision from the immutable completed runs:

```bash
.venv/bin/python scripts/build_draft03.py --compile --render
.venv/bin/python scripts/build_iteration02_notebook.py --execute
.venv/bin/python scripts/build_iteration02_slides.py
.venv/bin/python scripts/build_restart_readme.py
.venv/bin/python -m pytest -c pyproject.toml tests -q
```

The completed extraction and fit came from `worldmodels.restart_data` and `worldmodels.restart_experiment`. A new fit needs a fresh output directory and declared method identity, as specified in the iteration02 instruction. The original restart builders reproduce historical teaching artifacts and can change the old draft or slide entry point; use the current commands above for this revision. Optional replay of notebooks 10–15 uses `scripts/execute_restart_notebooks.py` and preserves their separate execution record.

The default teaching and publishing scripts consume the completed declared run. A new fitted run must be explicitly propagated through those input paths; do not overwrite an old run or quietly relabel its products. Data extraction can resume incomplete work only when its input and code identities match. Heavy arrays and fitted checkpoints remain local; small evidence tables, generated documents, vector figures, source generators and execution records support review. Local configuration uses this experiment's ignored `.env`; no paid LLM call is required for the pose/ridge experiments.

## Execute the progressive notebooks

| Notebook | Actual activity |
| --- | --- |
| [10: Question and inventory](10_restart_question_and_inventory.ipynb) | Rebuild the complete annotation inventory and trace cohort attrition. |
| [11: Video, pose and time](11_restart_video_pose_and_time.ipynb) | Decode an actual source frame, call the pose model, and inspect native timing. |
| [12: Representation and controls](12_restart_representation_and_controls.ipynb) | Verify prefix-only features, missingness and matched target support. |
| [13: Fitting and source folds](13_restart_fit_and_cross_validation.ipynb) | Fit a real ridge model and inspect the complete nested source evaluation. |
| [14: Results and uncertainty](14_restart_results_and_uncertainty.ipynb) | Read all paired source effects, anomaly rankings and condition tables. |
| [15: Manuscript claims](15_restart_manuscript_and_claims.ipynb) | Connect results to source-backed claims and generalization limits. |

Each notebook initializes its own paths and imports. Executed copies are in `artifacts/executed-restart`, with per-notebook hashes and status in [the execution record](docs/restart/notebook_execution.json). The [run ledger](docs/restart/run_record.json) records independent specialist work, verification and limitations. The current iteration02 deck has editable SVG slides, full-sentence explanations, tables, flowcharts and result figures, plus a vector PDF. It distinguishes measured findings from proposed masking and calibration experiments. Notebook [16: Evidence and mask design](16_iteration02_evidence_and_mask_design.ipynb) replays the score audit and tests exact-budget masking examples on real cached poses; it does not train a new model. Proposed notebooks 20–26 are specified in the research instruction. Previous slides are preserved as [the historical deck](slides/tutorial_before_restart.html).

The [official workshop page](https://med-wm.github.io/aaai27-mwm/) lists November 20, 2026 Anywhere on Earth, a 6–8-page full-paper route and a 2–4-page short-paper route, excluding references and appendices. The portal deadline still requires an independent OpenReview check. Submission is not performed by this project. The manuscript readiness report identifies unresolved scientific and author-information requirements.
