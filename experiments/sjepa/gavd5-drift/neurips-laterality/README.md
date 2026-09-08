# NeurIPS laterality: inductive notebook suite

This directory contains the new held-out-source-video workflow. The top-level legacy notebooks remain historical evidence and are not imported.

The registered GAVD cohort contains **93 source videos and 625 accepted pose
sequences**. Five source-level outer folds use 74–75 sources (436–553 sequences)
for training and 18–19 sources (72–189 sequences) for testing. See the
[canonical split reference](docs/TRAIN_TEST_SPLIT.md) for every fold, the
annotation census, inner-validation counts, and leakage controls.

## Run order

1. `00_protocol_and_governance.ipynb` — freeze the claim and check submission governance.
2. `01_cohort_and_target_audit.ipynb` — build a QC-only cohort and validate the paired-valid target.
3. `02_source_level_splits.ipynb` — create source-balanced outer and inner folds.
4. `03_fold_local_training.ipynb` — train label-blind encoders within folds and seeds.
5. `04_held_out_evaluation.ipynb` — score held-out sources and audit direct checkpoint token equivariance.
6. `05_aggregate_statistics.ipynb` — compute per-checkpoint gates, matched controls, and source bootstraps.
7. `06_external_subject_gate.ipynb` — validate—but do not execute—an optional subject-indexed external cohort contract.

## Research continuation: notebooks 07–10

The [research directions and rationale](RESEARCH_DIRECTIONS.md) connect the
completed results to four new step-by-step tutorials:

8. [07 — Research questions and diagnostics](07_research_questions_and_diagnostics.ipynb): review the actual findings, examine temporal pooling, and optionally compare the processed inputs with the original target.
9. [08 — Matched-budget masking](08_matched_budget_masking.ipynb): compare gait-informed and uniform prediction targets while hiding the same number of valid tokens. The real grid uses available MPS/CUDA acceleration and validates exact cached results before reuse. Its [parameter reference](docs/MATCHED_BUDGET_MASKING.md) explains the 1,200-update budget, full and pilot scopes, runtime design, and every visible setting.
10. [09 — Symmetry-aware JEPA](09_symmetry_aware_jepa.ipynb): compare augmentation with an explicit reflection loss, retaining prediction and feature-variation checks.
11. [10 — Past-only movement prediction](10_past_only_movement_prediction.ipynb): build timestamp-aware forecasting with leakage tests and simple motion baselines.

Their default training uses synthetic examples only. Real-data training requires
an explicit opt-in, and all new results are exploratory. Notebook 07 can read
available saved aggregate results without training. These additions do not
change the completed protocol, existing notebooks, or checkpoint implementation.
Their helper code is isolated under `laterality_extensions/`.

```bash
.venv/bin/python neurips-laterality/scripts/verify_research_notebooks.py --execute-smoke --save-executed
```

This separate checker runs the extension tests and fresh-kernel demonstrations.
It can save executed review copies and vector graphics under
`executed/research_extensions/`; canonical notebooks stay output-free. Editable
tutorial sources live in `tutorials/`, and `scripts/build_research_notebooks.py`
can regenerate selected notebooks with `--only`. The original builder and execution workflow below still
cover 00–06.

## Comparative masking and future prediction: notebooks 11–14

The [experiment specification](docs/COMPARATIVE_MASKING_PLAN.md) explains which
question each comparison addresses. The tutorials run independently in fresh
kernels, using short synthetic examples by default:

- [11 — Masking patterns and coverage](11_masking_patterns_and_coverage.ipynb) compares eight policies, their actual hidden counts, and the information left visible.
- [12 — Controlled masking pretraining](12_controlled_masking_pretraining.ipynb) pairs training conditions and displays the saved real-data recipe and planned workload.
- [13 — Encoder and predictor evaluation](13_masking_encoder_and_predictor_evaluation.ipynb) tests feature prediction, frozen movement readouts, missing observations, and source-level uncertainty.
- [14 — Future features and movement prediction](14_future_features_and_movement_prediction.ipynb) decodes predicted future features and compares them with observed-future diagnostics and past-only references.

```bash
.venv/bin/python neurips-laterality/scripts/build_research_notebooks.py --only 11 12 13 14
.venv/bin/python neurips-laterality/scripts/verify_comparative_notebooks.py --execute
```

The verifier saves executed teaching copies and vector figures in a new directory
under `executed/comparative_masking/`. Real-data training is explicitly enabled
inside the relevant notebooks after the workload is displayed. New empirical
results remain pending; successful synthetic execution verifies the software.

For whether to extend Notebook 06, see the [external evaluation assessment](docs/external_evaluation_assessment.md).
It recommends choosing a useful independent outcome and compatible dataset
before building an actual subject-held-out evaluation, while preserving the
current readiness check.

## Running the registered workflow (00–06)

The notebooks default to `LATERALITY_PROFILE=smoke`. Paper runs must be requested explicitly:

```bash
LATERALITY_PROFILE=paper uv run jupyter lab neurips-laterality
```

Before allocating paper compute, execute the complete synthetic integration test:

```bash
uv run python neurips-laterality/scripts/verify_suite.py --execute-smoke
```

This launches every notebook in a separate kernel and discards all generated smoke artifacts. The smoke score values are deliberately non-evidentiary.

## Inline figures and executed copies

Every notebook renders a profile-labeled inline figure plus its audit tables when executed. Canonical notebooks remain output-free so stale smoke or partial-paper output cannot masquerade as current evidence. To create a separately saved set with all inline output:

```bash
.venv/bin/python neurips-laterality/scripts/execute_notebooks.py --profile smoke
```

The saved copies appear under `neurips-laterality/executed/smoke/protocol_<digest>/<timestamp>/` and are git-ignored. The executor fails closed unless every code cell ran, no error output was stored, at least one PNG figure is embedded, and at least one separate result payload is visible; those counts are recorded in notebook metadata. Every smoke figure is visibly marked **SYNTHETIC SMOKE — NON-EVIDENTIARY**. A deliberate paper run uses `--profile paper --confirm-paper-run`; it preserves existing executed copies and refuses to overwrite a non-empty output directory. Figures contain aggregate pose-derived diagnostics only—never raw video, URLs, frames, or source identifiers.

Useful environment overrides:

```bash
LATERALITY_ARTIFACT_ROOT=/absolute/output/path
LATERALITY_FOLDS=0,1
LATERALITY_SEEDS=42,43
LATERALITY_VARIANTS=vanilla,reflection_augmented
```

Run notebooks in numeric order. A complete paper profile trains 5 folds × 5 seeds × 2 registered variants (50 independent encoders); subset overrides are useful for operational checks, but Notebook 05 marks their report incomplete. Valid checkpoints resume only when their protocol, cohort, split, implementation, runtime, fold, seed, variant, and source lists all match.

Default artifacts are nested under a protocol-digest directory, so a methodology revision cannot overwrite or silently reuse an older run. Existing pre-v2.1 paper artifacts remain untouched but are not valid evidence for the strengthened estimands.

See `RUNBOOK.md` for the staged execution plan, current verification status, real-data dry-audit command, and the exact boundaries between compute, evidence, and submission readiness. Split counts and semantics are maintained in [`docs/TRAIN_TEST_SPLIT.md`](docs/TRAIN_TEST_SPLIT.md).

Generated artifacts are excluded from git. Every checkpoint and result records the protocol digest, cohort digest, source IDs, fold, seed, and variant. Run `uv run python neurips-laterality/scripts/verify_suite.py` for the lightweight test and notebook-structure checks.

Artifacts contain linkable source-video identifiers and derived pose representations. Git exclusion is not a release determination: do not redistribute manifests, poses, embeddings, predictions, or checkpoints until the reviews in `governance/status.json` explicitly permit it.

The optional external gate additionally requires `LATERALITY_EXTERNAL_MANIFEST`
and `LATERALITY_EXTERNAL_GOVERNANCE`. The latter must be scoped to that exact
external dataset; the GAVD governance file is deliberately rejected. The
optional `LATERALITY_EXTERNAL_POSE_ROOT` confines accepted pose paths to one
directory. Notebook 06 checks the process environment and then local `.env`
files for only these three names, without displaying their values.

With neither required setting present, Notebook 06 finishes its preflight at
100% and reports the optional study as `not configured / not run`; this is not
an error and does not block notebooks 00–05. A partial configuration or an
invalid supplied contract remains red and fail-closed.

See [`docs/EXTERNAL_EVALUATION_GATE.md`](docs/EXTERNAL_EVALUATION_GATE.md) for
the complete preparation checklist. The files under `governance/` are
structure-only templates; filling them with invented approvals or subject IDs
would not authorize an evaluation. A green Notebook 06 validates the manifest
contract but still does not run a model or create external evidence.

## What this fixes

- the encoder no longer sees outer-test source videos;
- labels do not enter the primary representation objective;
- long source videos do not dominate training or metrics;
- invalid coordinate sentinels do not define the target;
- mirror behavior is evaluated on held-out sources;
- strict token equivariance is measured before a read-out against the paired initial encoder;
- native mirror residuals are squared per checkpoint before seeds are aggregated;
- absolute held-out predictive utility is required in addition to beating a random initialization;
- odd/even, free/zero-origin, learned/random, and single/two-pass effects have capacity-matched lanes;
- combined nuisance and learned-plus-nuisance lanes test incremental utility beyond measured shortcuts;
- statistical intervals resample source-video clusters and state that they are conditional on the fitted cross-validation pipeline;
- governance and unseen-person limitations are explicit machine-checked gates.

## What remains intentionally limited

The source video—not the person—is the independent unit. The defensible statement is post-development, within-GAVD cross-validated held-out-video performance. “Training-induced symmetry” is conditional on the BlazePose left/right schema, pose preprocessing, architecture, fixed identity-channel action, registered seeds, and measured controls. It is not symmetry discovered without supplied structure. An unseen-person claim requires a separately reviewed dataset with persistent subject identifiers and an implemented external evaluation; Notebook 06 only validates its prerequisites. Folder names remain annotations, not diagnoses, and no clinical claim is supported.
