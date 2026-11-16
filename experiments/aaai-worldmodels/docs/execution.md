# Execution and evidence status

The project has completed real GAVD pose extraction, classifier refitting, three-seed forecasting development, and one declared objective-weight sensitivity. The final cohorts are the person-reviewed derivatives below. Earlier pilot outputs remain preserved as development evidence. They must not be substituted for the final cohorts or counted as additional independent observations.

The joint final evaluation completed all four frozen comparisons. The [completion record](test_execution.json) pins their result files and source-level CSVs. All eight current notebooks completed real execution, including the updated direct-binary video demonstration in notebook 06. The [notebook execution record](notebook_execution_real.json) reports 88 executed code cells, current source/output hashes, and released final-result display. This report records execution and provenance rather than repeating the final performance tables.

## Current cohorts

| Cohort | Frames and prefix | Windows | Sources | Train / validation / test sources |
| --- | --- | ---: | ---: | --- |
| `pilot-input-clean-v1` | 16 sampled frames; 8-frame prefix | 405 | 89 | 53 / 19 / 17 |
| `extended-input-clean-v1` | 32 sampled frames; 16-frame prefix | 407 | 91 | 53 / 19 / 19 |

These are separate eligible cohorts. Their different source counts follow duration and pose-quality requirements. A window is not an independent person, and a YouTube source does not establish a unique patient. Both cohorts use the same five dataset labels and source-grouped partition assignments. They sample every second annotated source frame, so sixteen future frames do not imply the same number of seconds across videos.

The refreshed inventory originally contained 666 selected sequences from 103 source IDs. Media eligibility retained 652 sequences from 98 sources. The study uses bounded prefixes of eligible sequences, not every frame of every walking video. Confidence, missingness, frame numbering, actual sample rates, annotation identity and source provenance remain explicit.

## What happened, in order

1. `pilot-input-v1` tried one sequence per eligible source. It retained 80 windows from 80 sources. Its cerebral-palsy test source lost its selected sequence to pose filtering, and the comparison refused to proceed with a missing class. Sources were not reassigned to repair the split.
2. `pilot-input-v2` expanded selection to all media-eligible sequences, with sixteen sampled frames per sequence. It retained 494 windows from 96 sources in 695.59 seconds. The five-class baseline and direct-binary development experiments used this earlier cohort. These results are now quarantined as development artifacts.
3. `extended-input-v1` decoded sources in groups, reused verified raw pose observations, and extracted 32-frame prefixes. It added annotation-box pelvis containment and acquisition-quality metadata. It retained 414 windows from 93 sources in 262.4 seconds.
4. The forecast revision trained on normal training sources only. Seeds 27, 41 and 73 each used 200 shared uniform-pretraining updates, followed by 300 updates for each of `direct_det`, `residual_det` and `residual_slv`. Every arm used the same saved source schedule. The latent arm used sixteen context-prior samples for evaluation. Failed scientific criteria were retained rather than used to hide arms.
5. A training-only gradient diagnosis motivated exactly one secondary audit: coordinate weight 10 instead of 1 for `residual_det`. Each seed restarted from its original shared pretrained checkpoint, used its original 300-update schedule and a fresh optimizer, and changed only that loss weight. There was no further weight search. Weight 1 remains primary. See [the gradient evidence](training_gradient_audit.json) and [the amendment](revision_amendments.md).
6. Independent visual review found scene cuts and person identity changes in whole stroke sources `CMls0LwK5OU` from validation and `eBvzFkcvScg` from training. Both complete source groups were excluded before joint final-test release. The review did not infer a person's role or diagnosis from appearance. The [exclusion ledger](source_exclusions.json) records inspected frames, reviewers and the contact-sheet hash.
7. The short classifier cohort was rebuilt from verified raw MediaPipe caches and fresh annotation CSVs, with the person gate applied before eight-frame-prefix normalization. All short-cohort encoder/classifier fits were repeated. The direct binary primary remained the protocol-declared mixed-adaptation JEPA probe, even though another JEPA variant ranked higher on clean validation.
8. The long cohort removed seven abnormal training/validation windows. Its normal training rows, normal validation rows and every test row remained byte-identical, including validity and metadata order. Forecast checkpoints and stochastic validation predictions were therefore inherited. This is **checkpoint reuse, not retraining**. The [reuse certificate](forecast_reuse_certificate.json) records equality for 84 normal-training windows from 17 sources, 27 normal-validation windows from six sources, and 59 test windows from 19 sources.

The clean classifier refit retained seed 27 and the five-epoch rule. The frozen implementation derives its stage-update count from the retained normal training windows, producing **60 updates per stage instead of 95** in the earlier cohort. Data support, person gating and effective update count changed together. Their combined effect must not be attributed solely to the two source exclusions.

## Setup and indispensable inputs

Run commands below from `experiments/aaai-worldmodels` using Python 3.12:

```bash
uv sync
uv run python -m ipykernel install --user --name medwm-gait --display-name "Med-WM gait (Python 3.12)"
```

The experiment loads its own `.env` through the project's dotenv aliases. Do not copy credentials into notebook outputs. Local prerequisites are the current `data/gavd` annotations, their manifest, the authorized source videos under `data/videos-gavd`, and `data/models/pose_landmarker_lite.task` under the repository root. This model path is currently fixed in the extended and clean preparation adapters. These experiment stages do not need an LLM API call.

The commands below describe the recorded dependency graph. Existing run directories are immutable. Replaying the graph requires an isolated working copy or a separately declared study identity with consistent paths. In particular, `clean_data` uses fixed parent/output paths; changing only one command's output name will not redirect its dependencies.

## Strict reproduction order

**First create the sixteen-frame raw pose cache.** The clean adapter requires a verified raw cache for every retained candidate sequence, including sequences that later fail a window-quality check:

```bash
uv run python -m worldmodels.workflow prepare --full --max-frames 16 --output artifacts/runs/pilot-input-v2
```

**Then create the extended observations.** This stage reuses matching raw observations, computes missing frames once per source, and records per-frame person-gate outcomes:

```bash
uv run python -m worldmodels.extended_data --output artifacts/runs/extended-input-v1
```

**Then complete and freeze normal-only forecast development.** Development scores training and validation only. The freeze checks execution, identity and completeness; an unfavorable scientific result remains eligible for the predeclared final comparison:

```bash
uv run python -m worldmodels.revision develop --input artifacts/runs/extended-input-v1 --output artifacts/runs/revision-development-v1
uv run python -m worldmodels.revision freeze --development artifacts/runs/revision-development-v1
```

**The optional-in-general weight sensitivity is a prerequisite of this recorded clean-data workflow.** Reproducing this specific run requires its declared training-gradient evidence and the already-approved one-weight amendment. Do not invent a new gradient justification or change its weight based on a final result:

```bash
uv run python -m worldmodels.objective_audit develop --reference artifacts/runs/revision-development-v1 --output artifacts/runs/objective-audit-development-v1
uv run python -m worldmodels.objective_audit freeze --development artifacts/runs/objective-audit-development-v1
```

**Next provide the reviewed person-identity evidence and create the clean derivatives.** `clean_data` requires all of the following before it starts:

- `extended-input-v1` and both completed forecast-development directories, including their freeze files and validation CSVs.
- The original sixteen-frame caches under `artifacts/poses`, matched against current CSV, source-video, pose-model and extractor identities.
- The independently reviewed contact sheet `artifacts/research_analysis/person_review.png` and its documented source/frame decisions. `clean_data` consumes its hash and the declared exclusions; it does not perform the visual review.

The added review generator reproduces the seven selected sequences and their frame selections from the actual source videos and extended raw-pose records. Its default output preserves the original reviewed image:

```bash
uv run python scripts/build_person_review.py
```

That command was executed on the real videos, and its regenerated contact sheet was visually checked. Its sidecar records code, video and pose hashes. Rendering pixels can vary across environments. The historical `person_review.png` remains preserved; it was not retrospectively attributed to this newer generator.

In a fresh reproduction with no original review artifact, generate the required path and have reviewers inspect the displayed frames before accepting the documented exclusions:

```bash
uv run python scripts/build_person_review.py --output artifacts/research_analysis/person_review.png
uv run python -m worldmodels.clean_data
```

This creates `pilot-input-clean-v1`, `extended-input-clean-v1`, the filtered validation derivatives under `revision-validation-clean-v1`, and the exclusion/reuse certificates. It verifies the long-cohort equality claims before checkpoint inheritance. The short-cache lineage contains 641 reviewed candidate sequences; 405 produce retained windows.

**Refit the short classifiers and the separate acquisition controls.** The first command also generates legacy five-class test files automatically. Its existence does not authorize inspecting those files before joint release. During this refit only status, identities and validation fields were inspected:

```bash
uv run python -m worldmodels.workflow compare --input artifacts/runs/pilot-input-clean-v1 --output artifacts/runs/pilot-comparison-clean-v1 --epochs 5 --seed 27
uv run python -m worldmodels.binary_clean develop --prepared artifacts/runs/pilot-input-clean-v1 --baseline artifacts/runs/pilot-comparison-clean-v1 --output artifacts/runs/binary-development-clean-v1
uv run python -m worldmodels.acquisition_control develop --prepared artifacts/runs/extended-input-clean-v1 --output artifacts/runs/quality-development-clean-v1
```

The binary grid remains RF depth 4 or 8 and probe regularization `C` of 0.01, 0.1, 1 or 10. Classifier seed 27 is fixed. Validation selects hyperparameters within an arm; it does not change the mixed-JEPA primary. `refit_lineage.json`, `selection.json` and `seal.json` retain this distinction.

**Release all fixed final evaluations together.** The coordinator performs this once after independent review and all overlapping decisions are sealed:

```bash
uv run python scripts/release_and_evaluate.py
```

The release writes `docs/test_release.json`, runs the clean binary and acquisition evaluations, then the original frozen forecast and objective-audit evaluators. The latter consume their parent data because the complete test subset has a checked equality certificate. A sidecar pins the exact primary-test CSVs consumed by the objective audit. Successful completion produces `docs/test_execution.json`; the initial release file alone is not completion evidence.

Only after that completion record exists should the released analysis and final notebook display be run:

```bash
uv run python scripts/analyze_research.py
WM_DATA_MODE=real WM_RELEASE_TEST_RESULTS=1 uv run python scripts/execute_notebooks.py --data-mode real --timeout 600
```

Real notebook mode defaults to the clean short input and clean comparison paths. `WM_PREPARED_DATA` and `WM_COMPARISON_RUN` are explicit overrides, so record them if changed. The runner checks the kernel interpreter and records notebook-source, executed-output and input hashes. Read `docs/notebook_execution_real.json` to distinguish a fresh full pass from stale notebook sources or an interrupted replay.

## Verification and unresolved status

The coordinator's complete current project suite passed **43 tests in 7.03 seconds**, including the new direct-binary video adapter. That adapter also passed five focused tests in 0.93 seconds and a real training-video smoke execution without refitting. The full-suite command was:

```bash
uv run pytest -c pyproject.toml tests -q
```

Earlier real-mode execution passed all eight tutorials with final-test display disabled. After release, all eight clean-cohort tutorials completed a real replay. Notebook 06 was then extended with the direct-binary video call and successfully replayed again in 14.02 seconds. **All four joint evaluations and all eight current notebooks are complete, with 88 executed code cells and current source/output hashes verified.** The latest notebook report preserves earlier unchanged notebook executions and the newer capstone execution; it does not describe them as one simultaneous kernel session.

The original `pilot-input-v1`, `pilot-input-v2`, `pilot-comparison-v2`, and earlier `binary-development-v1`, `v2` and `v3` remain quarantined development records. No historical checkpoint is relabeled as a clean refit. The original normal-only revision and objective-audit checkpoints remain valid inherited artifacts under their equality certificate.

An earlier comparison automatically computed test files, and a pre-gate notebook persisted tables. The execution audit records that authors did not inspect metric values for development decisions. This is an internal source holdout with that audit limitation, not an independently preregistered or externally validated clinical cohort. Sparse independent sources, uncertain patient overlap, unresolved pose identity, short variable-duration horizons, and dataset-label uncertainty remain limits even when every software check passes.
