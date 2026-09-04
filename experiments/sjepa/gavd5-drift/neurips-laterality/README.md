# NeurIPS laterality: inductive notebook suite

This directory contains the new held-out-source-video workflow. The top-level legacy notebooks remain historical evidence and are not imported.

## Run order

1. `00_protocol_and_governance.ipynb` — freeze the claim and check submission governance.
2. `01_cohort_and_target_audit.ipynb` — build a QC-only cohort and validate the paired-valid target.
3. `02_source_level_splits.ipynb` — create source-balanced outer and inner folds.
4. `03_fold_local_training.ipynb` — train label-blind encoders within folds and seeds.
5. `04_held_out_evaluation.ipynb` — score held-out sources and audit direct checkpoint token equivariance.
6. `05_aggregate_statistics.ipynb` — compute per-checkpoint gates, matched controls, and source bootstraps.
7. `06_external_subject_gate.ipynb` — validate—but do not execute—an optional subject-indexed external cohort contract.

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

See `RUNBOOK.md` for the staged execution plan, current verification status, real-data dry-audit command, and the exact boundaries between compute, evidence, and submission readiness.

Generated artifacts are excluded from git. Every checkpoint and result records the protocol digest, cohort digest, source IDs, fold, seed, and variant. Run `uv run python neurips-laterality/scripts/verify_suite.py` for the lightweight test and notebook-structure checks.

Artifacts contain linkable source-video identifiers and derived pose representations. Git exclusion is not a release determination: do not redistribute manifests, poses, embeddings, predictions, or checkpoints until the reviews in `governance/status.json` explicitly permit it.

The optional external gate additionally requires `LATERALITY_EXTERNAL_MANIFEST` and `LATERALITY_EXTERNAL_GOVERNANCE`. The latter must be scoped to that external dataset; the GAVD governance file is deliberately rejected.

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
