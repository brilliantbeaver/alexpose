# Workflow and independent source review

This review covers the full-manifest revision, including the notebook/source/Slurm interfaces. It does not certify a real HAIC allocation or the current availability of remote assets.

## Execution and tutorial changes checked

- Preparation shard count is saved independently of the eight-worker concurrency cap. AMASS preparation, GAVD extraction and confirmation extraction use the same coordinator lock and allocation ledger.
- The core protocol contains ten recipes, 30 final fits and nine shared pretraining phases. The full 34-recipe protocol remains an explicit option. Tutorial result tables follow the selected protocol and identify omitted groups.
- The seven main notebooks and five experiment notebooks retain visible mathematical calculations. New cells explain full-manifest identity/duration selection, GAVD recording groups, weighted calibration, raw-motion/person aggregation, and a training-only linear gait-label probe.
- Source prediction loading and numerical verification accept memory-mapped exports. Verification reconstructs complete source-family support while reading metric tables in chunks; it retains duplicate and missing-row checks.
- The evaluation completion receipt hashes spatial/filter predictions, calibration, metric tables and the report. Regression checks reject a modified affine export or calibration file.
- Slurm submission inherits the saved account/partition. A new stage requested while a coordinator is active is explicitly reported as not queued. Confirmation stages check declarations before requesting GPUs.

Seventeen focused scheduler/tutorial tests passed, including shell syntax, relocated Slurm scripts, duplicate/ambiguous-submission handling, resource inheritance, core-plan controls, arbitrary shard counts and artifact tampering. Full notebook execution is recorded separately by the parent validation workflow; historical 94-cell receipts do not certify this revision.

## Independent adversarial findings and resolutions

The workflow reviewer read the separately authored AMASS cohort, preparation, training, evaluation and confirmation code. Three material scale/protocol failures were identified and returned to their owners:

1. **Frozen reviewed windows could shift during preparation.** The legacy interval extender could move a later reviewed window outside its approved interval and change its source-family identity. The planner now handles shared boundaries within the reviewed interval, and full-manifest preparation consumes the saved starts directly. A regression forbids calling the legacy extender in this path.
2. **The full source matrix still exported predictions eagerly.** The memory-mapped export condition originally depended on the hierarchical sampler, leaving the full matrix on the large compressed-array path. Source exports now use disk-backed arrays regardless of the selected sampler.
3. **The runtime projection multiplied fixed overhead by the update ratio.** Full-cohort hashing and prediction export were being treated as per-update work. Profiling now separates fixed phase overhead from measured optimization time before projecting the selected schedule.

The review also found that baseline outputs lacked a retained evaluation receipt despite a verifier comment implying artifact coverage. The added receipt and tamper regressions close that gap. Original source-code/fit receipts remain intact.

## Remaining evidence boundaries

AMASS filename labels identify candidates; technical geometry screening does not establish clinical gait labels. Source sampling coverage is reported because an update-limited fit can use the complete sampling pool without visiting every derived row. Confirmation requires a separately reviewed exposure history and frozen methods before test motion access.

GAVD groups related recordings and exact video copies, but an absent cross-video person mapping cannot establish person-level independence. Decoder-reported presentation timestamps are checked against the nominal frame grid; they are not independently verified capture timestamps. Gait-label probe results measure accessible label information and need their acquisition-confound controls, extraction exclusions and recording-group uncertainty. They do not establish pose accuracy or clinical validity.

HAIC body-model rendering, MMPose operators, actual resource use and complete source-data throughput still require allocated preparation/profiling runs. No GPU or scientific-result claim is inferred from local fixtures.
