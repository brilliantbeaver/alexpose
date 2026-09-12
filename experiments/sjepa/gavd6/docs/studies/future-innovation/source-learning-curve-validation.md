# Implemented source learning curve: audit, validation and execution status

The source-scaling study is implemented, with a separate protocol, source
reservation, CPU fitting/reporting path, expanded-data preparation, Slurm jobs,
and read-only inspection notebook. The local cohort audit and synthetic checks
are real executed work. **The expanded real-data learning curve has not run.**
The full-source media and annotation checkout are unavailable locally, and HAIC
rejected non-interactive authentication. This is incomplete execution, not a
scientific STOP and not evidence that larger training sets fail.

## What was reserved before expanded fitting

The [new run](../../../outputs/future-innovation-source-curve-dev-20260911-v2/)
preserves the [protocol](source-learning-curve-protocol.md), copied manifests,
historical-exposure records, participant policy, calibration and parent hashes.

| Inventory | Count | Meaning |
|---|---:|---|
| Full annotated sequences | 1,874 | One record per annotated sequence; not all verified eligible |
| Full source recordings | 348 | Recording IDs, not participants |
| Parent alignment candidates | 1,662 sequences / 332 recordings | Existing candidate evidence from the 50-clip experiment |
| Necessary annotation-length conditions | 1,728 sequences | At least 64 annotated frames and sufficient span; continuity and pose still need checks |
| Previously verified eligible cohort | 50 clips / 43 recordings | The completed cached gate |
| Conservatively marked exposed recordings | 128 | Union of FI and supplied historical GAVD/gait-parity inventories |
| Reserved confirmation | 44 recordings / 178 annotated sequences | No expanded processing, fitting, donors or evaluation in this study |
| Development | 304 recordings / 1,696 annotated sequences | Counts before expanded availability/pose eligibility |
| Explicitly identified participants | 0 known IDs; total unknown | No participant registry was supplied |

The source reservation is relative to the supplied exposure manifests. The
historical 642-sequence/94-recording GAVD inventory is used conservatively; the
retained laterality cohort's full raw identity archive was not newly recovered.
These records do not prove that every unmarked recording has never appeared in
another analysis. Confirmation requires a final exposure reconciliation and, if
participant-level claims are intended, an explicit participant registry. No
sequence/video identifier is treated as a person identifier.

The [machine-readable audit](../../../outputs/future-innovation-source-curve-dev-20260911-v2/reports/cohort-audit.json)
and [source reservation](../../../outputs/future-innovation-source-curve-dev-20260911-v2/config/source-reservation.csv)
retain exact identities and counts. The initial eligibility table distinguishes
parent-verified clips, parent pose failures, and unprocessed sequences. Necessary
annotation-length conditions are reported separately from actual eligibility.
Old exclusions caused by the 50-clip selection cap are not pose failures.

## How the experiment isolates the effect of training size

The experiment keeps the repaired joint ridge model, safe input preprocessing,
ordered temporal features, four controls, source balancing, teacher projection
and contextual target. It changes the amount of development training data and
makes the penalty comparable across that change. It also explicitly removes the
old two-clip source cap: every eligible sequence in a selected recording enters
the fit, with equal total weight per recording. Therefore each result reports
both recordings and clips; 40 recordings is not the original 50-clip experiment.

Five fixed outer folds determine the held-out recordings. For each fold, three
independently ordered source lists define nested prefixes near 40, 80 and 160
training sources. All clips from a selected source stay together. If known
participant links connect recordings, those connected groups also stay together.
The full-training endpoint is fitted once and referenced by all three lists.
Repeated source subsets measure sensitivity to data composition, not optimizer
randomness. The final subset plan is frozen only after expanded eligibility and
teacher-audit checks pass; its nominal sizes are not inferred from the old cache.

The original solver minimizes a sum of weighted errors. With weights summing to
n training clips, the new wrapper uses `lambda = n * rho`, with rho selected from
`[0.0025, 0.025, 0.25, 2.5, 25, 250]` separately for RGB and skeleton blocks.
This matches the old numeric grid at the declared 40-window anchor. Every inner
fit uses its own n. The exact baseline candidate, common RGB reference, finite
joint grid and tie/failure policy remain in place. The historical implementation is byte-preserved and retains fixed summed-penalty behavior.

Scores pool predictions on the same evaluation sources at every size. Reports
include RGB-only and all arm scores, per-subset matched increments, paired
source-bootstrap intervals, and raw teacher-unit squared errors. Raw error helps
interpret the curve because the declared predictive R² denominator uses the
applicable training mean and can change as training size changes. Masks are
intersected across all required fits; held-out observations never select them.

## Implementation and critical review

| Component | Implementation and checks |
|---|---|
| Audit/reservation | `fi_scaling_cohort.py`; explicit exposure registry, participant-connected groups, reserved confirmation role and deterministic fold/subset identities |
| Expanded input stages | `fi_scaling_data.py`; existing candidate builder, decoder, pose processing, token pooling, nuisance construction and readiness arithmetic |
| Fitting | `future_innovation_scaling/fi_scaling_nested.py` adds normalized penalties and source-group orchestration while importing the original model, controls and selection rules |
| Saved models and reports | `fi_scaling_training.py`; unique fit identities, all candidate records, selected coefficients, preprocessing/donors, predictions, paired uncertainty and an SVG curve |
| CLI | `run_source_learning_curve.py` and the isolated `future_innovation_scaling.fi_scaling_cli` module; old gate initialization still requires 50 clips |
| Slurm | Jobs 20–23 under `slurm/future-innovation-scaling/` and `submit-source-learning-curve.sh`; separate CPU/GPU stages, five-fold fit array, dependencies and explicit roots |
| Notebook | [23_source_learning_curves.ipynb](../../../23_source_learning_curves.ipynb), generated from `build_source_learning_curve_notebook.py`; inspection does not fit or encode |

Review identified and repaired two additional integrity gaps before freezing the
real study. First, a valid hash for a per-window receipt does not by itself check
the decoded frames and poses named in that receipt. Encoding now checks those
underlying artifacts. Second, relocating a teacher path must not authorize a
different teacher. The data contract compares scientific teacher identity with
the frozen parent, independently of runtime path strings. Focused tests reproduce
both altered-artifact cases and require rejection.

The numerical verifier reconstructs selected coefficients and training scalers,
inner group partitions, donor assignments and saved predictions. It recomputes
candidate pooling/selection, displayed outer training losses and final scores.
It does not refit every rejected candidate coefficient matrix. Rehashed wrong
prediction units are rejected. Typed baseline-only artifacts retain their exact
baseline prediction after reload. Failed required candidates prevent a complete
scientific report. Parent snapshots include digests, sizes, modification times
and file inventories; reused parent arrays retain their original bindings.

A broader historical-bridge check exposed a compatibility issue in the first
implementation: its sealed contract fingerprints the entire original FI code,
CLI and Slurm directories. Even adding a new command invalidated that identity.
The final implementation therefore lives in `future_innovation_scaling/` and
`slurm/future-innovation-scaling/`, with a dedicated script entry point. The
historical files were restored byte-for-byte. Their combined fingerprint again
equals `0729a925cde4c8441b615630d7dc3df4bf543d3be6d35f2628e4ec98ef4e82bf`.
All 22 bridge checks, including reconstruction of the sealed real pilot and its
tamper tests, pass. No verification rule was relaxed.

The first scaling root without the `-v2` suffix is retained as an implementation
attempt and blocked input audit. The final `-v2` root freezes the isolated code
before any expanded real-data fitting. Both reservations use the same source
identities and scientific policy. This amendment changes code organization and
compatibility, not the model, target, source sampling or decision thresholds.

## Calibration and actual commands

The fixed source fixtures use 80 synthetic recordings with two clips each,
five outer folds, three source-subset repetitions, the full 36-pair joint grid
plus exact baseline, all four arms, and 2,000 paired source-bootstrap draws. With
80 sources, the valid training sizes are 40 and all 64 outer-training sources;
80/160 are explicitly unavailable. Four fixtures yield 80 distinct nested fit
jobs. These are software checks, not real-data sample-size results.

| Fixed fixture | At 40 training sources | At all 64 training sources |
|---|---:|---:|
| Mean matched increment, three RGB-only draws | −0.0001866 R² | −0.0001022 R² |
| Planted temporal matched increment | +0.742955 R² | +0.749573 R² |
| Planted real-minus-shuffle | +0.751649 R² | +0.750505 R² |

The RGB-only seeds are 261401–261403; the temporal seed is 261404. The null mean
must be at most +0.03, and the two planted comparisons must exceed +0.10 at both
sizes. These thresholds were declared before calibration. Individual null draws
can have small positive estimates. The [final calibration record](../../../work/artifacts/source-learning-curve-20260911/calibration-final-v3/calibration.json)
retains full precision, source-subset plans, selected model objects and candidate
outcomes. Earlier calibration attempts are retained; the final attempt matches
the frozen implementation after review changes.

The focused tests exercise reservation, participant grouping, nested prefixes,
identical-endpoint reuse, finite-sample penalty normalization, held-out scaler
independence, exact scoring with source multiplicity, selected-model reload,
rehashed wrong predictions, duplicate-writer locks, nested input receipts and
teacher relocation. Synthetic and focused tests never authorize scientific
advancement.

The focused suite includes 13 new scaling tests, including exact equality with
the historical nested fit when the optional scaling behavior is disabled.
The final FI suite ran 138 tests: 136 passed and two skipped. The separate
22-test bridge suite passed, giving **158 passed and two skipped** across the
two suites. Notebook 23 executed all five code cells without errors; source and
executed cells match. Slurm syntax checks and the final submission dry run passed.
The skipped existing checks require the external
official teacher checkout (`FI_TEST_VJEPA_ROOT`); no supplied checkout was
available. Syntax/import checks and `git diff --check` passed. Logs are retained
under [the validation artifacts](../../../work/artifacts/source-learning-curve-20260911/).

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_future_innovation_scaling.py' -v
.venv/bin/python -m unittest discover -s tests -p 'test_future_innovation_*.py' -v
.venv/bin/python scripts/research_directions/future_innovation/calibrate_source_learning_curve.py \
  --output-root work/artifacts/source-learning-curve-20260911/calibration-final-v3
.venv/bin/python scripts/research_directions/future_innovation/build_source_learning_curve_notebook.py
git diff --check
```

The [execution guide](../../../slurm/future-innovation/SOURCE_LEARNING_CURVE.md)
contains the complete local freeze command and HAIC submission/resumption
commands. Completed real-data stages do not exist yet; completed fitting reuse
was tested on an injected synthetic source-held fit, not an expanded real cohort.
The actual new pose/teacher path has not been exercised on HAIC. Existing tests
cover the reused decoder, controls, causal teacher interface and old resumption
contracts; injected checks cannot establish the availability or runtime of new
media processing.

The final root retains [65 implementation files and their digests](../../../outputs/future-innovation-source-curve-dev-20260911-v2/implementation/manifest.json).
The [final execution record](../../../work/artifacts/source-learning-curve-20260911/execution-record-final.json)
links the exercised commands and logs. The [preservation check](../../../work/artifacts/source-learning-curve-20260911/parent-preservation-final.json)
confirms the parent artifact snapshot and the identical source reservation across
the two implementation attempts. Historical real-run reconstruction passed
without modifying their reports or relaxing their software identity checks.

## Concrete blocker and the next executable step

The documented remote probe was:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=10 tedmui@haic.stanford.edu 'hostname'
```

Outside the sandbox it returned exit code 255 and
`Permission denied (keyboard-interactive)`. The shorter `haic` alias did not
provide a usable connection either. No host-key checks were disabled and no
Slurm job was submitted. The local checkout contains the full manifests and
the 50-window teacher cache, but not the full-source video/annotation checkout
or the teacher checkpoint required to encode additional clips.

The actual local `prepare` command also returned exit code 1 at the first missing
annotation file, `data/gavd_full/annotations/GAVD/data/GAVD_Clinical_Annotations_1.csv`.
Its full command and traceback are retained in the execution record. The frozen
audit contains 50 parent-verified sequences, two historical pose failures and
1,822 sequences without completed pose processing. None of those 1,822 has been
reported as newly verified eligible.

After authenticated HAIC access is available, transfer the exact frozen study
and matching parent snapshot, supply the recorded model/annotation resources,
and execute preparation followed by cache/audit/plan. Only then can the CPU jobs
measure the curve. A growing matched increment could support a frozen
confirmation experiment. Better overall prediction with a flat increment would
favor a separate target or representation study. The current work establishes
neither outcome on the expanded real cohort.
