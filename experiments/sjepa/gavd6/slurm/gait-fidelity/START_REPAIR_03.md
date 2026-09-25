# Readout repair and protected AMASS confirmation

This is a new experiment in a new work directory and code release. It reuses the
completed `walking-core-01` and `jepa-response-02` artifacts without modifying them.
The frozen experiment cutoff remains **September 25, 2026, 8:00 AM Pacific**
(`2026-09-25T15:00:00Z`). The paper submission deadline is separate.

The experiment trains **12 new frozen-encoder readouts**, with no new pretraining:
two representations (`jepa_delta_v1`, `jepa_endpoint_v1`), two objectives, and seeds
17, 29, 43. `scalar_low` reduces only the scalar response coefficient to 0.1 of its
inherited value. `dense_change` uses the temporal response waveform and matches
its auxiliary gradient RMS to `scalar_low` at the same initial readout, using only
training batches. Both keep the inherited coordinate loss and geometry coefficient.
Initialization, training sampling, updates, and optimizer settings remain matched.

Both new arms, the four original response arms, and direct/base are evaluated:
**nine methods × three seeds = 27 fits**. The primary comparison is delta/dense
versus delta/scalar-low on waveform error, on the fixed ViTPose extractor family.
Response error and its interval are an explicit tradeoff. No method is selected
from development scores for confirmation. This tests a repair beyond a coefficient
reduction; it does not uniquely identify sparse percentile gradients as the cause.

## 1. Copy a new release from the local checkout

```bash
PATH="$PWD/.venv/bin:$PATH" bash slurm/gait-fidelity/sync-to-haic.sh \
  tedmui@haic.stanford.edu:/hai/scratch/tedmui/alexpose/experiments/sjepa/gavd6 --apply
```

Use the **new release path** printed by this command. The generic `run.sh setup`
suggestion it prints belongs to the original study; use the repair command below.
Keep both old releases unchanged. Pulling code into an old frozen checkout can
invalidate its recorded code identity.

## 2. Initialize and inspect on HAIC

```bash
export GF_ASSETS=/hai/scratch/tedmui/alexpose/experiments/sjepa/gavd6
export GF_RELEASE="$GF_ASSETS/releases/gait-fidelity/REPLACE_WITH_NEW_RELEASE_ID"
export GF_REPAIR="$GF_ASSETS/outputs/gait-fidelity/readout-repair-03"
export GF_PYTHON=/hai/scratch/tedmui/envs/synthetic-training-cu124/bin/python

bash "$GF_RELEASE/slurm/gait-fidelity/repair.sh" setup "$GF_REPAIR" \
  --response-work "$GF_ASSETS/outputs/gait-fidelity/jepa-response-02" \
  --deadline-utc 2026-09-25T15:00:00Z --max-jobs 4 --gpu-hours 48
bash "$GF_RELEASE/slurm/gait-fidelity/repair.sh" preflight "$GF_REPAIR"
bash "$GF_RELEASE/slurm/gait-fidelity/repair.sh" exposure-audit "$GF_REPAIR"
bash "$GF_RELEASE/slurm/gait-fidelity/repair.sh" audit-expansion "$GF_REPAIR"
```

Setup hashes completed dependencies and binds actual checkpoints, predictions,
data, selected update counts, and cohort authority files. It can take several
minutes on a large filesystem. It rejects incomplete or altered source studies.
There is no need to know the exposure-ledger path to start development: the audit
checks paths saved by the parent, including its existing reservation CSV. It
reports missing evidence and writes `exposure-review-required.csv` with `unknown`
statuses. **That worksheet is not reviewed evidence.**

## 3. Run development readouts and the preparation benchmark

```bash
bash "$GF_RELEASE/slurm/gait-fidelity/repair.sh" launch "$GF_REPAIR" --stage development
```

This submits a CPU coordinator, then six small training-only calibrations and
twelve GPU readouts. It evaluates the complete matrix using development exports.
Readout training logs are under `attempts/fit-*/ATTEMPT/result/training.jsonl`.
Readout durations must be measured on HAIC; fixture times are not GPU estimates.
The configured allocations are 30 minutes per calibration and 60 minutes per fit,
with four concurrent workers and a shared 48 GPU-hour ceiling. These are caps,
not predictions of execution or queue duration.

When development is complete, run:

```bash
bash "$GF_RELEASE/slurm/gait-fidelity/repair.sh" launch "$GF_REPAIR" --stage benchmark
```

The benchmark renders and runs the two pose estimators on two deterministic,
already-open development windows using the exact slim condition panel. It does
not read protected people's trajectories. Its measured fixed setup and slower
window cost determine confirmation admission with a 1.5 safety multiplier, a
one-hour queue allowance, and a two-hour evaluation allowance. A two-hour GPU
allocation bounds the benchmark. Missing or failed windows do not yield admission.

Only one coordinator for this repair work directory may be active. Requesting a
different stage while one is pending/running reports that it was not queued;
repeat the desired launch after the active stage finishes. Repeating a completed
stage verifies and returns its artifacts. Ambiguous `sbatch` responses preserve
the reservation; do not delete the ledger to force a duplicate submission.

## 4. Review exposure evidence and lock confirmation

The source plan retains all **14 named-walking candidate people**, with two
metadata-selected distinct motions per person. The 28-window panel has three
intervention levels (0°, 5°, 15°), two physical orientations, two cameras, two
observation conditions and two extractors. At 128 frames it requires 86,016
rendered frames and 172,032 extraction frame-passes before reference QC. Naming
variants reuse the extracted data. None of these repetitions increases the
independent participant count.

Inspect `exposure-audit.json`, `cohort/plan.json`, and the actual historical
reservation/training/evaluation records. A reviewer must establish that each
planned original-test person was unused in fitting, tuning and inspected outcomes.
If history is missing, retain `unknown`: development can still be reported as
exploratory, but unused-person confirmation cannot be claimed or launched.

The reviewed CSV must contain exactly the planned people and these columns:
`person_id,canonical_person_id,original_split,reserved,exposure`. Source admission
requires preserved `original_split=test`, `reserved=false`, and documented
`exposure=unexposed_verified`. Do not replace unknown values merely to pass the
gate. Record the reviewer and specific evidence used:

```bash
bash "$GF_RELEASE/slurm/gait-fidelity/repair.sh" lock-confirmation "$GF_REPAIR" \
  --exposure-ledger /absolute/path/to/reviewed-repair-confirmation.csv \
  --reviewed-by 'Reviewer name' \
  --evidence 'Specific historical manifests, run inventories and reservation review'
```

This binds the fixed 27 completed method/seed checkpoints, calibration-bearing
repair signatures, source identities, exposure evidence, statistical settings,
and code before protected preparation. It rejects incorrectly labelled methods,
partial fit inventories, unknown exposure, changed data identities, and later
checkpoint replacement. There is no automatic winner selection or test retuning.

## 5. Run admitted confirmation

```bash
bash "$GF_RELEASE/slurm/gait-fidelity/repair.sh" launch "$GF_REPAIR" --stage confirmation
```

Admission must fit the measured remaining time and total allocation cap. Up to
four preparation shards render the fixed windows; one GPU allocation then exports
all 27 fitted models' predictions and evaluates them. Source reference QC can
remove invalid windows, but it cannot replace them with a more favorable motion.
Coverage reports planned, retained and missing people/windows. Partial coverage
is labelled explicitly and does not receive the full-protocol confirmation flag.

Every worker is independently supervised against the absolute cutoff, including
rendering and inference. Queued jobs are cancelled by the coordinator at the
cutoff; only this run's recorded job IDs are eligible. A late-starting worker
launches no workload. Stopped attempts remain visible and charged until Slurm
accounting resolves them. No deadline extension or reduced-update fallback is
applied automatically.

## Status, progress and reports

```bash
bash "$GF_RELEASE/slurm/gait-fidelity/repair.sh" status "$GF_REPAIR"
squeue -u "$USER"
find "$GF_REPAIR/attempts" -name training.jsonl -print
```

Use `tail -f` on the exact printed training log or `slurm-JOBID.out` path. Status
preserves Slurm's `PENDING` state instead of describing a queued allocation as
model training. Coordinator receipt `state=submitted` records submission history;
the live Slurm state is authoritative for whether the controller is still running.

Reports and machine-readable tables:

```text
development/evaluation/report.md
development/evaluation/summary.json
development/evaluation/per-person.csv
development/evaluation/comparisons.json
confirmation/evaluation/report.md
confirmation/evaluation/summary.json
confirmation/evaluation/per-person.csv
confirmation/evaluation/comparisons.json
```

```bash
cat "$GF_REPAIR/development/evaluation/report.md"
bash "$GF_RELEASE/slurm/gait-fidelity/repair.sh" verify "$GF_REPAIR"
```

Primary intervals use paired person means after averaging the three fixed seeds.
Per-seed estimates and a descriptive crossed bootstrap show seed sensitivity.
Development and confirmation remain separate. Power sensitivity uses the actual
development contrast and reports its assumptions; it does not guarantee a
significant confirmation. Fourteen fresh people can support a large-effect test,
but generally cannot establish a small JEPA advantage with high power. A response
interval crossing zero does not establish noninferiority, and statistical
significance is not a clinical meaningfulness claim.

`audit-expansion` records CMU/GAVD readiness without automatically expanding the
protocol. CMU folders are not certified independent people; GAVD video labels
are not dense reference kinematics. Unknown identities, exposure, timing or
missing extraction assets leave those extensions unadmitted. This avoids adding
unverifiable participant counts or mismatched real-video metrics to the paper.

## Local end-to-end software fixture

```bash
bash slurm/gait-fidelity/repair.sh fixture /tmp/gait-fidelity-repair-fixture
```

This trains real tiny core/response/repair models and exercises calibration,
benchmarking, locking, sharded preparation, inference, evaluation and verification.
Its generated confirmation identities are distinct from its training identities.
All outputs explicitly deny human-subject, GPU throughput, power or clinical
evidence. Keep fixture scores out of the paper's scientific results.

Implementation verification: 217 tests passed, including the complete retained
fixture. See the [review and validation record](../../docs/studies/gait-fidelity/reviews/repair-implementation-20260924.md)
and [manuscript-ready methods text](../../docs/studies/gait-fidelity/manuscript/repair-experiment-methods.md).
