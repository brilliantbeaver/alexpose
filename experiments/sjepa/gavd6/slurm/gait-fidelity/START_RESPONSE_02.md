# Launch the core-informed JEPA follow-up

Use a new run named **`jepa-response-02`**. It keeps the original paired-change primary comparison and adds coordinate-only readouts to the same frozen encoders. The complete plan contains **9 pretraining phases, 18 readouts and 18 final models across seeds 17, 29 and 43**. The deadline is **September 25, 2026, at 8 AM Pacific** (`2026-09-25T15:00:00Z`). The total cap remains **48 H100-hours**.

Read the [scientific amendment](../../docs/studies/gait-fidelity/methods/core-to-followup-20260924.md) and [core interpretation](../../docs/studies/gait-fidelity/results/core-analysis-20260924/README.md). This is a development experiment. The old 18-phase default remains available for reproduction.

## 1. Transfer a new immutable release from the Mac

Run this in an interactive Mac terminal from the GAVD6 repository. The explicit Python path avoids the unavailable Python 3.10 selected by the ancestor `.python-version` file. SSH may prompt for HAIC authentication.

```bash
PATH="$PWD/.venv/bin:$PATH" bash slurm/gait-fidelity/sync-to-haic.sh \
  tedmui@haic.stanford.edu:/hai/scratch/tedmui/alexpose/experiments/sjepa/gavd6 --apply
```

Save the printed **new release path**. The transfer does not overwrite the parent checkout. Ignore the generic new-cohort setup command printed by the sync tool; this experiment uses the existing prepared parent instead.

## 2. Initialize and launch from an authenticated HAIC terminal

Replace `REPLACE_WITH_RELEASE_ID` with the exact release ID printed above. Run this block in Bash so that a failed prerequisite stops execution.

```bash
set -euo pipefail
export GAVD6_ROOT=/hai/scratch/tedmui/alexpose/experiments/sjepa/gavd6
gf_followup_release="$GAVD6_ROOT/releases/gait-fidelity/REPLACE_WITH_RELEASE_ID"
gf_parent_work="$GAVD6_ROOT/outputs/gait-fidelity/walking-core-01"
gf_followup_work="$GAVD6_ROOT/outputs/gait-fidelity/jepa-response-02"
test -f "$gf_followup_release/gait-fidelity-release.json"

unset PYTHONHOME PYTHONPATH
export GF_PYTHON=/hai/scratch/tedmui/envs/synthetic-training-cu124/bin/python
bash "$gf_followup_release/slurm/gait-fidelity/run.sh" setup-followup \
  "$gf_followup_work" --parent-work "$gf_parent_work" \
  --include-base-readouts --deadline-utc 2026-09-25T15:00:00Z

source "$gf_followup_work/session.env"
bash "$GF_ROOT/slurm/gait-fidelity/run.sh" preflight "$GF_WORK"
bash "$GF_ROOT/slurm/gait-fidelity/run.sh" plan "$GF_WORK"
bash "$GF_ROOT/slurm/gait-fidelity/run.sh" launch "$GF_WORK" --max-jobs 8
```

Setup checks the complete parent receipts, immutable original code, prepared arrays, source population and actual selected update counts. It imports all five core model families under both objectives, including direct/base. Existing child settings cannot be changed through setup; use the saved child session to resume an already initialized matching run. Do not edit a frozen run to bypass a failed check.

The first GPU stage calibrates coefficients using the fixed training-only rule and profiles all new phases. It either admits the full 27-phase matrix or records a resource refusal and retains permitted parent diagnostics. Admission retains the 48-GPU-hour cap, 2-hour queue allowance and complete three-seed controls. It also reserves two CPU wall hours for final evaluation. With diagnostics and recovery reserves, at least **14 hours plus the profiled fitting duration** must remain at admission. Submit early enough for that allowance; 8 AM is the results cutoff, not a permissible start time.

## 3. Monitor and verify

```bash
source /hai/scratch/tedmui/alexpose/experiments/sjepa/gavd6/outputs/gait-fidelity/jepa-response-02/session.env
bash "$GF_ROOT/slurm/gait-fidelity/run.sh" status "$GF_WORK"
bash "$GF_ROOT/slurm/gait-fidelity/run.sh" report "$GF_WORK"
```

After the complete evaluation is published, reconstruct it in a CPU allocation:

```bash
srun --account=mind --partition=hai --cpus-per-task=4 --mem=64G --time=04:00:00 \
  bash "$GF_ROOT/slurm/gait-fidelity/run.sh" verify "$GF_WORK"
```

The primary result remains in `evaluation/response-comparisons.json`. The new `evaluation/readout-control-comparisons.json` contains the base-readout comparison, comparisons with the core controls and the coupling-by-readout interaction. `evaluation/per-person.csv` includes the zero-response benchmark and separate successful-prediction and failure-penalty contributions. Read these alongside waveform, coordinate, nuisance, direction and coverage results.

No remote job was submitted during preparation: the assistant's noninteractive SSH session lacked HAIC keyboard-interactive authentication. Local tests and the CPU fixture do not verify H100 compatibility, source-runtime cost or scheduler availability.
