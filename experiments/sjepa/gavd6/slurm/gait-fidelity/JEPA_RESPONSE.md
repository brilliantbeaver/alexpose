# Run the JEPA response follow-up on HAIC

**24 September amendment:** for the core-informed 27-phase matrix and the new
September 25, 8 AM Pacific cutoff, use [START_RESPONSE_02.md](START_RESPONSE_02.md).
The instructions below retain the original 18-phase protocol and deadline.

This is a separate experiment named `jepa-response-01`. It reuses the completed
`walking-core-01` data and baseline predictions, then trains nine new models.
There is no new rendering, pose extraction, GAVD processing or confirmation-set
evaluation. Read the [scientific protocol](../../docs/studies/gait-fidelity/methods/jepa-response.md)
for the hypotheses and controls, and [notebook F](../../notebooks/gait_fidelity/experiments/F_jepa_response.ipynb)
for the calculations.

The core run must finish successfully before follow-up setup. Keep its checkout,
configuration and outputs unchanged. In particular, **do not pull these code
changes into the checkout used by an active `walking-core-01` run**.

## 1. Copy a separate code release

In your **Mac terminal**, from the local GAVD6 checkout:

```bash
bash slurm/gait-fidelity/sync-to-haic.sh \
  tedmui@haic.stanford.edu:/hai/scratch/tedmui/alexpose/experiments/sjepa/gavd6 --apply
```

Save the new release path printed by the command. The transfer creates a separate
directory under `releases/gait-fidelity`; it does not overwrite the core checkout.
The general setup command printed at the end is for a new source cohort. For
this follow-up, use step 2 instead.

## 2. Initialize the follow-up after the core completes

In your **HAIC terminal**, set the release path to the exact path printed above.
Your existing `GAVD6_ROOT` remains the asset checkout. Replace the release
placeholder before running the block:

```bash
gf_followup_release="$GAVD6_ROOT/releases/gait-fidelity/REPLACE_WITH_RELEASE_ID"
gf_parent_work="$GAVD6_ROOT/outputs/gait-fidelity/walking-core-01"
gf_followup_work="$GAVD6_ROOT/outputs/gait-fidelity/jepa-response-01"

unset PYTHONHOME PYTHONPATH
export GF_PYTHON="/hai/scratch/$USER/envs/synthetic-training-cu124/bin/python"
bash "$gf_followup_release/slurm/gait-fidelity/run.sh" setup-followup \
  "$gf_followup_work" --parent-work "$gf_parent_work"
```

Setup verifies the completed parent and records the prepared data, original
receipts, source identity and actual profiled update schedule. It refuses an
incomplete or changed parent. It references existing arrays read-only and does
not copy or regenerate the cohort. The parent admission amendment, including
any excluded singleton training person, remains part of that provenance.

If `jepa-response-01` already exists, resume its saved session using step 3.
Do not delete it, repeat setup over it, or edit its frozen configuration.

## 3. Check and launch the saved child run

```bash
source "$GAVD6_ROOT/outputs/gait-fidelity/jepa-response-01/session.env"
bash "$GF_ROOT/slurm/gait-fidelity/run.sh" preflight "$GF_WORK" &&
bash "$GF_ROOT/slurm/gait-fidelity/run.sh" plan "$GF_WORK" &&
bash "$GF_ROOT/slurm/gait-fidelity/run.sh" launch "$GF_WORK"
```

Check that the plan contains three representation variants, seeds 17, 29 and 43,
and **18 optimization phases: nine pretraining phases and nine readouts**.
Calibration/profiling and representation diagnostics are additional worker
stages. The child trains with the same update counts selected by the parent,
even if the parent used the registered half schedule.

The first GPU stage calibrates auxiliary-loss coefficients on training data and
measures runtime. It admits the complete matrix only when the measured schedule
fits the budget and results cutoff. A refusal is a recorded resource decision;
it must not be bypassed by dropping a control, reducing seeds or editing a saved
coefficient.

| Reserved work | H100-hours |
| --- | ---: |
| Eighteen fits | 36 |
| Calibration and profiling | 2 |
| Representation diagnostics | 4 |
| Recovery allowance | 6 |
| **Maximum additional allocation** | **48** |

The cutoff is **September 24, 2026, at 6 PM Pacific**, equivalent to
`2026-09-25T01:00:00Z`. Admission includes a two-hour queue allowance. Failed
allocations consume the same budget. Unfinished child work stops at the cutoff;
the parent is never stopped or modified. GPU-hours measure total allocation
across workers, not elapsed wall-clock time. Queue availability remains external
to the program.

## 4. Monitor, read and verify

In every new HAIC shell, first source the child session:

```bash
source "$GAVD6_ROOT/outputs/gait-fidelity/jepa-response-01/session.env"
bash "$GF_ROOT/slurm/gait-fidelity/run.sh" status "$GF_WORK"
```

The status and coordinator log distinguish admission refusal, incomplete work,
failed attempts and completed evaluation. A successful job submission is not a
completed result. After completion:

```bash
bash "$GF_ROOT/slurm/gait-fidelity/run.sh" report "$GF_WORK"
srun --account=mind --partition=hai --cpus-per-task=4 --mem=64G --time=04:00:00 \
  bash "$GF_ROOT/slurm/gait-fidelity/run.sh" verify "$GF_WORK"
```

Read the primary paired comparison, coverage and position/nuisance errors
together. Notebook F opens saved tables and response plots without submitting
jobs. Start its kernel from a shell that has sourced the **child** session.
Without a child session, the notebook executes small mathematical examples and
labels them as software illustrations.

The local fixture tests verify software behavior. The HAIC profile is still
required to validate the actual CUDA environment, memory use and throughput.
