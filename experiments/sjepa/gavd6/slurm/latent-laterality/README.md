# Latent-laterality HAIC jobs

**Current decision:** seed-7 AMASS v2 validation completed and the uniform
control reproduced SG-JEPA's gain. Confirmation stopped. Do not submit jobs
15/16 under the current plan; keep the test split sealed. See the
[completed findings](../../docs/studies/latent-laterality/study-so-far.md).

The numbered files cover three workflows; they are **not** one submission chain.
See the [HAIC run guide](../../docs/studies/latent-laterality/haic-run-guide.md)
for the environment contract, submission commands, and scientific decisions.

| Workflow | Jobs | Order and decision points |
| --- | --- | --- |
| Paired AMASS v2 | `11`–`16` | `11`–`14` completed; validation stopped `15`/`16`. |
| Historical AMASS v1 diagnostic | `02`–`03` | `02` → `03`. The recorded v1 gate failed; it does not authorize v2 training. |
| Deferred GAVD/source-route screen | `04`–`10` | `04` → `05` → `06`; select a single-source route before `07` → `08`; inspect the gate before `09`, then review seed 7 before `10`. |

Job `01` creates the shared neutral AMASS tensors. Run it only when those
inputs need building. Reuse completed manifests and gate outputs; use a fresh
run root for a new experiment.

For v2, job `13` trains correction-first, SG-JEPA, and the uniform control at
seed 7. Job `14` evaluates validation only. Submit `15` only if validation
supports SG-JEPA over correction-first without an even-channel penalty and
the uniform control does not reproduce the gain. Job `15` repeats the two-arm
contrast at seeds 19 and 31; `16` reads the sealed test split for all three seeds.
Use `afterok` dependencies on the entire preceding array for `13` → `14` and
`15` → `16`. Do not automate the scientific decision between `14` and `15`.

All jobs require exported `GAVD6_ROOT` and `AMASS_RUN_ROOT`. Study artifacts
default to `$AMASS_RUN_ROOT/latent-laterality`; override
`LATENT_LATERALITY_RUN_ROOT` to select another run root. Individual jobs declare
their additional inputs near the top. Use absolute paths as in the run guide;
relative data/output paths resolve from the gavd6 checkout.

Submit from `$GAVD6_ROOT` after preparing its `uv` environment. Jobs use
`uv run --no-sync` and write `slurm-ll-*.out` / `slurm-ll-*.err` in the submission
directory. Array logs include both the job and task IDs. Resource requests,
arm mappings, seeds, and CLI arguments remain explicit in each batch file.

`ll-common.sh` provides shared setup, input/output checks, and array-index
validation. Each job sources it through `GAVD6_ROOT`, so it also works when
Slurm executes a spool copy. Deploy the whole directory together. Jobs reject
existing output manifests, non-empty output directories, and symlink outputs.
Job `01` instead relies on the converter's compatibility checks to resume valid
existing tensors. A benchmark exit status of `2` is a scientific stop: inspect
its diagnostics and gate decision before submitting training.
