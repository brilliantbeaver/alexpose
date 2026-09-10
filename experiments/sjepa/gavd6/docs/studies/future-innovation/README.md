# Future Innovation: from question to decision

**Status checked 10 September 2026:** Experiment 0 is implemented. This local
checkout contains alignment overlays under `outputs/future-innovation/gate-v1`,
but no completed gate report in that run. Preparation artifacts do not establish
a scientific result. This status does not describe live HAIC jobs.

The question is whether skeleton history helps a small predictor after it has
already seen the recent video and recording details. We first test raw skeletons
against controls. S-JEPA training and adapter distillation are later questions.

## Follow the five notebooks

Each notebook starts with a question, runs or inspects its part of the experiment,
and ends with an interpretation and next action. Teaching examples explain the
same operations on small generated inputs. Kernels are independent; full
execution follows 00 → 04 through the shared run artifacts.

| Notebook | Question | Existing pipeline stage |
| --- | --- | --- |
| [00 · Question and worked example](../../../notebooks/experiments/future_innovation/00_question_and_worked_example.ipynb) | What does a skeleton correction add to a video baseline? | Initialize or verify the run |
| [01 · Cohort and alignment](../../../notebooks/experiments/future_innovation/01_cohort_and_alignment.ipynb) | Are the windows aligned and sources separated? | Candidate construction and pose extraction |
| [02 · Teacher features and validity](../../../notebooks/experiments/future_innovation/02_teacher_features_and_validity.ipynb) | Are the inputs past-only and the targets meaningful? | Feature cache and validity audits |
| [03 · Predictors and controls](../../../notebooks/experiments/future_innovation/03_matched_predictors_and_controls.ipynb) | Does correctly paired motion help under a fair comparison? | Nested fitting and Slurm arrays |
| [04 · Results and next decision](../../../notebooks/experiments/future_innovation/04_results_and_next_decision.ipynb) | What does the complete evidence justify next? | Existing gate decision and sealed report |

The default `teach` mode uses small generated examples, including only two CPU
updates for one residual head. Those examples have no empirical interpretation
about gait. Figures are generated inline from the displayed values and tensor
contracts; no separate diagram-generation workflow is required.

## Inspect an existing run

Set the environment before starting Jupyter, or edit the setup cell:

```bash
export GAVD6_ROOT=/path/to/gavd6
export FI_TUTORIAL_MODE=inspect
export FI_RUN_ROOT=/path/to/existing/future-innovation/run
```

A relative `FI_RUN_ROOT` resolves from `GAVD6_ROOT`. `inspect` reads manifests,
configuration, overlays, audit summaries and reports. It never downloads a
teacher, trains a model, submits a job, or rewrites a result. Missing local
files are reported as unavailable; the notebooks do not infer remote job state.

Notebook 04 verifies the existing report seal using the production verifier.
The seal covers the decision and narrative. This checks report integrity;
it does not rerun pixel audits, raw-data validation or out-of-fold scoring.
Copy the entire `reports/` directory for sealed report inspection. Cohort and
alignment inspection additionally needs the corresponding manifests and QC
files; large teacher caches and checkpoints are unnecessary for reading reports.

## Execute or recover the real experiment

Use the [HAIC run guide](../../../slurm/future-innovation/README.md) for input
setup, then the [separate notebook launchers](../../../slurm/future-innovation/NOTEBOOKS.md):

```bash
bash slurm/future-innovation/submit-fi-notebooks.sh all
```

They force `execute` mode and call the same production CLI as the original jobs:

```text
initialize → cohort → poses → cached teacher → validity audits → five fitting tasks → report
```

The stage commands retain their caching, compatibility checks, locks and
resumption behavior. Reuse the existing run root to recover interrupted work.
Small CPU heads reuse the expensive teacher features. Completed report
inspection does not require either stage to run again. Every notebook attempt
retains its outputs and command logs under `FI_RUN_ROOT/notebook_runs/`. Notebook
03 uses the full frozen selection grid and all 75 final heads; teaching settings
never enter execution mode. Notebook 04 fails for an incomplete measurement,
while a complete negative result remains a successful execution.

For interactive execution on allocated resources, set `FI_TUTORIAL_MODE=execute`
and an explicit `FI_RUN_ROOT`, then run all cells of 00–04 in order. Without a
fold override, notebook 03 runs all five folds sequentially. The original
`submit-fi-pipeline.sh` remains available for direct CLI execution.

The [experiment specification](../../../notes/future-innovation-distillation/experiment-0-guide.md)
contains the full design and implementation sketches. The production code and
each run's saved contracts define its actual execution; conceptual snippets
in the specification are not a second pipeline.

## Interpret completion and evidence separately

| State | Next action |
| --- | --- |
| No local report | Inspect or obtain the run artifacts; do not infer a scientific outcome |
| Incomplete STOP | Recover execution or repair invalid measurement |
| Complete valid STOP | Preserve the result and inspect which effect threshold failed |
| INCONCLUSIVE | Report the instability and define any further measurement in advance |
| ADVANCE | Follow the next measurement stage in the proposal; adapter training remains disallowed |
| Synthetic result | Use only as a software demonstration |

Read validity checks before treating a STOP as evidence against the hypothesis.
The no-skeleton comparison is a required reported diagnostic without an extra
numerical cutoff in the implemented gate. If that control reproduces the gain,
state the unresolved attribution even if the automatic gate advances.

Source videos are the independent sampling units; they are not verified people.
The target is a full-clip contextual feature at frames 38–39, while inputs use
frames 0–31. Background targets can retain person information through attention.
These limits remain applicable when every numerical check passes.

## Maintain one current interpretation

When real results arrive, update this overview with the run ID, report path,
supported conclusion, unresolved explanation and next action. Update the study
index at the same time. Add result-specific commentary to a retained executed
notebook, with an explicit artifact reference. Do not turn teaching values into
reported measurements or rewrite the experiment's thresholds after seeing them.

The [notebook maintenance and verification guide](../../../scripts/research_directions/future_innovation/future_innovation_tutorial_guide.md)
documents selective regeneration, fresh-kernel checks and measured runtimes.
