# Future Innovation: from question to decision

**Current protocol: `direct-v2`, 50 clips.** Experiment 0 now tests how much
skeleton coordinate/confidence history adds beyond RGB, nuisance inputs and a
matched head with validity flags. The gate samples 50 eligible clips from the
available full-GAVD pool; the full dataset is reserved for the real experiment.
Background quality, person/background edit sensitivity, and background-target
gain reduction are no longer prerequisites. Source holdout, timing boundaries,
training-only selection and paired controls remain required. Read the
[current protocol and decision rules](direct-gate-protocol.md).

The five-notebook structure is unchanged. New runs fit four arms × three seeds ×
five folds, producing 60 final heads. No completed real `direct-v2` prediction
result is available in this checkout. An ADVANCE recommends designing the
full-GAVD JEPA comparison; it does not establish a JEPA training benefit.

**Preserved legacy evidence:** copied run `haic-GOjuXSEB` completed a 50-clip cache
but stopped at sensitivity 1.360, below its frozen 2.0 threshold; all other six
checks passed. It never tested the prediction hypothesis. The new protocol was
adopted after that rejection and retains a separate run identity. See the
[investigation and independent review](notebook-run-investigation.md).

## Follow the five notebooks

Each notebook starts with a question, runs or inspects its part of the experiment,
and ends with an interpretation and next action. Teaching examples explain the
same operations on small generated inputs. Kernels are independent; full
execution follows 00 → 04 through the shared run artifacts.

| Notebook | Question | Existing pipeline stage |
| --- | --- | --- |
| [00 · Question and worked example](../../../notebooks/experiments/future_innovation/00_question_and_worked_example.ipynb) | What does a skeleton correction add to a video baseline? | Initialize or verify the run |
| [01 · Cohort and alignment](../../../notebooks/experiments/future_innovation/01_cohort_and_alignment.ipynb) | Are the windows aligned and sources separated? | Candidate construction and pose extraction |
| [02 · Teacher features and validity](../../../notebooks/experiments/future_innovation/02_teacher_features_and_validity.ipynb) | Are prefix inputs isolated and cached targets usable? | Feature cache and validity audits |
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

## Execute or recover the 50-clip gate

Use the [HAIC run guide](../../../slurm/future-innovation/README.md) for input
setup, then the [separate notebook launchers](../../../slurm/future-innovation/NOTEBOOKS.md):

```bash
bash slurm/future-innovation/submit-fi-notebooks.sh all
```

They force `execute` mode and call the same production CLI as the original jobs:

```text
initialize → 50-clip cohort → poses → cached teacher → readiness → five fitting tasks → report
```

The stage commands retain their caching, compatibility checks, locks and
resumption behavior. Use a new root for direct-v2; reuse that root to resume it.
Legacy roots retain their original requirements. An unfinished pose stage restarts
candidate processing; a complete frozen cohort is reused.
Small CPU heads reuse the expensive teacher features. Completed report
inspection does not require either stage to run again. Every notebook attempt
retains its outputs together in `FI_RUN_ROOT/notebook_runs/haic-<batch-id>/`;
command logs live separately under `FI_RUN_ROOT/logs/notebooks/`. Notebook
03 uses the full frozen selection grid and all 60 final heads when direct readiness
checks pass; teaching settings never enter execution mode. A verified audit
rejection ends 02–04 with an explicit blocked outcome, no fitting, and an
unsealed diagnostic STOP with incomplete predictive measurement. Other
incomplete or corrupt evidence still fails execution. Complete negative
results remain successful executions.

For interactive execution on allocated resources, set `FI_TUTORIAL_MODE=execute`
and an explicit `FI_RUN_ROOT`, then run all cells of 00–04 in order. Without a
fold override, notebook 03 runs all five folds sequentially. The original
`submit-fi-pipeline.sh` remains available for direct CLI execution.

The [direct gate specification](direct-gate-protocol.md) defines the current
comparison. The [original guide](../../../notes/future-innovation-distillation/experiment-0-guide.md)
remains historical design context. The production code and
each run's saved contracts define its actual execution; conceptual snippets
in the specification are not a second pipeline.

## Interpret completion and evidence separately

| State | Next action |
| --- | --- |
| No local report | Inspect or obtain the run artifacts; do not infer a scientific outcome |
| Incomplete STOP | Recover execution or repair invalid measurement |
| Complete valid STOP | Preserve the result and inspect which effect threshold failed |
| INCONCLUSIVE | Report the instability and define any further measurement in advance |
| ADVANCE | Design the full-GAVD JEPA comparison; adapter training remains disallowed |
| Synthetic result | Use only as a software demonstration |

The primary increment compares the real-skeleton head with the matched
no-skeleton head. It must be positive on average, in each seed and in at least
90% of paired source-bootstrap draws, alongside the retained ridge/shuffle/mismatch
criteria. The 95% interval is reported separately; containing zero leaves the
increment uncertain even if the 90% rule passes.

Source videos are the sampling units; they are not verified people. Inputs use
frames 0–31, while the target at frames 38–39 is contextualized by all 64 frames.
The comparison concerns teacher features rather than decoded future movement.
Background and recording cues remain possible explanations of parts of that target.

## Maintain one current interpretation

When real results arrive, update this overview with the run ID, report path,
supported conclusion, unresolved explanation and next action. Update the study
index at the same time. Add result-specific commentary to a retained executed
notebook, with an explicit artifact reference. Do not turn teaching values into
reported measurements or rewrite the experiment's thresholds after seeing them.

The [notebook maintenance and verification guide](../../../scripts/research_directions/future_innovation/future_innovation_tutorial_guide.md)
documents selective regeneration, fresh-kernel checks and measured runtimes.
