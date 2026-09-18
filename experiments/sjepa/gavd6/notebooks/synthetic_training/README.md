# Choose synthetic lessons from a student's training response

These tutorials implement [synthetic training selection](../../notes/research-agenda/proposals/synthetic-training-selection.md). A pose estimator receives a common short training update. A selector uses the resulting changes on unlabeled video to choose its next labeled AMASS lesson. Independent reference coordinates measure whether that decision improves real accuracy.

The source sequence is **00 → 01 → 02 → 03 → 07**. Notebook 02 runs independently for each source student. Real preparation and deployment follow in 04 and 05. Notebook 08 exchanges the students' already selected lessons. Notebook 06 explicitly opens either early or confirmation human references. The source launcher does not run real evaluation automatically.

| Notebook | Action | Read the result as |
| --- | --- | --- |
| [00 · Question and assets](00_question_and_assets.ipynb) | Inspect manifests, configured student roles, and actual file availability | A readiness inventory, not successful model loading |
| [01 · Prepare source data](01_prepare_source_data.ipynb) | Render separated AMASS probe/lesson/diagnostic/context/reference data and prepare COCO replay | A labeled source task whose appearance and conventions still need inspection |
| [02 · Measure source trials](02_measure_source_trials.ipynb) | Probe one source estimator and fork all lesson/replay branches at equal budgets | Measured source utilities and the opportunity for selection |
| [03 · Fit and freeze selectors](03_fit_and_freeze_selectors.ipynb) | Fit on training students and choose settings on excluded validation students | A source-selected teaching rule and fair comparison set |
| [07 · Source mechanism report](07_source_mechanism_report.ipynb) | Compare selected utility, simple explanations, and student-specific outcomes | The continuation decision before real reference outcomes |
| [04 · Prepare real evaluation](04_prepare_real_evaluation.ipynb) | Check views/crops, select recording-disjoint groups, and export human annotation templates | Independent context and measurement data, with labels still to annotate |
| [05 · Choose and adapt](05_choose_and_adapt.ipynb) | Choose lessons from unlabeled GAVD context and save student predictions | Frozen deployment decisions, including the held architecture |
| [08 · Exchange selected lessons](08_exchange_selected_lessons.ipynb) | Apply the students' selected lessons to one another at equal budgets | Whether each student benefits more from its own choice |
| [06 · Measure real accuracy](06_measure_real_accuracy.ipynb) | Read the explicitly selected human reference split and compare saved predictions | Real gain, uncertainty, and the supported claim level |

Use the [HAIC setup guide](../../slurm/synthetic-training/README.md) for packages, public checkpoints, shared motion-preservation variables, new asset requirements, commands, and output paths. Its pilot JSON uses four source students and one held architecture. It defines small trial budgets, not a validated training recipe.

Run each notebook in a fresh kernel from top to bottom. Its setup cell discovers the checkout, imports the study's Python module, and reads `RunConfig.from_env()`. Substantive implementation lives in [synthetic_training](../../src/gavd6_sjepa/research_directions/synthetic_training), keeping the notebooks focused on the experiment and its interpretation.

## Start with the smallest useful decision

First establish that the available rendering assets and trainable models work. Inspect the rendered RGB task and landmark convention before collecting source trials. A colorized mesh, random encoder, model-generated reference pose, or repeated checkpoint with a different name cannot stand in for the required experiment.

The renderer uses one fixed camera per clip and fits the full body trajectory inside the image. When motion needs more room, camera distances increase by the same factor across resolution settings for that motion and viewpoint. This preserves their distance ratios, but a person can appear smaller than the requested height fraction. Each clip's `scene.json` records the framing adjustment and the range of projected person heights; inspect these alongside the images when assessing the resolution conditions.

Then ask whether lesson gains vary and whether a selector exploits that variation on excluded source students. Extra training may help all students equally; that is not personalized teaching. A source utility table also does not guarantee sim-to-real transfer.

The final claim requires three distinct comparisons:

1. Full-budget replay tests whether the complete procedure is useful.
2. Snapshot and source-progress controls test the information supplied by target prediction change.
3. The held architecture tests transfer beyond the students used for source fitting and selection.

A nearest-neighbor selector can establish the information effect. A JEPA-specific interpretation additionally requires video-feature comparisons while keeping the response information fixed. The default protocol estimates visible 2D landmarks, not disease, hidden-joint truth, forces, or clinical 3D biomechanics.

The crossover is an additional check of personalization. It cannot by itself identify target response as the cause; current weakness or model-family information might already explain student-specific choices. Keep the matched source-progress comparison as the primary mechanism test.

## Editing tutorials and preserving results

[build_notebooks.py](../../scripts/research_directions/synthetic_training/build_notebooks.py) owns the cell sources. Regenerate the checked-in output-free notebooks after editing it:

```bash
.venv/bin/python scripts/research_directions/synthetic_training/build_notebooks.py
```

The executor saves every executed attempt under `ST_RUN_ROOT/notebook_runs`, including partial outputs when a cell fails. Array students and real evaluation splits have separate directories. Logs and notebook metadata record where each attempt ran. New scientific conditions should use a new run root rather than mixing their artifacts with an older trial.

The [validation record](VALIDATION.md) separates completed local interface checks from GPU and scientific experiments still to run.
