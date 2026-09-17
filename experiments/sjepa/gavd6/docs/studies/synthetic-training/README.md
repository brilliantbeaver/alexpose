# Synthetic training selection

This study tests whether an estimator's response to a short common training probe helps choose useful synthetic training lessons for real deployment. The distinctive question is whether **prediction changes on unlabeled target video** add useful information beyond the estimator's current predictions, labeled synthetic weaknesses, and source learning progress.

The [proposal](../../../notes/research-agenda/proposals/synthetic-training-selection.md) explains the hypothesis and its limits. The [notebook guide](../../../notebooks/synthetic_training/README.md) is the step-by-step experiment entry point. The [HAIC guide](../../../slurm/synthetic-training/README.md) contains package/model setup, configuration, data requirements, and Slurm commands.

## Experiment sequence

1. Prepare full-body AMASS lessons with compatible textured renders and twelve visible 2D landmark labels. Keep probe, lesson, diagnostic, context, and reference motions separate. Add a fixed labeled COCO replay subset.
2. For each source student, measure predictions and diagnostic errors before and after the same probe. Fork every lesson and replay comparison from the same post-probe checkpoint, with equal remaining budgets.
3. Fit selectors to gains measured against independent synthetic references. Select settings using excluded source students and data, then freeze the real experiment.
4. Select GAVD recording groups by checked view/crop metadata. Use unlabeled context to choose lessons. Save predictions on independent reference images, including a held architecture that never entered source fitting.
5. Exchange the students' preselected lessons at equal budgets, retaining agreed choices as well as differences. Then explicitly evaluate against human visible-landmark annotations, first on early recordings and later on untouched confirmation recordings.

The source launcher runs 00, 01, the per-student 02 array, 03, and 07. It never opens GAVD reference outcomes. The real stages 04, 05, 08, and 06 are separate submissions.

The real experiment is target-landmark-free adaptation with supplied person crops and checked viewing metadata. It does not claim fully automatic person detection or an entirely unannotated video-selection process. Independent reference boxes are drawn separately for scoring.

## What each positive result would establish

| Finding | Supported interpretation |
| --- | --- |
| A lesson improves on equal-budget replay | Synthetic adaptation can help this estimator and setting |
| Lesson choice improves on fixed/random/balanced training | Selection matters beyond ordinary augmentation |
| The full teacher beats matched controls without target prediction change | Observing the target response supplies useful teaching information |
| That effect appears on the excluded architecture and real references | The source teaching rule transfers under the evaluated conditions |
| Frozen JEPA features improve matched response selectors | Those video features add practical value; encoder comparisons alone do not isolate the JEPA objective |

The primary real metric averages visible-landmark distance normalized by independent reference body-box scale, first within frames and then within recordings. The same human visibility mask is used for every method. Missing predictions are penalized rather than omitted. Recording-level uncertainty does not by itself establish transfer across many lesson-selection settings.

## Implementation and remaining empirical work

The implementation is in [src/gavd6_sjepa/research_directions/synthetic_training](../../../src/gavd6_sjepa/research_directions/synthetic_training). It provides the experiment's source-data roles, trainable released pose-model adapters, feature/response comparisons, source selector fitting, explicit real deployment, and independent-reference evaluation. The [notebook builder](../../../scripts/research_directions/synthetic_training/build_notebooks.py) and [Slurm directory](../../../slurm/synthetic-training) expose these steps as research workflows.

The source contains no synthetic replacement for a missing released pose model, pretrained encoder, textured rendering asset, or human real reference. A successful local code check is not evidence that adaptation improves GAVD accuracy. Real HAIC checkpoint loading, EGL rendering, throughput measurement, annotation completion, and all scientific results remain experiments to run with the actual assets.

The sample configuration has only two teacher-training students and two source-validation students. It supports initial feasibility testing. A significant transfer claim requires enough independent learning behaviors and real settings to exclude simple model-family, optimization, context, and ordinary-augmentation explanations. A clean null result at an early stage should narrow the claim rather than trigger an automatic increase in training scale.

The [local workflow validation record](../../../notebooks/synthetic_training/VALIDATION.md) documents the completed notebook and launcher checks separately from the unexecuted HAIC experiments.
