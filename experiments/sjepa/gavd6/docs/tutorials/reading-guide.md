# Tutorial reading guide

Begin with the question a study answers. The active study asks whether a repair
method can remove tracking failures while preserving movement supported by the
video. It uses the complete declared clip for offline restoration. The
[study overview](../studies/motion-preservation/README.md) states the current
evidence and the [protocol](../studies/motion-preservation/protocol/README.md)
defines the comparison. Real pretrained-model evaluation is still pending.

## Choose a path

| Your question | Reading path | What to look for |
| --- | --- | --- |
| How do we preserve movement during repair? | [Motion-preservation notebooks 00–05](../../notebooks/motion_preservation/README.md) | Movement retention and achieved tracking-error removal, with strengths selected on separate calibration people. |
| How does this project's S-JEPA work? | [Model internals](sjepa_model_internals.md), then the [historical classification notebooks](../../notebooks/README.md#gait-representation-and-classification) | Tensor shapes, visible and masked tokens, gradient ownership and teacher updates. |
| Does skeleton history add to an RGB predictor? | [Prediction gate](../studies/future-feature-prediction/gate/README.md), then notebooks 00–04 in [the index](../../notebooks/README.md#future-feature-prediction) | The real-skeleton increment relative to the matched no-skeleton reference, with source-grouped evaluation. |
| Does more source diversity change that conclusion? | [Source-scaling overview](../studies/future-feature-prediction/scaling/README.md), then [notebook 23](../../notebooks/source_scaling/23_source_learning_curves.ipynb) | Recording counts, nested source subsets and the frozen development decision. |
| Which teacher targets can a skeleton student use? | [Accessibility overview](../studies/future-feature-prediction/accessibility/README.md), then [notebooks 19–22](../../notebooks/target_accessibility/README.md) | The declared reference, accessible predictive information and the separate, still-proposed student-training test. |

These are separate studies. Their notebook numbers do not define one continuous
training pipeline, and their scores should not be pooled. Future-feature targets
are contextual video-teacher features: selected target frames were encoded with
the complete teacher clip. Predicting them is not the same task as predicting
future joint positions.

## Prerequisites

To read the tutorials, know what a training example, held-out evaluation and
baseline are. Basic Python and NumPy help with executable examples; PyTorch
tensors and automatic differentiation help with the model-internals guide.
You can follow the study question and result interpretation without running a
GPU job.

Before executing, use the study's setup guide and confirm three things:

1. **Environment:** the project Python environment and the dependencies required
   by that study are available. Start the notebook in a fresh kernel and run
   cells from top to bottom.
2. **Inputs:** the configured data, manifests and model assets exist. A path in a
   configuration is not proof that a checkpoint loaded or a dataset was read.
3. **Run identity:** choose the intended mode and run directory before starting
   the kernel. Keep executed notebooks and their outputs with that run; retain
   the original configuration and source reservations.

For the active study, the [motion-preservation launch
guide](../../slurm/motion-preservation/README.md) specifies data paths, body-model
assets, pretrained motion and flow checkpoints, and runtime configuration.
Notebook 00 inventories availability; notebook 02 actually loads the configured
models. The optional GAVD stress stage also requires the earlier source-reservation
CSV. Recover that record before using the reserved sources.

## Work through the current experiment in three parts

| Part | Notebooks | Expected output and the next decision |
| --- | --- | --- |
| Establish the problem | 00 · Data and question; 01 · Controlled pairs | An inventory of people, sources and model settings, followed by paired examples of real events and tracking failures. Inspect event timing, event-plus-error cases and examples with identical observed skeletons. |
| Establish the evidence and comparison | 02 · Prior, flow and baselines; 03 · Train and calibrate | Cached repairs and video-motion evidence with backend provenance, then fitted models and calibration-selected strengths. First check whether the real frozen prior erases supported events and whether simple flow rules already solve the problem. |
| Evaluate and inspect failures | 04 · Preservation and repair; 05 · GAVD visual stress | Retention versus achieved error-removal results and a development decision, followed by optional real-video galleries and ambiguity inspection. GAVD trajectories are not 3D ground truth. |

Use the [six-notebook guide](../../notebooks/motion_preservation/README.md) for
individual files and commands. Training, calibration, development and final
people remain separate. The final event family opens only after the development
decision; it is not a routine extra notebook to run while tuning. The pilot
launcher submits stages 00–04 without automatically opening that final test or
submitting GAVD stress inspection.

A useful result retains more supported movement at comparable achieved error
removal. A smoother trajectory by itself cannot establish that result. Inspect
the strongest calibrated simple baselines, ambiguous examples and uncertainty
before attributing a gain to a learned model. A successful small temporal gate
would not, by itself, validate pretrained S-JEPA or V-JEPA features.

## Know what an execution proves

| Tutorial or mode | What it does | What its outputs support |
| --- | --- | --- |
| Motion preservation, explicit `demo` mode | Uses generated motion and stand-in estimators. | Understanding the pipeline and checking software behavior. |
| Motion preservation, default real mode | Uses configured real data and pretrained models. | An experiment only after the required assets load, stages complete and the protocol's evaluation conditions are met. |
| Prediction gate, default `teach` mode | Runs small generated examples. | Understanding the estimator and decision logic. |
| Prediction gate, `inspect` mode | Reads saved artifacts from the selected run. | Inspection of recorded evidence; it does not refit models or rerun the real stages. |
| Prediction gate, `execute` mode | Runs the configured real pipeline stages with an explicit run root. | Stage completion and applicable checks, followed by the study's numerical evaluation. |
| Accessibility notebooks 19–22 | Read retained evidence and run labeled synthetic calculations; notebook 21 also defaults to CPU reconstruction of selected models. | Checks of the available cached comparison. Notebook 21's optional file-integrity-only mode does not verify numerical predictions. None of these notebooks trains a new student. |

The gate notebooks use `FI_TUTORIAL_MODE` and `FI_RUN_ROOT`; the motion study
uses its own configuration and run-root settings. Follow each study's guide
instead of assuming the mode names are interchangeable. Missing historical
artifacts can prevent inspection even when a source notebook opens correctly.
A saved output records an earlier execution; it does not demonstrate a successful
run of today's source.

## Read equations alongside the experiment

For each expression, identify its inputs, target, reference and evaluation group
before interpreting the score. The same notation can describe different targets
in different studies. In particular, an R² difference is meaningful relative to
its stated reference and cohort, and a confidence interval for a development
comparison does not turn those sources into an untouched final test.

The [model-internals notation table](sjepa_model_internals.md#2-notation-and-standard-configuration)
explains its tensor dimensions. A coordinate, a learned feature and a clinical
measurement have different meanings even if all three are stored as arrays.
Consult the local definition when moving between tutorials.

## Maintain a tutorial

For generated notebooks, edit the [owning builder](../../scripts/research_directions/research_notebook_builder_guide.md)
and regenerate the source notebook. Keep executed copies under an identified run
root. Historical classification notebooks and the AMASS visualization are direct
sources and retain their historical outputs. The [repository layout](../repository/layout.md)
explains canonical paths and the retained source snapshots needed for historical
replay.
