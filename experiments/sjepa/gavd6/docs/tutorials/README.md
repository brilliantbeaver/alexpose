# Tutorials

Start with the [reading guide](reading-guide.md) to choose a research question,
check prerequisites, and distinguish a teaching example from an experiment.
For the current objective, begin with **motion preservation during tracking
repair**.

| Learning path | What you should understand by the end |
| --- | --- |
| [Motion preservation: six staged notebooks](../../notebooks/motion_preservation/README.md) | How controlled movement events and tracking failures differ, how video informs repair, and how to compare retained movement at comparable error removal. Real pretrained-model evaluation remains pending. |
| [Future-feature prediction: question to decision](../studies/future-feature-prediction/gate/README.md) | How source grouping, teacher context, matched predictors and uncertainty determine whether skeleton history adds predictive information beyond RGB. The repaired development result is STOP. |
| [Target accessibility: evidence to a proposed student experiment](../../notebooks/target_accessibility/README.md) | Why predicting teacher features and improving a trained student are separate claims, and what the cached comparison supports. |
| [S-JEPA gait model internals](sjepa_model_internals.md) | How coordinates become tokens, how masked teacher-feature prediction works, and how gradients, teacher updates and downstream pooling fit together. |

The [notebook index](../../notebooks/README.md) lists all studies, including the
separate source-scaling experiment and historical reflection, classification and
force-prediction work. Read a study's overview before interpreting its notebook
outputs.

The model-internals guide describes the historical gait-classification
implementation. Its [figure generator](make_class_figures.py) produces the SVG
illustrations in [figures/](figures/). It provides architectural background;
the current motion-preservation protocol defines the active experiment.
