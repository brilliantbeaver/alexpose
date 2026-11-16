# Response to adversarial review

The initial narrative is preserved in `manuscript_initial.md`. The first empirical source draft is preserved in `revisions/01_initial_results_draft/`; that revision had no PDF because the official style's compiler dependencies were unresolved. The revised PDF incorporates the independent scientific and visual findings below. The reviewer did not author the model or manuscript generator. Final acceptance must refer to the exact delivered artifact hashes.

| Finding | Revision and verification |
| --- | --- |
| The primary forecast gain could conceal a stronger simple comparator. | The abstract and main result table explicitly show the stronger current-posture-conditioned regression. The text explains that this model predicts a different future at each horizon; it does not repeat a static output. |
| A positive shifted-context effect does not complete the bilateral-dynamics argument. | The main results retain the uncertain own-leg contrast and the unfavorable static comparison. The abstract states that the full timing-specific bilateral advantage is not established. |
| Shrinkage does not outperform the unshrunk bilateral model. | Shrinkage remains secondary. Its complete results and selected beta counts are reported without claiming a win over the primary model. |
| The observation gate incorrectly referred to equal halves. | The generator now states the 16-sample prefix and eight-sample suffix explicitly. |
| The horizon could be mistaken for an endpoint-only error. | The methods define the primary error as the average across all eight future observations, nominally 0.1 to 0.8 seconds. |
| Preparation and modeling have different window counts. | The data section explicitly records 2,818 prepared windows, three common-support exclusions, and 2,815 modeled windows. All values are generated from run metadata. |
| Attrition differs substantially by condition. | The data section reports 141/287 retained normal sequences and 168/183 myopathic sequences. It explains that duration, geometry, identity, and observation support all contribute, and that selection may affect abnormality ranking. |
| Pooled out-of-fold scores come from different fitted models. | The limitations state the score-comparability assumption and possible cross-fold offsets. Conditional source intervals do not represent full retraining uncertainty. |
| Labels in the method diagram touched box edges. | The source generator uses shorter labels while preserving the observation, context, fitting, and scoring relationships. The SVG, PDF, and PNG are regenerated together. |
| The official style requires pdfTeX. | A portable TinyTeX environment supplies pdfTeX 1.40.29, with compiler distribution identity and commands recorded. The AAAI style and bibliography files are unchanged. |
| The first appendix comparison table exceeded its column. | Short labels identify shrinkage variants, with an explicit caption definition. Final compilation must contain no overfull boxes or unresolved references. |

The positive findings remain visible: bilateral history reduces mean normal-source forecasting error by 33.5% relative to persistence, and the predeclared excess-error score has AUROC 0.793. These claims concern this exploratory source-grouped cohort. They do not establish clinical generalization, neural mechanism, or treatment-response modeling.

Author names and affiliations remain explicitly pending. The workshop uses single-blind review, so these fields must be completed before submission. No submission has been performed.
