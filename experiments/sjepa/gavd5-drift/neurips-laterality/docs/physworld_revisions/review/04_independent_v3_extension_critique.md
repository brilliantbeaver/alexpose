# Independent adversarial extension review of paper v3

Reviewed 9 September 2026. This is a fresh review of `paper_v3.md` against the completed motion/region grid and the implementations. The separate v1 review remains preserved.

The central quantitative claims now match the retained evidence. The main table includes the direct-pose control and MAE; Appendix A correctly labels the new trained-versus-initial source bootstrap as exploratory and conditional on fitted models. The description of teacher-distribution cross-entropy and the gait-pooled VICReg term is technically consistent with the implementation. There is no remaining contradiction between the real training objective and the MSE predictor diagnostic.

The following corrections and additions would materially strengthen the next version.

| Priority | Location | Adversarial concern | Concrete revision |
|---|---|---|---|
| Necessary | §2 array table, final row | “Common mask count” can be read as requiring all clips in a batch to have the same number of targets. Connected-region draws can have different valid counts across clips. | Say “each clip's target count is matched across paired arms; counts may vary between clips.” State that target losses are averaged within each clip, then across clips. |
| Necessary | §5 bootstrap paragraph | The unqualified “2,000-resample intervals” follows historical reflection estimates with separate inference procedures. | Scope this count to the latest mask intervals and the new Appendix A reanalysis. Retain historical results under their original provenance. |
| Important | §3 recipe | The paragraph specifies the historical 0.6 mask parameter but gives no current 0.5 parameter or numerical precision. Readers cannot reconstruct which recipe produced the central table. | Add a compact latest-recipe statement: 1,200 updates, batch 20, current gait-derived mask fraction 0.5, constant EMA momentum 0.999, CUDA BF16 training with FP32 weights/loss reductions/evaluation. The existing architecture/AdamW values can be referenced rather than repeated. |
| Important | §4.3 learned readout | The much larger motion summary and high regularization boundary frequency are described only partly. | State mean-summary dimension 960 and mean-motion dimension 2,890 including ten support fractions. Link this to 74–75 training videos and the planned common alpha-grid expansion, without implying that expansion will necessarily fix the deficit. |
| Important | §4.3 intervention | “MAMP-style” does not define which part of the published method is tested. | Identify the official-code motion sampler convention, validity adaptation, and unchanged teacher-feature target. Make clear that the full MAMP motion-target method is not reproduced. |
| Important | §4.3 structured result | The structure intervention is less fully specified than its claim warrants. | Give the latest connected region as six graph-connected landmarks over eight of sixteen blocks, with about 9.9% valid input hidden and a separate count-matched random reference. Report visible graph-neighbor fraction 0.988→0.512 alongside temporal brackets 0.699→0. |
| Useful | §4.4 predictor diagnostic | “Useful clip correspondence” sounds semantically stronger than the measured contrast. | Prefer “sensitivity to correct clip–target pairing under this diagnostic.” Give 375/375 positive trained fold/seed/mask rows, noting their dependence; initial controls are positive in 33/75 rows per experiment and repeated across experiments. |
| Useful | Reproducibility paragraph | Appendix A links the script but not its retained machine-readable result. | Link `review/extension_recomputed_summary.json` so a reviewer can inspect exact values and provenance without running the script. |

## Bias and control interpretation

The initial baseline is now properly described as an initial S-JEPA encoder within a pipeline that already contains a pretrained pose detector, anatomical identities and a supervised ridge fit. This is an important correction: it prevents a reader from interpreting the 0.223 R² result as a system with no prior training or supervision anywhere.

The paired contrast estimates the effect of this pretraining recipe under these readout choices. A negative value is informative even when the readout is imperfect, but it is not a lower bound on the amount of motion information in the trained features. The alpha-boundary observations are therefore a limitation of interpreting representation quality, rather than a reason to remove the unfavorable result. The current wording mostly preserves that distinction.

The newer experiments use all the same development recordings as the earlier studies. Source-disjoint outer evaluation blocks a direct fitting path from test recordings into a particular fitted model, while repeated inspection can still influence which questions, features and comparisons are attempted. Appendix A labels its new analysis correctly. The next version should keep the already inspected outer cohort described as development evidence and reserve independent confirmation for another cohort or reserved setting.

The mask audits are successful manipulation checks. They show that selection moves toward higher displacement and that connected regions remove measured local context. They do not establish that those quantities capture clinically useful motion or that the predictor relies on interpolation. Equal target counts control one aspect of supervision; geometry and content-dependent sampling can still change information content. The existing text that refers to different “observable clues” is appropriately limited.

The original/prepared-target agreement diagnostic has R² near the initial-encoder score, but those numbers should not be treated as a paired comparison. The diagnostic has 623 clips/92 sources, compares two formulas, and fits no held-out predictor; the learned table has 625/93. Its current explanation that it is neither a model score nor an information ceiling is essential and should remain.

## Cross-entropy and regularizer details

The latest loss forms teacher probabilities as `softmax((target − center)/0.06)` and predicted probabilities through logits divided by 0.10. Teacher probabilities are detached. Hidden-target cross-entropy is reduced by mean targets within clip, then mean clips when counts vary. The full-input auxiliary pathway pools valid tokens from the twelve gait landmarks, projects that summary, and applies `25×view-MSE + 25×variance-penalty + covariance-penalty`, with the resulting VICReg scalar multiplied by 0.05. These details support the present high-level description.

The teacher update uses constant momentum 0.999 in this grid; the target center is updated with momentum 0.9. Separate online and EMA-teacher readouts both underperform their initial controls, so the central discrepancy is not explained by evaluating only the EMA model. One need not include every constant in the main text, but a precise supplementary recipe should preserve them.

Because the auxiliary pathway uses unmasked training views, direct hidden-content isolation applies to the masked predictor pathway. It does not mean the online encoder is forbidden from ever observing those training coordinates through another self-supervised objective. The manuscript's present full-input regularizer description handles this correctly. A pipeline diagram should preserve that distinction visually.

## Selection and emphasis

The manuscript is now strongest when it places the completed latest grid at the center and treats the earlier reflection results as the reason for the investigation. The mean-motion initial/trained contrast has directly inspectable predictions, strong controls and newly quantified conditional uncertainty. Its relationship with the positive predictor diagnostic gives the paper a substantive finding beyond a list of nonsignificant masks.

The health motivation should remain specific without implying diagnostic validation. A signed coordinate-derived displacement score tests whether side information survives the representation pathway; different neurological and muscular conditions can affect gait through different mechanisms. Condition labels are neither affected-side labels nor ground truth for this score. Explicit references can motivate why asymmetry matters, while the present experiment addresses measurement access.

The main remaining concern is methodological compression rather than unsupported results. A reviewer should be able to identify the exact source population, target, encoder recipe, training labels, matched baseline and uncertainty scope for each central claim without inferring them from the notebook sequence. Addressing the items above would close that gap while keeping the abstract free of procedural numbers.
