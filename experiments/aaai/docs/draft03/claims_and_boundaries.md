# Claims and evidence

The paper contributes a controlled procedure for measuring cross-leg prediction and assessing its use in gait anomaly detection. Estimating hidden movement, predicting accurately with the available observations, and ranking abnormal recordings are separate outcomes.

| Claim | Evidence and scope |
| --- | --- |
| The fitted both-leg model improves prediction of hidden laboratory angles. | Mean error falls by 5.20%, with conditional paired interval [3.55%, 6.70%], across 128 healthy participants. All four fold means improve. |
| Both-leg prediction helps when neighboring GAVD leg joints are hidden together. | The model trained with random masks lowers error by 19.5% relative to its own-leg correction. The absolute-benefit interval is [0.010721, 0.021300]. Targets are estimated pose coordinates. |
| Anatomical training helps the focused missing-observation task. | Anatomical versus random training lowers grouped-hiding error by 7.71%, with interval [6.36%, 9.00%], on identical targets. The unchanged uniform-task improvement is uncertain. |
| The available observations matter when interpreting a predictor. | The both-leg model fails every outer timing check on the exact aligned inputs, despite sufficient recording counts. A nonzero response to an edit does not establish an adequate timing measurement. |
| The cross-leg score has an observed laboratory AUROC of 0.694. | R reaches AUROC 0.694. M reaches 0.958 and the fixed combination reaches 0.892, so incremental detection value is not established. |
| The method is auditable across measurement systems. | Saved predictions retain target keys, masks, validity, source roles and reference parameters. The laboratory study refits the algorithmic comparison on participant-separated angle signals. |

Giving corrections the same number of features does not make them equally flexible. The evidence compares the fitted models; it does not prove that every difference in modeling capacity has been removed. The task is masked completion, not future forecasting. Its targets, references and shared body normalization do not identify physiological coordination or a disease cause.

The successful core comparison uses ridge coordinate and angle predictors. It is not evidence that JEPA pretraining causes the completion gain. The anatomical mask family and residual construction have established precedents; the contribution is their controlled use to evaluate relational anomaly evidence.

Completion percentages are ratios of group mean losses, not averages of individual percentage changes. The additional completion intervals were calculated after inspecting results. They keep the fitted models, data roles, masks and chosen analysis fixed, so they do not include retraining uncertainty. The 28 normal GAVD recordings informed development; upload IDs do not establish independent participants. The external study covers healthy and stroke walking, not all four condition families.

The video study is `d66e1b126c9d9a9adf1b`. The independent participant study is `external_fb9d0d23b56ce29fd846`. The [paired completion audit](advisory/completion_audit_v1/README.md) preserves inputs, producer hashes, bootstrap draws and target-identity checks. The [evidence record](evidence.json) connects these measurements to the manuscript.
