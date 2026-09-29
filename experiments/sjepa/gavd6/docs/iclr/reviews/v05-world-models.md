# Version 05: independent world-models and methods review

Reviewed frozen `paper-v05.tex`, its full diff from v04 and the empirical laterality naming figure. This reviewer did not author or edit the manuscript. The separate numerical reviewer checks the newly printed paired assignment interval.

## Claim alignment and objections

| ID | Severity/status | Evidence | Correction or disposition | Residual limitation |
|---|---|---|---|---|
| W05-1 | Moderate provenance omission, resolved | Main now states renderer-derived person boxes are supplied to fixed estimators and cameras are fitted once across source-family geometries. This matches `preparation.py:482–508`. | Accurate disclosure; no additional correction. | Detector uncertainty and real-video acquisition remain outside study. |
| W04-2 | Minor clarity, resolved | “projector-head outputs from translated-view encoder means” now distinguishes the learned projection head from camera-projected reference joints. | Correct and clearer. | None. |
| W05-2 | Per-leg metric definition, resolved | Counterevidence now explicitly includes 180° failure costs for left/right excursion errors, matching `response_evaluation.py:53–59`. | Retain these costs if the numbers are moved to another table or summary. | A lower excursion error does not imply better waveform or anatomical assignment. |
| W05-3 | Figure/claim scope, acceptable | Empirical naming figure is readable at source size, uses distinct markers and line styles, and shows 0–100% scale. Caption identifies post hoc analysis, hierarchical aggregation, endpoint 4:1 weighting, and no chance baseline. Main explains direct versus frozen JEPA is a practical comparison. | No material correction requested. | Strong visual difference is not evidence of matched optimization or clinical side recovery. |
| W04-1 | Minor reproducibility, open | Main still does not print shared feature-response calibration or precise common auxiliary support. Both exist in supporting reproduction record. | v06 can add `lambdaJ=.1 sqrt(Gbase/max(Gdelta,GE))`, with 32 training batches and shared seeds/variants, and distinguish all-frame paired support from any-frame base support. | Only initial gradient scale is matched. |
| W05-4 | Editorial scope decision, accepted | Main probe paragraph removed to make room for empirical laterality. No teacher-probe success or representation-accessibility claim remains that depends on it. Diagnostic data/supporting record are retained. | The tradeoff is reasonable for the requested nine-page laterality focus. Do not subsequently claim that useful encoder features were learned but merely hidden by the readout. | Probe limitations still constrain any mechanistic interpretation. |

**No outstanding material method or claim misstatement was found.** All previous source-alignment corrections remain present: true physical mirrors are reprojected, the original objective is scalar plus geometry, dense loss averages pairs equally, references remain privileged, graph masks do not delete sequence positions, and deployment consumes one observed sequence. The task is not relabeled as forecasting or action-conditioned world modeling.

The original readout-weight versus dense repair finding remains stated with its numerical uncertainty even though its full plot moved out of the main paper. The main laterality results are correctly marked exploratory and do not replace the primary response test. The per-leg counterevidence also prevents an overbroad “direct is uniformly best” claim. These are useful editorial gains without new empirical support.

## Fixed rubric

| Dimension | Weight | Score /10 | Concrete reason and remaining weakness |
|---|---:|---:|---|
| Conference relevance and contribution |20%|6.5|Same diagnostic representation-learning contribution; data and optimization limits remain.|
| Claim accuracy and evidence support |20%|9.0|No material method misstatement; privileges and metric penalties clarified.|
| Evaluation and statistical rigor |15%|7.0|Additional laterality analysis is explicitly post hoc, not new independent evidence.|
| Scientific insight and positioning |15%|8.0|Side-assignment tradeoffs are explicit; no new mechanism or model-family conclusion established.|
| Reproducibility |10%|8.5|Supporting contract strong; compact main feature calibration/support still incomplete.|
| Clarity and narrative |10%|8.5|Better fit to laterality request, with a reasonable loss of probe detail from the main paper.|
| Figures |5%|8.5|Empirical naming plot improves laterality emphasis, while replacing a useful repair plot; overall quality remains comparable.|
| Submission fit |5%|8.0|Nine-page main body retained; final compiled audit remains required.|
| **Weighted total** |**100%**|**78.75 /100**|Plateau is deliberate: improved editorial focus does not remove remaining empirical weaknesses.|

The manuscript is now defensible within its stated scope. Independent confirmation, optimization sweeps, natural-video references and a more direct mechanistic control would be required to strengthen that scope; wording revisions cannot supply them.
