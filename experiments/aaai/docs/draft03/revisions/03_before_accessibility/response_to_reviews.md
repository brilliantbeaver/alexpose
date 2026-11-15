# Review responses

The review focuses on whether the paper presents a coherent contribution to gait anomaly detection and whether every material claim follows from the saved measurements.

| Review point | Resolution |
| --- | --- |
| Make the paper self-contained around a scientific problem. | The title, abstract, introduction and conclusion describe controlled cross-leg prediction. The manuscript explains its method and findings without a development-history narrative. Its source template and evidence snapshot are local to this paper. |
| Explain why completion belongs in an anomaly paper. | The paper distinguishes useful prediction, adequate observation support and abnormality ranking. The method tests each step explicitly while preserving posture and movement amplitude. |
| Name the actual comparators. | The 19.5% and 5.2% reductions compare bilateral prediction with the own-leg correction. The 7.7% reduction compares anatomical with uniform training on the same distal task. |
| Avoid choosing a favorable masking denominator. | U and E_leg remain separate fixed banks. The U-trained reference leads the context comparison, and both banks accompany the A-versus-U result. |
| Keep the independent participant result central. | The paper describes the separate laboratory population, refitted angle models, participant separation and 109/128 improvement count. It does not claim transfer of video weights. |
| Do not equate input width with effective capacity. | Methods report numerical Gram-matrix ranks and differences in feature bases. The claim concerns the fitted predictor families. |
| Disclose exploratory inference without narrating manuscript development. | Methods identify development-informed normal recordings, fixed model-fitting roles and post-outcome descriptive completion intervals. |
| Preserve the actual anomaly-ranking evidence. | Tables and text report local, relational and combined scores, paired differences and uncertainty. Completion benefit is not presented as improved detection accuracy. |
| Do not infer a neurological mechanism from a timing response. | The timing results retain every failed paired-support adequacy gate. A nonzero error response alone does not validate a coordination measurement. |
| Do not assign a cause to score-integration behavior. | The distribution audit describes score tails and bilateral covariance. Calibration and reference estimation remain possible explanations, not established causes or tested remedies. |

The [independent framing review](advisory/standalone_framing_review.md) supports this presentation and identifies the required methodological disclosures. The [completion audit](advisory/completion_audit_v1/README.md) checks the paired numerical comparisons directly. The [final independent review](advisory/standalone_manuscript_review.md) evaluates the current manuscript.

Archived advisory records retain the actual prompts and versions they reviewed. They are provenance records, not claims that another model approved the current text. Review advice is not clinical validation or a publication decision. The [final validation record](final_validation.json) identifies the manuscript version that passed the current artifact checks.

The final reviewer found no material numerical or claim issue. We adopted the optional precision change that describes the relational score by its observed AUROC. The shared-body-reference qualification remains explicit in the measurement methods; no claim of unconditional input independence is made.
