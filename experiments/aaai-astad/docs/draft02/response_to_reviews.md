# Responses to independent reviews

The reviews assess the same frozen results. No model, endpoint, mask, score, source role or exclusion was changed to improve the reported outcome. The initial manuscript is preserved in `revisions/01_before_reviews`. The reviewed prompt, raw response, model identity and input hash are retained for the actual Claude call. Independent GPT reviews additionally inspected repository code and saved artifacts.

## Confirmed corrections

| Criticism | Evidence checked | Resolution |
| --- | --- | --- |
| The detection bootstrap count was incorrectly written as 5,000. | GAVD paired intervals, external paired effects and timing summaries each record 1,000 draws. The separate normal-benefit analysis uses 5,000. | Corrected the manuscript to 1,000 for detection. Reported interval endpoints remain unchanged. |
| The nuisance row was labeled as observation quality alone. | `nuisance_distance` combines availability features and annotation-view fractions. It differs from `quality16`. | Renamed it availability and viewpoint distance in the table and generated documentation. |
| The boundary audit described an undecodable sequence. | One requested frame failed; 306 of 307 requested frames in that sequence decoded. | Corrected the unit to one frame. Counts and fits remain unchanged. |
| The M diagram blurred fitting and reserved normal roles. | M fits on fitting normals; R uses separate reserved reference normals. | Changed the M box to “fitting normals.” |
| The timing figure did not identify its plotted policy. | Its plotted values are the U-trained predictor on 27 outer-test normals. | Added that population and policy to the caption. All-policy gate failures remain separate. |
| The wording implied identical training target selections across masks. | Policies share definitions, schedules and nominal budgets, but selected tokens and valid supervision differ. | Corrected the method paragraph and specified uniform selection of A's permitted starts. |
| A favorable base-error fusion was omitted from the concise manuscript table. | Its saved focal AUROC is 0.700. | Added the row without claiming a supported improvement over M. |
| Matching width could be misread as matching effective capacity. | GAVD residual ranks include 111 versus 141; external ranks are 44 versus 62 at the same width. | Reworded the abstract and introduction. The observed completion gain is explicitly conditional on the fitted families. |
| A positive timing interval might be mistaken for coordination evidence. | The minus-four interval excludes zero, but the predictor fails adequacy on the aligned paired inputs. | Explicitly explained why that interval does not pass the interpretation gate. |

## Additional checks and limitations

The direct interpolation concern was tested after outcomes, using frozen targets and permitted inputs. It did not alter fitting or selection. In the independent laboratory data, interpolation has mean participant MSE 309.838, versus 206.262 for f, 201.038 for f+h and 190.584 for f+g. Every fold has the same ordering. The derivative audit saves all comparisons and provenance. This resolves a missing comparator; it does not remove the unequal-rank limitation.

The calibration concern is valid. A separate descriptive audit records every fold rather than selecting one convenient example. External normal R 99th percentiles exceed M's in all folds; normal left/right benefit correlations range from 0.54 to 0.82, while R uses a diagonal distance. These observations support a specific limitation of the fixed equal-weight recipe. They do not establish why the AUROC difference occurs. The manuscript reports them without fitting a better fusion after observing outcomes.

The independent GPT reviewer reconstructed 204 saved weighted within-fold AUROCs with maximum numerical discrepancy 1.11e-16. It also checked population counts, interpolation results, feature ranks and figure-input hashes. The raw-prediction verifier separately reconstructs all core mask-target errors, source aggregation and metrics. Model-review opinions do not substitute for these executions.

## Criticisms or suggestions not adopted as stated

Claude suggested claiming that sham and constituent-error controls do not degrade M, and attributing the loss specifically to R's content. We did not adopt that causal interpretation. Their point estimates are close to M, but a point estimate alone does not prove equivalence. Different scales, tails and calibration can also affect fusion. We display the controls and retain the narrower fixed-score conclusion.

Claude suggested that prior exploration could not plausibly create the negative result. We did not adopt a directional immunity claim. Adaptive redesign can affect both baselines and contrasts. The GAVD analysis remains exploratory, including its negative result.

Claude stated that the external interpolation comparison was absent, although the supplied manuscript already contained its four MSEs. The exact input prompt confirms their presence. The independent repository reviewer verified those values. We retained the comparison and treated this criticism as a review oversight, not a new missing execution.

One suggestion would remove the observed completion gain entirely because effective ranks differ. We instead distinguish two claims. The fitted bilateral family improves held-out completion relative to the declared own-leg family; this is an observed result. The gain does not isolate an intrinsic information quantity, physiological coupling, or an advantage over every equally flexible own-leg model. Both statements are necessary.

Rank-matched feature families, a normal-only calibrated fusion, repeated external trials and a new independent normal cohort would strengthen a future study. They require a new scientific identity and cannot be retroactively presented as the frozen primary analysis. The user's requested positive interpretation is developed separately in [draft03](../draft03/README.md), using the same results and explicitly labeled derivative analyses.

## Review sources

- [Independent external and scoring audit](reviews/gpt_external_and_score_audit.md).
- [Independent manuscript and numerical audit](reviews/gpt_manuscript_review.md).
- [Actual Claude manuscript critique](reviews/claude_manuscript_review.md) and [call provenance](reviews/claude_manuscript_provenance.json).
- [Calibration audit](reviews/calibration_audit/README.md).

Remaining scientific limits are unequal effective capacity, few GAVD normal references, adaptive redesign, camera/view/pose confounding, failed timing-support adequacy, conditional uncertainty, a bounded pose-baseline adapter and no demonstrated detection advantage. They are not concealed by a successful PDF build.
