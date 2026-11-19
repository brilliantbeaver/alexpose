# What the independent reviews changed

The first real-results draft is preserved in [revisions/01_initial](revisions/01_initial/). The current [manuscript](manuscript.md) is the revised version. Neither the primary schedule, fitted models, test cohort nor score direction was changed during this revision.

Two actual Claude calls reviewed the protocol and manuscript. The latter completed successfully in 127.17 seconds through the Claude CLI, which reported `global.anthropic.claude-opus-4-8`. Its prompt, raw response and invocation metadata are saved in [reviews](reviews/). GPT-6 Astra independently reviewed the protocol and the completed manuscript. A separate implementation audit compared code, checkpoint tensors, reference parameters and artifact roles. These are model-assisted reviews, not peer review or clinical review.

Raw reviews are preserved verbatim. They contain mistaken interpretations as well as useful advice. The Claude response also includes an irrelevant local session-hook remark; it is not part of the manuscript or its scientific evidence. The decisions below were based on verification rather than reviewer agreement.

## Revisions adopted

| Review point | Evidence checked | Revision |
| --- | --- | --- |
| The common-coordinate random control is more informative than only the changed-teacher latent control. | Primary learned/random coordinate AUROCs are 0.685736/0.700992; longer values are 0.689550/0.693364 in the saved ranking files. | Added both random-coordinate values to the main table and abstract, with the caveat that separately fitted heads still need not optimize equally. |
| The narrative leads too easily with a favorable raw-error comparison. | Calibrated full-error AUROC is 0.672769; the profile is 0.641495. Their paired interval includes zero. | Rewrote the abstract and operational takeaway around an unestablished learned/relational advantage. Kept the originally declared raw-error contrast as primary. |
| The model must not be said to falsify the biological hypothesis. | Full-context coordinate MSE exceeds own-context MSE in all four primary folds and three longer folds. The memories are not nested representations. | Replaced “falsification controls” with “diagnostic controls” and explicitly separated estimator inadequacy from absence of useful biological relationships. |
| Dataset selection differs by category. | Normal sources fall from 28 to 23 at the profile gate; exercise from 92 to 59; inebriated from 6 to 2. | Added a source-attrition table and restricted the population statement to recordings that pass the declared observation rule. |
| Related work needs the closest masked-gait and dependency precedents. | Directly verified the primary GenGait and Beyond Marginals records. | Added both references and narrowed novelty to the controlled audit. No numerical reproduction is implied. |
| SeeKer's confidence term was described incorrectly. | Its primary abstract weights keypoint conditional scores using keypoint-detector confidence. | Corrected “prediction confidence” to “keypoint-detector confidence.” |
| Fold roles and reference reuse need to be explicit. | The [reference audit](normal_reference_audit.json) finds 15 distinct normals across 20 reference assignments; pairwise overlaps are 0–2. Within-fold fitting/reference/test roles are disjoint. | Added the actual reuse counts and stated that folds are not independent replications. |
| Shrinkage and the variance floor need scale context. | No floor binds among 960 checked regional denominators. Primary latent-benefit variances are 0.04614–0.36244; coordinate-benefit variances are 0.00001629–0.006059. | Reported scales and the fixed, untuned shrinkage choice. Did not choose a different coefficient after viewing test scores. |
| Interval precision and AP prevalence need clearer presentation. | The raw machine-readable interval stores full precision; only 23 normals limit sampling. The weighted no-skill AP baseline rounds to 0.908. | Rounded prose intervals, added the prevalence baseline and emphasized coarse conditional uncertainty. Three-decimal AUROCs remain consistent identifiers of point estimates, not claims of precision. |
| The longer coordinate-loss summary had an overly broad sentence. | Fold 2 reverses the own/full ordering after longer training. | Corrected the supporting report to four primary folds and three longer-run folds. |
| The development-probe units and timing shift are easy to misunderstand. | The [probe audit](reviews/normal_reference_review_response.md) confirms temporal crops and standardized profile differences. | Explicitly stated that the edit shifts frames, not spatial position, and that profile changes are not raw MSE. |
| Results need a clearer uncertainty graphic. | All five displayed paired contrasts are derived from the same frozen primary source scores. | Added a vector forest plot with one positive direction, named comparators and explicit primary versus supporting status. |

## Suggestions rejected or narrowed

**“Reference sets necessarily overlap heavily.”** This is not implied by four sets of five drawn from 23 normals. The actual overlaps are zero to two sources per pair. Reference and fitting reuse still make the folds dependent, so the manuscript now discloses the measured overlap rather than the critic's assumption.

**“The whole-body shift should disappear under pelvis centering, so the score is a bug.”** The operation is temporal: frames 4:36 versus 8:40, a four-frame or approximately 0.267-second phase change under the same original-prefix scale. It is not translation in image coordinates. The reported +0.765 is a change in a standardized squared profile distance, not raw coordinate MSE. The median raw full-coordinate MSE change is +0.006606. The code and outputs do not support the alleged translation-invariance bug. We did not cap or winsorize the result to make it look reassuring.

**“The bootstrap endpoints are mathematically suspicious because they are discrete.”** A source-level rank statistic naturally has discrete bootstrap values. The stored floating-point digits do not indicate a calculation failure. The useful correction was to round the presentation and explain the small normal sample, not to alter the valid calculation.

**“Intervals covering zero establish a null, so multiplicity no longer matters.”** They do not establish equivalence or absence of an effect. The revised paper keeps an inconclusive conclusion, identifies unadjusted supporting comparisons and avoids a positive significance claim from selecting among them.

**“The learning-adequacy failure proves the score measures only representation idiosyncrasy.”** It prevents attributing the score to optimal information gain, but does not prove a unique cause of its rankings. The paper keeps inadequate learning, representation choices, calibration and observation differences as unresolved explanations.

**“The below-chance longer-run contrast sharpens a proven null.”** The 0.398 point estimate alone does not establish a significant reversal or a null. We retain the original direction and report it as an additional limitation of interpretation.

**“Promote the reference-free contrast or a favorable longer schedule to primary.”** This would change the central comparison after outcomes were known. Both remain supporting analyses. The 512-update primary is preserved.

**“Workshop fit includes assistive-technology-for-disability.”** The workshop's actual name is Automated Spatial and Temporal Anomaly Detection. The raw review's expanded phrase is inaccurate and was not adopted. Fit is based on anomaly detection in spatial and temporal observations.

## What revision cannot repair

The revised writing cannot supply more normal participants, erase earlier analyst exposure, validate a published comparator, or create clinical review. It cannot turn inconclusive intervals into evidence of superiority or absence. These remain explicit weaknesses in the [readiness assessment](submission_readiness.md). The next meaningful improvements are estimator-adequacy experiments, a validated published comparator, independent normal participants and qualified movement review.

## Final revision check

The [second GPT-6 Astra pass](reviews/revision_check_gpt_astra.md) found no substantive claim inconsistency. Its remaining label and grammar comments were corrected: random latent heads are named explicitly in the main table and uncertainty figure, and the teacher description uses “an exponential-moving-average teacher.” The final PDF review additionally removed Type 3 fallback glyphs from the method diagram and added an explicit reference to that figure. The primary results remained unchanged throughout these revisions.
