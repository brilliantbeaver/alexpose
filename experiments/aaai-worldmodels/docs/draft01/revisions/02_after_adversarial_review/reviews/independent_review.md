# Independent manuscript and evidence review

**Disposition: accepted as a compiled internal short-paper draft with bounded claims. Human authorship and submission metadata remain unresolved.** This is a review of the completed manuscript and empirical artifacts, not an acceptance prediction for the workshop and not permission to submit publicly.

The reviewed title is **Gait Forecasting for Medical World Models: A Source-Grouped Study**. The PDF contains six pages: the body extends through page four, references begin on that same page, and appendices occupy pages five and six. This fits the stated short-paper body range of two to four pages. All twelve inspected font entries are embedded, and rendered text-bound checks find no page-edge overflow. All six current pages and all three standalone figure previews were inspected visually. The final reproducible PDF rebuild has identical extracted text and rendered pixels on every page compared with the visually inspected version.

## Exact reviewed artifacts

| Artifact | SHA256 |
| --- | --- |
| `manuscript.pdf` | `ed369e5497864f9766207bc41a69f7dca17952ed337f0be94cddce6af9f5b39e` |
| `manuscript.md` | `62c26445c78086a2ef049b325cfa8b6d3beb0a008159c74426e6d93ab1f5e3de` |
| `manuscript.tex` | `0dc6539be692460c278d9cafc355d0742fa9c0a4429578f547ab39747497dea2` |
| Completed empirical `results.json` | `f14a6e533338c96da99b5085cb231b9de5cc47f7fe50a4c118225c42dc1fa1e8` |
| Accepted experiment implementation | `226b276ae3d0759624aec9ae6acff59d0cf282b795bc309db814c4f49fe8259c` |
| Frozen experiment protocol | `e7c331238c3ae61237336c11b36526afb20952ae31da51d9374962e98eafd299` |

Later content or figure changes reopen their associated checks. The reviewer provided pre-outcome critiques but did not author the implementation, original run results or manuscript. The independent empirical replay and remaining scientific limitations are documented in [the final experiment review](../../restart/final_adversarial_review.md).

## Evidence supporting the manuscript

The data account reconciles 666 selected annotation sequences, 652 media-eligible sequences, 641 sequences after historical whole-source exclusions, 2,818 prepared windows, and 2,815 final model windows from 456 sequences and 91 source videos. The primary forecast endpoint is based on 29 normal sources. The manuscript correctly distinguishes sources from independent patients and explains that extra windows improve temporal coverage rather than independent sample size.

The reviewer independently reconstructed all 2,818 prepared windows, checked all 96 source caches, and replayed all eighty saved real model checkpoints. Every saved float32 prediction matched exactly after storage conversion. Window errors, source aggregation, all paired contrast intervals, and all reported anomaly AUROCs were recomputed. All 94 model artifact hashes passed. Ten selected output-coordinate fits also matched an independent scikit-learn weighted Ridge calculation within approximately `6.2e-15`. The evidence is in `artifacts/restart_review/real_replay.json` and `prepared_input_audit.json`.

The manuscript's positive result is supported: normal-source bilateral MSE is 0.08063 versus persistence MSE 0.12126, a difference of 0.04063 with conditional source-bootstrap interval [0.02399, 0.05911]. The reported 33.5% is the difference divided by the mean persistence MSE, not a separately estimated average of individual percentage gains. Bilateral error is lower on 23 of 29 normal sources.

The manuscript also preserves the consequential limitations. Own-history gain is uncertain. The separately fitted timing control is worse, but the current-posture transition model is stronger than bilateral history. The complete declared timing hierarchy therefore fails. The optional shrinkage sensitivity does not improve the bilateral result. The secondary excess-error AUROC of 0.793 remains a fixed-direction exploratory annotation association, alongside the below-chance raw-error scores and every declared comparison.

## Findings and revision checks

| Finding raised during review | Disposition in the reviewed draft |
| --- | --- |
| Calling the prefix and suffix two equal halves misdescribes the 16/8 observation contract. | Replaced with the explicit sixteen-sample prefix and eight-sample suffix. |
| The eight-step MSE could be mistaken for error only at the final instant. | The methods state that it averages all eight future observations, nominally 0.1 to 0.8 seconds. |
| The preparation count and model count differed by three windows. | The data account explicitly identifies the common distal-target support gate and both counts. |
| Pooled out-of-fold ranking assumes comparable scores across different fitted models. | The discussion now states this assumption and the possibility of cross-fold offsets. |
| Aggregate attrition conceals annotation-dependent selection. | The data account now reports normal retention of 141/287 sequences and myopathic retention of 168/183, and explains that duration, geometry and observation gates all contribute. |
| The original method figure had text touching or crossing two box borders. | Labels were shortened and the regenerated vector figure fits its boxes without overlap. |
| The first PDF build failed because required fonts were unavailable. | The current PDF compiles with the recorded portable pdfTeX environment and the unchanged official style. Its fonts are embedded and its rendered pages are legible. |
| A seven-page earlier render left a potentially stale page preview after repagination. | The renderer now removes previous page-numbered outputs, and the final directory has no stale page seven. |
| Regeneration initially depended on a sibling study's author-kit location. | The generator now builds from the bundled official files. The author demonstrated byte-identical outputs with the sibling root deliberately unavailable, and this reviewer verified final text and pixel identity. |

The strongest improvement is the integrated distinction between prediction utility and a stronger timing interpretation. The abstract now presents the positive persistence gain and stronger static comparator together. The paper does not conceal the failed hierarchy, switch score direction, invent clinical diagnoses or describe a static-input model as a motionless output baseline.

## Systemic weaknesses and concrete next steps

**Scientific identification remains limited.** The experiment measures short-term relative articulation, without treatment input, disease progression, verified clinical state or neural measurements. Keep the observation-transition framing. A future medical world-model study should collect longitudinal clinical outcomes and explicit interventions before making counterfactual or treatment-response claims.

**Generalization remains internal.** Source IDs do not establish unique patients or sites, and earlier GAVD results informed the restart. More windows and nested folds do not remove those limits. A new external cohort with verified patient identifiers, source acquisition metadata and clinician assessments is the highest-value next experiment.

**The contribution is an evaluation method and empirical case study rather than a demonstrated state-of-the-art architecture.** Related own/opposite-leg prediction exists in the sibling study, and the new cohort does not include a matched neural comparator. The draft appropriately avoids first-method or superiority claims. A future benchmark comparison should hold observations, source partitions, valid target support and tuning budget constant.

**Quality and identity need additional measurement validation.** The new full-duration extraction is computationally verified, but person containment is not complete tracking or clinical identity adjudication. Duration and visibility filtering affect labels unequally. Blinded trajectory review and matched-view or sensor-reference studies would strengthen a subsequent claim. Such work should not retroactively delete unfavorable sources under the current run identity.

**Conditional intervals and pooled anomaly scores need careful presentation.** The draft now states fixed-fit bootstrap limits, overlap among training folds, lack of multiplicity adjustment and cross-fold score-scale assumptions. These qualifications must survive any abstract shortening or slide revision. No post hoc choice of score sign, threshold or winning secondary arm is warranted.

## Submission readiness

The document is ready for human scientific and author review. The empirical tables, scope, controls, rendering and short-paper body length have passed this independent check. The remaining submission items are concrete: replace the author and affiliation placeholders, confirm acknowledgments and relevant AI-use disclosure, check the live submission invitation and portal fields, and obtain the actual authors' approval of the final file. The paper has not been publicly submitted.

External validation and the stronger mechanistic study would improve the work, but they are not represented as completed prerequisites hidden behind the present workshop claims. The current draft can be assessed on its transparent exploratory contribution. Workshop acceptance remains an editorial decision.
