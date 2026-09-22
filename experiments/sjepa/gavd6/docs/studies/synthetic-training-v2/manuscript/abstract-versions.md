# ICLR abstract revisions: paired synthetic pose restoration

*19 September 2026 · Ten abstract versions. Version 10 incorporates the completed eight-person, seed-17 development evaluation after two rounds of independent adversarial review.*

Start with [version 10: expanded results](abstract-v10-expanded-results.md), the current 203-word abstract. It describes the implemented method and the latest evidence: direct training improves position and short-interval movement errors relative to spatial calibration, while filtering sometimes performs better on movement error and paired JEPA adds little over strong controls. Exploratory checks still show motion distortion. Repeated training runs and budget comparisons remain planned. The [version 10 review record](abstract-review-v10.md) documents independent scientific, contribution and writing critiques using the requested weighted rubric.

**Current title:** Predictive Feature Learning for Pose Restoration: Position Accuracy and Motion Preservation.

The three reviewers scored version 10's final reviewed candidate **75.00–77.75/100** under the requested rubric. The evidence-dependent scores stayed fixed across its two wording rounds. The highest historical score for versions 01–07 was **70.50/100**; those drafts used an earlier evidence package and a different figure, so the scores are not directly comparable. Versions 08 and 09 have not been scored under that rubric. These are authoring assessments, not acceptance estimates. Additional independent people and seeds, and independently reviewed movement references, remain necessary for stronger conclusions.

## Abstract versions and why they changed

Filenames use `abstract-vNN-description.md` so they sort by version and explain their focus. The date remains inside each document. Earlier versions preserve the evidence available when they were written, including plans that have since been completed. Version 10 is the current study abstract.

| Revision | Main decision | Concrete remaining weakness | Weighted score / 100 |
| --- | --- | --- | ---: |
| [01 — evidence first](abstract-v01-pilot.md) | Assemble the predecessor findings, paired method and reported pilot without hiding unfavorable controls. | Three histories compete for attention; anatomical proxies and timing criteria need explanation. | 61.25 |
| [02 — preservation question](abstract-v02-preservation.md) | Connect laterality to bilateral measurement and use the original study's matched response-free control. | The incremental contribution of latent prediction needs clearer positioning. | 64.25 |
| [03 — representation comparison](abstract-v03-jepa-vs-coordinates.md) | Separate learned target features from coordinates under shared supervision. | Reviewer found the raw-track reduction's baseline omitted; later versions repair it. | 66.75 |
| [04 — controls and uncertainty](abstract-v04-baselines.md) | Lead with coordinate pretraining, name the raw baseline and include the relevant intervals. | The trained readout's importance and predecessor mechanisms remain too compressed. | 68.50 |
| [05 — readout and mechanism](abstract-v05-encoder-readout.md) | Distinguish encoder pretraining from a trained temporal readout and specify shuffled pretraining. | Historical wording overstates prior attribution; calibration needs a narrower interpretation. | 69.75 |
| [06 — conference draft](abstract-v06-iclr.md) | Explain controls more plainly and restrict calibration to measuring what simple corrections recover. | Full bilateral scope and a few comparator/provenance details need a final pass. | 70.50 |
| [07 — ICLR draft before diagnostics](abstract-v07-pre-diagnostics.md) | Restore the full objective, name motion-capture provenance and the matched selector, and separate synthetic preservation from real transfer. | Calibration and reference checks were still pending when this version was written. | 70.50 |
| [08 — calibration and motion results](abstract-v08-calibration-motion.md) | Incorporate the completed calibration controls, displacement results and reference-support checks. | Two development people and one seed; broader generalization and real-video transfer remain unestablished. | Not scored |
| [09 — study scope and planned evaluation](abstract-v09-study-scope.md) | Center the research question and implemented method, explain technical terms, and distinguish initial evidence from the planned main study. | At drafting, the expanded experiments remained unrun; the newer seed-17 findings are incorporated in version 10. | Not scored |
| [10 — expanded results (current)](abstract-v10-expanded-results.md) | Incorporate the completed eight-person evaluation, the direct/calibration/filtering comparisons, weak paired-JEPA increment and exploratory motion distortion in about 200 words. | One seed and a narrow development population; broader confirmation, reference review and real-video transfer remain outstanding. | 75.00–77.75 |

Versions 01–07 contain their own abstracts, critiques, next changes, evidence links and all eight weighted component scores. Versions 08–10 contain their abstracts and evidence notes; versions 09 and 10 have separate independent-review records. Version 10's record retains both scoring rounds from all three reviewers. Earlier drafts are records of the reasoning process and may retain weaknesses explicitly repaired later. Revision 01 preserves an appended arithmetic correction: its components sum to 61.25, not the initially written 62.0. Cross-version tables and machine-readable scores use 61.25.

## Historical scores for versions 01–07

Scores are on a 0–10 scale. A score of 5 indicates a substantial unresolved weakness; 10 indicates no material weakness within the stated scope. The total is the weighted sum multiplied onto a 100-point scale. Half-point judgments are approximate; the decimal totals show arithmetic, not measurement precision.

| Version | Contribution 20% | Accuracy 20% | Rigor 15% | Insight 15% | Reproducibility 10% | Clarity 10% | Figures 5% | Fit 5% | Total / 100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [01](abstract-v01-pilot.md) | 6.5 | 8.0 | 3.5 | 5.5 | 5.5 | 6.0 | 7.5 | 7.0 | **61.25** |
| [02](abstract-v02-preservation.md) | 6.5 | 8.5 | 3.5 | 6.0 | 5.5 | 7.0 | 7.5 | 7.5 | **64.25** |
| [03](abstract-v03-jepa-vs-coordinates.md) | 7.0 | 8.5 | 3.5 | 7.0 | 5.5 | 7.0 | 7.5 | 7.5 | **66.75** |
| [04](abstract-v04-baselines.md) | 7.0 | 9.0 | 3.5 | 7.0 | 5.5 | 7.5 | 7.5 | 8.0 | **68.50** |
| [05](abstract-v05-encoder-readout.md) | 7.0 | 9.0 | 3.5 | 7.5 | 5.5 | 8.0 | 7.5 | 8.0 | **69.75** |
| [06](abstract-v06-iclr.md) | 7.0 | 9.0 | 3.5 | 7.5 | 5.5 | 8.5 | 7.5 | 8.5 | **70.50** |
| [07](abstract-v07-pre-diagnostics.md) | 7.0 | 9.0 | 3.5 | 7.5 | 5.5 | 8.5 | 7.5 | 8.5 | **70.50** |

For versions 01–07, rigor remains **3.5** and reproducibility **5.5** throughout. The figure is the same reviewed schematic, so its score stays **7.5**. Accuracy concerns the faithfulness of qualified claims; it does not imply that summary-only results have become independently verified. Submission fit concerns appropriate scope and form, not acceptance readiness. The [evidence and rubric record](abstract-20260919-evidence-and-rubric.md) defines each dimension and explains these historical limits.

## Earlier title suggestions from versions 01–07

These are the seven titles used before the completed diagnostics. Version 10 retains version 09's title, shown at the top of this page. The independent literature notes preserve additional exploratory wording.

| # | Title | Why it works | Risk or qualification |
| --- | --- | --- | --- |
| 1 | Does Paired Synthetic Supervision Preserve Movement in 2D Pose Trajectories? | Direct statement of the research question. | The completed paper still needs an informative answer; a question title cannot substitute for sufficient evidence. |
| 2 | Paired Synthetic Supervision for 2D Pose Restoration: Accuracy and Movement Preservation | Names the supervision and both accuracy and preservation. | Longer and less explicit about the representation-learning comparison. |
| 3 | Testing Latent Prediction for Paired Synthetic Pose Restoration | Makes the latent-versus-coordinate objective comparison clear. | Gives less prominence to the broader movement-preservation objective. |
| 4 | Coordinate Gains and Motion Fidelity in Paired Synthetic Pose Restoration | Organizes the observed coordinate gains around the unresolved motion question. | Could be read as a completed two-endpoint evaluation unless support limits are explicit. |
| 5 | Evaluating Paired Synthetic Supervision for Movement-Preserving Pose Restoration | Names the intended application and paired supervision. | “Movement-preserving” can imply success on a quick reading; retain “evaluating” and avoid using it as a model name. |
| 6 | What Does Latent Prediction Add to Paired Synthetic Pose Restoration? | A concise ICLR-facing question about representation value. | Emphasizes JEPA’s incremental role more than the user’s broader preservation question. |
| 7 | Paired Synthetic Pose Restoration: Testing Accuracy and Movement Preservation | Preferred in version 07: states the object of study and its evaluation without assuming preservation. | Its strength depends on substantive diagnostic evidence; cautious wording alone does not establish novelty. |

Title 7 was recommended before the completed diagnostics. Title 6 offered an alternative emphasis on the matched latent-versus-coordinate comparison. A title promising a “motion-preserving JEPA” would exceed the current results.

## Evidence available when versions 01–07 were written

This section records the earlier evidence state. The calibration, reference-support and trajectory diagnostics described below as pending are now complete; see the [downloaded-results analysis](../results/postrun-analysis-20260919/README.md) and version 08.

The argument draws on three distinct studies. Laterality contributes summary-reported results from recorded motion capture with artificial joint-name swaps; its uniform control matched the proposed mechanism. Original synthetic training contributes locally recomputed aggregate outcomes, including the absence of a response-personalization benefit over a matched control. The revised study contributes a summary-reported one-seed synthetic pilot with two development people: coordinate errors improve over unchanged tracks, but strong controls leave the incremental JEPA benefit unestablished.

The abstracts do not claim real-video restoration. The adjacent real-video reflection/classification probe has retained predictions but a different task and overlapping source videos across splits; its new exploratory reconstruction is documented, then excluded from the restoration argument. Analytic fixtures supply software evidence only. Calibration, reference-self timing and trajectory analyses remain pending because the first post-run check failed before producing them.

New calculations are labeled **exploratory**. The [recomputation script](abstract-20260919-recompute.py) and [summary output](abstract-20260919-exploratory-summary.json) check ratios from rounded reported v2 errors. They do not create new observations, rerun inference or recompute intervals. Independent audits also report exploratory original-estimator decomposition and the separate real-video classification reconstruction, without promoting either to confirmation.

## Independent reviews and pipeline illustration for versions 01–07

Three independent reviewers audited [laterality and real-data boundaries](abstract-laterality-audit-20260919-v01.md), [original synthetic outcomes and matched controls](abstract-original-study-audit-20260919-v01.md), and [ICLR fit and primary literature](abstract-literature-positioning-20260919-v01.md). They subsequently critiqued successive drafts and the final candidate. Their reports preserve the specific findings and dispositions rather than merely recording approval.

![Conceptual paired synthetic training pipeline with separate observed and reference paths, representation pretraining, coordinate readout fitting, evaluation, and pending checks.](images/abstract-training-pipeline-20260919-v01.svg)

The projected reference branch is available for training and scoring only. The coordinate readout is fitted using projected training references after encoder pretraining; neither targets nor the teacher are available at inference. “L/R” means signed horizontal ankle separation. The figure explains the method and marks uncompleted checks; it does not visualize measured restoration success.

The [SVG](images/abstract-training-pipeline-20260919-v01.svg) and [PDF](images/abstract-training-pipeline-20260919-v01.pdf) are vector files; the [PNG](images/abstract-training-pipeline-20260919-v01.png) is a review preview. Geometry checks examined 55 labels and 14 arrows, finding no text overlap, container overflow or arrow/text intersection. Independent visual review found no clipping or crossed data paths. Its [generator](images/build-abstract-pipeline-20260919-v01.py) refuses to overwrite outputs. The screen-scale figure should be simplified before reduction to a narrow paper column; empirical trajectory plots still require valid artifacts.

## What would strengthen the paper next

The cached-data checks and expanded seed-17 diagnostics are now complete: training-only calibration, reference-against-itself timing support, preselected trajectories, displacement, amplitude ratios and fitting history. The [complete eight-person analysis](../results/seed17-complete-analysis-20260919/README.md) supports version 10 and supersedes the earlier two-person findings as the latest evidence. Readout-only and calibrated results must not be conflated. The remaining training seeds and budget comparisons will assess repeatability; follow the [one-week research plan](../research/iclr-one-week-plan-20260919.md) for measurement validation and the broader comparison.

Then obtain more independent people and motions, repeated seeds and adequate optimization. Independently review the joint convention and collect dense real temporal references before making transfer or anatomical measurement claims. Changed camera views or longer windows require a newly declared experiment; they do not retroactively repair this pilot. Keep using the [single Slurm guide](../../../../slurm/synthetic-training-v2/README.md) for operational commands.

## Submission details checked on 19 September 2026

The official ICLR 2027 call concerns representation learning broadly; no special annual technical theme was found. Abstracts are due **19 September at 04:59 PDT** (18 September, 23:59 AoE); papers are due **26 September at 04:59 PDT** (25 September, 23:59 AoE). Titles and abstracts may be revised before the full-paper deadline while staying close to the originally registered scope. These are official schedule conversions, not a submission made by this task. [Author guide](https://iclr.cc/Conferences/2027/AuthorGuidelines), [conference dates](https://iclr.cc/Conferences/2027/Dates).

The guide's explicit formatting section specifies nine initial main-text pages; its later FAQ contains an inconsistent ten-page statement, so use the explicit initial formatting rule. No abstract word limit was verified on the inspected guide; this series uses roughly 200–250 words as an editorial choice. The final manuscript also needs an accurate statement of AI assistance. These drafts used AI assistance for literature review, evidence organization, writing, critiques and figure preparation; authors must describe the assistance actually used and verify the final submission. [Author guide](https://iclr.cc/Conferences/2027/AuthorGuidelines), [AI policy](https://iclr.cc/Conferences/2027/AIPolicyForAuthors).

For auditability, [historical scores and word counts](abstract-20260919-scores.json) cover versions 01–07. The [earlier validation results](abstract-20260919-validation.json) preserve the checks performed at that time. [Version 10 scores and validation](abstract-review-v10.json) retain both new review rounds, the current word count, local-link checks and confirmation that versions 01–09 remain unchanged. Versions 08 and 09 have no historical rubric score.
