# Readability review — 18 September 2026

Reviewer: a separate Codex worker reviewed the actual canonical notebooks, notebook README, coordinator-authored protocol, `workflow.build_report` source and completed fixture report/plot. This worker authored the notebooks and their builder, so **the notebook observations are a disclosed self-review, not independent validation of that work**. The protocol/report wording review is separate from their author. This review addresses whether a general technical reader can identify the question, comparison, result, uncertainty and next step; it does not certify scientific effects or repeat the implementation review.

Review state: **complete** for the actual artifacts identified below, including the subsequent `fixture-notebooks-20260918-final` run. The final report addresses the substantive textual ambiguities, and the later faceted plot resolves READ-8's overlapping labels. An unordered next-action paragraph remains a low-priority limitation. Findings READ-1 through READ-7 preserve the first inspection; their dispositions appear below. READ-8's original observation and subsequent verified correction are both retained.

## Findings

| ID | Priority | Actual wording or display and likely misunderstanding | Smallest adequate correction |
|---|---|---|---|
| READ-1 | Medium | The report question promises assessment of “timing or side-specific variation”; notebook 05 says “amplitude” and “event timing.” The implemented amplitude is the RMS of demeaned signed **horizontal projected ankle separation**. Timing is the time of positive local maxima of that signal, with explicit support rules. A reader could infer stride length, heel strikes or clinical gait events. | Name the signal and units in the report and notebook caption. Say explicitly that its maxima are operational 2D trajectory events, not heel strikes; RMS amplitude is dimensionless relative to the reference-box diagonal, not metric stride length. Display unsupported amplitude/timing counts beside coordinate results. |
| READ-2 | Medium | The report table column is “People”; its introductory sentence says “People are outer sampling groups.” In a fixture these are analytic grouping IDs, not observed individuals. “Independent evaluation scales” also leaves unclear whether independence refers to the model prediction or anatomical annotation. | For fixtures label the count “analytic person groups.” For source data use audited canonical people or verified recording groups as appropriate. Replace the scale phrase with “reference-box diagonals supplied separately from model predictions”; retain the distinct requirement for independent anatomical/temporal annotations. |
| READ-3 | Medium | The report lists `coordinate`, `direct`, `ordinary_jepa`, `paired_jepa`, `shuffled_jepa` and `initialized` without a nearby method legend. Notebook 04's narrative describes objective controls, but a reader must decode names or visit configuration files to understand the contrast. | Add a short legend: direct is an end-to-end denoiser; coordinate is matched masked clean-coordinate pretraining followed by a separately fitted readout; paired JEPA predicts clean-target embeddings; ordinary JEPA uses noisy-target embeddings; shuffled changes pairing; initialized fits a readout without encoder pretraining. Name SmoothNet-style as a practical comparator and retain that it is not an exact paper reproduction. |
| READ-4 | Medium | The generated-report template says only “Intervals and paired resampling draws are in `evaluation/`.” A reader sees many decimal places and a mean table without the interval's sampling unit, conditioning assumptions or missing-support status. The report poses a motion-preservation question but displays only coordinate means and a displacement plot. | State that paired intervals resample audited people with motions nested, condition on the fitted model, and exclude training/selection uncertainty; seed variability remains separate. Add a compact endpoint/support/decision summary, including insufficient motion endpoints. For fixtures say these intervals test arithmetic on analytic groups and are not population uncertainty. |
| READ-5 | Medium | The protocol's “per-frame coordinate control” can be read as an absolute no-history baseline. The implementation review establishes that its input origin and scale are computed over the common observed window. | Call it a “per-frame model with shared whole-window input normalization” in the comparison description and notebook 04. This preserves the intended preprocessing match while admitting the low-dimensional temporal information in normalization. |
| READ-6 | Low | Notebook 05 asks “Do coordinate improvements survive…?”, which presupposes an improvement. Notebook 08 says “Claims of novelty remain narrow,” which can sound like the proposed distinction is already established. | Ask whether a method reduces coordinate error **and** preserves supported motion. Describe motion-preserving extractor-shift evaluation and paired-JEPA advantage as hypotheses being tested, with novelty contingent on an empirical result beyond the nearest prior work. |
| READ-7 | Low | The current report ends with cost-import requirements, while several earlier gates require exposure audits, independent annotations and repeated finalists. A reader cannot easily tell which concrete task comes next or which branches remain pending rather than failed. | End the report with a short ordered next-action list appropriate to its evidence mode, keeping source audits, authorized source pilot, repeated finalists and real-reference calibration distinct. Preserve the historical personalization failure separately from unrun image-adaptation/video branches. |
| READ-8 | Medium, visual | The actual final `evaluation/accuracy-preservation.png` places roughly twenty method/extractor annotations in the upper-right cluster. They overlap, and several extend beyond the axes. A reader cannot reliably identify individual learned controls from this figure. Axis units and the `fixture-tested` title are readable; the report's tables retain method-level values. | In a future artifact revision, facet by extractor and use an external legend, numbered points or a cluster inset. Preserve all methods and the fixture label. Until then, use the report tables for individual comparisons and describe this figure as an overview only. This readability issue does not justify changing gates or selecting a method. |

The strongest competing explanation remains ordinary paired coordinate denoising plus a learned readout. Readability fixes should expose that comparison rather than make the latent objective sound inherently more meaningful.

## Required terminology audit

| Term | Observed use and verdict |
|---|---|
| self-supervised | No unqualified whole-pipeline self-supervised or label-free claim was found in the inspected notebook/report text. Retain “privileged clean-target synthetic supervision”; the matched readouts also use clean pairs. |
| gait | “Useful gait measurement” appears as a claim that diagnostics cannot establish. Keep this limitation. READ-1 is needed so projected ankle peaks and amplitude cannot be mistaken for validated gait events or clinical measurements. |
| world model | No world-model claim was found. Do not introduce one for an offline observed-window restorer. |
| ground truth | Reader-facing text uses targets, references and synthetic proxies instead. Preserve the distinction: projected SMPL-H centers are not independently annotated anatomy, and estimated confidence is not reference visibility. |
| independent | Ambiguous in “People” and “independent evaluation scales”; see READ-2. Reviewer independence is also limited by authorship, as disclosed above. Separate data independence, annotation independence, prediction-independent normalization and reviewer independence. |
| significant | No statistical-significance claim was found. Decimal precision, a supported bootstrap interval or a favorable fixture ordering must not be paraphrased as significant empirical benefit. Meaningful-effect margins and source uncertainty remain separate requirements. |

## Reader's five questions

| Question | Current answer a reader should be able to recover |
|---|---|
| What is tested? | Offline restoration of estimated 2D body-12 tracks with synthetic projected targets; whether paired latent pretraining adds value beyond matched coordinate learning. |
| Compared with what? | Unchanged/filtering controls, a SmoothNet-style practical MLP, a same-backbone direct denoiser, and objective/supervision/readout controls. The final report now supplies READ-3's method legend. |
| What is the result? | An executed CPU fixture can establish software behavior and artifact reconstruction. No new empirical advantage or real preservation result follows. Historical audit arithmetic has a separate evidence origin. |
| What is uncertain? | Target anatomical convention, source/extractor exposure, empirical effect size, supported timing/amplitude, training variability and real-transfer evidence. Source group bootstrap uncertainty alone covers only part of this. |
| What comes next? | Resolve source/audit/runtime prerequisites before an authorized pilot; obtain repeated source and independent real-reference evidence before calibrated confirmation. Notebook 07 saves only a development snapshot. |

## Final artifact inspection

The revised protocol, SHA-256 `fb9e9bac6cfd4e9236430df41d7124b5acddd32663aad8b496b6f843dcb56a33`, now defines ankle-separation amplitude/peak timing, qualifies evaluation-scale independence, and explains the static model's shared window normalization. It also distinguishes an admissible decision-specification structure from still-missing calibrated margins. The [protocol review follow-up](protocol-review.md#follow-up-inspection-of-protocol-clarifications) records exactly which specification findings are fixed and which empirical prerequisites remain pending.

After coordinator authorization, the notebook author corrected READ-5/READ-6 in the builder and regenerated canonical notebooks. Notebook 05 also now names the actual amplitude/timing signal (READ-1), labels fixture sampling groups as analytic IDs (READ-2), and explains bootstrap conditioning (part of READ-4). All six notebook contract tests pass after regeneration. These are self-authored corrections, not independent notebook validation.

The reviewer directly read the completed [fixture report](../../../outputs/synthetic-training-v2/fixture-validation-20260918-final/report.md), its [gate JSON](../../../outputs/synthetic-training-v2/fixture-validation-20260918-final/evaluation/gates.json), and visually inspected the actual [tradeoff PNG](../../../outputs/synthetic-training-v2/fixture-validation-20260918-final/evaluation/accuracy-preservation.png). Reviewed hashes:

| Artifact | SHA-256 |
|---|---|
| `report.md` | `f885095c54bfbd01c15c18afd809490f8f97882c5afaf934d799f63f67dae2c6` |
| `evaluation/gates.json` | `1d6ef9e38500886be4a282b6a4347299b708dc2e43c6a0f76ab79bcbe88b88c7` |
| `evaluation/accuracy-preservation.png` | `785e96db585f62a045e4115b73fdb73e894cf13c593b7fcc90fde14e5ecdfe9e` |

| Finding | Final disposition against those artifacts |
|---|---|
| READ-1 | Addressed in text. The report explicitly names 0.20-second displacement, signed ankle separation, demeaned RMS and operational positive maxima; it excludes stride length, heel strikes and clinical outcomes. Its support table shows incomplete amplitude/event coverage rather than presenting motion preservation as established. |
| READ-2 | Addressed by qualification. The table now says “Group IDs,” and nearby prose states that fixture IDs are analytic trajectories rather than sampled people. It distinguishes scale independence from predictions from independent anatomical annotation. The opening still uses generic “People” and “independent evaluation scales,” but the nearby explicit explanation resolves their meaning in this report. |
| READ-3 | Addressed. A method key distinguishes direct coordinate fitting, coordinate pretraining with a separate readout, initialized, ordinary/paired/shuffled JEPA, SmoothNet-style and static controls. |
| READ-4 | Addressed for this fixture report. The candidate is explicitly `paired_jepa`; positive reduction is comparator minus candidate in reference-box-diagonal units. The text names canonical-person/nested-motion resampling, retains render variants, excludes training/model-selection uncertainty and denies population interpretation for fixture intervals. Support counts are labeled correlated windows, not independent samples. |
| READ-5 | Addressed in the revised protocol, notebook 04 and actual report method key: the static arm retains shared whole-window normalization. |
| READ-6 | Addressed by the disclosed notebook-author corrections. The actual report asks a question and does not claim novel empirical superiority. |
| READ-7 | Residual low-priority presentation issue. Missing source/runtime/reference prerequisites and historical-versus-pending branch distinctions are explicit. They remain prose rather than an ordered next-action list. The protocol and notebook next-gate sections provide the sequence. |
| READ-8 | Open visual limitation, reported to the coordinator before any proposed implementation change. Text tables remain the readable comparison source. No production or builder change was made during final artifact review. |

The report and gate JSON agree exactly:

| Branch | Report status | Artifact status | Interpretation |
|---|---|---|---|
| A: image adaptation | `insufficient_evidence` | `insufficient_evidence` | Separate augmented-COCO/synthetic adaptation experiment is pending. |
| B: paired-JEPA advancement | `insufficient_evidence` | `insufficient_evidence` | Fixture execution cannot authorize scientific advancement. |
| Personalization | `fail` | `fail` | Historical panel remains closed on its retrospective opportunity diagnostic; this is not a new fixture-measured personalization effect. |
| Real transfer | `insufficient_evidence` | `insufficient_evidence` | Independent temporal references and held-recording evidence are absent. |
| Video | `insufficient_evidence` | `insufficient_evidence` | Optional incremental-benefit branch was not run. |

The numerical tables contain one training seed and three analytic group IDs per extractor. Neither the favorable paired-versus-coordinate fixture interval nor the less favorable practical-control comparisons establish empirical superiority, significance or failure of the scientific hypothesis. The report says so explicitly. No unqualified self-supervised, world-model, ground-truth or statistical-significance claim was found in the final report. “Useful gait measurement” appears only as an unsupported conclusion that latent diagnostics cannot establish.

**Final readability verdict:** a general technical reader can identify the restoration question, named comparisons, fixture-only result, resampling limits and missing empirical prerequisites without reading configuration code. Individual learned methods cannot be reliably identified from the current plot labels, and the final next actions could be ordered more clearly. This review does not endorse source effects, anatomical/temporal preservation, novelty or clinical use. Fresh-kernel notebook execution is separately recorded by the executor; this author does not relabel that execution as independent review of the notebooks.

## Verified follow-up: final notebook-run artifacts

The reviewer directly read the new [report](../../../outputs/synthetic-training-v2/fixture-notebooks-20260918-final/report.md) and [gate artifact](../../../outputs/synthetic-training-v2/fixture-notebooks-20260918-final/evaluation/gates.json), and visually inspected the new [faceted plot](../../../outputs/synthetic-training-v2/fixture-notebooks-20260918-final/evaluation/accuracy-preservation.png). The earlier run and its review above remain unchanged. This follow-up supersedes the earlier READ-8 disposition for the new run only.

| Artifact | SHA-256 |
|---|---|
| `fixture-notebooks-20260918-final/report.md` | `f73daaa1d1460a06967d6dfb6899ba77930171393cdb382a6254610761bbc495` |
| Its `evaluation/accuracy-preservation.png` | `4be8ff6e715fbff7ec923f3d311d054c32b3a555b806bfb9d8c3b97fc6cf68b4` |
| Its `evaluation/gates.json` | `1d6ef9e38500886be4a282b6a4347299b708dc2e43c6a0f76ab79bcbe88b88c7` |
| Updated `protocol.md` | `0cb35558323e335c637acfd5d0daa4dadbae18c7af6a67f500a27f5ecbe6b5e9` |

**READ-8: corrected in the new artifact.** Separate `fixture-held` and `fixture-source` panels replace dense text annotations. The twelve method names are readable in an external color/shape legend. Axis labels state the coordinate and 0.20-second displacement error units, and the title retains `fixture-tested`. Closely coincident points naturally remain close or overlap; the report tables give their exact values. There are no overlapping or cropped method-name annotations. The figure now supports an overview of the method comparisons without obscuring its legend.

The report text is identical to the prior reviewed report except for its run ID. Gate JSON values are identical: A, B, real transfer and video remain `insufficient_evidence`; historical personalization remains `fail`. The plot change does not alter effect estimates, uncertainty or gate interpretation. READ-1 through READ-6 remain addressed, and READ-7 remains a low-priority request to order the already stated next actions.

The appended protocol paragraph requires the actual calibration artifact and SHA-256, compatible schema/time contract, source or real-development evidence, reviewer/reference provenance, exclusion of candidate outputs and agreement with requested decision margins. Its last sentence explicitly says software cannot authenticate a human declaration and independent review must inspect references. This wording clarifies the validation boundary; it does not claim that calibration data or scientific margins have now been established. This readability review did not independently execute the calibration loader.

**Current readability verdict:** the new report and plot communicate the question, comparisons, fixture-only result and uncertainty limits. The remaining presentation issue is the ordering of next actions, not an unsupported claim of scientific success. All empirical, anatomical, real-transfer and confirmation prerequisites retain their previous status.
