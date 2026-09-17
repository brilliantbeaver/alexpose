# Scientific and visual revision record

> **Review scope: September 14 seven-proposal portfolio and its 15 diagrams.** The later synthetic-training direction has a [separate review](synthetic-teaching.md).

Reviewed 14 September 2026. The final seven proposals are in [proposals](../proposals/), with the decision in [README](../proposal-comparison.md). This record documents how independent criticism changed the drafts. It does not report new model results.

**The workflow had three substantive passes.** First, separate research agents reviewed world models, gait/data availability and the scientific options while the lead inspected the existing results and portfolio. Second, drafts were written with concrete endpoints, examples, baselines and stop conditions. Third, an independent methods review and a rendered-image review challenged those drafts, and the lead revised the final documents and regenerated the figures.

The [world-model memo](world-models.md), [gait/data memo](gait-data.md), and [scientific options](research-options.md) preserve initial reasoning. The [adversarial methods review](portfolio-methods.md) contains concrete second-pass suggestions. The [visual review](portfolio-figures-draft.md) records inspection of the draft figures. Those memos can describe issues present before the final corrections; the proposal files are the current recommendations.

**The revisions changed decisions, not just wording.**

| Proposal | Adversarial concern | Final revision |
| --- | --- | --- |
| P1: observation value | Active acquisition already optimizes useful information. New linear mixtures of existing coordinate gains are not strong task generalization. | Main novelty gate now requires a frozen selector to transfer to a different forecaster and a held observation failure. Weighted questions are explicitly compositional checks. Initial tracking may not inspect withheld full-resolution pixels. |
| P2: motion beyond joints | A twist fixture does not establish useful natural information; extra points can add current pose geometry rather than temporal evidence. | Natural estimated-flow headroom comes first. Defined future transport as a change in pelvis-relative point offset. Added identical static off-axis-point controls, raw-flow fusion, PCA and longer joint histories. GAVD joint annotations do not certify surface twist. |
| P3: repair judgment | MoMask is not yet useful. A pointwise MSE oracle overstates a block verifier's options, and edit-based retention is undefined for arbitrary natural clips. | Candidate usefulness is a prerequisite. Use the actual blocks/grid, evaluate retention alongside repair, and distinguish sampled tradeoffs from a certified frontier. Keep 25%/+15 points on the controlled event comparison; require natural reference-error confirmation separately. |
| P4: motor memory | The old 20-person test requirement exceeds the split. Personal forecasting already exists. Rendered video contains rotations absent from position-only support. | State the approximately eleven potentially eligible people before verification, with no power promise. Added autoregressive, donor, morphology, full-rotation and same-pixel image/random-feature controls. Removed motion-generator adaptation as a dependency. |
| P5: evidence-responsive futures | A mixture head and evidence fusion are established; more information need not narrow a particular forecast. | First require an unresolved failure in two ordinary heads after duplicate rejection. Test a raw-plus-residual correction against calibration and redundancy weighting. Require final proper-score benefit under shifted evidence while retaining useful complementary information. No uncertainty decomposition or shift-coverage guarantee is claimed. |
| P6: ambiguity witnesses | Geometric admissibility does not prove full video ambiguity or human feasibility. Failure to find an alternative does not prove uniqueness. | Name the track-and-geometry observation model, give image-aware baselines the same RGB, and use independent constraint checking. Report finding-changing witness discovery at equal total cost. No search failure is treated as certainty. |
| P7: useful teaching | A covariance difference can be indefinite, and favorable target fitting can leak into held-configuration selection. | Added rank 1/2, cross-validated positive-direction selection and rank zero. Lock the criterion before held-family outcomes, while permitting a new transform to fit allowed training features. Actual student loss and decision regret remain endpoints. |

P5 was rewritten more substantially after the first critique. Its proposed correction now keeps both raw extra features and a training-fitted prediction residual; it does not assume that the residual is independent or sufficient. Its low novelty score remains visible. P7 likewise remains lower priority despite its inexpensive first test. The portfolio does not turn mature techniques into invented breakthroughs by changing their names.

**The common data contract was also revised.** We distinguished 1,874 GAVD sequences from 348 recordings and from unverified people, and 201 AMASS split rows from 189 unique audited identities. The often quoted GAVD binary result was qualified using the paper's actual evaluation design. No disease, force or severity endpoint was inferred from unavailable labels. Full-body motion is the default, with Core11 as a controlled ablation.

A quantitative GAVD panel is now an explicit prospective annotation task on existing videos, with 240 frames, 480 visible-landmark placements, an 8–12-hour human allowance and a partial second review. Any image-coordinate head must train outside those recording IDs. Render-trained zero-shot and disjoint-source pseudo-label adaptation are separate protocols, not interchangeable claims. Without annotation, real-video evidence is qualitative.

**The second pass also improved readability.** Each proposal starts with one measurable question and an everyday movement example, then explains inputs, the small method, comparisons, and the decision. Added plain definitions of latent representation, oracle, material point, residual, covariance, proper score and headroom. Physical endpoints and illustrative controls are separated. Costs are stated as allowances subject to measured throughput, not known runtimes. All proposed effect sizes are labeled prospective choices.

**The vector figures were rendered and challenged.** There are fifteen SVGs: one ranking map and a mechanism/experiment pair for each proposal. The drawing primitives were adapted from the existing portfolio generator. Each SVG contains a title, description and an explicit statement that it is a schematic without measured outcomes. PNG previews and two contact sheets support visual inspection.

Independent review inspected both contact sheets, the ranking map and full-resolution P3/P5/P6/P7 figures. It found no visible clipping or obstructed labels. Final changes were:

- P5's title became “Check what added evidence really changes,” avoiding a false rule that confidence must always increase.
- P5's final icon now depicts intervals checked against an outcome instead of a lone point forecast.
- P3's experiment says “Block oracle; event retention,” avoiding an implied maximum-retention oracle.
- P4's support-memory arrow goes directly to the forecast, separately from the query observation.
- The overview distinguishes rank from proposal ID because P5/P6 rank differently from their identifiers.

The lead regenerated all fifteen SVGs and inspected the corrected full-resolution P5/P4/overview images. XML parsing and measured text-box checks passed with **zero text collisions or canvas overflow**. Independent visual inspection, rather than those text checks alone, assessed arrows and conceptual clarity. The [layout record](../figures/layout-checks.json) records the automated checks; [previews](../figures/previews/) preserve the rendered artifacts.

The final local review checked proposal/image links, exactly seven proposal documents and fifteen SVGs, and the absence of em dashes in the new Markdown, figure labels and drawing scripts. No research code, Slurm jobs or model weights were changed or executed for the portfolio. The previous decision page and earlier proposal files remain available for comparison.
