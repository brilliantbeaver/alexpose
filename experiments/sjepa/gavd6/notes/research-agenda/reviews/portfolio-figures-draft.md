# Independent visual and workflow review

Reviewed 14 September 2026. This review inspected rendered PNGs with `view_image`, not only SVG source or layout-check output. No figures were modified by this reviewer.

## Material inspected

- The complete mechanism contact sheet, covering all seven mechanism diagrams.
- The complete experiment contact sheet, covering all seven experiment diagrams.
- The full portfolio overview.
- Full-size previews of `05-mechanism`, `03-experiment`, `06-mechanism`, and `07-experiment` to check the most consequential wording and the dense experiment layout.
- The current proposal text for the related workflows, particularly P2, P3, P5, and P6.

The diagrams are concept schematics. They do not plot measured improvements, and their visible caveats clearly state that experimental outcomes are not shown.

## Layout verdict

**Pass, subject to the conceptual wording corrections below.** No visible text overlaps, clipped labels, arrow crossings through text, or cluttered connectors were found. Card spacing, reading order, contrast, and type hierarchy are consistent. The contact sheets remain legible for overview inspection, and the full-size figures provide comfortable reading. No layout redesign is needed.

The overview correctly separates ranking from proposal identity: fifth-ranked P6 precedes sixth-ranked P5. Its final instruction to pilot P1/P2 and check P3's oracle matches the portfolio's resource decision.

## Required conceptual correction

**`05-mechanism.svg`, main title:** replace “More evidence should earn more confidence.” with **“Confidence should respond to the evidence.”**

The current title implies that increasing evidence should monotonically increase confidence. P5 explicitly allows an informative observation to reveal unusual movement and increase uncertainty. Its experiment rewards improved proper scores, not universally narrower intervals. The subtitle already states the correct qualification, but the headline should not contradict it.

## Small scientific wording correction

**`03-experiment.svg`, first card, second bullet:** replace “Repair and retention oracle” with **“Repair oracle; event retention”** or **“Block oracle and retention.”**

The oracle selects among the actual repair blocks, candidates, and strengths using the repair objective. Its event retention is measured. One repair-optimal point does not automatically provide a maximum-retention oracle or a complete preservation-versus-repair frontier. The remainder of the figure correctly preserves the 25% repair requirement and the 15-point retention comparison at comparable achieved repair.

## Optional clarity improvements

**P5's final mechanism icon:** a single point-forecast curve is a weaker illustration of distributional evaluation than a simple pair of interval or density glyphs. The accompanying “Proper score and coverage” and “Width alone is insufficient” labels are accurate, so this is optional. If changed, show possible forecast distributions without invented numerical scores or a universally narrower after-distribution.

**Portfolio overview:** small labels distinguishing “Rank” from “Proposal ID” would make the P6/P5 ordering easier to interpret without reading the README. The current order is correct; do not renumber the actual proposal files to match rank.

## Workflow checks that passed

- **P1:** selection precedes expensive reveal. “Do not inspect all crops” and extraction-cost accounting prevent a false acquisition-saving story. Reuse across another forecaster is visually distinct from choosing the first query.
- **P2:** full-body joints come first; surface information is explicitly additional. Natural-motion prediction and strong compression comparisons are prominent. The twist image is labeled as a conceptual example rather than empirical evidence.
- **P3:** a useful repair candidate is an explicit prerequisite. Exact identity is allowed, and rejecting all damaging proposals is explicitly insufficient. Opposite local decisions and transfer to a new refiner are represented accurately.
- **P4:** support walking and query activity are separate. Correct-person future prediction, matched donors, raw support, and the small eligible cohort remain visible. No health or impairment inference is implied.
- **P5:** duplicates, overlap, and actual reveals are distinguished. The experiment requires an observed fusion failure, simple remedies, and proper forecast scoring. Only the main mechanism title needs correction.
- **P6:** candidate proposal and independent checking are separate. The subtitle limits witnesses to declared tracks and bounds, rather than every RGB pixel. The figure explicitly states that failed search does not prove uniqueness.
- **P7:** actual student training and physical forecast evaluation follow target selection. No-distillation, simple transforms, decision cost, and held-family comparisons remain visible. It does not present latent R² as the final scientific outcome.

## Completion status

The root agent should make the two wording corrections, rerender the affected figures/contact sheets, and inspect the corrected P5 title and P3 bullet. This memo records the first independent visual review, not a claim that those corrections have already been applied.
