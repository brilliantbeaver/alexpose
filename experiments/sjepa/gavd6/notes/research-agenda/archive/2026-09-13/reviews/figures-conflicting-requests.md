# Visual review: all proposal figures and the constraint workflow

**Historical review.** This concerns the former constraint-conflict candidate. Its figures are archived below. Contact sheets and the ranking map have since been regenerated for the [optical-flow replacement](../proposals/motion-beyond-joints.md); the current visual verdict is in [visual_flow.md](figures-motion-beyond-joints.md) and [visual_final.md](figures-final.md).

Actual rendered PNGs were inspected with `view_image`, not inferred from SVG/XML. The viewed files were:

- [Mechanism contact sheet](../figures/previews/contact-mechanism.png), covering all seven mechanism SVGs.
- [Experiment contact sheet](../figures/previews/contact-experiment.png), covering all seven experiment SVGs.
- [Former Proposal 7 mechanism at full size](../figures/previews/conflicting-requests-mechanism.png).
- [Former Proposal 7 experiment at full size](../figures/previews/conflicting-requests-experiment.png).
- [Portfolio ranking at full size](../figures/previews/proposal-comparison.png).

Together these expose all 15 requested figures. The latest rendered contact sheet and individual Proposal 7 experiment were re-inspected after the author incorporated semantic corrections.

## Layout verdict

All figures pass this visual inspection. Cards and text are aligned, whitespace is generous, and no text/line collisions, clipped labels or ambiguous arrow crossings were visible. The four-card mechanisms and three-card experiments keep the workflow readable. Arrows stay between cards. Proposal 7's full-size output labels have enough space and use both words and color, so color alone does not carry meaning.

The contact sheets have one blank final cell because there are seven panels. This is harmless for review. Individual SVGs remain the publication artifacts. Their schematic footers correctly state that no experimental outcomes are shown.

## Semantic review and revisions verified

Proposal 7's mechanism correctly separates motion proposals from independent bounds. It states that unsuccessful generation is not proof of impossibility. Its three outputs preserve unknown as an honest search outcome. The footer says that a learned model earns a role only if classical solvers leave substantial work.

The first reviewed Proposal 7 experiment used “Verify labels” and “Unresolved stays unknown.” I requested wording that would avoid suggesting unknown is a third ground-truth class. The latest rendered image now says **“Verify claims”** and **“Count unresolved queries too.”** Both fit the revised protocol and remain legible without wrapping or overlap.

The first reviewed Proposal 5 experiment said to stop if teacher responses added no information to raw motion. That could wrongly imply new information conditional on the complete deterministic rendering input. I requested an operational comparison instead. The latest contact sheet now says **“Stop if responses provide no equal-budget predictive benefit beyond direct motion training.”** It fits and expresses the actual experiment.

Other important visual caveats are present: Proposal 2 does not claim uniqueness after failed search; Proposal 3 judges actual student benefit rather than target R-squared; Proposal 4 requires proper-score evidence on real grouped motion; and Proposal 6 compares the same query with same-person versus matched-donor support.

The portfolio puts Proposal 7 at rank seven and describes it as using a prior only where verified search needs help. This is consistent with the substantive review. The footer distinguishes the initial Proposal 1 choice from Proposal 6's inexpensive independent signal test, and the figure states that its ranking is not an acceptance probability.

No remaining layout changes are required from this review. Any later diagram text edits should be rendered again before final delivery.
