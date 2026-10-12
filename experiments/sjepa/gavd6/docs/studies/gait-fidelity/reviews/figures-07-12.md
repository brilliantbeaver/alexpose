# Independent figure/workflow review: Gait Fidelity figures 07–12

Reviewer: masking-audit subagent (did not generate these figures). Date: 2026-09-21.

## Review method

Opened every assigned PNG with `view_image` at both the native export size and its separate 900-pixel export: 07-video-evidence, 08-data-review-view, 09-source-splits, 10-assignment-uncertainty, 11-transformation-contracts, and 12-evidence-workflow. Inspected text containment, text/line overlap, margins, reading order, arrow meaning, and distinctions between proposed work and measured evidence. This review is independent of automated `layout-check.json`.

## Initial findings

| Figure | Layout at both sizes | Scientific/workflow judgment |
| --- | --- | --- |
| 07: local video evidence | Pass: three spacious cards, clean baseline and captions, no crossing lines or overflowing text. | Appropriately separates immediately available footage, annotations, and independent clinical references. Disease-folder labels do not identify an affected limb; estimator agreement is not independent truth. |
| 08: data review view | Pass: aligned cards, one clear common cursor, lower annotations do not touch the box edges. | Correctly labels synchronized overlays and traces as planned work. Native timestamps and shared time avoid separate warping of the two legs. |
| 09: source splits | Pass for geometry: two unobstructed arrows and comfortably fitting text. | Minor wording: “supported independent unit” overstates independence of unverified filename-derived sources. Prefer “Report uncertainty by source family,” retaining the person-identity limitation. Source grouping is an operational safeguard, not a proof of independent people. |
| 10: assignment uncertainty | Geometry pass, arrow semantics require correction. | Two independent arrows currently map assignment 1 to a zero signed mean and assignment 2 to magnitude a. Both hypotheses enter both summaries. Merge the two hypothesis paths before branching into signed and magnitude summaries. The displayed mathematics is otherwise correct for the stated illustrative equal-probability case. |
| 11: transformation contracts | Pass: legible, spacious, no overlaps. | Minor precision: the joint-slot column should explicitly concern estimated-input values and associated metadata, while anatomical references stay fixed when simulating a naming error. Avoid a reading that all reference labels should also be permuted, which would erase the tested error. |
| 12: evidence workflow | Fail: body labels overflow the approximately 192-pixel cards and enter arrow space. Visible at native and 900-pixel sizes. | Sequence is sound, but long lines such as “Anatomy and timestamps,” “Small gradient smoke test,” “Strong matched controls,” and “Fresh independent groups” need shortening/wrapping. Confirming once means a frozen final protocol with the declared seed set and new groups, not one neural run. |

The automated geometry receipt reported no findings despite figure 12's overflow. Add text-within-assigned-card containment; nonintersection checks alone cannot catch this case.

## Required changes before accepting the image set

1. Figure 10: combine both assignments into a shared distribution node or unambiguous junction, then branch to the two summaries.
2. Figure 12: shorten or wrap all card labels with comfortable inner margins; keep arrows out of text space. Reopen both export sizes after rendering.
3. Apply the minor source-unit and input-versus-reference wording clarifications, or clarify them explicitly in nearby captions.

Initial status: changes required, followed by the reinspection below.


## Final reinspection and disposition

Reopened regenerated figures 09, 10, 11, and 12 at both native and 900-pixel widths. Figure 09 now says “Report uncertainty by source family” and explicitly leaves person identity/cross-source dependence unverified. Figure 10 visibly merges both hypothesis paths at a junction before branching into the two summaries, with no crossing lines or text collisions. Figure 11 keeps anatomical truth fixed while modifying input metadata. Figure 12 uses shorter labels fully contained inside the cards, leaving clear arrow space. Both required findings and both minor scientific-precision findings are resolved.

**Final decision: accept figures 07–12 for documentation.** All assigned figures were independently opened at both output sizes, and no unresolved clutter, text overflow, text/line collision, or misleading arrow remains. The figures accurately distinguish proposals and illustrations from empirical findings. This is a visual/workflow review, not validation that proposed gait methods will work.

The exact six SVGs and twelve PNG previews reviewed are hashed in `/private/tmp/gait-fidelity-visual-review-02-assets.json`; copy that receipt alongside this report if retaining the review in the study folder.
