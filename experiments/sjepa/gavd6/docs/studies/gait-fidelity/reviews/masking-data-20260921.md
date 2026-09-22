# Review of the masking and local-video extension

21 September 2026. This record covers the new research documents, conceptual SVGs and source audits. The older [proposal review](proposal-20260919.md) and historical move/validation receipts retain their original scope.

## Independent work and review

Three independent agents examined the masking implementation, raw-video collection, and primary masking literature while the coordinating author built the documents and figures. The masking and literature reviewers then visually opened all twelve figures at both 1,000-pixel native width and 900-pixel embedded width. They did not generate those figures. Changed figures were reopened after correction; acceptance is based on rendered images as well as geometric checks.

The video reviewer separately checked the coordinating author's integration against the actual inventory, timing limitations and GAVD overlap. The literature reviewer challenged the masking experiment's comparators, prior-art claims, target-information boundary and occlusion estimand. These reviews improve the proposal; they do not replace independent clinical annotation or a completed experiment.

| Finding | Change made |
| --- | --- |
| Synthetic v2 already uses stochastic masks. | Reframed the experiment as comparing mask structures; no claim of repairing a permanent mask in v2. |
| MS masks overshoot the nominal rate and underexpose central hips. | Retained a reproducible mask bank, per-joint/time coverage and achieved-budget requirements. |
| BlazePose-33 indices and clinical weights do not transfer directly to body12. | Required a named-joint graph adapter and separate testing of any region bias. |
| Token-level masking can pass while preprocessing still reveals hidden inputs. | Distinguished pretext masking after normalization from simulation of genuinely missing raw observations. |
| The end-to-end direct model does not use the pretraining mask. | Defined coordinate-pretraining versus paired-JEPA arms and retained direct training as a practical comparator. |
| A contextual reference-feature target differs from a raw-coordinate target. | Defined matched exposure as windows, query positions, validity and downstream labels, while stating the intended target-content difference. |
| An occlusion may remove all evidence needed to distinguish two true movements. | Separated recoverable cases from uncertainty/abstention cases; no deterministic recovery requirement for indistinguishable inputs. |
| A fresh partition cannot make familiar videos an untouched cohort. | Kept this collection developmental, with internal cross-validation labeled exploratory and fresh references/groups required for confirmation. |
| Twenty-eight source IDs already occur in GAVD. | Retained the exact overlap list and barred an unaudited independent-transfer claim. |
| Decoded timestamp checks do not verify the capture clock. | Narrowed the wording to readable, monotonic presentation timestamps and cadence. |
| Four reference subscripts rendered as empty squares. | Replaced them with ASCII Y00/Y01/Y10/Y11. |
| Workflow text overflowed narrow boxes. | Shortened the labels and added an automated text-within-box containment check. |
| Uncertainty arrows assigned each summary to only one hypothesis. | Merged both hypotheses before branching to signed-mean and magnitude summaries. |
| Color alone distinguished graph mask states. | Added hollow target markers while context markers remain filled. |

Full visual reviews: [figures 01–06](figures-01-06.md) and [figures 07–12](figures-07-12.md). Historical findings remain in those records; their later disposition sections identify the accepted revisions. [Final validation](../records/history/extension-validation-20260921.json) records current asset hashes and link/layout checks.

## Verification and limits

The figure builder parsed all SVGs, rendered 24 PNG previews, and checked text bounds, text intersections, text containment within assigned boxes, and orthogonal connector/text intersections. Independent image review caught defects that the initial geometry checks missed. All twelve diagrams are conceptual; none is presented as a measured restoration result.

The sampler audit was independently rerun with the retained script, reproducing the 63.276% achieved target fraction and the recorded joint/time frequencies. The raw-video audit hashes all 91 MP4s, checks all decoded frame timestamps and inspects every existing pose cache. The visual feasibility screen consists of 43 midpoint frames and 24 additional time samples, not exhaustive continuous annotation.

The local gallery has 91 original-video links and local thumbnails, source/label filters, timestamp insertion and JSON note export. Static checks validate its asset paths and JavaScript syntax. Live browser interaction and playback could not be checked in this session: the available browser-control service reported no browser, and an isolated headless launch did not start. This limitation applies to the gallery interface; the original videos were successfully decoded by ffprobe/ffmpeg. No external video upload occurred.

No new restoration training, source extraction, clinical annotation or confirmatory evaluation was performed. The sibling notebook, raw videos, historical study results and prior validation receipts were not edited. The principal unresolved work is the schema-safe graph sampler, a physical-time pose cache, independent annotations and a frozen new experimental comparison.
