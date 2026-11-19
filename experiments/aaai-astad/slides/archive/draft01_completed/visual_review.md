# Visual and evidence review

Reviewed on November 15, 2026. This record applies to the 40-slide deck built by `scripts/build_research_slides.py`. The deck now presents completed real results rather than a pending-results placeholder. The original 16-slide foundation deck remains in `archive/pre_neuromechanics/`.

All 40 PDF pages were rendered at 90 dpi and inspected in the four contact sheets under `review/`. The revised cohort, training, primary comparison, constituent controls, coordinate controls, reference/support sensitivity, source controls, categories, supervised classifiers, teacher diagnostic, longer schedule and submission pages were also inspected at full-page size. The review corrected a crowded final card, shortened an overlong takeaway, replaced the ambiguous “512 heads” heading with “512 head updates,” and avoided describing overlapping conditional intervals as demonstrated equivalence. The final inspected pages have no clipped labels, text collisions or unintended connector crossings.

The completed checks establish the following:

- The PDF has 40 pages, each 960 by 540 points, with a 16:9 aspect ratio.
- `pdfimages -list` reports no embedded raster images. Text, charts, shapes and flow diagrams remain vector content.
- All PDF word bounding boxes fall within their page boundaries. This check supplements visual inspection and does not independently prove that all overlaps are absent.
- All 109 local asset and reference links in the HTML resolve. External scientific references require internet access.
- There are 40 editable SVG pages, 40 rendered pages and four contact sheets. The eight detailed source figures remain available in the HTML notes.
- The active HTML, Markdown and slide README contain no em dashes.

## Evidence reconciliation

The builder reads full-precision metrics from primary run `3044566091261819`, longer-head sensitivity `eba2bd5a6e05efa1`, supplementary source control `7407af320461a092`, and the primary run's real fold-0 development controls. Both main runs have passed artifact verification. The builder also requires completed real summaries, all four folds, the declared trained/random-backbone/ridge target and observation-budget cells, and complete common-source coverage. It records the loaded evidence and file hashes in `result_input.json`. Values are rounded only for display.

The cohort slide separates the annotation inventory, media acquisition, pose QC and fixed profile-support gate. The 251-source evaluated cohort includes only 23 normal sources. None of these counts is presented as a verified count of independent people. Historical three-seed scores remain on their labeled historical slide and are not substituted for the new experiment.

The primary slide reports a paired AUROC difference of 0.101 with conditional interval [-0.035, 0.224]. The interval belongs to the difference, not to the individual bars. Constituent, random-backbone and local-coordinate controls are visible. The four-patch result and leave-one-reference-out range are labeled as sensitivities, with the reference range explicitly distinguished from a confidence interval. Category results are descriptive and dependent. The nuisance Random Forest and five-class capstone are explicitly supervised, so their information access is not confused with normal-only anomaly fitting.

The longer schedule is a retained sensitivity motivated by normal diagnostics. It does not replace the original primary or become a co-primary result. Its paired interval also includes zero. Random-backbone means a frozen randomly initialized encoder with fitted prediction heads; the slides do not describe the entire comparator as untrained.

The real teacher diagnostic distinguishes two coverage levels. Full three-region profiles under all paired edits remain scorable for five of twelve normal fitting sources. The separate eight-joint raw-loss diagnostic has 87 observed source-joint rows across twelve fitting sources. An unchanged own-context predictor can incur changed teacher-target KL. This is a contextual model-target effect, not a causal disease explanation or a decomposition of the nonlinear full-profile score. Training-source diagnostics are not labeled as held-out mechanism validation.

The final slide states that no learned JEPA advantage or clinical detection capability is established. Published anomaly comparators and independent validation remain missing. Toy signals, biological chains and future physical-world-model diagrams retain their illustrative or proposed status. A visually complete slide deck does not establish publication readiness.
