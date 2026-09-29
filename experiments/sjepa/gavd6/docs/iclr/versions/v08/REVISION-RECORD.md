# Manuscript portability cleanup — 26 September 2026

Reader-facing text in the main paper and both appendix versions now describes
the experiments independently of the local project organization. References to
directories, named files, notebook IDs and walkthroughs, machine checks, disk
capacity, and remote execution have been removed or replaced with descriptions
of the scientific procedures. The experimental sequence, participant splits,
losses, numerical results, and inferential limitations are retained.

The AI-use disclosure retains all human contribution and approval assertions;
its list of inspected materials now uses the general term “analysis materials.”
This is an editorial wording change, not a new human verification event.
Historical copies, including the formerly verbatim technical appendix, remain
in history/v08-before-local-reference-cleanup-20260926.zip and earlier snapshots.
The current technical reference has been edited for standalone readability.

Relative paths needed by LaTeX and bundle build instructions remain in the
source package, but no local filesystem or notebook references are printed in
the manuscript. Current reviews and checks are in
reviews/portable-manuscript-methods-review.md,
reviews/portable-manuscript-evidence-review.md, and qa/portable-manuscript-*.json.
The rebuilt default PDF remains nine main pages and 16 pages overall. All eight
figure sets, seven numerical table sources, and 13 evidence CSVs are unchanged.
The first human-authorship paragraph is byte-identical. The unexecuted 102-fit
instructional design is omitted from the technical reference because it is not a
completed experiment. A redundant question-mapping table was also removed from
the optional reference; its scientific qualifications remain in the methods and
experimental-sequence discussion. This removes an otherwise empty continuation
page. Older review and preservation statements below describe
their dated snapshots.

---

# Figure 1 workflow and gait illustration — 26 September 2026

Figure 1 now explains the experiment through original vector keypoint
illustrations and a three-stage workflow: create a controlled motion pair,
fit the feature model and readout on 112training people, then score restored
windows from 14development people. The original and edited icons each contain
the 12selected joints; stacked outlines represent motion windows. Their geometry
is hand-designed and explicitly marked illustrative, never presented as source
motion or a successful model reconstruction. The edit has no displayed numeric
magnitude or clinical interpretation.

Endpoint and delta are alternative training runs with the same base objective.
Training references supply teacher targets and coordinate supervision. At
evaluation, each observed window enters the restorer separately, while references
enter the scoring box only. The diagram now makes asymmetry-change error and
trajectory error visible, alongside posthoc left–right assignment. The shortened
caption points to the exact feature losses and retains the direct-fitting
exception and adaptive development reuse. The method section retains EMA,
score transforms, query support, and physical-mirror/naming distinctions.

I-JEPA Figure 3 and MotionBERT Figure 1 were inspected as design precedents for
separate training paths and a staged system overview. The artwork is original;
no published figure or invented empirical example was copied. The complete
record is evidence/figure1-design-provenance.json. Independent reviews are in
reviews/figure1-methods-review.md and reviews/figure1-evidence-review.md.

The figure remains legible at 5.5 by 3.15 inches. The manuscript retains nine main
pages and 16 total pages. The three-row Table 1, five-page appendix, numerical
results, and human-attested disclosure are unchanged. The earlier package is
preserved in history/v08-before-figure1-redesign-20260926.zip. Final source and
Overleaf rebuilds are documented in qa/figure1-rebuild.json; no new human
verification event or source training is claimed by this editorial revision.

---

# Comparison and appendix simplification — 26 September 2026

Table 1 now presents the three declared experimental questions, each with its
candidate, comparator, outcome, and encoder-adaptation distinction. It no longer
mixes model controls, feature targets, and readout repairs into six “Family” rows.
The initial core/change versus direct/change comparison remains distinct from
the practical direct-coordinate benchmark.

The appendix is reduced from twelve pages in ten sections to five pages in five
sections. A covers split/shape integrity and inference, B gives the essential
training specification, C shows the three primary paired effects, D explains
sensitivity diagnostics and reproduction, and E retains all 14 participant
profiles. Two vector figures replace repeated prose and comparison inventories.
The single main waveform figure still displays all eight procedures and both
objectives because it supports the across-procedure deterioration finding.

The full prior technical appendix is preserved byte-for-byte as
`supplement/technical-details.tex`, with instructions for typesetting it instead
of the concise appendix. Both source bundles include it and all seven numerical
tables. All existing evidence CSVs and six original figure sets are unchanged;
new forest estimates are selected from the three original comparison JSONs.
No new method average, fitted model, experiment, hypothesis test, or numerical
result was introduced. Details needed to interpret the findings remain in the
paper; exhaustive reconstruction details remain accessible in the supplement.

Independent evidence and methods reviewers checked scientific selection,
source-data preservation, the two new figures, and the shortened protocol.
Corrections clarify the squared angular residual, the rounded mask budget,
paired error differences, and cross-references in both appendix variants.
Review records are `reviews/comparison-selection-review.md` and
`reviews/appendix-simplification-review.md`. The earlier Codex adversarial report
below remains specific to its framing-stage snapshot; no new plugin review is
claimed for this revision.

The final paper keeps nine main pages, disclosure/references on pages 10–11,
and a five-page appendix on pages 12–16. Official style files, citation content,
and the human-attested AI disclosure are unchanged. This AI-assisted revision
records no additional human verification event beyond the existing attestation.
The prior files are in `history/v08-before-comparison-simplification-20260926.zip`.
Current structural, visual, and clean-build checks are recorded in
`qa/validation.json`, `qa/simplification-review.json`, and
`qa/simplification-rebuild.json`. Older records below describe earlier snapshots.

---

# Clarity revision — 26 September 2026

The abstract now defines the paired asymmetry-change outcome before presenting
errors. It explains that zero response always predicts no change, makes the error
difference from direct coordinate fitting explicit, and states why the interval
for the feature-difference comparison leaves its benefit uncertain. Decoder-loss
findings now refer specifically to knee-angle trajectory accuracy. The abstract
retains post hoc labeling for the geometric assignment check and the future-only
world-model motivation.

The introduction explains the geometry of the measurement, then introduces
feature vectors and the decoder before following the sequence of experimental
questions. Main methods and results distinguish model fitting from evaluation,
explain endpoint/delta objectives and frozen encoders, and separate failure
penalties from errors among successful outputs. The appendix simplifies support,
calibration, weighting, and notebook-reconstruction descriptions while preserving
all its numeric tokens, math, tables, captions and references.

An independent evidence review rechecked the original person/seed exports,
calibration receipt and relevant gait-fidelity notebooks. It reproduced the
zero-versus-direct interval and verified the eight waveform increases and the
readout-repair comparisons. The separate future-innovation notebook study is
not used to support this paper. A methods review checked the simplified main
text against the appendix. Corrections retained the post hoc scope, the exact
two-pixel and Euclidean-distance rules, and calibration by auxiliary gradient
magnitude rather than loss value. All review findings are addressed in
`reviews/clarity-evidence-review.md` and `reviews/clarity-appendix-review.md`.

No experimental result, fitted model, figure data, table value, or citation was
changed. The source equations and human-attested AI-use disclosure are retained.
This is an AI-assisted editorial revision under the user's direction; it records
no additional human verification event beyond the existing attestation. Prior
v08 files are preserved in `history/v08-before-clarity-20260926.zip`. Earlier
review reports below remain records of their respective snapshots.

The rebuilt manuscript retains nine main pages and 23 total pages. Current
structural/visual and archive checks are in `qa/clarity-review.json`,
`qa/clarity-rebuild.json`, and `qa/clarity-final-check.json`.

---

# Title and abstract refinement — 26 September 2026

The current title is **Evaluating JEPA-Inspired Motion Representations through
Geometry and Gait Asymmetry**, adopting the first recommended title. It names
the representation-learning question and the physical quantities used for
evaluation without claiming a demonstrated JEPA advantage or a completed world
model. The title is synchronized in the source, PDF metadata and package indexes.

The abstract now opens with the gait-measurement question and ends with the
contribution of readout comparisons and geometric checks of anatomical identity.
Following the user's concern about emphasis, references to world models/modeling
are reduced from six to three: one abstract mention as future motivation, one
introductory definition, and one discussion statement of the requirements for
future prediction. Its uncertain effect estimate is unchanged, and
its earlier task description explicitly reserves future-state prediction for
subsequent study. The main data description now identifies the selected AMASS
walking candidates. A short editorial pass replaces an aphoristic result sentence
and a compressed discussion triad with interpretations tied to the fitted
outputs, and uses direct descriptions of appendix illustrations.

All experimental results, numbers, figure assets and references are unchanged.
The human-authorship disclosure remains intact. This later editorial update is
AI-assisted under the user's direction; the recorded human attestation is
preserved and is not a new claim of an additional human verification event.
The previous package is preserved in
`history/v08-before-title-abstract-20260926.zip`. Earlier review records below
retain the titles and wording of their own snapshots. The new editorial review
and verification receipts are `reviews/title-abstract-review.md` and
`qa/title-abstract-review.json`; existing adversarial review results continue to
describe their original framing snapshot.

---

# Human authorship and final approval update

Following the authors' explicit confirmation on 26 September 2026, the AI-use
statement now records completed human origination of the research ideas, initial
drafting, final editing, verification of the full text/reported results/references,
and approval of the final manuscript. It continues to disclose the AI-assisted
framing, later drafts and revisions, analysis/code work and supporting reviews.
The author attestation is recorded in reviews/human-authorship-attestation.json.
This does not reclassify AI work as human-only or claim new empirical validation.

The full paper and appendix were reviewed for attribution consistency. The
scientific narrative retains authorial voice; completion and responsibility are
stated explicitly in the disclosure. Package instructions and indexes now agree
with the confirmed status. Earlier review statements about unconfirmed human
verification describe the earlier snapshot, preserved in
history/v08-before-human-attestation-20260926.zip. The updated PDF and both
source bundles retain the same scientific results and title.

---

# In-place v08 revision: JEPA, gait geometry, and research context

This update retains the requested title and the completed experiments. It replaces
v08's current source, PDF, bibliography, indexes and upload bundles; the previous
v08 deliverables remain preserved in
`history/v08-before-jepa-framing-20260926.zip`. Versions 01–07 are unchanged.

The central empirical result remains the relationship between representation
learning, downstream supervision, failure accounting and movement fidelity.
Self-supervised learning and JEPA now motivate that question in the abstract,
introduction and discussion. World-model research explains why a predictive
representation should preserve measurable physical state, while the text clearly
identifies this procedure as offline restoration with privileged synthetic targets.
It does not add a future-dynamics, planning, clinical or purely self-supervised result.

The clinical introduction uses four newly verified primary references: one JEPA/
world-model position paper and three empirical gait studies. Stroke, Parkinson's
disease and knee osteoarthritis motivate preserving side-dependent measurements;
the discussion keeps disease diagnosis and treatment response outside the evidence.
Symmetry is not treated as a universal definition of healthy gait.

The methodology now explains fixed coordinate-array and token indexing, inherited
person splits, source-family variants, input-only normalization, training-only
optimization/calibration and the unadmitted confirmation cohort. It distinguishes
these controls from adaptive development reuse and from the intentional changes
made during resampling and synthetic preparation. Appendix H maps the notebook
questions to the three completed stages and states the directional no-benefit
hypotheses as a retrospective formalization, retaining the recorded two-sided
intervals. Instructional and unexecuted notebook arms remain clearly identified.

Figure 6 adds all 14 participant profiles rather than favorable selected examples.
Direct coordinate restoration has lower response error than noisy observations for
10/14 people; versus zero response it does so for 14/14 under clear images and 1/14
under occlusion. Every fitted seed is shown, observed seed ranges are not labeled
confidence intervals, and the original costs and hierarchical weighting remain.
No pose reconstruction or clinical example was invented.

Independent methods, evidence and editorial reviewers requested and verified
corrections to joint-slot versus anatomical identity, available versus visible
normalization inputs, held conditions versus an additional confirmation cohort,
world-model scope, OpenCap's laboratory references, and sparse appendix pages.
Their detailed dispositions are in `reviews/*-framing-final.md`. The requested
Codex adversarial review completed with **approve / no material findings**; its
unmodified output, structured result and disposition are recorded separately in
`reviews/codex-adversarial-*`.
No automatic score increase is assigned for adding terminology or explanatory prose;
the earlier v08 synthesis 79.33/100 is historical context, not an acceptance forecast.

The current manuscript has nine main pages, disclosure/references on 10–11, and
appendices on 12–23. The unchanged official ICLR 2027 style and initial-submission
limit were reverified against the current author guidelines and fresh official ZIP.
All 23 pages were covered by visual inspection, with grayscale verification for the
new figure. Structural checks confirm 18 resolved citation keys, embedded fonts,
anonymous metadata, preserved historical files and no unresolved references,
missing glyphs or overfull boxes. Both upload/source archives independently rebuilt
all 23 pages with identical text and rasterized appearance; the full source archive
also regenerated all six figures and the numerical tables from its included data.

The Overleaf ZIP contains one root main.tex and only typesetting assets plus
instructions/checksums. Select XeLaTeX after upload. It was tested locally with
Tectonic; no Overleaf account was accessed and no remote compilation is claimed.
Scientific limits remain 14 reused development people, three fitted seeds,
unmatched initial auxiliary influence, no independent clinical/confirmation study,
and incomplete raw poses/checkpoints for end-to-end replication.

---

# Historical record of the initial v08 revision

The record below describes v08 before this in-place framing update, including its
then-current 20-page layout and review scores. Its artifacts are preserved in the
history snapshot; current file links refer to the updated v08 where applicable.

# Version 08 revision and independent review

The title is preserved. This version substantially rewrites the scientific argument and figures using completed experiments only. Versions 01–07 remain unchanged; the package validator verifies 212 historical-file hashes. No source training, new participants, external communication, or submission was performed.

## Strengthened contribution

The study evaluates whether reference-feature prediction yields faithful paired movement restoration. Its strongest empirical result is the tension between improved noisy-pose restoration and useful recovery of movement change: zero response outperforms every pooled neural mean, the delta–endpoint effect is uncertain with reversed coordinate-readout ordering, and the lower response point estimate coexists with worse post hoc geometric assignment. Readout weighting and explicit failure accounting explain which comparisons the completed results can support, without establishing a generic representation failure or a causal mechanism.

The initial claim and supporting contributions were written before manuscript rebuilding in `CONTRIBUTION.md`. The final paper preserves the original feature/measurement question and labels subsequent naming, stratification, sensitivity, and new plotting intervals exploratory. “Declared” does not imply preregistration.

## Consequential changes

1. **Question-led argument.** The abstract centers the zero benchmark and uncertain representation contrast, with intervals for highlighted advantages. Closest skeletal prediction and temporal restoration work now precedes the method. Results address restoration, incremental feature benefit, failure accounting, readout weighting, and anatomical naming. Implementation and complete inventories move to the appendix.
2. **Stronger interpretation of zero response.** All four observation-by-nominal-edit groups and six observation-by-estimator groups are reported for all 16 variants. Direct improves on zero in clear-image groups, but not occluded groups. Nominal edits are not mislabeled as true-magnitude bins; the required per-pair magnitudes are unavailable. For the pooled delta and endpoint variants, even their unconditional successful contributions exceed zero's score, so no nonnegative failure cost can reverse that observed ranking.
3. **Comparison limits beside the results.** Initial weighted auxiliary RMS gradients are 10% of base for endpoint and 0.001303714% for delta. The six new JEPA fits clip 11,999/12,000 pretraining updates. These constrain the fitted-procedure comparison without being treated as persistent influence or a causal explanation. Joint encoder adaptation is distinguished from frozen readouts in a compact table and figure grouping.
4. **Reliability with explicit denominators.** Failure probability, additive error contributions, and conditional successful-output error are separate. Every fixed cost—0, 180, 360, 720 degrees—is reported; the original primary remains 720. The delta–endpoint interval crosses zero at every cost. The roughly 74% cost contribution is arithmetic accounting.
5. **Transparent supervision repair.** Both retained encoders show original, low-scalar, and dense absolute waveform estimates and paired effects. Delta's primary dense-minus-low advantage remains 0.28° [−0.19, 0.75]; the favorable endpoint secondary is not promoted. The descriptive 93% ratio is subordinate to absolute effects and uncertainty. Deterioration is attributed to the tested scalar-plus-geometry package.
6. **A new visual system.** Mathematical state/feature relationships replace generic rounded-box diagrams. Coordinated empirical panels show all families, paired waveform effects, separate reliability diagnostics, original/low/dense repairs, and unconnected naming-condition small multiples. Figures use the actual 5.5-inch width, 8–10 pt typography, stable method colors plus shape/row encoding, vector PDF/SVG, embedded fonts, and grayscale verification. User-specified visual precedents were studied and recorded without importing results or fabricating pose examples.

## Independent review and dispositions

Three independent agents supplied four perspectives: scientific contribution/methods, evidence audit, statistical interpretation, and writing/figures. Their full records are in `reviews/`; evidence and statistics are explicitly separated within their shared panel. These are assistant reviews, not conference peer review or independent empirical replication.

| Substantive objection | Disposition | Remaining boundary |
|---|---|---|
| Contribution could reduce to elementary summary-metric information loss | Rebuilt around observed zero-baseline, readout, reliability, and naming comparisons; removed prose that answered the editorial brief rather than explaining the science | Empirical contribution remains bounded by the completed protocol |
| Zero response was buried; selective subgroup reporting was possible | Central result; all supported selected groups and all 16 variants retained | True per-pair magnitude bins cannot be reconstructed |
| Equal coefficient could imply matched auxiliary influence | Exact magnitude ratios and clipping count moved beside the affected contrast | Requires matched-influence/convergence refits for stronger attribution |
| Direct and frozen results could be read as a controlled representation ranking | Comparison table, row grouping, captions, and discussion state adaptation/exposure differences | No new matched-adaptation experiment |
| “Repair improves them” could imply all eight families were repaired | Abstract explicitly limits the follow-up to two frozen encoders | No all-family repair experiment |
| Scalar loss could receive causal blame for a two-term package | Compound objective is explicit; fixed-geometry repair is described separately | No complete scalar/geometry factorial or unique mechanism |
| Failure rate, penalty share, and conditional error could be confused | Separate axes, formula, conditional denominator, and full-cost inventory | Cost choice remains a scoring decision |
| Post hoc naming could replace an uncertain primary or imply chance/anatomical truth | Original primaries retained, naming status/support defined, no 50% baseline | Independent anatomical confirmation remains absent |
| Appendix notation could mean squared expectation | Square moved inside the expectation; dense pair reduction made explicit | None within the described formula |
| Coordinate-delta coefficient omitted from reconstruction details | Explicit formula and verified 3.79389230231159 value added | End-to-end assets remain incomplete |
| Figure annotation/title and table width defects | Corrected after template-sized prototypes and full-page renders | No remaining inspected collision or overflow |
| Figures drifted into later sections; appendix overflow | Float boundaries and consolidated discussion align questions; appendix ends on page 20 | Table-only penalty-inventory page is intentionally retained |
| Appendix C title lost leading glyphs in rendering | Renamed to “Pooled response-study inventory” and visually rechecked | No scientific change |

## Fixed-rubric comparison

Scores are arithmetic averages of all three panels on the established weights; no favorable panel is selected. Historical scores are retained. The synthesis rises from **75.75 to 79.33/100**, a **3.58-point** increase supported by concrete interpretation, analysis, reconstruction, and visual improvements.

| Dimension | Weight | v07 /10 | v08 /10 |
|---|---:|---:|---:|
| Conference contribution |20%|6.17|6.50|
| Evidence accuracy |20%|9.00|9.00|
| Evaluation rigor |15%|6.00|6.33|
| Scientific insight and positioning |15%|7.50|8.17|
| Reproducibility |10%|8.17|8.83|
| Clarity |10%|8.50|9.00|
| Figures |5%|8.67|9.17|
| Submission fit |5%|8.33|8.33|

Individual totals: methods 79.25→81.25; evidence/statistics 72.75→77.25; editor 75.25→79.50. The methods reviewer keeps contribution/evaluation unchanged; the other panels credit more complete exploratory analysis while retaining the same independent-sample limitations. Evidence accuracy and submission fit plateau. Detailed reasons and machine-readable values are in `reviews/scores.json` and each independent review.

## Completed checks and unresolved research limits

Verified the current official ICLR 2027 style and rules against fresh primary sources; the style ZIP is byte-identical to the retained official copy. The paper has nine main pages, disclosure/references on pages 10–11, and appendices on pages 12–20. Fonts are embedded, the exact title and anonymous metadata are preserved, all citations resolve, and every final page plus every grayscale figure was inspected. No overfull text box, missing glyph, or undefined reference remains. A portable archive rebuild regenerates figures/tables and yields the same text on all 20 PDF pages.

The evidence audit verifies 11 source-export hashes, 16 saved comparison entries, all 96 generated numerical table rows, eight plotted family effects, sign-sensitive contrasts, and eight calibration-bound source hashes. Twenty-one substantive claims are mapped to 68 verified artifacts. New analyses preserve the 14-person/three-seed pairing and are explicitly exploratory.

Writing cannot supply more independent people or seeds, untouched confirmation, natural-video anatomical references, matched influence and convergence experiments, trained external baselines, or the missing raw predictions/checkpoints. The completed package is technically reviewable and substantially stronger than v07; its score is not an acceptance probability, and no submission or administrative eligibility is asserted.
