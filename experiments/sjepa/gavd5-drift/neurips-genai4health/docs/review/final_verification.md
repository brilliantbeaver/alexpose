# Final verification and review disposition

Completed September 6, 2026 for *Before Health Agents Interpret Movement: Lessons from a Gait Representation Study*. This record supersedes the September 5 production summary. It records completed checks, not clinical validation, submission approval, or an acceptance prediction.

## Delivered artifacts

| Artifact | Current status |
|---|---|
| Main position paper | 8 PDF pages: 5 main text, 1 references, 2 appendix |
| Companion extended abstract | 3 PDF pages: 2 main text, 1 references; no separate workshop abstract track is advertised |
| Manuscript formats | Canonical LaTeX, generated Markdown, and compiled PDF for both documents |
| Anonymous style | Official NeurIPS 2026 style without options; blank PDF author metadata |
| Figure | New editable vector comparison of recording weights, with PDF and PNG copies and a reproducible generator |
| Supporting material | Revision strategy, submission metadata, masking-candidate assessment, notebook reviews, numerical supplement, and build instructions |
| Source bundles | Separate main-paper and companion bundles; numerical records in a separate bundle |

All 74 source files in the protected BrainBody snapshot match their contents at the start of this revision, including the notebooks and selected manuscript, figure, and build sources. Pre-existing worktree changes outside the GenAI4Health output were preserved. No numerical CSV values were edited.

## Evidence checks and their scope

The current checks reconstruct retained outputs and cohort records. They do not rerun model training, inference, pose extraction, or video decoding. In particular, the original fold checkpoints, embedding caches, and complete split registry are not available in this checkout. The older checkpoint-backed verification remains historical; it must not be cited as a newly repeated check. The current machine-readable result is `evidence/current_verification.json`.

Completed checks establish that:

- All 60 saved test prediction rows—three classifiers on the same 20 videos—agree with the aliased numerical supplement.
- Predictions reproduce 10/20 correct videos for pose summaries, 6/20 for learned features, and 6/20 for landmark availability. Balanced accuracies round to 0.44, 0.26, and 0.25, respectively.
- The five retained normal-validation records agree across their available copies and reproduce mean cosine 0.89 with equal clip weights and 0.70 with equal video weights. The same values and clip counts enter both calculations.
- The acquisition and quality ledgers reconstruct 666 clips from 103 videos initially and 639 clips from 97 videos after exclusions. Retained video roles reconstruct 377/131/131 clips and 59/18/20 videos for training/validation/test, with one role per upload.
- The portable numerical verifier passes using only Python's standard library. It explicitly normalizes line endings in memory because the historical byte records used CRLF and the current checkout uses LF. This resolves a portability difference without changing numerical records.

The full artifact-dependent verification is preserved separately and is not the default build. Requesting it without its required inputs fails explicitly rather than treating absent files as verified.

## Production and consistency checks

The build uses Tectonic, Pandoc, Poppler, and the existing Python environment. Both PDFs satisfy their chosen main-text page budgets without changing template margins or shrinking body text. Automated checks report embedded fonts, blank author metadata, no local author identifiers, no out-of-page text, no overfull or underfull boxes, no missing glyphs, and resolved citations and references. A dependency-source warning does not appear as a replacement character in either PDF.

Rendered previews of every page were reviewed, with additional full-size inspection of the main figure, main results/proposal pages, and companion results page. No visible clipping or overlap was found. The figure is vector-based and uses readable labels rather than a dense multi-panel diagnostic.

Markdown copies are generated from canonical LaTeX. Cell-by-cell comparison confirms agreement for all 42 main-paper table cells and 12 companion table cells. A conversion defect that dropped the leading “64” in one cell was corrected; the recording-unit example now contains the complete count in every format.

All three ZIP files pass integrity and allowlist checks. The main and companion source bundles exclude internal review documents and numerical source records. The numerical bundle remains a local review artifact pending the authors' release decision. Bundle status notices do not imply upload or approval.

The automated renderer removed one stale generated preview, `page_09.png`, after the main paper became eight pages. The old preview is recoverable from version control; no research data was removed.

## Adversarial review and corrections

Independent notebook and literature reviews challenged the central claims, prior-art treatment, numerical scope, and workshop fit. Resolved issues include:

| Issue | Final treatment |
|---|---|
| Weighting contrast could imply patient deterioration or lost predictive function | The same clips and recorded similarities are held fixed; only their weights change. Neither average measures a health outcome. |
| Cosine described as coordinate agreement | Defined as directional agreement, without a magnitude or retained-information interpretation. |
| Classifier result could imply JEPA is generally inferior | Limited to one recorded split and initialization, with missing untrained control and selection/test aggregation mismatch disclosed. |
| Quality gate confused with encoder validity | Visibility-based clip eligibility separated from the additional finite-coordinate validity requirement. |
| Synthetic masking setting treated as trained-run metadata | Unrecorded nominal masking setting removed as a verified runtime fact. |
| Generic documentation proposal presented as novel validation framework | Model Cards, V3, and DECIDE-AI credited; the evidence record is a specific proposed application. |
| Health agent, forecasting, or clinical benefit implied | These remain explicitly untested; a prospective interpretation evaluation is described. |
| Laterality or incomplete later experiments could inflate the result | Excluded from both manuscripts. |
| License treated as institutional authorization | MIT annotation licensing acknowledged separately from source-video conditions and unresolved institutional determination. |

The masking-candidate assessment also passed an independent prior-art review. It identifies a promising hypothesis but no completed trained comparison of masking policies. No masking benefit, neurological latent-space validity, or new masking experiment is claimed.

## Remaining author decisions and scientific limits

The evidence still comes from one split and initialization, with unknown cross-upload person identity, incomplete execution metadata, uncalibrated pose geometry, different diagnostic preprocessing, and inconsistent classifier aggregation. These require new experiments or additional records to resolve. They cannot be repaired through wording.

The main acceptance risks are incremental novelty and the prospective connection to generative health systems. The proposed evidence record has not been implemented or evaluated. A matched interpretation experiment would strengthen that contribution; a masking paper would require a separate controlled training comparison.

Before upload, the authors must confirm authorship, institutional and data-use requirements, source-record release decisions, and overlap with the BrainBody submission. The shared classification result is not independent evidence or a new dataset. The current workshop call and both venues' overlap policies should be checked at submission.

No clinical approval, consent, external release, deployment, new training run, or workshop submission was performed or fabricated during this revision.
