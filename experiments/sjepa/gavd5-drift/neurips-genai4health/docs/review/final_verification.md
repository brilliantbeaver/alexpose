# Final verification and review disposition

Completed September 5, 2026. This record describes completed checks; it does not certify clinical validity or acceptance.

## Delivered artifacts

| Artifact | Verified status |
|---|---|
| Main position paper | 9 PDF pages: 5 main text, 1 references, 3 appendix |
| Companion extended abstract | 3 PDF pages: 2 main text, 1 references |
| Anonymous style | `\usepackage{neurips_2026}` with no options; blank PDF author metadata |
| Main-source bundle | Explicit allowlist of 11 manuscript, style, figure, and numerical files; excludes companion and internal audits |
| Numerical bundle | Five files; 20 aliased test sources for each readout, five normal-validation source summaries, hashes, verifier |
| Reader copies | Generated Markdown from canonical LaTeX with the same claims and numbers |
| Original research package | No source notebooks or original drafts edited by this work |

A final byte-level comparison detected concurrent updates to notebooks 00, 06, 07, 08, and 09. Their normalized cell source text is unchanged from the initial extraction; differences are in other notebook content or serialization. Every input hash in the manuscript evidence manifest remains identical. The initial hashes and later comparison are retained in `reproducibility/notebook_inventory.json` and `review/concurrent_source_updates.json`. This resolves the consistency alert without overwriting another process's work or claiming that all notebook bytes stayed frozen.

Every page of both final PDFs was visually inspected, including figures, equations, tables, references, and page boundaries. The figure labels were enlarged, overlapping annotations were separated, and a floating appendix table was moved into the main evidence summary to remove an orphan heading and isolated final page. The final PDFs have no observed clipping, overlapping content, broken citations, or replacement glyphs. Automated checks found no text outside page bounds, no author-specific path strings, no unresolved citation markers, and blank author metadata.

Tectonic 0.17.0 built the required LaTeX style. PyMuPDF rendered page images because Poppler was not available on PATH. Tectonic reports benign font-configuration/lineno dependency warnings and one underfull vertical box; the produced PDFs were checked directly and contain no corresponding visible missing glyph or overflow. No template margin or body-font reductions were used.

## Numerical checks

- The core verifier recomputes the source-manifest and split-registry digests, checks final/stage checkpoint bytes, verifies role separation, and counts trained tensor parameters.
- Saved source predictions exactly reproduce all three methods' accuracy, balanced accuracy, and macro-F1.
- All six current checkpoint hashes and the normal-anchor embedding caches agree with their recorded lineage.
- The source-weighting contrast is recomputed from the same stored normal-validation vectors; the mean of five source means and the clip-weighted mean reproduce 0.701058 and 0.889061 to float32 tolerance.
- Cohort and split sums agree with the manuscript tables.
- The standalone numerical supplement reproduces the headline calculations using only Python's standard library.
- Both ZIP archives pass integrity checks and contain only allowed files. Public-facing text files were screened for local paths and author-specific identifiers. This is an author-anonymity check, not a guarantee of person anonymity.

## Adversarial findings resolved

| Finding | Disposition |
|---|---|
| Draft loss/architecture described an earlier implementation | Corrected to current checkpoint and notebook 04 |
| “Label-free” omitted annotation-informed scheduling | Narrowed to the loss and disclosed curriculum |
| Readout refit and aggregation mismatch omitted | Both stated in main paper and companion; source procedure preserved |
| Normal-anchor diagnostic preprocessing and pooling unclear | Disclosed short-gap interpolation, 12-joint pooling, and trained normal-only reference |
| Cosine could imply functional or clinical retention | Named embedding cosine and separated from function throughout |
| Local split registry could imply external preregistration | Wording corrected; limits of historical test secrecy disclosed |
| Rounded difference appeared inconsistent | Stated 0.148 is calculated from unrounded scores |
| Variance estimator unspecified | Denominators B and B−1 stated |
| V-JEPA 2 first-author spelling | Corrected to the publication's Mido Assran |
| Reviewers could not access local numerical evidence | Added aliased, self-contained numerical supplement |
| Manuscript figures too small / isolated appendix table | Re-rendered and visually corrected |
| Historical, simulated, or absent results risked promotion | Excluded from current empirical claims; statuses retained in audits |

The final independent editorial review records that all material issues it identified are resolved within the chosen position-paper scope. Its original comments remain as an audit trail, followed by the final resolution section.

## Limitations deliberately not “fixed” by wording

The study still has one fold/seed, no verified person identity, inconsistent validation/test readout aggregation, legacy crop geometry, partial runtime metadata, different diagnostic preprocessing, and no current clinical, causal-forecasting, or effective-consolidation evaluation. These are visible limitations, not completed repairs. Remedying them requires a separately versioned experiment or new data. The provided papers do not claim otherwise.

No institutional ethics determination, consent, clinical collaborators, author list, deployment, or workshop submission was fabricated. The authors must settle the actual data-use/ethics status, authorship, and any concurrent-submission obligations before upload. The companion abstract is not a separately advertised workshop track.
