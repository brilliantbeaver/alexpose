# Independent evidence review of the September progress report

Reviewed on 13 September 2026 against all seven top-level notebooks, the saved R1/E0 results, the OOF prediction records, current implementation, source/fold/cache checks, and the earlier review ledger. The full `docs/08-0913-PROGRESS.MD` draft was read. The clinical motivation paragraphs were also checked against their linked PubMed records. This review did not edit the report or rerun training. It is an independent evidence review alongside the separately requested `codex:adversarial-review` workflow.

## Assessment

No consequential numerical error or unsupported positive empirical conclusion was found. The report distinguishes the old complete R1 result from the newer corrected-objective single-fold demonstrations, retains the missing notebook-05 output, explains source grouping without calling it participant independence, and avoids treating the negative result as a causal proof about JEPA or physiology. Its controls preserve both estimators instead of presenting a post-selected winner as a separately validated model.

The AR-5 entry in `docs/03-0802-PHASE_LEDGER.md` explicitly states that the frozen R1 was not rerun after the center was changed from per-example to per-batch updating. The report's strong chronology statement is therefore supported by a direct project record, not merely inferred from file dates or missing checkpoint metadata.

## Suggested changes

### R1 — Include total window and padding counts in the main report

**Priority: P2, completeness and interpretation.** Section 4 describes overlapping windows and padding, and section 6 identifies fold 0's 395 training windows, but the report does not state the total number of laptop-profile windows or the number of clips requiring padding. These counts have been independently recomputed and help distinguish the number of optimization examples from the number of independent sources.

Suggested addition after the windowing paragraph:

> Across the 47 cached clips, this configuration creates 481 overlapping windows: 92 from normal clips, 167 from MS, and 222 from PD. Three short clips require padding. These windows remain observations from 35 source groups, rather than 481 independent walks.

The three padded clips contain 15, 29, and 26 retained frames, receiving 17, 3, and 6 repeated frames respectively. Listing filenames in the report is optional because the evidence ledger already records them. If these counts are added to the section-3 accounting table, change the class column headings to simply “Normal,” “MS,” and “PD”; a row counting frames or windows should not sit beneath headings that say all entries are “clips / sources.”

**Disposition suggestion:** Add the four-sentence data accounting detail or a compact table row; no new computation is required.

### R2 — Explain the few remaining technical diagnostic terms

**Priority: P2, tutorial clarity.** Most terms are defined, but “temperature-scaled feature distributions,” “running center,” and “teacher–student drift” remain compressed jargon in sections 5 and 9. These are consequential to the centering repair and to the interpretation of the saved diagnostics.

Suggested explanation:

> The target center is a running average of target features that is subtracted before matching them. A temperature controls how concentrated the resulting distribution over feature dimensions is; those dimensions are not the three diagnosis classes. Teacher–student drift here is a parameter comparison: one minus the cosine similarity, averaged across corresponding parameter tensors.

A short explanation should make clear that drift values of 0.0141–0.0171 are not an embedding error, a clinical distance, or a physical movement measure. The report does not currently make any of those incorrect claims; the suggested addition prevents ambiguity for the requested tutorial audience.

**Disposition suggestion:** Add plain definitions near the first uses; avoid introducing more equations unless the report needs an exact reproducibility appendix.

### R3 — Narrow the wording of the organizing null to the measured endpoint

**Priority: P3, conceptual precision.** Section 2 defines a null of “adds no predictive information,” then operationalizes it with the difference in macro-F1 between two particular training/probe pipelines. Predictive information in a representation is a broader claim than the performance of one readout, seed, and dataset; a representation could contain information that its current probe fails to use.

Suggested replacement:

> A corresponding operational null is that this training-and-probe procedure does not improve held-out macro-F1 over the specified simpler comparison.

This keeps the report's organizing hypothesis aligned with its explicit score while preserving the broader geometry hypothesis as motivation. The draft already correctly states that no preregistered formal test was completed.

**Disposition suggestion:** Prefer endpoint-specific language; this is a precision improvement rather than a correction to any reported statistic.

### R4 — Make the geometric baseline's range and validity rule explicit

**Priority: P3, method explanation.** Section 8 identifies means, mean-angle asymmetries, and range fields correctly, including the duplicated left ankle range. A reader would benefit from knowing that “range” is the maximum minus minimum over valid angles and that an angle is screened using the three landmark confidences. This connects the baseline to measurement quality and helps explain why large angle ranges can reflect outliers.

Suggested addition:

> Each range is the maximum minus minimum of the valid angles in a clip. The angle calculator screens triplets using their landmark confidences, so pose quality affects which measurements enter the summary, and a surviving outlier can enlarge the range.

The implementation uses the geometric mean of the three confidences against a configured threshold. That detail belongs in a method note if included; do not substitute an invented fixed threshold or describe all three confidences as separately required to exceed it.

**Disposition suggestion:** Add one sentence and a direct relative link to the upstream constructor or the method-evidence record.

### R5 — Optional retained notebook-01 example and model configuration detail

**Priority: P3, optional completeness.** The report covers the consequential outputs. It could also identify notebook 01's 155-frame demonstration with 61% fully detected frames and notebook 03's approximately 0.40 million trainable parameters. The example detection fraction must be explicitly scoped to that one normal clip; it is not a dataset average. The printed generic epoch settings are legacy configuration text and should remain omitted because actual update budgets govern these retained runs.

**Disposition suggestion:** Include only if needed for “all latest results” coverage; these details should not displace the evidence limitations or the complete current score tables.

## Claims verified without changes

| Claim group | Review result |
|---|---|
| Raw/cached cohort | 49 clips/37 sources before exclusions, 47/35 afterward; label counts and 8,693 frames agree. |
| Five-fold partitions | All clip/source counts and class denominators agree with the frozen registry. |
| Tokenization | Four frames × three channels = 12 values; eight blocks × 33 joints = 264 tokens; learned width 96. |
| Mask demonstration | Six distinct masks; 512-mask minima round to 0.74 visible and 0.81 targeted at least once over time; target-token mean rounds to 0.63. |
| Notebook 03 | 800 updates, 37 training clips/29 sources/395 windows, first-five versus last-five mean losses 4.990→1.112, final rank 9.5. |
| Notebook 04 | 400 updates from weights with restarted state; 0.600→0.600 on 10 clips/6 sources; rank 10.4 afterward. |
| Fixed readout | 161/264 token positions, all 33 joints covered over some time blocks. |
| Notebook 05 | Retained error before embedding extraction; no t-SNE, UMAP, or silhouette output; prerequisite checkpoints now exist. |
| R1 headline | F1 0.438 versus 0.667, accuracy 0.447 versus 0.660, correct clips 21/47 versus 31/47, PD recall 4/17 versus 10/17. |
| R1 score difference | S-JEPA minus RF = −0.2281703915, correctly rounded to −0.228. |
| Per-class metrics and confusion | Every displayed value agrees with the frozen artifacts and independent OOF rescoring. |
| Diagnostic ranges | Effective rank 7.71–9.51, standard deviation 0.566–0.643, loss 1.135–1.584, parameter drift 0.0141–0.0171 are correctly rounded. |
| Live fold-0 demonstration | 500 updates, S-JEPA 0.644 versus RF 0.915, rank 8.9; frozen fold-0 RF 0.915 and S-JEPA 0.467 are correctly distinguished. |
| Controls | All ten classifier scores are correctly rounded; largest listed point estimate is mean/std pose logistic regression 0.703. Its RF difference 0.036 is correct. |
| Uncertainty | No formal interval or test is attributed to the current retained notebooks; fold SD is distinguished from a confidence interval. |
| Chance reference | One-third is correctly rejected as a universal macro-F1 null, and correctly identified as expected accuracy for uniform random guessing. |
| Physical interpretation | No completed symmetry-mechanism, future-state-model, or diagnostic-validity claim is made. |

The introduction's Parkinson's claim is supported by [Lewek et al.](https://pubmed.ncbi.nlm.nih.gov/19945285/): their small early-PD cohort showed greater arm-swing asymmetry while arm-swing magnitude did not significantly distinguish groups. The MS coordination motivation is supported by [Pau et al.](https://pubmed.ncbi.nlm.nih.gov/35305428/), whose three-dimensional gait study found altered joint-pair coordination associated with disability. The report appropriately treats both as external motivation and does not transfer their clinical findings to this collection.

## Scope of this review

This pass checked the report's prose, tables, linked evidence, and stated figure captions. It did not inspect the new SVGs visually or assess page rendering. The separate slide/graphics checks and the adversarial-review revision record should cover those deliverables. No severe issue remains in the checked numerical claims; the suggestions above address completeness, endpoint precision, and tutorial comprehension.

The lead reports that R1 and R2 were applied, including moving the mixed-unit frame-count row into prose, adding 481 windows and three padded clips, and defining center, temperature, and drift. The optional notebook-01 example from R5 was also added. R3 and R4 remain editorial suggestions for the final revision; neither changes a verified result.
