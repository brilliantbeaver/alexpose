# Version 05: independent evidence and statistical review

Reviewed artifacts: `paper-v05.tex`, the new naming figure and its editable source, `laterality-naming-provenance.json`, and the independently reconstructed assignment contrasts. All source empirical data remain unchanged. The parent build reports nine main pages.

## Evidence auditor

The new naming figure correctly plots direct/base, delta/change, and endpoint/change for all three input naming conditions. Its nine values match the audited condition-person table. The figure has a 0-100% axis, separate color/marker/line encodings, readable eight-point labels at 5.5 by 1.9 inches, no invented uncertainty, and no chance line. It was visually checked at its output size. The caption identifies the geometric categories and thresholds, shared population, pooled conditions, and post hoc origin. This is a stronger use of the existing evidence than the replaced repair plot because it directly addresses the requested laterality focus; repair means and primary uncertainty remain in the text and primary-effect figure.

The added source-rendered person boxes and once-fitted cameras match the preparation implementation. They disclose an important boundary: detector uncertainty is outside the experiment. The per-leg 180-degree failure cost matches `_limb_metrics` in `response_evaluation.py`. Removal of the probe paragraph from the nine-page main narrative does not create an unsupported claim: the paper does not infer successful deployment-feature probing or a unique information-loss mechanism, and the diagnostic evidence remains documented in the audit.

| ID | Severity | Evidence/disposition | Remaining action or limitation | Status at v05 |
|---|---|---|---|---|
| E01-E07 | Prior substantive objections | All earlier geometry, side-scope, figure-source, denominator, probe-context, and paired-training corrections remain accounted for. | Raw prediction and external validation limits unchanged. | Resolved or bounded as recorded |
| E08 | Prior editorial opportunity | Naming conditions now have a scientific figure with verified means and provenance. | Main figure is descriptive and uses the same development data. | Resolved |
| E09 | Minor wording | Subsection “Response preservation does not establish geometric side recovery” can sound as though preservation was achieved. | Prefer “A lower response score does not establish geometric side recovery.” | No new evidence required. | Open editorial suggestion |
| E10 | Minor wording | Naming caption calls delta and endpoint “both JEPA readouts,” although both have the same scalar readout objective and differ in pretraining. | Prefer “both frozen JEPA variants.” | No scientific conclusion changes. | Open editorial suggestion |

The scientific content has no unresolved material evidence misstatement in this review. The remaining two suggestions improve the local precision of otherwise qualified prose.

## Statistical reviewer

The new assignment interval has the correct direction and units. The independently reconstructed endpoint-minus-delta improvement is -0.7688 percentage points with crossed interval [-1.5279,-0.1738]. Negating the estimate and reversing the interval endpoints gives delta excess failure 0.7688 [0.1738,1.5279], correctly rounded in v05 to 0.77 [0.17,1.53]. Multiplication by 100 converts the original rate units to percentage points. The paragraph labels the contrast exploratory and unadjusted and explicitly preserves the response primary; it does not manufacture a new confirmatory result.

This interval supports a descriptive discrepancy between the two outcomes in the observed panel. It does not demonstrate general harm from difference pretraining, resolve the uncertain response comparison, or establish that the two endpoints disagree significantly as a joint multivariate hypothesis. The paper makes none of those stronger claims.

| ID | Severity | Evidence/disposition | Residual limitation | Status at v05 |
|---|---|---|---|---|
| S01-S03 | Earlier reporting objections | Post hoc selection, exact level weights, and per-leg counterexample retained. | Secondary analyses remain selected and unadjusted. | Resolved in reporting |
| S04 | Substantial empirical limitation | Same 14 people, three seeds, adaptive development, limited optimization exploration and no external model baseline. | New experiments and prespecified confirmation are still needed. | Bounded, unchanged |
| S05 | Numerical check | Excess-failure estimate and interval have correct negation, endpoint order and percentage-point units. | The interval describes the sampled development procedure. | Passed |

Means in the new naming plot are averaged over seeds within each person, then equally over people, after the audited 4:1 endpoint reaggregation. No error bars are supplied; this is appropriate for a descriptive figure whose selected contrast is separately reported with its inferential limits. The axes and caption do not imply 50% is a chance threshold.

## Fixed rubric

| Dimension | Weight | Score /10 | Reason and remaining weakness |
|---|---:|---:|---|
| Conference relevance and contribution | 20% | 6.0 | Same limited but useful representation-evaluation study; no demonstrated incremental benefit or independent validation. |
| Claim accuracy and evidence support | 20% | 9.0 | Verified new assignment interval and source-rendering details; only minor precision edits suggested. |
| Evaluation and statistical rigor | 15% | 5.0 | Fundamental evidence limitations are unchanged. |
| Scientific insight and positioning | 15% | 7.0 | The plot and correctly qualified cross-metric counterexample now make the laterality argument concrete. |
| Reproducibility | 10% | 8.0 | Naming plot has editable source, precise aggregation and source hashes; raw-data reproduction remains unavailable. |
| Clarity and narrative | 10% | 8.0 | Stronger visual focus, with two minor labels to tighten and substantial technical detail compressed into nine pages. |
| Figures | 5% | 8.5 | New figure answers a relevant question with exact means and accessible encodings; no verified raw reconstruction panel is available. |
| Submission fit | 5% | 8.0 | Nine-page main-paper verification retained, with distinct reference/disclosure matter and bounded scope. |
| **Weighted total** | **100%** | **72.25/100** | Gain comes from a clearer scientific argument and figure, not new empirical support. |

For v06, tighten E09/E10 and preserve the immutable naming-figure dependency. No stronger statistical claim is recommended.
