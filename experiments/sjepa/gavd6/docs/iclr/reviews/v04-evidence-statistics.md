# Version 04: independent evidence and statistical review

Reviewed artifacts: `paper-v04.tex`, changes to `scripts/build_figures_v04.py`, normalization and optimizer implementation, source empirical exports, and previous independent numerical checks. The main text's nine-page fit was verified by the parent build process; this review does not substitute for a visual page audit. No new source experiment was conducted.

## Evidence auditor

The normalization equation matches `normalize_batch`: coordinate-wise median origin, isotropic norm of the coordinate-wise P95-P5 span, observed context excluding artificial hidden tokens, and fixed zero-origin/unit-scale fallback for inadequate support or invalid/tiny scale. The added AdamW, weight decay, warmup, cosine schedule, and gradient clipping constants match the implementation. The dense calibration expression correctly matches gradient energy with the low scalar coefficient, rather than claiming gradients remain matched through training.

The input/reference distinction remains clear. The restored coordinate network sees observed estimates; the clean projected coordinates are privileged teacher targets. The model-flow figure now labels a single-state forward path and its caption explains that paired readout fitting compares two outputs. This resolves E07 without implying paired states are required at deployment.

The probe paragraph now identifies 333 pairs from 112 encoder-training people, resolving E06. I independently inspected all 21 exported diagnostic JSONs: every deployment-encoder probe has larger MSE than its zero-response baseline, all use 333 pairs and 112 people, and delta's encoder/teacher MSEs are 47.868394/6.708453 squared degrees. The displayed 47.87/6.71 values are correct. The paper retains the limits of a single fixed linear probe.

| ID | Severity | Disposition and evidence | Residual limitation | Status at v04 |
|---|---|---|---|---|
| E01/E02/E04/E05 | Earlier major/moderate | Correct mirror geometry, operational side scope, assignment rule, and immutable version references remain intact. | No anatomical validation or direct equivariance test. | Resolved |
| E03 | Earlier moderate | Primary figure effects and interval types remain sourced from hashed JSON. | End-to-end raw-data reproduction unavailable. | Resolved |
| E06 | Earlier minor | Probe text now names the 112 training people and encoder exposure. | A failed fixed probe does not establish information absence. | Resolved |
| E07 | Earlier moderate | Single-state forward path is distinguished from paired fitting losses in figure and caption. | None within this diagram's scope. | Resolved |
| E08 | Editorial opportunity | Laterality percentages are accurate but still occupy a late paragraph, while figures emphasize scalar and waveform losses. | A descriptive naming-condition plot would foreground the requested laterality argument without new experiments. | Optional revision |

No unresolved material numerical or evidence misstatement was found. The new technical details do not justify a stronger empirical conclusion, and v04 does not make one.

## Statistical reviewer

The text now gives the correct 4:1 nonheld/held weighting for endpoint metrics and 2:1 for nonzero responses, resolving S02. These ratios reproduce the published condition-averaged metrics under complete reference support; direction accuracy remains governed by its separate eligibility rule. The three declared primary comparisons, person/seed units, distinct repair interval, unchanged cohort, adaptive development, and unadjusted laterality analysis all remain accurately described.

The paper avoids three common errors that would have materially overstated the results: it does not equate the 74% failure contribution with improved successful kinematics, does not equate the 93% waveform recovery ratio with exclusive causal attribution, and does not use the favorable endpoint-repair secondary comparison to replace the unresolved delta primary.

| ID | Severity | Disposition | Residual limitation | Status at v04 |
|---|---|---|---|---|
| S01 | Earlier major | Post hoc, exploratory, unadjusted laterality label retained. | Needs independent prespecified confirmation. | Resolved in reporting |
| S02 | Earlier moderate | Exact endpoint and response weights are now stated. | Direction uses different eligibility; avoid generic reweighting claims. | Resolved |
| S03 | Earlier moderate | Per-leg excursion counterexample retained; no universal direct-model dominance claimed. | Multiple objectives remain unresolved. | Resolved |
| S04 | Substantial empirical limitation | Fourteen reused people, three fitted seeds, no confirmation, limited budget/weight sweep and no external neural baseline. | New experiments needed; manuscript revision cannot remove this weakness. | Bounded, unchanged |

The highest-value next figure is a three-condition plot of geometric assignment failure for direct/base, delta/scalar, and endpoint/scalar. It should show correct/global/temporary naming, a 0-100% axis, distinct color/marker combinations, and no chance line. Person means averaged across the same three seeds are descriptive points; the caption must disclose the post hoc origin, shared population, and failure definition. The plot asks whether the modest scalar-response point advantage corresponds to better geometric naming. It must not claim an anatomical classifier or a controlled attribution of direct-versus-frozen differences.

## Fixed rubric

| Dimension | Weight | Score /10 | Reason and remaining weakness |
|---|---:|---:|---|
| Conference relevance and contribution | 20% | 6.0 | Same bounded controlled evaluation; incremental feature-prediction benefit remains unestablished. |
| Claim accuracy and evidence support | 20% | 9.0 | No unresolved material misstatement found; calibrated scope and counterexamples retained. |
| Evaluation and statistical rigor | 15% | 5.0 | Substantial underlying population, seed, adaptivity, and external-baseline limitations remain. |
| Scientific insight and positioning | 15% | 6.5 | Consistent scalar/trajectory/naming distinction; the cause and generality of the failures are untested. |
| Reproducibility | 10% | 8.0 | Weights, calibration, normalization, optimizer, and source-driven interval reconstruction are now explicit; raw artifacts remain absent. |
| Clarity and narrative | 10% | 8.0 | Definitions and single-state/paired-training distinction are clearer; laterality could be more visually central. |
| Figures | 5% | 8.0 | Diagram ambiguity resolved and quantitative figures trace actual exports; naming currently lacks a plot. |
| Submission fit | 5% | 8.0 | Official template and reported nine-main-page verification, with visual QA still separately required. |
| **Weighted total** | **100%** | **71.25/100** | Gains reflect documentation and resolved diagram issues; statistical rigor remains unchanged. |

No empirical claim should be strengthened in v05. A naming-condition figure can make the existing argument more accessible while leaving its inferential status unchanged.
