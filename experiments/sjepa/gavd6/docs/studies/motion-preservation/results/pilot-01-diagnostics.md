# Pilot 01: projection, reconstruction and flow diagnostics

Analysis of the completed [notebook 06](../../../../notebook_runs/motion-preservation/run-06/06_diagnose_repair_mechanism.ipynb), executed on 14 September 2026. Cell references count all notebook cells starting at one. The [extracted tables and provenance](pilot-01/diagnostics/provenance.json) preserve the displayed evidence without changing the notebook or experiment.

**The current repair pipeline has two distinct problems: the fixed-length projection damages already accurate observations, and the MoMask reconstruction introduces much larger errors before that additional projection. Video evidence changes in the intended direction for most matched pairs, but does not yet reliably favor the appropriate candidate in both explanations. Diagnose these mechanisms before another gate-training or scaling run.**

This extends the [analysis of notebooks 00–04](pilot-01-analysis.md). That earlier analysis could not distinguish event suppression from overshoot or inspect flow at the event locations. Notebook 06 supplies those missing comparisons. It does not provide a new trained-gate result or overturn the existing development stop.

**What the notebook actually establishes.** Execution completed in real mode on the existing `pilot-01` cache. There are 64 calibration cases from eight people and 64 development cases from eight different people. The event-plus-noise table has eight cases per role, one per person. The 128 constructed cases are not 128 independent participants. All 128 cases have an available conversion-only bridge. Final data was not evaluated; models and calibration were unchanged. Sources: notebook cell 5 and [inventory](pilot-01/diagnostics/inventory.csv).

The local result bundle contains the executed notebook, including seven displayed tables and ten plots. The original per-case diagnostic CSVs, full strength curves, bone-length arrays and selected trace NPZ files remain at the recorded HAIC path. Numerical statements below use notebook display precision, generally six decimals. Approximate graph readings are identified as such; no confidence intervals have been reconstructed from aggregate tables. Execution metadata does not record the exact source revision; reading the current implementation supports interpretation but does not reproduce the HAIC computation.

**The error scale explains why the repair target was missed.** The following comparison uses only the factorial cases containing both a real event and tracking noise. Observed-joint MSE is expressed in square centimeters by multiplying the displayed square-meter values by 10,000. It includes damage to previously accurate observed joints.

| Diagnostic candidate | Calibration MSE, cm² | Development MSE, cm² | Calibration retention | Development retention |
| --- | ---: | ---: | ---: | ---: |
| Raw input | 0.27 | 0.27 | 0.906 | 0.855 |
| Projected raw | 0.41 | 0.40 | 0.859 | 0.786 |
| Conversion only, without additional projection | 0.54 | 0.55 | 0.846 | 0.768 |
| MoMask, without additional projection | 170.94 | 20.94 | 0.282 | 0.333 |
| Truth-informed mixture, without projection | 0.19 | 0.17 | 0.981 | 0.944 |
| Same truth-informed mixture, projected | 0.34 | 0.28 | 0.919 | 0.882 |

Source: notebook cell 7, [event-plus-noise scores](pilot-01/diagnostics/event_noise_scores.csv). The two mixture rows use hidden reference truth and are diagnostic controls, not practical competitors. These rows also cover a narrower condition than notebook 04's overall noisy-case average, so their numbers should not be substituted for that official summary.

The square roots of the displayed aggregate MSEs correspond to approximately 5.2 mm RMS distance for raw input versus 130.7 mm and 45.8 mm for MoMask on calibration and development respectively. These are derived RMS distances, not mean per-joint Euclidean errors. The primary noise-removal metric averages per-case ratios within people; it is not one minus the ratio of the aggregate MSE columns above.

**Projection is directly implicated, including at zero strength.** Projected raw has person-averaged error removal of **−78.1% on calibration and −53.9% on development**. In the displayed zero-strength curves, the unprojected branch returns raw input with zero error removal; the projected branch already has this penalty. This establishes that accepting no candidate correction does not make the original projected pipeline an identity operation. Sources: notebook cell 11 and [zero-strength scores](pilot-01/diagnostics/zero_strength_scores.csv).

The truth-informed unprojected mixture removes **27.4% and 30.0%** of error on these event-plus-noise groups. Applying projection to that same mixture changes removal to **−52.5% and −12.7%**. The 25% target is therefore achievable by this privileged unprojected control in the reported groups. Its margin is modest, and a learned method would have to approach that control closely. This does not establish that a deployable method, every person, or the full calibration cohort can reach the target.

The oracle chooses the pointwise closest location on the raw-to-prior line segment before projection. It bounds this specific unprojected squared-error family. Its projected score is not an upper bound on all possible projected repairs; optimizing before a nonlinear projection is a different problem from optimizing the final projected result.

Length-estimation bias contributes but is not the complete explanation. Notebook cell 13 shows approximate mean absolute biases of 3–5 mm at the left ankle and about 6.5 mm at the left foot, with most other mean biases near zero. The reference itself varies over time: several segments have within-clip standard deviations around 1–4 mm. These are graph readings, not exact values from a locally available bone-length CSV.

Replacing the estimated lengths with reference-derived median lengths still gives raw-input removal of **−75.5% and −57.1%**. On clean, error-free inputs, even projection with reference median lengths introduces approximately 0.13 and 0.11 cm² MSE. Thus better length estimates alone cannot make this projection harmless. The fixed-length assumption and the algorithm that preserves directions while rebuilding chains also need scrutiny. The notebook does not quantify the relative contribution of each mechanism.

Gap handling is a second, separate zero-strength problem. `legacy_prior_strength0` and `projected_raw` agree on the displayed observed-joint scores, but their completion errors differ sharply. The legacy row has completion MSE **0.001105 versus 0.000034 m²** on calibration and **0.003500 versus 0.000155 m²** on development. Candidate values at missing positions remain active even at zero strength. A true no-change endpoint must specify how it treats the already imputed raw cache as well as observed positions.

**The MoMask path damages clean motion as well as suppressing events.** In the no-event, no-noise controls, raw error is zero. The unprojected MoMask path introduces **167.20 cm² on calibration and 26.88 cm² on development**, approximately 129 and 52 mm derived RMS distance. This rules out an explanation confined to difficult injected tracking errors. The conversion-only control introduces much smaller errors, 0.26 and 0.23 cm², although those are still important relative to the small raw-error budget. Source: notebook cell 9 and [clean controls](pilot-01/diagnostics/clean_scores.csv). Undefined noise-removal ratios and retention in these clean controls are expected: raw error is zero and no event is present. Absolute error remains interpretable.

In event-plus-noise cases, MoMask's MSE on previously accurate joints is **171.30 cm² on calibration and 18.34 cm² on development**. Its damage extends well beyond the localized corrupted positions. The six selected development traces all show substantial errors on previously accurate joints. In KIT::348 and ACCAD::Male2, these errors grow markedly over time. That pattern motivates a decomposition into root-trajectory, heading and root-relative pose error; the plots alone cannot establish which component caused it.

Additional projection changes MoMask MSE only from 170.94 to 171.88 cm² on calibration and 20.94 to 21.21 cm² on development. It cannot account for the dominant MoMask reconstruction error. However, “unprojected” means without the study's additional median-length projection: the HumanML3D inverse already restores fixed source bone lengths. Comparison with the bridge isolates the added learned reconstruction path, but does not prove that every normalization, temporal-alignment or reconstruction-specific integration choice is correct. See [pretrained model bridge](../../../../src/gavd6_sjepa/research_directions/motion_preservation/pretrained_models.py).

Signed event error resolves an uncertainty in the earlier report. MoMask has mean signed event errors **−0.718 and −0.667**, where zero preserves the descriptor and minus one returns to the unedited descriptor. Alongside retention of 0.282 and 0.333, and suppression in all six displayed foot-height traces, this supports genuine descriptor suppression in this pilot. It does not establish selective event erasure by an otherwise useful repair model: clean-motion reconstruction is also poor. Foot height is one coordinate, and the examples are six alphabetically selected development cases rather than a random sample or a calibration trace audit.

**The flow evidence is present, but its decision is inconsistent.** All 16 matched pairs pass the diagnostic's input/reference equality checks and have supported flow contrasts. Each role has eight pairs from eight people.

| Matched-pair diagnostic | Calibration | Development |
| --- | ---: | ---: |
| Available event-flow coverage, mean | 99.32% | 99.32% |
| Mean gap in real-event video, pixels | +0.0690 | +0.0305 |
| Mean gap in tracking-failure video, pixels | −0.0594 | −0.1171 |
| Mean real-minus-failure gap, pixels | +0.1284 | +0.1476 |
| Pairs with a favorable relative gap change | 8/8 | 7/8 |
| Pairs favoring the appropriate path in both videos | 3/8 | 2/8 |

Here, gap means prior transport error minus raw transport error. Positive favors raw; negative favors prior. Sources: notebook cell 15, [flow summary](pilot-01/diagnostics/flow_summary.csv) and [all displayed pairs](pilot-01/diagnostics/flow_pairs.csv). Counts and medians in this analysis are calculated from these displayed pair rows.

The favorable change in **15/16 pairs** is evidence worth following: changing the video generally changes support in the expected direction despite identical skeleton inputs. But only **5/16 pairs** have both desired signs. Real-event videos favor raw in only 4/8 cases in each role. The development median gap change is approximately 0.0647 pixels, smaller than its 0.1476-pixel mean; eight pairs provide limited protection against source-specific effects. These are descriptive paired contrasts, not a calibrated event classifier or a held-out classification result.

Sparse usable support is not the leading explanation for this particular matched cohort. Fourteen pairs have 100% coverage in both videos; the remaining two have about 94.56% in both. Equal coverage counts do not establish identical masks for those partial pairs, and availability is not correctness. Maximum clean-to-edited image separation ranges from approximately **4.1 to 10.9 pixels**, with all reported event positions in frame. These maxima are not typical displacement, frame-to-frame motion, or proof of visibility through occlusion.

Surface-reference flow EPE ranges from approximately **0.093 to 0.813 pixels**, over **34.0–94.2% foreground coverage**, in the displayed matched-case table. This is a different population and denominator from joint-neighborhood transport gaps. A subpixel mean contrast cannot be treated as statistically resolved merely because average surface EPE is small. Conversely, the high event-support coverage and geometric displacement argue against attributing the entire failure to missing flow or universally invisible events. Source: [matched-case flow diagnostics](pilot-01/diagnostics/flow_cases.csv).

**The dense strength curves narrow the next experiment without selecting it.** Notebook cell 11 evaluates strengths 0, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5 and 1. The full-prior curve's error grows steeply at larger strengths; retention also collapses there. Small corrections can have a different retention tradeoff. The linear MSE axes make the region near zero hard to assess precisely. The full `strength_curve.csv` and `strength_case_scores.csv` are needed to identify numerical repair minima. Do not infer a successful practical operating point from these plots, and do not replace notebook 03's calibration with a favorable development point.

**Recommended next experiments, in order:**

1. Establish an explicit no-change endpoint and compare projection off, the existing fixed-length projection, and a justified softer constraint using the same cached inputs. Keep originally accurate joints and completion error visible. Any changed repair rule needs consistent training and inference plus new calibration.
2. Audit the MoMask reconstruction path on clean and corrupted clips. Measure root trajectory, root-relative pose, heading and temporal alignment separately; verify normalization and padding against the author implementation. Treat reference-based realignment as a diagnostic only. A useful candidate must spend less error on accurate motion than it removes from failures.
3. Quantify the near-zero unprojected curves on calibration cases and inspect each person's result. Keep the 25% target fixed for this condition. If no practical candidate reaches it, document the negative result and define a justified new candidate or corruption condition rather than reducing the target after seeing outcomes.
4. Pursue the paired visual signal with a simple calibration-only baseline before another larger gate. Test whether support improves with a better repair candidate, and inspect the common-support event transitions for the two partial pairs. Candidate bias and event evidence need separate checks.

Notebook 06 does not evaluate the learned gate outputs or provide their raw logits. It therefore cannot attribute the previous classifier's near-chance probabilities to optimization, features or probability calibration. The new result is narrower and useful: projection has a demonstrated penalty, MoMask's current reconstruction damage is much larger, and the paired video contains a promising but insufficient decision signal. Keep final data reserved while testing these explanations on development conditions.
