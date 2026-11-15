# Independent audit of the external experiment and anomaly scores

Reviewed on 15 November 2026. This review examines the executed external run `external_fb9d0d23b56ce29fd846`, its immutable source snapshot, prepared participant registry, saved predictions and scores, and the current GAVD scoring implementation. I did not implement the external experiment. I implemented parts of the shared coordinate estimator and timing diagnostic, so this is independent of the external implementation, but not independent of every component in the project.

Only this review document was written. No scientific code, fitted model, score, participant assignment or result artifact was changed.

## Overall assessment

The external experiment is a useful and auditable method replication. I found no demonstrated participant leakage or arithmetic error that invalidates its principal result. All four executed source-snapshot hashes match the frozen protocol. I independently checked fitting identities, internal fitting identities and test identities in every fold. Each fold uses 76 healthy fitting participants, 20 separate healthy references and 44 or 45 test participants. Each participant has one outer-test assignment.

The strongest supported conclusion is specific: **the fixed equal-weight combination of the motion and relational scores performs worse than the motion score alone in this external setting, even though the fitted opposite-leg predictor improves normal prediction.** This is valuable evidence about the difference between predictive usefulness and added detection value. It is not evidence that opposite-leg information is generally useless.

| External result | Verified value | Interpretation |
| --- | --- | --- |
| Eligible participants | 128 healthy and 50 stroke, with 433 windows. | The independent sampling unit is the participant, not a window or mask. |
| Motion score M, weighted within-fold AUROC | 0.958125. | A strong descriptive movement baseline in this cohort. |
| Relational score R, weighted within-fold AUROC | 0.693750. | The score is not devoid of label discrimination. This alone does not establish clinical utility. |
| Fixed combined score S, weighted within-fold AUROC | 0.891875. | The particular fusion loses discrimination relative to M. |
| S minus M | -0.066250, conditional interval [-0.105625, -0.030625]. | A negative comparison for the frozen recipe, conditional on these fitted models and participant split. |
| Age/sex-balanced sensitivity | 45 pairs; S minus M is -0.080868, interval [-0.155819, -0.027613]. | The negative fusion result persists in this selected balanced test subset. |
| Normal prediction adequacy | All four outer recipes pass, with positive gain in all three internal folds. | This is an operational normal-data gate, not a significance test. |
| Outer-test healthy prediction | The both-leg MSE is approximately 4.8% to 5.5% lower than the own-capacity MSE in each fold. | Useful opposite-leg prediction and inferior score fusion coexist. |

Evidence is in `metrics.json`, `paired_effects.json`, `adequacy.json`, `heldout_prediction_error.csv`, and `age_sex_sensitivity.json` within the external run. Independent recomputation from saved participant profiles reproduces R to approximately 7.1e-15 and S to 3.6e-15. Rebuilding M from its CSV descriptors and fitting participants differs by at most 4.91e-7, consistent with floating-point serialization and reconstruction, rather than a meaningful score discrepancy.

## Findings that require clear manuscript corrections

### High priority: limit the negative conclusion to the tested estimator and fusion

The paper must not say that cross-leg information fails, that neurological coordination is uninformative, or that the external dataset disproves a world-model approach. The external normal-data gains and R AUROC contradict that broad interpretation. The experiment tests a particular shared ridge basis, a two-dimensional aggregation of prediction benefits, a diagonal reference distance and a fixed 0.5 mixture.

Use a sentence such as: “Opposite-leg observations improved normal completion in the laboratory-angle replication, but the prespecified relational-distance fusion reduced detection performance relative to the motion baseline.” Keep the GAVD timing result separate. GAVD timing fails adequacy on its much sparser paired input support; that does not overturn the external normal-completion result on a different observation task.

The external stroke MSE is also lower than healthy MSE for several constituent predictors. For example, fold 0 has base MSE about 131.9 for stroke and 204.3 for healthy participants. Prediction difficulty is therefore not a monotonic measure of health. Amplitude, range, timing, missingness and estimator bias can all affect squared error. Do not describe larger prediction error as necessarily more abnormal.

### High priority: equal input width is not equal effective model capacity

The external own and both correction branches each have 72 input columns, but their saved empirical ranks are **44 and 62**, respectively, in all four final fits. The GAVD branches each have 154 columns, with ranks **111 and 141**. These are material differences, not a minor numerical tolerance.

The shared baseline and own-capacity sham are worthwhile controls. However, the experiment cannot claim that it fully rules out model expressiveness as an explanation of a prediction gain. The own branch uses local interpolation features plus own-leg moments. The both branch uses local interpolation features plus contralateral interpolation features. Different bases have different rank, geometry and regularization behavior even when the number of columns and the ridge grid match.

Replace “capacity-matched” with “matched nominal input width and regularization grid, with an own-capacity control,” and report the ranks in the main methods or limitations. A future protocol can compare matched effective degrees of freedom or an explicitly rank-matched feature construction. That future comparison must be selected on development normals and must not replace the present test results after inspection.

Evidence: external `fold_*/fit_audit.json`, GAVD `fold_*/fitting_record.json`, and the feature construction in the frozen `external_iteration02.py:37` and `:170`.

### High priority: the fixed mixture does not establish calibration equivalence

M and R are both nonnegative dimension-normalized distances, but their calibration is not interchangeable. M uses a covariance model fitted on 76 external healthy participants or 16 GAVD fitting-normal recordings. R uses a separate diagonal reference fitted on 20 external or five GAVD references. Their finite-sample uncertainty, tail behavior, covariance assumptions and centers differ.

The observed normal distributions make this tangible. External fold 1 has test-normal median M of about 1.003 and median R of 0.154. GAVD U fold 1 has test-normal median M of about 0.620 and median R of 5.804. These are descriptive audits, not grounds to tune weights on test data. Dividing by feature dimension does not make the two terms probabilities, percentiles, or equally reliable anomaly channels.

R also ignores covariance between its left and right components. In the external normal-reference data, their correlations are approximately 0.823, 0.767, 0.572 and 0.544. Two standardized profiles such as `(2, 2)` and `(2, -2)` receive the same diagonal distance despite representing different bilateral patterns. Averaging three or four joint benefits per side discards further structure before the distance is calculated.

The manuscript should identify this as a limitation of the tested score construction. A poorer fixed fusion does not prove that no combination could add value beyond M. Keep the frozen equal-weight comparison as the primary result. Describe a calibrated or covariance-aware alternative only as a future, separately registered study. Do not choose a fusion weight from the current abnormal results. For GAVD, report the already saved leave-one-reference-out and variance-floor sensitivities alongside this limitation.

Evidence: `iteration_scoring.py:136`, `:147`, `:223`, `:233`; external `fold_*/reference_audit.json`; GAVD `iteration_runner.py:169` and saved U-bank source scores.

### High priority: describe the C3D inputs as processed biomechanical model outputs

The external code verifies `POINT:ANGLE_UNITS == deg`, membership in `POINT:ANGLES`, continuous raw frame numbering, and residual validity. It selects component index 0 of six synchronized PiG joint-angle channels. The manufacturer's tables identify the first component as hip flexion, knee flexion and ankle dorsiflexion on both sides. This supports the component choice. [Vicon angle definitions](https://www.vicon.com/support/faqs/what-are-the-lower-body-segment-angles-from-plug-in-gait/).

These are not raw marker coordinates, direct measurements of neural commands, or BlazePose equivalents. The dataset authors report manual gap filling and the standard PiG dynamic pipeline, including Woltring smoothing, before C3D export. The distinct MAT release adds further filtering and stride normalization; those MAT operations were not used here. Hip angles are defined relative to the pelvis, so the precomputed channels already incorporate shared biomechanical references. Masking angle channels does not undo that upstream model dependence. [Dataset methods](https://www.nature.com/articles/s41597-023-02767-y.pdf).

State that access restrictions apply to the six processed angle channels. Do not call them isolated physiological leg states, or claim that this experiment validates video-coordinate predictions. The external experiment is an adapted method replication with a different representation, reference size and participant population. It is not GAVD encoder transfer, a JEPA experiment, or a comparison on identical physical targets.

## Additional methodological limitations and concrete actions

### Medium priority: make the 15 Hz processing assumption explicit

The executed parser linearly resamples to 15 Hz using adjacent valid source observations. It does not apply a new explicit anti-alias low-pass filter before reducing the rate. Upstream PiG smoothing exists, but its MSE parameter does not establish a documented 7.5 Hz cutoff. This is a measurement limitation to audit, not evidence that aliasing caused the negative result.

Report the resampling policy and upstream processing accurately. In a future measurement audit, examine signal power above the target Nyquist frequency and compare an explicitly specified anti-alias procedure before any new fitted comparison. Do not silently replace the current arrays or call this run unfiltered raw-angle analysis. Evidence: frozen `extract_angles` at `external_iteration02.py:67` and its call to `data._resample`.

### Medium priority: internal selection and final scoring average observations differently

The external `internal_report` sums errors across mask-target presentations before averaging within each window and angle. Final `score_profiles` first averages masks within each identical window/angle/patch target. Because the fixed bank does not hide every physical target equally often, these are not exactly the same weighting. The GAVD internal report also pools mask-target rows within source and joint, while final detection uses the fuller target, window, sequence and source hierarchy.

This is not test leakage, and participant/source balancing remains present. It does mean that the internal adequacy objective is not identical to the final profile aggregation. Explain the internal selection objective separately. A future protocol should use the same target-first hierarchy for selection and final measurement, or explicitly justify the different risks. Do not claim that every repeated mask has already been collapsed before the current internal-selection loss.

Evidence: frozen external `internal_report:178` versus `score_profiles:203`, shared `retained_prediction.inner_candidate_report`, and `iteration_scoring.aggregate_target_profiles`.

### Medium priority: report the missing direct interpolation benchmark

The implemented local ridge consumes interpolated observations as features. That is not the same as reporting the error of the direct interpolation predictor. The external run does not save a direct interpolation comparison, and the GAVD runner's primary baseline is fixed to ridge. A useful correction relative to these fitted controls does not establish that the local prediction problem has been solved well relative to a strong simple temporal interpolator.

Do not write that the experiment outperforms interpolation unless a separately documented result exists. Either disclose this missing comparison or add it under an explicit supplemental analysis identity with the same targets and support. It must not be used to revise the existing selected recipes retrospectively. This distinction matters because an added-context improvement over a weak local solution would overstate the information interpretation.

### Medium priority: matching is a sound sensitivity, not removal of demographic or speed effects

The matching implementation uses only verified demographics and outer-fold membership. It enforces same reported sex, a five-year age caliper, no replacement, and within-fold pairs. The large invalid-edge cost makes the assignment maximize the number of valid matches before minimizing age differences at these cohort sizes. The saved audit confirms these constraints. Whole-pair bootstrap sampling preserves the matching structure, and models are not refitted on the matched subset.

The resulting subset has 45 healthy and 45 stroke participants, with fold pair counts 11, 12, 11 and 11. The mean absolute age difference is about 2.02 years. Five stroke and 83 healthy participants are outside this sensitivity. The target population changes, so this is not a second estimate of exactly the same unrestricted population.

Use “age/sex-balanced held-out sensitivity,” not “age and sex effects were removed.” The normative models and references still span a broad healthy age range. Walking speed, impairment severity, assistance, lesion characteristics and other clinical factors are not matched. Preferred-speed walking and participant speed metadata are described in the source dataset, but this protocol explicitly leaves speed matching false. Do not interpret the remaining group contrast as a specific neurological mechanism. [Dataset description](https://www.nature.com/articles/s41597-023-02767-y).

### Medium priority: preserve the selected-trial denominator

All 188 frozen selected files parsed, but ten healthy participants failed the declared window/target-support requirements. The analyzed cohort is therefore 178 participants, not all 188 people in the release. Only one predetermined dynamic trial per participant was used, leaving repeatability across trials untested. The lexical selection avoids outcome-directed trial selection, which is a strength, but it does not establish that the selected trial is representative of the participant's gait.

Show the 188 selected, 188 parsed, 178 eligible flow and the 128/50 analyzed groups. Do not call the ten exclusions download failures or imply that all released trials were analyzed. The saved eligibility ledger lists the exclusions. A future repeat-trial analysis can estimate within-person variation without increasing the claimed participant count.

## Implementation checks that passed

The following distinctions should remain visible in the paper and tutorials because they are strengths of the execution.

1. The saved source snapshot matches the recorded producer hashes. The later CLI replay-guard change is documented separately and does not purport to generate new outcomes.
2. Fitting, reference and outer-test participants are disjoint within each fold. Saved internal model identities match the corresponding internal training role. No abnormal participant is used for normal-only fitting or reference estimation.
3. Uniform masks hide exactly 24 of 48 angle-patch tokens. Angle values and velocities do not influence sampling. The two legs share the raw trial clock; no independent cycle alignment is applied.
4. The feature builder removes masked values before interpolation or moments are calculated. The existing access tests check that changing hidden values cannot affect features and that opposite-leg changes cannot affect the own branch. The zero correction bypasses the correction model exactly.
5. Final profiles collapse repeated masks before averaging targets and windows. Reference fitting uses only the declared reference participants. The saved four-distinct-patches support rule counts physical targets, not mask repetitions.
6. The principal AUROC summary compares scores within fitted folds, with fixed pair-count weights. It does not rely on pooled cross-fold score rankings. Bootstrap comparisons preserve participant membership and condition on the frozen models and split.

These checks make the external negative result interpretable. They do not remove the representation, capacity, calibration, selection and generalization limitations above.

## Suggested manuscript position

The strongest paper is a careful test of when useful conditional prediction becomes useful anomaly evidence. It can show a positive result for normal prediction alongside a negative result for the chosen anomaly fusion, and show why a stricter timing intervention becomes uninformative when permitted context is sparse. That is more defensible than promising that the method recovers neurological causes.

Before submission, correct the capacity wording, include the R-alone result and normal prediction gains, describe the processed PiG representation, disclose the selected-trial flow and unmatched speed, and separate fixed-fusion failure from general claims about cross-leg information. Keep the current scores frozen. Any new calibration, rank matching, interpolation benchmark or filtering analysis should have a separate identity and an explicit exploratory label.

The `draft02` manuscript directory was not yet populated with manuscript text when this audit began. Consequently, these are evidence-based implementation and claim-boundary findings, not a claim that every problematic phrase already appears in the final manuscript.
