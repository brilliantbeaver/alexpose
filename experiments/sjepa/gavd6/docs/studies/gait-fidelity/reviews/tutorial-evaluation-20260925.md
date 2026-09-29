# Evaluation tutorial audit, 25 September 2026

Scope: `lesson_evaluation.py` and the calibration explanation in
`lesson_walkthrough.py`. The scientific source, configurations, saved results,
protected cohorts and checkpoints were not changed.

## Findings from the source trace

The existing evaluation tutorial already exposes the main core computations:
projected knee angles; reference-only support intersected across every source
family; linear-interpolation P95−P5 excursion; right-minus-left excursion;
coordinate and 0.20-second displacement errors; geometric left/right assignment;
movement, nuisance and interaction contrasts; condition/window/motion/person
aggregation; and crossed people/seed bootstrap intervals. These calculations are
checked against retained tables or the independent production implementation.

The support used for evaluation is stricter than the paired training loss:
`evaluation.common_reference_support` intersects every row in a source family,
whereas `measurements.measurement_loss` and
`repair_objectives.repair_measurement_terms` intersect the two training endpoints
and both legs. Neither can use predicted geometry to remove reference frames.
Prediction failure keeps the reference-supported unit with the declared cost.
Endpoint excursion, endpoint waveform, response/nuisance, and interaction costs
are respectively 360, 180, 720, and 1,440 degrees. These are scoring conventions,
not measured angles from successful predictions.

The additional response, probe and repair calculations were underexposed. The
revisions add explicit code for:

- Direction eligibility from the *measured reference change* and saved tolerance;
  failed predictions remain incorrect in the reference-resolvable denominator.
- Response-score decomposition into successful-output and failure-cost
  contributions. The successful contribution assigns zero to failed eligible
  pairs, while a conditional-success mean omits them. Only the former has the
  same denominator as the complete score.
- The feature-difference ridge probe: preserved slot order, person-level hash
  folds, fitting-fold standardization, the dual solve with `n_train * alpha`, and
  equal-person mean squared error. A primal solve checks the small teaching
  example, followed by comparison with every production probe prediction.
- The repair's participant t interval after averaging paired seeds within each
  person, with a complete constructed grid and a replay of the downloaded
  ViTPose-only primary estimate when that evidence is present.

The GAVD group-weighted classification probe remains a separate example. Its
labels, weighting and ridge objective differ from the response regression probe.
No GAVD result is implied by the numerical demonstration.

## Corrected factual statement

The walkthrough said that the dense coefficient matched the coordinate-gradient
scale. `repair_training.calibrate_repair` actually sets

\[
\lambda_{\rm dense}=0.1\,w_{\rm inherited}
\sqrt{\frac{\sum_b\|\nabla L_{\rm scalar,b}\|_2^2}
{\sum_b\|\nabla L_{\rm dense,b}\|_2^2}}.
\]

It matches the weighted dense gradient RMS to the **low-scalar** angular gradient
RMS across the same fixed initial readout parameters and training-only batches.
The coordinate-gradient RMS is diagnostic, rather than the calibration target.
The walkthrough now states that distinction. Matching initial gradient magnitude
does not match gradient direction or establish equal strength throughout fitting.

## Interpretation boundaries checked

The response probe's held people were excluded from fitting a fold's ridge
coefficients, but were part of the encoder training population. Its teacher sees
projected references; its deployed encoder sees estimated observations. The
saved `probe_result` flag refers to the masked teacher branch and cannot establish
successful encoder transfer. The tutorial makes these boundaries explicit.

The repair primary compares delta-JEPA dense versus low-scalar waveform error
for ViTPose. The core and response primaries pool estimators and use crossed
bootstrap intervals. The repair's primary t interval averages the three seeds
within a person and is conditional on those fits; its crossed interval is
additional. All three stages reused development people. No interval removes
that research-history limitation, demonstrates clinical importance, establishes
equivalence, or supplies evidence of independent protected confirmation.

## Checks run

All code cells in the two affected lesson builders compile. The new response
accounting cells ran against an existing completed CPU fixture's endpoint table
and matched `response_metrics` and `response_failure_decomposition`. Constructed
edge cases explicitly checked a failed negative response, a below-tolerance
reference change and a reference-ineligible pair.

The ridge example matched production fold assignments, all held predictions,
equal-person error and the zero-response benchmark. Its dual coefficients also
matched the primal solution to numerical tolerance. The repair teaching grid
matched `paired_comparison` point estimates, every person effect and the t
interval. The downloaded ViTPose result independently reproduced its saved point
estimate and primary interval. These smoke checks wrote no experiment artifacts
and performed no training or cluster actions.

The parent task performs full notebook execution and final independent review
after integrating the other tutorial changes.
