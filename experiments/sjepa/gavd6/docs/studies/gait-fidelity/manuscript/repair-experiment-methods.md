# Controlled readout repair: methods text and reporting contract

**Status:** implemented extension; source results are pending. The text below
describes the declared method. Do not change it to past tense or insert effect
claims until the source completion receipts and coverage have been checked.
Software fixture scores are not scientific results.

## Objective comparison

We retain the fitted delta-feature and endpoint-feature JEPA encoders and train
new coordinate readouts. For each encoder and each of the original three seeds
(17, 29, 43), two readouts share initialization, endpoint sampling, optimizer,
update count, coordinate supervision, and a short-segment geometry penalty.
Encoders remain fixed. This produces twelve new fits without additional
representation pretraining.

Let \(\theta_{e,t,\ell}\) denote a projected image-plane knee angle for endpoint
\(e\in\{a,b\}\), aligned frame \(t\), and leg \(\ell\). Endpoints correspond
to paired physical-intervention conditions; the difference below is not a
temporal velocity. Define excursion asymmetry
\(A_e=(P_{95}-P_5)(\theta_{e,\cdot,R})-(P_{95}-P_5)(\theta_{e,\cdot,L})\).
The scalar objective penalizes error in \(A_b-A_a\):

\[
L_{\mathrm{scalar}}=\mathbb E_{\mathrm{pairs}}
\left[\left(\frac{(\hat A_b-\hat A_a)-(A_b-A_a)}{180}\right)^2\right].
\]

The dense objective penalizes the paired angular response at each supported
frame and leg:

\[
L_{\mathrm{dense}}=\mathbb E_{\mathrm{pairs}}\,
\operatorname{mean}_{t,\ell}
\left[\left(\frac{(\hat\theta_{b,t,\ell}-\hat\theta_{a,t,\ell})
-(\theta_{b,t,\ell}-\theta_{a,t,\ell})}{180}\right)^2\right].
\]

Both objectives use identical reference-determined support common to the two
endpoints and both legs. Prediction failures cannot remove supervised frames.
The low-scalar coefficient is 0.1 times the inherited scalar coefficient. For
each encoder/seed, the dense coefficient is fixed using 32 training-only batches
at the same initial readout:

\[
\lambda_D=\lambda_{\mathrm{low}}
\sqrt{\frac{\sum_j\|\nabla_w L_{\mathrm{scalar},j}\|_2^2}
{\sum_j\|\nabla_w L_{\mathrm{dense},j}\|_2^2}}.
\]

The coordinate term and inherited geometry coefficient are identical in the
two new arms. Matching applies to initial auxiliary gradient RMS before
clipping. It does not ensure equal gradients or updates throughout optimization.
Training logs retain component losses, gradient magnitudes, clipping, and
realized angle-gradient support. Dense supervision changes temporal information
as well as gradient support, so this comparison does not uniquely identify a
causal effect of percentile-gradient sparsity.

## Fixed comparison set and endpoint

Evaluation retains both new objectives for both encoders, the corresponding
original base and paired-change readouts, and direct-coordinate/base: nine
methods with three seeds each. The primary contrast is delta/dense versus
delta/low-scalar for waveform error on the predeclared ViTPose family. Response
error is reported with an interval as a tradeoff. Endpoint-feature contrasts,
other extractors, differences from original objectives, and the remaining
direct-coordinate gap are descriptive secondary analyses. No winning method
is selected for confirmation. The original coordinate-only base does not
contain the geometry penalty and is therefore a practical reference rather
than the clean matched-objective contrast.

## Protected-person confirmation

The slim metadata plan retains all fourteen original-test named-walking
candidate people: eleven from BioMotionLab and three from KIT. It selects two
distinct raw motions per person by a fixed metadata hash, then one planned
window per motion. Selection precedes protected reference access and uses no
model outcomes. It is conditional on reviewed unused-person history; absent
records do not establish non-exposure.

The panel uses intervention levels 0°, 5°, and 15°, two physical orientations,
two cameras, clear/occluded observations, and two pose-estimator families.
At 128 samples per window, the 28-window plan requires 86,016 rendered frames
and 172,032 estimator frame-passes before reference QC. Naming corruptions
reuse extracted tracks. These are repeated observations of fourteen people,
not additional independent samples. Reference QC exclusions are retained in
coverage accounting without substitute windows. The smaller motion panel
changes coverage and may increase within-person measurement noise.

The checkpoint inventory, exposure evidence, source identities, condition
panel, statistical protocol, and code are locked before protected preparation.
Development and confirmation are reported separately. Complete-population
confirmation requires all planned people/windows and supported primary
measurements; otherwise the result is explicitly limited to retained support.

## Uncertainty and reporting

We aggregate conditions within source windows, windows within motions, and
motions within people. Primary paired differences average the three fixed
fits within each person. We report the person mean difference and its 95%
paired t interval, individual-seed estimates, and a descriptive crossed
person/seed bootstrap. These estimates condition on a small set of trained
models; repeated seeds do not multiply the participant count.

Absolute response error includes a 720° failure penalty, and waveform error
includes a 180° failure penalty. We separately report failure rates, their
contribution to the aggregate error, successful-output contributions, and
conditional successful-output errors. Failed predictions are not deleted.
A favorable aggregate difference can reflect fewer geometric failures,
better successful waveforms, or both.

Power sensitivity uses the actual development paired-person variance and
explicit effect-size assumptions. Fourteen people are suited to checking
large paired effects, not reliably establishing a small advantage. No
noninferiority margin or clinical meaningful-change threshold is assumed;
an interval crossing zero is not evidence of preservation or equivalence.

## Populate the result section only from completed source artifacts

Use `summary.json`, `comparisons.json`, `per-person-by-extractor.csv`, coverage
tables, and `power-sensitivity.json` from each split's evaluation directory.
Report, in order:

1. Planned/retained people, motions/windows, reference eligibility and failed
   predictions; source composition and exposure-review evidence.
2. The fixed primary waveform difference and interval, followed by the
   response tradeoff and individual seeds.
3. Waveform and response failure decompositions and the matched-base/direct
   gaps. Avoid a successful-output-only headline.
4. Whether the direction and magnitude replicate in independent confirmation;
   keep development exploratory and do not pool the two sets.
5. The finite synthetic/source scope, three-seed limitation, limited power,
   and absence of dense real-video or clinical validation.

A null result remains a result of this fixed contrast. Do not switch the
primary extractor, method, penalty, cohort or endpoint after inspecting
confirmation. CMU and GAVD readiness inventories are not additional empirical
results for this experiment.
