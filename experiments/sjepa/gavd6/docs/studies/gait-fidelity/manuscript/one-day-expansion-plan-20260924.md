# One-day evidence expansion and mechanism plan

Prepared 24 September 2026. This is a proposed extension, not a completed experiment or a claim of independent confirmation. No new HAIC jobs were launched while preparing this analysis. Preserve the existing core and response-follow-up outputs and their original primary comparisons.

Implementation: the controlled readout experiment and protected slim AMASS
workflow are available in [START_REPAIR_03.md](../../../../slurm/gait-fidelity/START_REPAIR_03.md),
with [independent implementation review](../reviews/repair-implementation-20260924.md).
CMU and GAVD remain conditional readiness audits; they are not added to the
participant count or scientific result tables by this implementation.

## Recommendation

Prioritize **independent AMASS confirmation of the waveform cost of scalar response supervision, plus a tightly controlled attempt to mitigate that cost**. Use a smaller rendering matrix that retains all eligible confirmation people. A credible mechanism and an independently evaluated intervention have more scientific value than increasing correlated trajectory counts.

The two conditional extensions are a separate CMU whole-source transfer test and a GAVD recording-group transfer evaluation. Neither supplies an automatic increase in verified independent participant count. Neither belongs on the critical path if its data and protocol cannot be made ready promptly.

The strongest defensible prospective question is:

> Does the waveform cost of the tested scalar response objective reproduce on previously unopened people, and can dense temporal supervision mitigate it beyond a loss-weight adjustment?

Do not promise that JEPA will outperform direct training, that any comparison will be significant, or that the paper will be accepted.

## Verified capacity, rather than hypothetical sample size

| Available inventory or plan | Verified count | What it does not establish |
| --- | ---: | --- |
| Current core development evaluation | 14 people | Independent confirmation of hypotheses developed from these results |
| Named-walking AMASS confirmation plan | 14 candidate people, 97 motions, 199 windows | Remote availability or previously unexposed status |
| All-action AMASS confirmation plan | 18 candidate people | Forty or more new participants, or a walking-only population |
| Named-walking training plan | 113 people before screening | Actual number used by the completed fit; inspect the prepared bundle |
| Raw CMU inventory, excluded from approved cohort | 2,082 motions, 102 candidate folders | 102 distinct people |
| Full local GAVD metadata | 1,874 sequences, 348 source videos | 348 verified independent people, available videos, or dense reference poses |
| GAVD dataset-level normal label | 27 normal versus 321 abnormal video groups | A large balanced evaluation cohort |

AMASS counts come from `slurm/gait-fidelity/validation/manifest-census.json`, the three current AMASS manifests, and a fresh metadata audit. The named-walking confirmation candidates are 11 BioMotionLab people and three KIT people. Every candidate has at least three motions and nine planned windows. The exposure-reservation records on HAIC remain unverified locally.

The 14 new candidates would be evaluated separately from the 14 development people. Do not pool them and describe the result as an untouched 28-person test.

## Idea 1: confirm and explain the measurement–trajectory mismatch

### Why this is the strongest bet

The core person-level results contain a much larger and more consistent effect than the 0.3731-degree follow-up JEPA contrast. Averaging the three fitted seeds within each person:

| Paired waveform difference | Mean | Across-person SD | Positive person effects |
| --- | ---: | ---: | ---: |
| Direct/change minus direct/base | +7.208 degrees | 1.029 degrees | 14/14 |
| Original JEPA/change minus JEPA/base | +5.100 degrees | 1.921 degrees | 14/14 |

These are exploratory development estimates. They support prioritizing a replication; they are not estimates of the effect or variance of a new repair.

There is also a concrete structural issue to investigate. `torch_knee_excursion()` in `measurements.py` differentiates exact interpolated P95 and P5 quantiles. Away from ties, the scalar term can touch at most eight of the 256 knee-angle entries per 128-frame endpoint, or 3.125%. This excludes the separate short-segment penalty. The coordinate objective remains dense, and parameter sharing can transmit updates across timestamps.

Temporal permutation leaves an excursion unchanged. A local toy calculation verified both properties: 16 nonzero scalar-loss angle gradients among 512 entries across two endpoints, and an unchanged excursion after permutation despite a large waveform difference. This is a mathematical/software check, not an empirical gait result.

### Minimum controlled experiment

Reuse frozen encoders; do not retrain the representation matrix. Before opening confirmation outcomes:

1. Inspect the already completed follow-up intervals, failure decomposition, and feature diagnostics. Confirm that the apparent effects are not scoring or support artifacts.
2. On fixed training batches, measure auxiliary/coordinate gradient norms, cosine similarity, temporal gradient concentration, clipping, and short-segment penalties.
3. Compare existing base and scalar-response arms with a lower-weight scalar control and a dense paired angular-change objective. Match batches, head initialization, updates, and the training-calibrated gradient budget. Keep coordinate and geometry-penalty terms identical between newly matched arms.
4. Prefer the delta and endpoint frozen encoders across all three existing seeds; retain direct/base prominently as the practical comparator. If adding a repaired direct model, state its different end-to-end optimization opportunity. Fix the selected model matrix before confirmation.
5. If resources permit, include dense endpoint-angle supervision to test whether pairing contributes beyond dense geometry supervision. A softened-quantile scalar control would better separate gradient sparsity from temporal information content. Without it, avoid attributing a result uniquely to sparse gradients.

A simple candidate auxiliary is

\[
L_{\rm dense}=\operatorname{mean}_{t,\ell\in S}
\left[
\frac{(\widehat\theta_{b,t,\ell}-\widehat\theta_{a,t,\ell})
-(\theta^*_{b,t,\ell}-\theta^*_{a,t,\ell})}{180}
\right]^2.
\]

Use common reference-supported timestamps, both knees, and aligned synthetic endpoints. Retain coordinate supervision. This is a candidate intervention, not a novel loss formula by itself.

Temporal geometric supervision already has close precedents, including [Motion Guided 3D Pose Estimation](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123580749.pdf). [GradNorm](https://proceedings.mlr.press/v80/chen18a.html) is relevant precedent for gradient balancing. The potential contribution is the controlled explanation and independent evidence, not inventing temporal losses or balancing gradients.

### Slim AMASS confirmation panel

Retain all 14 candidates, subject to verified exposure and technical eligibility. Select two windows from distinct raw motions per person using an outcome-independent hash order. Freeze any reference-only replacement rule before model evaluation.

| Factor | Planned count |
| --- | ---: |
| People | 14 |
| Distinct-motion windows per person | 2 |
| Movement levels | 0, 5, 15 degrees |
| Physical states | Original and mirrored |
| Views | Two |
| Observation conditions | Clear and occluded |
| Extractors | Prespecified primary plus secondary |
| Frames per window | 128 |

This gives 28 underlying windows, 672 rendered clips and **86,016 rendered frames**. Two extractors require 172,032 extraction-frame passes. Including derived naming and no-change conditions gives 5,376 trajectory records, approximately 198 MB per raw array copy under the current layout. Independent N remains at most 14.

The existing full confirmation plan estimates 815,104 rendered frames. The slim panel cuts repeated-condition work by about 9.5 times while preserving its candidate person count. It samples fewer motions, which can increase within-person measurement noise and change the estimand; disclose this reduction.

A four-hour preparation target with four workers requires at least 1.49 completed render frames/second/worker, inclusive of both estimators, before setup, merging, and safety allowance. With eight workers the arithmetic threshold is 0.75. Benchmark the entire pipeline on already-open training windows. No measured local throughput currently justifies a runtime promise.

### Statistical decision rules

Freeze one primary repair contrast and its endpoint before confirmation. Report response, waveform, coordinate error, direction, failures, and the direct/base comparison together.

- A waveform improvement supports **mitigation of waveform error**.
- A claim of joint improvement requires evidence for both waveform and response improvement.
- A nonsignificant response deterioration does not establish response preservation.
- Noninferiority requires a justified margin fixed before opening confirmation results; there is no established clinical margin for this projected synthetic measurement.
- If the lower-weight scalar matches the dense objective, favor the loss-balancing explanation.
- If dense supervision only improves the quantity it directly targets, describe that limited benefit.
- If direct/base remains best, do not replace it with a weaker comparator.

Report person-level effects averaged across the three fixed fits, each seed separately, and appropriate paired uncertainty. The three seeds provide limited information about the population of possible training runs. A secondary crossed bootstrap is useful but does not create more independent seeds or people.

### Engineering limitations

This is not a ready-to-run option on `jepa-response-02`. Its evaluator is development-only, and the child disables confirmation/GAVD preparation. Existing confirmation locks bind the complete parent-style cohort and checkpoint inventory. The trainer also rejects changed downstream training settings against an upstream checkpoint's full training signature.

Implement a separate immutable extension that explicitly verifies encoder/data identities while permitting declared downstream overrides, records the reduced cohort, binds a cross-run checkpoint list, and evaluates confirmation without relabeling it development. Preserve existing runs and identity safeguards. Allow roughly a 2–4-hour engineering risk budget, with a small fixture and full preparation timing gate before committing to the source run.

## Independent adversarial power review

Two separate statistical reviewers checked the proposal. The primary calculations use a two-sided paired t-test at alpha=.05 and 80% power as an illustration conditional on independent, approximately normal person effects. They are not a guarantee for the actual hierarchical data.

| Independent people | Approximate detectable paired standardized effect |
| --- | ---: |
| 14 | 0.81 SD |
| 40 | 0.45 SD |
| 60 | 0.37 SD |

At N=14, a usual 95% paired-mean interval has half-width about **0.577 times the paired-person SD**. Using the original direct and JEPA waveform SDs only for illustration gives 80%-power detectable changes of approximately 0.83 and 1.56 degrees. New repair effects and the slim sampling design can have different variances.

For a 0.3731-degree improvement, hypothetical paired-person SDs of 1, 1.5, and 2 degrees require approximately **59, 129, and 228 people**, respectively. With 14 people and SD=1 degree, illustrative power is only about 25%. The actual follow-up paired SD has not been inspected locally.

Verdict: 14 fresh people can be a credible large-effect confirmation. They are a poor plan for reliably establishing a small JEPA advantage. Additional windows, renderings, frames and seeds cannot repair this by being counted as independent people. Power for the repair remains unknown until its development variance is assessed, and that assessment should include sensitivity to larger variance and smaller effects.

Statistical significance is not sufficient practical value. Report the absolute effect, interval, fraction of the observed waveform cost mitigated, remaining gap to direct/base, and any response/coordinate deterioration. Do not label an engineering threshold a clinical minimal important difference.

## Idea 2: a separate whole-source CMU transfer test

The local raw inventory contains 2,082 CMU motions across 102 candidate folders; 1,437 meet the 5.08-second minimum. All were excluded from the approved registry because person aliases are unresolved. The [CMU database](https://mocap.cs.cmu.edu/) warns that subject numbers need not uniquely identify people.

Holding the entire source outside fitting can support a source-transfer experiment if every actual training and exposure manifest confirms absence. Leave the existing registry unchanged. Numeric motion IDs require a trustworthy join to [motion descriptions](https://mocap.cs.cmu.edu/search.php) or blinded locomotion review before calling the panel walking.

Freeze models and select a fixed, metadata-ranked motion panel before seeing predictions. Render and extract poses for a smaller panel if throughput permits. A cheaper projected-joint/artificial-corruption experiment can cover many more motions, but is a distinct synthetic-corruption stress test, not confirmation of the original estimator-observation pipeline.

Report exact finite-corpus paired effects, eligible motions, folders, and exclusions. Without a crosswalk, do not call folders participants or bootstrap them as independent people. One held source does not establish a distribution over unseen source datasets.

**Adversarial verdict:** useful for breadth and domain transfer; fails as a shortcut to 102-person statistical power. Lower priority than the protected AMASS confirmation. Stop if identity/metadata preparation would displace the primary experiment or writing.

## Idea 3: conditional GAVD transfer with a correctly defined target

Use frozen AMASS restorers to test whether outputs retain information associated with GAVD's sequence labels. GAVD supplies clinical annotations and video metadata, not the paired dense joint references needed for this paper's knee-response and waveform endpoints. See the [official repository](https://github.com/Rahmyyy/GAVD).

Important audit findings:

- Dataset-level labels yield 27 normal versus 321 abnormal video groups. Gait-pattern labels instead yield 32 normal groups; the concepts disagree on 37 sequences from five videos. Choose one definition before performance analysis.
- The existing default split, simulated before media/exposure/group exclusions, gives six normal development groups and four normal confirmation groups.
- Only 21 of the 27 normal video IDs have any annotation span of at least 128 native frames; this is an upper bound before FPS and availability checks.
- Sixteen of the 27 normal video IDs occur in the previously inspected local collection. Excluding known exposure does not prove the remainder unexposed.
- Local media includes 91 clips and 88 historical pose caches. The caches use MediaPipe33 at nominal 15 Hz without the timestamps and missing/imputation records needed by the current body12/25 Hz protocol. They are not directly compatible. HAIC cache availability remains unknown.

If suitable assets are already available, use prespecified grouped out-of-fold binary probes, training-fold-only normalization, fixed regularization, and linked recording/person/repost components. Compare unchanged tracks, filtering, direct/base and fixed JEPA methods. Retain camera/height and availability controls, and within-view reporting where both classes have support. The current multiclass fixed-split evaluator needs a separate binary grouped-cross-validation path.

A group bootstrap is descriptive uncertainty conditional on the fitted folds/probes. Unknown person links and past exposure prevent automatically calling this participant-independent external confirmation. More folds or repeated seeds do not add normal groups.

An illustrative paired-binary calculation with discordance probability .2 yields an approximately 12.5-percentage-point detectable balanced-accuracy difference with all 27/321 groups, and roughly 27 points with the default 6/60 development groups. Actual group-averaged scores need their own variance assessment. This is a poor design for promising a two-to-five-point gain.

**Adversarial verdict:** conditional supplementary transfer only. Establish compatible caches, available media, grouping, exposure and plausible precision within the first one or two hours; otherwise remove GAVD from the day's critical path. Sparse manual keyframes could support visible-coordinate agreement, but cannot validate full-window waveform or percentile-excursion fidelity. An estimator's predictions cannot be promoted to ground truth.

## Execution order and stopping rules

1. **Hours 0–2:** retrieve and verify completed follow-up evidence; audit actual training exposure, protected AMASS candidates and HAIC assets; record the prospective contrast and candidate matrix. Inspect GAVD/CMU readiness in parallel without consuming the main preparation budget.
2. **Hours 2–4:** finish the separate protocol/runner, meaningful fixture checks, gradient audit and source throughput measurement on already-open training data. If these gates fail, stop expansion and complete the verified existing paper.
3. **Hours 4–10:** run the bounded development repair fits and prepare the locked confirmation observations; prevent evaluation outcomes from informing further model choices. New models must be frozen before their confirmation predictions are inspected. Write methods and existing results concurrently.
4. **Hours 10–14:** one locked confirmation analysis, person/seed effects, failure contributions, and final figures. Respect any earlier agreed experiment cutoff, including September 25 at 08:00 Pacific, rather than extending the existing child deadline.
5. **Remaining time:** complete the manuscript, references, limitations, PDF verification and submission buffer. Do not launch another sweep to rescue an unfavorable result.

Operational stopping may depend on availability, reference validity, throughput, coverage and the fixed deadline. It must not depend on whether a p-value is favorable. Preserve the original follow-up primary result, the chronology of new hypotheses, every declared arm and the distinction between development and confirmation.

## Alternatives considered and rejected for this deadline

- More augmentations or windows from the same people: precision/coverage benefits, no new independent participant count.
- Switching to all AMASS actions: only four additional approved test candidates and a changed scientific scope.
- Treating CMU folders or GAVD sequences as people: invalid independence claims.
- A large new pretraining sweep: insufficient time for controlled interpretation and independent validation.
- Predictor-retention as the headline innovation: useful if existing diagnostics motivate it, but [CAPI](https://arxiv.org/html/2502.08769v3) already studies predictor reuse; it does not address the population limitation.
- A nominal label-efficiency claim from using fewer readout labels: paired clean-reference pretraining already consumes privileged supervision, which must be counted.
- A new GAVD classifier score without group/camera controls or sufficient normal groups: weak evidence for the actual restoration question.

The primary plan is intentionally useful if the repair fails: an independently reproduced, well-characterized limitation can strengthen the paper. It cannot guarantee acceptance, and a larger-looking denominator cannot substitute for that evidence.
