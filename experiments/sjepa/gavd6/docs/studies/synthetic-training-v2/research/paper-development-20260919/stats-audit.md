# Statistical design for the next pose-restoration study

The current eight-person evaluation is useful development evidence. A substantive paper needs a fixed research question, an independently held evaluation population, and uncertainty that accounts for people and model training. Adding windows or collecting enough runs to cross a significance threshold would not provide those safeguards. The target should be a useful effect that survives a strong comparison, with sufficiently narrow uncertainty to distinguish improvement from negligible change.

This planning analysis independently reconstructed all 60 contrasts in the retained [person-level analysis](../../results/seed17-complete-analysis-20260919/README.md). It does not change those results, fit models, or evaluate new data. The accompanying [script](stats-sensitivity.py) and [verification record](stats-verification.json) retain source hashes, formulas and numerical checks.

## What the existing variation says

The eight development people each contribute four physical windows, four rendering conditions and three pose extractors. Their 384 track records are repeated measurements of eight people. Every neural result currently used here comes from seed 17. Differences between methods on the same people can be informative, but person resampling alone conditions on these particular trained models.

| Contrast | Position improvement, across extractors | Paired person SD as a percentage of comparator mean | Displacement improvement | Corresponding paired person SD |
| --- | ---: | ---: | ---: | ---: |
| Direct versus affine calibration | 7.15–8.51% | 4.97–6.04% | 6.17–8.31% | 2.98–6.02% |
| Direct versus static control | 1.71–6.04% | 4.14–8.93% | 4.80–5.99% | 3.89–6.82% |
| Paired versus shuffled JEPA | −0.053–0.057% | 0.120–0.238% | 0.023–0.055% | 0.039–0.126% |

Improvements are ratios of person-balanced means; the SD is the sample SD of each person's paired absolute error difference, divided by the comparator's mean. It is not the SD of individual percentage improvements. Position uses the original visible-reference metric and displacement uses 0.20-second position changes. [Full calculated variation](stats-pilot-variation.csv).

The tiny paired-versus-shuffled displacement effects illustrate a particular danger: their standardized effect sizes are approximately 0.44–0.62 because their person-to-person variation is also tiny. A larger sample might make a 0.05% improvement statistically distinguishable from zero without making it useful. The paper therefore needs an explicit smallest worthwhile improvement as well as a statistical test.

The variation estimates are unstable. Under a normal-difference assumption, eight observations give a 95% interval for the population SD spanning approximately 0.66 to 2.04 times the observed SD. Broader activities, new cameras, real videos and new seeds can add uncertainty that this calculation cannot estimate.

## A concrete confirmatory design

Keep the current eight people as development data. Finish the already planned seed and budget comparisons on that panel to check training stability and investigate the mechanism, but do not relabel it as an untouched test set. Record every method and hyperparameter tried there.

For the final study, freeze one candidate method and the strongest direct-coordinate temporal baseline before evaluating new test predictions. Prefer five predeclared training seeds for these methods and the decisive ablations. Three seeds are a useful constrained minimum, but neither count guarantees adequate precision for training variability. Run all declared seeds, retain failures, and do not select the best seed for the headline result. Equalize training data, coordinate supervision, model capacity where possible, checkpoint selection rules and tuning effort; report update counts and compute separately.

A suitable primary question is whether the selected method improves the error in 0.20-second joint displacement over the direct-coordinate model on a new, person-disjoint panel, while maintaining position accuracy. This preserves a metric already used in the study. Predeclare all-valid reference joints where the reference convention is defensible, a fixed reference scale within each window, a fixed activity/view/condition mixture, and person-balanced aggregation. Evaluate visible and hidden joints separately as secondary outcomes. A displacement result still needs amplitude, waveform and event diagnostics before it can support a general motion-preservation claim.

For a claim that the benefit comes from paired feature learning, the matched shuffled-pairing comparison must also support that interpretation. If the candidate receives coordinate and motion supervision, the primary direct baseline must receive that same supervision; a position-only baseline would confound the feature-learning method with the additional motion targets. If the final intervention changes the readout or teacher, its relevant matched ablation must use the same architecture and optimization budget. Beating affine calibration or an untrained encoder alone does not isolate the value of JEPA.

Freeze the ViTPose checkpoint as the held extractor family if that remains the intended transfer test; its current development results have already been inspected. Fresh-person performance for that fixed family can provide confirmation, while the same eight-person result cannot. Multiple checkpoints or cameras from one family are repeated conditions rather than independent model families.

The proposed real-data target is **40–60 new usable MoVi identities, if access, synchronized annotations and a consistent joint convention permit it**. The dataset's nominal subject count does not establish how many usable people remain after a separate real-data development/QC set and exclusion rules. The dataset audit has not established a person crosswalk between MoVi and BioMotionLab_NTroje or BMLrub. The conservative new training roster therefore excludes BMLmovi and all potentially overlapping BioMotionLab collections, unless verified identities establish separation. Removing BMLmovi alone is insufficient evidence of person independence. Check for other duplicate recordings or aliases across constituent datasets. If MoVi supplies fewer eligible identities, report the actual population and its sensitivity instead of manufacturing a larger sample from multiple cameras. A large independent synthetic panel and a smaller real test answer different questions and should have separate estimates.

## How much data could resolve a useful effect?

For an initial planning approximation, let the standardized paired effect be the mean person-level improvement divided by the SD of those person-level differences. The following exact noncentral-t calculations use a two-sided 5% test under independent normal paired differences. They condition on fixed trained models and omit unknown training-seed and real-transfer variation. [NIST explains the assumptions behind sample-size planning](https://www.itl.nist.gov/div898/handbook/prc/section2/prc222.htm); [Lakens discusses choosing a sample size for an explicit inferential goal](https://doi.org/10.1525/collabra.33267).

| Assumed standardized paired effect | People for 80% power | People for 90% power |
| ---: | ---: | ---: |
| 0.25 | 128 | 171 |
| 0.35 | 67 | 88 |
| 0.50 | 34 | 44 |
| 0.75 | 16 | 21 |

At 40 people the 80%-power sensitivity is approximately 0.454 paired SD; at 60 it is 0.368; at 100 it is 0.283. Those are conditional design sensitivities, not assurances about the eventual experiment. If three extractor-specific tests were all primary and each used a conservative alpha of 0.05/3, an effect of 0.50 SD would require 45 people for 80% power, instead of 34. Predeclaring one main held-family contrast and treating the others as supporting comparisons is usually clearer. [Full standardized table](stats-standardized-sample-size.csv), [fixed-sample sensitivity](stats-fixed-sample-sensitivity.csv).

For scale, use the direct-versus-static variability as an imperfect proxy, and posit a **5% true reduction**, rather than assume the observed improvement will recur. Doubling the pilot SD yields these illustrative results for the held ViTPose family:

| Endpoint | Assumed paired SD, relative to comparator mean | People for 80% power against zero | People for 90% power against zero |
| --- | ---: | ---: | ---: |
| 0.20-second displacement error | 13.63% | 61 | 81 |
| Visible position error | 17.86% | 103 | 137 |

With an assumed true reduction of only 2%, those same scenarios require 367 and 629 people, respectively, for 80% power. A one-week project should not be designed around recovering a very small effect from a limited real-data test. These figures are deliberately conservative relative to this narrow pilot, but they still omit seed variance and do not estimate the variance of a new JEPA-versus-direct contrast. [All scale scenarios](stats-pilot-scale-scenarios.csv).

A 5% true effect powered against zero is different from demonstrating an improvement greater than 5%. The latter requires a confidence bound above 5% and a planning assumption larger than 5%. A practical planning target is therefore 60 eligible independent people when available, with five seeds and an explicit sensitivity statement, followed by reporting the interval obtained. If only 40 people are available, the study can still be useful, with a correspondingly larger detectable effect. Do not claim that either target guarantees statistical significance.

## Specify useful margins before seeing the new test results

One possible engineering target is at least 5% lower displacement error with no more than a 2% increase in coordinate error. These numbers are illustrative design choices, not established clinical thresholds. Set final margins using annotation repeatability, a documented downstream measurement need, and development evidence about the effect's absolute size. Publish both absolute normalized-error differences and relative reductions so a percentage cannot hide a trivial denominator or poor absolute accuracy.

For coordinate noninferiority, define a positive margin for the largest tolerable increase in error and require the prespecified one-sided upper confidence bound to fall below that margin. This establishes only the declared tolerance, not identical accuracy. Require this alongside the primary motion result if the paper claims joint improvement. Set the alpha allocation or gatekeeping rule in advance rather than choosing it after the results.

For the paired-versus-shuffled comparison, a confidence interval that includes zero is inconclusive about equivalence. If practical equivalence is the intended conclusion, specify both equivalence bounds before testing and use the corresponding two one-sided procedure. Bounds such as ±2% require scientific justification; they cannot be chosen because the current differences happen to fit inside them. With alpha 0.05 per one-sided test, the usual interval formulation is containment of the 90% interval within both bounds. [Lakens's original equivalence-testing primer](https://journals.sagepub.com/doi/10.1177/1948550617697177).

## Estimate uncertainty at the correct levels

Create one error value for every person × training seed × method × declared condition. First average equally across the person's retained recordings or windows according to the frozen protocol, then across conditions using fixed weights. Keep every method paired on the same inputs, reference convention, missingness penalties and conditions. Give each person equal influence unless a different target population was explicitly justified.

For the main interval, resample people and training-seed blocks independently, using the same resampled seed indices for every person in a replicate. Retain the paired methods, conditions, cameras and windows together. Resampling a new seed independently for each person would wrongly treat one trained model as many independent fits. If seed labels are paired across methods, define that pairing in advance through the experimental design; sharing an integer label alone does not imply identical stochastic training histories. Deterministic calibration fits do not acquire five independent fits when copied into five seed rows.

This is a crossed design: trained models are evaluated on many people, and each person is evaluated with many trained models. In a simple variance decomposition, uncertainty of the average has person, seed and interaction terms proportional to 1/N, 1/K and 1/(NK). Increasing the person count cannot remove a seed-level uncertainty floor. Crossed resampling addresses that structure, but with only three to five seeds its calibration and tails remain uncertain. Before locking an algorithm-level inferential claim, check the proposed interval's coverage in simulated crossed designs spanning plausible person and seed variance. If coverage is inadequate, retain seed-conditional person intervals and describe the crossed result as a sensitivity analysis. Also show every seed's result and leave-one-person-out sensitivity. Seed repetitions on one fixed training set do not estimate variability across newly sampled training datasets, which would require additional independent splits and refitting. [Owen and Eckles study independent factor resampling for crossed arrays](https://arxiv.org/abs/1106.2125).

Treat datasets and extractor families as named evaluation domains. Two datasets do not provide a stable estimate of variability over all possible datasets, and three selected extractor families do not represent all pose estimators. Report separate domain results, including sample counts, before any prespecified combined average. Standard benchmark splits with only two held people can establish a benchmark score but cannot supply a well-powered general-population person comparison, regardless of their frame count.

For a claim that corruption changes the method's benefit, estimate a paired method-by-condition interaction directly: for each person and seed, subtract the clean-condition method difference from the corrupted-condition method difference. Use the same reference joints and scale policy where possible. A significant effect in one condition and a nonsignificant effect in another is insufficient evidence that the effects differ. [Gelman and Stern explain this distinction](https://doi.org/10.1198/000313006X152649).

## Coverage, motion events and stopping

Predeclare reference eligibility independently of predictions. Use only anatomical references whose projection, joint names, timestamps and visibility conventions have passed review. Keep sample flow counts from downloaded identities through eligible subjects, sequences, windows and joints, with specific exclusion reasons. Reviewers who assess reference quality should not choose clips on the basis of which method looks best.

Missing predictions remain failures in coordinate and displacement scoring. Report the rate separately, and keep the same declared population across methods. Unsupported amplitude or timing outcomes remain unavailable rather than zero. A supported-only result is conditional on its retained population and must be labeled that way. For nearly stationary reference windows, RMS ratios become unstable; choose the minimum reference amplitude rule before evaluation and report how many windows it excludes.

Event timing needs reference-event recall, predicted-event precision, extra and missed counts, and timing error conditional on matched events. Report person-balanced values as the inferential results and event-pooled totals descriptively. A method that deletes true peaks can achieve attractive precision or matched-event timing. The current ankle-separation peaks should remain labeled operational 2D peaks, without a heel-strike or clinical interpretation unless separately validated.

Freeze an eligibility rule, a deterministic subject selection order, a target sample and a maximum effort or deadline before opening test results. Screen for data quality without looking at method differences. If the project is resource-limited, stop at its declared bound and report the achieved precision; do not add subjects or seeds until a p-value crosses 0.05. Any sequential testing requires an explicit method and spending rule decided beforehand. Log failures and reruns, with reruns restricted to documented technical causes.

Choose one primary inferential contrast. Use a declared hierarchy or multiplicity adjustment for the small number of confirmatory secondary contrasts, and label the remaining condition, lag, amplitude and feature analyses exploratory. Save all prespecified results, including null and unfavorable ones.

## Two defensible paper outcomes

**A method result:** the final model improves the prespecified motion endpoint over a strong matched direct-coordinate model on new people, preserves position accuracy within the declared margin, and the pairing ablation supports the feature-learning mechanism. Real-data replication supports a transfer claim only to the evaluated dataset and annotation convention.

**A diagnostic result:** the accuracy-versus-motion tradeoff recurs on independently reviewed data and fresh people, with identified conditions that cause it. A strong version preregisters a joint claim such as lower coordinate error but higher amplitude distortion for a particular coordinate-trained model versus a particular calibration or filtering baseline, and requires both directional intervals to support it. The figures should display the joint distribution of coordinate and motion effects by person rather than selecting different winning methods for different metrics. An equivalence result about this specific pairing recipe can strengthen the diagnosis if its bounds are justified; a large p-value cannot establish that all JEPA approaches are ineffective.

Neither outcome is made novel by its p-value. The paper's contribution depends on whether the controlled comparison identifies a reproducible limitation or a reproducible solution, using credible references and a clear boundary on what was evaluated.

## Reproduce this planning analysis

```bash
/private/tmp/gavd6-stv2-cpu/bin/python \
  docs/studies/synthetic-training-v2/research/paper-development-20260919/stats-sensitivity.py
```

The script requires NumPy, pandas and SciPy and accepts `--source` and `--output` for other locations. The temporary interpreter is the existing local analysis environment, not a portable HAIC instruction. Outputs start with `stats-`; the experiment data remain read-only. Its checks compare all person differences against the independent balanced metric table, verify all 60 retained aggregate contrasts, check the null rejection probability and known noncentral-t calculation, and confirm that source hashes have not changed.
