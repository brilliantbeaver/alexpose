# Role

You are the research lead for an AI/ML study of gait observations for medical world models. Coordinate specialists in data, representation, modeling, evaluation, literature, teaching, visualization, and independent review. Use only the specialists needed for the current stage. Read the repository instructions and `.agents/README.md`, assign exclusive file ownership, and preserve other contributors' work.

# Goal

Develop a convincing, reproducible contribution to gait anomaly detection that is motivated by movement physiology and survives strong simple alternatives. The central question is: **Can a model recognize departures from normal movement organization beyond what is explained by motion amount, current posture, camera view, and pose quality?** Improve the evidence for this question before increasing model size. Acceptance is an aspiration, not a measurable probability or a reason to select favorable outcomes.

The [Med-WM workshop](https://med-wm.github.io/aaai27-mwm/) includes observation dynamics, uncertainty, validation, and benchmarks. Position this work as a tested movement-observation component. Short video forecasts do not establish disease progression, treatment response, or action-conditioned patient simulation.

# 1. Start from the evidence and declare the new study

Read `experiments/aaai-worldmodels/docs/draft01/`, the source notebooks and their executed copies, `docs/restart/`, `worldmodels/restart_data.py`, `worldmodels/restart_experiment.py`, and the completed `restart-full10hz-v1` and `restart-dynamics-v1` runs. Then read `docs/draft02/`, `docs/iteration02/literature_and_masking.md`, and notebook `16_iteration02_evidence_and_mask_design.ipynb`. The latter is an evidence and design laboratory, not a completed masking comparison. Define `PROJECT = experiments/aaai-worldmodels`. Paths beginning with `docs/`, `worldmodels/`, `artifacts/`, `scripts/`, `tests/`, and notebook filenames are relative to PROJECT unless given in full. Paths beginning with `data/`, `ambient/`, `notes/`, or `.agents/` are repository-relative. Discover the repository root from the checkout; do not hard-code a different user's home directory.

Preserve the original notebooks, draft01, raw pose caches, predictions, checkpoints, and all run identities. Create a new method identity, such as `medwm-reference-masking-v2`, with an immutable protocol, code revision, annotation and media hashes, pose-weight hash, grouping table, folds, seeds, mask policy, objective, and analysis version. Give post hoc analyses of existing predictions their own analysis identities. Do not silently substitute a new result into an old artifact.

The current evidence establishes the following starting point.

| Observation | Consequence for the next study |
| --- | --- |
| The evaluated cohort contains 2,815 windows, 456 sequences, and 91 source videos, including 29 normal sources. | Independent-source coverage constrains inference; window count is not patient count. |
| Bilateral prediction reduces normal-source error by 33.5% relative to persistence, from 0.12126 to 0.08063. | The observations support a useful short-horizon learned predictor. |
| A learned transition from current posture has error 0.07164. | Current posture is a mandatory strong baseline. A history encoder must earn its extra complexity. |
| The paired bilateral advantage over own history is 0.00111, with a conditional interval spanning zero. | Opposite-leg temporal information has not yet shown a reliable advantage across all relevant controls. |
| The predeclared excess-error score has source AUROC 0.793. | Referencing error to movement difficulty is a useful methodological question. |
| Negating persistence error gives a post hoc low-motion comparator with AUROC 0.806. | A favorable excess-error AUROC does not establish improvement beyond motion amount. Retain this comparator openly. |

The last comparison is a diagnostic identified after inspecting outcomes. It does not change the original score direction or acquire preregistered status. The existing intervals condition on fixed overlapping cross-validation fits. Earlier GAVD results influenced design, so all reuse of these data remains exploratory, including a newly named run. A fresh random split does not make the same sources untouched.

Historical JEPA classifier scores use different cohorts, tasks, and label access. Notebook 06 includes supervised probes and condition adaptation, whereas the present forecasting study fits normal sources only. Do not compare their headline numbers as a single leaderboard or describe supervised adaptation as unsupervised anomaly detection. Small demonstration fits and synthetic notebook outputs are teaching evidence, not final experiment results.

# 2. Define the contribution and falsifiable hypotheses

Pursue **reference-aware prediction of gait organization** as a working research direction. Its potential contribution is the joint evaluation of a conditional movement model, structured observation masking, and an anomaly score that must outperform explicit nuisance baselines. Normal-only prediction, skeletal forecasting, semantic tubes, and motion-aware masks already have precedents. Their combination is a proposal to test, not established novelty.

Freeze four hypotheses before fitting new candidates.

1. A compact learned transition improves normal-source forecasting over persistence and a current-posture transition under identical target support.
2. Structured masking improves performance beyond uniform masking when model, objective, visible-token budget, training exposure, and evaluation targets are held constant.
3. A normal-calibrated score improves abnormality ranking beyond the locked low-motion baseline, `-E_persistence`, within the same held-out folds. The candidate must also be compared with normal-only static-posture, prefix-motion, and missingness controls.
4. Any claimed coordination effect persists under matched time-order, observation-quality, and camera-view controls, and is replicated on an independent cohort when access permits.

These hypotheses are separable. Better forecasting does not imply better anomaly detection. A masking gain does not identify neurological circuitry. A strong annotation association does not establish useful forecasting. Report a failed hypothesis precisely and retain the successful, narrower contribution if the evidence supports it.

The most promising low-cost alternative to a large temporal model is a compact state containing current posture, a robust recent velocity estimate, and uncertainty about that estimate. Compare it with the existing current-posture transition and a history encoder. Only add phase or stochastic latent variables after the data support their estimation.

# 3. Use the full annotation inventory responsibly

Rebuild the inventory from `data/gavd/<literal-label>/*.csv`. Exclude `data/gavd/source/` from cohort counts because it is the pinned provenance copy. Verify against `data/gavd/README.md` and `source/manifest.json`. The full local inventory has 1,874 sequences, 458,116 annotation rows, and 348 source IDs. The five requested labels contain 666 sequences and 103 source IDs. Treat the repeated stroke entry as one condition.

Use all eligible normal, parkinsons, stroke, cerebral palsy, and myopathic sequences after documented identity, duration, timing, and observation gates. Report a flow from catalog to media, extraction candidates, usable observations, and common model support, with counts by label, source, sequence, window, and view. Preserve known reference mismatches and whole-source exclusions. Compare accepted and rejected material in a blinded, source-stratified visual audit. Record review coverage and unresolved identity cases.

All 32 annotated normal source IDs already fall within the five-label scope, and 29 survive the current model cohort. Processing the other seven categories cannot add normal-source support. Broader GAVD use should answer a declared transfer or specificity question. Partition its remaining labels before acquisition or fitting: pathological labels with adequate annotation support can form an additional abnormality evaluation; exercise, pregnancy, stylized walking, prosthetic use, and intoxication must retain their actual meanings. Do not automatically label every unusual movement as neurological disease.

Audit the one source that spans multiple full-inventory labels before extending the loader. The current `assign_splits` rejects mixed-label sources. Group all linked observations together and define how mixed sources enter each endpoint. Likewise, the current video lookup and pose preparation select five labels; do not claim full-dataset processing without extending and testing those boundaries in the new study. Keep ambiguous labels as an explicitly described challenge set rather than resolving them to improve a score.

Build a source-relation graph using source IDs, file hashes, mirrored uploads, known repeated participants, and manually verified visual evidence. Keep connected components together. Metadata and appearance can suggest duplication but do not prove identity. Unknown participant relationships remain a limitation. Do not use an external folder as an independent dataset if it contains copies of the same GAVD URLs. Audit `data/videos-ms` and similar caches by source identity before considering transfer.

Fix coverage and stopping rules before observing new outcomes. Complete the declared eligible inventory or stop for documented resource or access limits. Do not add videos, windows, seeds, or conditions until a p-value crosses a threshold. Estimate interval precision at the source or verified-participant level. Simulations may quantify precision under explicit effect assumptions, but cannot manufacture independent sample size.

# 4. Reuse and verify the Ambient ingestion boundary

Use the current checkout's `ambient/` package for shared video and pose interfaces. The existing experiment adapter in `worldmodels/pose.py` already calls `ambient.utils.youtube_cache.download_youtube_urls`. Its output must remain under `data/videos-gavd`.

```text
data/videos-gavd/
  sources/<source_id>.<extension>       New originals, one source copy.
  previews/<source_id>/                Derived previews and contact sheets.
  manifest.json                       Source, URL, hashes, availability, and exclusions.
  <existing-condition>/<source_id>.*   Existing originals reused in place.
```

Do not duplicate originals into condition folders. Reuse verified existing paths, and record failed, private, missing, or mismatched media. Downloading is not proof of correct content or annotation alignment. Keep media and bulky outputs local.

The current restart decodes each selected source sequentially and calls `ambient.pose.keypoint_extractor.SequenceKeypointExtractor.extract_from_image` on RGB frames. Preserve this shared interface. GAVD annotation frames are one-based. The source decoder index is zero-based. Convert exactly once and retain both values. Do not pass a second conversion through an image API. If calling the public sequence or video-frame API instead, respect its one-based arguments.

Retain the declared BlazePose joint order `(11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32)`, pixel-space input, decoded timestamps, nominal FPS, image dimensions, confidence, original missingness, source ID, sequence ID, and model hash. Do not reuse COCO indexes by position. The prediction targets are the two sets `(25, 27, 29, 31)` and `(26, 28, 30, 32)`.

Preserve complete eligible annotation spans, nominal 10 Hz nearest-native-frame sampling, 16 past observations, eight future observations, and stride eight for the main comparison. The prefix covers 1.5 seconds between endpoints. The primary error averages nominal future offsets 0.1 through 0.8 seconds; it is not an error only at 0.8 seconds. Retain actual times and quantify timing deviation. Never call interpolated or imputed joints measured observations.

The current contract uses confidence at least 0.5, at least 70% observed prefix and future joints, and common distal-target support. Those are study settings, not universal Ambient defaults. Pelvis centering is per frame; torso scale is estimated from the prefix only. This produces relative 2D articulation, not physical gait speed or measured anatomical angles. Both hips already enter the reference. An own-leg comparison therefore conditions on a shared body reference.

Keep model inputs separate from target availability. At deployment, predict a fixed set of future queries from the prefix without consulting future confidence, target visibility, or outcome-dependent gates. Apply future observation support only when computing loss or evaluation. Report abstention from prefix quality separately from offline target availability. Preserve the same eligible target support across all methods.

Add tests that alter every future coordinate and future validity bit while keeping the prefix fixed. Predictions, mask choices, normalization, anchors, and selected model settings must remain unchanged. Recomputing a future-relative target frame is permitted for evaluation; using it to condition the predictor is not.

# 5. Connect physiology to observable tests

Explain each neurological idea in ordinary language before defining a model. Walking coordinates many joints, and impaired movement can become more variable, more stereotyped, or systematically compensated. Therefore, both unusually large and unusually small prediction errors can be informative relative to a suitable normal reference. This is a motivation for two-sided analysis, not a claim that every disease has the same signature.

| Motivation | Observable question | Required control and claim limit |
| --- | --- | --- |
| Bilateral timing can be impaired in Parkinson's disease. | Does correctly aligned opposite-leg history add information at the same current posture? | Compare own history, a separately fitted window-specific time shift, current posture, and view strata. No basal-ganglia inference is justified. |
| Stroke can alter muscle coordination and produce asymmetric compensation. | Are left/right conditional errors or coordination profiles unusual relative to normal reference variation? | Preserve side labels and camera orientation; do not infer affected side without verified annotation. Video coordinates do not measure EMG modules. |
| Cerebral palsy can alter selective motor control and muscle coordination. | Are multijoint residual patterns or phase-dependent departures reproducible across sources? | Age, assistive devices, view, and observation quality can differ. Small support limits condition-specific claims. |
| Myopathic gait can reflect weakness and compensation. | Do proximal reference motion and distal prediction errors show distinct patterns? | Myopathy is not exclusively a central neurological disorder. No muscle-strength estimate is available from these videos. |
| Normal walking varies with speed and phase. | Does a conditional normal reference distinguish reduced movement from disrupted organization? | Use measurable 2D motion amplitude and validated phase proxies; do not rename them physical speed or heel strikes. |

Use the verified primary studies and limits in `docs/iteration02/literature_and_masking.md`. If gait events are inferred, validate them against blinded manual labels on a preselected subset, report timing error and failure rates, and freeze the estimator before evaluating disease association. Do not infer a cycle from less than adequate evidence. Require at least two credible prefix cycles for a phase-based arm, report its eligible population, and compare all controls on that same subset. Longer prefixes are a separate sensitivity study because they change coverage. A label such as gait phase remains a proxy unless measured events support it.

# 6. Test masking as a controlled design choice

Read primary work on VideoMAE, I-JEPA/V-JEPA, MAMP, Skeleton S-JEPA, SLiM, and ASMa. Follow the verified versions in the literature review, including the revised SLiM paper. Semantic/anatomical tubes, masked latent prediction, and motion-aware masking are established techniques. Action-recognition improvements do not establish gait-anomaly improvements. Distinguish a faithful reproduction from an adaptation to sixteen sparse joints and a small normal-source cohort.

Separate three operations: an observation can be naturally missing, an observed prefix token can be artificially hidden during training, and a future query can be predicted. Store these masks separately. The deployment task always hides the entire future suffix. Random reconstruction of observations on both sides of a gap is a different task and cannot be reported as forecasting.

For the main experiment, keep the future target set, target weights, and horizon fixed. Vary only additional masking of the observed prefix during training. At evaluation, use the full observed prefix for the primary comparison and a fixed set of shared corruption patterns for robustness. Do not score each policy only on the targets it finds easiest.

| Arm | Exact proposed intervention | Scientific reason |
| --- | --- | --- |
| M0: suffix only | Hide all future samples and retain every eligible prefix token. | Establish the direct forecasting reference. |
| M1: uniform prefix | Hide a fixed fraction of eligible observed prefix tokens without replacement. | Supply the matched random-mask control. |
| M2: time blocks | Hide contiguous spans of non-reference prefix tokens, using the same hidden-token budget. | Reduce reliance on adjacent-frame interpolation and test short tracking gaps. |
| M3: anatomical tubes | Hide a connected distal limb across a contiguous prefix span, with the same hidden-token budget. | Test whether another body region supplies useful information when local observations are removed. |
| M4: motion-balanced | Allocate the same hidden budget between high- and low-motion prefix tokens, computed from observed prefix differences only. | Prevent quiet motion or noisy fast tokens from dominating the task. |
| M5: phase-stratified, optional | Hide matched tokens across validated prefix-phase bins. | Test coordination coverage after the phase-estimation prerequisite is met. |
| M6: learned or adversarial, optional | Learn a constrained mask policy on normal training sources only, with fixed budget and coverage constraints. | Test adaptive task difficulty only if simpler masks leave a clear unresolved question. |

Start M1–M4 at the fixed rate of 40% of eligible non-protected tokens. Use deterministic rounding and report actual counts. M0 deliberately has more visible input and is an uncorrupted anchor, not a token-budget-matched masking arm. Protect the reference joints used for pelvis centering and torso scaling and retain the most recent observed target-joint anchor. If an implementation hides reference joints, compute its normalization from the truly visible support and treat it as a separate representation change.

For M3, add a paired control that uses random joint groups of the same sizes, interval lengths, and realized token counts. This separates anatomical grouping from mask geometry. A MAMP-inspired motion-focused policy is a secondary comparator for M4, with the same budget and a nonzero uniform sampling component. Define exact-budget handling before execution. A final block or tube may need truncation when observations are missing; record this explicitly and name such masks budget-matched block/tube priorities rather than claiming every mask is a complete tube. Report both nominal rate and realized eligible-token rate. Keep a shared window list across masks, with predeclared handling of unsupported windows. Draw source, sequence, and window hierarchically so long videos do not dominate training.

Motion-balanced masking should use confidence-qualified, timestamp-aware differences from the prefix. Quantile ties need a deterministic rule. Do not treat missing motion as zero motion. Use bounded sampling probabilities so every eligible region can be selected. Match mask counts, anatomical coverage, and observation-quality distributions where possible. The choice of which token to hide can reveal information about the original observation; therefore include a mask-pattern-only diagnostic and an equally budgeted randomized-policy control. Never derive masks from class labels, future motion, held-out errors, or target confidence.

Learned masks require extra optimization and can hide easy regions or focus on pose failures. Count their compute in the comparison, constrain region coverage and entropy, and detach target-outcome gradients from the policy. Report results with both equal training exposure and measured compute. Do not describe Gaussian time jitter as a reproduction of a learned stochastic-position method.

Keep mask randomness, temporal-position noise, and a stochastic future-state variable as different factors. First establish deterministic baselines. If adding stochastic futures, sample only from a prefix-conditioned prior at evaluation. A future-conditioned posterior may assist training but may not enter prediction. Measure a proper probabilistic score, interval coverage, and sharpness; best-of-many error and raw sample spread are insufficient evidence of calibrated uncertainty.

# 7. Use a small, interpretable experiment ladder

Stage A reanalyzes the frozen source predictions and audits nuisance explanations. It requires no retraining. Reproduce all displayed figures, inspect each source, compare within-fold ranking, and quantify the paired difference between excess error and low motion. Keep the historical directions and identify new diagnostic directions as post hoc.

Stage B fits simple reference models on the exact same new grouped folds. Include persistence, damped velocity, own-history ridge, bilateral ridge, the independently fitted matched shift, current-posture ridge, and a compact posture-plus-velocity model. Add normal-only distance models on static pose, prefix motion amplitude, and observation quality. A supervised Random Forest is a separate label-assisted benchmark with its own inner selection, not a normal-only competitor.

Stage C changes masking with one compact deterministic model and one fixed objective. Prefer a residual forecast head anchored to the latest observed articulation. Begin with M0–M4 and a small declared seed set, such as 27, 41, and 73. Seeds characterize optimization sensitivity; they do not multiply the number of subjects. Select training duration and capacity inside permitted normal development data. If high variance makes the comparison inconclusive, report it instead of running a larger search against abnormal labels.

Stage D tests representation objectives only after the masking comparison is interpretable. Compare coordinate-only prediction, latent prediction plus a coordinate head, and a matched regularization ablation using the same backbone, masks, optimizer budget, and data. The current SkeletonJEPA implementation is an adaptation with teacher targets and auxiliary losses; do not attribute any result to an official S-JEPA reproduction. Quantify latent variance, effective rank, teacher drift, feature sensitivity, and performance of an untrained encoder. Low latent loss alone can reflect collapse or moving targets.

Stage E evaluates one locked candidate externally and performs robustness tests. Stage F considers phase-conditioned or stochastic models only when earlier stages establish a reason. This sequence avoids an unmanageable Cartesian product of masks, objectives, noise types, context lengths, and cohorts.

Measure one complete pilot fold before scaling. Record wall time, parameter count, memory when available, optimizer updates, effective observed tokens, and all failures. Limit concurrent heavy jobs to one unless resource measurements justify more. Set resource ceilings before launching a grid. Stop a failed fit for numerical divergence, empty support, or invalid provenance, record it, and fix the cause under a new run identity where necessary.

# 8. Define the anomaly score and its calibration before testing

Let `E_model` and `E_persistence` be errors on identical observed future coordinates with identical hierarchical weights. Preserve the historical score `E_model - E_persistence`. It describes how the learned forecast compares with retaining the last pose. It is not a disease-specific normalization. Its favorable ranking must be read beside `-E_persistence`, which measures low observed displacement from the anchor.

For the new candidate, define the scalar score before implementing calibration. First compute `E_model,s`, the source-average future MSE using the declared coordinate/leg/window/sequence/source hierarchy. Compute prefix motion amplitude from adjacent valid distal-joint differences divided by their actual time gaps, and aggregate it with the same source hierarchy. Define `z_s` as two values: log prefix motion amplitude and prefix missingness. Use a fixed numerical floor of `1e-8` before logarithms; retain a floor-sensitivity analysis. These are observed 2D quantities, not physical walking speed.

Use out-of-fit normal development predictions to fit a small ridge reference `g(z_s)` for `log(E_model,s + 1e-8)`. Fit standardization and ridge strength using only normal development groups. Define the scalar residual `r_s = log(E_model,s + 1e-8) - g(z_s)`. This residual means that a source was easier or harder to predict than expected from its observed prefix motion and missingness. It is neither a signed coordinate residual nor the historical excess-error score.

Build the residual reference from normal sources held out from both the forecast fit and the nuisance-reference fit. Use an explicit nested cross-fitting schedule and save those memberships. Let `F_cal` be its smoothed empirical distribution, with midranks and endpoint smoothing declared in code. Freeze the proposed two-sided anomaly score `A_s = abs(2 * F_cal(r_s) - 1)`. Larger values indicate either unusually easy or unusually difficult prediction relative to the normal reference. This cross-fitted score is exploratory and has no automatic conformal guarantee. If data are insufficient for this nested reference, report the comparison as infeasible and retain the marginal normal-reference sensitivity rather than fitting on the same source outcomes.

The primary comparator is fixed as `-E_persistence,s`, the low-motion score already exposed by the current audit. It is a post hoc diagnostic in draft02 and a declared baseline in the next protocol. Secondary controls are normal-only distance from current posture, prefix motion alone, prefix missingness, and marginal calibration without nuisance conditioning. Adding camera view or a posture summary to `g` is a separately declared sensitivity because 29 normal sources cannot support many covariates. All scoring and aggregation choices must be frozen before inspecting new abnormal outcomes.

Normal calibration can reduce fold-scale differences but cannot guarantee removal of recording confounding. Report both calibrated pooled scores and within-fold comparisons. Keep marginal and conditional calibration as separate arms. If the conditional reference is unstable, retain the simpler marginal reference and state why. Do not advertise cross-fitted empirical ranks as exact conformal coverage.

Thresholds require a genuinely separate normal calibration mechanism. With `n` exchangeable normal calibration sources, split-conformal resolution is `1/(n+1)`. At a 5% error target, fewer than 19 such sources cannot yield the usual finite distribution-free cutoff; even at 19, the estimate is coarse. The current 29-normal-source total must also support training and evaluation. Report unavailable thresholds or infinite cutoffs honestly. Repeated windows cannot repair this limitation. Threshold-free ranking is the primary internal anomaly endpoint; clinical operating points require stronger calibration data.

Inspect whether candidate improvement survives comparison with motion amplitude, persistence error in both documented directions, static pose distance, prefix missingness, view, and a nuisance-only model. An offline anomaly score can use the observed suffix after it arrives. A prefix-only early warning score is a different endpoint and must not receive that suffix.

# 9. Freeze evaluation and generalization requirements

Use the source-relation component as the minimum grouping unit and verified participant identity when available. Keep every clip, window, visit, and duplicate in its group. Fit transforms, imputers, reference distributions, hyperparameters, thresholds, and learned masks on permitted training/development groups only. Release abnormal outer labels to the evaluator after candidate selection is frozen. Even then, historical exposure makes the GAVD study exploratory.

Preserve five outer folds and three inner folds for a direct continuation where feasible, but recalculate feasibility if deduplication merges sources. Do not force a fold design that lacks normal training or evaluation support. For every fold, publish actual normal and abnormal source counts and fit membership. Common target support is mandatory. Average coordinate errors within each leg, legs within windows, windows within sequences, and sequences within sources. Use component-level aggregation when a component contains several linked sources and declare its weighting.

The main new anomaly endpoint is the paired within-fold AUROC difference between the locked two-sided candidate and `-E_persistence,s`. Define the aggregate in advance as the number-of-positive-negative-pairs weighted average across folds. Also report equal-fold averaging, each fold's counts and scores, pooled calibrated AUROC, average precision with prevalence, and all named low-motion/static/missingness controls. Do not choose the baseline after seeing outer outcomes. Condition-wise analyses are exploratory, and they share the same normal reference group.

Use paired stratified resampling of independent groups for uncertainty. For within-fold estimates, preserve fold and label strata. State when intervals condition on saved fits. Separately quantify retraining variability with predeclared repeated grouped partitions or complete pipeline resampling if feasible. Repeated partitions and seeds remain correlated evaluations, not independent datasets. If testing a family of masking hypotheses, name the primary contrast and use a declared adjustment, such as Holm, for confirmatory claims. Otherwise report exploratory intervals without significance language. No adjustment repairs historical test exposure.

Use leave-view-out or acquisition-matched analyses only when every compared subset has adequate class and source support. Audit whether camera angle, framing, cadence proxy, resolution, confidence, and retention explain the score. Reweighting or matching can change the estimand and cannot correct unmeasured confounding; publish the unmatched result and coverage loss as well. Synthetic occlusion, noise, temporal shifts, and frame drops test robustness and model sensitivity. They are not neurological interventions or independent clinical cohorts.

External evaluation needs a written compatibility contract: participant IDs, clinical labels, consent/access terms, overlaps with GAVD, camera conditions, joint correspondence, units, time grid, and missingness. Freeze source-trained weights, mapping, and score before examining external labels. Any target-domain calibration uses a separate normal participant set and is reported as adaptation. A common-joint reduced model must be trained and evaluated under its own identity on both domains. Do not invent missing heel or foot-index joints to force compatibility. A marker-based or wearable dataset can test a movement hypothesis while failing to validate a video pose pipeline; label its role accurately.

Put this explicit conclusion in the output README: **GAVD supports a heterogeneous public-video benchmark, but GAVD alone does not establish patient-disjoint, site-disjoint, or prospective clinical generalization.** Source IDs are not patient IDs, labels are not independently confirmed diagnoses, and the normal-source ceiling remains small even with the full annotation inventory. More data are useful when they add independent coverage, not when they merely multiply correlated windows.

# 10. Build progressive, executable notebooks

Preserve notebooks 00–15 and the audit notebook 16. Build the next study from source generators with fresh notebook numbers. Each notebook must explain the question, inputs, units, computation, result, and interpretation in complete connected sentences. It must run from its own clean kernel and fail clearly when a required artifact is absent. Separate a quick demonstration from the full run command. Never replace missing real outputs with synthetic performance.

| Notebook | Required computation and teaching output |
| --- | --- |
| 20: Cohort, identity, and coverage | Rebuild all-label inventory, reconcile duplicate/source components, display attrition, and record external-data feasibility. |
| 21: Reference scores and nuisance controls | Recompute frozen scores; visualize motion versus error; fit normal-only calibration within development groups; explain score direction and threshold resolution. |
| 22: Masking laboratory | Display real valid/hidden/visible/query masks, count tokens and anatomical coverage, demonstrate future invariance, and show matched-budget policies. |
| 23: Models, objectives, and fitting | Fit the declared compact references and candidates; show source-balanced sampling, optimization health, collapse checks, and full run identities. |
| 24: Grouped and external evaluation | Generate per-source predictions, paired within-fold endpoints, confidence intervals, calibrated pooled scores, and transfer results or explicit access limitations. |
| 25: Robustness and movement interpretation | Run matched time controls, view/quality analyses, corruption tests, and condition profiles; separate observations from proposed physiological explanations. |
| 26: Manuscript evidence | Rebuild every table and figure, map each claim to artifacts, expose failed hypotheses, and verify slide/manuscript agreement. |

Have notebook 22 include one source frame decoded through Ambient, but reuse immutable full-run caches for the large study. Notebook 23 must distinguish real training from simply loading a checkpoint. Notebook 24 must display uncertainty at the independent-group level. Notebook 26 must show the complete candidate registry so unsuccessful arms cannot disappear from the narrative.

# 11. Write the paper and slides around the supported result

Use `docs/draft02/` for the standalone current manuscript and preserve its measured-versus-proposed boundary. Its contribution can be a careful anomaly-detection benchmark showing why forecast errors need a reference and strong controls. Keep the favorable forecasting result, the predeclared excess-error association, the stronger current-posture forecast, and the post hoc low-motion comparison together. Do not claim that a novel learned detector beats existing methods unless a matched evaluation demonstrates it.

For a later fitted iteration, create a new manuscript evidence identity. Lead with the scientific question and the answer actually measured. Explain persistence as retaining the most recent pose, masking as hiding selected observations, and a latent representation as a learned summary. Define AUROC and mean squared error on first use. Avoid acronym chains, isolated sentence fragments, false dilemmas, em dashes, and repeated cautionary slogans. Captions must identify the population, units, direction, sample size, and meaning of uncertainty without relying on the main text.

Revise the current slide entry point under `experiments/aaai-worldmodels/slides/` while preserving older decks. Use editable vector diagrams for the ingestion boundary, prefix/future timeline, anatomical joint mapping, mask alternatives, calibration flow, and nested group splits. Show measured results separately from proposed experiments. Include a table comparing what each neurological motivation permits us to measure, a figure showing nuisance controls, and a practical notebook roadmap. Generate all numeric figures from declared evidence files and inspect every rendered page at full size.

Do not present prospective masking gains as results or erase relevant controls to improve the narrative. A transparent methodological contribution with bounded positive evidence is stronger than an unsupported claim of disease understanding.

# 12. Use independent adversarial review and explicit completion gates

The coordinator owns the protocol, run ledger, and integration. A data specialist owns cohort identities, a model specialist owns training, an evaluation specialist owns frozen endpoints, a literature specialist owns primary-source claims, and a visualization specialist owns editable diagrams. Assign one writer per overlapping path. Reviewers must not review their own authored artifacts as independent work.

Before scaling, have the reviewer try to break future isolation, source grouping, common support, weighting, normalization, and mask-budget equivalence. Have them challenge whether motion amount or view explains every proposed improvement. Check that a fixed global feature permutation has not been mislabeled a temporal control; a linear model can absorb it. Validate a complete small execution before using large artifacts.

Before publishing a result internally, independently reconstruct source scores from window errors, verify all artifact hashes, compare checkpoint predictions with saved forecasts, and reconcile every displayed count. Check text against figures, captions against axes, and the main manuscript against the complete appendix. Verify bibliography titles and claims from primary sources. Render the manuscript and slides, inspect legibility, and fix clipping, overlaps, missing labels, and broken links.

Record issues with severity, evidence, exact remedy, and disposition. Accept only the exact final hashes that were reviewed. Any material edit reopens the relevant review. Use tests for scientific invariants and changed interfaces; do not count an import check as a full experiment. Preserve runtime failures and unverified prerequisites explicitly.

Completion requires a runnable notebook chain, immutable protocol and run manifests, full source-level evidence, a plain-language README addressing generalizability, an accessible manuscript, editable and visually checked slides, and an independent review with resolved blocking issues. If external data or a prerequisite cannot be obtained, complete all independent work and state the resulting claim boundary. Never substitute invented observations, expected gains, or an unexecuted model for missing evidence.
