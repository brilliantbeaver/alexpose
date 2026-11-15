Act as an independent, demanding ASTAD workshop reviewer and research advisor. This is a real research manuscript and completed data. Use no tools. Find mathematical, methodological, reporting, clinical or novelty errors. Do not invent advantages, results, identities, diagnoses or acceptance probabilities. Give severity, exact offending claim, why it matters, and a concrete edit or small feasible analysis. Score novelty, rigor, evidence strength, clarity, workshop fit and technical submission readiness from 1 to 5 with reasons. Differentiate an honest workshop audit/negative-result contribution from a strong new detector paper. Assess whether the current central message is coherent and useful despite inconclusive results, and how to improve the manuscript without changing the primary analysis after seeing test outcomes. All 23 retained normal sources overlap earlier development. Four outer source-separated folds; five reference normals per fold. 512-head updates primary; 4096 retained sensitivity motivated by normal validation before test inspection. All twelve notebooks have actually been run on real GAVD videos and model outputs; synthetic demonstrations remain clearly separate. No clinical expert reviewed the clips and no published detector is successfully reproduced. Critique the actual claims, not a hypothetical inflated paper. Plain full sentences, no em dashes. Conclude with prioritized revisions feasible before November 20. Your raw advice will be archived and independently checked before adoption.

FILE docs/draft01/manuscript.md
---
title: "Auditing Predictive Context for Gait Anomaly Detection"
abstract: |
  Does the rest of the body help predict a hidden joint differently in normal and non-normal gait recordings? Neuromuscular findings motivate this question, but predictive relationships in video are not direct measurements of neural control. We audit a skeleton joint-embedding model using matched own-joint, local-interval and full-body contexts, coordinate and latent targets, and source-separated normal references. The refreshed GAVD cohort yields 8,950 eligible windows from 251 source videos, including 23 normal sources. In four source-held-out folds, the latent predictive-benefit profile achieves AUROC 0.641 versus 0.541 for raw full-context error; their paired difference is 0.101 with a conditional 95% interval of [-0.035, 0.224]. Full-error reference calibration reaches 0.673, and a random encoder with fitted heads reaches 0.651. Longer head training does not establish a learned-encoder advantage. Reference omissions, observation budgets and contextual teacher targets materially affect interpretation. Because all retained normal sources overlap earlier development, these results are exploratory. The study provides a reproducible audit of a plausible anomaly explanation, while leaving its neurological validity and benefit beyond simpler controls unestablished.
---

# Introduction

An unusual walking pattern need not be difficult to predict. A repeated compensation may be highly regular, while ordinary walking observed from an unfamiliar camera may produce substantial prediction error. This distinction motivates a different question: how much does the rest of the body help predict a hidden joint, compared with that joint's own visible history? Consider a hidden left-ankle interval. Its prediction may change when the model additionally sees the opposite leg, arms and trunk. Either unusually weak or unusually strong predictive help could distinguish an observed pattern from a normal reference.

We investigate this question through matched skeleton-completion tasks. A normal-trained representation supports three information conditions, with restrictions enforced before contextual encoding. We compare their prediction losses, summarize the differences across body regions, and measure distance from a reference estimated using separate normal sources. Observable-coordinate targets and contextual latent targets distinguish movement prediction from representation prediction. The intended contribution is a controlled empirical assessment of this diagnostic, including its failure conditions, rather than a new skeleton architecture or a clinical model of disease.


\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/method.pdf}
\caption{Matched completion tasks hold the target and mask fixed while changing available observations before encoding. A normal-reference distance summarizes three regional loss differences. This is an operational diagnostic of fitted predictors, not a physiological or causal model.}
\label{fig:method}
\end{figure*}

# Related work and biological motivation

Skeleton S-JEPA learns action representations by predicting teacher embeddings from partial skeletal observations [@abdelfattah2024sjepa]. MAMP instead emphasizes masked motion prediction [@mao2023mamp]. Both motivate the prediction task, but their motion-aware masking differs from our uniform selection. VICReg supplies agreement, variance and covariance regularization [@bardes2022vicreg]. Normal-only gait reconstruction and prediction already appear in FSGait [@duan2024fsgait]. SeeKer estimates conditional keypoint densities and incorporates prediction confidence [@delic2025seeker], while STG-NF models pose-sequence likelihoods using graph-based normalizing flows [@hirschorn2023stgnf]. Our narrower question concerns the difference between matched information conditions and its regional normal-reference profile. The contribution is the matched-context audit and its falsification controls, not priority for normal-only gait prediction, conditional skeleton modeling or anatomical anomaly localization.

Clinical findings motivate examining relationships without determining their expected direction. Following stroke, reduced muscle-activation complexity can accompany impaired locomotion, while plantarflexor drive and nonparetic compensation complicate the relationship between walking speed and propulsion [@clark2010modules; @awad2020drive]. Parkinsonian gait involves stride-length regulation, and arm-swing amplitude and bilateral coordination need not respond identically to medication [@morris1994hypokinesia; @sterling2015armswing]. Cerebral palsy encompasses diverse activation and musculoskeletal constraints; crouch changes strength requirements across muscle groups [@steele2015synergies; @steele2012crouch]. Myopathic gait concerns muscle disease rather than a single neurological etiology. Longitudinal Duchenne observations connect weakness with varied trunk, pelvis and lower-limb adaptations [@vandekerckhove2025dmd].

These studies do not establish that projected joint predictability measures neural coordination. Mechanics and task constraints can themselves produce compact movement or activation patterns [@kutch2012synergies]. Our biological motivation therefore supports a testable movement-level question, not diagnosis, muscle-synergy recovery or inference of lesion location.

# Data and observation protocol

We use the refreshed GAVD annotations, whose public-source videos provide observational gait categories, sequence boundaries, views and bounding boxes [@ranjan2025gavd; @gavd2026catalog]. The binary endpoint is exactly the `normal` folder versus every other folder across all 12 categories. A positive label does not uniformly signify disease: the catalog includes nonpathological movement circumstances. Parkinsons, stroke, cerebral palsy and myopathic labels additionally define a separate five-class task with normal gait.

The annotation inventory contains 1,874 sequences from 348 source IDs, comprising 32 normal and 316 abnormal sources. Ingestion obtained 332 decodable videos, including 184 newly acquired videos. Pose-quality filtering retained 9,474 windows from 1,389 sequences and 311 sources, including 28 normal sources. The subsequent fixed profile-support gate retained 8,950 windows from 1,303 sequences and 251 sources: 23 normal and 228 abnormal. These are source-video identifiers, not verified independent people. Media failure and pose attrition are recorded rather than replaced with cached features.

We extract projected two-dimensional BlazePose landmarks from the annotated person using sequence bounding boxes [@bazarevsky2020blazepose]. We retain BlazePose indices 11–16 and 23–32, covering shoulders, elbows, wrists, hips, knees, ankles, heels and foot indices. These 16 joints alone enter the model and masking sampler. Coordinates with confidence below 0.5 remain missing. Resampling to 15 Hz permits interpolation only between adjacent valid observations and excludes videos below 15 frames per second. Each window contains 32 frames with stride 16. Coordinates are centered on the frame's pelvis and scaled using valid torso lengths from the first half of the window.

A token contains one joint's two-dimensional coordinates over four frames, giving 128 tokens per window. A token is observed only when all eight coordinates are finite. Windows require at least 75% observed tokens overall and 50% in each half. Invalid patches are numerically zeroed at projection and excluded from attention keys, targets and pooling. Missing observations are not reconstructed as training labels.

Before fold assignment, a deterministic bank of 16 uniform masks establishes profile eligibility using only availability and annotated time identities. Each source must supply at least four distinct eligible target patches for every readout joint. Readout uses 12 joints: elbows, wrists, knees, ankles, heels and foot indices. All 16 joints still participate in training and masking. Support is deduplicated across overlapping windows within each sequence; separate annotation sequences are not assumed to be independent physical observations.

# Matched prediction experiment

We first train a small skeleton JEPA using normal fitting sources only. Its encoder has two transformer blocks, four attention heads and width 64. Uniform sampling hides exactly half of the 128 tokens, independently of coordinates, velocity and motion intensity. A visible-context encoder predicts centered, sharpened distributions from a full-observation exponential-moving-average teacher. A VICReg term regularizes agreement and representation variance/covariance across masked views. It discourages collapse and redundancy; it does not repel disease classes. Missing targets remain excluded. With teacher representation $z^T$, center $c$ and predictor output $u$, the target is $q=\operatorname{softmax}((z^T-c)/0.06)$ and the prediction is $p=\operatorname{softmax}(u/0.1)$. Pretraining minimizes target cross-entropy plus 0.1 times VICReg. VICReg acts on projected, valid-token pooled online embeddings from two independently masked views; the second view also receives small coordinate noise. The teacher receives no gradient and is updated after the online optimizer step.

The primary recipe uses 512 pretraining updates followed by 512 updates for each prediction head, with batch size 16, AdamW learning rate 0.0003, weight decay 0.05 and teacher momentum 0.996. Training first samples a normal source uniformly and then a window from that source. Normal-development diagnostics are recorded every 64 updates. Fixed budgets establish a reproducible computational setting, not convergence. The primary study uses neither stochastic positional noise nor a learned stochastic latent variable.

After pretraining, we freeze the encoder, teacher and target center. Six independently fitted heads cross three context conditions with two target types. For target joint $j$ and time patch $t$, **own** context contains visible valid patches of joint $j$; **patch** context adds visible joints in the target four-frame interval; **full** context contains every visible valid token. Allowed tokens are gathered before the encoder's first attention layer. Restricted branches never receive cached full-context embeddings. A position query supplies the target identity without its measurements. Within each target type, heads share initialization, sampled windows, masks and update budgets, but have separate fitted parameters.

The coordinate target contains the eight normalized coordinates of the hidden patch, scored by mean squared error. The latent target is the frozen teacher's distribution, scored by KL divergence from teacher to prediction. All branches receive the same target. A requested target is eligible only when observed and accompanied by at least two visible valid own-joint patches. Eligibility filtering does not redraw masks. During head fitting, a joint is sampled uniformly from all 16 joints independently of visibility, and eligible hidden patches of that joint contribute to all six objectives. The same 16-mask bank is used for all sources and comparators at evaluation.

These are completion tasks: visible context can occur after the hidden patch. Shared pelvis and torso normalization already supplies a body reference to the own condition. The patch condition includes short motion as well as geometry. Moreover, the latent teacher can encode other joints in its target. Predictive benefit therefore means benefit for these fitted models under the declared body reference. It is neither conditional mutual information nor a causal measure of inter-joint coupling.

At the raw-input level, full context contains own context. For an ideal predictor that minimizes expected squared error, additional information therefore cannot increase the optimal risk. Our learners do not establish that bound: they separately re-encode each context through a frozen transformer, so their resulting memories need not be nested representations, and finite fitted heads may underfit or overfit. The observed loss difference measures model-dependent predictive gain, not mutual information, neural coupling or the intrinsic value of the added observations. Negative benefit on normal coordinate targets motivates a learning-adequacy check; it is not evidence that coordination or additional body information is biologically harmful.

Normal-development coordinate losses showed worse full-context than own-context error in all four folds under the 512-head-update schedule. This training-adequacy diagnostic motivated a separately retained 4,096-head-update sensitivity with the same 512 pretraining updates. The decision used normal development observations rather than abnormal ranking outcomes. The original schedule remains primary, and the longer schedule cannot replace it based on a favorable test result.

# Regional normal-reference profiles

For an eligible target, let $\ell^{o}$, $\ell^{p}$ and $\ell^{f}$ denote own, patch and full losses. We define signed predictive benefit as $b=\ell^{o}-\ell^{f}$. Positive values indicate lower error with full context; negative values remain valid observations. For each joint, we first average all eligible mask-target exposures within a window, then average windows within sequences and sequences within sources. Repeated mask exposures retain equal weight in the window mean; the support gate separately deduplicates annotated target patches. Each region then gives equal weight to its four joints: left knee/ankle/heel/foot, the corresponding right joints, and both elbows/wrists.

For source $s$, the three regional benefits form $b_s$. Five reserved normal sources estimate a reference mean $\mu_g$ and sample variance $v_g$ for each region. With $\bar v$ the mean regional variance, the anomaly score is

$$
D(s)=\frac{1}{3}\sum_{g=1}^{3}\frac{(b_{sg}-\mu_g)^2}{\max(0.5v_g+0.5\bar v,10^{-8})}.
$$

This two-sided distance accommodates unusually high or low predictive benefit. It is not a calibrated probability. Identically constructed distances from own, patch and full losses provide constituent controls. They are necessary because comparing $D$ with raw full loss changes subtraction, regional preservation and reference standardization together. Patch-context losses are retained as a control for information near the hidden interval.

# Evaluation and controls

Four outer folds hold out disjoint source IDs. Their test populations contain respectively 6, 6, 6 and 5 normal sources, plus 57 abnormal sources each. Five non-test normal sources per fold serve as development observations and the normal reference, leaving 12 or 13 normal fitting sources. Abnormal sources never fit the normal-only anomaly models or reference. They enter fitting only in separately identified supervised controls.

The primary contrast is latent benefit distance versus raw full-context KL on identical readout support. We compute AUROC within each fold and average using test-normal counts as weights. Average precision is reported with abnormal prevalence. Scores from independently fitted references are not pooled into one ROC curve. Supporting paired comparisons against full-distance and own-distance controls address whether subtraction contributes beyond constituent calibration; their multiplicity is explicit.

A paired bootstrap resamples source IDs within fold and binary label, preserving pairing between scores and recomputing the same weighted AUROC difference over 1,000 replicates. Its interval is conditional on fitted models, reference observations, fold assignments and observed source identities. It does not propagate retraining, reference-estimation or unknown participant dependence. Fold-specific results and leave-one-reference-out score changes accompany this interval. A wide interval is inconclusive, while a narrow interval concerns this fitted analysis rather than performance after new training or population sampling. Removing one reference source tests sensitivity of the reference calculation without retraining the predictors.

The coordinate-target experiment tests whether findings extend beyond contextual teacher targets. A random-backbone control retains fitted heads and matched draws. Its coordinate targets remain comparable, whereas its latent target changes with the teacher. A fixed source-balanced ridge control uses 256 sampled windows/masks and regularization 10, with separate joint/time/context regressions. It shares information access and scoring support, but not neural parameter sharing or optimization budget. Its minimum target-support check establishes estimability, not adequate learning or a strong linear optimum.

A matched-budget sensitivity selects four distinct annotated target patches per readout joint using a fixed hash. It does not equalize mask exposure, contextual information, temporal coverage or precision. Source-level exposure counts remain part of the audit. Additional normal-only controls fit regularized Gaussian distances to 36 handcrafted motion features or joint-availability and camera-view features. A separately supervised Random Forest uses only the latter observation features. Every imputer, scaler and estimator is fitted using the appropriate training sources. The separately supervised five-class comparison uses a Random Forest on handcrafted features and a frozen-encoder probe with the same outer folds, excluding inner reference normals from fitting. Source-label weighting prevents multiple labels or many windows from increasing a source's total evaluation weight. These classifiers receive abnormal labels and answer a different question from normal-only ranking.

Development diagnostics apply relative-leg timing shifts, whole-body shifts and observation noise/dropout to normal fitting sources. They measure fitted-model sensitivity rather than held-out mechanism validation or virtual diagnoses. A four-factor teacher diagnostic crosses original/edited predictions with original/edited teacher targets on eight unchanged arm/right-leg joints. Its raw-KL predictor, teacher and interaction terms sum algebraically to total change. They do not decompose the nonlinear three-region score.

The reproducibility record fixes manifest and pose hashes, model configuration, source roles, mask seed and checkpoint identity. Every reported comparison must reconcile its required method, fold, target and observation-budget cells; a summary file alone is insufficient evidence that all controls completed. Missing cells remain missing rather than being replaced by synthetic results.

# Results

## The primary advantage is not established

Table~\ref{tab:ranking} reports the fixed primary schedule and its separately retained training-budget sensitivity. Across 251 sources, the primary latent profile reaches AUROC 0.641 versus 0.541 for raw full-context KL. The paired difference is 0.101, with a conditional 95% bootstrap interval of $[-0.035, 0.224]$. This interval includes both a small disadvantage and a meaningful advantage. It neither establishes benefit nor demonstrates equivalence. Primary fold AUROCs are 0.588, 0.646, 0.629 and 0.716. Average precision is 0.949, but approximately 91% of evaluated sources are positive, so this value should not be read as performance on a balanced screening population.

\begin{table}[t]
\centering
\begin{tabular}{lrr}
\toprule
Score or control & 512 & 4,096 \\
\midrule
Latent benefit profile & .641 & .667 \\
Raw full latent error & .541 & .529 \\
Full-error reference distance & .673 & .666 \\
Own-error reference distance & .657 & .590 \\
Random encoder, fitted heads & .651 & .714 \\
Coordinate benefit profile & .686 & .690 \\
Ridge coordinate profile & .607 & .607 \\
Latent profile, four patches & .542 & .644 \\
Reference-free contrast & .527 & .398 \\
\bottomrule
\end{tabular}
\caption{Normal-count-weighted within-fold AUROC on the same 251 sources. Columns identify head-update budgets; the 512-update recipe remains primary. The reference-free contrast ranks full minus own latent loss. Ridge is unchanged. These are point estimates, not established rankings of methods.}
\label{tab:ranking}
\end{table}

A calibrated full-error distance reaches 0.673, exceeding the profile point estimate. The profile-minus-full-distance difference is $-0.031$ with interval $[-0.152, 0.096]$. Its difference from calibrated own error is $-0.015$ with interval $[-0.128, 0.101]$. Thus the raw-error comparison does not isolate a benefit from subtracting own and full losses. A random frozen encoder with trained heads reaches 0.651; its difference from the learned profile is also unresolved. Because its latent target changes, this control questions the learned system's anomaly advantage rather than isolating the encoder alone.

The coordinate profile reaches 0.686 and the fixed ridge profile 0.607. A separate normal-only distance on handcrafted motion features reaches 0.626; the latent-profile difference is 0.015 with interval $[-0.140, 0.179]$. Observation-only distance and a separately supervised observation-only Random Forest reach 0.474 and 0.451. Their below-chance directions are retained. These limited controls do not eliminate unmeasured view, identity, activity or recording confounds.

## Learning and reference sensitivity

Increasing head training to 4,096 updates raises the latent profile point estimate to 0.667, but a random encoder with fitted heads reaches 0.714. The longer-run profile-minus-raw-error difference is 0.138 with interval $[-0.008, 0.280]$. This run cannot establish a learned advantage or replace the primary analysis. Normal-development full-context coordinate MSE exceeds own-context MSE in all four primary folds and three longer-run folds. This remaining learning-adequacy limitation prevents interpreting the loss difference as the best achievable information gain. Pretraining's normal-development embedding effective ranks range from 5.19 to 8.39 and embedding standard deviations from 0.488 to 0.570. These finite-batch diagnostics detect nonconstant representations, not sufficient representation quality.

With four distinct target patches per readout joint, primary latent-profile AUROC drops to 0.542. Selecting four patches changes observation coverage as well as quantity; it does not diagnose a single cause. Independently omitting one of five reference sources per fold yields aggregate AUROCs from 0.585 to 0.673 for the latent profile and 0.635 to 0.748 for the coordinate profile, without retraining. These ranges describe reference perturbations, not confidence intervals or new test sets.

## Category and condition comparisons

The primary profile's descriptive AUROCs are 0.815 for exercise, 0.681 for stroke, 0.604 for myopathic, 0.565 for Parkinsons and 0.486 for cerebral palsy. Their abnormal source counts are respectively 59, 14, 25, 9 and 10. Categories reuse the same normal sources and are not independent tests. An exercise review clip contains squatting, illustrating why the binary endpoint can distinguish activity rather than pathology. Restricting positives to the four focal condition families yields 81 sources in total and AUROC 0.585 for the latent profile, 0.612 for calibrated full error and 0.625 for the random-encoder profile. These secondary results do not establish disease-specific discrimination.

The separately supervised five-class task also uses 81 source videos. Across four folds, Random Forest macro-F1 values are 0.521, 0.532, 0.412 and 0.621, versus 0.483, 0.189, 0.262 and 0.423 for the frozen-encoder probe. Its training labels and endpoint differ from normal-only anomaly ranking. This comparison provides no evidence that the current representation improves supervised condition recognition.

## Development probes expose interpretation limits

Twelve normal fitting sources supply development windows for timing and observation edits. Only five remain scorable under common support across all five versions, including dropout. Median paired latent-profile changes are $-0.014$ for relative-leg timing, $-0.037$ for whole-body timing, 0.004 for small coordinate noise and 0.068 for dropout. For coordinates, the corresponding changes are $-0.001$, 0.765, 0.007 and 0.049. The intended selective response to relative-leg timing is not demonstrated. These small, reused fitting samples and potentially distorted limb-to-torso geometry prevent biological validation.

The teacher diagnostic retains 87 unchanged-joint summaries across 12 normal fitting sources. Changing other joints leaves the own-context predictor component exactly zero, while the median source mean absolute teacher-only change in own KL is 0.0351. This confirms an interpretation hazard: contextual latent targets can change even when the local measurements and their own-context prediction are unchanged. It is an expected property of the target construction, not evidence of neurological coupling.

# Discussion and limitations


This is exploratory internal evaluation on a cohort overlapping earlier development. All 23 retained normal sources and 58 abnormal sources occur in the earlier retained index; 170 abnormal sources do not. Absence from that index does not prove absence of prior inspection. New folds prevent direct fitting/test overlap but do not erase analyst adaptation. Newly scored abnormal recordings compared with reused normals are not an independent replication.

Five reference sources and five or six test normals per fold severely constrain precision. Additional abnormal videos cannot replace normal diversity. Visibility selection, camera view, walking aids, repeated people and overlapping annotations may influence discrimination. Folder labels are observational, projected pose is imperfect, and neither diagnoses nor physical forces are measured. Finally, no validated published anomaly baseline is included. The fixed ridge control and supervised Random Forest cannot establish superiority over SeeKer, STG-NF or other published methods. Conclusions must remain conditional on the tested predictors, observation pipeline and source population.


These results narrow the research question. A predictive-context profile can rank some non-normal recordings, but the present evidence does not attribute that ranking to learned inter-joint relationships. Reference calibration, the target representation, finite learning and observation coverage all require separate checks. Failure of the fitted predictor to exploit more context should trigger a model-adequacy investigation before a biological explanation.

A stronger follow-up should first test predictors on synthetic processes with known dependencies and a nested or residual architecture that can retain the own-context solution. It should then freeze choices before examining new normal participants, validate an established pose-anomaly baseline on exactly the same observation support, and collect blinded movement annotations. External recordings with participant identity, measured strength or electromyography would be needed to evaluate neurological or neuromuscular explanations. More abnormal web videos alone cannot answer those questions.

# Conclusion

This real-video study audits a biologically motivated prediction-context diagnostic rather than claiming a new disease detector. Matched constituent scores, a random encoder with fitted heads, longer training and reference perturbations prevent a favorable raw-error comparison from becoming an unsupported representation claim. On the current GAVD cohort, the latent profile's advantage remains inconclusive and its neurological interpretation unvalidated. The reproducible protocol and its negative controls specify what a subsequent, independently evaluated study must establish.


FILE docs/context_study_sensitivity.md
# Additional real-data evidence

The 512-head-update experiment remains primary. The 4,096-update experiment was motivated by normal-validation head adequacy before the new outer-test scores were inspected. It uses the same frozen source roles, masks and 512-update pretrained backbones. Neither budget establishes a learned-encoder advantage.

| Method | Primary512 | Sensitivity4096 |
| --- | --- | --- |
| Latent benefit profile | 0.641 | 0.667 |
| Raw full latent error | 0.541 | 0.529 |
| Full-error reference distance | 0.673 | 0.666 |
| Own-error reference distance | 0.657 | 0.590 |
| Random encoder, fitted heads | 0.651 | 0.714 |
| Coordinate benefit profile | 0.686 | 0.690 |
| Ridge coordinate profile | 0.607 | 0.607 |
| Latent profile, four patches | 0.542 | 0.644 |
| Reference-free contrast | 0.527 | 0.398 |

The primary latent profile minus raw error difference is 0.101, conditional95%CI [-0.03508771929824558, 0.2242753623188406]. The corresponding longer-budget difference is 0.138, conditional95%CI [-0.008409610983981755, 0.27995804729214335]. Both intervals include zero. The random encoder's prediction heads are fitted in both budgets.

## Normal validation

| head_updates | fold | own_coordinate | full_coordinate | own_latent | full_latent | effective_rank | embedding_std |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 512 | 0 | 0.023 | 0.057 | 1.433 | 1.208 | 8.393 | 0.537 |
| 512 | 1 | 0.044 | 0.063 | 1.756 | 1.466 | 8.145 | 0.570 |
| 512 | 2 | 0.043 | 0.058 | 1.057 | 0.692 | 5.193 | 0.488 |
| 512 | 3 | 0.046 | 0.074 | 1.333 | 0.856 | 5.560 | 0.564 |
| 4096 | 0 | 0.018 | 0.021 | 1.325 | 1.223 | 8.393 | 0.537 |
| 4096 | 1 | 0.038 | 0.040 | 1.788 | 1.165 | 8.145 | 0.570 |
| 4096 | 2 | 0.036 | 0.034 | 0.851 | 0.517 | 5.193 | 0.488 |
| 4096 | 3 | 0.040 | 0.047 | 1.182 | 0.740 | 5.560 | 0.564 |

Coordinate full-context MSE remains above own-context MSE. This is a learner-adequacy limitation, not evidence that additional body information is intrinsically harmful.

## Development probes

| target_type | intervention | n_sources | median_paired_change | minimum | maximum |
| --- | --- | --- | --- | --- | --- |
| coordinate | relative_limb_shift | 5 | -0.001 | -0.145 | 0.060 |
| coordinate | whole_body_shift | 5 | 0.765 | -0.231 | 12.005 |
| coordinate | coordinate_noise | 5 | 0.007 | -0.023 | 0.273 |
| coordinate | observation_dropout | 5 | 0.049 | -0.171 | 0.304 |
| latent | relative_limb_shift | 5 | -0.014 | -0.041 | 0.279 |
| latent | whole_body_shift | 5 | -0.037 | -0.192 | 0.118 |
| latent | coordinate_noise | 5 | 0.004 | -0.026 | 0.065 |
| latent | observation_dropout | 5 | 0.068 | -0.050 | 0.768 |

Twelve fitting-normal sources supplied long windows; five remain scorable after common support across every edit, including dropout. These are signed paired score changes on that small development set. They do not validate a neurological mechanism or held-out generalization. Geometry reports separately measure damage at limb-to-torso connections.

For the contextual-teacher diagnostic, 12 fitting-normal sources contribute 87 source-joint summaries. The own-predictor-only change is exactly zero when its local history is unchanged. The median across sources of the mean absolute teacher-only own-KL change is 0.0351. This verifies that a latent target can change because other joints changed, even when the local joint and its own-context prediction are fixed. It is an expected property of contextual targets, not a neural mechanism.

## Traceability

The [manuscript evidence file](draft01/evidence.json) records source artifact hashes. The primary results remain in [context_study_results.md](context_study_results.md). The longer-budget and nuisance comparisons are supporting analyses, not replacements selected for a favorable outcome.


FILE docs/context_study_results.md
# Real GAVD results for the revised context study

This report was generated from a completed real-data artifact. It is separate from the earlier five-condition exploratory results in [results.md](results.md). No synthetic score enters the tables below.

## Cohort and acquisition

The current annotations contain 1,874 sequences and 348 source IDs. The acquisition ledger records 332 locally decodable videos. Pose-window quality checks retain 9,474 windows from 1,389 sequences and 311 sources. The fixed twelve-joint profile-support check leaves **8,950 windows, 1,303 sequences, and 251 sources**, including **23 normal sources**.

The profile check is based on observation validity and the fixed uniform masks before any fit or test score. Every retained source supplies at least four distinct physical patches for every readout joint. Category source memberships may overlap. These counts are source videos, not verified independent people.

| label | annotated_sequences | annotated_sources | pose_sequences | pose_sources | pose_windows | profile_sources | profile_windows |
| --- | --- | --- | --- | --- | --- | --- | --- |
| abnormal | 767 | 117 | 613 | 103 | 5143 | 92 | 5098 |
| antalgic | 35 | 10 | 29 | 9 | 116 | 8 | 102 |
| cerebral palsy | 64 | 11 | 56 | 10 | 390 | 10 | 390 |
| exercise | 234 | 98 | 203 | 92 | 1092 | 59 | 781 |
| inebriated | 23 | 8 | 9 | 6 | 41 | 2 | 21 |
| myopathic | 188 | 30 | 164 | 28 | 669 | 25 | 631 |
| normal | 291 | 32 | 135 | 28 | 482 | 23 | 404 |
| parkinsons | 47 | 11 | 35 | 10 | 184 | 9 | 179 |
| pregnant | 6 | 2 | 6 | 2 | 20 | 2 | 20 |
| prosthetic | 39 | 8 | 28 | 7 | 445 | 6 | 438 |
| stroke | 76 | 19 | 60 | 15 | 779 | 14 | 773 |
| style | 104 | 3 | 51 | 2 | 113 | 2 | 113 |

## Previous use of these recordings

| binary_label | prior_use | sources |
| --- | --- | --- |
| 0 | previous_study | 23 |
| 1 | not_in_previous_study | 170 |
| 1 | previous_study | 58 |

The new protocol was specified before its new score evaluation, but the dataset was previously used for exploratory work. All retained normal sources appeared in that earlier study. The outer folds provide source separation for each fitted model; they are not a newly sequestered clinical test cohort.

## Prespecified anomaly comparison

All normal-only models use the same outer source folds. Five normal reference sources per fold are excluded from encoder and predictor fitting. The primary statistic averages within-fold AUROC using normal-source counts. AP is averaged with the same weights and must be interpreted with each fold's abnormal prevalence, available in the machine-readable metrics.

| Method | AUROC | AP |
| --- | --- | --- |
| JEPA latent profile | 0.641 | 0.949 |
| JEPA full-context error | 0.541 | 0.930 |
| JEPA coordinate profile | 0.686 | 0.957 |
| Ridge coordinate profile | 0.607 | 0.948 |
| Untrained latent profile | 0.651 | 0.951 |
| Own-error profile distance | 0.657 | 0.953 |
| Full-error profile distance | 0.673 | 0.953 |
| Patch-error profile distance | 0.648 | 0.952 |
| JEPA profile, four patches | 0.542 | 0.932 |
| Reference-free full-minus-own contrast | 0.527 | 0.926 |

The primary paired difference, latent profile minus full-context error, is **0.101 AUROC**, with a conditional 95% bootstrap interval **[-0.035, 0.224]**. The bootstrap resamples sources within fold and binary label, carrying both methods together. It conditions on fitted models, chosen references, and folds. It does not include retraining uncertainty, reference-selection uncertainty, or unknown repeated participants.

The conditional interval includes zero. The experiment does not establish an advantage of the proposed profile over raw full-context error.

![Real source-held-out ranking](../images/context_real_comparison.svg)

*Bars are point estimates on the common cohort. The interval above refers to the paired primary difference, not to each bar separately.*

## Matched alternative explanations

| Comparator | Difference | CI_low | CI_high |
| --- | --- | --- | --- |
| Full-error profile distance | -0.031 | -0.152 | 0.096 |
| Own-error profile distance | -0.015 | -0.128 | 0.101 |
| Untrained latent profile | -0.009 | -0.127 | 0.108 |
| Ridge coordinate profile | 0.034 | -0.087 | 0.155 |

These additional paired intervals use the same fixed-fit bootstrap. The own- and full-error distances receive the same two-sided reference calibration as the proposed profile. An advantage over raw error alone cannot establish that subtracting errors adds information. These are exploratory control comparisons without a multiple-comparison correction.

## Reference sensitivity and newly used abnormal sources

| target_type | minimum | maximum |
| --- | --- | --- |
| coordinate | 0.635 | 0.748 |
| latent | 0.585 | 0.673 |

These are the smallest and largest aggregate AUROCs attainable by omitting one inner reference source in each fold while holding the models fixed. They are perturbation ranges, not confidence intervals. The reference-free contrast above remains a secondary comparison and is not promoted to a new primary endpoint.

A descriptive restriction retains all normal test sources and only the 170 abnormal sources absent from the previous study. Its full metrics are saved in `new_abnormal_descriptive_metrics.json`. Because the normal sources remain previously explored, this restriction is not external validation.

## Category descriptions

| Category | Sources | Folds | AUROC | AP |
| --- | --- | --- | --- | --- |
| abnormal | 92 | 4 | 0.579 | 0.851 |
| antalgic | 8 | 4 | 0.522 | 0.481 |
| cerebral palsy | 10 | 4 | 0.486 | 0.431 |
| exercise | 59 | 4 | 0.815 | 0.930 |
| inebriated | 2 | 2 | 0.667 | 0.375 |
| myopathic | 25 | 4 | 0.604 | 0.671 |
| parkinsons | 9 | 4 | 0.565 | 0.485 |
| pregnant | 2 | 2 | 0.500 | 0.250 |
| prosthetic | 6 | 4 | 0.543 | 0.418 |
| stroke | 14 | 4 | 0.681 | 0.640 |
| style | 2 | 2 | 0.455 | 0.250 |

Category summaries reuse normal sources and may reuse mixed-category abnormal sources. They are dependent. A category can lack abnormal support in a fold; its descriptive average uses supported folds only. Two- and three-source categories cannot support stable population estimates. Folder labels are not verified diagnoses.

## Separate supervised condition recognition

Both classifiers receive focal-condition training labels and exclude outer test sources and the inner normal reference sources. The JEPA encoder remains frozen after normal-only pretraining. Predictions average windows within sequences and sequences within source-label pairs; a mixed-label source has total metric weight one. These supervised macro-F1 scores are not comparable numerically with normal-only AUROC.

| Fold | Method | Macro_F1 | Balanced_accuracy | Sources |
| --- | --- | --- | --- | --- |
| fold_0 | random_forest | 0.521 | 0.483 | 21 |
| fold_0 | normal_jepa_probe | 0.483 | 0.483 | 21 |
| fold_1 | random_forest | 0.532 | 0.629 | 21 |
| fold_1 | normal_jepa_probe | 0.189 | 0.219 | 21 |
| fold_2 | random_forest | 0.412 | 0.433 | 20 |
| fold_2 | normal_jepa_probe | 0.262 | 0.300 | 20 |
| fold_3 | random_forest | 0.621 | 0.630 | 19 |
| fold_3 | normal_jepa_probe | 0.423 | 0.580 | 19 |

## Training and robustness

The run uses 512 JEPA pretraining updates and 512 updates for each independent context head. Every window uses the same sixteen-mask scoring bank, independent of motion intensity and pose validity. Ridge uses 256 source-balanced training draws. These are fixed budgets and do not establish convergence by themselves.

![Actual normal-validation curves](../images/context_training_curves.svg)

*Validation sources influence neither weights nor target centers. The curves describe this budget, not an optimized stopping rule.*

The run also saves the four-patch observation-budget sensitivity and leave-one-normal-reference-out scores. No score direction, fold assignment, or category selection is changed after test inspection. The development timing/observation controls are reported separately when completed. Their edits are not disease simulations. A published SeeKer or STG-NF comparison remains unvalidated for this partially observed sixteen-joint corpus; see the [feasibility report](published_baseline_feasibility.md).

## Reproducibility and claim boundary

The run directory is `artifacts/context_study/gavd/3044566091261819/`. Its protocol, source roles, fresh manifest, acquisition and eligibility ledgers, checkpoints, cache hashes, frozen references, source scores, and training histories support the reported tables. `scripts/summarize_context_study.py` regenerates this report and the vector plots. The [implemented protocol](context_study_protocol.md) explains the estimand and controls.

A result on this cohort cannot identify neural causes or establish clinical screening performance. The normal sample is small, diagnoses and participant identities are unverified, and pose exclusions select for visible joints. GAVD supports a bounded observational comparison. The [draft01 manuscript and readiness assessment](draft01/README.md) state which conclusions survive those limitations.


FILE docs/published_baseline_feasibility.md
# Feasibility of a published skeleton anomaly baseline

Checked on November 15, 2026. **Neither official implementation is currently a validated comparator for our 16-joint, partially observed GAVD protocol.** STG-NF is easier to execute locally. SeeKer is the closer scientific comparison because it predicts conditional keypoint distributions. Their main shared obstacle is the treatment of missing coordinates. A quick zero-fill adapter would change the statistical model and would not establish a published-method reproduction.

This review inspected the official repositories, read the primary papers, and ran limited mechanical checks on unmodified model code. It did not train either method, change our model, install dependencies, or produce GAVD detection scores. The current study's matched ridge predictor is a useful control, but it does not substitute for a validated published anomaly detector.

## What was inspected and tested

| Item | Pinned version or basis | Observed status |
| --- | --- | --- |
| [Official SeeKer repository](https://github.com/adelic99/seeker/tree/7e7ab66a703188616b0efd482947ff2bf2bb26ca) | `7e7ab66a703188616b0efd482947ff2bf2bb26ca` | Source inspected. The unmodified model constructor fails on this laptop because its masks are moved directly to CUDA. |
| [Official STG-NF repository](https://github.com/orhir/STG-NF/tree/edb5f3220332e160e4d20ce258787d5e2d7e0200) | `edb5f3220332e160e4d20ce258787d5e2d7e0200` | Source inspected. An unmodified 18-joint model produced finite CPU outputs on synthetic finite input. |
| Local runtime | Apple arm64; Python 3.12.13; PyTorch 2.15.0; NumPy 2.5.4 | CPU checks only. No CUDA, MPS, optimizer, checkpoint or training validation was performed. |
| Our observation protocol | [data.py](../aaai_gait/data.py), [study_data.py](../aaai_gait/study_data.py), and [research roadmap](draft00/research_roadmap.md) | Sixteen selected joints, 32 frames at 15 Hz, four-frame tokens, confidence filtering, retained NaNs and source-held-out evaluation. |
| Historical completeness audit | [full_window_index.csv](../artifacts/full_window_index.csv) | Read-only count of the completed earlier five-class cohort. This is not the final expanded twelve-folder study cohort. |

Temporary repository checkouts were inspected outside the project. They were not vendored into the tutorial package. The historical window index had SHA-256 `53a08a8dffdfe0a691879587076dd6b04e1668811895cfd84cdf329483284142` at review time.

The STG-NF check used `pose_shape=(2,32,18)`, eight flow steps, one level, zero hidden channels, its default permutation and affine coupling, `R=3`, and CPU execution. This constructed 744 trainable parameters. A batch of two finite synthetic windows returned two finite negative log-likelihoods. After initialization, replacing one coordinate in the first window with NaN made that window's score nonfinite while the second score stayed finite. Reconstructing the same model with 16 joints failed because its graph still had 18 nodes. These checks establish a narrow execution fact, not training correctness, convergence or anomaly performance.

SeeKer's parser check confirmed that `droppout` exists and `dropout` does not. Static inspection confirmed that the trainer does not define `joint_lnp_full_window`. Importing its trainer in the current tutorial environment stopped at the missing `tqdm` package. Constructing its model directly, without the trainer, stopped at the hard-coded CUDA operation. No patches were applied to hide these failures.

## Why both methods matter scientifically

SeeKer models keypoint locations with autoregressive conditional Gaussians and combines keypoint log scores with detector-confidence weights. It already occupies the space of conditional skeletal prediction, compositional anomaly scoring and confidence-aware scoring. Therefore, predicting a hidden joint from other joints, reporting per-joint surprise, or using confidence cannot by itself establish our novelty. Our proposed distinction is the controlled comparison of own-joint, patch-local and full context, including a two-sided normal-reference profile and tests of contextual teacher effects. That distinction still needs evidence. [SeeKer paper](https://arxiv.org/html/2506.18368v1)

STG-NF directly learns a distribution over pose sequences using graph-conditioned normalizing flows. It supports training on normal examples and scores sequence likelihood. Normal-only skeleton anomaly detection and learning normal motion distributions are therefore established prior art. Our study must test a specific information hypothesis rather than claim novelty from normal-only fitting. [STG-NF paper](https://arxiv.org/pdf/2211.10946)

The two published methods are different kinds of baselines. SeeKer predicts ordered keypoints. STG-NF gives a whole-window density. Neither is a natural replacement for the exact matched raw-KL control inside the own-versus-full experiment. Their role is an external anomaly-detection comparison on a declared common source cohort.

## SeeKer: close scientific fit, incomplete released execution path

The released model is a masked fully connected autoregressive network. The masks express a fixed ordering, which is different from the uniform random target masks used by our S-JEPA. Adapting the published baseline does not require replacing its autoregressive factorization with our masking task.

The following issues are visible at the pinned version:

| Area | Verified issue | Required action before a result is credible |
| --- | --- | --- |
| Installation | The README references a `requirements.txt` that is absent. Its clone command contains a placeholder repository name. | Record an explicit dependency environment and use the actual official repository. |
| Entry point | The parser defines `droppout`; `seeker.py` accesses `args.dropout`. | Repair the argument mismatch and test the command-line path. |
| Device handling | The model, connectivity masks and confidence scoring contain direct CUDA calls. | Make device handling explicit and register masks as buffers. Check numerical agreement on an available reference device. |
| Test path | `test()` calls an undefined `joint_lnp_full_window`. It also unpacks three values from a scorer that returns one unless its separate return flag is enabled. | Resolve the intended inference path against the paper and a reference case. Do not guess which method name was intended and immediately report results. |
| Skeleton layout | The entry point uses 18 joints. The causal-mask context boundary subtracts `18*2`; validation reshapes scores with 18 joints. | Parameterize the joint count and audit the full flattening, ordering, output alignment and scoring path. |

These findings come from separate source files: [README](https://github.com/adelic99/seeker/blob/7e7ab66a703188616b0efd482947ff2bf2bb26ca/README.md), [arguments](https://github.com/adelic99/seeker/blob/7e7ab66a703188616b0efd482947ff2bf2bb26ca/args.py), [entry point](https://github.com/adelic99/seeker/blob/7e7ab66a703188616b0efd482947ff2bf2bb26ca/seeker.py), [masked model](https://github.com/adelic99/seeker/blob/7e7ab66a703188616b0efd482947ff2bf2bb26ca/models/made/made_partial.py), [training and inference](https://github.com/adelic99/seeker/blob/7e7ab66a703188616b0efd482947ff2bf2bb26ca/training.py), and [evaluation](https://github.com/adelic99/seeker/blob/7e7ab66a703188616b0efd482947ff2bf2bb26ca/validation.py).

Several methodological details also need adaptation:

- Its dataset loader converts COCO-17 detections into an 18-joint layout by deriving a neck point and reordering joints. Our allowlist cannot be represented by applying that conversion to a 16-joint tensor. The shoulders, hips, heels and foot indices must keep their actual identities. [Dataset conversion](https://github.com/adelic99/seeker/blob/7e7ab66a703188616b0efd482947ff2bf2bb26ca/dataset.py)
- Its normalization uses the mean position and vertical standard deviation across the whole input window. Our normalization uses each frame's pelvis and a prefix torso scale. Use the shared normalization for a matched-input comparison, and report the change. [Normalization](https://github.com/adelic99/seeker/blob/7e7ab66a703188616b0efd482947ff2bf2bb26ca/utils/data_utils.py)
- Its sequence utility interpolates missing frames. Our protocol preserves missing observations and never bridges a missing-frame gap. Reuse our accepted windows rather than silently importing that interpolation policy. [Sequence processing](https://github.com/adelic99/seeker/blob/7e7ab66a703188616b0efd482947ff2bf2bb26ca/utils/sequence_utils.py)
- The released trainer selects epochs by validation AUROC. For datasets other than UBnormal, its loader assigns test data to the validation role. Neither choice can be reused for our normal-only model selection. Our outer test sources must never choose a checkpoint. [Dataset roles](https://github.com/adelic99/seeker/blob/7e7ab66a703188616b0efd482947ff2bf2bb26ca/dataset.py), [checkpoint selection](https://github.com/adelic99/seeker/blob/7e7ab66a703188616b0efd482947ff2bf2bb26ca/training.py)

For a finite, fully observed input, a feasible layout adapter would use `[batch,time,16,2]` in the fixed numerical joint order. The flattened dimension would be `32*16*2=1024`; the context boundary would depend on `2*16`, not 36. Both XY coordinates of one target joint must be excluded from the conditioning set for that target. Verify this with numerical perturbation or Jacobian tests for every target position. In particular, audit the trainer's one-keypoint output offset before choosing an inference implementation. A changed order is an adaptation and should be fixed before inspecting abnormal performance.

The confidence weights need a separate decision. MediaPipe confidence values have not been established as calibrated probabilities of a correct joint location in this corpus. A weighted score can still be reported as the published scoring heuristic, with an unweighted sensitivity analysis, but it must not be described as calibrated uncertainty or as a solution to missing conditioning variables.

## STG-NF: executable core, incompatible graph and missingness assumptions

The official setup targets Ubuntu, Python 3.8 and a CUDA GPU. Its environment file pins PyTorch 1.10.1, NumPy 1.23.3 and CUDA 10.2, along with Linux-specific packages. That environment should not replace the tutorial's working Python 3.12 environment. The successful CPU core check suggests that a small modern environment is feasible, but it does not validate the legacy data and training scripts. The loader still contains the removed NumPy alias `np.int`. [Official setup](https://github.com/orhir/STG-NF/blob/edb5f3220332e160e4d20ce258787d5e2d7e0200/README.md), [environment](https://github.com/orhir/STG-NF/blob/edb5f3220332e160e4d20ce258787d5e2d7e0200/environment.yml), [segmentation code](https://github.com/orhir/STG-NF/blob/edb5f3220332e160e4d20ce258787d5e2d7e0200/utils/pose_utils.py)

The model constructs the default OpenPose graph inside each flow step. Passing a new input shape does not change its adjacency matrix. The existing headless option is also not our 16-joint skeleton. A real adapter must introduce an explicit layout and pass it through every flow block. For our numerical order `[11,12,13,14,15,16,23,24,25,26,27,28,29,30,31,32]`, a documented candidate is the induced body graph with these undirected edges in original BlazePose indices:

```text
11-12, 11-13, 13-15, 12-14, 14-16,
11-23, 12-24, 23-24,
23-25, 25-27, 27-29, 29-31, 27-31,
24-26, 26-28, 28-30, 30-32, 28-32.
```

Add the 16 self connections and convert indices through the fixed order above. Preserve the declared adjacency normalization and hop settings. Do not pad to 18 with duplicate joints or invented face points. Check graph symmetry, reachability, normalized adjacency, gradient finiteness and a finite forward/inverse round trip. Using this graph is an explicit BlazePose adaptation, not an exact OpenPose benchmark reproduction. [Graph construction](https://github.com/orhir/STG-NF/blob/edb5f3220332e160e4d20ce258787d5e2d7e0200/models/STG_NF/graph.py), [flow graph use](https://github.com/orhir/STG-NF/blob/edb5f3220332e160e4d20ce258787d5e2d7e0200/models/STG_NF/model_pose.py)

The flow likelihood and its data-dependent normalization operate on every coordinate. They expose no observation mask or marginal likelihood over an observed subset. The confidence option in the trainer is not an invalid-coordinate exclusion mechanism. Replacing NaN with zero makes the flow model the filled observation, while dropping selected latent likelihood terms does not generally marginalize missing input coordinates because the flow mixes coordinates. [Likelihood and normalization](https://github.com/orhir/STG-NF/blob/edb5f3220332e160e4d20ce258787d5e2d7e0200/models/STG_NF/modules_pose.py), [training confidence handling](https://github.com/orhir/STG-NF/blob/edb5f3220332e160e4d20ce258787d5e2d7e0200/models/training.py)

## Missing observations are the deciding constraint

Our accepted windows need at least 75% valid four-frame tokens overall and 50% in each half. A valid token has finite XY for its joint in all four frames. Confidence below 0.5 creates missing coordinates. The model excludes invalid keys, targets and pooled contributions. A published baseline must state how it uses that same available information.

Multiplying a final likelihood term by zero does not undo the effect of a missing or filled value on other predicted terms. In SeeKer, a missing early keypoint may be part of later keypoints' conditioning sets. In STG-NF, missing inputs enter coupled transformations. Both need more than a loss mask.

The tempting complete-case fallback was checked against the saved historical cohort:

| Historical folder | Accepted windows | Accepted sources | Fully observed 32-frame windows | Sources with a fully observed window |
| --- | ---: | ---: | ---: | ---: |
| Normal | 482 | 28 | 11 | 5 |
| Parkinsons | 184 | 10 | 70 | 5 |
| Stroke | 779 | 15 | 143 | 9 |
| Cerebral palsy | 390 | 10 | 114 | 6 |
| Myopathic | 669 | 28 | 105 | 18 |
| Total | 2,504 | 91 | 443 | 43 |

The count selects `valid_token_fraction == 1` in the saved window index. The five remaining normal sources cannot support our four outer folds with five separate inner normal reference sources and a remaining normal fitting set. The new all-folder extraction must be audited separately, but adding abnormal folders does not by itself add normal sources. A complete-case result would also describe a strongly selected visibility population. It cannot stand in for the primary partially observed cohort.

There are three honest options:

1. **A fully observed secondary cohort.** Preserve the published density model and explicitly restrict all compared methods to the same complete cases. Proceed only if the final corpus supplies sufficient normal sources for a prespecified fitting and evaluation design. The historical count fails that gate. Shortening windows or lowering confidence thresholds would change the experiment and requires a separately declared sensitivity analysis.
2. **An observation-aware adaptation.** For example, a conditional predictor could read finite values plus explicit availability indicators and score observed targets. This is technically plausible, but changes SeeKer's conditioning model and requires missingness noninterference, calibration, coverage and nuisance tests. Label it a SeeKer-inspired adaptation until validated against the published method. It is not an exact marginal likelihood simply because missing targets are excluded.
3. **A later density-method study.** Proper integration over missing coordinates or a validated conditional-flow construction is a substantive extension. It is not a safe one-line repair before this deadline. For this submission, report the missing published comparator and keep claims limited to the matched internal controls.

Using low-confidence raw coordinates that our primary protocol excludes is another possible sensitivity analysis, but it changes the information supplied to the method. It cannot be presented as the same-observation comparison.

## Dependencies and a bounded validation plan

For an isolated SeeKer audit, direct imports require PyTorch, NumPy, SciPy, scikit-learn, `tqdm` and TensorBoard. STG-NF's model core requires PyTorch and NumPy; its surrounding scripts also use `tqdm`, TensorBoard, SciPy, scikit-learn, Pillow and Matplotlib. Most are already available in the tutorial environment, but `tqdm` and TensorBoard were absent during this review. These are observed import requirements, not a tested replacement lockfile. Create a separate `uv` environment and record resolved versions before a reproduction attempt. Avoid upgrading the running experiment environment to repair another repository.

The release licenses also differ: SeeKer is MIT, while STG-NF is distributed under Attribution-NonCommercial 4.0. Preserve source attribution and the relevant license if code is redistributed. This review did not redistribute either implementation. [SeeKer license](https://github.com/adelic99/seeker/blob/7e7ab66a703188616b0efd482947ff2bf2bb26ca/LICENSE), [STG-NF license](https://github.com/orhir/STG-NF/blob/edb5f3220332e160e4d20ce258787d5e2d7e0200/LICENSE)

The following is a planning estimate, not a measured completion schedule:

| Priority | Work package | Gate for moving forward |
| --- | --- | --- |
| First half day | Recompute completeness and source support on the final frozen twelve-folder corpus; choose the observation policy. | Sufficient independent source roles without changing the primary comparison after viewing results. Stop the complete-case route if normal support remains inadequate. |
| Next one to two days | Reproduce a small native-layout reference case in an isolated environment. Prefer STG-NF for immediate execution; use SeeKer if the inference alignment and released defects can be resolved against the paper. | Finite forward/backward passes, deterministic evaluation, correct score direction, stable checkpoint reload, and a reference result or independently checked analytical case. |
| Next one to two days | Add only the declared 16-joint layout and shared preprocessing, keeping a written change log. | Target-order or graph tests pass; no artificial joints; no silent missing-coordinate fill; no test-based checkpoint selection. |
| Remaining time | Fit normal-only folds and aggregate on the common eligible source set. | All source roles and exclusions are recorded, normal development diagnostics are adequate, and paired source-level intervals use the same metric as the point estimate. |

Executing and adapting a small model core within a week is plausible. **A validated published baseline on the full partially observed cohort is not currently a defensible promise for the November 20 deadline.** The missingness policy and scarce complete normal sources are the limiting evidence, even if code repairs are fast. The workshop date is verified on the [official ASTAD4 call](https://ismart.ece.mcgill.ca/astad4/); this review does not infer an unstated cutoff timezone.

For the present paper, prioritize the matched own/full/patch-local comparisons, local-coordinate target control, matched raw KL, ridge predictor and source-level uncertainty. If a published comparator fails the gates above, say so plainly. A supported study of predictive information can remain useful without claiming superiority over SeeKer, STG-NF or the wider literature. No current result should be called state of the art, and the absence of a runnable adapter does not diminish the prior-art relevance of either method.
