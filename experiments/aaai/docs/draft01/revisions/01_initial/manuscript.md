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
