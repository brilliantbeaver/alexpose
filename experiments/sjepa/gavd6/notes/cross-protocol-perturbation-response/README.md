# Cross-Protocol Perturbation Response Prediction

**Study status:** Experimental protocol; no results have been produced.

This document expands [Proposal 01](../world-model-extensions/proposals-03/01-cross-protocol-response.md) into a standalone study plan. All model choices, thresholds, splits, and claims described below are prospective.

## Abstract

Human balance recovery is commonly studied by applying a controlled disturbance during walking and measuring the subsequent corrective response. These experiments provide a stronger test of predictive motion representations than passive gait classification because the intervention is observed, its timing is known, and the model must predict events that occur after the prediction point. This study asks whether one second of pre-perturbation pelvis and foot kinematics contains person- and state-specific information about recovery beyond the average response to perturbation direction, magnitude, and gait phase. The primary target is a four-dimensional recovery vector describing first recovery-foot placement, peak pelvis-speed deviation, time to pelvis-speed stabilization, and maximum mediolateral pelvis excursion over the following two steps. Rather than predicting one trajectory, the model estimates a calibrated conditional distribution over these outcomes.

The study harmonizes two independent public perturbation datasets: the [Georgia Tech ground-translation dataset](https://repository.gatech.edu/entities/publication/73a7c133-6535-4a88-b81e-5c39df5efb3e), in which perturbation magnitude, direction, and onset vary, and the [Stanford Dryad balance-impairment dataset](https://datadryad.org/dataset/doi:10.5061/dryad.cnp5hqch3), which contains pelvis perturbations delivered in four directions and two magnitudes at a fixed gait phase. A common pelvis-and-feet interface avoids assuming that the protocols expose identical skeletons. The central comparison is between an intervention-only conditional mean, a parameter-matched raw-kinematics model, a frozen S-JEPA representation, and a frozen S-JEPA representation with a small outcome-free adapter. Each learned model predicts the residual recovery after subtracting the training-fold intervention-specific mean.

Evaluation is participant-held and includes perturbation combinations excluded from training. The primary transfer experiment trains on Georgia Tech, fits only a small affine calibration on participant-disjoint Dryad data, and evaluates untouched Dryad participants; reverse transfer is a sensitivity analysis. Prespecified endpoints include standardized mean absolute error, joint negative log likelihood, energy score, 80% interval coverage and width, and participant-cluster bootstrap intervals. The study is designed to distinguish four questions in order: whether the outcomes can be harmonized, whether pre-perturbation state is informative, whether S-JEPA adds information beyond raw kinematics, and whether that information survives a change in apparatus and processing pipeline. It does not test diagnosis, clinical fall risk, or treatment effects.

## Introduction

### Motivation

Most gait-learning systems are trained on passive observations. They recognize an activity, classify a clinical presentation, reconstruct a masked pose, or forecast an unperturbed continuation. These tasks can be useful, but a high score does not necessarily show that a representation captures how a person's movement changes after the world changes. Background, participant identity, recording setup, speed, cadence, or label-specific acquisition artifacts can all support prediction without representing a response mechanism.

A controlled walking perturbation creates a more constrained predictive problem. At a recorded onset time, the ground or pelvis is deliberately displaced or pulled. Direction, magnitude, and gait phase are observed. The model receives only information available before onset and must predict the recovery that follows. This temporal ordering does not by itself establish causality, but it sharply reduces ambiguity about which event precedes the response and allows intervention strength to be conditioned on directly.

The difficult scientific question is not whether a large perturbation produces a large average response. An intervention-only model can learn that relationship. The question is whether the person's pre-perturbation state explains why their response differs from the mean response to the same intervention. Relevant pre-state may include foot placement, pelvis velocity, stance side, phase estimation error, recent variability, and latent dynamical information not summarized by handcrafted features. If no such incremental signal exists, a learned representation has nothing defensible to recover.

### Why predict a response envelope

Recovery is not a deterministic continuation. Similar pre-perturbation states can lead to different valid steps because of unobserved muscle activation, attention, strategy selection, measurement noise, and natural trial-to-trial variation. Predicting a single best trajectory would collapse this uncertainty and could reward a model that regresses toward an average response. The proposed output is therefore a probability distribution over a small set of interpretable recovery outcomes. Its quality is judged both by predictive accuracy and by whether stated uncertainty agrees with observed frequency.

The target is intentionally smaller than a complete future motion sequence. Full reactive-motion forecasting is already an active area; for example, [Human Motion Prediction Under Unexpected Perturbation](https://openaccess.thecvf.com/content/CVPR2024/html/Yue_Human_Motion_Prediction_Under_Unexpected_Perturbation_CVPR_2024_paper.html) predicts reactive motion using latent differentiable physics. This study instead asks whether a compact predictive gait representation supports calibrated estimates of recovery summaries that can be defined consistently across two measurement protocols. That narrower object makes cross-protocol evaluation possible and connects model outputs to conventional biomechanical quantities.

### Why cross-protocol validation matters

Within one laboratory, a model can exploit apparatus-specific coordinates, filtering, perturbation timing, file organization, or participant routines. The Georgia Tech and Dryad datasets provide a useful shift because they were collected with different perturbation mechanisms and processing pipelines. Georgia Tech varies perturbation onset and supports phase-generalization tests. Dryad targets perturbations at 32.5% of the gait cycle and therefore cannot independently validate phase generalization, but it provides a stringent apparatus-transfer test with OpenSim outputs and repeated perturbations.

Cross-protocol transfer is not treated as pooling. Each protocol is first evaluated on its own terms. Transfer is attempted only after units, coordinate conventions, outcome definitions, and validity rules are fixed. The primary Dryad transfer endpoint uses its normal-walking condition, which is closest to the source setting. The ankle-brace, vision-blocked, and pneumatic-jet conditions are retained as named out-of-distribution stress tests rather than mixed into a favorable aggregate.

### Research question and hypotheses

The broad research question is:

> Can a model use motion immediately before a known walking disturbance to predict how an individual's short-term recovery will differ from the average response to that disturbance, and does that information generalize to new people, unseen perturbations, and a different experimental protocol?

The primary hypothesis is that pre-perturbation kinematics reduce person-macro standardized error by at least 10% relative to an intervention-only conditional mean for at least three of four recovery outcomes. The representation hypothesis is that frozen S-JEPA features improve joint negative log likelihood over an equal-capacity raw-kinematics model, with a positive participant-bootstrap interval. The calibration hypothesis is that nominal 80% prediction intervals achieve 75% to 85% empirical coverage while remaining narrower than intervals from raw quantile regression. The transfer hypothesis is that the S-JEPA model retains positive log-likelihood gain after only participant-disjoint affine protocol calibration.

These hypotheses are ordered. Representation quality is not interpreted if raw pre-state fails the identifiability test. Cross-protocol transfer is not interpreted if within-protocol prediction or calibration fails.

### Intended contribution and scope

The intended contribution is not a novel backbone. It is a mechanism-first evaluation of person-specific recovery predictability with five linked properties:

1. the perturbation is observed and explicitly conditioned on;
2. the model predicts residual response beyond the intervention-specific mean;
3. the output is a calibrated distribution over interpretable recovery quantities;
4. evaluation holds out both participants and perturbation combinations; and
5. the passed mechanism is tested across independent protocols with minimal adaptation.

A successful experiment would support a claim about short-term biomechanical response prediction. It would not support claims about diagnosis, prospective fall risk, clinical severity, treatment benefit, or measured internal forces.

## Methodology

### Problem formulation

Let $t_0$ denote the recorded perturbation onset. For trial $i$, define:

- $x_i \in \mathbb{R}^{T \times C}$: the final 1.0 second of valid pre-perturbation kinematics;
- $q_i \in \{0, 1\}^{T \times C}$: the corresponding channel-validity mask;
- $u_i$: the observed intervention descriptor; and
- $y_i \in \mathbb{R}^{4}$: the primary recovery-outcome vector.

The intervention descriptor contains the horizontal perturbation direction as a unit vector, normalized magnitude, onset phase encoded as sine and cosine, stance side, and protocol identity. A balance-condition indicator is included only in Dryad-specific analyses; it is not silently treated as a feature available in Georgia Tech.

The desired predictive distribution is

$$
p(y_i \mid x_i, q_i, u_i).
$$

To remove the response that is predictable from the intervention alone, fit a conditional baseline $m(u)$ using training participants only and define

$$
r_i = y_i - m(u_i).
$$

Learned response models estimate $p(r_i \mid x_i, q_i, u_i)$. At inference, the predicted residual distribution is translated by $m(u_i)$ to recover a distribution over $y_i$. This construction makes the intervention-only prediction an explicit zero-residual reference rather than a weak post hoc baseline.

### Data sources and roles

The Georgia Tech dataset is the primary development dataset. Its official record describes ground-translation perturbations whose magnitude, direction, and onset time vary, with biomechanical outcomes related to balance recovery, foot placement, and whole-body angular momentum. It is used for participant-held development, held direction-by-magnitude tests, and the primary phase-generalization analysis.

The Dryad dataset contains 10 healthy participants walking at 1.25 m/s under four balance conditions. Within each condition, participants experience four perturbation directions, two magnitudes, and two repetitions, with pelvis perturbations targeted to begin at 32.5% of the gait cycle. The release includes synchronized perturbation metadata, motion capture, and OpenSim-derived kinematics. Dryad is used first for an independent within-protocol check and then as the primary target domain for cross-protocol transfer. Because onset phase is fixed, it is not used as evidence of phase generalization.

Before modeling, an availability audit must confirm the exact files, participant identifiers, trial counts, sampling rates, perturbation timestamps, coordinate conventions, foot-contact signals, and valid observation windows. No count inferred from filenames will be used until it agrees with dataset documentation and parsed metadata.

### Event alignment and common kinematic interface

Each trial is aligned to $t_0 = 0$ using the recorded controller or perturbation metadata, not a response-derived estimate. The model context is the half-open interval $[-1.0, 0)$ seconds. No sample at or after onset may contribute to preprocessing, phase estimation, normalization, missing-value interpolation, or feature construction for that trial.

Both protocols are mapped into a right-handed, pelvis-centered coordinate system with anterior-posterior, mediolateral, and vertical axes. Horizontal heading is estimated from pre-onset motion only. Direction signs are verified using documented coordinate conventions and trial-level perturbation labels. A synthetic sign audit applies known coordinate reflections and checks that direction-dependent features and outcomes transform as expected.

The primary common state contains:

- pelvis position and velocity;
- left and right heel or foot-center position and velocity;
- gait phase and stance side;
- cadence and pre-onset walking speed; and
- validity flags for every channel.

Signals are low-pass filtered only with a causal or pre-onset-only procedure. The common sampling rate is set to the highest rate reliably supported by both releases after the availability audit, then locked before outcome modeling. Resampling uses antialiasing for downsampling and never extrapolates through $t_0$. Short gaps may be interpolated using a method fitted only to surrounding pre-onset samples; longer gaps remain missing and are represented by $q_i$.

All normalizers are estimated inside the training fold. Spatial outcomes are divided by participant leg length. Temporal outcomes are divided by the participant's pre-perturbation stride time. Velocity is normalized by leg length per stride time. When a participant-specific reference is allowed, it may use that participant's unperturbed or pre-onset walking because those observations are available before prediction; it may not use their post-perturbation test outcomes.

### Recovery outcome vector

The primary model predicts four scalar outcomes over the first two recovery steps. Exact event detectors, thresholds, and fallback behavior are fixed during the availability audit and then frozen.

#### 1. First recovery-foot placement

Identify the first valid foot contact after $t_0$. Compute the horizontal displacement from the pelvis to the contacting foot at contact, project it onto the horizontal perturbation direction, and divide by leg length:

$$
y_{i,1} = \frac{\langle f_i(t_{\mathrm{contact}})-p_i(t_{\mathrm{contact}}), d_i \rangle}{\ell_i},
$$

where $f_i$ is foot position, $p_i$ is pelvis position, $d_i$ is the unit perturbation direction, and $\ell_i$ is leg length. The anterior-posterior and mediolateral components are retained as secondary diagnostics, but the projected scalar is the preregistered primary endpoint so the target remains four-dimensional.

#### 2. Peak pelvis-speed deviation

Construct a phase-matched pre-perturbation stride template from data available before onset. Measure the largest absolute difference between observed horizontal pelvis speed and the template during the two-step recovery window, normalized by baseline walking speed:

$$
y_{i,2} = \max_{t \in \mathcal{W}_i}
\frac{\left|v_i(t)-\tilde{v}_i(\phi(t))\right|}{\max(\bar{v}_i,\epsilon)}.
$$

Here $\mathcal{W}_i$ is the recovery window, $\tilde{v}_i$ is the pre-onset phase template, $\bar{v}_i$ is mean pre-onset speed, and $\epsilon$ is a training-fold numerical floor.

#### 3. Time to pelvis-speed stabilization

Starting at $t_0$, find the earliest time at which pelvis speed enters a participant-specific tolerance band around the pre-onset phase template and remains there for a locked dwell duration. Divide elapsed time by pre-onset stride time. The tolerance and dwell duration are chosen using training participants only. If stabilization is not observable within the valid recording, the endpoint is marked unavailable rather than imputed from later or invalid data. The joint loss uses an outcome-validity mask, and missingness is reported by protocol and intervention.

#### 4. Maximum mediolateral pelvis excursion

Measure the maximum absolute pelvis displacement from its extrapolated pre-onset path along the mediolateral axis during the recovery window, divided by leg length:

$$
y_{i,4} = \max_{t \in \mathcal{W}_i}
\frac{\left|p_{i,\mathrm{ML}}(t)-\tilde{p}_{i,\mathrm{ML}}(t)\right|}{\ell_i}.
$$

An extra-recovery-step indicator is secondary and is included only if the same event definition can be executed in both datasets. It is never substituted for one of the four primary outcomes after results are observed.

### Input representations

All representation arms receive the same common state, intervention descriptor, validity information, and training examples.

1. **Intervention-only baseline.** The model receives $u_i$ but no pre-state. Its prediction is $m(u_i)$.
2. **Raw temporal model.** A compact temporal network processes the normalized common-state sequence and validity mask. It is the principal identifiability baseline.
3. **Frozen S-JEPA.** The common pelvis and foot signals are deterministically mapped to the corresponding S-JEPA tokens. Channels unavailable in a protocol are masked rather than synthesized. The encoder is frozen; only the response head is trained.
4. **S-JEPA with outcome-free adapter.** A rank-8 adapter may be trained on pre-perturbation walking windows using the original predictive objective and no recovery labels. The response head is then trained after the adapter is frozen.
5. **Random-encoder control.** An encoder with the same architecture but random frozen weights is paired with the identical response head.

The raw and representation heads use the same temporal pooling and intervention-conditioning design. Trainable parameter counts are matched within 5%; update counts, optimizer family, early-stopping budget, and hyperparameter search budget are identical. If exact matching requires widening the raw model, the raw model is widened rather than shrinking the proposed model.

### Intervention conditioning and residual predictor

The intervention vector is embedded by a small multilayer perceptron. Direction is represented continuously rather than as an arbitrary class index; phase uses sine and cosine to preserve circularity. The temporal representation of $x_i$ is concatenated with the intervention embedding and passed to a zero-initialized residual branch. With zero residual weights, the model begins at $m(u_i)$, making optimization failure less likely to create an artificial disadvantage for the baseline.

The primary probabilistic head is a three-component Gaussian mixture over the four standardized residual outcomes. For batch size $B$, it emits mixture logits of shape $[B, 3]$, component means of shape $[B, 3, 4]$, and diagonal log scales of shape $[B, 3, 4]$. The head therefore represents multiple plausible recovery modes while remaining feasible for small datasets. A full-covariance mixture is not used unless the participant count and numerical conditioning support it under a prespecified gate.

Training minimizes masked joint negative log likelihood. Scale floors and ceilings prevent degenerate variance estimates. Outcomes are standardized using training-fold statistics and transformed back to physical normalized units for reporting. The number of mixture components, scale constraints, and optimization budget are locked before any held-out participant or cross-protocol test is evaluated.

### Calibration

Calibration is performed on participants not used to fit model parameters. Marginal 80% intervals are derived from the predictive mixture and adjusted with one scalar expansion factor per outcome on the calibration participants. No per-person test calibration is permitted. Report empirical coverage and mean interval width for each outcome, participant, direction, magnitude, and protocol.

Calibration is considered acceptable only when aggregate participant-macro coverage lies between 75% and 85%, no major intervention stratum shows systematic collapse, and the intervals are narrower than those of the parameter-matched raw quantile baseline at comparable coverage. Calibration cannot rescue a model whose point or distributional accuracy is worse than the baseline.

### Leakage prevention and quality control

The following checks are mandatory:

- participant identifiers are disjoint across training, calibration, validation, and test partitions;
- repeated trials from one participant never cross partitions;
- held intervention combinations are absent from every training participant, not merely from one test participant;
- normalizers, templates, missing-data rules, phase models, and calibration maps are fitted inside the relevant training split;
- all input construction ends strictly before $t_0$;
- perturbation order, trial number, filename tokens, and post-event duration are excluded from inputs;
- shuffled intervention tokens and shuffled participant histories eliminate any genuine conditional gain;
- a protocol-only classifier is trained on the common representation to quantify remaining measurement mismatch; and
- every exclusion is logged with a reason before outcomes are inspected.

### Reproducibility

The implementation records dataset versions, file checksums, parser versions, coordinate transforms, event-detection parameters, split manifests, checkpoint hashes, random seeds, package lockfiles, and hardware. Derived trial tables include participant, protocol, condition, intervention, availability flags, and exclusion reason but no undocumented manual corrections. Three seeds—fixed before the final test—are used for learned models. The maximum trainable compute budget is 24 H100-hours across all folds, seeds, and adapter arms.

## Experiments

### Experiment 0: Availability and harmonization gate

For an implementation-oriented walkthrough, see the [Experiment 0 availability and harmonization guide](experiment-0-availability-harmonization-guide.md).

The first experiment does not train S-JEPA heads. It parses all Georgia Tech trials and at least two Dryad participants, aligns perturbation events, constructs the common state, and derives the four outcomes. Random trials from every direction, magnitude, and available condition are plotted in the harmonized coordinate frame and checked against metadata.

The gate passes if at least 90% of expected trials produce a valid pre-onset window and at least 90% produce each primary outcome under one implementation. Direction signs, stance labels, and contact events must agree with manual inspection on a prespecified audit sample. If one endpoint fails availability, its definition may be repaired using training/development data, but the repair is versioned and the gate is rerun from scratch. An endpoint cannot be silently dropped after model comparison.

### Experiment 1: Pre-state identifiability

This experiment asks whether individual pre-perturbation motion adds information beyond the intervention itself. Compare:

- the intervention-specific conditional mean $m(u)$;
- a linear model on handcrafted pre-state summaries;
- gradient-boosted raw summaries;
- a linear state-space or periodic-template predictor; and
- the parameter-matched raw temporal model.

Evaluation uses leave-one-participant-out outer folds on Georgia Tech. Within each outer fold, model selection and calibration use only the remaining participants. Report standardized MAE separately for all four outcomes and macro-average first within participant and then across participants.

The identifiability gate passes only if the raw temporal model reduces standardized MAE by at least 10% relative to $m(u)$ on at least three of four outcomes, the improvement is positive for a majority of held-out participants, and participant-cluster bootstrap intervals do not indicate that the aggregate gain is driven by one person. If this gate fails, representation experiments stop because there is no demonstrated person-specific pre-state signal to encode.

### Experiment 2: Representation value

Holding data, intervention conditioning, head design, and training budget fixed, compare raw temporal input, frozen S-JEPA, outcome-free rank-8 S-JEPA adaptation, and the random frozen encoder. The primary endpoint is participant-macro joint negative log-likelihood gain over `conditional mean + raw temporal state`. Secondary endpoints are energy score and outcome-specific standardized MAE.

The representation gate requires a positive 95% participant-cluster bootstrap interval for the joint negative log-likelihood gain. S-JEPA must also beat the random encoder and retain the same sign of gain across all three seeds. If raw state matches or exceeds S-JEPA, the response-prediction problem may remain valid, but there is no representation-learning contribution.

### Experiment 3: Held-intervention generalization

For each eligible direction-by-magnitude pair, remove that combination from all training and calibration participants and evaluate it on held-out participants. Hyperparameters remain fixed across combinations. Georgia Tech phase bins are additionally held out in contiguous ranges to test interpolation and extrapolation around gait phase. Direction, magnitude, and phase are never shuffled independently in a way that creates physically impossible labels.

Report performance by held combination and participant. The prespecified claim requires positive joint log-likelihood gain over the raw model for a majority of held combinations, not merely a positive pooled average. Dryad is excluded from the phase-generalization claim because its perturbations target one phase.

### Experiment 4: Distribution quality and calibration

Evaluate the three-component mixture against deterministic regression, diagonal Gaussian regression, and parameter-matched quantile regression. For each outcome, report 80% coverage, mean interval width, interval score, and coverage stratified by direction and magnitude. For the joint response, report negative log likelihood and energy score.

Calibration passes when participant-macro 80% coverage lies between 75% and 85% and mixture intervals are narrower than raw quantile intervals at comparable coverage. Reliability plots are generated from training/calibration decisions fixed before test evaluation. A model cannot be called calibrated solely because pooled coverage is correct while individual intervention strata are severely undercovered.

### Experiment 5: Cross-protocol transfer

The primary transfer direction trains all representation and response parameters on Georgia Tech. A small, prespecified set of Dryad calibration participants is used only to fit an affine location-and-scale map for each output and the interval expansion factors. Remaining Dryad participants are untouched test participants. Calibration and test identities rotate through a fixed participant-level scheme so every reported test prediction is out of participant.

The primary Dryad endpoint uses the normal-shoes condition. The ankle-brace, vision-blocked, and pneumatic-jet conditions are evaluated separately as distribution-shift stress tests. The model does not receive those conditions during Georgia Tech training, and their scores are never pooled with normal walking to improve the headline result.

Transfer passes if the S-JEPA model retains positive participant-macro joint log-likelihood gain over `conditional mean + raw state` after affine calibration only. Fine-tuning the encoder or response head on Dryad is prohibited for this endpoint. Reverse transfer—Dryad to Georgia Tech—is reported as a sensitivity analysis, with the same separation between parameter fitting, calibration, and untouched test participants.

### Experiment 6: Controls and ablations

The following controls test whether an apparent gain reflects the proposed mechanism:

| Control | Failure it detects | Expected behavior if the mechanism is genuine |
| --- | --- | --- |
| Intervention-only neural model with matched capacity | Gain from nonlinear intervention decoding alone | Remains below models using valid pre-state |
| Pre-state from another participant matched on intervention | Protocol or intervention leakage | Loses the person-specific gain |
| Temporal shuffle within the pre-onset window | Static anthropometry or summary leakage | Reduces gains requiring motion dynamics |
| Intervention-token shuffle | Outcome imbalance or trial-order leakage | Eliminates conditional prediction |
| Random frozen encoder | Head capacity rather than learned representation | Underperforms frozen S-JEPA |
| No protocol identifier | Apparatus-specific mismatch | Reveals how strongly protocol identity is needed |
| Validity mask only | Missingness leakage | Produces no material gain |
| Post-onset contamination audit | Future leakage | Changing post-onset samples leaves every input unchanged |
| Persistence and periodic templates | Easy dynamical continuation | Remain below the strongest learned model if added complexity is warranted |

Additional ablations remove phase, stance side, recent variability, or intervention magnitude one at a time. These are interpreted as dependence tests, not as evidence that the removed variable is causal.

### Metrics and statistical analysis

The primary metric is joint negative log likelihood on held-out participants and held interventions. Supporting metrics are:

- standardized MAE for each recovery outcome;
- energy score for the joint predictive distribution;
- 80% marginal interval coverage and width;
- interval score;
- participant-level win rate over the strongest raw baseline; and
- calibration error by intervention stratum.

Metrics are computed per trial, averaged within participant, and then averaged across participants so repeated trials do not make one person dominate. Uncertainty is estimated with participant-cluster bootstrap resampling. The primary representation comparison uses a two-sided 95% bootstrap interval on paired participant-level log-likelihood differences. The four outcome-specific MAE tests are governed by the prespecified three-of-four gate rather than selectively reporting the most favorable endpoint.

### Decision sequence

The study advances through the following fixed sequence:

| Stage | Required evidence | Consequence of failure |
| --- | --- | --- |
| Availability | At least 90% valid harmonized trials and outcomes | Repair parsing or stop |
| Identifiability | At least 10% MAE reduction on three of four outcomes | Stop representation claim |
| Representation | Positive bootstrap interval for S-JEPA joint NLL gain | Retain only raw response model |
| Held intervention | Gain across a majority of unseen combinations | Restrict claim to seen interventions |
| Calibration | 75% to 85% coverage for nominal 80% intervals | Do not claim a calibrated envelope |
| Transfer | Positive gain after affine calibration only | Restrict claim to the source protocol |

No threshold is relaxed after a later-stage failure. All failed gates, excluded trials, and protocol-specific effects remain part of the experimental record.

### Planned execution schedule

Days 1–2 cover data parsing, event alignment, outcome validation, and the availability and identifiability gates. Days 3–4 freeze split manifests, cache S-JEPA features, and run representation comparisons. Days 5–7 cover held-intervention and calibration experiments. Days 8–10 run cross-protocol transfer in both directions. Days 11–12 run three-seed replication, participant-cluster bootstrap analyses, and ablations. Days 13–14 are reserved for figure generation, error-case auditing, and verification that the final claims follow the prespecified decision sequence.
