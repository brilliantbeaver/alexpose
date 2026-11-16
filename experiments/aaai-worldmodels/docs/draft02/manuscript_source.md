# Reference-Aware Gait Anomaly Detection for Medical World Models

## Abstract

Gait video can describe how a person's movement changes, but prediction error needs a reference before it can serve as an anomaly score. A restricted yet repeatable gait can be easier to predict than ordinary walking. We present a source-grouped benchmark that compares learned forecasts with persistence, which repeats the last observed pose. Normal-only models predict eight future samples from public gait videos. Across 2,815 windows from 91 source videos, the bilateral model reduces normal-source forecast error by 33.5%. Its prespecified excess error above persistence ranks abnormal annotations with AUROC 0.793, compared with 0.304 for raw forecast error. A post hoc low-movement diagnostic reaches 0.806, showing why the reference itself must be evaluated. Current posture also forecasts better than bilateral history. These results provide a practical way to separate forecasting skill, movement amount, and anomaly ranking. They support an exploratory motor-observation benchmark, without establishing clinical diagnosis or a learned anomaly advantage over movement alone.

## Introduction

A medical world model needs observations that describe a patient's state and a model of how that state changes. Gait provides one possible observation. Videos reveal relative motion of the legs even when clinical sensors are unavailable. Before these observations support clinical simulation, however, we need to know what their prediction errors mean.

A common approach trains on normal movement and treats large prediction errors as evidence of abnormality. This assumes that abnormal movement is harder to predict. The assumption can fail when a restricted or repeatable pattern produces little change between frames. Repeating the last pose can then obtain a small error without describing healthy movement.

We address this problem with a reference-aware evaluation. The same observed future is scored against a learned forecast and against persistence. Their difference describes how much the learned model helps beyond holding the last pose fixed. We then evaluate the reference itself, the importance of movement history, and the effect of keeping source videos separate. The contribution is a reproducible benchmark and an interpretable score analysis. It provides a practical test of whether a gait observation model learns information useful for anomaly ranking beyond simple movement differences.

## Neurological motivation and related work

Walking coordinates body segments while maintaining support and advancing the body. Laboratory measurements show structured relationships between segment angles [borghese1996]. Bilateral timing is clinically relevant: Plotnik and colleagues measured changes in the coordination of left and right steps in Parkinson's disease [plotnik2007]. Studies of stroke and cerebral palsy link altered muscle-activation patterns with impaired walking [clark2010; steele2015]. These observations motivate testing relationships between the legs. They do not imply that every disorder increases prediction error or that video coordinates identify a neural mechanism. The myopathic annotation also describes a neuromuscular category, rather than a shared neurological cause across all included conditions.

MoCoDAD uses skeleton forecasts for anomaly detection [flaborea2023]. FSGait combines normal-only gait reconstruction and prediction to detect postural and temporal abnormalities [duan2024]. GenGait studies normative reconstruction [motta2026], while S-JEPA predicts hidden skeleton representations for action recognition [abdelfattah2024]. Normal-only gait prediction is therefore established prior art. Following the importance of simple motion references [martinez2017], our contribution evaluates forecast skill and anomaly scores under one timing, observation, and source-grouping contract.

## Data and observation contract

The pinned local GAVD inventory contains 1,874 sequences from 348 source IDs [ranjan2025]. The five selected annotations contain 666 sequences from 103 sources. Available-media checks retain 652 sequences from 98 sources. Two documented source exclusions for identity or scene problems leave 641 extraction candidates from 96 sources. Timing, geometry, observation, and common-target gates retain 2,815 windows from 456 sequences and 91 sources. These video groups are not verified independent participants.

Ambient performs 46,783 recorded pose-model calls over the available annotated spans. We sample nearest native frames on a nominal 10 Hz grid and retain actual frame numbers and times. Each window has 16 observed and eight future samples, with stride eight. The observed prefix spans 1.5 seconds. The final future sample occurs 0.767 to 0.834 seconds later. We evaluate all eight future samples, not only the final instant.

The contract uses 16 BlazePose joints, confidence at least 0.5, and at least 70% valid observations separately in the prefix and future. An annotation-box check rejects unresolved geometry, although it cannot guarantee identity in crowded scenes. Coordinates are centered on each frame's pelvis and divided by a prefix-only torso scale. They describe relative 2D articulation, not physical speed. Both hips define the reference, so the own-leg model does not have strictly unilateral visual access.

Observation gates retain different fractions of the annotations. For example, 141 of 287 normal extraction candidates remain, compared with 168 of 183 myopathic candidates. All 32 annotated normal source IDs are already in the selected scope. Other GAVD labels cannot increase this normal-source ceiling, and additional windows do not create independent people.

## Forecasts and anomaly scores

For each target leg, ridge regression predicts future knee, ankle, heel, and foot-index coordinates. This linear model penalizes large coefficients to limit overfitting. Inputs use eight fixed prefix positions, including both endpoints, with observation indicators. Missing values are filled from earlier prefix observations only. Outputs are trained only where future targets were observed.

The own-leg model uses that leg's history. The bilateral model adds the opposite leg. A separately trained timing control shifts the seven earlier opposite-leg samples by a window-specific offset while preserving their values, validity, and current anchor. The current-posture model learns a separate transition at each future time from the latest bilateral pose. Its output can change over time. Persistence instead holds the latest pose fixed. Damped velocity and declared shrinkage variants provide additional references.

Let $a$ be the latest observed coordinate, $y$ its observed future, and $\widehat y$ the learned prediction. Actual displacement is $d = y - a$ and predicted displacement is $u = \widehat y - a$. Excess squared error is $(d - u)^2 - d^2 = u^2 - 2du$. The identity also holds after averaging with the same observed-coordinate support and hierarchical weights. Excess error therefore combines prediction size and alignment with actual movement. Subtracting persistence does not remove every effect of movement amount, view, or pose noise.

Larger excess error is prespecified to indicate more abnormal gait. A negative value means forecasting improved on persistence. Other declared scores are raw bilateral error, persistence error, bilateral-minus-own error, and prefix missingness. Because all error scores require the observed future, they detect anomalies retrospectively within a sequence. They do not predict an abnormal event before it occurs.

## Evaluation

Five outer folds keep all sequences and windows from one source together. Models train only on normal sources outside the evaluation fold. Three inner source folds select regularization from 0.1, 1, 10, and 100. Scaling is refitted within every training split. A secondary analysis blends learned displacement with persistence using strengths 0, 0.25, 0.5, or 1, selected on inner normal-source validation. Abnormal labels do not select model settings.

All methods share observed future targets. We average coordinate errors within a leg and window, the two legs equally, then windows within sequences, sequences within sources, and sources equally. Training weights follow this hierarchy. Forecast evaluation uses 29 normal sources. Anomaly ranking compares 62 abnormal and 29 normal sources. AUROC is the probability that an abnormal source receives a higher score than a normal source; average precision also depends on prevalence.

The frozen analysis uses 5,000 source resamples for forecast intervals and 2,000 stratified source resamples for anomaly intervals. These intervals condition on fixed models with overlapping training folds. They exclude retraining and participant-population uncertainty. The study is exploratory because GAVD outcomes were available during research development.

This report adds an explicitly post hoc low-movement diagnostic: negative persistence error. It assigns larger scores to smaller observed movements without changing any frozen score direction. We compare it with excess error using 5,000 paired stratified source resamples. Within-fold rankings avoid comparing scores from different fitted models. These added analyses do not select a model, threshold, or clinical rule.

## Results

Normal-source mean squared error is 0.12126 for persistence and 0.08063 for bilateral forecasting. The paired reduction is 0.04063 [0.02399, 0.05911], or 33.5% of the persistence mean. Bilateral forecasting improves error on 23 of 29 normal sources. This establishes useful forecast information within the observation contract.

The gain over own-leg history is 0.00111 [-0.00253, 0.00524], and the gain over the trained timing control is 0.00591 [0.00141, 0.01134]. However, current posture forecasts better, with error 0.07164, a 40.9% reduction from persistence. Longer bilateral context is therefore not established as necessary. Shrinkage yields bilateral error 0.08378 and does not improve the unshrunk model.

Prespecified excess error achieves AUROC 0.793 [0.680, 0.896] and average precision 0.865 [0.788, 0.938]. Raw forecast error has AUROC 0.304 [0.178, 0.431], and persistence error has AUROC 0.194 [0.096, 0.300]. Abnormal-source prevalence is 0.681. These differences show that an error-based ranking depends strongly on its reference and interpretation.

The post hoc low-movement diagnostic has AUROC 0.806. The paired difference, excess error minus low movement, is {{paired_difference}}. This does not establish added anomaly-ranking value from the learned forecast. Excess-error AUROC within individual folds ranges from 0.653 to 0.931. Weighting those rankings by the number of normal-abnormal source pairs gives {{within_excess}}, compared with {{within_motion}} for low movement. Their paired difference is {{within_difference}}. The association persists without cross-fold pairs, while small source groups produce substantial variation. Complete scores and condition slices appear in the appendix.

## Discussion and limitations

The study supplies a useful observation-model benchmark and a concrete lesson for gait anomaly detection. A model can forecast normal movement well while its raw error ranks restricted abnormal movement as easier. Comparing forecasts with persistence exposes this distinction. Evaluating persistence as an anomaly diagnostic then tests whether the apparent benefit comes from learned behavior or movement amount. This sequence of checks is the methodological contribution.

The next model should demonstrate added information beyond current posture and movement-only scores. Promising tests include normal-only calibration conditional on prefix motion and view, and structured masks that hide a leg or a continuous time interval. Such masks could encourage use of coordination rather than nearby redundant samples. These are proposed extensions that require matched budgets, fixed targets, timing controls, and an untouched evaluation cohort.

GAVD supports exploratory evaluation across heterogeneous public-video sources, but does not establish clinical generalizability. Participants may recur across uploads, a video may contain several people, and local annotations do not independently verify diagnoses. View, age, walking task, assistive devices, acquisition quality, and unequal retention can influence scores. More overlapping windows cannot resolve these limitations or enlarge the 29 retained normal-source groups.

The model describes short-term articulation without treatment inputs, repeated clinical visits, or measured intervention outcomes. Patient-level decisions require external participant-disjoint data and validated clinical measurements. Within this scope, the benchmark separates forecasting skill from anomaly ranking and defines clear tests for stronger medical claims.
