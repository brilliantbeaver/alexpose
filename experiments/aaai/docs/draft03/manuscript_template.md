---
title: "Controlled Cross-Leg Prediction for Gait Anomaly Detection"
abstract: |
  Models that detect unusual walking often learn to predict normal movement and then look for prediction errors. However, an error can also arise because joints are hidden or the model has learned a weak predictor. We present a controlled way to examine how much the other leg helps predict a hidden movement after the visible target leg has already been used. We preserve an own-leg prediction and compare corrections that use either the same leg or both legs. We also compare three ways of hiding observations during training while keeping evaluation tasks fixed. The exploratory study includes 304 GAVD video recordings and an independent laboratory dataset of 178 participants. Compared with the own-leg correction, the both-leg model reduces normal prediction error by @@GAVD_COMPLETION_PERCENT@@% when neighboring leg joints are hidden in GAVD and by @@LAB_COMPLETION_PERCENT@@% in the laboratory data. Anatomical masking reduces error on the grouped GAVD task by @@MASK_COMPLETION_PERCENT@@% relative to random masking, although its advantage on scattered hiding is uncertain. Better prediction does not automatically improve detection: movement-only and combined anomaly scores reach AUROCs of 0.694 and 0.576 in the main GAVD comparison, and 0.958 and 0.892 in the laboratory study. The method provides a reproducible way to test cross-leg evidence while preserving posture and movement size as separate information.
---

# Introduction

Gait describes how a person walks. A person compensating for weakness may repeat an unusual movement consistently, while ordinary walking can become difficult to predict when one leg hides the other. An anomaly detector must account for these different reasons why prediction error might change.

The relationship between the legs offers possible information beyond the movement of either leg alone. Suppose that a short section of the left ankle trajectory is hidden. The left knee and foot may already provide enough information to reconstruct much of it. The right leg could supply additional information about the walking pattern, but a comparison must establish what it adds beyond this local explanation.

We study that question using regularized linear models fitted to joint positions over time. The first predictor uses the visible target leg. We keep this baseline fixed and learn two corrections: one receives additional features from the same leg, while the other receives features from both legs. Both predict exactly the same hidden coordinates. This provides a stronger comparison than adding the other leg to an uncorrected baseline, although equal feature counts do not eliminate every difference in model flexibility.

Scattered missing points can leave nearby joints available to reconstruct the answer, whereas hiding neighboring joints together removes some of that help. We compare training masks on fixed evaluation tasks to measure how the pattern of missing observations affects the result.

The contribution is a controlled procedure for assessing cross-leg prediction as anomaly evidence. Experiments on web videos and independent laboratory participants test prediction quality under explicit input restrictions. We evaluate anomaly ranking separately, because a relationship that reconstructs normal walking may still add little to posture and movement measurements.

\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/method.pdf}
\caption{The study compares three predictions of the same hidden movement and measures posture and movement size separately. The own-leg baseline stays fixed while the two corrections are fitted. Their error difference measures the benefit of the both-leg model under the specified inputs. The anomaly analysis then asks whether an unusual benefit helps distinguish abnormal recordings.}
\label{fig:method}
\end{figure*}

# Background and motivation

Self-supervised models learn from the data themselves, for example by predicting hidden observations. MAMP applies this approach to skeleton motion. S-JEPA predicts a learned representation supplied by a teacher network [@mao2023mamp; @abdelfattah2024sjepa]. Our core experiment instead uses ridge regression, a regularized linear model, to predict coordinates or measured angles. This keeps the answer the same across models without requiring a JEPA encoder.

VideoMAE hides the same image regions across time, while SLiM hides anatomical groups to reduce easy reconstruction from nearby observations [@tong2022videomae; @do2026slim]. Our comparison uses established mask families with equal hiding rates. Unlike motion-aware sampling in MAMP and S-JEPA, mask selection does not depend on movement speed or size.

Several anomaly methods already learn from normal movement. FSGait models posture and time, GenGait uses masked gait reconstruction, SeeKer estimates conditional keypoint distributions, and STG-NF estimates pose likelihoods [@duan2024fsgait; @motta2026gengait; @delic2025seeker; @hirschorn2023stgnf]. Our focus is the comparison needed to interpret added context. The target remains fixed, the local prediction is preserved, and prediction quality is checked again when fewer observations are available.

Clinical findings motivate preserving posture, movement size and timing. Stroke can produce different adaptations on the two sides [@clark2010modules; @awad2020drive], while Parkinsonian movement size and arm coordination can change differently [@morris1994hypokinesia; @sterling2015armswing]. Cerebral palsy combines motor-control and musculoskeletal limitations, and myopathic conditions involve muscle disease [@steele2015synergies; @steele2012crouch; @vandekerckhove2025dmd]. Because mechanics and task constraints also produce joint relationships [@kutch2012synergies], video prediction cannot identify a neurological cause.

# Data and evaluation design

## Video recordings and laboratory participants

GAVD contains web videos with annotated walking sequences and gait categories [@ranjan2025gavd; @gavd2026catalog]. We rebuild the inventory from the current condition CSV files. It contains 1,874 sequences from 348 upload IDs, of which 332 have decodable videos. Fixed pose-quality and lower-leg observation requirements retain 304 recordings (Table \ref{tab:attrition}). A confirmed duplicate pair fails those requirements, but different uploads could still show the same person.

\begin{table}[t]
\centering
\begin{tabular}{lrrr}
\toprule
Population & Units & Normal & Windows \\
\midrule
GAVD annotations & 348 & 32 & -- \\
Usable pose windows & 311 & 28 & 9,474 \\
Leg observations & 304 & 28 & 9,442 \\
Four-condition set & 90 & 28 & 2,594 \\
Laboratory, selected & 188 & 138 & -- \\
Laboratory, eligible & 178 & 128 & 433 \\
\bottomrule
\end{tabular}
\caption{The available data after each selection step. GAVD units are video uploads, while laboratory units are participants. Windows are short, overlapping portions of a recording and are not independent people. Normal laboratory participants are labeled healthy in that dataset.}
\label{tab:attrition}
\end{table}

The main comparison contains 28 normal recordings and 62 recordings labeled Parkinsons, stroke, cerebral palsy or myopathic, with respective counts of 10, 15, 10 and 27. A second analysis treats every non-normal folder as abnormal across all 304 recordings. One cerebral-palsy recording also has a generic abnormal annotation. It counts once in binary evaluation but is excluded from the separate five-class experiment, leaving 89 recordings. These folder labels are not independently confirmed diagnoses.

We also use a public laboratory dataset with simultaneous hip, knee and ankle flexion angles from both legs [@vancriekinge2023dataset]. One walking trial per participant is selected before inspecting results. Of 188 selected trials, ten healthy trials lack the required observations, leaving 128 healthy and 50 stroke participants. This allows a participant-separated test with a different measurement system. We fit new models to the angles rather than transfer video-trained weights.

## Keeping training and testing separate

The GAVD evaluation uses four outer folds. In each fold, 16 normal recordings fit the models, five other normal recordings define a reference distribution for cross-leg prediction, and seven normal recordings are held out for testing. Abnormal recordings enter testing only, except in the separately labeled supervised comparisons. Three internal folds within the fitting set select model settings. All sequences, overlapping windows and modified inputs from one recording stay in the same role.

This analysis is exploratory because the 28 normal recordings informed method development. The population, comparisons, roles, masks and scores were fixed before fitting the reported models. The five reference recordings do not select model settings or score weights. Models share training recordings across folds, so four folds are not four independent studies. The laboratory analysis follows the same separation at participant level, using 76 healthy participants for fitting, 20 for reference estimation and 32 for testing in each fold.

## Turning video into model inputs

We crop each annotated person, run BlazePose to estimate joint locations and restore the coordinates to the full image [@bazarevsky2020blazepose]. The extractor returns the first pose in the crop, so person selection and continuity can still fail. Measurements with confidence below 0.5 remain missing. We resample at 15 Hz, interpolating only between adjacent valid observations, and use overlapping 32-sample windows.

A token contains four consecutive valid samples of one joint. Sixteen torso, arm and leg joints supply 128 tokens per window. At least 75% must be valid overall and 50% in each half. Final readouts use eight knee, ankle, heel and foot-index measurements, while window-quality checks still include all 16 joints.

A shared body reference, denoted $Q$, centers coordinates on the pelvis and divides them by torso length measured in the first half of the window. We preserve trunk tilt, movement size and the timing shared by both legs. The own-leg restriction is therefore conditional on this reference; it does not exclude every opposite-side body measurement. Image coordinates and projected angles do not measure walking speed, ground clearance or clinical three-dimensional angles.

# Measuring the value of the other leg

## Three predictions of the same answer

Let $Y$ be the hidden four-sample coordinate block, $O$ the visible target-leg observations and $C$ the visible opposite-leg observations. Each leg includes its hip, knee, ankle, heel and foot index. We remove hidden and invalid inputs before constructing features. This prevents a hidden answer from returning through a derived angle, interpolation estimate or embedding, conditional on $Q$. Visible observations can occur before and after the hidden block, so the task is completion rather than future forecasting.

The three predictions are

\begin{align}
\widehat Y_0 &= f(O;Q),\\
\widehat Y_o &= f(O;Q)+\alpha_o h(O;Q),\\
\widehat Y_b &= f(O;Q)+\alpha_b g(O,C;Q).
\end{align}

The own-leg baseline $f$ uses interpolation estimates, observation-availability indicators, relative time and target-joint identity. Once fitted, it remains fixed. The own-leg correction $h$ adds summaries of that leg's movement over time, while the both-leg correction $g$ adds opposite-leg features. The multipliers control correction size and include zero, which reproduces the saved baseline exactly.

The corrections each have 154 features, but equal feature counts do not guarantee equal flexibility. Their numerical ranks are 111 and 141, indicating different numbers of independent feature directions under the recorded threshold. These are not ridge effective degrees of freedom, so gains apply to the fitted model families.

Normal-only validation selects settings. The both-leg model must beat both local alternatives in at least two of three internal folds and in average error. This practical rule is supported by software checks for known dependence, conditional independence, exact zero corrections and unchanged predictions when inaccessible inputs change. The generated examples do not simulate disease.

## Hiding observations during training and evaluation

All training methods hide exactly 64 of 128 tokens. Random masking, U, samples them uniformly. Whole-joint masking, J, hides all eight time blocks of one randomly chosen joint, then hides 56 additional tokens at random. Anatomical masking, A, hides a small body region over consecutive times, also totaling eight tokens, then fills the remaining 56 positions randomly. The regions cover the torso, either arm, or the knee-to-foot joints on either side.

The anatomical group probabilities and permitted starting times give every token the same chance of entering the initial eight-token block. After random filling, each token has hiding probability 0.5 under all three methods. This changes which observations disappear together while holding the total and individual probabilities fixed. It does not make the tasks equally difficult or guarantee equal numbers of usable targets after naturally missing measurements are removed. Masks are drawn before inspecting validity and are never redrawn to rescue an example.

Every method shares the same random-trained baseline, fitting recordings and presentation schedule. Unrelated random choices are kept separate from mask sampling, preventing changes in the mask generator from silently altering other parts of training.

Evaluation uses two fixed sets of 16 masks. The main task, scattered hiding, uses random masks and requires at least two visible blocks of the target joint. The grouped-hiding task removes the knee, ankle, heel and foot index together over eight consecutive frames. It scores these hidden blocks and requires at least two visible tokens at different times on each leg. Both tasks require valid answers and four distinct target blocks for every scored joint in a recording. All 304 recordings satisfy both tasks; repeating a mask does not increase the distinct-target count.

Training methods are compared within each task on identical targets. Anatomical versus random training on scattered hiding is the main masking comparison; grouped hiding is a separate diagnostic. Both tasks use the same five reference recordings but estimate their reference parameters separately.


# From prediction to anomaly scores

## Movement and cross-leg scores

The movement score $M$ describes how unusual posture and movement size are compared with normal recordings. Its 16 features include the level and variation of knee angles, ankle movement ranges, variability of ankle velocity, trunk tilt and left-right differences. We average windows within sequences and sequences within recordings before fitting missing-value replacement, robust scaling and a shrinkage covariance estimate on fitting-normal recordings. The score is squared Mahalanobis distance divided by 16, which measures departure from normal while accounting for feature variation. These full-observation features never enter a restricted predictor.

The cross-leg score $R$ measures how unusual the benefit from the both-leg model is. It begins with two errors on the same hidden answer:

\begin{equation}
b=\operatorname{MSE}(\widehat Y_o,Y)-\operatorname{MSE}(\widehat Y_b,Y).
\end{equation}

Here MSE is mean squared error, and positive benefit $b$ means that the both-leg correction predicts more accurately. We average repeated masks of one target, then targets within windows, windows within sequences and sequences within recordings. Averaging the four scored joints on each side gives every recording a left-target and a right-target benefit.

The five reserved normal recordings supply a reference mean and variance for each side. $R$ averages the two squared, variance-scaled distances from these means. Either an unusually large or an unusually small benefit can raise $R$; larger benefit is not a ranking of health. Each variance is stabilized by averaging it with the mean variance across sides and adding $10^{-8}$. This score treats the sides separately rather than modeling their covariance.

The combined score is fixed as $S=(M+R)/2$. Dividing by feature count makes the distances easier to compare, but does not turn them into equally calibrated probabilities. A fold that fails the internal prediction check would receive $R=0$ for all recordings. Every fitted model passes that internal check in this study.

## Comparisons and uncertainty

We compare these scores with a richer 36-feature movement distance, timing and waveform descriptors, and Isolation Forest. Additional controls replace the cross-leg benefit with the improvement from the own-leg correction, or use distances based on the prediction errors themselves. These controls examine whether another score component helps simply because it adds information beyond $M$.

Scores based on measurement availability and viewpoint assess recording-related differences, including a separately supervised classifier. For a published pose comparator, we adapt the authors' STG-NF implementation to the 16-joint graph and replace missing values with fitting-normal medians. Only five normal recordings have complete poses, so an unmodified complete-case comparison cannot use the same split. The reported adapter is distinct from a reproduction of the published method.

A separate five-class experiment fits Random Forest and learned/random JEPA-feature classifiers. The learned encoder trains on fitting normals with a slowly updated teacher and VICReg regularization against constant or redundant representations [@bardes2022vicreg].

AUROC measures how often an abnormal recording ranks above a normal recording. We calculate it within each fold and weight the average by the number of normal-abnormal pairs, avoiding comparisons across differently fitted score scales. We also report average precision with each fold's abnormal fraction.

Paired bootstrap intervals resample whole recordings within fold and label group, keeping all method scores together and preserving the fold weights. Laboratory intervals resample participants instead. These intervals describe uncertainty conditional on the saved models and selected analysis. They omit retraining uncertainty, unknown repeated people across uploads and method-development choices. Multiple comparisons are exploratory and receive no multiplicity adjustment.

## Checking sensitivity to the other leg's timing

The timing diagnostic keeps the answer and visible target leg fixed while taking opposite-leg observations from four frames earlier or later, about 0.27 seconds. We use the middle 32 frames of a 48-frame interval, without wrapping, padding or crossing recording gaps. Donor joint positions are expressed relative to their visible hip and attached to the original visible hip, using the unchanged body scale.

Only observations available in both the aligned and shifted versions are allowed, so hidden donor values cannot reappear through the edit. This can leave much less context than the completion task. We therefore check prediction quality again on the exact aligned inputs used by the timing comparison.

The measured effect is shifted error minus aligned error from the fixed both-leg model. A zero shift must reproduce the input, and changing only the other leg must leave the own-leg prediction unchanged. Noise, missing-observation and whole-body interval-shift controls accompany the test. Even a valid response would reflect both changed alignment and changed instantaneous geometry, rather than isolate a neurological timing mechanism.

\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/positive_completion.pdf}
\caption{The left panel compares four ways to complete laboratory angles; each line represents a test fold, and lower error is better. The right panel compares anatomical with random training on the two GAVD tasks. Positive percentages mean lower error with anatomical training; intervals crossing zero leave the direction uncertain. The 95\% intervals resample recordings while keeping fitted models fixed. The panels show different comparisons and should not be pooled.}
\label{fig:positive}
\end{figure*}

# Results

## Predicting hidden movement in GAVD

On grouped hiding, the random-trained both-leg correction reduces mean error across 28 held-out normal recordings from 0.080769 to 0.065020 relative to the own-leg correction. This is a @@GAVD_COMPLETION_PERCENT@@% reduction in mean error. The absolute benefit is 0.015748, with conditional 95% interval [0.010721, 0.021300] from 5,000 within-fold recording resamples.

All three training methods have positive benefit intervals on this task. In contrast, the benefit remains uncertain for random and anatomical training on scattered hiding. When neighboring joints disappear together, local geometry supplies less information and the opposite leg is more useful to these models. The targets remain estimated image coordinates, so this measures recovery of deliberately hidden observations rather than independently verified motion during natural occlusion.


## Prediction in independent laboratory participants

The laboratory model uses six simultaneous joint-angle signals with their common timestamps preserved. Each 32-sample window has 48 four-sample tokens, half of which are hidden. Twelve angle-level and angle-range features replace the video-specific movement features, and the same internal prediction rule selects corrections. The angles are processed measurements from a shared body model and are not interchangeable with BlazePose coordinates.

The both-leg correction improves healthy-participant error by 4.78% to 5.52% in each fold. On the same held-out targets, direct interpolation gives mean MSE 309.838 squared degrees, the own-leg baseline gives 206.262, the own-leg correction gives 201.038 and the both-leg correction gives 190.584. Every fold has this ordering (Figure \ref{fig:positive}). The corrections each have 72 features, but their numerical ranks are 44 and 62. As with GAVD, the gain applies to these fitted feature families.

Across participants, the both-leg correction reduces mean error by 10.454 squared degrees, with conditional interval [7.154, 13.387]. The relative reduction is @@LAB_COMPLETION_PERCENT@@%, with interval [3.6%, 6.7%], and @@LAB_IMPROVED@@ of @@LAB_NORMALS@@ healthy participants improve. These descriptive intervals use 1,000 paired resamples within the fixed folds and were calculated after inspecting the fitted results. They exclude training and analysis-selection uncertainty.

## How the training mask changes prediction

Anatomical training reduces both-leg error on grouped hiding from 0.065020 to 0.060009 relative to random training. The reduction is @@MASK_COMPLETION_PERCENT@@%, with conditional interval [6.4%, 9.0%]. All four fold means and 27 of 28 normal recordings improve. Because this task resembles part of the anatomical training pattern, the result shows a benefit from matching training to a particular missing-observation task.

The same comparison on scattered hiding gives only a 0.8% reduction, with interval [-1.0%, 2.6%]. Sixteen recordings improve, and one fold's mean becomes worse. This leaves the general advantage of anatomical masking uncertain. Both evaluation tasks were fixed before fitting. The paired error intervals are descriptive analyses of saved predictions calculated after outcome inspection.


\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/timing_limits.pdf}
\caption{The timing test uses the random-trained model on 27 normal recordings. On the left, negative benefit means the both-leg correction is less accurate than the own-leg correction on the aligned inputs available to each comparison. On the right, positive change means shifting the other leg increases error. The earlier shift has a measurable response, but poor aligned prediction limits its interpretation. Error bars are 95\% intervals from recording resamples.}
\label{fig:timing}
\end{figure*}

## Distinguishing normal and abnormal recordings

Table \ref{tab:detection} compares scores on the same recordings; prediction-based scores use scattered hiding. In the four-condition GAVD comparison, movement alone reaches AUROC 0.694 and the combined score reaches 0.576. Their paired difference, combined minus movement, is -0.118 with interval [-0.217, -0.025]. Across all abnormal folders, the scores reach 0.707 and 0.631, respectively, with difference -0.076, [-0.143, -0.009].

\begin{table}[t]
\centering
\begin{tabular}{lrr}
\toprule
Score or method & \shortstack{Four\\groups} & \shortstack{All\\folders} \\
\midrule
Movement score M & 0.694 & 0.707 \\
Cross-leg score R & 0.445 & 0.560 \\
Combined S (random) & 0.576 & 0.631 \\
\addlinespace
Movement, 36 features & 0.677 & 0.669 \\
M + own-leg improvement & 0.687 & 0.661 \\
M + correction errors & 0.684 & 0.685 \\
M + baseline error & 0.700 & 0.692 \\
Timing and waveform features & 0.588 & 0.631 \\
Isolation Forest & 0.551 & 0.593 \\
STG-NF, adapted & 0.376 & 0.435 \\
Availability and viewpoint & 0.389 & 0.520 \\
Supervised recording features & 0.461 & 0.506 \\
\addlinespace
Combined S (whole joint) & 0.620 & 0.644 \\
Combined S (anatomical) & 0.631 & 0.672 \\
\bottomrule
\end{tabular}
\caption{GAVD AUROC averaged within the four test folds as described in the text. Higher values indicate better ranking of abnormal recordings. The columns contain 62 abnormal recordings out of 90 and 276 out of 304. All rows use the same eligible recordings. The recording-feature classifier uses abnormal training labels; STG-NF uses the stated missing-value adaptation.}
\label{tab:detection}
\end{table}

For the main masking comparison, anatomical training raises the combined score's AUROC by 0.055 relative to random training, but its interval [-0.018, 0.127] includes zero. On grouped hiding, combined-score AUROCs for random, whole-joint and anatomical training are 0.668, 0.668 and 0.684. These concern a different prediction task and do not replace the scattered-hiding comparison.

With random training, average precision in the four-condition comparison is 0.763 for the combined score and 0.864 for movement alone. The abnormal fraction ranges from 0.682 to 0.696 across folds, helping explain the relatively high precision values. The own-leg-improvement and correction-error controls reach AUROCs of 0.687 and 0.684. They show why another prediction-derived score must be compared with movement alone rather than credited automatically to cross-leg information.

In the laboratory data, movement, cross-leg and combined scores reach AUROCs of 0.958, 0.694 and 0.892. Combining the scores changes AUROC by -0.066, with interval [-0.106, -0.031]. A fixed same-sex, within-five-year matching rule retains 45 healthy/stroke pairs and gives movement and combined AUROCs of 0.955 and 0.874, a difference of -0.081, [-0.156, -0.028]. Walking speed remains unmatched.

The separate five-class classifiers have mean macro F1 scores of 0.347 for Random Forest, 0.251 for learned JEPA features and 0.309 for random-encoder features. Macro F1 weights the five classes equally. These small supervised runs provide no demonstrated advantage for the learned encoder.


## What the timing test can establish

The timing comparison includes 27 normal recordings, with at least six in each test fold. However, intersecting visible observations leaves only about 1.75 editable lower-leg tokens on average. The both-leg predictor fails its adequacy check on aligned inputs in all 24 outer fold, training-method and offset combinations, as well as all 72 corresponding internal comparisons.

For random training, earlier opposite-leg observations increase error by 0.000393, with interval [0.000177, 0.000618]. Later observations change it by 0.000064, [-0.000232, 0.000368]. Although the first interval excludes zero, the predictor already performs inadequately before the edit on these sparse inputs. The result cannot support a reliable coordination interpretation (Figure \ref{fig:timing}). Sufficient recording counts do not guarantee enough useful observations for a particular prediction test.


# Measurement limits and implications

A fixed rule selects ten high-error and low-error examples across the five labels for video and pose review. They show camera motion, assistance, split screens, overlapping limbs and feet outside the image (Figure \ref{fig:cases}). Three-frame checks reveal visible errors but cannot establish continuous tracking or verify clinical labels.

\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{figures/failure_cases.pdf}
\caption{Estimated skeletons from three times in two recordings: a normal-labeled example with high prediction error and a Parkinsons-labeled example with low error. The axes use image coordinates divided by torso length. These examples illustrate why raw prediction error should not be interpreted as neurological severity.}
\label{fig:cases}
\end{figure}

An image-boundary check finds high-confidence lower-leg points outside the decoded image in 13 of 28 normal and 83 of 276 abnormal recordings. Pooled point fractions are about 0.43% and 0.24%; one requested frame fails decoding. Pose confidence alone does not establish a visible measurement. No models or exclusions change after this check. Too few normal recordings share a single view for a complete four-fold view-matched comparison, so recording conditions remain a possible influence on the scores.

Five normal GAVD references give uncertain cross-leg means and variances. Recalculating scores while omitting each reference or changing the variance floor exposes that sensitivity. The laboratory scores also have different tails: test-healthy 99th percentiles range from 6.30 to 33.94 for $R$ across folds, compared with 2.45 to 3.99 for $M$. Left/right reference-benefit correlations range from 0.54 to 0.82, whereas $R$ treats the sides separately. These observations suggest calibration questions, but do not establish why the fixed combination performs worse or demonstrate a remedy.

The laboratory experiment strengthens evidence for completion across measurement systems. Its scope remains healthy and stroke walking, with ten selected healthy trials excluded and one trial per participant. GAVD supplies more varied recordings, but its normal sample is small and informed development. Neither study establishes clinical deployment, and more windows cannot replace independent participants. Limited training budgets and the adapted pose comparator also leave the best attainable model performance unresolved.

# Conclusion

The other leg improves prediction under specific missing-observation tasks in GAVD and independent laboratory measurements. The controlled comparison makes this result interpretable by preserving the local baseline, using the same hidden answers and measuring the value of an additional own-leg correction.

The anomaly experiments show why prediction quality and detection value need separate evaluation. Posture and movement size remain informative even when walking is predictable, while a timing test can become unreliable after too many inputs are removed. The procedure helps researchers examine these conditions directly and provides a basis for testing better-calibrated cross-leg scores on new participants.
