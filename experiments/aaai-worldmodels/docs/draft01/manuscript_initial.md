# Testing Bilateral Predictive Information in a Video-Based Gait Observation Model

**Initial narrative, November 16, 2026.** This version was written before the restart results were available. It contains a proposed interpretation framework and no new empirical result. Bracketed citation keys map to `references.bib`. The final manuscript must replace the pending-results passage using verified run outputs.

## Abstract

Medical world models require reliable representations of changing patient state. Video-based gait provides an accessible view of movement, but a model that classifies gait or predicts a nearly static pose has not necessarily learned useful motion dynamics. We investigate whether correctly timed opposite-leg observations add predictive information beyond a leg's own history. Our protocol uses public GAVD videos, a shared pose-estimation pipeline, fixed-time observation windows, normal-only fitting, and source-grouped evaluation. Low-capacity predictors are compared with persistence, static posture, damped velocity, and temporally altered bilateral context. We distinguish forecast skill from timing sensitivity and from secondary abnormality ranking. The empirical results will determine which of these claims are supported. This study concerns short-horizon observed articulation and does not model treatment responses, disease progression, or clinical decisions.

## 1. Introduction

Patient state is partly expressed through movement. A gait observation model could summarize how a person's articulation evolves and provide an input to a future longitudinal medical model. Before treating such a component as evidence of meaningful dynamics, however, it should outperform simple motion references and show that correctly ordered observations matter. Medical world modeling cannot be established solely by attaching a disease classifier to a pose representation.

There is a neurological basis for studying coordinated movement. Laboratory gait has structured intersegmental relationships [borghese1996]. Bilateral phase coordination is affected in Parkinson's disease [plotnik2007]. Studies of stroke and cerebral palsy relate altered EMG modular organization to impaired movement [clark2010; steele2015]. These findings motivate a predictive coordination question, but do not make video coordinates a measurement of neural activity. Camera view, speed, compensation, and pose-estimation error can all affect observed predictability.

We ask whether the opposite leg contributes useful, timing-specific information to forecasts of a target leg. The investigation emphasizes an evidence hierarchy: improvement over persistence, additional benefit from bilateral context, sensitivity to the timing of that context, and only then association with annotated abnormality. We do not claim that bilateral gait analysis is new. Our intended contribution is a reproducible audit that combines broader sequence coverage, a fixed-time observation contract, and explicit controls for interpretation.

## 2. Related work

Skeleton S-JEPA learns representations by predicting latent targets for masked skeletal observations and evaluates action recognition [abdelfattah2024]. Its pretext task is distinct from strict future prediction. MoCoDAD generates future skeletons for anomaly detection [flaborea2023], while GenGait studies normative gait reconstruction [motta2026]. BioGait-VLM combines visual, language, and biomechanical information for clinical gait assessment [chen2026]. These studies establish relevant precedents and preclude a broad novelty claim for normal-only modeling, future skeleton generation, or biomechanics-inspired assessment.

Simple baselines remain essential. Martinez and colleagues showed that strong motion-prediction performance can arise from a reference that does not model motion [martinez2017]. A bilateral forecast must therefore demonstrate more than a low coordinate error. Its improvement should survive a static reference and a matched comparison that alters the temporal relationship of the opposite leg.

## 3. Data and observation model

GAVD contains clinically informed annotations for publicly available gait videos [ranjan2025]. The selected scope includes normal, Parkinson's, stroke, cerebral palsy, and myopathic labels. A folder label is treated as a dataset annotation rather than an independently verified diagnosis. The duplicate stroke entry in the task brief is represented once.

Videos remain under `data/videos-gavd`, with one source copy and separate previews. We use Ambient's pose interface and retain source IDs, sequence IDs, decoded times, frame numbering, confidence, and missingness. Frame-index conversion occurs once at the annotation-to-decoder boundary. New fixed-time windows have 16 past and eight future observations at a nominal 10 Hz, giving a nominal final forecast horizon of 0.8 seconds. Windows advance by eight observations. The final data report must document achieved timing, media failures, person-gate exclusions, and pose-quality attrition.

Coordinates describe image-plane articulation in a body-relative reference. They are not physical velocity or anatomical three-dimensional joint angles. When the reference uses both hips and shoulders, the own-leg condition already includes that shared reference. The bilateral comparison measures the additional contribution of opposite-leg articulation, not the difference between unilateral and bilateral raw sensory access.

## 4. Predictors and controlled comparisons

All learned motion predictors are fitted using normal training sources. A ridge residual predictor forecasts the target leg relative to its observed anchor. The own-leg model receives the target leg's past articulation. The bilateral model adds the opposite leg's history. A declared temporal control changes the alignment of the opposite-leg observations while preserving their values and missingness together. The final methods section must specify whether this control is independently fitted, applied at inference, or both.

Persistence repeats the last available target articulation. Static posture and damped velocity supply additional low-capacity references. Training-fold statistics determine all imputation and scaling. Future coordinates, masks, or confidence do not enter prediction or preprocessing. Every paired comparison uses identical observed future targets and identical eligibility support.

## 5. Evaluation

Five outer folds keep all observations from each source video together. Regularization is selected using inner normal-source validation. We first average error across observed coordinates in a window, then across windows in a sequence, then across sequences in a source. Sources receive equal final weight. The primary questions concern normal held-out forecast skill and the incremental value of bilateral context. Secondary abnormality scores retain their declared directions even when their rankings are unfavorable.

The previous study's holdout results were already known when this restart was designed. The new cross-fitting study is therefore exploratory internal evaluation. It cannot be called a newly untouched test or prospective validation. Conditional intervals over fixed out-of-fold source predictions do not account for every source of uncertainty from overlapping training folds and model selection. Source groups are also not verified unique participants.

## 6. Results

**Pending verified execution.** Insert the generated cohort flow, source-level primary effects, timing-control effects, all declared baselines, and secondary scores here. No result should be inferred from training loss or selected because it supports the intended story. Preserve failed hypotheses and compare their practical magnitude with uncertainty and source support.

## 7. Discussion and limitations

A favorable bilateral result would support a narrow statement about additional predictive information in observed articulation. It would not identify a neural circuit, demonstrate an EMG synergy, or establish an intervention mechanism. A timing effect without forecast skill would indicate sensitivity without useful prediction. An abnormality-ranking result without either primary effect would remain an association with dataset labels.

GAVD supports heterogeneous-source experimentation, but it cannot by itself establish patient-level clinical generalization. Participants may recur across uploads, recording conditions differ, and pose filtering can select an easier subset. More windows increase temporal coverage without increasing the number of independent patients. Two-dimensional coordinates and approximate person gates retain measurement limitations. External data with verified participants, clinical assessments, and longitudinal or intervention information are required for stronger medical-world-model claims.

The practical value of this study is an explicit standard for the observation layer: useful prediction, timing-specific information, and clinical interpretation are separate requirements. The final conclusion must state which requirements the completed experiment met.
