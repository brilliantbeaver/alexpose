# Research review and rationale for the restart

Reviewed on November 16, 2026. This document proposes a new experiment identity. It does not replace the historical protocol or describe the new experiments as completed. The input evidence includes the current README, `docs/final_scientific_review.md`, `docs/results_tables.md`, the pose preparation code, and the primary sources in `primary_sources.json`.

## What the completed study taught us

The previous study established a working video-to-pose pipeline, a source-aware evaluation, and a useful distinction between classification, forecasting, and uncertainty. Its results do not justify a larger collection of transformer variants as the next scientific step. The primary mixed JEPA probe had binary balanced accuracy 0.818 versus 0.788 for Random Forest on 17 sources, but the paired difference interval included zero. An untrained encoder reached the same balanced accuracy as the primary trained representation. The residual forecast improved persistence by only 0.000512 MSE, with a source interval that included zero. Shuffling the prefix had little effect, while making the prefix static improved the residual model. These values are historical local results, not new literature claims or new restart results.

The strongest lesson is that a model can classify gait or produce plausible forecasts without demonstrating useful ordered dynamics. In the same study, the relatively poor direct forecaster produced the strongest secondary excess-error abnormality ranking. That result motivates separating motion prediction from abnormality detection. It does not justify choosing the direct model as a new primary based on the exposed test result.

The new work should ask a narrower and harder question: **Does a normal-motion predictor use correctly timed information from the opposite leg to predict future articulation beyond the target leg's own history?** The contribution is a controlled audit over fixed-time observations and more of each available sequence. Bilateral prediction itself is not claimed as new, including in relation to the repository's ASTAD work.

## Neurological reasoning and its measurement limits

Human locomotion contains coordinated patterns across limb segments. Borghese, Bianchi, and Lacquaniti observed structured covariation of segment elevation angles during laboratory walking. This supports evaluating joint relationships, but their three-dimensional, cycle-normalized measurements are different from projected video coordinates. A low-dimensional video prediction model is not a direct measurement of a neural controller. [Primary paper](https://doi.org/10.1113/jphysiol.1996.sp021539)

Plotnik, Giladi, and Hausdorff quantified bilateral timing using contralateral heel strikes and found worse phase coordination in Parkinson's disease. This provides a direct rationale for a timing-sensitive bilateral comparison. Our video context-control effect is neither their phase coordination index nor an estimate of basal-ganglia function. Heel-strike labels and clinical severity are not available here. [Primary study](https://pubmed.ncbi.nlm.nih.gov/17503027/)

Clark and colleagues linked reduced EMG module complexity to impaired locomotor performance after stroke. Steele, Rozumalski, and Schwartz found reduced EMG synergy complexity in cerebral palsy. These studies motivate checking whether relational motion contains useful information. They also caution against treating low variability or easy prediction as evidence of healthy movement. The direction of a video forecast error cannot be derived directly from either study. [Stroke study](https://doi.org/10.1152/jn.00825.2009), [cerebral palsy study](https://doi.org/10.1111/dmcn.12826)

The selected abnormal labels describe heterogeneous conditions. Myopathic gait should not be presented as exclusively a central neurological disorder. No label supplies a measured lesion, muscle activation, treatment, or verified severity. A shared predictive comparison is therefore more defensible than a disease-specific mechanistic explanation. Predictability can change with movement amplitude, speed, camera angle, compensation, or pose noise even when neurological status is unchanged.

## Position relative to current AI work

Skeleton S-JEPA predicts representations of masked skeletal observations and evaluates action recognition. Its published training uses motion-aware masking, EMA targets, centering, and cross-entropy between latent distributions. The current local implementation changes several of those choices. Neither the paper nor the local classifier establishes future patient-state dynamics. [ECCV 2024 paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04755.pdf)

Simple motion baselines are scientifically necessary. Martinez, Black, and Romero demonstrated that a strong motion-prediction score can be achieved without learning motion. The new comparison should therefore include persistence and deliberately constrained alternatives before interpreting a learned model as evidence of dynamics. [CVPR 2017 paper](https://arxiv.org/abs/1705.02445)

Future skeleton generation for anomaly detection already exists in MoCoDAD. Recent gait work also includes normative reconstruction in GenGait and biomechanics-informed clinical assessment in BioGait-VLM. These approaches rule out broad novelty claims for normal-only skeletal modeling, generative abnormality scores, or biomechanics-inspired gait representations. Their data, targets, and validation regimes differ from this study, so their reported accuracies are not matched numerical baselines. [MoCoDAD](https://arxiv.org/abs/2307.07205), [GenGait](https://arxiv.org/abs/2604.01997), [BioGait-VLM](https://arxiv.org/abs/2603.08564)

## Accepted minimal experiment design

The coordinator selected the following feasible restart. The executable protocol and run manifests remain authoritative if implementation details are amended before outcome inspection.

| Component | Decision and reason |
| --- | --- |
| Observation | Decode available selected-label GAVD sequences on a nominal 10 Hz grid. Retain 24-frame windows with 16 past observations and eight future observations, with stride eight. Report achieved timing and missingness. |
| Ingestion | Reuse Ambient pose inference, source video storage, and explicit annotation-to-decoded frame conversion. Keep image units, confidence, source ID, sequence ID, and person-gate outcomes. |
| Population | Use all eligible sequences from the five requested labels, with documented attrition and existing identity exclusions. Repeated stroke in the brief remains one class. |
| Split | Use five source-grouped outer folds. Fit predictors on normal training sources only. Select regularization on inner normal-source validation. All windows from a source remain together. |
| Primary comparison | Compare normal held-out forecasting error with persistence, then compare own-leg and bilateral predictors on identical future support. |
| Timing control | Replace the opposite-leg history with a declared temporal shift while preserving its observations, mask, dimension, and target support. Distinguish an independently fitted control from inference-time perturbation. |
| Other baselines | Include a static-posture model and damped velocity. A cyclic reference is deferred unless enough observed cycles support an identified period. |
| Estimation | Use low-capacity ridge residual prediction to expose whether the input contains usable predictive information before adding model complexity. Fit imputers and scales inside training folds. |
| Aggregation | Average coordinates within a window, windows within sequence, then sequences within source. Report source counts and per-source paired differences. |
| Secondary analysis | Evaluate abnormality rankings with the score direction fixed before result inspection. Report all declared scores and condition-specific support. |

Own-leg input is conditional on a shared pelvis and torso reference if normalization uses both hips and shoulders. It must not be described as prediction from strictly unilateral raw observations. The bilateral comparison then asks about additional opposite distal-leg articulation beyond that common reference. Future coordinates, confidence, or missingness must not affect prediction inputs, preprocessing statistics, anchors, or model selection.

A bilateral model has more input dimensions than an own-leg model. Improved fit can therefore reflect capacity or correlated noise. A matched shifted-context model helps challenge a timing interpretation. An inference-only shift is useful, but can also cause out-of-distribution inputs. Its effect alone does not establish physiological coupling. Both forms should be named accurately, and any unperformed control remains an explicit limitation.

## Falsifiable hypotheses and interpretation

1. **Forecast skill:** the learned model improves source-level normal forecasting over persistence. Failure means the data and model did not demonstrate useful forecast skill at this horizon.
2. **Additional bilateral information:** correctly timed opposite-leg context improves on own-leg history and on the matched timing control. Failure means the experiment does not support the specific bilateral-dynamics interpretation.
3. **Abnormality association, secondary:** a declared predictive discrepancy differs between annotated normal and abnormal sources. This cannot substitute for either primary hypothesis, and its sign must not be reversed after seeing AUROC.

Positive evidence requires both forecast benefit and timing specificity. A normal-source gain without a timing effect supports a practical predictor but weakens an ordered coordination explanation. A timing effect without forecast benefit shows model sensitivity without establishing useful prediction. A favorable abnormality score without either primary result supports a descriptive association only.

The old holdout results have already been inspected. Whole-cohort cross-fitting is a fresh implementation and broader internal evaluation, but it is **exploratory**. It does not create an untouched test set or undo historical design selection. Fold-overlapping training sets also mean a naive bootstrap of fixed out-of-fold source scores captures conditional source variation, not all retraining uncertainty. Report this limitation next to intervals.

## Does GAVD meet generalization requirements?

GAVD is appropriate for an exploratory benchmark across heterogeneous public videos. Its paper reports 1,874 sequences and more than 400 visually screened subjects. The local pinned annotation derivative has 348 source video IDs; these quantities are not interchangeable. The five selected labels contain 666 annotated sequences and 103 source IDs before media and quality exclusions. Exact restart counts must come from the new data audit. [Dataset paper](https://arxiv.org/abs/2407.04190), [local inventory](../../../../data/gavd/README.md)

GAVD alone does not meet patient-disjoint, site-disjoint, prospective clinical-generalization requirements. Source IDs are useful grouping proxies, but unknown repeated participants and related uploads can cross source boundaries. Public web videos have selection, view, recording, annotation, and availability biases. More overlapping windows improve observation coverage but do not add independent patients. Selected conditions also do not represent every form of abnormal gait.

The highest-value next extension is external evaluation with verified participants, clinical assessments, repeated visits or measured interventions, and compatible pose definitions. A larger transformer or more windows cannot replace those missing forms of evidence. If such data are unavailable, a transparent benchmark or short methodological paper is more supportable than a clinical world-model claim.

## Workshop fit and submission requirements

The official Med-WM page lists November 20, 2026, Anywhere on Earth, as the extended deadline. Full papers are 6–8 pages; short, poster, and position papers are 2–4 pages, excluding references and appendices. The PDF must use the AAAI two-column format and remain below 50 MB. Review is single-blind and the workshop is non-archival. The OpenReview page returned a loading shell during this audit, so its live invitation deadline was not independently verified. [Official call](https://med-wm.github.io/aaai27-mwm/), [submission portal](https://openreview.net/group?id=AAAI.org/2027/Workshop/MEDWM)

Recommended framing: **a controlled test of the motor-observation layer needed by medical world models**. The model forecasts observed articulation over fractions of a second. It does not learn treatment actions, disease progression, counterfactual trajectories, or clinical decisions. A strong paper can show why those distinctions matter, including when the proposed dynamics hypothesis fails. Acceptance probability cannot be quantified from the available evidence.

## Adversarial review priorities

The final reviewer should attempt to reproduce data attrition and source grouping, alter future values to test target isolation, verify timing and common support, and recalculate all manuscript effects from source-level outputs. They should check whether gains are explained by static posture, amplitude, view, confidence, or dimensionality. They should also inspect whether the paper's contribution duplicates the sibling ASTAD study, whether exposed historical outcomes influenced the new narrative, and whether claims about patients, anatomy, or causality exceed the observations. Every material change after that review reopens the corresponding check.
