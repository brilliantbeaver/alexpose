# A standalone contribution to gait anomaly detection

The strongest defensible paper presents a method for measuring whether cross-leg context supplies useful evidence for gait anomaly detection. Its contribution is an executed measurement and evaluation procedure, supported by positive completion results and independently fitted participant data. It should explain what the procedure measures, when that measurement is supported, and how it relates to anomaly ranking. It can do this without discussing earlier drafts or narrating the development process.

The paper should not promise a detection improvement that its tables do not show. Instead, it can make a substantial methodological case: an anomaly detector needs to distinguish unusual movement, poor prediction under missing observations, and an unusual relationship between joints. Those possibilities are confounded by an unqualified reconstruction error. The reported procedure makes several of those distinctions testable. It does not resolve every confound or establish a clinical detector.

This advice uses the saved [completion audit](completion_audit_v1/README.md), [study results](../../iteration02/results.md), [limitations](../../iteration02/limitations.md) and [score-calibration audit](../../draft02/reviews/calibration_audit/README.md). No models, scores, evaluation populations or manuscript files were changed for this review.

## Recommended positioning

Use a direct title such as **Measuring the Value of Cross-Leg Context for Gait Anomaly Detection**. A slightly more methodological alternative is **Auditing Cross-Leg Evidence for Gait Anomaly Detection**. Both titles make the application explicit without implying a new best-performing detector or a recovered neurological mechanism.

The main research question can be:

> When does the opposite leg provide useful prediction evidence beyond a competent own-leg model, and what checks are needed before using that evidence to rank abnormal gait?

This is more specific than asking whether gait joints are related. It distinguishes the added context from an extra fitted correction, fixes the target, and requires the prediction comparison to work under the actual permitted observations. It also connects the completion experiment to a concrete anomaly-detection decision.

A suitable problem statement is:

> A reconstruction model may predict a joint accurately from nearby joints without learning a relationship that distinguishes abnormal gait. Conversely, sparse observations can increase prediction error even when the movement is ordinary. We evaluate cross-leg prediction evidence with a retained local model, a matched own-leg correction and identical hidden targets. The resulting measurements separate completion benefit, observation support and anomaly-ranking value.

“Separate” here means separately evaluate these quantities. It does not mean their causes have been statistically or biologically identified.

## Three concrete contributions

### 1. Measure added-context prediction against a retained local explanation

The method predicts the same hidden coordinate with three branches: a frozen own-leg baseline, an additional own-leg correction, and a both-leg correction. The two correction branches share nominal input width and the retained baseline. Masks and validity restrictions apply before feature construction. The principal measurement is the own-correction error minus the both-leg error, with repeated physical targets aggregated before source-level evaluation.

This is useful to anomaly detection because it asks whether the relational score has a competent local comparator. It avoids giving the other leg credit merely for adding a correction to a weaker unaugmented predictor. It also preserves local posture and movement evidence as a separate observable channel.

The positive evidence is concrete. On GAVD's fixed distal-leg diagnostic, the uniform-trained bilateral predictor lowers normal-source mean MSE by 19.5% relative to the own-leg correction. Independently fitted laboratory models lower healthy-participant mean MSE by 5.2%, with a conditional 95% interval of [3.55, 6.70]%, and lower error for 109 of 128 participants. These are improvements under the recorded predictor families. They are not estimates of intrinsic conditional information, and matched width does not remove the numerical-rank or inductive-bias differences.

Suggested contribution sentence:

> We introduce an observable-target evaluation procedure that measures the value of opposite-leg context after retaining a local predictor and comparing an additional own-leg correction on identical targets.

The novelty lies in the particular controlled measurement and its evaluation in this setting. Residual prediction, own-information controls and normal-reference distances are established ideas. Avoid an unsupported claim that this is their first use anywhere.

### 2. Evaluate observation structure separately from training-mask policy

The U, J and A policies each hide exactly 64 of 128 tokens and give every token a marginal hiding probability of one half. The two evaluation banks remain fixed across policies. This makes the experiment interpretable: a change in training can be compared within one observation task, while the focused distal-leg task asks a separate question about jointly unavailable nearby joints.

The positive masking result is a 7.7% reduction in bilateral completion MSE for anatomical rather than uniform training on the same E_leg targets, with a conditional interval of [6.36, 9.00]%. Twenty-seven of 28 sources improve. The uniform-task comparison is 0.8%, with an interval spanning zero. Reporting both banks gives the result its methodological value. It shows task-specific benefit rather than a universal claim that structured masking is better.

Anomaly models often face different missing observations at evaluation. The contribution is an explicit way to test how a reported relation depends on those observations. The sparse timing diagnostic strengthens this point: the exact paired-support inputs fail the normal-prediction adequacy checks even though the denser task passes. A model response to an input edit is therefore insufficient evidence of a meaningful movement relationship under that edited support.

Suggested contribution sentence:

> We separate training-mask policy from evaluation access using equal-budget masks and fixed target banks, and test prediction adequacy again when a perturbation changes the permitted observation support.

Do not call the mask families novel. Do not claim that the adequacy gate has been shown to reduce false alarms in deployment. It is an operational interpretability check here. No learned gate or general reliability model was evaluated.

### 3. Validate prediction evidence and anomaly ranking as distinct outcomes

The study combines uncontrolled web-video pose with an independent participant-level laboratory measurement. It reports normal completion, local movement distance, relational-reference distance and their fixed combination. All four folds remain visible, masks and sources are matched within comparisons, and the score uncertainty respects the recording or participant unit.

The laboratory result matters because the completion benefit also appears in separately fitted synchronized bilateral angle signals. It is not merely a favorable plot from the web-video model. The anomaly results then show what the particular score construction contributes. For example, laboratory relational distance has AUROC 0.694, while local movement distance has 0.958 and their fixed combination has 0.892. These are descriptive scores from different measurements, not evidence that every relational model is less useful. GAVD's focal comparison is also necessary: local movement has AUROC 0.694 and the fixed combination has 0.576.

These values can be reported directly as the present study's findings. There is no need to describe them in relation to a previous manuscript or to characterize the paper as a recovery from an earlier result. Their practical implication is that a demonstrable completion benefit does not determine the weight or calibration of an anomaly score.

Suggested contribution sentence:

> We evaluate the procedure on web-video gait and independently fitted laboratory participant data, showing where cross-leg context improves completion and quantifying its distinct contribution to anomaly ranking alongside preserved local movement evidence.

This does not establish sensor transfer, a new standard benchmark, better clinical sensitivity, or improved anomaly discrimination. The laboratory sample covers healthy participants and stroke, not all four GAVD condition families.

## Why this is relevant to anomaly detection

The significance should be explained through the decisions the procedure supports. An investigator can determine whether an apparent relational signal survives an own-leg correction, whether it depends on a particular missing-observation task, and whether the corresponding normal-reference distance adds useful ranking information. The saved target, support and source-level outputs make these checks inspectable.

For example, consider a hidden left ankle. If visible left-knee and foot measurements already predict it well, additional right-leg context may yield little measurable benefit. If several adjacent left-leg observations are hidden together, the same right-leg observations may help more. Neither result alone makes a person's gait healthy or unhealthy. A stable compensatory movement can be predictable, while an ordinary movement can be hard to complete from sparse measurements. This is why the local movement descriptors and the relational comparison answer different questions.

The neurological motivation is therefore a reason to preserve multiple observations, not a biological explanation of a score. Stroke, Parkinsonian, cerebral-palsy and myopathic labels describe heterogeneous impairments and adaptations. The present data do not identify lesions, muscle activation modules, strength, medication response or physiological phase control. The paper can be clinically motivated without claiming a clinical mechanism.

The contribution also has clear limits relative to prior work. GenGait already uses normal-only masked gait reconstruction and joint inconsistency. Beyond Marginals separates marginal, temporal and inter-variable evidence. SLiM already uses anatomical semantic tubes to address reconstruction shortcuts. The defensible distinction is the implemented comparison of added context, local prediction, observation access and anomaly conversion, with source-level and independent participant evidence. [GenGait](https://arxiv.org/abs/2604.01997v2), [Beyond Marginals](https://arxiv.org/abs/2509.15033), [SLiM](https://arxiv.org/abs/2603.10648v4).

## Procedural disclosures that must remain

Standalone presentation does not require a development-history narrative. It does require enough methodological provenance to prevent a reader from mistaking exploratory evidence for independent confirmation.

| Disclosure | Concise standalone wording and placement |
|---|---|
| The normal GAVD recordings informed method development. | Methods: “The GAVD evaluation is exploratory because its 28 normal recordings also informed method development. Within the reported experiment, source roles are separated for fitting, reference estimation and testing.” |
| The positive paired completion summaries were chosen after outcomes were inspected. | Methods or the relevant results caption: “The paired completion intervals are descriptive analyses of frozen predictions computed after inspection of the outcomes.” The abstract should at least call the study exploratory and avoid confirmatory language. |
| Endpoint and bank hierarchy matter. | Define the focal binary endpoint, required broad secondary, uniform reference task and separate distal-leg diagnostic. Report both banks. Do not promote the favorable diagnostic to a newly prespecified primary endpoint. |
| Cohort, masking and score choices were fixed before the reported fitting. | State which quantities were frozen before fitting and which summaries were subsequently derived. A clear procedural sentence replaces references to earlier drafts. |
| Equal width is an incomplete control. | Retain numerical feature ranks, distinct feature bases, and the fact that widths do not equalize effective capacity or regularization geometry. |
| The body reference is shared. | Own-leg restriction is conditional on pelvis and torso measurements used to define the body reference. It is not independence from every opposite-side measurement. |
| Training and final target aggregation differ slightly. | Keep the documented request-weighting versus repeated-target aggregation distinction. It affects what model selection optimizes. |
| The independent sampling unit is limited. | GAVD units are known recording groups with unresolved repeated people; laboratory units are participants. Windows, joints and mask draws are not independent patients. |
| Conditional intervals are narrower than full research uncertainty. | State the original-fold paired bootstrap, fixed weights, absent refitting/seed uncertainty, and unadjusted exploratory contrasts. The four fold fits are not independent replications. |
| Calibration and fusion are actually limited. | M and R are dimension-normalized distances, not comparable probabilities. R omits bilateral covariance and uses small normal references. Report the actual score comparisons and do not claim recalibration was tested. |
| Observation quality defines the cohort. | Retain attrition, confidence filtering, natural missingness, artificial withholding, pose-estimation error and the ten excluded laboratory controls. |
| Comparator scope matters. | Label the STG-NF implementation as a bounded adaptation and the JEPA runs as small probes. The core positive result uses observable coordinate or angle prediction; it is not evidence of a new S-JEPA representation advance. |

A compact integrated disclosure could read:

> This is an exploratory study. The GAVD normal recordings informed method development, while fitting, reference estimation and evaluation use separated source roles within the reported experiment. Cohorts, masks and scoring rules were fixed before fitting. The completion ratios and paired intervals are descriptive summaries of frozen predictions examined after outcomes. Intervals condition on the recorded fits and do not include retraining or protocol-selection uncertainty.

This gives the reader material provenance without discussing manuscript versions, rejected ideas or the conversation that produced the study.

## Recommended results narrative

Begin with the controlled prediction comparison and its measurement units. Present the 19.5% GAVD distal-task benefit and 5.2% laboratory benefit with named comparators. Next present A versus U within both fixed banks, so that the 7.7% result is immediately bounded by the uncertain uniform comparison. Then report M, R and S as the anomaly-ranking evaluation of the same measurement procedure. Finally report the sparse-support adequacy result and the calibration audit as limits on interpretation.

The calibration audit supports a specific observation: empirical normal-score distributions and left/right reference dependencies differ. It does not establish that calibration caused the fusion outcome or that a proposed recalibration would repair it. Likewise, all failed timing gates support withholding a timing-mechanism interpretation, not concluding that human gait lacks temporal coordination.

The conclusion can state:

> Cross-leg prediction supplies measurable completion benefit under declared observation tasks, but its use in anomaly detection requires separate validation of support, reference estimation and score combination. Our procedure makes these checks explicit and reproducible across web-video gait and independent laboratory participants. The results support task-specific completion and careful evaluation of relational anomaly evidence while preserving informative local movement measurements.

That is a positive methodological conclusion supported by the completed evidence. It is more credible than claiming a universal detector improvement, and more useful than describing the work as a sequence of unsuccessful attempts. Its importance depends on showing the full controlled procedure and its evidence, rather than declaring significance or novelty in the abstract.
