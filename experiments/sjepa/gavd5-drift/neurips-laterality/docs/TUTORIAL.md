# From gait symmetry to useful movement prediction

A guide to our research questions, findings, and next experiments. Notebook evidence reviewed on 8 September 2026; related literature and workshop information were last checked on 7 September 2026.

The next implementation is available in notebooks [15](../15_motion_weighted_masking.ipynb),
[16](../16_structured_masking_and_context.ipynb), [17](../17_motion_and_structure_pretraining.ipynb)
and [18](../18_motion_information_and_readout.ipynb). They examine the MAMP code's
motion weighting, structured target geometry, paired JEPA training and
motion-sensitive frozen readouts. The [source review and experiment specification](MOTION_STRUCTURED_MASKING.md)
records the literature checked on 8 September 2026 and the precise adaptations.
All four now default to real GAVD, displaying five source folds and seeds
42–46. The [GAVD run guide](MOTION_GAVD_WORKFLOW.md) explains preparation,
training enablement and loading the same complete grid in Notebook 18.
Real-input and masking audits reproduce the 625-clip, 93-video cohort; the
full new pretraining comparison remains unrun. Generated software checks are
explicitly separate. The empirical findings below are unchanged. This
implementation follows Direction A before expanding the forecasting work.

## Contents

- [Research question and current finding](#1-the-question-connecting-the-work)
- [Data and research trajectory](#2-the-data-and-the-research-trajectory)
- [What notebooks 11–14 have established](#what-has-run-in-notebooks-1114)
- [Real-data masking findings](#question-2-does-hiding-gait-relevant-landmarks-help-learning)
- [Mask patterns and worked examples](#what-notebook-11-teaches-about-a-fair-masking-comparison)
- [Feature prediction and missing observations](#what-notebook-13-teaches-about-prediction-and-missing-observations)
- [Future-feature prediction](#question-4-does-learning-future-features-help-predict-future-movement)
- [Comparative masking techniques and priorities](#4-is-our-fixed-landmark-selection-too-restrictive)
- [Novelty and related literature](#5-what-recent-research-changes-about-the-broader-novelty-claim)
- [Next research directions](#6-the-most-productive-research-directions)
- [Workshop fit and readiness assessment](#7-how-close-is-this-to-a-strong-workshop-paper)
- [Practical sequence toward the next paper](#8-a-practical-sequence-toward-the-next-paper)
- [Implementation prompt for the next notebook suite](#9-copy-ready-implementation-prompt-for-the-next-notebook-suite)
- [Evidence and figure reproduction](#evidence-and-figure-reproduction)

## 1. The question connecting the work

**What should a JEPA learn about human movement, and how can we tell whether pretraining has helped?**

Our experiments study sequences of estimated body landmarks, including shoulders, knees, and ankles. An encoder turns these coordinates into numerical features that a prediction model can use. During Joint-Embedding Predictive Architecture (JEPA) pretraining, part of the input is hidden, and a predictor estimates its features from the visible observations. A second, slowly updated encoder supplies the target features. This training does not use gait-condition labels.

The motivating idea is that predicting missing observations could encourage a model to represent meaningful relationships between body parts and their movement. For example, the visible hip and knee trajectory might help predict an obscured ankle. We evaluate that idea by holding the encoder's weights fixed—often called freezing the encoder—and testing whether its features help a simple regression model predict a movement quantity on videos excluded from training.

The latest real-data comparison, completed through Notebook 12, contains 50 trained encoders evaluated on 625 clips from 93 source videos. Selecting gait-relevant landmarks as hidden targets does not establish better movement prediction than selecting targets across the body. Both trained alternatives perform worse than their matched untrained encoder, even with the regression penalty selected using training videos only. Direct summaries of the pose coordinates give the best scores among these tested representations.

The trained predictor also carries clip-related information: it estimates its own clip's hidden teacher features more accurately than teacher features from a different video, while the initial control shows no consistent preference for correct targets. That result does not translate into better prediction of our left–right movement score. A predictor can exploit posture, viewpoint, or other clip-specific information without improving the movement distinction we want to recover. This is the most useful finding to investigate next.

Three questions are especially interesting now:

1. **Where does useful movement information become difficult to recover?** The recorded coordinates, prepared input, learned features, and final feature summary can each affect the result.
2. **Which left–right relationships should the representation preserve?** Overall movement and the side contributing more movement need different responses to reflection.
3. **Can learning these relationships improve prediction of future movement?** This would provide a stronger connection to temporal world models than a test of whole-clip consistency alone.

This tutorial separates measured results on gait recordings from synthetic demonstrations and proposed experiments. The real-data evidence covers two scattered masking policies. Motion-weighted and structured masks have been implemented and demonstrated, but their real-data training comparisons remain unrun. Forecast decoding in Notebook 14 is also a synthetic demonstration; it has not established a benefit on human gait recordings.

Section 4 examines whether our masking choices are too restrictive. It distinguishes the landmarks supplied to the encoder from those selected as hidden targets, explains established alternatives, and proposes a manageable comparison grounded in the current results.

## 2. The data and the research trajectory

### A curated movement dataset and a controlled evaluation

The completed evaluations use 625 accepted clips from 93 source videos selected from the Gait Abnormality in Video Dataset (GAVD). GAVD is a researcher-curated collection of online gait videos, with annotations covering normal, pathological, and other abnormal walking patterns. Its YouTube sources were assembled for gait analysis, rather than sampled as arbitrary internet videos. This clinical motivation is important, although our current prediction target is calculated from estimated coordinates and is not a diagnosis or an independently measured clinical score. [GAVD dataset paper](https://arxiv.org/abs/2407.04190)

We divide source videos into five groups. For each group, every clip from its videos is excluded from both encoder training and fitting the final regression model. We repeat training with five random seeds, which change the starting weights as well as sampled clips, hidden targets, and training views. Thus, two training choices require 5 groups × 5 seeds × 2 choices = 50 trained encoders. Notebook 12 has completed this full design, with 1,200 updates per encoder. These are repeated fits on the same recordings, rather than 50 independent datasets. Notebook 08 retains a separate completed comparison using an earlier training and evaluation implementation; its scores must not be pooled with the new ones.

Consider a video contributing ten clips. Testing on two of those clips while training on the other eight could allow the model to benefit from the same recording conditions or repeated movement. Keeping the entire video together prevents that overlap. When scoring, each video receives the same total weight, so a video with many clips cannot dominate simply through its clip count. This is careful recording-level separation; without verified participant identities, it cannot establish separation of people who might appear in different videos.

### How one question led to the next

**We began with an anatomical choice.** Gait-relevant landmarks seemed sensible targets for learning from missing body observations. The initial design used that selection throughout, however, so it could not show whether the anatomical choice caused an improvement.

**We then used laterality to test a known relationship.** Here, laterality means a signed contrast between movement on the left and right anatomical sides. For illustration, a contrast of +0.2 should become −0.2 when we reflect the coordinates and exchange the left/right landmark labels. This follows from the definition of the measurement; it does not require the person's gait to be symmetric.

**The reflection evaluation separated consistency from predictive value.** Adding mirrored clips during training reduced the specified disagreement between original and reflected features. Its effect on movement prediction remained uncertain: the estimated R² improvement was 0.004, with a 95% interval from −0.006 to 0.013. Ordinary pretraining also failed to establish an improvement over the matched initial encoder: the R² difference was −0.018, with an interval from −0.039 to 0.002. These intervals resample whole source videos while keeping the fitted models fixed. [Completed results notebook](../05_aggregate_statistics.ipynb)

A sign-reversal rule imposed on the final output can satisfy the expected relationship even with an untrained encoder. That control showed why satisfying a mathematical rule cannot, by itself, establish useful learning. The feature test also assumes that reflection exchanges anatomical tokens without changing their internal feature coordinates; other possible responses within those coordinates remain untested.

**We next investigated what might explain the gap.** Notebook 07 follows movement information through the pipeline. Notebook 08 supplied the first complete real-data comparison of gait-focused and all-landmark targets. Notebook 09 introduced explicit reflection training, and Notebook 10 asked whether future-feature training produces a useful representation of the observed past.

**Notebooks 11–14 make the comparisons more informative.** Notebook 11 separates which landmarks may be hidden from the shape of the missing observations. Notebook 12 trains paired encoders and now supplies a new complete real-data comparison. Notebook 13 separates feature-prediction accuracy from movement-prediction accuracy, examines both encoders, and selects readout penalties using training sources. Notebook 14 adds the missing step of decoding the JEPA predictor's future features into coordinates. The new real-data result retains the earlier warning about weak predictive benefit, while the added controls narrow what we should investigate.

For navigation, notebooks 00–05 contain the original protocol and evaluation; [06](../06_external_subject_gate.ipynb) checks prerequisites for external evaluation; [07](../07_research_questions_and_diagnostics.ipynb) provides diagnostics; [08](../08_matched_budget_masking.ipynb) compares masking choices; [09](../09_symmetry_aware_jepa.ipynb) tests reflection training; and [10](../10_past_only_movement_prediction.ipynb) introduces past-only forecasting. External participant-level validation has not yet been completed.

## 3. What the updated notebooks help us answer

### What has run in notebooks 11–14?

The four notebooks have different evidence levels. A completed mask-construction example verifies how observations are hidden; a completed real-data training grid can test whether that choice helps prediction on excluded videos.

| Notebook | Retained execution and what it establishes |
|:--|:--|
| [11 — Masking patterns and coverage](../11_masking_patterns_and_coverage.ipynb) | Eight policies run on generated coordinates. Their masks, coverage counts, and infeasible-budget cases can be inspected. This does not rank their usefulness on gait recordings. |
| [12 — Controlled masking pretraining](../12_controlled_masking_pretraining.ipynb) | Small synthetic examples and a complete real-data comparison of gait-focused versus all-landmark scattered targets: 25 paired jobs, 50 encoders, and 60,000 optimizer updates. |
| [13 — Encoder and predictor evaluation](../13_masking_encoder_and_predictor_evaluation.ipynb) | Its inline examples use generated data and three training updates. The same evaluation helpers also produced the real-data prediction and diagnostic tables saved by Notebook 12; those tables support the empirical findings below. |
| [14 — Future features and movement prediction](../14_future_features_and_movement_prediction.ipynb) | The retained teaching copy uses eight generated clips and four updates per arm. Observed future features decode accurately, while predicted future features decode poorly. No real-data forecasting performance has been retained. |

The latest real comparison contains 125,000 prediction records because each clip is evaluated repeatedly across seeds, masks, feature choices, and observation conditions. The underlying cohort remains 625 clips from 93 source videos. Neither those repeated records nor the repeated seeds increase the number of independent recordings.

### Question 1: Does the input still express the movement we want to predict?

Notebook 07 compares the original movement measurement with the same formula applied after input preparation. Preparation fills eligible short gaps, normalizes coordinates, and resizes sequences to a common length. These operations are useful for batching, but they can also change a quantity based on movement between observations.

In a read-only reconstruction, the two calculations have matching signs in about 70% of cases, with equal total weight per source video. Both calculations are finite for 623 clips from 92 sources; two clips have no finite recomputation. This measures agreement between two ways of calculating the target. It is neither a model's accuracy nor a ceiling on what a model could learn.

A simple example explains a separate concern about summarizing time. Let a left landmark follow positions (0, 1, 0, 1), and a right landmark follow (0, 0, 1, 1), at equally spaced times. Their average positions are both 0.5, but their median movement speeds are 1 and 0. Swapping the trajectories reverses the speed contrast while leaving the averages unchanged. A model given only those average positions cannot distinguish the cases. Averaged encoder features could still contain motion if the encoder has represented it before averaging.

The next useful comparison is to apply the measurement after each preparation step on the same clips, then compare time-averaged features with summaries that retain ordered movement. If a change restores measurement agreement and improves held-out prediction, it would identify a practical weakness. If agreement improves without better prediction, we would need to investigate representation learning or the regression model as well. The existing diagnostic motivates these tests but does not identify which operation causes the discrepancy.

### Question 2: Does hiding gait-relevant landmarks help learning?

The completed Notebook 12 comparison selects hidden targets either from twelve landmarks—the left and right shoulders, hips, knees, ankles, heels, and foot tips—or from all 33 landmarks. Both policies scatter targets across joint–time positions. A target token represents one landmark over four prepared time steps. The experiment changes the eligible target landmarks while keeping mask shape and the actual hidden count matched.

The important control is the number of valid targets actually hidden. Suppose there are four time blocks and every landmark is visible. Twelve landmarks provide 48 candidate tokens, while 33 provide 132. Hiding half would produce 24 targets in one condition and 66 in the other. The notebook instead hides the same number in both conditions, adjusting the shared count when observations are missing. This avoids attributing a change in training difficulty merely to hiding more input.

The paired models share starting weights, source-video draws, training views, and 1,200 optimizer updates. Both receive all 33 landmarks, use the same twelve-landmark pooling in the feature-variation regularizer, and use the same five left–right landmark pairs for their movement-feature summary. Consequently, the comparison tests **where masked prediction targets are chosen within an already anatomy-informed procedure**. Broader target eligibility does not remove every anatomical choice from training or evaluation.

The table below was independently recomputed from the saved test predictions; no training was launched for this review. The pretrained rows use the frozen teacher, the slowly updated encoder that supplies JEPA's training targets. “Direct pose summaries” means average landmark positions, their variation over time, average movement between prepared samples, and the fraction of observations available for each landmark. This control uses the same prepared input without an encoder.

| Features supplied to the regression model | R² | Mean absolute error |
|:--|--:|--:|
| Teacher encoder pretrained with gait targets | −0.023 | 0.0471 |
| Teacher encoder pretrained with targets across the body | −0.010 | 0.0461 |
| Matched encoder without pretraining | 0.048 | 0.0451 |
| Direct pose summaries | 0.130 | 0.0432 |
| Mean target from training videos | −0.011 | 0.0462 |

Every row covers all 625 clips and 93 videos. For each seed, predictions are combined across the five held-out groups before scoring, with equal total weight per video. The table then averages the five seed-specific scores. Direct pose and training-mean predictions do not depend on the training seed and are not five independent baseline runs. Imputation, scaling, and regression are fitted using training sources only; the regression penalty is selected through three inner source-separated groups.

R² compares squared prediction errors with variation in the observed target. A value of zero matches the source-weighted mean of the evaluated targets; a negative value has larger squared error. That evaluation mean is a mathematical reference, not a deployable predictor with access to test labels. The last table row is the usable control fitted from training videos only. Mean absolute error measures the average size of the error in units of the normalized movement contrast; smaller values are better, and these values are not percentages.

![The two pretrained teacher encoders have lower movement-prediction R-squared than the initial encoder and direct pose summaries. A separate paired interval for broader masking includes zero.](figures/tutorial_comparative_masking_results.svg)

*Figure 1. Movement prediction in the completed Notebook 12 comparison. The pretrained rows use teacher features. Circles show seed-specific scores and diamonds their means; the two seed-independent controls have one diamond each. The lower panel compares all-landmark with gait-focused targets using 2,000 paired source-video resamples. Its interval keeps the fitted models fixed and excludes uncertainty from retraining.*

Broader targets have a mean teacher-feature R² advantage of 0.012 over gait targets, with a saved 95% paired source interval from −0.018 to 0.050. The interval includes a benefit in either direction. We therefore cannot claim that all-landmark masking is better, that gait-focused masking is better, or that the two are equivalent. The result weakens the claim that this particular gait target set provides a demonstrated advantage for our endpoint. Its anatomical motivation remains a reasonable hypothesis, but the experiment does not validate the set as neurologically optimal.

The learned-over-initial comparison is more consistent. Both pretrained teachers have lower R² than the matched initial encoder in all five seeds. Looking at the actively optimized, or online, encoder does not reveal a hidden benefit: its average R² is −0.027 with gait targets and −0.020 with all-landmark targets, also below the initial encoder in every seed. This result concerns the features and regression procedure we tested; it does not establish that the encoder contains no movement information.

For example, imagine a feature that distinguishes a close-up side view from a distant frontal view. That distinction may help predict missing body features, yet contribute little to estimating which leg moves more. The predictor diagnostics below show why this kind of alternative explanation matters. They do not identify viewpoint as the cause in our data.

One unresolved issue is visible in the regression selection itself. Encoder summaries have 960 dimensions, compared with 264 for direct pose summaries. The tested penalties are 0.01, 0.1, 1, 10, and 100; larger values constrain the fitted weights more strongly. Training-source validation selects the largest value for every initial and online-encoder fit, and for 49 of the 50 pretrained-teacher fits. The search has reached its upper boundary. A wider, prespecified penalty range is a sensible next diagnostic, using training sources only. The current result already shows that allowing some penalty selection does not automatically establish useful pretraining, but it does not show that regularization has been adequately explored.

For historical context, Notebook 08 reported teacher-feature R² values of −0.540 with gait targets and −0.423 with uniform targets, using its earlier implementation and fixed penalty of 1. Those saved results remain intact. The new scores are less unfavorable, and direct pose summaries now have positive R², but training and evaluation implementation details changed together. We cannot attribute that numerical shift solely to penalty selection or treat it as a masking improvement. The controlled inference is the comparison between arms within the new grid.

### What Notebook 11 teaches about a fair masking comparison

The most important lesson from Notebook 11 is that the location and arrangement of missing information define different learning tasks, even when the number of hidden observations is the same. These are verified construction examples; their predictive value still requires training and evaluation.

**Count what is available before choosing what to hide.** The executed example has six time blocks and 33 landmarks, giving 198 possible tokens. One token is naturally unavailable, leaving 197 valid tokens. A scattered mask deliberately hides 18 of those valid tokens and leaves 179 as context. The naturally missing token contributes neither a training target nor visible evidence. Counting it as an additional supervised target would ask the teacher to supply a measurement that was never available.

**Separate landmark preference from mask shape.** Gait-focused and all-landmark scattered masks both hide isolated joint–time positions, but draw those positions from different pools. A fixed random twelve-landmark set tests whether the chosen anatomical set matters beyond restricting the pool to twelve points. The set stays fixed across runs while the masked time positions are redrawn. A soft gait preference keeps every valid landmark eligible but gives gait landmarks more probability. In the implemented starting setting, the sampling weights mix uniform all-landmark and uniform gait-landmark probabilities equally; this does not force half the sampled targets into each group.

For example, a valid wrist token has a chance of being selected under the soft preference, but no chance under the gait-only rule. A random twelve-point set may include wrists and facial points while excluding one ankle. Several sets must be chosen before looking at prediction scores, because selecting the best set afterward would turn a control into a search for a favorable result. None of these additional landmark-selection comparisons has a retained real-data training result yet.

**Check whether motion weighting changes selection as intended.** In a generated clip, the left ankle moves five times as fast as the right ankle, while the left elbow is stationary. Across 200 draws of one hidden target each, the left ankle is selected 128 times, the right ankle 30 times, and the left elbow once; other landmarks account for the remaining 41 draws. These counts combine each landmark's six time blocks. The sampler mixes 75% movement-based probability with 25% uniform probability, so every valid token remains eligible and a fivefold speed difference need not produce a fivefold selection difference. This verifies the intended sampling behavior. No model is trained in this example, and the counts provide no evidence of better movement prediction.

**Keep a whole trajectory whole.** In the six-block example, hiding three complete landmark trajectories produces 18 targets, the same count as an 18-token scattered mask. Their information content differs: scattered masking may leave an ankle's earlier and later positions visible, while a whole-ankle mask removes that landmark throughout the window. A connected three-landmark region over three blocks produces nine targets. The executed two-block interior gap hides 65 valid tokens because one of the 66 possible positions was naturally missing. The notebook reports these unequal counts openly; they are illustrations of different tasks, not a count-matched performance comparison.

At the real input length, 64 prepared time steps become sixteen blocks. A complete trajectory then contributes sixteen valid tokens, whereas a complete time block contributes 33 when all landmarks are visible. Two trajectories hide 32 tokens and one time block hides 33. No positive count can match both intact structures while leaving context in this fully observed example: the first shared multiple is all 528 tokens. The practical solution is a separate scattered reference for each structured mask, matched to its actual count. Trimming a trajectory to fifteen blocks or adding a lone cell would change the task being tested.

**Interpret the advertised masking percentage using its denominator.** With fully valid input, half of the twelve-landmark candidate pool is 96 tokens, about 18% of all 528 input tokens. In the completed real training schedules, actual hidden counts range from 38 to 95 per sampled clip, with a median of 81. The corresponding fraction of valid input hidden is about 7%–25%, with a median of 16%. These counts reflect the shared feasibility rule and varying observation coverage. The models therefore retain substantial context; the experiment does not test hiding half of every body sequence.

The practical question is whether a harder or differently arranged missing-information task improves the chosen movement outcome. Hiding a complete ankle trajectory might encourage use of the knee, hip, and opposite leg, but it might also remove evidence needed to recover genuine asymmetry. Both outcomes are plausible, which is why a defined mask and a successful software check are only the beginning of the experiment.

### What Notebook 13 teaches about prediction and missing observations

**Training loss and movement prediction answer different questions.** Every one of the 50 real-data encoders has lower average feature-prediction training loss in its last fifty updates than in its first fifty. Averaged across the 25 fitted models per policy, those window means decrease from 3.64 to 0.87 for gait targets and from 3.74 to 0.84 for all-landmark targets. The teacher changes during training, so these numbers describe optimization against a changing target. They cannot rank the usefulness of two different teachers' feature spaces.

A more informative check holds the evaluation masks fixed and asks whether the predicted features are closer to the correct clip's teacher features than to teacher features from another source video at the same valid target positions. All 150 trained diagnostic rows—two policies, 25 fitted models, and three evaluation masks—have lower mean error for the correct targets than for the mismatched targets. The matched initial controls do not consistently favor correct targets. These rows repeat models and recordings; they are not 150 independent replications. The outcome shows clip-related predictive information under this check, although posture or camera viewpoint could contribute alongside movement.

For illustration, predicting the features of a bent knee better than those of a different person's straight knee could satisfy this diagnostic without improving an estimate of left–right movement over the clip. The observable laterality score provides a common outcome for comparing encoders whose internal coordinates differ. On that shared outcome, the trained features remain weaker than the initial features. The saved diagnostics do not flag the clip-level target or prediction summaries as nearly constant, so “the features collapsed to one value” would also be too strong an explanation.

Notebook 13's own inline execution uses a small generated dataset and three updates. It first includes a known, recoverable movement signal to check that the evaluation can detect information when it is present. That is a software check, not a real-data result. Its small learned-versus-initial differences cannot establish the ranking of gait masking policies. The larger empirical statements in this section come from the evaluation tables saved by Notebook 12.

**The retained missing-observation tests are mild sensitivity checks.** Every model receives the same altered prepared coordinates. The conditions remove a short connected region on the left leg, the corresponding region on the right leg, or scattered positions, while retaining the laterality target from the unaltered clip. For a clip whose original score is +0.10, the question is whether the remaining observations can still predict +0.10; the experiment does not redefine the reference score after hiding the ankle or reconstruct its coordinates.

These tests usually remove four valid tokens, with a median removal fraction below 1% of valid input. Across the three altered-input conditions, gait-teacher R² ranges from −0.027 to −0.021, all-landmark teacher R² from −0.013 to −0.009, and direct-pose R² from 0.123 to 0.131. The broad ordering remains, but this does not establish reliable prediction under substantial occlusion. A weak predictor whose output changes little can look insensitive to a small gap.

Coverage also matters. The left-leg and scattered conditions evaluate 622 clips; the right-leg condition evaluates 621. All retain 93 sources. Three or four clips are recorded as unavailable because the requested gap contains no fully valid token to remove. They remain in the result records rather than silently disappearing. A claim about the change caused by corruption should compare each altered-input score with the unaltered score on those same available clips; comparing 622 cases with all 625 can mix an observation effect with a change in the evaluated set.

Finally, these masks are applied after interpolation and normalization. A removed coordinate may already have influenced that preparation. The results therefore concern sensitivity of the prepared representation to these small gaps. Notebook 13 also demonstrates removing raw observations before preparation, but a full raw-recording evaluation has not run. For a claim about genuinely absent observations, the withheld ankle measurements must be removed before any preparation that could use them.

### Question 3: Can explicit reflection training preserve information about each side?

Notebook 09 compares the base objective, mirrored training clips, and an explicit penalty for disagreement between corresponding original and reflected landmark features. The penalty encourages agreement after anatomical landmark exchange, with missing observations handled consistently.

Imagine left and right movement values of 3 and 2. Their average remains 2.5 after exchanging sides, while their difference changes from +1 to −1. A useful representation may need to support both quantities. In plain language, some information should stay the same under reflection, while information about which side contributes more should change predictably.

Perfect feature agreement is insufficient evidence of success: giving every clip the same feature vector would also make the comparison agree. The notebook therefore measures predictive performance and checks whether features vary across clips. These checks can expose a trivial solution, although feature variation alone does not establish useful movement content.

The retained run uses generated data and eight training updates. The explicit penalty slightly reduces its targeted reflection error without a displayed predictive advantage; there is no real-data result establishing that it helps gait learning. Its short run is a software and teaching demonstration, not a test of an adequately trained method. The extra reflection calculations also cost computation, so matching update counts alone would not support an efficiency claim.

A useful real-data outcome would be better movement prediction together with better geometric agreement. If only agreement improves, the contribution remains a consistency result. If prediction deteriorates, the next test should examine where the penalty is applied and how strongly, rather than assume that more symmetry training is beneficial.

### Question 4: Does learning future features help predict future movement?

Notebook 10 observes the first 0.8 seconds of a sequence and predicts landmark positions 0.25, 0.50, or 0.75 seconds beyond that boundary. For example, the middle horizon asks about an ankle around 1.3 seconds, using observations available through 0.8 seconds. Unlike the masked-learning notebooks, this forecasting implementation supplies only the twelve selected landmarks to its past and future encoders. Whether a broader input helps is a separate, untested comparison.

This changes the information available to the model. A whole-clip movement score can use the complete sequence; forecasting must prepare the input from the observed past alone. The notebook preserves timestamps and calculates its reference position and body scale using only that prefix. A test changes future coordinates and visibility while checking that the prepared input and prediction stay unchanged.

During pretraining, a predictor estimates features of a future window. During evaluation, a separate regression model maps the frozen encoder's past features to future coordinates. Notebook 10 therefore tests whether future-feature training improves the **representation of the observed past**. It does not decode the JEPA predictor's future feature vector or evaluate a sequence of predicted future states.

The comparisons include keeping each landmark at its last position, continuing its last estimated velocity, regression directly from past coordinates, the initial encoder, and training against a future taken from a different training video. That mismatched-future condition is important: if it performs similarly, correct temporal pairing has not demonstrated added value.

The retained Notebook 10 training results use synthetic oscillating trajectories and do not show a consistent advantage over direct past-coordinate regression or the initial encoder. Correctly paired and mismatched futures also perform similarly in that demonstration.

**Notebook 14 now tests the predictor's future features directly.** After training the forecasting model on training videos, it freezes the model and fits a coordinate decoder using observed future-teacher features from those training videos. At evaluation, the same frozen decoder receives either the observed future-teacher features or the features forecast from the past. The first condition checks whether the features and decoder can express the endpoint when the future is available; it is a diagnostic with access to the answer period. Only the second condition is a past-only forecast.

For example, suppose the endpoint is an ankle's position 0.50 seconds after the observed prefix. If the decoder is inaccurate even when supplied features of the observed future, we have not established an adequate route from features to ankle coordinates. If that diagnostic is accurate but decoding predicted future features fails, the complete forecasting route still needs improvement. Decoder behavior on imperfect predicted features is one possible explanation; the comparison alone cannot identify the cause.

The retained [executed Notebook 14](../14_future_features_and_movement_prediction.ipynb) uses eight generated clips from eight synthetic sources: six for training and two for testing, one seed, and four updates per training arm. It reports the following coordinate errors for correctly paired future training:

| Horizon after the observed prefix | Observed future features, decoded | Predicted future features, decoded |
|:--|--:|--:|
| 0.25 seconds | 0.011 | 2.51 |
| 0.50 seconds | 0.013 | 2.61 |
| 0.75 seconds | 0.025 | 2.64 |

These are source-balanced coordinate root mean squared errors in normalized pose units, where smaller values indicate more accurate coordinates. The normalization uses a pelvis reference and body width calculated from the observed prefix; these errors are not millimeters or calibrated three-dimensional distances. Every method at a given horizon uses the same 24 landmark endpoints from the two test clips. The large gap shows that the decoder can express these synthetic endpoints from observed future features much more accurately than from the predicted features produced by this very short training run. It is an observed failure of that software demonstration, rather than evidence against adequately trained JEPA forecasting.

Regression from the trained past encoder gives errors of 0.064, 0.068, and 0.072 across the same horizons, close to the initial encoder's 0.063, 0.069, and 0.074. Regression directly from past coordinates gives 0.077, 0.057, and 0.059. Training against a future from another training source also produces similar displayed results in this short run. There is no useful learned advantage to claim here. A low effective rank in the past-feature diagnostic is likewise unsurprising with only two distinct test prefixes; it cannot establish feature collapse.

The distinction between an interior gap and a future target remains essential. A missing middle interval can use observations on both sides, much like filling a missing ankle position between two visible steps. A future mask must use only the observed prefix. Notebook 14 checks that changing future values does not change the prepared past input, and keeps twelve- versus 33-landmark input as a separate comparison with identical future landmark endpoints. Predicting several horizons from one fixed prefix also does not constitute a recursive rollout.

Preparing the real recordings is feasible: a read-only check found 611 clips eligible for at least one horizon, from 93 videos, contributing 1,814 clip–horizon examples. Duration and landmark visibility affect which outcomes can be scored. These are repeated questions about existing clips, not new independent samples. The real-data decoder comparison and twelve- versus all-landmark input training remain awaiting results.

## 4. Is our fixed landmark selection too restrictive?

It is a reasonable starting prior, but too narrow a basis for concluding which masking strategy helps movement learning. The twelve-landmark choice has anatomical motivation; our experiments have not established that this exact set is neurologically optimal. “Gait-focused” or “anatomically motivated” describes the evidence more accurately than “neurologically validated.”

The concern also needs a precise account of what is fixed. A fixed number of hidden observations can make a comparison fair. Permanently restricting which landmarks can become targets limits the questions that comparison can answer.

### What is—and is not—restricted in the current notebooks

In notebooks 03, 08, and 09, the encoder receives all 33 landmarks. The original recipe selects hidden prediction targets from twelve of them, but draws a fresh set of hidden landmark–time tokens at each training update. The same anatomical list is reused; the actual missing observations change. Valid elbows, wrists, and other landmarks outside that list remain available as context, and the target encoder processes the unmasked full-body input.

There are additional anatomical choices elsewhere in the pipeline. The feature-variation regularizer summarizes the twelve gait landmarks from unmasked training views, including in Notebook 08's all-landmark masking condition. The laterality readout uses five bilateral pairs: shoulders, knees, ankles, heels, and foot tips. Hips belong to the masking set but not this readout. Notebook 10 makes a stronger restriction by supplying only twelve landmarks as input. Changing any of these choices answers a different question from changing the mask alone.

We have therefore already tested one important alternative: selecting the same number of hidden targets across all 33 landmarks. Both Notebook 08's retained comparison and Notebook 12's new grid fail to establish an advantage for gait-focused selection under their respective procedures. Several random twelve-landmark target sets would be another useful control; Notebook 11 demonstrates their construction, but no such real-data training comparison has been retained.

A numerical example reveals a second restriction. With 64 prepared time steps and four steps per token, all 33 landmarks provide 16 × 33 = 528 tokens. The twelve-landmark target set provides 16 × 12 = 192 candidates. Hiding half of those candidates hides 96 tokens, or about **18% of the full input**, assuming all observations are valid. Thus, the current “50%” setting does not hide half of the body sequence. Missing observations further limit the actual shared count.

This leaves many neighboring observations available. A short ankle gap may be predictable from the ankle immediately before and after it, without requiring a rich account of whole-body movement. That is a plausible explanation to test, rather than an established cause of our weak prediction results. In a JEPA, the targets are contextualized features, so ease of coordinate interpolation alone cannot establish how the learned predictor solves its task.

### Mask shape changes the information available for prediction

The following figure holds the hidden count constant while changing its arrangement. It makes visible why selecting a landmark set, choosing a masking percentage, and choosing a pattern over time are distinct decisions.

![Four illustrative six-landmark, six-time-block grids with twelve hidden cells each: scattered tokens, whole-joint trajectories, a connected-limb interval, and a middle temporal gap.](figures/tutorial_masking_patterns.svg)

*Figure 2. Four ways to hide 12 of 36 landmark–time cells. Blue cells are prediction targets; pale cells remain visible. This is a hypothetical six-landmark example, not the experimental landmark set or masking rate. Panel D illustrates completion, with observations before and after the gap.*

### Established and promising alternatives

The literature contains established baselines as well as newer proposals. Their evidence comes mainly from action recognition, pose reconstruction, or RGB video, so their relevance is a reason for comparison rather than a prediction that they will improve our gait endpoint.

**1. Scattered random masking with broader or softer landmark coverage.** Uniformly sampling joint–time tokens across the body is an important baseline, already present in Notebook 08. Random patch masking is central to MAE, whose image experiments favored it over the grid and block alternatives tested there. That result does not determine the best skeleton mask, but it shows why a simple random baseline deserves serious treatment. [MAE, CVPR 2022](https://arxiv.org/abs/2111.06377)

Two additional controls would clarify the anatomical claim. First, compare the selected gait set with several random twelve-landmark sets chosen before looking at outcomes. Second, give gait landmarks a sampling preference while keeping every valid landmark eligible. For example, an equal mixture of all-landmark and gait-only sampling would soften the exclusion rule. That mixture is a proposed starting comparison, not an established optimum. If it outperforms the fixed set, the useful ingredient might be anatomical emphasis combined with broader coverage.

The landmark scheme itself matters: several of the 33 points describe the face or hands, so uniform sampling of points is not uniform sampling of body regions. A body-region-balanced draw would be another reasonable control, provided its grouping is defined in advance. A fixed random subset and a new subset drawn at every update should also be distinguished: the first tests a particular restricted pool, while the second can eventually cover the whole body.

**2. Motion-weighted masking.** Instead of always favoring the same landmarks, assign higher masking probability to joint–time regions with more observed movement. A swinging arm may receive more targets in one clip, and a moving foot in another. Selection remains stochastic, so this need not always hide the single fastest-moving joint.

This is the most direct unrun real-data comparison with the source method. The original S-JEPA uses motion-weighted masking inherited from MAMP; our fixed gait-target pool is an adaptation. MAMP's mask-only ablation reports NTU-60 cross-subject linear-evaluation accuracy of 84.9% with motion-weighted masking versus 83.7% with random masking. MAMP also changes the prediction target to motion, so its overall performance cannot be attributed to mask selection alone. Notebook 11 now demonstrates an adapted motion-weighted sampler while retaining our JEPA feature target; its real-data predictive value still needs evaluation. [S-JEPA, ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04755.pdf), [MAMP, ICCV 2023, Table 8](https://openaccess.thecvf.com/content/ICCV2023/papers/Mao_Masked_Motion_Predictors_are_Strong_3D_Action_Representation_Learners_ICCV_2023_paper.pdf)

For laterality, movement magnitude is an imperfect guide. A limb that moves less can be essential to the left–right contrast, while an implausibly large jump can come from a tracking error. A practical adaptation should preserve a nonzero chance of masking slow-moving regions, exclude invalid observations, and control the influence of isolated jumps. These are proposed safeguards to evaluate, not evidence that motion-weighted masking already works here.

**3. Whole-joint trajectories and connected body regions.** A trajectory mask hides the same landmark throughout the input window. For example, withholding the left ankle for the whole clip removes the immediate before-and-after observations that could fill a short gap. VideoMAE's analogous “tube” mask repeats a spatial mask through the video. At the same 90% masking ratio, its reported Something-Something V2 accuracy is 69.6% for tube masking and 68.3% for independent random masking. Those are fine-tuned RGB action-recognition scores; the very high ratio should not be copied directly to sparse pose input. [VideoMAE, NeurIPS 2022](https://arxiv.org/abs/2203.12602)

A connected-region mask hides several anatomically linked landmarks for a continuous interval. For example, hide the left knee, ankle, and foot together while leaving the hip, trunk, and opposite leg available. This asks whether broader body relationships help when local cues are missing. Groups should follow anatomical connections rather than consecutive indices in a data array.

There is close skeleton-specific precedent. The SLiM preprint, revised in August 2026, masks connected anatomical regions over consecutive time blocks and varies temporal duration with region size. It combines teacher-feature prediction with global/local alignment and reports structured-mask comparisons. Its design makes connected-region masking a well-motivated comparator, while ruling out a broad novelty claim based on introducing anatomical tubes alone. [SLiM, 2026 preprint](https://arxiv.org/html/2603.10648v3)

I-JEPA also motivates predicting several connected targets from surrounding context, although its blocks are image regions rather than body parts. Its preference for multiblock masks, alongside MAE's preference for random masks in a different objective, argues for testing mask design within the actual learning task. Hiding an entire asymmetric limb can also make its movement genuinely ambiguous; greater difficulty does not guarantee more useful features. [I-JEPA, CVPR 2023](https://arxiv.org/abs/2301.08243)

**4. Missing time intervals and strictly future targets.** A temporal gap hides every landmark during part of a sequence. With observations before and after the gap, the model performs movement completion. MotionBERT uses joint-level and frame-level corruption, while PoseBERT studies missing poses and temporal blocks. Both provide precedent for varying which parts of time are observed, although their reconstruction objectives differ from our feature-prediction objective. [MotionBERT, ICCV 2023](https://arxiv.org/html/2210.06551v3), [PoseBERT](https://arxiv.org/html/2208.10211v2)

Hiding the entire future asks a different question. Observing through 0.8 seconds and predicting an ankle at 1.3 seconds requires continuation without later clues. Notebook 10 implements that information boundary, but has no retained real-data training result. The August 2026 Human-JEPA preprint offers adjacent evidence from human RGB video by comparing forecasting-oriented masks with other masking choices. Its results motivate testing this choice without establishing a benefit for our skeleton task. [Human-JEPA](https://arxiv.org/html/2608.21160v1)

Joint masks, middle gaps, and future masks can eventually be combined during training. First compare them separately so that a gain can be attributed to a particular source of supervision. Preparing a whole clip and hiding its future afterward can still expose future information through normalization or interpolation; the forecasting pipeline must continue to prepare its input from the observed past alone.

**5. Complementary masks and several levels of difficulty.** Instead of presenting two similarly incomplete views, deliberately hide different kinds of information in each. One view might retain the trunk while hiding moving limbs; another might retain those limbs and hide some trunk observations. ASMa studies complementary choices based on joint connectivity and frame motion, within a method that also changes the encoders and feature-alignment objective. It supports testing complementary coverage, rather than attributing the full method's gains to one mask. [ASMa, TMLR 2026](https://openreview.net/pdf?id=kIFo1q3VMS)

Short gaps, long gaps, small regions, and larger regions can also be sampled with different frequencies. This gives the encoder varied prediction tasks, but broad difficulty variation should follow an interpretable fixed-pattern comparison. A schedule that makes masking harder over training would be another hypothesis; we have no result showing it is preferable here.

For this signed endpoint, caution is needed about making incomplete views identical. Removing the more active leg changes the information available about laterality. A loss that encourages agreement regardless of which limb is missing could suppress the very distinction we want to retain. Evaluate side-sensitive prediction alongside any agreement between views.

**6. Learned or adaptive selection.** A small auxiliary model can learn which observations to reveal or hide. AdaMAE learns a sampling distribution over the tokens that remain **visible**, using a reconstruction-error-based training signal. This differs from a rule that hides high-motion regions. A pose adaptation could learn which observations make the remaining movement predictable, but would add parameters, computation, and another source of training instability. [AdaMAE, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Bandara_AdaMAE_Adaptive_Masking_for_Efficient_Spatiotemporal_Learning_With_Masked_Autoencoders_CVPR_2023_paper.html)

I would postpone this until simpler masks provide an informative baseline. A learned selector could prioritize pose-estimation errors because they are difficult to predict, or communicate information through the locations it chooses. Using complete examples to construct self-supervised training masks is not automatically label leakage, but a future-informed mask must never influence an input advertised as past-only. An adaptive method should also beat a simple motion-based rule after accounting for its extra cost.

### Changing the target or the loss is a separate experiment

Some promising methods broaden the learning signal without changing mask placement. V-JEPA 2.1 supervises visible as well as hidden tokens and intermediate encoder layers. AMR assigns greater reconstruction-loss weight to movement-rich regions and also changes the prediction architecture. Neither should be described as merely an alternative sampler. Keep the mask fixed when testing these changes, so a result can be attributed to what the model predicts or how its errors are weighted. [V-JEPA 2.1, 2026 preprint](https://arxiv.org/html/2603.14482v3), [AMR, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Sun_Exploring_Adaptive_Masked_Reconstruction_for_Self-Supervised_Skeleton-Based_Action_Recognition_CVPR_2026_paper.html)

The same distinction applies to broadening the regularizer's twelve-landmark summary or Notebook 10's input. Those choices deserve investigation, but changing them together with masking would prevent a clear explanation of the result.

### Which comparisons should we prioritize?

The most useful next experiment is a focused comparison of what information is hidden, rather than a large search over every available technique. The following order balances scientific value with the weakness of the current predictive results.

1. **Make the evaluation informative before expanding training.** Use training-source validation to check regression regularization and motion-sensitive feature summaries, retaining the untrained encoder and direct-pose controls. A masking comparison needs a way to reveal useful learned information if it is present.
2. **Test the anatomical exclusion rule.** Keep scattered masks and compare the gait-only and all-landmark references with several preselected random twelve-landmark sets and one soft gait preference. Judge random sets as a declared group rather than reporting whichever set gives the most favorable contrast. If a new condition changes the feasible shared token budget, repeat the reference conditions at that budget.
3. **Test motion and mask shape under all-landmark eligibility.** Prioritize motion-weighted selection because it is the missing canonical S-JEPA comparator, and connected-limb intervals because they remove nearby body cues. A whole-joint trajectory condition would distinguish missing one trajectory from missing a connected region. Ensure the body-region groups together cover the declared landmark pool; otherwise the structured mask quietly introduces another exclusion rule. Keep the feature-prediction objective and all other training choices fixed for these comparisons.
4. **Evaluate temporal prediction separately.** Complete the real-data past-only comparison, including direct past-pose regression, initial features, simple motion continuation, and mismatched futures. Compare twelve- and all-landmark inputs only as an explicitly separate question. Future prediction should be scored at the same times on the same supported observations.
5. **Add complex combinations only when simpler results justify them.** Complementary views, visible-token supervision, and learned selectors become more informative after identifying which mask types preserve useful movement. A favorable training loss alone is insufficient grounds to expand the method.

A modest training-only pilot can establish feasible settings before confirming a small declared comparison across the five video groups and five seeds. It should not select the winning recipe from the outer test results.

### How to keep these comparisons fair and interpretable

Match the number of genuinely observed tokens deliberately hidden, while recording the fraction of valid input this represents. Naturally missing landmarks must remain distinct from observations deliberately withheld for prediction. A whole-joint or whole-frame mask may only support certain counts; choose feasible budgets or report a controlled range instead of silently breaking the mask's structure to reach an arbitrary total. Overlapping blocks should not inadvertently count the same target multiple times.

For example, on the fully observed 16-time-block, 33-landmark grid, a complete joint trajectory contributes 16 hidden tokens and a complete time block contributes 33. A 96-token budget accommodates six trajectories, but not an integer number of complete time blocks. These two mask families have no common positive count short of hiding the entire input. One solution is to compare each structured mask with a scattered-mask reference matched to its actual count, and report the different budgets explicitly. Their effects relative to those references would not establish a direct ranking of the two structures at an identical budget.

Equal counts control the amount of missing input, not its difficulty. Record the length of missing intervals and which body regions remain visible. Give each mask family the same training-only tuning opportunity, with several feasible coverage levels rather than assuming that either the current percentage or a video paper's high percentage is suitable. Keep source separation, paired random draws where applicable, architecture, teacher construction, readout procedure, and training exposure consistent. The present encoder retains masked positions in its token sequence, so hiding more tokens does not automatically reduce its computation as in a visible-token-only encoder.

Evaluate intact recordings and prespecified missing-region tests. For example, withhold a short ankle interval from every method and predict the original whole-clip laterality score from the remaining observations. That uses our existing kind of readout; reconstructing the ankle coordinates would require a separate decoder fitted on training videos. Report the direction of the left–right contrast as well as the magnitude of prediction errors, so systematically shrinking asymmetric scores toward zero does not look like an unqualified success. Choose both sides for masking without a permanent left/right preference; reflected-pair comparisons must transform the masks along with the landmarks.

For a test of genuinely missing observations, withhold them before preparing the evaluation input, and calculate normalization and interpolation using only permitted observations or an independently specified reference. The untouched recording supplies the reference score, not hidden information for input preparation. Masking an already prepared sequence remains a useful representation-sensitivity test, but supports a narrower claim. This distinction concerns the evaluation boundary; the full-input teacher remains a legitimate source of self-supervised training targets.

If structured masks improve prediction only when the matching region is missing, that would support a specific robustness benefit. If they also improve intact-video prediction over the initial encoder, the evidence for useful pretraining would be broader. If random subsets or soft preferences match the fixed gait set, the exact anatomical selection has no demonstrated special advantage. If all learned alternatives remain weak, the next explanation should concern the input, objective, or evaluation—not an assumed need for still more elaborate masks.

The most promising contribution is a controlled account of **which missing-information tasks preserve useful movement differences, and under what observation conditions**. The literature already supplies strong masking ideas; our opportunity is to determine how they behave when the outcome depends on fine left–right movement rather than broad action identity.

## 5. What recent research changes about the broader novelty claim

The literature supports the general motivation, while narrowing what would count as a distinct contribution. The following are primary research sources checked for this review; preprints are identified where proceedings publication was not verified.

**Skeleton masking and feature prediction already have substantial precedent.** Section 4 explains why anatomical or motion-aware masking alone is too broad a novelty claim. The ICCV 2025 paper *Towards Efficient General Feature Prediction in Masked Skeleton Modeling* also predicts local and global skeletal features. Our opportunity is a controlled explanation of when particular training targets preserve a useful movement quantity. [Skeletal feature prediction](https://openaccess.thecvf.com/content/ICCV2025/papers/Sun_Towards_Efficient_General_Feature_Prediction_in_Masked_Skeleton_Modeling_ICCV_2025_paper.pdf)

GaitJEPA already combines masked and future-latent prediction on walking silhouettes. Its authors report IJCB 2026 acceptance, with evaluations of person identification and sex classification. Those tasks and inputs differ from our signed coordinate-based movement target, but the work rules out presenting the general combination of JEPA and gait as new. [GaitJEPA author repository](https://github.com/AVAuco/GaitJEPA)

**Combining geometric consistency with useful features is an established research problem.** The NeurIPS 2025 seq-JEPA paper separates representations serving invariant and transformation-sensitive tasks. Soft Equivariance Regularization, published at ICLR 2026, reports that deeper geometric supervision can improve equivariance scores while reducing classification performance, and studies intermediate-layer supervision instead. Our earlier separation between consistency and prediction should be situated within that literature. A stronger new result would identify which geometric relationship preserves signed body dynamics under the observation conditions in our data. [seq-JEPA](https://proceedings.neurips.cc/paper_files/paper/2025/hash/2f63d2963526bdd9ff1b8bcc2dc9905a-Abstract-Conference.html), [Soft Equivariance Regularization](https://proceedings.iclr.cc/paper_files/paper/2026/hash/3be6511c8f56d0dca4b5ed59fdf9b2f4-Abstract-Conference.html)

**What receives supervision matters.** The 2026 V-JEPA 2.1 preprint studies supervision of visible as well as hidden tokens and multiple encoder layers. Its findings motivate a small comparison of training targets that preserve local movement, rather than assuming that predicting any hidden features will retain the information our task needs. Its results concern much larger image/video models and do not establish an expected gain for this dataset. [V-JEPA 2.1](https://arxiv.org/html/2603.14482v3)

The August 2026 Human-JEPA preprint makes the evaluation of forecasting particularly relevant. It adapts a video JEPA to human imagery and reports that the future predictor adds almost no benefit beyond its encoder in an NTU-120 early-action evaluation using the first half of each clip. Improvements to the encoder and improvements from using predicted future features must therefore be measured separately. This motivates extending Notebook 10 beyond a probe of past features. Its RGB-video experiments are adjacent evidence, not results on our skeleton task. [Human-JEPA](https://arxiv.org/html/2608.21160v1)

**A world-model claim needs an evaluation of prediction over time.** The 2026 LeWorldModel preprint evaluates action-conditioned latent prediction and control. It also reports limitations of its feature-distribution regularizer when observations have little diversity. Together with LeJEPA's proposed regularizer, this suggests a possible comparison, not a guaranteed remedy for weak gait prediction. Our recordings contain observational walking without controlled actions; successful forecasting would support a movement-dynamics result, while intervention or planning claims would need additional evidence. [LeWorldModel](https://arxiv.org/html/2603.19312v3), [LeJEPA preprint](https://arxiv.org/html/2511.08544v3)

**Clinical transfer requires an appropriate movement representation and independent outcomes.** The July 2026 GaitEncoder preprint evaluates gait features against clinical outcomes with participant-separated tests, using an evaluation encoder distinct from its final all-participant model. Its preparation flips left strides into a common right-stride convention with the stated aim of avoiding laterality encoding. Those processed inputs would need reconsideration for our signed target; stride-normalized kinematics also differ from our timestamped image-space landmarks. Its strongest lesson here concerns study design, not a ready-to-use external benchmark. [GaitEncoder preprint](https://www.medrxiv.org/content/10.64898/2026.07.07.26357479v1.full), [author repository](https://github.com/rdmagruder/GaitEncoder)

## 6. The most productive research directions

### Direction A: Explain and recover access to movement information

This is the first priority because the completed masking comparisons show weak learned-feature prediction even before asking which anatomical targets are best. Notebook 12 adds evidence that both online and teacher features underperform their initial control, while Notebook 13's real-data diagnostics show that predicting the correct clip's features has become easier. Together with Notebook 07's measurement discrepancy, these findings point toward a mismatch among what is retained, what is predicted during pretraining, and what the readout is asked to recover.

The first readout improvement has now been implemented: penalty selection uses inner training-source groups. Its repeated selection of the largest candidate calls for a wider range, declared before examining the next outer-test scores. Other separate comparisons should retain timing more faithfully during preparation and preserve movement order in the feature summary. Keep the compared clips and outer groups fixed, and apply each change to initial and trained encoders alike. An improved regression model must not be mistaken for improved pretraining.

For example, if training-only penalty selection improves trained and untrained features equally, the main improvement concerns the readout. If preserving time substantially improves a direct movement baseline but leaves learned features unchanged, the training representation still deserves investigation. If a motion-sensitive feature summary reveals a repeatable learned-over-initial advantage, that would support the more specific explanation that the old summary obscured useful learned content.

The new real-data evaluation already compares the actively optimized encoder and the slowly updated teacher at the final checkpoint; neither has a learned-over-initial advantage. What remains missing is a real-data predictive learning curve at prespecified earlier checkpoints. Such a comparison could help select a defensible training budget through inner source validation, provided candidate pretraining also excludes those validation sources when the whole recipe is being selected. Neither the final checkpoint nor decreasing training loss proves why prediction is weak.

After establishing an informative evaluation, use Section 4's staged masking comparison to separate the effect of anatomical selection from movement weighting and the shape of missing observations. A gait-specific advantage that persists against matched random sets and structured alternatives would be more informative than the present two-choice comparison. Add visible-token or intermediate-layer supervision separately if the diagnostics justify it.

For a concrete test of observation loss, hide a short ankle segment and ask whether the remaining hip, knee, and opposite-leg observations help predict the laterality score calculated from the unaltered recording. Follow Section 4's preparation safeguards and compare with randomly placed gaps of the same size, retaining the same clips and reference targets. Such a result could show when learned body relationships help beyond direct measurements of the available coordinates. The reference would still be pose-derived, so clinical accuracy would remain a separate question.

The potential paper would explain how a seemingly sensible pretraining comparison changes when movement information and the prediction procedure are controlled. Its novelty would depend on demonstrating a generalizable mechanism or remedy; adjusting one regression parameter alone would be a modest contribution.

### Direction B: Test whether reflection changes the features in a predictable way

Our current feature test assumes that each numerical feature should stay unchanged after exchanging anatomical landmarks. A learned encoder could instead express reflection through a predictable change in those feature values.

For illustration, suppose two features encode “overall movement” and “left-minus-right movement.” Reflection should preserve the first and negate the second. Comparing both coordinates as if neither should change would penalize a useful representation. This example does not show that our encoders have learned such features; it identifies an alternative explanation worth testing.

A low-compute experiment could freeze each encoder and learn one shared transformation of feature values from original/reflected training pairs, after aligning anatomical landmarks. Evaluate that transformation on held-out videos. Applying it twice should return to the starting point. Requiring it also to preserve lengths is an additional modeling restriction that prevents shrinking all features to obtain better agreement; this restriction is not guaranteed by reflecting the input. Keep the original unchanged-feature score, and compare real pairs with mismatched pairs and trained encoders with their initial references. Failure of this restricted transformation would leave more general reflection responses untested.

If a transformation fitted on training sources explains held-out reflection behavior, it would support a response within the features that the original test does not capture. This expands the assumed reflection rule; simply rotating the coordinate system cannot turn a nontrivial response into an unchanged one. To support a useful-learning claim, relate that structure to signed prediction or forecasting, with independent evaluation. Failure to generalize, similar performance on mismatched pairs, or an equally strong result before training would weaken the learning explanation. Lower alignment error alone would not establish improved prediction.

This direction is a good match for NeurReps because it turns a concrete experimental ambiguity into a question about learned representations. Related equivariance methods already exist; the contribution would be the explanation and its predictive consequence in noisy body observations.

### Direction C: Make geometric structure improve observable future prediction

This has the greatest potential for a paper about movement world models, with substantially more empirical work required. Notebook 14 now implements decoding of observed and predicted future features alongside the direct past-coordinate and past-feature references. Its synthetic result shows a functioning observed-future decoder and a poor predicted-future route after four updates. The next step is a training-source pilot that tests whether the latter improves with an adequate budget, before committing to the real-data grid. Accurate decoding from observed future features alone can reflect directly available future positions and does not establish learned dynamics.

The observed-future diagnostic should stay separate from deployable forecasting. If it fails, the decoder or representation cannot yet express the endpoint adequately. If only the predicted-feature route fails, the forecast features or the decoder's response to them remain candidates for investigation. The same frozen decoder, common endpoints, and matched/mismatched future controls make this more informative than reporting past-feature regression alone.

For example, after observing through 0.8 seconds, ask whether a predicted future feature vector can recover the ankle position at 1.3 seconds. Repeat on the reflected input and check whether the predicted movement transforms appropriately. A model that improves its reflection score without improving position prediction has not demonstrated the desired forecasting benefit.

Retain the direct past-pose, initial-encoder, and mismatched-future controls. To test the value of temporal order, disrupt the association between earlier positions and their timestamps while keeping the last two observations fixed, preserving the immediate position and velocity references. Merely reordering observations together with their timestamps would leave their temporal information intact. Use common evaluable landmarks and clips when comparing methods, and report changes in coverage across horizons. If correct futures or correct temporal associations do not help, apparent success may rely mainly on static pose or smoothness.

Once one-step latent prediction is useful, a further implementation could feed predicted states into subsequent predictions. The current predictor accepts pose windows, so this would require a compatible feature-to-feature transition or an explicitly tested decode-and-re-encode procedure. Asking about several horizons from the same observed prefix does not yet implement repeated prediction. Compare reflection penalties at intermediate versus final features as a separate experiment. Synthetic occlusion and pose perturbations can test robustness to observation errors, but they are not physical interventions on a person.

A convincing result would show where geometric structure improves future movement prediction and where it costs accuracy. An observational forecasting model could support a focused world-model contribution without claiming action planning, clinical decision support, or a complete physiological simulator.

### When expanding external evaluation is worthwhile

Notebook 06 is worthwhile if it leads to an actual independent evaluation. Additional readiness checks alone add little scientific evidence. Select the outcome and dataset first, verify coordinate and timing compatibility, and establish simple baselines before transferring the selected method.

An external set could test generalization to new people where participant separation from development data can be verified and those people are excluded from training and model selection. Any unresolved overlap must be reported. An independently measured movement quantity would test more than reproducing the same coordinate formula. A clinical rating would support a clinical question only if the data, reference, and study design justify that interpretation. These are different goals, and one external dataset need not serve all of them. The existing [external evaluation assessment](external_evaluation_assessment.md) discusses candidate routes and their unresolved requirements.

Because the current recordings have already shaped our hypotheses, further comparisons on them should be described as development evidence. An independently specified evaluation would provide a stronger test of the selected explanation.

## 7. How close is this to a strong workshop paper?

The new complete grid improves the evidence beyond the earlier masking comparison. We can now report training-only readout selection, separate online and teacher results, and a predictor diagnostic showing clip-related learning without improved laterality prediction. The central anatomical-benefit hypothesis remains unsupported, and its explanation is unresolved. A credible paper could center on this specific gap and its controls; a claim of improved JEPA movement learning requires further evidence.

The following scores are editorial judgments on a five-point scale, not estimated acceptance probabilities. Five means a strong position on the stated dimension; one means substantial missing evidence or poor fit.

- **Importance of the question: 4/5.** Distinguishing useful movement learning from satisfying a training or consistency objective matters for body representation learning.
- **Experimental control: 4/5.** The complete source-separated, paired grid and matched hidden-token counts are strong. Training-only regression selection and both encoder readouts are now present. The penalty search boundary, additional anatomy-specific controls, and stronger missing-observation tests remain important gaps.
- **Evidence for the current bounded conclusion: 4/5.** The full grid supports reporting that this procedure fails to establish the proposed masking benefit. It does not support a universal negative claim.
- **Explanation of the result: 2/5.** Correct-versus-mismatched feature targets help rule out the simple account that training made no clip-related predictive progress. They do not identify whether preparation, temporal summaries, regularization, or the training objective explains weak laterality prediction.
- **Novel contribution beyond related work: 2/5 today.** The ingredients and general consistency/usefulness distinction have precedent. A specific mechanism, convincing correction, or new predictive result would strengthen novelty.
- **External generalization: 1/5.** No independent participant-level or independent-outcome evaluation has been completed.

For the proposed directions, the scores separate evidence already available from potential after successful experiments. “Evidence now” rates the support for investigating the question, not evidence that the proposed remedy works. Feasibility concerns completing a focused comparison with the existing data and tools; workshop potential assumes a clear, well-supported outcome.

| Research direction | Novelty potential | Evidence now | Feasibility | Workshop potential |
|:--|--:|--:|--:|--:|
| A. Recover movement information | 3/5 | 3/5 | 5/5 | 4/5 |
| B. Explain the feature response to reflection | 3/5 | 2/5 | 4/5 | 4/5 |
| C. Improve future movement prediction | 4/5 | 1/5 | 2/5 | 5/5 |

Direction A has the best near-term return because it uses completed experiments to investigate a concrete failure. Its strongest paper would explain more than a dataset-specific tuning issue. Direction B is an economical, focused geometry question with a clear way to reject its explanation, although demonstrating useful consequences is essential. Direction C offers the strongest temporal-model contribution, but its high potential score is conditional: no real-data forecasting benefit has yet been demonstrated.

### Which workshop audience is most appropriate?

**Foundation Models for the Brain and Body: topic fit 5/5.** Its call includes movement, video-derived pose, and evaluation of pretraining. Direction A addresses whether a body encoder learns useful information, while Direction C could add temporal evidence. A broad foundation-model capability would still require transfer across tasks or datasets. The posted paper deadline was 5 September AoE and has passed. Its separate interactive-demo call remains open until 19 September AoE, but a demonstration would be a different submission from a research paper. [Paper call](https://brainbodyfm-workshop.github.io/call-for-papers.html), [demo call](https://brainbodyfm-workshop.github.io/call-for-demos.html)

**NeurReps: topic fit 5/5; track choice matters.** Direction B directly concerns the geometry of neural representations. The four-page, non-archival extended-abstract track is the best format fit for the present controlled findings and developing explanation. The call advertises a nine-page archival proceedings track for more developed work. Its **Findings Track receives 2/5 for our current work**: it emphasizes high-impact collaborations between experimentalists and theorists, including biological discoveries with substantial geometric insight. It uses editorial review and has no page limit. “Findings” here does not mean an easier route for any negative experimental result. The posted deadline was 24 August AoE; no extension was verified. [Official track descriptions](https://neurreps.org/#cfp)

**Physical World AI: topic fit 4/5.** Articulated geometry and evaluation of learned physical representations are relevant, with Direction C offering a stronger temporal connection if completed. The present work does not evaluate contact, materials, or multimodal sensor fusion. The call advertises an archival deadline of 9 September and a non-archival window of 29 September–29 October, with eight-page papers or four-page extended abstracts. That later window is the most practical posted paper route among the previously considered venues for developing a focused result. [Official call](https://physworld-org.github.io/physworld.github.io/cfp/)

**Foundation Models for Temporal Systems (FMTS): topic fit 4/5 for Directions A and C.** This additional workshop explicitly includes irregular sampling, sparse observations, leakage-aware evaluation, and temporal consistency. It accepts four-page non-archival submissions, including preliminary and negative findings, until 15 September AoE. A completed analysis of movement timing or forecasting would fit; a general roadmap alone would not establish a temporal-learning contribution. [Official call](https://fmts-workshop.github.io/cfp.html)

**Embodied Spatial Reasoning: topic fit 3/5.** Body geometry and movement prediction connect to spatial and temporal reasoning, but we do not yet study an interacting agent, surrounding objects, or spatial memory. The workshop welcomes preliminary and negative results; its posted 5 September AoE paper deadline has passed. [Official call](https://embodiedsr.github.io/call-for-papers.html)

**GenAI4Health: topic fit 2/5.** Careful evaluation is relevant to clinical trust, but a coordinate-derived gait result does not supply the missing medical generative application or clinical benefit. Its 9 September AoE deadline is close, and the evidence should determine the claim rather than the available deadline. [Official call](https://genai4health.github.io/2026-NeurIPS/)

**Med-Reasoner: topic fit 1/5.** The current pose-only model does not address the workshop's medical vision-language reasoning problem. Its posted 5 September AoE deadline has passed. A future submission would require a genuine clinical reasoning task and evidence about that reasoning, not simply health-related input data. [Official call](https://med-reasoner.github.io/neurips2026/call_for_paper.html)

NeurReps and PhysWorldAI advertise archival options, while central NeurIPS guidance describes workshop papers as non-archival. This unresolved policy difference should be checked with organizers before choosing an archival route or relying on its compatibility with another submission. [NeurIPS workshop guidance](https://neurips.cc/Conferences/2026/WorkshopsGuidance)

### Why IAAI remains a different research trajectory

IAAI-27 has an application focus, giving the current work a fit score of **1/5**. Its emerging-applications track seeks pilot or early-deployment evidence and a credible path to use; its deployed-applications track requires production use and measured benefits. The current notebooks document neither. The clearest route would include an intended user and decision, an end-to-end system, evaluation against the existing workflow, a real user pilot, and evidence about failures and operation. These would be new application work, not a refinement of the present representation-learning results. [IAAI-27 call](https://aaai.org/conference/aaai/aaai-27/iaai-27-call/)

The posted IAAI deadline is 8 September AoE. Its restrictions on overlapping submissions explicitly exempt limited-audience workshops without archival proceedings; this corrects an overly broad reading of the policy. Each destination's own rules still need checking. The near deadline does not resolve the missing application evidence. [Submission policy](https://aaai.org/conference/aaai/aaai-27/iaai-27-call/)

## 8. A practical sequence toward the next paper

1. **Preserve and explain the completed masking results.** Lead with the new comparison's complete source coverage, both learned-versus-initial controls, and the uncertainty in the masking contrast. Explain why improved feature prediction does not establish improved movement prediction. Keep the older implementation's scores distinct.
2. **Resolve the smallest plausible explanations first.** Broaden the training-only regression penalty range, then compare temporal preparation and motion-sensitive summaries in separate steps. Once these controls are informative, test a limited motion-weighted or structured-mask comparison with a count-matched scattered reference. In parallel, the frozen-feature reflection test from Direction B remains a focused geometry question.
3. **Choose one main contribution.** If the information-recovery experiments identify a repeatable mechanism, develop a focused body-representation paper. If the learned reflection rule explains feature behavior with a useful consequence, develop the geometry paper. Pursue the larger forecasting study when its simple real-data controls establish a credible starting point.
4. **Confirm the selected result independently.** Freeze the principal choices before the external evaluation and make the intended generalization claim explicit. Keep clinical interpretation proportional to the available reference measurements.
5. **Write around the answer, including an unfavorable one.** A workshop paper should make one question, its decisive comparison, and its implications easy to follow. Avoid presenting the entire notebook roadmap as demonstrated novelty. Finalize the applicable ethics, data-use, and submission-policy documentation separately from the scientific assessment.

The project has progressed from a sensible anatomical training choice to a more demanding question about useful learning. The completed masking result gives us a concrete problem to explain, and the new notebooks provide ways to distinguish possible explanations. The strongest next paper will come from resolving one of those questions with a clear comparison and an observable movement outcome.

## 9. Copy-ready implementation prompt for the next notebook suite

The specification below has now been implemented in notebooks 11–14. Their small examples use generated movement to check the software, and Notebook 12's retained working copy now records an explicitly enabled, completed real-data gait-target versus all-landmark comparison. The other masking-family comparisons and real-data future-feature training remain unrun. Section 3 reports the evidence from the completed grid rather than assuming every implemented method has been evaluated.

The automatic missing-observation comparisons evaluate sensitivity to masking prepared coordinates. Notebook 13 also provides and tests a helper that removes raw observations before input preparation. A full dataset comparison using that helper needs a declared raw-recording corruption plan; the prepared-coordinate results alone would support a narrower sensitivity claim.

### Implemented notebooks and how to use them

| Notebook | What the reader can examine |
|:--|:--|
| [11 — Masking patterns and coverage](../11_masking_patterns_and_coverage.ipynb) | Eight policies, illustrated target patterns, natural missingness, and feasible versus impossible hidden counts |
| [12 — Controlled masking pretraining](../12_controlled_masking_pretraining.ipynb) | Paired training, a loss that preserves each clip's targets, retained reference settings, and a real-data workload preview |
| [13 — Encoder and predictor evaluation](../13_masking_encoder_and_predictor_evaluation.ipynb) | Feature prediction, separate online and teacher readouts, initial and direct-pose controls, missing-observation tests, and source-level uncertainty |
| [14 — Future features and movement prediction](../14_future_features_and_movement_prediction.ipynb) | A training-only decoder applied to observed and predicted future features, past-only baselines, and a separate twelve- versus 33-landmark input option |

Read these notebooks in order for the explanation, or use their inexpensive synthetic examples independently. For example, Notebook 12 demonstrates hiding whole joint trajectories with a scattered reference at the same count; Notebook 13 then examines movement readouts and feature variation. Those short examples cannot establish a preferred masking method for gait. The separate real-data section in the retained Notebook 12 has its training flag enabled because that run was performed; inspect the settings before executing it again.

The executed working copies of notebooks 11–14 now have commentary next to their retained outputs. An earlier [executed teaching copy of Notebook 14](../executed/comparative_masking/synthetic_9jdx6f_d/14_future_features_and_movement_prediction.ipynb) also has annotations; its reported synthetic results agree with the current working copy at the displayed precision. This review changed explanation cells only, without changing code, numerical outputs, or model artifacts.

The [experiment specification](COMPARATIVE_MASKING_PLAN.md) gives the comparison stages. The commands below regenerate output-free source notebooks and run synthetic verification. Regeneration replaces the annotated working copies, so archive those copies before using these commands; they were deliberately not run during this results review.

```bash
.venv/bin/python \
  neurips-laterality/scripts/build_research_notebooks.py --only 11 12 13 14
.venv/bin/python \
  neurips-laterality/scripts/verify_comparative_notebooks.py --execute
```

The verifier saves executed copies and vector figures in a new directory under `executed/comparative_masking/`. Real-data training remains an explicit notebook choice after a compact workload summary is displayed; the complete machine-readable workload remains in `real_plan["workload"]`. The real runner explains its workload, hardware, memory strategy, and recovery policy once, then prints only one start and one plain-language outcome for each fold/seed job. Notebook 12 also uses `notebook_progress.py` for one updating display of the active fold, seed, masking arm, optimizer position, sampled loss, elapsed time, and adaptive ETA. Completed compatible results and evaluations are reused. An interrupted fold/seed job can resume from a verified checkpoint that contains both independently optimized masking arms at the same update boundary. Notebook 12 saves that state every 300 updates per arm, reducing large checkpoint writes while bounding repeated work after an interruption.

The following prompt is retained as the implementation specification. It can also guide a review or a further extension; its proposed filenames now refer to the implemented notebooks linked above.

### Role and objective

Act as an AI/ML researcher and research software engineer specializing in skeletal representation learning and JEPA. Extend the tutorial notebooks in `neurips-laterality` to compare masking policies, the encoders trained with them, and the resulting predictor performance. The central question is which missing-information tasks help preserve useful movement differences on recordings excluded from training.

Begin by reading the applicable project instructions, `docs/TUTORIAL.md`, notebooks 00–10, their editable tutorial sources, the relevant training and evaluation helpers, and the retained real-data results. Check the primary papers cited in Section 4 before implementing a named method. Follow the user's plain-research-writing preferences. Use subagents for independent implementation, testing, and adversarial review, with explicit file ownership to avoid conflicting edits.

Implement and verify the notebook suite, but do not launch the full real-data training grid automatically. Default execution must use inexpensive synthetic examples. Provide a clearly labeled, explicitly enabled path for real-data experiments, display its aggregate counts before training starts, and retain the complete job table in the plan object without dumping it into the routine log. Never invent performance values or substitute synthetic examples for missing empirical results.

### Preserve the completed research

Keep notebooks 00–10, their tutorial sources, `laterality/`, the existing protocol, saved results, checkpoints, and executed copies unchanged. In particular, do not extend `laterality_extensions/masked_learning.py` in place: modifying that implementation can invalidate compatibility checks for the completed masking comparison. Put new policies and runners in new modules, reusing unchanged components where appropriate.

The next unused notebook number is 11 at the time of this prompt; check again before creating files. Follow the existing editable-source convention with `tutorials/research_11.py` and subsequent numbered sources. Extend `scripts/build_research_notebooks.py` carefully and generate only the selected new notebooks with `--only`; its default behavior regenerates every notebook in its mapping. Preserve output-free source notebooks and save executed teaching copies separately. Add focused test files without weakening existing tests, and verify that protected files remain unchanged.

### Create four connected tutorials

1. **`11_masking_patterns_and_coverage.ipynb`.** Explain the policies and implement their construction before meaningful model training. Show what is hidden and what remains observable, using readable vector graphics. Demonstrate eligibility, valid-token counts, temporal spans, anatomical coverage, and impossible-budget cases. Clearly separate naturally missing observations from deliberate training masks.
2. **`12_controlled_masking_pretraining.ipynb`.** Implement a shared training runner with matched initial states, source sampling, augmentations, encoder, predictor, teacher construction, and optimizer exposure. Separate the anatomical-selection comparison from the motion and mask-shape comparison. Include a tiny synthetic execution and an explicit real-data configuration, with safe saving and reuse of completed compatible runs.
3. **`13_masking_encoder_and_predictor_evaluation.ipynb`.** Evaluate both the pretraining predictor and the usefulness of frozen encoder features. Include initial encoders, direct-pose summaries, and training-mean controls. Aggregate source-held-out predictions correctly, inspect declared online/teacher checkpoints, and evaluate prespecified missing-observation conditions. Explain why a lower feature-prediction loss may coexist with weak movement prediction.
4. **`14_future_features_and_movement_prediction.ipynb`.** Build on the question in Notebook 10 without rewriting it. Separate regression from past features from decoding the JEPA predictor's future features. Provide the training-only decoder control described below, simple forecasting references, and tests excluding future information from the input. Keep twelve- versus all-landmark input as a separate configurable comparison with the same future landmark endpoints. Its implementation and synthetic tests are required; real training requires explicit enablement.

Each notebook should introduce its question and dependencies, explain consequential design choices before the code, and use small code cells that expose the method. Move repetitive machinery into readable helpers. Explain each displayed result in terms of what it supports, what it leaves unresolved, and what outcome would weaken the hypothesis. Avoid unexplained condition nicknames, long decimal strings, local run identifiers in the narrative, and claims of neurological validity unsupported by the experiment.

### Separate the experimental questions

Before implementing the training grid, write a compact experiment specification distinguishing:

- **Landmark eligibility:** gait-only targets, all-landmark targets, fixed random subsets, and a soft gait preference, with scattered mask geometry held constant.
- **Mask shape and movement weighting:** scattered, motion-weighted, complete-trajectory, and connected-region masks under the same declared landmark eligibility.
- **Time prediction:** an interior missing interval with observations on both sides versus a future interval with observations only before it.
- **Other model choices:** input landmarks, regularizer pooling, temporal feature summaries, predictor capacity, and target/loss changes. These belong in separate comparisons.

For the first masked-learning comparisons, retain the existing 33-landmark input, twelve-landmark pooling of unmasked regularizer views, five-pair laterality summary, and coordinate-derived target. Describe these remaining anatomical choices openly. Do not call the all-landmark masking condition free of all anatomical guidance. Do not describe the extension as an exact reproduction of the original paper's training schedule.

### Define the masking policies precisely

Implement a common policy interface that receives the permitted observations, token-validity mask, budget specification, and a dedicated random generator. It should return the deliberate mask, unique target positions, remaining-context counts, and a coverage summary. It must not receive the laterality label or clinical-condition annotations.

- **Gait-only scattered masking:** preserve the existing twelve-landmark eligible set and redraw joint–time targets at every update.
- **All-landmark random masking:** sample without replacement from every valid joint–time token, with equal initial sampling weight.
- **Fixed random twelve-landmark sets:** preselect several sets with separate recorded selection seeds, reuse them across folds and training seeds, and report them as a declared group. Random individual landmarks and random bilateral pairs are different controls; name them accordingly. Do not select a favorable subset after seeing prediction results.
- **Soft gait preference:** keep every valid token eligible. A starting rule is an equal mixture of uniform all-landmark probabilities and uniform gait-landmark probabilities, followed by weighted sampling without replacement. Specify the fallback when no gait candidates are valid. This mixture does not guarantee that exactly half the sampled targets come from each component, and its weight is a proposed comparison setting rather than a validated optimum.
- **Motion-weighted masking:** make greater reliable movement increase the probability of being hidden, consistent with MAMP and S-JEPA. Specify how movement is computed, its timing units, normalization, treatment of missing transitions, and control of isolated jumps. Retain a uniform sampling component and define behavior for stationary or unusable motion. Keep any motion-target reconstruction objective separate from this sampler.
- **Whole-joint trajectories:** hide selected landmarks throughout the entire input window wherever their observations are valid. Do not shorten the trajectory to satisfy a count while retaining the whole-trajectory label.
- **Connected-region intervals:** use explicit anatomical connections to select a connected region and a consecutive time interval. Ensure the region definitions collectively cover the declared candidate pool. Array-index adjacency is not an anatomical graph.
- **Interior temporal gaps:** hide every valid landmark in a contiguous interior interval while retaining observations before and after it. Evaluate this as completion; implement future-only masks separately in the forecasting tutorial.

Complementary views, learned selectors, broad mixtures of mask families, and visible-token supervision should remain optional follow-ups. Do not add them to the primary grid merely because the interface can support them.

### Make token-budget feasibility explicit

Preserve the current four-step tokenization and validity definition unless a separately named experiment changes them. Count valid input tokens, deliberately hidden valid tokens, remaining valid context, and the fraction hidden. Missing measurements cannot become supervised targets, and overlapping blocks cannot count a position more than once.

Match realized hidden counts exactly for compatible comparisons. Require at least one valid context token. When a new policy lowers the shared feasible count, rerun its reference conditions at that count. Record the actual interval lengths, landmark coverage, and body-side coverage rather than relying on a policy name or requested percentage.

Check counts within each batch as well as between policies. The existing prediction path flattens selected targets and reshapes them by batch, assuming the same count for every example. If structured masks and missingness produce unequal counts, either enforce a feasible equal-count invariant or implement a new dense or packed target path that preserves sample and token identities. Do not reuse the old reshape on unequal counts. Define loss weighting explicitly—for example, average over each example's hidden targets before averaging examples—and test that targets cannot be mixed between clips.

Do not require an impossible combination of exact counts and intact mask shapes. With sixteen time blocks and 33 fully valid landmarks, whole trajectories contribute multiples of sixteen and whole time blocks contribute multiples of 33. Their only common positive count is the whole input, which leaves no context. Use separate count-matched scattered references, a declared coverage-range analysis, or separate experiments for these families. Never trim a structured mask or add isolated cells silently to achieve an arbitrary count, and never describe approximate matching as exact.

### Control training, randomness, and model selection

Use the established five source-video outer groups and five training seeds for the declared full comparison. Every clip from an outer-test video must remain excluded from encoder training, selection of policies or checkpoints, and fitting the movement readout. Repeated seeds represent repeated training choices on the same recordings, not new independent participants.

Use separate reproducible random streams for initialization, source sampling, geometric views, policy-specific mask draws, random-subset selection, and evaluation corruption. An adaptive policy's additional random draws must not change which training clips its reference model receives. Pair compatible initial weights, source schedules, augmentations, and update counts; save evidence that these controls hold. The realized mask itself should differ when mask shape is the intended comparison.

Keep architecture, predictor capacity, teacher update rule, learning-rate schedule, regularizer, batching, training exposure, and readout procedure fixed within a masking-only comparison. If a later experiment changes predictor capacity or encoder dimensions, retain a matched initial control for each architecture and state which parameters can still be paired. Report compute or runtime as well as update counts; equal updates need not imply equal cost.

Extract and display the complete real-data recipe from the saved Notebook 08 comparisons as the starting reference. Do not inherit a small synthetic demonstration's settings for real training. Preserve its 1,200-update budget for a direct extension, or document a different budget selected through training-only evidence and apply it consistently to the compared methods. List all model, optimizer, schedule, teacher, masking, and regularization settings before starting a run.

Predeclare a small primary comparison and a limited training-source validation procedure. Use inner source-separated validation for regression penalties and any mask, budget, or checkpoint choices. Do not use the outer test results to select the best policy or early stopping point. If an inner validation result is intended to assess an entire pretraining recipe, also exclude those validation sources from that candidate's encoder training. Distinguish this from tuning only a supervised readout on representations already pretrained within the outer training set.

Check the current evaluation's ability to detect useful information before expanding training. In a synthetic test with a known movement signal, demonstrate that an appropriate simple readout can recover it; this verifies the evaluation pipeline without implying a real-data gain. Treat subsequent choices informed by the already inspected GAVD results as development decisions, with independent evaluation needed for stronger confirmation.

### Evaluate three different kinds of prediction

**First, measure the JEPA predictor's behavior.** Evaluate hidden-feature prediction on a fixed, declared bank of evaluation masks and include appropriate initial and mismatched-target controls. Preserve its normal pathway: online/context encoder to predictor, with the teacher supplying targets. Record feature variation for targets and predictions, feature norms, effective rank, and a clearly defined normalized error with its denominator. Compare online and teacher features through separately fitted frozen readouts at prespecified checkpoints, using training-source validation for any selection. Substituting the teacher into the predictor's input pathway would be a separate experiment.

Errors against each arm's own learned teacher are not a common measurement scale: a model can make prediction easier by producing less variable targets. Do not rank useful learning by raw training loss or assume that normalization fully resolves this issue. A common frozen teacher could support a separate comparison trained against that shared target. Do not score existing predictors directly against a different teacher's vectors without an explicitly fitted and controlled alignment; their latent coordinates are not automatically compatible. The between-method usefulness claim should rest on a shared observable movement outcome.

**Second, evaluate frozen-feature movement prediction.** Fit the same training-only imputation, scaling, and ridge-selection procedure for every representation. Retain the matched initial encoder, direct-pose summaries, and a mean target estimated from training sources. Keep online and teacher results distinct, and use a common, declared primary endpoint rather than selecting whichever metric favors a method.

Use source-balanced R² as the primary laterality score and mean absolute error as a complementary error measure. Report sign behavior where useful, while explaining instability for targets close to zero. Within each seed, pool all out-of-fold predictions before computing the metric; then summarize seed-specific scores. Do not average fold R² values or silently replace the analysis with an ensemble of seed-averaged predictions. Include evaluated clip and source counts for every table.

**Third, evaluate future-feature prediction in Notebook 14.** After fitting the forecasting model on training sources, freeze it and fit a decoder from observed future-teacher features to future coordinates using training videos only. Freeze that decoder and evaluate it on both observed and predicted future features from held-out videos. The observed-future condition checks whether the features and decoder can express the endpoint; it is a diagnostic with access to the future, not a deployable forecasting baseline. Demonstrate this path synthetically first and leave real-data results absent until their experiment runs.

Compare with regression directly from past coordinates, regression from past encoder features, an initial encoder, persistence, recent-velocity continuation, and an encoder trained against mismatched futures. Sample mismatched futures from other training sources at the same horizon, balancing sources before clips. Keep horizons, prediction units, and evaluated observations identical across the relevant methods. Report actual observation times when sampling does not land exactly on a requested horizon.

Notebook 10's existing past-feature regression must retain its current interpretation. It does not decode the JEPA predictor's future feature vector. Do not call multiple forecasts from one fixed prefix a recursive rollout; recurrent prediction requires a compatible latent transition or an explicitly tested decode-and-re-encode procedure.

### Test missing observations without exposing withheld information

Create a shared, prespecified bank of evaluation corruptions, including scattered gaps and connected missing regions on both sides of the body. Every compared model must receive the same corrupted observations. Keep the laterality score from the unaltered recording as the reference; predict that score from the remaining observations. The unaltered recording may already have missing measurements. Actual coordinate reconstruction requires its own decoder and metrics.

For a claim about genuinely missing data, remove the selected raw observations before interpolation and normalization, using only permitted observations or independently specified reference geometry to prepare the input. A test masking already prepared coordinates should be identified separately as representation sensitivity. Report corruption feasibility, unavailable predictions, and changes in coverage; do not silently discard difficult cases or make each method's test set more favorable.

Ordinary full-input teacher targets during self-supervised training remain legitimate. For past-only forecasting, however, input preparation, context-derived motion scores, and any adaptive selection must use only the observed prefix. The teacher can use the declared future window for training targets. Changing future values must not change the input available to the context encoder or its deterministic predictions.

### Require tests before interpreting real results

Add focused tests that establish all of the following:

- Identical seeds and inputs reproduce masks; outputs contain no duplicate targets, no invalid targets, and at least one valid context token.
- Fixed subsets remain fixed while masks are redrawn; soft preference retains non-gait eligibility; motion-weighted sampling hides higher-motion tokens more often in a repeated-sampling test, with valid stationary and noisy-input behavior.
- Complete trajectories, connected regions, and temporal gaps satisfy their definitions. Infeasible budgets fail clearly or follow an explicitly documented alternative.
- Reflection exchanges coordinate signs, anatomical identities, validity, and mask locations consistently. Both anatomical sides can be selected.
- Withheld content does not enter the relevant student or evaluation pathway. Hold the selected mask fixed when perturbing hidden values to test direct content isolation. Motion-based selection during ordinary self-supervised training can legitimately respond to changes in the complete permitted observation; future-informed selection remains forbidden in a past-only input. Changing labels cannot alter label-blind pretraining when splits and random streams are fixed.
- Teacher parameters receive no gradients; online and teacher checkpoints are stored and identified separately.
- Fitting functions cannot access outer-test sources. Altering outer-test values leaves trained parameters and selected hyperparameters unchanged.
- Perturbing future coordinates, visibility, and timestamps that remain after the boundary leaves prepared context and past-only predictions unchanged. Moving an observation across that boundary is a different test.
- Constant or nearly constant features trigger an explicit diagnostic rather than appearing to demonstrate successful learning.
- Aggregation rejects duplicate predictions, cross-fold source overlap, incomplete declared coverage, and incompatible paired comparisons. A partial run can be summarized only with its actual scope clearly labeled.
- Fresh-kernel execution completes the synthetic tutorials and renders their figures without relying on hidden notebook state.

Tests should verify the implementation and scientific controls. They must not require a proposed mask to outperform a baseline; an unfavorable empirical result is a valid outcome.

### Retain results and quantify uncertainty carefully

Use separate output locations for the new experiments and refuse to overwrite completed results. Save configurations, data and split references, subset definitions, budgets, realized coverage, source schedules, loss components, declared checkpoints, and per-clip predictions with their source, fold, seed, condition, target, and prediction.

Only reuse a completed run when its data, split, policy, model, objective, schedules, random seeds, and implementation are compatible. Corrupted, incomplete, duplicated, or mismatched saved results must fail clearly. Save atomically. If resuming training is supported, retain and restore optimizer, teacher, model buffers such as target centers, every sampler's random state, schedules, and global update position. Test agreement with an uninterrupted run within declared numerical tolerances; otherwise label interrupted runs incomplete.

For paired uncertainty, resample complete source videos while keeping their clips, conditions, and seed predictions together. State that these intervals condition on the fitted models and do not represent full retraining uncertainty. Report variability across seeds separately. Keep full precision in machine-readable results and use sensible rounded values in the tutorial.

Before any real run, display the selected conditions, folds, seeds, updates, approximate workload, output locations, and whether the request is a pilot or the complete declared comparison. Do not launch a large Cartesian product of all masks, ratios, models, and endpoints by default.

### Definition of done

The implementation is complete when the new source notebooks validate and execute their synthetic examples in fresh kernels; mask and evaluation tests pass; protected research files are unchanged; and the explicit real-data path can validate its inputs and display its workload without starting training. The documentation must distinguish implemented methods, software demonstrations, completed empirical comparisons, and analyses still awaiting results.

Use independent reviewers to check mask definitions and budgets, source separation and evaluation, and the clarity of the explanations. Fix substantive findings and rerun the affected checks. In the handoff, state which notebooks and helpers were added, what was executed, what remains unrun, and which scientific conclusions the retained evidence supports. Update the tutorial and its PDF when new results exist, without inventing gains or treating a useful anatomical prior as established neurological validity.

End of implementation prompt.

## Evidence and figure reproduction

This overview combines the completed results in notebooks 00–05, the diagnostic procedures in 07, the earlier real-data comparison associated with 08, the new complete real-data comparison in 12, and the retained teaching executions of 09–14. Absence of inline output was not treated as absence of a saved experiment. No real-data symmetry-training or future-feature forecasting result was found in the reviewed material.

The current masking table and Figure 1 use one pooled held-out score per seed, followed by averaging those scores. The plotted interval is the saved Notebook 12 teacher-feature contrast, calculated from 2,000 resamples of the 93 source videos while keeping each video's clips and all paired seed predictions together. It conditions on fitted models and excludes retraining uncertainty. The distinct Notebook 08 summary retains its earlier 20,000-resample exploratory analysis, which is not the interval plotted here.

The [new result-figure generator](figures/make_comparative_findings.py) verifies the saved summary files, checks paired prediction coverage, and independently recomputes source-balanced scores before drawing Figure 1. Its [numerical summary](figures/tutorial_comparative_masking_summary.json) retains full-precision aggregate values for the new comparison. The [earlier generator](figures/make_tutorial_figures.py) and its [summary](figures/tutorial_masking_summary.json) remain available for the distinct Notebook 08 evidence. The [mask-pattern illustration generator](figures/make_tutorial_masking_patterns.py) draws Figure 2 and verifies that its four hypothetical panels hide equal numbers of cells; it does not run an experiment.

The review checked the new grid's 25 paired jobs and all 50 complete training histories, compared the saved prediction tables with an independent metric calculation, and inspected predictor diagnostics, readout selection, and corruption coverage. Recorded forecasting numbers come from the executed synthetic notebook; raw per-example forecast artifacts were not retained for that demonstration, so differences smaller than its displayed precision were not inferred. Notebook commentary was checked by independent masking, evaluation, and forecasting reviewers, and its code and outputs were preserved. No new model training was performed.

Run the [PDF build script](build_tutorial.sh) after editing this Markdown source to regenerate [TUTORIAL.pdf](TUTORIAL.pdf). Notebooks 00–10, their source tutorials, model implementations, and empirical artifacts were not edited during this documentation review.
