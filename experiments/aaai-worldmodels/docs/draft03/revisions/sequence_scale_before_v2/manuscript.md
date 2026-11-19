# Separating Movement Prediction from Gait Anomaly Detection: A Reference-Controlled Benchmark

## Abstract

A compact model can predict future leg positions well without providing a better measure of gait abnormality. We demonstrate this distinction in a benchmark for movement observations in medical world models. Models learn only from videos annotated as normal and predict eight future observations from sixteen past observations. The internal evaluation contains 2,815 windows from 456 sequences and 91 GAVD source videos, including 29 normal sources. A learned transition from current posture reduces normal-source forecast error by 40.8% relative to repeating the last pose. We then compare six training-mask policies with matched hidden-input budgets and an unmasked prefix baseline, keeping the model and future targets fixed. Anatomical and motion-balanced masks do not improve on uniform masking. We separately test a locked score from the posture-plus-velocity model that asks whether a source is easier or harder to predict than expected from its movement amount and missing observations. Its within-fold AUROC is 0.462, compared with 0.811 for a low-motion baseline that ranks smaller displacement over the observed future as more abnormal. Frozen transfer to a separate dataset with 90 coded participants gives 0.591 and 0.762, respectively. The contribution is a reproducible benchmark that establishes useful short-horizon forecasting and makes its limits as evidence for anomaly detection explicit.

## Introduction

Walking depends on the coordinated movement of many joints. Changes in that coordination can make movement more variable, more restricted, or more dependent on compensation. A model that learns how normal observations evolve could help describe such changes. For a medical world model, this would provide a testable observation component: a way to represent movement and check whether a short future is consistent with the observed past. Establishing this component requires evidence about the measurements and the prediction task before stronger claims about patient state or intervention are possible.

Prediction offers an appealing route to gait anomaly detection. A model can learn from examples annotated as normal and assign an unusualness score when a new sequence does not follow its expectations. However, an error score can reflect several things at once. A nearly stationary leg may be easy to predict even when its movement is clinically restricted. A fast-moving leg may be difficult to predict because the camera or pose estimator provides poor observations. A model may also forecast well because current posture already constrains the short future, without using the detailed order of earlier movement.

These possibilities motivate our central question: can a model recognize departures from normal movement organization beyond what is explained by movement amount, current posture, camera view, and observation quality? We address this question with compact models and explicit controls. We first compare learned forecasting with simple references. We then test whether hiding anatomically or temporally organized parts of the past improves a fixed forecasting model. Finally, we evaluate an anomaly score against a low-motion comparator and normal-only posture and quality controls. Each question has its own endpoint because success on one does not imply success on the others.

The study makes three contributions. First, it provides a reproducible observation and evaluation contract that keeps related video windows together, isolates the future from model inputs, and compares methods on identical observed targets. Second, it measures the value of compact state descriptions and six budget-matched masking policies and an unmasked prefix baseline, identifying a strong current-posture reference. Third, it tests reference-aware anomaly scoring internally and in frozen transfer, making the gap between useful prediction and incremental anomaly ranking explicit. The contribution is a controlled benchmark and an empirical finding about evaluation. Normal-only forecasting, anatomical masking, and motion-aware masking already have substantial precedents.

![The benchmark asks two separate questions: can a model predict future articulation, and does its error identify departures beyond simple references? Each GAVD window contains sixteen past and eight future observations at nominal 10 Hz. Methods use the same evaluation windows and future target support, with their declared past inputs. Future availability never conditions a prediction.](figures/observation_task.svg)

The benchmark asks two separate questions: can a model predict future articulation, and does its error identify departures beyond simple references? Each GAVD window contains sixteen past and eight future observations at nominal 10 Hz. Methods use the same evaluation windows and future target support, with their declared past inputs. Future availability never conditions a prediction.

## Movement motivation and related work

Laboratory studies provide reasons to examine how joints relate over time. Bilateral gait timing differs in the Parkinson's disease population studied by Plotnik and colleagues [@plotnik2007]. Studies using muscle recordings associate altered coordination complexity with post-stroke walking and cerebral palsy [@clark2010; @steele2015]. These findings motivate tests of timing and multijoint organization. They do not give a unique interpretation to a video prediction error. Our measurements contain projected joint positions rather than muscle activity, verified foot contacts, neural signals, or muscle strength. We therefore test observable relationships without inferring a neural controller or affected side.

Human-motion forecasting already uses simple references such as repeating the latest pose, which can be strong at short horizons [@martinez2017]. Gait-specific self-supervised detection also predates this work [@duan2024]. Our question concerns the additional evidence needed to interpret forecasting as an abnormality measure under heterogeneous video observations. We do not compare headline scores across studies that use different cohorts, labels, pose formats, or supervision. A classifier adapted with condition labels answers a different question from a predictor fitted only on normal sources.

Masking deliberately hides observed input during training. It can discourage a model from relying on a nearby measurement that makes a task too easy. VideoMAE uses temporal tubes in video representation learning [@tong2022]. MAMP uses motion-informed masking for skeletal action representations [@mao2023]. S-JEPA predicts learned numerical summaries of withheld observations rather than their coordinates; a separate target encoder produces the summaries that guide learning [@abdelfattah2024]. A small summary-matching loss alone cannot establish a useful representation because an encoder could discard information.

Recent skeleton methods make the relevant precedents especially clear. The revised SLiM method uses connected anatomical tubes, while ASMa combines complementary motion and spatial masking [@do2026; @anand2026]. Our policies adapt these ideas to sixteen intermittently observed joints and a small normal-source cohort. They change the input to one coordinate predictor rather than reproduce those larger action-recognition systems. This design tests the proposed information constraints before changing the representation objective or increasing model size.

## Videos, joints, and prediction targets

### Internal and external populations

We rebuild the local GAVD inventory from the condition folders, excluding its pinned provenance copy from counts [@ranjan2025]. The complete inventory contains 1,874 sequences and 348 source identifiers. The five study annotations, normal, parkinsons, stroke, cerebral palsy, and myopathic, contain 666 sequences from 103 sources. Media availability leaves 652 sequences from 98 sources. Preserving two known source-reference exclusions leaves 641 extraction candidates from 96 sources. Quality and common-target gates produce the cohort in Table 1. We reuse verified full-duration pose caches rather than extract only the first few frames of each sequence.

| GAVD annotation | Sources | Sequences | Windows |
| --- | --- | --- | --- |
| cerebral palsy | 8 | 52 | 518 |
| myopathic | 27 | 168 | 845 |
| normal | 29 | 141 | 465 |
| parkinsons | 11 | 37 | 292 |
| stroke | 16 | 58 | 695 |

The internal model cohort after common observation gates. Source counts are video identifiers, not independently verified patients.

All 32 annotated normal sources already occur within this five-label scope, and 29 survive the model gates. Processing other GAVD categories cannot increase this normal-source support. Those categories include exercise, pregnancy, stylized walking, and prosthetic use, which retain their actual meanings. We audit their inventory without calling every unusual movement neurological disease. A source-relation table checks exact duplicate media and known relationships before splitting. Each group contains all sources connected by those verified relationships. No additional linked component is established in the retained cohort, so its 91 components are single source videos. Unrecognized repeated people remain possible.

For transfer, we use the separately released KOA-PD-NM video dataset [@kour2020; @kour2022]. The local cache contains 189 originals from 95 dataset-coded participants. Byte-identical copies in another folder are counted once. Two files cannot be decoded. Processing all 187 decodable recordings with the fixed gates retains 773 windows from 168 recordings and 90 coded participants: 27 normal, 49 osteoarthritis, and 14 Parkinson's. All recordings from one coded participant stay together. We report these counts separately because video sources and coded participants are different grouping units; their names alone do not establish independent people.

### What the model observes

A pose is an estimated arrangement of body joints in an image. We use Ambient's shared RGB-image extraction interface with a pinned BlazePose weight file. Sixteen positions describe the shoulders, arms, hips, knees, ankles, heels, and foot indices, which are landmarks near the fronts of the feet. We subtract the midpoint between the hips, our pelvis reference, in each frame and divide by a torso scale estimated from past observations only. The scale is the median valid distance between the hip midpoint and shoulder midpoint in the prefix. These coordinates describe relative two-dimensional articulation in prefix-torso units. They are not physical walking speed or measured three-dimensional joint angles. Both hips enter the shared reference, including in an own-leg model.

Each window has sixteen past observations and eight future queries on a nominal 10 Hz grid, with stride eight between windows. The past spans 1.5 seconds between endpoints. Primary error averages all eight future offsets from 0.1 through 0.8 seconds, rather than only the final offset. We select native frames nearest the grid, retain actual decoded times, and never call an interpolated joint a measured observation. GAVD frames are one-based and decoder indices zero-based; conversion occurs once at the decoder boundary.

Confidence must be at least 0.5. At least 70% of prefix and future joints must be observed, and both distal legs must pass common target-support gates. The eight predicted joints are both knees, ankles, heels, and foot indices. Predictors receive fixed future queries from the prefix. Future confidence and visibility determine which errors can be evaluated after the future arrives. They never select predictions, masks, anchors, or prefix normalization. Tests change every future coordinate and validity bit while requiring predictions and prefix preprocessing to remain unchanged. Future-relative target coordinates and evaluation support may change.

![A real normal-window example compares the learned current-posture transition and persistence on the same left-ankle x coordinate. Time zero is the latest prefix sample; the shaded area contains future queries. Coordinates use prefix-torso units. The example is the first eligible window in identifier order, selected without using forecast accuracy; it illustrates the task rather than population performance.](figures/forecast_example.svg)

A real normal-window example compares the learned current-posture transition and persistence on the same left-ankle x coordinate. Time zero is the latest prefix sample; the shaded area contains future queries. Coordinates use prefix-torso units. The example is the first eligible window in identifier order, selected without using forecast accuracy; it illustrates the task rather than population performance.

## Compact models and controlled masking

### References before larger models

Persistence repeats each target joint's latest observed position. It is both a short-horizon forecasting reference and a measure of how far the future moves from that anchor. A damped-velocity reference extrapolates half the latest observed rate of change using actual time gaps. Learned ridge regressions predict displacement from the same anchor. Ridge regularization limits coefficient magnitude in the small training population. We compare own-leg history, bilateral history, current bilateral posture, and a separately fitted opposite-leg timing control. The timing control applies a window-specific shift while retaining the anchor. It changes alignment rather than simply renaming feature columns.

The new compact state contains current posture, a robust recent velocity estimate, and uncertainty-related summaries of that estimate. Velocity is the median of valid adjacent differences over the latest five prefix intervals. Median absolute deviation and valid-pair counts describe its variability and support. This arm therefore adds both movement and observation information; its effect cannot be attributed to velocity alone. A matched current-posture ridge is the mandatory simple alternative. It learns a mapping from the current joint positions to future positions, whereas persistence copies those positions unchanged. An untrained random encoder with a fitted coordinate head checks whether a trained nonlinear representation is needed.

For masking, one multilayer perceptron has two hidden layers of width 32 and a residual coordinate head, with 31,936 parameters. It receives visible prefix coordinates, visibility indicators, latest observed anchors, and prefix times. Hidden values are filled only after masking. The objective is future-coordinate squared error with equal leg weighting. A learned representation here is the numerical summary inside this predictor. This experiment uses coordinates as targets rather than a teacher encoder or latent-target objective.

### A fixed task with different hidden past observations

Natural missingness, deliberate training masking, and future queries are stored separately. The entire future stays hidden in every arm. Additional prefix masks hide 40% of eligible non-protected tokens, with deterministic rounding; a token is one joint at one time. All observed reference joints and each joint's latest observed point are protected, including the target anchors. The unmasked prefix baseline retains every observed prefix token and therefore has more visible input. It provides a full-input reference for the six budget-matched corruption policies.

Uniform masking samples eligible tokens without replacement. Time blocks prioritize contiguous intervals. Anatomical tubes prioritize connected limb groups over time. Random joint groups match the anatomical policy's group sizes, intervals, and realized token counts. Motion-balanced masking distributes the budget across low- and high-motion prefix tokens, using confidence-qualified differences and actual time gaps. A motion-focused comparator favors higher-motion tokens. Both retain a uniform sampling component, so every eligible region can be selected. Missing motion is not zero motion, and ties have a deterministic rule.

Missing observations can prevent a complete block or tube from reaching the exact budget. The frozen implementation then truncates or completes its priority list. These are therefore budget-matched block and tube priorities rather than invariably complete tubes. Four fixed mask banks per window make exposure auditable. A mask-pattern-only diagnostic removes residual coordinate inputs to test the information in the chosen locations; the shared persistence anchor is still added at the output. Main evaluation uses the full observed prefix for every policy, with shared corruptions reserved for robustness.

## From forecast error to an anomaly score

### Why a forecast needs an explicit reference

Mean squared error, or MSE, averages squared differences between predicted and observed coordinates. Lower error means a closer forecast, in squared prefix-torso units. We compare the learned forecast and persistence on exactly the same future. Historical excess error is learned MSE minus persistence MSE. A negative value favors the learned forecast, while a higher value has the declared abnormality direction. This measures forecast gain but does not automatically remove movement amount.

For one observed coordinate vector, let a be the latest pose, y the future, and d the model's predicted displacement from a. The difference in squared errors is:

E_f - E_p = ||d||² - 2 <d, y - a>.

Here E_f and E_p are learned and persistence errors. The inner product measures alignment between predicted displacement and observed displacement y minus a. The identity shows that excess error can vary when movement amount changes even if the predicted displacement stays fixed. Common-weight averaging preserves that dependence. We therefore retain negative persistence error as a low-motion comparator: it assigns greater abnormality to less displacement from the anchor. This diagnostic direction was discovered in earlier outcome inspection and locked as a baseline for the continuation. It does not gain preregistered historical status.

### A normal reference for movement difficulty

The prespecified candidate uses the posture-plus-velocity ridge forecaster and asks whether a source is easier or harder to predict than expected from its past. First, we average future MSE within source. Prefix movement uses valid adjacent distal-joint differences divided by actual time gaps. We also retain prefix missingness. A small ridge regression g predicts log forecast MSE from log prefix movement and missingness using out-of-fit normal development predictions. Each such prediction belongs to a normal source excluded from the model that produced it. Subtracting this expected log error gives a scalar residual. Separate normal sources define how unusual the residual is:

r_s = log(E_f,s + epsilon) - g(z_s); A_s = abs(2 F_cal(r_s) - 1).

The index s denotes a source, E_f,s its hierarchical forecast MSE, and epsilon the numerical floor 0.00000001. The two values in z_s are log prefix movement and missingness. The function F_cal is the smoothed empirical cumulative distribution of residuals from normal calibration sources. It indicates where a residual falls in that reference. The absolute-value transformation assigns a high score A_s to unusually easy or unusually difficult prediction. This scalar residual measures forecast difficulty; it is not a signed coordinate residual or the historical excess score.

Normal sources have three separate roles. A development group fits the forecaster. Out-of-fit predictions within that group fit g. A disjoint normal calibration group supplies residuals and is excluded from both fits. The outer test group enters none of these operations. We rotate calibration membership three times per outer fold and average tail scores. The matched posture model uses the same schedule, testing whether the larger state adds value under the same reference mechanism. These anomaly components use smaller forecast-fitting pools than the full-development-pool forecasts reported in Table 2.

Separation is costly when normal support is small. Each calibration set has seven or eight sources, and its forecaster uses fifteen or sixteen. Smaller out-of-fit models supply the nuisance regression's training predictions, which may change error scale. Empirical ranks are coarse and have no automatic conformal coverage guarantee. We therefore report ranking rather than a clinical operating threshold. Repeated windows cannot increase independent calibration support.

## Evaluation and frozen transfer

All sequences and windows from a source-relation component stay in one outer fold. Five outer folds retain the earlier source partition; three inner normal-source folds choose ridge strength and training duration. Selection, scaling, masks, and calibration do not use abnormal training labels. Nevertheless, earlier GAVD outcomes informed the study, so every internal result remains exploratory. A new method name or split cannot make reused sources untouched.

Errors average observed coordinate discrepancies within leg, give legs equal window weight, average windows within sequence, and average sequences within source. Training samples source, sequence, and window hierarchically so long videos do not dominate. Mask policies share the model, objective, windows, targets, three seeds, and normal-selected duration. Neural predictions are averaged across seeds before computing error. Seeds measure optimization sensitivity rather than additional subjects.

The primary anomaly endpoint is the paired within-fold AUROC difference between the conditional two-sided candidate and low motion. AUROC is the probability that an abnormal group ranks above a normal group, with half credit for ties. Fold AUROCs are weighted by their normal-abnormal pair counts. We also report equal-fold and pooled rankings. Average precision summarizes precision across recall levels and depends on abnormal prevalence. Neither metric alone is clinical diagnostic accuracy. Secondary controls include normal-only posture distance, motion/quality distance, prefix movement, missingness, marginal calibration, and matched posture calibration.

Primary 95% intervals use 5,000 paired bootstrap resamples of groups within fold and binary label. Forecast intervals resample the 29 normal components. These intervals condition on saved overlapping fits and omit unknown participant links and complete pipeline retraining variation. Two additional grouped partitions repeat posture and posture-plus-velocity fits as correlated sensitivities. Mask comparisons are exploratory; we make no confirmatory significance claims across the family.

External transfer freezes all weights, transforms, nuisance regressions, directions, and normal references. For each of fifteen calibration components, we average windows within recording and recordings within coded participant. Prefix covariates follow the same hierarchy. We then apply the component's normal-reference transform and average its participant tail scores. Aggregation must precede this nonlinear calibration. The change from source videos to repeated participant recordings may alter score variance. No external participant fits or calibrates the model. Paired external intervals use 5,000 participant bootstrap draws.

![The locked posture-plus-velocity forecaster supplies the conditional two-sided anomaly score. Its smaller normal forecast-development fits and disjoint calibration groups differ from the full-pool forecasts in Table 2. The comparator is negative persistence error, which assigns higher abnormality to smaller realized displacement over the same observed future segment. The left panel shows pair-count-weighted within-fold AUROC on 91 GAVD sources. The right panel uses all 90 external coded participants without target-domain fitting. Lines are conditional 95% paired group-bootstrap intervals; differences compare the same resamples.](figures/ranking_comparison.svg)

The locked posture-plus-velocity forecaster supplies the conditional two-sided anomaly score. Its smaller normal forecast-development fits and disjoint calibration groups differ from the full-pool forecasts in Table 2. The comparator is negative persistence error, which assigns higher abnormality to smaller realized displacement over the same observed future segment. The left panel shows pair-count-weighted within-fold AUROC on 91 GAVD sources. The right panel uses all 90 external coded participants without target-domain fitting. Lines are conditional 95% paired group-bootstrap intervals; differences compare the same resamples.

## Results

### Current posture supplies useful short-horizon prediction

The current-posture transition has normal-source MSE 0.07174, compared with 0.12126 for persistence (Table 2). Its paired absolute improvement is 0.04952, with conditional 95% interval [0.03447, 0.06628], a relative reduction of 40.8%. This establishes useful prediction relative to the tested reference. It does not establish that the model has recovered a physiological state or needs the detailed order of past movement.

| Forecast | MSE x1,000 | 95% interval |
| --- | --- | --- |
| Persistence | 121.26 | [93.84, 149.90] |
| Damped velocity | 575.19 | [375.68, 828.96] |
| Own-leg history | 81.74 | [64.25, 100.33] |
| Bilateral history | 80.63 | [62.73, 99.97] |
| Current-posture transition (legacy) | 71.69 | [55.73, 88.66] |
| Current-posture transition | 71.74 | [55.71, 88.79] |
| Posture plus velocity | 76.93 | [60.96, 93.49] |
| Random-feature ridge | 105.20 | [85.17, 125.55] |
| Normal mean pose (post hoc) | 108.98 | [89.97, 130.25] |
| Mean displacement (post hoc) | 122.35 | [92.71, 152.75] |

Normal-source forecasting over the eight nominal future offsets, 0.1 through 0.8 seconds. Lower MSE is better. Units are squared prefix-torso units; values and conditional source intervals are multiplied by 1,000. The separately implemented legacy and current posture transitions are both retained.

Adding velocity and its support summaries gives MSE 0.07693. Its paired improvement over posture is -0.00519 [-0.01022, -0.00046], so this state does not earn its extra information here. Bilateral history also improves on persistence, but current posture is stronger. The bilateral advantage over own-leg history has an interval spanning zero. Both alternate partitions retain the direction of the posture comparison. These results make current posture a necessary reference for any claim about learned temporal organization. 

A model might improve on persistence merely by predicting an average normal pose at each future query. To check that simpler explanation, a separately declared post hoc diagnostic adds two normal-training intercept priors, without tuning: a mean pose for each future query and an anchor plus mean displacement. They have MSE 0.10898 and 0.12235, respectively. Posture improves on the mean-pose prior by 0.03724 [0.02539, 0.05150]. This supports the value of current positions beyond a constant prior within an outcome-exposed exploratory comparison.

### Structured masks do not improve the fixed task

Uniform masking gives MSE 0.07761, compared with 0.07962 for anatomical tubes and 0.07916 for motion balancing (the masking comparison). With improvement defined as uniform error minus candidate error, the differences are -0.00201 [-0.00426, -0.00002] and -0.00155 [-0.00311, -0.00013]. These exploratory comparisons do not support a benefit. The tube advantage over geometry-matched random groups is 0.00042 [-0.00086, 0.00173], which also does not establish an anatomical advantage. Every seed and policy remains in the supplementary registry.

![Forecasting with six prefix-corruption policies and an unmasked prefix baseline on the same 29 held-out normal source videos. Points are source-average MSE multiplied by 1,000; lines are conditional 95% source-bootstrap intervals. Predictions are averaged over three fixed seeds before errors are computed. The dashed line is the compact current-posture transition. Primary evaluation retains the full observed prefix. Structured masks do not improve on uniform masking under this contract.](figures/masking_results.svg)

Forecasting with six prefix-corruption policies and an unmasked prefix baseline on the same 29 held-out normal source videos. Points are source-average MSE multiplied by 1,000; lines are conditional 95% source-bootstrap intervals. Predictions are averaged over three fixed seeds before errors are computed. The dashed line is the compact current-posture transition. Primary evaluation retains the full observed prefix. Structured masks do not improve on uniform masking under this contract.

Shared prefix dropout, small coordinate noise, and opposite-leg time shifts do not reverse the tube or motion-balanced comparison with uniform masking. These perturbations test observation dependence rather than neurological interventions. Phase, learned-mask, latent-objective, and stochastic-future extensions remain deferred under the declared evidence gates. A broader search after these outcomes would be a new study.

### Forecast skill and anomaly ranking separate

The locked conditional score has within-fold AUROC 0.462, compared with 0.811 for low motion. The paired difference is -0.348 [-0.506, -0.191]. Matched posture calibration also performs weakly, and its paired comparison with the larger state does not establish an advantage. Thus the present normal reference does not recognize annotation-defined departures better than the locked movement comparator. Table 3 keeps the simple alternatives and other score definitions visible.

| Anomaly score | Within-fold | Pooled | Pooled AP |
| --- | --- | --- | --- |
| Conditional two-sided | 0.462 | 0.462 | 0.651 |
| Posture two-sided | 0.386 | 0.394 | 0.636 |
| Low motion | 0.811 | 0.806 | 0.879 |
| Conditional one-sided | 0.500 | 0.490 | 0.674 |
| Marginal two-sided | 0.556 | 0.545 | 0.688 |
| Static-posture distance | 0.404 | 0.397 | 0.646 |
| Motion/quality distance | 0.557 | 0.566 | 0.763 |
| Low prefix motion | 0.749 | 0.764 | 0.840 |
| Prefix missingness | 0.426 | 0.427 | 0.646 |
| Bilateral excess error | 0.774 | 0.793 | 0.865 |
| Posture/velocity excess error | 0.802 | 0.811 | 0.866 |

Internal rankings on 29 normal and 62 abnormal GAVD sources. Higher values are better. Within-fold AUROC weights each fold by its number of normal-abnormal pairs; pooled AUROC and average precision (AP) combine all sources. Abnormal prevalence is 0.681. Scores use different reference mechanisms as explained in the text.

The bilateral excess score has pooled AUROC 0.793, while low motion has 0.806. Their within-fold rankings differ because those compare only sources evaluated by the same fold model. The favorable excess-error association remains compatible with a substantial contribution from movement amount. This does not make low motion meaningless: restricted movement can itself matter clinically. It shows that the association alone does not demonstrate additional knowledge of movement organization.

Frozen transfer gives candidate AUROC 0.591 and low-motion AUROC 0.762, with paired difference -0.172 [-0.357, 0.009]. The interval includes zero, and the candidate does not establish improvement. Parkinson's and osteoarthritis slices follow the same direction and share normal participants. A pre-score audit found uncertain person continuity when an assistant overlaps one Parkinson's participant. The fixed exclusion sensitivity retains 89 participants and gives 0.597 versus 0.759, with difference -0.162 [-0.349, 0.020]. It removes the only retained severe-Parkinson's participant and therefore narrows severity coverage.

### Nuisance checks constrain interpretation

Matching within fold and annotated view on prefix movement and missingness retains 40 of 91 sources. Thirty-eight enter four folds with adequate matched class support. The paired candidate-minus-low-motion difference is -0.409 [-0.651, -0.161]. There is no distance caliper, so matching does not guarantee similar observations; pair distances and coverage loss remain visible. Most sources have multiple views, and single-view strata are small. These analyses cannot establish camera invariance or remove unmeasured recording differences.

External native frame rates are associated with labels even after sampling at 10 Hz. Age, reflective markers, clothing, scene, and full-image person selection also differ. Some prefix covariates leave the normal training range, and finite calibration tails can saturate. External ranking therefore measures transfer under documented observation shift rather than isolated disease physiology. A source-identification diagnostic on the normal training pool likewise supplies no evidence that learned features remove recording identity relative to random features.

## Discussion and scope

The positive forecasting result identifies a compact, reproducible capability: a normal-only model predicts relative leg articulation better than repeating the last pose. The strongest tested reference uses current posture rather than a long history. The benchmark also establishes why forecasting and abnormality ranking need separate validation. Neither a forecast-error reduction nor a favorable excess-error AUROC establishes an incremental anomaly detector.

Several explanations remain for the conditional score's performance. Movement amount may carry annotation information that the nuisance regression discounts. The normal cohort may be too small for a stable conditional reference. Disjoint training and calibration roles reduce fitting support, while coarse tails lose ranking detail. Out-of-fit and refitted forecasting models can also differ in error scale. The study does not identify which explanation dominates or rule out better conditional models with stronger independent normal coverage.

The masking finding is similarly bounded to one compact coordinate model and this sparse observation task. Anatomical or latent masking may help with other representations or larger populations. Here the policies do not earn an advantage over a matched uniform control. A useful next study should add independent normal coverage and validated observations that resolve these ambiguities before increasing model capacity.

GAVD supports a heterogeneous public-video benchmark, but GAVD alone does not establish patient-disjoint, site-disjoint, or prospective clinical generalization. Source IDs are not patient IDs, and folder labels are not independently confirmed diagnoses in this analysis. The external dataset adds participant-coded transfer evidence, but diagnoses and cross-dataset participant independence were not independently reverified. Byte hashes exclude identical files rather than reencoded copies or repeated people. Quality-dependent retention also changes the evaluated population.

For medical world models, the contribution is a tested movement-observation benchmark with useful forecasting and explicit anomaly controls. Short gait forecasts do not establish disease progression, treatment response, or action-conditioned patient simulation. Verified independent participants, comparable acquisition, and clinically grounded movement measurements are needed before a claimed organizational effect can support stronger medical use.

## References

See references.bib and the formatted bibliography in the PDF.

## Supplementary material

## Complete candidates and implementation

| Every declared forecast | MSE x1,000 | 95% interval |
| --- | --- | --- |
| Bilateral history | 80.63 | [62.73, 99.97] |
| Damped velocity | 575.19 | [375.68, 828.96] |
| Anatomical tubes | 79.62 | [62.21, 98.26] |
| Anatomical tubes (seed 27) | 82.54 | [64.42, 102.23] |
| Anatomical tubes (seed 41) | 81.48 | [63.92, 100.32] |
| Anatomical tubes (seed 73) | 84.44 | [65.26, 104.58] |
| Motion-balanced masking | 79.16 | [61.28, 98.76] |
| Motion-balanced masking (seed 27) | 82.93 | [64.59, 102.80] |
| Motion-balanced masking (seed 41) | 83.01 | [64.70, 102.81] |
| Motion-balanced masking (seed 73) | 82.22 | [63.68, 102.67] |
| Motion-focused masking | 77.75 | [60.31, 96.82] |
| Motion-focused masking (seed 27) | 82.76 | [64.76, 102.48] |
| Motion-focused masking (seed 41) | 79.92 | [62.56, 98.93] |
| Motion-focused masking (seed 73) | 82.17 | [63.73, 102.48] |
| No prefix masking | 78.28 | [59.59, 99.44] |
| No prefix masking (seed 27) | 83.66 | [63.75, 105.81] |
| No prefix masking (seed 41) | 80.60 | [61.70, 101.66] |
| No prefix masking (seed 73) | 81.41 | [61.82, 103.12] |
| Mask-pattern diagnostic | 138.46 | [111.44, 166.94] |
| Random joint groups | 80.04 | [62.41, 99.08] |
| Random joint groups (seed 27) | 85.00 | [66.01, 105.43] |
| Random joint groups (seed 41) | 81.67 | [64.02, 100.59] |
| Random joint groups (seed 73) | 82.75 | [64.45, 102.49] |
| Time blocks | 78.43 | [61.07, 97.03] |
| Time blocks (seed 27) | 83.22 | [64.52, 102.92] |
| Time blocks (seed 41) | 81.25 | [63.57, 100.14] |
| Time blocks (seed 73) | 81.65 | [63.26, 101.41] |
| Uniform masking | 77.61 | [60.12, 96.69] |
| Uniform masking (seed 27) | 83.12 | [64.14, 103.63] |
| Uniform masking (seed 41) | 79.12 | [61.70, 98.30] |
| Uniform masking (seed 73) | 82.57 | [64.15, 102.51] |
| Own-leg history | 81.74 | [64.25, 100.33] |
| Persistence | 121.26 | [93.84, 149.90] |
| Current-posture transition | 71.74 | [55.71, 88.79] |
| Posture plus velocity | 76.93 | [60.96, 93.49] |
| Shifted opposite history | 86.54 | [66.90, 107.95] |
| Current-posture transition (legacy) | 71.69 | [55.73, 88.66] |
| Random-feature ridge | 105.20 | [85.17, 125.55] |
| Normal mean pose (post hoc) | 108.98 | [89.97, 130.25] |
| Mean displacement (post hoc) | 122.35 | [92.71, 152.75] |

Complete candidate registry at the primary eight-query horizon, including all optimization seeds and the mask-pattern diagnostic. Lower values are better. Seeds do not multiply the 29 normal source groups.

The method identity is `medwm-reference-masking-v2`, and the fitted run is `reference-v2-core-v1`. Its request pins the code revision, nine implementation hashes, cohort identity and membership, protocol freeze, versions, seeds, and grids. Independent acceptance preceded new outer fits. GAVD remains exploratory after earlier outcome exposure. External preparation, transfer, identity sensitivity, robustness, and post hoc diagnostics have separate identities. Older predictions, checkpoints, drafts, and run identities are preserved.

The core records 2235 cached fits, including inner selection, cross-fitting, mask seeds, and alternate partitions. This is an execution count rather than independent evidence. The core took 323.5 seconds on the recorded Apple arm64 host with one heavy job. Fresh external extraction used 17,138 real pose calls and took 466.5 seconds. Timing depends on the host and caches. The ledger retains the pilot, failures, memory observations, updates, and checkpoint parameter counts.

The perceptron has 832 input features and two 32-unit GELU hidden layers. Its coordinate head starts at zero displacement. AdamW uses learning rate 0.001, weight decay 0.001, gradient clipping at 1, and hierarchical batches of sixteen sources. Seeds are 27, 41, and 73. Inner normal-source validation chooses 100 or 200 updates using the unmasked arm at seed 27, then all policies share that duration. Ridge strength is selected from 0.1, 1, 10, and 100. No transformation uses an outer abnormal outcome.

The exported mask-exposure table reports each selected outer model's actual hidden and eligible token counts for every bank. Across corruption policies, rounding produces aggregate realized rates from 40.024% to 40.034%; corresponding policies have identical counts. These repeated training banks are exposure records, not independent observations. The model-health table retains training-prefix feature variance, effective rank, update count, parameters, and runtime. Effective rank summarizes how variance is distributed across feature directions; its range of 5.59 to 11.59 in the 32-dimensional representations does not establish clinical usefulness or view invariance.

Current-posture ridge has 24 inputs: eight target-joint coordinate anchors and known-support indicators. The 64-feature posture-plus-velocity input adds median rates, median absolute deviations, and valid-pair counts. Both predict all future queries directly as residuals from the anchor. Future availability enters loss and evaluation only. The untrained encoder coordinate probe and mask-pattern diagnostic remain separate controls, not additional successful method families. The later mean-prior diagnostic is `reference-v2-mean-prior-posthoc-v1`. It minimizes the declared equal-leg masked coordinate objective using only each outer fold's normal training sources. Each observed coordinate receives its source/sequence/window weight divided by the observed-coordinate count of its leg. The weighted intercept is fitted separately for each query and coordinate, either directly or after subtracting the latest observed anchor. All held-out future coordinates can be poisoned without changing the fitted means or forecasts. Both untuned priors and their paired comparisons remain visible, with post hoc status.

## Calibration details and all rankings

Within each outer normal development set, three rotations choose calibration sources C and forecast-development sources R. A forecaster fitted on R predicts C and outer test sources. Three out-of-fit partitions within R produce errors for fitting the nuisance regression. Each such forecaster selects its regularization only within its own normal training pool. C never enters tuning or nuisance fitting. Exact memberships and hashes are stored for all fifteen components and the matched posture family.

The nuisance regression standardizes log movement and missingness within its fitting set. Leave-one-source-out selection chooses ridge strength from the four-value grid. For n calibration residuals, cumulative probability is (0.5 plus the number of smaller residuals plus half the number of ties) divided by n plus one. Endpoints remain finite, and extreme residuals can tie. Marginal calibration ranks log forecast MSE without nuisance conditioning. A one-sided arm ranks conditional residuals directly. Neither replaces the primary after outcomes.

With n separate exchangeable normal calibration sources, usual split-conformal resolution is one divided by n plus one. Fewer than nineteen cannot supply the usual finite 5% upper-tail cutoff. Our seven- or eight-source subsets are smaller, and the cross-fitted mixture has no automatic coverage guarantee. We therefore report no 95% clinical normality threshold. The floor is fixed at 0.00000001; notebooks inspect numerical floors and empirical tail resolution. A separately reviewed post hoc sensitivity, `reference-v2-floor-posthoc-v2`, refits only the nuisance reference at floors 0.0000000001, 0.00000001, and 0.000001. All 91 final source scores, pairwise ranking directions, and primary intervals remain unchanged. Its first replay attempt stopped when it omitted the producer's intermediate CSV write/read step. That omitted step changes coarse empirical scores at machine precision and can change exact ranking ties, so the failed attempt and corrected producer path are both preserved. The original primary scores are unchanged.

![Source-average persistence and bilateral-forecast MSE for all 91 GAVD source videos. Each point is one source; colors identify dataset annotations. The dashed line marks equal error. Both axes use logarithmic scales in squared prefix-torso units. The scatter motivates explicit movement controls and does not identify a physiological mechanism.](figures/motion_and_error.svg)

Source-average persistence and bilateral-forecast MSE for all 91 GAVD source videos. Each point is one source; colors identify dataset annotations. The dashed line marks equal error. Both axes use logarithmic scales in squared prefix-torso units. The scatter motivates explicit movement controls and does not identify a physiological mechanism.

The internal score file retains every raw error, historical-direction excess score, nuisance baseline, and calibrated arm. Table 3 does not choose a new favorable direction after outcomes. Its average precision uses abnormal prevalence 62/91. Per-fold counts, equal-fold averaging, condition profiles, seeds, and the four-query horizon remain in the source evidence. Conditions share the same normal reference.

| External score | AUROC | 95% interval | AP |
| --- | --- | --- | --- |
| Conditional two-sided | 0.591 | [0.466, 0.703] | 0.790 |
| Posture two-sided | 0.528 | [0.407, 0.641] | 0.772 |
| Low motion | 0.762 | [0.663, 0.854] | 0.909 |
| Conditional one-sided | 0.638 | [0.514, 0.750] | 0.807 |
| Marginal two-sided | 0.404 | [0.286, 0.524] | 0.654 |
| Forecast MSE | 0.443 | [0.322, 0.564] | 0.727 |
| Forecast minus persistence | 0.740 | [0.605, 0.858] | 0.840 |
| Low prefix motion | 0.779 | [0.665, 0.877] | 0.886 |
| Prefix missingness | 0.586 | [0.449, 0.717] | 0.743 |

Every declared external score on 27 normal and 63 abnormal coded participants. Higher AUROC and AP are better. These marginal intervals use 2,000 participant bootstrap draws within label strata and condition on fixed GAVD models. The principal paired contrast uses 5,000 draws. Abnormal prevalence is 0.700.

The external abnormal endpoint combines Parkinson's and osteoarthritis. It therefore measures dataset-defined abnormality rather than neurological specificity. Averaging fifteen source-trained calibration components does not create fifteen independent validation cohorts. Bootstrap uncertainty resamples coded participants conditional on fixed models. Raw and excess controls use the same component predictions and targets.

## Physiology, robustness, and observation quality

The timing control moves coordinate-validity pairs together and preserves the current anchor. A fixed global feature permutation would be inadequate because linear regression can absorb it. A separately fitted window-specific shift and an inference shift are therefore distinct controls. Any effect reflects dependence within this representation; it is not an estimate of basal-ganglia function, muscle modules, or selective motor control.

Dropout removes 40% of eligible non-protected prefix tokens. Coordinate noise has standard deviation 0.02 prefix-torso units. Patterns are fixed by window ID and shared across models; future targets and support stay unchanged. Opposite-leg stress changes one leg's past when predicting the other, preserving its latest observed anchor. The source table retains all four conditions and seven models, including observed-prefix replays.

The source-identification diagnostic uses thirteen normal sources in outer fold zero's training pool with at least two annotated sequences. A fixed regularized classifier learns identity from the first sequence and evaluates the others. The forecaster already used these sources, and sequences may overlap in physical frames. This is a recording-information diagnostic rather than biometric validation or new-person generalization. Random encoder features are retained; the learned features show no source-invariance advantage.

Original GAVD videos remain under `data/videos-gavd`, with verified paths reused without duplication. Source and sequence IDs, media hashes, frame origins, decoder times, joint order, dimensions, confidence, and original missingness remain explicit. External originals stay in their existing owner-dataset cache. The external pipeline shares the quality contract except the unavailable GAVD person-box gate. Across retained GAVD windows, the actual final offset ranges from 0.767 to 0.834 seconds, and the maximum query-grid deviation is 0.0412 seconds. External query offsets match the nominal grid to floating-point precision. The sampling summary retains all gap and deviation definitions. Full-image selection can follow an assistant, motivating the whole-participant sensitivity rather than selective deletion of inconvenient frames.

A partially blinded postfit audit inspects thirty selected stills from ten of the ninety-six candidate GAVD sources, comparing accepted and rejected material. It records crowding, occlusion, and unresolved identity cases rather than verifying continuous tracking. External inspection covers 282 selected stills from 94 decodable coded participants before efficacy scoring. Stills cannot verify every frame or a diagnosis. Uncertain overlap for `PD_SV_001` stays in the primary population and is excluded only in the predeclared sensitivity. Both population sizes remain visible.

## Reproduction and interpretation record

Notebooks 20 through 26 rebuild the inventory, reference scores, mask invariants, fitting evidence, grouped and external endpoints, robustness, and claims. Each initializes its own paths and kernel. A fresh Ambient call teaches extraction, while full experiments reuse immutable caches. A small real normal-only teaching fit is labeled separately from the research run. Missing prerequisites cause explicit failure rather than synthetic performance.

The builder reads only completed, hash-verified runs. Editable prose is `docs/draft03/manuscript_source.md`; its separate generator is `scripts/build_next_draft03.py`. The older compatibility command continues to write preserved draft02. Every figure has source data or a selection rule. The ledger connects claims to runs and uncertainty methods. Generated Markdown, LaTeX, PDF, references, vector figures, build records, slides, and review identify the exact checked artifacts.

The supported contribution is a controlled benchmark showing that useful compact forecasting can coexist with an anomaly score that fails its strongest simple reference. Improved calibration, latent prediction, phase conditioning, and stochastic futures require new declared experiments with adequate independent support. They are not results of this study.
