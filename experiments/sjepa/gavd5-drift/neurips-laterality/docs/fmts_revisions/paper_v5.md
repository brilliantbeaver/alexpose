# Hidden-Feature Matching and Movement Estimates Diverge in a Gait JEPA

## Abstract

We evaluate whether learning to predict hidden skeleton features improves access to a movement quantity in human gait. A small skeleton Joint-Embedding Predictive Architecture (JEPA) is trained without movement labels, then frozen while a separate regression estimates a signed left–right speed contrast. In source-video cross-validation on 625 clips from 93 videos, trained predictors more consistently favor hidden features from the correct clip over another video's features. With the same expanded summary and regression procedure, however, mean source-balanced $R^2$ falls from 0.22 at initialization to 0.10–0.11 after training. Motion and region masks leave their advantage over matched random masks uncertain. This focused evaluation shows why the success of latent prediction should be assessed alongside the usefulness of features for a specified observable of an evolving system. The measured gap concerns whole-clip gait representations; future movement prediction remains untested on real data.

## Introduction

Walking is an evolving system: coupled body landmarks change position over time, and movement depends on their trajectories. Gait rhythm and coordination motivate temporal modeling, although video-derived coordinates describe only part of the physical process. Temporal world modeling aims to represent such a system well enough to predict its observable behavior. We test access to one movement measurement within that larger objective.

We study that question with S-JEPA [@sjepa], using a left–right contrast of median coordinate speeds. Our central null is no improvement from JEPA training over matched initial weights for this readout on held-out source videos. The no-improvement null also motivated the registered reflection study. The expanded readout and mask questions developed on the same cohort and remain exploratory.

The contribution is a domain-specific evaluation relevant to temporal foundation models: success on latent prediction needs to be considered alongside access to observable movement. Scaling, broad transfer, persistent temporal state and gait simulation remain outside the demonstrated scope.

## Study design

**Data and observable.** Quality checks retain 625 GAVD clips from 93 source videos [@gavd]. Each archive has shape $T_i\times33\times4$: three estimated coordinates plus landmark visibility, with original frame numbers and frame rate. Coordinate units are image normalized with inferred depth; they are not calibrated physical speeds.

The movement path uses observed coordinates, pelvis centering and a body-width scale, without gap filling or resizing. For five pairs—shoulders, knees, ankles, heels and foot tips—we calculate median speeds $m_{L,k},m_{R,k}$ using original timestamp differences. Both sides must be observed at both ends of at least eight common transitions for every pair. The signed target is

$$y=\frac{1}{5}\sum_{k=1}^{5}\frac{m_{L,k}-m_{R,k}}{m_{L,k}+m_{R,k}+10^{-8}}.$$

Anatomical reflection exchanges left/right landmarks and validity flags and flips the centered horizontal coordinate, giving $y(Mx)=-y(x)$. Hips define the pelvis and are excluded from the contrast because centering equalizes their speed magnitudes. The statistic describes relative median speed, without measuring phase, stride timing or loading. Reversing a trajectory with its sampling intervals preserves speed magnitudes, so recovering this target alone cannot demonstrate learned temporal dynamics.

**Encoder input.** A landmark is observed when visibility is at least 0.45 and all coordinates are finite. A separate encoder path fills interior gaps of at most four samples, treating filled positions as valid. After pelvis centering and body-width normalization, coordinates and validity are linearly resized to 64 positions by sequence index. Original time intervals are discarded. Resized validity must reach 0.999; invalid coordinates are zero. Each joint's four-position patch projects 12 coordinates to 96 features, giving a $16\times33$ grid. A patch is valid only if all four positions are valid. Anatomical identities and prepared positions survive this operation; the original trajectory, duration and cadence need not.

![Completed model. Poses and tokens are schematic: skeleton colors mark anatomical left/right; token colors mark visible (teal), hidden (orange) and missing (gray) positions. Scattered/motion-weighted and connected-region masks may leave context on both sides. Full clips reach the teacher and regularizer; target y supervises the separate readout.](assets/v5/figures/training_pipeline_compact.svg)

**Training.** The grid uses 1,200 AdamW updates per encoder (architecture and settings in Appendix A). The online encoder zeroes hidden patch content before adding joint and position embeddings; the predictor substitutes learned mask tokens at target positions. Cross-entropy matches centered teacher feature distributions at hidden valid positions, averaged within clips and then across clips. A teacher follows the online weights by exponential moving average without gradients. A full-view VICReg term [@vicreg], weighted 0.05, pools twelve gait joints, including hips, across two geometric views and passes them through a projector. Thus the total objective also accesses complete clips. Training uses CUDA BF16 and evaluation FP32.

We compare three motion conditions and two region conditions across five folds and five seeds: 125 encoders. The motion conditions use random sampling, a MAMP-style sampler [@mamp], and a robust motion mixture. Motion masks sample across all 33 joints using a count derived from the twelve-joint gait pool. Connected regions hide six connected landmarks over eight of sixteen blocks, with a separate count-matched random reference. Each comparison matches initial weights, source draws, views and update counts. This grid uses neither reflection augmentation nor an explicit reflection loss (Appendix B distinguishes their evidence).

**Held-out readout.** Five outer folds keep each video's clips and derived views together. Fresh encoders fit only outer-training videos with source-balanced sampling. All overlapping windows, mirrors and augmentations inherit the source split. The encoder is frozen while ridge regression, a linear model with a coefficient penalty, estimates $y$. Three inner source folds select the penalty using only outer-training labels; imputation and scaling are fitted within each inner training set and then refitted with the regression on all outer-training sources. Inner-validation clips may have contributed without labels to outer-training encoder fitting.

Both encoders use bilateral mean features (960 inputs), expanded with standard deviations, absolute adjacent-block feature changes and ten validity-support fractions (2,890 inputs). We pool outer-test predictions within each seed, weighting each video equally, then average the five $R^2$ values. Paired 95% source-bootstrap intervals resample videos with their clips and seed predictions together (2,000 resamples). They condition on the fitted models and existing splits, excluding uncertainty from retraining, alternative partitions and development choices. Repeated seeds and mask draws do not add independent videos.

## Findings

**Movement readout falls after training.** The expanded initial representation reaches mean $R^2=0.223$, compared with $0.101$–$0.114$ for trained teachers (Figure 2). All five conditions decrease in every seed, with greater mean absolute error. Paired trained-minus-initial intervals are below zero in every condition. Final online encoders also score below their matched initial encoders.

| Training mask | Mean $R^2$ | $\Delta R^2$ | 95% source interval |
|:--|--:|--:|:--|
| Random (motion) | 0.114 | -0.109 | [-0.170, -0.041] |
| MAMP-style | 0.113 | -0.110 | [-0.165, -0.051] |
| Motion mixture | 0.114 | -0.108 | [-0.171, -0.042] |
| Random (region) | 0.109 | -0.113 | [-0.166, -0.060] |
| Connected region | 0.101 | -0.122 | [-0.183, -0.055] |

Table 1: trained teacher minus matched initialization. Intervals resample source videos, conditional on fitted models; differences are calculated before rounding.

Expanding the initial readout raises $R^2$ from 0.071 to 0.223 ($+0.152$). The added components mix temporal variation, absolute increments and observation support; standard deviation discards order, and absolute increments discard direction. Their combined gain therefore does not isolate sensitivity to temporal order. The initial control retains the pretrained pose detector, input preparation, anatomical identities and supervised regression. 

![Different outcomes from the same fitted models. A: frozen movement readout; points are five seed scores, and short vertical marks their mean. B: correct-clip feature error is lower in 375 of 375 trained diagnostic checks, versus 33 of 75 distinct initial checks. These checks reuse models and videos.](assets/v5/figures/learning_results.svg)

**Correct-clip features become easier to match.** Each diagnostic compares source-weighted mean squared feature error against the correct clip and valid targets from another source, holding the evaluated clips and hidden positions fixed. All 375 trained checks favor correct targets; 33 of 75 initial checks do so. The trained checks cover 125 fitted models under three evaluation masks (left-leg gap, right-leg gap and scattered gap). Initial checks cover 25 matched fold/seed controls under the same three masks, with copies shared between experiment families counted once. Visible context and full-clip teacher targets share pose, viewpoint and missingness, as well as motion. Those cues can support correct-clip matching without isolating hidden movement. Teachers and target distributions change during training, so raw training losses cannot rank representation quality across conditions.

**Changing mask geometry does not establish better readout.** The inspection draws confirm that motion samplers select higher-motion targets and that connected regions remove immediate temporal brackets. Motion masks hide approximately 17% of valid all-landmark tokens; regions hide 9.9%. Their readout comparisons therefore use distinct random references. All three mask-comparison intervals span zero (Appendix B). Their benefits remain uncertain; the intervals do not establish equivalence.

**Preparing input changes the measurement.** A saved comparison recalculates the target after input preparation: 623 clips from 92 sources have finite values on both paths, with source-weighted sign agreement 70.4% and calculation-agreement $R^2=0.218$. This is agreement between two calculations of the observable, not a learned model score. Neither the achievable readout accuracy nor an individual preprocessing cause follows from it.

## Interpretation and next experiment

The predictor favors the correct clip more consistently after training, while the same regression procedure estimates its left–right speed contrast less accurately from trained features. Poorer readout performance does not imply complete destruction of movement information. In particular, 49/125 teacher fits and 96/125 online fits select the largest ridge penalty, 10,000. Wider training-only selection, support-only controls and summaries of matched dimension would help separate feature geometry from readout limitations. Source holdout does not establish participant independence, chronological holdout or an untouched development cohort. Recording and pose-estimation cues remain possible explanations.

A next experiment would decode future coordinates from past-only inputs at common valid endpoints and physical-time horizons. Fit and select decoders on training sources only; compare last-position, constant-velocity, direct-pose, initial-encoder and mismatched-training-future controls. Future coordinates and visibility must not change prefix preparation, normalization, masks or predictions. The retained past-feature readout and future-feature decoding studies have synthetic training evidence; the real-data horizon census establishes eligibility only. Decoding observed future-teacher features diagnoses the decoder; decoding features predicted from the past evaluates forecasts. Multiple horizons from one prefix are not recursive rollouts.

Forecasting and simulation require tests of future behavior beyond this whole-clip result.

## Related work

Prediction targets and available context influence which variation a representation retains. Skeleton methods predict learned features [@sjepa] or explicit motion [@mamp]; connected-body masks reduce nearby cues [@slim]. Action recognition evaluates the resulting features for labels, while video anticipation and action-conditioned planning assess other behavior [@vjepa2]. Work separating invariant and transformation-sensitive features further exposes this dependence on the task [@seqjepa]. Our gait contrast adds an observable-specific test with matched initial weights. We borrow masking ideas without comparing the complete methods.

## References

References are generated from the version-specific bibliography using numeric NeurIPS citations.

## Appendix A. Measurement and implementation details

Of 666 selected annotations, 642 match pose archives and 625 pass quality checks. The motion target count is half the batch's smallest valid twelve-joint gait pool, rounded down, then sampled across all 33 joints.

The completed recipe uses 96 features, encoder/predictor depths 4/2, four attention heads, batch size 20, 1,200 AdamW updates, learning rate 0.001, weight decay 0.05 and teacher momentum 0.999.

The archive coordinates are $x=(x_0+uW_c)/W$, $y=(y_0+vH_c)/H$, and $z=dW_c/W$, where crop origin and size are $(x_0,y_0),W_c,H_c$, frame size is $W,H$, and $u,v,d$ are pose estimates. Unequal image-axis scaling and inferred depth remain after body normalization. The scale is the median of valid shoulder and hip widths. The movement path requires observed hips; the encoder uses the median valid pelvis midpoint for missing hips (zero if none is available). No boundary extrapolation fills gaps.

The teacher receives the unaugmented prepared clip; the masked online path receives a geometric view. Two full online views use rotations up to 8 degrees in the horizontal-depth plane and translations up to 0.03 in image-plane coordinates. Target/predictor temperatures are 0.06/0.10; the teacher center uses momentum 0.9. Gradient norm is clipped at 1. The twelve-joint regularizer pool is shoulders, hips, knees, ankles, heels and foot tips. Its projector applies two 96-to-96 linear layers separated by GELU, a smooth activation. It combines invariance, variance and covariance penalties with internal weights 25, 25 and 1, then overall weight 0.05.

The expanded summary uses common left/right validity within each pair; increments require consecutive common valid blocks. Each pair contributes differences and sums of means, standard deviations and mean absolute feature increments (six 96-vectors), plus two support fractions. Thus $5\times6\times96+10=2,890$. Missing support gives zero summary values. Temporal positions in this summary are prepared block indices, not original timestamps. Ridge penalties are selected from $0.01,0.1,1,10,100,1000,10000$.

Source-balanced $R^2$ uses weights $w_{vi}=1/(Vn_v)$ and weighted test mean $\bar y_w$:

$$R_w^2=1-\frac{\sum_{vi}w_{vi}(y_{vi}-\hat y_{vi})^2}{\sum_{vi}w_{vi}(y_{vi}-\bar y_w)^2}.$$

The test mean defines this metric's denominator, not a deployable fitted baseline. Outer test folds contain 19, 19, 19, 18 and 18 videos and 189, 182, 72, 77 and 105 clips. The retained provenance audit checked archive IDs and hashes, source membership and absence of test-source training draws. Those checks concern file integrity and split adherence, not cross-upload content deduplication. Persistent person IDs and complete annotation retrieval/version records are unavailable, preventing a person-level contamination audit or full reconstruction of acquisition history. Source holdout alone provides no evidence of generalization under distribution shift.

## Appendix B. Supporting results and reproducibility

MAMP-style, motion-mixture and region masks minus their own random references give $\Delta R^2=-0.0008$, $+0.0005$ and $-0.009$, with intervals $[-0.020,0.015]$, $[-0.015,0.015]$ and $[-0.033,0.018]$. A 264-value coordinate/validity summary reaches $R^2=0.035$; it is a limited direct-pose baseline, not a ceiling on coordinate information.

These exploratory percentile intervals use bootstrap seed 812, retain all five fitted seeds and have no familywise or development-selection adjustment. The numerical supplement preserves aggregate seed scores and recorded intervals. The executed report identifies 125,000 prediction rows and 200 pooled score rows; a saved audit records their recomputation. Raw predictions and checkpoints are unavailable in the revision environment, so this revision checks aggregate consistency without claiming to repeat that bootstrap or inference.

A separate reflection-augmentation comparison has real-data summary evidence: augmentation reduces the specified teacher-token reflection discrepancy by $\Delta q=-0.008$ $[-0.010,-0.007]$, while its readout difference is $\Delta R^2=0.004$ $[-0.006,0.013]$. This uses its own recipe, four inner readout folds and compact summary; it is not pooled with the motion grid. An explicit reflection loss has synthetic evidence only. Likewise, a readout forced to take the form $g(x)=[h(x)-h(Mx)]/2$ reverses sign algebraically without establishing learned movement information.

Clinical gait asymmetry motivates the observable [@stroke], but GAVD annotations identify neither affected side nor treatment response. Uncalibrated coordinates and absent clinical reference measurements preclude a clinical interpretation of $y$. Public annotation access does not authorize redistribution of separately hosted videos or derived poses; neither is included in the supplement.

**AI assistance.** An AI assistant supported author-side evidence reconciliation, literature checking, manuscript revision, figure programming and adversarial checks. No new model training was performed for this revision. Responsibility for the submitted text, figures, claims and references remains with the authors.
