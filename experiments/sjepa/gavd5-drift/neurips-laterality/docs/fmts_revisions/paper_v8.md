# S-JEPA Training Improves Same-Clip Matching but Worsens a Left-Right Pose-Speed Readout

## Abstract

We ask whether training a skeleton Joint-Embedding Predictive Architecture (S-JEPA) makes it easier to predict a score that contrasts left- and right-side pose-coordinate speeds. We compute the score from original timestamps in 625 clips from 93 GAVD source videos. We train 125 models across five masks, five source-video folds, and five seeds. For each run, we freeze the final teacher and matched initialization, fit a ridge readout for each on training sources, and evaluate both on held-out sources. Training lowers mean source-balanced $R^2$ from 0.223 to 0.101-0.114; all five conditions decline in all five seeds. A separate diagnostic favors the matching clip in all 375 trained comparisons, compared with 33 of 75 distinct comparisons at initialization. These comparisons reuse models and videos, so they are not independent trials. Even so, the evaluations move in opposite directions: same-clip preference becomes more consistent while speed-score prediction becomes less accurate.

## Introduction

Masked prediction lets temporal models learn from sequences without behavior labels. S-JEPA predicts teacher features for hidden parts of a sequence from visible context. That objective measures feature matching, not whether a later readout can estimate a chosen movement quantity. We test that distinction in gait.

Walking must be measured over time because a single pose does not show how body landmarks move. Pose sequences remain incomplete: camera angle and estimation error affect the coordinates, while forces and foot contact are unobserved. We study a narrow quantity available from the poses themselves, a signed contrast between left- and right-side coordinate speeds. The score is not a clinical measure of gait asymmetry.

We compare two evaluations of the same S-JEPA models [1]. The first asks whether predictions from visible pose patches are closer to hidden teacher features from the same clip than to features from another video. The second freezes the encoder and uses ridge regression to predict the left-right speed score. Each trained encoder is paired with its own initialization, and whole source videos are held out. After training, every recorded diagnostic comparison favors the matching clip, yet mean $R^2$ for the speed score falls by 0.108-0.122 across mask conditions.

Masked skeleton models are usually judged by action recognition [1] [4] [5]. Video anticipation and predictive world models instead assess future behavior [6] [7]. Our study uses S-JEPA as a controlled case and tests a continuous movement readout before and after training. The finding is specific to this dataset, model, target, and readout, but it shows why feature matching and downstream measurement should be evaluated separately.

## Study design

### Left-right speed score

After quality checks, the dataset contains 625 clips from 93 GAVD source videos [2]. Each clip provides 33 pose landmarks with three estimated coordinates, visibility, frame numbers, and frame rate. The coordinates use image normalization and inferred depth rather than calibrated physical units. For the target calculation, we keep the original timestamps and gaps, center each observed pose at the pelvis, and divide the coordinates by body width.

We use five left-right landmark pairs: shoulders, knees, ankles, heels, and foot tips. Each pair must have at least eight transitions for which both sides are observed at both endpoints. For each valid transition, we divide a landmark's Euclidean displacement in the three normalized coordinates by the elapsed time. Let $m_{L,k}$ and $m_{R,k}$ be the median left- and right-side speeds for pair $k$. We define

$$y=\frac{1}{5}\sum_{k=1}^{5}\frac{m_{L,k}-m_{R,k}}{m_{L,k}+m_{R,k}+10^{-8}}.$$

Positive scores mean that the left-side landmarks have higher median pose-coordinate speeds on average; negative scores mean the opposite. Anatomical reflection changes the sign. Reversing the order of a sampled trajectory and its time intervals leaves the score unchanged because the calculation uses speed magnitudes. The score therefore does not test temporal direction, gait phase, stride timing, or loading.

### S-JEPA training

The model input follows a separate preparation path (Figure 1). It fills interior gaps of up to four samples, centers coordinates at the pelvis, scales them by body width, and resamples each clip to 64 equally spaced sequence positions. When the pelvis is missing, this path uses a fallback estimate; unlike the target path, it does not give the encoder original elapsed times or clip duration. Four positions from one joint form a patch, producing $16\times33$ tokens; invalid patches are masked out. Appendix A gives the visibility rules and tensor dimensions.

S-JEPA uses visible patches to predict teacher features at hidden positions. The teacher sees the complete prepared clip, and its weights follow an exponential moving average of the online encoder. A VICReg term [3] also makes pooled features from two geometric views agree while discouraging collapsed or redundant features. Neither the speed score nor gait-category labels enter training.

We test five mask conditions. The motion family uses random targets, MAMP-style motion-weighted targets [4], or a mixture of random and motion-weighted targets. The region family samples targets either uniformly or from connected body landmarks. Each structured mask has a count-matched random control with the same initialization, data draws, views, and number of updates. Across five outer folds and five seeds, this gives 125 trained models, each run for 1,200 updates. This grid does not use reflection augmentation or a reflection loss.

![The target and encoder follow separate paths. The left branch computes the left-right speed score from observed transitions and original timestamps. The center branch fills short gaps and resizes the clip for S-JEPA training. The right branch freezes the initialized encoder or trained teacher, fits separate ridge readouts by the same procedure on outer-training sources, and evaluates them on held-out sources.](assets/v8/figures/training_pipeline_compact.svg)

### Evaluation against matched initialization

Five outer folds keep all clips and derived versions from a source video together. For each fold, the encoder trains on the other four folds only; we then freeze its initialization and trained weights. Within the outer-training data, three source-separated inner folds choose the ridge penalty. Imputation and scaling are fitted only on each inner-training partition and then refitted on all outer-training sources. No outer-test clip enters encoder training, readout fitting, or model selection. An inner-validation clip may still have contributed without labels to encoder training.

The expanded readout used for the main results has 2,890 inputs: feature means, standard deviations, absolute changes between consecutive blocks, and ten observation-support fractions. For each seed, we pool predictions from all five test folds, give each source video equal total weight, and compute one $R^2$ before averaging across seeds. The paired 95% intervals were computed from 2,000 source-video resamples while keeping fitted models and splits fixed. They do not include retraining, new splits, or uncertainty from analysis choices.

## Results

### Left-right speed prediction falls after training

With the initialized encoder, the ridge readout reaches mean source-balanced $R^2=0.223$. Across the five mask conditions, trained teachers average 0.101-0.114, a drop of 0.108-0.122 (Table 1). Each condition is below its matched initialization in all five seeds, and every condition-level interval is below zero. Mean absolute error rises from 0.0415 to 0.0436-0.0440. The final online encoders also have lower mean $R^2$ than their initial weights in every condition.

| Training mask | Trained teacher $R^2$ | $\Delta R^2$ | 95% source-bootstrap interval |
|:--|--:|--:|:--|
| Random (motion) | 0.114 | -0.109 | [-0.170, -0.041] |
| MAMP-style | 0.113 | -0.110 | [-0.165, -0.051] |
| Motion mixture | 0.114 | -0.108 | [-0.171, -0.042] |
| Random (region) | 0.109 | -0.113 | [-0.166, -0.060] |
| Connected region | 0.101 | -0.122 | [-0.183, -0.055] |

Table 1: Mean results across five seeds. Each row compares a trained teacher with its matched initialized encoder. We pool held-out predictions across the five folds within each seed before computing source-balanced $R^2$. Differences are calculated before rounding. The intervals were computed by resampling sources while holding fitted models fixed.

### Trained models favor the matching clip

The same-clip diagnostic asks a different question. With visible context and hidden positions fixed, we compare prediction error against teacher features from the matching clip and from another source video. Prediction error is lower for the matching clip in all 375 trained comparisons. At initialization, the matching clip has lower error in 33 of 75 unique comparisons; the duplicated initial control shared across the two mask experiments is counted once (Figure 2). Because these comparisons reuse videos and models, they are not independent trials. Shared pose, viewpoint, or missingness can provide the cue, and raw errors cannot be compared across trained models because each has its own teacher space and scale.

![Panel A shows source-balanced left-right speed scores for the initialized encoder and final trained teacher encoders; dots are seed scores and ticks are means. Panel B shows the same-clip diagnostic: 375/375 trained comparisons and 33/75 distinct initialization comparisons favor the matching clip. The diagnostic rows reuse videos and models.](assets/v8/figures/learning_results.svg)

### Secondary checks do not identify a cause

At initialization, the expanded readout raises $R^2$ from 0.071 for mean features alone to 0.223. It adds variation, feature-change, and observation-support summaries together, so the gain cannot be assigned to temporal order. The baseline also includes a pretrained pose detector, input preparation, joint identities, and supervised ridge regression.

No structured mask has a clear readout advantage over its count-matched random control. The MAMP-style, motion-mixture, and connected-region effects are -0.0008, +0.0005, and -0.009 in $R^2$; all three intervals include positive and negative values. The mask effects remain uncertain.

We also apply the speed formula to the prepared encoder input at uniform relative time steps. On the 623 clips from 92 sources with finite values on both paths, equal-source weighting gives 70.4% sign agreement and $R^2=0.218$ between the calculations. No regression is fitted, and no uncertainty interval was computed. Because timing, interpolation, normalization, and valid support all change, this check neither identifies a cause nor sets a performance ceiling.

## Discussion

The same-clip diagnostic can use any cue that distinguishes one clip from another, including pose, viewpoint, or missingness. The readout instead needs variation aligned with the left-right speed score. Training may strengthen the first set of cues without helping the second. We cannot identify the mechanism, but the two metrics should not be treated as substitutes.

The tested ridge readout may miss information available to another model or feature summary, and its largest penalty is selected in many fits. The encoder-prepared and timestamped scores also agree only modestly. Source-video splits keep related clips together, but they do not ensure participant independence; the data lack person identifiers, a chronological split, and an external cohort, and the target and expanded readout were developed on this cohort. The score is not clinically validated. We verified aggregate arithmetic, but missing raw predictions and checkpoints prevented us from rerunning inference or the bootstrap. Together, these limits prevent us from claiming that S-JEPA training generally removes movement information.

For FMTS, this is an evaluation result rather than a claim about foundation-model scale. Temporal models should be tested on the quantities they are expected to support, not only on self-supervised objectives. Our masked task can use context from both sides of a hidden patch, so it does not test forecasting. A direct follow-up would predict future coordinates from a prefix in physical time and compare against last-position, constant-velocity, direct-pose, and initialized-encoder baselines. We have not evaluated forecasting or recursive rollout on real gait data. Here, S-JEPA training makes same-clip matching more consistent but reduces ridge-readout accuracy for the left-right speed score. The two metrics should be reported separately.

## References

[1] M. Abdelfattah and A. Alahi. [S-JEPA: A joint embedding predictive architecture for skeletal action recognition](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04755.pdf). *Computer Vision - ECCV 2024*:367-384, 2024.

[2] R. Ranjan, D. Ahmedt-Aristizabal, M. A. Armin, and J. Kim. [Computer vision for clinical gait analysis: A gait abnormality video dataset](https://doi.org/10.1109/ACCESS.2025.3545787). *IEEE Access*, 13:45321-45339, 2025.

[3] A. Bardes, J. Ponce, and Y. LeCun. [VICReg: Variance-invariance-covariance regularization for self-supervised learning](https://arxiv.org/abs/2105.04906). *International Conference on Learning Representations*, 2022.

[4] Y. Mao, J. Deng, W. Zhou, Y. Fang, W. Ouyang, and H. Li. [Masked motion predictors are strong 3D action representation learners](https://arxiv.org/abs/2308.07092). *Proceedings of the IEEE/CVF International Conference on Computer Vision*:10181-10191, 2023.

[5] J. Do, Y. Chen, G. Youk, and M. Kim. [Less is more: Compact-token masked feature prediction for skeleton representation learning](https://arxiv.org/abs/2603.10648v3). *arXiv preprint arXiv:2603.10648v3*, 2026.

[6] M. Assran, A. Bardes, D. Fan, Q. Garrido, R. Howes, M. Komeili, M. Muckley, A. Rizvi, C. Roberts, K. Sinha, A. Zholus, S. Arnaud, A. Gejji, A. Martin, F. R. Hogan, D. Dugas, P. Bojanowski, V. Khalidov, P. Labatut, F. Massa, M. Szafraniec, K. Krishnakumar, Y. Li, X. Ma, S. Chandar, F. Meier, Y. LeCun, M. Rabbat, and N. Ballas. [V-JEPA 2: Self-supervised video models enable understanding, prediction and planning](https://arxiv.org/abs/2506.09985). *arXiv preprint arXiv:2506.09985*, 2025.

[7] H. Ghaemi, E. B. Muller, and S. Bakhtiari. [seq-JEPA: Autoregressive predictive learning of invariant-equivariant world models](https://proceedings.neurips.cc/paper_files/paper/2025/hash/2f63d2963526bdd9ff1b8bcc2dc9905a-Abstract-Conference.html). *Advances in Neural Information Processing Systems*, 38, 2025.

[8] K. K. Patterson, I. Parafianowicz, C. J. Danells, V. Closson, M. C. Verrier, W. R. Staines, S. E. Black, and W. E. McIlroy. [Gait asymmetry in community-ambulating stroke survivors](https://doi.org/10.1016/j.apmr.2007.08.142). *Archives of Physical Medicine and Rehabilitation*, 89:304-310, 2008.

## Appendix A. Implementation and evidence details

**Data and input preparation.** The dataset starts with 666 selected annotations; 642 have matching pose archives and 625 pass quality checks. Archive coordinates are $p_x=(x_0+uW_c)/W$, $p_y=(y_0+vH_c)/H$, and $p_z=dW_c/W$, where $(x_0,y_0)$ is the crop origin, $(W_c,H_c)$ is the crop size, $(W,H)$ is the frame size, and $(u,v,d)$ is the pose estimate. Body scale is the median valid shoulder or hip width, with a fallback of 1.

The target requires observed hips. The encoder path replaces a missing pelvis with the clip's median valid pelvis midpoint, or zero if none exists. It fills a short gap only when observed endpoints surround it. A point is observed when visibility is at least 0.45 and its coordinates are finite; after resizing, interpolated validity must be at least 0.999. Invalid coordinates are set to zero, and all four positions in a patch must be valid for its token to be used. Each $T_i\times33\times4$ input becomes $16\times33$ tokens of width 96.

**Training settings.** The width-96 encoder has depth 4; the predictor has depth 2. Both use four heads and batch size 20. AdamW uses learning rate 0.001, weight decay 0.05, gradient clipping at 1, and teacher moving-average rate 0.999. Training uses CUDA BF16 and evaluation uses FP32. Geometric views rotate by up to 8 degrees and translate by up to 0.03; the teacher sees the unmodified input. Teacher and predictor temperatures are 0.06 and 0.10, with center rate 0.9. The width-96 VICReg projector has two layers and loss weights 25, 25, and 1; its total-loss multiplier is 0.05.

For each batch, the motion-target count is half the smallest number of valid tokens among the twelve gait joints, rounded down; targets may still come from all 33 joints. MAMP-style weights use temperature 0.8. The mixture uses 25% uniform and 75% motion weighting, capped at the 95th percentile of positive scores.

**Readout and source weighting.** Means and standard deviations use blocks in which both members of a pair are valid; feature changes require consecutive common-valid blocks. Missing summaries are zero. Each pair contributes six 96-value vectors: left-right differences and sums for three summaries, plus two support fractions. The full readout has 2,890 inputs. Ridge candidates are $0.01,0.1,1,10,100,1000,10000$.

For $V$ videos with $n_v$ clips, $w_{vi}=1/(Vn_v)$ gives every source equal total weight. After the five outer folds are pooled within a seed, $\bar y_w=\sum w_{vi}y_{vi}$ and

$$R_w^2=1-\frac{\sum_{vi}w_{vi}(y_{vi}-\hat y_{vi})^2}{\sum_{vi}w_{vi}(y_{vi}-\bar y_w)^2}.$$

The weighted held-out mean is used only to define $R^2$; a deployable constant predictor would estimate that mean from training sources. The five test folds contain 19, 19, 19, 18, and 18 videos, with 189, 182, 72, 77, and 105 clips. Source-level checks confirm the split assignment.

**Supporting checks and limits.** Recorded mask-inspection draws show that sampled motion masks hide about 17% of valid tokens and region masks hide 9.9%. Connected-region sampling reduces targets with both immediate temporal neighbors visible from 69.9% to zero. Against their random controls, MAMP-style, mixture, and connected-region masks have $\Delta R^2=-0.0008$, $+0.0005$, and $-0.009$, with intervals $[-0.020,0.015]$, $[-0.015,0.015]$, and $[-0.033,0.018]$. These intervals are exploratory and unadjusted for multiple comparisons or development choices.

A limited 264-input coordinate and validity baseline reaches $R^2=0.035$; it is not an upper bound on pose information. The largest ridge penalty, 10,000, is selected in 49/125 trained-teacher and 96/125 trained-online expanded-readout fits. The range may constrain regularization, but a larger value is not known to help. Only aggregate scores and prior intervals remain; raw predictions and checkpoints are needed to rerun inference or the bootstrap. Synthetic demos exercise prefix-only code but provide no real-gait forecast or rollout evidence.

The dataset lacks person identifiers and complete acquisition records. Clinical gait asymmetry motivates the speed score [8], but pair averages can cancel, affected-side labels are unavailable, and the score is not clinically validated. Videos and poses are excluded from the supplement because public annotation access does not authorize redistribution. **AI assistance.** An AI assistant helped check evidence and literature, revise text and figures, and conduct critical reviews. The authors remain responsible for the final paper.
