# Masking alternatives for the next gait study

The strongest small experiment is to compare **uniform masks, joint tubes, and anatomical tubes while holding the masked-token budget and each token's marginal masking probability constant**. Evaluate all three training recipes on the same frozen uniform-mask task. Use structured evaluation as a separately named diagnostic. This tests whether removing easy interpolation during training improves the actual gait estimator, rather than merely changing which questions the model answers.

This review checks primary papers available through November 2026, including recent author implementations. It is a targeted methods review, not an exhaustive priority search. None of the proposed gait ablations below has been trained or evaluated. The suggested mask probabilities were checked algebraically and by exhaustive enumeration of the proposed anatomical cores, without using GAVD outcomes.

## Correct the description of the original methods

**MAMP is not uniformly random masking by default.** It predicts motion and preferentially masks tokens with greater motion intensity. In the paper, a joint token contains several consecutive frames. Motion intensity is calculated from temporal differences, converted to sampling weights, and combined with Gumbel noise to select the masked tokens. In schematic notation:

```text
pi = softmax(motion_intensity / temperature)
masked_indices = top_k(log(pi) + Gumbel_noise)
```

The weights are not the final marginal inclusion probabilities when sampling multiple tokens without replacement. This distinction matters when comparing masking distributions. [MAMP, ICCV 2023, Section 3.4](https://arxiv.org/html/2308.07092).

The official implementation additionally normalizes mean absolute patch-to-patch motion by the sample's maximum motion before applying the temperature. It retains the lowest Gumbel scores and masks the highest. Its `motion_aware_tau <= 0` branch uses ordinary random masking. Therefore a uniform variant is available, but should be named a variant rather than the published default. The inspected [official implementation](https://github.com/maoyunyao/MAMP/blob/main/model_mamp/transformer.py) has file SHA256 `9c29a96ff475c92908baef74a0cfbf820a8a74e6462866adf4feca2c96950431`.

**The original skeleton S-JEPA also uses motion-aware masking inherited from MAMP.** It predicts target-encoder representations with centered and sharpened distributions, an EMA target encoder, and cross-entropy. Its published setup uses four-frame patches, 120 frames, a 0.9 masking ratio, and ordinary learnable spatial and temporal positional embeddings. A 0.9 ratio on that dense action-recognition grid does not establish that 90% masking is sensible on our 128-token, incomplete 2D gait grid. Our uniform masking, coordinate readout and VICReg choices are declared adaptations. [S-JEPA, ECCV 2024, Sections 3 and 4.2](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04755.pdf).

Stochastic positional embeddings are a separate design axis. StoP samples uncertain target locations for masked image modeling. It does not mean uniformly sampling missing skeleton tokens, and it is not the positional-embedding method specified by the original skeleton S-JEPA paper. Keep it out of this masking-only comparison so that a gain has an interpretable source. [StoP paper, revised February 2024](https://arxiv.org/abs/2308.00566).

## What recent work actually changes

| Primary work and verified status | Masking or related mechanism | Relevance and boundary for this project |
| --- | --- | --- |
| [Graph-based SkeletonMAE, ICCV 2023](https://www.openaccess.thecvf.com/content/ICCV2023/papers/Yan_SkeletonMAE_Graph-based_Masked_Autoencoder_for_Skeleton_Sequence_Pre-training_ICCV_2023_paper.pdf). | Selects predefined anatomical joint sets for feature masking and reconstruction using a graph-based architecture. | Anatomical masking predates this project. Do not confuse this paper with the different spatial-temporal SkeletonMAE at ICME Workshops 2023. |
| [GFP, ICCV 2025](https://arxiv.org/html/2509.03609v1). | Uses MAMP's motion-aware mask while changing prediction to hierarchical learned features with a lightweight target generator. | It is evidence about target design and computational efficiency, not a new independent mask sampler. Avoid reproducing its whole architecture in a masking ablation. |
| [ASMa, TMLR 2026](https://23wm13.github.io/ASMa/), with [method equations](https://arxiv.org/html/2602.06251v1). | Uses complementary high-degree/low-degree joint selection and high-motion/low-motion frame selection, followed by feature alignment and distillation. | Its complementary views are not simply a binary mask and its inverse. The temporal choices depend on motion, so this is literature context rather than the preferred main alternative. |
| [Motion topology masking, Scientific Reports, February 2026](https://www.nature.com/articles/s41598-026-39330-9). | Mixes topology and motion weights, samples joints, and maps them to body parts to mask a region. | Grouping can change the final masked-token count. Its motion-weighted version is not velocity-independent, and its results concern action recognition. |
| [AMR, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Sun_Exploring_Adaptive_Masked_Reconstruction_for_Self-Supervised_Skeleton-Based_Action_Recognition_CVPR_2026_paper.html), with [full method](https://arxiv.org/html/2606.11450v1). | Decouples decoder granularity and weights reconstruction using local motion energy. Its implementation details retain MAMP motion-aware masking. | “Adaptive” primarily describes reconstruction guidance here. Changing the sampler, target size and loss weighting together would not isolate masking. |
| [SLiM, arXiv v4, November 5, 2026](https://arxiv.org/html/2603.10648v4). Its [author record](https://arxiv.org/abs/2603.10648) reports NeurIPS 2026. | Semantic Tube Masking selects anatomical regions across consecutive time tokens. Proposal duration varies inversely with spatial width. | This is directly relevant prior work on avoiding local interpolation. Anatomical tubes and that motivation cannot be claimed as our invention. |

Three details prevent misleading implementations:

1. ASMa uses degree-proportional weights for one spatial view and complementary weights for another; the latter need normalization when interpreted as a categorical sampling distribution. Its temporal views choose high or low motion frames. A motion-independent complement-pair control below is therefore our simpler control, not an ASMa reproduction.
2. The Scientific Reports mixture is `P = alpha*P_topology + (1-alpha)*P_motion`. Setting alpha to one removes the motion term, but still leaves body-part expansion, token-budget and group-frequency choices to specify. A topology-only adaptation must report its realized masks.
3. AMR's energy is the mean squared displacement from a joint's within-window mean. The reconstruction weight grows with that energy. Low-amplitude movement may be medically relevant, so emphasizing high energy cannot be assumed beneficial for gait abnormality detection.

SLiM samples target mask ratios between 0.5 and 0.9 and uses repeated tube proposals. Its appendix explicitly distinguishes requested from realized coverage. It replaces content before adding joint identity embeddings. The official [mask generator](https://github.com/KAIST-VICLab/SLiM/blob/main/slim/data/masking.py) uses contiguous slices of declared joint-group lists rather than an explicit graph-connectivity test. Do not copy its Kinect joint indices into BlazePose or assume that numerical adjacency means anatomical adjacency. The inspected file SHA256 is `b655049895f36e1afe0c6ff94ffc7e7fb961e37d2cbe1cfc2580d1fbd07f502a`.

These papers report different objectives, datasets, backbones and compute budgets. Their action-recognition rankings do not establish a best masking policy for GAVD. The important contribution here would be a controlled explanation of when mask structure helps conditional prediction and real gait detection.

## Run three primary masking recipes

Use the same 16-joint allowlist, 32 frames, four frames per token, and an 8-by-16 token grid. Let `m[t,j]=1` mean that a token is hidden. Every primary recipe masks exactly 64 of the 128 tokens before validity filtering. No recipe below reads coordinates, velocity, confidence, diagnosis or a model's attention when drawing a mask.

| Recipe | Exact construction | Scientific question |
| --- | --- | --- |
| U: uniform reference. | Sample 64 distinct token indices uniformly from 128. | How well does the original observation task work? |
| J: joint tube plus uniform fill. | Choose one of the 16 joints uniformly. Hide its eight temporal tokens, then choose 56 additional tokens uniformly from the remaining 120. | Does preventing same-joint temporal interpolation improve training, while local neighboring joints remain available? |
| A: anatomical tube plus uniform fill. | Draw an eight-token anatomical core using the rule below, then choose 56 additional tokens uniformly from the remaining 120. | Does also hiding a connected local part over a short interval help beyond hiding one joint? |

These are controlled adaptations for this study, not faithful reproductions of entire published systems. The uniform fill is deliberate: it keeps the total budget fixed and avoids making tube construction a large new subsystem.

### Define the anatomical core without changing joint frequencies

Use these disjoint groups. The hip stays in the shared torso group, so a lower-limb core is explicitly distal to the hip.

| Group | BLAZEPOSE_33 indices | Width, w | Consecutive temporal tokens, h | Probability of selecting the group |
| --- | --- | ---: | ---: | ---: |
| Torso anchors | 11, 12, 23, 24 | 4 | 2 | 4/16 |
| Left elbow and wrist | 13, 15 | 2 | 4 | 2/16 |
| Right elbow and wrist | 14, 16 | 2 | 4 | 2/16 |
| Left distal leg | 25, 27, 29, 31 | 4 | 2 | 4/16 |
| Right distal leg | 26, 28, 30, 32 | 4 | 2 | 4/16 |

For a selected group, use `h=8/w`. Pick one of the disjoint aligned blocks of length h uniformly. For h=2, the starts are 0, 2, 4 and 6. For h=4, the starts are 0 and 4. Every core therefore contains eight tokens. No data or mask wraps around the interval boundary.

For any token in a group of width w, the probability of belonging to the core is:

```text
P(core) = (w/16) * (h/8) = 1/16.
P(hidden after uniform fill) = 1/16 + (15/16)*(56/120) = 1/2.
```

J has the same core probability, and U also masks each token with probability one half. Thus U, J and A match their total hidden count and first-order token probabilities. They differ in which tokens disappear together. Matching these quantities does not equalize task difficulty, higher-order dependence, valid context counts or optimization behavior. Those differences must be measured.

For example, A can hide the left knee, ankle, heel and foot index for eight consecutive frames. A predictor must then use observations outside that interval, the shared reference, and any permitted right-leg context. This is a clearer probe of reliance on local geometry than independently hiding an ankle while its knee and heel remain visible. It still does not measure a neurological cause.

## Separate training effects from changes in the question

Use the existing frozen U evaluation bank of 16 masks and seed 17 for the primary score. Keep the original source registry, endpoints, coordinate targets, aggregation, normal references, local-motion score, fusion and within-fold comparison unchanged. The main masking result compares training recipes **within this same evaluation column**.

Start with the inexpensive retained-baseline coordinate model. Keep the already selected U-trained `f` fixed, then fit each recipe's own-capacity correction `h` and both-leg correction `g` using the same normal fitting sources and declared budgets. Evaluate all three on U. This isolates the effect of the correction-training mask without changing the retained local baseline. The normal adequacy gate still compares each `f+g` with both `f` and that recipe's `f+h`.

If the learned JEPA candidate passes the earlier adequacy gates, run a separate encoder-pretraining ablation with U, J and A. Keep the downstream coordinate-head training masks fixed to U in that ablation. Do not simultaneously change encoder masks and head masks and then attribute an improvement solely to representation learning. Match updates, source sampling, batch size, target dimension and regularization. Keep the random-encoder comparison.

Report all three predeclared recipes. If a single operational recipe must be selected, choose using source-grouped fitting-normal validation only. Reference normals may fit score parameters after selection; they must not select a mask. Abnormal AUROC must not choose the training policy. All earlier exploration and the new ablation remain exploratory.

Add J and A evaluation banks as secondary columns if support allows:

| Training recipe | U evaluation, primary | J evaluation, diagnostic | A evaluation, diagnostic |
| --- | --- | --- | --- |
| U | Required | Secondary | Secondary |
| J | Required | Secondary | Secondary |
| A | Required | Secondary | Secondary |

Compare training recipes within a column using matched target keys and support. Fit a separate normal score reference for each diagnostic evaluation bank. U/U versus A/A is not an isolated training comparison because both the training and the question changed. Report primary and diagnostic eligibility separately. Do not force every mask family and timing edit onto one tiny all-controls intersection.

The fixed-target distal-leg timing control should retain its existing U parent-grid mask protocol for every training recipe. Altering its mask geometry at the same time would create another experiment. A later paired geometry diagnostic may hold Y fixed and deliberately remove its neighboring joints, but its estimand and support must be separately declared.

## Keep complementary and adaptive alternatives bounded

**One inexpensive complementary control is sufficient.** Compare 16 independent U evaluations with eight independently drawn U masks and their eight complements. Each complement also masks 64 tokens, and every token is masked once in each pair. Average and count targets using the existing aggregation. This can test mask Monte Carlo variability. It is not additional independent patient evidence. It also does not change the marginal distribution of a single uniform mask. For a training version, compare two complement-paired forward passes against two independent passes, not against one pass.

**Do not add motion-aware sampling to the main three-arm experiment.** The latest request supports studying alternatives, but the earlier velocity-independent requirement is best preserved by U, J and A. Describe MAMP, ASMa and motion topology weighting accurately in the literature review. If a motion-aware training sensitivity is later authorized as a separate protocol, keep U evaluation and state that mask patterns themselves now depend on the underlying motion. Computing a test mask from a hidden trajectory changes the information available through the mask and defeats a simple hidden-value noninterference interpretation.

**Defer learned attention masks, aggressive mask-ratio curricula and StoP.** They introduce mask selection, support, optimization or positional-uncertainty changes beyond the question at hand. A learned sampler on very few normal sources could emphasize camera or pose artifacts. A favorable action-recognition result is not enough to assume that large or fast movement is the important gait signal.

## Checks that can invalidate the comparison

1. Verify exactly 64 hidden tokens, allowlist membership, deduplicated groups and bilateral symmetry. Empirically audit per-token frequencies against the analytical one-half probability before applying missing-data rules.
2. Draw masks before validity filtering. Never redraw to recover a preferred joint or easier target. Report the realized number of valid targets, own-leg observations and other-leg observations by recipe.
3. Keep mask flags separate from missing measurements. Invalid measurements are not zero-valued skeleton observations. Masked content must be inaccessible before feature construction and attention; positional identities can remain available consistently.
4. Audit legacy eligibility. The old own-joint rule requiring visible patches of the very target joint would discard the selected J tube. The new own-leg branch needs a declared branch-appropriate context rule. Preserve primary U cohort definitions and document any change; do not silently claim that the historical selector already supports the new diagnostic.
5. Preserve target values, normalizers and references across own-capacity and both-leg comparisons. Keep source weighting and equal joint readout weights. Match nominal training budgets and report effective valid-target exposures as well.
6. Re-run known-dependency, conditional-independence, zero-gate and hidden-input invariance fixtures for each recipe. Then report normal prediction adequacy on U and on any diagnostic support used for interpretation.
7. Report mask support and losses even if a recipe fails. A stronger anatomical occlusion can make the task unlearnable with this data. Worse error is not evidence of a richer world model.

The publishable question is whether a controlled mask structure helps the model learn useful contralateral information that transfers to a fixed observation task and contributes to real-source gait detection. The mask pattern itself is established methodology. The proposed study must earn its novelty through the controlled measurement, its result, and its limits.

## Assessment of the focused mixed-policy alternative

The coordinator also proposed a smaller clinical emphasis: Bmix draws U half the time and otherwise masks a left or right distal-leg block containing eight tokens before uniform fill to 64; Tmix draws U half the time and otherwise masks eight randomly selected joints for all eight temporal positions. Both are implementable and motion-independent. Their fixed U evaluation is the correct primary comparison.

Bmix is not a pure change in co-masking geometry. Averaged across time, it masks each lower-limb token with probability about 0.5167 and each other token with probability about 0.4833. Sampling a contiguous block from all seven possible starts also masks central temporal positions more often than endpoints. These are modest but real differences from U. Describe Bmix as a combined anatomical-structure and target-emphasis policy unless a matched-frequency control is supplied. The balanced A construction above removes that ambiguity without requiring another training arm.

Tmix has one-half marginal masking probability for every token, but its structured component concentrates all 64 hidden tokens into full joint trajectories. Bmix concentrates only eight tokens into its anatomical core and fills the rest uniformly. Tmix versus Bmix therefore differs in structured-mask strength as well as shape. The single-joint eight-token core J above is the closer matched-budget geometry control. If retaining Tmix, present it as a stronger temporal stress policy and audit learning adequacy rather than attributing its difference solely to tube versus body-part geometry.

A possible exact frequency-matched control for Bmix chooses the same leg, but independently draws a two-patch contiguous interval for each of its four joints, followed by the same uniform fill. This preserves each joint/time token's marginal core probability while breaking shared cross-joint interval selection. It is a useful substitute for another optional arm, not a reason to expand into a large mask search.
