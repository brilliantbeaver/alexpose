---
title: "Refining Section 4.1 of the original draft"
subtitle: "Stochastic regional masking: replacement text and verified explanation"
date: "8 October 2026"
geometry: margin=0.85in
fontsize: 11pt
colorlinks: true
linkcolor: black
urlcolor: blue
---

This note supplies a replacement for Section 4.1 of `wacv2007-v1.pdf` and explains the evidence behind it. The top-level notebooks remain the source of truth. Implementation details below come from the functions those notebooks call; new diagnostic calculations are identified separately. The original PDF and notebooks have been preserved.

## Replacement text for Section 4.1

### 4.1. Stochastic regional masking

During training, S-JEPA predicts numerical descriptions of selected parts of a skeleton sequence using the remaining observations. A *mask* identifies the positions to withhold from the context network. Each 32-frame window includes all 33 body landmarks. Four successive frames for one landmark form a token, giving eight time blocks and 264 tokens per window. The twelve landmarks described below guide how masks are sampled; the model continues to receive the full set of landmark positions before masking.

A fresh mask is drawn for each training example from seven overlapping body regions: head, mouth, left arm, right arm, trunk, left leg and right leg. Each draw chooses a region, a span of one to six consecutive time blocks, and a start position that keeps the span inside the window. Span length and the possible start positions are sampled uniformly. The selected positions are added to the mask until at least 158 tokens have been selected, or 36 region draws have been attempted. Because the last region is added in full, the final hidden fraction can exceed the nominal target of 60%.

The sampling weights reflect an anatomical design choice motivated by gait analysis. Regions containing either shoulder, hip, knee, ankle, heel or foot index receive a multiplier of 1.5. The weight is then divided by the average number of regions containing the region's landmarks, before the weights are converted to selection probabilities. This adjustment partly accounts for shoulders and hips belonging to two regions. Arm regions receive the multiplier through their shoulders, so elbows, wrists and hand landmarks also inherit that region weight. These rules do not give every highlighted landmark the same hiding frequency, or make its final frequency 1.5 times that of other landmarks.

The sampler retains at least one visible token from the twelve-landmark set somewhere in the window. That token may belong to a shoulder, so the rule does not ensure visible hips or legs in every example. Overlapping draws can hide a joint for all eight time blocks, and a whole time block can be hidden when context remains elsewhere. In notebook 02's separate bank of 512 masks generated with seed zero, the mean hidden fraction was 63.3%. Every landmark was visible in at least one time block in at least 74.0% of those sampled windows. Availability as context varied across time: for example, left-hip context occurred in 51.6% of samples at the first block but 7.0% at the fifth block.

These observations describe the sampling rule, rather than the masks encountered throughout model training. Clinical gait studies motivate attention to body position and coordination, but do not validate this particular twelve-landmark choice or its weighting factor. The study includes no controlled comparison with uniform, motion-dependent or alternative anatomical masking. Its results therefore leave the contribution of this masking design to classification performance unresolved.

## What required correction, and why

The original section already recognized that the twelve landmarks were not a fixed mask and that the masking design had not been evaluated separately. Those qualifications should be retained. Its main weaknesses were an incomplete sampling rule, an ambiguous summary of coverage, and insufficient separation between clinical motivation and evidence of benefit. The visible editorial marker “Rewrite:” should also be removed.

**The multiplier alone does not describe the sampler.** The implementation applies both the 1.5 anatomical multiplier and a correction for overlapping region membership. Omitting the second operation prevents an accurate reconstruction of the procedure. It also obscures why trunk draws have a lower probability than head draws, even though the trunk contains selected landmarks. The probabilities below apply to one region draw; they are not final joint-hiding probabilities.

| Region | Landmark indices | Probability per draw |
|:--|:--|--:|
| Head | 0–8 | 12.7854% |
| Mouth | 9, 10 | 12.7854% |
| Left arm | 11, 13, 15, 17, 19, 21 | 16.4384% |
| Right arm | 12, 14, 16, 18, 20, 22 | 16.4384% |
| Trunk | 11, 12, 23, 24 | 9.5890% |
| Left leg | 23, 25, 27, 29, 31 | 15.9817% |
| Right leg | 24, 26, 28, 30, 32 | 15.9817% |

The exact rule is $w_G=b_G/m_G$, followed by $P(G)=w_G/\sum_H w_H$. Here $b_G$ is 1.5 when region $G$ contains a selected landmark and 1 otherwise; $m_G$ is the mean number of region memberships among its landmarks. The twelve selected indices are 11, 12, and 23–32. The factor 1.5 is a configured design choice; the retained experiment does not establish that it is optimal.

**“Approximately 60%” conceals useful implementation details.** The code rounds $0.6\times264$ to 158 and adds complete region spans without trimming the last draw to that budget. This explains the diagnostic average of about 63.3%. The one-to-six-block limit applies separately to each region draw. Several overlapping draws can cover all eight blocks for the same joint.

**Coverage has two different denominators.** “Visible somewhere in a window” counts a window once if a joint has any context token. “Visible token fraction” counts individual joint–time positions. The former can look reassuring while a joint is rarely available at particular times. Notebook 02 reports both measures, including the low middle-block hip coverage. Its minimum 74.0% window coverage is an observation from 512 sampled masks, not a guarantee for each window, each batch or a saved training run. The guarantee in the code concerns one selected-set token somewhere in each window. In the first seed-zero demonstration, both complete legs are hidden for the entire window.

**The original 5,000-mask summary should not be presented as notebook output.** Notebook 02 saves a 512-mask diagnostic. Its printed mean is rounded to 0.63; the replacement's 63.3% comes from recalculating that same seeded bank, which gives 0.6327607. A separately reproduced 5,000-mask, seed-zero calculation yields 77.84–78.31% hiding for shoulders, 76.87–77.26% for hips, 62.06–62.54% for the other leg landmarks, and 63.61–64.30% for elbows and wrists. For facial landmarks, head-region tokens are hidden 59.65% of the time and mouth tokens 52.18%. Their joint-weighted average is 58.29%. Thus the original “58% for facial landmarks” is compatible with an aggregate across all eleven facial landmarks, but it conceals different rates for the two regions. The replacement uses the notebook's own bank and keeps the expanded calculation in this verification note.

**Clinical relevance supplies a rationale, not a measured benefit.** Notebook 02 maps landmarks to candidate gait measurements, but the table does not show that those physical measurements were recovered or clinically validated in this experiment. Its introductory suggestion that motion-based masking would conceal the relevant disease signal is also an untested explanation: withheld information is used as a prediction target, so withholding alone does not demonstrate loss of useful learning. The paper can state that this implementation uses anatomical weights without claiming that motion-dependent masking is unsuitable for MS or PD.

## Related wording that should remain consistent

If the original landmark illustration is retained, replace its caption with:

*The highlighted landmarks are both shoulders, hips, knees, ankles, heels and foot indices (BlazePose indices 11, 12 and 23–32). They influence the weights used to sample body regions during training. All 33 landmarks remain part of the input, and a new regional mask is drawn for each example. Weights include both a factor of 1.5 for regions containing highlighted landmarks and a correction for overlapping region membership. Highlighting therefore indicates the sampling design, rather than a fixed hidden set or a measured contribution to classification accuracy.*

The reflection limitation belongs beside the augmentation description or in the Discussion, with a cross-reference if needed. Reflection swaps left and right landmark slots in the context input, while the target mask and unreflected reference retain their original indices. Consequently, a physical joint selected as a reference target can appear in the opposite-side context slot. This is a mismatch within training examples; it is not evidence of overlap with held-out test recordings. The retained experiments do not measure its effect on accuracy. Correcting the text does not produce results for a corrected training procedure.

The classification readout also needs separate wording: the fixed seed-zero mask selects 161 **output** positions from the target encoder after it receives the complete window. It does not hide those input coordinates during classification. Confusing that readout with the changing training masks would give readers the wrong account of how predictions are made.

## Evidence and verification

Notebook cell indices here are zero-based and include Markdown cells. Links are relative to this note's folder.

| Evidence | What it establishes |
|:--|:--|
| [Notebook 02](../../02_anatomical_mask_and_tokenization.ipynb), cells 7, 9, 10–13 and 16–17 | Token count; twelve indices; 512-mask saved diagnostic; coverage definitions and first-mask demonstration. |
| [Mask implementation](../../sjepa/masking_v2.py), `JOINT_GROUPS`, `_region_weights`, `sample_target_mask` | Seven regions, membership correction, draw probabilities, budget, span lengths, stopping rule and context guarantee. |
| [Notebook audit](evidence/notebook-audit.md) and [stored checks](evidence/notebook-audit-checks.json) | Independent deterministic 5,000-mask diagnostic; separate classification readout and reflection correspondence checks. |
| [Section-specific verification](evidence/section-4.1-verification.md) | Rechecked cell/line mapping and numerical reproduction for this standalone refinement. |
| [Literature audit](evidence/venue-and-literature.md) | Primary-source support and limits for retained scientific claims. |

Abdelfattah and Alahi's [S-JEPA paper (ECCV 2024)](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04755.pdf) supplies the relevant skeleton-prediction reference. Its motion-aware rule increases the sampling probability of joints with larger motion; it should not be described as always selecting the highest-motion joints. That comparison is background to this implementation, rather than an experiment performed here. [Filli et al. (2018)](https://www.nature.com/articles/s41598-018-22676-0) and [Plotnik et al. (2020)](https://www.nature.com/articles/s41598-020-68263-0) support gait-related clinical motivation without validating the selected mask. The rewritten section therefore needs no claim that the present design is clinically established or superior to those alternatives.

The numerical mask checks require no pose re-extraction, model retraining or change to the notebooks. They verify what the documented sampler does. Determining whether the anatomical weights improve learning would require a new, matched experiment.
