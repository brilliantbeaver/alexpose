# Neurologically Motivated Self-Supervised Learning for Bilateral Gait Geometry

**[Authors and affiliations to be inserted in the RISEx template]**

## INTRODUCTION

Self-supervised motion features are attractive for AI systems that must interpret people from incomplete observations. A central question is whether success at predicting a hidden feature means that the representation retains a useful physical relation. We test this question with bilateral gait geometry. Reflection exchanges left and right body landmarks and reverses a signed contrast of their movement speeds. This relation is relevant to gait asymmetry in several neurological and musculoskeletal conditions [1], while remaining independent of a diagnostic label.

## MATERIALS AND METHODS

The study uses 625 quality-checked GAVD pose clips from 93 source videos [2]. Each clip is prepared independently. All 33 landmarks remain in a 64-by-33-by-3 input, and four-frame patches yield a 16-by-33 time-by-joint grid. A skeleton JEPA predicts teacher features for masked grid positions from their visible context [3]. No label, target, or affected side supervises encoder training.

We define a left-minus-right median-speed contrast over five paired joints. The null hypothesis is that JEPA does not improve its frozen ridge-regression estimate over a matched initial encoder. Five outer folds exclude complete source videos from pretraining and readout fitting; five seeds repeat each condition. Mask comparisons share starting weights, source draws, training views, and updates. We weight sources equally and resample sources, not clips, for paired intervals.

## RESULTS AND DISCUSSION

| Outcome | Before JEPA | After JEPA |
|:--|--:|--:|
| Bilateral readout $R^2$ | 0.223 | 0.101–0.114 |
| Correct-clip target preferred | 33/75 checks | 375/375 checks |

All five trained-minus-initial contrasts were negative ($-0.109$ to $-0.122$); their saved source intervals lay below zero. Motion-weighted and connected-region masks had no clear advantage over their own count-matched random references. The predictor therefore learns within-clip feature correspondence while the tested encoder-plus-readout pipeline produces a weaker bilateral movement estimate.

The evidence has limits that matter for interpretation. Teacher feature scales differ between conditions. The expanded readout adds temporal summaries and missing-data support while increasing dimension. Moreover, preparation changes the target calculation: its agreement with the original calculation is 70.4% by sign and $R^2=0.218$. The study does not show that JEPA destroys geometric information, only that this training recipe and readout fail to improve this observable.

## CONCLUSIONS

A latent prediction diagnostic can improve even when a prespecified geometric readout declines. Human-aware robotics may benefit from testing learned motion features against task-relevant observables before using them downstream. Future work should evaluate past-only motion prediction with persistence and velocity baselines, a stable target path, and new source or participant groups.

## REFERENCES

[1] Lewek MD et al. *Gait Posture* 31:256–260, 2010. [2] Ranjan R et al. *IEEE Access* 13:45321–45339, 2025. [3] Abdelfattah M, Alahi A. *ECCV*, 2024.
