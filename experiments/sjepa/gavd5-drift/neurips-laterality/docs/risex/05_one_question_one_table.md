# Neurologically Motivated Self-Supervised Learning for Bilateral Gait Geometry

**[Authors and affiliations to be inserted in the RISEx template]**

## INTRODUCTION

Robotic systems that perceive people need motion features whose useful geometric content can be checked. We test a skeleton JEPA using bilateral gait. Under reflection, a skeleton's left and right landmarks exchange places, and a left-minus-right speed contrast changes sign. This property provides a direct target for evaluating a representation. Gait asymmetry motivates the setting because it occurs in several health conditions [1], but our score is derived from estimated pose coordinates and is not a clinical endpoint.

## MATERIALS AND METHODS

We used 625 GAVD clips from 93 source videos [2]. Clip-local preparation keeps all 33 landmarks in a 64-step sequence; four-frame patches make a 16-by-33 grid. A JEPA predicts hidden teacher features from visible grid positions [3]. The bilateral target averages normalized speed differences for five left–right joint pairs.

The null hypothesis was no gain over an encoder with the same initial weights. Five outer folds hold out complete source videos from both encoder training and frozen ridge-readout fitting. Five seeds repeat each comparison. Matched mask conditions share starts, source draws, views, and updates. Scores weight source videos equally; paired resampling uses whole sources.

## RESULTS AND DISCUSSION

| Check | Result | Reading |
|:--|:--|:--|
| Initial encoder readout | $R^2=0.223$ | Reference before JEPA |
| Trained teachers | $R^2=0.101$–$0.114$ | Lower in all five arms |
| Trained minus initial | $-0.109$ to $-0.122$ | Each reported interval below zero |
| Correct clip feature target | 375/375 trained checks | Higher within-clip correspondence |
| Alternative masks | intervals cross zero | No clear gain over random masks |

JEPA training made its predictor discriminate a correct target clip from a mismatched source more consistently, yet the frozen readout estimated bilateral movement less accurately. The feature test can use pose, view, or missingness cues, so it is a diagnostic rather than proof of useful motion representation. The movement result also depends on the expanded readout and target preparation; it identifies a limitation of this pipeline.

## CONCLUSIONS

Within-clip latent prediction is insufficient evidence that an articulated-motion representation preserves a downstream geometric quantity. The proposed evaluation pattern can guide human-aware perception research before features are transferred to robot-facing tasks. It does not demonstrate robot performance, control, or clinical use.

## REFERENCES

[1] Patterson KK et al. *Arch Phys Med Rehabil* 89:304–310, 2008. [2] Ranjan R et al. *IEEE Access* 13:45321–45339, 2025. [3] Assran M et al. arXiv:2301.08243, 2023.
