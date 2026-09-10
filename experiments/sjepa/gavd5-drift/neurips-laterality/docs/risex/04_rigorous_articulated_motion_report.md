# Neurologically Motivated Self-Supervised Learning for Bilateral Gait Geometry

**[Authors and affiliations to be inserted in the RISEx template]**

## INTRODUCTION

Learning from videos of people could help robots anticipate human motion, but an internal feature match is an incomplete test of what a representation preserves. We evaluate a skeleton JEPA with a geometric relation that has a known answer. Reflection swaps left and right anatomy and reverses a signed speed difference. Gait asymmetry has been studied in stroke, Parkinson's disease, cerebral palsy, and muscular disorders [1–3]. Those studies motivate the relation; this work neither diagnoses these conditions nor estimates disease severity.

## MATERIALS AND METHODS

We selected 625 pose sequences from 93 GAVD source videos [4]. Processing is clip-local. It retains all 33 landmarks in a 64-by-33-by-3 array, then packs four frames per token to preserve a 16-by-33 spatial-temporal grid. The online encoder receives visible tokens, while a predictor estimates hidden teacher features. The teacher follows the online encoder by exponential moving average. This pretraining uses no movement target or disease label.

The target averages normalized left-minus-right median speed differences for five anatomical pairs. A frozen ridge readout is fitted only on outer-training sources. Five outer source-video folds keep test clips, transformed views, encoder training, and supervised readout fitting separate. Five seeds share initialization, source draws, views, and updates within each mask comparison. We report source-balanced scores and paired source resampling that keeps all clips and seed predictions from a source together.

## RESULTS AND DISCUSSION

![Study design and primary result: correctly matched latent features improve while the bilateral movement readout declines.](risex_onepage_evidence.png)

The initial encoder achieved $R^2=0.223$. Trained teachers achieved $0.101$–$0.114$; every training arm was lower than its matched initialization (difference $-0.109$ to $-0.122$). Conversely, correct-clip targets had lower feature error in 375/375 trained diagnostic checks, versus 33/75 at initialization. Alternative motion and connected-region masks gave intervals spanning zero relative to matched random masks.

The result is a controlled warning for learned articulated-motion features. It remains conditional on the target and readout: preparation changes the score, and the larger readout combines motion, dimension, and missingness changes. It therefore cannot establish a general loss of geometric information or a robotics failure.

## CONCLUSIONS

Latent feature correspondence and recovery of bilateral movement can disagree. A geometric observable, matched initialization, and source-held-out test split provide a compact evaluation pattern for AI representations of articulated motion.

## REFERENCES

[1] Patterson KK et al. *Arch Phys Med Rehabil* 89:304–310, 2008. [2] Lewek MD et al. *Gait Posture* 31:256–260, 2010. [3] Brændvik SM et al. *Front Neurol* 10:1399, 2020. [4] Ranjan R et al. *IEEE Access* 13:45321–45339, 2025.
