# Neurologically Motivated Self-Supervised Learning for Bilateral Gait Geometry

**[Authors and affiliations to be inserted in the RISEx template]**

## INTRODUCTION

Masked representation learning is often evaluated through its own latent prediction loss or a broad downstream benchmark. For articulated motion, a geometric observable offers a sharper check. In a skeleton, horizontal reflection exchanges left and right joint labels. A signed contrast of their speeds must reverse. This relation gives a label-free way to ask whether a learned feature space still supports an interpretable movement measurement. It is relevant to human-aware robotics, where person-motion features may later guide prediction, monitoring, or interaction, although no such system is evaluated here.

## MATERIALS AND METHODS

We used 625 GAVD pose clips from 93 source videos [1]. Each independently processed clip retains 33 landmarks, resized to 64 steps and grouped into four-step patches. The resulting 16-by-33 grid keeps each time-block and anatomical position distinct. A skeleton JEPA predicts teacher features at masked positions from visible context [2]. The encoder never receives the movement target.

Our target averages normalized left-minus-right median speeds over shoulders, knees, ankles, heels, and foot tips. We compare each trained teacher with its identical initialization using a frozen ridge readout. Five outer folds exclude entire source videos from both stages; five seeds repeat every condition. Count-matched motion and region masks have their own random references. Whole-source paired bootstrap intervals condition on the saved fitted models.

## RESULTS AND DISCUSSION

| Observable outcome | Initial | After JEPA |
|:--|--:|--:|
| Bilateral readout $R^2$ | 0.223 | 0.101–0.114 |
| Correct clip has lower feature error | 33/75 checks | 375/375 checks |

Across five training arms, trained minus initial $R^2$ was $-0.109$ to $-0.122$, with each reported source-bootstrap interval below zero. Motion-weighted and connected-region masks showed no clear improvement over their count-matched random references. Thus the predictor learned a reliable preference for its own clip's features while the trained encoders gave a weaker readout of the bilateral movement score.

The gap should not be over-interpreted. The teacher is condition-specific, so feature errors cannot rank representations across methods. Also, the expanded readout changes its dimension and includes missing-observation information. Recomputing the score after input preparation gave 70.4% sign agreement and $R^2=0.218$ with its original calculation. These checks limit the conclusion to the tested pipeline.

## CONCLUSIONS

For this articulated-motion setting, improved latent correspondence did not translate into improved recovery of a known geometric observable. Evaluation of self-supervised features should include task-level physical quantities alongside latent prediction diagnostics.

## REFERENCES

[1] Ranjan R et al. *IEEE Access* 13:45321–45339, 2025. [2] Abdelfattah M, Alahi A. *ECCV*, 2024.
