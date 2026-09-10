# Neurologically Motivated Self-Supervised Learning for Bilateral Gait Geometry

**[Authors and affiliations to be inserted in the RISEx template]**

## INTRODUCTION

Robots that work around people need compact descriptions of human motion. A useful description should retain geometric relations that matter to a later task, rather than only make hidden inputs easy to predict. Human gait offers one such relation. Reflecting a pose exchanges left and right joints and reverses a signed difference between their movement speeds. Gait asymmetry is clinically relevant in stroke and Parkinson's disease, although the score studied here is not a diagnosis [1]. We ask whether a skeleton JEPA retains this bilateral movement signal after self-supervised training.

## MATERIALS AND METHODS

We used 625 accepted pose clips from 93 GAVD source videos [2]. Each clip contains 33 landmarks. After within-clip normalization and resizing, we retain all landmarks in a 64-by-33-by-3 input; four-frame patches form a 16-by-33 grid. A JEPA encoder receives visible tokens and predicts features of masked tokens supplied by a slowly updated teacher encoder [3]. The signed target averages normalized left-minus-right median speeds for shoulders, knees, ankles, heels, and foot tips.

The null hypothesis was that training would not improve a frozen ridge-regression readout over the same encoder at initialization. We use five outer source-video folds and five training seeds. Test videos are excluded from encoder and readout fitting. Matched arms keep initial weights, source draws, views, and update counts fixed while changing the mask.

## RESULTS AND DISCUSSION

| Test | Result |
|:--|:--|
| Initial frozen encoder | $R^2=0.223$ |
| Trained teacher encoders | $R^2=0.101$–$0.114$ |
| Trained minus initial | $-0.109$ to $-0.122$, five arms |
| Correct-clip feature target preferred | 375/375 trained checks; 33/75 initial checks |

Training improved a diagnostic of within-clip feature correspondence while reducing accuracy for the bilateral readout. Motion-weighted and connected-region masks did not resolve this: their differences from count-matched random masks had intervals that crossed zero. The result is relevant to AI systems that represent articulated people, because success on a latent prediction task did not imply preservation of this downstream geometric quantity.

## CONCLUSIONS

For the tested data, JEPA recipe, and frozen readout, hidden-feature prediction and bilateral movement recovery disagree. The study provides a compact diagnostic for evaluating learned articulated-motion representations. It does not evaluate a robot, forecast future motion, or establish clinical utility. Future work should test a past-only human-motion prediction task with simple motion baselines and independently specified data.

## REFERENCES

[1] Patterson KK et al. *Arch Phys Med Rehabil* 89:304–310, 2008. [2] Ranjan R et al. *IEEE Access* 13:45321–45339, 2025. [3] Assran M et al. arXiv:2301.08243, 2023.
