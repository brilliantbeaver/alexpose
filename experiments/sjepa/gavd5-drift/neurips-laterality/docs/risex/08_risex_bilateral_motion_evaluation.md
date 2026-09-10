# Neurologically Motivated Self-Supervised Learning for Bilateral Gait Geometry

Alexander Mui\*, Penelope Inouye

Computer Science & Engineering, ASDRP, Fremont, CA, USA

\*Corresponding author: Alexander Mui; email: alexander.mui@students.asdrp.org

## INTRODUCTION

Stroke-related walking asymmetry and asymmetric arm swing in Parkinson's disease [1,2] motivate side-sensitive representations for human-movement understanding and rehabilitation robotics. We test whether a joint-embedding predictive architecture (JEPA), which predicts hidden features rather than coordinates, improves recovery of a signed bilateral speed contrast. Our exploratory null hypothesis is no improvement over matched initialization.

## MATERIALS AND METHODS

We use 625 quality-checked GAVD pose clips from 93 source videos [3]. For five joint pairs (shoulders, knees, ankles, heels and foot tips), the target is

$$
y=\frac{1}{5}\sum_{k=1}^{5}\frac{m_{L,k}-m_{R,k}}{m_{L,k}+m_{R,k}+10^{-8}}.
$$

Here $m$ is median speed in body-normalized pose coordinates, using original timestamps and at least eight jointly observed transitions per pair. Reflection swaps sides and gives $y(Mx)=-y(x)$. This pose-derived measure has no independent clinical validation.

Clip-local centering, scaling, short-gap filling and resampling produce $64\times33\times3$ inputs. Four-frame patches retain a $16\times33$ time–joint grid; masking zeroes features without removing positions. Joint identity is preserved, while resampling changes the original clock.

Our S-JEPA-inspired model [4] uses 96-channel features, a four-layer encoder and two-layer predictor. Masked cross-entropy matches teacher-channel distributions; the teacher receives no gradients and follows exponential averaging (0.999). A 0.05-weighted VICReg term [5] encourages agreement and nonconstant features pooled over 12 gait joints across two complete views. Neither target scores nor neurological labels enter pretraining. Two motion-based masks and a connected-region mask have count-matched random controls within each mask family: five conditions, five folds and five seeds give 125 runs, each with 1,200 updates and batches of 20. Paired conditions share initialization, source draws and views.

Five outer folds exclude entire source videos from pretraining and readout fitting. All encoder comparisons use the same 2,890-feature summary of bilateral means, temporal variation and observation support. A ridge readout (linear regression with shrinkage) uses three inner source-separated folds for training-only scaling and penalty selection. Held-out predictions are pooled across outer folds; sources receive equal total weight, and scores are averaged across seeds. Paired 95% intervals use 2,000 whole-source bootstrap draws with fitted models fixed.

## RESULTS AND DISCUSSION

Training reduced bilateral readout accuracy across all five conditions (Table 1). For the motion-family random control, trained-minus-initial $\Delta R^2=-0.109$ [−0.170, −0.041]. All five marginal intervals lie below zero; structured-mask comparisons with their random controls span zero.

*Table 1. Source-balanced held-out scores. Higher $R^2$ and lower mean absolute error (MAE) are better; ranges cover five trained conditions.*

| Predictor | $R^2$ | MAE |
|:--|--:|--:|
| Training-source mean | −0.011 | 0.0462 |
| Matched initial encoder | 0.223 | 0.0415 |
| Trained teachers | 0.101–0.114 | 0.0436–0.0440 |

Correct-clip teacher targets yield lower feature error than targets from another source in 375/375 trained checks, versus 33/75 at initialization. These repeated fold/seed/mask checks establish clip correspondence, which can also reflect viewpoint or missingness. They do not isolate movement prediction.

Original and preprocessed target calculations agree in sign on 70.4% of source-weighted comparisons (623 clips, 92 sources); 49/125 teacher readouts select the largest tested ridge penalty. These observations limit interpretation of the readout deficit. Analyses are exploratory; intervals omit retraining and selection uncertainty, and video separation does not verify participant independence.

## CONCLUSIONS

Under this recipe, clip-specific hidden-feature matching coexists with poorer recovery of bilateral movement by the ridge readout. For human-movement models, the result supports testing a fixed geometric target alongside the training objective. Time-preserving inputs and a wider readout-penalty search should be evaluated before drawing conclusions about suitability for rehabilitation or future-motion prediction.

## REFERENCES

[1] Patterson KK et al. *Arch Phys Med Rehabil* 89:304–310, 2008. [2] Lewek MD et al. *Gait Posture* 31:256–260, 2010. [3] Ranjan R et al. *IEEE Access* 13:45321–45339, 2025. [4] Abdelfattah M, Alahi A. *ECCV 2024*, LNCS 15090:367–384. [5] Bardes A et al. *ICLR*, 2022; arXiv:2105.04906.
