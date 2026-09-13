# Adversarial review of the optical-flow replacement

Independent reviewer: evidence-audit agent. Review date: 12 September 2026. Reviewed `07-motion-beyond-keypoints.md` after the world-model agent's draft; applied the substantive revisions below directly. No body-model, optical-flow or forecasting experiment has been performed.

## Verdict and ranking

The replacement is stronger than the old constraint-conflict reserve. Agree with the integrated order: **1, 7, 2, 6, 5, 3, 4**, provided readers understand what that ranking means. Proposal 7 earns second place for conceptual upside, an inexpensive disconfirmation test and compatibility with Proposal 1's observation assay. It does not yet have a higher demonstrated probability of useful natural forecasting gains than Proposal 2 or Proposal 6.

Keep conditional novelty at 3/5 and feasibility at 3/5. Significance and visual explanatory potential can reasonably be 5/5 if the required results occur. A visible twist example is highly feasible; a useful new forecasting principle has not been established. No numerical ICLR acceptance probability follows from these ordinal judgments.

## Failures found and applied edits

| Failure or loophole | Revision applied |
| --- | --- |
| An ideal axial twist can leave forward-kinematic joints unchanged but alter the actual surface-regressed landmarks or checkpoint channels. | Preserved full-input verification, added a numerical tolerance and outcome-separation requirement, linked the algebra script explicitly as an ideal-chain check, and fixed the input boundary to 22 joint positions. Rotations are targets or labeled ceilings only. |
| A generated alias might exploit arbitrary pose ranges or pose-blend artifacts. | Required declared angle/velocity ranges and stated that kinematic validity is not human biomechanical validation. The hour-24 stop includes rendering artifacts and actual-input mismatch. |
| A hidden edit label or constant-speed twist is an easy classifier/extrapolation task. | Kept constructed pairs as diagnostic fixtures. Forecast heads train on unedited motion, and the primary study tests unedited held-out trials. Added angular-velocity extrapolation and extra off-axis landmarks. |
| The draft could imply that alias construction proves a natural-world forecasting problem. | Added the distinction between pairwise information loss and its unknown prevalence or utility in natural motion. |
| AMASS mesh trajectories could be mistaken for direct measurements of skin motion or clinical twist. | Explicitly described fitted body parameters and model-derived rendered references, citing AMASS. The claim remains a controlled representation study. |
| An offline tracker or flow pair straddling the cutoff can inspect future pixels. | Required every flow pair to end before the cutoff and every offline tracker to process only the prefix. |
| A method could select easy future-visible points or abstain on errors without paying a coverage cost. | Fixed anatomical query points using cutoff visibility, common masks across methods, separate visible/occluded results and coverage, every eligible point in the primary comparison, separate abstention calibration. |
| A global displacement target could be explained by root drift. | Specified pelvis-relative point displacement normalized by body height, with absolute displacement separately reported. Added local segment geodesic orientation error. |
| Four choices of K do not identify a mathematical information minimum. A token may silently contain an arbitrarily large tensor. | Replaced the minimum claim with an empirical budget among tested encodings. Fixed token dimension and time history; added bytes, downstream computation and separate full flow extraction cost. |
| Generic flow-plus-skeleton reasoning was framed as less crowded than it is. | Added H-Flow as a direct related prior and simulated-correspondence neighbor, alongside H-MoRe and prior articulated flow work. |
| An excessive reproduction penalty trivially deletes all real motion. | Required development-fixed weights, noise-removal and motion-retention results, and testing tolerance explicitly. Unavailable original weights cannot be described as empirical failure of the original model. |
| Camera and appearance controls omitted false motion on a stationary body. | Added moving texture, repeated patterns, static person with moving camera and the limitation of homography compensation. |
| A hundred aliases could be repeated variants of one source clip. | Required distinct source windows plus counts of people/trials; grouped subjects and variants; people-level bootstrap where identity is known. |
| The latest failed increment could be used as proof of this proposed mechanism. | Added actual run increments and explicitly denied that causal inference. |

## Remaining scientific risks

**The strongest baseline may already solve it.** A full-RGB mesh estimator followed by a simple orientation predictor may retain this state cheaply. More off-axis landmarks may be a simpler interface. A residual-token method needs meaningful robustness or efficiency beyond these comparisons. Frozen RGB features with enough capacity may also absorb the information without any explicit flow representation.

**The reference may contain fitting conventions.** Natural AMASS histories are unedited, but their body rotations were still inferred by model fitting. A small model can learn stable fitting patterns without recovering a clinically relevant latent variable. Broadening to another acquisition source or body fitting method is more informative than adding many render variants. That additional corroboration must fit the week or be stated as a future limitation, not silently assumed.

**Observability differs from predictive usefulness.** Dense surface motion can distinguish two past histories while adding no information about their future. The manufactured fixture cannot resolve this. The natural-motion pilot must measure actual future outcomes against strong prefix-only extrapolation and direct flow heads.

**A useful prior can have deliberate exceptions.** A skeletal guidance term may penalize the constructed motion while improving aggregate estimation. The claim should be a measured tradeoff under relevant conditions, not that every such constraint is invalid. H-MoRe's implementation and stationary-joint handling are not verified because its author link failed. H-Flow's complete model is not reproduced by evaluating one equation.

**A compact state is not cheaper input acquisition.** SEA-RAFT still processes image pairs; tracking adds cost. The result can claim compact downstream prediction state if supported, but total inference must include extraction. A skeleton-only deployment cannot recover information that was never observed merely through distillation.

## Exact claim ladder

1. Verified finite pairs prove loss under the specified input map and tolerance.
2. Estimated prefix flow recovering their actual rendered outcomes shows that the extra observation can help in that controlled setting.
3. Unedited held-out future gains show practical utility on the declared body-model distribution.
4. Beating direct fusion, extrapolation, extra landmarks and RGB mesh baselines supports the proposed method rather than the general value of additional information.
5. Transfer to new movement, rendering and acquisition conditions strengthens the case for a general predictive-state principle.

Stop at the highest rung actually supported. The proposal currently offers a plan to climb this ladder, not evidence that its upper rungs are reachable in one week.

## Review links

- [Revised Proposal 7](../../proposals/motion-beyond-keypoints.md)
- [Independent optical-flow candidate audit](optical-flow-candidates.md)
- [World-model literature and release audit](optical-flow-world-models.md)
- [Prior front-runner review](adversarial-front-runners.md)

Visual review of the replacement diagrams is assigned to the world-model agent. This report does not claim to have re-inspected the newly rendered Proposal 7 images.
