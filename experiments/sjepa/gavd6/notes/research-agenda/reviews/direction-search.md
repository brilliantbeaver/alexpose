# New direction search after withdrawing the P2 probability estimate

**Historical first-round decision.** The subsequent request to continue led to a [new synthetic teaching proposal](../proposals/synthetic-training-selection.md) and a [further independent review](synthetic-teaching.md). The conclusion below describes the earlier candidates, not the final outcome of the continued search.

**Decision, 14 September 2026: no new flagship cleared the independent review.** This is a research decision record, not a proposal to run another null experiment. No new experiments, checkpoint inference, video annotation, or HAIC jobs were performed. The previously quoted 40–55% probability for P2 remains withdrawn.

The request was to find a new, significant JEPA contribution using the existing GAVD and AMASS resources, eight H100s, and the remaining four days to the abstract and eleven days to the paper. Three independent agents examined methods, data and usable models, and scientific weaknesses. Candidates were revised and returned for a second adversarial review. The lead also searched related work and examined the existing study evidence.

The reviewers were not asked to approve a predetermined idea. Their objections changed the decision. One reviewer retained a reusable observation model as a bounded research hypothesis; another rejected it as the deadline recommendation. None endorsed a high-probability significant-paper claim.

## What the existing evidence supports

The [shared evidence record](../references/portfolio-evidence.md) remains the basis for this search. The future-feature study does not show a useful physical-motion benefit from JEPA distillation. The MoMask experiment does not establish useful tracking repair. The positive optical-flow contrast occurs in 15 of 16 paired cases, but only five pairs obtain both desired absolute preferences. It does not establish a JEPA-specific advantage or real-video repair accuracy.

These observations do not prove that a new JEPA method will fail. They mean that its positive mechanism cannot be treated as already demonstrated.

There are three separate requirements:

1. Additional observations contain useful information.
2. A proposed method uses that information more effectively, or enables a meaningful capability beyond established alternatives.
3. The effect is substantial and general enough to support the intended paper.

The earlier ideation repeatedly had a stronger argument for the first requirement than for the second. A better implementation or larger experiment does not automatically close that gap.

## Finalist A: correct noise bias in nonlinear JEPA targets

**Constructive mechanism.** A nonlinear encoder applied to a zero-mean noisy measurement does not, in general, average to the clean encoding. With Gaussian noise covariance `Sigma`, let `T` denote expectation over added noise. For an already noisy observation `y`, the target

`2 f(y) - E_eta[f(y + eta)]`

has expectation `(2T - T^2) f(x)`. Under suitable smoothness and a small covariance, it cancels the leading covariance-order bias. Averaging ordinary noisy targets reduces variance without removing this bias.

**Why it was rejected as a flagship.** The correction is related to established nonlinear measurement-error estimation and simulation extrapolation. [Stefanski (1989)](https://doi.org/10.1080/03610928908830159) studies unbiased nonlinear functions of a Gaussian mean and corrected scores. [Cook and Stefanski (1994)](https://doi.org/10.1080/01621459.1994.10476871) develops SIMEX. [Nonlinear Noise2Noise](https://arxiv.org/abs/2512.24794) already studies nonlinear-target bias, although its particular transform and loss choices differ from correcting an arbitrary neural encoder.

Exact inverse-noise correction can amplify high-frequency components and have unacceptable variance. More importantly, recovering the clean JEPA objective does not establish that this objective improves physical forecasting. The method would depend on both useful clean-target teaching and useful noise correction, neither of which the local results establish.

AMASS has clean motion references, so the missing-clean-target setting would be imposed experimentally. GAVD lacks clean 3D references, but also lacks a validated tracking-noise distribution. Different crops or trackers do not provide independent zero-mean measurement errors. The strongest theoretical assumptions and the real deployment problem do not currently coincide.

**Decision: reject for this deadline.** This could become a longer research program with real sensor-noise evidence, but it is not an endorsed next experiment.

## Finalist B: a reusable visual observation model for motion priors

**Constructive capability.** Keep a motion generator fixed. Learn an observation model that evaluates how well each candidate motion explains the video, separately from how common the motion was in training. Reuse that observation model with a different motion prior and a different pattern of observed frames.

This would be more substantial than accepting or rejecting one repair. It could support updating a distribution over possible motions as observations arrive, while retaining ambiguity when the video cannot distinguish the alternatives.

**A better starting component was found.** [SkeletonDiffusion](https://arxiv.org/abs/2501.06035) is a published stochastic human-motion predictor with native AMASS support. Its [official repository](https://github.com/Ceveloper/SkeletonDiffusion) links public [model files](https://huggingface.co/SkeletonDiffusion/ModelCheckpoints/tree/main) and an inference notebook. The release was inspected; its weights were not loaded locally. This is a useful resource finding, not evidence that candidate coverage is sufficient for the proposed study. Public pretraining may overlap the local AMASS people, so a checkpoint evaluation cannot automatically claim unseen pretraining identities.

**The statistical construction is sound but established.** Write `H` for shared observed history, `M` for an unknown candidate continuation, and `Z` for features of a complete selected observation set. A calibrated joint-versus-product discriminator can estimate

`s(M,Z,H) = log p(Z | M,H) - log p(Z | H)`.

For a replacement prior `q(M | H)`, weighting its samples by `exp(s)` approximates updating that prior with the observation. The interpretation requires adequate support and an unchanged conditional observation distribution. It concerns the encoded observations, not necessarily the full RGB information.

[Amortized neural ratio estimation](https://proceedings.mlr.press/v119/hermans20a.html) and [direct neural ratio estimation](https://arxiv.org/abs/2311.10571) already provide this operation. A generic neural ratio estimator using the same frozen features implements the central idea, so it is a primary predecessor rather than an optional ablation.

A proposed two-classifier correction was also checked. If full and history-only classifiers use exactly the same positive motion distribution and the same proposal sampler, their ideal logits can cancel the training-motion-to-proposal density ratio. This is correct conditional-density-ratio algebra. It is not a new theorem. Finite-data errors, saturation, inconsistent history inputs, and proposal-support mismatch can break the practical cancellation. Because the renderer can generate matched candidate-observation pairs, a one-classifier simulator-based ratio estimator is also a strong, simpler alternative.

**The human-motion precedents are close.** [Appearance as Reliable Evidence](https://doi.org/10.1016/j.cag.2025.104404), with a [public implementation](https://github.com/Zipei-Chen/Appearance-as-Reliable-Evidence-implementation), already combines appearance evidence with a generative motion prior. Its per-sequence optimization differs from an amortized latent observation model, but the central evidence-versus-prior principle is occupied. [Closed-loop Diffusion Planning](https://rfa-cldp.com/) already reweights sampled human futures using incoming observations without another diffusion pass. The project description also identifies the candidate-coverage limitation.

**Adversarial revisions.** Do not multiply scores for overlapping video windows. A whole-observation score is not an incremental likelihood; overlapping windows can count the same evidence repeatedly. Score the complete permitted observation set once at each update, unless a valid history-conditioned factorization has actually been learned. Never inject the true continuation into the main candidate bank. Never call squared JEPA prediction error an observation likelihood without a distributional model and calibration evidence.

**Why it did not clear the recommendation bar.** The remaining contribution would require substantial natural-motion and real-video advantages over geometric/flow likelihoods and ordinary ratio estimation using identical features. It would also require useful candidate support, nuisance robustness, and meaningful reuse across genuinely different priors. The local results establish none of these. The proposal therefore adds dependencies to P2's unresolved representation question.

**Decision: retain as an unendorsed research hypothesis, not the recommended deadline bet.** A large practical transfer result could make it publishable. The review supplies no basis to predict that result or to call this direction stronger than P2.

## Finalist C: replace repeated video encoding with predicted features

**Constructive mechanism.** Use motion prediction or feature transport between expensive encoder calls, and refresh when a cheap signal indicates that cached features have become inaccurate. This could reduce computation while preserving downstream motion measurements.

**Why it was rejected as a flagship.** [Eventful Transformers](https://arxiv.org/abs/2308.13494) already reuse video computation and update selected tokens. [EP-ViT](https://journals.sagepub.com/doi/10.3233/FAIA250790) uses motion vectors, residuals, cached features, and a degradation-aware refresh mechanism. A JEPA predictor could be an implementation choice, but the proposed general capability is not new. Gait-specific measurements would improve evaluation without by themselves establishing the conceptual contribution.

**Decision: reject as the new paper direction.** A practical speed improvement is plausible, but it is a different success criterion from the requested significant contribution.

## Broader routes excluded during the search

The search also found close work on several general principles that would otherwise sound attractive:

| General pitch | Close primary work | Consequence |
| --- | --- | --- |
| Preserve dense human information during predictive adaptation | [Human-JEPA](https://arxiv.org/abs/2608.21160) | Frozen anchors and forecasting masks alone are not a new direction. |
| Ground JEPA in physical state | [PSG-JEPA](https://arxiv.org/abs/2608.06799), [XP-JEPA](https://arxiv.org/abs/2608.24044) | Adding AMASS state supervision needs a materially different capability. |
| Make latent rollouts temporally consistent | [Semigroup-JEPA](https://arxiv.org/abs/2609.10464) | Multi-step consistency alone is insufficient. |
| Give latent distances a useful physical meaning | [Decision-Metric Alignment](https://arxiv.org/abs/2608.18746) | Physical metric calibration needs more than another probe or distance loss. |
| Maintain motion-aware memory | [Flow Equivariant World Models](https://arxiv.org/abs/2601.01075) | Motion transport and memory are already a substantial existing line of work. |

This is a targeted literature search, not a proof that no novel extension exists. A nearby paper does not automatically invalidate an application or a new empirical finding. In these candidates, however, the nearby method explains the main proposed mechanism, while the distinct practical advantage is still unmeasured.

## What the decision means

Do not launch a new training campaign on the strength of this ideation exercise. Do not replace the withdrawn probability with another numerical forecast or an impressive name. Do not imply that independent reviewers endorsed a promising flagship when their conclusions were conditional or negative.

There is a legitimate longer-term question about reusable visual measurement models, and a useful newly identified motion prior. Neither is evidence of a novel result within the remaining deadline. The current review therefore does not justify changing the research direction or escalating compute.

The unmet requirement is not more architectural complexity. It is a supported reason that the proposed capability will improve a consequential outcome beyond its strongest established alternative. Finding that reason remains open.
