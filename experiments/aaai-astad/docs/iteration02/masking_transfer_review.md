# What video and JEPA masking research can contribute to this gait study

Reviewed on November 15, 2026. This is a research and implementation proposal. No new gait model was trained, no primary artifacts were replaced, and no core code was edited.

**Recommendation:** retain the existing uniform 64-of-128 mask as the reference. The strongest small comparison is against a per-joint balanced random mask and a short temporal-block mask with exactly the same per-joint budget. This separates the effect of leaving enough own-joint evidence from the effect of hiding a coherent movement interval. Add complementary scheduling only as a coverage experiment. Treat stochastic positional embeddings as a separate regularization experiment, and defer learned adversarial masks until these simpler comparisons pass.

There is no verified universally best mask for gait anomaly detection. Image and video benchmark improvements motivate hypotheses. They do not establish which masks improve detection in GAVD, whose normal population is small and whose landmark observations are incomplete.

## First distinguish five different design choices

| Choice | What changes | What must not be confused with it |
|---|---|---|
| Input masking | Which observed joint-time tokens the predictor may read. | Removing an unreliable measurement because its confidence is low. |
| Target selection | Which hidden answers contribute to the loss. | Every token absent from the encoder necessarily being a scored target. |
| Multiple or complementary views | How masks are related across presentations of one window. | Receiving more training passes for free. |
| Stochastic positional embeddings | The predictor's positional query features. | Moving joints, choosing different mask locations, or sampling a physical future. |
| Dense or auxiliary prediction losses | Which visible and hidden representations receive supervision. | A new input-mask pattern. |

The missing-observation mask remains separate from all five. Sample the learning mask independently of coordinates, velocity, labels, and confidence. Then exclude unavailable observations and unavailable answers. Do not redraw until a convenient mask appears.

## What the primary literature actually supports

| Method and primary source | Verified mechanism | Transfer judgment for this project |
|---|---|---|
| [I-JEPA, Assran and colleagues, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Assran_Self-Supervised_Learning_From_Images_With_a_Joint-Embedding_Predictive_Architecture_CVPR_2023_paper.pdf) | Predicts several sizeable target blocks from a separately sampled context region, with overlapping target positions removed from context. Informative context and target scale are important to its image-representation results. | The useful idea is to prevent purely local completion shortcuts. A rectangle across numerical joint indexes is not an anatomical region. A skeleton adaptation must define temporal intervals and anatomical sets explicitly. |
| [VideoMAE, Tong and colleagues, NeurIPS 2022](https://arxiv.org/abs/2203.12602) | Repeats one spatial mask across time. Its pixel-reconstruction experiments support very high masking rates because videos are redundant. The [official generator](https://github.com/MCG-NJU/VideoMAE/blob/14ef8d856287c94ef1f985fe30f958eb4ec2c55d/masking_generator.py) confirms temporal repetition. | A joint tube means hiding that joint throughout the clip. It removes same-joint temporal evidence. This is useful for a separately named missing-joint reconstruction test, but is incompatible with the present own-joint eligibility rule. Do not transfer its 90–95% masking rate to 128 incomplete skeleton tokens. |
| [V-JEPA, Bardes and colleagues, 2024](https://arxiv.org/html/2404.08471v1) | Combines unions of smaller and larger spatial blocks extended through time. Its reported feature-learning comparison favors multi-block masks over random tubes and causal variants. Two masks share one teacher computation. | Large coherent omissions are credible alternatives to scattered omissions, but the result depends on the objective. “Short-range” and “long-range” in this recipe describe spatial block scales; they are not simply short and long temporal gaps. Reuse the principle, not its image-grid geometry or approximately 90% hidden fraction. |
| [V-JEPA 2, 2025](https://arxiv.org/abs/2506.09985), and [V-JEPA 2.1, 2026](https://arxiv.org/html/2603.14482v1) | V-JEPA 2 extends predictive learning to large-scale video and action-conditioned planning. V-JEPA 2.1 retains the earlier masking strategy and adds dense supervision of visible and hidden tokens, intermediate-layer supervision, and other changes. | V-JEPA 2.1 is current relevant evidence, but its gains do not identify a new superior mask. Dense supervision is a separate objective ablation. Keep hidden-target coordinate loss as the anomaly measurement; visible-token reconstruction is not an independent test of hidden prediction. |
| [VideoMAE V2, Wang and colleagues, CVPR 2023](https://arxiv.org/abs/2303.16727) | Dual masking gives the encoder and decoder smaller token subsets, reducing computation. Its [official code](https://github.com/OpenGVLab/VideoMAEv2/blob/29eab1e8a588d1b3ec0cdec7b03a86cca491b74b/dataset/masking_generator.py) separates tube and running-cell generators. | This is a decoder-efficiency method, not merely two complementary input masks. Our predictor has only 128 positions and one transformer layer, so decoder subsampling is low priority and would alter target exposure. |
| [CoMA, Li and colleagues, 2025 preprint](https://arxiv.org/html/2511.05929v1) | Uses complementary visible subsets in two branches, together with a particular model-update arrangement and a hierarchical transformer. | Complementary mask scheduling is an inexpensive idea to test. Do not attribute the paper's entire efficiency gain to complementation alone, or call a simple schedule a reproduction of CoMA. At a 50% ratio, a mask and its complement have equal budgets. |
| [ColorMAE, Hinojosa, Liu and Ghanem, ECCV 2024](https://arxiv.org/abs/2407.13036) | Generates data-independent structured masks by filtering random noise. It does not require a learned masking network. | This is a useful precedent for adding structure while remaining independent of movement intensity. A temporal or skeleton-graph filter is plausible. An image-plane filter across the arbitrary joint-index axis is not. Boundary handling and equal marginal coverage need verification. |
| [ADIOS, Shi and colleagues, ICML 2022](https://proceedings.mlr.press/v162/shi22d.html), and [CL-MAE, Madan and colleagues, WACV 2024](https://openaccess.thecvf.com/content/WACV2024/html/Madan_CL-MAE_Curriculum-Learned_Masked_Autoencoders_WACV_2024_paper.html) | ADIOS trains a mask function against representation agreement. CL-MAE learns masks with a curriculum that changes from cooperative reconstruction to adversarial reconstruction. | Both add a learned selection process and extra optimization. They are not the same algorithm. On this small corpus, hard examples may be tracking defects or unusual camera views. These are later ablations, with a fixed common evaluation mask bank and an explicit compute comparison. |
| [StoP, Bar and colleagues, ICML 2024](https://proceedings.mlr.press/v235/bar24a.html) | Adds Gaussian randomness to masked-token positional embeddings. Its formulation ties the noise map to the context projection to discourage a trivial zero-noise solution. The masking policy remains a separate choice. | A low-cost regularization ablation is feasible. It does not create a diverse physical motion model or calibrated uncertainty. The final published title is *Stochastic positional embeddings improve masked image modeling*, rather than the earlier OpenReview title. |

The official I-JEPA and V-JEPA sources were also inspected at pinned revisions: [I-JEPA multi-block collator](https://github.com/facebookresearch/ijepa/blob/52c1ae95d05f743e000e8f10a1f3a79b10cff048/src/masks/multiblock.py), [V-JEPA multi-block collator](https://github.com/facebookresearch/jepa/blob/51c59d518fc63c08464af6de585f78ac0c7ed4d5/src/masks/multiblock3d.py), and [V-JEPA 2 multi-sequence collator](https://github.com/facebookresearch/vjepa2/blob/204698b45b3712590f06245fbfba32d3be539812/src/masks/multiseq_multiblock3d.py). They distinguish encoder and predictor selections. Their batchwise trimming and variable block unions must not be copied without auditing their effect on skeleton target coverage.

## A concrete laptop-feasible comparison

The current window contains 32 samples at 15 Hz, spanning 31/15 = 2.07 seconds. A token contains four consecutive XY samples of one joint. There are eight temporal positions and 16 joint positions, giving 128 tokens. Keep exactly 64 learning-mask tokens hidden in each main comparison.

The allowlist remains **11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32**. No face, hand, or additional landmark enters a mask or input. The primary lower-limb readouts remain 25–32; this does not reduce the training allowlist.

| Proposed arm | Exact construction before measurement validity | Question it isolates |
|---|---|---|
| U: existing uniform reference | Choose 64 of the 128 positions uniformly without replacement. | Preserve the completed protocol's sampling reference. |
| B: balanced random | Independently choose four of eight temporal positions for every joint. | Does reliable availability of own-joint history help learning? |
| T: balanced short temporal blocks | Partition time positions into `[0,1]`, `[2,3]`, `[4,5]`, `[6,7]`. For each joint, uniformly choose two of these four blocks to hide. | Compared with B, does hiding adjacent movement samples reduce easy temporal interpolation? |
| C: complementary scheduling, optional | Present a mask and its Boolean complement in a paired schedule. Compare against two independently generated masks for the same window under the same presentation budget. | Does finite-sample exposure balance improve stability or efficiency? |

U, B, and T all hide each joint-time position with marginal probability 0.5. B and T additionally leave exactly four visible positions per joint before missingness. T sometimes joins adjacent selected blocks into a longer gap, which must be reported in its gap histogram. Fixed two-position boundaries are an intentional limitation of this first simple design.

For one illustrative joint, `X` means hidden and `O` means visible. These are patterns, not measured gait results.

| Pattern | Temporal positions 0 through 7 |
|---|---|
| One balanced random draw | `X O X O O X O X` |
| One short-block draw | `X X O O X X O O` |
| Its complement | `O O X X O O X X` |
| Whole-joint tube | `X X X X X X X X` |

B is a necessary control for T. Even with complete observations, under U a hidden target has fewer than two visible same-joint patches with probability

```text
sum_{k=0,1} C(7,k) C(120,64-k) / C(127,64) = 0.05485654.
```

This is an exact combinatorial calculation, not an experimental result. B and T remove that particular failure before observation missingness. Therefore, a T-versus-U gain could arise from better context availability rather than temporal structure. The T-versus-B comparison addresses that alternative.

Do not sample a four-position contiguous interval uniformly over starts 0 through 4 and call every position equally likely. Its hidden probabilities are `0.2, 0.4, 0.6, 0.8, 0.8, 0.6, 0.4, 0.2`. If a future proposal uses these intervals, declare the unequal probabilities or balance exposure explicitly. Circular manipulation of mask indexes is different from circularly wrapping motion data; never wrap the observations to fabricate a walking transition.

A mixture of U and T may eventually combine easy and difficult tasks, but first compare the pure arms. Otherwise a favorable mixture hides whether target support or temporal difficulty mattered. Anatomical multi-block masks should likewise follow, with an explicit list of anatomy groups and a proof or measurement of each joint's inclusion probability.

## Why whole-joint and whole-leg tubes need a separate endpoint

The current `ContextConfig` requires two valid visible patches of the target joint. Under a complete joint tube, all eight patches of a selected joint are hidden. Every corresponding target therefore fails the existing own-joint condition. Relaxing that condition changes the question; it cannot be a silent implementation adjustment.

A future own-leg predictor can sometimes reconstruct a fully hidden ankle from a visible knee and heel. That is a legitimate geometry experiment, but it does not provide the historical own-joint comparator. Hiding an entire target leg also removes the local baseline on which the retained-baseline proposal depends. Use such masks only for clearly separate missing-part tests, not to manufacture a larger benefit from the opposite leg by denying the local model its required evidence.

Likewise, a future-suffix mask is a forecasting task. A model using later visible frames to complete an interior gap is an interpolation/completion model. Neither the word “JEPA” nor a temporally contiguous mask turns it into prospective forecasting.

## Keep the comparison scientifically fair

1. **Separate training policy from evaluation policy.** First change training masks only and score every trained arm with the same frozen, historical 16-mask bank. This keeps target identities, context challenges, source eligibility, and readout weights common. Then evaluate all arms on one prespecified short-block challenge bank as a secondary test. A train-policy by evaluation-policy table reveals whether a model merely specializes to its own mask distribution.
2. **Keep information access identical within each contrast.** Local, own-capacity, both-leg, random-encoder, and ridge alternatives must receive the same permitted observations and identical hidden targets for a particular draw. Draw before considering validity. Never reveal a hidden coordinate through a full-context embedding, phase estimate, donor hip, or mask-learning decision.
3. **Keep the cohort and missingness rule fixed.** Record valid context counts, requested and scored targets, unique joint-patch coverage, gap lengths, per-source counts, and no-context/no-target failures. Training masks can change usable supervision despite equal nominal budgets. Report this; do not hide it by replacing difficult windows. If a challenge bank cannot support a source, disclose its separate coverage instead of silently redefining the primary population.
4. **Match presentations and optimization.** Keep source sampling, example order, initialization, optimizer updates, EMA updates, loss normalization, and encoder/predictor token budgets fixed. Compare two complementary views against two independent views, not against one. Teacher reuse can save work, but two student forwards still cost work. Report actual time, memory, and supervised-target exposure rather than claiming equal compute from equal epoch counts.
5. **Freeze target and score semantics.** Keep the coordinate target and normal-reference scoring recipe constant. JEPA's contextual teacher can change with the encoder, so latent loss alone cannot isolate the value of a mask. Do not tune the mask, ratio, gap size, or score sign on abnormal labels or outer-test discrimination.
6. **Use normal development for learning adequacy.** Test known-dependency fixtures, local-prediction adequacy, residual usefulness, and sensitivity to aligned versus shifted opposite-leg inputs. A harder reconstruction loss is not automatically a better representation. Artificial mask difficulty is not a simulated neurological disease.

Complementary evaluation is attractive for coverage: eight 50% masks and their complements give every token exactly eight requested exposures across 16 draws. But this changes the historical bank and potentially source eligibility. Use it as a separately named sensitivity, with a new bank hash. Repeated exposure is not a new independent target or participant.

Content-adaptive masks need an additional safeguard. A selector that reads a hidden answer can encode information about that answer in the pattern of visible positions. It also changes the sampled target distribution. For a future learned-mask arm, train the selector only on fitting-normal sources, freeze it before evaluation, and retain the data-independent common evaluation bank. Do not call its training distribution uniform or claim that confidence-independent selection has been preserved.

## What the current implementation would need to change

These are planned changes, not edits made in this review. Prefer an additive `aaai_gait/masking.py` and a new protocol version. Preserve the old functions' default behavior so draft01 can still be reproduced.

| Current location | Required planned change |
|---|---|
| `model.py:84`, `uniform_mask` | Retain it unchanged as the reference. Add an explicit policy interface returning Boolean `[B,128]` masks and a serializable policy description. Test U, B, and T cardinality, marginal coverage, and device determinism. |
| `model.py:115`, `_visible_indices` | Preserve equal nonzero visible counts. Variable unions from image multi-block generators violate this contract unless explicitly padded or budgeted. The first proposed arms all keep 64 visible positions. |
| `model.py:276`, `SJEPA.loss` | Pass the policy to both the main draw and the VICReg view draw. Currently, supplying the first mask manually still leaves the second view uniformly sampled. A one-line replacement therefore does not specify the complete training intervention. |
| `model.py:296`, second-view randomness | Separate the VICReg mask stream and coordinate-noise stream. They currently share the main masking generator. A policy that consumes a different number of random values would otherwise change later masks and noise. Keep a separate positional-noise stream as already implemented. |
| `training.py`, `train_sjepa`, and `model.py`, `latent_residual` | Carry the policy through the general tutorial training entry point. Keep the residual scorer's fixed evaluation bank explicit; it currently constructs uniform masks internally. Do not accidentally change both training and evaluation through one global switch. |
| `context_prediction.py`, `fit_context_models` and `_head_summary` | Route pretraining and head-training masks through declared policies. Keep validation banks fixed. Separate sampled target-joint RNG from mask RNG; they currently share `head_rng`, so sampler changes can alter which joint is trained next. |
| `context_prediction.py:397`, `make_mask_bank` | Preserve the old bank by default. Let new evaluation protocols supply an explicit bank and its hash; do not regenerate from an ambiguous mask ID alone. |
| `linear_context.py`, `fit_ridge_context` and `score_ridge_context` | Give ridge the same declared training policy for matched comparisons and the exact scoring bank. Its fitting draw is presently hard-coded to `uniform_mask`. |
| `predictive_profile.py`, `freeze_profile_eligibility` | Retain historical eligibility. Create a separately named coverage audit for each secondary bank. This function presently enforces 16 masks, two same-joint context patches, and historical readout coverage. |
| `study_runner.py` and saved provenance | Record training policy, view policy, bank hash, target counts, RNG streams, code hashes, and compute. Reject checkpoint/cache combinations with a different policy identity. |

The lightweight encoder already gathers visible tokens before attention. Preserve this behavior. The implementation's zeros for unavailable coordinates are numerical placeholders whose attention keys are excluded, not an imputation method that makes missing motion observed.

## StoP deserves a separate, small experiment

The current `SJEPA.forward` adds `spe_sigma * context_projection(noise)` only at hidden predictor queries. The same projection maps context features, which follows the central weight-tying idea in the [StoP formulation](https://arxiv.org/html/2308.00566v2). It does not jitter XY coordinates or choose a different mask. In this implementation, `spe_sigma` multiplies standard-normal samples, so it is a standard-deviation multiplier; do not copy a paper's variance parameter without checking that convention.

There is also an important configuration limit: `ContextConfig.model_config()` currently leaves `spe=False`, and the context-study coordinate heads do not use the JEPA predictor. Enabling StoP elsewhere in a tutorial therefore does not mean that the completed context study used it. A future experiment must explicitly enable it during encoder pretraining, save that setting, and compare against the same masks, data, and random seed streams without StoP.

For short skeleton sequences, position carries anatomical identity and precise time. Deliberately making those features less precise may help general representation learning while harming the very timing sensitivity being tested. Require a normal-prediction and alignment-sensitivity check before interpreting a downstream gain. Inspect noise scale and representation variance. Do not present samples caused by query noise as calibrated predictive intervals or distinct plausible physiological futures.

## Decision for the coming implementation

Implement mask-only checks and U/B/T support censuses first. They require no neural training. If T and B preserve adequate normal support, run a small, matched normal-development comparison before committing to the full source folds. A simple failure to learn the controlled dependency is a stop signal, not a reason to increase masking severity.

For the deadline, the strongest possible result is an explanation of **when structured omissions force useful cross-leg information instead of local interpolation, while preserving enough local evidence for an honest comparator**. Changing a standard mask alone is not substantial novelty. A credible gain must survive the balanced-random control, common evaluation masks, simple predictors, observation-quality checks, and the unchanged real-source evaluation. Learned mask generators, multiple masking ratios, dense-loss redesign, and large pretrained video backbones should not all enter the same experiment.

## Consolidated implementation choice

The canonical instructions select U/J/A: uniform, a single-joint eight-token core, and a width-balanced anatomical eight-token core, each filled to 64 hidden positions. These laws have matched marginal probabilities and keep the primary U evaluation bank fixed. The balanced per-joint and temporal-block variants reviewed above remain alternative designs, not additional mandatory arms. The separate RNG, VICReg-view, support and exposure safeguards apply to the selected laws.
