# Independent masking audit for Gait Fidelity

Inspected 2026-09-21. This is a code and small deterministic-check audit, not a new model-training result. The sibling multiple-sclerosis project was read only. All checks used the existing `gavd6/.venv/bin/python`, with `PYTHONDONTWRITEBYTECODE=1`.

## Main finding

The suggested technique is worth a controlled experiment, but Gait Fidelity should describe the intervention accurately: **compare different stochastic mask structures**, rather than claiming that its predecessor still permanently hides fixed clinical joints. The multiple-sclerosis notebook documents a repaired fixed-mask defect; synthetic-training-v2 already varies masks across examples and updates.

## What the notebook and implementation do

`../../multiple-sclerosis/02_anatomical_mask_and_tokenization.ipynb` (zero-based notebook cells):

- Cell 1 explains the old fixed 12-joint mask and context starvation; cells 10–11 define the anatomical prior. It is BlazePose-33: both shoulders, hips, knees, ankles, heels, and foot indices. This twelve-joint *subset of 33* differs from synthetic-training-v2's body12 schema.
- Cells 7–9 define one token as four frames of one joint. Laptop configuration: 32 frames, 33 joints, eight time blocks, 264 tokens. Cells 12–13 explicitly distinguish per-window coverage from exact joint/time coverage.
- Cells 16–17 make an honest visualization: one motion sequence, eight newly sampled masks, separate mask-draw and time-block clocks, all joint/time bits in a timeline. The seed-zero first draw can hide both legs for the entire window.

`sjepa/masking_v2.py:43–51` defines seven fixed connected anatomical groups (head, mouth, two arms, trunk, two legs). The *groups* are fixed but their selection and time intervals are sampled. This is stochastic anatomical-region/time masking, not a learned graph and not a graph neural network.

`masking_v2.py:84–105` approximately corrects repeated membership in groups. A clinical boost applies to any group containing a clinical joint, including both arms because they contain shoulders. Its `clinical_bias=1.5` is a region-selection weight, not a final joint-level mask-probability ratio.

`masking_v2.py:108–165` samples whole regions over contiguous spans, adding their union until approximately 60% of tokens are hidden. A span is at most 75% of the window, but overlapping spans can hide a joint for its entire window. It guarantees one clinical-set context token somewhere, which may be a shoulder; there is no lower-limb or per-block context guarantee. The nominal ratio is not an exact budget.

`masking_v2.py:168–189` returns independent per-example masks using an explicit NumPy RNG. `train_v2.py:205–220` samples new masks each update, makes student augmentations, and predicts teacher features at masked positions. Diagnosis labels are ignored in this training loop. Multiple masks per sequence repeat the input and increase compute; this requires budget matching in any comparison.

## Mask coverage measured locally

Re-executed `mask_bank_stats(33, 8, n_masks=512, seed=0)` with defaults. Receipt: `/private/tmp/gait-fidelity-masking-stats.json`.

| Check | Result |
| --- | ---: |
| Configured / realized target fraction | 60% / 63.276% |
| Minimum over joints, context somewhere in window | 74.02% of draws |
| Minimum over joints, target somewhere in window | 80.86% of draws |
| Left / right hip hidden for entire window | 21.48% / 22.85% of draws |
| Left hip context by exact time block | 51.56%, 22.66%, 10.35%, 9.57%, 7.03%, 9.77%, 22.46%, 49.22% |

The implementation passes its loose window-coverage rationale while strongly undersampling hip context in middle blocks. This is a useful illustration of why every-joint-ever-visible is insufficient. It is not evidence that the trained representation fails or that a particular revised sampler performs better.

## Token leakage and information boundaries

`sjepa/models.py:113–134` tokenizes every joint/time slot, applies key-padding so context queries cannot attend to hidden tokens, and replaces hidden outputs with shared mask tokens before prediction. Its LayerNorm and feed-forward operations are token-local. `PredictorV2` adds joint/time position tags (`161–200`), and teacher processing is detached (`282–297`). A small real-model check changed all artificially hidden post-normalization input values by large random amounts: maximum change in the returned context representation was **0.0**. This supports the local masking implementation, under that test, rather than assuming that tokenization of the full array leaks hidden values.

Preprocessing is a different boundary. `sjepa/data.py:164–181` computes framewise pelvis and torso normalization *before* masking, using hips/shoulders that may later be hidden. A deterministic perturbation of hidden hip coordinates changed retained wrist coordinates by up to **0.2303 normalized units** in a synthetic check. Thus the stronger claim “no hidden-coordinate information enters context” is false at the preprocessing level. Whether this matters depends on the declared task: a training augmentation may legitimately start from the full observed input, whereas simulating truly unavailable joints needs context-derived normalization. It is not leakage of independent clinical ground truth; it is pretext-task information passing through normalization.

`data.py:130–160` says “short gaps,” but the actual implementation fills every finite-bounded gap without a length cap, interpolates the confidence channel, and fills entirely absent channels with zero. A ten-frame gap was filled in a 20-frame synthetic check. Preserve raw coordinates, original timestamps, observed flags, and interpolation flags for Gait Fidelity, and do not use this cache as dense reference truth.

`sjepa/augment.py:40–87` applies student-only random rotation, scale, translation, and reflection with joint-slot swapping; teacher gets the original sequence. For side-specific restoration, do not inherit this invariance objective without an explicit transform contract. Record the transform, move coordinates/confidence/validity/masks consistently, and compare in a shared coordinate/side convention. Anatomical side naming, camera reflection, and physical movement mirroring need separate definitions. Unrecorded student reflection plus original teacher targets can encourage loss of the sign being studied.

Raw confidence, natural absence, and artificial training hiding are three different concepts. MS's normalized cache contains interpolated values and has no separate raw-observed tensor in model input. Gait Fidelity's existing schema already has a stronger separation.

## Existing synthetic-training-v2 behavior

`src/gavd6_sjepa/research_directions/synthetic_training_v2/training.py:189–206` uses a random joint order plus a cyclic time-block start, masks only patches with an observed input frame, and matches a target budget over observed patches. With complete 64-frame body12 input, patch length four, and 50% masking, a direct check found **eight of 16 whole-body blocks hidden, zero partly hidden blocks**. At complete support, random joint ordering does not change this particular budget. A cyclic interval may straddle the window boundaries; it is not always a single contiguous interval in the displayed linear window.

`models.py:105–125` replaces hidden coordinates/confidence before projection, retains joint/time query identities, and permits missing positions as output queries. It does not remove their parameter gradients in the same way as the old MS fixed-mask architecture. Do not generalize the MS fixed-mask starvation proof to this architecture.

`training.py:407–426` combines artificial masks with natural missing-input queries; target validity gates loss after the mask is sampled. Target visibility/validity must never drive the student mask. `training.py:455–461` already records mask/query hashes and realized fractions. Extend this provenance rather than replacing it.

At only one observed patch, the existing mask can hide that sole patch. A proposed context guarantee needs an explicit degenerate-input policy: no artificial hiding when insufficient context exists, then retain and report naturally uninformative windows. Never manufacture observed joints or silently remove all hard cases.

## Body12 mapping and porting constraints

Existing order (`contracts.py:10–15`): shoulders 0/1, elbows 2/3, wrists 4/5, hips 6/7, knees 8/9, ankles 10/11.

Candidate graph regions:

- left arm `(0,2,4)`; right arm `(1,3,5)`;
- trunk `(0,1,6,7)`;
- left leg `(6,8,10)`; right leg `(7,9,11)`.

Edges must be explicit and schema-versioned. Body12 has no heels or toes. Missing feet cannot be manufactured by remapping indices or claimed as observed contact measurements. A direct call to the sibling sampler with 12 joints raises `IndexError: index 12 is out of bounds for axis 1 with size 12`.

Blindly porting `clinical_bias` also fails conceptually: every body12 region above touches at least one shoulder/leg joint, so multiplying every region by 1.5 cancels when probabilities normalize. If a leg preference is tested, state separate region weights explicitly and preserve identical left/right marginal policies.

## Recommended bounded experiment

1. **Primary 2×2 mechanism comparison:** current stochastic time blocks versus unbiased graph-time masks, crossed with `coordinate` versus `paired_jepa`. The `coordinate` arm is masked coordinate pretraining followed by a frozen encoder/new readout; it provides the stage-matched alternative to feature prediction. Existing `direct` is end-to-end and never calls `_mask` (`training.py:158–163`), so retain it as a practical benchmark, not as if it automatically receives the masking intervention.
2. Use identical source splits, batch draws, paired targets, target-support rules, architecture, readout budget, seeds, and eligible-token masking budgets. Match realized rates and duration distributions, not just the number 0.5 in two configuration files. Record group-connectivity after intersection with naturally observed inputs; missing input can break a region's observed connectivity.
3. If the primary contrast is worth pursuing, add independent random-token masks to distinguish generic stochastic hiding from structured hiding, then a mild bilateral leg-bias ablation. A mask with shuffled anatomical memberships but matched group sizes/spans can isolate the anatomical prior. Fixed clinical masking is a historical failure demonstration, not the primary modern comparator.
4. Do not use motion magnitude, diagnosis, affected-side labels, reference error, or reference visibility to choose the main mask. Motion-aware masking is a relevant ablation; the sibling code's assertion that it is “contraindicated” is stronger than its evidence. High-motion selection can undersample low-motion targets, but that does not establish harm without comparison.
5. Evaluate on unchanged independent observations and on held observation-gap families. Artificial holes used to train a model are not clinical evidence. Test reference-verified unequal left/right motion, low-amplitude movement, true movement changes, identity swaps, and long/short observation gaps without forcing bilateral symmetry.
6. Select the policy on development data, freeze it, and confirm on new source people and nuisance families with paired person-level effects and crossed training-seed uncertainty. Keep coordinate accuracy and nuisance resistance guardrails alongside the prespecified gait-response endpoint. No promise that masks produce a significant result.

## Required implementation checks before a training run

- Sampler seed/resume reproduces masks; different examples/updates vary; model-independent mask traces are shared across objective comparisons.
- Exact mask budget, no-target/no-context cases, and naturally missing patches are explicit.
- Per-joint × exact-time context and target frequencies, side balance, maximum hidden runs, and query-distance distributions are plotted. Report context-availability denominators separately from reference-valid loss denominators.
- Perturbing hidden input values cannot affect student predictions after the declared preprocessing boundary; perturbing privileged reference data cannot affect masks or student normalization.
- Every anatomical slot receives useful context exposure across the bank. Nonzero query-embedding gradients alone do not prove that a joint has ever been observed as context.
- Permuting/reflection-transforming joints moves confidence, validity, masks, and side metadata together; inverse-transform tests recover original signed quantities.
- Teacher receives training references only; downstream readout inputs remain deployable observations. Graph policy names and graph schema join checkpoint signatures and output manifests.

Small-check receipt: `/private/tmp/gait-fidelity-masking-checks.json`. No experiment code was modified and no new neural model was fitted.
