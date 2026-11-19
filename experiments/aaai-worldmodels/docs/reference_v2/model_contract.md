# Compact forecasting and canonical mask contract

This implementation belongs to `medwm-reference-masking-v2`. It adds new modules, `worldmodels/v2_models.py` and `worldmodels/v2_masks.py`, while preserving all historical implementations and completed runs. The public model API implements a coordinate-prediction stage. It neither selects folds nor evaluates abnormal labels. Model selection, nested calibration, and the final anomaly endpoint belong to the separately reviewed evaluation runner.

## What the predictor receives and produces

The inference boundary is `predict_forecaster(checkpoint, prefix, observed, times, hidden=None)`. It accepts exactly `[N,16,16,2]` normalized prefix coordinates, Boolean natural observation support `[N,16,16]`, and increasing actual decoder times `[N,16]`. Supplying a 24-sample full window is rejected rather than silently exposing future information. Natural missing coordinates may be NaN because they are erased before coordinate operations. A deliberately hidden coordinate may also be NaN at inference and cannot affect the output once the hidden mask is fixed.

The output has shape `[N,8,8,2]`. Joint order is `(25,27,29,31,26,28,30,32)` in BlazePose IDs, which places the four left-leg targets before the four right-leg targets. Outputs always request the same eight future times and joints, independently of whether future observations will be available. The caller applies a shared observation mask when computing loss or error. Normalized coordinates describe relative 2D articulation; actual prefix time differences are measured in seconds.

`fit_forecaster(prefix, observed, times, future, future_valid, metadata, config)` receives outcome arrays through distinct fitting-only arguments. Future coordinates may have shape `[N,8,16,2]` in the full declared joint order or `[N,8,8,2]` in target order. Target residuals subtract each target joint's latest visible prefix coordinate. Unknown target anchors invalidate corresponding future targets. Every training window must have at least one measured future coordinate for each leg; the caller's stricter common-support gate remains authoritative.

Metadata must contain source ID, sequence ID, and `label=normal` for every training row. Any abnormal label is rejected. Optional `group_id` identifies the audited grouping component. The current accepted cohort has one source per component. The runner must reject a multi-source component until an explicit component/source/sequence/window weighting extension is reviewed; this implementation uses group/sequence/window weighting and must not silently reinterpret a merged component. Optional `partition`, when supplied, must be `train` for all rows. The caller must provide only the permitted training groups; the API records their identity but cannot infer whether a supplied group was externally held out.

## Three compact prediction modes

| Mode | Information used | Implementation |
| --- | --- | --- |
| `ridge_posture` | Current eight-joint target posture and whether each anchor is known. | A 24-dimensional feature vector and output-specific masked ridge fits. This is a new compact reference, not a relabeled historical static ridge checkpoint. |
| `ridge_posture_velocity` | Current target posture plus recent robust velocity, velocity dispersion, and observation counts. | A 64-dimensional feature vector. Velocity is the coordinatewise median of valid adjacent-pair slopes among the last five prefix intervals. Dispersion is median absolute deviation from that median. The measured-pair count is retained separately. |
| `mlp` | Relative full-prefix articulation, actual visibility, current anchors, known-anchor flags, and actual prefix times. | An 832-dimensional feature vector, two 32-unit GELU hidden layers, and 128 residual outputs. The model has 31,936 parameters. The output head begins at zero, so its initial forecast is persistence. |

For MLP inputs, coordinates are causally forward-filled only after deliberate hiding. Before the first visible observation, a joint's stored coordinate is zero, with visibility and known-anchor indicators retained. Each temporal coordinate is expressed relative to that joint's latest visible anchor. Features concatenate 512 relative coordinates, 256 visibility bits, 32 anchor coordinates, 16 known-anchor bits, and 16 times relative to the last prefix observation, divided by 1.5 seconds. Actual time features do not claim exact future physical-time sampling.

MLP standardization is fit only from causally filled permitted-training feature banks. These features contain no per-bank hidden coordinate. Reference and anchor protection means the normalization and residual origin remain available under every canonical corruption. Feature means and scales are computed with equal group, sequence, and window weights; near-constant scales are set to one. Ridge standardization uses the same permitted-training hierarchy.

## Canonical mask policies

An observed token contains one joint's xy coordinate pair at one prefix sample. The mask arrays keep natural validity, deliberate corruption, visible context, and protected observations distinct. Every observed shoulder or hip used by the normalization is protected: BlazePose joints 11, 12, 23, and 24. Each joint's latest observed prefix sample is also protected, even if it occurs before the last prefix frame.

M1 and the structured/secondary policies hide `floor(0.4 * eligible_count + 0.5)` eligible observations, capped only to leave support when fewer than two eligible tokens are available. Unsupported windows fail explicitly. M0 applies no artificial prefix corruption. No policy receives future values, future availability, labels, or prediction errors.

| Policy name | Arm | Exact behavior |
| --- | --- | --- |
| `none` | M0 | Retain every naturally observed prefix token. |
| `uniform` | M1 | Randomly permute eligible tokens and hide the exact required count. |
| `time_blocks` | M2 | Randomly order four-sample contiguous time blocks over non-protected joints. Hide their observations in that priority order until reaching the exact count. |
| `anatomical_tubes` | M3 | Randomly order four-sample blocks of left distal leg, right distal leg, left arm, and right arm. Group sizes are 4,4,2,2. |
| `motion_balanced` | M4 | Reserve at least 20% of the corruption budget for uniform draws. Split remaining measured-motion tokens into low/high within-prefix ranks, shuffle each stratum, and alternate priorities, with the first stratum randomized. |
| `random_groups` | M3 matched control | Shuffle the twelve non-reference joint IDs into a partition with sizes 4,4,2,2. Keep that partition through the same four-sample temporal blocks. Match the anatomical arm's exact total hidden count. |
| `motion_focused` | M4 secondary control | Reserve the same uniform fraction. Sample measured-motion observations without replacement using bounded rank-based weights between one and two. |

Structured priorities can truncate their final block to meet an exact budget. The audit records this. Thus these are exact-budget tube/block priorities, not an assertion that every realized mask is a complete intact tube. Natural missingness and protected anchors can also change the realized count in any candidate block. Random-group and anatomical masks match group-size and time-extent distributions and final token counts; their realized partial-block geometry is reported rather than assumed identical.

Motion is measured only from adjacent prefix pairs whose coordinates were both observed, divided by their actual time interval. No valid pair means unknown motion, represented by NaN, rather than measured zero velocity. Stable rank sorting handles ties without consulting outcomes. The uniform component gives unknown-motion observations positive inclusion probability. Motion ranks are estimated within the available prefix, so no global threshold uses evaluation outcomes. This is an explicit adaptation, not a reproduction of MAMP, ASMa, or SLiM.

An adaptive mask can carry information about an observation that it subsequently hides. `pattern_only=True` is an optional residual diagnostic: coordinate and anchor-value features are zeroed before the learned residual head, while the visibility and time pattern remain. The final output still adds persistence anchors. The endpoint therefore needs the separately declared randomized-policy control. A fixed-mask hidden-value invariance check does not establish absence of this pattern information.

## Objective, training, and model identity

The current configuration supports only `objective=coordinate`. Each window's loss first averages measured coordinate errors within each leg, then gives the two legs equal weight. It averages those window losses in a batch drawn uniformly across independent groups, uniformly across a group's sequences, and uniformly across a sequence's windows. This sampling matches the declared group/sequence/window hierarchy in expectation. Longer clips do not obtain larger expected total training mass.

Output-specific ridge fits likewise normalize each window's contribution separately within each leg. Missing future targets do not become observed zeros. They are excluded from the solve and loss. Their array storage is sanitized before tensor subtraction to keep gradients finite. Scaling and ridge penalties are selected by the caller's permitted normal development groups, not abnormal labels.

The proposed MLP training configuration uses CPU computation with one Torch/BLAS thread, AdamW with learning rate 0.001 and weight decay 0.001, gradient clipping at one, and up to sixteen distinct groups per update. A batch contains all groups if fewer than sixteen are available. Every configuration uses the same 32-unit backbone and sampling-seed schedule. Seeds 27, 41, and 73 characterize optimization variation; they do not add independent sources. Normal-inner validation selects a common step budget from 100 and 200 using M0 seed 27, as declared by the coordinator, before applying it to every mask policy.

Each fit precomputes four seeded mask banks per training window, except M0, which uses one identical full-prefix bank. At each update, bank choice is independent of the source sampler. This bounds laptop runtime while retaining multiple corruptions per window. It is a finite-corruption-bank training regime, not on-the-fly unlimited random masking. All policies use the same number of updates and sampled windows, and the audit records their total visible and hidden observations. M0's greater visible-token count remains an explicit uncorrupted reference.

Checkpoints contain the complete configuration, target order, source/group/sequence/window membership, input and metadata hashes, preprocessing parameters, weights, parameter count, seed, fitting time, optimizer history, mask-bank hashes/counts, and package versions. The runner separately pins module hashes and the group-isolated split identity. Reloaded prediction is checked against the original checkpoint. CPU seeded fits reproduce exact predictions in the tested environment.

The MLP checkpoint reports effective rank and mean standard deviation of its hidden features before and after training on normal training prefixes only. It also records the maximum parameter magnitude and finite loss checks. These are optimization and representation-health diagnostics, not evidence that latent dynamics or clinical mechanisms were learned. No latent-target model, stochastic prior, phase estimator, or disease classifier is implemented by this stage. Those changes require the coordinator's later gate and new identities.

## Checks actually performed

The two new focused test modules verify natural-missing/hidden-value isolation, exact-budget matching, protected references and anchors, actual-time motion differences, unknown-motion selection, future-shape rejection, causal filling, equal-leg loss and gradients, hierarchical weights, an independent ridge solver, seed reproducibility, normal-only fitting, and checkpoint serialization/replay. Their initial execution passed 24 tests; final execution and source hashes are recorded by the coordinator after review.

The bounded real-data pilot uses only normal sources outside historical outer fold zero and reports timing and numerical health. It never evaluates those held-out normal sources or any abnormal source. Pilot checks and current implementation identities are in `model_pilot.json`; small development checkpoints remain local under `artifacts/model-development-v2/`. A pilot is implementation evidence, not a full experiment or an efficacy result.
