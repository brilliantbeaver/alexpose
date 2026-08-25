# Repair and run the three JEPA variants as a scientifically valid rapid experiment

## Workspace

`/Users/theodoremui/dev/alexpose/experiments/sjepa/gavd6`

Primary package:

- `src/gavd6_sjepa/gait.py`
- `src/gavd6_sjepa/amass.py`
- `src/gavd6_sjepa/train_amass.py`
- `tests/test_amass_core11_jepa.py`

## Goal

Implement the minimum model, training, and evaluation changes required to compare three JEPA architectures scientifically:

1. `paired_shared_no_cross`: shared branchwise encoder and predictor, without cross-attention.
2. `reflection_equivariant`: symmetric, swap-commuting cross-attention.
3. `paired_unconstrained`: direction-specific, untied cross-attention.

The experiment should answer:

> At matched model capacity and training exposure, does symmetric cross-branch fusion outperform no fusion and unconstrained fusion?

Prioritize rapid experimentation and trustworthy results. Do not spend time on provenance hashes, elaborate manifests, resumable jobs, atomic artifact machinery, or production hardening. Preserve unrelated existing changes in the dirty worktree.

## Background and confirmed problems

The existing reflection-equivariant encoder is geometrically correct: it commutes exactly with reflection. Its poor final checkpoint was primarily a learning-architecture problem, not a failed mirror operator.

The current paired encoders and EMA teachers use cross-branch attention, but `TokenPredictor` restores masked tokens and predicts each branch independently. The teacher therefore constructs cross-dependent targets using complete branches while the predictor lacks cross-branch interaction after mask-token insertion.

The reflection run:

- Beat the current standard through approximately epoch 36.
- Reached its best total loss near epoch 31.
- Regressed substantially afterward.
- Showed much less late regression in a small three-seed diagnostic when teacher cross-attention was weakened or disabled.

Additional problems:

- Cross-attention residuals are added at full strength without a gate or LayerScale.
- The model called `standard` is actually a paired, shared, no-cross encoder. It must not be treated as a single-view JEPA.
- `paired_unconstrained` has nearly twice as many encoder parameters as `reflection_equivariant` at the same width, invalidating a clean weight-tying comparison.
- Only final checkpoints are saved; checkpoint selection does not use validation data.
- Histories omit teacher entropy, KL divergence, complete VICReg components, and representation-health metrics.
- The current run completed only seed 7. Planned seeds 19 and 31 are absent.

## Required architecture changes

### A. Shared no-cross baseline

- Rename the current `standard` variant to `paired_shared_no_cross` throughout active code, CLI choices, outputs, and tests.
- Keep one shared branchwise encoder applied independently to original and mirrored branches.
- Keep a shared branchwise predictor without cross-attention.
- Keep paired inputs, masks, JEPA loss, and parity-resolved VICReg so its objective remains comparable to the paired models.
- Do not add a true single-view model in this task. That would be a fourth arm with a different objective.

### B. Reflection-equivariant model

Retain the symmetric encoder construction:

\[
F(a,b)=[g(a,b),g(b,a)].
\]

Add a symmetric paired predictor that:

1. Restores mask tokens and positional embeddings in both branches.
2. Processes the complete reconstructed branch pair with symmetric self/cross layers.
3. Returns predictions at masked positions from both branches.

The predictor must commute with branch swapping exactly, including after mask-token insertion.

Keep the target encoder as an EMA copy of the online encoder. Do not permanently remove cross-attention from only the target encoder.

Gate every cross-attention residual using a shared scalar or LayerScale that preserves swap equivariance. Initialize its effective scale near `0.1`. Apply the same design to the paired predictor.

### C. Paired-unconstrained model

- Add an unconstrained paired predictor with separate self-attention and directional cross-attention modules for `A -> B` and `B -> A`.
- Restore mask tokens before paired predictor interaction, matching the information flow of `reflection_equivariant`.
- Use the same initial cross-residual scale as `reflection_equivariant`.
- Expect this model to violate branch-swap commutation. That is intentional.

### D. Capacity matching

- After implementing the predictors, count every trainable parameter affecting optimization: online encoder, predictor, cross gates, and VICReg projector(s).
- Exclude the frozen EMA copy from the trainable count.
- Adjust embedding width and/or feed-forward width per variant until total trainable parameter counts are within approximately 5% of one common budget.
- Keep depth, mask fraction, window exposure, and optimizer updates matched unless a documented compute-matched secondary run is added.
- If matching is impractical, stop and report the remaining difference before launching expensive runs.
- Do not attribute an effect to weight tying while capacity remains materially different.

## Loss and metrics

Keep the same primary objective for all variants:

```text
total_loss = jepa_loss + vicreg_weight * vicreg_loss
```

For the first repaired experiment, retain the current:

- Teacher and predictor temperatures.
- Mask fraction.
- Yaw augmentation.
- VICReg and odd-channel weights.
- Optimizer family.
- Weight decay.

Do not redesign VICReg simultaneously with the architecture repair.

Extend loss reporting to include:

- JEPA cross-entropy.
- Teacher entropy, `H(q)`.
- KL divergence, `D_KL(q || p) = JEPA cross-entropy - teacher entropy`.
- Even VICReg invariance, variance, and covariance.
- Odd VICReg invariance, variance, and covariance.
- Total VICReg loss and total loss.

Compute KL from the same teacher and predictor distributions used in JEPA cross-entropy. Test numerically that:

\[
\text{cross-entropy}=\text{teacher entropy}+D_{KL}(q\|p)
\]

within floating-point tolerance.

## Training-loop changes

Refactor the monolithic fixed-update trainer into an epoch-oriented workflow:

```text
for each epoch:
    train for one complete epoch of matched window exposures
    evaluate the validation split with model.eval() and no gradients
    record training and validation metrics
    save the latest checkpoint
    update the best-checkpoint decision
```

Use a maximum of 100 epochs and approximately 10–15 validation epochs of early-stopping patience.

### Checkpoint selection

1. Reject checkpoints with non-finite metrics or obvious representation collapse.
2. For `reflection_equivariant`, also require the complete commutation audit to pass.
3. Among eligible checkpoints, select the lowest validation KL.
4. Report validation JEPA cross-entropy alongside KL, particularly when teacher entropies are similar.

Do not select checkpoints using training loss. Do not automatically use the final epoch.

## Validation and test evaluation

Create a shared `evaluate_variant(...)` path used by every architecture.

Requirements:

- Use the existing identity-disjoint train, validation, and test splits.
- Use fixed validation masks and augmentation draws so every variant receives the same evaluation task.
- If inexpensive, average several deterministic mask/view draws per validation window.
- Never update the model, projector, target center, or EMA state during validation or test evaluation.
- Evaluate the test split exactly once after architecture and hyperparameters are frozen, using the selected best checkpoint.

Return at least:

- JEPA cross-entropy, teacher entropy, and KL.
- All even/odd VICReg components.
- Even/odd feature energy and feature variance.
- Effective rank and mean pairwise cosine for pooled even/odd representations.
- Layerwise and final commutation residuals for `paired_shared_no_cross` and `reflection_equivariant`.
- The expected nonzero commutation residual for `paired_unconstrained`.

Health metrics prevent a collapsed or trivial teacher from winning merely because its KL is small. Do not interpret low JEPA/KL alone as proof of a useful representation.

If a downstream target or probe already exists and is inexpensive, freeze each selected encoder and evaluate all variants with the identical probe, data split, regularization-selection procedure, and metric. Without downstream evaluation, restrict conclusions to masked prediction, geometry, and representation health.

## Run plan

### Stage 1: implementation checks

- Add focused unit tests for predictor output shapes, mask restoration, cross-gate behavior, and commutation.
- Confirm that `paired_shared_no_cross` and `reflection_equivariant` commute in the online encoder, EMA encoder, predictor, and masked predictions.
- Confirm that `paired_unconstrained` generally does not commute.
- Run all relevant existing tests.

### Stage 2: smoke experiment

- Run every variant for 2–5 epochs using the smoke configuration.
- Verify finite losses and correct cross-entropy decomposition.
- Verify non-collapsed even/odd features.
- Verify validation, checkpoint selection, checkpoint reload, and evaluation.

### Stage 3: full pilot

- Run seed 7 for all variants on the full AMASS Core11 training split.
- Use the same window order, masks, augmentations, and exposure budget within the seed.
- Inspect validation curves before launching additional seeds.
- If reflection still exhibits late regression, retain the best validation checkpoint and report the trajectory rather than tuning against test results.

### Stage 4: scientific comparison

- Freeze the architectures, hyperparameters, and checkpoint-selection rule.
- Run paired seeds 7, 19, and 31 for all variants.
- Compare variants within matching seeds.
- Report each seed plus the mean and standard deviation.
- Test only the selected checkpoint from each completed run.

## Rapid tuning budget

If tuning is necessary, keep it small and validation-only:

| Setting | Candidates |
|---|---|
| Learning rate | `1e-4`, `2e-4` |
| Initial paired-model cross scale | `0.1`, `0.25` |
| Predictor depth | Hold fixed initially |

Give `reflection_equivariant` and `paired_unconstrained` the same number of tuning trials. Do not inspect test results while choosing settings.

## Acceptance criteria

The task is complete when:

- [ ] All variants have student/teacher/predictor information flow appropriate to their architecture.
- [ ] Cross-attention is gated and stable at initialization.
- [ ] Trainable parameter counts are within approximately 5%, or the mismatch is surfaced before full training.
- [ ] Reflection commutation holds through the encoder, EMA teacher, predictor, and masked outputs.
- [ ] Training and validation log cross-entropy, entropy, KL, complete VICReg components, and representation-health metrics.
- [ ] Best checkpoints are selected exclusively using validation results and survive reload/evaluation.
- [ ] Test evaluation uses only selected checkpoints and does not mutate model state.
- [ ] A full seed-7 pilot completes for every variant under a matched exposure schedule.
- [ ] If resources permit, seeds 19 and 31 complete under the frozen protocol.
- [ ] The report distinguishes optimization/geometry evidence from downstream representation utility.

## Deliverables

- Minimal source changes implementing the corrected architectures and evaluation loop.
- Focused tests for the architecture contracts and validation behavior.
- A concise command or script for smoke, full seed-7, and three-seed runs.
- Training and validation histories for each run.
- Best checkpoints.
- One summary table containing:
  - Parameter count.
  - Runtime.
  - Selected epoch.
  - Validation metrics.
  - Test metrics.
  - Representation-health metrics.
  - Commutation results.
- A short explanation of remaining scientific limitations.

## Working rules

- Inspect the dirty worktree before editing and preserve unrelated user changes.
- Prefer the smallest coherent implementation over framework or artifact refactors.
- Do not launch expensive full runs until architecture tests, metric decomposition, validation selection, and a smoke run pass.
- Do not use the test split for tuning, checkpoint selection, or early stopping.
- Do not describe the no-cross baseline as single-view.
- Do not claim an equivariance advantage unless capacity and training exposure are adequately matched.
