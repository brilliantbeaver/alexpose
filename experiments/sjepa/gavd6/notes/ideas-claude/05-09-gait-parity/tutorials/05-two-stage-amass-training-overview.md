# Two-stage AMASS training for GaitParity

This guide introduces a proposed way to pretrain every GaitParity S-JEPA variant. The same model first learns from broad AMASS motion and then continues learning from AMASS walking intervals located with BABEL.

The two stages solve different problems:

1. **Broad-motion pretraining** gives the encoder varied examples of how a human body moves.
2. **Walking-focused continuation** concentrates that general representation on gait timing, alternating legs, stance, swing, and foot motion.

Neither stage uses clinical outcomes. Both use the same self-supervised JEPA task: hide part of the motion and predict the teacher encoder's representation of the hidden part.

> **Implementation status.** This is a proposed full-study training protocol. The repository already contains the three matched model variants and their GAVD feasibility loop in `gavd6/gait_parity_jepa.py`, plus the AMASS inventory builder in `gavd6/scripts/make_amass_inventory.py`. It does **not** yet contain the production AMASS-to-core-schema converter, BABEL manifest builder, or command-line AMASS trainer named in these tutorials. Commands that use `prepare_amass_windows.py`, `build_babel_walking_manifests.py`, `train_gait_parity_amass.py`, or `audit_gait_parity_pretraining.py` describe the interface those scripts should implement; they are not currently runnable files.

## 1. The complete path

```text
AMASS motion files
        |
        | audited subject split and common skeleton conversion
        v
broad AMASS windows
        |
        | Stage 1: masked JEPA training
        v
broad-motion checkpoints
        |
        | load model, teacher, and projector weights
        | reset optimizer and learning-rate schedule
        v
BABEL-selected AMASS walking windows
        |
        | Stage 2: the same masked JEPA training objective
        v
gait-focused checkpoints
        |
        | freeze encoder; no more representation learning
        v
participant-held-out clinical readouts
```

BABEL enters only between the two stages. It answers, “During which part of this AMASS recording is the person walking?” It does not tell the S-JEPA what to predict.

## 2. What each stage contributes

| Stage | Training data | Information given to the model | Intended benefit |
|---|---|---|---|
| 1. Broad motion | Participant-disjoint AMASS windows from compatible human motion | Coordinates, validity masks, and synthetic mirrored coordinates | General movement structure, source breadth, and protection against learning only one narrow gait pattern |
| 2. Walking continuation | Clean walking windows cut from the same AMASS universe using BABEL | The same coordinates and masks; no BABEL category | Better alignment with gait cycles and the later signed gait target |

The Stage 1 checkpoint remains a real study artifact. Stage 2 does not replace it. Comparing the two tells us whether gait-focused continuation helped beyond broad-motion pretraining.

## 3. What BABEL does—and does not do

BABEL is used to build the Stage 2 sampling manifest. Its action categories and timestamps are not included in a training batch.

Correct:

```python
batch = {
    "coordinates": coordinates,       # [batch, frames, joints, 3]
    "valid_tokens": valid_tokens,     # [batch, time_patches, joints]
}
```

Incorrect:

```python
batch["action_label"] = "walk"       # Do not give this to the model.
batch["babel_category_id"] = 17       # Do not predict this either.
```

BABEL therefore changes **which windows are sampled**, not the JEPA loss applied to those windows.

## 4. The three variants must travel together

Every stage trains these variants:

- `standard`
- `paired_unconstrained`
- `reflection_equivariant`

The scientific question is architectural. The data cannot change between variants. For a given seed and matching regime, all three must share:

- the same eligible AMASS subjects and source motions;
- the same subject split;
- the same original windows;
- the same original/mirror exposure;
- the same mask schedule and eligible tokens;
- the same augmentation draws where the architecture allows it;
- the same tuning opportunities and stopping rule;
- paired seed numbers;
- an explicit exposure-matched or compute-matched budget.

The Stage 2 run for `standard`, seed 7, must descend from the Stage 1 `standard`, seed 7 checkpoint. It must not start from the best checkpoint belonging to another seed or variant.

## 5. Shared skeleton and reflection contract

The full GaitParity study specifies a common **core-11** schema as the primary input. A separate core-13 shoulder sensitivity analysis may be run, but it must never be mixed into the primary result.

The current feasibility module hard-codes 33 MediaPipe landmarks. It is useful as a tested model reference, but it must be parameterized before production AMASS training. The production data adapter must:

1. convert AMASS SMPL-family parameters into 3D joint positions;
2. map those positions into the frozen core-11 ordering;
3. keep anatomical left and right labels intact;
4. express the sequence in the frozen body-centred coordinate frame;
5. create the exact anatomical reflection by negating the declared lateral axis and swapping every left/right joint pair;
6. carry a validity mask through windowing and masking;
7. pass the involution test `mirror(mirror(x)) == x` within the declared tolerance.

Do not independently change this adapter between Stage 1 and Stage 2.

## 6. Checkpoint transition between stages

Stage 2 is a new, traceable training phase rather than an accidental resume.

Load from Stage 1:

- student encoder weights;
- target/teacher encoder weights;
- predictor weights;
- parity regularization projector weights;
- target-centering state and other model buffers.

Start fresh for Stage 2:

- optimizer state;
- learning-rate scheduler;
- Stage 2 step counter;
- Stage 2 early-stopping record.

This gives Stage 2 its own controlled learning-rate schedule while preserving everything the representation learned during Stage 1. The Stage 2 manifest must record the exact parent checkpoint hash.

## 7. What may determine a checkpoint

Checkpoint selection may use only nonclinical training and validation evidence:

- held-out AMASS masked-prediction loss;
- representation variance and effective rank;
- even- and odd-channel energy;
- layerwise commutation error for the equivariant model;
- throughput, memory, and stability failures;
- a prespecified stopping rule.

Do not choose the Stage 1 duration, Stage 2 duration, BABEL policy, seed, or checkpoint after inspecting stroke or Parkinson's outcomes. That would allow the clinical test set to shape pretraining.

## 8. The two matching regimes

One comparison cannot perfectly match both data exposure and GPU cost. Record two regimes when resources permit:

### Exposure-matched

All variants see the same number of original windows, mirrored windows, masked tokens, and optimizer updates. Compute and wall time may differ.

### Compute-matched

All variants receive the same prespecified compute allowance. The number of windows or updates may differ.

Name the regime in every run ID and checkpoint. Never describe a compute-matched run as exposure-matched.

## 9. Required checkpoint family

For each seed and regime, the minimum complete family is:

```text
stage1 / standard
stage1 / paired_unconstrained
stage1 / reflection_equivariant

stage2 / standard                 <- corresponding Stage 1 parent
stage2 / paired_unconstrained     <- corresponding Stage 1 parent
stage2 / reflection_equivariant   <- corresponding Stage 1 parent
```

With three seeds and two matching regimes, this is 36 runs. Start with one seed and the exposure-matched regime as an engineering check. Expand only after all lineage and geometry tests pass.

## 10. Evidence gates

Do not proceed merely because a loss curve decreased.

Stage 1 is ready to become a Stage 2 parent when:

- its input, split, configuration, and weight hashes are recorded;
- the standard and paired models pass their declared health tests;
- the equivariant model passes layerwise commutation tests in train and evaluation mode;
- the odd channel is not an all-zero or near-zero representation;
- all matched variants completed the frozen budget.

Stage 2 is ready for frozen downstream readouts when the same checks pass again and its manifest points to the correct Stage 1 parent.

## 11. What a result would mean

- **Stage 2 improves all variants:** gait-focused continuation helped, independently of the parity architecture.
- **Only the equivariant model improves:** the gait-focused distribution may interact usefully with internal parity structure.
- **Stage 2 changes nothing:** broad motion was sufficient, or the walking filter added too little information.
- **Stage 2 hurts:** the gait subset may be too small, too repetitive, biased by BABEL coverage, or trained for too long.
- **The odd channel collapses:** the run is invalid even if mirror error is numerically perfect.

The comparison must distinguish these possibilities rather than reporting only the final clinical score.

## 12. Continue with the practical tutorials

1. [Stage 1: broad AMASS pretraining](./06-stage-1-broad-amass-pretraining.md)
2. [Stage 2: BABEL-guided walking continuation](./07-stage-2-walking-focused-continuation.md)

The detailed study requirements remain authoritative in [GaitParity Study methodology](../METHODOLOGY_LONG_TERM.md).
