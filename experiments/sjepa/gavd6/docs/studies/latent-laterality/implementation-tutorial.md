# Implementation tutorial: repaired AMASS baselines and AMASS-Gauge

This tutorial turns the diagnostic AMASS artifacts into a valid baseline suite,
then replaces the easy window-local swap screen with a sequence-consistent
benchmark capable of testing SG-JEPA. It is deliberately ordered so that an
invalid coordinate, missingness, or masking contract cannot be hidden by a
successful training run.

The existing files in `outputs/repaired-jepa-seed7-v2` are local diagnostic
artifacts. Do not modify, resume, or use them in a model leaderboard. They show
that the training scaffold runs and that capacity matching is possible, but all
four checkpoints predate the repaired paired-mask contract. The directory also
contains a standard-only `run_config.json` and one-row `summary.csv` alongside
four independently produced checkpoints.

## 1. Target result

The implementation has three layers:

1. **Repaired clean AMASS baseline:** gauge-neutral coordinates, explicit
   validity input, branch-specific orbit-closed masks, complete provenance, and
   common downstream evaluation.
2. **AMASS-Gauge benchmark:** one latent convention path per full sequence,
   ambiguous switch boundaries, clean hard negatives, sparse or absent anchors,
   and shared corruption draws.
3. **SG-JEPA comparison:** the learned method is compared with continuity,
   correction-first, fixed-reflection, generic paired, and oracle controls.

Do not start layer 3 until layers 1 and 2 pass their gates.

## 2. Work on new immutable namespaces

Never write a repaired run into an old output directory. Use new schema and run
identifiers, for example:

```text
core11-gauge-neutral-v1/
amass-gauge-v1/
runs/repaired-baseline-seed7-v1/
runs/sg-jepa-screen-seed7-v1/
evaluations/sg-jepa-test-v1/
```

Make every writer refuse a nonempty output directory. A run that changes code,
input hashes, corruption settings, or seeds receives a new run ID.

Before implementation, run the current tests and save the result with the
development notes:

```bash
.venv/bin/python -m unittest \
  tests.test_convert_amass_core11 \
  tests.test_amass_core11_jepa \
  tests.test_swap_probe -v
```

## 3. Build `core11-gauge-neutral-v1`

### 3.1 Why the converter must change

`scripts/convert_amass_core11.py::estimate_forward_frame` currently falls back
to a vector constructed from the named left and right hips. It also consults
the named hip axis when deciding whether a travel-derived frame is acceptable.
Those operations can encode the absolute token convention into coordinate
signs before a semantic swap is applied.

An unanchored benchmark is valid only when the coordinate frame is unchanged
by a global left/right token relabeling.

### 3.2 Introduce a new schema instead of changing `core11-v1`

In `scripts/convert_amass_core11.py`:

1. Bump `CONVERTER_VERSION`.
2. Set a new schema such as `core11-gauge-neutral-v1`.
3. Set a new coordinate-frame name such as
   `pelvis-travel-or-random-yaw-v1`.
4. Keep the old conversion readable; never silently reinterpret old archives.

Add the policy to `ConversionConfig` and to the configuration fingerprint. A
safe initial policy is:

- **Confirmatory traveling stratum:** require a sufficiently long and straight
  pelvis trajectory. Derive forward from trajectory PCA and sign it using net
  displacement. Never read a named bilateral joint.
- **Stationary stress stratum:** retain the world horizontal orientation or use
  a deterministic sequence-hash-derived yaw. Keep this stratum separate until
  its leakage audit passes.

Do not use a hip axis even as a quality check. A decision that switches methods
after reading named hips is itself a side-dependent channel.

A useful function boundary is:

```python
def estimate_gauge_neutral_frame(
    pelvis_world: np.ndarray,
    pelvis_valid: np.ndarray,
    *,
    sequence_key: str,
    policy: str,
    min_travel_m: float,
    min_travel_straightness: float,
) -> ForwardFrame:
    """Return a frame that never reads a named bilateral joint."""
```

For deterministic random yaw, derive the seed with the repository's stable
hashing convention from `sequence_key` and a domain string such as
`gauge-neutral-stationary-yaw-v1`. Do not use Python's process-randomized
`hash()` and do not reuse the corruption seed.

### 3.3 Keep the three transformations typed

The implementation must keep these operations separate:

- Sensor reflection `M`: changes the declared coordinate channel and never
  changes token names.
- Semantic permutation `P`: exchanges token-indexed coordinates, validity,
  confidence, targets, and masks without negating coordinates.
- Global chart randomization: re-expresses the saved semantic path and teacher
  convention together; it must not change the physical motion.

Apply the hidden sensor reflection independently of the semantic path. Record
both bits in the corruption manifest. Independence should be tested from the
saved draws, not assumed from two adjacent random-number calls.

### 3.4 Preserve audit information

The existing archives contain pelvis world coordinates and invertible frame
transforms. Retain enough information to reconstruct resampled world-space
Core11 coordinates exactly. Add a round-trip test rather than duplicating a
large world-coordinate array unless storage permits it.

Each manifest row should include:

```text
schema
coordinate_frame
forward_policy
forward_method
source_sha256
conversion_fingerprint
canonical_frames
valid_fraction
stationary_stratum
```

Publish an attrition table by source corpus, identity, split, and rejection
reason before training.

### 3.5 Converter tests

Extend `tests/test_convert_amass_core11.py` with fixtures that prove:

1. Globally permuting left/right token names does not change the physical frame.
2. The main policy never returns either legacy hip-facing method.
3. Translation and yaw behave as declared.
4. Stationary behavior is deterministic and independent of semantic seeds.
5. The body-to-world round trip reconstructs every valid point.
6. Train, validation, and test identities remain disjoint after attrition.

The old tests that require hip-facing fallback remain tests of the old schema;
new tests must target the new schema explicitly.

## 4. Make validity part of every learned input

### 4.1 Current limitation

`Core11WindowDataset` loads validity and the mask sampler uses it, but the
encoder receives only zero-filled coordinates. Consequently, an invalid joint
and a valid joint at the coordinate origin are indistinguishable.

### 4.2 Token-level validity embedding

Add an explicit validity mode to `TrainConfig`, for example:

```python
validity_input: str = "binary-embedding-v1"
```

At tokenization time, combine the coordinate token with a learned binary
embedding:

```python
coordinate_token = coordinate_projection(flattened_segment)
valid_fraction = valid_frame.float().mean(dim=segment_axis)
validity_token = validity_projection(valid_fraction.unsqueeze(-1))
token = coordinate_token + validity_token + time_position + joint_position
```

Using the per-segment valid fraction retains more information than reducing the
segment to one Boolean. If the scientific contract requires conservative
all-frames validity, store both `all_valid` and `valid_fraction`, but specify
which one enters the model.

Change encoder interfaces so validity cannot be accidentally omitted:

```python
encoder(coordinates, valid_patch, keep_mask=...)
target_encoder(coordinates, valid_patch)
```

For paired branches, lift validity with `mirror_validity`; never infer validity
from whether coordinates happen to be zero. Under a semantic path, apply the
same blockwise `P` action to validity that is applied to coordinates.

### 4.3 Missingness augmentation

The existing AMASS manifest is almost entirely valid, so explicit missingness
must be generated during training. Use domain-separated deterministic seeds.
Include:

- isolated distal-joint dropout;
- short contiguous temporal gaps;
- whole bilateral-pair gaps;
- boundary-centered gaps for difficult semantic events; and
- identically distributed gaps around clean pseudo-boundaries.

Keep geometric target masks separate from observation validity. An observed
joint can be selected as a JEPA target; an invalid observation cannot.

### 4.4 Validity tests

Add tests showing that:

- identical zero coordinates with validity zero and one produce different
  tokens;
- invalid targets are never sampled;
- `P` and `M` transport validity correctly;
- paired validity is orbit-closed;
- the mask-only leakage control cannot predict event boundaries materially
  better than the input-free path prior; and
- clean performance is not degraded beyond the prespecified tolerance.

## 5. Lock the repaired paired-mask contract

The current source already implements `orbit_closed_target_masks` and writes
`paired_mask_contract` in new checkpoint payloads. The required work is to
verify that path and produce new checkpoints; old checkpoints do not inherit a
code repair retroactively.

For branch zero mask `m`, branch one must hide `P(m)`:

```text
mask_B[t, P(j)] = mask_A[t, j]
valid_B[t, P(j)] = valid_A[t, j]
```

Run at least these existing tests after every relevant model change:

```bash
.venv/bin/python -m unittest \
  tests.test_amass_core11_jepa.AmassCore11JepaTests.test_orbit_closed_masks_hide_physical_counterparts \
  tests.test_amass_core11_jepa.AmassCore11JepaTests.test_streaming_views_lift_paired_masks_but_not_standard_masks \
  tests.test_amass_core11_jepa.AmassCore11JepaTests.test_streaming_paired_mask_lifts_one_sided_validity \
  tests.test_amass_core11_jepa.AmassCore11JepaTests.test_validation_selected_checkpoint_survives_reload -v
```

Also add a direct copyability test: perturb the reflected physical counterpart
that should be hidden and verify that the prediction cannot read the perturbed
value through the other branch.

Every new paired checkpoint must contain:

```text
metadata.paired_mask_contract = branch-specific-p-closed-v1
```

The single-branch control should record `single-branch-v1`.

## 6. Replace the training artifact contract

### 6.1 Prevent overwrite and partial-matrix ambiguity

`src/gavd6_sjepa/train_amass.py` currently permits independently launched jobs
to target one shared directory. That is how four checkpoints survived beside a
standard-only configuration and summary.

Choose one of these safe layouts:

1. One job owns the entire seed/arm matrix and updates an expected-run ledger
   atomically; or
2. Every seed/arm writes an immutable child directory, followed by a read-only
   aggregation job.

The second layout is safer for Slurm arrays:

```text
repaired-baseline-v1/
  expected_runs.csv
  runs/seed-7/standard_sjepa/
  runs/seed-7/paired_shared_no_cross/
  runs/seed-7/reflection_equivariant/
  runs/seed-7/paired_unconstrained/
  aggregate/summary.csv
```

No training worker may write `aggregate/summary.csv`.

### 6.2 Required per-run artifacts

Each seed/arm directory should contain:

```text
effective_config.json
environment.json
hashes.json
data_manifest.json
capacity_and_compute.csv
history.csv
validation_predictions.parquet
best.pt
STATUS.json
COMPLETE.json
```

Write `STATUS.json` as `running`, then atomically replace it with `complete` or
`failed`. `COMPLETE.json` must include hashes for every final artifact and an
explicit `test_split_evaluated` flag.

Record the configuration *after* `capacity_matched_config` resolves embed and
feed-forward dimensions. Include the git commit, dirty diff hash, lockfile
hash, manifest hash, conversion fingerprint, GPU, Slurm job ID, wall time,
peak memory, optimizer updates, and paired-mask contract.

### 6.3 Do not rank architectures by self-KL

Each arm predicts its own EMA teacher distribution. Validation KL is therefore
usable for checkpoint selection within an arm, but it is not a common
cross-model endpoint.

Train identically specified frozen readouts for each representation using only
training identities. Use the same target definitions, regularization search,
identity split, and evaluation code for every arm. The primary comparisons
should include:

- odd and even common-target NMAE/NMSE;
- sign accuracy only above a prespecified nonzero threshold;
- clean preservation;
- feature variance and effective rank;
- path recovery and calibration for gauge-aware arms; and
- per-identity paired differences with identity-clustered intervals.

Save window/sequence predictions so every aggregate can be recomputed.

## 7. Rerun the repaired clean baseline before SG-JEPA

At minimum rerun these four matched arms:

```text
standard_sjepa
paired_shared_no_cross
reflection_equivariant
paired_unconstrained
```

Add `standard_mirror_aug` as an inexpensive control if the paper will compare
fixed-reflection inductive biases. Match optimized parameters, source windows,
mask and augmentation draws, optimizer updates, checkpoint-selection policy,
and downstream readout capacity.

The run is accepted only if:

- all expected seed/arm statuses are complete;
- every checkpoint reloads exactly;
- all paired checkpoints record the mask contract;
- tied-model float32 commutation residual is at most `1e-5`;
- required feature variances exceed the declared minimum;
- every summary row traces to saved predictions; and
- no test tensor is opened.

The current command-line entry point can remain, but the target invocation
should use a new directory and new neutral manifest. Treat this as a target
interface until the preceding code changes exist:

```bash
AMASS_RUN_ROOT=/path/to/amass-neutral \
AMASS_CONVERSION_MANIFEST=/path/to/amass-neutral/manifests/core11-gauge-neutral-v1.csv \
AMASS_TENSOR_ROOT=/path/to/amass-neutral/core11-gauge-neutral-v1 \
AMASS_OUTPUT_DIR=/path/to/runs/repaired-baseline-seed7-v1 \
AMASS_RUN_ID=repaired-baseline-seed7-v1 \
AMASS_PROFILE=full \
AMASS_DEVICE=cuda \
AMASS_SEEDS=7 \
AMASS_EVALUATE_TEST=0 \
AMASS_RUN_TRAINING=1 \
uv run --no-sync python -m gavd6_sjepa.train_amass
```

## 8. Generate AMASS-Gauge paths before windowing

### 8.1 Refactor the E0 generator

`src/gavd6_sjepa/swap_probe.py` currently samples one anchored path per
nonoverlapping 64-frame window and enumerates clean plus one fixed-duration
segment. Keep that code as the E0 diagnostic. Build the full generator in a new
module such as `src/gavd6_sjepa/amass_gauge.py`.

Use sequence-level records:

```python
@dataclass(frozen=True)
class SequenceGaugeDraw:
    sequence_id: str
    corruption_draw: int
    gauge_path_rle: tuple[tuple[int, int], ...]
    switch_frames: tuple[int, ...]
    semantic_scope: str
    sensor_reflection_bit: int
    latent_chart_bit: int
    anchor_frames: tuple[int, ...]
    occlusion_seed: int
    noise_seed: int
```

Generate this record once for every `(sequence_id, corruption_draw)`. Slice the
already-corrupted sequence afterward so overlapping windows inherit identical
states on shared frames.

### 8.2 Path families

Use a declared mixture rather than one fixed event:

- 20% clean paths;
- global swaps;
- one segment lasting 1, 2, 4, or 8 blocks;
- repeated low-rate Markov switches; and
- evaluation-only off-grid boundaries and partial limb-chain swaps.

The in-model semantic action should initially swap all five Core11 bilateral
pairs. Partial swaps are misspecification tests, not ordinary training labels.

### 8.3 Make boundaries difficult without making the task impossible

The current continuity model sees two sharp, fully observed discontinuities.
Replace that cue with ambiguity that a longer-context model can resolve:

1. Identify candidate boundary strata on training data only: limb crossings,
   turns, low speed, approximately bilateral motion, and naturally missing
   distal joints.
2. Apply boundary-centered occlusion or interpolation to semantic events.
3. Apply the same nuisance distribution around sampled clean pseudo-boundaries.
4. Keep information before and after the boundary so sequence inference remains
   possible.

Do not merely increase global Gaussian noise. That can destroy both local and
long-context evidence without testing structured inference.

Run a frozen difficulty grid on training/calibration identities. Freeze the
selected severity before validation interpretation and certainly before test.

### 8.4 Expand path inference

The twelve-path enumeration in E0 is insufficient for variable duration and
repeated switches. Implement a two-state hidden semi-Markov model or an
equivalent finite-state dynamic program:

- Viterbi gives the hard MAP path.
- Forward-backward gives block and edge marginals.
- A duration prior handles variable segment lengths.
- Anchored scoring uses the supplied chart bit.
- Unanchored scoring minimizes over the one global flip or evaluates the
  equivalence class directly.

Fit edge-head weights on training identities. Fit the duration prior and one
temperature on an identity-disjoint calibration subset. The posterior must be
calibrated before entering the JEPA loss.

### 8.5 Leakage and consequence gates

Before representation training, require:

1. The mask-only control's relative path-NLL gain over the input-free prior is
   at most 1%.
2. An absolute unanchored convention probe is near chance, with upper 95%
   confidence bound below 0.55 AUROC.
3. Oracle correction improves swapped-event common-target odd error by at least
   5%.
4. Oracle remains at least 5% better than continuity.
5. The validation set contains a prespecified minimum number of
   intermediate-entropy paths; one ambiguous window is not sufficient.

If gates 3 and 4 cannot both hold, report the transparent corrector and stop
the SG-JEPA architecture experiment.

## 9. Implement the corruption-aware comparison

Give every learned arm the same clean pretraining budget and the same
corruption-aware training examples. Required controls are:

```text
raw continuity + dynamic programming
corruption-trained standard S-JEPA + gauge head
fixed-reflection JEPA + the same gauge head
generic paired temporal transformer
synchronization-first correction + S-JEPA
SG-JEPA hard MAP
SG-JEPA structured posterior
oracle path
uniform 50:50 posterior
```

Train the gauge head with the same synthetic edge labels for every applicable
arm. Then freeze it. Calibrate the structured posterior on held-out training
identities and detach its weights inside the JEPA prediction loss. Otherwise,
the supposed correspondence posterior can become an unconstrained
loss-selecting gate.

Preserve parity explicitly:

- even representations and targets must be invariant to `P`;
- odd representations must transport sign under `P`;
- signed odd output is allowed only with an independent anchor; and
- unanchored output is evaluated as `{y, -y}` or an equivalent symmetric
  distribution.

## 10. Evaluation and promotion protocol

Use seed 7 to screen all arms. Promote only SG-JEPA and the strongest
non-oracle baseline to seeds 19 and 31. This avoids spending the full budget on
dominated arms while preserving a confirmatory contrast.

Validation reports must include:

- clean and corrupted odd/even targets;
- path Hamming up to global flip and when anchored;
- switch F1 with declared tolerance;
- segment IoU;
- structured path NLL, Brier score, and reliability bins;
- MAP-versus-posterior deltas by entropy stratum;
- clean false-switch and preservation costs; and
- sequence-then-identity macro effects with identity bootstrap intervals.

Open the test split once, after architecture, severity, priors, anchors, seeds,
and margins are frozen. Use ten shared deterministic corruption draws. The test
evaluator writes to a new directory and records a seal preventing silent
reruns.

## 11. Add GAVD only after the controlled benchmark works

GAVD should be an ecological audit and optional domain-adaptation stage, not a
replacement for controlled AMASS training.

1. Operate on raw detector outputs before any named-hip body frame.
2. Preserve confidence and validity as explicit channels.
3. Split and bootstrap by source video.
4. Use extractor disagreement only to propose candidates.
5. Have blinded humans label coherent swaps, partial errors, tracking errors,
   no swaps, and indeterminate cases.
6. Keep probability-sampled and candidate-enriched lanes separate.

Only after verified natural events survive transparent correction should
AMASS-to-GAVD continued pretraining be claimed as necessary.

## 12. Practical implementation order

Use this order because each step is a dependency of the next:

1. Add new converter schema and gauge-neutral frame tests.
2. Convert a small traveling-only smoke cohort and run the absolute-gauge leak
   probe.
3. Add validity inputs and missingness tests.
4. Run the paired no-copy and checkpoint metadata tests.
5. Replace the run artifact layout and test interrupted/partial matrices.
6. Run the repaired seed-7 clean baseline.
7. Implement sequence-level paths and matched pseudo-boundary nuisance.
8. Tune benchmark difficulty on training/calibration identities.
9. Run E1 integrity gates.
10. Screen corruption-aware arms at seed 7.
11. Promote the decisive comparison to seeds 19 and 31.
12. Freeze the protocol and execute the sealed test once.

## 13. Definition of done

The repair is complete only when all of the following are true:

- [ ] No coordinate-frame decision reads named left/right joints.
- [ ] Global token relabeling does not change the physical frame.
- [ ] Validity is an explicit encoder input for every learned arm.
- [ ] Missingness, semantic permutation, and sensor reflection remain typed.
- [ ] Paired masks are branch-specific and orbit-closed.
- [ ] Fresh checkpoints record their mask and input contracts.
- [ ] Every run is immutable, complete, hashed, and independently aggregatable.
- [ ] All arms see shared windows, corruption draws, updates, and targets.
- [ ] Cross-arm claims use common downstream endpoints rather than self-KL.
- [ ] Sequence paths are consistent across overlapping windows.
- [ ] Clean pseudo-boundaries prevent nuisance-mask leakage.
- [ ] Oracle is consequential and materially better than continuity.
- [ ] Posterior evaluation has enough genuinely ambiguous cases.
- [ ] Three-seed conclusions use identity-clustered intervals.
- [ ] Test evaluation is sealed and executed once.

Until this checklist passes, the old AMASS checkpoints remain useful only for
loading, smoke tests, and historical diagnostics.
