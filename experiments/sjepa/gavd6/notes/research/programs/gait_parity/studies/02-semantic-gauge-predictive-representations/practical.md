# Practical guide: running the two-week semantic-gauge study

This is the execution document. It distinguishes files and commands that exist
now from the target interface that still must be implemented. The scientific
specification lives in [proposal.md](./proposal.md); mathematical definitions
and the glossary live in [theory.md](./theory.md).

## 1. The finish line

The two-week deliverable is complete when it contains:

1. an immutable, gauge-neutral AMASS-Gauge dataset manifest;
2. repaired no-copy masks and a validity-aware common input contract;
3. raw/Viterbi, correction-first S-JEPA, fixed-reflection, generic paired, and
   SG-JEPA comparisons;
4. three seeds for SG-JEPA and the strongest non-oracle baseline;
5. one sealed corpus-qualified identity-disjoint test evaluation with shared corruption
   draws;
6. identity-clustered statistics and calibrated anchored/unanchored outputs;
7. a clear simpler-method and null-result branch; and
8. an optional GAVD audit whose claims are excluded unless its gate passes.

Do not make an external force cohort a dependency. Do not add a loopy graph,
product group for individual joints, architecture search, or clinical endpoint
unless the core result is already complete.

## 2. What exists today

### Code and data map

| Path | Status | Purpose |
| --- | --- | --- |
| `manifests/amass_core11_conversion.csv` | Available locally | Current 8,854-sequence Core11 conversion manifest |
| `manifests/amass_subject_splits.csv` | Available locally | Corpus-qualified identity split source |
| `scripts/convert_amass_core11.py` | Available | Converts AMASS to 30 Hz, 11-joint tensors with provenance |
| `src/gavd6_sjepa/gait_parity_jepa.py` | Available | Standard and paired fixed-reflection model components |
| `src/gavd6_sjepa/amass_core11_jepa.py` | Available | Streaming dataset, losses, checkpointing, and evaluation |
| `src/gavd6_sjepa/train_amass.py` | Available | `train-amass-core11` entry point |
| `slurm/train-amass-core11-full.sbatch` | Available, not matrix-safe | Currently selects seed 7 and standard S-JEPA; do not assume it launches all arms |
| `outputs/repaired-jepa-seed7-v2/` | Available, diagnostic only | Four pre-orbit-mask seed-7 artifacts; config/summary reflect only the last arm |
| `scripts/download_gavd_full.py` | Available | Resumable unique-source downloader; expects a sequence manifest not yet built by a standalone script |
| `work/artifacts/real/poses/` | Available, small | 96 sequences from 18 source videos, one pose extractor |
| `scripts/evaluate_gavd_core11_probe.py` | Available | Existing frozen GAVD probe; not a semantic-gauge evaluator |

No SG-JEPA model, sequence-level gauge generator, gauge evaluator, paired GAVD
extractor, or human-audit tool exists yet. Commands below labeled **target
interface** describe what implementation should expose; they are not runnable
today.

### Current AMASS snapshot

| Split | Identities | Sequences | Eligible sequences (at least 64 frames) | Windows |
| --- | ---: | ---: | ---: | ---: |
| Train | 151 | 7,217 | 6,921 | 79,535 |
| Validation | 19 | 769 | 722 | 5,936 |
| Test | 19 | 868 | 824 | 8,220 |

The current manifest has `valid_fraction=1.0` throughout. Existing training
therefore does not establish behavior under missing joints. The histories in
`repaired-jepa-seed7-v2` report 79,552 exposures because 79,535 windows are
padded to 2,486 complete batches of 32, repeating 17 examples in the last
batch. That count is expected. The exact old input remains unauthenticated
because its manifest and code hashes were not saved with the copied run.

### Current checkpoint verdict

The stored checkpoints are useful only to test loading, feature extraction,
and the old architecture. Every checkpoint lacks
`metadata.paired_mask_contract`; the models were trained before branch-specific
orbit-closed masks. This exposes a physical counterpart only in paired models
with cross-branch interaction, but all paired arms still need one recorded mask
contract. The top-level run configuration and summary were also
overwritten by the final standard-only invocation, and test evaluation was
off. Do not copy their validation KL into a model leaderboard.

Before any new run, snapshot:

- Git commit and dirty diff;
- Python lockfile hash;
- raw/conversion/split manifest hashes;
- effective per-arm model configuration after capacity resolution;
- corruption-generator version and manifest hash;
- paired-mask contract;
- Slurm job ID, GPU model, wall time, peak memory, and measured GPU-hours.

## 3. Data flow

```text
AMASS source + subject registry
            │
            ▼
  canonical Core11 sequence ── gauge-neutral coordinate conversion
            │
            ├── clean kinematic targets and identity split
            │
            ▼
 sequence-level gauge path g(t) + global chart bit b + sensor bit C
                    + occlusion/noise + anchor
            │
            ▼
 immutable AMASS-Gauge manifest (all random seeds and actions recorded)
            │
            ▼
 overlapping 64-frame windows inherit the same sequence path
            │
            ├── training/validation loaders
            └── sealed deterministic test draws shared by every model
```

### 3.1 Build a gauge-neutral coordinate contract

The existing conversion frequently uses named hips to determine facing and
handedness. That leaks the semantic answer if $P$ is injected afterward.
Create `gauge-neutral-v1` with these rules:

1. use world/trajectory information that does not inspect left-versus-right
   names;
2. exclude or separately stratify near-stationary windows where trajectory is
   undefined;
3. sample a hidden lateral sensor reflection independently of the semantic
   gauge so coordinate sign cannot reveal token convention;
4. apply the same sensor action to coordinates only, and the semantic action
   to all token-indexed fields only; and
5. preserve raw world coordinates for leakage audits.

Before GPU training, fit a strong absolute-gauge classifier on coordinates,
validity, metadata, motion ID, corpus ID, and frame flags. Pass only if AUROC is
near 0.5 and the upper identity-bootstrap 95% CI is below 0.55. Relative-switch
evidence may remain predictable; absolute root convention should not.

Only 2,607/8,854 current sequences use the side-neutral travel-based facing
method. Simply excluding all named-hip fallbacks would leave about
10,921/942/1,551 train/validation/test windows and 107/14/15 represented split
identities, rather than 151/19/19. Emit an attrition table before training.
Prefer a genuinely orientation-free/random-yaw contract that can retain valid
stationary motion; otherwise revise the sample-size, runtime, and uncertainty
plan to the retained identities.

### 3.2 Generate corruption before windowing

Generate one path per full sequence and corruption draw. Never sample an
independent path for each overlapping window; doing so assigns contradictory
states to the same source frames.

Each manifest row records:

| Field | Meaning |
| --- | --- |
| `identity`, `motion_id`, `split` | Independent unit and immutable split |
| `corruption_draw` | Deterministic draw ID shared across models |
| `block_frames` | Four in the minimum model |
| `gauge_path_rle` | Run-length encoded coherent Core11 swap path |
| `switch_frames` | Exact event boundaries before window slicing |
| `semantic_scope` | `core11_all_pairs` for in-model data; partial scopes are out-of-model |
| `sensor_reflection_bit` | Independent hidden coordinate-frame action |
| `latent_chart_bit` | Random global semantic re-expression applied jointly to the teacher convention and saved gauge path; required for unanchored batches |
| `occlusion_seed`, `occlusion_mask` | Exact missingness action |
| `noise_seed`, `noise_scale` | Exact coordinate perturbation |
| `anchor_blocks`, `anchor_error_rate` | Supplied anatomical convention evidence |
| `generator_version`, `source_sha256` | Reproducibility |

Use 20% clean training examples. Sample in-model local segments of 1, 2, 4,
and 8 blocks and repeated low-rate Markov switches. Include evaluation-only
off-block boundaries and partial joint-chain swaps to measure misspecification.

The provisional confirmatory cell is one 4-block/16-frame local segment with
moderate boundary-centered distal-joint occlusion and noise. Freeze exact
levels using validation data before opening test. Use 5–10 deterministic test
draws, identical across arms.

### 3.3 Validity is an input

Add a learned validity embedding or explicit validity channel to every learned
arm. Invalid coordinate zero must be distinguishable from a valid joint at the
origin. AMASS has no detector confidence, so do not invent it; the GAVD adapter
may add confidence as a separate declared channel. Apply $P$ and every mask to
validity/confidence exactly as to coordinates.

## 4. Integrity gates before full training

### Gate 0A: typed transformations

Unit tests must prove:

- $P^2=I$, $M^2=I$, and $P\ne M$ on a non-symmetric fixture;
- $P$ exchanges token contents but does not negate coordinates;
- $M$ exchanges pairs and negates only the declared mediolateral channel;
- validity, confidence, targets, and masks follow the correct token action;
- sequence corruption is identical on overlapping source frames; and
- relative-edge logits are invariant to a global semantic chart flip; and
- applying the saved inverse path reconstructs the clean tensor exactly before
  added noise/occlusion.

### Gate 0B: no-copy masks and checkpoints

For paired branch zero mask $m$, branch one must use $P(m)$, not the same
indices. A copyability test must show that neither branch can attend to the
other branch's physical target. Fresh checkpoints must contain:

```text
paired_mask_contract = branch-specific-p-closed-v1
```

Also require optimizer/reload equality, deterministic evaluation, nonzero
even/odd variance, and tied-model float32 commutation residual at most (10^{-5}).

### Gate 0C: benchmark consequence

On validation data:

- an oracle path must materially improve the corrupted common-target odd error;
- an unanchored absolute-offset probe must remain at chance;
- a relative edge oracle must reconstruct paths up to one global bit; and
- the uniform 0.5-on-every-edge baseline must score poorly on relative-edge
  discrimination even though it appears “uncertain.”

If the oracle provides no benefit, the corruption does not affect the claimed
representation endpoint. Stop before a full matrix.

## 5. Model arms and expected behavior

| Model | Training | Expected clean behavior | Expected corruption behavior | What a surprise means |
| --- | --- | --- | --- | --- |
| Raw continuity + HMM/Viterbi | CPU; no encoder | No representation | Strong when gait is smooth/visible; weaker at occluded boundaries | If it solves every realistic cell, SG-JEPA is unnecessary |
| Corruption-trained standard S-JEPA + gauge head | Same data/validity | Strong ordinary baseline | Can learn detection, but a detached hard correction may lose uncertainty | If it matches SG-JEPA, joint marginalization adds no value |
| Repaired fixed-reflection JEPA + same head | Uses known $M$ paired lift | Exact anatomical-reflection commutation | No guarantee for latent semantic $P$ | Success would show useful transfer from chirality, not semantic-gauge novelty |
| Generic paired temporal transformer | Capacity/FLOP sensitivity | Flexible, more compute | May fit swaps without exact parity or calibration | A win means constraints hurt or capacity matching was incomplete |
| Synchronization then S-JEPA | Edge classifier + Viterbi, corrected input | Strong decisive baseline | Good when one hard path is enough | Matching SG-JEPA favors the simpler modular system |
| SG-JEPA | Soft relative posterior inside predictive loss | Must be within 2% of best clean arm | Should preserve even content, transport odd content, and abstain when ambiguous | Gain only at extreme corruption narrows the claim |
| Oracle path | Evaluation-time upper bound | Same clean target | Maximum attainable benefit from correct transport | Small oracle gap means the task is not consequential |
| Uniform/global 50:50 | Evaluation-time lower bound | Uninformative | Correct global symmetry but poor edge information | Prevents calling maximal uncertainty a success |

Seed-7 ablations remove one item at a time: marginalization (hard MAP), temporal
messages, parity projection, and validity input. Only the single decisive
ablation is repeated for seeds 19 and 31.

Every learned gauge-head arm receives the same synthetic edge labels, BCE
supervision, masks, corruption draws, and anchor exposure. Freeze and calibrate
the edge head after BCE training. On a fixed held-out calibration subset, fit
the switch prior and one CRF temperature using structured equivalence-class
path NLL, then detach its posterior weights inside the JEPA prediction loss;
otherwise model-specific label access or loss-driven gating would confound the
architecture comparison. The calibration subset does not update edge-head
weights and is not the sealed test set.

## 6. Commands

### Available now: inspect and rerun the fixed-reflection scaffold

Dry configuration, without test access:

```bash
AMASS_RUN_ROOT=/hai/scratch/tedmui/alexpose/experiments/sjepa/gavd6/data/amass/outputs \
AMASS_PROFILE=full AMASS_DEVICE=cuda AMASS_SEEDS=7 \
AMASS_EVALUATE_TEST=0 AMASS_RUN_TRAINING=0 \
uv run --no-sync train-amass-core11
```

After the repaired mask and metadata tests pass, run each seed or Slurm-array
task into a unique directory:

```bash
export AMASS_RUN_ROOT=/hai/scratch/tedmui/alexpose/experiments/sjepa/gavd6/data/amass/outputs
export AMASS_RUN_ID=repaired-baseline-seed7
AMASS_RUN_TRAINING=1 AMASS_PROFILE=full AMASS_DEVICE=cuda \
AMASS_SEEDS=7 AMASS_NUM_WORKERS=4 AMASS_EVALUATE_TEST=0 \
AMASS_VARIANTS=standard_sjepa,paired_shared_no_cross,reflection_equivariant,paired_unconstrained \
uv run --no-sync train-amass-core11
```

Do not launch independent variants into one reused directory until summary and
configuration consolidation is fixed. Do not change the current Slurm time
limit and assume a three-seed matrix will fit; use a seed job array and unique
run IDs.

### Target interface: not implemented yet

Implement small composable commands with these responsibilities:

```text
scripts/build_amass_gauge_manifest.py   # neutral conversion + sequence paths
src/gavd6_sjepa/train_amass_gauge.py    # learned arms; validation only by default
scripts/evaluate_amass_gauge.py         # sealed shared-draw test evaluation
scripts/extract_gavd_pose_pair.py       # two extractor outputs in raw image frame
scripts/audit_gavd_laterality.py        # probability/enriched samples + adjudication
```

The intended usage should look like this after implementation:

```bash
uv run --no-sync python scripts/build_amass_gauge_manifest.py \
  --source-manifest manifests/amass_core11_conversion.csv \
  --source-tensor-root /hai/scratch/tedmui/alexpose/experiments/sjepa/gavd6/data/amass/outputs/core11 \
  --subject-splits manifests/amass_subject_splits.csv \
  --coordinate-contract gauge-neutral-v1 \
  --output-root /hai/scratch/tedmui/amass-gauge-v1

AMASS_GAUGE_RUN_TRAINING=1 AMASS_GAUGE_EVALUATE_TEST=0 \
AMASS_GAUGE_VARIANT=sg_jepa AMASS_GAUGE_SEED=7 \
AMASS_GAUGE_MANIFEST=/hai/scratch/tedmui/amass-gauge-v1/manifest.parquet \
AMASS_GAUGE_DATA_ROOT=/hai/scratch/tedmui/amass-gauge-v1 \
AMASS_GAUGE_OUTPUT_DIR=/hai/scratch/tedmui/runs/sg-jepa-seed7 \
uv run --no-sync python -m gavd6_sjepa.train_amass_gauge

uv run --no-sync python scripts/evaluate_amass_gauge.py \
  --frozen-protocol configs/amass-gauge-v1-frozen.json \
  --run-index outputs/amass-gauge-run-index.csv \
  --split test --evaluate-once
```

The final interface may use different names, but it must default to no test
evaluation, refuse an output directory with incompatible run metadata, and
write atomically after every completed arm.

## 7. GPU plan and dynamic workflow

Treat one current four-arm suite as 12 reference-GPU hours and cap the study at
eight suite equivalents (about 96 GPU-hours on the same hardware). Stop at
failed gates rather than using
the cap as a target.

| Phase | Days | GPU cap | Main work | Exit condition |
| --- | --- | ---: | --- | --- |
| Specification and integrity | 1–2 | 6 h | Hashes, neutral frame, sequence generator, raw/HMM/oracle, masks, leakage tests | Absolute-gauge upper CI <0.55; oracle consequence present |
| Integration | 3 | Included above | Short seed-7 health runs only; profile I/O and FLOPs | All artifacts reload; no-copy/commutation pass |
| Repaired baseline | 3–4 | 12 h | Fresh clean/corruption baseline suite, validation only | Common-target metrics valid; no collapse |
| Learned-arm screen and finalists | 3–7 | Up to 54 h | Screen learned arms at seed 7; run seeds 19/31 only for SG-JEPA and the selected baseline | Ceiling/reserve, not required spend; finalists frozen on validation |
| Ablations | 6–8 | 8 h | Four seed-7 component removals | One causal explanation identified or gain judged uninterpretable |
| Confirmation/contingency | 8 | 4 h | Repeat one decisive ablation or recover one failed job if budget remains | Architecture, temperature, and protocol frozen |
| Sealed AMASS test | 9 | 4 h | Shared 5–10 draws, no tuning | Test marker written; evaluation cannot silently rerun |
| Optional paired GAVD extraction | 1–8 | 8 h | Hard-capped extraction only; retrieval/annotation use CPU | Audit proceeds or is reported incomplete without delaying AMASS |
| Statistics and writing | 10–14 | None expected | Aggregate identity-first; figures; claims table | Reproducible package and bounded conclusion |

Run full-GAVD retrieval, paired extraction, and annotation preparation in
parallel from day 1. This parallel lane may produce a useful audit but must not
delay the complete AMASS result. The existing downloader is resumable, but
first convert notebook-only sequence-manifest creation into a standalone,
hashable step and merge shard reports before quoting coverage.

## 8. Evaluation and aggregation

### 8.1 Common targets

Do not compare each model's own EMA KL as the primary endpoint. Use common
coordinate/kinematic targets or one frozen canonical teacher for anchored
evaluation. In unanchored evaluation, globally randomize and orbit-score the
teacher's odd targets so its canonical convention cannot act as an undeclared
anchor. Recommended common probes are:

- **odd:** right-minus-left ankle-speed energy and right-minus-left leg
  excursion, with a training-defined near-zero zone for sign scoring;
- **even:** forward speed and total lower-limb motion energy;
- **structured equivariant:** future centered joint coordinates, scored only
  after the declared path alignment or global orbit minimization; and
- **gauge:** relative edges and paths up to one global flip.

These are representation/mechanism targets. They are not force measurements.

### 8.2 Metrics

| Question | Primary or diagnostic metrics |
| --- | --- |
| Relative path | Equivalence-class path NLL; edge Brier/log loss; Hamming error minimized over global flip |
| Switch events | AUPRC; event precision/recall/F1 within ±1 block; segment IoU; boundary error |
| Anchored odd readout | Normalized MAE, predictive NLL, sign accuracy outside near-zero zone |
| Unanchored odd readout | Orbit MAE, symmetric-mixture NLL, calibration and sharpness; not two-sign coverage alone |
| Even/clean preservation | Common-target error and 2% clean non-inferiority margin |
| Selective use | Risk-coverage curve and area under selective-risk curve |
| Representation health | Feature variance, covariance, effective rank, odd energy |
| Geometry/integrity | Exact parity, commutation, no-copy masks, absolute-gauge leakage, sensor-bit decodability |
| Resources | Parameters, FLOPs, wall time, GPU model, peak memory, actual GPU-hours |

### 8.3 Correct aggregation and estimands

First stitch or average overlapping-window logits and latent predictions onto
each unique source-sequence block; otherwise overlapping source frames are
counted repeatedly. Compute full-path and switch-event metrics once per
sequence and corruption draw. For additive common-target losses, deduplicate
source target blocks before averaging (or preregister an explicitly
window-weighted estimand).

For additive losses such as sequence-level path NLL, Brier/log loss, Hamming,
and MAE, for each model seed and test identity:

1. calculate sequence/draw metrics with the same corruption draws;
2. average draws within sequence;
3. average sequences within identity; and
4. compare models on the same identity and paired seed.

Then report every seed, the seed mean/range, identity-level effect size, a
10,000-resample identity-cluster bootstrap 95% CI, and a paired identity-level
randomization test. Do not turn three seeds or thousands of overlapping windows
into the nominal sample size.

Pair seed IDs 7, 19, and 31 across SG-JEPA and the selected baseline. The
confirmatory effect is the average of the three seed-specific identity effects;
bootstrap identities jointly with the fixed seed set. This CI is conditional
on those seeds, while the per-seed range is descriptive.

Nonlinear metrics are not averaged from window-level versions. Prespecify both
the pooled **micro** estimand and an **identity-macro** sensitivity estimate for
AUPRC, event F1, segment IoU, and boundary error. Reliability/ECE and
risk-coverage/AURC are pooled diagnostics. Every cluster-bootstrap replicate
resamples identities (or source videos for GAVD) and recomputes the complete
nonlinear statistic, including event matching and curve construction.

### 8.4 Artifact contract

Each run directory should be immutable and self-contained:

```text
run-id/
├── effective_config.json
├── environment.json
├── hashes.json
├── capacity_and_compute.csv
├── corruption_manifest.sha256
├── seed-<n>_<variant>_best.pt
├── seed-<n>_<variant>_history.csv
├── seed-<n>_<variant>_validation_predictions.parquet
└── COMPLETE.json
```

The sealed evaluator writes test predictions to a new evaluation directory,
not into training directories. A central index is built by reading each
`COMPLETE.json`; it is never maintained by letting separate jobs overwrite one
shared `summary.csv`.

## 9. How to read possible results

| Observed pattern | Interpretation | Next action |
| --- | --- | --- |
| Oracle transport does not improve odd common-target error | Corruption is not consequential for the endpoint | Stop or redesign the endpoint; do not train SG-JEPA |
| Absolute gauge is decoded from “unanchored” inputs | Coordinate/metadata leak acts as an anchor | Fix neutralization and regenerate data |
| Raw/HMM fixes at least 95% of natural candidate events | Representation architecture is unnecessary for this setting | Report correction result; stop main architecture claim |
| SG-JEPA lowers path NLL but not common-target error | Better tracking only | Do not claim representation improvement |
| SG-JEPA improves odd error but harms clean/even by more than margin | Robustness tradeoff | Report honestly; inspect parity or capacity constraint |
| Hard MAP equals marginalization and calibration is equal | Posterior mixture adds no measured value | Prefer the simpler hard corrector |
| Marginalization improves ambiguous cells and risk-coverage without clean cost | Intended mechanism supported | Repeat decisive contrast across seeds and report bounded claim |
| Fixed-reflection model solves semantic $P$ | Known chirality transfers better than predicted | Reframe novelty around empirical transfer, not a unique new architecture |
| Generic paired transformer wins | Extra flexibility/compute matters more than gauge structure | Audit FLOPs and constraints; do not claim inductive-bias benefit |
| Unanchored global entropy collapses despite chance-neutral data | Model violates the output contract or exploits leakage | No identifiability-aware claim |
| Only severe synthetic cells show gains | Controlled stress robustness | Remove real-world/broad usefulness language |

## 10. GAVD audit lane

Audit raw image-coordinate outputs before any body frame using named hips. Use
two extractors only to diversify candidates. For the probability sample, draw
source videos within fixed camera-view and quality strata with known inclusion
probabilities. For the enriched sample, oversample low confidence, extractor
disagreement, limb crossings, and model-ranked switches; never use it for an
unweighted prevalence estimate.

Two raters, blinded to model score, label:

- coherent whole-lower-body convention swap;
- whole-limb-chain or joint-specific swap;
- person-track/identity error;
- no swap; or
- indeterminate anatomical visibility.

Adjudicate disagreements and report rater agreement. The source video is the
bootstrap unit. Pass the broad ecological gate only if the probability sample's
inclusion-weighted proportion of retrievable source videos containing a local
coherent event has a one-sided 95% lower bound above 1%, with at least 20 such
events across 10 videos and two view strata, consequential change in an odd
feature, and failures remaining after the transparent corrector. Report
retrieval response by stratum and restrict inference to retrievable videos if
nonresponse cannot be adjusted. Enriched cases support taxonomy and examples,
not frequency or unweighted calibration. Global per-video naming differences
do not justify a temporal representation model.

## 11. Final reproducibility and claim checklist

- [ ] Test identities were never used for architecture, severity, temperature,
  anchor, checkpoint, or margin selection.
- [ ] Effective model configs—not only a nominal base config—were saved.
- [ ] Every paired checkpoint records the repaired mask contract.
- [ ] Data, code, split, corruption, and environment hashes resolve.
- [ ] Gauge paths are defined at sequence level and agree across overlaps.
- [ ] The unanchored leakage probe passed before training.
- [ ] Common targets, rather than per-model EMA KL, drive the main comparison.
- [ ] Results aggregate to identities before inference.
- [ ] Every seed and failed job is visible; no best-seed selection occurred.
- [ ] Wall time, FLOPs, peak memory, and GPU-hours accompany parameter counts.
- [ ] An unanchored result is scored as a distribution/orbit, not forced sign.
- [ ] Partial swaps are labeled out-of-model rather than counted as coherent
  gauge successes.
- [ ] GAVD prevalence uses probability weights; enriched samples are separate.
- [ ] Kinematic targets are not called force, balance, diagnosis, or clinical
  outcomes.
- [ ] The conclusion follows the outcome table, including the simpler-method or
  null branch.
