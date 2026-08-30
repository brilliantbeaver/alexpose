# Latent Laterality implementation prompt

**Role**: You are an expert JEPA researcher specializing in human motion,
reflection equivariance, structured latent-variable models, and careful
experimental design.

**Task**: Carefully inspect the current Latent Laterality study and implement,
in one coherent pass, the local code needed for the next decisive experiments.
Do not stop after writing a plan. Reuse the existing implementation where it is
sound, make the smallest scientifically meaningful changes, run focused tests
and CPU synthetic smoke experiments, and leave the real AMASS/GAVD data and
GPU runs for the user on HAIC.

Read these sources before changing code:

- `docs/studies/latent-laterality/README.md`
- `notes/latent-laterality/theory.md`
- `notes/latent-laterality/experiments.md`
- `docs/studies/latent-laterality/swap-probe.md`
- `docs/studies/fixed-reflection-baselines/protocol.md`
- `docs/studies/frozen-core11-probe-results.md`
- `outputs/swap-probe-seed7/summary.csv`

## Research objective

Determine whether a predictive motion model can preserve useful odd
(right-minus-left) information when bilateral token identity changes locally,
while remaining honest about the global sign when no anatomical anchor exists.
The experiment should answer two questions:

1. Does SG-JEPA outperform a transparent temporal swap corrector once the
   correspondence problem is genuinely ambiguous?
2. For transfer to real gait video, which representation source is most useful:
   AMASS-only pretraining, full-GAVD pretraining, or AMASS followed by GAVD
   self-supervised adaptation?

Do not assume that GAVD condition labels supply laterality. Stroke,
Parkinsonian, myopathic, and cerebral-palsy labels describe gait patterns; they
do not reliably identify an affected anatomical side. Use them only for
secondary condition readouts. Signed laterality must come from a controlled
coordinate convention, known synthetic transformations, or an independent
side anchor.

## Start from the evidence already present

The checkpoints in `outputs/repaired-jepa-seed7-v2` predate the repaired
branch-specific orbit-closed mask contract. They may be used for loading and
diagnostic checks, but not as final baselines.

The existing swap probe also shows that its corruption is too easy:
continuity, learned MAP, structured posterior, and oracle correction perform
almost identically. Treat this as a stop signal. Do not build a large SG-JEPA
training study around a benchmark that a transparent continuity rule already
solves.

The current frozen AMASS-to-GAVD probe does not show an advantage over raw or
random features, and its 96-sequence evaluation is source-confounded. This
motivates a clean source-transfer comparison on the full GAVD cohort rather
than assuming AMASS-only pretraining will transfer.

## Implement only the research-critical repairs

First, make the input representation scientifically valid. Add a
gauge-neutral AMASS frame that never reads named left/right joints when choosing
orientation. For the main experiment, it is acceptable to restrict AMASS to a
traveling stratum whose forward direction is estimated from pelvis trajectory
and signed by net displacement. Keep sensor reflection `M`, semantic token
permutation `P`, and global chart choice as separate operations.

Make observation validity an explicit encoder input. Invalid zero-filled joints
must not be indistinguishable from valid joints at the origin, and invalid
tokens must never become JEPA targets. Transport validity through `P` exactly
as coordinates are transported.

Enforce branch-specific orbit-closed masks for every paired model:

```text
mask_B[t, P(j)] = mask_A[t, j]
valid_B[t, P(j)] = valid_A[t, j]
```

Add focused tests for gauge neutrality, explicit validity, parity, and the
no-copy mask invariant. Avoid broad infrastructure work that does not change
the validity or interpretation of an experiment.

## Replace the easy swap probe with a sequence-level benchmark

Keep E0 as a historical diagnostic and implement the real benchmark at full
sequence scope before cutting windows. Overlapping windows must share the same
latent convention on shared frames. Include clean paths, global swaps,
variable-duration swapped segments, and repeated low-rate switches.

Make boundaries ambiguous without erasing all motion evidence. Prefer events
near turns, limb crossings, low speed, or bilateral motion. Apply short
boundary-centered gaps or interpolation to both true switch boundaries and
matched clean pseudo-boundaries so the missingness pattern does not reveal the
answer. Keep evidence before and after each boundary so longer-context
inference can still help.

Implement a two-state temporal model that provides both a MAP path and
calibrated block/edge marginals. Evaluate unanchored paths up to one global
flip. Prevent the posterior from becoming an arbitrary loss gate by calibrating
it separately and detaching it inside the JEPA prediction objective.

Do not proceed to SG-JEPA training unless the revised benchmark satisfies all
conditions on held-out synthetic or calibration identities:

- A mask-only model improves relative path NLL over an input-free prior by no
  more than 1%.
- An absolute convention probe on unanchored examples remains near chance; its
  upper 95% confidence bound should remain below `0.55` AUROC.
- Oracle correction improves swapped-event odd error by at least 5% over
  continuity.

If the oracle does not beat continuity, the simpler corrector is the research
result. Report it and stop expanding the architecture.

## Build the smallest decisive model comparison

Implement these common-interface controls:

- raw coordinates with temporal correction;
- standard S-JEPA;
- standard S-JEPA with mirror augmentation;
- a capacity-matched paired-unconstrained control;
- reflection-equivariant JEPA with repaired paired masks;
- correction-first S-JEPA;
- SG-JEPA with a structured correspondence posterior;
- a uniform 50:50 posterior ablation;
- oracle correction; and
- raw-coordinate and random-encoder downstream baselines.

Use the same windows, corruption draws, masks, optimizer-update budget, and
downstream readout for comparable learned arms. Do not compare architectures by
self-KL because each model has its own EMA teacher. Compare them through common
odd/even kinematic targets and the same frozen readout.

For the source-transfer question, support three training routes with the same
encoder and objective wherever possible:

- AMASS only;
- full GAVD only; and
- AMASS pretraining followed by label-free GAVD adaptation.

Keep AMASS splits identity-disjoint. Split GAVD by source video before any
encoder or probe fitting because multiple sequences cut from one video are not
independent examples. Train GAVD routes only on outer-training videos and keep
condition labels out of the JEPA loss. Balance sampling at the source-video
level so prolific videos or conditions do not dominate. Preserve detector
confidence/validity and use the same Core11 ordering and parity definitions
across AMASS and GAVD.

Implement one GAVD-to-Core11 adapter for every GAVD route. It must not use a
named left/right hip axis to define an allegedly unanchored orientation. Use a
gauge-neutral travel frame where reliable; otherwise retain a declared
image-space chart and evaluate odd outputs only up to global sign. Do not hide
this domain difference inside the encoder.

Screen the three data routes with standard S-JEPA and the
reflection-equivariant model first. Advance only the strongest route and its
most informative baseline to the more expensive SG-JEPA comparison. This keeps
the experiment focused on whether structured correspondence adds value rather
than multiplying every architecture by every dataset. Use seed 7 for screening,
then confirm only the selected SG-JEPA contrast with seeds 19 and 31.

## Evaluate the claims that matter

Report clean and corrupted odd/even NMAE or NMSE, signed accuracy only when an
independent anchor exists, unanchored error up to global sign, path Hamming
error, switch F1, posterior NLL/Brier score, clean false-switch cost, and
feature variance. Aggregate AMASS results by identity and GAVD results by
source video. Report MAP-versus-posterior differences by uncertainty stratum so
the structured posterior is credited only when its uncertainty is useful.

The key comparisons are:

1. oracle versus continuity, which establishes whether correspondence matters;
2. SG-JEPA versus correction-first S-JEPA, which tests whether structured
   uncertainty belongs inside representation learning;
3. reflection-equivariant versus standard S-JEPA, which tests whether explicit
   parity preserves odd information; and
4. AMASS-only versus GAVD-only versus AMASS→GAVD, which tests whether clean 3D
   motion, in-domain video, or staged adaptation provides the best transfer.

Treat GAVD condition classification and natural swap detection as secondary
ecological evaluations. They cannot establish clinical validity, and a signed
claim is not allowed without side ground truth or an independent anchor.

## Local work versus HAIC work

Complete all source changes, focused unit tests, synthetic fixtures, CPU smoke
runs, and simple command-line entry points locally. Do not add elaborate run
ledgers, checkpoint registries, dashboards, generic orchestration, or extensive
provenance machinery. Retain only the configuration fields and outputs needed
to interpret the experiment.

The user will handle all HAIC-only work: AMASS and full-GAVD manifests, dataset
paths and downloads, AMASS conversion, Slurm submission, GPU training, real
checkpoint selection and storage, and final evaluation runs. Do not SSH to
HAIC, modify real manifests, move checkpoints, or attempt full training from
this environment.

At handoff, give the user a short ordered HAIC run guide containing:

- the exact scripts or commands to run;
- the manifest, tensor-root, and checkpoint inputs the user must supply;
- the order of the benchmark gate, source-transfer screen, and SG-JEPA run;
- the small set of result tables the user should return; and
- explicit stop conditions when continuity or a simpler representation wins.

## Output

Implement the complete local research path rather than producing another long
proposal. Keep code changes narrow, run the focused tests and synthetic smoke
experiments, and summarize:

- what was implemented;
- what the local evidence shows;
- whether the revised benchmark is ready for real training; and
- exactly what the user must run next on HAIC.
