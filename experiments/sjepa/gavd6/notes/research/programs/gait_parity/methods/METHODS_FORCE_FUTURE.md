# Study 03 methodology: fixed-reflection signed-force validation

> **Status:** retained biomechanics-validation protocol for the fixed-reflection
> baseline. Use it only after the repaired Study 01 baseline, GAVD data contract,
> and architecture/objective choice are frozen. It does not define Study 02's
> semantic-gauge method.
>
> **Companion proposal:** [README_FORCE_FUTURE.md](../proposals/README_FORCE_FUTURE.md)

## 1. Scope, target, and claim

The study evaluates a frozen, repaired AMASS-pretrained JEPA for participant-held-out prediction of a signed force asymmetry:

$$
y_{prop} = \log(J_R / J_L).
$$

`J_R` and `J_L` are integrated, positive propulsive impulses from force plates. Use natural logarithms and an audited reporting-limit rule; do not turn low-force contacts into arbitrary extreme ratios. Force data never enters pose extraction, JEPA training, event detection, or model selection.

The outcome is biomechanical association, not clinical diagnosis, causation, prognosis, or treatment response. Paretic/more-affected side and force sign are distinct variables and are analysed separately.

## 2. Data and independence

Use a stroke cohort for the primary test only after verifying participant IDs, bilateral contacts, force coverage, calibration, and split-half target reliability. Require at least 30 eligible stroke participants after force-quality screening; this is a feasibility floor, not a power claim.

Use a Parkinson's cohort only as a locked replication: audit fields and participant IDs, create its partition, and seal prediction outcomes before architecture or tuning decisions. MoVi walking trials test real camera variation with actors grouped across views. GaitRec is a target/compatibility sanity check, not a substitute for the primary cohort.

AMASS contributes pretraining data only. Record source dataset, subject, sequence, motion type, frame rate, and mapping provenance; exclude any known participant/source overlap with downstream cohorts.

The participant is the independent unit. Keep all of one participant's cycles, trials, visits, medication conditions, camera views, and reflected copies in one outer fold. Aggregate cycle to trial to participant before scoring or bootstrapping.

## 3. Input, reflection, and repaired JEPA

Use a frozen Core11 (or separately declared compatible) skeleton contract: joint order, coordinates, body frame, frame rate, scaling, bilateral permutation, visibility semantics, and transform replay are all versioned. Anatomical reflection swaps bilateral joints and their validity/confidence, negates only the mediolateral coordinate, and leaves time, forward, and vertical axes unchanged. Test reflection twice on real data and visually inspect replayed overlays.

The JEPA must already have passed these gates:

- branch-specific orbit-closed masks: `m_B[t,p(j)] = m_A[t,j]`;
- equivalent closure for validity, motion energy, tube/block masks, and paired draws;
- validity/confidence embeddings and attention padding masks in every encoder/predictor path;
- exclusion of invalid coordinates and displacement targets from losses and pooling; and
- synthetic copyability tests covering mirror counterparts, temporal neighbours, joint neighbours, and cross-attention receptive fields.

For every claimed equivariant layer, test swap-then-process equals process-then-swap through online encoder, EMA teacher, predictor, and masked prediction. Exact commutation is necessary but not evidence of useful force prediction. Monitor even/odd variance, effective rank, energy, cosine structure, teacher entropy, and finite norms so a zero odd channel cannot pass as success.

## 4. Model and training controls

Freeze the architecture/objective chosen by the GAVD protocol. Record its pretraining provenance, EMA schedule, visible-token budget, MAMP condition, checkpoints, code/environment hashes, source-window exposures, parameters, FLOPs, throughput, memory, and GPU-hours.

The two co-primary comparisons are:

| Model | Required output rule | Purpose |
| --- | --- | --- |
| `reflection_equivariant` | exact odd readout, no bias in odd linear head | proposed encoder-wide structure |
| `paired_unconstrained` | identical paired inputs, cross-attention, and odd readout | generic paired-fusion control |
| `odd_output` | $(q(E(x))-q(E(Mx)))/2$ | cheap output-repair control |

All models train/evaluate on matching original and mirrored participant windows. Retain `standard_sjepa`, mirror augmentation, raw kinematics, random encoder, side-agnostic, nuisance-only, and appropriate ST-GCN/MotionBERT-style baselines as secondary context. Report exposure-matched and compute-matched configurations independently.

## 5. Participant-safe evaluation

Create frozen outer participant folds and grouped inner folds. Within each outer-training split, fit normalization, feature pooling, readout hyperparameters, calibration, and low-label subset selection. Never use an outer-test participant for any of these decisions or for target-domain SSL.

Evaluate recorded and reflected pairs together: score `(x,y)` and `(M(x),-y)`, average their loss, then give the participant one aggregated contribution. This prevents a common-side prediction from benefiting from prevalence or from reflected copies becoming pseudo-participants.

Report a frozen encoder/readout analysis first. If supervised fine-tuning or target-train SSL adaptation is permitted, report it in separate rows and use only outer-training participants. It is not zero-shot transfer.

## 6. Endpoints, uncertainty, and falsification

Primary endpoints are the paired, participant-orbit-averaged MAE differences of `reflection_equivariant` versus `odd_output` and versus `paired_unconstrained` at the largest prespecified label budget. Pre-register a reliability-based practical MAE margin and use simultaneous 97.5% participant-bootstrap intervals. Report all paired seed deltas; seeds are training replicates, not extra participants.

Secondary endpoints are normalized MAE, untruncated $R^2$, calibration slope/bias, sign accuracy outside a prespecified near-zero zone, and learning-curve area. Use nested participant subsets of 4, 8, 16, and all eligible people to test label efficiency.

Run a frozen corruption suite: missing joints, pose noise, one-sided motion attenuation with known sign/dose, coordinate-frame perturbation, temporal gaps, and real camera changes in MoVi. A left/right swap is a falsification test, not a robustness target. Compare against static/side-agnostic and nuisance features to expose non-gait shortcuts.

## 7. Completion rules

| Gate | Requirement |
| --- | --- |
| Target | audited force reliability and participant-safe cohort |
| Architecture | repaired masks/validity, copyability test, commutation, and healthy parity channels |
| Fairness | matched controls, paired seeds, sealed manifest, exposure and compute reports |
| Primary evidence | practical benefit over both co-primary controls on held-out participants |
| Generality | locked Parkinson's replication and MoVi geometry results, reported whether positive or negative |

If a gate fails, narrow the conclusion. Do not weaken grouping, relabel the target, substitute GAVD classification, select a checkpoint after test inspection, or convert an inconclusive interval into equivalence. Release code, contracts, splits, manifests, participant-level predictions where permitted, and the complete negative/positive result record.
