# Proposal: semantic-gauge predictive representations

## Executive decision

**Conditional go, with a narrower claim.** The original direction combined a
valuable identifiability observation with too many dependencies: repaired
fixed-reflection training, full GAVD retrieval, two pose extractors, human
annotation, a new gauge architecture, and an independently side-anchored force
cohort. That is not a credible two-week study.

The revised study asks one controlled representation-learning question on
AMASS and runs a GAVD phenomenon audit in parallel. The force/stability cohort
is future validation. The result can still be significant: it tests whether a
predictive learner should marginalize uncertainty about *what its input tokens
mean*, not merely become invariant to a known geometric augmentation.

The project earns a strong claim only if SG-JEPA beats a transparent temporal
corrector and synchronization-plus-S-JEPA on a common target. If the corrector
wins, the simpler method is the scientific result.

## Research question

Pose models usually assume that a row named `left_ankle` continues to describe
the same anatomical ankle. In video that assumption can fail during crossing,
occlusion, blur, or tracking instability. Coordinates can remain plausible
while the left/right token assignment switches for part of a sequence.

The research question is:

> **Under intermittent latent bilateral token relabeling, can a predictive
> motion representation recover the identifiable relative correspondence,
> preserve parity-sensitive information for an independently anchored readout,
> and outperform correction-first baselines while remaining calibrated about
> an unanchored global sign?**

This question is insightful because it separates three problems normally
collapsed into “mirror robustness”:

1. a camera or coordinate-frame action;
2. a physical anatomical reflection; and
3. uncertainty about which observed token names an anatomical side.

Only the third is latent in this study. The distinction changes both the model
and the legitimate output.

## Hypotheses

- **H1 — relative recovery:** temporal evidence can identify whether adjacent
  blocks use the same convention, with the one-sided 95% lower confidence bound
  for identity-mean equivalence-class path-NLL improvement over the fixed
  input-free Markov prior above zero, even when the global convention is hidden.
- **H2 — representation value:** in one prespecified local-swap-plus-occlusion
  regime, the one-sided 95% lower confidence bound for
  $(E_{base}-E_{SG})/E_{base}$ exceeds 0.05 against the strongest non-oracle
  corrector/synchronizer, after equal identity weighting.
- **H3 — ambiguity integrity:** global randomization prevents above-chance
  absolute-gauge leakage; relative-edge probabilities remain calibrated; and
  an independent anchor improves signed kinematic recovery. A 50:50 global
  root is imposed by the unanchored model, so it is an integrity assertion, not
  evidence of learned calibration.
- **H4 — no robustness tax:** against the strongest baseline, the one-sided
  95% upper bound for the relative clean-error increase
  $E_{SG}/E_{base}-1$ is below 0.02, and the upper bound for the absolute edge
  Brier-score difference is below 0.01.
- **H5 — ecological relevance, conditional:** coherent local convention events
  occur in the probability-sampled, retrievable GAVD source-video frame with a
  one-sided 95% lower prevalence bound above 1%, with at least 20 confirmed
  events across 10 videos and two view strata. Failure of H5 narrows the paper
  to controlled robustness; it does not invalidate H1–H4.

The 5%, 2%, and 0.01 margins are proposed preregistration values. They may be
changed using validation-only pilot evidence before the test set is opened,
but the final values and rationale must then be frozen. H1, H2, and H4 are
tested in that fixed gatekeeping order at one-sided $\alpha=0.05$; testing stops
at the first failed gate. H3 is an integrity requirement, H5 licenses only the
ecological extension, and secondary severity contrasts use Holm correction.

## Why the problem matters

### Representation learning

Ordinary augmentation says that a transformation is known and the learner
should ignore it or respond predictably. Here the transformation is unknown
and temporally structured. Forcing invariance can destroy the signed content
needed later; forcing a single convention can teach confident errors. The
representation must instead keep invariant content, transport equivariant
content, and retain a distribution over unresolved transport.

This makes the work relevant beyond gait. Sensor arrays, multi-object tracks,
keypoint schemas, and entity slots all face semantic correspondence errors.
The transferable idea is to couple a small structured correspondence posterior
to predictive latent learning and to type downstream outputs by what is
identifiable.

### Biomechanics

Right-minus-left step length, joint excursion, contact timing, and related
functionals have meaningful sign only when side correspondence is trustworthy.
The revised study tests whether that information survives corruption and
whether the model knows when it may attach an anatomical sign. It does not
claim that kinematics identify ground-reaction force, diagnosis, balance, or
treatment response. Those require measured outcomes and participant-safe,
side-anchored validation.

## Relation to the previous Ideas 05 and 09 program

[Idea 05](../../../../ideas/05-signed-laterality-decodability/) was a post-hoc
measurement instrument: freeze an S-JEPA, decode a constructed signed
left-minus-right scalar, compare with raw coordinates, and ask whether a known
anatomical mirror negates the readout. [Idea 09](../../../../ideas/09-reflection-equivariant-symmetry-axis/)
was an architectural inductive bias: supply $x$ and its known anatomical
reflection $Mx$, enforce exact layerwise equivariance, and expose even/odd
channels. The joined GaitParity program carried those ideas toward a future
side-labeled force endpoint.

That approach assumes the action is known, global, and deliberately applied.
It asks “does the model obey this mirror?” Study 02 asks “which local token
convention generated this observation, what can be inferred only relatively,
and what should the model output when the global answer is absent?” It adds:

- a latent temporal correspondence process rather than a supplied mirror;
- posterior-aware masked prediction rather than fixed equivariance alone;
- anchored and unanchored training/evaluation as separate regimes; and
- a decision contract that returns a signed value only with independent
  anatomical evidence.

The old models remain essential negative and capacity controls. They are not
expected to solve a semantic permutation merely because they commute with a
physical reflection.

## Related work and the remaining gap

No individual ingredient is new. The contribution must be framed as a tested
integration, and any “first” claim should be avoided unless a submission-time
search supports it.

| Area | What prior work already provides | What this study tests beyond it |
| --- | --- | --- |
| Predictive skeleton learning | [S-JEPA](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/4755_ECCV_2024_paper.php) predicts masked skeleton embeddings; [seq-JEPA](https://arxiv.org/abs/2505.03176) learns invariant/equivariant content from supplied actions | The action is a latent semantic correspondence with a posterior and an unanchored global offset |
| Known reflection | [Chirality Nets](https://proceedings.neurips.cc/paper_files/paper/2019/file/1f88c7c5d7d94ae08bd752aa3d82108b-Paper.pdf) builds coordinate reflection plus side exchange into pose networks | Distinguish a known physical reflection from an unknown token-name permutation that may switch locally |
| Learned symmetry/canonicalization | [Learned canonicalization](https://proceedings.mlr.press/v202/kaba23a.html), [probabilistic symmetrization](https://proceedings.neurips.cc/paper_files/paper/2023/hash/3b5c7c9c5c7bd77eb73d0baec7a07165-Abstract-Conference.html), and [SymPE](https://proceedings.iclr.cc/paper_files/paper/2025/hash/c7138635035501eb71b0adf6ddc319d6-Abstract-Conference.html) learn or randomize symmetry choices | Couple a time-local bilateral correspondence distribution to masked motion prediction and parity-typed decisions |
| Learned actions | [Winter et al.](https://proceedings.neurips.cc/paper_files/paper/2022/hash/cf3d7d8e79703fe947deffb587a83639-Abstract-Conference.html) separate invariant content from a learned action component, including permutations; [SEN](https://proceedings.mlr.press/v162/park22a.html) maps unknown input actions to known feature actions | Infer a time-indexed correspondence path, test its calibration, and compare joint predictive marginalization with correction-first use |
| Gauge methods | [Gauge Equivariant Transformer](https://proceedings.neurips.cc/paper/2021/hash/e57c6b956a6521b28495f2886ca0977a-Abstract.html) handles arbitrary local coordinate frames on manifolds | Use a finite gauge analogy for anatomical token semantics, not mesh coordinate orientation |
| Synchronization | [Permutation synchronization](https://proceedings.neurips.cc/paper_files/paper/2013/hash/3df1d4b96d8976ff5986393e8767f5b2-Abstract.html) aggregates noisy pairwise correspondences | Learn relative evidence from predictive motion and compare synchronization-first with joint learning |
| Pose-swap correction | [PoseFixeR](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1009989) detects and corrects left/right leg reversals using gait structure; [SwapPF](https://www.scitepress.org/Papers/2026/144493/144493.pdf) uses gait periodicity and particle filtering | A direct threat and required baseline: determine whether posterior-aware representation learning adds anything beyond correction |
| Ambiguous pose outputs | [DiffPose](https://openaccess.thecvf.com/content/ICCV2023/papers/Holmquist_DiffPose_Multi-hypothesis_Human_Pose_Estimation_using_Diffusion_Models_ICCV_2023_paper.pdf) predicts multiple plausible human poses under monocular ambiguity | Restrict the multiple hypotheses to a declared semantic orbit and test anchored versus quotient-valued decisions |
| Geometry-aware correspondence | [Telling Left from Right](https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_Telling_Left_from_Right_Identifying_Geometry-Aware_Semantic_Correspondence_CVPR_2024_paper.html) shows that visual geometry/orientation cues can resolve correspondence | State the impossibility result only when such cues and other anchors are excluded or balanced |

The defensible prospective contribution is:

> a predictive representation study in which coherent bilateral token identity
> is a local latent variable; relative correspondence is inferred and
> calibrated over time; prediction marginalizes that uncertainty; and the
> output is signed only when an independent anchor fixes the component's
> global convention.

## Audit of `outputs/repaired-jepa-seed7-v2`

### What is present

The directory contains four seed-7 histories and checkpoints, each trained for
100 epochs on AMASS Core11. The effective models use 64-dimensional embeddings
and are closely parameter matched:

| Variant | Parameters | Selected epoch | Validation KL | Feature health at selection | Programmed masked commutation |
| --- | ---: | ---: | ---: | --- | ---: |
| `standard_sjepa` | 821,866 | 95 | 0.2555 | variance 0.0695; effective rank 16.21 | Not applicable |
| `paired_shared_no_cross` | 821,866 | 98 | 0.2653 | even 0.0873 / 17.16; odd 0.0775 / 16.01 | 0 |
| `reflection_equivariant` | 821,860 | 98 | 0.2791 | even 0.0896 / 16.56; odd 0.0785 / 15.27 | 0 |
| `paired_unconstrained` | 822,214 | 98 | 0.2304 | even 0.0830 / 15.93; odd 0.0844 / 15.64 | 2.7574 |

Feature-health entries are variance / effective rank. The standard arm is the
only one whose surviving summary records runtime: 7,326 seconds for 100 epochs,
0.405 GiB peak allocated CUDA memory, and 236,170 updates at the selected
epoch 95 (248,600 updates were executed across the full history). The
user-observed full-suite time of about 12 hours is the planning budget.

These KL values must not be used to rank architectures. Each encoder predicts
its own EMA teacher with a different target distribution and teacher entropy.
A lower self-KL is not a common-target representation result.

### Fatal limitations for scientific interpretation

1. The four checkpoints omit `metadata.paired_mask_contract` and predate the
   branch-specific $P$-closed mask implementation. In variants with
   cross-branch interaction, the legacy same-index mask can leave a physical
   counterpart visible in the other branch and permit copying.
   Zero commutation does not detect that leakage.
2. `run_config.json` and `summary.csv` were overwritten by the last
   independently launched standard-only arm. Runtime and memory for the other
   arms are lost; the directory is not an immutable four-arm run.
3. `evaluate_test=false`; there is no test result. Validation used one
   deterministic corruption/mask draw and one training seed.
4. Histories report 79,552 training-window exposures per epoch because 79,535
   windows are padded to 2,486 complete batches of 32; the final batch repeats
   17 examples. This is expected, not a manifest discrepancy. However, the run
   records absolute HAIC paths rather than manifest/code hashes, so the exact
   old input still cannot be authenticated from the copied artifacts alone.
5. The top-level configuration records a nominal 96-dimensional template,
   while capacity resolution produces the effective 64-dimensional checkpoint
   configurations. Future summaries must record both, per arm.
6. No checkpoint was trained or evaluated on semantic $P$-swaps, explicit
   missingness inputs, a gauge posterior, calibration, an anchor, an
   orbit-valued output, or a common downstream target.

### What the current run establishes

It establishes that the training/loading scaffold executes, close parameter
matching is possible, the selected features are not obviously constant, and
the tied layers obey their programmed known-reflection algebra under the old
mask. It establishes neither a repaired fixed-reflection comparison nor
semantic-gauge usefulness.

### How the new experiments expand it

The revised experiment retains Core11, corpus-qualified identity-disjoint splitting, the EMA
teacher, capacity accounting, and commutation tests. It adds a gauge-neutral
coordinate contract, sequence-consistent corruption manifests, validity-aware
tokens, repaired orbit-closed masks, a relative-state posterior, anchored and
unanchored losses, common-target probes, identity-level statistics, and
immutable one-run directories. The existing weights support pipeline diagnosis
only; confirmatory baselines are rerun.

## Data

### AMASS-Gauge: the core causal benchmark

[AMASS](https://amass.is.tue.mpg.de/) unifies motion-capture corpora in one body
parameterization. The current local Core11 manifest contains 8,854 sequences
and 151/19/19 train/validation/test identities. It yields 79,535/5,936/8,220
overlapping 64-frame windows after the repository's end-window rule.

For each full sequence—not independently for overlapping windows—the benchmark
samples a piecewise-constant path $g_t$, applies the coherent five-pair Core11
token permutation $P^{g_t}$, and saves every switch, occlusion, noise, sensor
reflection, and anchor. Overlapping windows inherit the same path. The core
cells are:

1. clean;
2. global swap;
3. one local segment of 1, 2, 4, or 8 blocks;
4. repeated Markov switches;
5. a local segment plus boundary-centered occlusion/noise;
6. sparse trusted anchors; and
7. partial joint/limb swaps as out-of-model stress tests.

The provisional primary regime is one 4-block (16-frame) local swap with
moderate bilateral distal-joint occlusion and noise. Exact magnitudes are fixed
from validation or, if available, the GAVD audit before test access. Five to ten
deterministic test corruption draws are shared by every model.

The present conversion is not gauge neutral: 64.60% of sequences use a
named-left/right `hip_facing_fallback`, and another 5.95% use a related fallback
after anatomy/trajectory disagreement. Applying $P$ after that frame can leak
absolute convention. Before training, build `gauge-neutral-v1` without a
side-labelled fallback, balance a hidden sensor reflection independently of
the semantic bit, and require a strong raw-feature absolute-gauge probe to have
AUROC near 0.5 with upper 95% CI below 0.55.

Only 2,607 of 8,854 current sequences use the travel-based facing method. A
naive exclusion of every fallback would leave roughly 10,921/942/1,551
train/validation/test windows and 107/14/15 represented split identities,
dropping four current test identities. The neutral conversion must therefore
publish an attrition table before training and either retain stationary motion
under a genuinely orientation-free/random-yaw contract or revise sample-size,
runtime, and inference claims to the retained cohort.

AMASS supplies canonical joint identity and kinematics, not ground-reaction
force. The controlled odd targets are right-minus-left ankle-speed energy,
joint excursion, or contact-timing proxies defined before fitting. Clean speed
and total motion energy are even controls.

### GAVD: optional ecological audit

The [GAVD paper](https://arxiv.org/abs/2407.04190) reports 1,874 sequences and
links/annotations for more than 450 public source videos; the
[official release](https://github.com/Rahmyyy/GAVD) distributes the metadata;
researchers retrieve videos independently under platform, ethics, and copyright
requirements. It has camera-view and gait-pattern annotations but no verified
anatomical-side labels or kinetics.

The local cache has only 96 sequences from 18 source videos and one MediaPipe
pipeline. It is inadequate for prevalence or transfer claims. The audit runs in
two lanes:

- a probability-stratified source-video sample for weighted prevalence; and
- a separate candidate-enriched sample for error taxonomy and ranking stress
  tests.

Two blinded human raters label `correct`, `coherent swap`, `partial/joint
error`, `person-track error`, or `indeterminate`, followed by adjudication.
Pose-extractor disagreement creates candidates, not truth. A broad ecological
claim requires a probability-sample, inclusion-weighted estimate whose
one-sided 95% lower bound exceeds 1%, plus at least 20 confirmed local coherent
events across at least 10 source videos and two view strata, consequential
odd-feature change, and failures remaining after transparent correction. The
inference frame is retrievable source videos; retrieval response is reported
by stratum, and no wider claim is made if nonresponse cannot be adjusted.
Enriched cases establish taxonomy and examples only. Otherwise GAVD is an
audit/null result, not support for SG-JEPA in the wild.

## Methodology

### Model

SG-JEPA lifts every block into the pair ([x,Px]) with a shared encoder. The
sum/difference projection produces exact even and odd features. A small edge
head scores whether adjacent blocks share the same convention. An exact
two-state chain conditional random field combines those scores with a switch
prior to obtain the relative path posterior. The JEPA predictor marginalizes
the transported odd targets rather
than forcing a hard convention. [Theory](./theory.md) defines the normalized
mixture and unanchored quotient loss.

The minimum implementation is a chain. Skip edges and cycle losses are deferred
because a chain has no cycles. The model represents one coherent Core11
(\mathbb Z_2) action; a richer product group for partial joint errors is future
work.

### Decisive comparisons

| Arm | Question isolated |
| --- | --- |
| Raw continuity + Viterbi/HMM | Can kinematics alone repair the sequence? |
| Corruption-trained standard S-JEPA + gauge head | Does ordinary predictive learning plus uncertainty suffice? |
| Repaired fixed-reflection JEPA + same head | Does known anatomical-reflection structure transfer to unknown semantic relabeling? |
| Compute/capacity-matched unconstrained paired transformer | Are gains just a second branch or more computation? |
| Relative synchronization + corrected input + ordinary S-JEPA | Is correction before representation learning enough? |
| SG-JEPA | Does joint posterior-aware predictive learning add value? |
| Oracle path; uniform posterior | Upper mechanism bound and non-informative lower baseline |

Hard-MAP transport, no temporal messages, no parity split, and no validity
embedding are seed-7 ablations. Output odd/even symmetrization and mirror
augmentation are inexpensive secondary controls. The strongest validation
baseline—not the easiest one—is carried to the three-seed test.

### Endpoints

The primary hierarchy is:

1. length-normalized equivalence-class path negative log-likelihood,
   $-(K-1)^{-1}\log[q(g^*\mid x)+q(g^*\oplus1\mid x)]$, which scores full-path
   posterior mass per identifiable relative decision, up to its unanchored
   global flip; then
2. a shared-target parity-sensitive prediction error in the primary corruption
   regime.

Secondary endpoints include edge Brier/log loss, switch AUPRC, event F1 with a
fixed one-block tolerance, segment intersection-over-union, path Hamming error
minimized over the global flip, anchored odd normalized MAE and sign accuracy,
unanchored orbit MAE and symmetric-mixture NLL, even-task preservation,
risk-coverage curves, feature variance/effective rank, and measured FLOPs and
GPU-hours.

Each corruption draw is identical across arms. Results stitch overlapping
evidence to unique sequence blocks, aggregate draws to sequence, then sequences
to equal-weight identity. The independent units are the test identities
retained by the frozen neutral manifest; the current source split has 19, but
neutral-conversion attrition may reduce this count. Pair seed IDs 7, 19, and 31
between SG-JEPA and the selected baseline. The confirmatory effect averages
the three seed-specific identity effects; its identity bootstrap resamples
identities jointly and is conditional on this fixed seed set. Report every seed,
the descriptive seed mean/range, a 10,000-resample identity-cluster bootstrap
CI, and a paired identity-level randomization test. Nonlinear event and curve
metrics are recomputed within each bootstrap replicate. Test is evaluated once
after architecture, corruption, temperature, anchors, and primary margins are
frozen.

## Two-week experiment and decision plan

The hard budget is eight current-suite equivalents, approximately 96
reference-GPU hours if each 12-hour suite occupies one accelerator:

| Work | Budget | Decision value |
| --- | ---: | --- |
| Integrity, profiler, leakage/oracle sanity | 6 GPU-h | Prove the benchmark is nontrivial and gauge neutral |
| Repaired four-arm Gate-0 baseline | 12 GPU-h | Replace the invalid old-mask artifacts |
| Seed-7 learned-arm screen, then seeds 19/31 for SG-JEPA and the selected baseline | Up to 54 GPU-h | A ceiling/reserve, not a requirement; nonfinal arms stay seed 7 |
| Four seed-7 ablations | 8 GPU-h | Attribute a gain to marginalization, temporal evidence, parity, or validity |
| Decisive-ablation/job-failure contingency | 4 GPU-h | Spend only after a validation gate |
| Sealed AMASS test inference | 4 GPU-h | Shared 5–10 draws, run once |
| Optional paired GAVD extraction | 8 GPU-h | Hard stop; retrieval and annotation remain CPU work |

Days 1–2 freeze claims, hashes, splits, common metrics, and the neutral
corruption generator; implement raw/Viterbi/oracle baselines and repaired mask
tests. GAVD retrieval and annotation preparation start in parallel. Day 3 runs
short integration tests only. Days 3–8 run the validation-only learned matrix
and ablations. Day 8 freezes the winning contrast and calibration. Day 9 opens
the AMASS test once. Days 10–12 run clustered statistics and figures; days
13–14 write and package the controlled claim.

Full training stops if the gauge-neutrality or oracle-consequence gates fail.
Architecture expansion stops if the correction-first baseline is within 5%
and equally calibrated. The GAVD workflow never blocks completion of the
controlled AMASS paper.

## Claims licensed by possible outcomes

| Outcome | Licensed conclusion |
| --- | --- |
| SG-JEPA wins the paired common-target and calibration comparison, including moderate corruption | Posterior-aware latent correspondence improved controlled predictive motion representation under the coherent-swap model |
| SG-JEPA improves path detection but not common-target prediction | Better correspondence inference, not a representation-learning gain |
| Corrector/synchronizer matches SG-JEPA | Use the simpler correction pipeline; architecture claim rejected |
| Gain appears only under severe synthetic errors | Synthetic robustness result with limited practical scope |
| Unanchored outputs choose a confident global sign | Identifiability contract failed even if point accuracy is high |
| Natural GAVD gate passes | Add carefully bounded evidence that coherent local convention errors occur in uncontrolled video |
| Natural GAVD gate fails | Report the audit; make no in-the-wild necessity claim |

## Limitations and responsible interpretation

- The coherent Core11 swap is narrower than real pose-estimation error.
- A clean AMASS teacher is an anchor; unanchored batches must explicitly hide
  and quotient its global convention.
- Synthetic corruption labels make this controlled corruption-supervised
  representation learning, not purely self-supervised learning.
- Population asymmetry or a handed coordinate frame can leak absolute side;
  the gauge-neutrality probe is mandatory.
- Three seeds and the test identities retained by the neutral conversion (at
  most the current 19; about 15 under naive fallback exclusion) give limited
  uncertainty about model training and corpus transfer.
- GAVD source videos may include repeated people, and its labels are dataset
  annotations rather than diagnoses produced by this project.
- Neither dataset licenses force, balance, prognosis, treatment, or clinical
  deployment claims.

The most worthwhile outcome is not necessarily a new architecture. It is a
clear empirical boundary between errors that need only transparent correction
and ambiguities that require uncertainty-aware representation and decision
interfaces.
