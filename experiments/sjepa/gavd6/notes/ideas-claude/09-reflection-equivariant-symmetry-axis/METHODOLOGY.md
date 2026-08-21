# Idea 9 methodology: a layer-tested reflection-equivariant Skeleton-JEPA

This is the implementation and analysis contract for Idea 9. It turns [README.md](./README.md) into a falsifiable experiment and an executable notebook sequence. It is deliberately stricter than putting a shared left/right head on an ordinary encoder: that head can make a *final scalar* odd, but cannot make the preceding representation equivariant. The distinction is the scientific point.

- [`nb_09a_equivariant_encoder_contract.ipynb`](../../../nb_09a_equivariant_encoder_contract.ipynb) is the executable architecture contract, smoke-mode training path, and core matched comparison.
- [`nb_09b_equivariant_futures_and_reach.ipynb`](../../../nb_09b_equivariant_futures_and_reach.ipynb) is the pre-registered future simulator, decision table, and non-clinical multi-view reach scaffold.
- [`nb_09c_gavd_matched_jepa_contract.ipynb`](../../../nb_09c_gavd_matched_jepa_contract.ipynb) freezes the complete local-GAVD cohort, preprocessing, objective, health gates, and matching ledger.
- [`nb_09d_gavd_matched_jepa_training.ipynb`](../../../nb_09d_gavd_matched_jepa_training.ipynb) trains fresh paired-seed standard, paired-unconstrained, and reflection-equivariant JEPAs.
- [`nb_09e_gavd_matched_jepa_audit.ipynb`](../../../nb_09e_gavd_matched_jepa_audit.ipynb) reloads every checkpoint and independently audits collapse and the numerical geometry contract.

All GAVD results are source-video-level and descriptive unless every encoder is trained inside an outer source-disjoint fold. Folder names are dataset annotations, not diagnoses. Neither AMASS nor MoVi contains a clinical outcome and neither can establish clinical benefit.

## 1. Question and claim ladder

The primary question is not merely whether a score changes sign under mirroring. Any scalar model can be made exactly odd after the fact:

$$
q_{\mathrm{odd}}(x) = \frac{q(x) - q(Mx)}{2}.
$$

Instead, the experiment asks whether retaining left/right reflection structure **through the encoder** improves a signed gait target relative to this cheap output repair, and relative to an equally large paired encoder without the required weight ties.

| Level | Claim | Necessary evidence | Not established by |
|---|---|---|---|
| C0 | Implementation is correct | mirror tests and layerwise commutation | a good downstream score |
| C1 | Representation is alive | variance, effective rank, pair diversity, non-zero odd energy | exact oddness alone |
| C2 | Encoder is genuinely equivariant | every claimed layer satisfies the swap test | an antisymmetric readout alone |
| C3 | Equivariance beats output repair | paired held-out comparison against `odd_output` | `standard_one_view` alone |
| C4 | Gain is not generic two-view fusion | matched comparison against `paired_unconstrained` | extra parameters or compute |
| C5 | Signed output is stable to real cameras | actor-held-out MoVi calibrated-view test | synthetic yaw rotation |
| C6 | There is clinical value | participant-held-out clinical-target performance and replication | AMASS, MoVi, or GAVD labels |

Every notebook reports the highest level it earned and withholds all higher ones. A zero odd channel passes an oddness test exactly; it has not passed C1, much less C3.

## 2. Mathematical contract

### 2.1 Anatomical mirror

Let $M$ negate the anatomical horizontal coordinate and swap every left/right landmark in the complete BlazePose pairing. It is an involution: $M(Mx)=x$. The lower-body-plus-shoulder pairs used for the signed target are `(11,12)`, `(23,24)`, `(25,26)`, `(27,28)`, `(29,30)`, and `(31,32)`; the input mirror must also use face, arm, hand, and leg pairs so it remains a valid skeleton.

Apply the mirror to raw coordinates, then use the identical interpolation, pelvis-centering, body-scale normalization, and temporal-resize pipeline as the original input. Applying it after a non-equivariant preprocessing step invalidates the test.

### 2.2 Paired lift

The equivariant model never processes $x$ in isolation. Its first internal state is

$$
H^0(x) = [\phi(x), \phi(Mx)].
$$

For a mirrored input, the slots exchange:

$$
H^0(Mx)=[\phi(Mx),\phi(x)]=S H^0(x),
$$

where $S[a,b]=[b,a]$. This is valid even when patch embedding $\phi$ is otherwise ordinary: the pair carries the regular two-element representation.

### 2.3 Every layer, including the EMA teacher

Every online-encoder layer, target/EMA-encoder layer, predictor layer, normalization path, and cross-branch interaction must obey

$$
F_\ell(SH) = S F_\ell(H).
$$

`nb_09a` measures maximum absolute and relative residuals per layer on random inputs, smoke/real windows, masks, and both float32 and the training precision. Branch-specific normalisation statistics, untied positional tables, independently sampled dropout masks, or an asymmetrically updated EMA teacher can violate the condition without producing a runtime error.

The model must have symmetric cross-branch interaction, not two disconnected copies. A safe construction is:

$$
F([a,b])=[g(a,b),g(b,a)],
$$

where $g$ may cross-attend from its first argument to its second. The shared reversed call proves commutation. `paired_unconstrained` has the same paired inputs, cross-branch communication, depth, masks, and parameter budget, but not this tie; it is expected to fail the commutation test.

### 2.4 Readout and its limit

At the final pair,

$$
h_+=\frac{h_\mathrm{orig}+h_\mathrm{mirr}}{2},\qquad
h_-=\frac{h_\mathrm{orig}-h_\mathrm{mirr}}{2}.
$$

A zero-bias head on $h_-$ is exactly odd **only after the encoder pair has passed the swap contract**. It is a useful health diagnostic, but cannot make an ordinary backbone equivariant; in general

$$
f(E(Mx)_L)-f(E(Mx)_R)\ne-[f(E(x)_L)-f(E(x)_R)].
$$

For the co-primary comparison, both paired models use the same final protection. Let $H(x)$ denote
one complete paired-state forward pass; then evaluate it once more from the mirrored input:

$$
q_{\mathrm{pair\mbox{-}odd}}(x)=\frac{d(H(x))-d(H(Mx))}{2}.
$$

This construction is exactly odd even when the paired-unconstrained interior fails to commute: replacing
$x$ by $Mx$ merely reverses the two terms. Both models make the same two paired-state passes. A
difference can therefore be attributed to internal swap-preserving organization, not an easier output
head. The direct $h_-$ head remains a labelled ablation, never the privileged primary readout.

## 3. Models and comparisons

| ID | Training input / architecture | Final score | Question answered |
|---|---|---|---|
| `standard_one_view` | one ordinary S-JEPA branch | ordinary $d(x)$ | Practical context only |
| `odd_output` | same ordinary checkpoint evaluated on $x$ and $Mx$ | $(d(x)-d(Mx))/2$ | Is output repair sufficient? |
| `two_view_free` | ordinary encoder and free two-view fusion | ordinary joint readout | Does a second view alone help? |
| `paired_unconstrained` | paired branches with cross-branch fusion but no swap ties | shared odd wrapper $\tilde d$ | Is generic paired fusion enough? |
| `equivariant_encoder` | paired lift, swap-commuting layers and EMA teacher | shared odd wrapper $\tilde d$ | Does encoder-wide organization help? |

The co-primary effects are participant-orbit-averaged error differences:

$$
\Delta_\mathrm{repair}=\operatorname{MAE}(\mathrm{equivariant})-\operatorname{MAE}(\mathrm{odd\_output}),
$$

$$
\Delta_\mathrm{pair}=\operatorname{MAE}(\mathrm{equivariant})-\operatorname{MAE}(\mathrm{paired\_unconstrained}).
$$

Negative values favour the equivariant encoder. Compare each seed only with the same seed; never compare best-run to best-run.

## 4. Dataset roles and split rules

| Dataset | Role in Idea 9 | Split / exclusion rule | Claim boundary |
|---|---|---|---|
| Synthetic smoke cohort | Exercises operators, model, trainer, artifacts, figures | deterministic synthetic subjects | C0/C1 software check only |
| GAVD | Local integration and descriptive feasibility | source video is unit; provenance held constant; full outer-fold retrain needed for held-out estimate | no clinical accuracy/generalization claim |
| AMASS locomotion | Generic label-free JEPA pretraining and scale test | split people before windows; exclude known downstream people/source datasets; record source provenance | broad non-clinical motion only |
| Stroke force cohort, when governed | Primary clinical target evaluation after pretraining | participant-disjoint nested folds; force never enters generic pretraining | bounded signed-force association, not diagnosis |
| MoVi | Calibrated real-camera view test | actor-held-out; all views/motions of one actor together | real-view stability plus mirror behaviour, not clinical value |
| Parkinson's cohort, when governed | One-time replication | untouched participant manifest; open after architecture freezes | condition-specific replication only |

AMASS and MoVi are not interchangeable. AMASS supplies movement diversity for learning; it has no clinical target. MoVi tests whether a body-frame score stays stable through real projection and visibility changes; it is held out from generic pretraining by default. Rigid yaw augmentation tests coordinate-frame sensitivity, not camera invariance.

The current GAVD curriculum contains condition labels in later stages. For the causal architecture experiment the default path is label-free JEPA pretraining and every model sees the same source windows and masks. A compatibility ablation retaining the legacy group loss must use it identically for every arm and be labelled transductive.

## 5. Training and evaluation protocol

Record, for every model/seed, sources, original/reflected windows, visible/masked tokens, optimizer updates, learning-rate schedule, masks, hyperparameter trials, parameters, FLOPs, wall clock, precision, teacher momentum, and code/data hashes.

A paired model necessarily evaluates two branches, so no one comparison matches examples, updates, parameters, and compute perfectly. Report both:

1. **Exposure-matched:** same sources, windows, masks, updates, and tuning opportunities.
2. **Compute-matched:** same FLOPs/GPU-hours, accepting different update counts.

Do not describe either as matching the other quantity. The paired-unconstrained control receives the same number of paired-state forward passes as the equivariant model.

The student predicts masked target-encoder features; the EMA teacher has the identical paired architecture. Before a target is opened, fix training-only health gates for per-dimension variance, covariance effective rank, mean pairwise cosine, masked-JEPA prediction quality, odd energy by source, even/odd energy ratio, response to controlled unilateral attenuation, and source/provenance decodability from side-agnostic nuisance features. A near-zero $h_-$ fails even if commutation is machine-precise.

The GAVD arm reuses Idea 5's signed left-minus-right target and source grouping only as a representation diagnostic. An honest clinical arm retrains the entire pretraining recipe inside each outer participant fold, chooses all settings inside inner folds, and produces one aggregated prediction per test participant. Report signed-target MAE, low-label prefixes, calibration, raw-kinematic baseline, corruption robustness, output oddness, layerwise commutation, and compute; bootstrap at the participant level.

## 6. Possible futures

Numbers displayed by `nb_09b` are simulated shapes, never observations.

| Future | Observable pattern | Licensed conclusion | Withheld conclusion |
|---|---|---|---|
| F1 interior advantage | C0/C1 pass; both deltas materially below zero | Encoder-wide mirror organization helps beyond output repair and generic fusion | clinical deployment/generalization |
| F2 geometry without utility | commutation and health pass; tie with `odd_output` | cheap output repair is sufficient for this task | equivariance is generally better |
| F3 generic pair advantage | beats `odd_output`, ties `paired_unconstrained` | linked two-view fusion helped | swap ties caused the gain |
| F4 exact but collapsed | commutation passes, odd-energy/rank/positive-control fail | implementation may be geometric but uninformative | any representation/target claim |
| F5 invalid experiment | layer failure, leakage, mismatched exposure, or nuisance-control success | repair the pipeline | every positive claim |
| F6 real-camera failure | synthetic mirror passes, MoVi view stability fails | reflection rule does not imply camera robustness | real-world view invariance |

Only F1 is an architecture win. F2 is a valuable negative result, F3 prevents a generic-fusion overclaim, and F4/F5 are failures of evidence rather than weak positives.

## 7. Notebook execution contract

Both notebooks default to `GAVD_MODE=smoke`. Smoke mode uses a planted sign-flipping toy signal solely to test plumbing; every artifact is labelled `illustrative`. `nb_09a` now uses exactly 05a's startup requirements: real mode activates only when the selected curriculum-final checkpoint **and** `poses/` cache exist under the usual GAVD artifact directory; otherwise it falls back to smoke mode. `nb_09b`, like 05b, requires no checkpoint or pose cache at all. An optional `IDEA9_RUN_ID`, output directory, and comparison manifest can make a real architecture run traceable, but their absence does not stop the contract or futures notebooks. Neither notebook downloads AMASS, MoVi, or clinical data, and neither labels a full-corpus GAVD encoder as source-held-out generalization.

`nb_09a` mirrors `nb_05a`: environment and frozen contract; mirror/swap operators; paired modules and unconstrained control; layer/teacher tests and health gates; smoke or explicit-real run; core comparison; figures; JSON bundle. It uses Jupyter's standard `python3` kernel metadata, and embeds its saved architecture-contract PNG in the notebook output so the visual audit remains available after headless execution.

`nb_09b` mirrors `nb_05b`: frozen practical margins and scoring; the six futures; decision table and expected-shape panels; AMASS split/overlap scaffold; MoVi actor-held-out scaffold; JSON bundle.

`nb_09c` through `nb_09e` default to a deterministic CPU smoke run. A real run requires an explicit,
versioned `GAIT_PARITY_RUN_ID`; the device remains CPU unless `GAIT_PARITY_DEVICE=cuda` is set. The CPU
profile uses every canonical GAVD sequence at reduced width and temporal density. The GPU profile uses
denser windows, the 96-wide four-layer encoder, more epochs and paired seeds, and CUDA autocast. Both
profiles use the same centered/sharpened JEPA target and the same even/odd VICReg anti-collapse terms.

Run the sequence twice when resources permit: once with `GAIT_PARITY_MATCHING=exposure`, which fixes
updates and orbit exposure, and once with `GAIT_PARITY_MATCHING=compute`, which fixes a predeclared
analytic token-parameter budget while allowing update counts to differ. The proxy is recorded as a
proxy rather than mislabeled exact FLOPs; measured wall time, parameters, actual exposures, precision,
and device are recorded separately. The audit covers online and EMA encoders, training and evaluation
modes, masked and unmasked tokens, checkpoint reload, float32, and CUDA autocast precision when CUDA is
selected.

## 8. Threats, stop conditions, and references

A scalar output can be odd while all hidden layers violate the reflection rule. GAVD's small and provenance-confounded source structure makes full-corpus probes transductive. AMASS aggregation creates subject/source-overlap risk. Independent dropout, branch-specific statistics, asymmetric EMA updates, and unmatched pair masks break exact commutation. MoVi views are correlated observations, so the actor is the split and uncertainty unit. Skeletons do not recover force, EMG, spasticity, transverse-plane dynamics, or a diagnosis.

Stop and repair if a layer fails its numerical contract, $h_-$ collapses, the matching tolerance is exceeded, AMASS overlap cannot be ruled out, or a side-agnostic nuisance channel recovers a signed target.

References: Abdelfattah and Alahi, *S-JEPA* (ECCV 2024); Assran et al., *I-JEPA* (CVPR 2023); Bardes et al., *V-JEPA* (2024); Cohen and Welling, group-equivariant networks (ICML 2016); Mahmood et al., AMASS (ICCV 2019); Ghorbani et al., MoVi (*PLOS ONE*, 2021); Ranjan et al., GAVD (*IEEE Access*, 2025); Kapoor and Narayanan (2022) and Varoquaux (2018) on leakage and small-sample validation.
