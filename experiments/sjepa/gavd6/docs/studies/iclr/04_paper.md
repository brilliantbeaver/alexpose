# When Predictive Features Miss Motion: Measuring Temporal Accessibility in Skeleton JEPA

**Research manuscript draft, 11 September 2026.** This document distinguishes
retained real-data findings, newly implemented measurements and a prospective
distillation method. It does not report a trained student benefit or establish
readiness for an ICLR main-track submission. The companion tutorial provides
implementation and evidence links; this text develops the scientific argument.

## Abstract

Prediction of learned features need not preserve useful motion information.
We connect two gait studies and implement a measurement path toward selective
future distillation. In a retained skeleton-JEPA study with 625 clips from 93
videos, a matched movement readout falls from 0.223 R² at initialization to
0.101–0.114 after training, despite consistent hidden-feature correspondence.
In a repaired experiment with 50 clips from 43 videos, adding skeletons to an
RGB reference yields −0.00024242 R² on contextual video-teacher prediction.
RGB complementarity, however, differs from skeleton-only accessibility.
Our new source-held panel measures the predictive increment of a skeleton
history block beyond current posture and observation support. Its primary
effect is +0.001994 R², with a conditional 95% source-bootstrap interval of
[−0.006673, +0.012608]: no supported temporal lead. The block also changes
confidence features and regularization, so this estimate does not isolate motion.
Exact reflection/time-reversal constructions and a controlled readout fixture
separate geometric consistency from direction-sensitive information.
We propose selecting future teacher directions for extra student-accessible
temporal value, with an exact no-transfer option. The current contribution is
a calibrated measurement framework and empirical motivation; improved student
learning and independent confirmation remain to be demonstrated.

## 1. Introduction

A predictive representation is useful relative to the distinctions it preserves.
Action recognition may group walking examples despite changes in view or which
leg moves faster. Measuring left–right movement requires those differences;
anticipating a future position can require recent direction even when current
posture is identical. One feature-prediction task need not preserve all three.

Joint-Embedding Predictive Architectures predict target representations from
partial context. [S-JEPA](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04755.pdf)
applies this idea to hidden skeleton features, evaluated primarily through action
recognition. Avoiding reconstruction of every noisy coordinate can be useful,
but leaves open which movement quantities remain accessible.

We connect two development experiments. A laterality study finds improved
hidden-feature correspondence but worse linear access to a signed speed contrast.
Repaired future-innovation Experiment 0 instead finds no supported skeleton gain
after an RGB reference when predicting contextual video-teacher features.
Different targets, cohorts and R² denominators prevent ranking the two results.

RGB complementarity also differs from skeleton-only accessibility. Shared motion
information may be predictable from skeletons without improving a model that
already sees RGB. Information requiring both modalities may improve their joint
predictor while remaining inaccessible to a skeleton-only student. Distillation
therefore needs evidence aligned with the student's inputs.

We implement a bounded source-held panel of a history-feature block beyond
current posture and support, plus symmetry and temporal-observability calibrations.
We then specify selection of future teacher directions as a subsequent experiment.
The measured contribution is the separation and testing of these evidence levels;
useful student transfer and broader world-modeling claims remain unestablished.

## 2. Observables, representations and prediction boundaries

### 2.1 A movement measurement with known reflection behavior

Both cohorts derive from [GAVD](https://arxiv.org/abs/2407.04190), a video dataset
for gait analysis. The laterality cohort contains 625 accepted clips from 93 source videos. For
each of five bilateral landmark pairs, both sides must be observed at both ends
of at least eight recorded transitions. Coordinates are centered at the pelvis
and normalized by body width. Let \(m_{L,k}\) and \(m_{R,k}\) be median coordinate
speeds, using the original time intervals. The scalar target is

\[
y=\frac{1}{5}\sum_{k=1}^{5}
\frac{m_{L,k}-m_{R,k}}{m_{L,k}+m_{R,k}+10^{-8}}.
\]

The pairs are shoulders, knees, ankles, heels and foot tips. Hips establish the
origin and are excluded because centering equalizes their speed magnitudes.
This target describes relative coordinate speed, not phase, contact, loading or
clinical affected side. Image-normalized coordinates and inferred depth are not
metric three-dimensional measurements.

An anatomical reflection M flips the centered horizontal coordinate and swaps
left/right identities, confidence and validity. Then \(y(MH)=-y(H)\). Reversing
a trajectory and its physical sampling intervals preserves speed magnitudes,
so \(y(RH)=y(H)\). Laterality is therefore reflection-odd and time-even: it
changes sign under one operation and remains unchanged under the other.
Recovering it does not establish direction-sensitive forecasting.

### 2.2 A contextual video-teacher target

Experiment 0 predicts a projected, 256-dimensional person-region representation
from a frozen V-JEPA 2.1 teacher. Inputs cover frames 0–31; selected target tokens
correspond to frames 38–39, but were encoded using the full 64-frame clip. The
target can consequently reflect observations after its nominal frames. This
is a declared contextual-feature task, not a measurement of an isolated future
physical state. The nominal eight-frame horizon also corresponds to 0.267–0.800
seconds across the cached recordings.

The cache contains RGB/nuisance inputs, prefix skeletons, projected pooled
targets and matching metadata. It does not contain mirrored teacher responses,
dense token maps or future skeleton trajectories. New parity-resolved teacher
features or explicitly bounded future-block targets require new encoding and
corresponding provenance and time-boundary checks. [V-JEPA 2.1's dense-feature
design](https://arxiv.org/abs/2603.14482v3) motivates such targets but does not
establish their gait-specific usefulness.

### 2.3 Student-accessible temporal prediction

Let H contain skeleton history and permitted observation/timing channels. Define
C as a fixed function of H retaining current posture and observation support.
For a target \(Y_h\) at a declared horizon, write

\[
u_h(H)=\mathbb{E}[Y_h\mid H]-\mathbb{E}[Y_h\mid C].
\]

For finite second moments and a common squared-error metric, the population
risk reduction equals \(\mathbb{E}\|u_h(H)\|^2\). This follows by decomposing
\(Y_h-\mathbb{E}[Y_h\mid C]\) into the residual conditional on H and \(u_h\);
their expected cross product is zero. These are standard conditional-expectation
identities. They are not a new theorem, mutual-information estimator or guarantee
for a regularized predictor trained on a small cohort.

The operational quantity is instead a difference between held-source errors of
declared finite model families. It can be negative because of estimation error,
regularization or an unsuitable representation. A null linear increment does
not prove that the population conditional means coincide. Moreover, ordinary
squared-error distillation already estimates a conditional mean in the ideal
large-data limit. Selective targets must demonstrate an advantage in finite-data
learning, robustness or downstream utility.

![The student's inputs remain on the observed side of the prediction boundary.](figures/02_information_boundary.svg)

*Figure 1. Future observations supply training targets and scoring outcomes.
They do not enter the evaluated student's prefix. A current-state reference
and a history predictor must share the same declared support information.*

## 3. A source-held accessibility panel

We implement a new development measurement using the existing 50 clips and 43
source videos. It preserves the teacher cache, source partitions, target units,
raw skeleton controls, input-support preprocessing and scoring from direct-v3.
It changes the question by replacing the RGB reference inputs with information
available to a skeleton-only student.

The support panel uses 331 fixed inputs: per-joint confidence and validity means
in four ordered eight-frame bins, frame-31 confidence and validity, and decoded
frame rate. The posture panel adds the 66 valid x/y coordinates at frame 31,
giving 397 inputs. Invalid endpoint coordinates are imputed using applicable
training sources only. Coordinates inherit prefix-derived normalization; this
reference is a prefix-normalized endpoint rather than an independently measured
instantaneous image. Observation support deliberately remains available across
the prefix because otherwise tracking patterns could be mistaken for a coordinate
history benefit.

The added skeleton block contains 924 fixed features: ordered-bin coordinates,
valid adjacent-frame velocities, confidence and observation/transition support.
Velocities use only valid adjacent observations; imputation never creates
movement. Raw histories form four arms before the common constructor: original,
four-frame-block shuffle, different-source clip mismatch within the current
partition, and no-skeleton coordinates/confidence with original validity.
All arms receive the recipient's same reference block. Donor matching uses
declared context metadata and no target values.

The common reference does not exhaust observation-quality information. Its
confidence means include all frames, while S averages confidence over valid
observations and the no-skeleton arm zeros those values. The primary contrast
therefore concerns the entire declared coordinate/confidence feature block,
including its additional regularized routes, rather than isolated motion.

Shuffle changes observation support and can create new block-boundary transitions;
mismatch changes which body's observations are supplied. A favorable control
comparison narrows explanations but does not isolate causal dynamics. Multiple
noisy postures can also improve state estimation without requiring a learned
direction-of-time mechanism.

Each fit minimizes a source-weighted, summed objective,

\[
\sum_i w_i\|Y_i-b-X_iW_x-S_iW_s\|^2
+\lambda_x\|W_x\|_F^2+\lambda_s\|W_s\|_F^2.
\]

Here X is the safely transformed reference, S the skeleton summary and Y the
target standardized with current training-partition statistics. The squared
Frobenius norm is the sum of squared matrix entries. Source weights give every
video equal total influence and are normalized to sum to the training clip
count. The intercept is unpenalized. The solve uses float64 arithmetic. Entirely
unobserved, constant or insufficiently supported input columns have zero influence
in all partitions; target standardization and masks remain separate.

Five outer source folds measure excluded-source prediction. Three inner source
folds choose positive penalties from \(\{0.1,1,10,100,1000,10000\}\), using the
same 36 joint combinations per arm. A separately selected shared reference is
also an exact `baseline_only` candidate. Inner losses pool source-weighted error
sums consistently; ties within the frozen tolerance prefer the baseline, then
stronger penalties. Failed required candidates prevent a complete scientific
measurement. The model is deterministic and is not replicated under several
seed labels.

The named primary contrast is real-minus-no-skeleton in the posture panel. The
support panel and other arm comparisons are descriptive. A prospective development
lead requires the primary paired 95% interval above zero and positive point
differences against shuffle and mismatch. This new exploratory rule does not
revise direct-v3's original scientific thresholds or its STOP.

For teacher feature j, predictive R² compares source-weighted squared prediction
error with the error of the outer-training-mean reference. In training-standardized
units that reference is zero. Valid target dimensions are the intersection of
training-derived masks across outer folds; scores pool predictions before
averaging featurewise R². Two thousand paired bootstrap draws resample whole
sources, retaining multiplicity and all associated clips. The intervals condition
on the saved models and exclude repeated fitting, selection and adaptive redesign.

## 4. Reflection and temporal observability as calibration

Reflection and time reversal provide interpretable tests when their action on
the observable is explicit. M and R commute under the declared landmark and
timestamp transformations and each squares to identity. For any feature function
h and signs \(a,b\in\{-1,+1\}\), define

\[
P_{a,b}h(H)=\frac{h(H)+a h(MH)+b h(RH)+ab h(MRH)}{4}.
\]

The projected feature gains factor a under M and b under R. This construction
separates four transformation types, but its exactness is algebraic. Zero features
satisfy the same rules. Feature energy and rank can detect that particular collapse,
yet positive energy still does not establish useful movement information.

Our deterministic calibration checks the two involutions, their commutation,
physical-interval reversal and four observable parity types. Group errors are
zero on the constructed fixture; the largest recorded projector error is
\(5.55\times10^{-17}\), below the prespecified \(10^{-12}\) tolerance. A zero-feature
control has exact symmetry and zero predictive R² on the balanced nonzero target.
These results verify the construction rather than a learned model.

A second fixture has 48 generated sources, each contributing a closed path and
its reversal. Both paths end at the same posture and have the same coordinate
distribution and observation support, but their last signed vertical velocities
are opposite. Sources 0–35 train a fixed-penalty linear readout; sources 36–47
are held out. The target is that last observed velocity, which defines a possible
constant-velocity continuation. It is not measured future human motion.

| Synthetic readout inputs | Held-source R² |
|---|---:|
| Current posture | 0.000000 |
| Coordinate mean, standard deviation, absolute changes and support | 0.000000 |
| Observation support only | 0.000000 |
| Ordered signed velocities | 0.999999999978 |

The generator deliberately places opposite targets behind identical order-even
summaries. This supplies a precise failure case for those summaries, not evidence
that all temporal pooling is order-blind: contextual tokens may encode order
before pooling. No JEPA training, GAVD forecasting or reflected-video encoding
occurs in this calibration.

## 5. Empirical evidence and its limits

### 5.1 Hidden-feature correspondence and movement readout disagree

The retained laterality grid contains five training conditions across five source
folds and five seeds. With a common expanded readout, all trained teachers score
below their matched initialization in every seed. The expanded summary includes
means, standard deviations, absolute adjacent-block changes and support, giving
2,890 inputs compared with 960 for means alone.

| Laterality representation | Mean R² | Trained minus initial | Retained 95% source interval |
|---|---:|---:|---|
| Matched initialization | 0.222544 | — | — |
| Motion random | 0.113725 | −0.108819 | [−0.170093, −0.041219] |
| MAMP-style mask | 0.112890 | −0.109653 | [−0.165071, −0.051351] |
| Motion mixture | 0.114209 | −0.108334 | [−0.170640, −0.042307] |
| Region random | 0.109446 | −0.113098 | [−0.166192, −0.059623] |
| Connected region | 0.100777 | −0.121766 | [−0.182787, −0.055281] |

Meanwhile, correct-clip targets yield lower predictor error in all 375 trained
diagnostic checks, compared with 33 of 75 distinct initial checks. This supports
learning of correspondence, but neither identifies its mechanism nor turns
repeated checks on the same models into independent samples. Mask comparisons
against their respective random references have intervals crossing zero.

Several explanations remain open. Expanding the initial readout increases R²
by 0.151716, but simultaneously changes dimension, temporal summaries and support
information. The largest tested ridge penalty wins for 49/125 teacher and
96/125 online expanded readouts, motivating a wider shared training-only grid.
Input preparation also differs from target calculation: it fills gaps and
resizes by sequence index, whereas the target retains original timing and gaps.
The retained path comparison has 70.4% sign agreement and R² 0.218. This discrepancy
neither bounds achievable accuracy nor identifies a single processing cause.

These are readout-access findings, not proof that motion information was erased.
The current review recomputes retained seed and contrast arithmetic, but raw
laterality checkpoints and prediction rows are absent from the copied bundle.
The reported bootstrap intervals are retained results, not newly regenerated
intervals. Original reflection experiments and the later mask grid also use
different training/readout recipes and should not be combined as one factorial
intervention.

### 5.2 Repaired RGB complementarity remains unsupported

The original residual implementation suffered from unsupported input directions,
extreme scaling of newly observed missingness, and forced selection among
corrections that all lost to the inner reference. Direct-v3 replaces that construction
with the joint model, safe preprocessing and exact fallback. Its calibration
and independent prediction reconstruction support interpreting the repaired
negative result.

| Direct-v3 predictor | Predictive R² | Gain over shared RGB reference |
|---|---:|---:|
| RGB/nuisance reference | 0.334215168 | — |
| Real skeleton | 0.333972751 | −0.000242417 |
| Time shuffle | 0.334246668 | +0.000031500 |
| Clip mismatch | 0.335452594 | +0.001237425 |
| No skeleton, validity retained | 0.334215168 | 0.000000000 |

The matched increment is −0.000242417, with conditional 95% interval
[−0.001474019, +0.000919797] and 36.35% positive paired draws. Fourteen of twenty
selected fold-arm models use the exact baseline. All 740 pooled candidates are
valid. All selected joint models retain the reference's RGB penalty of 100,
so a changed selected RGB penalty does not explain these particular increments.

The original direct-v2 scores remain the result of the earlier implementation.
Its post-hoc weight removal and zero-initialization interventions are diagnostic,
not independent evidence of a repaired improvement. The increase in the RGB
reference between protocols is likewise not a skeleton gain.

Unlike the teacher-feature experiment, laterality's scalar R² uses the weighted
held-out mean in its denominator. We preserve both definitions and do not compare
their absolute levels. Both cohorts are development data, and source holdout
does not establish participant holdout across recordings.

### 5.3 Student-accessibility measurement

The new panel completes all required fits and retains all 256 target dimensions.
All 1,480 pooled candidates are valid; 17 of 40 selected fold-arm models use exact
baseline fallback. Read-only verification reloads and refits selected linear
models and preprocessing, reconstructs 102,400 held-out prediction rows and
16,000 bootstrap rows, and checks the saved report. Parent and source-run artifacts
remain unchanged. Inner candidate losses are checked for pooling and selection,
but are not independently refitted in this verification.

| Cached-panel predictor | Support reference | Posture reference |
|---|---:|---:|
| Shared reference | 0.008229092 | 0.005890997 |
| Real skeleton | 0.013864827 | 0.013261709 |
| Time shuffle | 0.012433535 | 0.007692185 |
| Clip mismatch | 0.008229092 | 0.005890997 |
| No skeleton, validity retained | 0.007295537 | 0.011267966 |

*Table 4. Predictive R² on the existing contextual teacher target. The posture
panel's real-minus-no-skeleton contrast was named primary before fitting.*

The primary estimate is **+0.001993744 R²**, with paired 95% interval
**[−0.006672679, +0.012608431]** and **68.35%** positive draws. The result is
`no_supported_temporal_lead` under the prospective exploratory rule. Real-minus-
baseline is +0.007370712, but the real model selects stronger reference-block
regularization in two folds, and the validity-retaining no-skeleton model gains
+0.005376968 itself. The reference difference therefore cannot be attributed
entirely to coordinate history. The matched increment is the relevant declared
comparison, and it remains uncertain.

Adversarial reconstruction identified an additional interpretation limit: 182
finite confidence summaries differ between X's raw means and S's valid-conditioned
means. Real-arm fits retain 132 confidence columns, versus zero in no-skeleton.
Thirty-one confidence columns exactly duplicate standardized X columns in every
fold. Duplicate columns with penalties \(\lambda_x\) and \(\lambda_s\) have
effective penalty \((\lambda_x^{-1}+\lambda_s^{-1})^{-1}\), so they alter
regularization as well as nominal representation. The current evidence does not
quantify their contribution to the estimate. The preserved negative conclusion
does not establish that confidence and recording quality were fully controlled.

The secondary support panel gives real-minus-no-skeleton +0.006569290, interval
[−0.000071482, +0.014709426], with 97.3% positive draws. Its real-minus-shuffle
contrast is only +0.001431291, interval [−0.003920839, +0.008541424]. It also omits
current posture from the reference. Selecting this more attractive result after
fitting would change the scientific question. Both panels show that these fixed
skeleton summaries predict little of the full contextual teacher vector under
this training regime; they leave target suitability, model family and data
quantity unresolved.

![Paired uncertainty in the cached comparison.](figures/06_cached_panel_results.svg)

*Figure 2. Differences in predictive R² with paired 95% source-bootstrap intervals, conditional on fitted models. Teal marks the prospectively designated posture-panel comparison. The secondary panel does not replace it; every displayed interval includes zero.*

## 6. A prospective selective-distillation method

Before isolating coordinate motion in a new panel, include every confidence,
validity and transition-support summary once in a common block, identically
across arms. Vary only coordinates and displacements; use that common block as
the confidence/support-only reference and control its coefficients and penalty. These
are prospective amendments requiring a new frozen experiment, not changes to
the completed comparison.

The next method experiment would select teacher directions by their source-held
temporal value. Let U contain a small number of constrained teacher directions,
learned using training sources only. Cross-fitted current-state predictions
\(\widehat m_C\) define residual targets in common raw teacher units. A student
would minimize an auxiliary objective of the form

\[
\mathcal L=\mathcal L_{\text{S-JEPA}}+
\beta\,\mathbb E\big\|g_\theta(H)-
U^\top[Y_h-\widehat m_C(C)]\big\|^2.
\]

This is a proposed experiment, not a completed student implementation or result.
Ranks and weights include an exact zero-transfer choice. Normalization,
cross-fitting, direction learning and selection must stay inside the relevant
training partitions. U must have a fixed scale, such as orthonormal columns,
to prevent trivial loss reduction by shrinking targets. The comparison should
retain both original-target measurements and a common downstream outcome.

Reflection-resolved targets are one candidate, not a requirement that all useful
motion be asymmetric. A teacher's even/odd components require actual paired
encodings of original and reflected video in a compatible feature basis. They
cannot be inferred by negating an existing pooled vector. Their motion content
must be tested because teacher reflection sensitivity can encode view or appearance.

A focused target-construction study could pair similar current postures with
different incoming velocities. Within-recording future-feature differences may
attenuate shared appearance, but this requires more same-source windows and an
empirical cancellation check. Pair construction must use prefix information only,
with sources split before pairing. Controlled motion under independently varied
appearance provides a useful mechanism test; naturally matched videos remain
observational evidence.

The decisive student comparison holds architecture, data and training opportunity
fixed across ordinary S-JEPA, direct motion prediction, complete teacher
distillation, unselected residual distillation, a dimension-matched target chosen
without temporal information, and the proposed selection. Evaluation must include
observable future motion, current-state and simple velocity baselines, and
independent sources or participants. A lower selected-code loss alone cannot
establish useful transfer.

## 7. Related work and contribution boundary

Motion-sensitive skeleton pretraining is established by
[MAMP](https://openaccess.thecvf.com/content/ICCV2023/html/Mao_Masked_Motion_Predictors_are_Strong_3D_Action_Representation_Learners_ICCV_2023_paper.html);
[SLiM](https://arxiv.org/abs/2603.10648v3) studies compact feature prediction and
semantic tube masks. [MC-JEPA](https://arxiv.org/abs/2307.12698) combines motion
and content learning. Our local results do not reproduce or invalidate these
complete systems; they motivate evaluating specific observables alongside the
learned prediction objective.

Invariant/equivariant separation also has clear precedents in
[SIE](https://proceedings.mlr.press/v202/garrido23b.html) and
[seq-JEPA](https://proceedings.neurips.cc/paper_files/paper/2025/hash/2f63d2963526bdd9ff1b8bcc2dc9905a-Abstract-Conference.html).
The four-parity construction is a standard finite-group projection, used here
as an interpretable calibration. Neither it nor an added reflection loss
constitutes the proposed novelty.

The [Modality Focusing Hypothesis](https://zihuixue.github.io/MFH/index.html)
studies when crossmodal knowledge transfer helps; [C2VL](https://arxiv.org/abs/2405.20606v2)
transfers visual-language information into skeleton representations. Future
privileged information is central to [Overlooked Poses](https://arxiv.org/abs/2208.01302)
and [Spectral-Guided Physical Dynamics Distillation](https://openreview.net/pdf?id=P6F4MxtOKp),
the latter including spectral enhancement and human-motion evaluation. The
prospective distinction is therefore narrow: selection by incremental temporal
accessibility, explicit observable checks and demonstrated avoidance of unsuitable
transfer. Conditional future representations themselves have a long history,
including [predictive-state regression](https://proceedings.neurips.cc/paper/2015/file/9a3d458322d70046f63dfd8b0153ece4-Paper.pdf).

[BioGait-VLM](https://arxiv.org/abs/2603.08564v1) combines observed video,
language and biomechanical information for gait assessment. Its temporal evidence
aggregation is a further reason to avoid a broad gait-distillation novelty claim;
our proposed future-prefix student comparison addresses a narrower question.

## 8. Discussion

Hidden-feature correspondence, anatomical symmetry, temporal accessibility and
student utility are distinct properties. The completed comparisons do not
establish the last of these. The next decisive evidence is a source-held student
comparison with bounded future targets, common motion outcomes and matched
alternatives, followed by confirmation with participant identities or reliable
geometry. The inspected 43-source cache refines that question without establishing
broad transfer. Separate-horizon prediction would still not demonstrate recursive
rollout, and these observations contain no actions or closed-loop decisions.
The immediate objective is to test when transferring future information improves
measured movement prediction, including a valid choice to transfer nothing.

## Reproducibility and evidence statement

The repository retains the original laterality and future-innovation evidence,
source inventories, calibration definitions, separately versioned panel code,
selected-model artifacts and numerical verification. Teacher-cache reuse is
read-only. The current laterality check covers retained aggregate arithmetic;
the repaired future-innovation check reconstructs predictions and paired scores.
The accompanying [tutorial](01_critique_and_research_tutorial.md) and
[panel protocol](02_cached_panel_protocol.md) document those boundaries and
execution commands. The prospective selective student experiment has no reported
real-data performance.

## Ethics and AI-use statement

Video-derived poses can reveal personal information and retain demographic or
recording biases. The coordinate-speed contrast has no demonstrated clinical
diagnostic validity, and public annotation access does not authorize video
redistribution. Source grouping does not guarantee distinct participants.
AI assistance supported evidence organization, implementation, literature
checking, figure programming and manuscript drafting. Authors remain responsible
for source verification, experimental interpretation and the submitted claims.
