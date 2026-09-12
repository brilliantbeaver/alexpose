# Future-innovation distillation after the repaired development STOP

Research assessment, 11 September 2026. This is a proposed research direction,
not a frozen protocol or a record of new fits. It preserves the completed
direct-v2 and direct-v3 results. No new cohort, teacher encoding, student training,
or HAIC job was run for this assessment. Literature links were checked on the
date above; the novelty assessment is provisional, not an exhaustive priority
claim.

**The recommended next question is which future teacher information a skeleton
student can learn, and whether selecting that information improves its motion
representation.** Keep the RGB-conditioned comparison as a distinct scientific
measurement. A negative result for that comparison does not answer the entire
distillation question.

The [repaired result](direct-v3-repair-validation.md) gives a reliable starting
point: on 50 clips from 43 sources, the shared RGB reference scored 0.334215 R²,
the real-skeleton arm scored 0.333973, and the matched increment was
−0.00024242 R². Its conditional 95% source-bootstrap interval was
[−0.00147402, +0.00091980], with 36.35% positive draws. The model selected exact
baseline fallback for 14 of 20 fold-arm fits. The original +0.05 effect rule and
other direct-v3 criteria remain failed. Nothing proposed here converts that
STOP into an ADVANCE.

Those intervals describe the saved fits. They do not measure uncertainty from
repeating model selection, changing the representation, or adapting the study
after inspecting these sources. They cannot be used as a general upper bound on
the usefulness of skeletons or as a sample-size calculation for a new study.

**Separate complementarity, accessibility and temporal value.** The current RGB
reference includes representations of the entire observed prefix, including
global pooling, last-prefix person pooling and background pooling. It already
has temporal video information. Experiment 0 asks whether explicitly supplied
skeletons help the tested predictors beyond those representations and recording
conditions. A skeleton-only student instead needs teacher information that can
be predicted from its own input. We should also establish that history helps
beyond current posture and observation support.

| Quantity | Comparison | What a positive result establishes |
|---|---|---|
| RGB complementarity | RGB/nuisance plus real skeleton versus its matched control | Supplying skeleton coordinates helps this RGB predictor |
| Skeleton accessibility | Skeleton input predicts teacher targets on held-out sources | Some teacher information is learnable from the student's input |
| Temporal value | Ordered skeleton history versus current posture and support controls | The tested history representation adds predictive value |
| Distillation benefit | Student trained with the proposed targets versus matched training baselines | The training method improves an independently evaluated representation |

These quantities are neither interchangeable nor an information-theoretic
decomposition of R². In particular, an RGB-conditioned gain is neither necessary
nor sufficient for skeleton-only predictability. Two elementary population
examples make this precise. They are mathematical examples, not GAVD results:

* Shared signal: let RGB summary X, skeleton history S and target Y all equal a
  zero-mean variable U. RGB predicts Y perfectly, so adding S gives zero gain.
  Nevertheless, S predicts Y perfectly too. A distillation gate requiring a
  positive RGB increment would reject information the student can access.
* Jointly required signal: let X and S be independent, equally likely −1/+1
  variables, and let Y = X S. Together they predict Y perfectly. From S alone,
  the best squared-error prediction is zero. A conditional correction that
  requires both modalities cannot be copied exactly by a skeleton-only student.
  This example assumes a predictor capable of multiplication; it is not a
  claim that the repaired linear model learns this interaction.

Skeletons here are extracted from RGB. With a fixed extractor, they are a
function of the observed pixels. Supplying that function cannot add information
to an ideal predictor already given all those pixels, but it can improve a
finite predictor's representation and learning efficiency. The frozen pooled
RGB features are a lossy summary, so this observation does not force the actual
direct-v3 increment to be zero.

**Define innovation relative to the student's information.** Let H contain the
available skeleton history, confidence, validity and permitted deployment
metadata. Let C be a declared function of H containing current posture and the
support/metadata controls, with earlier coordinate motion removed. Let Y_h be
the teacher representation of a precisely bounded future interval. In the
population, under squared loss, define

\[
u_h(H) = \mathbb{E}[Y_h\mid H] - \mathbb{E}[Y_h\mid C].
\]

An expectation here is the ideal average prediction given the specified inputs.
The difference is the part of that prediction made possible by history beyond
C. Since C is available in H, this target is a function of the student's input.
For finite second moments and a common fixed target metric,

\[
\mathbb{E}[u_h(H)\mid C]=0,
\qquad
\mathbb{E}\|Y_h-\mathbb{E}[Y_h\mid C]\|^2
-\mathbb{E}\|Y_h-\mathbb{E}[Y_h\mid H]\|^2
=\mathbb{E}\|u_h(H)\|^2.
\]

The second identity says that the ideal reduction in error equals the size of
the history-dependent prediction. It follows from conditional expectation and
orthogonality of prediction errors. It is not a new theorem, an estimator, or a
finite-sample guarantee. Learned, regularized predictors need not satisfy these
identities exactly, and they can still lose on unseen sources.

This formulation also makes an important limit explicit: ordinary squared-error
distillation already approaches the conditional mean with sufficient data and
capacity. Selecting a predictable subspace must therefore earn its place by
improving finite-data learning, robustness or efficiency. The equation alone is
not a novel distillation algorithm.

**Make the first follow-up inexpensive and decisive.** Use the inspected cache
for a separately versioned development panel. Reuse safe input preprocessing,
source splits, target units, exact fallback, nested selection and verification.
Before new fitting, specify a small, common model family and search budget for:

| Input | Purpose |
|---|---|
| Support and permitted metadata only | Establish the observation-quality reference |
| Current pose plus support and permitted metadata | Establish the student's current-state reference |
| Ordered skeleton history plus the same controls | Measure student-accessible history prediction |
| Time-shuffled history, mismatched history, and validity-only controls | Test timing, pairing and support explanations |
| Existing RGB/nuisance reference and RGB plus skeleton | Retain the original complementarity question |

Use the same cached Y for this first panel. It isolates the change in question
from changes to the teacher target. Current pose must have an explicit
missing-joint rule; normalization based on the prefix must be disclosed rather
than described as purely instantaneous sensing. A matched no-history model
should receive the same support channels and modeling opportunities. Keep the
original controls for comparison; a support-preserving order diagnostic would
be an additional, separately defined control, not a replacement of its result.

Start with the repaired linear family. One prespecified low-rank alternative can
test whether restricting the number of learned directions helps, but a broad
architecture search on 43 sources would make interpretation harder. If a frozen
S-JEPA encoder is added, include its matched initialization and raw kinematic
features. Its joint schema and pretraining-source provenance need checking.
The repository's [Core11 probe](../frozen-core11-probe-results.md) does not
currently establish a held-source advantage of the local trained representation.

Plot source-held performance and the temporal increment against training-source
count, with refitting and inner selection at each size. Repeated fixed source
subsamples measure development sensitivity; they are not extra independent
sources or optimization seeds. Include realistic low-amplitude planted signals
at the actual feature dimensions and sample sizes. The earlier strong synthetic
positive controls established software sensitivity, not power for subtle gait
motion. A curve on this small cohort can motivate further data collection but
cannot reliably extrapolate the required sample size.

Interpret the panel prospectively. Good skeleton-only prediction with no gain
over current pose suggests static or support-related transfer rather than
temporal innovation. A positive history increment with little RGB increment is
consistent with useful shared motion information. Weak prediction for every
skeleton input leaves data quantity, pose quality, model class and target choice
unresolved; it does not by itself identify which one failed.

**Audit the target before scaling the student.** The current target pools
person-region tokens at frames 38–39 after encoding all 64 frames. It may
describe past observations and observations after frame 39. This is compatible
with the recorded contextual-target protocol, but it does not establish that
the target isolates future motion. In the saved manifest, the nominal eight-frame
horizon ranges from 0.267 to 0.800 seconds because frame rates differ. A new
physical-motion study should specify context duration, forecast delay and target
interval in seconds, together with a tested frame-sampling implementation.

Three target mechanisms deserve focused tests rather than assumptions:

| Hypothesis | Discriminating test | Interpretation boundary |
|---|---|---|
| Person pooling obscures local limb dynamics | Compare a fixed small body-region token layout with pooled tokens | A positive difference identifies a representation effect, not proof of the original target's failure |
| The contextual target contains much static or overlapping information | Compare it with a separately encoded, explicitly bounded future block | Target units and meaning change; report a new experiment |
| Skeleton motion is predictable but teacher features are poorly aligned with it | Compare future-kinematic prediction and teacher-to-kinematic readouts on held-out sources | GAVD automatic future poses are noisy pseudo-labels, so use independent annotations or controlled data for validation |

V-JEPA 2.1 was explicitly designed to improve dense spatial and temporal
features. That motivates testing its local tokens; it does not demonstrate that
our pooled GAVD vectors contain a recoverable gait signal.
[V-JEPA 2.1 paper](https://arxiv.org/abs/2603.14482).

The existing per-window files contain baseline[2382], skeleton[32,33,4],
person[256], background[256] and matching[11]. They do not retain dense token
maps, future skeletons or alternative horizon targets. New local-token targets,
future-block encoding and future-pose checks therefore need new extraction and
corresponding audit evidence. They cannot be reconstructed from the 256 pooled
target values. Keep the parent cache immutable and create a new binding.

For a mechanism test, render matched motion sequences under independently varied
appearance, background and camera. Compare different histories that pass through
similar current poses but have different velocities and futures, as well as the
same motion under different appearance. This makes the desired temporal signal
observable and separates camera changes from body dynamics. Split underlying
motion sequences and subjects before creating rendering variants. AMASS provides
body-motion sequences suitable for such a controlled rendering study, subject
to access and use conditions; it is not itself a ready paired RGB benchmark.
[AMASS paper](https://openaccess.thecvf.com/content_ICCV_2019/html/Mahmood_AMASS_Archive_of_Motion_Capture_As_Surface_Shapes_ICCV_2019_paper.html).
Synthetic controls establish behavior in the generator, not causal validity on
uncontrolled GAVD videos.

**A particularly informative challenge is similar current pose with different
subsequent motion.** Two histories can pass through similar postures with
different velocities, such as a leg moving forward versus returning. Select
pairs using current-pose, view, tracking-support and prefix-motion information
only, then measure whether their observed futures differ in the predicted way.
The current-pose baseline and ordered-history model face the same candidate
futures. Stronger performance by history tests an ambiguity that instantaneous
posture cannot resolve. Controlled rendering supplies a precise version of this
test; naturally matched GAVD examples supply a more realistic, imperfect version.

One concrete target candidate is a within-recording future contrast,
Y_h(i) - Y_h(j), for the same tracked person under approximately matched current
pose, camera and support. Train differences of student predictions to match
these teacher differences. Shared appearance may cancel, making temporal change
more visible than it is in whole-vector regression. This cancellation is a
hypothesis: neural features need not separate appearance and motion additively,
and camera motion can remain. Include a simple pose/velocity forecast baseline,
static-pose and shuffled-history controls, and independent kinematic readouts.
Pair selection must not inspect the future targets or final test performance.
Relative targets also leave an arbitrary constant offset, so retain an anchored
or ordinary skeleton representation objective rather than claiming that pair
loss alone identifies an absolute state representation.

This experiment needs more temporal windows per source and probably new teacher
encoding; the current cache has at most two windows per source. Multiple windows
make within-recording comparisons possible, while independent source count still
determines the evidence for generalization. Split sources before forming pairs.
Real matched examples remain observational comparisons, not causal interventions.

**The proposed method should select temporal targets and permit no transfer.**
After the panel identifies student-accessible temporal signal, estimate a small
teacher subspace using training sources only. A subspace is a set of linear
combinations of teacher features; reduced-rank regression or partial least
squares is a transparent starting point. Candidate ranks such as 0, 2, 4, 8 and
16 are design suggestions, not frozen choices. Rank zero is an exact no-transfer
option. Train-derived normalization and fixed-norm or orthonormal directions
prevent a target map from making its loss small merely by shrinking to zero.

Select directions by source-held temporal prediction beyond C, then check their
value against teacher reliability and independent motion measurements. High
repeatability alone can select static appearance, and high prediction alone can
select tracking artifacts. Do not rank directions using the final test data, or
optimize an easier projected metric and call it an improvement of the original
256-feature gate. Report raw-target performance alongside the selected-code
metric and use a common downstream evaluation for all methods.

A possible student objective combines ordinary S-JEPA training with prediction
of selected future residuals:

\[
\mathcal L = \mathcal L_{\mathrm{S\text{-}JEPA}}
 + \beta\sum_h
 \|p_h(f(H))-\operatorname{stopgrad}\{P_h(Y_h-\hat m_{C,h}^{(-k)}(C))\}\|^2.
\]

Here f is the skeleton encoder, p is a small prediction head, P is a selected
and frozen target projection, and m is a current-state predictor fitted without
the source used to form that training target. The superscript denotes this
cross-fitting. All target construction, including projection selection, remains
inside the outer training partition; inner validation must be excluded from
every corresponding learned transform. Subfit predictions must be converted
into common raw teacher units before residual targets are combined. Include
beta=0 and preserve ordinary skeleton learning when transfer is unsupported.

This is a candidate objective requiring its own implementation and calibration.
Cross-fitting does not automatically repair model mismatch or regularization.
Retain a direct, jointly regularized reference and a matched current-pose-only
student to detect improvements caused by refitting C. There should be no second
unrestricted RGB correction route. Compare residual-target construction with
direct future-target learning; a residual loss is useful only if it actually
improves the intended behavior.

An auxiliary target computed entirely by a small regression of H can simply
teach the student to reproduce that regression. This is a concrete risk, not a
theoretical contribution. Comparisons must show added value from future video
supervision and from the proposed selection, rather than only compression of
hand-engineered skeleton features.

**Establish novelty against the closest work.** The contribution cannot rest on
combining the names V-JEPA and S-JEPA, or on observing that some teacher knowledge
is inaccessible to a student.

| Existing result | Consequence for the proposed paper |
|---|---|
| [Thoker and Gall, ICIP 2019](https://arxiv.org/abs/1910.04641) transfer RGB action knowledge to pose students using paired unlabelled examples | RGB-to-skeleton distillation itself is established |
| [Modality Focusing Hypothesis, ICLR 2023](https://zihuixue.github.io/MFH/index.html) links transfer to task-relevant shared teacher information | Student accessibility must be operationalized beyond restating shared-information intuition |
| [C2KD, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Huo_C2KD_Bridging_the_Modality_Gap_for_Cross-Modal_Knowledge_Distillation_CVPR_2024_paper.html) addresses modality imbalance and misaligned teacher guidance with customized distillation | Selective or student-adapted distillation alone is insufficient novelty |
| [C2VL](https://arxiv.org/abs/2405.20606) transfers vision-language knowledge into skeleton representations | Include a relevant cross-modal skeleton baseline, adapted with its supervision differences made explicit |
| [S-JEPA, ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/4755_ECCV_2024_paper.php) predicts masked skeleton representations | Establish what future video supervision adds over skeleton self-supervision |
| [Yang et al., ICML 2025](https://proceedings.mlr.press/v267/yang25aj.html) quantify modality interactions and apply them to targeted distillation | Separating redundancy and synergy is already an active research area; do not label R² differences as a new information decomposition |

The more defensible contribution is a tested method for choosing future teacher
targets using the student's temporal prediction ability, with an explicit
no-transfer choice and evidence that the choice improves downstream learning.
Its novelty would come from the temporal formulation, a reliable selection
procedure, and empirical transfer under changes of recording conditions. The
literature search does not establish that this exact combination is unclaimed;
a full manuscript-stage review remains necessary.

**Build the paper around a falsifiable comparison.** Train all students with the
same backbone, skeleton data, training budget and readout-selection opportunities.
Compare ordinary skeleton self-supervision, direct full-feature distillation,
unselected future-residual distillation, unconditional low-rank distillation,
and the proposed temporal selection. Include raw kinematics, a simple future-pose
prediction objective, and matched random/initial encoders where relevant.
Current-time teacher targets distinguish generic visual transfer from future
prediction. Shuffled or mismatched teacher pairing tests whether correct
alignment supplies the benefit.

The primary result should be a prespecified motion-sensitive downstream task
under source or subject separation, supported by future retrieval and prediction
measurements. Retrieval should use appearance/view/current-pose-matched
distractors; ranking different cameras or clothing is an easy shortcut. A
within-source temporal retrieval diagnostic can hold appearance more nearly
fixed, but multiple candidates from one source still count as one statistical
cluster. Gait events need suitable annotations; the GAVD repository says its
gait-event annotations are limited. Clinical label prediction is a secondary
research outcome, not clinical validation.
[GAVD data documentation](https://github.com/Rahmyyy/GAVD).

Use GAVD for in-the-wild transfer and a dataset with explicit participants for a
second test. NTU RGB+D offers paired RGB and skeleton data and participant/view
evaluation conventions; it is an action benchmark, so success there does not
establish clinical gait utility. Its skeleton format differs from GAVD's and
requires a declared adapter.
[NTU dataset documentation](https://rose1.ntu.edu.sg/dataset/actionRecognition/).
Measure pose-extraction cost as well as student inference cost when making a
deployment-efficiency claim.

**Allocate new sources after choosing the scientific question.** Keep the
inspected 43 sources as development data. Inventory the remaining usable,
independent recordings and separate a training expansion from an untouched
confirmation allocation before examining model scores. Fifty new independent
sources can supply a replication, but replacing the original cohort with them
leaves roughly forty training sources in each outer fold. It does not test a
substantially larger training regime.

Choose confirmation size using a prespecified practically meaningful effect on
the new primary task and source-level variability including refitting. Learning
curves and power simulations inform that choice; the saved-model bootstrap alone
does not. When estimating training sensitivity, all duplicate instances of a
source must remain grouped. If the available source count cannot support the
desired precision, report a pilot rather than promise a powered confirmation.

The original direct-v3 thresholds continue to govern that original question.
Any new target or student-only comparison needs its own frozen endpoint,
minimum effect, multiplicity treatment, controls and stability rules. Rank,
horizon and target selection belong inside development/inner training. Report
all prespecified horizons or one prespecified aggregate, rather than advancing
because any tested horizon happened to win. Keep fitted-model bootstrap
uncertainty separate from repeated-fit variation and stochastic training seeds.

| Stage | Evidence needed before the next stage | Reason to stop or redirect |
|---|---|---|
| Cached student-accessibility panel | Reproducible history contribution beyond current pose/support, or a specific target/data hypothesis supported by diagnostics | No distinguishable temporal value; no identified reason to expect scaling to help |
| Bounded teacher/kinematic audit | Target responds to the intended future motion and has student-predictable structure | Predictable kinematics but inaccessible teacher target suggests another target; failure of both suggests data/pose problems |
| Limited distillation comparison | Benefit over ordinary skeleton learning, direct teacher matching and equally restricted alternatives | Target selection merely makes its own loss easier or adds no downstream value |
| Independent confirmation | Frozen method meets the new task's effect, control and uncertainty rules on new sources/participants | Retain the negative result and stop that method; do not tune on the confirmation set |

The high-value negative outcome would be a replicated account of where temporal
information is lost and which distillation gates fail to predict student
benefit. That requires controlled comparisons and broader evidence than the
current 50 clips. A single repaired software failure or a small-cohort null
result is a useful technical report, but is not yet a strong general world-model
contribution. Conversely, a positive student result would establish a useful
predictive representation; action-conditioned planning or causal world-model
claims would need additional experiments.

The immediate recommendation is to specify and run the cached accessibility
panel before spending the next teacher-encoding allocation. Keep the existing
STOP visible, and use the resulting evidence to decide whether to expand data,
change the target, or begin a bounded distillation study.
