# From laterality to useful future prediction

## A critical tutorial for the next S-JEPA study

**Research question:** can we teach a skeleton model future information that it
can actually predict, while preserving meaningful differences in how the body
moves?

The most promising connection between the laterality study and future-innovation
distillation is the need to measure what a representation retains. A training
objective can improve while an important movement measurement becomes harder to
recover. A video teacher can also provide a well-defined target that is largely
unhelpful for a particular skeleton predictor. The next study should use these
findings to choose its targets and comparisons, rather than add a more elaborate
network before resolving the question.

This tutorial separates three kinds of work. The laterality results and the
repaired direct-v3 result are **completed evidence with stated limits**. The
cached accessibility panel and accompanying tutorials are a **separately
versioned completed development investigation**. New teacher targets, larger student
training and independent-source confirmation are **subsequent experiments**.
An attractive diagram or successful synthetic example does not move a result
from the third category into the first.

Read this document alongside the [laterality paper](../../../../gavd5-drift/neurips-laterality/docs/fmts_revisions/paper_v8.pdf),
the [direct-v3 repair assessment](../future-innovation/direct-v3-repair-validation.md),
and the [prospective cached-panel protocol](02_cached_panel_protocol.md). The
new notebooks follow the laterality suite's sequence, starting at 19, and live
at the top level of this repository.

![Two completed findings motivate a sharper prediction question.](figures/01_evidence_bridge.svg)

*Figure 1. The two studies supply complementary evidence. Their targets,
cohorts and R² definitions differ, so the values belong to separate comparisons.
The proposed bridge asks which future information is accessible from skeleton
history and useful for observable movement.*

## Step 1 — Establish what the existing experiments actually measured

### The laterality study measures a specified movement contrast

The laterality experiment starts with 625 accepted clips from 93 source videos.
A pose detector supplies estimated landmark coordinates and visibility. For
five bilateral landmark pairs, the target compares median left and right
coordinate speeds. If the two sides have median speeds \(m_L\) and \(m_R\),
their contrast is approximately \((m_L-m_R)/(m_L+m_R)\). The full target
averages this quantity over the five pairs.

This is a useful observable: a number calculated from measured trajectories,
with an explicit rule for missing landmarks and timestamps. Its dimensionless
scale permits comparisons within the stated normalization. It is not a clinical
affected-side label, a measure of loading, or a complete description of gait.
Opposite differences at different joints can cancel. Estimated depth and image
coordinates are also not calibrated three-dimensional positions.

The S-JEPA model learns to match teacher features at hidden skeleton positions.
After training, a separate ridge regression predicts the movement contrast from
frozen encoder features. Ridge is a linear prediction rule whose coefficient
penalty limits how strongly it fits the training examples. Keeping that readout
procedure the same for the trained model and its initialization is a substantive
strength: it tests whether training improves accessible information under the
declared readout.

With the expanded feature summary, initialization scores **0.222544 R²**. The
five trained-teacher conditions score **0.100777–0.114209**. Every trained
condition falls below its matched initialization in every seed. The mean
trained-minus-initial differences range from **−0.108334 to −0.121766 R²**;
their retained source-bootstrap intervals all lie below zero. These numbers
support a negative result for that training recipe, cohort, target and readout.
They do not show that all JEPA features lose all motion information.

The expanded summary itself is an important finding and an unresolved explanation.
It increases the readout from 960 to 2,890 inputs by adding feature standard
deviations, absolute adjacent-block changes and ten observation-support fractions.
Initialization improves from **0.070827 to 0.222544 R²**, a gain of **0.151716**.
That gain combines more features with several kinds of information. Standard
deviation discards order, absolute changes discard direction, and support can
describe tracking quality. A matched-dimension ablation is needed before calling
the improvement evidence of temporal-order learning.

At the same time, correct-clip hidden targets receive lower predictor error than
different-source targets in **375 of 375 trained checks**, compared with **33
of 75 distinct initial checks**. The predictor has learned correspondence. That
correspondence could depend on posture, view, identity-related appearance in the
pose pattern, observation support, or movement. These checks reuse fitted models
and videos; 375 is not a count of independent demonstrations of generalization.

### Experiment 0 measures skeleton complementarity to a video representation

Future-innovation Experiment 0 uses 50 clips from 43 source videos. The predictor
observes frames 0–31. Its target is a 256-dimensional vector obtained by projecting
pooled person-region video-teacher features at frames 38–39. The teacher encoded
the full 64-frame clip, so those features can reflect later observations as well
as the selected target frames. The experiment predicts **contextual teacher
features**, rather than isolated future body positions.

The repaired direct-v3 comparison uses a safely scaled, jointly regularized
linear model. One coefficient block maps RGB/nuisance features to the target;
another maps ordered skeleton summaries. Separate penalties control those two
blocks. Inner source validation selects the penalties and can select the exact
RGB-only reference instead. This avoids forcing an unsupported correction into
the prediction.

The shared RGB reference scores **0.334215 R²**, the real-skeleton arm
**0.333973**, and their difference is **−0.00024242**. The no-skeleton arm selects
the exact reference in every fold, so the matched real-minus-no-skeleton
increment is the same number. Its 95% paired source-bootstrap interval is
**[−0.00147402, +0.00091980]**, with **36.35%** positive draws. Fourteen of the
twenty selected fold-arm models use exact baseline fallback. The original
effect and control criteria remain failed: this is a **complete development
STOP**.

“Complete” means the required fitting and numerical evaluation succeeded.
“Development” means these sources were inspected while the method was repaired.
“STOP” means this comparison does not justify advancing to the proposed larger
training experiment under its frozen decision rules. It is a scientifically
usable negative result, not a failed software job.

### Keep the claims and evidence levels visible

| Claim | Evidence now available | Appropriate interpretation |
|---|---|---|
| The laterality training grid improves correct-clip feature matching. | Retained notebook outputs and reconciled aggregate records. | Correspondence improves under those diagnostics. |
| The expanded laterality readout performs worse after training. | Five-seed aggregates and retained paired source intervals. | A specific observable becomes less accessible to this readout. |
| Reflection augmentation improves useful laterality prediction. | Separate comparison: ΔR² = +0.004, interval [−0.006, +0.013]. | Movement benefit remains uncertain. |
| An explicit reflection loss improves real-data forecasting. | Synthetic tutorial only. | No real-data conclusion. |
| The repaired RGB-conditioned gate finds a skeleton benefit. | Reloadable models, held-out predictions and independently checked arithmetic. | No supported benefit; retained STOP. |
| Skeleton-only history adds teacher prediction beyond the posture/support reference. | Completed cached panel: matched increment +0.001994, interval [−0.006673, +0.012608]. | The named comparison has no supported temporal lead. |
| Selected future targets improve a trained S-JEPA student. | Proposed student comparison. | No demonstrated distillation benefit yet. |

The laterality revision can recompute retained seed means and contrast arithmetic.
Its raw predictions and checkpoints were unavailable in the reviewed local
bundle, so it cannot repeat model inference or independently regenerate those
source-bootstrap intervals here. The direct-v3 bundle supports a stronger local
reconstruction check. This distinction concerns available evidence, not a reason
to silently discard either result. See the [laterality aggregate
recomputation](../../../work/artifacts/iclr-bridge-2026-09-11/laterality/aggregate_recomputation.json).

## Step 2 — Read the notebooks as an evidence chain

Notebooks combine explanation, executable code and saved outputs. Those three
parts can come from different moments in a study. A cell describing a real-data
option does not establish that the option ran; an output displaying source
counts does not establish a completed forecasting comparison.

The laterality sequence has a useful progression. Notebooks 00–06 establish the
original protocol and source-held evaluation. Notebook 09 demonstrates a
reflection-loss experiment on generated data. Notebook 10 introduces physical
time and past-only movement forecasting. Notebook 14 separates a decoder applied
to observed future features from the same decoder applied to predicted features.
Its retained example uses eight synthetic clips and four training updates.
Notebooks 17–18 contain the real masking grid and the expanded movement-readout
analysis that supports the current paper.

Notebook 14 also illustrates a useful failure: its observed-future decoder has
RMSE approximately **0.011–0.025**, while applying that decoder to predicted
future features gives **2.507–2.637**. These are results from the small synthetic
demonstration, not GAVD forecasting performance. They show why a representation
that can describe observed future motion may still yield unusable forecasts.
The proposed real comparison remained disabled. In Notebook 18, retained outputs
coexist with a source cell that has no execution count; the current notebook
file therefore does not by itself establish a fresh, successful Run All.

The future-innovation notebooks 00–04 follow the question, cohort/alignment,
teacher cache, matched predictors and final decision. Historical executed copies
preserve the old implementation. Regenerated source notebooks explain the repair;
the separately executed direct-v3 copies verify the revised run. Existing
reflection notebooks elsewhere in this repository are not interchangeable with
either study: some archived outputs predate later masking fixes.

The broader notebook portfolio supplies useful tools but no substitute for these
source-held comparisons. Some foundation notebooks learn a label-informed
representation from rows later used in descriptive evaluation. Those scores are
not independent-source clinical validation. The frozen Core11 probe likewise
does not establish a trained-feature advantage, and its overlapping source split
blocks a stronger generalization claim. Early signed-excursion tutorials also
give the raw baseline the same coordinate summary used to construct the target;
its high score is an engineered reference, not an independent discovery. Read the
[future evidence audit](../../../work/artifacts/iclr-bridge-2026-09-11/future/evidence-audit.md)
before using an older notebook's verdict label as a paper claim.

Use the new sequence to make the research questions explicit:

| Notebook | Read or run it to establish |
|---|---|
| [19 — Evidence and observability](../../../notebooks/iclr_bridge/19_evidence_and_observability.ipynb) | Which claims are measured, which are algebraic, and which still need experiments. |
| [20 — Symmetry and temporal information](../../../notebooks/iclr_bridge/20_symmetry_and_temporal_information.ipynb) | Why reflection, temporal order and prediction are distinct properties. |
| [21 — Student-accessible future features](../../../notebooks/iclr_bridge/21_student_accessible_future_features.ipynb) | How to inspect the separately fitted cached accessibility comparison. |
| [22 — Selective future distillation](../../../notebooks/iclr_bridge/22_selective_future_distillation.ipynb) | How to calibrate target selection and design the subsequent student study. |

The [notebook inventory](../../../work/artifacts/iclr-bridge-2026-09-11/future/notebook-inventory.csv)
records the inspected files. Treat synthetic teaching output, saved real-data
results and newly executed measurements as separate evidence categories. New
tutorials should read the retained panel by default; merely opening them should
not trigger another real-data refit.

## Step 3 — Separate information available to the predictor from information in the target

Let **H** denote the student's observed skeleton history, including confidence,
validity and permitted timing metadata. Let **C** be a declared reference drawn
from H: current posture together with observation support. Let **Y** denote a
future teacher target. The key question is whether H improves prediction of Y
beyond C.

This differs from adding H to RGB features. If RGB and skeletons both describe
walking phase, RGB may already predict the target well enough that adding
skeletons changes nothing. A skeleton-only student can nevertheless learn the
shared phase information. Conversely, if a target requires an interaction
between RGB and skeletons, a successful two-input correction may be impossible
to reproduce from skeletons alone.

![Inputs, targets and the evaluation boundary.](figures/02_information_boundary.svg)

*Figure 2. The student receives only the observed prefix and quantities known
when prediction is requested. Future observations can define training targets
and later evaluation outcomes. Their presence in a teacher target does not
authorize their use in the student's inputs.*

Under ideal squared-error prediction, define the temporal contribution as

\[
u(H)=\mathbb{E}[Y\mid H]-\mathbb{E}[Y\mid C].
\]

The conditional expectation is the best average prediction given the indicated
information. Because C is contained in H, the difference is a function of
student-available inputs. Under finite second moments and a fixed target metric,
the corresponding population risk reduction is

\[
\mathbb{E}\|Y-\mathbb{E}[Y\mid C]\|^2
-\mathbb{E}\|Y-\mathbb{E}[Y\mid H]\|^2
=\mathbb{E}\|u(H)\|^2.
\]

To understand the equality, write the first error as the second error plus u.
The cross term averages to zero because the remaining error has zero conditional
mean given H. This is a standard property of conditional expectation. It is not
a new theorem, a finite-sample guarantee or an information-theoretic decomposition
of the observed R² values.

A trained ridge model is an imperfect approximation. Limited training sources,
regularization and a restricted feature family can make its held-out increment
negative even when useful information exists in the population. The accessibility
panel therefore measures **prediction accessible to a finite declared model
family**. Ordinary squared-error distillation already approaches a conditional
mean in an ideal unlimited-data setting; target selection must earn its value
through better finite-data learning or downstream performance.

## Step 4 — Use reflection and time reversal to test different information

Reflection M reverses the centered horizontal coordinate and exchanges left and
right anatomical landmarks, together with confidence and validity. Applied
twice, it returns the original sequence. For the signed speed contrast,

\[
y(MH)=-y(H).
\]

A temporal reversal R instead presents the observed trajectory in the opposite
order while reversing its sampling intervals consistently. Speed magnitudes
are preserved, so this laterality measurement satisfies

\[
y(RH)=y(H).
\]

The first equation makes laterality a useful reflection-sensitive observable.
The second explains why laterality alone cannot establish direction-sensitive
motion understanding. A model might recover which side moved more without
knowing whether a particular leg is moving forward or returning.

![Reflection and reversing time change different properties.](figures/03_reflection_and_time.svg)

*Figure 3. Swapping anatomical sides changes the sign of the laterality target;
reversing the sequence preserves its speed magnitudes. A future-motion test must
add an outcome for which prior direction matters. The illustrated poses and
trajectories are schematic.*

For any encoder h, the two constructed features

\[
z^+(H)=\tfrac12[h(H)+h(MH)],\qquad
z^-(H)=\tfrac12[h(H)-h(MH)]
\]

are respectively unchanged and sign-reversed by M. These are called the even
and odd parts. The identities follow directly from M² = I. They hold for an
untrained encoder and for an encoder that returns zero for every input.
Consequently, exact reflection behavior is a useful consistency test but gives
no accuracy guarantee.

A stronger check combines transformation behavior with observable prediction.
For example, compare a post-hoc odd projection of initialization, the same
projection of a trained encoder, and a model trained with a reflection objective.
Give each the same readout opportunities and account for the cost of evaluating
both H and MH. Also examine the unprojected features. Otherwise, an imposed
algebraic rule can be mistaken for a learned property.

Teacher parity requires its own data. To construct
\(Y^-(V)=[f(V)-f(MV)]/2\), encode both the video and its mirrored version. The
current pooled cache does not contain f(MV), and mirroring only the skeleton
while reusing the original video target cannot produce it. Dense-token comparisons
also require spatial alignment. Even a valid odd video target may describe
asymmetric clothing, view or image text; it becomes a motion target only through
supporting measurements.

Specify the transformation semantics as well. Flipping an image, reflecting a
centered coordinate system and repairing accidentally exchanged joint labels
are different operations. None automatically simulates a change in a person's
impairment. If an architecture processes both original and reflected branches,
hide corresponding landmark content in both: otherwise the second branch can
reveal the very token the first branch is asked to predict. This paired-mask
check complements equivariance tests. Also avoid calling pooled features
automatically sign-blind; a non-equivariant encoder may encode side identity in
its channels before pooling.

## Step 5 — Preserve the repairs while changing the scientific question

The repaired infrastructure is valuable because it prevents several convincing
but incorrect explanations. An unsupported input column now has exactly zero
model influence. A feature that is always missing or constant in training cannot
acquire an arbitrary effect when it appears in a held-out clip. Training-only
statistics determine imputation, means, scales and supported-feature masks.

This matters at the observed sample size. The old implementation divided a newly
observed missingness feature by a training scale floor of 10⁻⁸, producing values
of 100 million. Its redundant RGB residual map also had far more coefficient
directions than the training examples could constrain. Fitting the reference's
in-sample residuals with another unrestricted map could undo its regularization.
Those were confirmed implementation failures, not evidence that skeleton motion
is intrinsically unhelpful.

The repair fits the two input blocks together and lets inner selection choose
an exact baseline. That protects against forced corrections but cannot guarantee
better performance on new sources. A candidate that wins inner validation can
still lose in the outer fold. Failed candidates remain visible, and a missing
required comparison makes the measurement incomplete.

Target scaling stays separate from input preprocessing. A training-derived
target mask identifies dimensions whose variance permits meaningful evaluation;
an input-support mask must never erase held-out target variation. Selected
models save the target means and scales needed to recover raw teacher units.
These requirements also apply to any future learned subspace or residual target.

The laterality study has additional unresolved issues that should guide its next
extension. Its target uses original timestamps and unfilled observations, while
its encoder input fills short gaps, uses whole-clip fallbacks and resizes by
sequence index to 64 positions. The retained comparison between those calculation
paths has **70.4% sign agreement** and **R² = 0.218** on the common finite cases.
This confirms disagreement between paths; it neither identifies the responsible
step nor bounds attainable model accuracy. In particular, uniform speed scaling
largely cancels in a left/right ratio, so lost duration alone is not an established
cause. Preserve physical time and audit each operation in the next forecasting
study rather than attributing the discrepancy prematurely.

Regularization also deserves a prospective check. The largest tested readout
penalty, 10,000, wins in **49/125 trained-teacher** and **96/125 final-online**
expanded-readout fits. A wider common grid, support-only references and equal-sized
summaries can test whether the deficit depends on the readout. They may fail to
recover it; the result must be retained either way. The specific 100-million
input explosion diagnosed in future innovation has not been established in the
laterality scaler.

Finally, laterality's inner folds select the readout after its encoder has learned
without labels from all outer-training sources. That is a defined readout-selection
protocol. It is not fully nested selection of the entire pretraining method.
If the next experiment uses inner validation to choose an encoder recipe or a
learned target map, those fits must also exclude the inner-validation sources.

### Do not silently combine the studies' R² values

Both studies balance source videos, but their reference errors differ:

| Study | Target and denominator | Interpretation |
|---|---|---|
| Laterality | One scalar; squared deviation from the source-weighted held-out mean. | How much held-out variation in the movement contrast the readout explains. |
| Future innovation and cached panel | Teacher features; squared error of the outer-training-mean reference, then mean featurewise R². | Prediction relative to a reference available from training sources. |

Both are legitimate declared statistics. Replacing either after seeing results
would change the experiment. Their absolute numbers should not be averaged or
used to claim that one study learns a better representation than the other.

The source bootstrap repeatedly samples whole source videos with replacement,
retaining each sampled source's clips and multiplicity. Paired draws compare
models on the same sampled sources. Its intervals condition on the saved fits;
they omit new training, hyperparameter selection, split variation and adaptive
redesign. Five seeds, 2,000 draws and many overlapping clips do not create more
independent people. Source separation also leaves participant overlap unresolved
when participant identifiers are absent.

## Step 6 — Run the smallest informative real-data extension

The cached accessibility panel changes the reference inputs while preserving the
teacher target, cohort, five outer source folds, three inner folds, safe
preprocessing, ordered skeleton summaries and raw controls. This makes it an
inexpensive way to test the new question before re-encoding video.

Its **support reference** has 331 inputs: confidence and validity averaged in
four ordered eight-frame bins, the frame-31 confidence/validity values, and
recorded frame rate. Its **posture reference** adds 66 frame-31 x/y coordinates,
giving 397 inputs. Invalid endpoint coordinates are missing values handled by
training-only preprocessing. Those coordinates inherit the parent's prefix-based
normalization, so this reference is a prefix-normalized endpoint, not a separately
observed instantaneous pose.

The added skeleton block retains the 924 ordered-bin features from direct-v3.
The real, time-shuffled, mismatched and no-skeleton histories are constructed
before the common feature builder. Every arm receives the recipient's same
reference inputs. Every arm has the same 36 joint penalty combinations and exact
baseline candidate. The no-skeleton history preserves time-varying validity.
The reference includes raw confidence means, while real history additionally
includes confidence means conditioned on valid observations. These are not the
same feature under missingness. Some standardized history confidence columns
also duplicate reference columns, creating an additional regularized route.
The primary contrast therefore changes a complete coordinate/confidence feature
block, not motion alone. Its nuisance adjustment is a declared finite feature
construction rather than an exhaustive removal of recording information.

For a cleaner subsequent study, include confidence, validity and transition
support once and identically across arms, then vary coordinate/displacement
features while controlling the reference route. Do this in a new frozen panel;
do not replace the completed estimate after seeing it. The present evidence
does not identify how much confidence or repeated-feature regularization
contributes to the small measured difference.

The **named primary contrast** is real-minus-no-skeleton in the posture panel.
The support panel and other comparisons are descriptive. This prevents choosing
whichever reference happens to yield the most attractive result. It remains a
development comparison on inspected sources, and its contextual target cannot
establish bounded physical forecasting.

The separate protocol defines a development lead only when the primary paired
95% interval lies above zero and the real arm's point score exceeds shuffle and
mismatch. This is an exploratory follow-up rule, not an amendment that changes
direct-v3's original +0.05 gate or turns its STOP into an ADVANCE.

From the repository root, the workflow is:

```bash
.venv/bin/python scripts/research_directions/iclr_bridge/run_cached_panel.py freeze \
  --source-run outputs/future-innovation-direct-v3-dev-20260911 \
  --output-root outputs/iclr-bridge-cached-20260911 \
  --protocol-document docs/studies/iclr/02_cached_panel_protocol.md

.venv/bin/python scripts/research_directions/iclr_bridge/run_cached_panel.py run \
  --output-root outputs/iclr-bridge-cached-20260911

.venv/bin/python scripts/research_directions/iclr_bridge/run_cached_panel.py verify \
  --output-root outputs/iclr-bridge-cached-20260911
```

The freeze step binds the implementation, specification and source evidence
before fitting. The run step fits or resumes that bound experiment. Verification
reloads selected models, reconstructs predictions and recomputes scores and
bootstrap summaries. It does not independently refit every saved inner candidate;
the candidate ledger remains the record of those selection fits. A changed
implementation requires a separately versioned experiment.

The [sealed panel report](../../../outputs/iclr-bridge-cached-20260911/reports/panel-report.json)
now records a complete measurement. All 1,480 pooled candidates are valid, all
256 target dimensions remain evaluable, and 17 of 40 selected fold-arm models use
exact baseline fallback. The independent check reconstructs the selected linear
fits, 102,400 prediction rows and 16,000 bootstrap rows. It checks inner loss
pooling and selection but does not independently refit all candidate models.
Parent and source-run artifacts remain unchanged.

| Predictor | Support-panel R² | Posture-panel R² |
|---|---:|---:|
| Shared reference | 0.008229 | 0.005891 |
| Real skeleton | 0.013865 | 0.013262 |
| Time shuffle | 0.012434 | 0.007692 |
| Clip mismatch | 0.008229 | 0.005891 |
| No skeleton, validity retained | 0.007296 | 0.011268 |

The named posture-panel increment is **+0.00199374 R²**, with paired 95%
interval **[−0.00667268, +0.01260843]** and **68.35%** positive draws. The saved
status is **no supported temporal lead**. This is a completed uncertain result,
not a failure to run the comparison.

Real history improves over the posture reference by +0.00737071, but the
no-skeleton arm already improves by +0.00537697. The real model also selects a
stronger penalty on the reference block in two folds. Its baseline gain therefore
cannot be described as a pure motion contribution. The matched real-minus-
no-skeleton contrast was chosen prospectively to compare the complete history
block with its validity-retaining control. Its observed estimate is smaller
and uncertain. This methodological choice preceded the new held-out scores.

The secondary support panel looks more encouraging: its matched increment is
+0.00656929, with 97.3% positive draws. Its 95% interval still includes zero
([−0.00007148, +0.01470943]), and its real-minus-shuffle interval is
[−0.00392084, +0.00854142]. It also removes current posture from the reference.
Promoting that result after seeing the table would bypass the question we froze.
The measured outcome supports a bounded target/pose investigation before
larger student training, while preserving direct-v3's separate STOP.

## Step 7 — Let the result choose the next experiment

![A staged workflow from evidence to independent confirmation.](figures/04_research_workflow.svg)

*Figure 4. Each stage resolves a different uncertainty. A positive synthetic
check supports implementation; a positive cached comparison supplies a development
lead; improved student performance and independent-source confirmation are later
requirements. A negative result can stop a branch without invalidating the study.*

| Observed pattern | What it suggests | Next bounded experiment |
|---|---|---|
| History predicts teacher features, but current posture works as well. | Static pose or support may explain accessibility. | Compare current-time supervision and future targets before claiming innovation. |
| History beats posture and matched controls. | The tested history representation improves prediction; its temporal mechanism remains unresolved. | Check a bounded future target and independent movement outcomes. |
| History helps only on a few sources or preprocessing choices. | The result may depend on source conditions or representation details. | Audit those mechanisms using training data; confirm a frozen choice on new sources. |
| All skeleton predictors remain weak. | Data amount, pose quality, target content and model family remain unresolved. | Test kinematic forecasting and teacher-to-motion readouts before scaling a student. |
| Distillation improves teacher-code matching but harms movement evaluation. | The selected target may still reward the wrong information. | Retain the negative transfer result and revise target selection separately. |

A learning curve can help distinguish sample limitations from a stable plateau.
At several training-source counts, refit transforms and repeat inner selection;
evaluate the same declared outcome on excluded sources. A curve based on 43
sources is a development diagnostic, not a reliable extrapolation to the required
sample size. New sources should be allocated deliberately between training
expansion and confirmation before inspecting their results.

## Step 8 — Build a temporal challenge that current posture cannot solve

The most interpretable new challenge contains similar current poses followed by
different motion. Consider a leg passing through the same position while moving
forward in one example and returning in another. Current position alone is
ambiguous. Recent observations identify velocity and can disambiguate the future.

![Matched current posture with different incoming motion and futures.](figures/05_matched_pose_futures.svg)

*Figure 5. Histories that pass through a similar current pose can imply different
next positions. The illustration is a synthetic mechanism, not a pair of measured
GAVD examples. A useful history representation should resolve this ambiguity
beyond a matched current-state reference.*

Begin with generated motion whose phase, direction, left/right amplitude and
missing observations are controlled independently. Verify that the declared
shuffle reduces the planted temporal signal. A simple velocity model should be
strong here; it is a required baseline, not an inconvenient competitor. Calibrate
weak signals as well as easy ones so that success is not limited to unrealistically
large effects.

The implemented [symmetry calibration](../../../work/artifacts/iclr-bridge-2026-09-11/laterality/symmetry-calibration.json)
provides a concrete starting example. It creates 48 sources, each with a closed
path and its reversal; 36 sources train the readout and 12 are held out. The two
paths share their endpoint posture and order-even summaries but have opposite
last velocities. Current posture, support, and mean/standard-deviation/absolute-
change summaries score R² = 0 on this deliberately balanced signed-velocity
target. Ordered signed velocities score 0.999999999978. This verifies the
constructed readout distinction; the target is observed velocity and a defined
continuation, not measured future human motion, and no JEPA is trained.

Next, construct controlled renderings in which motion stays fixed while appearance
and camera change. Split the underlying motion sequences and subjects before
creating rendering variants. Finally, form naturally matched GAVD pairs using
prefix posture, view and support only. A within-recording future difference,
Y(i)−Y(j), may reduce shared appearance, but the cancellation must be measured.
Pair selection must not inspect the future target or final test score.

This study requires more windows per recording and likely new teacher extraction.
The current cache has too few same-source windows for a strong paired study, and
its pooled vectors cannot recover dense body-region tokens or future skeletons.
Future-block encoding, mirrored teacher targets and different physical-time
horizons each need a new cache binding and the corresponding boundary audits.

## Step 9 — Distill selected information only after accessibility is established

The proposed method selects a small number of teacher directions: linear
combinations of its feature coordinates. A low-rank model limits how many such
directions are learned, which is helpful when independent training sources are
scarce. Rank zero represents an exact decision to provide no extra distillation
target.

Fit current-state and history predictors inside training sources. If residual
targets are used, obtain them with cross-fitting: each source's target uses a
baseline fitted without that source. Convert subfit predictions to common raw
teacher units before combining them. Fit and select the target map within the
same source boundaries, constrain its scale, and preserve a reference metric on
the original target. Otherwise, shrinking the target or choosing an easy
projection can lower the loss without improving the student.

Train the same small S-JEPA architecture with and without the selected target.
Keep the ordinary skeleton objective as an anchor. Equalize source data,
architecture, training budget and downstream readout opportunity across:

1. Ordinary S-JEPA and matched initialization.
2. Direct kinematic forecasting or motion-prediction supervision.
3. Complete teacher-feature distillation.
4. Unselected future-residual distillation.
5. An equally small teacher target selected without temporal information.
6. The proposed selection by held-source temporal value, including no transfer.

The primary outcome should be observable movement or a prespecified downstream
task. Teacher-code prediction supplies mechanism evidence. Laterality remains
one useful readout, supplemented by direction-sensitive future positions,
velocity or phase measurements. Evaluate observed-future features through the
same decoder as predicted-future features, following laterality Notebook 14:
the former tests whether the target representation can express the outcome,
whereas the latter tests forecasting.

A successful method paper must demonstrate that selection predicts **when
distillation helps** and improves a common student evaluation beyond strong
alternatives. Success on a target chosen specifically to be easy to predict does
not establish that claim.

## Step 10 — Position the contribution against the closest work

The building blocks have substantial precedents. [S-JEPA](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04755.pdf)
already predicts hidden skeleton features, and [MAMP](https://openaccess.thecvf.com/content/ICCV2023/html/Mao_Masked_Motion_Predictors_are_Strong_3D_Action_Representation_Learners_ICCV_2023_paper.html)
uses explicit motion targets and motion-guided masking. [SIE](https://proceedings.mlr.press/v202/garrido23b.html)
and [seq-JEPA](https://proceedings.neurips.cc/paper_files/paper/2025/hash/2f63d2963526bdd9ff1b8bcc2dc9905a-Abstract-Conference.html)
already separate invariant and equivariant representations. A reflection branch
or a new mask by itself is therefore a weak novelty claim.

Future supervision is also established. [Overlooked Poses](https://arxiv.org/abs/2208.01302)
distills privileged later poses for human-motion prediction. [Spectral-Guided
Physical Dynamics Distillation](https://openreview.net/pdf?id=P6F4MxtOKp), published
at ICLR 2026, uses future trajectories and spectral enhancement, including human
motion experiments. [Modality Focusing](https://zihuixue.github.io/MFH/index.html)
explains why crossmodal teachers may fail, while [C2VL](https://arxiv.org/abs/2405.20606v2)
already transfers visual-language knowledge to skeleton-only inference.

The stronger prospective contribution is the combination of a **specific
history-increment selection rule, observable transformation checks, and evidence
that the rule improves transfer or correctly rejects it**. Reflection supplies
a controlled diagnostic; it should not obscure useful symmetric motion. The
[literature memo and bibliography](../../../work/artifacts/iclr-bridge-2026-09-11/literature/literature-and-positioning.md)
record the checked sources and limits of this targeted novelty review.

The [BioGait-VLM preprint](https://arxiv.org/html/2603.08564v1) also combines temporal video aggregation and textualized kinematics for gait classification. Its temporal evidence branch aggregates observed frames; it does not establish the proposed prefix-only selection and future transfer criterion. This nearby application further limits novelty claims based only on adding temporal features or biomechanics to a teacher.

Rank the research paths accordingly. Selective student-accessible future
distillation is the strongest method direction if its benefit is demonstrated.
Matched-current-pose contrasts are a focused target-construction extension.
A systematic observable-dynamics evaluation study is a credible alternative
when it generalizes across representations and datasets. The existing two
negative studies alone do not establish a broad main-track claim about JEPA.

## Step 11 — Build an eight-page argument around measured evidence

An eight-page paper needs one central claim. A suitable eventual claim is that
training-only estimates of temporal accessibility identify teacher targets that
improve skeleton learning under limited data, while preserving specified motion
observables. This is currently a hypothesis. The draft must describe only the
results that actually exist and mark the remaining comparison as prospective.

| Main-text space | Argument and evidence |
|---|---|
| Page 1 | Scientific motivation and the two retained failures, with their different scopes visible. |
| Page 2 | Related work and the distinction between complementarity, accessibility and distillation benefit. |
| Page 3 | Precisely bounded inputs/targets and the selection method with rank zero. |
| Page 4 | Controlled temporal ambiguity, reflection checks and calibrated failure cases. |
| Pages 5–6 | Real student comparisons on common observable outcomes, with uncertainty. |
| Page 7 | Target/rank/control ablations and independent-source or participant-separated transfer. |
| Page 8 | Interpretation, limitations and reproducibility. |

For a strong main-track submission, the remaining evidence should include a
complete real student comparison, a reliable motion benchmark with participant
identities or controlled geometry, and confirmation that the mechanism extends
beyond the inspected GAVD cohort. A world-model claim additionally needs tested
state prediction over time. Agentic claims need actions, decisions and evaluated
closed-loop consequences; gait videos alone provide none of those.

As checked on 11 September 2026, the [ICLR 2027 author
guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines) specify nine initial
main-text pages and ten during discussion/camera-ready, with references and
appendices outside that limit. The user-requested eight-page form is an editorial
choice within the initial limit. The guide's later FAQ has inconsistent wording;
use the explicit initial-submission paragraph and recheck the live guidance.
The [official call](https://iclr.cc/Conferences/2027/CallForPapers) lists September
18 and 25, 2026 for abstracts and papers. Scientific readiness should determine
the submission, rather than forcing missing experiments into that short interval.

## Step 12 — Review the mechanism and writing adversarially

Before claiming an advance, trace each favorable result to a saved model, target
definition, source split and score. Challenge the easiest alternative explanations:
an improved RGB refit, unsupported columns, unequal feature support, a changed
target mask, an easier learned target, a selected outer-test outcome, or duplicated
seed results from a deterministic model. Recompute predictions and aggregates;
checksums alone cannot detect incorrect arithmetic saved consistently.

For representation claims, compare initialization and simple kinematics at equal
readout opportunity. For symmetry claims, compare to the exact post-hoc algebraic
construction. For temporal claims, require a direction-sensitive or future outcome
and a valid input boundary. For distillation claims, require a trained student's
independent evaluation. For generalization, count independent sources and people,
not repeated windows or bootstrap draws.

Retain the negative branch of every experiment. A selection procedure that
reliably chooses no transfer in an unsuitable regime may be useful, but that
claim itself needs repeated controlled cases and real comparisons. The valuable
research contribution is a more dependable relationship between a training
target and the movement information a representation supports. The proposed
workflow makes that relationship measurable before committing to larger models.
