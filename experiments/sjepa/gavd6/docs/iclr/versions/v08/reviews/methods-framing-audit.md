# Audit for the in-place v08 predictive-learning framing revision

This read-only audit covers the completed source/configuration/export contract and notebooks 00–07 and A–G. It adds no fitting, dataset transformation, new statistical test or notebook execution. Source pointers below are relative to `src/gavd6_sjepa/research_directions/`; notebook cell numbers are one-based and include markdown cells.

## Recommended scientific framing

Self-supervised learning and world-model research can supply the motivation, but their names must not change the executed experiment's supervision. The model predicts missing reference features within a complete observed window. Its teacher receives projected synthetic references; coordinate readouts and paired angular objectives also use those references. It is therefore an **SSL-inspired predictive representation procedure evaluated with privileged paired synthetic supervision**, not a demonstration of self-supervised learning from unannotated natural observations. It learns neither a tested future transition model nor an action-conditioned rollout or planner. Bidirectional access to the whole window is intentional offline restoration, not leakage in a claimed forecasting task.

Suggested introduction prose:

> Predictive representation learning motivates learning features that retain the state information needed for downstream decisions. For movement measurement, that requirement includes the magnitude, timing and anatomical side of a change after features are decoded. We test one part of this motivation: whether an S-JEPA-inspired feature predictor, trained with privileged synthetic reference poses, makes a paired movement response more recoverable than the tested coordinate-learning procedures. This is a controlled restoration study, not a future-prediction or planning evaluation.

Notebook 00 cell 3 explicitly identifies within-window feature prediction and excludes future-window forecasting. Notebook 03 cells 3–16 traces observations, tokens, predictor and coordinate decoder; notebook 04 cells 10–16 places projected references in the training teacher and losses. Model input allow-lists and the bidirectional token implementation are in `synthetic_training_v2/models.py:16,45–62,94–125`; teacher construction is in `gait_fidelity/training.py:594–625`. The literature audit in `methods-initial.md` supplies the verified primary S-JEPA, PoseBERT, MotionBERT and SmoothNet scope. Calling the completed system simply a “world model” would imply capabilities not measured here.

## Initial question and stage-specific nulls

No explicit `H0`/`H1` statements were found in the source notebooks. Saved configurations declare candidate/comparator identities and endpoints; those declarations are stronger evidence than retrospective prose claiming an original null formulation. Define `D_ps = e_comparator,ps − e_candidate,ps`, after the executed condition→window→motion→person reduction. Positive improvement favors the candidate. Let μ denote its mean under the stated population/seed weighting.

The manuscript can **retrospectively formalize** the directional no-benefit hypotheses as `H0,j: μj ≤ 0` versus `H1,j: μj > 0`, while retaining the original two-sided descriptive intervals and avoiding any claim that a new one-sided test was performed.

| Scientific question | Recorded comparison and endpoint | No-benefit quantity | Status |
|---|---|---|---|
| Initial operational question: does reference-feature prediction help beyond direct coordinate adaptation under the original change package? | Core JEPA/change versus direct/change; pooled movement-response error | μcore = mean(error direct/change − error core JEPA/change) | Candidate/comparator declared in `walking-core/config.json`; formal null wording is retrospective. This is not a contrast against the stronger direct/coordinate arm. |
| Does coupled feature-difference supervision add value beyond continuous endpoint-feature supervision? | Delta/change versus endpoint/change; pooled movement-response error | μresponse = mean(error endpoint/change − error delta/change) | Response primary retained in `jepa-response/config.json`. Coordinate-readout contrast and readout interaction are secondary. |
| Does dense paired angular supervision add waveform benefit beyond reducing scalar weight? | Delta/dense versus delta/low scalar; ViTPose waveform error | μrepair = mean(error delta/low − error delta/dense) | Nested `repair.statistical_protocol` in repair config is authoritative; inherited top-level evaluation still names the earlier response comparison. |

The initial research motivation is broader than the first operational comparison. A single null about “SSL/world models preserve gait” has no unique comparator, outcome or estimand and was not tested. Likewise, “preserves waveform” needs an explicit tolerance to become a noninferiority claim. All three configs have `meaningful_margin_deg: null`; repair specifies no response noninferiority claim. No clinical-effect, equivalence or noninferiority margin may be supplied retrospectively as original rigor.

The retained estimates are 0.688° [−0.640, 1.977] and 0.373° [−1.110, 1.760] response improvement under crossed person/seed intervals, and 0.279° [−0.195, 0.753] waveform improvement under repair's person-t interval. Their uncertainty leaves average benefit unresolved. Do not say a null was accepted, equivalence established, or a general family failure demonstrated. The three stages reuse people and form a development sequence, not independent replications.

Provenance: `docs/studies/gait-fidelity/methods/core-to-followup-20260924.md:3,7,21–41` documents an amendment after core inspection, preserving the response primary and adding base readouts; `methods/jepa-response.md:130–155` describes descriptive development inference. The core/response `evaluation` config fields and repair's nested protocol fix the executed contrasts. Timestamps/hashes document local artifacts, not an independent preregistration registry. The initial v08 audit records those timestamps.

## Notebook arc and completed versus instructional work

| Notebook(s) | What they establish or explain | What cannot be claimed |
|---|---|---|
| 00 | Research quantities and the three completed stages; distinguishes generated fixtures from retained evidence | Fifteen notebooks are not fifteen experiments. |
| 01 | Cohort identity, physical references, pairs, naming corruption, input-only normalization and inverse transform | Filename screens or projected joints do not certify clinical gait or anatomical truth. |
| 02–03 | Fixed joint/time grid, stochastic masks, channels, transformer and residual coordinate readout | Shape/value parity does not establish accuracy or GPU compatibility. |
| 04 | Hierarchical training draws, CE/VICReg and coordinate/angular losses, optimizer/EMA order, frozen versus direct fitting | Scratch CPU updates are software checks; the scheduled source pipeline supplies study fits. |
| 05–06 | Reference-fixed evaluation, separate assignment denominator, hierarchical means and paired intervals | More renderings, windows, seed rows or bootstrap draws do not create participants or untouched evaluation. |
| 07 | Reconstructs completed core→response→repair results from `outputs/iclr`; labels results versus illustrations | Its constructed waveforms are not restored source examples. |
| A | Coordinate/JEPA and output-objective comparison logic | Only graph-time source fits completed; the full multi-mask factorial is not completed evidence. |
| B | Anatomical topology/duration controls and mask audits | No completed source comparison establishes the value of graph topology or interval duration. |
| C | Direct, static, temporal-refinement and calibration control designs | Direct fits and deterministic controls completed; static/learned temporal-refiner source comparisons are not in the completed core. |
| D | Initialized and shuffled-reference controls, including donors confined to a training person/stratum | Shuffling reference windows does not randomize movement-pair coupling in the later delta auxiliary. |
| E | Re-pairing/per-example-label controls and distribution audit | These controls were not fitted in the completed studies; re-pairing is not the later readout repair. |
| F | Feature/coordinate residual auxiliaries, support, gradients and calibration; completed response evidence | Shared coefficient does not equalize auxiliary influence; source findings remain finite-procedure comparisons. |
| G | Dense/low-scalar losses, common support, calibration and frozen readout updates; completed repair evidence | Equal initial gradient RMS neither persists automatically nor isolates gradient sparsity as a mechanism. |

Use the notebooks as a **reconstruction path for the scientific argument**, not as evidence that every instructional branch was run. Main text should motivate the question, specify its comparison, then report the source result and the next question it generated. A compact appendix notebook map is sufficient; the main paper need not narrate notebook execution chronologically.

## Person boundaries and concrete leakage protections

| Boundary | Audited implementation evidence | Accurate statement and limit |
|---|---|---|
| Identity before augmentation | `gait_fidelity/cohort.py:123–152,196–207`; inherited `synthetic_training_v2/contracts.py:58–93` | Audited canonical identities join to original person splits; exact duplicate motion hashes cannot cross identities/splits. Original train/validation/test map to train/development/locked confirmation. This relies on the supplied identity audit; it is not biometric proof against unknown aliases. |
| Windows and derived variants | `cohort.py:181–209`; `data.py:136–168`; `preparation.py:517–528` | Nonoverlap window starts are frozen before rendering. A source family binds person, raw-motion hash and split; all edits, mirrors, views, naming/occlusion conditions and estimators inherit that split. Random rendered-row splitting was not used. |
| Fit population | `training.py:43–72,403–408,557–573` | Pair construction includes only train rows; held intervention is rejected; losses use selected train endpoint indices. Core/response ledgers record 112 training people and 315,840 available training endpoint rows. |
| Held settings | `preparation.py:466,506`; inherited contracts `:88`; `data.py:139–140` | The 15° edit and ViTPose family are absent from training. They are development condition shifts, not a substitute for an untouched person test. |
| Masked normalization | `training.py:288–321`; notebook 01 cells 22–25 | Origin/scale use observed, unhidden context only; fixed fallback does not depend on hidden or reference values. Per-window test-time normalization uses that window's observations, not fitted development statistics. |
| Inputs versus targets | `data.py:99–120`; `models.py:16,45–62,105–125`; `training.py:797–814` | Student/inference dictionaries contain only xy, native confidence, observed mask and timestamps. No reference validity, reference scale, labels, person or intervention metadata enter deployed predictions. Teacher/loss supervision is privileged and explicit. |
| Calibration | `response_calibration.py:139–173,184–231`; `repair_training.py:69–81,261–305` | Gradient calibration selects train-only pairs at fresh initialization and performs no fitting update. Repair verifies unchanged model hashes. The coefficient is not selected by minimizing development error. Subsequent design decisions nevertheless used development results. |
| Spatial controls | `gait_fidelity/evaluation.py:208–238` | Offset/affine correction fitting rejects nontrain records and uses training hierarchy weights. Applying a fitted correction to a development observation is not refitting it on development labels. |
| Confirmation | `data.py:92–95,218–224`; `training.py:403–404`; `repair_training.py:72–73`; `cohort.py:286–299` | Ordinary fit/load paths reject confirmation, and explicit admission requires a locked declaration/checkpoints. All completed configs state `confirmation_admitted: false`. No independent test outcome exists here. |

**Important precision:** do not claim that development reference tensors are never opened in the training process. Core training calls `validate_bundle(bundle)` before train-only pair selection; response calibration similarly validates the ordinary full bundle. Those validators may read development reference arrays for consistency or hashing. The audited claim is that development references do not enter optimization/calibration losses. Repair's context additionally validates only a lazy training subset. Neither distinction prevents adaptive reuse of development outcomes.

As a local identity cross-check, the union of people in the retained response probe's train/test folds contains 112 training identities and has empty intersection with the 14 development identities in the per-person result CSV. This is consistent with the fit/evaluation separation, but is not a replacement for revalidating every original row: full raw bundle/manifests are not in the compact packet. Probe folds hold people out only from a ridge fit; their encoders already saw those training people. `response_diagnostics.py:20–47,52–105` fits feature standardization inside each ridge training fold, and explicitly labels the result a training-population diagnostic.

## What exact shape preservation means

The completed configurations all use `window_size=128`, `patch_size=4`, `width=96`, 12 named joints, and 25 Hz. At the restoration interface,

\[
X,\widehat Y\in\mathbb R^{B\times128\times12\times2},\quad
c\in\mathbb R^{B\times128\times12},\quad
O\in\{0,1\}^{B\times128\times12},\quad
t\in\mathbb R^{B\times128}.
\]

Targets have matching xy shape and separate boolean validity/visibility. Packing is

\[
[B,128,12,5]\to[B,32,12,20]\to[B,384,96]
\to[B,384,8]\to[B,128,12,2].
\]

The first two operations regroup four consecutive frames of one joint; `patch*12+joint` fixes token identity. Every slot remains even if entirely missing. Artificial masking zeros xy/confidence/usable channels and retains time/position; natural absence zeroes xy/usable while finite native confidence can remain. The decoder exactly reverses the token indexing to recover the original frame/joint grid. A paired loss may view the batch as `[B/2,2,128,12,2]`; this rearranges already adjacent endpoint rows, not time or anatomical coordinates. Source: `models.py:105–125,128–141,227–236`; `training.py:808–814`; notebook 03 cells 3–16 and G cell 3.

This is **interface and indexing preservation**, not preservation of the raw AMASS dataset's size or contents. Preparation deliberately selects/windows sequences, resamples recorded rotations by SO(3) interpolation and translations/dynamic shape linearly on one physical-time grid, applies knee edits and physical mirroring, renders cameras/occlusion, selects body-12 projections, and multiplies condition records. Source: `motion_preservation/motion_data.py:82–112`; `gait_fidelity/preparation.py:448–535`. It does not stretch each leg to its own phase or normalize limbs independently. Input normalization changes coordinates by one shared isotropic scale and translation, preserving angles before its inverse; projection and intentional corruptions can change measured content.

Training/development have different leading dataset row counts because held conditions are excluded from fitting: 1,645 training windows ×192 records =315,840 rows, versus 155 development windows ×360 records =55,800. The candidate plan's 113 training/15 development people and 2,171/214 windows are **pre-admission plans**, not final counts; use completed ledger/report counts of 112/14 and 1,645/155. Shapes and populations were not modified during this manuscript audit.

## Inference and validation claims that remain unavailable

The primary uncertainty machinery preserves matched methods within person/seed cells. Core/response use crossed resampling of 14 people and three fitted seeds (2,000 draws, seed 731); repair averages seeds first and uses `t.975,13 sd/sqrt(14)`, conditional on those fits. These intervals do not account for the adaptive research history, establish equivalence or identify clinical usefulness. Auxiliary imbalance, almost universal new JEPA pretraining clipping, separate teachers and missing movement-pair randomization still limit mechanism claims. The framing revision cannot remove those existing limits.

Current source notebooks have cleared execution counts. The retained refresh receipt reports 64 passed targeted tests, passed F/G calculations and a passed 07 walkthrough, but explicitly marks full fixture execution blocked by disk exhaustion and HAIC validation false. Current raw notebook hashes match that receipt for F/G; the other 13 differ. Retained executed copies of 00–03 differ from current code only in the helper import path, and retained 07 differs only in optional export settings; their substantive calculation cells match. This supports a narrow statement about retained software checks, not “all current notebooks were executed successfully.” No new tests were run for this audit. Source: `docs/studies/gait-fidelity/records/tutorial-refresh-20260925.json` and retained notebooks under `outputs/gait-fidelity/tutorial-transparent-20260925/notebooks` and `notebook-walkthrough-20260925`.

Suggested compact main-text sentence:

> Canonical-person splits precede windowing and rendering, and every derived variant retains its source split. Optimization and calibration use training pairs only; normalization excludes references and artificially hidden values, and inference preserves the fixed 128-frame, 12-joint grid using observations alone. The 14 evaluated people are training-disjoint development participants reused across stages, not an untouched test cohort.

Suggested appendix qualification:

> These guards establish the implemented loss/input and known-identity boundaries. They do not certify an uninspected evaluation history, unknown cross-dataset aliases, all upstream estimator training provenance, or end-to-end replication from the compact packet. The notebook calculations explain and check the source procedure; only retained completed fits supply scientific outcomes.
