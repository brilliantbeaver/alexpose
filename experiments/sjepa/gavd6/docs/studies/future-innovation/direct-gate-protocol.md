# Experiment 0: the 50-clip direct skeleton gate

This is the current `direct-v2` protocol, adopted on 11 September 2026 after
inspection of the legacy teacher-selectivity rejection. It is an explicit
protocol amendment, not the original preregistration or a result from the new
prediction comparison. New run directories freeze it before fitting. Existing
`legacy-v1` runs retain their original rules and artifacts.

## Question and primary comparison

Does past skeleton coordinate/confidence history improve a small predictor of
held-out teacher features after RGB, recording details and observation-validity
flags are already available? Skeletons are extracted from RGB. A positive result
would establish an advantage of supplying that representation to the tested
predictors, rather than new information unavailable in the video.

The primary estimate is the source-balanced, out-of-fold R² of the real-skeleton
head minus the R² of the matched no-skeleton head. Both receive the same RGB and
nuisance inputs and have the same number of parameters; no-skeleton zeros x, y
and confidence while retaining the time-varying validity mask. Gain over the
RGB+nuisance ridge baseline remains a separate required comparison. A larger
residual head can improve the RGB prediction alone, so gain over ridge by itself
cannot establish an advantage from skeleton coordinates.

## Cohort and temporal boundary

Experiment 0 selects **50 eligible clips** from the available full-GAVD candidate
pool, with **at least 25 source videos and at most two clips per source**. Candidate
order, valid window choice and source folds use fixed hash-based ordering.
Selection stops at 50 before teacher features or prediction scores are inspected.
Unselected candidates beyond that point are not pose-quality failures. The full
dataset is reserved for the subsequent real experiment, under a separate plan.

A clip has 64 contiguous, exactly decoded source frames and aligned annotation
boxes. The predictor receives frames 0–31. Skeleton input has shape
`[32, 33, 4]`: normalized x, normalized y, confidence and validity. Normalization
uses only that prefix. Original per-joint validity rules remain, but there is no
45% whole-body-coverage cutoff and no 90% crop-retention cutoff. At least one
joint must have observations at adjacent prefix frames, and context/target person
token regions must be usable. Missing videos, failed decoding, unavailable boxes,
and an entirely unusable skeleton cannot supply this comparison. Coverage and
retention values and optional overlays are saved for interpretation.

The frozen V-JEPA 2.1 teacher uses the pinned official encoder and checkpoint.
At 384×384, 16-pixel patches and two-frame tubelets produce a 24×24 spatial grid.
Future tokens are removed before attention for the input encoding. The input
concatenates global prefix pooling, last-prefix person-region pooling and prefix
background-region pooling, followed by nuisance features. Its target is a fixed
256-dimensional projection of person-region features at frames 38–39, encoded
with **the full 64-frame clip**. The target can therefore reflect observations
after frame 39. This is prediction of contextual teacher features from a prefix;
it is not a measurement of decoded future gait or a trained S-JEPA model.

RGB, pose and camera-motion summaries use only the prefix. Nuisance inputs also
include frame rate, sequence duration/relative position, source dimensions, view,
box motion, confidence and missingness. Duration/relative position are explicitly
permitted offline metadata. If background pixels, adjacent-frame background flow,
or background tokens are unavailable, their summaries are zero and separate
prefix support fractions identify the missing measurements. No background quality
score or minimum background extent is required. Recording cues can still explain
part of the contextual target.

## Retained checks and fitted comparisons

Three windows chosen before teacher inference check repeatability and invariance
of prefix features to randomized future pixels. The existing float32 repeat/cache
tolerance is 1e-6; the leakage tolerance is the greater of 1e-6 and twice measured
repeat error. Person-target training variance, artifact lineage, source splits,
finite arrays and complete predictions are still checked. These are checks that
the requested comparison can be computed without leakage or corrupt inputs.

The direct gate omits person/background pixel replacements, the sensitivity ratio
and edit-direction prerequisites, and the background-target arm/gain-reduction
rule. It produces no selectivity contact sheets and makes no claim of teacher
selectivity. Legacy records keep those requirements.

Five source-disjoint outer folds contain all clips from each held-out source.
Three inner source folds select ridge penalty, residual-head weight decay and
update count using training sources only. All scaling, imputation, target masks
and ridge residuals are fitted within the applicable training partition.
Source holdout does not establish participant independence. Sources that enter
this development gate cannot subsequently be described as an untouched test
cohort; the full experiment needs its own frozen development/evaluation plan.

Four arms share the person target and fold-local ridge baseline: real skeleton,
four-frame time-block shuffle, different-source clip mismatch, and no skeleton.
Shuffle moves coordinates, confidence and validity together. Mismatch donors are
chosen separately inside each training, validation and test partition using
context metadata, without crossing source-fold boundaries. The temporal control
therefore changes the alignment of observation support as well as coordinates.

The existing width-64 heads, seeds 7/19/31, ridge alphas
0.1/1/10/100/1000, AdamW decay 0.01/0.1, and update choices 25/50/100/200 remain.
There are **60 final heads** (5 folds × 3 seeds × 4 arms), plus inner selection.
Teacher features are cached once. No shortened teaching configuration enters a
real run. The smaller synthetic fixture exists only for implementation tests.

## Scores and next decision

Scores pool held-out predictions within each seed, balance source videos, and
average featurewise R² using the recorded training-variance mask. Seed scores
are then averaged; predictions are not ensembled and fold R² values are not
averaged. The report includes both gain over ridge and the paired skeleton
increment, per-seed results, 2,000 paired source-bootstrap draws and 95% intervals.
Intervals resample saved out-of-fold predictions and are conditional on those
fitted models; nested model selection and training are not repeated in each draw.
Repeated seeds and bootstrap draws add no independent source videos.

`ADVANCE` requires complete valid evidence and all of the following:

- Mean real-head gain over ridge is at least 0.05 R².
- Real gain is at least twice the nonnegative shuffled gain; mismatched-clip gain
  is at most 0.01 R².
- Real gain is positive in all three seeds, reaches 0.05 in at least two seeds,
  and is positive in at least 90% of paired source-bootstrap draws.
- The mean real-minus-no-skeleton increment is strictly positive, positive in
  each seed, and positive in at least 90% of paired source-bootstrap draws.

These retain the original effect/control criteria except the removed background
criteria, and add an explicit matched-capacity requirement. A 95% interval that
contains zero still leaves uncertainty, including when a 90%-positive decision
rule passes. The report states the interval and decision threshold separately.
`INCONCLUSIVE` means point checks passed but stability did not. A complete `STOP`
is a measured failure of at least one required comparison; an incomplete `STOP`
is an execution or input problem with no predictive conclusion.

`ADVANCE` recommends designing the **full-GAVD JEPA experiment**, comparing trained
features, matched initial encoders and raw skeleton history with frozen source
splits and training-only readout selection. Decision fields
`allow_full_experiment` and `allow_jepa_training_comparison` record that
recommendation; these notebooks never launch that training. Adapter distillation
remains disallowed. Synthetic and incomplete runs cannot authorize either step.

## Execution and reproducibility

Keep notebooks 00–04 and the existing HAIC allocation/dependency structure.
Set a **new** `FI_RUN_ROOT`, optionally `FI_EXPERIMENT_PROTOCOL=direct-v2` (the
new-run default), and use the [notebook launcher](../../../slurm/future-innovation/NOTEBOOKS.md).
Do not repurpose the legacy run directory. Cohort rules, nuisance schema and
control contracts differ, so old fitted folds and caches are not silently imported.

Within the new run, completed cohorts, per-window teacher caches, audits and
folds are verified and reused. An interrupted pose stage before cohort freeze
restarts candidate processing; per-window pose resumption is not implemented.
Model fitting remains CPU based. Empty background regions no longer abort cache
construction, but real media availability, the installed detector and H100 teacher
inference still require verification on HAIC. The local test record distinguishes
injected software fixtures from real data and real pretrained-teacher evidence.

The JSON decision retains checkpoint, manifest, readiness, protocol, run/config
and code hashes. The report and decision are sealed together only for a complete
measurement. See the [implementation review record](notebook-run-investigation.md)
and [validation record](../../../slurm/future-innovation/VALIDATION.md).
