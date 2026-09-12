# Source learning curve for repaired Experiment 0

Protocol `source-learning-curve-v1`, specified before expanded-cohort fitting.
The historical direct-v2 and direct-v3 STOP results remain unchanged. This is a
development study of sample size, not a student-training experiment.

## Question and fixed scientific boundary

Does increasing the number of training recordings improve contextual teacher
prediction, and does it increase the advantage of real skeleton history over
matched validity information? We retain `joint-ridge-v1`, `supported-input-v1`,
`training-target-v1`, `ordered-bins-v1`, all four raw controls, 2,382 RGB/nuisance
inputs, 924 skeleton summaries, the original 256-column projection, and frames
0–31 as inputs. The person target at frames 38–39 is encoded using all 64 frames.
This target is contextual, not an isolated future state. Coordinate/confidence
routes and their interpretation remain those of direct-v3; this study does not
claim to isolate coordinate motion after exhaustive quality adjustment.

## Inventory, prior exposure, and confirmation reservation

Freeze the full sequence/video manifests, the parent candidate and eligibility
records, and every supplied prior-exposure manifest. Count annotation records,
candidate windows, verified eligible windows and unprocessed windows separately.
Selection exclusions after the old 50-window cap are not pose failures. Prior
pose failures are recorded separately from pending eligibility. Missing media or
model resources are execution blockers, not new scientific exclusions.

All sources in supplied historical experiment manifests are development sources.
The initial local reservation conservatively includes the 642-sequence,
94-source historical GAVD cohort as well as the 50-window FI cohort and retained
real gait-parity manifests. This is an exposure registry, not a new diagnosis
analysis. Labels are never used to choose splits. Reservation is relative to
this documented registry; unknown historical exposure cannot be ruled out.

An optional explicit `video_id,participant_id` registry links recordings of the
same known participant. Connected recordings remain together across every
boundary. Blank participant IDs are unknown, never distinct inferred people.
The source manifest's annotation `id` means video ID, not participant ID.
Without participant metadata, report participant count as unknown and make only
recording-disjoint claims. A new participant/exposure registry after freeze
requires a new reservation; do not quietly move confirmation sources.

Rank unexposed connected source groups by a fixed hash with seed 261101. Reserve
ceil(20% of these groups) for confirmation before expanded pose processing or
teacher inference. All clips in a reserved source are held out of the learning
curve, transforms, donor pools, training and evaluation. Do not replace failed
or unavailable confirmation sources with better-looking sources. Confirmation
encoding and evaluation are deliberately not implemented by the development
commands; they require a separately frozen confirmation experiment.

The rest form development data. Unlike the old gate's two-clip cap, use all
eligible annotated sequences in a selected source, one deterministic 64-frame
window per sequence. This source-cap amendment is necessary to test the full
inventory. Report source and clip counts at every size: 40 sources is not the
same intervention as the historical 50 clips. Source weights give every recording
equal total influence. Availability and eligibility may lower all advertised
counts; do not fabricate a nominal 1,800-window complete cohort.

## Nested source sampling and regularization

Keep five fixed outer source-group folds and three inner source-group folds.
Each development source is evaluated once per size/repetition in its outer fold.
The exact evaluation sources and windows remain fixed across sizes and sampling
repetitions. For each outer training pool, use independently hashed group orders
with subset seeds 261201, 261202, 261203. Prefixes select approximately 40, 80,
160, and all eligible training sources; keep known participant groups intact,
possibly exceeding a nominal size. Include every eligible clip from a selected
source. Sizes exceeding a fold's pool are recorded as unavailable, never capped
and relabeled. Fit the all-sources endpoint once per fold and reference that
artifact from every repetition; it is not three independent fits. Collapse any
other identical source subsets by their identity. Sampling repetitions measure
data-composition sensitivity, not optimization stability.

For a fit with n windows, each source has equal total weight and weights sum to
n. Retain the summed-loss solver but set lambda_x=n*rho_x and lambda_s=n*rho_s.
Both rho grids are [0.0025, 0.025, 0.25, 2.5, 25, 250]. These equal the direct-v3
grid divided by the declared 40-window anchor. Thus effective regularization
per unit average loss stays comparable across sample sizes and inner subfits.
At n=40 the numeric penalties equal the old grid. This does not reproduce the
old inner selection exactly; that implementation used fixed summed penalties.
Save nominal anchor penalties, rho, n, weight sum and actual penalties for every
fit. The intercept remains unpenalized. No neural model or extra feature family
enters this comparison.

Use the shared selected RGB reference, all 36 joint pairs per arm, and exact
baseline fallback. Reuse direct-v3 pooled source-weighted inner errors, ties
(baseline within 1e-10 + 1e-8 times loss; otherwise stronger penalties), numerical
failure policy, and target scaling. All learned statistics are fit within the
applicable source subset. Inner splits, preprocessing IDs, donor assignments,
all candidate losses, selected models and actual penalties are retained.
Any failed required candidate makes the scientific comparison incomplete.

## Scoring, uncertainty, and interpretation

Pool outer held-out predictions within each subset repetition. Use the declared
predictive R², whose reference is each fit's outer-training mean in its own
training-standardized units. Intersect training-derived valid target dimensions
across every required fit and size; held-out variation never defines the mask.
Report RGB-only, all four arms, real-minus-RGB, real-minus-no-skeleton,
real-minus-shuffle, mismatch-minus-RGB, fallback counts, and per-repetition
values. Average scores across subset repetitions, not predictions. At the all
endpoint identical referenced scores have zero subset-composition spread, not
evidence from repeated optimization.

Use 2,000 paired whole-source bootstrap draws, seed 260905, with the same source
multiplicities across arms, sizes and repetitions. Preserve within-source clips.
Report 95% intervals and fractions positive separately, including largest-minus-
smallest matched increment. Draws condition on saved fitted models and omit
retraining and adaptive development. Known participant grouping prevents fitting
leakage; source bootstraps still need caution if several sources share a person.

Because the training-mean R² denominator changes with training size, additionally
report source-weighted raw teacher-unit mean squared errors. Use their paired
change when interpreting absolute learning curves. Do not silently replace the
declared R² with test-mean centering. Numerical verification reconstructs models,
predictions, training transforms, masks, candidate selection and report arithmetic.

The largest feasible endpoint is the prospective decision endpoint. Preserve
direct-v3 effect rules: real-minus-RGB >=0.05, real gain >=2*max(shuffle gain,0),
mismatch gain <=0.01, positive matched increment, and >=90% positive paired draws
for real gain and matched increment. Also report each subset repetition; a
deterministic solver has no stochastic-seed pass. Synthetic runs can never
advance. A positive matched-increment growth estimate with >=90% positive paired
draws is a development trend, not proof of monotonic scaling or a confirmation
pass. A flat/uncertain increment alongside lower raw prediction error motivates
a separate target/representation study. Missing requested sizes are explicit
feasibility limitations; missing required fits/data are incomplete execution,
not a scientific STOP. No student training launches automatically.

## Execution and preservation

Use a new sibling output root. Freeze protocol text, model policy, manifests,
source reservation, software fingerprint, runtime, parent cache/config/audit
lineage and digest/size/mtime snapshots. Parent artifacts are read-only.
Expansion requires original full-source media, per-frame annotation boxes and
the identical pose model. Rebuild alignment candidates with existing code;
require every old candidate ID/window boundary to agree. Run the same pose and
feature constructors. Copy the original projection bytes. The development cache
binds the new cohort and new teacher audits; parent evidence is not relabeled as
evidence for new clips. Verified parent cache entries may be reused only for
matching window/input definitions and retain original bindings and lineage.

Stages use independent locks and atomic receipts. Completed stages verify before
reuse. Interrupted pose/cache items without completion receipts are recomputed;
completed mismatched items fail. Scientific artifacts are sealed and read-only
after reporting. Relocation is explicit: copy the frozen root unchanged and
provide runtime parent/media/model locations whose contents match frozen hashes.
CPU fitting never loads teacher weights; missing expanded arrays cannot trigger
unrequested teacher inference. The local environment can audit and calibrate;
actual expanded results require the completed data/cache/audit stages.
