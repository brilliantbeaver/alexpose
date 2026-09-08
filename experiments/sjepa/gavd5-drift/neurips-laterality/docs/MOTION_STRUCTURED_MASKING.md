# Motion and structured masking: the next controlled study

Primary sources reviewed on 8 September 2026. This extension implements and
tests masking alternatives, with four new notebooks numbered 15–18. Its
generated examples are software evidence. No new real-data masking benefit is
claimed. The completed findings in `TUTORIAL.md` and notebooks 00–14 retain
their original scope.

The [verification record](MOTION_STRUCTURED_VERIFICATION.md) documents fresh
notebook executions, passing extension tests, visual checks, and the two
existing broader-suite failures specific to this checkout's prerequisites.

## Why this direction follows from the current results

The tutorial reports a complete comparison of gait-only and all-landmark
scattered targets. The trained predictor distinguishes correct clip targets
from mismatched targets, while frozen trained features do not outperform the
matched initial representation on the laterality endpoint. This makes
**recovering movement information** the first research priority.

Two explanations are economical to investigate. First, scattered masks may
leave enough neighboring observations to make local completion relatively easy.
Second, averaging features over time may suppress useful movement variation.
Neither explanation follows automatically from the observed result. The new
suite measures available context cues, tests a recoverable movement signal,
and applies the same summaries to initial and trained encoders.

The raw pose cache is available locally; the revised GAVD workflow has prepared
the 625-clip cohort and five source folds using the original protocol. New full
training-grid checkpoints remain absent. The tracked numerical summaries support reading the old
results and recovering their configuration, but cannot supply model features.
The [GAVD workflow guide](MOTION_GAVD_WORKFLOW.md) explains exact-inventory
recovery, input preparation and the shared five-fold/five-seed execution path.
The notebooks never create replacement empirical results from generated data.

## What the primary sources establish

| Source | Relevant design and evidence | Consequence for this experiment |
|---|---|---|
| [MAMP, ICCV 2023, §3.4 and Table 8](https://arxiv.org/html/2308.07092) | Motion-derived probabilities guide stochastic target selection. Its mask-only ablation compares random and motion masking; its full method also predicts coordinate motion. | Implement motion selection while keeping our teacher-feature target fixed. Do not attribute the full method's gains to the sampler. |
| [Official MAMP implementation](https://github.com/maoyunyao/MAMP/blob/main/model_mamp/transformer.py) | `motion_aware_random_masking` averages absolute patch displacement, normalizes by the clip maximum, and removes the largest Gumbel-perturbed weights. | Distinguish the implementation convention from the paper's unnormalized intensity equation. Test agreement on fully observed arrays. |
| [S-JEPA, ECCV 2024, §3](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04755.pdf) | Uses motion-based masking and contextualized full-input teacher features, centering, cross-entropy and an EMA teacher. | Motion selection is a directly relevant omitted comparator in the completed gait adaptation. |
| [VideoMAE, NeurIPS 2022, §3.3](https://arxiv.org/html/2203.12602) | Repeats the spatial mask throughout time to reduce access to redundant temporal content. Its tube/random comparison uses RGB action recognition. | Hide entire landmark trajectories. Its high RGB masking ratios do not determine appropriate pose budgets. |
| [I-JEPA, CVPR 2023, §3 and Table 6](https://arxiv.org/html/2301.08243v3) | Predicts multiple target regions from an informative context with target/context overlap removed. Its ablations favor this design within its image objective. | Connected targets are worth testing, but an image block is not an anatomical region, and mask quality depends on the learning objective. |
| [SLiM, arXiv v3, §3.3](https://arxiv.org/html/2603.10648v3) | Uses connected anatomical subsets over consecutive spans, with duration varying inversely with region size; also changes tokenization and alignment losses. | Connected anatomical tubes already have close precedent. Our fixed-volume comparison isolates mask geometry and does not reproduce the full SLiM method. |

The evidence comes mainly from action recognition rather than signed gait
regression. These sources motivate controlled comparisons; they do not establish
that a moving region is always the most useful region to hide. A less active
limb can carry the clinically interesting contrast, and tracking noise can
produce large coordinate differences.

## Precise masking definitions

Let `x[t,j,c]` contain prepared coordinates and `V[b,j]` indicate that every
observation in a four-step token is valid. Every deliberate mask is a subset
of `V`, contains unique targets and leaves at least one valid context token.
Natural missingness is not a prediction target.

### Motion selection

`mamp_motion` implements the official code's mean absolute displacement across
corresponding offsets in consecutive temporal patches. Replicate padding uses
the second patch score for the first. The log weight is

\[
a_{bj}=\frac{I_{bj}}{\max_{b'j'} I_{b'j'}\,\tau+10^{-10}}.
\]

Gumbel top-k samples without replacement. A temperature of 0.8 is prespecified.
Missing transitions contribute only when both endpoints are observed; the
maximum is computed over eligible tokens. Zero usable motion gives uniform
weights. This validity extension and the 33-landmark input are adaptations.
The sampler has no label argument. It predicts latent features, not MAMP's
coordinate-motion targets.

`robust_motion` reuses Notebook 11's implementation unchanged. It takes median
Euclidean transition speeds within a token, requires at least two valid
transitions, and clips positive token scores at their 95th percentile. With
scores `s` and `N` valid tokens, the initial draw weights are

\[
p_i=\frac{1-\beta}{N}+\beta\frac{s_i}{\sum_j s_j},\qquad \beta=0.75.
\]

Stationary or unusable motion falls back to uniform sampling. The uniform
component retains slow-region coverage. These weights are not final inclusion
probabilities under sampling without replacement. Neither rule guarantees that
both sides are hidden in each draw. The median and clipping are proposed
safeguards, not a demonstrated improvement over MAMP.

All training inputs use the existing resized sequence. MAMP scores are
displacements per prepared token; robust scores are per prepared step. Neither
is measured in metres per second. Original timestamps would be required for
a physical-speed comparison. MAMP's absolute-axis sum is also sensitive to
general rotations; the Euclidean score has different geometric behavior.

### Structured selection

- **Connected regions:** six landmarks connected through the declared anatomical
  graph, hidden for half the prepared window. Region definitions collectively
  cover all 33 landmarks. Uniform selection over the fixed region bank is not
  uniform landmark inclusion, and the graph includes declared head/body links.
- **Whole trajectories:** three landmarks hidden at every valid token throughout
  the window. A trajectory is never shortened to satisfy a budget.
- **Interior completion:** all valid landmarks hidden in a contiguous interior
  interval occupying one quarter of the token grid, with observations retained
  before and after it. This remains a completion task.

Missing cells within a nominal region remain missing. Each structure determines
its realized target count per clip; its scattered reference hides that exact
count. A six-landmark, eight-block region has 48 targets when fully observed.
Three full trajectories also have 48 on this grid, but their missingness can
produce different realized counts. Separate comparisons avoid conflating these
cases. Whole-frame intervals have counts in multiples of 33 and full trajectories
in multiples of 16 when fully observed; no positive proper common count exists.

The context audit records how often a hidden token retains both immediately
adjacent temporal observations or a graph neighbor at the same time. It
describes available information. It cannot prove that a contextualized-feature
predictor uses coordinate interpolation.

## Training controls and their practical limits

The new runner imports the existing model, source sampler, objective primitives
and checksum-validated atomic artifact format. It lives in new modules; the
old training and masking code is unchanged. All arms use identical initial
encoder/predictor/teacher and projector parameters, source schedules, geometric
views and optimizer exposure. Separate mask RNGs depend on the arm, step, fold
and seed. Geometric views are generated once on CPU and shared across arms.

The dense prediction path keeps clip identities even when target counts vary.
It computes the mean target cross-entropy within each clip and then averages
clips. The target center uses the same per-clip reduction. The teacher receives
no gradient. Input masking zeros hidden patch embeddings before the online
encoder, retaining positional placeholders; it preserves the current model's
behavior rather than reproducing the official papers' visible-token pruning.
Fixed-mask perturbation tests verify direct hidden-content isolation through
the predictor. The full-input regularizer is retained equally in every arm.

All encoder inputs still contain 33 landmarks, regularization still pools the
twelve gait landmarks from unmasked views, and the endpoint still uses five
bilateral pairs. Changing any of those is a separate experiment. Ordinary
self-supervised mask selection can inspect the full permitted training clip;
mask locations can consequently communicate motion information. A fixed-mask
perturbation test does not rule out that selection channel.

The plan reads the tracked Notebook 08 recipe from
`docs/figures/tutorial_masking_summary.json`: 1,200 updates, batch 20, width 96,
encoder/predictor depths 4/2, four heads, AdamW learning rate 0.001 and weight
decay 0.05, EMA 0.999, regularizer weight 0.05. Other fixed settings are betas
(0.9, 0.95), constant schedules, center momentum 0.9, predictor/teacher
temperatures 0.10/0.06, gradient norm limit 1, rotation up to eight degrees and
translation up to 0.03. These settings do not come from the synthetic examples.

The primary motion and region experiments contain 125 encoders and 150,000
updates across five folds and five seeds. Full trajectories and completion are
optional follow-ups. No full grid starts by default. Equal updates do not imply
equal cost; the runner retains elapsed time for each paired job. Completed
compatible jobs can be reused. Interrupted jobs restart from their declared
seed; optimizer resume is not implemented in this extension.

## Evaluation and decision rules

1. **Use existing encoders first when their artifacts are available.** A result
   loaded through `comparative_training.load_comparison` can be passed to
   `motion_readout.evaluate_motion_readouts`, avoiding another pretraining run.
   Notebook 18 also provides `evaluate_retained_comparison`: it recomputes the
   expected identity from current data, code and runtime before loading a saved
   job. Set `LATERALITY_RETAINED_COMPARISON` to a compatible Notebook 12 result
   directory to enable this path. New readouts are saved under
   `artifacts/motion_structured/retained_readouts`, preserving the old files.
2. **Check the readout's sensitivity.** The positive control contains zero-mean
   oscillations with known bilateral amplitude differences. Mean-plus-motion
   summaries must recover this synthetic signal on excluded sources. This test
   does not require any mask to improve real-data performance.
3. **Report two declared summaries for every encoder.** Mean-plus-motion adds
   bilateral standard deviations, absolute consecutive increments and support
   fractions to the original mean summaries. Initial and trained encoders receive
   identical summaries. Extra dimensions change readout capacity and must be
   accounted for through the matched initial control.
4. **Choose ridge penalties on training sources.** The grid spans 0.01 to 10,000;
   boundary selections are reported. This selects only the readout. Selecting
   an entire pretraining recipe requires separately excluding inner validation
   sources from encoder training. No outer-test result selects a mask, summary
   or checkpoint. Final checkpoints are the real runner's declared evaluation;
   earlier saved checkpoints support a separately declared analysis.
5. **Separate prediction tasks.** Online and teacher readouts, initial features,
   direct pose and the training mean test useful movement information. A fixed
   evaluation-mask bank, own-teacher errors and mismatched targets test JEPA
   prediction. Own-teacher loss does not provide a common feature scale.
6. **Preserve source identity and coverage.** Pool outer predictions per seed
   before scoring, weight each video equally, and report seed variation. Strict
   aggregation rejects duplicate predictions, missing declared cells and source
   overlap between folds. Source-bootstrap contrasts retain paired seeds and
   conditions and condition on fitted models. They exclude retraining uncertainty.

Improvement shared by initial and trained encoders implicates the readout.
A repeatable learned-over-initial benefit revealed by temporal summaries would
support the pooling explanation. Failure of both calls for investigating
preparation timing and target design before adding complex masking mixtures.
This already inspected cohort supplies development evidence; independent
participant and outcome validation remain necessary for stronger conclusions.

For raw missing-data claims, remove observations before preparation using
Notebook 13's boundary. For future prediction, use Notebook 14's prefix-only
preparation, fixed horizons, observed-future decoder check and simple forecasting
controls. The new samplers do not turn interior completion into forecasting.

## Running the extension

From the repository root, using its Python environment:

```text
python neurips-laterality/scripts/build_research_notebooks.py --only 15 16 17 18
python -m unittest discover -s neurips-laterality/tests -p test_motion*.py -v
python neurips-laterality/scripts/verify_motion_notebooks.py --execute
python neurips-laterality/scripts/verify_motion_notebooks.py --execute --data-mode gavd
```

Open notebooks 15, 16, 17 and 18 in order. Source notebooks remain output-free;
verification retains executed copies and vector figures separately under
`executed/motion_structured`. The verifier forces real training off. Its default
mode explicitly selects generated software checks; `--data-mode gavd` exercises
real cohort/split loading and the full training-mask audits without launching
the real training grid.

All four notebooks default to `DATA_MODE="gavd"` and prepare/reuse the real
cohort and splits. In Notebook 17, set `RUN_TRAINING=True` in the configuration
cell, or use `LATERALITY_RESEARCH_RUN_REAL=1` (study alias:
`LATERALITY_MOTION_RUN_REAL=1`). `LATERALITY_MOTION_EXPERIMENTS` defaults to
`motion,regions`; `LATERALITY_DEVICE` selects the device. Change the visible
`FOLDS`/`SEEDS` declaration for a pilot and keep the same configuration in 18.
Inspect the workload before enabling `run_gavd_grid`. Notebook 18 uses
`collect_gavd_grid`, which cannot start encoder training.

Real study artifacts are isolated under `artifacts/motion_structured`, with
content-checked training/evaluation jobs and complete grid indexes. The
low-level `plan_mask_study` remains available without raw data for workload
inspection. The notebooks' default path validates real inputs first and fails
clearly if the registered inventory cannot be recovered. No old notebook,
protocol, result or checkpoint is rewritten.
