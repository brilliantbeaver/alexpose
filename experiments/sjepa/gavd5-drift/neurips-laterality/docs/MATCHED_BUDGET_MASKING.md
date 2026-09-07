# Matched-budget masking: parameter rationale

This reference explains the optional real-data experiment in Notebook 08. The
experiment is exploratory and does not change the completed protocol or its
registered results. Its purpose is to compare gait-target and uniform-target
masking while holding the realized hidden-token count and the main training
budget fixed.

## Why the experiment uses 1,200 optimizer updates

There is no mathematical or empirical result in this project showing that an
encoder requires exactly 1,200 updates. The value is used as a matched-compute
reference to the completed paper training in Notebook 03.

The registered paper profile trained every encoder for 300 epochs with a batch
size of 20. Each outer-training fold contains 74 or 75 source videos. The
source-balanced sampler therefore uses four updates per epoch:

$$
\left\lceil\frac{75\text{ sources}}{20\text{ positions per batch}}\right\rceil
=4\text{ updates per epoch}.
$$

Consequently,

$$
300\text{ epochs}\times4\text{ updates per epoch}
=1{,}200\text{ optimizer updates per encoder}.
$$

Notebook 08 specifies optimizer updates directly rather than using an epoch
argument. Giving both masking recipes 1,200 updates prevents one recipe from
receiving more optimization simply because it is the proposed method. Matching
the earlier per-encoder budget also makes the computational scale easier to
compare with the completed experiment.

This is a control, not a convergence guarantee. A different masking objective
can learn at a different rate even when it receives the same number of updates.

## Compute accounting

The enabled real-data experiment defaults to the full design: five outer folds,
five seeds, two masking arms, and 1,200 updates per encoder. This gives 50
encoder jobs and 60,000 optimizer updates. A progress display may show 2,400
steps for one fold/seed comparison because that comparison trains the two arms
in sequence:

$$
2\text{ masking arms}\times1{,}200\text{ updates per encoder}
=2{,}400\text{ updates per paired comparison}.
$$

The 2,400 value is therefore not an increase in the per-encoder budget. The
real-data experiment remains explicitly enabled by the user; within that
experiment, the full five-fold, five-seed grid is the default scientific scope.

| Quantity | Notebook 03 paper training | Notebook 08 optional grid |
|---|---:|---:|
| Source-held-out folds | 5 | 5, inherited unchanged |
| Seeds per fold | 5 | 5, using 42–46 |
| Training recipes | 2 | 2 |
| Encoder jobs | 50 | 50, newly trained unless an exact completed comparison is validated and reused |
| Updates per encoder | 1,200 | 1,200 |
| Total optimizer updates | 60,000 | 60,000 |
| Batch size | 20 | 20 |
| Sampled sequence positions per encoder | 24,000 | 24,000 |
| Sampled sequence positions over the grid | 1,200,000 | 1,200,000 |

The sampled-position counts describe repeated optimizer exposure. They do not
increase the cohort beyond its accepted sequences or create additional
independent observations. The source-balanced sampler first balances source
videos, then selects sequences within them.

## Why the original run appeared much slower

The first version of the optional runner used the `LearningSettings` CPU
default. It did not consult Notebook 03's device resolver, even on a machine
where Apple Metal Performance Shaders (MPS) were available. The introductory
synthetic example also set PyTorch to one CPU thread because that setting works
well for its tiny batches. Unless it was changed later, the much larger real
experiment inherited the same one-thread setting.

The comparison with a recent execution of Notebook 03 was also misleading.
That execution validated and reused all 50 existing checkpoints; it did not
train 50 encoders again. Notebook 08, by contrast, initially created a fresh
random output directory for every completed comparison and had no path for
finding and reusing it on the next run. An interruption could consequently
waste a finished arm or cause completed work to be repeated.

The audited early CPU timing projected approximately 8.5–10.8 hours for the
complete pre-acceleration grid. Historical Notebook 03 checkpoints trained on
MPS had a median recorded time of about 221 seconds per 1,200-update encoder,
which corresponds to about 3.1 hours for 50 freshly trained encoders at that
median rate. Both figures are approximate, hardware- and load-dependent
references. The CPU value is a projection rather than a completed full-grid
measurement, and the MPS value comes from the earlier Notebook 03 training
implementation. Neither is a post-change benchmark or a promise about Notebook
08 wall time. Thermal throttling, other processes, compilation warm-up, and
suspending the computer can all change the observed duration.

After the acceleration changes, a local MPS check ran 50 updates in each arm,
or 100 full-size encoder updates in total. The measured training portion was
about 11.2 seconds, approximately 0.112 seconds per update. Linear extrapolation
gives about 1.9 hours for 60,000 uncached updates. This is a short implementation
check rather than a completed-grid measurement: it excludes some fixed setup,
evaluation, validation, and saving costs, and a multi-hour run may slow under
thermal or competing-system load. It is evidence that the accelerated path is
active, not a guaranteed completion time.

## Acceleration policy without changing the comparison

The acceleration work changes execution, not the declared experiment. Device
selection in automatic mode prefers an available accelerator—MPS on supported
Apple systems or CUDA on supported NVIDIA systems—and uses CPU only when no
supported accelerator is available. If the user explicitly requests MPS or
CUDA and that backend is unavailable, setup stops with a clear error instead
of silently moving the experiment to CPU. This distinction prevents a long CPU
run from beginning under the false impression that an accelerator is active.
The selected device is shown in progress and retained with the result.

CPU execution remains supported. Its thread count is configurable for the
machine and workload rather than being forced to inherit the one-thread choice
used by the small synthetic demonstration. The selected count should also be
recorded with the run. Thread-level parallelism changes how numerical kernels
are scheduled; it does not change the source rows, masks, update count, or loss
being optimized.

Two implementation changes remove repeated transfers and dispatch overhead:

- The fixed fold arrays can remain resident on the selected device while the
  sampler continues to index exactly the predeclared source-balanced rows. Data
  residency is a storage decision. It does not make an outer-test source an
  input to representation training, and the source-separation assertions remain
  in force.
- The two views used by the VICReg term can be concatenated along the batch
  dimension for their independent random transformations and encoder pass, then
  split back into their two original groups before pooling and loss calculation.
  Transformer attention remains within each sequence; examples do not attend
  across the batch. The operation therefore draws the same augmentation
  distribution, computes the same two-view objective, and preserves the
  masked-prediction-plus-VICReg recipe while reducing dispatch overhead.

A completed fold/seed comparison is reusable only after its saved lineage and
comparison contract are validated. Validation covers the cohort and source
split, settings, masking arms, implementation identity, paired initialization,
source draws, and realized hidden-token counts. An absent, incomplete, or
incompatible result does not count as reused. Saving and validating at this
granularity makes a resumed full grid substantially cheaper without treating
stale work as current evidence.

These changes do not reduce the 1,200 updates, omit a fold, select a favorable
seed, or weaken the matched-arm checks. Any observed numerical differences
between CPU, MPS, and CUDA still need to be understood as backend-dependent
floating-point behavior, not as a change in the scientific design.

## The smaller run is a conditional pilot

The acceleration policy does not silently replace the full grid with fewer
seeds. Setting `LATERALITY_RESEARCH_MASKING_SCOPE=single_seed_pilot` retains all
five outer folds and both masking arms but runs only the predeclared seed 42 at
1,200 updates. That scope trains 10 encoders for 12,000 total updates, one fifth
of the full grid. Retaining every fold still gives each accepted source one
held-out prediction under that seed. The notebook labels this mode when it
starts and when it completes; `full` remains the default.

Such a result would estimate performance conditional on seed 42 and the stated
training recipe. It would not estimate the mean over seeds 42–46, show that the
direction is stable across initializations, or replace the full-grid result.
Choosing whether to run the other seeds after looking at the seed-42 outer-test
scores would remain an exploratory decision, not an independent confirmation.
The enabled real-data default remains the full 50-encoder design.

## Rationale for each visible setting

| Parameter | Optional Notebook 08 value | Rationale and limitation |
|---|---:|---|
| `REAL_FOLDS` | 0–4 | Preserve the registered source-video test assignments. Redrawing folds after seeing earlier results would create a different development analysis. |
| `REAL_SEEDS` | 42–46 | Match the registered initialization count and make training-seed variation visible. Five seeds do not create five independent datasets. |
| Recipes | Gait targets and uniform targets | Isolate target-location selection. Both arms use the same realized number of hidden valid tokens. |
| `REAL_STEPS` | 1,200 | Match the registered per-encoder update count. This is not a demonstrated optimum. |
| `batch_size` | 20 | Match the registered paper batch size and its source-exposure scale. |
| `embed_dim` | 96 | Match the registered paper encoder width. |
| `encoder_depth` | 4 | Match the registered paper context-encoder depth. |
| `predictor_depth` | 2 | Match the registered paper predictor depth. |
| `heads` | 4 | Match the registered paper attention-head count. |
| `learning_rate` | 0.001 | Reuse the registered initial learning rate. Notebook 08 holds it constant rather than applying Notebook 03's cosine decay. |
| `weight_decay` | 0.05 | Match the registered paper regularization value. |
| `vicreg_weight` | 0.05 | Match the registered paper weight on the feature-variation regularizer. |
| `ema_momentum` | 0.999 | Reuse the registered starting teacher momentum. Notebook 08 keeps it fixed; Notebook 03 increases it toward 1.0. |
| `mask_fraction` | 0.5 | Use a simple, visible exploratory fraction. The common realized count, rather than the percentage alone, is what makes the two candidate sets comparable. Notebook 03 used 0.6. |
| `ridge_alpha` | 1.0 | Avoid selecting the read-out penalty from held-out results in the tutorial. This fixed value is not known to be optimal. A final study should predeclare it or select it using training sources only. |

The architecture, source roles, initialization seeds, optimizer-update count,
batch size, and several regularization values are deliberately comparable with
Notebook 03. The masking policies, mask fraction, learning-rate schedule, and
teacher-momentum schedule are not identical. The experiment should therefore be
described as a matched-budget comparison, not as a reproduction of the earlier
training protocol.

## How to decide whether 1,200 updates are sufficient

The registered Notebook 03 curves show that most loss reduction occurred early,
while smaller improvement continued between epochs 200 and 300. That observation
supports 1,200 as a plausible full-run budget, but it cannot establish the right
budget for a changed masking objective.

A principled budget study would declare several checkpoints in advance, such as
400, 800, and 1,200 updates, and compare training-only diagnostics or an inner
validation outcome. The same stopping rule must apply to both masking recipes.
Outer-test sources must remain unavailable when the budget is selected. If the
loss and training-only outcome have stabilized well before 1,200 updates, a later
registered study could justify a smaller budget. If they are still changing, it
could justify studying a longer budget. Either decision would define a new
experiment and should not be made by repeatedly inspecting the current outer-test
scores.

## Interpretation boundary

Equal updates control one important source of unfairness, but they do not imply
equal wall-clock time, floating-point operations, optimization difficulty, or
convergence. The comparison should report elapsed time and, if efficiency is a
claim, a compute-matched analysis. A favorable held-out result would support the
narrow conclusion that the selected masking policy helped under this declared
training recipe. It would not show that 1,200 updates are universally necessary
or that the recipe is optimal for other datasets or tasks.
