# Architecture and objective tutorial audit — September 25, 2026

This review traces the current source implementation and the completed source
configurations. Changes are confined to `notebooks/gait_fidelity/lesson_model.py`
and `lesson_response.py`; model, training, evaluation and retained evidence files
are unchanged.

## Source correspondence

The body-12 architecture in `synthetic_training_v2/models.py` receives estimated
coordinates, confidence, observed flags and physical timestamps. Four consecutive
frames of one joint form a token. The saved source model maps
`[B,128,12,5] → [B,384,20] → [B,384,96]`, uses four encoder Transformer layers,
and uses two further layers for the JEPA feature predictor. Missing tokens remain
output queries. Neither reference support nor reference coordinates enter the
deployment encoder. The readout emits four two-coordinate corrections per token,
which are added to usable observations after reshaping. Missing observations use
a zero normalized addition base.

The original tutorials already explicitly derived channel packing, positions,
self-attention, residual decoding, coordinate loss, centered feature
cross-entropy, the translated-view regularizer, and teacher/center updates. Those
calculations were retained. Notebook 03 now separately exposes the feature
predictor's Transformer → LayerNorm → linear path and checks both its values and
its input gradients against the production predictor.

All three completed source configurations use `sampling='person_motion'`.
`gait_fidelity/training.py:hierarchical_pair_groups` and
`draw_hierarchical_pairs` sample a person, raw motion, source window and paired
condition/intervention in four uniform conditional draws, with replacement.
Notebook 04 now reconstructs that hierarchy, prints exact pair probabilities,
verifies equal person probabilities, and compares seeded draws with production.
The previous blanket description of closed permutation cycles is restricted to
the optional `matched_cycles` full-protocol/re-pairing controls. These controls
were not fitted in walking-core-01. The `repaired_change` objective denotes
re-paired endpoints and differs from the subsequent readout-repair study.

## Response extension

The paired auxiliary requires a queried token and all reference frames at both
endpoints; the base JEPA loss requires any valid reference frame at each queried
endpoint token. The revised F tutorial exposes both masks without substituting
one for the other. An unsupported auxiliary returns a graph-connected zero and
reports zero supported pairs; its missing support cannot be presented as a
successful zero-error result. Base supervision remains available independently.

F now reconstructs a complete small-model forward pass with explicit
cross-entropy, translated-view projector/regularizer, centered residuals,
paired/endpoint auxiliaries, and combined objective. Production calls serve as
parity checks after the calculations. Forward values and every trainable
parameter gradient are compared. The coordinate-difference example includes
invalid-reference NaN placeholders, masks them before arithmetic, and compares
both values and gradients with production.

Initialization calibration uses all trainable parameters, with unused gradients
counted as zero. Squared gradient norms are summed across 32 training batches;
their RMS is the square root of this sum divided by batch count and parameter
count. One shared JEPA coefficient scales the larger initial auxiliary RMS to
10% of base RMS. It does **not** independently set both auxiliaries to 10%.
F demonstrates the reduction on one generated batch, labels that demonstration,
and reconstructs the actual saved coefficients separately when the compact
receipt is available.

The final generated-batch step exposes the production order: combine losses,
backpropagate once, clip the combined gradient, update the student with AdamW,
update the teacher by moving average, then update the center from pre-update
teacher tokens with equal supported-endpoint weight. It verifies absence of
teacher/readout gradients during feature pretraining and the exact moving-average
and center equations.

The result reader uses canonical `summary.json`, `comparisons.json` and
person-level condition tables. It distinguishes an explicitly selected source
or fixture run from downloaded compact evidence. An incomplete selected response
run is not silently replaced by another run's completed results. Probe-fold
wording now distinguishes holding people out of the ridge fit from holding them
out of encoder pretraining. Similar estimates with broad intervals are described
as unresolved rather than equivalent.

## Validation and remaining scope

- Executed all nine architecture and eleven training calculation cells against
  the actual walking-core configuration and plan. The launcher and scheduler
  status cells were deliberately omitted from this read-only calculation check.
- Executed all eleven response code cells, including complete-loss/all-parameter
  gradient parity, unsupported auxiliary handling, the scratch optimization
  step, and the compact evidence readers.
- Confirmed source-sized encoder/output shapes `(1,384,96)` and `(1,128,12,2)`.
- Confirmed the generated hierarchy draws agree exactly with production and
  its person probabilities sum to one third for each of three example people.
- Retained calibration reconstruction gives the same source coefficients:
  shared JEPA `0.012646811176583523`, coordinate `3.79389230231159`.
- All calculations use fresh CPU teaching models. No retained checkpoints,
  source training arrays, HAIC processes or result tables are mutated.

The new readout-repair lesson, notebook build/integration, full kernel execution
and independent adversarial review are coordinated separately by the parent
task. Scratch examples test implementation correspondence; they do not test
source training convergence or reproduce the scientific evaluation population.
