# Data and teaching-flow review — September 25, 2026

Scope: notebook source cells in `lesson_data.py`, `lesson_start.py`, and
`lesson_experiments.py`. The review traced preparation, data validation,
normalization, masking, pair formation, training sampling, and reference donor
selection without changing scientific source code or retained evidence.

## Reference notebooks and teaching approach

The multiple-sclerosis notebooks introduce the task and vocabulary before
describing the computation. Notebook 03 uses a shape table to distinguish
frames, landmarks, tokens, features, and batch examples; its split discussion
distinguishes clips from source videos and warns that recording IDs are not
verified person IDs. Its training explanation separately identifies student
inputs, teacher inputs, loss support, optimization, and teacher averaging.
Notebook 02 places the token and mask diagrams beside the quantities they
describe. These are useful teaching patterns for gait fidelity.

Their numerical choices do not transfer to this study. Gait fidelity uses
body-12 joints, five channels, input-only window normalization, and a paired
reference-restoration task. Its masks have exact budgets and its latest
training sampler gives people, raw motions, and windows equal probability at
successive levels. Copying the reference study's 33-landmark normalization,
clinical-region bias, approximate mask fraction, or source-video sampler
would misstate the implemented experiment. Several model operations in the
reference notebooks themselves use shared helpers; the gait tutorials retain
their existing more explicit source-parity calculations.

## Substantive additions

1. Notebook 01 reconstructs population and factorial counts from the loaded
   bundle. It separately reports people, raw motions, source windows, actual
   condition records, and per-window grid completeness. The fixture does not
   silently inherit the human study's 55,800-record count.
2. Split assertions visibly check canonical people, raw motion hashes, and
   source families. Training must retain its original training split and
   exclude the held intervention and held estimator. Original test identities
   are reserved for the separate locked confirmation loader.
3. Pair formation is written out as a metadata-only operation and compared
   against `training.paired_indices`. The table displays no-change pairs and
   makes repeated baseline exposure explicit. Notebook 04 owns the subsequent
   hierarchical sampling derivation, avoiding a second implementation here.
4. The three naming conditions are explicit NumPy transformations. Coordinates,
   confidence, and availability move together; timestamps and references remain
   fixed. All conditions are compared with `data.apply_naming`, and applying
   each twice recovers the inputs. The text distinguishes naming errors from
   physical mirroring before rendering.
5. Notebook 00 connects walking core, JEPA response, and readout repair while
   marking omitted full-matrix controls as implemented but unfitted. Experiment
   E now distinguishes endpoint re-pairing from the completed readout repair,
   and identifies closed-cycle batching as the `matched_cycles` protocol.
6. Experiment A's aggregation description now includes raw-motion averaging
   between windows and people; it no longer suggests that all windows receive
   equal weight regardless of their parent motion.

## Source correspondence and retained boundaries

- `preparation.prepare` generates movement geometry before physical mirrors,
  freezes the camera across the paired states, extracts estimated coordinates,
  and only then applies naming interventions. The no-change state reuses the
  complete baseline input and target arrays.
- `data.validate_bundle` and `synthetic_training_v2.contracts.validate_records`
  establish the identity, time, input/target, and held-condition boundaries.
  Notebook assertions expose selected checks and do not replace the validator.
- `training.paired_indices` forms only training pairs. Its order is part of the
  seeded sampler's behavior, so the visible implementation preserves it exactly.
- `training.hierarchical_pair_groups` and `draw_hierarchical_pairs` implement
  the latest `person_motion` sampler. `paired_batch_cycles` and
  `draw_pair_batch` remain relevant to the extended re-pairing control, not to
  the completed core, response, and repair source runs.
- Existing normalization, mask sampling, token ordering, donor matching,
  projection, and fixed-filter calculations were preserved. The observed,
  artificially hidden, reference-valid, and synthetically visible masks retain
  different roles.

No reference geometry, identity, intervention label, or evaluation scale is
added to the inference input dictionary. Additional views and intervention
records provide condition coverage, not independent participants. The local
source report is walking core; compact `outputs/iclr` evidence additionally
contains the later response and readout-repair results.

## Validation

All code cells emitted by the three owned lesson sources compile. Eight
selected data/math cells executed directly against the production analytic
fixture with 16 frames, three generated people, and two windows per person
(576 records), without writing checkpoints or changing saved source data.
The 144-row training pair table matched production bit for bit. Correct,
whole-window-swap, and temporary-swap transformations matched production and
passed their double-swap checks. The existing projection, inverse
normalization, hidden-coordinate perturbation, and empty-context fallback
checks also passed. These checks establish code correspondence and execution;
the generated fixture contributes no scientific result.

Experiment E's two code cells also executed with the actual latest core plan
and an empty result ledger. Its illustrative cycle checks passed, and omitted
group-L recipes produced an explicitly labeled empty receipt table without
attempting an unfitted comparison.

The parent revision runs the generated notebooks and conducts the independent
adversarial review after integrating all lesson changes. Source-study arrays
are not available locally, so the new array-level demonstrations are validated
with fixture data; retained result interpretation uses the downloaded evidence.
