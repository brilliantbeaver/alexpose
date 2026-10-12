# Independent mathematical review of the revised tutorials

September 25, 2026. This review was performed by an agent that did not author
the new repair lesson or the evaluation and walkthrough lessons reviewed here.
The reviewer read the production operations and executed the emitted teaching
cells independently. No production code, notebook source under review, retained
evidence, checkpoints, or large generated artifacts were modified.

## Findings resolved during review

1. In `lesson_repair.py`, step 4 describes a mean of squared scalar errors but
   placed the square outside the bracket following `mean_i` in its displayed
   equation. The parent moved the square inside the averaging brackets to
   avoid reading it as the square of a mean. The executable calculation already
   implemented the correct mean of per-pair squared errors.
2. In `lesson_evaluation.py`, the prose introducing the final person/seed plot
   described averaging over supported source families without naming the
   raw-motion level. The parent corrected it to conditions within windows,
   windows within raw motions, and motions within people. The computed
   per-person results already followed the production hierarchy.

Both corrections were inspected in the final sources. The parent also changed
the scratch initialization to seed only the CPU default generator, preserving
CUDA random state, and added a source-study-kind/non-fixture guard before
replaying source calibration receipts. No numerical algorithm mismatch or
unresolved review finding remains in the reviewed new calculations.

## Repair loss and optimization checks

The source trace covered `repair_objectives.repair_measurement_terms`,
`angle_gradient_support`, `repair_training._forward`, `_gradient_audit`,
`calibrate_repair`, and `train_repair`, including inherited input-only
normalization and the coordinate readout.

The visible repair correctly:

- Forms the common reference frame mask across both endpoints and both legs
  before checking predictions. Missing input observations and missing
  references remain separate.
- Applies the saved two-pixel segment threshold, 16-frame minimum, and 80%
  coverage requirement. The third constructed pair is excluded from angle
  supervision while supported coordinates remain in coordinate supervision.
- Computes the angle using the guarded absolute-cross-product `atan2` formula,
  linear percentiles, right-minus-left excursion, and state-b-minus-state-a
  response. The dense objective concerns differences between states at matched
  times, not temporal velocity.
- Reduces frames and legs within each eligible pair before averaging pairs.
  The geometry penalty uses the same reference support. A collapsed prediction
  cannot delete the troublesome frames.
- Masks invalid target coordinates before subtraction. The coordinate loss's
  squared normalized units are distinguished from evaluation NLE.
- Counts every readout parameter, including unused zero-gradient parameters,
  in calibration. It sums squared gradient energies across batches before
  taking the square root. The dense calibration target is the downweighted
  scalar gradient strength, rather than the coordinate gradient strength.
- Replays matched initial readouts and endpoint batches for separate optimizers,
  preserves the encoder and teacher, applies the inherited warmup/cosine
  schedule and combined-gradient clipping, and deploys on the four allowed
  observation fields without references.

All emitted G code cells executed independently, including the two optimizer
updates for each repair arm and the six retained calibration reconstructions.
The full G cell sequence passed again after the CPU-random-state and source
receipt-guard changes.
Additional adversarial checks compared values and coordinate gradients in 12
randomized prediction cases, including exact limb collapse and unequal
reference support. Replacing every invalid-reference placeholder with a large
finite coordinate left support and losses unchanged. Six percentile cases with
tied order statistics and different admitted-frame counts matched production
values and gradients exactly. The empty-angle-support case retains a connected
zero, while the full trainer's rejection of an unsupported batch is explained.

The teaching implementation remains a CPU illustration using a fresh random
encoder. These checks do not establish convergence, clinical validity, or HAIC
numerical equivalence under different hardware precision.

## Evaluation, probes, and uncertainty

The new response accounting cells executed on a generated production-fixture
family and their constructed denominator edge cases. Direction eligibility uses
the measured reference change and keeps failed predictions in the eligible
denominator. Successful and failed contributions retain the original population
and reconstruct the saved scoring rule; the conditional successful mean is
identified as a different denominator.

The visible ridge regression agrees with `response_diagnostics.person_ridge_probe`:
the complete person remains in one fold, standardization and target centering
use only fitting-fold rows, the dual system includes the mean-loss factor
`n * alpha`, and final errors give people equal weight. The constant-feature
case executes. Both the source predictions and an independent small primal
solve agree with the visible dual solution. The explanation correctly limits
this to the encoder's training population and separates teacher-input reference
features from deployed observed-input features.

The repair interval code agrees with `repair_evaluation.paired_comparison` and
uses ViTPose for the declared primary contrast. It averages the three matched
seed effects within each person before applying the Student-t interval. The
downloaded result reconstructs to improvement **0.2788559691 degrees**, with
primary interval **[-0.1949135069, 0.7526254451]**. An incomplete person/seed
method grid was explicitly rejected during this review. The text retains the
development-reuse, conditional-on-fitted-seeds, and non-equivalence limitations.

## Completed-study walkthrough

All 15 code cells in notebook 07's emitted source executed independently with
figure export disabled. The checks verified all 69 selected evidence-file
hashes, shared participants and seeds, retained comparator equality, additive
failure accounting, the crossed bootstrap intervals, and the repair's primary
Student-t interval. Eleven figures were assembled in memory. This execution
checks computations and figure construction; rendered layout review remains a
separate integration check.

The distinction between measured figures and constructed illustrations is
preserved. The scalar-shift illustration makes a measurement property concrete
without attributing that specific failure to fitted models. The readout repair's
calibration and figure captions preserve the scalar-gradient matching target
and its limitation to initialization. No new scientific claim is inferred from
successful tutorial execution.
