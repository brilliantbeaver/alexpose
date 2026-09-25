# Readout-repair implementation review

The implemented experiment tests whether dense temporal supervision improves
waveform fidelity beyond reducing the original scalar response loss weight.
It reuses completed encoders, trains twelve matched readouts, evaluates a fixed
27-fit comparison set, and prepares a smaller confirmation panel while retaining
every planned original-test person. The launch guide is
[START_REPAIR_03.md](../../../../slurm/gait-fidelity/START_REPAIR_03.md).

The training, cohort/preparation, and evaluation modules were implemented by
separate agents. Cross-review then examined code written by other agents: a
methods reviewer challenged training and confirmation admission, the training
implementer reviewed orchestration and legacy checkpoint compatibility, and the
methods reviewer independently tested admission accounting. These are software
and methods reviews, not evidence that the scientific intervention succeeds.

## Material findings and resolutions

| Finding | Resolution and verification |
| --- | --- |
| A same-seed scalar checkpoint could be called dense in the confirmation manifest. | Match the declared method to completed checkpoint phase, encoder, policy, representation and objective; independent adversarial regression. |
| A partial fit inventory could unlock source confirmation. | Require all fixed 27 method/seed fits before access; independent adversarial regression. |
| Preparation admission omitted the evaluation GPU reservation. | Include the final two GPU-hours in cost and reservation checks; independent near-budget regression. |
| A measured duration could pass while the full Slurm allocation could not fit. | Admit against both measured time and declared allocation limits, with queue/evaluation/grace allowances; independent near-cutoff regression. |
| Resume admission counted completed preparation twice. | Verify completed receipts, exclude completed work and count active reservations once; resume regression. |
| The CMU audit read an inventory that deliberately excluded CMU. | Read the raw inventory while preserving unresolved person identities and exclusion from the experiment; audit regression. |
| An unreceipted prediction or merged directory could prevent recovery. | Preserve unpublished artifacts and regenerate from immutable inputs; completed publication is verified before reuse. |
| A mutable admission file could invalidate completed evaluation. | Bind immutable result, prediction, lock and merge artifacts; leave repeated operational admission receipts outside completion hashes. |
| Rendering could continue after the controller disappeared. | Supervise each GPU worker's process group against the absolute cutoff, using the existing tested supervisor. |
| Direct confirmation evaluation could bypass worker accounting. | Source CLI requires `launch --stage confirmation`; direct CLI evaluation remains development-only; regression. |
| A reservation in an additional discovered ledger could be overwritten by a new ledger. | Treat explicit reservations and non-test roles as conflicts across discovered ledgers; regression. |
| Strict attribution expected a string `none` for the direct model's policy. | Preserve the actual legacy `None` checkpoint representation; regression using a real trained direct checkpoint. |

## Scientific boundaries retained

- The clean primary comparison is delta/dense versus delta/scalar-low, with
  identical coordinate and geometry terms, initialization, sampling, updates,
  optimizer and frozen encoder. Original coordinate-only base is a practical
  comparator; it has no geometry penalty.
- Calibration uses only training data. Dense matching applies to initial
  auxiliary gradient RMS before clipping, not every optimizer update. Logs
  expose clipping and component losses. The contrast changes temporal
  information as well as gradient support, so it cannot uniquely identify
  percentile-gradient sparsity as the cause of earlier harm.
- Person-level uncertainty averages the three fixed seeds within each person.
  Secondary comparisons and crossed bootstrap intervals are descriptive. The
  14 candidate confirmation people are not multiplied by windows, renderings,
  estimators or seeds. They can support a large-effect test; small advantages
  remain poorly powered unless the measured paired variance is very low.
- Response and waveform failure penalties are decomposed at the same
  hierarchical weights as the primary errors. Failed predictions are retained.
  A response interval containing zero is not evidence of noninferiority.
- Confirmation requires actual reviewed exposure evidence and freezes all
  methods before access. Missing references and failed reference QC are
  reported without replacement. Development and confirmation are never pooled.
- GAVD and CMU receive readiness audits, not invented participant counts or
  new unreviewed analyses. GAVD lacks paired dense kinematic truth; CMU folder
  aliases do not establish independent people.

## Execution evidence

The final suite passed **217 tests**. The retained end-to-end fixture completed
in **182.55 seconds**, with 30 core fits, 18 response fits, 12 repair fits, six
calibrations, and two complete 27-fit evaluations. Its three generated
confirmation identities and six windows were fully retained. Three integration
assertions also verified identity separation and idempotent artifact validation.
Shell syntax, Python compilation, CLI help and immutable-release packaging
checks passed. These times describe the tiny CPU fixture, not HAIC throughput.

The machine-readable validation record is
[`../records/repair-validation-20260924.json`](../records/repair-validation-20260924.json).
The retained software fixture lives outside the scientific results at
`outputs/gait-fidelity-repair-validation-20260924`.

HAIC execution is not implied by local validation. Noninteractive SSH from this
environment returned `Permission denied (keyboard-interactive)`. No source jobs
were submitted here. Source throughput, queue timing, source QC retention and
the actual confirmation exposure ledger remain external checks. The saved core
configuration points to
`outputs/synthetic-training-v2/full-01/inputs/person-reservations.csv` on HAIC;
the new audit checks that location without assuming its entries are reviewed.
