# Gait-fidelity tutorial refresh, 25 September 2026

This revision connects the existing mathematical tutorials to the completed
walking-core, JEPA-response and readout-repair implementation. The source
notebooks contain their equations and Python calculations directly. Production
preparation, retained fitting and artifact publication remain canonical runner
operations rather than alternative tutorial implementations.

## Implementation and teaching changes

The investigation traced preparation, identity splits, condition construction,
pairing, normalization, masking, token packing, representation and prediction,
training losses and updates, inference, measurement, failure accounting and
uncertainty. The multiple-sclerosis notebooks informed the stepwise presentation
and explicit intermediate-array inspection; their model and experimental
assumptions were not imported into this study.

The tutorial sequence now contains fifteen notebooks and 141 code cells:
00–06, experiment lessons A–G, and the completed-evidence walkthrough 07.

| Area | Substantive addition or correction |
| --- | --- |
| Data | Explicit condition counts, identity boundaries, paired endpoints and joint naming permutations, with source equality checks. |
| Architecture | Predictor Transformer, normalization and projection path with output and input-gradient comparisons; original attention, packing and coordinate decoding derivations retained. |
| Core training | Current person→motion→window→pair sampling, exact sampling probabilities and seeded draw parity. Optional complete-cycle controls are identified as outside the completed core. |
| Response pretraining | Complete feature loss, translated-view regularizer and paired auxiliary computation; full gradient comparison, missing-support behavior, calibration and update order. |
| Readout repair | New G notebook derives angle geometry, percentile interpolation, common support, scalar/dense/geometry terms, all-head gradient-energy calibration and matched readout updates. Both model and AdamW states are compared with the production loss path. |
| Evaluation | Explicit direction eligibility and failure costs, response ridge probes and fold standardization, plus repair's participant-level t interval and incomplete-pairing rejection. |
| Experimental interpretation | Latest core/response/repair lineage, matched participants, estimator scope and calibration target connected to the saved evidence. |
| Execution | Invoking-interpreter kernels, core/full fixture selection, relocated links, source-directory protection, and progress/failure receipts that replace stale success. |

The [algorithm guide](../../../../notebooks/gait_fidelity/docs/ALGORITHM_GUIDE.md)
maps each calculation to its notebook and source comparison. F's feature-response
extension and G's frozen-readout repair are distinct from E's endpoint re-pairing
control. The main workflow rejects a follow-up session that has no core
preparation bundle and directs readers to the appropriate standalone lesson.

## Independent adversarial review

Three investigators split data/teaching, architecture/training and evaluation.
They then reviewed one another's changes and the new repair implementation.
The final integration and disagreements were resolved by the parent agent.

Concrete corrections from review include the actual response comparison schema,
the default held-estimator exclusion, scalar-loss mean-of-squares notation,
raw-motion aggregation, CPU-only random-state isolation, and the distinction
between probe folds and encoder-training participants. Dense calibration is
explicitly matched to the weighted low-scalar gradient, not the coordinate
gradient. Fresh readouts share initialization for matching seeds and are fitted
separately.

Numerical stress cases covered unequal reference support, collapsed predicted
limbs, invalid reference placeholders, tied percentiles, missing auxiliary
support and incomplete method/person/seed pairings. The runner was separately
fault-tested for initialization failure, a later notebook failure, stale pass
receipts, source hashes, protected paths and restoration of `JUPYTER_PATH`.

Detailed records:

- [Architecture and response audit](tutorial-architecture-20260925.md)
- [Data and teaching audit](tutorial-data-clarity-20260925.md)
- [Evaluation audit](tutorial-evaluation-20260925.md)
- [Independent mathematical review](tutorial-independent-math-review-20260925.md)
- [Independent flow review](tutorial-independent-flow-review-20260925.md)
- [Execution-runner fault review](tutorial-runner-20260925.md)

## Validation scope

The current source structure compiles all 141 code cells and requires cleared
outputs. Sixty-four targeted tests passed across tutorial workflow, measurements,
masking, response training/evaluation and repair training/evaluation. The complete
G calculation passed after the final optimizer-state comparison was added;
all eleven F cells passed after the final evidence-path correction.
Notebook 07 executed all fifteen cells with eleven figure outputs and no errors,
including verification of all 69 compact-evidence file hashes and reconstruction
of the retained primary comparisons. Its exported figures and G's gradient-support
illustration were visually checked. The response and repair session guards were
also exercised directly, before any tutorial cache could be created.

The integrated fresh-core fixture run is still awaiting sufficient local disk
space. Two attempts stopped with `OSError: [Errno 28] No space left on device`
during retained checkpoint/prediction writing. Their temporary fixture directories
were removed; no source study was restarted or changed. These failed attempts
are not reported as completed tutorial execution. Final execution status and
artifact hashes will be recorded here after the storage issue is resolved.
The [machine-readable validation record](../records/tutorial-refresh-20260925.json)
retains the current notebook hashes and the exact scope of completed checks.

A baseline inventory covers 139 production/source-evidence files, including the
downloaded research packet. Their hashes were unchanged after implementation and
independent review. Local generated examples establish numerical correspondence
and software execution; they do not reproduce HAIC/H100 training or provide new
scientific confirmation. No protected-test or GAVD results were introduced.
