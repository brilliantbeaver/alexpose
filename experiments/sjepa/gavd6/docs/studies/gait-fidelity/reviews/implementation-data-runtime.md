# Data preparation and runtime review

Reviewed on September 22, 2026. This review covers the new Gait Fidelity implementation, its inherited synthetic-training interfaces, and the boundary between software checks and source evidence.

The data implementation author tested the data and visualization components and independently reviewed the coordinator, configuration and profiling code written by another agent. The training implementation author independently identified the interval-length problem described below. This division provides an independent check of the orchestration and integration decisions; the data author's own unit tests are not represented as an external review of that author's code.

## Findings and corrections

| Finding | Consequence | Correction |
| --- | --- | --- |
| The inherited source windows begin at 5 and 8 seconds. Extending both from 64 to 128 frames makes them overlap. | Removing the second window leaves the shuffled-reference control without a different window from the same training person. | The second start moves deterministically to 10.12 seconds when needed, within the same retained recording. The original audited start remains in provenance. Every 64-frame part of each extended interval receives a new kinematic screen. Preparation and merging require at least two admitted training source families per person. |
| A renderer that refits the camera for every movement level changes both movement and observation geometry. | A measured response could partly reflect camera adjustment. | Camera pose is fitted once to the union of a source family's original, mirrored and edited geometry, then fixed across its movement levels. The image-space occluder is also fixed across those levels. |
| Renaming only coordinate channels leaves confidence or missingness attached to the wrong side. | Inputs become internally inconsistent. | Global and temporary naming interventions permute coordinates, native confidence and observation flags together. Anatomical reference arrays remain unchanged. |
| A coordinator can disappear between writing a merged bundle and updating its ledger. | A later launch could fail on the existing directory or reuse an unrelated merge. | The coordinator checks completed shard identities before adopting an existing merge. Dataset publication now writes and validates an unpublished staging directory before renaming it into place, so an interrupted archive write does not publish a partial dataset. |
| A completed-parent flag alone does not verify the checkpoint supplied to a dependent phase. | A modified upstream artifact could enter a nominally completed dependency. | Workers verify retained completion receipts and artifact hashes before loading dependencies. |
| Runtime profiling can accidentally become an outcome-based selection step. | Different methods could receive different training opportunities after their scores are inspected. | The profiling decision uses elapsed time and allocation bounds only. It selects the full or reduced update count for the complete matrix. Profile checkpoints are separate from final fits. |

The explicit no-change condition is a second endpoint with `movement_state="no_change"`, while the unique baseline has `movement_state="baseline"`. Both have zero intervention magnitude. Baseline selection must use the state or endpoint field, rather than assuming every zero-magnitude record is a distinct baseline. The no-change observations and references are exact copies and provide a deterministic zero-response check; they are not independent repeated measurements.

## Numerical and integration checks

The data test suite contains fourteen tests covering physical-time preservation, serialization and content hashes, confirmation-reference access, side-channel permutations, split leakage, missing-input preservation, reference consistency, physical reflection, geometry failures, extended nonoverlapping intervals, complete shard merging and portable viewer generation. An injected failure during target serialization leaves the public dataset path absent; retrying can publish a valid bundle without replacing an existing dataset.

A separate source-branch integration test uses the actual identity-registry and reservation joins, source hash checks, new locomotion screening, movement editing, fixed-camera fitting, `StudentSpec` construction, score-preserving extraction adapter, naming interventions and bundle merge. It replaces licensed body reconstruction and hardware-dependent rendering/estimator backends with explicitly synthetic test implementations. The test verifies that held estimator families and held movement levels remain absent from training, that new source intervals retain their original audit parent, and that each source/camera has one camera hash across physical conditions. Review contact sheets are also produced through the ordinary image-writing path.

These commands passed locally:

```bash
.venv/bin/python -m unittest discover -s tests/gait_fidelity -p test_data.py -v
.venv/bin/python -m unittest discover -s tests/gait_fidelity -p test_source_pipeline.py -v
```

## Evidence that still requires HAIC

The source preparation path retains the working synthetic-training environment checks for Torch 2.6.0 with CUDA 12.4, Torchvision 0.21.0 and compatible MMCV operators. Local mock execution does not establish that those GPU operators, licensed SMPL-H assets, EGL rendering or the three configured MMPose checkpoints work in the current HAIC allocation. The source preflight, preparation workers and timing profile must run there and retain their receipts.

The new extended intervals remain algorithm-screened development material until the user reviews the synchronized source/reference display. A successful kinematic screen is neither a validated locomotion classifier nor a human audit. The knee edits pass fixed geometry, continuity, low-foot displacement and additional-ground-penetration checks, but those checks do not establish physically validated dynamics or a model of neurological disease. The implementation labels them as synthetic kinematic stress tests.

No fresh confirmation population or clinical reference annotations are created by these tests. Additional GPU capacity increases the amount of the declared comparison that can run; independent people, valid references and retained uncertainty remain necessary for interpreting the resulting measurements.
