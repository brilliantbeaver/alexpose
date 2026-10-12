# Execute the focused JEPA response follow-up

[Proposal](../README.md) · [Scientific protocol](jepa-response.md) · [HAIC commands](../../../../slurm/gait-fidelity/JEPA_RESPONSE.md) · [Measurement protocol](evaluation.md)

The follow-up changes the initial training of an encoder, which converts joint trajectories into learned feature vectors. A later coordinate readout converts those features into position corrections while the encoder is frozen, meaning its parameters stay fixed. Each final model therefore needs two optimization phases, one for each training stage.

**Active execution contract: 23 September 2026.** The separate `jepa-response-01` child trains nine models after `walking-core-01` and its evaluation complete. Its maximum additional allocation is **48 H100-hours**, with results cut off at **September 24, 2026, 18:00 America/Los_Angeles** (`2026-09-25T01:00:00Z`). Up to eight one-GPU workers may run concurrently when their dependencies and resource reservations permit.

This contract replaces the earlier broad eight-H100 capacity example as the guide for the active follow-up. The registered 34-recipe full matrix remains available as a separate program; its 102 final fits are not part of this allocation. Local validation has exercised the implementation on CPU. Source H100 runtime and scientific outcomes are still pending.

## 1. Bind the completed parent without changing it

Use a new immutable source release and a child work directory outside the parent. Setup verifies the complete parent matrix, evaluation, original source identity and artifact receipts. It refuses active or unresolved parent work, mismatched results or changed arrays. The original checkout and run remain unchanged throughout child execution.

The child inherits the prepared AMASS-derived bundle, model, mask fraction, optimizer, sampling, physical-time grid, measurement settings, split roles and the parent's actual selected update counts. Its binding also retains admission amendments and exclusions. There is no new rendering, pose extraction, GAVD processing or confirmation-set evaluation. Population sizes are read from the bound manifest.

The parent's nominal source schedule was 2,000 pretraining updates and 2,000 readout updates, with a registered half schedule available to its own profiler. The child uses whichever schedule the completed parent actually selected. It cannot shorten that schedule again to fit its deadline.

## 2. Retain all three variants and all three seeds

| Pretraining variant | Seeds | Pretraining phases | Frozen-readout phases | Final models |
| --- | --- | ---: | ---: | ---: |
| Paired latent residuals: `jepa_delta_v1` | 17, 29, 43 | 3 | 3 | 3 |
| Independent latent endpoints: `jepa_endpoint_v1` | 17, 29, 43 | 3 | 3 | 3 |
| Paired coordinate residuals: `coordinate_delta_v1` | 17, 29, 43 | 3 | 3 | 3 |
| **Total** | Three seeds per variant | **9** | **9** | **9** |

Each pretraining phase has its own explicit variant identity and feeds one matching readout. All readouts use `paired_change`. The plan therefore contains **18 optimization phases**; calibration/profiling and representation diagnostics are additional worker stages. Parent baseline predictions are reused rather than refitted.

The sequence is: completed parent → child setup and verification → calibration/profiling → admitted pretraining and dependent readouts → representation diagnostics → development evaluation and numerical reconstruction. A readout waits for its matching complete checkpoint, and an admitted child diagnostic stage waits for the full eighteen-phase matrix.

## 3. Calibrate losses before measuring feasibility

The first GPU worker sets the weights of the added loss terms using a fixed calibration rule. It uses 32 fixed training batches at seed-17 initialization. It performs no optimization and uses no development outcomes. Its receipt binds the coefficients to code, data, training settings, formulas and support rules. Final fits initialize afresh, and both JEPA variants use the same calibrated coefficient across all seeds.

The same worker profiles every new variant's pretraining and readout type using short timing runs. These runs do not select a scientific winner. The projection includes fixed setup costs and measured update time, then applies a **1.25 timing factor**. Admission requires all projected fits to fit within the 36 H100-hour fitting allowance and each phase within the 119-minute workload limit of its two-hour allocation.

Wall-time admission follows the saved dependency graph with up to eight workers. It reserves four further hours for diagnostics, six for recovery and two for queue uncertainty before the absolute cutoff. Queue availability remains external to the program. If these requirements fail, the matrix is refused; there is no branch that drops a control or seed, reduces updates or edits the coefficient to obtain a result.

## 4. Account for allocation and the cutoff

| Reserved work | H100-hours |
| --- | ---: |
| Eighteen optimization phases | 36 |
| Calibration and profiling | 2 |
| Representation diagnostics | 4 |
| Recovery allowance | 6 |
| **Maximum additional allocation** | **48** |

An H100 is the GPU used by the source study. H100-hours sum allocated GPU time across workers. Eight simultaneous one-hour allocations consume eight H100-hours. Failed attempts remain in the ledger and consume their assigned budget; retries use the recovery allowance. The saved run permits at most two attempts per phase. A diagnostics retry retains its four-hour allocation.

The coordinator checks the remaining complete schedule before admitting more work. Training saves an interrupted checkpoint near the cutoff, the worker supervisor enforces the deadline, and cancellation applies only to recorded child job IDs, including pending allocations. The child never cancels or modifies the parent. The fixed two-hour queue allowance is a resource assumption, not a guarantee that queued work will start on time.

A profile refusal records `FOLLOWUP_MATRIX_NOT_ADMITTED` and can preserve parent-only representation diagnostics. Work that starts but cannot complete the whole matrix is recorded as `FOLLOWUP_INCOMPLETE`. Neither status supports a selected partial comparison, and no experiment stops because an interim scientific outcome is unfavorable.

## 5. Keep the result reviewable

Retain `parent-binding.json`, the child configuration and plan, calibration and timing receipts, attempt directories, complete checkpoints, saved development predictions and the evaluation tables. The primary contrast is delta JEPA versus endpoint JEPA on person-balanced response error; the five parent methods and coordinate-delta arm supply additional context.

The report must distinguish successful submission, resource admission, phase completion and completed evaluation. Read coverage, coordinate error, waveform error and nuisance sensitivity alongside the primary estimate. The verification command reconstructs published statistics from verified predictions in temporary storage and reports any disagreement.

CPU mathematical checks, the end-to-end generated-data fixture and the [dated implementation review](../reviews/jepa-response-20260923.md) establish local behavior. They do not establish CUDA operator compatibility, measured source throughput or a JEPA advantage. Those facts require the actual HAIC preflight, profile and completed source comparison.

## Scope of later experiments

The broader full matrix tests alternative masks, topology controls, temporal refiners, additional endpoint labels and re-pairing. It requires a separate resource decision rather than spare capacity being silently added to this child. Fresh confirmation would also require a separately admitted population, justified margins and a comparison frozen before its outcomes are inspected. This follow-up supplies synthetic development evidence only, even if every registered phase completes before the cutoff.
