# Eight-H100 execution-plan review

21 September 2026, Pacific time. The coordinating author revised the proposal for the user's assumptions of eight continuously available H100s and experiment setup within one hour. Two independent reviewers inspected the scientific comparisons and compute schedule. A third agent generated Figure 17; the compute reviewer independently opened its native and 900-pixel renders. This record summarizes their findings and final dispositions.

**Final disposition: accepted as a prospective execution specification.** The review verifies the plan and its enumeration. It does not establish measured H100 throughput, completed implementation, launched jobs, statistical significance or clinical validation.

## Scientific and matrix checks

The scientific reviewer independently reconstructed the 34 expected recipe cells and verified all 102 unique recipe/seed outputs. The manifest contains 33 reusable pretraining phases, 84 frozen readouts and 18 end-to-end phases. Every dependency has the correct seed, objective and mask; the 135-node optimization graph has no missing dependencies or cycles.

The initial 96-output design was expanded to 102 to include direct end-to-end models with per-example measurement supervision and valid re-paired-change supervision. This gives the practical direct method the same four loss regimes examined in the coordinate-pretrained and JEPA graph-policy comparisons.

Review also led to explicit statements that initialized encoders have no pretraining mask or query loss, that frozen readouts share deployment-style inputs, and that re-pairing must preserve endpoint exposure and allowed strata while recomputing valid target differences. A declared matching tolerance limits pairing-specific claims. The resource fallback preserves all recipe cells and seeds rather than selecting favorable completed results.

## Compute and scheduling checks

The compute reviewer independently verified the midnight-September-22 start, Thursday-noon freeze and Friday-18:00 upload scenario: 480 and 720 gross H100-hours, respectively. The planned allocations sum to 360 hours, leaving 120 hours of reserve before the results freeze. The 51/204/408-hour sensitivity scenarios are arithmetic illustrations for 102 complete recipe fits, not measured runtimes.

Corrections explicitly charge profiling, smoke checks and retries to the same study budget, reserve active-job costs atomically, and dispatch ready jobs without unnecessary wave barriers. The official AoE deadline conversion and abstract-deadline timezone are explicit. Historical timing receipts remain distinguished from H100 measurements, and optional clinical acquisition remains separate from the synthetic schedule.

## Visual and generated-document checks

The independent reviewer opened both Figure 17 renders and found no clipping, overlapping labels or misleading throughput claim. The diagram explains shared preparation, eight one-GPU workers and frozen evaluation; it labels the one-hour setup and capacity as planning assumptions. Final figure hashes remained unchanged after a repeat build.

The generated paper and visual gallery include the new plan and figure. Link/fragment, inline-JavaScript syntax, budget, dependency and artifact-preservation checks are recorded in the [validation receipt](../records/compute-validation-20260921.json). Live browser interaction was not exercised in this update. Earlier interactive arithmetic review remains applicable to the unchanged JavaScript/CSS, rather than constituting a fresh browser test.

## Accepted versions

| Artifact | SHA-256 |
| --- | --- |
| `methods/execution.md` | `36199c81c0eeabb4214996514bf547989010fbb10c7d0c133c85d09e9c2fd5c7` |
| `records/compute-plan-20260921.json` | `7d5fdcfc855d624e2d1c206129817e2d99e77d53ccca0d40a3ea1e1acfa14f63` |
| `images/17-parallel-execution.svg` | `e2a735ca3a8f54b3b28118d96e2f8c8933a529ed9131407fb646d7379018f1f7` |
| `images/previews/17-parallel-execution.png` | `9fc64c632345d5a24b12507bc57ce2097015f5fc50b8405732858b3c8b149162` |
| `images/previews/17-parallel-execution-900.png` | `ab38a3725966a41870803840614db65600cae57242d3640f541e2ec9c3b5cf65` |
