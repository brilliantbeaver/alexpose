The next run should test whether pretraining makes movement change more recoverable, while explicitly checking whether the result depends on the paired-change readout. The [completed core analysis](../results/core-analysis-20260924/README.md) motivates this addition: direct coordinate training was the strongest method, the added change loss worsened waveform error in all five model families, and JEPA's response advantage over initialized features remained uncertain.

This is a **24 September 2026 development-protocol amendment made after examining the core and before inspecting any new source-child results**. It is implemented as the explicit `--include-base-readouts` option in a new run, recommended name **`jepa-response-02`**. It preserves the [original follow-up](jepa-response.md), its primary contrast, and existing frozen runs. The user-specified new completion deadline is **25 September 2026 at 8 AM Pacific**, or `2026-09-25T15:00:00Z`.

**The core changes what a convincing result would mean.** A gain against direct/change would not establish practical superiority, because direct/base is substantially better. Lower nuisance error cannot establish preserved movement, because insensitive outputs and fewer failed measurements can also lower that score. A smaller response error can coexist with poor absolute measurements when endpoint errors cancel. Accordingly, the follow-up must distinguish improved movement information, dependence on the downstream objective, prediction reliability, and practical performance against direct/base.

The hypothesis remains that coupling latent prediction residuals across movement states helps beyond independent continuous endpoint regression. The existing two auxiliaries satisfy `L_delta = L_endpoint − mean(e_a · e_b)` on identical support. Their comparison holds the architecture, support and shared training-calibrated coefficient fixed. Separately evolving teachers, mask-dependent normalization and common-bias cancellation still limit a unique mechanism claim. Neither the core nor a low auxiliary loss demonstrates that the encoder has lost or recovered a particular physical variable.

**The revised matrix uses the same nine new encoders with two readouts each.**

| Pretraining variant | Question addressed | Frozen readouts per seed | Seeds |
| --- | --- | --- | --- |
| `jepa_delta_v1` | Does coupling endpoint residuals help? | Coordinate-only `base`; existing `paired_change` | 17, 29, 43 |
| `jepa_endpoint_v1` | Is continuous endpoint supervision sufficient? | The same two readouts | 17, 29, 43 |
| `coordinate_delta_v1` | Does a coordinate-space difference objective also help? | The same two readouts | 17, 29, 43 |

This requires **nine pretraining phases and eighteen readout phases: 27 optimization phases and 18 final models**. The additional cost over the original follow-up is nine readouts. Each pair consumes the exact same frozen encoder checkpoint, matched readout initialization and inherited training exposure. Pretraining is not repeated. Direct/base and both readout objectives of all five core families are imported from the parent's verified predictions, supplying 30 existing neural prediction exports without retraining them.

The architecture, data, source selection, train/development assignments, mask distribution, optimizer, update counts and paired-change coefficient remain inherited from the parent. The supplied parent summary records 2,000 pretraining, 2,000 readout and 4,000 end-to-end updates; setup binds the authoritative actual schedule from the full parent receipts. The experiment introduces no new renderings, pose extraction, GAVD processing, patients or confirmation-set access.

**Keep one primary comparison and make the new comparisons explicit.** The primary remains delta JEPA versus endpoint JEPA with `paired_change` on person-balanced movement-response error. The base-readout comparison and interaction are prespecified secondary development analyses, not replacements selected after looking at child outcomes. Let

```text
D_base   = error(endpoint JEPA, base)   − error(delta JEPA, base)
D_change = error(endpoint JEPA, change) − error(delta JEPA, change)
interaction = D_change − D_base
```

Positive `D` favors residual coupling. A positive interaction means its advantage is larger with the change readout; it does not establish that either final model is accurate. The new evaluator saves the base-readout contrast, each new model versus its matching plain-pretraining parent, each versus matched initialized features, and each versus direct/base. It preserves the same crossed person/seed bootstrap, without treating multiple readouts as independent people or seeds. The secondary intervals are descriptive and have no multiplicity adjustment or clinical useful-effect threshold.

| Possible result | Interpretation to retain |
| --- | --- |
| Delta improves on endpoint with both readouts, without worse coordinates or waveforms | Evidence that the benefit is not confined to the paired-change readout, within this tested procedure. |
| Delta improves only with the change readout | A conditional interaction with downstream supervision; practical and general representation benefits remain separate questions. |
| Delta improves only with the base readout | The paired-change readout can conceal or counteract the benefit; retain the original primary result and report the secondary finding transparently. |
| Endpoint improves similarly to delta relative to plain JEPA | Additional continuous supervision may explain the benefit; uncertain differences do not establish equivalence. |
| Coordinate-delta improves relative to plain coordinate pretraining | Difference supervision may help outside latent prediction; compare within-family gains before attributing a common mechanism. |
| Feature probes improve but neither readout improves | Information accessibility under the probe has not translated into useful restoration. |
| Primary response improves because failed measurements become less frequent | A reliability benefit; accuracy among successful predictions needs a separate interpretation. |
| Scalar response improves while waveform or coordinate errors worsen | A fidelity tradeoff, not general preservation. |
| No clear effect | Report the unresolved comparison, uncertainty and the core's direct/base result; do not select another primary. |

**Diagnostics address explanations that the core export could not resolve.** The existing response evaluator already exports signed bias, direction accuracy, limb-specific excursions, actual-reference response curves, held-dose strata and detailed coverage. The new option adds a zero-response benchmark and the exact hierarchy-preserving decomposition

```text
all-attempted response error
  = successful-output error contribution
  + 720° × failed-contrast fraction.
```

Both contributions use the same eligible denominator and the same windows-within-motions-within-people reduction. Successful-case conditional error is also retained but remains descriptive, since it changes support. The zero-response benchmark predicts no change without using a reference as an input; its error is the absolute reference change. It tests whether a favorable absolute score can be explained by small reference responses. The declared primary continues to include actual small responses and failed predictions.

Read these diagnostics alongside nuisance error, level accuracy, waveforms and coordinates. The detailed nuisance and per-window tables allow their failure mechanisms to be examined as well. No-change controls and actual-response curves should expose attenuation, amplification and offset; camera-dependent projected references must remain separate.

Keep the fixed training-only encoder/predictor/teacher probes from the original protocol. They use metadata-selected families and person-separated folds. Compare masked with deployment inputs and true movement changes with independently masked identical observations. Probe results explain the final models; they must not select coefficients, drop models, change stopping, or turn the reused development population into confirmation data. The current experiment still has no new latent re-pairing control and cannot establish a unique causal role for anatomically correct pairings.

**Resource admission remains a measurement.** The total cap remains 48 H100-hours: 2 for calibration/profiling, 36 for fits, 4 for feature diagnostics and 6 for recovery. Failed allocations remain charged. The revised profiler times all three pretraining variants and both readout objectives, sharing each probe's pretraining checkpoint across its readouts. It admits all 27 phases at the parent's actual update counts or records a refusal. It cannot silently reduce seeds, discard the endpoint control or shorten updates.

For scale only, applying the saved parent's representative phase estimates to nine pretraining and eighteen readout phases gives about 6.06 GPU-hours before the new auxiliary overhead; a 25% timing margin makes that 7.58. This is not a measured child runtime or admission result. The real profiler must include calibration, additional backward computations, prediction export, receipt verification, memory and I/O costs.

The new launch reserves **two CPU wall hours for final evaluation**, in addition to four hours for diagnostics, six hours for recovery and two hours for queue uncertainty. Admission needs at least 14 hours plus the measured dependency-aware fit duration remaining before the cutoff. A later deadline does not permit waiting until the final few hours to start. Pending reservations and actual completed allocation charges remain within existing scheduler limits; if runtime exceeds the admitted envelope, incomplete work stays explicitly incomplete.

**Launch from the complete parent on HAIC.** The six copied summary files on the Mac cannot initialize a source follow-up. Setup needs the original parent path, frozen code identity, preparation manifest and arrays, full ledger, checkpoints and completion receipts. It checks that the parent is finished and that no confirmation arrays are included. It binds both readout objectives of the core controls and never writes into the parent.

Use the [new launch instructions](../../../../slurm/gait-fidelity/START_RESPONSE_02.md) to create a new immutable release, initialize `jepa-response-02` with the explicit deadline and readout-control option, then preflight and launch it. If an older child already exists, do not change its config, deadline or plan in place. This implementation preserves the original 18-phase default and refuses attempts to use setup to change a saved child's matrix or deadline.

At preparation time, automated SSH access to HAIC was rejected with `Permission denied (keyboard-interactive)`. This prevented inspection or submission of remote jobs from the assistant session. Local validation and a software fixture establish software behavior only; CUDA profiling and actual submission remain HAIC steps. The [readiness record](../records/followup-launch-readiness-20260924.json) records completed checks and execution status.
