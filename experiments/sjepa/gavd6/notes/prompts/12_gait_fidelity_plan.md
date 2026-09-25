# A focused JEPA study of movement-response preservation

## 1. Research question and chosen direction

Test whether **coupling prediction errors across paired movement states during JEPA pretraining improves the size and direction of the movement changes recovered by its frozen representation**.

The current implementation predicts clean-reference features separately for each endpoint of a movement pair. Explicit supervision of the difference between movement states enters later, when training the coordinate readout with the encoder frozen. This creates a plausible limitation: the readout may be asked to recover movement information that pretraining did not make accessible.

Two other implementation details matter. The predictor is discarded before readout training, so useful information could depend on that predictor; meanwhile, the coordinate readout adds corrections to the observed coordinates, so reasonable restoration performance alone does not demonstrate useful learned features. Neither observation proves a failure mechanism.

Archived synthetic-training-v2 results show little separation between paired and shuffled JEPA. Those findings motivate stronger controls, but the ongoing Gait Fidelity core experiment must establish its own results.

The candidate directions are:

| Direction | Hypothesis and distinguishing evidence | Decision |
|---|---|---|
| **Couple errors across movement pairs** | Matching differences during pretraining improves downstream movement response beyond equally supported endpoint regression. A benefit over that control would support residual coupling as a useful mechanism. | Implement as the main experiment. |
| **Add continuous endpoint feature regression** | Existing distribution-matching supervision provides insufficiently precise targets. If endpoint regression matches the paired improvement, additional feature supervision explains the benefit without requiring coupling. | Implement as the principal control. |
| **Retain or expose predictor features** | Useful movement information becomes more accessible after the predictor and is poorly transferred to the frozen encoder. Better predictor probes would motivate a later capacity-matched architectural experiment. | Diagnose; defer additional restoration models. |
| **Separate bilateral features or predict temporal derivatives** | Explicit left–right interactions or change targets could improve signed response or timing. Invertible feature rearrangement alone changes no information, and extra processing requires capacity and supervision controls. | Defer within the deadline budget. |

Difference supervision has substantial precedents, including [Sobolev Training](https://proceedings.neurips.cc/paper/2017/file/758a06618c69880a6cee5314ee42d52f-Paper.pdf) and [augmentation-aware self-supervision](https://proceedings.nips.cc/paper_files/paper/2021/hash/94130ea17023c4837f0dcdda95034b65-Abstract.html). [V-JEPA 2.1](https://arxiv.org/html/2603.14482v3) also motivates examining dense feature grounding. The contribution must therefore come from a clear mechanism and convincing movement-preservation evidence, rather than presenting an auxiliary difference loss as inherently novel.

## 2. The controlled experiment

Add three representation variants, each using seeds **17, 29 and 43**:

| New variant | Pretraining addition | Downstream training |
|---|---|---|
| `jepa_delta_v1` | Match differences between endpoint prediction errors | Existing frozen readout with `paired_change` |
| `jepa_endpoint_v1` | Regress endpoint features independently | Identical frozen readout |
| `coordinate_delta_v1` | Match differences between coordinate prediction errors | Identical frozen readout |

This requires **nine pretraining phases and nine readout phases**. Reuse the completed core comparisons, including ordinary paired JEPA, coordinate pretraining, initialized features, shuffled-reference JEPA and direct coordinate training.

Keep the architecture, masks, sampled endpoints, normalization, optimizer and deployment inputs unchanged. All variants use the **actual update schedule selected by parent-run profiling**, including any registered reduction. Readouts receive matched initialization and endpoint exposure.

### Specify the auxiliary objectives precisely

For endpoints $a,b$, let $p_i$ be predicted features, $t_i$ the teacher features, $c$ the existing teacher center, and $H(v)$ subtract the mean across feature channels. Define the detached-target residual

$$
e_i =
H\!\left[
\frac{p_i}{\tau_s}
- \mathrm{stopgrad}\!\left(\frac{t_i-c}{\tau_t}\right)
\right],
$$

using the existing temperatures.

With feature dimension $D$, compare

$$
L_{\Delta}=\frac{\|e_b-e_a\|^2}{2D},
\qquad
L_{\mathrm{endpoint}}=
\frac{\|e_a\|^2+\|e_b\|^2}{2D}.
$$

Their difference is

$$
L_{\Delta}-L_{\mathrm{endpoint}}=-\frac{e_a^{\top}e_b}{D}.
$$

Thus, with identical support and one shared coefficient, the comparison isolates an explicit coupling between endpoint errors. Retain the original JEPA objectives and add no parameters.

For coordinate pretraining, convert each normalized coordinate residual to a common observation-derived pair scale:

$$
s_{ab}=\frac{s_a+s_b}{2},\qquad
r_i=\frac{s_i}{s_{ab}}\left(\hat{x}_i^{\mathrm{norm}}-y_i^{\mathrm{norm}}\right).
$$

Use $\|r_b-r_a\|^2/4$ per frame, then average across the four frames in a token. The divisor accounts for two endpoints and two coordinate dimensions; do not divide by four again.

For all auxiliaries:

- Admit only tokens queried at both endpoints with all four reference frames valid at both endpoints.
- Average tokens within each pair, then average supported pairs equally.
- Retain each endpoint’s original base loss when auxiliary support is absent.
- Record support by person and condition; unsupported pairs are never reported as successful zero errors.

### Fix coefficients without development tuning

Use 32 fixed training batches at seed-17 initialization. Measure squared gradient norms over **all trainable parameters**, treating unused gradients as zero. Encoder-only calibration would fail for coordinate pretraining because its zero-initialized output head initially blocks encoder gradients.

Choose

$$
\lambda_J =
0.1\sqrt{\frac{G_{\mathrm{base}}}
{\max\!\left(G_{\Delta},G_{\mathrm{endpoint}}\right)}},
$$

where each $G$ sums squared gradient norms across calibration batches. Both JEPA variants share this coefficient across all seeds. Calibrate the coordinate coefficient analogously against its own base loss.

Reject zero or nonfinite calibration quantities. Save the calculation as a verified receipt, restore random-generator states, and initialize final fits afresh. Log gradient components and clipping frequency: the calibration controls the initial scale, not the entire optimization trajectory.

## 3. Diagnostics, evaluation and interpretation

Before interpreting restoration scores, examine where movement information is accessible.

Use a fixed training-only diagnostic panel containing at most three metadata-selected source families per person, with one registered movement pair per family. Preserve joint and temporal token positions when extracting encoder, predictor and teacher features.

Fit standardized ridge probes of reference movement change using three folds separated by person, a fixed mean-loss ridge penalty of $0.01$, and no hyperparameter search. Compare masked pretraining inputs with unmasked deployment inputs. These probes measure linear accessibility; failure does not prove that information is absent.

Additional diagnostics will:

- Compare teacher-feature differences for true movement pairs with identical-reference pairs subjected to independent training masks and normalization.
- Measure feature variation across examples at fixed joint/time positions, preventing positional embeddings from dominating a pooled rank diagnostic.
- Record teacher and predictor difference norms, auxiliary support, gradient contributions and clipping.
- Compute cross-entropy, teacher entropy and KL divergence from identical probabilities and reductions before state updates. Historical logged cross-entropy minus entropy is unsuitable for this calculation.

These measurements diagnose the proposed mechanism without selecting variants or stopping experiments based on favorable results.

**Primary comparison:** `jepa_delta_v1` versus `jepa_endpoint_v1` on the existing person-balanced `response_error`. This measures error in the change of right-minus-left projected knee excursion, using actual reference geometry.

Pair methods by person and seed; aggregate correlated windows within people before applying the existing crossed person/seed bootstrap. Report effect sizes, uncertainty and seed consistency. Three seeds do not replace independent people.

Add secondary measurements of:

- Signed response bias and direction accuracy where the reference change exceeds the saved one-degree tolerance.
- Separate left and right excursions, coordinate accuracy and temporal waveform fidelity.
- Nuisance-induced changes and prediction failures.
- Response plots against actual reference changes, separated by camera, corruption and held-out conditions.

Missing predictions remain failures. Conditional plots retain coverage counts and cannot replace the declared population.

Interpret outcomes according to a fixed table:

| Finding | Supported interpretation |
|---|---|
| Paired auxiliary improves over endpoint regression | Coupling residual errors helps under the registered observation procedure. |
| Endpoint regression obtains the same improvement | Additional continuous feature supervision is a sufficient explanation. |
| Coordinate-difference supervision matches the JEPA gain | The remedy is not specific to latent prediction. |
| Latent loss improves without decoded response improvement | Better pretraining fit has not established useful restoration features. |
| Response improves while position or nuisance errors worsen | A measured tradeoff; no claim of improvement on every objective. |
| Differences remain uncertain | An inconclusive comparison, with its uncertainty reported. |

The delta objective permits shared endpoint bias, and independent normalization can introduce feature differences unrelated to physical movement. Neither latent distance nor low auxiliary loss establishes anatomical fidelity.

Without a new pretraining re-pairing control, the study will not claim that correct anatomical pairing uniquely causes any benefit. Synthetic findings also cannot establish clinical validity or real-video transfer.

## 4. Integration and HAIC execution

Create a separate run, **`jepa-response-01`**, using a new immutable code release. Preserve the active `walking-core-01` checkout, configuration, checkpoints and results.

Add one public launcher command:

```text
run.sh setup-followup CHILD_WORK --parent-work PARENT_WORK
```

It initializes the registered follow-up protocol. Existing `preflight`, `launch`, `status`, `report` and `verify` commands then operate on the child run.

Implementation requirements:

- Introduce `representation_variant` in phase identities, pretraining deduplication, checkpoint signatures and upstream validation. Include the auxiliary formula version, support rule and calibration receipt hash. Existing core/full plan identities remain unchanged.
- Add explicit reuse of a verified, prepared Gait Fidelity dataset. The current `source_bundle` option selects a legacy roster and must not be repurposed.
- Bind the child to parent configuration, code identity, actual profiling schedule, prepared-array hashes, admission amendment and completed prediction receipts. Finalize these bindings after the parent finishes; never fabricate child preparation jobs.
- Open parent data read-only, without rendering or copying pose arrays again. Run parent-checkpoint diagnostics with the recorded parent code and interpreter, writing only into child output paths.
- Freeze the calibration protocol before execution; calculated coefficients become verified derived artifacts rather than edits to the frozen configuration.
- Add a child-specific profiler. It must not inherit the existing automatic half-budget choice or 60-GPU-hour evaluation reserve.
- Keep reference coordinates, intervention labels, pair identifiers and true anatomical assignments outside deployment inputs.

The follow-up uses the same prepared AMASS cohort and split boundaries. It introduces no new dataset, GAVD extraction or confirmation-set exposure.

The agreed resource limits are:

| Allocation | Maximum additional H100-hours |
|---|---:|
| Eighteen fitting phases | 36 |
| Calibration and profiling | 2 |
| Representation diagnostics | 4 |
| Recovery reserve | 6 |
| **Total** | **48** |

Core work has priority; child GPU work begins after the core finishes. Admission requires measured phase runtimes, verification/export overhead, dependency-aware wall time and a two-hour queue allowance to fit before **September 24, 2026, at 6 PM Pacific**.

If the complete matrix cannot fit, retain diagnostics and do not launch a selectively reduced matrix. Count failed allocations against the limit. At the cutoff, checkpoint and stop remaining child jobs; report any incompleteness without discarding unfavorable results.

## 5. Validation and deliverables

Before HAIC submission, require:

- Algebra and gradient tests for both latent losses, common-bias cancellation, teacher detachment and coordinate scale conversion.
- Tests for partial reference validity, unsupported pairs, correct averaging and nonfinite calibration.
- Distinct checkpoints for all variants, strict upstream matching and deterministic interrupted/resumed training.
- A small end-to-end fixture covering calibration, all three variants, frozen readouts, prediction export and paired evaluation.
- Compatibility checks preserving existing core/full plans and default training behavior.
- Tampered-parent rejection, read-only dataset reuse, budget exhaustion and deadline-stop tests.
- An independent final review of leakage, matched exposure, extra supervision, statistical aggregation and the correspondence between configured and executed experiments.

Deliver one focused tutorial showing the equations, tensor operations, gradient checks and response plots, alongside updated HAIC instructions and a study-methods writeup. Label fixture outputs as software validation.

HAIC profiling remains the execution test for real GPU compatibility and cost. The paper-facing outcome will be a bounded claim about whether pretraining preserves useful bilateral movement response—and what the matched controls reveal about the reason.
