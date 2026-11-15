<!-- iteration02-execution-status -->
> This is the preserved design or feasibility record. Consult the [implemented protocol](protocol.md), [verified results](results.md), and [external execution report](external_replication_results.md) for current execution status.

# Verification of the U, J and A masking design

**Result: the proposed primary mask laws pass exact enumeration and sampler checks.** No GAVD data, videos, model weights or training code were loaded. This verifies the proposed masking construction, not its clinical value or an implementation of the full experiment.

Run the standalone, standard-library-only checker from the repository root:

```bash
python3 experiments/aaai/docs/iteration02/verify_mask_design.py
```

The checker only prints JSON. Its saved output is [mask_design_verification.json](mask_design_verification.json), and its source is [verify_mask_design.py](verify_mask_design.py). The exercised sampler uses Python's `random.Random`; it does not recreate the historical Torch mask bank from its seed alone.

## Exact results

| Property | U | J | A |
| --- | --- | --- | --- |
| Allowed joints | Exactly the declared 16. | Exactly the declared 16. | Exactly the declared 16. |
| Grid | 8 temporal positions by 16 joints. | Same. | Same. |
| Hidden tokens before validity filtering | Exactly 64. | Exactly 64. | Exactly 64. |
| Core construction | Not applicable. | 16 possible joint tubes, each with 8 tokens. | 16 possible anatomical blocks, each with 8 tokens. |
| Probability that any token belongs to the selected core | Not applicable. | 1/16. | 1/16. |
| Final masking probability of every token | 1/2. | 1/2. | 1/2. |
| Left-right reflection and time-reversal symmetry | Holds by uniform sampling. | Core family is closed under both. | Core family is closed under both. |
| Spatial connectivity and temporal wrap | No structured core. | Connected singleton joint; no temporal wrap. | Each declared anatomical group is connected; every interval is contiguous and in bounds. |

J and A each partition the complete grid into 16 disjoint eight-token cores. Each token occurs in exactly one possible core. Selecting a core uniformly therefore gives core probability 1/16. Filling 56 of the remaining 120 tokens uniformly gives:

```text
P(hidden) = 1/16 + (15/16)*(56/120) = 1/2.
```

For A, choosing a group with probability `width/16` and then its aligned temporal block uniformly is exactly equivalent to choosing one of the 16 cores uniformly. Choosing the five groups with equal probability would be a different policy and would break the marginal-probability claim.

The checker also verified all 8,128 token pairs. Their probabilities sum to `choose(64,2)=2016`, as required by the fixed budget. In J and A, two tokens in the same core are jointly hidden with probability 9/34. Tokens in different cores are jointly hidden with probability 21/85. U has probability 63/254 for every pair. Thus the policies match first-order masking probabilities while intentionally changing which observations disappear together. They do not match task difficulty or all higher-order dependencies.

An additional 6,144 sampled masks passed the fixed-budget and allowlist checks. These are software checks, not independent examples supporting a research result.

## The secondary E_leg bank asks a different question

The proposed bank has two sides, four aligned two-patch starts and two repeats, giving 16 masks. Each mask hides its eight-token distal-leg core and fills uniformly to 64 hidden tokens. Only the core supplies scored targets.

Before missing-data filtering, this gives 64 distinct target tokens and 128 target instances. Each lower-limb joint-time target appears twice. Repeats are not additional distinct patches. Every readout joint has eight possible temporal target positions before filtering.

E_leg deliberately has a different masking distribution: averaging over its fixed core schedule and random fill, lower-limb tokens are hidden with probability 8/15, and the other tokens with probability 7/15. This is appropriate for a named leg-occlusion diagnostic, but it must not replace E_U or be described as having the same 0.5 marginal probabilities as the three primary training laws. Diagnostic score references, support counts and results must remain separate.

## Required production checks

1. Preserve and hash the historical E_U bank. Record the RNG library, version, seed and actual bank bytes. Identical numeric seeds in Python and Torch do not imply identical masks.
2. Save a target indicator separately from the hidden-input mask. For E_leg, the 56 fill tokens are hidden context, but are not scored targets. Primary U evaluation keeps its historical target and eligibility definition.
3. Give each recipe's `h` and `g` the same training mask, target keys and weights. Keep the U-trained `f` weights fixed across recipes. Different permitted inputs can change its prediction; fixed weights do not imply identical predictions across different masks.
4. Explicitly define branch-appropriate training and diagnostic eligibility. Applying the legacy minimum-visible-own-joint rule to the selected J tube would reject its central training example, because that joint is hidden throughout the window. Preserve the old selector for primary U comparisons; report a separate context rule and population for J or leg-block diagnostics.
5. Apply validity only after drawing the mask. Do not redraw an invalid core or fill slot. Report valid target counts, own-leg context, other-leg context and rejected examples. Exact nominal masking probabilities do not imply identical usable supervision after filtering.
6. Verify that unknown hidden values cannot influence derived features, normalization beyond the declared shared reference, embeddings, selection decisions or predictions. The standalone checker cannot verify these model-level properties.
7. Retain original source/time target keys when checking distinct-patch support. A repeated mask or an overlapping window must not manufacture an additional physical target interval. Keep all variants within the same source role.
8. Keep primary comparisons within E_U. Within any secondary bank, use the same targets, references and support for compared methods. Do not call a U/U versus A/E_leg difference a masking-training effect.

The report establishes that U, J and A can support a controlled masking comparison. The remaining work is to verify the actual production sampler, information restrictions, eligible populations and fitted predictor behavior before reporting a gait result.

## Review of canonical Sections 5.1 through 5.4

The written U/J/A recipes agree with this exact enumeration. Their fixed U-trained baseline, policy-specific paired corrections, unchanged primary evaluation, separate encoder ablation, separate randomness streams and explicit shared reference resolve the main design concerns. The E_leg schedule also matches the verified construction. No mathematical error remains in those definitions.

Three specification details should be made explicit before production fitting:

| Priority | Detail | Concrete instruction |
| --- | --- | --- |
| Medium | The new own-leg context selector is named but its minimum context is not yet numerical. | A minimal declared rule is at least two valid, visible own-leg joint-patch tokens at two distinct temporal positions, with a fully valid target patch. Apply it identically to new correction-training arms and structured diagnostics. Retain the historical per-target U rule and the already declared eight-readout source gate for primary evaluation. This minimum is an engineering rule, not proof of adequate information. Record opposite-leg counts; do not silently exclude zero-opposite-context cases to inflate a gain. |
| Medium | “Separate normal score references” could be read as requiring another five normal source identities. | Use the same five reserved reference identities with bank-specific fitted reference parameters. If E_leg lacks usable reference support, report that limitation and restrict the diagnostic rather than borrowing fitting or test sources. |
| Low | Equal probabilities can be confused with identical finite realized token counts. | Say “matched marginal masking probabilities,” then report actual token and valid-target exposure. Sixteen random evaluation draws need not hide each token exactly eight times. Preserve the bank bytes rather than assuming the seed alone identifies them. |

These are implementation clarifications, not reasons to add another mask arm or redesign the scientific endpoint. The mathematical core and the primary comparison can proceed under the stated boundaries.
