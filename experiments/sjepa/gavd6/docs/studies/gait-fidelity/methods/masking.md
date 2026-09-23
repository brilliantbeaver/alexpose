# Stochastic anatomical masking for gait fidelity

21 September 2026. Source inspection and a small sampler audit are complete. The training comparisons below are proposed; no new restoration model has been trained.

[Research proposal](../README.md) · [Local video audit](../data/local-videos.md) · [Visual guide](../images/gallery.html)

## What to test

A pose encoder needs examples in which a joint is visible to learn how its observations relate to the rest of the body. Permanently hiding the hips, knees or ankles removes those examples. **Stochastic graph-time masking** instead chooses a connected anatomical region, such as a hip–knee–ankle chain, and hides it for a sampled interval. Subsequent presentations can expose that region and hide another. Here, “graph” means the joints and their anatomical connections; it does not require replacing the encoder with a graph neural network.

![Three illustrative masks hide a left leg, right arm and trunk, while the same joints provide context in other draws.](../images/03-changing-graph-masks.svg)

*A single time block from three possible draws. These are explanatory masks, not outputs from a trained model. An anatomical label remains attached to its joint even when the coordinate is hidden.*

The useful hypothesis is that exposing every joint while practicing recovery from connected gaps can improve restoration under occlusion **without shrinking genuine bilateral differences or their changes**. This must be tested against equally masked coordinate training. Structured masks may also make recovery too easy through a stereotyped gait prior, or remove too much evidence from the more informative leg. Neither an anatomical graph nor a low prediction loss guarantees preserved movement.

## What the requested notebook already does

The inspected [anatomical-mask notebook](../../../../../../multiple-sclerosis/02_anatomical_mask_and_tokenization.ipynb) uses the current [graph-time sampler](../../../../../../multiple-sclerosis/sjepa/masking_v2.py). It has already replaced the earlier permanent 12-landmark mask with fresh per-example masks. The notebook's old single-draw animation could show both hips hidden throughout its repeatedly replayed window; replaying that animation does not sample new masks. The sibling project's [mask visualization audit](../../../../../../multiple-sclerosis/docs/12-mask-visualization.md) explains this distinction.

| Inspected path | Actual behavior | Implication for this experiment |
| --- | --- | --- |
| MS notebook 02 and `masking_v2.py` | Samples connected BlazePose-33 regions over contiguous time blocks; overlapping regions may hide a joint throughout one window. | Reuse the idea and coverage diagnostics after adapting the anatomical schema. |
| MS region weighting | Applies approximate overlap correction and boosts regions touching a designated clinical set. | Hips and shoulders can remain overrepresented. A nominal 1.5 region weight is not a measured 1.5 joint probability. |
| MS token layout | A token represents a joint over a block of consecutive frames; indexing is time first, then joint. | Confirm the same layout when converting masks, and hide the complete coordinate content of each selected token. |
| Synthetic v2 [`_mask`](../../../../src/gavd6_sjepa/research_directions/synthetic_training_v2/training.py) | Already draws stochastic masks from observed tokens, with a random time start and joint ordering. | This study compares policies; it does not repair a permanent anatomical mask in v2. |
| Synthetic v2 complete-input example | At 64 frames, four-frame patches and a 50% mask, eight of sixteen whole-body blocks are hidden. | Random joint ordering does not create partial-limb masks when the budget ends on a complete time block. |
| Synthetic v2 training arms | `coordinate` uses masked coordinate pretraining and a later readout; `direct` trains end to end without this pretraining mask. | Use `coordinate` versus `paired_jepa` for the matched objective-by-mask comparison, and retain `direct` as the practical benchmark. |

Do not directly import the MS sampler into the current body12 pipeline. Its groups contain BlazePose indices up to 32, and a body12 call fails with an out-of-bounds index. Body12 has shoulders, elbows, wrists, hips, knees and ankles; it has no heel or toe landmarks. Create groups by audited joint names, validate every edge, and derive the reverse mapping explicitly. The same region-bias rule would boost every body12 group because arms contain shoulders and all other regions contain clinical joints, cancelling the intended relative preference. Start with no clinical bias and treat any later leg weighting as a separate experiment.

### A small coverage audit reveals the remaining problem

We drew 512 masks from one advancing seed-0 generator using the current MS sampler, 33 joints, eight time blocks, target ratio 0.60 and clinical bias 1.5. The achieved masked-token fraction was **63.28%**. Every joint supplied context somewhere in at least 74.0% of windows, yet the left hip supplied context in only **7.0%** of draws at the fifth time block, compared with **51.6%** at the first. The entire left hip was hidden in 21.5% of windows and the right in 22.9%. These are descriptive sampler diagnostics, not estimates of model quality or clinical benefit; larger banks in the sibling audit show the same central-interval imbalance.

![The coverage audit checks every joint-time slot rather than only whether a joint is visible somewhere in a window.](../images/05-coverage-audit.svg)

*A joint can be visible in many windows and still rarely supply context at their centers. Retain both window coverage and joint-by-time frequencies.*

Finite intervals that must fit inside a window include central positions more often than endpoints, and overlapping regions compound that imbalance. A maximum length per region does not cap the union of multiple regions. Equalize temporal exposure through a declared sampler design or compare a coverage-balanced variant; do not silently change sampling until a preferred model wins. Cyclic sampling balances relative starts but can split an apparent gap across the two ends of a crop, so it must be labeled separately from a physically contiguous occlusion. For physical missingness, generate the interval process on a longer parent recording before cropping and retain actual run lengths.

The retained [audit JSON](../evidence/masking-bank.json) and [reproduction script](../scripts/audit_masking.py) describe this check. Source hashes distinguish the inspected implementation from future revisions.

## Keep the input and target contract explicit

![Observed validity, random hiding, encoder context and reference-valid loss are separate stages.](../images/04-mask-contract.svg)

*Proposed contract. Artificial training masks are absent from ordinary deployment; genuine missing observations and their validity indicators remain.*

Maintain four separate arrays: detection availability, artificial training mask, image visibility, and reference validity. Artificial hiding selects available input tokens; natural detector failures remain absent regardless of the draw. A hidden image joint may still have a valid synthetic reference, while a visible real joint may have no independent reference. Zero coordinates must not stand in for all these states.

Use an advancing, reproducible RNG keyed by run, epoch/update, sample presentation and worker. Keep mask randomness independent from model initialization and nuisance generation so changing one policy does not silently change the training examples. Save compact mask receipts and achieved counts. Each eligible joint-time slot must have a nonzero chance of serving both as context and target. Verify that property over several seeds and the actual patch/window shapes, including sparse-input examples; a nominal probability cannot supply observations that the dataset never contains.

Remove artificially hidden coordinates before any cross-token operation that could reveal their values through velocities, bone vectors, interpolation or normalization. Declare whether the experiment masks already-normalized tokens or simulates raw observation loss. In the first case a shared normalization computed before hiding can expose aggregate information about hidden inputs; report it and include a stricter context-only sensitivity check. In the second case compute the deployable transform from retained observations, with a fixed fallback for insufficient context. Never normalize from reference poses or affected-side labels. All compared arms must receive the same permitted information.

For paired JEPA, the complete valid reference sequence can supply teacher features during training. That is privileged supervision, not deployment input. Replace invalid reference coordinate values before teacher projection, carry validity, and verify that changing invalid numeric content cannot affect valid target features. Explicit missing-query tokens can remain in the model. Normalize coordinate and feature losses by their own eligible target counts, record dropped examples, and check actual gradients for every joint embedding across a mask bank. Missing targets must not become numeric zero labels.

The MS training path also applies student-only geometric augmentation, including reflection, while the teacher sees the original sequence. Do not copy that representation-invariance objective unchanged into signed gait restoration. Record each transform and its inverse, move input coordinates, confidence, validity and mask indices together, and compare targets in an explicitly shared convention. An imposed error in estimated joint naming leaves anatomical reference labels fixed. The [source review](../evidence/masking-source-review.md) distinguishes a passing token-level hidden-value check from information that passes through preprocessing.

Handle very sparse inputs explicitly. If no usable context remains, skip additional artificial hiding and retain the natural-failure record under a declared scoring policy. Do not manufacture observed joints or drop difficult patients merely to meet a context guarantee. Context coverage and reference-valid loss coverage require separate denominators.

## Run the registered masking comparison

Use the existing window length, patch size and 50% nominal mask budget for the implementation check. For the main study, choose a reviewed interval length and camera panel that support the declared measurement, then hold those choices fixed across policies. Match the **realized** hidden fraction among available tokens, not only the configuration value. The six objective/policy combinations cross two pretraining objectives with three masking policies; the [eight-H100 plan](execution.md#3-run-the-complete-matched-matrix) evaluates each with and without paired-change readout supervision and with all three seeds:

![Coordinate and paired-JEPA pretraining are each crossed with current blocks, random joint-time tokens and graph-time masks.](../images/06-matched-mask-experiment.svg)

*Each arm receives the same observations and supervised coordinate readout protocol. This grid concerns pretraining; direct end-to-end training is a separate benchmark.*

| Policy | Purpose | Remaining difference to expose |
| --- | --- | --- |
| Current stochastic time blocks | Reproduce the established v2 policy on the new data. | On complete input, it often removes all joints together. |
| Uniform joint-time tokens | Establish whether spreading visibility across joints is sufficient. | Isolated tokens differ from graph regions in temporal persistence and spatial structure. |
| Connected graph-time regions | Hide sampled limb/trunk regions over intervals, with left/right symmetry in sampling. | Overlapping groups can bias joint exposure and overshoot the token budget. |

This comparison alone cannot attribute a gain to anatomical connectivity. Include the registered **topology-shuffled control** matching group sizes, temporal intervals, mask counts and marginal joint exposure while breaking anatomical adjacency, and random-joint intervals with the same run-length distribution. Run these controls under both objectives and both readout losses, regardless of the initial graph result. Reject or explicitly report residual mismatches when the constraints cannot be matched. A fixed anatomical mask remains an optional historical failure demonstration; its different marginal exposure makes it unsuitable as the connectivity control.

Keep the direct end-to-end model, calibration and temporal filtering on the same population. Adding masking augmentation to direct training would be a separate declared arm with matched observations and supervision. The full matrix fixes graph-time as the comparison policy for initialized and shuffled-reference controls; the initialized encoder has no pretraining mask or query loss. All frozen readouts use the same observed inputs without additional artificial masking. The [execution plan](execution.md) crosses mask policy and paired-change supervision prospectively so that the interaction can be evaluated without selecting favorable combinations afterward.

Hold people, clips, reference mapping, update budgets, readout capacity, tuning budget and evaluation conditions fixed. Report compute as well as updates. A one-seed execution check identifies numerical or data failures, without selecting which registered cells receive seeds 17, 29 and 43. The complete matrix receives that seed set. CPU mask/gradient checks precede full jobs; actual H100 measurements determine whether the predeclared common update budget fits the deadline.

“Same target exposure” means the same source windows, query positions, validity rules and downstream coordinate labels, with shared batch and mask receipts within each policy. JEPA predicts features contextualized by the reference sequence, whereas the coordinate objective predicts coordinate values; those target contents differ by design. Report that distinction and keep the shuffled/initialized controls rather than implying the targets contain identical information.

### Test a gait-preservation claim, rather than masked-token loss

The focused proposal chooses the [signed projected knee-excursion difference](evaluation.md#primary-engineering-endpoint) for the first response comparison, together with position, displacement and waveform checks. It uses the right-minus-left difference in 95th–5th percentile image-plane knee angle. Reference-only validity, common paired-cell time support and an all-attempted failure policy must be frozen before training. Clinical step-time asymmetry needs independently labeled contacts and side identity; ankles alone do not establish them.

The primary development comparison should ask whether graph masking reduces error under a held-out occlusion family while preserving the reference change in a signed bilateral quantity. Require a response-error guardrail and clean-input position guardrail alongside the occlusion improvement. Set useful margins from reference resolution/repeatability before opening confirmation results. Retain waveform error, missing outputs, event coverage where valid, and at least one outcome excluded from the training loss. Evaluate realistic image occluders and natural estimator failures separately: anatomical neighbors do not necessarily disappear together in an image.

Separate recoverable cases with retained side-specific evidence from cases where identical supplied observations admit different true movements. The latter require uncertainty or abstention rather than a deterministic recovery claim. A verified limb label can resolve naming without revealing hidden motion.

Keep all conditions from one source identity together and compute paired contrasts within people when identity is known. Independent people and trained seeds, rather than token draws or rendered windows, determine the uncertainty of the scientific claim. The eight already inspected development people remain development. The local MS/PD collection also remains development after this inspection. Grouped or nested cross-validation can support internal exploration here; confirmation requires fresh, independently referenced person/source groups, with the claim bounded by the identities actually verified.

## Positioning and decisions

[MAMP (ICCV 2023)](https://arxiv.org/abs/2308.07092) already uses motion-aware masking, and [S-JEPA (ECCV 2024)](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04755.pdf) adopts that sampling approach for feature prediction. [SkeletonMAE (ICCV 2023)](https://arxiv.org/abs/2307.08476) combines skeleton topology and masked pretraining. More directly, [Hui et al. (Scientific Reports, February 2026)](https://www.nature.com/articles/s41598-026-39330-9) select anatomical groups with motion-weighted sampling and mask temporal segments. Graph masking by itself is therefore established prior art.

The potential contribution is evidence about when a masking policy preserves or distorts genuine gait response, and a remedy validated beyond its training quantities. Motion weighting deserves an ablation because restricted movement can itself be informative; it should not be declared universally unsuitable for MS or PD. Start with input-independent, side-balanced sampling to avoid encoding a clinical answer in the pattern. Do not choose the masked side from a diagnosis or assume MS always causes unilateral impairment.

Launch the registered matrix after the graph adapter, coverage bank, preprocessing information check and gradient smoke test pass. If gains disappear after matching temporal persistence or direct supervision, report that boundary. The [execution plan](execution.md) preserves the controls when adapting to measured cost. No result from the current audit establishes that graph masking will improve JEPA or achieve statistical significance.
