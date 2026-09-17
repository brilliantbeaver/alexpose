# Independent adversarial review of proposals 3 to 6

Reviewed 12 September 2026. Scope: proposals 3, 4, 5, 6 and the common evidence/execution contract. No training was run. These are objections to the proposed inference and execution, with concrete wording or design changes. They are not experimental findings.

## Blocking or high-priority revisions

### 1. Forecast preprocessing can leak the future before the network sees an input

**Where:** Common contract, AMASS bridge; proposals 3 to 5. Proposal 6 already recognizes most of this risk.

The common contract preserves physical timestamps but does not impose a universal causal conversion rule. HumanML3D-style velocity and contact channels, whole-clip floor estimation, interpolation and canonicalization can encode information from samples beyond the observation cutoff. Clamping the generated prefix does not repair that leak. This matters especially when a proposal seeks only a small incremental future-prediction gain.

**Proposed wording:** “For every forecasting experiment, derive the model-visible prefix independently of the unavailable suffix. Its floor estimate, normalization, imputation, velocities, contacts and boundary channels may use only permitted observations or training-fitted constants. Perturb or replace the withheld suffix and require the final prefix tensor to remain unchanged. Reconstruction proposals may use their explicitly declared complete observation window.”

Keep the assertion on the actual input tensor, not just the dataloader's frame-index list.

### 2. Proposal 4 does not yet establish that the conditional mean is uninformative

**Where:** Proposal 4, first 48-hour gate.

A mean-error improvement below 1% as a point estimate can be an underpowered positive effect, or a poorly fitted mean head. The distributional model can gain CRPS partly by improving its mean. If so, “information missed by conditional-mean gates” overstates the evidence.

**Required change:** Prespecify an equivalence margin and require the upper uncertainty bound on mean-prediction improvement to fall below it, after comparing flexible well-calibrated mean baselines. Independently require a positive distribution-score improvement. If the equivalence test is inconclusive, call the finding “better probabilistic prediction with unresolved mean contribution.”

**Proposed wording:** “The mean-only blind-spot claim requires an equivalence result, not failure to reject a mean gain. Otherwise report the distribution gain without claiming that mean-based evaluation cannot see it.”

The elementary ±a construction remains a valid illustration, but cannot substitute for this empirical distinction.

### 3. Proposal 5's learned response projection has a zero-signal solution

**Where:** Proposal 5, “Learn a small projection and adapter.”

A learned projection can map every teacher response to zero. It then looks perfectly appearance invariant and easy for the student to predict. Small response norms also make an unnormalized stability measure look favorable. A later direct-forecasting head could carry all useful information while the response objective has collapsed.

**Required change:** Fix response normalization in training, constrain the projection scale and rank, and require nonzero held-out response variance plus prediction of independent future differences. Evaluate response stability relative to nuisance-only differences at matched response magnitude. Zero-rank is a declared no-response baseline, not a successful invariant representation.

**Proposed wording:** “Use a training-fitted, scale-constrained projection and retain a no-response baseline. A response direction must have measurable held-out variance and predict real future differences; a stable zero vector fails the mechanism.”

This also makes the unspecified “criterion for responses that transfer” more concrete.

### 4. Proposal 3's decision-policy claim is broader than the planned independent teacher evidence

**Where:** Proposal 3, held-teacher-family evaluation and final claim.

One video family and one motion family give one directional family-transfer test. Many ranks, horizons and layers do not create independent teacher families. A rule with several tuned thresholds can fit that small development universe, even if the final source groups are held out.

**Required change:** Describe the one-week result as a pilot of transfer between two declared families. Freeze the entire rule, including shrinkage, rank and distillation weight selection, before exposing the second family's student outcomes. Report both directions only if each rule is fixed independently without looking at its held-out family. A broad “general distillation decision rule” claim requires more independent teacher/task families later.

**Proposed wording:** “The sprint can establish one held-family transfer result. It cannot estimate universal teacher-selection reliability from correlated layers and ranks.”

Also note that history increment beyond current pose/velocity is a candidate utility heuristic, not a necessary condition for helpful distillation. A teacher can help a finite student learn a useful current-state function even when history contains no additional population information.

### 5. Proposal 3's compute-saving and oracle-regret claims need a complete accounting

**Where:** Proposal 3, comparison with short student pilots and selected-policy regret.

Computing cross-fitted teacher scores for every candidate may cost more than brief student training. Counting only neural updates would give the proposed rule an unfair saving. The candidate with the best noisy final score is also an optimistically selected oracle, so regret against that winner is not an unbiased estimate of true utility.

**Required change:** Include rendering, teacher inference, reference fitting, projections, selection and student pilots in the selection-compute budget. Freeze the policy before final outcomes. Report regret against the best *observed evaluated candidate* as a descriptive benchmark, or evaluate the oracle's candidate choice on independent groups. Use paired training seeds and group uncertainty for policy contrasts.

**Proposed wording:** “Any claimed training savings include all selection costs. Oracle regret is relative to the best measured candidate in this finite panel, with its selection uncertainty stated.”

Clarify whether the twelve-configuration cap includes all comparison methods, ranks and teacher settings. Otherwise “12 configurations × three seeds” can silently expand into a much larger method matrix.

### 6. Distributional and stochastic prediction metrics remain underspecified

**Where:** Proposals 4 and 6.

Proposal 4 lists four quantities and two horizons, but does not say whether the two-component mixture is joint or eight independent scalar mixtures. Marginal CRPS cannot establish joint coordination. Combining angular and translational CRPS in raw units makes the result depend on units. Trunk yaw also wraps at the angle boundary.

**Required change:** Declare eight scalar marginals for the first experiment, standardize each score with a training-fitted fixed scale, report raw-unit scores separately, and restrict the claim to marginal uncertainty. Use a circular score or prefix-relative unwrapped angle within a declared range. A joint-distribution claim needs a joint proper score and a corresponding model.

Proposal 6 uses a stochastic MDM but does not define the primary point forecast. Fix the same number of samples and use their mean trajectory, or choose a fixed deterministic decoding rule, before evaluating joint error. Never use the sample closest to the actual future. Report a proper distributional score separately.

**Proposed wording:** “Sample count and aggregation are fixed across arms; evaluation never chooses a sample using the true future.”

## Material feasibility and interpretation risks

### 7. Proposal 6's strongest resource constraint is eligible people, not GPUs

**Where:** Proposal 6, person/activity audit; common scheduling recommendation.

The available archive is not yet known to contain twenty held-out people with walking, two separate query activities, reliable annotations, matched donor support and demonstrably unexposed query motions. Adding held-activity transfer makes this intersection smaller. The proposal correctly has a stop rule, but its scientific success should not be treated as high feasibility until the intersection is counted.

**Required change:** Schedule a metadata-only eligibility audit before any MDM adaptation. Record eligibility after each requirement, separately for training, development and test people. If fewer than twenty qualify, remove this from the sprint's flagship candidates rather than relax the split after seeing outcomes.

**Wording correction:** Replace “previously unseen person” in the opening question with “person held out from adapter training,” unless pretraining identity exclusion is verified too. The later paragraph already makes this distinction; the headline should match it.

### 8. Cross-activity memory still needs a residual-behavior baseline

**Where:** Proposal 6, comparator table.

Correct-person support can outperform donors because of body proportions, posture, handedness, motion amplitude, source-specific tracking or a richer estimate of current state. The proposal has several of these controls, but a strong raw-memory baseline should receive the same twenty seconds and feed learned summaries directly into a query forecaster. Otherwise the new model may win because only it receives a sufficiently expressive support representation.

**Required change:** Make the direct baseline explicit: identical support/query observations, a same-capacity raw support encoder and no frozen prior. Add a query-only model with the same total trainable parameter budget. If this matches performance, the data may contain personal predictive information, but the world-model adaptation claim fails.

The preceding literature is close. Even a 5% forecast gain should be framed as an evidence result about transferable personal information, not as first-of-its-kind personalization.

### 9. Proposal 5 needs a declared no-future matching rule for evaluation pairs

**Where:** Proposal 5, “similar current pose but different histories and later movement.”

Selecting held-out pairs because their futures differ can enrich the endpoint and give an artificially favorable average response separation. This is not automatically input leakage, but it changes the evaluated population. The primary untouched-motion forecast helps, provided it remains independent of pair selection.

**Required change:** Form evaluation pairs from prefix-only distance and fixed source constraints, or explicitly call future-stratified pairing an enriched secondary assay. All methods must receive the identical pairs. Pair construction using future labels must not be described as label-free.

**Proposed wording:** “The primary forecast includes all eligible untouched test motions. The paired secondary test matches using prefix information only; any future-stratified diagnostic is reported separately.”

### 10. Common terminology sometimes overstates independence

**Where:** Proposal 4, “500 independent motion windows.”

Windows grouped by trial are not independent simply because there are five hundred of them. Replace with “approximately 500 windows, grouped by original trials and people; report the number of independent groups.” Trial bootstrap is still weaker than person bootstrap when repeated trials share a known person. The common contract otherwise handles this well.

## Overall assessment

The common evidence contract is unusually careful about the failed historical gate, source versus person counts, synthetic edits and pretrained-model overlap. Its most important missing universal rule is a suffix-invariance audit on actual forecast inputs.

Proposal 3 has the best infrastructure fit but a narrow novelty opening and a small independent decision panel. Proposal 4 has the most serious inference risk: improved uncertainty is easier to establish than information that specifically escapes an adequate conditional-mean test. Proposal 5 needs a noncollapse definition before the response criterion is executable. Proposal 6 could show an attractive real capability, but public-data eligibility and strong raw-support prediction determine whether the one-week plan is viable.

No reviewed proposal is invalid in principle. Each has a clear point where a cheap baseline, an information boundary or a missing-data intersection can defeat the intended headline. The recommended revisions make those outcomes visible before significant GPU spending.
# Applied revision record

The following substantive revisions were applied directly to proposals 3 through 6 after this review. The common causal preprocessing revision remains owned by the root author.

- Proposal 3 now counts all selection costs, caps the complete pilot matrix, treats the best measured candidate as a noisy benchmark, and limits its generalization claim to one directional held-family test. It also states that history information beyond the current state is not a necessary condition for useful distillation.
- Proposal 4 now specifies eight scalar marginal mixtures, fixes scales and angle handling, prohibits best-of-sample evaluation, includes ordinary forward-KL distillation as a strong baseline, and requires an interval-based equivalence test before claiming a blind spot in mean prediction. The text separates independent people and trials from the number of overlapping windows.
- Proposal 5 now freezes a training-fitted projection, requires nonzero response variance and independent predictive value, matches nuisance-change magnitudes, and evaluates the primary endpoint on all eligible untouched test motions. Future-enriched pair diagnostics are separate from prefix-only matching.
- Proposal 6 now counts eligible people after all metadata filters, separates adapter holdout from foundation pretraining exposure, fixes eight-sample mean forecasts for every arm, and includes equally rich raw-support and parameter-matched query-only baselines.

These revisions prevent several false-positive interpretations. They do not remove the principal scientific risks: ordinary distribution distillation may explain proposal 4, Jacobian or relational distillation may explain proposal 5, and cross-activity support may contain too little information for proposal 6.
