**Independent advisory review by GPT-6 Astra. November 14, 2026.**

These proposals are hypotheses, not new measured results. I reviewed the advisory brief, both archived documents, and the target construction in `aaai_gait/model.py`. I did not train models. Following the user's clarification, the binary endpoint is **normal-folder gait versus gait in every other folder**. Exercise, style, and pregnant belong to the abnormal endpoint under this definition. This endpoint does not establish pathology or verified diagnosis.

My recommendation is to investigate one claim: **Abnormal-labeled gait reduces the predictive benefit of other joints, even when a joint's own motion remains predictable.** This is more memorable and falsifiable than “JEPA struggles on uncontrolled video.” A suitable provisional title is “When Joints Stop Explaining Each Other: Predictive Coordination for Gait Anomaly Detection.” Do not use that title as an established result until the experiment supports it.

**1. Three ranked paper concepts.**

**First: Measure the additional predictive value of other joints.**

The central claim is that subtracting a same-joint prediction loss from an all-joint prediction loss isolates useful relational information that raw prediction error mixes with individual motion difficulty. Train both predictors on normal sources. A joint that is unusual but internally smooth might be easy to interpolate from its own observations while contradicting the relationships learned between joints.

The difference from prior work must be stated narrowly. Conditional skeleton likelihood is established by [SeeKer](https://arxiv.org/abs/2506.18368), while [Beyond Marginals](https://arxiv.org/abs/2509.15033) already separates temporal and inter-variable dependencies. The contribution would be a tested gait-specific prediction contrast, accompanied by controls showing exactly what information causes its response. Neither dependence anomalies nor subtracting a reference score is a new general idea.

The usable GAVD asset is synchronized joint trajectories in the existing pose cache, together with diverse abnormal categories for a transfer challenge. Contact annotations and verified camera pairs are unnecessary. The decisive experiment is a paired comparison of the contrast against raw discrepancy, followed by a controlled disruption of relative joint timing while retaining the individual trajectories. Include a local-target control because contextual teacher embeddings can themselves create apparent relational dependence.

The concept fails as an improvement if the full predictor gains no normal-validation predictive benefit, if the contrast responds mainly to missingness, or if its paired interval excludes the prespecified useful discrimination gain. It is feasible in six days with two small predictors and existing cached sequences. Its feasibility does not depend on obtaining all 245 additional sources.

**Second: Determine whether contextual targets manufacture apparent coordination sensitivity.**

The claim is that some apparent response to changed context comes from the teacher incorporating that context into its target, rather than from predicting the masked joint's movement. In the implementation, `target_encoder(x)` receives the entire observed window. Changing another limb can therefore change a target embedding even when the target joint patch is identical. This is intended contextual representation behavior, not automatically a software defect.

The methodological contribution would be an explicit target-sensitivity decomposition, separating how an intervention changes the prediction from how it changes the target. Existing [physical-surprise criticism](https://arxiv.org/abs/2611.01815) establishes the broad interpretability problem; this proposal identifies a concrete architectural mechanism to test. I have not verified priority for this particular decomposition, so it must not be advertised as the first analysis of contextual targets.

GAVD supplies real pose trajectories for perturbations and the saved model family. Hold a target patch fixed, replace eligible context, and evaluate four losses combining original or modified prediction with original or modified teacher target. Repeat with a local coordinate target, an untrained encoder, and an adequately trained encoder. Preserve the observation support and use interventions within the same source where possible.

The decisive outcome would be a large contextual-target response that explains a reproducible discrepancy between latent surprise and coordinate prediction. It fails if target context sensitivity is negligible, or if accounting for it leaves the substantive interpretation unchanged. This is computationally the easiest concept. I rank it second because a mechanical property of attention becomes a paper only when it demonstrably changes an anomaly conclusion.

**Third: Test transfer from named conditions to the broader abnormal-gait taxonomy.**

The claim is that performance measured on the original five labels misstates broad abnormal-gait detection, and that relational scores transfer differently from marginal motion scores. Freeze selection using the original development cohort and evaluate additional abnormal folders without retraining or score-direction changes. Report category-specific transfer, including exercise, style, and pregnant as abnormal-labeled gait.

The contribution would be a reproducible source-audited transfer protocol and a supported explanation of transfer failure or success. Merely adding labels is too weak. GAVD's 245 sources outside the selected classes are the relevant asset, but they are annotation references until usable cached media and poses are verified. Their absence of normal labels is a substantial limitation.

The decisive experiment compares fixed methods across original and newly evaluated abnormal categories against an explicitly reused normal reference. It fails as a generalization claim if results track acquisition differences, if usable category support is negligible, or if the normal-reference reuse is concealed. Six-day feasibility is moderate only for already cached data. This direction should support the first concept rather than displace it. The main binary evaluation should cover every folder with usable data, with explicit missing-category and attrition reporting.

**2. The strongest mechanism and the weakness of the previous framing.**

The archived review is appropriately skeptical, but “evaluate optimization, quality, and temporal information” is a program of work rather than a paper thesis. Its ordered-versus-orderless contrast also changes ordinary within-joint predictability. A difference need not reflect coordination. Narrowing the information restriction to other joints gives the comparison a clearer interpretation.

For target joint patch `j,t`, construct two predictions of the same target `y[j,t]`. The self predictor sees only observed patches of joint `j`. The full predictor also sees other observed joints. Both retain joint identity and temporal positions. Use identical uniform random masks, valid targets, training sources, and loss scaling. The attention restriction must operate before contextualization and remain enforced through every layer. Pooling already contextualized tokens would leak the excluded information.

Define:

`C(x) = mean_valid_targets[loss(y, prediction_full) - loss(y, prediction_self)]`.

A larger value means less predictive benefit from other joints. Fix this direction in advance. Report both constituent losses so that cancellation cannot hide two failing predictors. Equal parameter counts and matched optimization help, but held-out normal predictive performance must establish that the full model actually exploits its additional information.

The primary mechanistic target should be a fixed standardized coordinate patch, or a frozen local patch embedding that cannot see surrounding tokens. Fit standardization on normal training sources. Repeat the contrast with the existing contextual JEPA targets as a secondary comparison. A local linear embedding largely changes the coordinate metric; it does not create a novel learned representation. If coordinates win, write a paper about predictive information and scoring, without claiming a demonstrated JEPA advantage.

The current latent cross-entropy is over feature channels. Its difference is not a skeleton likelihood ratio or a causal measure of neurological coordination. A Gaussian conditional-density extension could produce a density ratio between two specified conditional models, but it introduces variance fitting and is unnecessary this week.

This framing is ambitious because it demands both sensitivity and specificity: the score must react to disrupted relationships, and that reaction must help on real abnormal labels. Synthetic timing changes alone are not disease simulations. The [MiLeTS 2026 study on univariate anomalies](https://kdd-milets.github.io/milets2026/accepted%20papers/1camera_ready_paper_anomalies_mostly_univariate.pdf) already uses marginal-preserving corruptions to investigate dependence. The distinctive contribution must be what the controlled analysis establishes about gait and contextual prediction.

**3. A small experiment matrix and a credible negative result.**

Use one fixed development split with two initialization seeds initially. Keep split and initialization identifiers separate. Expand only the selected comparison to the existing three source partitions as explicitly exploratory replication.

| Experiment | Required comparisons | Decision it resolves |
| --- | --- | --- |
| Training gate. | Existing JEPA at 64, 256, 1,024, and, if needed, 4,096 updates; one prespecified EMA warm-up control. | Is inadequate optimization still a sufficient explanation? |
| Information gate. | Local-target self predictor, full predictor, and their contrast; identical masks and training budgets. | Do other joints provide measurable predictive benefit on normal validation? |
| Mechanism test. | Original inputs, matched timing disruption, and observation deletion; local versus contextual targets. | Does the contrast respond to relational evidence or target/context construction? |
| Detection test. | Contrast, its two constituent losses, adequately trained JEPA discrepancy, normal embedding distance, and one normal-only coordinate baseline. | Does the mechanism improve the defined binary endpoint? |
| Transfer test. | The fixed leading comparison on usable additional abnormal folders. | Does the finding extend beyond the original named conditions? |

Use ridge prediction with and without other joints as a cheap, indispensable baseline. If it reproduces the gain, the finding concerns the scoring principle rather than transformer capacity. Include SeeKer only if a faithful adaptation can be validated quickly; otherwise state that the coordinate comparator is a baseline inspired by conditional prediction, not a reproduction of SeeKer.

For timing interventions, prefer longer cached sequences with overlapping crops. Shift a limb trajectory relative to the others, use a common interior support, and avoid circular wraparound discontinuities. Compare with a synchronous crop shift, which preserves relative timing. Document boundary exclusions and newly absent observations. Do not claim that individual dynamics are preserved exactly unless the transformation and evaluated support establish that property. Neither transformation provides a verified pathological counterfactual.

A sensible example of a minimum useful effect is an AUROC gain of 0.05, chosen before final evaluation and justified as an editorial research target rather than clinical utility. Estimate paired uncertainty for the same source-weighted endpoint. An interval whose upper bound is below 0.05 excludes that gain under the sampling assumptions. An interval spanning harm and a 0.10 improvement is inconclusive, even if its point estimate is zero.

A meaningful negative result needs a successful information gate, a clear response to the controlled intervention, a stable normal-validation learning curve, and replication of the relevant pattern. If the full model never learns to use other joints, failure to detect abnormal coordination is an implementation or information limitation. If it demonstrably uses them, responds selectively to timing disruption, and still offers no useful label discrimination with a sufficiently narrow interval, the negative finding is scientifically stronger: the tested relational information is insufficient for this endpoint. That still does not automatically make a strong paper.

For a strong method paper, require all three: demonstrated normal predictive benefit, selective timing-intervention sensitivity after the local-target control, and a broad-label discrimination gain meeting the chosen useful-effect target with a paired interval excluding zero. Require category-level evidence that the gain is not entirely one acquisition subgroup. If these conditions fail, claim a strong mechanism paper only when the target-sensitivity experiment identifies and replicates a specific explanation that changes an empirical conclusion.

**4. Evaluation discipline and the confounds that could defeat the paper.**

All fitting and model selection for the main detector should use normal training and normal validation sources. If abnormal validation labels select architecture, hyperparameters, or score combinations, describe the resulting protocol as anomaly-informed selection. Keep supervised five-class RF and encoder probes in a separate secondary analysis. Their abnormal training labels make their AUROC an inappropriate normal-only performance claim. All eleven abnormal categories are unseen during normal-only fitting. The seven additional categories are unseen categories specifically for the supervised or condition-adapted comparator trained on the original four abnormal conditions. Do not conflate these two meanings of unseen.

Aggregate windows to sequences and sequences to sources using a prespecified rule, such as means at each level. Report source-level binary AUROC and AP, per-category support, and source-clustered paired intervals. For category-specific analyses of sources containing several abnormal labels, retain source clustering across all their rows. Do not treat repeated windows or the overlapping three splits as independent samples. AP must accompany the abnormal prevalence at the identical aggregation unit.

The shortage is normal sources, not abnormal windows. Additional abnormal folders cannot repair a five-source normal calibration set. No useful 5% false-positive guarantee follows from this dataset configuration. Nineteen exchangeable calibration units merely permit a finite threshold under the existing rank rule; they do not establish precise performance.

All original tests have been inspected. Preserve their old results and explicitly label new analyses exploratory. A reshuffle does not erase inspection. Audit whether newly selected sources were previously used anywhere relevant, freeze their manifest before evaluation, and separate genuinely new abnormal-source evidence from reused normal-reference evidence. The latter means the combined AUROC is not an entirely untouched confirmatory evaluation.

The most serious scientific confound is common measurement error: camera motion, normalization, and pose tracking can create cross-joint dependence without biological coordination. Source-held-out evaluation and high-coverage subsets help, but neither proves a physiological mechanism. Shared normalization can leak information across joints even when attention is restricted; define identical fixed normalization and describe the remaining conditioning explicitly. Missingness order, source-specific category labels, short windows, attrition, and unknown repeated people remain material limitations. Camera-view metadata do not establish paired views or participant identity. Sparse event annotations cannot validate phase inference across normal gait.

The recent [JEPA driving study](https://arxiv.org/abs/2608.01336) already challenges domain-confounded novelty scores. [FSGait](https://openaccess.thecvf.com/content/ACCV2024/html/Duan_FSGait_Fine_Grained_Self-Supervised_Gait_Abnormality_Detection_ACCV_2024_paper.html) and [GenGait](https://arxiv.org/abs/2604.01997) already occupy predictive and reconstruction-based gait anomaly detection. These verified precedents make a generic negative-result or robustness paper insufficiently distinctive. My targeted search does not establish exhaustive novelty, and any stronger priority statement remains unverified.

**5. Six days with decisions that prevent an unfocused submission.**

On November 14, freeze the claim, target definition, masks, metrics, effect threshold, and source-use ledger. Audit cached coverage of additional folders. Stop cohort expansion if it requires substantial new extraction.

On November 15, complete the optimization and normal-information gates. If neither the neural nor ridge predictor benefits from other joints, abandon a coordination-detector claim and assess whether contextual-target sensitivity supports the second concept.

On November 16, complete the paired timing and observation interventions. If the effect disappears with local targets or common observation support, make that mechanism the finding rather than disguising it as robustness.

On November 17, lock implementations and evaluate the original exploratory cohorts plus the fixed usable transfer cohort. Freeze analysis choices afterward. Distinguish uncertain results from evidence excluding the useful gain.

On November 18, write around one supported finding, including unsuccessful controls, source counts, and paired intervals. Retain classification only if it answers a specific secondary question.

On November 19, audit every claim, reproduce the main table from saved predictions, verify anonymity and the submission cutoff, and prepare the final manuscript. If the evidence remains inconclusive, submit a candid exploratory workshop study rather than a purportedly validated new detector. November 20 should be reserved for submission, not discovering the central result.
