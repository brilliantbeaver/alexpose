# What the model advisors changed, and what they did not establish

> **Stage note.** This document records the proposal and its review before the expanded real-data study. The [current executive summary](EXECUTIVE_SUMMARY.md) and [draft01](../draft01/README.md) describe the completed experiments and revised conclusions.

This record explains the research decisions behind the revised [main review](README.md), [roadmap](research_roadmap.md), and [executive summary](EXECUTIVE_SUMMARY.md). The consultations generated and challenged ideas. They did not train a detector, establish novelty, or validate a clinical claim.

## The consultations were actually run

| Advisor | Interface and recorded identity | Main contribution |
| --- | --- | --- |
| Claude Opus | Three completed calls through the local `claude` CLI. All reported `global.anthropic.claude-opus-4-8`. | Challenged the original plan, reviewed revisions, and critiqued the new neuromechanical direction. |
| GPT-6 Astra | An independent agent explicitly configured as `gpt-6-astra`. | Proposed full-joint versus own-joint predictive benefit and identified the contextual-teacher target as a central interpretation problem. |
| GPT-6 Sol | An independent agent explicitly configured as `gpt-6-sol`. | Checked data feasibility, intervention design, source-level inference, and a practical four-fold evaluation using scarce normals. |
| GPT-6.1 Sol clinical-literature advisor | An additional agent explicitly configured as `gpt-6.1-sol`. | Checked 14 primary clinical/biomechanical studies and the distinction between observable movement and hidden biological causes. |

Claude received the research brief and earlier documents as text, with tools disabled. It did not independently inspect the repository or browse citations. The GPT advisors could inspect the specified local evidence and primary sources. The coordinator verified the main numerical and literature claims and made the final research choices. After the independent first round, advisors received selected competing ideas and objections. Agreement in the later round is therefore not three independent confirmations.

The raw records are retained: [Claude first round](advisory/claude_review.md), [Claude second round](advisory/claude_followup_review.md), [GPT-6 Astra](advisory/gpt_6_astra_review.md), and [GPT-6 Sol](advisory/gpt_6_sol_review.md). Invocation records and CLI responses are in the same folder. **Raw advice contains mistakes and superseded recommendations. The revised documents, not the unfiltered reviews, state the adopted plan.**

## The user's label clarification changed the study

The initial brief considered distinctions between different kinds of non-normal activity. The user clarified the endpoint: the normal folder is normal, and all remaining folders are abnormal gait. That definition now governs every main document.

The additional folders can broaden the abnormal-gait test. They are not healthy controls. The distinction between an abnormal gait label and a confirmed medical diagnosis remains necessary. No advice is allowed to override the requested binary mapping.

The fresh census also changed the feasibility assessment. It found 245 source IDs outside the original selection, no additional normal sources, no source with both normal and another label, and only 758 rows with gait events. A separate file check found local video filenames for 49 of the 245 extra sources. These observations rule out several convenient but unsupported plans, including assuming natural normal-versus-abnormal pairs or plentiful normal gait-phase labels.

## Ideas we accepted

**Use a specific information comparison as the central question.** The earlier plan covered optimization, quality, temporal information, and uncertainty. That was too broad to organize one strong paper. The new comparison asks what other joints add beyond a target joint's own motion. It has a clear intervention, a simple linear baseline, and an outcome that can fail.

**Pair the mechanism study with broader abnormal-category evaluation.** The seven additional folders are a useful test of the fixed method's behavior beyond the original named conditions. They do not create new normal calibration data. Their usable coverage must be measured before claiming an all-folder result.

**Make target construction a first-class control.** A full-context teacher may place information about other joints inside a target embedding. The all-joint predictor could therefore have an advantage even when the target joint itself has not changed. Local coordinate or local-patch targets are needed to interpret that advantage. A four-loss comparison separates prediction changes, target changes, and their interaction descriptively.

**Use paired perturbations and report their limits.** Comparing a clip with a specified edit holds much of the source content fixed. Relative limb timing, whole-body timing, and observation changes test different explanations. They do not establish a medical mechanism or create new clinical subjects.

**Use the limited normal sources more efficiently.** Four-fold source cross-fitting lets each eligible normal source receive an out-of-fold prediction. The primary summary averages within-fold ranking performance rather than pooling uncalibrated scores across fitted models. This remains exploratory because earlier data and results were inspected.

## Advice we corrected or rejected

| Suggestion or implication | Why it was insufficient | Final decision |
| --- | --- | --- |
| Removing quality-related score variation proves that tracking quality caused the discrimination. | Movement can affect tracking quality. Adjustment can remove real signal, and matching is observational. | Use paired interventions for narrow measurement-effect claims. Treat adjustment as a secondary association analysis. |
| The untrained encoder's classification win proves that the normal-only anomaly score loses to an untrained detector. | The untrained result uses a supervised probe with abnormal training labels. | Keep the supervision settings separate. A normal-only untrained score would require its own experiment. |
| Quality adjustment can make the result meaningful regardless of training adequacy. | An uninformative model does not become a valid mechanism test through regression. | Require optimization and held-out information checks first. |
| Calibration arithmetic could serve as the strong paper if the main idea fails. | The finite-sample rank formula is established, and five versus nineteen is not an order-of-magnitude gap. | Keep calibration as an important limitation and supporting analysis. |
| The full-context gain must exceed the local-target gain. | Coordinate MSE and latent cross-entropy have different scales. This comparison has no general interpretation. | Require substantive evidence within each target type, not a cross-scale numerical inequality. |
| A wide interval or failed significance test establishes a negative result. | Such evidence may be inconclusive. | Define a useful effect before evaluation and distinguish exclusion of that effect from insufficient precision. |
| Adding all abnormal folders costs little because labels already exist. | Annotation availability does not guarantee downloadable videos or valid poses. | Profile acquisition and extraction, freeze eligibility rules, and report actual coverage. |
| A generic corruption benchmark or dependence-anomaly method is new. | RobustGait, Beyond Marginals, and the MiLeTS benchmark analysis already cover closely related ideas. | Require a specific gait finding or a demonstrated contextual-target mechanism. |
| Every outcome of the proposed experiment would make a publishable paper. | Undertraining, weak controls, and imprecise estimates can leave no strong conclusion. | Set explicit evidence requirements and stop unsupported claims. |

Claude acknowledged several first-round errors in its second response. That second response still required correction: it compared local and contextual loss gains too directly and sometimes treated inconclusive transfer as a strong negative result. Those statements were not adopted. This is why model consultation was used to generate objections rather than to vote on scientific truth.

## The final choice

The main direction is the matched cross-joint predictive-benefit study, with broader abnormal-category evaluation and observation controls. The specific alternative is a contextual-target mechanism study if it explains a substantive change in the anomaly conclusion. Learned stochastic latents, dense phase supervision, and verified multi-view training are deferred because their supporting data or implementation are not yet established.

The earlier numerical score remains 45/100. The plan is more focused and ambitious, but the empirical evidence is unchanged. A strong workshop submission will depend on the experiments in the roadmap, not on the confidence or number of its advisors.

## The neurological research extension

The user then asked for a direction grounded in the biological origins of stroke, Parkinson's disease, myopathic gait, and cerebral palsy. We ran a third Claude consultation and new GPT reviews. The new [research document](neurological_research_direction.md) contains the accepted synthesis and eight vector illustrations. The raw records are [Claude's neuromechanics response](advisory/claude_neuromechanics_review.md), [Astra's review](advisory/neuromechanics_gpt_astra.md), [Sol's feasibility review](advisory/neuromechanics_gpt_sol.md), and the [clinical study ledger](advisory/neuromechanics_clinical_evidence.md).

The biological studies changed the assumption about direction. Reduced muscle-control complexity or a stable adaptation does not imply less predictable keypoints. We therefore added a prospectively specified, two-sided normal-reference distance over three target regions. The earlier one-sided score remains a comparator. This is a proposed protocol revision before new scoring, not a newly observed improvement.

The near-term contribution remains about observable movement relationships. The longer-term proposal is a forward physical model that retains multiple compatible mechanisms and tests which extra measurement resolves the ambiguity. Generative GaitNet, Bidirectional GaitNet, OpenSim Moco, and newer physiological simulation papers make a generic “world model for gait” novelty claim untenable. The proposed additional contribution must be demonstrated against that prior work.

A further coordinator search found earlier work on probabilistic musculoskeletal modeling, gait-model identifiability, and Bayesian gait-state estimation under different sensor configurations. The final novelty table includes these studies. Uncertainty estimation and sensor choice are established ideas; the proposed contribution needs a specific impairment-ambiguity benchmark and a validated improvement at equal measurement cost.

| New advice or claim | Decision after review |
| --- | --- |
| Use abnormal gait's potentially regular organization as the motivation. | Accepted as a testable hypothesis. A larger predictive benefit alone is not proof of biological compensation. |
| The two matched own/full models already exist. | Corrected. The existing project has JEPA components, but these matched branches and their new scoring experiment still need implementation. |
| Make five-class group separation the primary test. | Rejected for this study. The primary endpoint remains normal-only binary abnormal-gait ranking; recognition and movement annotations are secondary. |
| Call the new cross-fitted study confirmatory. | Rejected. The existing population has already been examined. This remains exploratory internal evidence. |
| Regress away posture, error magnitude, or quality and interpret the remainder as mechanism. | Rejected as a causal argument. Use matched baselines and controlled edits; observational adjustment cannot establish cause. |
| Use a raw 16-joint score as the only baseline for a 12-joint profile. | Corrected. The primary raw-KL comparison now uses identical targets, groups, weights, and source support. |
| Treat all very similar trajectories as indistinguishable causes. | Corrected. Similarity must be assessed relative to a declared observation model and available measurements. |
| Run a directed joint map this week using current predictors. | Deferred. Direction-specific information claims need additional matched context restrictions and optimization. |
| Interpret an absent significant medication effect as no coordination change. | Corrected in the clinical discussion. A small study's nonsignificant result does not establish equivalence. |
| Strength, EMG, and muscle causes can be inferred uniquely from GAVD poses. | Rejected. The video observations leave those causes underdetermined; external measurements are needed. |

Independent review also tightened reference estimation, minimum target support, source-level aggregation, and the four-loss decomposition. It caught overlapping legends and an ambiguous measurement arrow in the figures; those layouts were revised. All plotted curves and profiles are explicitly illustrative, and all new models and experiments are explicitly proposed.
