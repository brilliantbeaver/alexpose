# Independent evidence and novelty audit

Research memo for proposal selection. Read-only evidence review completed 12 September 2026, local time. This is not a new experiment and does not revise any frozen experimental verdict.

## What the actual results permit

| Evidence | Exact support | Forbidden inference |
| --- | --- | --- |
| Latest expanded run | 1,403 eligible windows, 290 recordings; all 12 stage attempts passed; 50 unique fits | The run was incomplete, or there were three independent endpoint replications |
| RGB learning curve | R² rises from 0.451045 to 0.662830; approximate raw MSE falls 39.5% | Skeleton-only learning saturates, or increasing independent participants caused the improvement |
| Skeleton addition | Endpoint real-minus-RGB is 0.000357099, saved conditional 95% interval [0.000041374, 0.000669686] | Exactly no information, useful student transfer, or a practically adequate effect |
| Frozen decision | The 0.05 gain requirement fails by roughly a factor of 140; `development_stop` | Positive bootstrap fractions override the prospective utility threshold |
| Correspondence control | Mismatched skeleton gain is 0.000602912, 1.69 times the real-skeleton point gain | Mismatch is statistically superior; its paired difference interval is absent |
| Student-input panel | Historical posture-conditioned matched increment 0.001993744, interval [−0.006672679, 0.012608431] | A demonstrated skeleton-only temporal signal |
| Laterality readout | Initialization R² 0.222544 versus 0.100777–0.114209 across five trained conditions | Anatomical information was erased, all nonlinear decoders would fail, or actual clinical laterality was measured |

Evidence anchors: [run analysis lines 11–15, 36–49, 51–84, 92–118](../../../../../notebook_runs/future-innovation/haic-run-02/ANALYSIS.md); [strategy lines 15–22](../../../../../docs/studies/future-feature-prediction/scaling/research-strategy.md); [paper laterality table and preprocessing discussion, lines 273–294](../../../../../docs/studies/future-feature-prediction/manuscript/paper.md). The line numbers above are review references, not Markdown line fragments.

The notebook's saved verification receipt is meaningful, but the original models and predictions are absent locally. The analysis is not a new numerical reconstruction of those artifacts. The 39.5% raw-error change was extracted from the saved SVG and should be replaced by the remote table before publication. Bootstrap uncertainty is conditional on the fitted models. Reused endpoint labels and correlated representation configurations must not be counted as independent experiments.

The main scientific distinction is between three questions:

1. Does skeleton history add information after RGB has already been supplied? The new curve answers this.
2. Can the skeleton history predict useful target information beyond a state summary available to the skeleton student? The corrected expanded experiment remains unrun.
3. Does teaching that information improve an actual neural student? This has not been measured.

The first result cannot substitute for either of the other two. The latest run is a strong reason to stop optimizing its original average target score. It is not proof that a different teacher target will work.

## Data and observation constraints that should change the proposals

I independently read `manifests/gavd/gavd_full_sequences.csv` with pandas. It contains 1,874 rows, 348 unique `video_id` values and 348 unique URLs. Grouping by `video_id` and counting `gait_pattern_annotation` values gives **347 videos with one presentation label and one video with two**. This makes acquisition style a serious explanation for presentation prediction, even under recording holdout. It does not establish that every such classifier uses a shortcut.

The number of distinct recordings for each presentation is: abnormal 117, exercise 98, normal 32, myopathic 30, stroke 19, cerebral palsy 11, parkinsons 11, antalgic 10, inebriated 8, prosthetic 8, style 3, pregnant 2. These are inventory counts, not counts after processing or development splits. Rare classes cannot support broad diagnosis claims in a one-week study. The metadata have no participant-ID column. Paper-level subject counts must not be presented as independently verified subject-disjoint evaluation for this manifest.

The expanded run reserved 43 recordings before development, of which 41 appear to have media available. Their pose eligibility is unknown. Do not use them to choose this portfolio, pilot thresholds, select a teacher, or repair a baseline. Full-body AMASS is the much stronger source for geometry ground truth and known observation interventions. GAVD supplies a real-world observation stress test and, when appropriate, a descriptive presentation task.

Critical implementation facts:

- The cached skeleton covers observed frames 0–31 only. Future pose outcomes require new extraction or ground-truth motion data. See [strategy lines 173–175](../../../../../docs/studies/future-feature-prediction/scaling/research-strategy.md).
- Teacher tokens nominally at frames 38–39 were encoded with frames 0–63. Their physical information support is broader than the nominal target timestamp. This is target contextualization, not evidence of leakage into the student input. See [analysis line 118](../../../../../notebook_runs/future-innovation/haic-run-02/ANALYSIS.md).
- The historical no-skeleton control removes confidence as well as coordinates, while retaining validity. A new coordinate-only claim needs a new matched-quality protocol. See [analysis line 103](../../../../../notebook_runs/future-innovation/haic-run-02/ANALYSIS.md).
- The separate cached-panel audit found 182 raw-versus-valid-conditioned confidence differences and 31 standardized confidence columns duplicated across input blocks. Duplicates alter regularization. See [adversarial review line 12](../../../../../docs/studies/future-feature-prediction/accessibility/review.md).
- A uniform number of video frames is not a uniform physical prediction horizon. Preserve original timestamps, and normalize using only permitted observations.
- Image-derived GAVD coordinates are not metric 3D ground truth. A second model's output or agreement between two estimators is not an independent physical measurement.

## The old portfolio already occupies much of the obvious idea space

| Existing proposal group | Ideas already proposed | What would be a rebrand |
| --- | --- | --- |
| [Round 1](../../../../archive/research-planning/proposals-01) | Intervention fingerprints, physics versus typicality, minimum normalizing edits, coordination graphs, simulator-teacher residuals, depth correction, device-body interaction | A frozen model as a generic scorer, a response Jacobian, a joint graph, another correction or normality score |
| [Round 2](../../../../archive/research-planning/proposals-02/README.md) | SourceSwap, past-only surplus, adaptive examination, counterfactual dose, cycle innovation, side-anonymous asymmetry, structure ladder | Generic view consistency, order shuffling, active sensing, or another controlled-edit localization study |
| [Round 3](../../../../archive/research-planning/proposals-03/README.md) | Cross-protocol perturbation response, future distillation, pose auditor, pre-impact horizon, sparse anchors, full-body surplus, intent frontier | Another uncertainty envelope, confidence auditor, temporary-side tracker, or arm-to-foot predictor |
| [ICLR strategy](../../../../../docs/studies/future-feature-prediction/scaling/research-strategy.md) | Conditional target spectrum, matched-state futures, temporal-support audit, decoder audit, distributional supervision | Renaming any of these and claiming a new direction |

The new set can legitimately develop one of these into a more decisive research object. It should say what changed: the target, falsifier, deployment decision, or evidence requirement. Merely adding LoRA does not supply that change.

## New primary literature makes generic transfer selection insufficiently novel

The strongest new collision is [Information-Theoretic Criteria for Knowledge Distillation in Multimodal Learning](https://arxiv.org/html/2510.13182v1), arXiv 2510.13182. It proposes predicting KD benefit from whether teacher-student mutual information exceeds student-label mutual information. It tests actual students on synthetic, image, sentiment and omics data. Its theorem has a restricted jointly Gaussian linear setting, a sample-to-dimension condition, bounded/aligned teacher predictions and sufficiently small distillation weight. It uses teacher and optimal-student scalar predictions, not an unrestricted theorem about arbitrary embedding information. Nevertheless, it directly occupies the broad claim of predicting whether cross-modal KD helps and selecting teacher modalities. It must be a comparator, not just a passing related-work citation.

[ATLAS: Attainable Target Learning with Student-Observable Graphs for Heterogeneous Cross-Modal Distillation](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7346910), posted 24 August 2026, is an even closer collision for generic attainable target construction. Its primary indexed abstract describes a warm-up student's sample graph and graph-spectral projection of teacher features and decisions onto attainable targets, with EEG-to-ECG and EEG-to-pupil experiments. Direct full-page fetch returned 403, so this audit verified the indexed primary abstract, not the complete method or implementation. Its existence is enough to prohibit claiming student-observable target projection as a new general principle.

[Could Student Selection Be the Missing Piece for Efficient Distillation?](https://openreview.net/pdf?id=x4kreAhV1c), an indexed ICLR 2026 submission, proposes a training-free cross-model neural-tangent-kernel score to predict post-distillation performance. Its role is student-architecture selection; our narrower candidate would fix the student and select temporal supervision. The broad pretraining-selection claim still overlaps. The venue decision was not verified.

Additional comparisons are [MFH](https://arxiv.org/abs/2206.06487), [MST-Distill](https://arxiv.org/abs/2507.07015), [Robust Cross-Modal Knowledge Distillation for Unconstrained Videos](https://arxiv.org/abs/2304.07775), [A Good Teacher Adapts Their Knowledge for Distillation](https://openaccess.thecvf.com/content/ICCV2025/papers/Qian_A_Good_Teacher_Adapts_Their_Knowledge_for_Distillation_ICCV_2025_paper.pdf), and [LogME](https://proceedings.mlr.press/v139/you21b.html). These cover shared useful information, routing, removal of irrelevant modality content, student-adapted supervision, and cheap transfer assessment.

[A Controlled Study of Feature-Based Knowledge Distillation Across Student Designs](https://arxiv.org/abs/2608.08294) also reinforces a design concern: equal auxiliary coefficients do not imply equal gradient scales. Record gradient norms and give comparison methods equal tuning opportunity. Do not convert a larger gradient into a claimed better target.

## Candidate A: predict the value of a temporal teaching decision

**Question.** For a fixed small skeleton student and a fixed training budget, can a cheap, source-held measurement predict whether a chosen teacher layer, time support and target rank improves independent future-motion learning beyond the no-transfer student, including when transfer should be rejected?

**Meaning.** A teacher may be easy to imitate because it repeatedly describes posture. That does not mean teaching it helps predict the next movement. The study measures the *decision to teach*, with actual students as the ground truth, rather than evaluating only compressed teacher features.

**Method.** Use the corrected reference-conditioned spectrum already proposed in docs 07, with posture/quality, order-invariant denoising and local velocity references. Use common raw target units and nesting that rebuilds every map inside training groups. Compare scalar scores, selected ranks and a zero-transfer option. A small first panel should include full target, whitened target, residual target, residual CCA, selected target and no transfer. Cross-modal complementarity, teacher task readout, ordinary teacher prediction, and a short student pilot are strong decision baselines.

**Decisive result.** Hold out a teacher family or a dataset when testing the score. The score must predict *paired downstream student gain*, not merely its own target loss. Evaluate ranking, harmful-choice frequency and realized performance at a fixed selection budget. Include predeclared negative cases, so an always-transfer rule cannot look successful by selection. A useful double dissociation is high teacher R² with no student gain and lower teacher R² with better motion, reproduced beyond one teacher family.

**One-week simplification.** Use AMASS true future joint outcomes and controlled renders from an already usable rendering pipeline. Start with two teacher families and a small target list, reusing teacher encodings across student runs. If paired video generation or teacher encoding is unavailable on day 1, this is not a one-week primary choice. Do not describe GAVD's current cache as sufficient for independent future-motion evaluation. The first two days should produce a corrected accessibility screen and a pilot set of actual student outcomes, not only a new eigenspectrum.

**Honest originality judgment.** Moderate at best as an algorithm; stronger as a rigorous empirical finding about temporal supervision decisions. To support ICLR significance it must beat elementary residual CCA, an information-theoretic KD criterion, and simply running a short student pilot. A positive effect on one GAVD cohort would not meet that bar. I would retain this as the evidence-grounded continuation, but I would not rank it the most novel conceptual proposal after the new literature checks.

## Candidate B: establish the information timestamp of a frozen video representation

**Question.** Does a representation assigned to a future frame change when only later video changes, and does removing that later support improve the validity of forecast evaluation without discarding useful motion information?

**Change from docs 07.** The existing strategy already suggests this audit. A stronger contribution would provide a support profile across several public video models and a target-construction rule whose advantage is measured on future-motion decoding and actual students. Merely showing that bidirectional attention reads later tokens is expected and insufficient.

**Minimal experiment.** Keep a prefix and the scored target block identical. Change only suffix blocks using controlled renders with natural continuations and nuisance-matched splice controls. Measure target changes as a function of replacement time. Compare full-context targets, bounded-block targets and position-matched padding conventions. Then ask whether nominally identical forecasting tasks have different effective horizons or different rankings of students. A position or out-of-distribution padding artifact must not be called temporal reasoning.

**Novelty boundary.** [Latent Video Prediction Learns Better World Models](https://arxiv.org/abs/2605.15618) already measures temporal direction and several world-model robustness properties. The distinction would be temporal attribution of a target and its consequence for forecasting, not another reversal benchmark. This is fast and strongly motivated, but only an unexpectedly consequential cross-family result could carry a paper. Otherwise it belongs in the methods appendix.

## Candidate C: evidence-preserving completion, only if the other audit sharpens it

The promising conceptual reversal is to test whether a normal-motion prior makes an estimate look better while removing the unusual movement that should be measured. This differs from trying to normalize gait. However, data-consistent diffusion, anomaly-preserving reconstruction and constrained motion estimation are established. A simple reprojection constraint or nullspace projection cannot itself be sold as novel. The proposal needs a preservation-versus-recovery experiment with known withheld truth, a Pareto improvement from a small adaptation, and a clear distinction from the earlier physics-versus-typicality and pose-auditor proposals. This candidate is being developed independently by the gait-model reviewer; no duplicate final proposal should be added here.

## Decision advice

Use the latest negative evidence as a reason to demand an independently measurable outcome and a strong non-foundation-model baseline. It increases confidence that another global teacher-feature score is the wrong headline. It does not increase the probability that any particular rescue will work.

Among continuations tied directly to the completed run, Candidate A has the clearest executable decision. Its novelty is narrower than the previous strategy suggested because ATLAS and the cross-modal complementarity criterion now need explicit comparison. Candidate B is an excellent cheap audit, but a weaker standalone bet. Candidate C offers a sharper conceptual reversal if the method genuinely preserves unusual evidence while recovering missing structure. None presently supports a calibrated probability of producing an accepted ICLR paper.
