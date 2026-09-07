# Manuscript, literature, and workshop review

Reviewed September 6, 2026. This is an internal editorial and adversarial review, not an acceptance prediction. It supersedes this file's September 5 venue and framing assessment. The source BrainBodyFM manuscripts and notebooks were not edited.

## Recommended position

The strongest distinct-from-laterality submission is a position paper about the evidence that should accompany movement features when a health agent uses them. A suitable title is *Before Health Agents Interpret Movement: Evidence from a Gait Representation Case Study*. The position should be stated directly: a movement summary supplied to a health assistant should identify what was measured, whose recordings contributed, and which task has established its usefulness. An embedding-similarity score alone cannot substantiate a statement about a person's mobility.

The cleanest empirical example holds the recordings, stored vectors, and model checkpoints fixed while changing only the averaging rule. Sixty-four normal-annotated validation clips yield cross-checkpoint cosine similarity of 0.89 with equal clip weights and 0.70 with equal upload weights. One upload supplies 60 clips, receiving about 94% of the first average and 20% of the second. This is a transparent demonstration of how a summary can mainly describe one recording. Neither number measures clinical stability, preserved normal-gait function, or decline in a patient.

The classifier comparison provides a complementary task-level check. Simple pose summaries classify 10 of 20 held-out videos correctly, compared with 6 of 20 for learned features and 6 of 20 for landmark availability. Balanced accuracies are 0.44, 0.26, and 0.25. These are descriptive results from one split and one initialization, with a mismatch between validation and test aggregation. They justify retaining direct-input comparisons; they do not establish a general failure of JEPA or the causal effect of pretraining.

A paper organized around these examples is more coherent than one attempting to cover source transfer, continual learning, forecasting, and clinical utility as four equal empirical contributions. Forecasting should remain a short boundary or follow-up requirement because no qualifying result exists.

## Documents reviewed and how to reconcile them

The review read the current GenAI4Health paper and extended abstract, their revision strategy and existing review, and the substantive BrainBodyFM Markdown manuscripts: the original draft, V2, V3, the readiness guide, and the explanatory paper. The explanatory paper's methods, notebook map, and historical corrections are useful for identifying which evidence belongs to which experiment. Its internal readiness scores are judgments, not experimental measurements or acceptance estimates.

| Source | Role in this revision |
|---|---|
| BrainBodyFM V3 | Latest refined laterality-centered manuscript. Preserve it; do not repackage its main results as a new GenAI4Health contribution. Its appendix provides the concise classifier comparison. |
| Original BrainBodyFM draft and V2 | Historical writing and methodology context. Their excessive precision, study labels, and broad claims should not be imported. |
| BrainBodyFM explanatory paper and readiness guide | Detailed tutorial and evidence history. Some artifact-availability statements describe an earlier checkout state and require current verification. |
| Existing GenAI4Health manuscript and abstract | A valid starting position, but overemphasize local execution details and repeat broad evidence boundaries instead of making the most informative finding easy to understand. |
| GenAI4Health numerical supplement and notebook reviews | The appropriate place for detailed traceability and independently checked calculations. Their presence does not establish independent retraining, preregistration, or historical test secrecy. |

The 625-clip/93-video, five-fold/five-initialization experiment is the separate laterality study. The non-laterality encoder/readout evidence concerns 639 clips from 97 videos and one completed split. These totals, objectives, model sizes, and evaluation repetitions must never be exchanged. The new paper can acknowledge overlapping recordings and the narrower classifier coverage without organizing its argument around “Study A” and “Study B.”

## Workshop fit and current requirements

The best fit is the workshop's trust/evaluation topic, with ambient health as the prospective application. The current evaluated component is non-generative; the argument concerns the information a future generative assistant would consume. That connection is defensible as an evidence-grounded position, but remains a relevance risk and should be stated without implying an implemented agent. The audience includes ML and healthcare researchers, clinicians, and policy specialists. [Official call](https://genai4health.github.io/2026-NeurIPS/).

The live call lists research papers up to nine pages, demonstration and position papers up to five, excluding references and appendices. It does not list an extended-abstract track. The deadline is now September 9, 2026, 23:59 AoE. Submissions are anonymous, use the unmodified anonymous NeurIPS 2026 package, and have no rebuttal. Position papers should support a clear thesis and engage counterarguments. Therefore, keep the extended abstract as a companion synopsis, choose one submission, and avoid treating near-duplicate versions as separate works. [Submission requirements](https://genai4health.github.io/2026-NeurIPS/).

This assessment does not authorize submission or resolve the project's recorded ethics, data-use, or derived-pose release decisions. Authors must also verify concurrent-submission obligations, authorship, the exact portal, and the final PDF before uploading.

## Contribution and novelty calibration

### What is specific and useful

The same-vector weighting comparison is an interpretable reanalysis with only one changed analytic choice. It reveals why the number of clips alone is an inadequate description of the supporting sample. The task comparison then asks whether features are useful for a stated prediction problem, rather than equating representational stability with usefulness.

The proposal can make these observations actionable through a short evidence record that accompanies a movement summary. A worked example should contain actual study quantities while making clear that the runtime interface has not been implemented or user-tested:

| Record field | Worked example for this study |
|---|---|
| Quantity | Same-clip feature similarity between the normal-only and final encoder |
| Observations | 64 fixed clips from five validation uploads |
| Averaging rule | Equal upload weight; also disclose the clip-weighted result |
| Concentration | One upload supplies 60 of the 64 clips |
| Task evidence | No fixed functional-retention endpoint measured across the two encoders |
| Supported interpretation | Coordinates changed during model training |
| Clinical interpretation | No evidence here about longitudinal change in an individual |

This record is a specialization of established model reporting, not a new clinical validation framework. Its prospective value is that a downstream consumer would have enough information to distinguish a model-comparison statistic from a patient measurement. Whether it actually changes clinician or agent behavior remains an empirical question.

### What is established prior art

Model cards already address intended use, evaluation conditions, metrics, and caveats. Digital-measurement frameworks already distinguish sensor verification, analytical validation, and clinical validation. Grouped evaluation, simple baselines, and leakage prevention are also established. These must be acknowledged explicitly. The new contribution is the worked movement example and the concrete interpretation it supports, not the invention of these principles. [Model Cards](https://arxiv.org/abs/1810.03993), [V3 framework](https://www.nature.com/articles/s41746-020-0260-4).

### Claims that remain unsupported

Do not claim a validated biomarker, clinical-grade gait measurement, patient-level generalization, successful continual-learning repair, a causal explanation for the baseline ordering, improved clinician trust, an implemented health agent, or a forecasting world model. Also avoid presenting missingness performance as proof of a learned shortcut. A lower linear-probe score cannot establish that information is entirely absent from an encoder.

## Editorial changes with the highest value

The abstract should introduce the prospective movement-input problem, state the position, then use only the decisive evidence: the 0.89 versus 0.70 same-vector comparison and the dominance of one upload. The classifier can be summarized in one plain sentence rather than listing every metric. It should close with the bounded contribution, without adding a formulaic slogan.

The methods need one connected account: GAVD is a curated resource; this subset contains normal and four condition annotations; pose quality filtering retains 639 clips from 97 videos; training and validation use 59 and 18 videos; testing uses 20. The model is a compact S-JEPA-inspired adaptation, trained to predict hidden pose features. Exact architecture and loss details belong in the appendix. GAVD's clinician-informed curation deserves credit, while the project must not imply independent verification of diagnoses.

The results should explain weighting before relying on the term “source.” Use “original video” or “upload” initially. Show the five upload contributions and two resulting averages in a clean vector graphic. Keep the classifier as a small table using correct-video counts and balanced accuracy, with macro-F1 available in the supplement if needed. Use two decimal places for displayed scores and whole percentages for the concentration example. Preserve higher precision in machine-readable verification, where it is useful.

The discussion should connect the worked record to a concrete proposed use, such as a clinician reviewing a movement summary after a perception-model update. Separate changes in model coordinates from changes in observed movement. A proposed human-factor experiment could compare interpretation of the same summaries with and without the evidence record, but must be clearly identified as future work.

The appendices should retain the cohort/role counts, exact trained implementation, readout definition, and the aggregation mismatch. Remove prose about local folders, protocol names, file digests, execution counters, and archived experiments unless it directly changes interpretation. The detailed review directory can preserve the forensic history.

## Primary literature verified for the revised argument

| Reference and bibliography key | Narrow supported use and limitation |
|---|---|
| Mitchell et al., *Model Cards for Model Reporting*, FAT* 2019, DOI 10.1145/3287560.3287596; `mitchell2019modelcards` | Model documentation can report intended use, evaluation populations and conditions, metrics, and caveats. Our evidence-record proposal should acknowledge this precedent. No clinical benefit from the proposed record is established. [Author paper](https://arxiv.org/abs/1810.03993). |
| Goldsack et al., *Verification, analytical validation, and clinical validation (V3)*, npj Digital Medicine 3:55, 2020, DOI 10.1038/s41746-020-0260-4; `goldsack2020v3` | Distinguishes the kinds of validation needed for digital measurement tools. Use as established clinical-measurement context, without claiming that the current gait pipeline has completed those stages. [Primary article](https://www.nature.com/articles/s41746-020-0260-4). |
| Vasey et al., *DECIDE-AI*, Nature Medicine 28:924–933, 2022, DOI 10.1038/s41591-022-01772-9; `vasey2022decideai` | Supports attention to actual performance, safety, and human factors in early live clinical evaluation. It is reporting guidance for those studies; following a checklist alone does not establish methodological quality, and this retrospective representation study is not such a live evaluation. [Primary article](https://www.nature.com/articles/s41591-022-01772-9). |
| Endo et al., *GaitForeMer*, MICCAI 2022, LNCS 13438:130–139, DOI 10.1007/978-3-031-16452-1_13; `endo2022gaitforemer` | A positive counterexample: it reports improved prediction of clinician-rated MDS-UPDRS gait severity after pretraining. Its pretraining combines self-supervised forecasting with activity-label supervision, and its clinical target and evaluation differ from this case. Do not compare its score directly with ours or describe its full pretraining as label-free. [Author paper](https://arxiv.org/abs/2207.00106). |
| Assran et al., *I-JEPA*, CVPR 2023; `assran2023ijepa` | Predicts target representations and explicitly describes the approach as non-generative. Supports the method description, without implying a health agent or clinical capability. [Author paper](https://arxiv.org/abs/2301.08243). |
| Assran et al., *V-JEPA 2*, 2025; `assran2025vjepa2` | Separates action-free representation pretraining from additional action-conditioned world-model training and robotic planning evaluation. This illustrates the extra evidence needed for planning, not a capability inherited by our model. [Author paper](https://arxiv.org/abs/2506.09985). |
| Ranjan et al., *Computer Vision for Clinical Gait Analysis: A Gait Abnormality Video Dataset*, IEEE Access 13:45321–45339, 2025; `ranjan2025gavd` | GAVD contains curated online recordings with clinically informed annotations. The project-specific 639/97 subset is not the whole dataset, and category annotations do not supply longitudinal outcomes for this study. [Author paper](https://arxiv.org/abs/2407.04190). |

The four new bibliography entries above were added without rewriting existing references. Other existing citations remain appropriate within their previously documented scope: S-JEPA for skeletal latent prediction, VICReg for variance/covariance regularization, and grouped-evaluation/leakage literature for familiar evaluation safeguards.

## Adversarial review: objections the paper must answer

**“This is ordinary evaluation practice illustrated by an underperforming small model.”** That is a serious novelty concern. The response is to acknowledge the established practices and center the unusually interpretable same-vector weighting example. A position contribution can explain the consequences for movement inputs to health agents without alleging a new algorithm or a general failure of predictive learning.

**“Equal upload weighting is not automatically better.”** Correct. The two averages answer different questions. Equal upload weighting matches a random-upload interpretation; neither average represents equal patient weighting without reliable identity information. The paper should report both and specify its intended unit, rather than naming one universally correct.

**“Cosine was never a clinical measure.”** Correct, and no clinician misinterpretation has been observed here. The position concerns how the score should be communicated if used downstream. Avoid claiming demonstrated false reassurance, observed clinical harm, or measured distrust. The useful statement is that the available quantity does not answer a clinical-stability question.

**“The classifier comparison is procedurally imperfect.”** Disclose the validation/test aggregation mismatch near the result and treat the comparison as descriptive. The weighting argument survives this weakness because it uses the same stored vectors. Neither post hoc confidence intervals nor extra decimal places can recover absent split and training variability.

**“The paper says little about generative AI.”** The current model is non-generative and there is no completed agent. Make the intended interface explicit and small: future health assistants may consume movement summaries, and their interpretation should remain limited to the evidence supporting those summaries. This is a position-track argument, with weaker fit to a frontier-methods research paper and no fit to a functioning-system demonstration.

**“Positive gait-pretraining work contradicts this position.”** GaitForeMer provides useful evidence that pretraining can help a clinical gait task. Acknowledge it. Its success is compatible with the claim that each proposed use needs its own task and population evidence; the present small run cannot arbitrate the value of an architecture family.

**“Another reporting form adds overhead without helping.”** Keep the worked record short, reuse available evaluation metadata, and propose a test of whether it reduces unsupported interpretations. Avoid assuming that documentation by itself improves human decisions.

## Bottom line

The position-paper route is defensible, while novelty and direct GenAI relevance remain material review risks. The revised argument should be constructive: show exactly how a movement statistic changes with its supporting sample, pair that observation with a task-level check, and demonstrate how to report its limited meaning. No writing change can supply missing clinical outcomes, external validation, a health-agent evaluation, or repeated encoder runs.
