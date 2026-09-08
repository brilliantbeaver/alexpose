# From gait symmetry to useful movement prediction

Research overview and workshop assessment, 7 September 2026.

## The question connecting the work

Can we make self-supervised learning preserve useful information about how a body moves, and show that this information improves prediction on recordings excluded from training?

Our inputs are sequences of estimated body landmarks, such as shoulders, knees, and ankles. An encoder converts these coordinates into numerical features. During JEPA pretraining, a predictor uses features from the visible input to predict features of a hidden region, with a separate, slowly updated encoder providing the training targets. We then test whether the learned features help a simple prediction model estimate movement. Predicting hidden features successfully during training and improving a useful movement prediction are separate outcomes.

The new tutorials investigate where that connection might succeed or break down. They examine input preparation, the choice of hidden body regions, explicit reflection training, and prediction of future movement. The examples below illustrate these questions; only results identified as measured findings should be read as evidence about the gait recordings.

## How our research has developed

Our starting point was domain-informed pose learning: use gait-relevant landmarks to guide the hidden-feature prediction task. This provided a reasoned design choice, but the original experiment used that same landmark selection throughout. It could not establish whether the selection itself helped.

We then made left–right movement a testable question. Here, laterality means a signed contrast between movement on the two anatomical sides. For illustration, if a clip has a contrast of +0.2, reflecting its coordinates and exchanging the left/right landmark labels should produce −0.2. This expectation follows from the measurement and transformation; it does not assume that the person's gait is symmetric.

The [completed evaluation](../05_aggregate_statistics.ipynb) used 625 accepted gait clips from 93 source videos in the Gait Abnormality in Video Dataset (GAVD). We divided the videos into five groups, kept each test group separate from encoder training and prediction-model fitting, and repeated training with five random initializations. Two training recipes gave 50 trained encoders, each with a matched untrained reference. This is substantial controlled experimentation, although different videos do not necessarily identify different people.

The central finding was that adding mirrored training clips improved the specified agreement between original and reflected features, while its predictive benefit remained uncertain. The estimated improvement in predictive fit was about 0.004 in R², with a 95% interval from −0.006 to 0.013. Both trained recipes still had worse feature agreement than their matched initial encoders under the particular reflection rule tested. Pretraining also did not establish better movement prediction than initialization. These findings are narrower than saying that JEPA cannot learn useful geometry.

A rule imposed on the final prediction can guarantee the required sign reversal, even with an untrained encoder. That observation motivated the next stage: investigate whether the input, training task, and feature summary preserve movement information that helps prediction. The new notebooks pursue those possibilities through distinct comparisons.

## Four questions the new notebooks help answer

### 1. Does preparing or summarizing the input obscure the movement we want to predict?

[Notebook 07 — Research questions and diagnostics](../07_research_questions_and_diagnostics.ipynb) follows the information from observed landmarks through preparation, encoding, and the final feature summary.

Consider two simple one-dimensional trajectories. A left landmark follows positions (0, 1, 0, 1), while a right landmark follows (0, 0, 1, 1). Both have average position 0.5, although their median movement speeds differ. Exchanging their trajectories preserves both averages and reverses the left–right speed contrast. This constructed example shows why an average position can miss a movement distinction. Averaging encoder features could behave differently if those features already contain motion information.

The notebook also compares the original movement measurement with the same formula applied after input preparation. A read-only calculation gives matching signs in about 70% of cases when each source video receives equal total weight. Both calculations are available for 623 clips from 92 sources; two clips have no finite recomputation. This is a comparison of measurement procedures, not prediction accuracy.

The discrepancy supports investigating preparation, but does not identify its cause or prove that information is irretrievably lost. Further comparisons would separate interpolation from temporal resizing and compare time-averaged features with motion-sensitive summaries. If those changes recover better held-out prediction, they would locate a specific weakness in the current pipeline. Notebook 07 demonstrates the problem and provides the initial diagnostic; it does not yet complete these attribution experiments.

### 2. Does hiding gait-relevant landmarks lead to more useful learning?

[Notebook 08 — Matched-budget masking](../08_matched_budget_masking.ipynb) compares two training tasks. One selects hidden targets from twelve landmarks: the left/right shoulders, hips, knees, ankles, heels, and foot tips. The other selects targets across all 33 landmarks.

A simple example explains the main control. With four time blocks, twelve landmarks provide 48 possible landmark–time regions, while 33 provide 132. Hiding half would give the models 24 and 66 targets, respectively, changing both target location and the amount of visible context. The tutorial instead hides the same actual number in both conditions—24 in this fully observed example—and adjusts that common count when observations are missing.

The models share their starting weights, source-video draws, and training budget. After pretraining, their encoders are frozen, and the same kind of prediction model is fitted using training videos only. Comparison with the untrained encoder tests whether learning added value; comparison with direct movement summaries tests whether these learned features improve prediction beyond readily available pose information.

A reliable predictive advantage for gait-selected targets would support this particular anatomical choice under the tested conditions. It would not establish that anatomy always helps, because different target locations can still differ in difficulty. Several preselected random twelve-landmark groups would help determine whether the chosen anatomy matters beyond using a smaller target set. That additional control is supported by the code but is not part of the current two-recipe grid.

Real-data training has begun, with completed paired runs covering part of the planned evaluation. Full coverage across the five source groups and five initializations remains incomplete at this review, so the available runs do not yet support a full-dataset conclusion.

### 3. Does explicit reflection training preserve useful left–right information?

[Notebook 09 — Symmetry-aware JEPA](../09_symmetry_aware_jepa.ipynb) compares the base training objective, adding mirrored examples, and adding a penalty when corresponding original and reflected landmark features disagree.

For a simple illustration, left and right movement values of 3 and 2 have average 2.5 and difference +1. Exchanging sides preserves the average and changes the difference to −1. Useful features may need to retain both the overall movement and which side contributes more. Making every whole-clip representation identical under reflection could erase the second distinction.

The explicit penalty encourages features at exchanged anatomical landmarks to match; it does not guarantee perfect agreement. The notebook checks prediction separately and examines whether features still vary across clips. Without that check, identical features for every clip could achieve perfect reflection agreement while carrying no information about differences in movement.

Better held-out prediction alongside better agreement would support the proposed training change. Better agreement with unchanged prediction would establish a consistency benefit only; worse prediction would suggest that the constraint or its strength discards useful information. The implemented comparison matches optimizer updates, but the additional reflection calculations cost more, so an efficiency claim would require a separate compute-matched comparison.

Retained executions demonstrate the method on synthetic data. No retained real-data result yet establishes a benefit from the explicit penalty.

### 4. Does predicting future features improve prediction of actual future movement?

[Notebook 10 — Past-only movement prediction](../10_past_only_movement_prediction.ipynb) moves beyond summarizing an entire clip. It observes the first 0.8 seconds and asks about landmark positions 0.25, 0.50, or 0.75 seconds after that observation boundary.

For example, after observing an ankle moving forward, can learned features help predict its subsequent position better than keeping it at its last position or continuing its recent velocity? These simple alternatives are important because a smooth movement can be predictable without representation learning.

The input is prepared using only observations available before the prediction boundary, including the reference position and body scale. A test changes future coordinates and visibility while leaving the prediction input unchanged. This checks that future information has not entered through preparation.

During training, a predictor learns to estimate future-window features. Afterwards, a separate regression model uses the frozen encoder's features of the observed past to predict future landmark positions. The comparison also includes a regression model given the past coordinates directly. Another encoder is trained with futures taken from different training videos. If correctly paired futures improve prediction over this mismatched-future control and the untrained encoder, that would support a benefit from learning the actual temporal relationship.

The measured endpoint is therefore the usefulness of the past representation for forecasting. The notebook does not decode the JEPA predictor's future feature vector into a rollout, simulate interventions, or evaluate clinical outcomes. Its retained training results are synthetic demonstrations; real-data forecasting remains unestablished.

## What these directions could contribute to a paper

The completed study supports a bounded empirical paper about the difference between geometric agreement and useful pretraining. The extensions could support a distinct paper if they complete a new comparison and explain its outcome. Their contribution cannot rest on presenting familiar training ingredients under new names: masked skeleton-feature prediction and motion-aware target selection already appear in [S-JEPA](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04755.pdf), while transformation-aware self-supervised learning has precedent in [SIE](https://proceedings.mlr.press/v202/garrido23b.html).

My assessment is that Notebook 08 offers the shortest route to a new empirical result because its controlled real-data comparison is already underway. Notebook 07 can help interpret that result and identify whether preparation or the feature summary needs attention. Notebook 09 offers a focused geometry question, but still needs real-data evidence. Notebook 10 has the clearest route toward a temporal world-model contribution, with more work remaining before that connection is demonstrated.

A well-supported adverse or inconclusive result can still be informative if the experiment rules out a plausible explanation or identifies a consequential limitation. An incomplete comparison, however, cannot establish either success or failure. Similar point estimates alone do not prove equivalence.

### Current multi-dimensional assessment

The following scores assess readiness for a **new empirical paper based on the extensions**, taking the completed study as background. They are editorial judgments: 1 means a major gap, 3 means promising but incomplete, and 5 means strong support for the intended claim. They are not probabilities of acceptance.

- **Importance and clarity — 4/5.** Whether body-informed training improves useful movement representations is clear and relevant. The eventual paper should choose one primary comparison and answer it fully.
- **Experimental design — 4/5.** Matched initializations, source separation, equal hidden-target counts, and simple baselines address important alternative explanations. Additional anatomy controls and fair compute comparisons are still needed for broader claims.
- **Completed evidence for a new finding — 2/5.** The input diagnostic is complete, but masking covers only part of the planned evaluation; the other training extensions have no retained real-data results. Finish the chosen comparison before ranking methods.
- **Demonstrated novelty — 2/5.** There is a plausible opportunity for a useful empirical finding, but no completed extension yet establishes it. Show what the new evidence adds beyond existing masking and symmetry methods.
- **Evidence beyond the current task and sources — 1/5.** No independent participant evaluation, clinical endpoint, or real-data forecasting benefit has been established. Add the validation appropriate to the chosen claim rather than implying all are required for a narrow workshop paper.
- **Readiness of a focused paper argument — 3/5.** The questions and interpretation rules are clear enough to structure a paper. Its central new result and uncertainty analysis remain incomplete.

These scores should not be averaged into an acceptance estimate. Strong design does not substitute for completed evidence, and a narrow paper may be credible without claiming broad clinical or foundation-model capabilities.

### Which workshops and conferences fit the evidence?

These are topic-fit judgments, separate from readiness and deadline availability.

**NeurReps: Symmetry and Geometry in Neural Representations — 5/5.** This is the closest conceptual fit for the completed laterality study. Its call explicitly invites methods for invariant and equivariant representations, learning group structure, dynamics of neural representations, and applied work in motor control. The reflection test asks a direct representation question: does self-supervised training preserve a known transformation of articulated movement, and does that geometric agreement coincide with useful prediction? The controlled result that these outcomes separate is informative for this audience, even though it is not a general theory of equivariant learning. The non-archival four-page extended-abstract track expressly welcomes early-stage and negative findings, making it a better format for the incomplete extensions than the nine-page archival proceedings track. [Official call](https://neurreps.org/#cfp)

**Foundation Models for the Brain and Body — 5/5.** Its call explicitly includes video-derived pose, movement, and evaluation of whether pretraining helps. Our present question directly matches that audience, without establishing a broadly transferable foundation model. [Official call](https://brainbodyfm-workshop.github.io/call-for-papers.html)

**Physical World AI — 4/5.** Reflection of articulated poses and evaluation of geometry-aware representations fit its geometry and evaluation themes. Real forecasting evidence would strengthen the world-model connection; materials, contact dynamics, and multimodal sensing are outside the current experiments. [Official call](https://physworld-org.github.io/physworld.github.io/cfp/)

**Embodied Spatial Reasoning — 3/5.** The reflection study concerns the geometry of an articulated body, and the planned past-only task asks whether a representation of observed motion helps predict subsequent landmark positions. These are meaningful connections to spatial structure and temporal dynamics. The limits are equally important: the experiments do not model an agent and surrounding objects, object permanence, spatial memory, 3D scene reasoning, interaction, or physics-based simulation. The paper track welcomes analyses and negative results, so the completed reflection study could be an honest fit if framed as a narrow analysis of spatial structure in learned body representations. The incomplete forecasting extension should not be used to claim a world model or an embodied agent. [Official call](https://embodiedsr.github.io/call-for-papers.html)

**GenAI4Health — 2/5.** Careful evaluation is relevant to trustworthy health AI, but the current outcome is a pose-derived movement measure. We have not evaluated a medical generative application or clinical benefit, so a health-centered contribution needs a more direct, evidenced connection. [Official call](https://genai4health.github.io/2026-NeurIPS/)

**Med-Reasoner: Medical Reasoning with Vision-Language Foundation Models — 1/5.** The current work is outside this workshop's central problem. Med-Reasoner seeks medical vision-language models that connect visual findings with clinical knowledge and support interpretable diagnostic or clinical reasoning. Our model receives estimated pose coordinates rather than images and language, and the endpoint is a coordinate-derived movement contrast rather than a diagnosis, clinical decision, or validated medical measurement. The study's separation of source videos and its attention to data governance are useful research practices, but they do not supply the missing vision-language or medical-reasoning contribution. [Official call](https://med-reasoner.github.io/neurips2026/call_for_paper.html)

**IAAI-27: Innovative Applications of Artificial Intelligence — 1/5 today.** IAAI evaluates innovative AI applications in the real world. Its deployed-applications track requires a production system used by end users with meaningful performance data and measurable benefits. Its emerging-applications track still requires early deployment or pilot-stage results and a clear path to full deployment. This project has careful research controls, software checks, and a source-held-out evaluation, but it has no deployed system, intended end-user workflow, pilot, or evidence of practical benefit. The paper is therefore a better fit for a research venue than an IAAI application track. [Official call](https://aaai.org/conference/aaai/aaai-27/iaai-27-call/)

NeurReps' 2026 submission deadline was 24 August AoE, so it is no longer a route for a new submission this year. If a late route becomes available, the completed reflection study should be framed around the geometry of learned representations and the diagnostic value of the negative result. The unfinished masking and forecasting work should appear only as motivated future tests, rather than being presented as a second completed contribution. Its proceedings track would require a self-contained, highly developed paper and is archival; the extended-abstract track carries no dual-submission restriction. [Track and dual-submission details](https://neurreps.org/#cfp)

As of 7 September 2026, Brain & Body's posted paper deadline of 5 September AoE has passed; its strong fit does not establish that a new paper can still be submitted. [Submission dates](https://brainbodyfm-workshop.github.io/call-for-papers.html)

PhysWorldAI lists an archival deadline of 9 September and a later non-archival window of 29 September–29 October, accepting papers up to eight pages or extended abstracts up to four. That later window may provide a practical route for completing a focused comparison. [Submission options](https://physworld-org.github.io/physworld.github.io/cfp/)

Embodied Spatial Reasoning allowed a single non-archival, double-blind paper format of four to eight pages, including preliminary and negative findings, but its deadline was 5 September AoE. Its separate demo track required an interactive embodied system or physically grounded world model, which the present notebook suite does not provide. [Format and policy](https://embodiedsr.github.io/call-for-papers.html) [Demo requirements](https://embodiedsr.github.io/call-for-demos.html)

GenAI4Health's extended deadline is 9 September AoE. Its research track requires evidence supporting the central claim even for work in progress; its position-paper track permits evidence-grounded arguments without new experiments. A notebook roadmap alone would need substantial development to serve either purpose. [Track requirements](https://genai4health.github.io/2026-NeurIPS/)

Med-Reasoner's extended deadline was 5 September AoE. Its long and short papers are non-archival, but its four-page short-paper track would still require a self-contained contribution to medical vision-language reasoning. Reframing the present pose-only evaluation as medical reasoning would make the paper less accurate. A future fit would require a genuinely multimodal clinical question, clinically grounded evidence, and evaluation of the reasoning process itself. The call also requires disclosure of dataset licensing and governance. [Submission requirements](https://med-reasoner.github.io/neurips2026/call_for_paper.html)

IAAI-27 accepts electronic submissions until 8 September 2026 AoE, but the deadline should not determine venue choice. IAAI is single-blind and expects author and affiliation information. Unlike the non-archival workshops above, it does not allow a substantially similar paper to be under review elsewhere during its review period. [Track requirements and submission policy](https://aaai.org/conference/aaai/aaai-27/iaai-27-call/)

### What IAAI would require before this becomes an appropriate application paper

The central gap is an application, rather than another representation-learning result. The following changes are prerequisites for the emerging-applications track; the deployed track needs production use beyond them.

1. **Name one decision and one user.** Define a concrete user, setting, decision, and action following the model output. For example, a movement-research analyst could use a system to prioritize recordings for manual review. This is only a possible direction, not a validated use case. Do not describe the current laterality score as a diagnostic or treatment recommendation.
2. **Build the end-to-end system around that workflow.** The current notebooks are research procedures, not a user-facing application. An application would need an input contract, pose-quality checks, an output that a user can interpret, a defined escalation path for low-quality or uncertain cases, and a record of what was reviewed. For a health-related setting, this design would also need appropriate privacy, consent, licensing, and institutional review.
3. **Show that the system helps in its intended setting.** Evaluate it against the current workflow on data and cases representative of use. Measure outcomes meaningful to that user, such as review time, agreement with an independently defined reference, missed-case rate, or decision quality. A higher R² on the coordinate-derived laterality target alone does not establish application value.
4. **Run a genuine pilot with the people who would use it.** Record the number and type of cases, user interaction, failures, overrides, and the consequences of errors. A source-held-out split establishes a useful research control; it cannot replace field evidence with end users.
5. **Document operation and failure handling.** Report how pose estimation failures, occlusion, different camera conditions, repeated appearances of one person, and distribution shifts are detected and handled. Include monitoring, versioning, rollback or disablement conditions, and human oversight. IAAI values candid accounts of redesign and failure in real operation.
6. **Demonstrate a credible path beyond the pilot.** State the needed integration, compute and latency constraints, maintenance plan, ownership, and deployment partner. A co-author from the organization operating the system would strengthen evidence that the application is real. Do not claim readiness for a clinical deployment without the corresponding validation and governance.

The existing work can contribute to this future pathway by providing an explicit check that an attractive geometric property is not confused with useful predictive performance. For IAAI, that check would become one safety or quality-control component of a larger system, rather than the paper's main claim.

## The next decisions that would most strengthen the work

1. Complete one primary comparison before expanding the method grid. For the current masking experiment, finish the declared source groups and initializations, keep training choices independent of test performance, and quantify uncertainty by resampling whole source videos. A reduced study can be reported honestly as a pilot, but should not be selected because its partial results look favorable.
2. Explain why the result occurred. If gait-target masking helps, use preselected random landmark groups to test the anatomical explanation. If all learned features remain weak, use Notebook 07's proposed preparation and feature-summary comparisons to investigate why. A lower training loss alone cannot resolve either question.
3. Choose validation that matches the paper's claim. For useful temporal learning, complete Notebook 10's real-data comparison against its simple motion and mismatched-future baselines. For generalization to new people or clinical relevance, select an independent dataset and meaningful outcome first. [Notebook 06](../06_external_subject_gate.ipynb) checks external-data prerequisites; it does not perform that evaluation. Another collection scored only by the same coordinate formula would test transfer of that measurement, not clinical validity.
4. Keep the eventual paper focused and submission-ready. Use the research trajectory to motivate one question rather than turn every notebook into a separate contribution. Distinguish the new result from the existing laterality paper, check the selected venue's submission policies, and document the outstanding ethics and data-use determinations before submission. The project record currently marks those reviews unresolved; this is a separate requirement from scientific merit.

The research has progressed from asking whether an encoder follows a known left–right relationship to testing which learning choices preserve information useful for movement prediction. We now have concrete comparisons that can support or weaken those hypotheses. Completing one of them, with its controls and uncertainty intact, would provide a stronger basis for the next workshop paper than broadening the claims around the results already available.
