# Independent advisory brief

The user wants a strong AAAI ASTAD workshop paper using the GAVD data already available. The date is November 14, 2026, and the deadline is November 20. We are revising research documents, not running new experiments in this turn. Give candid, imaginative, technically specific advice. Do not inflate evidence or guarantee acceptance. Use plain language and complete sentences.

## Current evidence

The project has eight runnable tutorials, tested data and model code, and saved real-data experiments. The five classes are normal, parkinsons, stroke, cerebral palsy, and myopathic. The selected annotations contain 666 sequences from 103 source videos. After pose quality checks, the cohort has 450 sequences from 91 sources and 2,504 overlapping windows. There are no verified participant IDs or diagnoses. Splits protect source IDs, not proven people. All three test splits have been inspected.

Random Forest source-weighted macro-F1 is 0.267, 0.482, and 0.333. Validation-selected JEPA scores are 0.244, 0.252, and 0.331. An untrained encoder with a supervised probe scores 0.293, 0.327, and 0.348. Normal-only masked discrepancy AUROC is 0.200, 0.604, and 0.583. Suffix forecasting scores are 0.138, 0.417, and 0.567. Normal embedding distance scores are 0.323, 0.625, and 0.250. Five or six normal calibration sources give an infinite threshold for the finite-sample 5% rule. At least 19 calibration units are needed merely for a finite threshold, not adequate precision.

Normal pretraining used only 45, 66, and 63 optimizer updates. With EMA 0.996, the algebraic initialization coefficients are 0.835, 0.768, and 0.777. Losses still change substantially. Progressive training adds 777, 747, and 825 updates and changes optimizer state. A single positional-noise run performed worse. These observations concern a short training recipe, not a general limitation of JEPA.

The model uses 2D XY coordinates, 16 BlazePose joints (11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32), 32 frames at 15 Hz, four-frame joint patches, width 64, two encoder layers, one predictor layer, an EMA teacher, latent cross-entropy, and VICReg. Uniform random masking independent of motion is a firm requirement for the main model. Missing observations remain missing and are excluded from attention, loss, and pooling. Random masking is completion; the suffix model is separate. The supervised probe uses the online encoder on complete observed windows. Condition labels influence staged sampling. A learned stochastic latent prior and posterior are not implemented.

## Newly verified GAVD opportunities and limits

A fresh scan of the per-condition CSV files found 1,874 sequences, 458,116 rows, 348 source IDs, and 12 labels. The source cache was excluded from counting.

| Label | Sequences | Source IDs |
| --- | ---: | ---: |
| abnormal | 767 | 117 |
| antalgic | 35 | 10 |
| cerebral palsy | 64 | 11 |
| exercise | 234 | 98 |
| inebriated | 23 | 8 |
| myopathic | 188 | 30 |
| normal | 291 | 32 |
| parkinsons | 47 | 11 |
| pregnant | 6 | 2 |
| prosthetic | 39 | 8 |
| stroke | 76 | 19 |
| style | 104 | 3 |

There are 245 sources outside the selected five classes. None contains a normal-labeled sequence. These are additional annotation references, not guaranteed usable media or sources untouched by every earlier local project. No source ID mixes normal with another label. There are 213 sources with multiple camera-view labels, but repeated views of the same person are not verified. Videos can contain different demonstrators.

Only 758 rows contain gait-event annotations: 751 initial contacts and seven other events. Their support by label, shown as rows/sequences/sources, is abnormal 359/34/6, exercise 33/3/1, myopathic 40/7/2, parkinsons 56/2/1, and style 270/34/1. No normal sequence has contact labels. Do not assume abundant phase ground truth or normal contact calibration.

For normal-labeled rows, the dataset field says Normal Gait for 35,975 rows and Abnormal Gait for 5,365. This field alone does not prove acquisition domain, diagnosis, or patient status. Exercise, style, and pregnant are not automatically diseases or guaranteed healthy controls. Do not label every non-normal category pathological. Metadata do not establish causal pairs.

## Prior art already verified

Skeleton S-JEPA, ECCV 2024, established masked skeleton latent prediction. MAMP has a uniform random masking branch. VICReg and StoP are established methods. FSGait, ACCV 2024, uses normal-only reconstruction and prediction with joint and frame anomaly outputs. GenGait, a 2026 preprint, uses a normal-only Transformer masked autoencoder, joint inconsistency scores, and normative reconstructions; its abnormal test patterns are simulated. GaitJEPA, IJCB 2026, uses masked and future latent prediction on silhouettes. SeeKer, ICCV 2025, already uses confidence-weighted keypoint likelihoods. STG-NF, ICCV 2023, models normal skeleton distributions.

Asleep at the Wheel (arXiv 2608.01336) examines domain confounding in JEPA anomaly scores. The study at arXiv 2611.01815 critiques physical-surprise interpretations. Ren and colleagues introduced background-corrected likelihood ratios in 2019. Generic confidence weighting, positional noise, stochastic latents, normal pretraining, or applying an established model to another dataset is insufficient novelty by itself.

## Requested advice

1. Propose three distinct, ambitious, testable paper concepts. For each, state the central claim, the difference from prior art, the GAVD asset it actually uses, the decisive experiment, the failure criterion, and feasibility within one week. Rank them.
2. Choose the strongest direction. Explain whether the previous draft aims too low or asks for evidence that cannot be obtained this week. Offer one specific mechanistic idea where justified.
3. Specify a small decisive experiment matrix, appropriate baselines, evaluation units, and a policy for previously inspected test data. Separate normal-only detection from supervised classification. Explain what makes a negative finding scientifically informative rather than a consequence of inadequate training.
4. Identify the most serious confounds. Distinguish an inconclusive interval from evidence excluding a useful gain. Flag unverified literature claims rather than inventing citations.
5. Give a six-day plan with explicit decision points. Do not assume that new clinical collaborations or data collections are already available. Do not provide acceptance probabilities.

Write about 1,500 to 2,500 words. Avoid em dashes. You are a critic and advisor. The coordinating author will integrate the advice rather than use model agreement as evidence. Do not read credentials or .env files. Do not edit the main documents, train models, or download videos.


You are the Claude advisor. Put special emphasis on a memorable scientific contribution, the risk that a reasonable-sounding plan is still incremental, and what can be demonstrated with GAVD in six days. Critique and improve the prior review below. Do not merely agree with it. You have no tools in this consultation. Treat the supplied measurements and census as evidence; flag any new reference that requires verification.

# Draft 00: Critical assessment of novelty and publication readiness

**Assessment date: November 14, 2026.** This is an internal research review of the eight AAAI tutorial notebooks, their implementation, and the saved real-data experiments. Scores are editorial judgments about the evidence available today. They are not acceptance probabilities or official AAAI reviewer scores.

## My judgment

**The project is a strong experimental starting point, but the current results do not yet support a convincing paper claiming a novel or superior gait anomaly detector.** I would lean toward rejection of that claim in its present form. There is a plausible ASTAD workshop paper in a carefully controlled empirical investigation, provided it establishes a specific finding beyond the observation that a briefly trained model performed poorly.

My current scores are **2/10 for methodological originality, 3/10 for empirical strength, and 9/10 for workshop relevance**. The weighted research-readiness assessment is **45/100**. The substantial work in data processing, tutorials, reproducibility, and controls earns credit, but it cannot substitute for an original, supported scientific conclusion.

The most promising question is: **When does a skeleton prediction discrepancy measure a gait difference, and when does it measure pose-estimation error, recording differences, or incomplete training?** The current experiments motivate that question. They have not yet answered it. A more ambitious method paper could develop a way to separate measurement uncertainty from movement uncertainty and demonstrate that the separation improves useful detection on independent people.

I recommend two different priorities. For the November 20 deadline, strengthen the controlled study and narrow its claims. For a longer research program, build and validate the uncertainty separation described in the [research roadmap](research_roadmap.md). Adding more named components to S-JEPA is unlikely to make this work more persuasive by itself.

## 1. Evidence used for this assessment

I reviewed the progression from [inventory](../../00_research_and_gavd_inventory.ipynb) and [pose quality](../../01_video_pose_and_quality.ipynb) through [masking](../../02_joint_tokens_and_uniform_masks.ipynb), [normal pretraining](../../03_normal_only_jepa_and_vicreg.ipynb), [adaptation](../../04_progressive_condition_adaptation.ipynb), [anomaly scoring](../../05_forecast_anomalies_and_explanations.ipynb), [classification](../../06_capstone_rf_vs_jepa.ipynb), and [ablation planning](../../07_ablation_and_paper_evidence.ipynb).

The checked-in tutorial notebooks have cleared outputs. The numerical evidence here comes from [results.md](../results.md), [results_summary.csv](../results_summary.csv), saved run metrics, and checkpoint training histories. I also inspected the relevant model, training, and evaluation code. This review did not retrain the models or independently repeat the full video extraction. The earlier [verification record](../validation.md) reports 42 passing software tests and notebook execution checks; those establish implementation behavior rather than research novelty.

The accompanying [evidence.json](evidence.json) records the exact numerical inputs to this review and their local artifact paths. The [scorecard](scorecard.csv) makes the scoring arithmetic inspectable. Earlier planning documents are treated as plans where they differ from completed measurements.

## 2. Scored assessment

The scale is 0 for absent evidence, 1–3 for weak evidence, 4–6 for partial support, 7–8 for strong support, and 9–10 for exceptional support. The weights express my view of what matters for this particular research submission. A high score in relevance cannot compensate for an unsupported central claim.

| Criterion | Score / 10 | Weight | Reason for the score |
| --- | ---: | ---: | --- |
| Methodological originality | 2 | 20% | Skeleton JEPA, normal-only gait modeling, masked prediction, positional noise, and variance regularization all have close precedents. No new mechanism has an isolated, demonstrated benefit. |
| Importance of the research question | 7 | 10% | Distinguishing unusual movement from unreliable visual measurement is useful for anomaly detection and healthcare research. The present data support only a limited version of that question. |
| Technical and methodological soundness | 6 | 15% | Source separation, missing-token accounting, causal prefix normalization, and train-only fitting are strengths. Training sufficiency, teacher dynamics, comparison budgets, and annotation timing need more evidence. |
| Empirical strength | 3 | 20% | Three short runs exist, but the trained representations lose to an untrained control, anomaly ranking is unstable, and key explanatory controls are missing. |
| Generalization evidence | 2 | 10% | There are 91 retained source videos, no verified cross-video participant identities, and no independent cohort. Some class-specific test sets contain only one source. |
| Reproducibility and research utility | 8 | 10% | Runnable tutorials, source manifests, exclusion records, saved predictions, model checkpoints, and explicit baseline definitions are valuable. Public-media availability still limits exact reconstruction elsewhere. |
| Strength of the scientific insight | 4 | 10% | The observations identify useful research questions, but do not yet isolate why pretraining or predictive surprise failed. |
| ASTAD topic fit | 9 | 5% | Normal-reference detection, temporal prediction, uncertainty, robustness, and healthcare applications are directly relevant. |

The weighted sum is 44.5/100, reported as **45/100** to avoid implying calibrated precision. This is the maturity of the current research case, not a score for the amount or quality of engineering effort.

Different paper framings lead to different judgments:

| Intended paper | Present judgment | What would change my judgment? |
| --- | --- | --- |
| A new, superior S-JEPA gait detector | Not supported. | A distinct technical contribution and matched experiments demonstrating its benefit over strong anomaly baselines. |
| A controlled study of skeleton predictive learning on GAVD | Promising but incomplete. | Adequate optimization, relevant prior-method comparisons, and a reproducible explanation of one failure mechanism. |
| A clinically generalizable classifier for five diagnoses | Not supported. | Verified diagnostic and participant information, independent evaluation, and meaningful operating-point performance. |
| A tutorial and reproducible application resource | Already useful. | A research paper still needs an original empirical or methodological contribution beyond the tutorials. |

ASTAD explicitly welcomes applications, benchmarks, uncertainty, and distribution-shift research. That creates room for a strong empirical paper without a new architecture. It does not make a tutorial automatically an original research contribution. An ASTAD paper would be an AAAI workshop paper; it should not be represented as an AAAI main-conference result. [Official ASTAD call](https://ismart.ece.mcgill.ca/astad4/).

## 3. What the measured results actually establish

### Classification has not demonstrated a pretraining benefit

The accepted cohort contains **2,504 windows from 450 sequences and 91 source videos**. Every method uses the same retained sequences within each split. The following values are source-weighted five-class macro-F1, rounded to three decimals.

| Method | Seed 42 | Seed 43 | Seed 44 |
| --- | ---: | ---: | ---: |
| Random Forest | 0.267 | 0.482 | 0.333 |
| Normal-pretrained JEPA | 0.244 | 0.214 | 0.290 |
| Progressively adapted JEPA | 0.108 | 0.252 | 0.331 |
| Validation-selected JEPA | 0.244 | 0.252 | 0.331 |
| Untrained encoder with a fitted probe | 0.293 | 0.327 | 0.348 |
| Mean-pose control | 0.230 | 0.250 | 0.322 |
| Pose-availability and camera-view control | 0.221 | 0.204 | 0.243 |

Random Forest and the untrained encoder each exceed the selected trained JEPA on all three splits. This is evidence against claiming a demonstrated representation-learning advantage under the executed recipe. It does not establish that JEPA cannot learn useful gait representations. The mean-pose results also make it important to test whether a trained model uses temporal order rather than mostly static posture or recording characteristics.

The nuisance control performs above the fitted prior baseline, but that observation does not prove that the JEPA uses the same shortcuts. A nuisance predictor can reveal information in the acquisition process; attributing a particular model's errors to that information requires additional intervention or matched-subset analysis.

The RF's supervised abnormal-versus-normal AUROC is 0.840–0.909, while its five-class macro-F1 is substantially lower. That suggests different levels of difficulty for binary separation and category recognition. These AUROCs use labeled abnormal examples during fitting. They are not evidence that a normal-only anomaly detector achieved those results. Likewise, “normal-pretrained JEPA” describes encoder training; its classification probe is trained with all five labels.

### Normal-only anomaly ranking is inconsistent

These are source-level or source-label-level rankings, with source clustering retained in uncertainty calculations.

| Normal-only score | Seed 42 AUROC | Seed 43 AUROC | Seed 44 AUROC |
| --- | ---: | ---: | ---: |
| Masked latent discrepancy | 0.200 | 0.604 | 0.583 |
| Future-suffix latent discrepancy | 0.138 | 0.417 | 0.567 |
| Normal embedding distance | 0.323 | 0.625 | 0.250 |

The score direction was fixed as higher meaning more abnormal. Reversing it after inspecting test performance would create a new, retrospectively selected method. The results justify reporting unstable ranking under this recipe, with the wide intervals in the [full results](../results.md). They do not justify choosing the best seed or score as the headline result.

Normal calibration has only 6, 6, and 5 sources across the three splits. The implemented finite-sample 5% rule therefore produces an infinite threshold and zero sensitivity by construction. There is no demonstrated usable 5% false-positive operating point. A high average precision also needs comparison with the abnormal prevalence at the same evaluation unit; this test cohort contains many abnormal examples.

The fixed-position versus positional-noise comparison is similarly preliminary: seed-42 macro-F1 is 0.244 versus 0.172. This single short experiment supplies no evidence of a benefit from stochastic positions. It cannot establish that stochastic positions are generally harmful.

### The optimization budget is a central unresolved issue

Reading the saved checkpoints adds an important qualification to the phrase “three epochs.” The normal-only models received very few optimizer updates:

| Seed | Normal masked updates | Prediction loss, first to last epoch | Final projector standard deviation | Additional progressive updates |
| --- | ---: | ---: | ---: | ---: |
| 42 | 45 | 5.095 → 2.284 | 0.102 | 777 |
| 43 | 66 | 3.685 → 1.719 | 0.259 | 747 |
| 44 | 63 | 4.032 → 0.799 | 0.216 | 825 |

These are recorded epoch means. With constant teacher momentum 0.996, expanding the EMA recursion gives an initial-teacher coefficient of approximately 0.835, 0.768, and 0.777 after those normal runs. This is an algebraic coefficient, not a measured percentage of “random features.” Nevertheless, it makes teacher warm-up a serious hypothesis to examine. The prediction loss is still changing markedly, and there is no convergence demonstration.

The mean projector standard deviations are also below VICReg's target of one. This does not prove that the encoder has collapsed: the projector and encoder are different objects, and variance alone is not a measure of useful class information. It means that declaring collapse fully resolved merely because VICReg is present would be premature.

Progressive training changes the data, number of updates, and optimization trajectory. Its optimizer is recreated at each stage. Consequently, the current comparison cannot isolate the effect of condition order. A matched pooled-training control and a compute-matched normal-only control are necessary before attributing a result to staged adaptation. These details are visible in [training.py](../../aaai_gait/training.py) and [workflow.py](../../aaai_gait/workflow.py).

## 4. The novelty boundary is narrower than the original proposal suggests

The literature check below is targeted rather than exhaustive. It is sufficient to reject several broad originality claims, but it cannot certify that a proposed future method is the first of its kind.

| Proposed contribution | Relevant prior work | Assessment for this project |
| --- | --- | --- |
| Predict masked skeleton embeddings with an EMA teacher. | [Skeleton S-JEPA, ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04755.pdf). | This is the inherited core architecture. The current smaller 2D implementation is an adaptation. |
| Uniform random masking across joint-time tokens. | [MAMP](https://arxiv.org/abs/2308.07092) and its [official uniform-mask branch](https://github.com/maoyunyao/MAMP/blob/main/model_mamp/transformer.py). | A controlled design choice, not a new masking principle. MAMP's motion targets and its motion-aware masking must remain distinguished. |
| Prevent collapse with variance and covariance regularization. | [VICReg](https://arxiv.org/abs/2105.04906). | Established regularization. It does not force health-condition clusters apart. |
| Add stochastic positional embeddings. | [StoP, ICML 2024](https://proceedings.mlr.press/v235/bar24a.html). | A transfer hypothesis whose benefit has not been demonstrated here. |
| Train on normal gait and localize abnormalities through reconstruction or prediction. | [FSGait, ACCV 2024](https://link.springer.com/chapter/10.1007/978-981-96-0960-4_19), describes normal-only reconstruction and prediction with sequence, frame, and joint outputs. | Normal-first predictive gait modeling and fine-grained anomaly localization are already established research directions. |
| Use a normal-only masked transformer to detect unusual joints. | [GenGait, 2026 preprint](https://arxiv.org/html/2604.01997v2), uses a normative masked autoencoder and joint inconsistency scores. | This is a particularly close conceptual precedent. Its multi-camera setting and simulated abnormal patterns differ from this GAVD experiment. Neither study's numerical results are directly comparable. |
| Apply JEPA to gait representation learning. | [GaitJEPA, official IJCB 2026 repository](https://github.com/AVAuco/GaitJEPA), combines masked and future latent prediction on silhouettes. | The skeleton input and anomaly task differ, but “JEPA for gait” alone is not an original contribution. |
| Account for joint confidence in anomaly scoring. | [SeeKer, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Delic_Sequential_keypoint_density_estimator_an_overlooked_baseline_of_skeleton-based_video_ICCV_2025_paper.html), uses confidence-weighted keypoint conditional likelihoods. | Confidence weighting and per-joint scoring alone are insufficient novelty. This is an important comparison method. |
| Learn normal skeleton distributions for anomaly detection. | [STG-NF, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/papers/Hirschorn_Normalizing_Flows_for_Human_Pose_Anomaly_Detection_ICCV_2023_paper.pdf). | Random Forest is a useful supervised baseline, but it does not replace a normal-only skeleton anomaly baseline. |
| Show that JEPA prediction error can respond to nuisance variation. | [Asleep at the Wheel, 2026 preprint](https://arxiv.org/abs/2608.01336), and [When Does Latent Prediction Error Support Physical Surprise Claims?, 2026](https://arxiv.org/abs/2611.01815). | The general warning already exists. A gait paper needs a specific mechanism, controlled evidence, or a useful evaluation contribution beyond repeating that warning. |
| Introduce learned stochastic latent variables to JEPA. | [A Stochastic Latent Variable Generalization of JEPA, November 2026 preprint](https://arxiv.org/abs/2611.09302). | This extension is unimplemented here and already has direct prior art. A gait-specific uncertainty claim needs its own validation. |

There is also recent [multimodal GAVD classification work](https://link.springer.com/article/10.1007/s44163-026-02522-3) using RGB and skeleton representations. Its modalities, binary task, and cohort differ. It should inform related work and comparison design, rather than supply a leaderboard number to compare against this five-class subset.

Choosing a useful combination of established methods can support an application paper. The contribution then needs to be a demonstrated capability, a meaningful efficiency or supervision tradeoff, a new evaluation resource, or a generalizable empirical insight. At present, the project has not established such a capability from the learned model. Careful implementation is its strongest completed contribution.

## 5. What a skeptical reviewer is likely to challenge

### The effective sample size is much smaller than the window count

The five selected annotation folders contain 666 sequences and 103 source IDs before filtering. The analysis retains 450 sequences and 91 source IDs, excluding 216 sequences for media, duration, frame-rate, or pose-quality reasons. That is approximately 32% sequence attrition. A reviewer needs to see whether exclusion varies with condition, view, and motion severity, because quality filtering can change the population being studied. See the [data review](../data_audit.md).

The 2,504 overlapping windows are not 2,504 independent people. Source separation is a substantial safeguard, but repeated people, demonstrators, or acquisition settings may still appear across different uploads. One Parkinsonian test source in seeds 42 and 43 is too little support for a stable Parkinsonian-generalization claim. No bootstrap can create the missing participant diversity.

### The statistical summaries answer different questions

The headline macro-F1 values are source-weighted, while the currently reported paired bootstrap differences are sequence-weighted with source-cluster resampling. Those intervals cannot be attached directly to the source-weighted differences as if they estimated the same quantity. For example, seed 43 has a negative sequence-weighted interval for selected JEPA minus RF, but that is not an interval for the headline source-weighted difference.

Only 550, 558, and 722 of 1,000 classification resamples retain all five classes. This is another sign of fragile class support. The three source splits overlap and also change initialization and masking seeds; they are not three independent patient studies or a clean estimate of initialization variance. Use a consistent primary metric and uncertainty calculation, and show which uncertainty remains unmeasured.

### A gait-category label is not a verified diagnosis

The five labels are a defensible restricted study cohort. They do not exhaust abnormal walking, and their shared motivation does not make them one homogeneous neurological population. In particular, the myopathic category should not be described as a single central neurological etiology. The current annotations do not establish an independently verified medical diagnosis for each recorded person.

Frame timing and person assignment also deserve systematic checks beyond a few illustrative frames. If those are wrong, more sophisticated learning will model the wrong movement. Claims about physiological mechanisms require independent kinematic or expert evidence; latent visualizations and attention maps cannot provide it on their own.

### The project separates prediction tasks correctly, but the claims must follow

Uniform masking usually leaves both earlier and later observations around a target. That is completion. The separate suffix experiment genuinely restricts the predictor to a prefix, but evaluation admits sequences using future pose coverage. It is retrospective prediction on evaluable futures. The five-class probe uses an observed complete window.

None of these experiments demonstrates prediction of disease onset, deterioration, falls, or another future clinical outcome. Those require longitudinal targets and an evaluation organized around the time when a decision could have been made. The current model also predicts latent targets rather than decoded future skeletal coordinates.

## 6. Which innovations would make the research stronger?

The following scores concern the promise of a well-executed research direction. They do not upgrade the scores for the current implementation. Higher feasibility means more realistic before the present deadline.

| Direction | Potential originality / 10 | Potential impact / 10 | Deadline feasibility / 10 | My priority |
| --- | ---: | ---: | ---: | --- |
| A controlled benchmark separating optimization failure, pose quality, and movement information | 6 | 7 | 8 | First for ASTAD. |
| A normal-motion model that separates observation uncertainty from movement uncertainty | 7 | 8 | 3 | First for a longer method paper. |
| Phase-aware and laterality-preserving representations with tested sensitivity to meaningful movement changes | 6 | 8 | 3 | A focused extension if reliable movement annotations can be obtained. |
| A constraint preserving normal anomaly scores during condition adaptation | 4 | 6 | 5 | Secondary; replay, distillation, and frozen references already have strong precedents. |
| Adding positional noise or a generic stochastic latent alone | 2 | 4 | 4 | An ablation, not the central contribution. |

The first direction becomes valuable if it isolates an explanation through a controlled intervention, such as changing teacher warm-up while matching the remaining training procedure. A loss of discrimination on a quality-matched subset would support a narrower association or robustness finding; observational matching alone does not prove that recording quality caused the original discrimination. Observing that a nuisance feature predicts a label is also insufficient. A controlled change must alter the suspected mechanism while keeping the relevant comparison fixed.

The second direction asks the model to distinguish two situations: an uncertain estimate of ordinary walking, and a well-observed movement inconsistent with the normal reference. Its technical contribution would need to go beyond multiplying residuals by confidence, since SeeKer already uses detector confidence. Paired recordings or independently checked poses are likely to be more valuable than another loss term.

The third direction addresses a real tension in the representation objective: useful invariance to camera or measurement variation can also remove subtle movement information. A promising contribution would show that a specified training constraint preserves sensitivity to validated asymmetry or timing differences while reducing sensitivity to measurement changes. Generic phase encoding or a visually appealing joint heatmap would be insufficient.

The [roadmap](research_roadmap.md) defines concrete controls, falsification criteria, and a deadline plan for these directions. All proposed main-model experiments retain uniform masking over the same 16 allowed joints. Motion-aware sampling is not required to pursue any of them.

## 7. A defensible paper and a clear stopping rule

A suitable working title is **“Evaluating Skeleton Predictive Learning for Gait Anomaly Detection in Uncontrolled Video.”** It names the completed kind of study without implying clinical utility or methodological priority. If the quality-versus-movement hypothesis is later demonstrated, the title can become more specific.

Three contribution statements could be defensible after the missing controls are completed:

1. We provide a reproducible, source-grouped evaluation of normal-reference skeleton prediction and supervised gait-category recognition on a documented GAVD subset.
2. We measure how optimization budget, pose observation quality, and temporal information affect the relationship between predictive discrepancy and annotated gait abnormality.
3. We identify a reproducible failure mechanism and provide a controlled mitigation or a clearly delimited negative result, compared with appropriate normal-only and supervised baselines.

The first statement is largely supported now. The second is only partially supported. The third is a research target. They must not all be written in the past tense merely because they make a coherent paper outline.

Before submission, I would require adequate training diagnostics, a relevant normal-only comparison, an estimate and interval for the same primary metric, and one completed explanation of a failure. An independent clinical dataset would materially increase impact, but the workshop study can remain a carefully bounded GAVD investigation if it avoids clinical generalization claims. Stronger main-conference ambitions would need a substantially broader empirical and conceptual contribution.

The official deadline is **November 20, 2026**, with up to seven content pages plus two reference-only pages, double-blind review, and the AAAI-27 author kit. Archival proceedings are intended but not yet confirmed. These details were checked on the [workshop page](https://ismart.ece.mcgill.ca/astad4/) on the assessment date; the page does not specify the cutoff timezone.

If the remaining work produces only another short run and a larger list of model components, I would keep this as an exploratory technical report rather than claim a strong innovation paper. If it isolates why the score succeeds or fails under controlled conditions, the same project could support a useful ASTAD contribution even without beating every classifier.


# Research directions and experiments that would change the assessment

This document accompanies the [critical review](README.md). Everything below is a proposed experiment unless explicitly described as an existing result. The aim is to produce a supported contribution, not to add every possible extension before the workshop deadline.

## 1. First establish what the current model has learned

The normal anchors received only 45–66 optimizer updates. Before interpreting their performance as a property of latent prediction, investigate training sufficiency on development data. Use separate variables for the source split and model initialization seed, so that their effects can be distinguished.

| Priority | Controlled experiment | Keep fixed | What it can establish |
| --- | --- | --- | --- |
| Essential | A prespecified update ladder, for example 64, 256, 1,024, and 4,096 updates | Training sources, architecture, loss, joint set, and sampling policy | Whether the result depends on a very short optimization budget. These are diagnostic budgets, not a claim that 4,096 guarantees convergence. |
| Essential | Teacher warm-up or momentum schedule versus the current constant EMA | Total updates, initialization, data draws, and masks | Whether teacher dynamics explain a meaningful part of the short-run result. |
| Essential | Online-encoder and target-encoder probes | Checkpoint, train/validation sources, probe search, and evaluation | Whether the chosen readout contributes to the apparent failure. The current readout differs from the original Skeleton S-JEPA paper. |
| Essential for a curriculum claim | Progressive, pooled, and reversed-order condition exposure | Total updates, approximately equal per-condition exposure, normal replay, and optimizer-reset policy | Whether order contributes beyond additional compute and data. Include a normal-only continuation with the same update budget. |
| Essential for a dynamics claim | Correct context, swapped context, and orderless-context prediction | Teacher targets, allowed observations, target tokens, and scoring | Whether useful prediction depends on the temporal context. |
| Essential for an anomaly claim | A normal-only skeleton anomaly baseline | Cohort, observation policy, training-source access, and aggregation | Whether the failure or improvement is specific to the proposed scoring/model choice. |

Record training and normal-validation prediction loss, encoder variance and effective rank, projector variance, teacher-to-online drift, and downstream validation probes. Keep the normal-only anomaly setting separate from the supervised probe setting. A probe trained with abnormal labels does not become unsupervised because its encoder was pretrained on normal data.

Compare against an appropriate subset of [STG-NF](https://github.com/orhir/STG-NF), [SeeKer](https://github.com/adelic99/seeker), and the concepts in [FSGait](https://link.springer.com/chapter/10.1007/978-981-96-0960-4_19), in addition to the existing distance score. A simple coordinate reconstruction or motion-prediction baseline can help isolate the value of latent targets. Document adaptations to the 16-joint topology, missing observations, and sequence duration. Do not compare a complete-case baseline against a different, easier cohort without also reporting a common-cohort comparison.

The strongest useful outcome could be either positive or negative. If adequate training closes the untrained-encoder gap, the original failure was at least partly a recipe issue. If it does not, a well-controlled failure analysis becomes more credible. Neither outcome should be decided in advance.

## 2. Near-term direction: isolate the information in predictive surprise

### Research hypothesis

After sufficient optimization, raw latent discrepancy may combine static geometry, ordinary prediction difficulty, observation quality, and temporal coordination. The additional predictive value of ordered context may be more informative about coordination than the raw discrepancy alone.

This is a hypothesis, not a conclusion from the current AUROCs. Some pathological gait can remain highly regular and predictable. A temporal score therefore cannot replace a separately assessed static-posture or normal-space score.

### A concrete experiment

Freeze one adequately trained normal-reference teacher, including its centering statistics and temperature. Train two predictors using only the same normal training sources and the same observed inputs. The dynamic predictor receives an ordered temporal context. The comparison predictor receives a permutation-invariant summary of the same context. Both retain joint identities and target positions. Match capacity and training budget, and record exactly how confidence and visibility information enter each predictor.

Define the removed order precisely. One implementation could embed each visible four-frame joint patch independently, without a context-time embedding, and pool patches within each joint before any temporal contextualization. It would preserve motion within a four-frame patch while removing order between those patches. Pooling already contextualized transformer outputs would not achieve this control. Test that permuting the eligible context patches within each joint, while keeping the target queries and visibility summaries fixed, leaves the comparison predictor's output unchanged. Label the control as removing order between patches rather than all temporal information.

For each common valid target token, let `p` be the frozen teacher's feature-channel distribution, `q_dynamic` the dynamic prediction, and `q_orderless` the orderless prediction. Define the exploratory contrast:

```text
temporal_contrast = mean_valid_targets[
    CE(p, q_dynamic) - CE(p, q_orderless)
]
```

More negative values mean that ordered context provides greater predictive benefit. A relatively high value means reduced benefit relative to the normal reference distribution. The two losses must use identical targets, temperatures, and valid-token support. This contrast is also a difference of KL discrepancies because the shared target entropy cancels. **It is not a likelihood ratio over skeleton sequences:** the channel softmax is not a normalized probability density over possible motions.

Removing order can also remove the temporal pattern of missingness. Control this explicitly through identical visibility summaries and a matched high-coverage subset, and repeat the test with controlled observation masks. Otherwise the experiment may measure missingness order rather than movement order. A position-only predictor and context swapping provide complementary checks with different failure modes.

Use uniform masking over the existing 16-joint grid for the main completion experiment. A causal suffix experiment is separate and must use prefix-only normalization and inputs. Preserve the original calibration and test procedures while introducing the contrast as a new exploratory score. Fix its sign before evaluating new test data.

The background-correction idea itself has substantial prior art, including [Ren et al., NeurIPS 2019](https://proceedings.neurips.cc/paper/2019/hash/1e79596878b2320cac26dd792a6c51c9-Abstract.html). The possible contribution is a demonstrated gait-specific explanation and useful scoring behavior under controlled confounds, not priority over subtracting a background score.

### Evidence that would support or falsify it

Measure raw discrepancy and the contrast on normal and abnormal held-out sources, then repeat on a prespecified high-quality subset and within supported view strata. Use the same source groups for paired comparisons. Report AUROC, prevalence-aware AP, and the response to controlled perturbations. Do not pool source windows as independent observations.

The hypothesis is weakened if correct and swapped context perform similarly, if the contrast is driven mainly by observation coverage, or if it removes sensitivity to annotated gait differences. Prespecify a minimum useful gain before independent evaluation and calculate a paired interval. An interval that excludes that gain can reject the proposed improvement at that magnitude; a wide interval spanning both meaningful benefit and no benefit is inconclusive. A negative result remains informative if the controls establish what the model actually uses and the uncertainty supports the stated conclusion.

## 3. Longer-term direction: separate observation uncertainty from movement uncertainty

### The contribution worth pursuing

A useful detector should distinguish a poorly observed ordinary movement from a well-observed movement outside the normal reference. Simply treating a large residual or high predictive variance as abnormal cannot make this distinction. Confidence-weighted joint scores already exist in [SeeKer](https://openaccess.thecvf.com/content/ICCV2025/html/Delic_Sequential_keypoint_density_estimator_an_overlooked_baseline_of_skeleton-based_video_ICCV_2025_paper.html).

A stronger contribution would be an identifiable separation between uncertainty in the measurement and variability in the movement, supported by paired observations. The model should have an observation-quality output and a movement-deviation output whose behavior can be tested separately. If evidence is too poor, an explicit “insufficient observation” outcome is preferable to a confidently interpreted movement score, but its coverage and errors must also be evaluated.

### Data and model design

Obtain a small reference set with repeated views, independently checked keypoints, or a validated motion reference. Two pose estimators on the same image provide useful disagreement measurements, but their agreement is not ground truth and their errors can be correlated. Manual or external reference measurements are needed to interpret that disagreement.

Train an observation model on paired measurements and a normal-motion predictor on the normal training partition. If learned stochastic latents are used, distinguish the conditional prior available at inference from the target-informed posterior used during training. Use a stated probabilistic objective and evaluate prior samples. Report negative log predictive density only if the implementation defines a normalized predictive density; otherwise use an appropriate proper sample score in a fixed target space. Evaluate representation quality separately so that uncertainty improvements do not hide a loss of useful information.

The [November 2026 stochastic-latent JEPA preprint](https://arxiv.org/abs/2611.09302) already motivates learned predictive uncertainty and diverse completions. Transferring that machinery is a baseline. The proposed new contribution would be separating the sources of uncertainty and demonstrating better movement decisions under realistic pose error.

### Cross movement status with observation quality

| Evaluation condition | Prespecified expected behavior |
| --- | --- |
| Normal movement, reliable observation | Low movement-deviation score and low observation uncertainty. |
| The same normal movement, degraded observation | Increased observation uncertainty; movement decisions should remain stable where adequate information survives. |
| Independently annotated movement difference, reliable observation | Increased movement-deviation score, without requiring poor tracking. |
| Movement difference with degraded observation | Calibrated uncertainty or reduced coverage; report whether the detector misses the movement difference. |

Use paired image-level degradations when testing the entire visual pipeline. Coordinate noise is a useful controlled skeleton experiment, but it does not reproduce all failures of a visual pose estimator. Prespecify degradation strengths and have the retained movement content checked. A temporal shuffle or synthetic asymmetry is a controlled perturbation, not a newly labeled patient or a validated disease simulation.

Evaluate discrimination, normal false alarms, error-versus-uncertainty association, calibration where applicable, and coverage-versus-risk curves. Show abstention rates separately by condition and source. A model that improves conditional accuracy by rejecting most difficult cases has not necessarily improved screening usefulness.

The hypothesis fails if the proposed movement score rises mainly with tracking error, if the model suppresses genuine movement differences as uncertainty, or if gains disappear against a simple confidence-aware baseline. An independent participant-identified cohort would be the decisive test of impact.

## 4. Longer-term direction: preserve relevant movement sensitivity

A second method question is whether representation learning can become insensitive to nuisance variation while retaining subtle asymmetry and temporal coordination. Adding VICReg does not specify which variations should be preserved. Conversely, making every normal gait look identical can erase useful differences.

One concrete design is to distinguish transformations that should preserve the prediction from changes that should affect a measured kinematic output. Preserve left/right anatomy, estimate phase using the available context only, and retain physical timestamps or cadence alongside any phase-aligned representation. Whole-sequence phase alignment in a forecasting input could leak future information. Time normalization can also erase the timing abnormality being studied.

An auxiliary target could be a predeclared, independently validated set of asymmetry and inter-joint timing measurements. Compare the model with and without the target, with an equal-budget coordinate-prediction baseline, and with direct classical features predicting the same measurements. Since this adds kinematic supervision or a structured pretext target, describe that supervision clearly. It is not purely generic latent self-supervision anymore.

The important result would be a controlled tradeoff: nuisance transformations have less effect on the representation while expert-validated movement differences remain detectable. Measure both sides. Synthetic timing changes can test sensitivity, but establishing clinical relevance requires real annotations and agreement analysis. [FSGait](https://link.springer.com/chapter/10.1007/978-981-96-0960-4_19) and [GenGait](https://arxiv.org/abs/2604.01997) already make fine-grained gait analysis relevant prior art; joint localization alone is not enough.

I would defer a normative reconstruction or treatment recommendation claim. The present system has no validated coordinate decoder, and producing a plausible alternative movement does not establish a safe or appropriate rehabilitation target.

## 5. Treat adaptation and stochasticity as focused secondary experiments

For staged condition adaptation, keep the normal checkpoint immutable, as the code already does. A stronger future question is whether one shared representation can preserve normal-score calibration while improving recognition. Compare a frozen anchor, ordinary replay, distillation, pooled training, and a proposed explicit score-preservation constraint under matched budgets. The new claim must concern a measured retention-versus-recognition tradeoff. Keeping a backup checkpoint alone is good practice rather than a new continual-learning method.

For positional stochasticity, first establish the deterministic reference and then repeat matched runs with a small, prespecified noise grid. Preserve initialization, masks, batch draws, and update counts. Investigate whether changes affect pose-noise sensitivity and movement sensitivity differently. The existing single seed supplies a pilot observation only.

For stochastic latent variables, compare a deterministic predictor, simple fixed noise, and a learned prior/posterior model at matched encoder capacity and training budget. At test time, sample from the prior without the target. Report sample count, runtime, diversity, calibration, and representation quality. A fixed-sample-count minimum residual can be evaluated retrospectively once the future is observed, but it is not a proper predictive-density score or a forecast-selection rule available before that observation. It depends on sample count and needs separate calibration and matched sampling cost. Learned uncertainty should not be added until there is a concrete uncertainty question and data to test it.

## 6. Preserve a valid evaluation after inspecting the current tests

All three current test splits have been inspected. They can support the original descriptive results and transparently labeled follow-up analyses. A new partition of the same reviewed sources does not automatically become untouched confirmatory evidence. Preserve the existing results, record the timing of every new hypothesis, and use genuinely unexamined participant or source data for a new confirmatory claim.

For development, source-grouped internal resampling can help choose schedules without introducing window leakage. For final evidence, audit repeated people and duplicated media as far as the available metadata permit. Record verified identity groups separately from source IDs. More windows from the same recording do not solve a shortage of independent people.

Choose the evaluation unit before analysis. For five-class recognition, retain the current source-weighted macro-F1 as a primary outcome and calculate paired uncertainty for that same weighting. Keep sequence-weighted results as a separate outcome. For normal-only detection, state exactly how mixed-label source rows and source clustering are handled. Report per-class support, attrition, and uncertainty rather than only pooled performance.

For the implemented rule, the finite-sample threshold rank is `ceil((n + 1) * (1 - alpha))`. At `alpha = 0.05`, at least 19 normal calibration units are needed even to obtain a finite threshold. Nineteen is a mathematical minimum, not a sample-size recommendation for precise clinical performance. The units and test population also need the relevant exchangeability assumptions; unknown repeated people and distribution shift weaken that interpretation. See [Angelopoulos and Bates](https://arxiv.org/abs/2107.07511) for the broader calibration framework.

Avoid strengthening calibration language by using multiple windows from five people as if they were many independent normal subjects. If the available data cannot support a useful operating point, report ranking and the limitation honestly. A separately prespecified exploratory operating point can be informative, but does not establish a 5% guarantee.

## 7. A realistic plan for the remaining days

This is a prioritized schedule, not a promise that all training will finish within a laptop budget. Measure one run's duration and choose a defensible scope. Quality of the comparison matters more than the number of variants.

| Date | Main deliverable | Decision rule |
| --- | --- | --- |
| November 14–15 | Fix the paper question; audit a stratified sample of timing/person alignment; inspect training and teacher dynamics on development data. | If annotation problems are material, repair and disclose them before interpreting model behavior. Preserve the original results as superseded where appropriate. |
| November 15–16 | Run the update-budget diagnostic and one normal-only anomaly comparison. | If pretraining remains far from a meaningful training regime, prioritize this over positional or latent stochasticity. |
| November 16–17 | Complete one temporal-context or pose-quality experiment, with a prespecified scoring rule. | Choose one mechanism for the paper. Treat hypotheses formed after the current tests as exploratory. |
| November 17–18 | Finish matched evaluation, consistent intervals, attrition summaries, and a small representative failure gallery. | If no specific finding is supported, narrow the manuscript to an explicitly exploratory report. |
| November 18–19 | Write the seven-page paper, check every numerical claim, and review anonymity. | Do not turn proposals into completed contributions or omit unfavorable controls. |
| November 20 | Final source check and submission before the verified system cutoff. | Confirm the cutoff timezone in the submission system; the workshop page does not state it. |

A compact seven-page allocation could be one page for the question and closest prior work, one for cohort and protocol, one for the model and diagnostic contrast, two for controlled experiments, and two for failure analysis, interpretation, and limitations. Adapt the allocation to the actual findings. The [ASTAD requirements](https://ismart.ece.mcgill.ca/astad4/) allow two additional pages for references only.

The strongest paper will answer one difficult question clearly. If training is the main explanation, document that mechanism rather than claiming a general failure of world models. If pose quality dominates, show the controlled evidence and the limits of the mitigation. If temporal context adds useful information under matched conditions, demonstrate that contribution without implying diagnosis, disease progression, or clinical deployment.
