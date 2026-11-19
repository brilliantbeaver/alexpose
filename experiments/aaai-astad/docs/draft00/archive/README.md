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
