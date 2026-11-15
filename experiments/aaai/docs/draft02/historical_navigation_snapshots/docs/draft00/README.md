# A clearer research plan for detecting abnormal gait

The [next-iteration instructions for Astra](../../../../notes/aaai/05-iterate-02-INSTR.md) now specify retained-baseline prediction, matched U/J/A masking experiments and systematic documentation updates. The [research brief](../iteration02/README.md) and [planning slides](../../slides/tutorial.pdf) describe that proposed work. The results reported below remain historical evidence; the new experiments have not been run.

> **Completed real-data update, November 15, 2026.** The matched-context experiment and its longer-training sensitivity have now run. The learned latent profile reaches AUROC 0.641, but its interval versus raw error includes zero and matched calibrated or random-encoder controls have higher point estimates. Read the [current executive summary](EXECUTIVE_SUMMARY.md), [grounded manuscript](../draft01/manuscript.md), and [readiness assessment](../draft01/submission_readiness.md). The research ideas below explain the proposal; they are not evidence of an established neurological or JEPA advantage.

> Implementation update. The proposal below motivated the executable [context study](../context_study_protocol.md), taught in [Notebooks 08 through 11](../../08_biology_cohort_and_source_folds.ipynb). Read the [real-data results](../context_study_results.md) and [draft01 manuscript](../draft01/README.md) for the evidence and submission judgment. Illustrative numbers below remain teaching examples.

**Revised November 14, 2026, after consultations with Claude Opus and GPT advisors.** Start with the [executive summary](EXECUTIVE_SUMMARY.md) for the coming week. The new [neurological research direction](neurological_research_direction.md) explains the biological motivation and a more ambitious world-model program. The [research roadmap](research_roadmap.md) specifies the experiments. Browse the [illustrated visual guide](images/index.html), or read the [advisory synthesis](advisory_summary.md) for accepted and rejected advice.

## The question in plain language

**Can we detect unusual walking by checking whether the legs and arms move together in familiar patterns?**

When someone walks, we can look at each body part separately, or we can look at how the body parts move together. These are different sources of information. An ankle can follow a smooth, repeating path while its timing relative to the other leg is unusual. A model that only asks whether that ankle is easy to predict may miss the difference.

We propose to learn the relationships found in the normal gait videos, then test whether those relationships help distinguish the abnormal gait videos. This does not assume that every abnormal gait has poor coordination. It asks whether information about movement relationships adds anything useful beyond the motion of each joint on its own.

Following the user's definition, **`normal` is the only normal folder. Every other GAVD folder is abnormal for the binary detection task.** This includes `exercise`, `style`, and `pregnant`. These folders are not healthy negative controls. Their abnormal gait labels do not establish a medical diagnosis.

## A concrete example: hide part of the left ankle's movement

Imagine watching a short skeleton animation. A small section of the left ankle's path is hidden. We ask two predictors to fill in the same missing section.

| Predictor | What it can see | The question it answers |
| --- | --- | --- |
| Predictor A: the ankle alone | The left ankle's visible positions elsewhere in the clip. | Can this ankle's own movement tell us where the missing positions should be? |
| Predictor B: the ankle and the rest of the body | The same visible ankle positions, plus the other visible allowed joints. | Do the other leg, hips, knees, and arms help predict those missing positions? |

Neither predictor sees the hidden ankle positions. Both receive exactly the same target positions and random mask. Both learn from normal training videos only. The extra body information is the intended difference between them.

Here, “the ankle alone” describes the information available to the learned predictor. Both predictors still use the same body reference to center and scale the coordinates. That shared preprocessing already uses the hips and shoulders.

Suppose Predictor B makes fewer errors on normal validation videos. That tells us the other joints contain useful information about the hidden ankle movement. We can then ask whether the same extra help is present in abnormal videos. We repeat the comparison across the allowed joints and different randomly hidden sections, then combine the results for each source video.

This example explains a masked-completion task. The visible context may include observations before and after a hidden section. It does not, by itself, demonstrate prediction of future disease or future clinical events.

## Why looking at the joints together might matter

Consider a simple illustration with two leg trajectories. In the first sequence, the left and right legs alternate in a familiar walking pattern. In the second, we keep the individual trajectories but shift the timing of one leg in software. Each leg can still repeat its own movement, while the relationship between the legs changes.

Predictor A has little reason to notice that the other leg changed. Predictor B was trained on relationships in normal walking, so the timing change may reduce the help it gets from that other leg. That difference is what we want to measure.

The edited sequence is a demonstration of the idea, not a simulated diagnosis. A real experiment must check that an edit has not simply created broken anatomy, discontinuities, or missing observations. Those controls are specified in the [roadmap](research_roadmap.md).

### A numerical illustration

The following numbers are **made up to explain the comparison. They are not GAVD results.** Prediction error is shown in arbitrary units; smaller values mean a more accurate prediction.

| Illustrative situation | Error using the ankle alone | Error using the ankle and other joints | How much the other joints help |
| --- | ---: | ---: | ---: |
| A familiar movement relationship | 0.20 | 0.08 | 0.12 |
| A changed relationship that remains unfamiliar to the normal model | 0.20 | 0.18 | 0.02 |
| An abnormal pattern whose relationships remain predictable | 0.20 | 0.09 | 0.11 |
| An unusually tight relationship | 0.20 | 0.03 | 0.17 |

In the first row, the other joints reduce error by 0.12. In the second row, they reduce it by only 0.02. The ankle alone is equally predictable in these two examples, but the usefulness of the other joints differs.

The third row is important. An abnormal gait may remain predictable both on its own and in relation to the rest of the body. Our proposed score could miss it. That is why we must compare the score with direct measures of posture and distance from normal examples. We should not assume that one kind of prediction error detects every abnormal gait.

The fourth row shows a different possibility: the joints help more than expected. The biological review motivates checking this possibility too. Unusually tight movement relationships can differ from normal relationships, even when prediction is accurate. The numbers do not show that a particular condition causes this effect.

For readers who want the exact calculation:

```text
help from other joints = prediction error using the joint alone
                         - prediction error using all visible joints
```

A larger value means the additional joints helped more. Keep the original hypothesis that some abnormal patterns show less help as a one-sided comparison. The new, separately specified test looks for departures in either direction across a small set of body regions. Both tests must be fixed before new scoring. Neither may be chosen or reversed because it looks favorable on the test set.

![Two matched predictors estimate the same hidden joint patch.](images/predictive_benefit.svg)

*The diagram shows the proposed comparison. Any predictive benefit must first be demonstrated on held-out normal sources.*

## How neurological and muscle mechanisms sharpen the question

Abnormal walking can be a reliable way to keep moving despite a limitation. After stroke, the legs may contribute differently. In Parkinson's disease, movement size and rhythm can change differently. Cerebral palsy can affect selective control while muscle properties change with growth. Myopathic gait starts from muscle-related problems in many cases, so it should not be described as one neurological disease. The [new evidence-based review](neurological_research_direction.md#2-what-the-four-condition-families-actually-suggest) explains these distinctions with primary clinical studies.

![Different biological origins can produce overlapping visible movement effects.](images/condition_routes.svg)

*These routes motivate hypotheses. They do not identify the cause of an individual GAVD clip.*

This suggests a more specific question: **Can we detect abnormal walking that remains predictable by checking whether different body regions receive an unusual amount of help from the other joints?** Imagine that the ankle is still easy to predict, but the whole-body information helps one leg much more than the other. A single average error could hide that pattern. We propose to compare a small three-region profile with the normal reference, while checking posture, camera view, and tracking quality.

![An illustrative profile shows both less and more predictive help than expected.](images/support_profile.svg)

*The values are invented. The profile describes prediction, not muscle force, neural activity, or diagnosis.*

The longer-term world model would represent a controller, a physical body, ground contact, and the camera separately. Its creative target is to report several compatible explanations when two limitations produce similar visible motion, then test which extra measurement would distinguish them. This requires simulation and independent clinical data. It is a research path beyond this week's GAVD experiment, described with diagrams in the [world-model proposal](neurological_research_direction.md#6-the-more-ambitious-and-creative-world-model-program).

## What would make this scientifically useful?

The central contribution would be evidence that **how body parts move together helps detect abnormal gait beyond what their individual movements reveal**. That is more specific than saying a transformer can classify walking videos.

Three results are needed. First, the other joints must provide useful predictive information on normal videos. Second, controlled timing changes must affect that information in a way that observation damage does not explain. Third, the comparison must improve detection on real abnormal labels, including additional categories where enough usable sources are available.

There is also a subtle model issue. A JEPA teacher turns the hidden ankle patch into a learned target. If that target already contains information about the rest of the body, Predictor B may gain an advantage simply because it can see information that the teacher put into the answer. To check this, repeat the comparison using the ankle coordinates themselves as the target. The coordinate target describes the hidden ankle patch without a contextual transformer mixing in other joints. Both versions use the same declared body-centering and scaling convention.

If the effect appears only with the contextual target, we must investigate that explanation before calling it a movement-relationship finding. Coordinate error and latent error use different scales, so their raw numerical sizes should not be compared directly.

A suitable working title is **“Predictable but Abnormal: Testing Joint Relationships in Gait Anomaly Detection.”** The proposed dissociation must be demonstrated before using it as a result claim. Stronger language about physiological coordination should wait for independent supporting evidence.

## What this adds to the existing notebooks

The existing notebooks ask whether a model can predict missing skeleton information and whether its learned features help classification. The new study adds two matched predictors so we can ask **which visible information helps**, rather than treating every prediction error as evidence of abnormality.

The current results do not establish the proposed benefit. This is the question for the coming week's experiments. If the ankle-only predictor already performs as well as the whole-body predictor on normal videos, the proposed comparison has little useful information to work with. If the benefit exists but does not distinguish abnormal labels, that is a different outcome. The research should explain which situation occurs, rather than assume success.

## What GAVD can support

A fresh scan found **1,874 sequences, 458,116 annotated rows, and 348 source-video IDs**. The duplicate source CSV cache was excluded. These are annotation counts before additional downloading and pose filtering. See the [census record](advisory/gavd_opportunity_audit.json).

| Folder | Binary label | Sequences | Source IDs |
| --- | --- | ---: | ---: |
| normal | Normal | 291 | 32 |
| parkinsons | Abnormal | 47 | 11 |
| stroke | Abnormal | 76 | 19 |
| cerebral palsy | Abnormal | 64 | 11 |
| myopathic | Abnormal | 188 | 30 |
| abnormal | Abnormal | 767 | 117 |
| antalgic | Abnormal | 35 | 10 |
| exercise | Abnormal | 234 | 98 |
| inebriated | Abnormal | 23 | 8 |
| pregnant | Abnormal | 6 | 2 |
| prosthetic | Abnormal | 39 | 8 |
| style | Abnormal | 104 | 3 |

There are **1,583 abnormal sequences from 316 source IDs** and 291 normal sequences from 32 source IDs. One source appears under both `abnormal` and `cerebral palsy`. Keep it in one partition and count it once in the overall binary result.

The seven folders outside the original five-class experiment provide **245 additional source IDs**. None supplies a normal-labeled source. A filename check found local video files for 49 of them: 30 abnormal, 13 exercise, two antalgic, two inebriated, one prosthetic, and one style. No pregnant source had a matching local file. Decoding and pose quality were not verified by this check.

Broader evaluation is feasible to pursue, but an all-folder result cannot be promised before media checks. Attempt acquisition under a fixed time budget, record failures, and use every source meeting the declared quality rule. Keep downloads in `data/videos-gavd`. Report sequence counts and source counts separately.

No source ID contains both normal and another label, so there is no ready-made within-video normal-versus-abnormal comparison. There are 213 sources with multiple camera-view labels, but same-person or synchronized views are unverified. Only 758 frame rows contain gait-event annotations, and none belongs to normal gait. Dense normal phase ground truth is therefore unavailable.

The [GAVD paper](https://arxiv.org/abs/2407.04190) describes a clinically screened collection. Our claims must use the exact local cohort rather than treating historical subject counts as verified participant IDs in these files.

## The novelty boundary

The originality claim must be narrower than “JEPA for gait” or “model joint relationships.”

| Prior work | What is already established | What our study must add |
| --- | --- | --- |
| [Skeleton S-JEPA](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04755.pdf) | Masked skeleton latent prediction. | A controlled scoring comparison and a demonstrated gait finding. |
| [FSGait](https://openaccess.thecvf.com/content/ACCV2024/html/Duan_FSGait_Fine_Grained_Self-Supervised_Gait_Abnormality_Detection_ACCV_2024_paper.html) and [GenGait](https://arxiv.org/abs/2604.01997) | Normal-only predictive or reconstructive gait anomaly analysis. | Evidence about other-joint information beyond a joint's own motion. |
| [GaitJEPA](https://github.com/AVAuco/GaitJEPA) | JEPA-style gait representations from silhouettes. | A distinct question, not priority over applying JEPA to walking. |
| [SeeKer](https://openaccess.thecvf.com/content/ICCV2025/html/Delic_Sequential_keypoint_density_estimator_an_overlooked_baseline_of_skeleton-based_video_ICCV_2025_paper.html) | Conditional keypoint density and confidence-aware scores. | Benefit beyond conditional prediction and confidence weighting. |
| [Beyond Marginals](https://arxiv.org/abs/2509.15033) | Separating temporal, marginal, and inter-variable behavior. | A gait-specific mechanism checked with local targets and real labels. |
| [The MiLeTS 2026 benchmark analysis](https://kdd-milets.github.io/milets2026/accepted%20papers/1camera_ready_paper_anomalies_mostly_univariate.pdf) | Cross-channel diagnostics and marginal-preserving perturbations. | Findings about real gait and contextual targets, not a new synthetic-corruption claim. |
| [RobustGait](https://arxiv.org/abs/2511.13065) | Gait corruption and extractor-bias analysis. | A specific prediction mechanism rather than a degradation table. |
| [Asleep at the Wheel](https://arxiv.org/abs/2608.01336) and [the physical-surprise study](https://arxiv.org/abs/2611.01815) | Limits of interpreting JEPA error as the intended anomaly. | A concrete explanation or controlled improvement in the gait setting. |

Uniform masking, [VICReg](https://arxiv.org/abs/2105.04906), [stochastic positions](https://proceedings.mlr.press/v235/bar24a.html), and [stochastic latent variables](https://arxiv.org/abs/2611.09302) are established ingredients. Combining them is not the proposed innovation. Background subtraction also has precedent in [Ren and colleagues' work](https://proceedings.neurips.cc/paper/2019/hash/1e79596878b2320cac26dd792a6c51c9-Abstract.html).

This targeted search does not certify a first-of-its-kind claim. If a matched linear predictor reproduces the gain, the finding may concern the scoring principle rather than JEPA. The paper should say so.

## What the current experiments establish

The accepted five-class cohort contains **450 sequences, 91 sources, and 2,504 windows**.

| Supervised classification method | Seed 42 macro-F1 | Seed 43 macro-F1 | Seed 44 macro-F1 |
| --- | ---: | ---: | ---: |
| Random Forest | 0.267 | 0.482 | 0.333 |
| Validation-selected trained JEPA | 0.244 | 0.252 | 0.331 |
| Untrained encoder with a fitted probe | 0.293 | 0.327 | 0.348 |

These source-weighted results do not show a representation-learning advantage. All three methods use abnormal training labels for classification. Their comparison is not a normal-only anomaly experiment.

| Normal-only score | Seed 42 AUROC | Seed 43 AUROC | Seed 44 AUROC |
| --- | ---: | ---: | ---: |
| Masked latent discrepancy | 0.200 | 0.604 | 0.583 |
| Future-suffix discrepancy | 0.138 | 0.417 | 0.567 |
| Normal embedding distance | 0.323 | 0.625 | 0.250 |

The ranking is unstable. The normal anchors received only **45, 66, and 63 optimizer updates**, while losses still changed substantially. At EMA momentum 0.996, the initial-teacher coefficients remain approximately 0.835, 0.768, and 0.777. These are algebraic coefficients, not percentages of random features. Optimization and teacher warm-up must be checked before interpreting failure.

Progressive training adds 777, 747, and 825 updates while changing data exposure and optimizer state. It cannot isolate condition-order effects. Keep five-class recognition as a secondary result unless an additional matched experiment answers a clear question.

Five or six normal calibration sources produce an infinite threshold under the implemented finite-sample 5% rule. Additional abnormal folders do not fix that shortage. Full measurements remain in [results.md](../results.md) and [evidence.json](evidence.json).

## Scores after the advisory discussion

The measured evidence has not changed, so **research readiness remains 45/100**. An improved plan is not an improved experimental result. Scores are editorial judgments, not acceptance probabilities.

| Current criterion | Score / 10 | Weight |
| --- | ---: | ---: |
| Demonstrated methodological originality | 2 | 20% |
| Importance of the question | 7 | 10% |
| Technical and methodological soundness | 6 | 15% |
| Empirical strength | 3 | 20% |
| Generalization evidence | 2 | 10% |
| Reproducibility and utility | 8 | 10% |
| Supported scientific insight | 4 | 10% |
| ASTAD fit | 9 | 5% |

The weighted sum is 44.5 before rounding. [scorecard.csv](scorecard.csv) preserves the calculation. The following scores concern potential contributions if their required evidence is obtained.

| Proposed direction | Potential contribution / 10 | Feasibility this week / 10 | Decision |
| --- | ---: | ---: | --- |
| Cross-joint predictive benefit with local-target controls and broad evaluation | 8 | 7 | Main direction. |
| Preserve multiple physical explanations and select a useful extra measurement | 8 | 2 | Longer-term world-model program; requires simulation and clinical validation. |
| Explain when contextual teacher targets change an anomaly conclusion | 7 | 8 | Specific alternative if supported. |
| Broader category evaluation without a mechanism | 5 | 7 | Useful evaluation, insufficient as the main innovation. |
| Separate measurement and movement uncertainty using independent reference poses | 8 | 3 | Longer-term work. |
| Calibration arithmetic or another noise ablation | 3 | 9 | Supporting material. |

## The standard for a strong result

A strong positive result needs a complete chain: other joints provide predictive benefit on held-out normal sources; the effect survives a local target; controlled relationship changes affect the score beyond what measurement damage explains; and the score improves a prespecified abnormal-gait endpoint against serious baselines. Category results must show whether one recording subgroup accounts for the gain.

A strong negative result needs equally clear support. It could show that contextual targets change a substantive anomaly conclusion, or that a validated relationship-sensitive score has no useful discrimination within a sufficiently precise interval. An undertrained model, a wide interval, or a nonsignificant difference does not establish either conclusion.

The roadmap proposes four-fold source cross-fitting so each scarce normal source receives an out-of-fold prediction. This remains **exploratory internal evaluation**, because earlier results were inspected. New grouping does not erase that history. New abnormal sources can provide prospectively scored evidence, but the complete binary evaluation still reuses a previously studied normal population.

The [ASTAD call](https://ismart.ece.mcgill.ca/astad4/) lists November 20, 2026 as the deadline, seven content pages plus two reference-only pages, and double-blind review using the AAAI-27 author kit. The plan targets a strong supported workshop contribution. Acceptance is not guaranteed, and the workshop should not be represented as the AAAI main conference.
