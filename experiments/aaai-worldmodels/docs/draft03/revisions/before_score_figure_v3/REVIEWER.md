# Reviewer assessment of draft03

## Overall assessment

**This is a credible Med-WM benchmark and validation study with a convincing short-horizon forecasting result. Its main acceptance risk is the strength and distinctiveness of the scientific contribution, rather than whether the paper can display a larger sequence count.** The clinical annotation, varied public videos, explicit controls, frozen external transfer, and reproducible artifacts give the work substance. The completed evidence supports useful movement prediction and a careful test of anomaly interpretation. It does not establish a superior gait anomaly detector or an advantage from the tested structured masks.

I would expect reviewers to respond more favorably when the paper leads with its 456 annotated walking sequences and explains what those observations let us test. I would expect a serious methodological objection if those sequences were presented as 456 independent trials or patients. The strongest presentation combines sequence-led scale with the recording groups that protect evaluation from correlated observations.

This is a research assessment informed by separate data, literature, and statistical audits. It is not an actual conference review, an acceptance probability, or a claim that dataset size determines acceptance. The evidence behind the judgments is linked below.

## Emphasize GAVD's real scale and clinical annotation

The recommended description is:

> GAVD was introduced as the largest publicly accessible collection of abnormal-gait videos annotated by expert clinicians. Its complete inventory contains 1,874 normal and abnormal walking sequences. Our study evaluates 456 unique annotated gait sequences, represented by 2,815 forecast windows.

This credits a substantive contribution of the dataset. It also states precisely what was evaluated. The largest-resource claim belongs to the published video collection; our model did not evaluate all 1,874 sequences. The claim should retain its video and abnormal-gait scope and its attribution to the dataset's introduction. [GAVD primary paper](https://arxiv.org/html/2407.04190v1), [author repository](https://github.com/Rahmyyy/GAVD).

An unrestricted claim about every publicly available clinically annotated gait dataset would be difficult to defend. Other modalities have different and sometimes much larger clinical resources. GaitRec, for example, contains 75,732 bilateral force-plate walking trials from 2,084 patients and 211 controls. This is not a matched video benchmark or a performance comparator; it illustrates why the modality and counting unit must be specified. The literature audit did not perform an exhaustive current census. [GaitRec publication](https://www.nature.com/articles/s41597-020-0481-z), [scale verification](reviewer_support/literature_scale.md).

Expert clinical annotation is a strength even though this study does not independently confirm every diagnosis from medical records. Describing the labels as clinically informed gait annotations preserves that distinction without understating their value.

| Coverage stage | Unique gait sequences | Interpretation |
| --- | ---: | --- |
| Full local GAVD catalog | 1,874 | The annotated resource, excluding the duplicated provenance copy. |
| Five gait annotations in this study | 666 | Normal, Parkinson's, stroke, cerebral palsy, and myopathic annotations. |
| Media available | 652 | The reachable local material before reference and quality exclusions. |
| Extraction candidates after known reference exclusions | 641 | The declared eligible material before observation gates. |
| Evaluated common-support cohort | 456 | The sequences that contribute to the reported forecasting study. |
| Normal sequences within that cohort | 141 | The observations supporting normal forecasting, distributed across 29 recording groups. |

| Evaluated annotation | Sequences | Forecast windows | Original recording groups |
| --- | ---: | ---: | ---: |
| Normal | 141 | 465 | 29 |
| Parkinson's | 37 | 292 | 11 |
| Stroke | 58 | 695 | 16 |
| Cerebral palsy | 52 | 518 | 8 |
| Myopathic | 168 | 845 | 27 |
| **Total** | **456** | **2,815** | **91** |

These are substantial movement observations for a heterogeneous clinical-video benchmark. They are a smaller foundation for claims about independent patients, disease-specific effects, or clinical error rates. Both statements can be true because they describe different kinds of coverage.

## What a gait sequence establishes

The GAVD annotation protocol describes uninterrupted walking in one camera-relative direction, with at least two gait cycles in a complete annotated sequence. That definition supports presenting sequences as meaningful walking observations. The implementation retains the published sequence IDs and then extracts overlapping forecasting windows from them. It does not independently detect turns throughout every retained pass. A complete sequence's cycle description also does not guarantee two cycles in the model's shorter 1.5-second prefix.

Separate passes can share a person, body proportions, movement habits, diagnosis, camera geometry, clothing, and pose-estimation errors. Removing a turn improves task consistency but does not remove those common influences. The appropriate distinction is between a sequence as an observational unit and the group used to evaluate generalization and estimate uncertainty. Repeated-observation concerns are well established in neuroscience and medical machine learning. [Lazic, 2010](https://link.springer.com/article/10.1186/1471-2202-11-5), [Saeb et al., 2017](https://pubmed.ncbi.nlm.nih.gov/28327985/).

The current data provide direct reasons to preserve that hierarchy:

- Seventy-three of the 91 recordings contribute more than one retained sequence. One normal recording contributes 30 sequences.
- Seventeen pairs of retained sequence IDs share annotated source frames. Those pairs involve 34 sequence IDs across three recordings. Shared images can contain different people, so this is not proof of duplicate participants or identical poses. It does show that different sequence IDs do not guarantee disjoint footage.
- Of the 2,815 windows, 2,712 share a native frame with another window. There are 155 overlapping window pairs across sequence IDs.
- Three retained sequences contain changing view labels. These changes do not establish physical turns, but the audit cannot certify every retained sequence as newly verified turn-free walking.

All known overlaps remain within their original recording groups and outer folds. The audit finds no recording, sequence, or relation component crossing outer folds. This is a strength of the existing evaluation design. Unknown repeated people across different uploads remain a limitation. The detailed counts, frame intersections, and replay commands are in the [sequence audit](reviewer_support/sequence_audit.md).

## How persuasive is the statistical evidence?

The forecasting result is persuasive within the declared exploratory evaluation. I would not dismiss it simply because the normal reference has only 29 recording groups. The paired effect is large, its conditional interval is clearly positive, and the result survives a reasonable change in observational weighting. Those features support a useful capability. They do not establish a clinically meaningful patient benefit or make smaller incremental effects easy to estimate.

| Comparison | Paired effect and conditional 95% interval | What it supports |
| --- | --- | --- |
| Current posture versus persistence, original primary weighting | MSE improvement **0.04952 [0.03447, 0.06628]**, or **40.8%** relative reduction | Useful short-horizon prediction with each original recording given equal weight. The 29 normal groups contain 141 gait sequences. |
| Same forecasts, equal weight per normal sequence | MSE improvement **0.05766 [0.04178, 0.07967]**, or **41.6% [35.9%, 46.9%]** relative reduction | The forecasting advantage persists under sequence weighting. This is a separately identified post hoc sensitivity using whole-recording resampling. |
| Current posture versus a normal mean-pose prior | MSE improvement **0.03724 [0.02539, 0.05150]** | The tested constant-pose prior does not explain the full forecasting gain. This control was added post hoc. |
| Bilateral versus separately fitted shifted history | MSE improvement **0.00591 [0.00147, 0.01110]** | The observed alignment helps relative to this particular perturbation. Own history remains competitive, and current posture is stronger. |
| Conditional anomaly score versus low motion, internal | AUROC difference **-0.348 [-0.506, -0.191]** | The locked candidate does not improve on the declared movement comparator under the internal endpoint. |
| Same comparison, external coded participants | AUROC difference **-0.172 [-0.357, 0.009]** | Frozen transfer does not demonstrate an advantage. The interval does not precisely establish the external population disadvantage either. |

The equal-sequence sensitivity makes the sequence-level observational unit concrete. It averages window errors within each of 141 normal sequences and then gives the sequences equal weight. Its 5,000 bootstrap draws sample entire recording components within the original folds, carrying every selected component's sequences and multiplicity together. The denominator changes with the selected sequence count. A large recording therefore receives more point-estimate weight than under the primary analysis, while its observations still travel together during resampling. This is a different weighting estimand, not a larger independent cohort. [Sensitivity results](sequence_weighting_results.json), [independent result replay](reviewer_support/sequence_sensitivity_result_review.json).

The original result remains 40.8%. The 41.6% sensitivity does not replace it or acquire preregistered status. The failed first sensitivity attempt stopped at a pandas metadata check before estimating effects; its declaration and failure record are preserved under a separate identity.

Several boundaries matter when a reviewer asks about significance:

1. **The normal forecasting comparison uses 141 sequences, but its uncertainty preserves 29 recording groups.** The original mean gives each recording equal weight; the sensitivity gives each sequence equal weight. Both describe the same frozen predictions under different weighting choices.
2. **The internal anomaly endpoint ranks 91 recordings.** Its AUROC cannot be relabeled as an AUROC over 456 independent sequences. A sequence-level anomaly endpoint would require a separately defined score, calibration mechanism, weighting, and uncertainty analysis.
3. **The intervals condition on saved cross-validation fits.** They do not include all training and model-selection variability, and unknown repeated participants can weaken the recording-independence assumption. Earlier GAVD outcomes influenced the study, so the internal evidence remains exploratory.
4. **Normal calibration is more constrained than the overall sequence count suggests.** Each disjoint calibration set has only seven or eight normal recording groups. The 141 normal sequences do not create 141 independent calibration subjects or justify a 95% clinical normality guarantee.
5. **Multiple exploratory comparisons require restrained language.** The manuscript can report effect sizes and intervals. It should not invent p-values, claim confirmatory significance across the masking family, or treat seeds and bootstrap repetitions as extra subjects.

These constraints do not erase the positive forecasting result. They explain why the same study can support a sizeable technical improvement while remaining uncertain about smaller coordination effects and clinical generalization. The complete numerical reasoning and independent replay are in the [statistical advisory](reviewer_support/statistical_review.md).

## How I expect reviewers to receive the paper

| Review dimension | Current assessment | Reason |
| --- | --- | --- |
| Dataset relevance and observational coverage | A substantive strength | The work uses clinician-annotated video, hundreds of gait sequences, multiple gait annotations, and heterogeneous recording conditions. The full resource and analyzed subset are clearly distinguished. |
| Evaluation discipline | A strength | Related observations remain grouped, target support is matched, simple alternatives are retained, and the external pipeline is frozen. The frame-overlap audit explains why those protections matter. |
| Positive forecasting evidence | Persuasive within its scope | The paired gain is large and remains positive with equal sequence weights and with a normal mean-pose control. It concerns relative 2D articulation, not a measured patient benefit. |
| Anomaly-method improvement | Unsupported by the completed experiment | The locked candidate does not beat low motion. The paper should explain this result rather than imply that a favorable score establishes new neurological knowledge. |
| Structured-mask contribution | An informative controlled comparison | The equal-budget and random-group controls are useful. The tested anatomical and motion-balanced policies do not improve on uniform masking. This is not evidence that all structured masking fails. |
| Algorithmic novelty | The main vulnerability | Current-posture regression and established masking families are familiar. The new contribution must be the reusable evaluation question, protocol, and empirical finding, with a clear distinction from the closest gait-anomaly work. |
| Medical generalization | Limited but honestly described | Recording IDs are not verified patient IDs. External coded-participant transfer is valuable, while acquisition, age, quality, and identity differences remain. |
| Reproducibility and clarity | A strength | The study retains unsuccessful arms, source evidence, executable notebooks, editable figures, and independent numerical checks. These support trust but do not replace a scientific contribution. |

The [official Med-WM call](https://med-wm.github.io/aaai27-mwm/) includes datasets, benchmarks, uncertainty, and validation. A tested movement-observation component therefore has a legitimate workshop role. The paper does not need to claim disease-progression simulation or intervention modeling to fit that scope.

My present preference is a focused benchmark or validation paper. A short, 2–4 body-page submission could make the central finding especially clear, with complete experiments and audits in the supplement. A full, 6–8 body-page benchmark remains defensible if the main text explains the reusable advance over existing evaluation practice and uses its additional space for substantive evidence. The current manuscript's format eligibility alone does not settle that contribution question. Negative comparisons do not determine the appropriate format either; the amount of useful, well-explained evidence does.

I would expect mixed reactions to a full paper whose novelty appears to be a small predictor and a failed new score. I would expect a more constructive review of a paper that establishes a concrete benchmark for separating forecasting skill from anomaly-ranking validity, shows the same issue under frozen transfer, and provides enough artifacts for other researchers to test that distinction. This is a conditional judgment about the argument reviewers will assess, not an estimate of acceptance odds.

## Changes with the greatest scientific payoff

The sequence-first dataset description, scoped largest-video claim, sequence-to-recording hierarchy, and clustered weighting sensitivity are addressed in this revision. They improve the paper's accuracy and make its scale easier to appreciate. Further gains should come from evidence and community value rather than stronger adjectives.

First, make the benchmark easy for another group to use. Provide the permitted sequence inventory, grouping and fold membership, exact forecast targets, mask definitions, normal-only fit rules, score directions, and baseline code. Retain both sequence errors and recording summaries. Raw-media rights and access limits should remain explicit. A precise, reusable evaluation advance is more convincing than an unsupported first-of-its-kind claim.

Second, sharpen the comparison with the closest gait-anomaly approaches. Explain what ambiguity this benchmark resolves that an ordinary forecasting or classification leaderboard leaves open. A matched existing method on this exact observation contract could strengthen a future method-comparison claim. Headline scores from different cohorts or supervised adaptation cannot supply that comparison.

Third, prioritize independent normal coverage in the next experimental iteration. All 32 annotated normal GAVD recording identities are already in scope, and 29 survive the current gates. More conditions or more windows from those recordings cannot supply missing independent normal people. New data should add verified participants and recording conditions, with separate development, calibration, and evaluation roles fixed before outcomes are inspected.

Fourth, pursue a small additional model only when it tests a specific competing explanation. Normal-only nearest-posture retrieval or robust recent-velocity extrapolation could help establish what current posture already explains. Any analysis on previously exposed GAVD outcomes remains exploratory and needs its own identity. A larger search against abnormal labels would not repair the evidence gap.

Finally, connect physiology to validated observations. Verified walking events, consistent person tracking, and acquisition-matched clinical reference data would strengthen a future claim about movement organization. Projected joint error alone does not identify muscle coordination, an affected side, or a neural mechanism.

## Recommended central message

> We evaluate 456 clinically annotated gait sequences from a large public gait-video resource. A compact model learns useful short-term articulation, while controlled comparisons show that forecasting skill and abnormality ranking are distinct validation problems. The benchmark preserves related-recording groups and exposes the strongest simple alternatives, providing a reproducible movement-observation test for medical world models.

This message gives reviewers a positive contribution they can assess. It preserves the genuine observational scale and the statistical protections that make the findings credible. The scale statement is strongest when it communicates what was observed, while the inference statement explains what the study can generalize beyond.

## Evidence and revision record

- [Revised manuscript](manuscript.md) and [PDF](manuscript.pdf).
- [Sequence inventory, overlaps, and annotation audit](reviewer_support/sequence_audit.md).
- [Primary literature and workshop-scope verification](reviewer_support/literature_scale.md).
- [Independent statistical advisory](reviewer_support/statistical_review.md).
- [Declared post hoc sequence sensitivity](reviewer_support/sequence_sensitivity_protocol.json) and [independent result replay](reviewer_support/sequence_sensitivity_result_review.json).
- [Sequence-weighted results](sequence_weighting_results.json) and [per-sequence forecast errors](sequence_level_forecast_errors.csv).
- [Pre-revision snapshot](revisions/sequence_scale_before_v2/snapshot_manifest.json), preserving the earlier manuscript, figures, build evidence, and acceptance record.

The reporting revision does not refit models, alter cohort eligibility, change the frozen primary score, or relabel an old run. Its new sequence-weighted analysis is explicitly post hoc. The revised publication has a new evidence identity; earlier exact-artifact acceptance remains tied to the earlier hashes.
