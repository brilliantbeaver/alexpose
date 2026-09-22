# Independent review of abstract version 10

19 September 2026. Three independent agent reviewers challenged two successive candidates against the completed seed-17 evidence. Their roles were scientific accuracy, conference contribution, and editorial clarity. All used the same eight dimensions, weights and scoring anchors in both rounds. The root author reconciled their critiques and retained responsibility for the final wording. This process produced no new experimental observations and did not rerun training.

The final draft is [abstract-v10-expanded-results.md](abstract-v10-expanded-results.md), with a 203-word abstract. Candidate A was 187 words; candidate B was 201 words. All three reviewers found candidate B ready with no essential wording corrections. The final text adds the contribution reviewer's optional “on average” qualification to the ankle-separation finding and changes the following conjunction for readability. Scores below are the reviewers' candidate-B scores; that minor final edit was not assigned a new score.

**Rubric and consistent scoring**

Scores use a 0–10 scale: 5 indicates a substantial unresolved weakness, 8 indicates a strong treatment with remaining limits, and 10 indicates no material weakness within the claim's scope. The total is the sum of each score multiplied by its percentage weight and divided by ten. The figures score concerns the existing [paired-effects plot](../results/seed17-complete-analysis-20260919/paired-effects.png) and [amplitude diagnostics](../results/seed17-complete-analysis-20260919/amplitude-diagnostics.png), which all reviewers inspected. It does not reward an abstract for containing a figure.

| Dimension | Weight | Science A → B | Contribution A → B | Editorial A → B |
| --- | ---: | ---: | ---: | ---: |
| Relevance and contribution to the conference | 20% | 6.5 → 6.5 | 6.5 → 6.5 | 7.0 → 7.0 |
| Claim accuracy and evidence support | 20% | 8.0 → 9.0 | 8.5 → 9.0 | 8.5 → 9.0 |
| Evaluation and statistical rigor | 15% | 6.0 → 6.0 | 6.0 → 6.0 | 6.5 → 6.5 |
| Scientific insight and positioning against related work | 15% | 7.0 → 7.5 | 7.0 → 7.0 | 7.0 → 7.5 |
| Reproducibility | 10% | 8.0 → 8.0 | 8.0 → 8.0 | 8.0 → 8.0 |
| Clarity and narrative | 10% | 7.5 → 8.5 | 8.0 → 8.5 | 7.5 → 8.5 |
| Figures | 5% | 7.5 → 7.5 | 8.0 → 8.0 | 8.0 → 8.0 |
| Submission fit | 5% | 8.0 → 8.5 | 8.0 → 8.0 | 8.0 → 8.5 |
| **Weighted total / 100** | **100%** | **71.75 → 75.75** | **73.50 → 75.00** | **74.75 → 77.75** |

Each reviewer held evaluation rigor, reproducibility and figures constant across wording rounds because the evidence and figures did not change. Contribution scores also remained constant. Higher accuracy and clarity scores reflect reduced risk of misinterpretation, not stronger experimental results. The science and editorial reviewers additionally credited clearer presentation of the measurement tradeoff and submission scope; the contribution reviewer left those scores unchanged. The reviewers' final range is 75.00–77.75, with a mean of 76.17/100. These subjective assessments are not acceptance probabilities or a prediction of conference review scores.

Versions 01–07 used the same weights but evaluated an earlier, smaller evidence package and a conceptual figure. Their historical scores remain unchanged and should not be interpreted as a controlled comparison with this expanded-data review. [Machine-readable scores and validation](abstract-review-v10.json) retain both candidates and their component scores.

**Critiques and their disposition**

| Challenge | Revision or reason for retaining the text |
| --- | --- |
| The earlier current abstract reports two people and describes the expansion as unrun. | Version 10 reports the completed eight-person development evaluation and three pose estimators, while keeping the larger research question and remaining repetitions open. |
| Candidate A does not identify the narrow activity population. | The empirical sentence now specifies treadmill walking and people excluded from training. Normal treadmill recordings from one corpus remain an evidence limitation. |
| “Improves joint positions” obscures what was measured. | The result now states reductions in position error and error in position changes over 0.20 seconds. |
| Direct training's advantage over spatial calibration could be mistaken for superiority over all motion controls. | The same sentence explains that smoothing neighboring frames sometimes produces lower movement error. The author notes identify the overall HRNet and combined-corruption comparisons. |
| Hidden-joint amplitude and event results come from an added analysis, not the primary endpoint. | The abstract labels them exploratory. Author notes distinguish all-valid synthetic references and fixed scaling from the unsupported primary amplitude aggregate. |
| “Amplitude” and “occluded” add unnecessary jargon. | The abstract uses “variation in horizontal ankle separation” and “synthetic joints hidden from view.” It retains the signed-separation interpretation rather than inaccurately calling it an absolute ankle distance. |
| Averaged amplitude ratios do not imply that every sequence is amplified. | The final author edit adds “on average.” The notes retain the per-window ratio definition and its limits. |
| “Untrained features” could imply that the whole initialized control is untrained. | The method sentence explains the separately trained output network, and the notes explicitly describe the initialized fixed extractor plus trained output network. Reviewers found the abstract clear in context. |
| “Shuffled references” leaves the intervention unclear. | The abstract describes reference sequences from different time windows; notes preserve the same-person and rendering/extractor restrictions. No complete destruction of correspondence is claimed. |
| “Further experiments examine” can suggest that repetitions have already been evaluated. | The final sentence explicitly calls those experiments planned. It promises an assessment of consistency and uncertainty, not a statistically significant JEPA benefit. |
| A numerical headline might make the result more concrete. | The contribution reviewer judged the comparator, direction, time interval and empirical scope sufficient at this length. The 7.15–8.51% position reduction relative to affine calibration is retained in the evidence notes rather than replacing the motion-control qualification. |
| A more emphatic title could imply successful preservation. | The existing title is retained. It names the evaluation objectives without describing the method as motion-preserving. |

**What the revision cannot resolve**

The completed experiment uses one training seed and eight development people performing normal treadmill walking. Further seeds do not add independent people. The synthetic joint references require independent anatomical review, and no real-video transfer has been demonstrated. Direct training, frozen-encoder methods and the static control differ in objectives, optimization and some architectural details, limiting causal attribution. Shuffled windows can contain similar movement, so their small difference from paired pretraining constrains this recipe without proving that temporal correspondence is generally irrelevant.

The paired-effects figure clearly presents person-level uncertainty conditional on the fitted seed; the amplitude figure preserves unsupported entries but lacks person-level dispersion. Those scientific limitations remain after prose revision. Existing temporal refinement and skeletal representation work also limits generic novelty claims. The [earlier literature review](abstract-literature-positioning-20260919-v01.md) and [version 09 review](abstract-review-v09.md) document that positioning. The present contribution is the controlled empirical question and the emerging distinction between coordinate accuracy and movement fidelity, whose reliability and broader significance still require evidence.

**Evidence used**

Reviewers consulted the [complete diagnostic analysis](../results/seed17-complete-analysis-20260919/README.md), its [verification record](../results/seed17-complete-analysis-20260919/verification.json), detailed comparisons and figures. Scientific checks also examined the [training implementation](../../../../src/gavd6_sjepa/research_directions/synthetic_training_v2/training.py). The analysis independently reconstructed saved aggregates and timing summaries, but original neural prediction arrays are absent locally. This review does not claim a new raw-prediction replay, new motion annotations or completed additional training runs.
