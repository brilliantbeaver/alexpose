# Executive summary: what the completed study supports

**Updated November 15, 2026.** The proposed experiment is now implemented and has been run on real GAVD videos. The strongest defensible paper is a careful study of the limits of prediction-based gait anomaly scores. The current results do not support a claim that the learned JEPA detects abnormalities better than its simpler controls. The [new manuscript](../draft01/manuscript.md), [real results](../context_study_results.md), and [additional controls](../context_study_sensitivity.md) replace the earlier implementation plan as the account of completed work.

The plain question remains useful: if we hide part of the left ankle's movement, does seeing the other joints help more, or less, than it usually does in normal recordings? We trained a predictor that sees the ankle's visible positions, another that also sees nearby body positions, and another that sees every permitted visible joint. They predict the same hidden patch. We then compare their errors with a reference built from separate normal videos.

That comparison is a measurement of what these fitted models can use. It is not a direct measurement of communication between neurons or coordination between muscles. A limited predictor can fail to use useful information. A contextual teacher can also change the target when another joint changes, even though the ankle itself stays the same. Both limitations matter in the completed experiments.

![The comparison uses the same hidden targets and separates model fitting, normal references and test recordings.](../draft01/figures/method.svg)

## What actually ran

The fresh annotation inventory has 348 source videos and 1,874 sequences. We obtained 332 decodable videos, including 184 newly downloaded files. Pose filtering and the fixed target-support rule retained 8,950 windows from 1,303 sequences and 251 sources, including 23 normal sources. Every category outside the `normal` folder is positive in the binary endpoint. This includes nonpathological movement circumstances, so this is not a uniformly clinical disease endpoint.

Four source-separated folds were run with 512 JEPA pretraining updates and 512 prediction-head updates. A separately retained sensitivity used 4,096 head updates after normal-development errors raised concerns about learning adequacy. The primary schedule was not changed after looking at test performance. Both runs include coordinate and latent targets, a random encoder with fitted heads, matched loss controls, fixed ridge predictors, observation-budget checks and normal-reference sensitivity. Supplementary experiments include motion features, visibility and view features, and a supervised five-class comparison.

| Comparison on the same 251 sources | Primary AUROC |
| --- | ---: |
| Learned latent benefit profile | 0.641 |
| Raw full-context latent error | 0.541 |
| Calibrated full-error distance | 0.673 |
| Random encoder with fitted latent heads | 0.651 |
| Learned coordinate benefit profile | 0.686 |
| Random encoder with fitted coordinate heads | 0.701 |
| Simple motion-feature distance | 0.626 |

The profile's difference from raw error is 0.101, with a conditional 95% interval from -0.035 to 0.224. Its difference from the motion-feature control is 0.015, with an interval from -0.140 to 0.179. These intervals include zero. They concern this fitted analysis, not uncertainty from new training, new reference samples or unidentified repeated people.

![Every displayed paired interval includes zero. The raw-error comparison alone does not establish a learned representation advantage.](../draft01/figures/paired_differences.svg)

Longer head training does not settle the question: the latent profile reaches 0.667 while the random-encoder control reaches 0.714. In the separate five-class task, the handcrafted-feature Random Forest has higher macro-F1 than the frozen JEPA probe in every fold. We should report these outcomes rather than select a more favorable comparison.

## What the neurological reasoning contributes

Stroke, Parkinson's, cerebral palsy and muscle disease can affect gait through different combinations of control, strength, mechanics and adaptation. The [clinical review](neurological_research_direction.md) provides the primary evidence and examples. These findings motivate comparing movement relationships, but do not identify the cause of a GAVD clip. Myopathic gait is a muscle-related category, not one neurological disease.

A stronger future world model would distinguish control, body mechanics, ground contact and camera observations. It would retain multiple explanations when video cannot distinguish them, then identify which additional measurement would resolve the ambiguity. This requires validation with participant information and relevant clinical or physical measurements. Renaming an unconstrained latent variable after a brain process would not provide that validation.

![The longer-term model separates the physical process from camera observation.](images/physical_world_model.svg)

## What to do during the remaining week

| Date | Concrete priority | What would count as progress |
| --- | --- | --- |
| November 15 | Finish independent manuscript review, artifact verification, notebook execution and figure checks. | A coherent draft whose claims match the completed results. |
| November 16 | Implement and validate one published pose-anomaly comparator under the same missing-observation and source rules. | A tested adapter and comparable source predictions, or a clearly documented remaining gap. Do not silently remove difficult videos. |
| November 17 | Test a predictor that can retain the own-context solution, first on synthetic processes with known dependencies and then on normal development recordings. | Evidence that the learner can use added information. Treat any new architecture as a new exploratory experiment. |
| November 18 | Obtain independent movement descriptions if qualified reviewers are available. Audit repeated people, views and activity differences. | Recorded agreement and uncertainty, without retrospective diagnosis. The prepared review clips are not completed human review. |
| November 19 | Freeze the evidence and revise around the supported contribution. Check anonymity, references, figures and page limits. | A technically complete workshop submission with unresolved weaknesses stated plainly. |
| November 20 | Complete author checks and submit through the workshop system if the team chooses to proceed. | The workshop lists November 20; verify the cutoff in the submission system because its public page does not state a timezone. |

The [official ASTAD page](https://ismart.ece.mcgill.ca/astad4/) requests double-blind AAAI-27 submissions with up to seven content pages and two reference pages. The manuscript is being prepared to those requirements. No submission has been made.

## How strong is the paper now?

The completed work is much more reproducible and scientifically defensible than the earlier proposal, but it is not yet a strong new-detector paper. Only 23 normal source videos survive filtering, all overlap earlier exploration, each fold uses five reference normals, and the main interval remains wide. Video sources are not verified independent participants. A published anomaly baseline and independent clinical validation are still missing. More abnormal clips do not repair those limitations.

Use the [submission-readiness assessment](../draft01/submission_readiness.md) for current scores and priorities. Earlier scores and advisory conversations in this folder describe previous stages, not new performance evidence. Claude and GPT critiques are useful checks; their agreement is not a scientific result or an acceptance forecast.
