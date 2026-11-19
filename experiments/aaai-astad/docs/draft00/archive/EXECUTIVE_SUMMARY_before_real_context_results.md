# Executive summary: the research to do this week

**Recommendation, revised November 14, 2026.** Spend the coming week testing whether abnormal walking can remain predictable while the relationships between body regions differ from normal. Use the full GAVD definition: the `normal` folder is normal, and every other folder is abnormal gait. The new [neurological research review](neurological_research_direction.md) explains the clinical evidence, examples, and longer-term world-model direction.

This gives the paper a clear question. A person can walk abnormally while repeating the same movement smoothly. A model may predict that movement accurately and still miss the abnormality. We should test whether relationships between joints provide information that a joint's own trajectory does not.

For example, predict the left ankle twice. One predictor sees its own observed positions. The other also sees the remaining allowed joints. Both predict the same hidden positions and receive the same random mask. The difference in their errors measures how much the additional joints helped. Summarize this benefit for left-leg, right-leg, and arm targets, then ask whether the profile differs from the normal reference. It can differ through either less or more help. Keep the earlier loss-of-help score as a declared comparison.

Both predictors use the same body reference for centering and scaling. “The ankle alone” restricts the learned prediction context; it does not remove that shared preprocessing.

Claude Opus, GPT-6 Astra, GPT-6 Sol, and an additional GPT-6.1 Sol clinical-literature advisor reviewed the ideas. Their advice helped sharpen the plan, but model agreement is not scientific evidence. The [advisory synthesis](advisory_summary.md) explains corrections, including the fact that the matched predictors still need implementation.

![The proposed experiment trains on normal sources, compares matched predictors, and evaluates held-out videos.](images/neuromechanics_experiment.svg)

*This is the proposed workflow, not a completed experiment. All illustrations are also available in the [visual guide](images/index.html).*

## What the clinical research changes

Stroke can alter selective control and the contribution of each leg. Parkinson's can affect movement size differently from rhythm and coordination. Cerebral palsy combines an early brain disturbance with changing control and body mechanics. Myopathic gait concerns muscle-related problems and should not be treated as one brain disease. The primary studies and their limits are explained in the [condition-by-condition review](neurological_research_direction.md#2-what-the-four-condition-families-actually-suggest).

The common research idea is adaptation under constraints. A person may keep moving through a repeated strategy that is unusual but easy to predict. This motivates studying altered joint relationships. It does not mean that GAVD can reveal muscle force, dopamine state, lesion location, or the cause of an individual clip.

## Why this is a stronger paper target

The current result is that short JEPA training did not improve on the untrained encoder's supervised classification probe. That is useful, but it does not explain why. The normal models received only 45 to 66 optimizer updates, so inadequate training remains a plausible explanation.

The proposed study asks what information makes prediction useful, checks that interpretation with controlled changes to the same sequence, and evaluates the consequence across abnormal gait categories. A successful paper would explain a property of the prediction model and demonstrate its value for detection.

## What we can use now

The annotation inventory contains 1,874 sequences from 348 source videos. There are 291 normal sequences from 32 sources, and 1,583 abnormal sequences from 316 sources. The existing processed five-class cohort has 450 usable sequences from 91 sources.

Seven additional abnormal folders provide 245 source IDs outside the original selection. A filename check found local videos for 49 of those sources, but their decoding and pose quality still need checking. The others require attempted acquisition. Process them under a fixed rule and report what succeeds. More abnormal videos broaden the test but do not add normal examples for calibration.

No source video is labeled with both normal and abnormal gait. We cannot assume natural matched pairs. We can compare a sequence before and after a specified edit to test model behavior. These edited sequences are not new patients or simulations of particular diagnoses.

## The work for the coming week

| Day | Work to complete | Evidence needed before proceeding |
| --- | --- | --- |
| Sunday, November 15 | Freeze labels, source groups, profile definition, metrics, and acquisition rules. Check extra videos and pose overlays. Fix a blinded movement-annotation rubric if reviewers are available. Time one training run. | A reliable data ledger, a score fixed before new results, and a realistic compute budget. |
| Monday, November 16 | Implement and train matched predictors with and without other-joint context. Check a longer normal-training schedule and a target containing only the hidden joint patch. Fit the profile reference using inner normal sources. | The full predictor must gain useful information from other joints. The small normal reference must be reported. |
| Tuesday, November 17 | Test relative limb timing changes, a whole-body timing control, and observation degradation. Compare local and contextual JEPA targets. | The interpretation must survive the controls. Uncertain results remain uncertain. |
| Wednesday, November 18 | Freeze the method and evaluate the accepted abnormal-category cohort. Compare raw JEPA error, normal-space scores, linear prediction, and a validated normal-only baseline. | Paired source-level results, category support, and honest uncertainty. |
| Thursday, November 19 | Write around the supported finding. Rebuild tables from predictions and review claims and anonymity. | A clear contribution backed by completed experiments. |
| Friday, November 20 | Finish checks and submit before the verified cutoff. | A finished manuscript rather than unfinished central experiments. |
| Saturday, November 21 | Archive submitted results and prepare reproducibility material. | A stable record of the work already submitted. |

The deadline is November 20, so Saturday is for follow-up documentation. Check the exact cutoff timezone in the submission system. [Official workshop requirements](https://ismart.ece.mcgill.ca/astad4/).

## The larger research direction

Build a physical forward model that separates movement control, muscles and body geometry, ground contact, and camera observations. Its distinctive target should be **multiple compatible explanations**, followed by a test of which extra measurement distinguishes them. Generating gait from weakness or estimating anatomy from gait already has substantial prior work. The [novelty review](neurological_research_direction.md#7-where-novelty-would-actually-come-from) explains what must be added.

![A future benchmark would preserve uncertainty when different simulated mechanisms produce similar observations.](images/mechanism_ambiguity.svg)

*The example is a proposed experiment, not an established pair of equivalent clinical conditions.*

Keep this simulator and clinical-validation program outside the six-day implementation. This week should produce a focused motion-prediction finding. If time becomes tight, cut extra stochastic-latent experiments and directed joint maps before cutting the local-target, geometry, or source-separation controls.

## What success should mean

Require a consistent chain of evidence. Other joints must improve prediction on held-out normal sources. That improvement must remain meaningful without a contextual transformer constructing the target, conditional on the shared body reference. Controlled changes to limb relationships must affect the score in a way that observation damage does not explain. Finally, the score must add useful discrimination on real abnormal labels, with results reported by category and source.

If that chain fails, investigate the reason. A precise finding that contextual teacher targets change the interpretation of anomaly scores could still be substantial. An undertrained model or a wide confidence interval is not such a finding. Do not replace a failed claim with a favorable seed, a reversed score, or the assertion that calibration arithmetic alone makes a strong paper.

The readiness score remains 45/100 because this consultation adds ideas and a data census, not new model performance. The opportunity is stronger than the original plan: one testable mechanism, broader abnormal-gait evaluation, and explicit standards for deciding what the paper can claim. The [roadmap](research_roadmap.md) gives the implementation and evaluation rules.
