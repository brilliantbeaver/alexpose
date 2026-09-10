# RISEx submission assessment for `paper_v8`

Assessed 10 September 2026 against the [RISEx call for papers](https://conference.albertarobotics.ca/call-for-papers/) and its supplied one-page template. This is an editorial assessment, not an acceptance prediction.

## Recommendation

Do not submit V8 unchanged. RISEx accepts a one-page archival paper about current robotics and AI research. V8 is an eight-page study of skeleton JEPA representations and has no robot, policy, control loop, interaction task, or robotics benchmark. Its best route to relevance is as a compact AI-methods report for **human-aware robotic perception**: before a robot uses a learned representation of a moving person, it should be checked for preservation of task-relevant geometric information. That is a plausible motivation, not an evaluated robotics claim.

The deadline listed by RISEx is 10 September 2026, 11:59 PM UTC. The call does not state a prior-publication or dual-submission policy. Because RISEx describes the one-page paper as archival, authors should obtain an answer from the organizers before submitting work that is also intended for another archival venue.

## Multidimensional editorial grade

Scores use a five-point scale: 5 is compelling for this venue; 3 is credible but incomplete; 1 has little support. They are not probabilities.

| Dimension | V8 | One-page target | Reason |
|:--|--:|--:|:--|
| AI-methods relevance | 3 | 3 | JEPA, structured masking, and representation evaluation are in scope as AI. |
| Robotics relevance | 1 | 2 | Human motion is relevant to human-aware robots, but no robot-related task is evaluated. |
| Technical soundness | 4 | 4 | Source-held-out folds, matched starts, fixed update budgets, and source-cluster resampling are clear strengths. |
| Evidence strength | 3 | 3 | The result is consistent across five seeds, yet conditional on one development cohort and fitted models. |
| Novelty | 2 | 3 | The contribution is a focused diagnostic/evaluation finding, not a new JEPA method. |
| Clarity at one page | 1 | 4 | V8 cannot be reduced verbatim; a single question and one results table can be clear. |
| Reproducibility | 3 | 3 | Complete prediction grids support the main mask results; some reflection results survive only as summaries. |
| Publication strategy | 1 | 2 | An archival one-page version is hard to justify unless RISEx's overlap policy is confirmed. |

The current overall fit is **2/5**. A carefully scoped one-page snapshot can reach **3/5** as an AI-methods contribution, but cannot honestly become a strong robotics paper without new work.

## What the paper can claim

The defensible contribution is a controlled test of whether masked latent prediction preserves a known geometric relation in articulated motion. In 625 pose sequences from 93 source videos, all 33 landmarks are retained as a 16-by-33 spatiotemporal token grid after four-frame patching. Five source-held-out folds prevent a video and its clips from appearing in both encoder training and test evaluation. Across five seeds, trained encoders match their own clip's hidden teacher features more often than mismatched-source targets, but their frozen ridge readouts estimate the bilateral speed contrast less well than matched initial encoders. This is a useful warning for representation learning; it does not establish poorer robotic perception, clinical validity, or a general limitation of JEPA.

## Central reviewer risks and actions

| Risk | Why a reviewer may object | Action in the drafts |
|:--|:--|:--|
| “Why robotics?” | The study has no robot task. | State the human-aware perception motivation once; do not claim deployment or control. |
| “The endpoint changed during processing.” | Prepared and original targets have only 70.4% sign agreement and \(R^2=0.218\). | Name this as a limitation, rather than presenting a representation-wide conclusion. |
| “The readout is the problem.” | The expanded summary changes dimension, temporal features, and missingness support; ridge penalties often reach the grid maximum. | Attribute the result to the tested training-and-readout pipeline. |
| “One page hides the evidence.” | The protocol and uncertainty are essential. | Retain the split unit, matched control, seed count, primary numbers, and a scoped limitation. |
| “Why an archival page?” | The current call leaves overlap policy unstated. | Confirm policy before upload; otherwise use these drafts as a poster handout or future submission seed. |

## Evidence selected and deliberately omitted

The final candidate uses the most interpretable completed result: initial-encoder \(R^2=0.223\), trained-teacher \(R^2=0.101\)–\(0.114\), and trained-minus-initial differences from \(-0.109\) to \(-0.122\) across five training arms. It retains the correct-versus-mismatched target diagnostic (375/375 trained checks; 33/75 initial checks) because it explains the apparent contradiction.

It omits the synthetic-only explicit reflection-loss result, the incomplete whole-trajectory and interior-gap comparisons, detailed license language, and historical experiments. The mask comparison is condensed because all three effects cross zero. Clinical examples motivate bilateral geometry but are not used as labels or outcomes. This selection follows the one-page template's standard headings: Introduction, Materials and Methods, Results and Discussion, Conclusions, and References.

