# Figure 1 evidence and interpretation review

## Latest keypoint and workflow concept

Reviewed anew on 2026-09-26. This assessment concerns the latest figure with two illustrative gait-keypoint icons, training on 112 people, and evaluation on 14 development people. It supersedes the earlier encoder/teacher-graph assessment. I read the current plotting source and caption, inspected the replacement PNG, and checked the paired scoring operation in the implementation. Only this review file was changed.

The new concept communicates the experimental question more directly and does not fabricate empirical evidence. The two icons use twelve hand-designed 2D points for shoulders, elbows, wrists, hips, knees, and ankles. The visible label **“Illustrative keypoints”** and stacked-window cards distinguish them from observed frames or reconstructed predictions. They illustrate the location of a synthetic knee edit; no angle, measured improvement, clinical condition, or participant identity is asserted. The builder's 28° planar rotation is explicitly a drawing choice and is not a scientific intervention parameter. The actual experiment applies nominal 0/5/10/15° edits to the 3D motion before projection; the graphic must not be used as a geometric validation of those edits.

The workflow's 128-frame, 12-joint labels and 112-training/14-development-person counts match the retained experiment. Three fitted seeds are named without treating them as extra people. The caption explicitly states development reuse, so the final panel does not imply independent confirmation.

Training paths remain distinct: noisy estimated poses enter the student and predictor; clean projected references supply teacher targets. Endpoint and delta occupy separate cards with **OR**, **“same base loss,”** and **“separate runs.”** These labels describe alternative auxiliary choices, rather than two simultaneous losses or a shared fitted teacher across procedures. Details of EMA updates, detached targets, centered/scaled score residuals, and query support remain in the main methods and Equation 2. Their omission from this conceptual figure does not change the depicted experiment or imply pure self-supervision.

The frozen encoder is followed by a trained coordinate readout, and the readout's reference-coordinate supervision is visible. The caption retains the separate direct-fitting procedure that updates encoder and readout jointly. The evaluation input is one observed window, a or b, and references point to scoring rather than prediction. The figure depicts no pair concatenation, future-frame forecasting, action-conditioned rollout, or reconstructed success example. Naming is explicitly marked post hoc. Zero response and direct fitting need not be added as parallel diagram paths because their different roles remain explicit in the main comparisons; adding a zero-baseline trajectory would be incorrect.

## Findings and disposition

**F1-E01 — Closed by replacement.** The earlier graphic's phrase “same masked locations” was corrected in its intermediate revision and is absent from the current conceptual figure. Exact common-query support remains specified in the methods. That prior terminology issue does not apply to this design.

**F1-E02 — Closed.** Independent restoration and paired scoring are now explicit in both the graphic and caption. The scoring box says **“Paired asymmetry-change error,”** its reference arrow is labeled **“References a,b,”** and the footer reads **“Restore separately; score the change from both windows.”** The caption states that paired outputs and references determine asymmetry-change error, while trajectory and post hoc naming checks are scored per window. This matches `response_contrasts`: predicted and reference asymmetry are each differenced between the original and edited endpoint before their absolute discrepancy is scored. I re-inspected the rebuilt PNG and exact caption after this correction.

**Final disposition: approved for scientific fidelity and readable interpretation.** No unresolved material numerical, clinical, training-access, feature-objective, or paired-scoring misstatement remains in the latest concept. The figure distinguishes an illustrative edit, separate objective fits, independent window restoration, and reference-based evaluation without asserting successful clinical reconstruction or independent confirmation. Final PDF rendering and package checks are separately recorded by the editor.

Implementation authority: `src/gavd6_sjepa/research_directions/gait_fidelity/evaluation.py:90` (`response_contrasts`); `training.py` (student/reference branches and fresh frozen-encoder readout fitting); `response_objectives.py` (separate endpoint/delta objectives and common-query support); and `src/gavd6_sjepa/research_directions/synthetic_training_v2/models.py` (`RestorationModel.forward` and encoder-only EMA update). The displayed participant counts were previously reconciled to completed export/ledger counts rather than planned configuration counts.

Final wording check: the diagram now says **“Knee-angle trajectory error”** and the caption says **“knee-angle trajectories.”** This terminology matches the evaluated angular waveform outcome and does not change the approval above.

## Final reviewed artifact hashes

The following hashes bind the corrected keypoint/workflow concept and caption reviewed above.

- `paper-v08.tex`: `b726551c3e59666e29c0dbe2f6fefc5b55df1ba285a977cf41519bb9e98fe011`
- `scripts/build_figures.py`: `06f42606c966d85fc7e47acf6115d5de53590ae29f3e3cf82e4d7d339dde90ba`
- `figures/method.pdf`: `f8d6ed4f2203ec9f097bdf6b988cecc5d1db0bf519c98224d0968afb53816c44`
- `figures/method.png`: `2e7ba763009078a37f3d7ea94103e8af90e7b4b4c148327f701d2371170d222a`
- `figures/method.svg`: `b3a75bd29a1c5770e6d9b0e217d645c67e3db2c1a39a37d58134d76d072c583a`
