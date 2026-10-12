# Terminology and explanation audit

This editorial pass responds to the request to rename the two auxiliary losses and explain technical language throughout the compact writeup. The lead editor reviewed the complete manuscript, both captions, and the labels in the unchanged figures, then checked the revised descriptions against v08's methods, Appendix B, and technical supplement. This is a terminology and fidelity review, not a new independent scientific review or a new experiment.

## Loss names and their meaning

| Source terminology | Reader-facing name and explanation |
|---|---|
| Endpoint auxiliary | **Per-sequence feature-matching loss**. Compare predicted and teacher features separately for the original clip and edited clip. |
| Delta auxiliary | **Paired feature-change loss**. Compare the student's edited-minus-original feature difference with the teacher's corresponding difference. |
| Auxiliary | Explained as an additional training-error term, or loss. The short manuscript uses the descriptive names rather than the unexplained label. |
| Endpoint | Retained only in the mapping to the original figure. Explicitly means a member of the motion pair, not the beginning or end frame. |
| Shared-error cancellation | Explained directly: two inaccurate individual predictions can still have an accurate difference if they share an error. Feature agreement does not directly measure knee angles. |

The original result figure is unchanged. Its `endpoint` and `delta` labels map to the new names in the immediately preceding prose. No numerical result was relabeled as a different experiment.

## Whole-document terminology pass

| Area | Clarification or replacement |
|---|---|
| Pose, trajectory, anatomical assignment | Joint positions, their movement over time, and correct left–right naming. |
| JEPA and representations | Expanded JEPA and described learning numerical features by predicting reference features. The concept caption explains encoders, predictor, inputs, conditioning variable, and mismatch measure. |
| Gait asymmetry | Introduced as differences between legs, then defined specifically as right-minus-left knee-angle excursion for this experiment. |
| AMASS and windows | Identified AMASS as recorded 3D body motion; windows are short clips. The dataset subset names remain proper names. |
| Projection | Described as mapping joints into camera images; the metric paragraph explicitly distinguishes 2D image angles from anatomical 3D accuracy. |
| Ankle-height gating, tapering, mirroring | Edits are activated according to ankle height and faded at clip boundaries; mirroring reflects the body and exchanges sides. |
| Occlusion and detector uncertainty | Blocking visibility and errors in locating a person. Estimators receive known person locations. |
| Development, held conditions, confirmation | Identified the development set as the people used for comparisons, explained the withheld edit/estimator, and stated that reserved test participants were not evaluated. |
| Normalization | Positions are shifted and rescaled using only available observations. |
| Tokens and model dimensions | Replaced token jargon with groups of four samples per joint. Retained a four-layer transformer and explained that it combines information across joints and time; secondary width/head dimensions remain in the source methods. |
| Student and teacher | The student converts noisy joint observations to features. A predictor learns clean-reference features supplied by a teacher whose parameters follow a weighted running average of the student's. |
| Cross-entropy and anti-collapse regularization | Cross-entropy compares teacher and predicted feature probabilities; additional penalties discourage identical features across clips. |
| Pretraining, readout, frozen encoder | Initial feature learning; a small output network mapping features to coordinates; keeping encoder parameters fixed. The different update schedules remain explicit. |
| Coordinate pretraining, shuffled references, initialized features | Pretraining to predict coordinates, incorrect reference pairings, and untrained encoders. |
| Excursion, response, waveform, localization | Defined the percentile range, change in asymmetry, angular error over both legs and time, and position error divided by the enclosing image-box diagonal. |
| Failure costs | Explicit worst-case costs for invalid predictions where the reference permits measurement, with original 720°/180° values. |
| Seeds and averaging | Different randomized training runs; scores aggregate by clip, motion, person, and run. |
| CI, crossed bootstrap, person-t | Expanded confidence interval; explained separate resampling of people and runs, and person variation after averaging runs. |
| Core, change, low scalar, dense | Base JEPA model; additional response and short-segment penalties; reduced weight on a single-number response; angle-change supervision for each leg and timestamp. These explanations decode all model labels in the result figure. |
| Gradient influence and adaptive reuse | Different initial contributions to the parameter-update signal; repeatedly using development results to guide later experiments. |
| Checkpoints, monocular reconstruction | Saved model parameters; reconstruction from one camera. |
| Ambiguity, calibrated uncertainty, ensemble disagreement | Conflicting trajectories fitting the same observations; error ranges adjusted against separate reference measurements; disagreement among independently trained models. |
| Marker-based reference, interpolation, priors | Motion capture using body-mounted markers; filling gaps from neighboring frames; anatomical assumptions. |
| Matched coverage and selective reporting | Reporting the same proportion of whole trial pairs and checking whose measurements are withheld. |

Published titles, model names, dataset names, and standard numerical/statistical notation remain intact. The prose explains their role where needed; it does not turn the short brief into a glossary of bibliographic titles.

## Fidelity and visual checks

The changes preserve completed cohort sizes, synthetic perturbations, training schedules, reported errors, the exploratory visibility finding, and uncertainty about all three primary comparisons. The clean-reference training information remains explicit. Restoration is not described as forecasting, physical understanding, clinical validation, or older-adult generalization. The future proposal still uses independent 3D references, observation-only no-change controls, simple comparison methods, separate participants, complete-pair reporting, and the limits of a finite trajectory search.

The build checks passed: three body pages plus one reference page, unchanged template and figure assets, no unresolved references or overflowing text. All four pages were rendered and visually reviewed after the substantive revision; page 1 was rechecked after the final caption wording adjustment. Font size and page geometry were not reduced. The original study files and experimental graphics were preserved. Final hashes are recorded in `qa/verification.json`.
