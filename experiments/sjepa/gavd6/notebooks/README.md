# Notebook index

The current [motion-preservation study](../docs/studies/motion-preservation/README.md) has six stages. Other folders preserve distinct historical experiments or shared data inspection. Numbers specify sequence within each folder; 19–23 retain established references.

Canonical notebooks are listed below. Old `foundations`, `idea05_signed_laterality`, `idea09_reflection_equivariance` and `iclr_bridge` paths contain compatibility links, not additional experiments. Outputs in historical classification and AMASS notebooks are preserved.

## Motion preservation during tracking repair

[current; real-model evaluation pending](../docs/studies/motion-preservation/README.md)

| Notebook | Canonical file |
| --- | --- |
| 00 · The question, available data, and the first decision | [00_data_and_question.ipynb](motion_preservation/00_data_and_question.ipynb) |
| 01 · Construct real-event and tracker-error pairs | [01_make_controlled_pairs.ipynb](motion_preservation/01_make_controlled_pairs.ipynb) |
| 02 · Ask the frozen prior, then check the video | [02_prior_flow_and_baselines.ipynb](motion_preservation/02_prior_flow_and_baselines.ipynb) |
| 03 · Learn what to preserve, then lock the comparison | [03_train_and_calibrate.ipynb](motion_preservation/03_train_and_calibrate.ipynb) |
| 04 · Does the method preserve more motion at the same repair quality? | [04_preservation_and_repair.ipynb](motion_preservation/04_preservation_and_repair.ipynb) |
| 05 · Inspect movement preservation on GAVD video | [05_gavd_visual_stress.ipynb](motion_preservation/05_gavd_visual_stress.ipynb) |

## Future feature prediction

[historical development studies; no established student benefit](../docs/studies/future-feature-prediction/README.md)

| Notebook | Canonical file |
| --- | --- |
| 00 · What can past skeleton motion add? | [00_question_and_worked_example.ipynb](future_innovation/00_question_and_worked_example.ipynb) |
| 01 · Align clips, joints and source folds | [01_cohort_and_alignment.ipynb](future_innovation/01_cohort_and_alignment.ipynb) |
| 02 · Teacher features, prefix isolation and lineage | [02_teacher_features_and_validity.ipynb](future_innovation/02_teacher_features_and_validity.ipynb) |
| 03 · Select jointly regularized predictors and controls | [03_matched_predictors_and_controls.ipynb](future_innovation/03_matched_predictors_and_controls.ipynb) |
| 04 · Measure the increment and decide what follows | [04_results_and_next_decision.ipynb](future_innovation/04_results_and_next_decision.ipynb) |
| 23 — Does more source diversity help repaired Experiment 0? | [23_source_learning_curves.ipynb](source_scaling/23_source_learning_curves.ipynb) |
| 19 — Evidence and observability: what did the two studies establish? | [19_evidence.ipynb](target_accessibility/19_evidence.ipynb) |
| 20 — Reflection, time direction and observable motion | [20_temporal_information.ipynb](target_accessibility/20_temporal_information.ipynb) |
| 21 — Student-accessible prediction of contextual teacher features | [21_target_accessibility.ipynb](target_accessibility/21_target_accessibility.ipynb) |
| 22 — Selective future distillation: a falsifiable next study | [22_distillation_design.ipynb](target_accessibility/22_distillation_design.ipynb) |

## Reflection equivariance and signed-laterality probes

[historical controls and probes](../docs/studies/reflection-equivariance/README.md)

| Notebook | Canonical file |
| --- | --- |
| Notebook 09a: Reflection-equivariant encoder contract (Idea 9, core arm) | [01_encoder_contract.ipynb](reflection_equivariance/01_encoder_contract.ipynb) |
| Notebook 09b: Possible futures and reach scaffolds for Idea 9 | [02_futures_and_reach.ipynb](reflection_equivariance/02_futures_and_reach.ipynb) |
| 09c. Freeze the full-GAVD GaitParity training contract | [03_gavd_contract.ipynb](reflection_equivariance/03_gavd_contract.ipynb) |
| 09d. Train the three matched GAVD JEPAs | [04_gavd_training.ipynb](reflection_equivariance/04_gavd_training.ipynb) |
| 09e. Audit GAVD checkpoint health and reflection geometry | [05_gavd_audit.ipynb](reflection_equivariance/05_gavd_audit.ipynb) |
| Full-GAVD GaitParity replication — CPU | [06_cpu_replication.ipynb](reflection_equivariance/06_cpu_replication.ipynb) |
| Full-GAVD GaitParity replication — CUDA GPU | [07_gpu_replication.ipynb](reflection_equivariance/07_gpu_replication.ipynb) |
| Train the three JEPA variants on AMASS 11-landmark lower-body | [08_amass_training.ipynb](reflection_equivariance/08_amass_training.ipynb) |
| Frozen AMASS-11-landmark lower-body JEPA probes on GAVD | [09_gavd_frozen_probe.ipynb](reflection_equivariance/09_gavd_frozen_probe.ipynb) |
| Notebook 05a: Signed-laterality decodability probe (Idea 5, core arm) | [01_probe.ipynb](signed_laterality/01_probe.ipynb) |
| Notebook 05b: Reflection reach and the possible futures of Idea 5 | [02_futures_and_reach.ipynb](signed_laterality/02_futures_and_reach.ipynb) |

## Gait representation and classification

[historical label-informed curriculum](../docs/studies/gait-classification/README.md)

| Notebook | Canonical file |
| --- | --- |
| The shortest useful definition | [00_sjepa_basics.ipynb](gait_classification/00_sjepa_basics.ipynb) |
| One CSV is one sequence | [01_gavd_data.ipynb](gait_classification/01_gavd_data.ipynb) |
| Why sequence-level extraction matters | [02_pose_extraction.ipynb](gait_classification/02_pose_extraction.ipynb) |
| Required de-duplicated table | [03_keypoint_masking.ipynb](gait_classification/03_keypoint_masking.ipynb) |
| Tutorial scale and curriculum scale | [04_staged_training.ipynb](gait_classification/04_staged_training.ipynb) |
| Load the completed curriculum checkpoint | [05_representation_analysis.ipynb](gait_classification/05_representation_analysis.ipynb) |
| The capstone question | [06_gait_classifiers.ipynb](gait_classification/06_gait_classifiers.ipynb) |

## Force prediction from frozen motion representations

[negative feasibility result](../docs/studies/force-prediction/README.md)

| Notebook | Canonical file |
| --- | --- |
| Exploratory frozen-JEPA probe on StrokePiG | [01_strokepig_probe.ipynb](force_prediction/01_strokepig_probe.ipynb) |

## Shared data inspection

[AMASS SMPL-H/DMPL visualization](amass/01_visualize_amass_smplh_poses.ipynb) inspects raw motion and body-model conversion.

## Editing and execution

Generated notebooks are owned by their [builders](../scripts/research_directions/research_notebook_builder_guide.md); edit the builder before regenerating. Classification notebooks and the AMASS visualization are direct sources and retain their historical outputs. Keep new executed copies under an identified run root, separate from source notebooks.

For the current workflow, use the [motion-preservation guide](motion_preservation/README.md) and [Slurm guide](../slurm/motion-preservation/README.md). Historical output roots, run IDs, stopped decisions and held-out data remain unchanged.
