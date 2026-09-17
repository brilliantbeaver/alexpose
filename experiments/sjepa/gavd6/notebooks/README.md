# Notebook index

Start with the current [motion-preservation study](../docs/studies/motion-preservation/README.md) and its [six-stage guide](motion_preservation/README.md). The [tutorial reading guide](../docs/tutorials/reading-guide.md) explains prerequisites, expected outputs and the difference between demonstrations, saved evidence and real execution.

Other folders preserve distinct historical experiments or shared data inspection. Numbers specify sequence within each folder; 19–23 retain established references and do not extend the current experiment.

Canonical notebooks are listed below. Old `foundations`, `idea05_signed_laterality`, `idea09_reflection_equivariance` and `iclr_bridge` paths contain compatibility links, not additional experiments. Outputs in historical classification and AMASS notebooks are preserved.

## Synthetic training selection

The [nine-notebook tutorial](synthetic_training/README.md) implements source lesson trials, response-based selection, held-architecture deployment, and independent GAVD landmark evaluation. Follow 00, 01, 02, 03, and 07 for the source experiment; 04, 05, 08, and 06 cover real-video preparation, deployment, crossover, and scoring. Start with the [HAIC setup guide](../slurm/synthetic-training/README.md). Local implementation checks are complete; real GPU experiment results remain to be generated.

## Motion preservation during tracking repair

[Current study; real pretrained-model evaluation pending](../docs/studies/motion-preservation/README.md)

Read 00–01 to establish controlled events and tracking failures, 02–03 to build the evidence and calibrate comparisons, then 04 to evaluate preservation at comparable error removal. Notebook 05 is an optional real-video stress inspection. The [workflow guide](motion_preservation/README.md) explains asset requirements, the explicit `demo` mode and the reserved final evaluation.

| Notebook | Canonical file |
| --- | --- |
| 00 · The question, available data, and the first decision | [00_data_and_question.ipynb](motion_preservation/00_data_and_question.ipynb) |
| 01 · Construct real-event and tracker-error pairs | [01_make_controlled_pairs.ipynb](motion_preservation/01_make_controlled_pairs.ipynb) |
| 02 · Ask the frozen prior, then check the video | [02_prior_flow_and_baselines.ipynb](motion_preservation/02_prior_flow_and_baselines.ipynb) |
| 03 · Learn what to preserve, then lock the comparison | [03_train_and_calibrate.ipynb](motion_preservation/03_train_and_calibrate.ipynb) |
| 04 · Does the method preserve more motion at the same repair quality? | [04_preservation_and_repair.ipynb](motion_preservation/04_preservation_and_repair.ipynb) |
| 05 · Inspect movement preservation on GAVD video | [05_gavd_visual_stress.ipynb](motion_preservation/05_gavd_visual_stress.ipynb) |

## Future feature prediction

[Historical development studies; no established student benefit](../docs/studies/future-feature-prediction/README.md)

Notebooks 00–04 teach the prediction gate; 23 covers a separate source-scaling experiment; 19–22 examine evidence and target accessibility. Start with each study overview. The gate defaults to generated `teach` examples; the [accessibility guide](target_accessibility/README.md) specifies the retained artifacts needed by 19–22.

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
| 01 · Define the reflection-equivariant encoder contract | [01_encoder_contract.ipynb](reflection_equivariance/01_encoder_contract.ipynb) |
| 02 · Explore reflection-based future and reach tasks | [02_futures_and_reach.ipynb](reflection_equivariance/02_futures_and_reach.ipynb) |
| 03 · Freeze the full-GAVD training contract | [03_gavd_contract.ipynb](reflection_equivariance/03_gavd_contract.ipynb) |
| 04 · Train three matched GAVD JEPA variants | [04_gavd_training.ipynb](reflection_equivariance/04_gavd_training.ipynb) |
| 05 · Audit GAVD checkpoints and reflection geometry | [05_gavd_audit.ipynb](reflection_equivariance/05_gavd_audit.ipynb) |
| 06 · Reproduce the full-GAVD comparison on CPU | [06_cpu_replication.ipynb](reflection_equivariance/06_cpu_replication.ipynb) |
| 07 · Reproduce the full-GAVD comparison on CUDA | [07_gpu_replication.ipynb](reflection_equivariance/07_gpu_replication.ipynb) |
| 08 · Train three JEPA variants on AMASS lower-body landmarks | [08_amass_training.ipynb](reflection_equivariance/08_amass_training.ipynb) |
| 09 · Probe frozen AMASS-trained features on GAVD | [09_gavd_frozen_probe.ipynb](reflection_equivariance/09_gavd_frozen_probe.ipynb) |
| 01 · Measure signed-laterality information in frozen features | [01_probe.ipynb](signed_laterality/01_probe.ipynb) |
| 02 · Explore signed-laterality and reflection tasks | [02_futures_and_reach.ipynb](signed_laterality/02_futures_and_reach.ipynb) |

## Gait representation and classification

[historical label-informed curriculum](../docs/studies/gait-classification/README.md)

| Notebook | Canonical file |
| --- | --- |
| 00 · Learn the S-JEPA architecture | [00_sjepa_basics.ipynb](gait_classification/00_sjepa_basics.ipynb) |
| 01 · Inspect GAVD data and sequence boundaries | [01_gavd_data.ipynb](gait_classification/01_gavd_data.ipynb) |
| 02 · Extract and inspect pose sequences | [02_pose_extraction.ipynb](gait_classification/02_pose_extraction.ipynb) |
| 03 · Construct keypoint masks | [03_keypoint_masking.ipynb](gait_classification/03_keypoint_masking.ipynb) |
| 04 · Run the staged training curriculum | [04_staged_training.ipynb](gait_classification/04_staged_training.ipynb) |
| 05 · Inspect learned motion representations | [05_representation_analysis.ipynb](gait_classification/05_representation_analysis.ipynb) |
| 06 · Evaluate gait-classification readouts | [06_gait_classifiers.ipynb](gait_classification/06_gait_classifiers.ipynb) |

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
