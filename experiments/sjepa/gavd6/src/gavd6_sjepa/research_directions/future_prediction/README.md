# Future-feature prediction implementation

These are historical feasibility gates for future video-teacher prediction. The experiment measures the increment from skeleton history conditional on RGB and declared nuisance inputs. It does not establish student distillation benefit. The [study overview](../../../../docs/studies/future-feature-prediction/README.md) retains the negative development decisions.

Read the package in five groups:

| Group | Files and responsibility |
| --- | --- |
| Data and protocol | [contracts.py](contracts.py) binds configurations and software; [cohort.py](cohort.py) selects recordings/windows; [video_pose.py](video_pose.py) decodes frames and skeletons; [source_inventory.py](source_inventory.py) locates videos. |
| Teacher features | [vjepa.py](vjepa.py) adapts the video teacher; [token_regions.py](token_regions.py) defines token regions; [feature_cache.py](feature_cache.py) stores targets; [nuisance_features.py](nuisance_features.py) defines permitted context; [controls.py](controls.py) constructs matched controls. |
| Validity and cache reuse | [validity_audits.py](validity_audits.py) checks pixel-intervention validity; [readiness.py](readiness.py) checks the distinct direct-gate input contract; [cache_reuse.py](cache_reuse.py) verifies inherited data. |
| Model fitting | [residual_models.py](residual_models.py) and [nested_training.py](nested_training.py) implement the earlier residual comparison; [joint_models.py](joint_models.py), [joint_training.py](joint_training.py) and [joint_calibration.py](joint_calibration.py) implement the repaired joint-ridge comparison and synthetic calibration. |
| Results and execution | [metrics.py](metrics.py), [decisions.py](decisions.py), [reporting.py](reporting.py), [direct_report.py](direct_report.py), [joint_report.py](joint_report.py) score their corresponding versions; [cli.py](cli.py), [notebook_workflow.py](notebook_workflow.py), [inspection.py](inspection.py), [smoke.py](smoke.py) provide execution and inspection. |

The model versions and validity contracts deliberately remain separate. Combining them into one training or audit file would obscure which scientific comparison was performed. The small control and token modules expose independently tested mathematical operations; short length alone is not duplication.

Current software fingerprints enumerate the canonical source and launch files. The [replay snapshot](../../../../scripts/archive/code_layout_20260913/README.md) preserves the original module paths needed by old serialized predictors. Never rewrite a saved receipt to claim a new implementation has an old identity.
