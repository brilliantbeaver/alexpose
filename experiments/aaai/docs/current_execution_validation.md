<!-- iteration02-navigation -->
> Historical study or planning record. Its scientific results and run identities remain unchanged. The executed [iteration02 study and draft02](draft02/README.md) contain the active results, six new notebooks and independent participant replication.

# Current notebook execution and artifact validation

All twelve notebooks passed in independent fresh kernels on November 15, 2026, using the prepared real GAVD workspace. The sum of recorded notebook durations is **133.96 seconds**. Every one of the 73 code cells executed, and the saved outputs contain no error outputs. The authored notebooks were not rebuilt or modified during this validation. Executed copies are separate artifacts.

This validates local execution and output provenance. It does not establish a learned JEPA advantage, clinical usefulness or generalization to independent people.

## Command and environment

The command was run from `experiments/aaai`:

```bash
.venv/bin/python -u scripts/validate_notebooks.py --real
```

The runner creates a temporary kernelspec tied to the experiment's own interpreter, starts a new kernel for each notebook, and closes it afterward. It explicitly sets `AAAI_DEMO=0`, `AAAI_EPOCHS=3` and `AAAI_RUN_SPE_ABLATION=0`. Root `.env` loading therefore does not turn this run into synthetic mode. Existing compatible training artifacts can be reused; fresh kernels do not imply fresh training from scratch.

The interpreter is CPython 3.12.13 on macOS arm64, from `experiments/aaai/.venv/bin/python`. The observed packages include PyTorch 2.15.0, NumPy 2.5.4, pandas 2.3.3, nbclient 0.11.0, nbformat 5.11.1 and ipykernel 7.4.0. The repository's `ambient` package is available through the experiment environment.

The [execution report](../artifacts/real_notebook_checks/execution_report.json) records durations and completion. The [captured command output](../artifacts/real_notebook_checks/validation_stdout.log) records the pass sequence. The [output audit](../artifacts/real_notebook_checks/output_audit.json) records code-cell execution, video outputs, and hashes of the authored and executed notebooks.

## Per-notebook results

| Executed notebook | Cells | Seconds | Status |
| --- | ---: | ---: | --- |
| [00_research_and_gavd_inventory.ipynb](../artifacts/real_notebook_checks/00_research_and_gavd_inventory.ipynb) | 12 | 2.45 | Passed |
| [01_video_pose_and_quality.ipynb](../artifacts/real_notebook_checks/01_video_pose_and_quality.ipynb) | 14 | 3.96 | Passed |
| [02_joint_tokens_and_uniform_masks.ipynb](../artifacts/real_notebook_checks/02_joint_tokens_and_uniform_masks.ipynb) | 16 | 3.92 | Passed |
| [03_normal_only_jepa_and_vicreg.ipynb](../artifacts/real_notebook_checks/03_normal_only_jepa_and_vicreg.ipynb) | 15 | 4.71 | Passed |
| [04_progressive_condition_adaptation.ipynb](../artifacts/real_notebook_checks/04_progressive_condition_adaptation.ipynb) | 13 | 7.22 | Passed |
| [05_forecast_anomalies_and_explanations.ipynb](../artifacts/real_notebook_checks/05_forecast_anomalies_and_explanations.ipynb) | 15 | 38.66 | Passed |
| [06_capstone_rf_vs_jepa.ipynb](../artifacts/real_notebook_checks/06_capstone_rf_vs_jepa.ipynb) | 16 | 21.35 | Passed |
| [07_ablation_and_paper_evidence.ipynb](../artifacts/real_notebook_checks/07_ablation_and_paper_evidence.ipynb) | 15 | 6.84 | Passed |
| [08_biology_cohort_and_source_folds.ipynb](../artifacts/real_notebook_checks/08_biology_cohort_and_source_folds.ipynb) | 16 | 17.50 | Passed |
| [09_matched_context_prediction.ipynb](../artifacts/real_notebook_checks/09_matched_context_prediction.ipynb) | 12 | 17.62 | Passed |
| [10_source_profiles_and_classifier_comparison.ipynb](../artifacts/real_notebook_checks/10_source_profiles_and_classifier_comparison.ipynb) | 14 | 4.15 | Passed |
| [11_interventions_and_manuscript_evidence.ipynb](../artifacts/real_notebook_checks/11_interventions_and_manuscript_evidence.ipynb) | 14 | 5.58 | Passed |

The displayed cell count includes Markdown and code cells. These times cover the notebook walkthroughs, cached artifact loading and their bounded calculations. They are not the wall-clock cost of full acquisition, extraction or the two research training schedules.

## Real studies and illustrative cells have different provenance

The [real-data latest pointer](../artifacts/context_study/gavd/latest.json) was checked before and after execution. It selects **primary run `3044566091261819`**, with 512 JEPA pretraining updates and 512 updates per independent prediction head. It does not select the longer schedule simply because that schedule has a higher point estimate. The 4,096-head-update run, `eba2bd5a6e05efa1`, remains a separately retained sensitivity with the same 512-update pretraining.

The foundation notebooks 00–07 use the earlier focal five-condition workflow. Notebook 03 displays 2,504 real windows and a compatible normal anchor in `artifacts/gavd/e9a0527c16adec52`. Their [historical results](results.md) and [historical verification record](validation.md) do not become results of the revised context experiment. Notebook 04 verifies that progressive adaptation leaves the normal anchor unchanged. Notebook 07 explicitly leaves the optional positional-noise ablation disabled in this execution pass.

The primary-study notebooks 08–11 use the current all-twelve-folder manifest. Their small teaching calculations remain clearly separate:

- Notebook 08 labels its three-row predictive-help example as invented values.
- Notebook 09 fits an eight-update real-data teaching exercise. That exercise is not the completed 512-update primary model or a convergence demonstration.
- Notebook 10 uses an invented three-region matrix to explain reference standardization, then loads the completed real study in separate cells.
- Notebook 11 uses synthetic motion and a tiny two-update model to demonstrate paired edits and the four-factor identity. Its right-ankle factor table is explicitly artificial. Separate cells load the actual primary-run development-control protocol and geometry results. The real eight-joint teacher measurements remain in their [source-joint artifact](../artifacts/context_study/gavd/3044566091261819/fold_0/development_controls/teacher_decomposition/source_joint_components.csv) and [evidence report](context_study_sensitivity.md).

A `GAVD` execution-mode label therefore identifies the data-backed workflow; it does not relabel deliberate illustrative cells as empirical findings.

## Output inspection

Notebook 08 emitted two inline HTML video outputs with embedded MP4 data: one normal example and one stroke-folder example. The corresponding preview files decoded successfully, with 96 and 120 frames respectively. Their first and midpoint frames were visually inspected in the [preview contact sheet](../artifacts/real_notebook_checks/08_embedded_video_frames.png). This confirms that the notebook supplies real walking-video context. It is a limited visual spot check, not independent clinical annotation or comprehensive playback review.

The same notebook displays 1,874 annotation sequences from 348 source IDs, 9,474 pose-QC windows, and the fixed outer folds. The outer-test normal counts are 6, 6, 6 and 5, with 57 abnormal sources in each fold. The stricter profile-support gate produces the 251-source study cohort.

Notebook 10 displays a completed, nonsynthetic primary summary with 251 sources, 23 normals, 1,303 sequences and 8,950 windows. Its result cells contain the actual ranking table, paired bootstrap, reference sensitivity and all eight fold/method rows of the separately supervised RF/probe comparison. The latent profile AUROC is 0.641495 and raw full-context AUROC is 0.540809. Their paired difference is 0.100686 with conditional 95% interval [-0.035088, 0.224275]. The interval includes zero. These are loaded measured results, not a pending placeholder.

Notebook 11 displays the completed primary summary and the real fold-0 control protocol with `synthetic=False` and `training_sources_only=True`, followed by measured geometry summaries. Twelve fitting-normal sources supplied long windows; five retain all full-profile support under every paired edit. That coverage differs from the raw eight-joint teacher diagnostic's 87 observed source-joint rows across twelve sources. These are development sensitivity checks, not held-out mechanism validation. Neither real-result notebook emitted a missing-artifact or missing-control fallback message.

## Acquisition and run verification

The acquisition ledger records **332 decodable videos out of 348 annotated source IDs**, including 184 newly acquired recordings. Sixteen sources remain failed or unavailable. This is an availability count, not a count of patients or complete independent clinical cases. Fresh pose QC retains 311 sources and 28 normal sources; the frozen profile-support cohort retains 251 sources and 23 normals. Exclusions remain visible in the ledgers and [results report](context_study_results.md).

Both completed research runs have separate passed verification reports:

| Study | Artifact verifier | Checks | Required files |
| --- | --- | ---: | ---: |
| Primary, 512 head updates | [Primary verification](../artifacts/context_study/gavd/3044566091261819/verification.json) | 8 passed | 173 |
| Sensitivity, 4,096 head updates | [Sensitivity verification](../artifacts/context_study/gavd/eba2bd5a6e05efa1/verification.json) | 8 passed | 173 |

These reports check current corpus and mask identities, fold roles, losses, references, checkpoints, condition-classifier artifacts and metric consistency. They do not establish clinical validity. The [supplementary source-control summary](../artifacts/context_study/gavd/3044566091261819/source_controls/7407af320461a092/summary.json) records 251 sources shared with the primary comparison. The longer schedule and controls remain supporting analyses, not replacement primary endpoints.

## Separate software tests

A separate full-suite execution used the same CPython 3.12.13 experiment environment and command `.venv/bin/python -m pytest tests -q`. It completed with **105 passed in 16.07 seconds**. The exact invocation and output are preserved in [final_pytest.log](draft01/reviews/final_pytest.log). This test result is distinct from the twelve notebook executions and the real-data artifact verifiers. No duplicate test run was needed for this documentation pass.

## What remains unestablished

There is no completed independent human phenotype review or agreement analysis. The annotation template is blank evidence infrastructure, not expert validation. SeeKer and STG-NF have not been reproduced as validated published comparators for this partially observed sixteen-joint protocol. The [baseline feasibility review](published_baseline_feasibility.md) describes that gap.

Source IDs are not verified participant identities. All 23 retained normal sources were present in the earlier development index. Missing media, pose visibility and camera differences can affect the selected cohort. The five-source normal references and conditional bootstrap cannot represent full training, sampling or population uncertainty. Neither the primary nor longer schedule establishes a learned-encoder advantage. The [results](context_study_results.md), [sensitivity report](context_study_sensitivity.md) and [manuscript](draft01/manuscript.md) bound the scientific conclusions separately from execution success.
