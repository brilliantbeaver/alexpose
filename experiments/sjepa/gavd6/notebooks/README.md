# Notebook index

Study folders live directly under `notebooks/`; there is no additional
`experiments/` level. Notebooks are grouped by purpose. Numeric prefixes define the order only within
their directory; they are not a single global sequence. Notebooks 19–23 retain
their original numbers so existing study references remain recognizable.

## Laterality, future distillation and source learning curves: 19–23

Read 19–22 in order for the bridge from the laterality findings to a possible
skeleton-only student. Notebook 23 inspects a separate experiment: whether more
training recordings improve the repaired RGB-plus-skeleton comparison. It uses
its own source-learning-curve run, rather than the cached panel in notebook 21.

| Number | Notebook | Purpose and execution |
|---:|---|---|
| 19 | [Evidence and observability](iclr_bridge/19_evidence_and_observability.ipynb) | Reconcile the two studies using saved evidence and a synthetic metric example. |
| 20 | [Symmetry and temporal information](iclr_bridge/20_symmetry_and_temporal_information.ipynb) | Explain reflection, time reversal and temporal prediction through labeled synthetic examples. |
| 21 | [Student-accessible future features](iclr_bridge/21_student_accessible_future_features.ipynb) | Inspect the saved accessibility comparison; by default, refit selected CPU models to verify predictions without writing to the run. |
| 22 | [Selective future distillation](iclr_bridge/22_selective_future_distillation.ipynb) | Calibrate selection ideas on synthetic data and explain the remaining student experiment. |
| 23 | [Source learning curves](future_innovation/23_source_learning_curves.ipynb) | Read the cohort reservation, training subsets and available results from `FI_RUN_ROOT`; no fitting or teacher extraction. |

The [19–22 guide](iclr_bridge/README.md) explains their evidence
paths and how to regenerate and execute them. The
[source learning-curve guide](../slurm/future-innovation/SOURCE_LEARNING_CURVE.md)
covers notebook 23 and its Slurm launcher. Historical executed copies stay in
their original artifact directories. The files here are output-free sources.

Regenerate these five source notebooks from the repository root:

```bash
uv run python scripts/research_directions/iclr_bridge/build_notebooks.py
uv run python scripts/research_directions/future_innovation/build_source_learning_curve_notebook.py
```

## Future Innovation: question to decision

Start with the [study overview](../docs/studies/future-innovation/) for the
current evidence and execution boundary. These five notebooks work independently
in fresh kernels. `teach` is the default and uses generated examples;
`FI_TUTORIAL_MODE=inspect` reads an explicit `FI_RUN_ROOT` without training,
downloads, submissions or report writes. `FI_TUTORIAL_MODE=execute` runs the full
Experiment 0 stages in order with an explicit `FI_RUN_ROOT`. Use the separate
[notebook HAIC launchers](../slurm/future-innovation/NOTEBOOKS.md) to schedule
fresh kernels, including the five-fold CPU array, and retain executed copies.
The historical [direct-v2 50-clip gate](../docs/studies/future-innovation/direct-gate-protocol.md) uses
four matched arms, no background-quality/selectivity prerequisite, and a primary
real-minus-no-skeleton comparison. The
[direct-v3 repair](../docs/studies/future-innovation/direct-v3-repair-validation.md)
and [source learning curve](../docs/studies/future-innovation/source-learning-curve-protocol.md)
are separate development experiments. Use the execution guide for the intended
protocol and a new run root; historical runs keep their rules.
A verified teacher-audit rejection finishes 02–04 with **TRAINING BLOCKED**,
retains the diagnostic STOP, and skips fitting. The predictive measurement
remains incomplete. Missing or corrupt evidence and unexpected execution
failures still raise errors; see the [run investigation](../docs/studies/future-innovation/notebook-run-investigation.md#run-haic-gojuxseb-verified-audit-rejection).

| Order | Notebook | Question |
| --- | --- | --- |
| 00 | [Question and worked example](future_innovation/00_question_and_worked_example.ipynb) | What could skeleton history add? |
| 01 | [Cohort and alignment](future_innovation/01_cohort_and_alignment.ipynb) | Are windows aligned and sources separated? |
| 02 | [Teacher features and validity](future_innovation/02_teacher_features_and_validity.ipynb) | Are inputs past-only and targets meaningful? |
| 03 | [Predictors and controls](future_innovation/03_matched_predictors_and_controls.ipynb) | Does correctly paired motion help? |
| 04 | [Results and next decision](future_innovation/04_results_and_next_decision.ipynb) | What does the evidence permit next? |

Edit [the builder](../scripts/research_directions/future_innovation/build_future_innovation_notebooks.py),
then regenerate selected notebooks. The [maintenance guide](../scripts/research_directions/future_innovation/future_innovation_tutorial_guide.md)
documents source checks, fresh-kernel verification and executed-copy storage.

## Foundations

Run these in order to reproduce the original GAVD S-JEPA workflow.

| Order | Notebook | Purpose |
|---:|---|---|
| 00 | [S-JEPA from first principles](foundations/00_sjepa_from_first_principles.ipynb) | Minimal learning graph and tensor contract |
| 01 | [GAVD manifest and YouTube](foundations/01_gavd_manifest_and_youtube.ipynb) | Source manifest and video cache |
| 02 | [Extract and watch skeletons](foundations/02_extract_and_watch_skeletons.ipynb) | Pose extraction and alignment checks |
| 03 | [Neurologic keypoint masking](foundations/03_neurologic_keypoint_masking.ipynb) | Mask parser, sampling, and target assertions |
| 04 | [Pretrain S-JEPA on normal gait](foundations/04_pretrain_sjepa_on_normal.ipynb) | Five-stage checkpoint lineage |
| 05 | [Inspect latent motion](foundations/05_inspect_latent_motion.ipynb) | Representation-health and geometry audits |
| 06 | [Health-condition classifiers](foundations/06_capstone_health_condition_classifiers.ipynb) | Leakage-aware downstream readouts |

These notebooks are their own authoritative source. Edit them directly.

## AMASS utilities

| Notebook | Purpose |
|---|---|
| [Visualize AMASS SMPL-H poses](amass/01_visualize_amass_smplh_poses.ipynb) | Inspect raw `*_poses.npz` recordings and rendered SMPL-H motion |

The repository's current HPC submission scripts are in
[`slurm/`](../slurm/). They cover AMASS conversion/training and the swap probe;
there is no checked-in generic notebook-07 launcher.

## Research experiments

| Experiment | Order | Notebook | Authoritative builder |
|---|---:|---|---|
| Signed laterality | 01 | [Probe](idea05_signed_laterality/01_probe.ipynb) | [`build_signed_laterality_probe_notebook.py`](../scripts/research_directions/signed_laterality/build_signed_laterality_probe_notebook.py) |
| Signed laterality | 02 | [Futures and reach](idea05_signed_laterality/02_futures_and_reach.ipynb) | [`build_signed_laterality_futures_notebook.py`](../scripts/research_directions/signed_laterality/build_signed_laterality_futures_notebook.py) |
| Reflection equivariance | 01 | [Encoder contract](idea09_reflection_equivariance/01_encoder_contract.ipynb) | [`build_reflection_encoder_contract_notebook.py`](../scripts/research_directions/reflection_equivariance/build_reflection_encoder_contract_notebook.py) |
| Reflection equivariance | 02 | [Futures and reach](idea09_reflection_equivariance/02_futures_and_reach.ipynb) | [`build_reflection_futures_notebook.py`](../scripts/research_directions/reflection_equivariance/build_reflection_futures_notebook.py) |
| Reflection equivariance | 03 | [GAVD contract](idea09_reflection_equivariance/03_gavd_contract.ipynb) | [`build_reflection_gavd_notebooks.py`](../scripts/research_directions/reflection_equivariance/build_reflection_gavd_notebooks.py) |
| Reflection equivariance | 04 | [GAVD training](idea09_reflection_equivariance/04_gavd_training.ipynb) | [`build_reflection_gavd_notebooks.py`](../scripts/research_directions/reflection_equivariance/build_reflection_gavd_notebooks.py) |
| Reflection equivariance | 05 | [GAVD audit](idea09_reflection_equivariance/05_gavd_audit.ipynb) | [`build_reflection_gavd_notebooks.py`](../scripts/research_directions/reflection_equivariance/build_reflection_gavd_notebooks.py) |
| Reflection equivariance | 06 | [CPU replication](idea09_reflection_equivariance/06_cpu_replication.ipynb) | [`build_reflection_replication_notebooks.py`](../scripts/research_directions/reflection_equivariance/build_reflection_replication_notebooks.py) |
| Reflection equivariance | 07 | [GPU replication](idea09_reflection_equivariance/07_gpu_replication.ipynb) | [`build_reflection_replication_notebooks.py`](../scripts/research_directions/reflection_equivariance/build_reflection_replication_notebooks.py) |
| Reflection equivariance | 08 | [AMASS Core11 training](idea09_reflection_equivariance/08_amass_core11_training.ipynb) | [`build_reflection_amass_training_notebook.py`](../scripts/research_directions/reflection_equivariance/build_reflection_amass_training_notebook.py) |

Edit the builder for a generated experiment notebook, then regenerate it. Do
not hand-edit generated cell source because the next build will overwrite it.

## Running notebooks

From the `gavd6` project root:

```bash
uv sync
uv run jupyter lab
```

For a non-interactive run, keep the executed copy outside the source tree:

```bash
mkdir -p work/artifacts/notebook_runs
uv run jupyter nbconvert \
  --to notebook --execute notebooks/path/to/notebook.ipynb \
  --output-dir work/artifacts/notebook_runs
```

`work/artifacts/notebook_runs/` is ignored because executed notebooks are run
artifacts. Scientific outputs should continue to use each notebook's explicit,
versioned artifact contract.
