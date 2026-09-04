# Notebook index

Notebooks are grouped by purpose. Numeric prefixes define the order only within
their directory; they are not a single global sequence.

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
| Signed laterality | 01 | [Probe](experiments/idea05_signed_laterality/01_probe.ipynb) | [`build_signed_laterality_probe_notebook.py`](../scripts/research_directions/signed_laterality/build_signed_laterality_probe_notebook.py) |
| Signed laterality | 02 | [Futures and reach](experiments/idea05_signed_laterality/02_futures_and_reach.ipynb) | [`build_signed_laterality_futures_notebook.py`](../scripts/research_directions/signed_laterality/build_signed_laterality_futures_notebook.py) |
| Reflection equivariance | 01 | [Encoder contract](experiments/idea09_reflection_equivariance/01_encoder_contract.ipynb) | [`build_reflection_encoder_contract_notebook.py`](../scripts/research_directions/reflection_equivariance/build_reflection_encoder_contract_notebook.py) |
| Reflection equivariance | 02 | [Futures and reach](experiments/idea09_reflection_equivariance/02_futures_and_reach.ipynb) | [`build_reflection_futures_notebook.py`](../scripts/research_directions/reflection_equivariance/build_reflection_futures_notebook.py) |
| Reflection equivariance | 03 | [GAVD contract](experiments/idea09_reflection_equivariance/03_gavd_contract.ipynb) | [`build_reflection_gavd_notebooks.py`](../scripts/research_directions/reflection_equivariance/build_reflection_gavd_notebooks.py) |
| Reflection equivariance | 04 | [GAVD training](experiments/idea09_reflection_equivariance/04_gavd_training.ipynb) | [`build_reflection_gavd_notebooks.py`](../scripts/research_directions/reflection_equivariance/build_reflection_gavd_notebooks.py) |
| Reflection equivariance | 05 | [GAVD audit](experiments/idea09_reflection_equivariance/05_gavd_audit.ipynb) | [`build_reflection_gavd_notebooks.py`](../scripts/research_directions/reflection_equivariance/build_reflection_gavd_notebooks.py) |
| Reflection equivariance | 06 | [CPU replication](experiments/idea09_reflection_equivariance/06_cpu_replication.ipynb) | [`build_reflection_replication_notebooks.py`](../scripts/research_directions/reflection_equivariance/build_reflection_replication_notebooks.py) |
| Reflection equivariance | 07 | [GPU replication](experiments/idea09_reflection_equivariance/07_gpu_replication.ipynb) | [`build_reflection_replication_notebooks.py`](../scripts/research_directions/reflection_equivariance/build_reflection_replication_notebooks.py) |
| Reflection equivariance | 08 | [AMASS Core11 training](experiments/idea09_reflection_equivariance/08_amass_core11_training.ipynb) | [`build_reflection_amass_training_notebook.py`](../scripts/research_directions/reflection_equivariance/build_reflection_amass_training_notebook.py) |

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
