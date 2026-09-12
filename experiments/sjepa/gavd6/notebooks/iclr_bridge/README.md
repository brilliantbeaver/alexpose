# Laterality to future distillation: notebooks 19–22

These tutorials connect the laterality findings, repaired Experiment 0 and a
separately fitted student-accessibility comparison. Their original numbers
continue the external laterality series. Read them in order:

| Notebook | What it establishes |
|---|---|
| [19 — Evidence and observability](19_evidence_and_observability.ipynb) | What the saved results support, and why the two studies' R² values answer different questions. |
| [20 — Symmetry and temporal information](20_symmetry_and_temporal_information.ipynb) | How reflection and temporal order differ, with explicitly synthetic examples. |
| [21 — Student-accessible future features](21_student_accessible_future_features.ipynb) | What the cached comparison measured and how to verify its selected models and predictions. |
| [22 — Selective future distillation](22_selective_future_distillation.ipynb) | How to test target-selection ideas before a separate student-training experiment. |

The [research tutorial](../../docs/studies/iclr/01_critique_and_research_tutorial.md)
provides the scientific context. For the separate dataset-size experiment,
continue to [23 — Source learning curves](../future_innovation/23_source_learning_curves.ipynb)
and its [execution guide](../../slurm/future-innovation/SOURCE_LEARNING_CURVE.md).

## Open and run

Start Jupyter from the repository root:

```bash
uv run jupyter lab notebooks/iclr_bridge/
```

Each notebook finds the repository by walking upward from its working directory,
so it also works when its kernel starts in this folder. Run cells from the top
in a fresh kernel. The saved evidence must be available at
`work/artifacts/iclr-bridge-2026-09-11/` and
`outputs/iclr-bridge-cached-20260911/`; notebook 19 also inventories the retained
`outputs/future-innovation-direct-v3-dev-20260911/` reports. Missing required
evidence raises an error.

Notebook 21 defaults to CPU reconstruction of selected models. It writes no
scientific artifacts. In an interactive session, setting `RECONSTRUCT_MODELS =
False` in section 2 selects file-integrity inspection without refitting; that
mode does not verify numerical predictions. The other notebooks read saved
evidence and run labeled synthetic calculations. None launches a new real-data
comparison or student training.

## Maintain and retain executed copies

Edit the [canonical generator](../../scripts/research_directions/iclr_bridge/build_notebooks.py),
then regenerate from the repository root:

```bash
uv run python scripts/research_directions/iclr_bridge/build_notebooks.py
```

Keep the source notebooks output-free. For a batch execution, choose a new
directory directly under `work/` so previous executed copies remain intact:

```bash
export ICLR_NOTEBOOK_OUTPUT="work/iclr-notebooks-$(date +%Y%m%dT%H%M%S)"
mkdir "$ICLR_NOTEBOOK_OUTPUT"
uv run jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=1800 \
  --output-dir "$ICLR_NOTEBOOK_OUTPUT" \
  notebooks/iclr_bridge/19_evidence_and_observability.ipynb \
  notebooks/iclr_bridge/20_symmetry_and_temporal_information.ipynb \
  notebooks/iclr_bridge/21_student_accessible_future_features.ipynb \
  notebooks/iclr_bridge/22_selective_future_distillation.ipynb
uv run python scripts/research_directions/iclr_bridge/check_deliverables.py \
  --executed-dir "$ICLR_NOTEBOOK_OUTPUT" \
  --output "$ICLR_NOTEBOOK_OUTPUT/deliverable-audit.json"
```

This output directory has the same depth as the source folder, preserving the
notebooks' relative document links. The audit requires all four notebooks,
matching source cells and completed execution, and checks the retained external
source snapshot and document links. It requires the historical evidence files;
it is not a portable substitute for numerical experiment verification.

Historical executed notebooks remain in
`work/artifacts/iclr-bridge-2026-09-11/executed_notebooks/`. They describe the
source version used at that time, including its former location at the project
root. Do not overwrite them to make them match a regenerated tutorial.
