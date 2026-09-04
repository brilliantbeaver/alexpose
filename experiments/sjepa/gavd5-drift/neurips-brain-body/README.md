# NeurIPS BrainBodyFM workshop package

This folder is the home of the notebooks, paper drafts, and submission notes for the NeurIPS 2026 Workshop on Foundation Models for the Brain and Body. It is intentionally a workshop package inside the larger `gavd5-drift` experiment, not a second copy of the experiment state.

## Directory contract

| Location | Contents |
|---|---|
| `neurips-brain-body/` | Workshop notebooks and planning records |
| `neurips-brain-body/docs/` | Current workshop paper, source, PDF, and readiness guide |
| `../data-gavd/` | Shared GAVD manifests |
| `../work/` | Shared checkpoints, derived poses, results, and experiment scripts |
| `../cache/`, `../images/`, `../.env` | Shared cache, tutorial figures, and configuration |
| `../docs/figures/`, `../docs/references.bib`, `../docs/neurips_2026.sty` | Shared paper figures, bibliography, and workshop style |
| `../neurips-laterality/` | The separate laterality paper package and its figures |

Notebook setup cells resolve the experiment parent from either this folder, the experiment folder, or `ALEXPOSE_ROOT`. Generated data and results continue to go to `../work`; moving the notebooks does not fork their artifact lineage.

## Notebook map

- `00`–`06` are the executable data, training, representation, and readout path.
- `07`–`09` are BrainBodyFM-directed diagnostics and follow-up experiments.
- `nb_05a`–`nb_05d` are laterality probes shared with the sibling `neurips-laterality` paper package; their scientific figures and paper live there.

Run the numbered pipeline in dependency order. Notebook 04 produces the checkpoints consumed by notebooks 05 onward. The paper currently excludes stale or mixed-lineage results from notebooks 08 and 09; see [the readiness guide](docs/neurips-brain-body.md) before treating saved notebook output as current evidence.

## Run locally

From `experiments/sjepa/gavd5-drift`:

```sh
uv sync
uv run jupyter lab neurips-brain-body
```

The Colab badges in notebooks 00–06 point to their paths in this folder. Configuration remains in the experiment-level `.env` file.

## Build the paper

From `neurips-brain-body/docs`:

```sh
tectonic bbfm2026_paper_draft.tex
```

The LaTeX source resolves the shared style, bibliography, and generated figures from the parent experiment's `docs/` directory. The Markdown source uses the same shared figure files.

## Planning records

`FINAL_ACTION_PLAN.md`, `PAPER_REVISION_PLAN.md`, `REGENERATION_CHECKLIST.md`, `SUBMISSION_READINESS_EXECUTIVE_SUMMARY.md`, and `WORK_STATUS_AND_DECISION_POINT.md` are pre-revision planning snapshots from 2026-09-03. They are retained for audit history; the current paper and readiness guide supersede their status statements.
