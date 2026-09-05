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

## Current data and split contract

The five raw manifest folders contain 666 sequences from 103 source videos and 140,641 annotated frames. On the local date 2026-09-04, 657 sequences from 100 videos and 137,690 annotated frames were metadata-public. By condition, that public population is normal 291/32/41,340, Parkinson's 47/11/10,426, stroke 75/18/32,930, myopathic 184/29/33,992, and cerebral palsy 60/10/19,002 (sequences/videos/annotated frames).

Metadata-public is not decoded-frame or pose-QC usability. The decoded-span candidate upper bound is 656 sequences, 99 videos, and 137,232 annotated frames because `n93bgWhLZk4` is public but its 15-second media is too short for annotations through frame 458. The current measured decode audit contains 655 eligible sequences from 98 sources and 135,804 annotated frames: `n93bgWhLZk4` is terminal-short, while `hGNKzkCF4J8` remains a retryable acquisition failure. Notebook 02's verified real-mode audit found all 655 locked pose caches structurally and provenance ready; the predeclared 0.50 neurologic-joint coverage gate retained 639 sequences from 97 sources and 134,259 annotated frames. By condition, that pose-QC cohort is normal 276/30, Parkinson's 46/11, stroke 74/18, myopathic 183/28, and cerebral palsy 60/10 (sequences/sources).

The dated targeted retry ledger remains at `docs/youtube_antibot_retry_2026-09-04.csv`; later full-download attempts supersede its partial-cache counts. Notebook 01 now distinguishes terminal attrition from retryable acquisition errors, records bounded client attempts, resumes from valid cached files, and continues on the measured eligible subset unless `GAVD_STRICT_DOWNLOADS=1` is explicitly set.

Protocol v2 uses five deterministic outer folds. Before availability/QC attrition, every fold assigns 60 sources to training, 20 to validation, and 20 to sealed testing. The input-manifest SHA-256 is `7fd559e5105b11011a3e5c194b7ccc29729c56491c424745834df39884123b5a`; the split SHA-256 is `ff3518b87b1d1fa7d95efb1aea1711773137a21699967cb8015edb8d845ccbe1`. These hashes are deterministic outputs of the dated snapshot and protocol-v2 module. Fold 0 / seed 42 now has current label-free training, latent-audit, readout, temporal, and normal-retention artifacts; folds 1--4 and additional seeds remain pending.

## Notebook map

- `00`–`06` are the executable data, training, representation, and readout path.
- `07`–`09` are BrainBodyFM-directed diagnostics and follow-up experiments.
- `nb_05a`–`nb_05d` are laterality probes shared with the sibling `neurips-laterality` paper package; their scientific figures and paper live there.

Run the numbered pipeline in dependency order. Notebook 04 produces the checkpoints consumed by notebooks 05 onward. Every numbered notebook now retains an evidence-aware SVG summary and exports matching SVG/PNG/PDF paper assets through `scientific_visuals.py`; blocked prerequisites render as status figures rather than fabricated metrics. Earlier model, geometry, classifier, temporal, laterality, and repair metrics are archived and are not comparable with protocol-v2 results. See [the readiness guide](docs/neurips-brain-body.md) before treating saved notebook output as current evidence.

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
