# Repository layout and naming

The repository is organized by **scientific study within conventional file roles**. Begin at the [study index](../studies/README.md); the current study is [motion preservation](../studies/motion-preservation/README.md). The machine-readable [ownership registry](studies.json) connects each study to its documentation, implementation, notebooks, tests and cluster execution.

## Study structure

```text
docs/studies/
  motion-preservation/          current tracking-repair experiment
  future-feature-prediction/    gate, scaling, accessibility, manuscript
  latent-laterality/            unknown left/right correspondence
  reflection-equivariance/     known anatomical reflection and probes
  gait-classification/          historical staged representation learning
  force-prediction/             StrokePIG feasibility study
  perturbation-response/        deferred cross-protocol study

notebooks/<study>/              ordered analysis or execution stages
src/gavd6_sjepa/                importable study methods and shared data code
scripts/research_directions/    notebook/figure builders and execution wrappers
slurm/                         study-specific cluster jobs
tests/<study-or-component>/    grouped tests and shared fixtures

notes/research-agenda/          current alternatives and shared references
notes/archive/research-planning/  earlier portfolios and broad ideation
docs/history/                  preserved publications and methodology history
docs/repository/               organization, naming and migration records
```

Each study README presents roughly four chunks: protocol, execution, results, and references/development. Future-feature prediction instead separates its gate, scaling and accessibility experiments from the manuscript. These measurements have different questions and cohorts; placing them under one parent does not pool their effects or establish successful distillation.

Consolidation means giving each record an owner and a clear reading order. Independently frozen protocols, individual notebook stages and dated result records remain separate documents. Source code belongs in an importable package, rather than being copied into documentation folders.

## Naming rules

- Use an established method, measurement or scientific question for the study name. Avoid an idea number, conference venue, unexplained acronym or an implied result as the primary label.
- Use lowercase `kebab-case` for study/document directories and Markdown filenames; use `snake_case` for Python modules and notebook stems. Python import paths must remain valid identifiers.
- Let the parent directory provide context: `scaling/research-strategy.md` is clearer than repeating the study name in every filename. Prefer short descriptive names such as `proposal.md`, `data-harmonization.md`, `results.md` and `paper.md`.
- Notebook prefixes describe execution order. Existing 19–23 references are retained; other stems are shortened without changing their scientific sequence.
- Conventional names such as `README.md`, `__init__.py` and study-local `build_notebooks.py` may repeat in distinct namespaces. A global basename-uniqueness rule would encourage unnecessarily long names.
- Dates belong on immutable snapshots, run identities and dated assessments. Do not manufacture `final`, `final-v2`, or new method acronyms to distinguish ordinary working files.
- Describe the historical lower-body subset as an **11-landmark lower-body representation**, including heel/forefoot surface landmarks. `core11-v1` remains a legacy schema identifier, not a claimed community standard. Existing serialized keys, user-facing command names and checkpoint names retain their original spelling. Current Python module paths use the descriptive code names.

## Representative changes

| Earlier path or name | Canonical location |
| --- | --- |
| `notes/world-model-extensions/proposals-04/01-preserve-real-movement.md` | [Motion-preservation proposal](../studies/motion-preservation/protocol/proposal.md) |
| `notes/world-model-extensions/proposals-04/README.md` | [Research agenda](../../notes/research-agenda/README.md) |
| `notes/world-model-extensions/proposals-04/00-evidence-and-execution.md` | [Shared experiment guidelines](../../notes/research-agenda/references/experiment-guidelines.md) |
| `notes/world-model-extensions/proposals-04/02-iterate-ideas.md` | [JEPA research-directions prompt](../../notes/archive/research-planning/prompts/jepa-research-directions.md) |
| `docs/studies/iclr/07_scaling_result_and_research_strategy.md` | [Scaling research strategy](../studies/future-feature-prediction/scaling/research-strategy.md) |
| `docs/studies/iclr/04_paper.md` | [Historical manuscript](../studies/future-feature-prediction/manuscript/paper.md) |
| `notebooks/idea09_reflection_equivariance/08_amass_core11_training.ipynb` | [AMASS training notebook](../../notebooks/reflection_equivariance/08_amass_training.ipynb) |
| `notebooks/foundations/06_capstone_health_condition_classifiers.ipynb` | [Gait classifiers](../../notebooks/gait_classification/06_gait_classifiers.ipynb) |
| `notebooks/iclr_bridge/21_student_accessible_future_features.ipynb` | [Target accessibility](../../notebooks/target_accessibility/21_target_accessibility.ipynb) |
| `notes/world-model-extensions/proposals-01/` through `proposals-03/` | [Earlier research planning](../../notes/archive/research-planning/README.md) |

The [documentation migration record](migration-2026-09-13.json) and its [review](organization-review.md) describe the earlier study organization. The subsequent [code organization](code-organization.md) covers implementation names, merged handlers, grouped tests and exact historical replay.

## Compatibility and evidence

Canonical navigation uses the new study names. Older notebook paths remain relative symlinks where launchers or historical workflows use them. Legacy documentation entry points redirect to the new overviews. These paths identify the same source, not additional experiments or independent evidence.

The redundant `notes/world-model-extensions/proposals-04/` redirects were subsequently removed after updating incoming document links. Its remaining follow-up prompt now lives with the archived planning prompts; the table above records all four destinations. Historical migration receipts retain the paths recorded at the time.

Original numerical protocols and result identities retain their paths and exact bytes. Current Python modules, scripts and Slurm jobs use the concise names described in [code organization](code-organization.md); historical software paths are preserved inside an [exact replay snapshot](../../scripts/archive/code_layout_20260913/README.md). Current code has a new software identity. Old receipts are not rewritten to conceal that change.

Source notebooks and builders were updated together; saved execution output is retained. The refactor changes neither motion schemas nor cohort membership, model equations, calibration rules or reserved evaluation sets. The result registry keeps its established run IDs and paths. Registered historical bundles remain absent locally.

## Shared and local files

AMASS/GAVD manifests, raw data, body-model assets and checkpoints remain shared resources with their own provenance. Being ignored by Git does not make them disposable. Local result collections remain at their established paths and are linked from the owning study; no run bundle is silently renamed to match a new documentation label.

The pre-migration source snapshot is retained locally at `work/repository-organization/20260913/source-before.tar.gz`, with file hashes and move records beside it. This includes untracked source work present before organization. No scientific artifacts were deleted, and no final evaluation set was reopened.
