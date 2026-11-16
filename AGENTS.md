# Working in AlexPose

This guide applies throughout the repository. `CLAUDE.md` is a relative symlink to this file.
Read any more specific instructions in the directory you change, and inspect Git status first.

For authorized research delegation, consult [.agents/README.md](.agents/README.md).
It indexes shared AI/ML roles, coordination rules and generated native profiles.
Use only the specialists needed for the task; saving the library does not start a study.

## Purpose and code ownership

AlexPose processes gait videos into poses, movement features, and classifications.
Prefer existing functionality in `ambient/` for reusable ingestion, video processing,
pose estimation, gait features, classification, configuration, and storage.
Experiments should consume these shared interfaces. Keep study-specific models,
protocols, and analyses inside their experiment folders until they have a demonstrated
reusable purpose. Shared code must not acquire dependencies on individual experiments.
Preserve existing public interfaces or update their callers, tests, and documentation together.

## Repository map

| Path | Responsibility |
| --- | --- |
| [ambient/](ambient/) | Reusable Python implementation; the primary starting point for shared behavior. |
| [data/](data/) | Dataset annotations, provenance, local media, models, and runtime data; each has a different Git policy. |
| [server/](server/) | FastAPI app, routers, services, middleware, and backend integration. |
| [frontend/](frontend/) | Next.js/React interface, components, hooks, and frontend tests. |
| [config/](config/README.md) | Application settings, environment overlays, model definitions, and prompts. |
| [tests/](tests/) | Shared Python unit, property, integration, API, and performance tests. |
| [examples/](examples/) | Small examples of shared APIs; verify them against current implementations. |
| [scripts/](scripts/) | Development helpers, diagnostics, startup scripts, and test utilities. |
| [experiments/](experiments/) | Research notebooks, local packages, protocols, and study-specific outputs. |
| [docs/](docs/README.md) | Architecture, usage, analysis, development guidance, and historical reports. |
| [notes/](notes/) | Research instructions and working plans; these do not establish implemented behavior. |

Root `cache/`, `logs/`, `debug_sessions/`, and root `realtime/` are local runtime/sandbox paths.
The similarly named `ambient/realtime/` contains shared code and must remain visible to Git.
Treat source and configuration as evidence of current behavior. Some README links and
older plans are stale; verify destinations and check the relevant implementation.

## Shared code navigation

- [core/](ambient/core/) defines interfaces, frames, data models, and configuration.
  The current CLI/server use `ConfigurationManager` in [config.py](ambient/core/config.py).
  [configuration.py](ambient/core/configuration.py) also contains YAML/environment managers;
  inspect callers before choosing between these APIs.
- [utils/youtube_cache.py](ambient/utils/youtube_cache.py) provides `download_youtube_urls`
  and cache lookup helpers. Prefer these over a new experiment-specific downloader.
- [gavd/](ambient/gavd/) loads and transforms annotations and coordinates GAVD processing.
- [video/](ambient/video/) handles video decoding, extraction, and YouTube processing.
- [pose/](ambient/pose/) contains `PoseEstimatorFactory`, `SequenceKeypointExtractor`,
  estimator backends, keypoint structures, joint conventions, and angle geometry.
- [analysis/](ambient/analysis/) contains `FeatureExtractor`, gait analysis, temporal
  measurements, and symmetry analysis. Feature sets depend on options and input format.
- [classification/](ambient/classification/) contains feature vectors, conventional
  classifiers, ensembles, prompt management, and LLM classification.
- [storage/](ambient/storage/) provides structured storage, SQLite, and backup services.
- [cli/](ambient/cli/) defines the Click command group and its registered commands.
- [realtime/](ambient/realtime/) implements buffering, tracking, pose estimation,
  stream processing, and gait analysis for live input.

### Where to make changes

| Task | Start here | Tests and documentation |
| --- | --- | --- |
| Downloading and video ingestion | [youtube_cache.py](ambient/utils/youtube_cache.py), [video/](ambient/video/) | [video integration tests](tests/integration/test_video_pipeline.py), [YouTube tests](tests/integration/test_youtube_pipeline.py), [installation](docs/guides/installation.md) |
| GAVD annotation inventory | [processing scripts](data/gavd/) and [shared loader](ambient/gavd/gavd_processor.py) | [dataset workflow](data/gavd/INSTRUCTIONS.md), [processing tests](data/gavd/test_processing.py) |
| Pose extraction and joint geometry | [keypoint extractor](ambient/pose/keypoint_extractor.py), [factory](ambient/pose/factory.py) | [keypoint tests](tests/pose/test_keypoints.py), [angle tests](tests/ambient/pose/test_joint_angles.py), [extraction guide](docs/analysis/keypoint-extraction.md) |
| Movement features and classifiers | [feature extractor](ambient/analysis/feature_extractor.py), [classifiers](ambient/classification/) | [feature integration tests](tests/ambient/test_enhanced_feature_integration.py), [classifier tests](tests/ambient/classification/), [feature guide](docs/analysis/feature-extraction.md) |
| Configuration and storage | [config.py](ambient/core/config.py), [storage/](ambient/storage/) | [core tests](tests/ambient/core/), [storage tests](tests/storage/), [configuration guide](docs/guides/configuration.md) |
| Web or live-stream behavior | [routers](server/routers/), [services](server/services/), [frontend](frontend/), [realtime](ambient/realtime/) | [API tests](tests/api/), [realtime tests](tests/ambient/realtime/), [live-stream docs](docs/realtime/README.md) |

## Dataset and research conventions

Use [data/gavd/README.md](data/gavd/README.md) as the GAVD inventory authority.
It describes annotations and provenance, not downloaded videos or extracted poses.
Its [source manifest](data/gavd/source/manifest.json) pins the upstream snapshot.
Do not double-count `source/` CSVs alongside the per-condition sequence CSVs.
GAVD videos and previews belong under `data/videos-gavd`; prefer one source directory
with a `previews/` subdirectory instead of duplicating videos into condition folders.
The media README and selected manifests can be tracked; the bulk media stays local.

Keep source IDs, sequence IDs, frame numbering, decoded times, joint indexing,
confidence, missingness, image dimensions, and coordinate units explicit.
GAVD CSV frame numbers and `SequenceKeypointExtractor` frame arguments are one-based;
decoded OpenCV frame indexes are zero-based. Convert exactly once at the boundary.
Keep the declared pose format explicit; BlazePose-33 and COCO joint indexes differ.
Image coordinates and projected angles are not physical speed or measured 3D anatomy.
Take confidence thresholds and preprocessing from the applicable API or study protocol.
For example, the AAAI study's quality settings are not universal Ambient defaults.
Group related clips and windows by source before evaluating generalization.
Folder labels are dataset annotations, not independently verified clinical diagnoses.

Use [exp5](experiments/exp5/) for historical feature/RF notebooks,
[multiple-sclerosis](experiments/multiple-sclerosis/README.md) for its source-grouped study,
and [aaai](experiments/aaai/README.md) for the ASTAD tutorials and controlled prediction study.
Read each experiment's protocol, environment, and artifact prerequisites first.
Preserve historical results, frozen run identities, provenance, and manuscript snapshots.
A changed scientific method needs a new declared identity; never relabel an old output.

## Setup and commands

Run the following from the repository root unless a different directory is stated.
The root [pyproject.toml](pyproject.toml) requires Python >=3.12; prefer Python 3.12
for compatibility with the research projects. Some experiments require exactly 3.12.x.
Root `uv.lock` and frontend `package-lock.json` are tracked; retain them when applicable.

```bash
uv sync --extra dev
uv run python -m ambient.cli.main --help
uv run python -m ambient.cli.main analyze --help
uv run uvicorn server.main:app --reload --host 127.0.0.1 --port 8000
```

The installed `alexpose` entry point uses the same Click CLI. Inspect command help
before analysis. Despite its URL examples, `analyze` currently requires an existing
local path at argument validation. Use the shared download helpers for URLs.
An analysis may download models or call a paid provider; a help check does neither.
The backend command starts the app; inspect configuration and startup effects first.

From `frontend/`, use the scripts declared in [package.json](frontend/package.json):

```bash
npm ci
npm run dev
npm run lint
npm test -- --runInBand
```

Each experiment's `pyproject.toml` defines its own dependencies and environment.
For example, these commands run from the root against the AAAI project:

```bash
uv sync --project experiments/aaai
uv run --project experiments/aaai pytest -c experiments/aaai/pyproject.toml experiments/aaai/tests -q
```

Initialize the pinned comparator only when working with that baseline:
`git submodule update --init -- experiments/aaai/vendor/STG-NF`.
Use [the experiment README](experiments/aaai/README.md) for data prerequisites and notebook launch commands.

## Configuration and verification

Read [config/README.md](config/README.md). The main settings are `config/alexpose.yaml`;
`development.yaml`, `production.yaml`, and `heroku-production.yaml` provide overlays.
`llm_models.yaml` and `classification_prompts.yaml` define model settings and prompts.
Use the root [.env.example](.env.example) as a template for local `.env` settings.
Never print credentials or copy real values into tracked templates or notebook outputs.

Choose tests for the changed interface and its callers. Root pytest discovery targets
`tests/`; it does not automatically verify every experiment. Check markers before selection.
Start with a focused local test, for example:

```bash
uv run pytest tests/pose/test_keypoints.py -q --no-cov
```

[pytest.ini](pytest.ini) takes precedence over the root pyproject pytest settings and
adds coverage and strict configuration checks. Consult the [audit](docs/development/gitignore-audit.md)
for verified commands and any existing environment or configuration failures.
The [Makefile](Makefile) is a convenience wrapper; `test-fast` selects `not slow`,
which does not guarantee exclusion of integration or hardware tests. `make docs` is a placeholder.
The active [CI workflow](.github/workflows/comprehensive-testing.yml) runs only a placeholder
`test-summary` job. Its green status does not establish that the substantive test suite ran.

## Documentation and Git hygiene

Update affected API documentation, examples, and relevant tests when behavior changes.
Start with [core architecture](docs/architecture/core-components.md),
[contributing](docs/development/contributing.md), or the relevant analysis/study guide.
For generated notebooks, slides, figures, and reports, edit the source generator first,
then regenerate and check the exported content against its inputs.
Record what ran and what remains unverified. Keep speculative plans distinguishable from results.

Before adding files, inspect `git status --short` and the scoped diff.
Use `git check-ignore --no-index -v <path>` to diagnose ignore behavior, including tracked paths.
A negated matching rule means the path is allowed. Ignore changes do not untrack existing files.
The [ignore audit](docs/development/gitignore-audit.md) explains deliberate local-data exclusions,
metadata exceptions, lockfile differences, and existing tracking conflicts.
Preserve unrelated work and pinned submodules. Do not stage datasets, credentials,
environments, or large run products as an incidental part of a code change.
