# Repository ignore-rule audit

## Intended policy, recorded before editing

The root policy should keep shared source, tests, annotations, documentation, safe configuration templates, and intentional publication deliverables visible. Local credentials, environments, dependencies, runtime databases, model checkpoints, downloaded media and declared experiment caches should remain excluded.

The audit will anchor the root realtime sandbox, allow selected documentation inside media folders, cover environment-file variants, and preserve reviewed test fixtures. Nested experiment policies for retained pose caches, transient run directories, reveal.js distribution assets and lockfiles will remain in force. No tracked file will be removed from the index, and dependency locks will not be regenerated.

The initial working tree contains the unrelated untracked file `notes/14. create AGENTS.md`. It is outside this task. The STG-NF submodule is initially at `edb5f3220332e160e4d20ce258787d5e2d7e0200`.


## Scope and decisions

This audit inspected the root ignore file and all 14 nested project ignore files outside dependencies and the STG-NF submodule. Only the root and `frontend/.gitignore` needed edits. The 177-line [agent guide](../../AGENTS.md) records repository navigation and development practices. Root `CLAUDE.md` is a relative symlink to that guide.

| Policy | Revision and reason |
| --- | --- |
| Shared realtime code | Changed `realtime/` to `/realtime/`. The old unanchored rule also hid new shared modules, tests, frontend components, and documentation. The root sandbox remains ignored. |
| Local environment files | Ignore `.env` and `.env.*` at any depth. Explicitly allow `.example`, `.sample`, and `.template` variants, including names such as `.env.production.template`. Frontend rules have the same exceptions because a nearer ignore file can override root rules. Safe names are not proof of safe contents. |
| Media-directory metadata | Ignore contents under `data/videos*`, `data/youtube*`, `data/models`, and existing `data-videos*`/`video-data*` locations. Reopen directories for traversal, then allow selected README and manifest filenames. Standard data media/model locations also allow case-insensitive `LICENSE*` and `checksums.sha256`. Arbitrary JSON, pose arrays, and prediction CSVs remain hidden there. README extension matching explicitly covers `.md` and `.MD`. |
| Databases | Replace global `*.db` filtering with rules for root databases and data/output/experiment paths. Include SQLite sidecars. Small maintained databases under `tests/fixtures` stay visible. |
| Model checkpoints | Replace global `*.pt` filtering with experiment/output checkpoint locations and the data model store. Do not globally ignore `.pkl` or `.npz`, which can be deliberate research evidence. A small `.pt` fixture under `tests/fixtures` stays visible. |
| Existing media extension policy | Retain the existing global MP4, MP3, and AVI exclusions because older download/example flows can write outside central stores. Add explicit `tests/fixtures` exceptions. Other intentional assets in those formats need a reviewed path-specific exception; PDF/SVG assets remain visible. No new global MOV/WebM/MKV exclusion was added. |
| Caches and dependencies | Consolidate repeated Python cache/environment rules. Ignore root runtime cache and Node dependencies wherever installed. Preserve the two existing tracked architecture records under `cache/artifacts/real/idea9/20260818/` with exact file and ancestor exceptions. Other cache records remain ignored. |
| Schemas and packaging | Remove generic `*.manifest` and `*.spec` exclusions, which could hide maintained schemas or build source. Retain ordinary Python packaging outputs and narrow root runtime stores. Remove unused legacy template boilerplate rather than introducing new technology-wide exclusions. |
| Intentional deliverables | Keep notebook sources, figure generators, SVG/PDF slides, dataset annotations, provenance, and reviewed scientific outputs visible unless a specific existing project policy excludes their output directory. |

Git cannot reinclude a file beneath an excluded parent directory. The media and retained-cache rules therefore reopen the required ancestor directories before allowing selected files. An ignore change does not stop tracking an indexed file. See the [official Git ignore rules](https://git-scm.com/docs/gitignore).

### Nested policies deliberately preserved

- `experiments/aaai/.gitignore` excludes local `artifacts/`, render scratch, and `uv.lock`. Its executed evidence may exist locally without being in Git. This audit does not change the study's publication/storage contract.
- `experiments/multiple-sclerosis/.gitignore` deliberately retains pose keypoints and small run provenance while excluding checkpoints and content-addressed acceleration caches. It is not replaced with AAAI's policy.
- Penny and S-JEPA studies keep their own cache, environment, work-directory, PDF-scratch, and laterality-output exclusions. Existing reveal.js distribution exceptions in `experiments/penny/gavd2` and `experiments/sjepa/gavd2` remain effective despite the root `dist/` rule.
- Root `/docs/penny/` and declared S-JEPA scratch paths remain ignored as existing local mirrors/work areas. This audit does not reinterpret them as maintained documentation.

### Lockfiles

The root lockfile and ten experiment `uv.lock` files are tracked and remain visible. Frontend `package-lock.json` is also tracked. AAAI explicitly ignores its untracked local `uv.lock`; this preserves the earlier study-specific pyproject-only delivery policy. It differs from the usual uv project recommendation to commit the lockfile, so AAAI dependencies are less precisely reproducible from Git alone. A future policy change should be deliberate and separate from this audit. No lockfile or dependency resolution was changed. See [uv's project lockfile guidance](https://docs.astral.sh/uv/concepts/projects/layout/#the-lockfile).

## Verification matrix

All 69 path checks passed after the final changes, both with the local Git configuration and with `core.ignorecase=false` to check portability to case-sensitive systems. Some paths below are hypothetical probes, not files created or discovered on disk. They check how Git would treat a new file in that location. Existing paths were also tested with `--no-index`, so their tracked status did not mask the rule behavior.

“Visible” means eligible for normal Git discovery, not staged or approved for publication. “No matching rule” is visible. A matching rule beginning with `!` is also visible. The responsible rule shown is the final verbose match from `git check-ignore --no-index -v`; its line number belongs to this revision. See [the official check-ignore documentation](https://git-scm.com/docs/git-check-ignore). Do not infer ignored status from a verbose line or exit code alone when that line is a negation.

| Path | Expected | Observed | Responsible rule |
| --- | --- | --- | --- |
| `ambient/realtime/new_module.py` | Visible | Visible | `No matching rule` |
| `docs/realtime/new_guide.md` | Visible | Visible | `No matching rule` |
| `frontend/components/realtime/NewPanel.tsx` | Visible | Visible | `No matching rule` |
| `tests/ambient/realtime/test_new.py` | Visible | Visible | `No matching rule` |
| `realtime/local-output.txt` | Ignored | Ignored | `.gitignore:81:/realtime/` |
| `.env` | Ignored | Ignored | `.gitignore:3:.env` |
| `.env.local` | Ignored | Ignored | `.gitignore:4:.env.*` |
| `.env.production` | Ignored | Ignored | `.gitignore:4:.env.*` |
| `experiments/aaai/.env.local` | Ignored | Ignored | `.gitignore:4:.env.*` |
| `frontend/.env.production` | Ignored | Ignored | `frontend/.gitignore:35:.env.*` |
| `.env.example` | Visible | Visible | `.gitignore:5:!.env.example` |
| `.env.production.template` | Visible | Visible | `.gitignore:10:!.env.*.template` |
| `frontend/.env.example` | Visible | Visible | `frontend/.gitignore:36:!.env.example` |
| `frontend/.env.production.template` | Visible | Visible | `frontend/.gitignore:41:!.env.*.template` |
| `experiments/aaai/.env.example` | Visible | Visible | `.gitignore:5:!.env.example` |
| `data/gavd/normal/example.csv` | Visible | Visible | `No matching rule` |
| `data/gavd/source/manifest.json` | Visible | Visible | `No matching rule` |
| `data/gavd/source/LICENSE` | Visible | Visible | `No matching rule` |
| `data/gavd/README.md` | Visible | Visible | `No matching rule` |
| `data/videos-gavd/README.md` | Visible | Visible | `.gitignore:134:!/data/videos*/**/README.[mM][dD]` |
| `data/videos-gavd/manifest.json` | Visible | Visible | `.gitignore:136:!/data/videos*/**/manifest.json` |
| `data/videos-gavd/checksums.sha256` | Visible | Visible | `.gitignore:137:!/data/videos*/**/checksums.sha256` |
| `data/videos-gavd/source_id/README.md` | Visible | Visible | `.gitignore:134:!/data/videos*/**/README.[mM][dD]` |
| `data/videos-gavd/source_id/video.mp4` | Ignored | Ignored | `.gitignore:153:*.mp4` |
| `data/videos-gavd/source_id/previews/clip.webm` | Ignored | Ignored | `.gitignore:124:/data/videos*/**` |
| `data/videos-gavd/source_id/poses.npz` | Ignored | Ignored | `.gitignore:124:/data/videos*/**` |
| `data/videos-gavd/source_id/predictions.csv` | Ignored | Ignored | `.gitignore:124:/data/videos*/**` |
| `data/videos-gavd/source_id/download.log` | Ignored | Ignored | `.gitignore:124:/data/videos*/**` |
| `data/models/model.pth` | Ignored | Ignored | `.gitignore:128:/data/models/**` |
| `data/models/README.md` | Visible | Visible | `.gitignore:142:!/data/models/**/README.[mM][dD]` |
| `data/models/manifest.json` | Visible | Visible | `.gitignore:144:!/data/models/**/manifest.json` |
| `experiments/aaai/artifacts/checkpoint.pt` | Ignored | Ignored | `experiments/aaai/.gitignore:4:artifacts/` |
| `experiments/exp5/checkpoint.pt` | Ignored | Ignored | `.gitignore:113:/experiments/**/*.pt` |
| `data/runtime.sqlite3` | Ignored | Ignored | `.gitignore:102:/data/**/*.sqlite3` |
| `tests/fixtures/small.db` | Visible | Visible | `No matching rule` |
| `tests/fixtures/small.pt` | Visible | Visible | `No matching rule` |
| `tests/fixtures/small.mp4` | Visible | Visible | `.gitignore:156:!/tests/fixtures/**/*.mp4` |
| `ambient/storage/schema.spec` | Visible | Visible | `No matching rule` |
| `uv.lock` | Visible | Visible | `No matching rule` |
| `experiments/multiple-sclerosis/uv.lock` | Visible | Visible | `No matching rule` |
| `experiments/aaai/uv.lock` | Ignored | Ignored | `experiments/aaai/.gitignore:3:uv.lock` |
| `frontend/package-lock.json` | Visible | Visible | `No matching rule` |
| `experiments/aaai/12_observation_population_and_video_context.ipynb` | Visible | Visible | `No matching rule` |
| `experiments/aaai/slides/images/slide-01.svg` | Visible | Visible | `No matching rule` |
| `experiments/aaai/slides/tutorial.pdf` | Visible | Visible | `No matching rule` |
| `experiments/aaai/scripts/gait_slide_viewer.py` | Visible | Visible | `No matching rule` |
| `experiments/multiple-sclerosis/artifacts/eval/results.json` | Visible | Visible | `No matching rule` |
| `experiments/multiple-sclerosis/artifacts/keypoints/example.npz` | Visible | Visible | `No matching rule` |
| `ambient/__pycache__/module.pyc` | Ignored | Ignored | `.gitignore:20:__pycache__/` |
| `.venv/lib/example.py` | Ignored | Ignored | `.gitignore:13:.venv/` |
| `frontend/node_modules/package/index.js` | Ignored | Ignored | `frontend/.gitignore:4:/node_modules` |
| `outputs/demo/node_modules/package/index.js` | Ignored | Ignored | `.gitignore:58:node_modules/` |
| `frontend/.next/server/page.js` | Ignored | Ignored | `frontend/.gitignore:17:/.next/` |
| `build/wheel.whl` | Ignored | Ignored | `.gitignore:26:build/` |
| `cache/artifacts/runtime.npz` | Ignored | Ignored | `.gitignore:74:/cache/**` |
| `experiments/sjepa/gavd2/slides/reveal/dist/reveal.js` | Visible | Visible | `experiments/sjepa/gavd2/.gitignore:22:!slides/reveal/dist/**` |
| `experiments/penny/gavd2/slides/reveal/dist/reveal.css` | Visible | Visible | `experiments/penny/gavd2/.gitignore:22:!slides/reveal/dist/**` |
| `data/videos-other/weightgait/license.txt` | Visible | Visible | `.gitignore:135:!/data/videos*/**/[Ll][Ii][Cc][Ee][Nn][Ss][Ee]*` |
| `data/videos-other/weightgait/license_text` | Visible | Visible | `.gitignore:135:!/data/videos*/**/[Ll][Ii][Cc][Ee][Nn][Ss][Ee]*` |
| `CLAUDE.md` | Visible | Visible | `No matching rule` |
| `AGENTS.md` | Visible | Visible | `No matching rule` |
| `data/runtime.sqlite3-wal` | Ignored | Ignored | `.gitignore:103:/data/**/*.sqlite*-*` |
| `data/videos-other/turning-in-place/README.MD` | Visible | Visible | `.gitignore:134:!/data/videos*/**/README.[mM][dD]` |
| `data/videos-gavd/source_id/LICENSE.txt` | Visible | Visible | `.gitignore:135:!/data/videos*/**/[Ll][Ii][Cc][Ee][Nn][Ss][Ee]*` |
| `data/videos-gavd/source_id/.env.production` | Ignored | Ignored | `.gitignore:124:/data/videos*/**` |
| `cache/artifacts/real/idea9/20260818/idea9_architecture_contract.json` | Visible | Visible | `.gitignore:79:!/cache/artifacts/real/idea9/20260818/idea9_architecture_contract.json` |
| `cache/artifacts/real/idea9/20260818/idea9_architecture_contract.png` | Visible | Visible | `.gitignore:80:!/cache/artifacts/real/idea9/20260818/idea9_architecture_contract.png` |
| `cache/artifacts/real/idea9/20260818/extra.json` | Ignored | Ignored | `.gitignore:74:/cache/**` |
| `data/videos-gavd/source_id/other.json` | Ignored | Ignored | `.gitignore:124:/data/videos*/**` |

To repeat one check, run from the repository root:

```bash
git check-ignore --no-index -v ambient/realtime/new_module.py
git check-ignore --no-index -v data/videos-gavd/README.md
git check-ignore --no-index -v data/videos-gavd/source_id/video.mp4
git ls-files -ci --exclude-standard
git status --short --untracked-files=all
git diff --check
```

The first command should have no matching output and exit 1. The second should show a negated README exception, and the third an excluding media rule. Nonexistent probe paths do not need to be created. All matrix rows were checked programmatically by comparing the final pattern with the expected visibility; no probe media or model files were written.

## Newly visible local files

Comparing untracked-file inventories before and after narrowing exclusions exposed six existing dataset documentation/license files:

- `data/videos-gavd/README.md`.
- `data/videos-ms/koa-pd-nm/README.md`.
- `data/videos-other/turning-in-place/README.MD`.
- `data/videos-other/weightgait/README.md`.
- `data/videos-other/weightgait/license.txt`.
- `data/videos-other/weightgait/license_text`.

Their combined size is 30,626 bytes. Their contents were reviewed as storage guidance, dataset descriptions with source/citation information, or dataset deposit/license text. No credentials or bulk data were found. These files were left unchanged and untracked. The rest of the newly visible files are this task's guide, symlink, and audit. The unrelated notes file remains untouched. No downloaded media, pose caches, bulk prediction files, or real environment files became visible. This check used Git's filtered file list and the six small documentation/license files, without recursively opening media or model contents.

## Existing tracking conflicts and unresolved policy issues

`git ls-files -ci --exclude-standard` initially found 965 indexed paths matching ignore rules. It now finds 922. The reduction is not a deletion: 48 realtime source/documentation/test/note paths no longer match, and five already indexed local assets now match newly scoped exclusions.

| Current group | Indexed paths matching an ignore rule | Treatment |
| --- | ---: | --- |
| `experiments/sjepa/gavd5-drift` | 917 | Existing nested cache/output exclusions conflict with tracked historical evidence. Preserve the index and evidence; a separate archival decision is needed before any untracking. |
| `data/models/pose_landmarker_full.task`, `pose_landmarker_lite.task`, `rf_example.pkl` | 3 | Existing tracked model assets now match the local model-store rule. They remain tracked and unchanged. Whether to keep these specific distributions/examples or migrate them to a versioned download manifest needs a separate decision. |
| Two links in `outputs/019fbc29-5178-76b3-96a3-5d12cb12878e/presentations/final-visual-review/node_modules/` | 2 | Existing indexed `@oai/artifact-tool` and `lucide` dependency links now match `node_modules/`. Keep them unchanged; investigate their original packaging intent separately. |

The two tracked `idea9_architecture_contract` JSON/PNG records remain visible through exact exceptions. No `git rm --cached`, staging, data migration, submodule update, commit, or push was performed. The STG-NF revision remains unchanged.

Ignore rules are a discovery policy, not a credential scanner or a publication approval. Exact-name README/manifest/license exceptions can expose sensitive text if someone puts it there. Review their contents before adding them. Conversely, a new intentional media fixture outside `tests/fixtures` or a publication asset inside an explicitly ignored run directory needs a narrowly scoped exception or an appropriate maintained output location.

## Repository commands and lightweight checks

The configuration authority is root `pyproject.toml` and `uv.lock`, which require Python >=3.12, rather than the root README's older 3.11+ statement. The existing root interpreter is 3.14.6 and the AAAI interpreter is 3.12.13. AAAI requires >=3.12,<3.13. Commands below used `--no-sync` to inspect the existing environments without updating dependency resolutions. This does not prove a fresh installation on another machine.

| Check | Working directory | Observed result |
| --- | --- | --- |
| `uv sync --help` | Root | Passed. Setup syntax and the declared `dev` extra were checked; dependency installation was not run. |
| `uv run --no-sync python -m ambient.cli.main --help` | Root | Passed. The module is the current Click CLI. |
| `uv run --no-sync python -m ambient.cli.main analyze --help` | Root | Passed. The command's implementation requires an existing local video path despite URL examples. |
| `uv run --no-sync alexpose --help` | Root | Passed. Confirms the installed console entry point. |
| `uv run --no-sync uvicorn --help` | Root | Passed. `server/main.py` defines `app`; no backend or startup hooks were launched. |
| `npm run dev -- --help` | `frontend/` | Passed. Confirms the Next.js dev command without starting a server. |
| `npm run lint -- --help` | `frontend/` | Passed. Confirms ESLint wiring, not a clean lint run. |
| `npm test -- --listTests --runInBand` | `frontend/` | Passed. Discovered four Jest test files; did not execute their assertions. |
| `uv run --no-sync pytest tests/pose/test_keypoints.py -q --no-cov` | Root | 26 passed and 6 failed in untouched tests/code. See failure details below. |
| `uv run --no-sync pytest tests/ambient/pose/test_joint_angles.py -q --no-cov` | Root | 27 passed and 1 failed in untouched tests/code. See failure details below. |
| `uv run --project experiments/aaai --no-sync pytest -c experiments/aaai/pyproject.toml experiments/aaai/tests/test_iteration02_masking.py -q` | Root | 7 passed. Uses the experiment's own pytest configuration and environment. |
| Guide link/path and symlink check | Root | All 71 repository-relative links resolve; `CLAUDE.md` links to `AGENTS.md`. |
| Ignore-policy matrix | Root | All 69 expectations pass in both case modes, including negation chains and retained artifacts. |
| `git diff --check` | Root | Passed. |

The six keypoint failures concern two outdated MediaPipe factory assumptions, an absent `_process_extractor` attribute, two `Keypoint` constructions missing required `id`/`name`, and an empty 33-joint set rejected by validation. The joint-angle failure expects timestamp `0.167`, while serialization returns `0.17`. These failures are recorded rather than repaired in this documentation task. No assertion results from the root test suite are represented as fully green.

Root `pytest.ini` supplies discovery under `tests/`, coverage defaults and strict options. It takes precedence over root pyproject pytest settings; experiment test files are not automatically covered by root discovery. The active `.github/workflows/comprehensive-testing.yml` only echoes that checks are skipped. Archived workflows are not current CI coverage. `Makefile`'s `test-fast` selects `not slow`, which can still include integration or hardware cases, and its docs target is a placeholder.

Setup commands (`uv sync`, `npm ci`), the complete test suites, live servers, notebook execution and production builds were inspected rather than run. No training, video downloads, paid model calls, or long experiment runs were launched. Existing source/configuration, data, lockfiles and submodule revisions were preserved.
