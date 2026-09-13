# Code organization and independent review

The September 2026 refactor organizes implementation around study responsibilities and short operation names. It changes source paths and software identity, while retaining scientific equations, data splits, protocol decisions and saved evidence. Start with [Python modules](../../src/gavd6_sjepa/source_module_guide.md), [scripts](../../scripts/navigation_guide.md), [cluster jobs](../../slurm/README.md) or [tests](../../tests/README.md). The [migration manifest](code-migration-2026-09-13.json) records the exact old-to-new mapping.

## Structure and naming

| Area | Result |
| --- | --- |
| `src` | Study-local names such as `jepa.py`, `amass_training.py`, `benchmark.py`, `inference.py` and `gavd_probe.py`. Prediction, scaling and accessibility use `future_prediction`, `source_scaling` and `target_accessibility`. Shared files identify conversion, downloads, artifact IO and result administration. |
| `scripts` | Study-local operations such as `build_notebooks.py`, `inspect_fits.py`, `evaluate_failures.py` and `probe_residual_head.py`. Scaling orchestration is Python under `scripts/research_directions/source_scaling`, beside its calibration and notebook tools. |
| `slurm` | Study/group directories with operation names. Shared-data jobs and reflection controls have their own folders. Laterality jobs are grouped into paired-AMASS, source-transfer and legacy protocols. Dependency order remains explicit in submission scripts. |
| `tests` | **52 flat files consolidated into 32 test modules**, grouped by study or infrastructure. Shared fixture/root helpers live in `tests/support.py`. All 355 original test methods remain independently discoverable. |

Parents provide scientific context; filenames identify an operation or mathematical component. Standard names such as `cli.py`, `__init__.py` and `build_notebooks.py` can repeat in separate namespaces. A global basename-uniqueness requirement was replaced with scoped layout/discovery checks. Established dataset/model names remain; local skeleton nicknames were removed from current implementation filenames. Serialized `core11-v1` fields and established command names retain compatibility spelling.

Examples:

| Before | After |
| --- | --- |
| `reflection_equivariance/jepa_model_architecture.py` | `reflection_equivariance/jepa.py` |
| `latent_laterality/laterality_corruption_inference.py` | `latent_laterality/inference.py` |
| `future_innovation/fi_joint_training.py` | `future_prediction/joint_training.py` |
| `motion_preservation/adapters.py` | `motion_preservation/pretrained_models.py` |
| `motion_preservation/learning.py` | `motion_preservation/repair_models.py` |
| `motion_preservation/gavd.py` | `motion_preservation/gavd_stress.py` |
| `build_reflection_gavd_notebooks.py` | `build_gavd_notebooks.py` |
| `test_future_innovation_scaling_notebooks.py` | `future_feature_prediction/scaling/test_notebooks.py` |

## Consolidation decisions

- Merged the 69-line sequence-benchmark command handler into `latent_laterality/benchmark.py` and the 72-line GAVD-probe handler into `reflection_equivariance/gavd_probe.py`. Each invocation now lives with the implementation it exclusively serves.
- Merged the prediction-gate notebook shell helper into its shared environment helper. The CLI, notebook and cached-repair launch paths retain their distinct dependency graphs and failure behavior.
- Consolidated small tests around input data, teacher features, decision rules, training, diagnostics and notebook execution. Shared helpers replace copied fixtures and test-to-test imports; test bodies and scientific assertions were preserved.
- Kept earlier residual fitting separate from repaired joint-ridge fitting, and input-readiness checks separate from pixel-intervention validity audits. These implement different scientific comparisons.
- Kept integrity-only inspection separate from numerical verification that can refit models. Similar reporting names do not imply equivalent execution.
- Kept the three diagnostic scripts separate: inspecting saved fits, probing a synthetic failure mechanism and evaluating cached-result failures answer different questions.
- Kept large model, conversion and inference modules separate. Some already exceed 1,000 lines; combining them would increase the reading burden. A later split would need its own numerical-equivalence and serialization review. The [prediction package guide](../../src/gavd6_sjepa/research_directions/future_prediction/README.md) groups its remaining files by responsibility.

## Independent criticism and resolution

Independent agents first audited source/provenance, execution/Slurm, and naming/tests. After implementing separate areas, they reviewed one another's changes. The workflow adapted when checks exposed specific failures.

| Review finding | Resolution |
| --- | --- |
| Renamed Python classes can make old fitted Joblib objects unloadable. | Preserved exact historical source and paths in a checked replay archive. Current code has an honest new identity; the existing swap-head compatibility export remains narrowly scoped. |
| Broad source globs or old hard-coded directories can silently omit fingerprint inputs. | Current fingerprints explicitly require canonical modules, shared inputs and nested launch paths; missing required files fail. No old hash is substituted for changed code. |
| The gate and strict scaling/accessibility readers do not have the same resumption policy. | Preserved their actual behaviors: the gate records changed code/runtime provenance, while strict readers reject incompatible software. |
| Test concatenation can shadow fixtures or silently duplicate/drop cases. | Audited module bindings and before/after method inventories; all 355 methods collect exactly once. Reused source fixtures were extracted into ordinary helpers. |
| Tests that retrieve old paths from Git HEAD would fail after committing the rename. | Historical numerical comparisons read the immutable replay archive instead. |
| Number removal can accidentally change Slurm job order or dependencies. | Submission scripts retain explicit order and `afterok`/`afterany` semantics. All 45 resource-directive blocks are unchanged. |
| A broad path rewrite touched the replay manifest and normalized exposure CSV bytes. | Restored both from the original archive, excluded replay material from rewrites, and verified every archived digest and both exposure resources. |
| Split path literals and editable notebook metadata still named old builders/modules. | Updated startup locators, source pointers, cell sources and deterministic IDs together; retained execution outputs and counts. |
| Installed console scripts still imported the removed CLI module. | Refreshed the editable project installation without changing dependency versions; verified `gavd6 --help` and the training entry-point target. |

## Validation and evidence boundary

The full suite and focused reruns finished with **344 passed and 11 expected skips**, covering numerical models, masking, calibration, source separation, artifact integrity, notebook execution and scheduler behavior. Final collection contains exactly 355 unique tests. All 36 canonical notebooks retain their execution output, counts, cell metadata and attachments; only five editable-source metadata pointers were updated. The link audit found no newly broken local links; 52 preexisting historical/artifact gaps remain. Detailed results are in the migration manifest. Notebook checks compare all 26 generated sources against their builders; six reflection notebooks also start from four working-directory contexts. The four real-kernel cases initially required permission to open local Jupyter sockets and passed when rerun with that permission.

The replay check verified **287 archived files**, reproduced all three original gate/scaling/accessibility fingerprints, and confirmed snapshot imports never fell through to the current checkout. The current fingerprints differ, as they should. All original numerical protocols, recorded exposure resources and notebook execution evidence remain intact. No held-out experiment or new real-model training was run.

[Replay instructions](../../scripts/archive/code_layout_20260913/README.md) explain how to use old fitted objects. Three small documentation redirects retain links from unchanged numerical protocols; they do not provide duplicate executable code. The compact archive is retained with the repository; a full local recovery archive also preserves the original source notebooks and outputs. Historical datasets, checkpoints and absent run bundles remain external dependencies.

For routine validation after future changes:

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -t . -v
PYTHONPATH=src .venv/bin/python -m gavd6_sjepa.workspace_validation.notebooks
.venv/bin/gavd6 --help
git diff --check
```
