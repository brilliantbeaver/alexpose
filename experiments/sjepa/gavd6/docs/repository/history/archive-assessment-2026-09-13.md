# Repository relevance and archive assessment

**Assessment date: 13 September 2026.** The selected objective is **preserving real movement while repairing tracking failures**, as specified by [Proposal 01](../../../notes/world-model-extensions/proposals-04/01-preserve-real-movement.md) and implemented in the [motion-preservation notebooks](../../../notebooks/motion_preservation/README.md).

**Recommendation:** make motion preservation the primary entry point; reclassify the earlier classification, reflection, laterality and teacher-distillation programs as historical studies or optional comparisons. Archive superseded design material first. Keep scientific evidence, audited identities and shared implementation available. Physical relocation of research code needs a separate dependency and compatibility change.

This assessment moves or deletes nothing. The accompanying [file inventory](archive-assessment-2026-09-13.json) assigns a recommendation to every source/workflow file and summarizes ignored artifacts by bundle. Its categories describe relevance and retention, not permission to delete files.

## What was examined

The snapshot contains **797 source/workflow files: 681 tracked and 116 untracked**, including three tracked manifest records hidden by current ignore rules. Static analysis parsed all **202 Python sources** and code cells from all **36 notebooks**, with no syntax failures. It recorded imports, existing local documentation references, notebook outputs, sizes and content hashes. Targeted reading covered the selected proposal and execution contract, current implementation and launchers, older study decisions, source ownership, output organization and compatibility tests.

A filesystem inventory covered **11,193 local files**, approximately **855 MiB of logical file content**, excluding `.venv`. Ignored outputs are assessed as **97 bundles/groups**, not individually interpreted scientific arrays. The approximately 1.4 GiB installed environment was measured separately, not reviewed package by package. Logical sizes are not guaranteed recoverable disk space: filesystem compression, cloning and retained backups can change that relationship.

The repository's Git root is the parent `alexpose` project. This analysis is scoped to `gavd6`; it does not establish that no parent-project or external consumer uses an old import path. No HAIC filesystem, job state, remote model load or new scientific result was inspected. The source snapshot predates these two assessment files.

| Recommendation | Source files | Meaning |
| --- | ---: | --- |
| Keep current study | 46 | P1 implementation, notebooks, launchers, tests and primary design material |
| Keep shared support | 36 | Current dependencies, data utilities, environment, compatibility and artifact tools |
| Keep reference | 25 | Useful measurement, model-access and data-inspection material |
| Refresh navigation | 9 | Retain the entry point, update what it calls current |
| Preserve historical evidence | 67 | Results, protocols, figures and limitations from earlier studies |
| Archive candidate | 171 | Superseded design, teaching and supporting assets; move with their links |
| Archive after dependency review | 199 | Older executable workflows and tests; no blanket move |
| Defer alternative | 19 | Other current portfolio proposals and their associated assets |
| Retain existing archive | 203 | Already historical, including compatibility adapters and identity-audit records |
| Optional generated cleanup | 22 | Figure previews and TeX intermediates, conditional on source/final-artifact retention |

## The scientific boundary for relevance

The initial P1 experiment asks whether video evidence helps a small adapter retain genuine movement at a comparable amount of tracking-error removal, beyond calibrated flow and simple baselines. It uses complete AMASS motion, whole-body conversion, controlled rendering, frozen motion/flow models, separate people for training/calibration/development/final evaluation, and a guarded observational GAVD stage.

Consequently, lower JEPA feature loss, better five-condition classification, an RGB-conditioned skeleton increment, and reflection consistency are not primary success criteria. They remain useful motivation or optional comparisons. S-JEPA/V-JEPA features must earn a role beyond the implemented coordinate/flow baselines; their historical presence does not make their full training stacks current dependencies.

The [implementation validation](../../../notebooks/motion_preservation/VALIDATION.md) records CPU/demo and notebook checks, not a successful real-data preservation result. Retention must therefore protect the intended experiment and its provenance without presenting the new direction as already validated.

## Keep in the working study

| Files or family | Why they remain relevant |
| --- | --- |
| `src/gavd6_sjepa/research_directions/motion_preservation/` — all 11 Python files | Configuration, manifest-backed motion, geometry, rendering, prior/flow adapters, learning, metrics, orchestration, GAVD inspection and reporting form the selected experiment. |
| `notebooks/motion_preservation/` — six notebooks, README and validation | The actual research sequence and the distinction between demonstration and real evidence. |
| `scripts/research_directions/motion_preservation/{build_notebooks,execute_notebook}.py` | Notebook source ownership and retention of executed/failed outputs. |
| `slurm/motion-preservation/` — 10 files | Pilot configuration, stage dependencies, explicit final evaluation and separate GAVD launch. |
| `tests/test_motion_preservation_*.py` — six files | Protect movement geometry, model interfaces, calibration, identity grouping, ambiguity and reservation boundaries. |
| `proposals-04/01-preserve-real-movement.md`, `00-evidence-and-execution.md`, `02-literature-and-access.md`, `optical-flow-directions.md`, portfolio README and review | Selected question, constraints, baseline obligations and access assumptions. Retain the P1 diagrams and shared portfolio map. |
| `notebooks/amass/01_visualize_amass_smplh_poses.ipynb` | Useful inspection of raw SMPL-H/DMPL motion; unlike the old Core11 training objective, its data-inspection role remains directly useful. |
| AMASS/GAVD inventory, identity, download and shared artifact utilities | Necessary acquisition and provenance support, even where the P1 runner does not invoke every utility. |
| `pyproject.toml`, `uv.lock`, `.env.example`, `.gitignore`, local `.env` and working `.venv` | Maintain the executable environment. The current optional extra overlaps older dependencies; do not remove packages merely because an older study is archived. Keep `.env` private. |

The four immediate local manifest dependencies are:

- `manifests/amass/amass_raw_inventory_eligible.csv`
- `manifests/amass/amass_subject_registry.csv`
- `manifests/amass/amass_subject_splits.csv`
- `manifests/gavd/gavd_full_sequences.csv`

These files are ignored by Git but are **current research inputs**. Preserve their exact contents and the audit history. The current subject registry is byte-identical to the tracked `manifests/archive/amass_subject_registry.audited.csv`; the duplicate provides useful provenance, while the active path is hard-coded by the loader. The raw inventory, full-video manifest/summary and Core11 conversion ledger also retain acquisition or ablation value.

## Dependencies that prevent a blanket code archive

The following chains were confirmed in source and by importing `workflow` and `gavd` in the existing environment:

```text
motion_preservation/data.py
  -> data_foundations/amass_core11_conversion_pipeline.py
     (load_amass_sequence, not the Core11 output representation)

motion_preservation/gavd.py
  -> future_innovation/fi_video_pose.py
     -> data_foundations/gavd_pose_extraction_primitives.py
     -> future_innovation/fi_contracts.py
        -> shared_infrastructure/artifact_io_operations.py
```

`fi_contracts.py` also contains a deferred import of `fi_joint_models.py`. The conservative static closure contains 17 local modules: 11 P1 modules and six other files. That deferred model import did not execute in the import check; it is not evidence that the P1 experiment fits FI ridge models.

**Keep the converter and FI decoder dependencies at their existing paths now.** A later refactor can establish a shared raw-AMASS loader and exact-frame decoder, then verify P1 against those interfaces. Moving the entire Future Innovation directory today would break the GAVD module. Renaming the Core11 converter based on its title alone would break motion loading.

There are additional compatibility constraints:

- `command_line_interface.py` resolves many old handlers by string; `pyproject.toml` still exposes `train-amass-core11`.
- The root package lazily exports the reflection model API. Existing archive import adapters refer to the current reflection/laterality modules, and `test_archived_compatibility_paths.py` checks those relationships.
- `fi_contracts.code_fingerprint()` includes FI Python files, shared files, the CLI, dependency configuration and FI Slurm launchers. Scaling has its own software identity checks. Relocating launchers or editing a dependency lock is not scientifically neutral for every frozen resume path. Read-only readers and resume checks also differ; verify the specific operation rather than assuming every move invalidates every reader.
- `tests/test_research_directory_ownership.py` expresses the policy that active code must not import archive modules. Shared functionality belongs in a shared layer, not in an archive that the new study depends on.

For the first cleanup, use historical status labels and navigation changes. Retain old executable packages and their tests together until these boundaries are resolved.

## Archive or demote these study families

| Candidate | Evaluation and proposed treatment |
| --- | --- |
| `notes/world-model-extensions/proposals-01/`, `proposals-02/`, `proposals-03/` | 51 files documenting earlier portfolios. Archive as dated design history, with figures and reference ledgers. They should not compete with P1 as current execution instructions. |
| `notes/world-model-extensions/brainstorm/`, `codex_ideas.md`, older shared `images/` | Earlier concept exploration, including Wrench-JEPA/force and forecasting agendas. Archive linked documents and assets together; retain useful citations through the current reference ledger. |
| `notes/cross-protocol-perturbation-response/`, `notes/future-innovation-distillation/`, `notes/latent-laterality/`, older `notes/prompts/` | Superseded or different experimental objectives. Preserve decision history, but remove from the primary run order. Keep current P1 constraints accessible even when the prompt that introduced them becomes historical. |
| `notebooks/foundations/` | All seven belong to the earlier augmented-normal/classification curriculum. Data teaching can remain reference material, but the 159-sequence curriculum and condition classifiers are not the P1 experiment. Archive executed versions before making lighter teaching copies. |
| `notebooks/idea09_reflection_equivariance/`, `notebooks/idea05_signed_laterality/` | Nine reflection and two signed-laterality notebooks. Preserve as historical representation/control studies; they do not establish event preservation under tracking repair. |
| Reflection/laterality source directories, associated builders and Slurm jobs | Seven reflection and ten laterality Python modules. Historical execution capability, with possible optional representation/corruption comparisons. Demote now; relocate only after checking shared imports, public exports, handlers and adapters. |
| `notebooks/future_innovation/`, FI and scaling code/builders/Slurm | Preserve the completed negative gates and scaling inspection. These evaluate teacher-target accessibility, not restoration. Keep the decoder dependency above; treat the remaining 23 FI modules and eight scaling modules as historical/optional execution infrastructure. |
| `docs/studies/iclr/`, `notebooks/iclr_bridge/`, bridge source/builders | Valuable motivation, cached-panel evidence and an earlier manuscript thesis. Keep a stable historical bundle; rewrite a future P1 manuscript separately. The recently edited `07_scaling_result_and_research_strategy.md` remains a useful research transition record, not the current method contract. |
| `notebooks/evaluate_strokepig_frozen_jepa.ipynb`, StrokePIG result and outputs | Archive together as a negative force-prediction feasibility study. Force inference is not the P1 target, and the negative result should remain inspectable. |
| Old `docs/tutorials/` and root `images/` | Historical S-JEPA teaching assets. Move with their referring tutorials/README sections, not independently by extension. |

Other **current** `proposals-04` alternatives deserve a different label. P2–P7 are reserves, not obsolete findings. P7's optical-flow measurement discussion supports P1, even though its forecasting objective is separate. Retain common measurement notes, checkpoint/access records, current figure generators and baseline reviews. Keep optional projects out of the P1 launch sequence.

Existing `docs/history/`, `notes/archive/`, `scripts/archive/`, `src/gavd6_sjepa/archive/`, `manifests/archive/` and `work/archive/` already have the right historical role. Moving them again adds little. Compatibility adapters are small and useful; their archive location does not make them dead code.

## Evidence and protected cohorts must survive

Keep the laterality uniform-control failure, the FI direct-v2/direct-v3 STOPs, the cached bridge's unsupported temporal lead, the expanded learning-curve inspection, and the frozen-probe limitations. They explain why P1 is worth testing and prevent recycling failed claims. Their datasets, metrics and questions differ; archiving must not collapse them into one result.

Especially retain `notebook_runs/haic-run-02/23_source_learning_curves.ipynb`, `ANALYSIS.md` and its analysis files. This is local evidence for the expanded study. It is not a substitute for the original remote models, predictions and reports. Preserve raw and processed data lineage when relocating older pose outputs or augmentation cohorts; unusual movement should never be discarded merely because an old cohort was filtered for a different task.

The P1 GAVD stage requires the earlier FI `config/source-reservation.csv`. No file with that name was found in this checkout outside the excluded environment. The full-video manifest does not recover the 43 held source IDs. Preserve/recover the actual reservation artifact before real GAVD inspection; do not create a replacement split. AMASS person splits and the reserved final event family likewise remain binding after a study is renamed or archived.

**Installed-artifact discrepancy:** all 16 entries in `experiment-registry.json` have absent canonical and legacy paths here. `outputs/` contains only `.DS_Store`. The [migration validation record](../output-organization-validation.md) describes a completed migration, retained backups and successful readers, but its referenced receipt and bundles are not installed in this snapshot. This establishes a location mismatch, not a failed or fictitious earlier migration. Do not use that record to authorize deletion of supposed local duplicate backups. Recover or locate the actual bundles, then verify them where they exist.

## Storage cleanup: prioritize verified generated material

| Local candidate | Observed logical size | Recommended handling |
| --- | ---: | --- |
| Seven foundation source notebooks | 27.2 MiB total; about 26.7 MiB serialized output payload | Preserve an identified executed copy, then clear outputs from teaching sources. Notebook 01 alone is 24.4 MiB. Moving files within the checkout does not reduce Git history. |
| Four `work/artifacts/source-learning-curve-*` collections | About 355.7 MiB combined | Archive calibration histories together. Exact duplicate selection JSON exists across seven calibrations, but different designs/software/receipts may still matter. Prefer a verified compressed archive over deleting whichever directory looks oldest. |
| `work/artifacts/notebook_runs/future_innovation/` | About 169.7 MiB | Repeated synthetic verification and teaching runs. Retain the runs cited by validation documents and representative successful/failure cases before pruning redundant executions. |
| ICLR `tex-deps/`, `tools/`, `tectonic.tar.gz` | 64.8, 51.5 and 20.7 MiB | Optional offline build cache/tooling. Preserve finished manuscript artifacts and enough build provenance; retain cached dependencies if offline reproduction matters. |
| `tmp/sjepa_writeup/` and `output/{docx,pdf}/` | 7.9 and 0.57 MiB | Repeated render directories are stronger cleanup candidates than their final documents. Check final exports and the generator first. |
| Ordinary Python/Ruff/Matplotlib caches and loose `.DS_Store` | About 4.4 MiB in 308 files | Low-risk cleanup candidates once unused. Do not apply a recursive cache/metadata purge inside frozen evidence bundles whose snapshots include those files. |
| `proposals-04/images/previews/` and TeX intermediates | Included in the 22 optional source-file candidates | Preserve SVGs, figure generator, layout/review records and final manuscript outputs; check inbound references before removal. |

The calibration duplicate check found four selection-file groups, each with seven identical copies, representing roughly 140 MiB of repeated logical content. This is evidence of a compression/deduplication opportunity, **not** evidence that their containing run bundles are interchangeable. Shared writable hard links would also couple supposedly independent artifacts.

Do not classify `.venv`, cached model weights, AMASS data/identity records, `work/artifacts/real/poses/`, or all `.npz`/`.csv` files as disposable. Cached models may be needed by retained readers; poses may be the only locally retained derivation. Their relevance and replacement cost must be evaluated by provenance, not filename extension.

## Recommended sequence

1. **Update navigation.** Point the root README, docs/study/notebook/notes indexes, source guide and script guides to P1. Label old studies as historical, optional comparison or deferred. Keep dated records intact and clarify that the installed `outputs` catalog is currently absent.
2. **Preserve the current uncommitted work.** The snapshot includes 116 untracked source/workflow files, including the new study. Git history alone cannot recover them. Establish a checked-in or verified backup snapshot before moving them.
3. **Archive design history as linked bundles.** Start with proposals 01–03, earlier brainstorming, old implementation prompts and teaching material. Preserve small overviews at established entry points and check local links after any move.
4. **Reduce generated bulk with receipts.** Snapshot executed foundation notebooks; retain cited verification runs and final exports; then clear teaching outputs and prune verified redundant renders/caches. Record what was retained and where.
5. **Handle executable history separately.** Keep old package paths initially. If physical relocation is still useful, first separate the shared loader/decoder interfaces, preserve historical software identities, update CLI/adapters/builders and assess parent-repo callers.
6. **Verify the resulting layout.** Check P1 imports and focused tests, notebook generation/startup, shell launch paths, archive compatibility and documentation links. Run historical artifact readers only against located, verified bundles. Do not reopen final evaluation data or resubmit stopped experiments as a cleanup check.

The immediate payoff is a clear P1 working surface with recoverable scientific history. The highest-confidence changes are navigation and design-history organization; the highest storage payoff is in generated notebook/calibration/build material. Neither requires deleting negative evidence or weakening the experiment's input and holdout boundaries.
