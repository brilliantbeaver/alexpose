# Decision log and dependency board

2026-09-15. The coordinator read the complete47 plan/audit/literature/experiments and relevant historical code. The latest present manuscript was v9. No applicable AGENTS.md was found in the repository/ancestor checks. The working tree already contained extensive user changes in manuscript v8/v9, repository navigation and unrelated studies. Historical files and unrelated edits were preserved; only an additive study registry entry is integrated.

| Work | Owner / authority | Dependencies | Status |
| --- | --- | --- | --- |
| Historical evidence ledger | implementation_audit, read-only history | four complete plan docs | retained means checked, missing inference artifacts disclosed |
| Literature critic | literature, primary sources + assigned note | execution-date search | complete targeted refresh; not exhaustive systematic review |
| Data/contracts/causality | coordinator, new modules/tests only | frozen interfaces | implemented; synthetic contract checks |
| Model/loss/training | literature follow-up, assigned modules/tests | separate context/target API | E1/E2 implemented; CUDA pending |
| Evaluation/uncertainty | implementation_audit follow-up | model/checkpoint and window API | strong baselines and frozen-fit evaluation implemented |
| Notebook/HPC/runbook | execution_design, assigned directories | pure stage/config API | output-free notebooks, explicit execution, dry-run scheduler |
| Integration/registry | coordinator only | all owners | additive integration, no historical rebuild |
| Independent adversarial review | separate Codex CLI, read-only | scoped prompt + changed-file list | protocol completed; implementation/evidence gate tracked separately |

The host exposes four concurrent slots including the coordinator; work was scheduled within them. Agents were given disjoint editing boundaries and later concrete follow-ups, not asked to solve the whole mission. Their outputs were inspected and tested; consensus was not treated as validation.

## Decisions and deviations from the larger plan

- Preserve the negative historical result and the separate uniform-control scientific stop. Do not recover nonexistent raw predictions by simulation.
- Require existing **complete-bout pose exports** as the first real-data route. Raw-video pose extraction is not implemented; fail closed if absent. Original full videos are still explicitly checked for content/PTS alignment. This avoids pretending a pose extractor or full RGB experiment exists.
- Use versioned JSON rather than guessing legacy CSV schemas. Explicit source roles are supplied; no random split can silently promote exposed sources into a fresh test.
- Retain all eligible bouts/windows before endpoint scoring. No eight-sequence cap, short initial crop or target-dependent SSL admission.
- Use `cohort_scope=historical_overlap` for the first matched index/time contrast, with audited source membership. Full-cohort expansion is a new run with `full_allowed`, not a hidden change to an old checkpoint.
- Keep the index adapter and 2D centered-CE recipe labeled adaptations. Historical whole-clip replay remains separate and currently cannot reproduce inference without its missing artifacts.
- Declare prepared-feature masking, not observation-level withholding. Add a separate mask-before-preprocessing branch only if that stronger claim is investigated.
- Replace the originally considered distant same-bout target control with wrong-horizon-order. Short bouts would otherwise be excluded only from the control. This preserves target multiset/count, with a narrower interpretation.
- Enforce separate original-time endpoint and strict80ms teacher interval. Cross-review caught a tolerance implementation that initially blurred this distinction; regression coverage now distinguishes them.
- The selected final update is fixed. Retained intermediate checkpoints are not automatically tuned; extending the optimizer horizon needs a new run.
- Use one explicit outer development split and disclose that readout-inner held groups were SSL-visible. Do not call this fully nested encoder retraining.
- Freeze the ratio-of-mean-errors seed estimand and point/CI engineering gate before any test. Synthetic success never authorizes real expansion.
- Retain per-bout shards and complete role arrays. Current training/evaluation assemble role arrays in RAM. There is no silent cap; real full-cohort RAM/throughput remains to be measured before expansion. A future lazy-sharded loader must preserve sampling/digests and is not claimed present.
- E3, RGB transfer, uncertainty calibration and clinical/physical endpoints remain gated. No favorable score is required for scientific completion; a well-supported direct-baseline win or JEPA null is an acceptable outcome.
- Independent implementation review found and triggered fixes for prepared-cache verification, partial-summary recovery, Slurm runtime-resume identity and governing-protocol binding. Downstream train/development readers verify preparation receipts; test caches have a separate data receipt. Failed summary attempts are preserved in fresh retry directories. Runtime-only checkpoint selection is excluded consistently from scientific identity/task grids and requires one explicit task when submitted. Protocol and decision documents are snapshotted and hashed; later changes reject old runs.
- Full workflow fault injection exposed macOS `/var` versus `/private/var` aliases in artifact paths. Explicit output paths are now canonicalized consistently; no alternative data locations are searched. Historical-overlap runs reject final test locking before any untouched test is opened.
- Final independent HPC cross-check found that the information-audit stage fits the configured CUDA direct forecaster but its launcher had requested CPU resources only. Allocate an accelerator for that stage, verify the emitted pilot DAG's device/resource contract, and retain a new software run identity after the launcher correction. The earlier synthetic execution remains preserved; CPU fixture success alone never established cluster readiness.
- Independent follow-up review found two final-test receipt-chain gaps: derived inventory authorization and direct-reader authentication of the analysis-lock receipt. Authenticate inventory00 before preparation or test opening, and authenticate07-lock inside the shared lock verifier. Matching editable JSON fields alone are insufficient. Fault tests mutate source/role metadata or a selection plus matching opening marker and require rejection before contents are read.
- Missing secondary-horizon inner-validation support must not block an estimable primary0.50s analysis. Serialize an explicit `not_estimable` secondary decoder with training-support reasons and produce unavailable predictions for that slot; do not borrow another horizon's targets or fitted parameters. Missing primary validation support still fails closed. This is a training-only support decision, not a change to final-test eligibility.

## Exact missing inputs for empirical work

An explicit output root, all seven manifest paths, full-video/pose content digests and complete PTS-aligned causal pose exports; audited historical exposure/identity/reservation/source-role assignments; a compatible HAIC CUDA environment/interpreter and authorized scheduler resources. Untouched test availability is not established. No remote directory inspection is needed or authorized to provide these.
