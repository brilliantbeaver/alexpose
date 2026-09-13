# Local result organization validation

Completed on 12 September 2026 Pacific time (13 September UTC).

The nine original top-level bundles were relocated under
`outputs/studies/{future-innovation,fixed-reflection-baselines,latent-laterality}`.
The [catalog](../../outputs/README.md) indexes 16 study/run/dataset/evaluation
entries and 21 preserved notebook files. Collection entries overlap their child
entries and must not be added together when calculating storage use.

## Preservation and reader checks

- All **863 original files** match their before-migration hashes, byte sizes and
  nanosecond modification times in both the organized bundles and retained
  backups. The original paths resolve through relative compatibility links.
- All five recorded dependency snapshots pass. The repaired comparison and
  accessibility panel read the organized parent/source bundles through the
  unchanged absolute paths in their contracts.
- The unchanged direct-v3 verifier reconstructed **20 selected models, 51,200
  prediction rows and 8,000 bootstrap rows**. Its development decision remains STOP.
- The unchanged accessibility verifier reconstructed **40 selected arm models,
  102,400 prediction rows and 16,000 bootstrap rows**. The supplement checked
  1,480 candidate display records and 40 fitted-state diagnostic records. Its
  outcome remains `no_supported_temporal_lead`.
- Original gate cache/readiness reading, both frozen-study readers and the
  swap-probe encoder checksum pass. These checks do not imply expanded training
  or re-execution of raw media processing.
- The three implementation identities for historical future innovation, scaling
  and the cached bridge are unchanged. Fitting modules, protocol contracts,
  model files and scientific reports were not edited.
- All **38 relative documentation links** in relocated archived notebooks
  resolve through two navigation links outside the run roots. Notebook bytes
  remain unchanged.

The original copied gate references 3,545 unavailable raw-processing artifacts
across its candidate/cohort receipts. These are **receipt references**, not an
additional claim about unique missing files. They were absent before relocation
and are enumerated in the catalog's `unavailable_source_evidence`. The legacy
AMASS collections lack comprehensive hash seals and receive
`inventory_recorded`; they are not mislabeled as numerically verified.

The [migration receipt](../../outputs/.organization/relocations/2026-09-13T040646.982820_0000-ebd8d885/relocation.json)
contains original snapshots, backup paths and reader results. Additional checks
are recorded under `outputs/.organization/`. Backups are retained, adding
approximately 560 MiB of storage. The [organization policy](output-organization.md)
documents commands, restart semantics, rollback and cross-machine limits.

## Automated validation

| Check | Result |
|---|---|
| `test_experiment_result_organization.py` | 14 passed: preserved copies, unchanged contracts, retained backups, aliases, rollback, copy/reader failures, stale verification, path safety, alternate HAIC layout and archived notebook links |
| `test_future_innovation_launch.py` | 13 passed, 1 optional integration test skipped because `FI_LAUNCH_TEST_CALIBRATION` was not supplied |
| `test_future_innovation_scaling_notebooks.py` | 12 passed, including actual kernels and external inspection preserving the entire run snapshot |
| `test_research_directory_ownership.py` | 3 passed |
| Shell syntax and `git diff --check` | Passed |
| ID-based notebook submission preview | Correct canonical run/parent and external inspection logs; dry run only |

The final relevant suite accounts for **42 passing tests and one skip**. Local
Jupyter kernels required execution outside the socket-restricted sandbox; the
real-kernel checks passed with that access. Synthetic scheduler fixtures used
temporary storage; no Slurm job was submitted.

The separate `test_unique_project_filenames.py` check fails on three pre-existing
`__init__.py` files (root package, future-innovation scaling, and ICLR bridge).
`git ls-tree HEAD` confirms those duplicates precede this change. No new file
introduced that collision, and this unrelated filename policy was left unchanged.
