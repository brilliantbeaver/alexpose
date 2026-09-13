# Experiment result organization

Organize results by study and stable run identity. Keep each run's configuration,
manifests, cache receipts, models, predictions, quality checks and reports together.
The [curated registry](experiment-registry.json) records purpose and evidence
boundaries. The generated [local index](../../outputs/README.md) and
[catalog](../../outputs/catalog.json) record installed locations and observed state.
The catalog is a dated local view, not a live HAIC job monitor.

See the [completed local migration and validation](output-organization-validation.md)
for preservation checks, reader reconstruction, test results and retained backups.

```text
outputs/
  README.md                         generated human-readable index
  catalog.json                      generated locations, status and verification
  studies/
    fixed-reflection-baselines/<run-id>/
    latent-laterality/<run-or-collection-id>/
    future-innovation/<run-id>/
  inspections/<run-id>/<batch-id>/  new notebook copies
  .organization/
    verification-*.json             independent check receipts
    relocations/<batch>/            snapshots and migration/rollback receipt
    backups/<batch>/                retained original directories
  <legacy-name> -> studies/...      compatibility links after migration
```

## Identity and evidence

A run ID identifies one frozen scientific setup and its execution attempt. Use
`<comparison>-<protocol-or-cohort>-<scope>-<YYYYMMDD>[-<attempt>]` for new IDs, for
example `source-learning-curve-available-dev-20260912`. Use a new ID when the
protocol, reserved cohort or frozen fitting software changes. Preserve existing
IDs, including older names such as `iclr-bridge-cached-20260911`.

The cached accessibility comparison belongs to the future-innovation study;
the ICLR documents link to it as publication evidence. The laterality collection
retains its existing benchmark/training/evaluation structure. Its seven child
entries are indexed separately; do not sum collection and child sizes as if they
were independent copies.

The catalog separates:

| Field | Meaning |
|---|---|
| `kind`, `description`, `limitations` | Curated experimental purpose and evidence boundary |
| `location`, `legacy_path`, `canonical_path`, `origin_hint` | Current storage and historical origin; origin hints are not live remote checks |
| `parents` | Logical dependencies, including cached teacher input and frozen encoder reuse |
| `protocol`, `declared_run_id`, `contract` | Saved experiment identity and contract checksum |
| `counts` | Counts from identified receipts, keeping initial inventory separate from processed data |
| `execution_state` | Availability and stage reached according to saved artifacts |
| `scientific_decision` | Reported STOP, development outcome or not evaluated |
| `verification` | Method, timestamp, snapshot binding, unavailable evidence and whether numerical reconstruction occurred |
| `checkpoints` | Per-checkpoint hashes, filename-derived variant/seed and matching history/config coverage for legacy collections |
| `inspections` | Notebook location, recorded input root, execution status and verification metadata |

Notebook execution success does not imply completed training. The two
`future-innovation-source-curve-dev-20260911*` bundles contain frozen setups, but
no expanded fitted models or learning-curve report. The `-v2` attempt revised
calibration and software while preserving the same source reservation. Keep both.

The older `repaired-jepa-seed7-v2` collection contains four variants, while its
run-level config and summary describe the last standard-only invocation. The
catalog identifies that coverage without inventing missing provenance or
relabeling the checkpoints as results of the later mask repair.

## Read and refresh

Run from the project checkout using the existing Python environment:

```bash
bash scripts/workspace_management/experiment-results.sh refresh
bash scripts/workspace_management/experiment-results.sh resolve gate-v2
bash scripts/workspace_management/experiment-results.sh verify
bash scripts/workspace_management/experiment-results.sh verify --readers
```

`refresh` reads bundles and writes only `outputs/catalog.json` and
`outputs/README.md`. It indexes newly created `outputs/studies/<study>/<id>`
directories as unclassified runs until their interpretation is registered.
Unregistered legacy roots are surfaced explicitly. Editing the registry changes
navigation/interpretation, never a frozen run's scientific identity.

`verify` checks installed configuration and artifact hashes and exact dependency
snapshots. Unsealed historical collections receive `inventory_recorded`, not an
assertion of independently established provenance. Missing required models,
cached arrays or scored artifacts fail verification. Original raw-processing
files omitted from the copied gate are reported under
`unavailable_source_evidence`; they are not silently treated as verified.

`verify --readers` additionally invokes the unchanged gate cache/readiness reader,
the repaired direct-v3 numerical verifier, the accessibility verification
supplement, the frozen-study readers, and the swap-probe encoder hash check.
Numerical reconstruction is reported only for the comparisons actually
reconstructed. These operations do not train students, rerun teacher inference,
or reopen raw video. Changed bundle snapshots make earlier catalog verification
stale. Machine-readable receipts remain under `.organization/`.

## Start and inspect new source-learning-curve runs

The public scaling launcher accepts logical IDs as well as explicit paths:

```bash
export GAVD6_ROOT=/path/to/gavd6
export FI_PARENT_ID=gate-v2
export FI_RUN_ID=source-learning-curve-available-dev-20260912
unset FI_PARENT_ROOT FI_RUN_ROOT
# Set the model/media variables in the HAIC source-learning-curve guide.
bash slurm/future-innovation-scaling/launch/submit.sh check
```

An installed ID resolves through the registry. A new `FI_RUN_ID` selects
`outputs/studies/future-innovation/<id>` without creating it during selection.
An explicit root that conflicts with an ID is rejected. A registered but missing
parent is an error; use the actual `FI_PARENT_ROOT` on a machine whose historical
bundle layout has not been organized. The launcher prints its selected run and
parent, and continues to supply `FI_SCALING_ROOT` to the unchanged internal jobs.

The public launcher defaults `FI_INSPECTION_ROOT` to
`outputs/inspections/<run-id>`. Notebook-only submissions write their Slurm logs
and submission IDs there, leaving the inspected run untouched. Each notebook
batch has its own directory and execution receipt. The inspection root is bound
to the saved study contract and cannot be reused for a different run. Original
notebook files already inside preserved bundles remain in place.

Direct Python calls to the notebook runner retain the legacy output convention
when no external inspection root is supplied. For external inspection explicitly:

```bash
.venv/bin/python slurm/future-innovation-scaling/launch/notebooks.py \
  --run-root outputs/studies/future-innovation/future-innovation-source-curve-dev-20260911-v2 \
  --inspection-root outputs/inspections/future-innovation-source-curve-dev-20260911-v2
```

## Relocation and rollback

```bash
bash scripts/workspace_management/experiment-results.sh migrate
bash scripts/workspace_management/experiment-results.sh migrate --apply
```

The first command prints the exact source/destination plan without modifying
bundles. Applying it is serialized against other migrations and follows these
steps:

1. Verify retained artifacts and unchanged production readers before copying.
2. Copy entire registered bundles with file metadata preserved. Compare every
   file's hash, size and nanosecond modification time against the original.
3. Retain original directories under `.organization/backups/<batch>/` and place
   relative compatibility links at their former locations.
4. Run the unchanged readers against the organized bundles. Their frozen
   absolute references continue to work through the compatibility links.
5. Publish the catalog and a receipt containing original snapshots, backup paths
   and verification results. Recheck the retained originals.

An exception during copying, switching or verification restores the original
paths. Partial/organized copies are retained beside the relocation receipt.
Destination collisions, nested migration roots and symlinks inside bundles are
rejected before adoption. A later explicit rollback uses the recorded receipt:

```bash
bash scripts/workspace_management/experiment-results.sh rollback \
  outputs/.organization/relocations/<batch>/relocation.json
```

Rollback validates registered paths and original backup snapshots, refreshes the
catalog, and retains the organized copies. Stop writers before migrating or
rolling back a run; the organization lock serializes this tool, not Slurm jobs.
Keep the original backups until a separate retention decision is made.

Compatibility links preserve local reading, not arbitrary cross-machine
relocation. On another machine, legacy readers still need their declared
dependency paths to resolve to the verified bundles. The scaling reader supports
a parent override, but older cached readers do not. Never rewrite frozen
contracts to substitute paths. Likewise, archived initialization/resume code may
compare canonical paths or frozen software identity; these completed historical
runs are for inspection, and changed experiments use new IDs.

The original parent snapshots cover complete inventories, including `.DS_Store`.
Two navigation links, `outputs/studies/docs` and `outputs/studies/slurm`, preserve
the relative documentation links embedded in archived notebooks. They sit outside
every run, are excluded from study discovery, and leave notebook bytes unchanged.
Keep new notes, cleanup operations, inspections and publication exports outside
frozen source trees. The catalog/migration modules intentionally live outside
the fitting packages, and the launcher changes are outside the frozen fitting
fingerprints. No numerical protocol file needs to change for organization.

`outputs/`, executed notebooks and backups remain ignored by Git. The registry,
organization tools, tests and this policy are tracked. Figures and manuscripts
intended for publication remain in study documentation and cite identified runs.
