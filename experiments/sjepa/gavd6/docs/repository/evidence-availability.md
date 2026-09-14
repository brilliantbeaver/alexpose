# Evidence availability

Checked on **13 September 2026**. This inventory explains unavailable local artifacts cited by the study documents. It preserves the original repository-relative paths so an artifact owner can locate the correct run. A run registered in the [experiment registry](../studies/experiment-registry.json) is not necessarily installed in this checkout.

The live study-document audit found 37 references to 32 distinct unavailable paths. These are historical evidence and generated local catalogs; the documentation update did not regenerate them. Links marked **unavailable here** lead to this inventory, not to a replacement experiment. Numerical claims remain historical reports unless their supporting artifacts can be inspected independently.

Saved execution notebooks also retain historical navigation links. Their [companion guide](historical-notebooks.md) supplies current destinations without changing the recorded files. Those links do not imply missing experiment outputs.

## Available preserved evidence

The [accessibility evidence directory](../studies/future-feature-prediction/accessibility/evidence/README.md) contains selected report and verification copies. In particular, the [preserved panel report](../studies/future-feature-prediction/accessibility/evidence/cached-panel-report.json) matches its source-manifest SHA-256 (`e182537f7864750e3458b37d2f0518f918a33384327d442db4c881aba55914f0`). These copies support inspection of the reported numbers. They do not contain the complete original run, fitted models or raw predictions needed for fresh reconstruction.

The [archived code snapshot](../../scripts/archive/code_layout_20260913/README.md) preserves the earlier source layout and numerical protocols. It is a source archive, not a recovered result bundle. Current source notebooks explain workflows; they must not be substituted for missing executed notebooks.

## Local output catalog and relocation receipt

The catalogs are generated for a particular local installation. The [output organization guide](../studies/output-organization.md) describes how to refresh a catalog from installed bundles. Refreshing it does not recover a missing run or the earlier migration receipt.

- `outputs/.organization/relocations/2026-09-13T040646.982820_0000-ebd8d885/relocation.json`
- `outputs/README.md`
- `outputs/catalog.json`

## Target accessibility

The following original artifacts are absent from this checkout:

- `outputs/iclr-bridge-cached-20260911`
- `outputs/iclr-bridge-cached-20260911/config/frozen-protocol.md`

## Direct-v3 gate

The following original artifacts are absent from this checkout:

- `outputs/future-innovation-direct-v3-dev-20260911/config/frozen-protocol.md`
- `outputs/future-innovation-direct-v3-dev-20260911/config/runtime-contract.json`
- `outputs/future-innovation-direct-v3-dev-20260911/notebook_runs/local-cached/00_question_and_worked_example.ipynb`
- `outputs/future-innovation-direct-v3-dev-20260911/notebook_runs/source-snapshots/generator-direct-v3.py`
- `outputs/future-innovation-direct-v3-dev-20260911/reports/gate-decision.json`
- `outputs/future-innovation-direct-v3-dev-20260911/reports/gate-report.md`
- `outputs/future-innovation-direct-v3-dev-20260911/reports/numerical-verification.json`
- `outputs/future-innovation-direct-v3-dev-20260911/source_snapshots/fitting-code-contract.json`

## Source scaling

The following original artifacts are absent from this checkout:

- `outputs/future-innovation-source-curve-dev-20260911-v2`
- `outputs/future-innovation-source-curve-dev-20260911-v2/config/source-reservation.csv`
- `outputs/future-innovation-source-curve-dev-20260911-v2/implementation/manifest.json`
- `outputs/future-innovation-source-curve-dev-20260911-v2/notebook_runs/manual-20260912T160403-4d747a14/23_source_learning_curves.ipynb`
- `outputs/future-innovation-source-curve-dev-20260911-v2/notebook_runs/manual-20260912T164103-70df86ed/23_source_learning_curves.ipynb`
- `outputs/future-innovation-source-curve-dev-20260911-v2/notebook_runs/manual-20260912T193406-e6c6f7ba/23_source_learning_curves.ipynb`
- `outputs/future-innovation-source-curve-dev-20260911-v2/notebook_runs/manual-20260912T221427-a0d77ebc/23_source_learning_curves.ipynb`
- `outputs/future-innovation-source-curve-dev-20260911-v2/reports/cohort-audit.json`

## Latent laterality

The following original artifacts are absent from this checkout:

- `outputs/latent-laterality/amass-benchmark-seed7-v2-chart-paired/benchmark_gates.csv`
- `outputs/latent-laterality/amass-benchmark-seed7-v2-chart-paired/gate_decision.json`
- `outputs/latent-laterality/amass-benchmark-seed7-v2-chart-paired/sequence_metrics.csv`
- `outputs/latent-laterality/amass-benchmark-seed7/benchmark_gates.csv`
- `outputs/latent-laterality/amass-benchmark-seed7/gate_decision.json`
- `outputs/latent-laterality/amass-gauge-v2-seed7-validation/evaluation_contract.json`
- `outputs/latent-laterality/amass-gauge-v2-seed7-validation/gauge_path_metrics.csv`
- `outputs/latent-laterality/amass-gauge-v2-seed7-validation/gauge_readout_predictions.csv`
- `outputs/latent-laterality/amass-gauge-v2-seed7-validation/gauge_readout_summary.csv`
- `outputs/swap-probe-seed7/summary.csv`
- `outputs/swap-probe-seed7/validation_edge_metrics.csv`

## Restoring access

Recover the exact run from its original storage or an identified backup, then verify its recorded manifest and provenance before restoring links. Use the registry's original and organized paths to establish run identity. A newly executed run needs its own identity and evidence record; it cannot replace a historical result silently.
