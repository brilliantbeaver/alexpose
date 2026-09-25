# Compact evidence packet for the gait-fidelity paper

This is a file recommendation, not a claim that remote files were copied or
verified. It covers the three real HAIC work directories under
`/hai/scratch/tedmui/alexpose/experiments/sjepa/gavd6/outputs/gait-fidelity/`:

- `walking-core-01`
- `jepa-response-02`
- `readout-repair-03`

The latest supplied repair status reports development complete and confirmation
unlocked/incomplete. The packet must preserve that distinction. Never substitute
files from `gait-fidelity-repair-validation-20260924`: those are software fixtures.

Two independent read-only reviews challenged scientific sufficiency and storage
cost. Their substantive findings are incorporated below.

## 1. First transfer: existing compact results and provenance

Exact relative-path allowlists are provided separately:

- [Walking core](paper-evidence-walking-core-files.txt)
- [JEPA response](paper-evidence-jepa-response-files.txt)
- [Readout repair](paper-evidence-readout-repair-files.txt)

Resolve each list against its own real work directory. Inventory existence and
byte sizes before transfer; the lists are not instructions to ignore missing
files. The same generic filenames in different work directories are different
artifacts and must not be flattened into one directory.

The [download helper](../scripts/download_paper_evidence.py) implements these
allowlists, adds successful response calibration/representation diagnostics,
and downloads into local `outputs/iclr/{walking-core,jepa-response,readout-repair}`.
Run from the local repository, where SSH can request the usual HAIC login:

```bash
.venv/bin/python docs/studies/gait-fidelity/scripts/download_paper_evidence.py
.venv/bin/python docs/studies/gait-fidelity/scripts/download_paper_evidence.py --apply
```

The first command inventories remote files without downloading their contents.
The second streams a compressed packet over one SSH connection, verifies SHA-256
hashes, and preserves a timestamped transfer inventory. No files are written on
HAIC. Defaults enforce 10 MiB per source file and 50 MiB total selected content.
Missing required files are reported; an applied partial packet exits with status
3. Oversized required files or a total over the cap block transfer. Existing
different local files are preserved and require a fresh `--output` directory.

For all three runs, retain `config.json`, `plan.json`, `frozen.json`, and
`ledger.json`. Together they record architecture/loss settings, declared method
and seed matrices, code/config hashes, completed attempts, actual update and
training-exposure counts, inherited checkpoints, and resource accounting.
Inspect the successful results in `ledger.completed`, rather than assuming
every file found under `attempts/` belongs to a final model. Profile jobs and
failed/retried fits are not additional experimental seeds.

The scheduler copies completed training report dictionaries into the ledger.
Thus a complete ledger generally replaces separate copies of every
`fit/receipt.json` or `result/receipt.json` for scientific interpretation. If
the ledger exceeds the transfer budget, export its successful completion
metadata and scalar training reports on HAIC, retaining original paths/hashes
and an attempt-status/accounting summary. Alternatively select the corresponding
small successful fit receipts. Do not collect both representations by default.

### Walking core

| Files | Purpose |
| --- | --- |
| `report.md`, `evaluation/summary.json`, `evaluation/comparisons.json` | Original result framing, actual training budget, primary effects and uncertainty |
| `evaluation/per-person.csv` | Recompute paired person/seed comparisons for all methods, including simple baselines |
| `evaluation/by-condition-person.csv` | Endpoint metrics by extractor, camera, naming and observation condition; includes source/motion context where recorded |
| `evaluation/coverage.csv`, `evaluation/coverage-per-person.csv` | Reference eligibility and prediction success, including losses of support |
| `evaluation/calibration.json` | Parameters, support and fitting rules for simple training-only coordinate corrections |
| `evaluation/complete.json` | Published artifact identities and configuration binding |
| `data/admission.json`, `data/mask-audit.json`, `cohort/summary.json` | Prepared-data counts, mask matching diagnostics, and planned cohort capacity |

Also retain `evaluation/coverage-by-source-motion.csv` if present; it provides a
compact dataset/motion coverage breakdown. Planned counts in `cohort/summary.json`
must not replace actual post-screening population counts.

### JEPA response

| Files | Purpose |
| --- | --- |
| `report.md`, `evaluation/summary.json`, `evaluation/comparisons.json` | Response experiment and retained parent predictions, primary effects and uncertainty |
| `evaluation/per-person.csv` | All response and imported control methods; response failure decomposition and zero-response baseline where recorded |
| `evaluation/readout-control-comparisons.json` | Base versus paired-change readouts, matched controls and coupling/readout interactions |
| `evaluation/response-by-condition-person.csv` | Person/seed response and endpoint metrics stratified by extractor, camera, naming, obstruction, physical state and held intervention |
| `evaluation/response-curves-person.csv` | Person-level descriptive response curves, intervention levels and prediction coverage |
| `evaluation/coverage.csv`, `evaluation/coverage-per-person.csv` | Endpoint and response-pair denominators and failures |
| `evaluation/complete.json` | Published result identities |

`response-summary.json`, `response-comparisons.json`, and `response-report.md`
duplicate the canonical summary/comparisons/report files; do not copy both
versions. `response-no-change-person.csv` is a subset of the curve file.
`parent-binding.json` is useful as a standalone provenance record, but the same
binding is embedded in this run's full `config.json`; it is not required twice.

The curve file averages seeds and pools extractors/naming. It cannot support
ViTPose-only response curves or training-seed variability claims. Use the
condition-person table for the preserved extractor/seed strata.

### Readout repair

| Files | Purpose |
| --- | --- |
| `development/evaluation/report.md`, `summary.json`, `comparisons.json` | Fixed dense-versus-low-scalar comparison, response tradeoff, seed/person effects, original scalar/base/direct comparisons |
| `development/evaluation/per-person.csv` | Secondary pooled-extractor results |
| `development/evaluation/per-person-by-extractor.csv` | Essential ViTPose primary data and both response/waveform failure decompositions |
| `development/evaluation/coverage.csv`, `coverage-per-person.csv` | Retained measurement support and endpoint/response failures |
| `development/evaluation/complete.json`, `development-complete.json` | Publication/completion records, prediction and checkpoint identities |
| `exposure-audit.json` | Documents the unresolved independent-confirmation status |

`development/evaluation/power-sensitivity.json` is optional planning evidence,
not an observed-performance result. `cohort/plan.json` is useful for describing
the proposed confirmation population, but it does not establish an evaluated N.

## 2. Small control and representation evidence for the paper

Resolve dynamic paths through the successful entries in each run's ledger, not
the newest-looking attempt directory. These are additional small artifacts to
inventory, not an instruction to copy entire attempt directories.

| Run and relative pattern | Reason |
| --- | --- |
| Response: `attempts/followup-profile/<successful-attempt>/profile/calibration/calibration.json` | Actual auxiliary coefficients and training-only calibration evidence |
| Repair: `attempts/calibrate-<variant>-<seed>/<successful-attempt>/result/calibration.json` | Six receipts supporting initial gradient matching, coefficients and shared initial head/encoder identities |
| Response: `attempts/followup-diagnostics/<successful-attempt>/diagnostics/diagnostics-summary.json` | Inventory of the completed diagnostic exports |
| Response: the individual `diagnostics/<parent-or-child-phase>/diagnostics.json` files referenced by that summary | Representation statistics and person-grouped probes needed to assess collapse/sensitivity hypotheses |

Repair calibration result dictionaries are also embedded in the ledger. The six
standalone calibration JSONs are optional file-level integrity evidence; omit
them in the minimal packet when the complete ledger is retained. The retained
software fixture was checked independently: all 39 core, 27 response and 12 repair
fit reports exactly match their ledger result dictionaries. This validates the
storage choice, not the results of the real experiments.

The diagnostic summary alone omits the detailed statistics/probes. Copy the
small per-representation JSONs, not `features.npz`, other feature arrays, or
entire diagnostic folders. Probe results are training-person feasibility
diagnostics, not independent held-out evidence. Initial gradient matching does
not establish matching throughout optimization or a unique sparsity mechanism.

Retain `gait-fidelity-release.json` from each distinct `code_root` if available
and small; it identifies the deployed source package. Local working files must
not silently substitute for an older run's frozen code.

The AMASS/GAVD manifests already available in the local checkout need not be
copied again when their hashes match the authorities used on HAIC. They describe
inventory and identity/split rules, not which observations the completed models
actually used.

## 3. Compact derived exports for stronger cross-run conclusions

These files **do not currently exist by these names**. Generate them on HAIC
only if making the corresponding claims; export summaries without modifying
the frozen experimental files.

1. **`paper-export/population-and-qc.json`**: actual unique canonical IDs per
   split, raw-motion/source-window counts, source datasets, exact extractor
   identities, condition levels, planned versus retained coverage and exclusion
   reasons. Include source manifest hashes and training/development identity
   intersections. The expanded bundle manifest remains on HAIC. Preparation
   shard receipts or a streaming metadata reader can supply the census; avoid
   loading a giant manifest on the login node merely to copy it.

2. **`paper-export/person-extractor-metrics.csv`**, for core/response where
   needed: method, seed, person, extractor, waveform/response error, respective
   failure rates and 180°/720° failure contributions, successful-output
   contributions and conditional error, zero-response baseline, and separate
   eligibility/success counts. Preserve conditions → source window → raw motion
   → person aggregation. The core condition-person table lacks response metrics;
   the response tables lack waveform failure decomposition. Derive missing
   values from existing `per-window.csv`/`responses.csv` on HAIC, joining source
   families to their motion hashes. Do not infer response-pair failures from
   endpoint counts. Retain held-intervention strata if claiming generalization
   to the held level. The existing repair extractor table already covers its
   fixed primary and decompositions.

3. **`paper-export/training-curves.csv`**, only for optimization/undertraining
   claims: loss components, update number, coefficients, gradient norms,
   clipping/support diagnostics, method and seed. Strip per-update endpoint,
   target and pair-index arrays from histories on HAIC. Never interpret a
   changed training loss scale as an improvement in restoration accuracy.

4. A few **matched waveform examples**, only if needed for a figure: deterministic
   or explicitly stratified selection of input/reference/restoration snippets
   shared by all shown methods, with timestamps and identifiers. This requires
   a small deliberate export; summary tables cannot reconstruct trajectories.
   Do not copy all predictions to obtain a handful of examples.

## 4. Transfer size and exclusions

Check actual HAIC bytes before copying. The real core `per-person.csv` already
copied locally is 150,336 bytes; this establishes the utility of compact person
tables, not the size of uninspected remote condition tables. Even tiny software
fixtures have 10–18 MB `per-window.csv` files, so fixture sizes cannot forecast
full-study transfer sizes.

A reasonable initial transfer policy is **at most 10 MiB per uncompressed file
and 50 MiB total selected uncompressed data**, with a size inventory first.
These are suggested limits, not measured sizes or experiment parameters. If an
essential condition table exceeds them, gzip the complete table or generate the
required complete aggregate on HAIC, then re-inventory the actual transfer
bytes. Do not truncate rows, omit failed methods, or select favorable people.
Report missing files and exclusions explicitly.

Exclude by default:

- Entire `evaluation/`, `attempts/`, `data/`, or diagnostic directories.
- Checkpoints (`.pt`/`.pth`), predictions, pose/feature arrays (`.npy`/`.npz`),
  videos, rendered frames and raw AMASS/GAVD data.
- `per-window.csv`, `responses.csv`, `nuisance.csv`, `interaction.csv` and full
  expanded bundle/cohort manifests.
- `history.json`, `batch-pair-cycles.json`, `training-pairs.npy`, unfiltered logs
  and detailed per-row sampling files. A `.json` extension does not imply small.
- Existing PNG plots unless convenient and small; publication figures can be
  regenerated from the numeric tables.

Include a transfer inventory with source run/path, byte size, SHA-256 and missing
or oversized status. Keep the large artifacts on HAIC. This is an analysis and
paper-evidence packet, not a self-contained rerun package; copied configs retain
remote paths and are not local launch configurations.

## 5. What these files can and cannot establish

- Core, response and repair development reuse people and inherited predictions.
  Do not add participant counts or call the stages independent replications.
- Preserve the original primary endpoints: the response experiment targets
  response error, while the repair targets waveform error on ViTPose. Pooled
  extractor means and held-extractor means are different comparisons.
- Compare common person/window populations, exact extractor identities,
  reference support and metric versions before constructing a unified table.
  Repeated deterministic baselines across seed rows are not extra trained seeds.
- A better mean can reflect fewer geometric failures, better successful outputs,
  or both. Person-level decompositions and coverage are necessary to distinguish
  these explanations.
- Current repair results are development evidence. A future confirmation packet
  would additionally require `confirmation/lock.json`, the reviewed ledger path
  bound by that lock, confirmation completion/coverage, and the same evaluation
  allowlist under `confirmation/evaluation/`. None can be replaced by a planned
  cohort or fixture result.
- These tests do not by themselves establish clinical accuracy, general JEPA
  superiority, or predictive world-model rollout capability. No significant
  advantage, equivalence margin, or acceptance probability follows from file
  collection alone.
