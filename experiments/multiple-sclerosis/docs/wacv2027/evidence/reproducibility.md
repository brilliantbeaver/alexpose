# Reproducibility record

The revised manuscript describes the experiment in the seven top-level notebooks. Its primary numerical evidence is the retained September capstone matching notebook 06, with the later corrected-RF execution reported separately. The supplied PDF's conflicting numerical claims were corrected from these records. This document distinguishes reproducing saved-prediction analysis from recreating the original data collection or neural training.

## Evidence hierarchy and run identities

1. **Top-level notebooks define the intended study.** Notebooks 00–05 contain saved executed cells. Notebook 06 has seven code cells with no execution counts or saved outputs; its numerical Markdown reports the historical full-v1 experiment. It creates a fresh output directory when executed and does not load a newest result implicitly.
2. **The matching retained artifacts verify notebook claims.** The primary run is `artifacts/runs/full-v1/9496e61b050f/laptop-1d09c8e1eea5/capstone-20260921T011305448893Z/`. Its elapsed time is 215.981 seconds, and its source-weighted macro-F1 is 0.456624 for S-JEPA, 0.451893 for mean pose and 0.411263 for the historically defective RF. These match the notebook's rounded description.
3. **Added analyses are labeled as reanalysis.** Metric recalculation, paired source-resampling intervals, one-column RF verification, the original-draft forensic checks, and additional mask-bank calculations were performed during manuscript preparation. They are not saved notebook outputs or additional independent clinical observations.
4. **The corrected execution is separate.** `artifacts/runs/full-v1/9496e61b050f/laptop-1d09c8e1eea5/capstone-20261009T060955236580Z/` supplies the corrected RF result (source-weighted macro-F1 0.325184). Its UTC timestamp is October 8, 2026 at 23:09:55 in Los Angeles. It does not supersede the notebook-primary table without explanation.
5. **Older and standalone runs remain distinct.** The `g1`/R1 evaluation has 47 clips and different experimental settings. Standalone Fold-0 models and training visualizations from notebooks 03–05 are not the same experimental objects as the five capstone folds. None are pooled to create extra replicates.

The JSON audit retains its initial `primary_metrics` name for compatibility, which points to the latest corrected execution. For the manuscript, use the explicit **`notebook_primary_metrics`** and **`corrected_verification_metrics`** fields. The corresponding run fields are also explicit. `historical_metrics` is an alias for the notebook-primary metrics.

## Recompute the saved-prediction audit

Run from the experiment root, the directory containing the seven top-level notebooks:

```sh
.venv/bin/python docs/wacv2007/evidence/results_audit.py
```

This command reads the existing registry, caches, OOF rows, provenance and manifests. It independently calculates weighted confusion matrices, scalar metrics, per-class scores, fold summaries, and paired source-resampling intervals. It checks complete and correctly attributed OOF coverage, hashes every artifact referenced by fold manifests, compares the two saved runs, and refits the small RF baseline for the feature-correction check. It writes only the audit reports under `docs/wacv2007/evidence`; it does not extract poses or train neural models.

The outputs are `results-audit.json`, `results-audit.md`, and `results-numerical-consistency.md`. The primary manuscript bootstrap settings are in `original_draft_audit.historical_saved_draft_seed_bootstrap`: **10,000 draws, seed 20261004**, sorted source IDs, label-stratified counts 16 Normal/13 MS/12 PD, identical source draws for all methods, and weight one for every sampled source occurrence. The script also retains a separate 20,000-draw seed-20260920 sensitivity analysis under `bootstrap`, referring to the corrected execution. Do not mix its intervals with the manuscript's 10,000-draw estimates.

For the primary notebook predictions, the manuscript-setting bootstrap gives:

| Quantity | Point | Conditional 95% percentile interval |
|---|---:|---:|
| S-JEPA source-weighted macro-F1 | 0.456624159 | [0.328714413, 0.578793142] |
| S-JEPA minus mean pose | 0.004731283 | [−0.160496452, 0.173998638] |
| S-JEPA minus historical RF | 0.045361249 | [−0.133446835, 0.220113928] |

These are descriptive sensitivity intervals conditional on fixed OOF models and predictions, class-source counts and split. They do not include retraining, stage selection, seed, split or development uncertainty. Unknown participant overlap can violate independent-source resampling assumptions, and overlapping training sets induce model dependence. No population superiority or equivalence test is claimed.

The RF check uses the original fixed hyperparameters and partitions. Current extraction reproduces all 88 corrected predictions; replacing only `left_ankle_range` with `right_ankle_range` reproduces all 88 historical predictions. This operation occurs in an in-memory feature matrix. It does not modify the extractor, cached poses, saved predictions, or checkpoints.

The forensic search likewise operates on in-memory copies of historical predictions. It enumerates 176 possible one-clip prediction changes to diagnose the supplied PDF. The unique matching counterfactual is recorded as a diagnostic, not exported as experimental data. The actual saved-prediction results are always reported as the study evidence.

## Figures and paper builds

The revised figures are generated with:

```sh
.venv/bin/python docs/wacv2007/scripts/figure_pipeline_v5.py
.venv/bin/python docs/wacv2007/scripts/figure_results_v5.py
```

The pipeline script writes `figures/pipeline-v5.pdf` and a text-containment check under `qa`. Its training box uses the accessible term “label classifiers.” The results script explicitly reads the notebook-associated historical OOF file and independently forms source-weighted fold F1 and row-normalized confusion matrices. It writes `figures/folds-and-errors-v5.pdf`. Display indices **0–4 match the registry and partition table**. Straight horizontal connectors pair S-JEPA and mean pose within each held-out fold; they do not connect different folds or indicate a temporal trajectory.

The earlier v3 figures and their generator, `scripts/make_figures.py`, remain archived for that revision. Its older figure displays registry folds 0–4 as labels 1–5 and connects each method across fold indices. The v4 pipeline is also retained with `scripts/figure_pipeline_v4.py`. Use the figure filename referenced by the particular retained manuscript when rebuilding an earlier revision; do not silently replace an archived visual with a later layout.

A retained revision can be rebuilt with its version number; for example:

```sh
python3 docs/wacv2007/scripts/build_papers.py 3
```

The build script uses `tectonic`, `pdftotext`, and `pdftoppm`, with the retained official WACV 2027 style and bibliography style. It snapshots the bibliography as `references-vN.bib` if that version's snapshot does not yet exist, and normalizes the manuscript's `\bibliography{...}` line to that version-specific filename. This is a small source normalization during the first build; an existing snapshot is preserved on later builds. It writes the PDF plus build logs, extracted text and page images under `docs/wacv2007/qa`, and leaves the original supplied v1 PDF untouched when building the revised versions. The build tools and TeX package availability must be installed separately; the Python environment alone does not supply them.

The reference audit can be rebuilt from retained metadata without initiating new literature searches:

```sh
.venv/bin/python docs/wacv2007/evidence/build_reference_audit.py
```

`audit_references.py` instead queries live Crossref metadata and is an optional refresh step, not required to regenerate the retained audit. Candidate matches were checked by the research agents against primary sources before acceptance; no human review is claimed. Rerunning a search does not itself verify a citation or its associated scientific claim.

The deterministic mask/readout calculations are documented in `notebook-audit.md` and stored in `notebook-audit-checks.json`. They use `sjepa.masking_v2.sample_target_mask`: a separate seed-0 bank of 5,000 masks for the expanded diagnostic, and the first seed-0 mask for the 161-position readout check. The notebook's own seed-0 bank has 512 masks. These bank sizes are distinct and neither reconstructs the actual masks used throughout saved model training.

## Software and computation

The historical capstone provenance records the following environment. The audit and corrected execution have the same listed Python/scientific-package versions; their platform string is macOS 27.0.1 instead of historical macOS 26.6.2.

| Component | Recorded version or setting |
|---|---|
| Python | 3.12.13 |
| NumPy | 2.4.6 |
| PyTorch | 2.13.0 |
| scikit-learn | 1.9.0 |
| SciPy | 1.18.0 |
| joblib | 1.5.3 |
| Historical platform | macOS-26.6.2-arm64-arm-64bit |
| Corrected/audit platform | macOS-27.0.1-arm64-arm-64bit |
| Neural device | mps |
| Default neural dtype | torch.float32 |
| Float32 matmul precision | highest |
| CPU threads in capstone provenance | 1 |
| Deterministic algorithms enabled | false |
| cuDNN benchmark / deterministic flags | false / false |

Additional packages observed in the audit environment are Matplotlib 3.11.1, OpenCV (`opencv-python`) 5.0.0.93, MediaPipe 1.0.0, UMAP 0.5.12, threadpoolctl 3.6.0 and filelock 3.32.2. These are **audit-environment observations**; the capstone provenance does not prove that every pose cache was originally produced with those exact extraction-library versions.

The historical and corrected runs both use seed 42, the same data, the same configuration, and the same source partitions. All S-JEPA hard predictions coincide, but checkpoint bytes and embedding arrays differ. Both report computed folds. They therefore provide same-seed repeat execution evidence, not independent-seed, independent-person or independent-cohort replication. Byte-identical neural retraining is not promised on this nondeterministic MPS configuration.

The current pipeline source supplies the inspected training methods, but the older all-code provenance is an aggregate hash rather than a complete archived source snapshot. The historical ankle defect is independently reproduced by the one-column verification; this should not be generalized into a claim that every historical execution detail can be recovered from current code alone.

## Input identities and hashes

The registry's embedded `registry_sha256` is a canonical JSON-content digest, not the SHA-256 of the pretty-printed file. It is `9496e61b050fc8b62ccbc59cb737376396205a3356b323733b9b8915d56600e1`. The embedded dataset inventory digest is `93e3ed716d608f503b77381692f833e8b027b85d9aa6d3580fae70eebb30e75b`. Actual file hashes are listed below. Every one of the 88 pose-cache file hashes is retained in REG `inventory.cache` and both run provenance `inputs`; the audit verifies the current bytes against this frozen inventory.

| Input file | SHA-256 of file bytes |
|---|---|
| `artifacts/eval/full-v1/fold_registry.json` | `4fbb7a63ec7072048979bb1bf6b1d14df3339469cd8d4e17462d352e770cc4df` |
| `artifacts/eval/full-v1/exclusions.json` | `5cdf75d825eef8592fc2dc00439d44ee1cea411339125064f4a539f44049bddf` |
| `docs/wacv2007/wacv2007-v1.pdf` | `d96bfb92a5ea2fb041b8c2d05307e8f7bd6f79b3f4cf82eebc51b677d47326fa` |
| HIST `oof.json` | `70d9b62a8e1d949f4910921f409abd974529e70d059d71f6b10875e77d0438f5` |
| HIST `results.json` | `1e3b02e3b6ed4733e225de59cb770382f04a49e390d5bdd5e3d4376ac693e6ca` |
| HIST `provenance.json` | `e821dc51ee7dfc93b1cee919d91619e8aa11d34b693fb75bf5bcd02ba2135381` |
| CORR `oof.json` | `8ea21a90ea72e30346cd7063e6e55a973cbe8ae9a0503c755fb9cbcf918c9de0` |
| CORR `results.json` | `c4a5567e03335c3927a22c824c20c93b8d45404542a55f079d2c32bacf95dffe` |
| CORR `provenance.json` | `b371d90040aa19b88ba05323abbe6587d79089b113e2390573f8fb09cba0a19f` |

The notebooks were audited at these file identities. Cell numbers and execution-status claims in the ledger refer to these versions.

| Notebook | SHA-256 | Code cells with saved execution counts / total |
|---|---|---:|
| `00_overview_and_video_gallery.ipynb` | `dc5e657887d09abf5cb1160c0f2d21d0ce6cbff5b221373e98cd1189bec5afaa` | 6 / 6 |
| `01_pose_extraction_from_raw_video.ipynb` | `8e6545638a9976b1372a925ffa7c4f3fc27e90f0f92b09400da3c91cd2adb6ab` | 9 / 9 |
| `02_anatomical_mask_and_tokenization.ipynb` | `2c6c5c7a2c5bc776fa3e85f654e53c962779c3e2f44ab181be527583251d34d1` | 10 / 10 |
| `03_sjepa_model_and_pretrain_normal.ipynb` | `8907eebfa53c8cd223be9937cd40fbfcc8ebb021103cebe65aa1d66b88c4a48a` | 8 / 8 |
| `04_progressive_finetune_ms_pd_vicreg.ipynb` | `7002adc86615299a121a152719b9d0cc0221193400e76f2b7b642402516f83f2` | 7 / 7 |
| `05_representation_visualization.ipynb` | `a0f6f8f7d9b4522b8eae0e5044932c4481db03f3ddafc33c87bf15464d8b1d47` | 8 / 8 |
| `06_capstone_rf_vs_sjepa.ipynb` | `5547d761a220f990ff8be7a79819a9b873e91f74f50c61b9ac05351441554c4c` | 0 / 7 |

A small read-only check for the report-recorded inputs is:

```sh
.venv/bin/python - <<'PYCHECK'
import hashlib, json
from pathlib import Path
report = json.loads(Path('docs/wacv2007/evidence/results-audit.json').read_text())
for filename, expected in report['hashes'].items():
    actual = hashlib.sha256(Path(filename).read_bytes()).hexdigest()
    assert actual == expected, filename
print('Recorded audit input hashes match.')
PYCHECK
```

The full audit additionally checks the registry's internal digest and the manifest-linked checkpoint/embedding hashes. Hashes establish byte identity to retained files; they do not independently authenticate diagnoses, participant identities, original capture times, acquisition conditions, or source-media rights.

## Preservation and limits

The manuscript audit preserved the supplied v1 PDF, all top-level notebooks, raw videos, pose caches, OOF files, model checkpoints, split registry and original result files. The RF extractor correction was already present in the working tree when audited; this manuscript work did not introduce or silently rewrite it. New documents, analyses, figures and review outputs were written under `docs/wacv2007`. Re-running the audit updates its derived reports and audit timestamp, not the original study artifacts.

No new pose extraction or neural model training was performed for this manuscript audit. The additional computation was fixed-prediction analysis, deterministic mask diagnostics and the explicitly described small RF verification refits. Previously saved corrected-run evidence remains separate from newly generated paper analysis.

The cache audit does not reauthenticate raw video bytes or reconstruct original per-frame timestamps and detection-gap masks. It cannot establish participant-level independence, clinical adjudication, consent for secondary health inference, source-video redistribution permissions, or an ethics-review determination. Those facts are absent from the reviewed experiment records. The paper accordingly reports collection-label prediction with source grouping, and leaves clinical validation and causal temporal-benefit experiments unrun.

For claim-specific provenance and the exact corrections made to the supplied draft, see [claim-ledger.md](claim-ledger.md), [results-numerical-consistency.md](results-numerical-consistency.md), [notebook-audit.md](notebook-audit.md), and [reference-audit.md](reference-audit.md).
