# Data audit for reference-aware gait modeling v2

The new cohort identity is `reference-v2-cohort-v1`. This audit rebuilds the complete annotation inventory, verifies actual source-video and raw-pose-cache hashes, and reconstructs every prepared window from its native sequence cache. It fits no model and does not inspect new forecasting outcomes.

The inventory has 1,874 sequences, 458,116 annotation rows, and 348 source IDs. The upstream source CSVs are not counted again. The five requested labels contain 666 sequences from 103 source IDs. Media eligibility retains 652 sequences from 98 sources, and the two preserved whole-source identity exclusions leave 641 extraction candidates from 96 sources. All 96 source-cache and actual original-video hashes match the frozen preparation. Fresh annotation hashes match every selected sequence.

All 2,818 prepared windows from 456 sequence caches reconstructed exactly, with maximum coordinate difference 0. The predeclared common distal-target support gate removes 3 windows, leaving 2,815 windows from 456 sequences and 91 source videos. The arrays are reused in place by hash and `prepared_index`; no media or tensors are duplicated. The original nominal 10 Hz, 16-prefix/8-future, confidence 0.5, separate 70% prefix/future support contract remains unchanged.

## Source relations and generalization

The grouping graph joins original source IDs only when original footage is byte-identical. There are no verified cross-source-ID byte-identical originals in the scanned local GAVD-related files. Therefore the evaluated cohort has 91 singleton components, and its seed 2701 five folds are identical to the historical source folds. This does not prove that all 91 sources show independent people. Mirrored, edited, or reencoded uploads and repeated participants can evade byte hashes. Source and component IDs remain proxies, not patient identifiers.

The complete annotation inventory has one mixed-label source, `wRntYsztIEY`, with `cerebral palsy` and `abnormal` annotations. It remains a single source component. Its cerebral-palsy sequences are already in the primary scope. Any broader-label transfer must exclude this shared source from its new-source test or retain it in the same training/evaluation component, with the mixed endpoint declared explicitly.

GAVD supports a heterogeneous public-video benchmark, but GAVD alone does not establish patient-disjoint, site-disjoint, or prospective clinical generalization. All 32 annotated normal source IDs already belong to the selected scope, with 29 in the final cohort. The other seven labels add coverage but no additional normal source support. Exercise, pregnancy, stylized walking, prosthetic use, and intoxication retain their actual annotation meanings. More overlapping windows do not create more independent patients.

## Frames, time, support, and view

Annotation frame numbers are one-based and decoded native indexes are zero-based. Reconstructed native mappings, source-window identities, decoder timestamps, and prefix-only normalization agree exactly with the frozen arrays. All retained windows have finite increasing actual timestamps; adjacent intervals range from 0.06667 to 0.13333 seconds. This is nearest-native-frame sampling on a nominal clock, not interpolation or exact physical-time resampling. Target coordinates describe pelvis-relative 2D articulation in prefix-torso units. They are not physical speed or measured 3D anatomy.

`data_attrition_by_label.csv` traces catalog, media, extraction candidates, prepared windows, and common support. `data_coverage_by_view.csv` preserves each literal view annotation and counts sources separately from windows. Future availability gates describe which outcomes can be measured retrospectively; they are not inputs to the forecasting model. The evaluation owner computes source-level prefix-motion and missingness covariates inside the declared model evaluation.

## Blinded visual audit

The audit samples 10 sequences from 10 sources, covering accepted and rejected sequences within each selected label. Three native frames per sequence provide 30 panels. The sheet conceals the strata, source IDs, and gate outcomes; the separate selection ledger retains those identities. This is a deterministic, outcome-independent sample, not comprehensive full-duration tracking or a new clinical-label review. The generated selection is recorded in `data_visual_qa.json`; the actual inspection is recorded separately in `data_visual_review.json`.

## Local external-data opportunity

The local KOA-PD-NM cache has 189 videos from 95 coded participants, including 30 normal, 15 Parkinson's, and 50 osteoarthritis participants. The bundled demographic PDF distinguishes participant numbering separately within each severity group, so subject keys include group and severity. The 189 videos under `data/videos-ms/koa-pd-nm` are exact byte copies of the chosen `data/videos-more/koa-pd-nm` originals; they must not be counted twice. No chosen external file is byte-identical to a GAVD original, but undetected reencoded overlap remains possible.

The descriptor lists 191 videos and 96 participants; local severe-Parkinson footage has 3 videos from 2 participants rather than 5 videos/3 participants. Participant `PD_SV_003` is absent and `PD_SV_002` has only direction 01. The original description's 31 PD videos also differs from the 32 implied by pairing all 16 demographic participants. These discrepancies are retained rather than filled by assumptions. `data_external_inventory.csv` records every existing video hash, clone relation, subject key, nominal FPS, image size, and first/last decode check.

The metadata audit found 187 videos with successful first/last decode checks. Invalid files remain explicitly listed in `data_external_summary.json`; their presence is not usable observation evidence. The external metadata audit performs no pose extraction or model fitting. A frozen video model could be evaluated through the same Ambient image API after a separate compatibility/extraction protocol. Any normal target-domain calibration is adaptation and needs a separate normal participant set. The local cache alone does not establish untouched historical exposure. A preimplementation text search found no dataset-name, DOI, or source-filename references in existing experiment code/reports and notes, but cannot establish what researchers previously viewed.

The 656 sequence clips in `data/videos-ms/gavd-sequences` are explicitly traced to the same GAVD source IDs and annotation frames by their own manifest. They are derived observations of this dataset, not external validation. Turning-in-place video contains a distinct PD task with no normal group, and WeightGait is a controlled healthy-participant perturbation dataset. Their native formats and labels require separate contracts before any clinical comparison.

## Reproduce and consume

From the repository root, run the experiment Python 3.12 environment:

```bash
experiments/aaai-worldmodels/.venv/bin/python experiments/aaai-worldmodels/scripts/audit_v2_data.py
experiments/aaai-worldmodels/.venv/bin/python -m pytest -c experiments/aaai-worldmodels/pyproject.toml experiments/aaai-worldmodels/tests/test_v2_data.py -q
```

The first command creates the declared cohort only if absent, then regenerates descriptive reports and review material. Completed cohort manifests are immutable; use a new output identity to change eligibility, grouping, or preparation. Consumers call `worldmodels.v2_data.load_cohort()` to obtain `(x, valid, metadata, times)`. The metadata keeps `prepared_index`, `component_id`, and `outer_fold`. The new identity pins the old array hashes, annotation snapshot, actual source hashes, pose-weight hash, code revision, and audit implementation. No paid provider, model download, new media download, or pose-model inference is required by this audit.
