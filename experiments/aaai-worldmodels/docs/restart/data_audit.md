# GAVD audit for the research restart

This audit re-read the current annotation CSVs, hashed every local source video referenced by the five selected labels, and checked the existing prepared arrays against their recorded fingerprints. It did not use a prior notebook's summary as its inventory. Machine-readable counts and input identities are in [data_audit.json](data_audit.json). The read-only audit script is saved locally under `artifacts/restart_audit/audit.py`.

## Annotation and media accounting

| Label | Annotated sequences | Annotated source IDs | Candidate sequences after existing exclusions | Candidate source IDs |
|---|---:|---:|---:|---:|
| Normal | 291 | 32 | 287 | 31 |
| Parkinsons | 47 | 11 | 47 | 11 |
| Stroke | 76 | 19 | 64 | 16 |
| Cerebral palsy | 64 | 11 | 60 | 10 |
| Myopathic | 188 | 30 | 183 | 28 |
| Total | 666 | 103 | 641 | 96 |

The selected annotations contain 140,641 rows. The media inventory first retains 652 sequences from 98 sources. Nine sequences lack a local video, four have the established video-reference mismatch, and one annotation exceeds the available source duration. The two established whole-source identity exclusions then remove eleven stroke sequences. This leaves 641 candidate sequences, 96 sources and 135,633 annotation rows. No exclusion was changed after inspecting new model results.

The candidate sequences cover 132,911 unique source frames. Seventeen overlapping sequence pairs share 2,722 annotated source frames, concentrated in `R8LRCiTvUz8`, `jzJIpY6vRLo` and `v1SoZ_S31pk`. These observations are related and must remain in the same source group. Exact duplicate source/frame-tuple windows are removed by the restart adapter; partly overlapping observations remain dependent.

All 100 locally available original video files matched their recorded media-manifest SHA-256 fingerprints. No two different source IDs had identical complete video-file bytes. That check cannot detect recoded reposts, edited excerpts or a participant appearing in different source IDs. Person identity remains unresolved beyond the available annotation and visual checks.

## What the previous experiment actually covered

| Prepared cohort | Windows | Source IDs | Temporal coverage |
|---|---:|---:|---|
| `pilot-input-v2` | 494 | 96 | At most the first 16 stride-two sampled frames per sequence. |
| `pilot-input-clean-v1` | 405 | 89 | The same bounded temporal prefix with person gating and source exclusions. |
| `extended-input-v1` | 414 | 93 | At most the first 32 stride-two sampled frames per sequence. |
| `extended-input-clean-v1` | 407 | 91 | The same bounded temporal prefix after the source exclusions. |

The existing extended raw cache contains 595 files, each with exactly 32 observations. All checked CSV identities, current video identities and annotated frame sequences matched. After the two manual exclusions, these caches cover 18,569 unique sampled source frames. Sampling every second annotated native frame across the complete candidate sequences would require 67,125 unique source frames, so the former cache covers less than one third of that full-duration sampling opportunity. The restart uses its separately declared nominal 10 Hz grid rather than treating these different frame sets as equivalent.

The historical clean forecasting cohort has 29 normal, 11 parkinsons, 16 stroke, eight cerebral-palsy and 27 myopathic sources. Its final horizon ranges from approximately 1.067 to 2.133 seconds because it fixes a native-frame count instead of elapsed time. All four inspected prepared cohorts have source-disjoint recorded partitions, exact artifact-hash agreement, finite normalized arrays and no disagreement with the original source partition assignment.

The earlier test outcomes have already been inspected. Changing the model or assigning new folds cannot create an untouched test population. New cross-validation on these sources is an exploratory reanalysis. It can reduce dependence on a small historical split, but it does not remove prior researcher exposure to these data.

## Full GAVD expansion

The complete annotation collection has 1,874 sequences, 458,116 rows, 12 labels and 348 source IDs. The five requested labels already include every source annotated as normal. The other seven labels add 1,208 sequences and 245 previously unselected source IDs.

| Additional label | Sequences | Source IDs | New source IDs beyond the five selected labels |
|---|---:|---:|---:|
| Abnormal | 767 | 117 | 116 |
| Antalgic | 35 | 10 | 10 |
| Exercise | 234 | 98 | 98 |
| Inebriated | 23 | 8 | 8 |
| Pregnant | 6 | 2 | 2 |
| Prosthetic | 39 | 8 | 8 |
| Style | 104 | 3 | 3 |

One source, `wRntYsztIEY`, has both cerebral-palsy and abnormal annotations. It must remain in a single group if categories are expanded. No filename-matched video for any of the 245 additional source IDs was found under `data/videos-gavd`, `data/videos-more`, `data/videos-ms`, `data/cache`, `data/youtube` or the root cache. This is a local availability audit, not an internet availability test. The detailed evidence is saved under `artifacts/restart_audit/full_gavd_availability.json`.

Additional labels can support a separately specified condition-shift or acquisition-shift assessment. They cannot be relabeled as the same disease endpoint, and exercise, pregnancy or style labels do not automatically denote disease. Adding them does not increase the number of available normal source IDs. Any expansion should be fixed before fitting and reporting its outcomes, with explicit label eligibility, source grouping and stopping rules. Sampling until a significance threshold is crossed would bias the evidence.

## Recommended cohort and generalizability claim

The largest presently supported candidate cohort for the requested labels is the 641 sequences from 96 source IDs after the established exclusions. Full-duration pose extraction can use more movement from these sources and can recover sequences whose initial frames were uninformative. It cannot establish more independent participants. Quality filtering must be reported after extraction, without replacing failed sources or adjusting thresholds to improve model results.

The fresh adapter and its output contract are documented in [data_contract.md](data_contract.md). It uses shared Ambient pose inference, nominal 10 Hz native-frame sampling, 24-observation windows with 16 context observations, conservative annotation-box gating, separate prefix/future quality thresholds, prefix-only scale estimation and explicit common target support. The first real source pilot made 293 MediaPipe calls in 5.84 seconds and retained eleven windows from six sequences. Four focused contract tests passed before full extraction began. Final full-run counts are recorded separately by the completed run's `preparation.json`.

GAVD supports a heterogeneous observational feasibility benchmark. Its source count and unknown participant identity do not satisfy patient-level, site-level or clinical deployment generalizability requirements. Labels are dataset annotations rather than independent diagnostic verification. Age, disease severity, treatment, assistive devices, demonstrations, camera perspective and source editing may differ systematically between categories. Neither a large frame count nor many overlapping windows resolves those limitations. A stronger manuscript needs independently collected participants and recording settings, patient identifiers, verified diagnoses, and external evaluation of forecast and calibration behavior.

## Completed full-duration extraction

The frozen `restart-full10hz-v1` run made **46,783 actual MediaPipe model calls** across all 96 candidate sources in 990.00 seconds. It retained **2,818 windows from 456 sequences and 91 source IDs**. No new source-video download was needed, and no earlier pose cache was reused in this run.

| Label | Retained windows | Retained sequences | Retained source IDs |
|---|---:|---:|---:|
| cerebral palsy | 518 | 52 | 8 |
| myopathic | 845 | 168 | 27 |
| normal | 465 | 141 | 29 |
| parkinsons | 293 | 37 | 11 |
| stroke | 697 | 58 | 16 |

There were 4,361 duration-eligible candidate windows before quality filtering and deduplication. 64 sequences had no complete 24-observation window. The run records every rejection and gate outcome in its attrition and person-gate tables. The maximum retained-window deviation from the nominal 10 Hz grid was 0.0412 seconds, and the final forecast horizon ranged from 0.7674 to 0.8342 seconds.

The complete replay reconstructed all 2,818 retained windows from their saved native-frame pose records with **zero array difference**. It checked every target-support mask, source partition, frame tuple, timestamp, window ID, source-pose payload hash, artifact hash and attrition count. Five focused contract tests passed. The verifier reuses the shared normalization API and therefore establishes artifact consistency, not independent clinical validation. The exact running extraction source is preserved under the run's `source_snapshot/` directory.

The prepared arrays and metadata are ready for a separately identified source-grouped model study. This expanded observation cohort retains the generalizability limits described above. More dependent windows do not establish more independent patients, and source groups do not establish verified subject identity.
