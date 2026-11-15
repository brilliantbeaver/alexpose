# Verification record

> Historical foundation record. This document describes the earlier five-condition experiment. The revised twelve-folder study is documented in [the current protocol](context_study_protocol.md), [real results](context_study_results.md), and [critical review](context_study_review.md). Its manuscript is in [draft01](draft01/README.md).

This record distinguishes runnable implementation checks from real research measurements. It was created on November 14, 2026 and is updated after the full extraction audit.

## Environment and authored materials

The experiment's own Python 3.12 environment was resolved and installed with `uv sync`. It imports the repository `ambient` package in editable mode. The authored deliverables include eight Jupyter notebooks, 14 SVG figures, a 16-slide local HTML deck, its Markdown source, and a browser-generated PDF export.

Each notebook was executed in a fresh kernel using the experiment interpreter. The validation runner forces synthetic mode and writes executed copies and timing records under `artifacts/notebook_checks/`. The first and second complete passes succeeded for all eight notebooks. A further complete pass covered the missing-token revisions and final source-specific media links, with all eight notebooks passing again. Synthetic plots and scores are software evidence only.

The model, data, and orchestration tests cover current annotation hashes, source-group splitting, whitelist restriction, uniform mask frequencies, hidden/future-coordinate noninterference, stopped teacher gradients, EMA timing, invalid-key and invalid-target exclusion, deterministic inference, train-only preprocessing, preservation of the normal anchor, paired prediction ordering, and finite-sample anomaly calibration. **42 tests passed** in the experiment environment after the validity and forecasting revisions. A two-worker real-media smoke check also verified process-isolated extraction before the full four-worker run.

## Visual inspection

All 14 SVGs were rendered and independently inspected. A browser geometry check found no SVG text outside the 1200-by-600 canvas and no horizontal slide overflow. A contact sheet and selected slide renders were inspected for readability, crossing arrows, and overlap. The diagram labels were corrected where the first draft overstated what VICReg guarantees or described the wrong calibration rule. Independent components no longer have misleading sequential arrows.

Five training-source bounding-box overlays were also inspected. Each displayed box enclosed the intended visible person in that sampled frame. This spot check does not establish full-timeline alignment or clinical label validity. The poolside cerebral-palsy example has a small person and remains difficult for pose estimation; no semantic exclusion was inferred from the snapshot.

The HTML deck uses local assets and needs no CDN. It was printed to `slides/tutorial.pdf` using Chromium. Browser inspection supports layout validity; it is not a claim that the deck was opened in PowerPoint.

## Real media and pilot learning

The current cohort contains 666 annotated sequences from 103 source IDs. A dated retrieval pass made 99 of the 103 sources locally decodable, including 81 newly downloaded recordings. Four sources remained missing or undecodable. New media was written only beneath `data/videos-gavd/`.

The original eight-sequence extraction pilot rejected all windows under an all-joints-complete rule. The revised validity-aware protocol was applied to the same poses and retained 12 windows from six sequences. No visibility threshold was loosened. A token is valid only when all four frames of its joint are observed. Windows need 75% valid tokens overall and 50% in each half. Missing values are excluded from attention, pooling, target centering, and prediction loss.

A normal-only optimization check on the seven retained normal pilot windows produced finite loss and gradients. All 12 pilot windows produced finite anomaly scores, and invalid tokens had zero scoring coverage. This establishes technical interoperability between real ambient poses and the model. It does not establish held-out classification accuracy, clinical usefulness, or five-class coverage.

## Full extraction

The complete 666-sequence manifest was processed with four isolated MediaPipe workers. Of these, 654 sequences had decodable source media, including two verified caches, and 12 referenced unavailable sources. The fixed validity rule retained 2,504 windows from 450 sequences and 91 sources. All five labels remain in all three partitions for seed 42. Its Parkinson's test class has only one source, and stroke validation has only one source. These limitations materially weaken the precision and stability of per-class claims.

## Review and fixed issues

The [adversarial review](adversarial_review.md) records the detailed findings and their resolution. Important repairs include prefix-only scale estimation, exclusion of upsampled sources to preserve raw-time forecasting boundaries, source-aligned pilot downloading, shared audit/training configuration, separate positional-noise random streams, corpus hashes that include pose provenance, normal-anchor identity in adaptation caches, training-only source-aware scaling, exact-source links for explanations, and source-cluster intervals for anomaly ranking.

Existing unrelated changes in `notes/gavd/` and `notes/aaai/` were preserved. No paper submission, publication, or message to another person was performed.

## Remaining scientific limits

A working implementation and a finite training loss do not establish representation quality. The source-held-out cohort can still contain duplicate participants or recording shortcuts. GAVD labels and public-video annotations require semantic review. The validity and camera-view baseline can expose some confounding but cannot prove its absence. Finite-sample 5% calibration remains unsupported with only six normal validation sources. Full data results, when produced, must remain explicitly exploratory until reviewed and replicated across prespecified splits and an independent cohort.

## Actual training and inference runs

The real cohort was trained and scored with seeds 42, 43, and 44 using three epochs per normal/progressive stage. Seven classification methods, masked and future predictive surprise, and the prespecified normal-embedding-distance baseline were evaluated. The seed-42 positional-noise ablation was also executed. See [results](results.md) for measured values and their limitations. The results do not support a claim that the short JEPA training improves on the untrained encoder.

Full extraction actually used the repository Python 3.14.6 environment with MediaPipe 0.10.31 and NumPy 2.2.6, recorded in `artifacts/extraction_environment.json`. Model training and notebook validation used the experiment Python 3.12 environment. Per-run package versions and pose-file hashes preserve this distinction; the recommended environment must not be described as the environment that produced the original pose caches.

All eight notebooks also passed independent-kernel execution on the actual retained GAVD corpus. A held-out sequence passed the video-to-pose-cache-to-classifier path and matched the saved capstone probabilities within 1e-6. The inference output correctly marked it as absent from training. The forecast notebook was then checked again after separating its output path from the richer CLI evaluation report.

The prepared workspace root `.env` now selects `AAAI_DEMO=0`, so normal notebook launches use the verified real corpus. The validator explicitly sets its requested mode, and `smoke` explicitly uses synthetic data. New installations without that local setting still default to synthetic mode. Existing API keys and unrelated environment values were preserved.
