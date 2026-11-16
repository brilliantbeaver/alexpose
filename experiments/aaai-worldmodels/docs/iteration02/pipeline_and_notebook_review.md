# Review of the actual ingestion and notebook evidence

This review informed the next-iteration instruction. It inspects the current code and saved execution records without refitting a forecasting model or changing a completed cohort. The checkout is `/Users/alexmui/dev/alexpose`; a different home-directory path in the research brief is not an additional data source.

## The current pipeline

`worldmodels/data.py::build_inventory` reads per-condition sequence CSVs, preserves source and sequence identities, checks the schema and positive one-based frame numbers, and excludes the upstream `source/` copy from counts. It audits all labels but returns the five selected labels for this study. Media eligibility uses existing source files and manifest information. It is not itself a fresh pose-quality or complete source-content check.

`worldmodels/pose.py::download_sources` deduplicates source IDs and delegates to `ambient.utils.youtube_cache.download_youtube_urls`, explicitly setting `data/videos-gavd/sources`. `find_video` reuses existing originals and detects conflicting copies by hash. Full-inventory expansion needs an explicit label/lookup extension because these study adapters currently select five categories. No replacement downloader is needed.

`worldmodels/restart_data.py::_extract_source` sequentially decodes selected sources with OpenCV. It converts decoded index to one-based annotation number once and calls Ambient's RGB image API, `SequenceKeypointExtractor.extract_from_image`. The image API receives no frame index to convert again. The public Ambient video-frame interface instead accepts a one-based frame number, which matters if an alternative adapter uses that entry point.

The source cache retains the full BlazePose-33 output, frame numbers, decoded indexes, decoder times, nominal times, dimensions, confidence, annotation hashes, video hash, pose-weight hash, and relevant implementation identities. A sequence-specific annotation-box gate is applied after shared source inference. Unresolved geometry is invalidated. Pelvis containment cannot guarantee the intended identity when people overlap.

`sampled_spans` chooses nearest native frames on a nominal 10 Hz grid within contiguous annotation spans. `supported_window` calls the shared study normalization in `worldmodels/pose.py`, then checks the separate prefix and future observation fractions. That normalization centers each frame on its pelvis and obtains scale from observed prefix torso lengths. The retained array uses the explicit sixteen-joint BlazePose subset. It is relative 2D articulation, not calibrated physical displacement.

The prepared cohort contains 2,818 windows. `worldmodels/restart_experiment.py` removes three windows at the common distal-target support gate, leaving 2,815. Training and reporting retain source/sequence/window relationships. Normal-only fits and inner regularization choices are distinct from the older supervised classifier notebooks. Future validity is allowed to define observed evaluation support, but not predictor inputs.

## What the notebooks actually did

| Notebook group | Evidence inspected | Interpretation |
| --- | --- | --- |
| 00–02 | Inventory, source identity, pose extraction, declared joint tokens, uniform and suffix masks. | Working ingestion and representation examples; their short-prefix contract differs from the full-duration restart. |
| 03–05 | Demonstration loss terms, normal-only JEPA training examples, condition adaptation, forecasting, stochastic outputs, and persistence comparisons. | Demonstration steps and sample dispersion do not establish trained model superiority or calibrated uncertainty. |
| 06–07 | Saved pilot classifier comparisons, supervised probes, threshold limitations, and later objective-control tables. | Different cohorts and label access prevent pooling these scores with the restart anomaly study. An untrained encoder and simple references remain essential controls. |
| 10 | Rebuilt complete annotation inventory and cohort attrition. | Audits coverage rather than equating frame count with sample size. |
| 11 | Decoded a real source frame and made a fresh Ambient pose call. | Demonstrates the actual image inference boundary and native timing. |
| 12 | Changed future values and validity while checking prefix features. | Establishes an implementation isolation check, not a clinical validity result. |
| 13 | Fitted a small real normal-source ridge model and inspected the completed nested evaluation. | Its displayed training error is didactic; saved outer-source results supply held-out evidence. |
| 14–15 | Read complete forecasting, anomaly, condition, provenance, and claim records. | Support the current findings and their stated limits. |
| 16 | Added a frozen score audit and matched mask examples on real observations. | New execution is a design and evidence laboratory, with no new learned-model results. |

The root notebooks 00–15 are source notebooks with cleared outputs. Executed copies reside under `artifacts/executed*`; the directory identity matters because several historical copies share filenames. The restart execution record documents six successful independent-kernel notebooks and 30 code cells. Source and executed hashes are recorded in `docs/restart/notebook_execution.json`. Earlier synthetic, pilot, and revised-real directories must not be presented as one run or one population.

The current results were traced through `results.json`, `source_scores.csv`, `source_anomaly_scores.csv`, `cohort.csv`, saved fold memberships, checkpoint manifests, and the manuscript evidence validator. Draft03 adds an analysis identity for paired low-motion and within-fold diagnostics. It preserves the original predictions and score directions. The reviewer independently checks frozen hashes and numerical summaries.

## Consequences for the next experiment

The first task is to challenge movement and observation explanations, not to expand the model grid. Current posture is the strongest reported forecast reference, and low motion is a strong anomaly diagnostic. Anatomical masking should be compared with random groups of the same size and time extent. Motion-aware masking needs a uniform sampling component and a mask-pattern-only control because selection itself can reveal hidden information.

Protect the observed shoulders and hips used for normalization when artificially hiding prefix coordinates. Retain the latest target anchors for a residual predictor. Keep target queries fixed without reading future availability, and score every method on the same observed suffix. These details prevent a mask comparison from becoming an unintended change in coordinate reference, available information, or sample population.

The source-level normal sample remains small. Conditional calibration, phase estimation, and stochastic futures are prospective hypotheses with explicit feasibility checks. Their proposed mathematical form and execution order appear in the instruction. No data inspection in this revision establishes their future performance.
