# Data and evaluation protocol

This project studies whether a predictive skeleton representation detects deviations from a learned normal-gait reference and distinguishes five supplied gait labels. It does not establish a clinical screening or diagnostic system. The primary endpoint is source-level binary balanced accuracy for normal versus the selected non-normal labels. Five-class macro-F1 is secondary. AUROC, average precision, class recall, sensitivity, specificity, and prevalence accompany the primary result.

## Fresh annotation accounting

The inventory was computed by parsing every current per-sequence CSV under `data/gavd`, excluding the upstream `source/` cache. It verifies the ten-field schema, one sequence/source/label per file, positive one-based frames, no repeated frame within sequence, unique sequence IDs, and agreement between filename and label directory. The source manifest fingerprint and every current sequence-file fingerprint are recorded. The source manifest remains the upstream provenance authority; an inventory is not a new download of upstream main.

| Supplied label | Sequence CSVs | Annotation rows | Source videos |
|---|---:|---:|---:|
|normal|291|41,340|32|
|parkinsons|47|10,426|11|
|stroke|76|33,934|19|
|cerebral palsy|64|20,346|11|
|myopathic|188|34,595|30|
|Selected union|666|140,641|103|
|All 12 GAVD labels|1,874|458,116|348|

Stroke appeared twice in the brief and is included once. Restricting the first comparison to these five labels gives a clear, auditable target population. It also narrows the claim. A model trained this way has not demonstrated recognition of every abnormal gait. The unselected GAVD labels can support a later external-condition stress test, with its rules fixed before looking at results. Do not add those labels to the final test set after tuning against them.

Parkinsonian, stroke, and cerebral-palsy annotations motivate movement-control questions. Myopathic gait should be described as neuromuscular rather than automatically grouped as a central neurological disease. The folder names do not independently verify etiology, diagnosis, medication state, disease severity, or whether a video depicts a demonstration. That uncertainty matters more than choosing a plausible neuroscience story after seeing classification errors.

## Existing media and exclusions

The current media manifest lists 100 of 103 requested source videos. Existing originals are under `data/videos-gavd/<label>/<video-id>.<container>`. We preserve this layout and resolve it before downloading. New downloads use `data/videos-gavd/sources/<video-id>.<container>` through Ambient's `download_youtube_urls`. New previews belong under `data/videos-gavd/previews/`; do not duplicate originals into more condition folders.

The fresh annotation cohort contains 652 media-eligible sequences from 98 source IDs. Fourteen sequences are excluded before pose processing: nine have no local video, one source is too short for its requested annotation range, and four have a documented video-reference mismatch. The mismatch is normal source `2LbERGI34lA`; its four annotated ranges show a map or wide street view rather than the expected walking subject. The existing exclusion is preserved even if a file is later downloaded again. The original CSVs remain unchanged.

Media eligibility is not pose eligibility. Image-size differences, annotation-time frame-rate uncertainty, edits, multiple people, and pose-estimator errors remain possible. The existing still-image review does not verify every frame or clinical label. A paper needs a documented person/reference review and pose attrition table. Report the requested sources, available sources, reference-approved sources, usable sequences, and usable windows separately.

## Partitions are established before fitting

Seed 27 allocates source IDs within label to approximately 60% training, 20% validation, and 20% test. Every sequence and window from a source inherits the same partition. The source table is frozen before quality filtering, so an unavailable source does not get replaced to improve the result. Mixed labels within a source cause an explicit failure requiring reconciliation.

The current post-media-filter source counts are:

| Label | Training sources | Validation sources | Test sources |
|---|---:|---:|---:|
|normal|19|6|6|
|parkinsons|7|2|2|
|stroke|10|4|4|
|cerebral palsy|7|2|1|
|myopathic|16|6|6|
|Total|59|20|19|

The single cerebral-palsy test source is a serious limitation. Twelve sequences from it do not create twelve independent people. Keep this pilot split fixed, report the limitation, and use a separately declared source-grouped repeated or nested cross-validation study for a more credible paper estimate. Each fold must retrain normal pretraining, adaptation, feature imputers, scalers, and probes from that fold's training sources. Subject identity is not established by YouTube IDs; a person or edited repost can appear in multiple source videos. Audit duplicate media and recognizable repeated subjects before claiming person-level generalization.

No validation or test skeleton participates in self-supervised pretraining. Stage A uses only normal training sources. Stage B starts from a copy of Stage A and adds selected abnormal training sources, with normal replay and a fixed checkpoint identity. The unchanged normal-only checkpoint remains available for one-class anomaly scoring. Training on abnormal observations changes the expected normality reference and therefore requires a separately named detector.

## Pose and window contract

The extraction adapter decodes original source videos with OpenCV and uses Ambient's public `SequenceKeypointExtractor.extract_from_image` for each requested RGB frame. GAVD frame numbers remain one-based. The adapter converts to the zero-based decoded index once, using `frame_num - 1`. Image extraction does not perform another frame conversion. Frame numbers, decoded indexes, presentation timestamps, nominal times, measured FPS, image dimensions, confidence, and missing poses remain explicit.

The default uniform sampling step is two source frames. This does not guarantee exactly 15 Hz because videos have different measured rates. Metadata records the actual nominal sampled rate, and RF temporal features use that per-window value. A paper should resample on audited presentation timestamps to a fixed time grid or explicitly compare fixed-frame and fixed-time protocols. A 16-frame pilot can be shorter than a full gait cycle and is a computational smoke test, not a justified final temporal horizon.

Cache keys include the current CSV bytes, video bytes, MediaPipe model bytes, Ambient extractor source, adapter source, and extraction settings. The cache preserves failed poses at their requested frame positions. The adapter extracts the first full-image pose. All pilot outputs therefore remain unverified for person identity and cannot support paper claims until reviewed. Ambiguous person selection requires review rather than pretending that the bounding box has already identified the correct person. Annotation boxes cannot be applied blindly when downloaded and annotation images differ in scale or framing.

The 16-joint contract and transformations are documented in [joints.md](joints.md). Overlapping windows are allowed within a source but never across partitions. Gaps are rejected. Both models receive the same accepted windows, the same joints, and the same missingness decisions. Window-level metrics alone are not evidence of source generalization.

## Fair classical and learned comparisons

The classical baseline follows exp5's lower-limb angle summaries and uses shared Ambient geometry. The 37-field extension adds arm and foot summaries. Imputation is fit on training observations only. The Random Forest has 200 trees, depth 8, and minimum leaf size 2 by default. The experiment uses scikit-learn directly for its declared 37-field matrix and explicit source weights; Ambient's existing `RFGaitClassifier` wraps `GaitFeatureVector` and its own validation workflow. Shared pose and angle code remains reused, while this study-specific evaluation policy stays local. It is a newly fitted baseline on this study's cohort, not a reused cached exp5 result.

JEPA's classification baseline is a frozen encoder plus a training-only standardized logistic probe. Source weights equalize each source's contribution within a class and then equalize classes. Each approach must share the same input eligibility and training labels. If tuning hyperparameters, use the same declared development budget and choose them using validation sources only. Include a random untrained encoder and simple prediction controls before attributing an advantage to self-supervised dynamics learning.

Predictions are averaged over windows within sequence, then over sequences within source. This prevents a long sequence from dominating its source. Binary abnormal probability is one minus the normal-class probability, and the fixed probability threshold is 0.5 for the primary supervised comparison. Report the number of sources in every class alongside scores. The paired class-stratified source bootstrap resamples identical source indexes for both methods. Its interval estimates uncertainty conditional on this small test cohort. With one cerebral-palsy source, the bootstrap cannot estimate between-source variation for that class.

The one-class anomaly detector uses a different threshold protocol: calibrate its residual scores on normal validation sources, never on final test data. The finite-sample upper order statistic for a 5% false-positive target is infinite with only six normal calibration sources. This means no finite threshold supports that nominal guarantee here. `validation_threshold_report` records `threshold: null` and an explicit insufficient-calibration status. More calibration sources or a separately declared less stringent exploratory target are needed. Do not replace the infinity with a convenient empirical quantile while retaining the 5% guarantee.

## Does GAVD establish generalizability?

GAVD provides useful heterogeneity in viewpoint, clothing, recording quality, and motion appearance. Its refreshed annotation size is large enough to support an exploratory workshop study if the source-level limitations are respected. However, 140,641 annotated frames are not 140,641 independent patients. The selected labels have only 103 source videos before attrition, their true person count is unknown, and diagnosis and acquisition factors are incompletely controlled. Source-disjoint testing is necessary but insufficient for clinical generalization.

A defensible workshop claim is an auditable feasibility study of normal-reference skeleton prediction under source shifts, with explicit negative results and nuisance controls. A stronger clinical claim needs independent cohorts, subject/site separation, verified diagnoses, held-out recording settings, calibration analysis, and a prospective or external assessment. A successful paper cannot be guaranteed by a large derived-window count or a visually separated embedding plot.

## Reproduction and historical precedents

`artifacts/inventory.json` records the exact current annotation and partition fingerprints. `artifacts/inventory.csv` contains every selected sequence, its original source ID, media exclusion reason, and partition. `save_inventory` refuses to overwrite a changed inventory, requiring a new run directory.

The reviewed precedents are `experiments/exp5/01_extract_features.ipynb`, `02_classify_features_rf.ipynb`, and the multiple-sclerosis README and pose-extraction notebook. Exp5 demonstrates Ambient sequence extraction and hip/knee/ankle summaries. The multiple-sclerosis project explains why related clips must stay in source groups and how independently retrained folds avoid leakage. This project consumes the shared Ambient APIs directly; it does not import an older study's models, cached cohort, or claimed scores.
