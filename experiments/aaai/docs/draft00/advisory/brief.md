# Independent advisory brief

The user wants a strong AAAI ASTAD workshop paper using the GAVD data already available. The date is November 14, 2026, and the deadline is November 20. We are revising research documents, not running new experiments in this turn. Give candid, imaginative, technically specific advice. Do not inflate evidence or guarantee acceptance. Use plain language and complete sentences.

## Current evidence

The project has eight runnable tutorials, tested data and model code, and saved real-data experiments. The existing five-class experiment uses normal, parkinsons, stroke, cerebral palsy, and myopathic. The user has now explicitly defined the expanded binary task: normal is the only normal folder, and every other GAVD folder is abnormal gait. The selected annotations contain 666 sequences from 103 source videos. After pose quality checks, the cohort has 450 sequences from 91 sources and 2,504 overlapping windows. There are no verified participant IDs or diagnoses. Splits protect source IDs, not proven people. All three test splits have been inspected.

Random Forest source-weighted macro-F1 is 0.267, 0.482, and 0.333. Validation-selected JEPA scores are 0.244, 0.252, and 0.331. An untrained encoder with a supervised probe scores 0.293, 0.327, and 0.348. Normal-only masked discrepancy AUROC is 0.200, 0.604, and 0.583. Suffix forecasting scores are 0.138, 0.417, and 0.567. Normal embedding distance scores are 0.323, 0.625, and 0.250. Five or six normal calibration sources give an infinite threshold for the finite-sample 5% rule. At least 19 calibration units are needed merely for a finite threshold, not adequate precision.

Normal pretraining used only 45, 66, and 63 optimizer updates. With EMA 0.996, the algebraic initialization coefficients are 0.835, 0.768, and 0.777. Losses still change substantially. Progressive training adds 777, 747, and 825 updates and changes optimizer state. A single positional-noise run performed worse. These observations concern a short training recipe, not a general limitation of JEPA.

The model uses 2D XY coordinates, 16 BlazePose joints (11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32), 32 frames at 15 Hz, four-frame joint patches, width 64, two encoder layers, one predictor layer, an EMA teacher, latent cross-entropy, and VICReg. Uniform random masking independent of motion is a firm requirement for the main model. Missing observations remain missing and are excluded from attention, loss, and pooling. Random masking is completion; the suffix model is separate. The supervised probe uses the online encoder on complete observed windows. Condition labels influence staged sampling. A learned stochastic latent prior and posterior are not implemented.

## Newly verified GAVD opportunities and limits

A fresh scan of the per-condition CSV files found 1,874 sequences, 458,116 rows, 348 source IDs, and 12 labels. The source cache was excluded from counting.

| Label | Sequences | Source IDs |
| --- | ---: | ---: |
| abnormal | 767 | 117 |
| antalgic | 35 | 10 |
| cerebral palsy | 64 | 11 |
| exercise | 234 | 98 |
| inebriated | 23 | 8 |
| myopathic | 188 | 30 |
| normal | 291 | 32 |
| parkinsons | 47 | 11 |
| pregnant | 6 | 2 |
| prosthetic | 39 | 8 |
| stroke | 76 | 19 |
| style | 104 | 3 |

There are 245 sources outside the selected five classes. None contains a normal-labeled sequence. These are additional annotation references, not guaranteed usable media or sources untouched by every earlier local project. No source ID mixes normal with another label. There are 213 sources with multiple camera-view labels, but repeated views of the same person are not verified. Videos can contain different demonstrators.

Only 758 rows contain gait-event annotations: 751 initial contacts and seven other events. Their support by label, shown as rows/sequences/sources, is abnormal 359/34/6, exercise 33/3/1, myopathic 40/7/2, parkinsons 56/2/1, and style 270/34/1. No normal sequence has contact labels. Do not assume abundant phase ground truth or normal contact calibration.

For normal-labeled rows, the dataset field says Normal Gait for 35,975 rows and Abnormal Gait for 5,365. This field alone does not prove acquisition domain, diagnosis, or patient status. The user explicitly defines exercise, style, pregnant, and every other non-normal folder as abnormal gait for the binary endpoint. Honor that definition. Do not use those folders as healthy negative controls. Abnormal gait annotations still do not establish a confirmed medical diagnosis. Metadata do not establish causal pairs.

## Prior art already verified

Skeleton S-JEPA, ECCV 2024, established masked skeleton latent prediction. MAMP has a uniform random masking branch. VICReg and StoP are established methods. FSGait, ACCV 2024, uses normal-only reconstruction and prediction with joint and frame anomaly outputs. GenGait, a 2026 preprint, uses a normal-only Transformer masked autoencoder, joint inconsistency scores, and normative reconstructions; its abnormal test patterns are simulated. GaitJEPA, IJCB 2026, uses masked and future latent prediction on silhouettes. SeeKer, ICCV 2025, already uses confidence-weighted keypoint likelihoods. STG-NF, ICCV 2023, models normal skeleton distributions.

Asleep at the Wheel (arXiv 2608.01336) examines domain confounding in JEPA anomaly scores. The study at arXiv 2611.01815 critiques physical-surprise interpretations. Ren and colleagues introduced background-corrected likelihood ratios in 2019. Generic confidence weighting, positional noise, stochastic latents, normal pretraining, or applying an established model to another dataset is insufficient novelty by itself.

## Requested advice

1. Propose three distinct, ambitious, testable paper concepts. For each, state the central claim, the difference from prior art, the GAVD asset it actually uses, the decisive experiment, the failure criterion, and feasibility within one week. Rank them.
2. Choose the strongest direction. Explain whether the previous draft aims too low or asks for evidence that cannot be obtained this week. Offer one specific mechanistic idea where justified.
3. Specify a small decisive experiment matrix, appropriate baselines, evaluation units, and a policy for previously inspected test data. Separate normal-only detection from supervised classification. Explain what makes a negative finding scientifically informative rather than a consequence of inadequate training.
4. Identify the most serious confounds. Distinguish an inconclusive interval from evidence excluding a useful gain. Flag unverified literature claims rather than inventing citations.
5. Give a six-day plan with explicit decision points. Do not assume that new clinical collaborations or data collections are already available. Do not provide acceptance probabilities.

Write about 1,500 to 2,500 words. Avoid em dashes. You are a critic and advisor. The coordinating author will integrate the advice rather than use model agreement as evidence. Do not read credentials or .env files. Do not edit the main documents, train models, or download videos.
