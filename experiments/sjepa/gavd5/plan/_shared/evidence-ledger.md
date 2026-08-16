# Evidence ledger for notebooks 00 to 06

This ledger records the facts used to select the seven proposals. A saved output is evidence about this run. It is not automatically evidence about a new person, camera, or clinic.

## Data and independence

| Quantity | Saved value | Consequence |
|---|---:|---|
| Canonical sequences | 96 | Useful for pipeline development, not 96 independent people |
| Canonical source videos | 18 | Primary independent unit for available evaluation |
| Canonical normal sources | 1 | Broad normal-gait generalization cannot be estimated |
| Parkinson's sources | 2 | Enumerate both possible single-source holdouts |
| Stroke sources | 3 | Report every source-level result |
| Myopathic sources | 10 | Largest source support, still small |
| Cerebral palsy sources | 2 | Enumerate both possible single-source holdouts |
| Added-normal sequences | 63 accepted of 64 candidates | Secondary data with a different annotation and extraction pathway |
| Full curriculum | 159 sequences from 35 videos | Larger clip count does not remove source and provenance dependence |

Source: project `README.md`, notebook 01 outputs, and notebook 06 leakage tables.

## Model and training

- Every 64-frame sequence is divided into 16 four-frame positions.
- Thirty-three joints times 16 positions gives 528 possible joint-time tokens.
- Only 12 shoulder, hip, knee, ankle, heel, and foot-index landmarks may be masked as targets.
- The nominal eligible-mask target is 0.60, but the batch-safe sampler uses the least-visible sample. The saved mean eligible fraction fell from 0.551 at the end of Stage 0 to 0.423 at the end of Stage 4.
- The completed curriculum used 600 epochs and 11,400 optimizer updates.
- Stage 0 uses normal sequences. Stages 1 to 4 add a condition-aware group loss, so they are label-informed.
- Final JEPA loss was 0.477845 and final VICReg loss was 8.418068.
- Final feature standard deviation was 0.413745 and mean pairwise cosine was 0.609342.
- Normal-anchor cosine fell from 0.954 after Stage 1 to 0.594197 after Stage 4, evidence of substantial drift.

Source: notebook 04 saved outputs and project `README.md`.

## Representation geometry

Notebook 05 pooled each canonical sequence into a 384-dimensional vector using means and standard deviations. Saved values:

| Diagnostic | Value | Interpretation |
|---|---:|---|
| Cosine silhouette | 0.008975 | Does not show clean five-group separation |
| Minimum centroid distance | 0.036718 | Smaller than mean within-condition distance |
| Mean centroid distance | 0.292119 | Descriptive only |
| Mean within-condition distance | 0.119521 | Conditions overlap materially |
| Closest centroids | Myopathic and cerebral palsy | Descriptive, not clinical similarity |

The 0.036718 pooled-space cosine distance is not comparable to notebook 04's 0.364318 Euclidean group-loss centroid distance. They use different representations and metrics.

## Readouts and leakage

| Lane | Accuracy | Balanced accuracy | Macro F1 | Exposure boundary |
|---|---:|---:|---:|---|
| All-96 stratified S-JEPA | 0.793 | 0.889 | 0.821 | All 16 test videos overlap classifier training; all 29 test rows trained the encoder |
| All-96 missingness only | 0.448 | 0.466 | 0.429 | Same sequence-level split |
| Exact earlier 68-row S-JEPA | 0.714 | 0.730 | 0.742 | All test videos overlap; encoder saw all rows |
| Video-grouped binary readout | 0.849 | 0.874 | 0.826 | Probe split is grouped, but encoder saw all 159 rows |
| Video-grouped five-class readout | 0.653 | 0.603 | 0.625 | Two folds; encoder saw all 159 rows |

These values establish in-corpus decodability. They do not estimate unseen-source performance.

## Artifact lineage warning

The augmented experiment is documented with fingerprint prefix `d0acc262`, while the locally available artifact set has also been observed with a canonical fingerprint prefix `dba24a`. Some documented augmented checkpoint filenames are absent from the current local cache. Before comparing checkpoints, bind every result to one manifest, experiment fingerprint, file hash, config, and cohort. Do not combine metrics across lineages.

## Evidence levels

| Level | Supported statement |
|---|---|
| Direct implementation fact | The model, masking, preprocessing, and readout code paths exist and saved checks ran |
| Direct run fact | The recorded run stayed numerically finite, did not reach total constant collapse, and drifted across stages |
| Corpus-specific descriptive fact | Labels and two scalars are decodable inside the exposed corpus |
| Open question | Whether temporal order remains in tokens after current pooling |
| Open question | Whether any method generalizes to unseen source videos when the encoder is trained strictly inside the outer fold |
| Unsupported | New-patient, cross-clinic, causal, diagnostic, disentangled, or full world-model claims |

## Primary literature anchors

- Abdelfattah and Alahi, [S-JEPA](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/4755_ECCV_2024_paper.php), ECCV 2024.
- Assran et al., [I-JEPA](https://arxiv.org/abs/2301.08243), CVPR 2023.
- Bardes et al., [Revisiting Feature Prediction for Learning Visual Representations from Video](https://arxiv.org/abs/2404.08471), 2024.
- Bardes, Ponce, and LeCun, [VICReg](https://arxiv.org/abs/2105.04906), ICLR 2022.
- Ranjan et al., [GAVD](https://arxiv.org/abs/2407.04190), IEEE Access 2025.
- Kapoor and Narayanan, [Leakage and the Reproducibility Crisis in Machine-Learning-Based Science](https://arxiv.org/abs/2207.07048), Patterns 2023.
- Varoquaux, [Cross-validation failure: Small sample sizes lead to large error bars](https://pubmed.ncbi.nlm.nih.gov/28655633/), NeuroImage 2018.
