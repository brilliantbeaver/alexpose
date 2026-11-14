# Technical methods and uncertainty record for v9

This companion preserves the exact evaluation details behind the more accessible v9 manuscript. It describes the same notebook-associated experiment and retrospective checks used in v8. No new neural model was trained for the revision, and no saved prediction was changed.

## Meaning of the scores

The primary predictions are the 88 held-out clip rows from the notebook-associated September execution identified in [reproducibility.md](reproducibility.md). They cover 41 source-video IDs: 16 Normal, 13 MS and 12 PD. A source ID is not a verified participant ID.

If source s has n_s clips, every clip from that source receives weight 1/n_s. For true label a and predicted label b, the weighted confusion count is:

`C[a,b] = sum_i 1(y_i=a and prediction_i=b) / n_source(i)`.

For each label a, `F1[a] = 2*C[a,a] / (sum_b C[a,b] + sum_b C[b,a])`; macro-F1 is the mean of the three label scores. This combines all test predictions before scoring. It is not the mean of per-source F1 scores, and it does not aggregate each recording to a single predicted label. The equal-clip comparison uses weight one for every clip. The fold-summary column is the arithmetic mean and population standard deviation (`ddof=0`) of five source-weighted scores. F1 calculated from pooled predictions need not equal the mean of fold F1 values.

## Retrospective source resampling

The notebook evaluation code has no saved execution outputs; the manuscript's primary results are verified against the execution matching its written results. The following resampling analysis was performed for manuscript checking and is not presented as a saved notebook output.

Use **10,000 draws**, NumPy random seed **20261004**, and sorted source IDs. Within each label group, sample source IDs with replacement, preserving counts of 16 Normal, 13 MS and 12 PD sources. Include every clip from each sampled source. Every sampled occurrence contributes total weight one, distributed equally among its clips; a source drawn twice contributes twice. Use the same draws for every compared method. Recalculate pooled source-weighted macro-F1 for each method, then subtract the comparator score from the S-JEPA score on that draw. Report the 2.5th and 97.5th percentiles. Thus positive differences favor S-JEPA.

| Quantity | Observed value | 2.5th–97.5th percentile interval |
|:--|--:|:--|
| S-JEPA source-weighted macro-F1 | 0.456624159 | [0.328714413, 0.578793142] |
| S-JEPA minus mean pose | 0.004731283 | [−0.160496452, 0.173998638] |
| S-JEPA minus historical notebook RF | 0.045361249 | [−0.133446835, 0.220113928] |
| S-JEPA minus corrected verification RF | 0.131440055 | [−0.046114011, 0.303792121] |

These intervals describe sensitivity to the sampled source composition while holding the trained models, saved predictions and label-specific source counts fixed. They do not include changes due to retraining, initialization, source partitions, stage selection or other development choices. Training sets overlap between folds, so their fitted models are dependent. Participant overlap across source IDs is unresolved. These limitations prevent interpretation as a complete measure of population uncertainty or a confirmatory test of superiority or equivalence. The paper reports the observed RF advantage without claiming that it will hold in every new sample.

The separate 20,000-draw, seed-20260920 calculation retained in the wider audit concerns a different diagnostic setting. Its values must not be substituted for the manuscript intervals above. The original v1 interval repeated for RF and mean pose is unsupported; the [numerical consistency audit](results-numerical-consistency.md) explains its reconstruction and correction.

## What the positive interpretation supports

S-JEPA's fixed features were obtained through label-free encoder training and used by a supervised linear classifier. Their held-out score shows that this complete procedure produces features predictive of the supplied collection labels. It does not isolate the increase in useful information caused by training: that would require an otherwise identical untrained-encoder comparison. It also does not establish that the features encode specific neurological impairments, calibrated physical gait measurements, or a causal benefit from frame order. These questions are explicitly left for follow-up experiments.

The historical RF has a duplicated ankle-range input. Its source-weighted score is 0.411262910; after correction it is 0.325184104. The versions remain separately identified. Under equal-clip scoring, S-JEPA is 0.397033957 and historical RF is approximately 0.397316, so their common rounded value of 0.397 does not support an S-JEPA lead on that measure. The main comparison concerns the specific feature/classifier procedures and fixed settings tested here.

## Training, mask and coordinate details

The exact learning equations and updates are in [method-details.md](method-details.md). [notebook-audit.md](notebook-audit.md) records the implementation behind the seven top-level notebooks, including the 33-landmark input, stochastic region masks, twelve-landmark weighting set, membership correction and limited context guarantee. It also distinguishes training masks from the fixed 161-position output readout and explains the reflection correspondence issue. [claim-ledger.md](claim-ledger.md) maps consequential statements to their evidence.

MediaPipe image fractions are converted to pixel coordinates before cleaning and interpolation. The encoder and mean-pose control then use hip-centered, torso-scaled coordinates; the RF uses the cleaned pixel coordinates. The unchanged visibility channel under spatial normalization has already undergone the cleaning and interpolation step. Nominal 15-fps sampling does not guarantee equal elapsed time for every 32-frame window.

## Reproducing the checks

From the experiment directory, with the retained project data and dependencies available:

```sh
.venv/bin/python docs/wacv2007/evidence/results_audit.py
.venv/bin/python docs/wacv2007/scripts/verify_delivery.py
```

The numerical audit writes its results under `docs/wacv2007/evidence` and uses in-memory copies for the RF-defect and original-draft forensic checks. It does not change the notebooks, cached poses, checkpoints or saved predictions. Run identities, file hashes and environment details are in [reproducibility.md](reproducibility.md). The Overleaf source package is intended to compile the manuscript; it does not include the original video collection or a complete executable training environment.
