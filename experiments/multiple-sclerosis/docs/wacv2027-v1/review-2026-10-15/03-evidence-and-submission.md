# Evidence and submission notes

This is an author-facing companion to the replacement passages, not supplementary material to upload unchanged. It includes local paths and source identifiers. The manuscript text is in [02-overleaf-replacements.md](02-overleaf-replacements.md); the complete independent audits are in [evidence/](evidence/).

## Which results the revision uses

The primary comparison is the complete saved execution at:

```text
artifacts/runs/full-v1/9496e61b050f/laptop-1d09c8e1eea5/
  capstone-20261009T060955236580Z/
```

`oof.json` supplies the held-out predictions, `results.json` supplies the retained metrics and training-stage summaries, and `provenance.json` identifies the code and feature-cache inputs. The historical comparison is the sibling directory `capstone-20260921T011305448893Z`. Neither experiment was rerun for this review.

The original PDF uses the September RF results but omits its duplicated ankle-range field. The current notebooks have October outputs surrounded by some September narrative. The revision uses October throughout and explicitly identifies the earlier RF as historical. This is a correction of the baseline implementation, not a claim that S-JEPA improved between executions.

## Verified numbers for the manuscript

| Method | Equal-source macro-F1 | Equal-clip macro-F1 | Correct clips | Fold mean ± population SD |
|---|---:|---:|---:|---:|
| Frozen S-JEPA + linear classifier | 0.456624 | 0.397034 | 36/88 | 0.438068 ± 0.195980 |
| Corrected joint-angle RF | 0.325184 | 0.320940 | 30/88 | 0.305256 ± 0.039217 |
| Mean pose + linear classifier | 0.451893 | 0.409840 | 37/88 | 0.427332 ± 0.132210 |
| Visibility + linear classifier | 0.317565 | 0.318057 | 29/88 | 0.284918 ± 0.104069 |
| Majority class | 0.259398 | 0.202125 | 23/88 | 0.159394 ± 0.021986 |
| Historical RF with duplicate field | 0.411263 | 0.397316 | 37/88 | 0.395384 ± 0.097960 |

The current S-JEPA–RF difference is 0.131440, which rounds to 0.131. The historical difference is 0.045361, which rounds to 0.045. Compute differences before rounding. Macro-F1 of 0.457 is not 45.7% clip accuracy: S-JEPA correctly labels 36/88 clips, or 40.9% without source weighting.

There are 91 raw clips and 88 retained clips from 41 source recordings. The retained label counts are 24 Normal, 29 MS and 35 PD clips, from 16, 13 and 12 source recordings respectively. Each fitted encoder and classifier uses 24 training sources. An additional 8–9 validation sources select the training stage, making 32–33 development sources per fold when training and validation are counted together. The paper must not imply that the complete procedure uses only 24 sources or that the 41 sources are 41 verified participants.

## How the checks were performed

[verify_results.py](evidence/verify_results.py) recalculates the results directly from the saved predicted labels, independently of the project's metric functions. It checks pooled and per-class scores, every confusion cell, fold summaries, test membership, and checksums for all 88 retained pose caches. It also compares the two experiments' cached features, checkpoint tensors and embedding arrays. It does not fit a classifier or retrain a neural model.

The two cached RF matrices differ in only `left_ankle_range`. Reverting that single corrected assignment in memory also reproduces the complete older code-content hash. Together with the saved prediction records, this identifies the origin of the RF change. The learned-model tensors differ slightly between executions, but the seed and source folds are unchanged and all S-JEPA test labels agree. These are repeated executions of the same design, not an independent-seed study.

Run the verification from the experiment directory:

```sh
.venv/bin/python docs/wacv2027-v1/review-2026-10-15/evidence/verify_results.py
```

The script writes [results-verification.json](evidence/results-verification.json), including exact values and the source-bootstrap specification. The recorded check passed. Its model-reading code uses PyTorch to inspect trusted local checkpoints; no network access or source-video processing is needed. The check requires the existing local artifacts and should not be presented as a standalone public reproduction package.

The notebook audit covers all 124 cells in notebooks 00–06, saved numerical outputs, scientifically relevant plots and representative animation/video frames. It traces the active implementation instead of inferring methods from notebook filenames. A separate inexpensive mask-sampling check verifies the denominator correction; [the recorded mask and history check](evidence/mask-and-history-verification.json) also confirms that the notebook 03 and current capstone fold-0 checkpoints contain identical serialized payloads after their temporary archive-directory names are removed. A video-metadata check confirms that integer frame strides approximate a 15-Hz target rather than enforce an identical physical cadence. These are verification analyses, not newly trained model comparisons.

## Claim-to-evidence guide

| Proposed statement | Evidence | Boundary to retain |
|---|---|---|
| A trained representation supports held-out label prediction. | Current `oof.json`, frozen target-encoder readout and training-only logistic regression in `sjepa/full_experiment.py`. | The classifier and checkpoint selection use labels; prediction is of collection categories, not independently verified diagnoses. |
| The trained-feature pipeline exceeds the handcrafted RF tested here. | Independently recomputed source F1 0.456624 vs 0.325184 and clip F1 0.397034 vs 0.320940. | Inputs, classifier family and checkpoint selection differ between pipelines. This does not isolate a causal pretraining effect. |
| Encoder training uses a small video set. | Registry: 24 sources, 45–58 clips, 190–238 windows per training fold. | Pretrained pose estimation and 8–9 validation sources are also part of the procedure; no learning curve measures sample efficiency. |
| Training fits the hidden-feature task while features retain variation. | Notebook 03 diagnostic summaries; current capstone stage endpoints and logs. | Loss/rank are training diagnostics; they do not identify neurological information or establish improvement over a random encoder. |
| The masking implementation uses an anatomical preference. | `masking_v2.py`, 5,000-mask check with seed 42. | Region weights and observed joint frequencies are different quantities. No downstream mask comparison was performed. |
| Predictions extend to excluded source recordings. | Registry membership and per-fold checkpoint training membership. | Different source IDs may contain the same person; the development history does not establish a prospectively untouched external test collection. |

## Workshop fit and preparation

The relevant contribution is markerless behavioral analysis using learned pose representations. The study contains no neural signals, and its contribution does not need to be described as neural decoding. The current workshop requirements, official author-kit verification, live links and primary literature checks are recorded in [the venue audit](evidence/paper-and-venue-audit.md#venue-requirements-and-fit).

For the revised paper, use the unchanged official review style, anonymous author information and the assigned submission identifier. Apply the replacements as one revision, update figures and captions, resolve citation/table labels, and inspect the rebuilt PDF at final column size. The supplied original PDF fits the content limit, but that does not verify the page count of a newly assembled manuscript. The LaTeX fragments are a revision aid; they are not a submission or a complete replacement for the original Overleaf source.

The main results can be understandable without statistical terminology dominating the narrative. R14 preserves a reproducible uncertainty analysis for readers who want it, including the fact that resampling can reverse the observed ordering. Retaining that information in supporting material lets the main text explain the finding and its scope directly.

## Primary literature that shapes the recommended wording

The [published S-JEPA paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04755.pdf) establishes the skeleton feature-prediction framework, including cross-entropy, centering/sharpening and a momentum target. The current experiment is an adaptation, not the introduction of S-JEPA. [MaskFeat](https://openaccess.thecvf.com/content/CVPR2022/html/Wei_Masked_Feature_Prediction_for_Self-Supervised_Visual_Pre-Training_CVPR_2022_paper.html) provides masked visual-feature prediction as related work, including handcrafted targets. These sources support the operations described in the paper, not a claim that the present features have validated clinical meaning.

The [2021 OpenPose gait study](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008935) and [2024 clinical extension](https://journals.plos.org/digitalhealth/article?id=10.1371/journal.pdig.0000467) support the potential of defined video-derived gait measurements. They should not be cited as direct validation of this collection's BlazePose pipeline. The original bibliography already contains these studies, so the proposed corrections require precise attribution rather than a large increase in references.

R3c adds two close precedents omitted from the original bibliography: [Kaur et al.'s neurological gait classification study](https://pubmed.ncbi.nlm.nih.gov/36126031/) and [GaitForeMer](https://arxiv.org/abs/2207.00106). The former was checked through its primary abstract, publisher metadata and authors' code documentation; the full journal manuscript was inaccessible. The latter was checked in full and combines motion forecasting with activity classification during pretraining. The proposed paragraph stays within these verified facts. Ready-to-merge entries are in [additional-references.bib](additional-references.bib).

## Review record

Three parallel audits separately examined notebooks/methods, numerical results and the paper/venue. The resulting replacement text then underwent a fresh Codex adversarial review. [04-review-disposition.md](04-review-disposition.md) records the review target, findings and fixes. The independent contexts provide additional scrutiny within the available Codex model family; they are not model-family diversity or independent scientific replication.
