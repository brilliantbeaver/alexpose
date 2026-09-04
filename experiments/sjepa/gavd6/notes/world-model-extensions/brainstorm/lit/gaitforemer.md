# GaitForeMer: Self-Supervised Pre-Training of Transformers via Human Motion Forecasting for Few-Shot Gait Impairment Severity Estimation

## Problem

Clinical gait severity labels are scarce. GaitForeMer tests whether forecasting motion on a large public skeleton dataset can reduce the labels needed to estimate Parkinson's disease gait impairment on the 0 to 4 Movement Disorder Society Unified Parkinson's Disease Rating Scale, or MDS-UPDRS.

## Method in five sentences

1. GaitForeMer encodes forty frames of 3D skeletons with a transformer and predicts the next twenty frames with a non-autoregressive transformer decoder.
2. During pretraining, a linear head also predicts one of sixty NTU RGB+D action classes from the encoder representation.
3. The pretraining loss gives equal weight to future-pose L1 error and supervised action classification over 56,880 public skeleton sequences.
4. Clinical videos are lifted to 3D with VIBE, normalized, split into 100-frame clips, and used to fine-tune both the forecasting branch and a four-class severity head.
5. The best schedule fine-tunes both branches for fifty epochs and then the class branch alone for fifty epochs.

## Headline numbers

The clinical cohort contains 54 participants scored independently by three movement-disorder neurologists. Scores 3 and 4 are merged because severe cases are scarce. Under participant leave-one-out cross-validation, Table 1 reports macro F1 of 0.76, precision of 0.79, and recall of 0.75. The same architecture trained without NTU pretraining reaches 0.60 macro F1, while the previous OF-DDNet result is 0.58. Figure 2 reports 0.56 macro F1 with 25 percent of the clinical training data, versus 0.52 for ST-GCN with all data and 0.58 for OF-DDNet with all data.

## What it makes possible here

This is the direct baseline for the claim that motion prediction can improve label efficiency in clinical gait analysis. It also establishes a harder standard than binary normal-versus-abnormal detection: few-shot severity under person-level separation. A GAVD proposal must ask what prediction provides beyond a representation transfer gain. Useful extensions include calibrated generative surprise, interpretable counterfactual repair, pathology-type transfer, or a test of whether normal-only AMASS adaptation helps or erases impairment cues.

## Limitations

Pretraining is not fully self-supervised because its action head uses NTU labels. The clinical cohort is private and small. The paper merges the two most impaired categories and breaks annotator ties randomly. It does not model rater uncertainty, external-site shift, or common video shortcuts. Forecast quality is assessed qualitatively, so the evidence does not show that forecast error itself measures severity. The model is trained from scratch on NTU, which is disallowed for the present program.

## Access status

Full text and the official GPL-3.0 repository were read on 2026-09-03. Code is public. The clinical data are not public. The repository's Stanford Box checkpoint link returned HTTP 404 on 2026-09-03, so no usable pretrained checkpoint was verified. GaitForeMer is prior work and an evaluation baseline, but it is not an admissible starting checkpoint unless the broken link is restored.

## Sources

- Full paper and version record: https://arxiv.org/abs/2207.00106
- Official code: https://github.com/markendo/GaitForeMer
- Broken checkpoint URL: https://stanfordmedicine.app.box.com/s/d43piha9pos9xneisfgxln394tc81oyp
