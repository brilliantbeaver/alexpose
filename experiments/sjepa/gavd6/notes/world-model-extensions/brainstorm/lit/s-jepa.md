# S-JEPA: A Joint Embedding Predictive Architecture for Skeletal Action Recognition

## Problem

Skeleton representation models often learn by reconstructing missing joint coordinates. Those coordinates contain detector noise and provide little context on their own. S-JEPA asks whether predicting contextual representations of hidden joints produces more transferable action features. A joint-embedding predictive architecture (JEPA) predicts internal vectors, called latent representations, rather than reconstructing its input.

## Method in five sentences

1. The method divides a 3D skeleton sequence into four-frame joint tokens and selects 90 percent of them with motion-aware masking.
2. A view encoder processes the visible tokens after random rotation, translation, or reflection, while a target encoder processes the complete sequence.
3. A five-layer transformer predictor estimates the target encoder representations at the hidden token locations.
4. Cross-entropy matches sharpened probability distributions from predicted and target representations, and batch centering prevents one target dimension from dominating.
5. The target encoder follows the view encoder by an exponential moving average, so gradients train only the view encoder and predictor.

## Headline numbers

Table 1 reports frozen linear-evaluation accuracy of 85.3 and 89.8 percent on NTU60 cross-subject and cross-view, 79.6 and 79.9 percent on NTU120 cross-subject and cross-setup, and 92.2 and 53.5 percent on PKU-MMD phases I and II. Table 2 reports fine-tuned accuracy of 93.1 and 97.6 percent on NTU60 and 90.3 and 91.3 percent on NTU120. Table 4 reports transfer accuracy of 71.4, 74.2, and 70.9 percent on PKU-MMD phase II after pretraining on NTU60, NTU120, and PKU-MMD phase I. Table 5 shows collapse when the exponential-moving-average target update is removed: linear accuracy falls to 1.6 percent on NTU60 and 0.83 percent on NTU120. Pretraining used 1,200 epochs, effective batch size 256, and eight NVIDIA A100 GPUs.

## What it makes possible here

S-JEPA supplies a direct baseline for asking whether hidden-motion prediction contains pathology information. Its predictor exposes joint-by-time surprise rather than only a pooled class embedding. The current repository already implements a smaller paper-aligned variant, so a first GAVD error-map experiment can reuse the model contract and controls. The strongest extension would convert the discriminative predictor into a conditional keypoint sampler or compare its error geometry with a publicly downloadable motion prior. That generative step is not part of the paper.

## Limitations

The paper evaluates action identity on clean benchmark skeletons, not pathology, severity, laterality, clinical shift, or noisy in-the-wild monocular video. All headline datasets provide 3D skeletons. The reported training recipe is long and uses eight A100 GPUs. The method predicts latent distributions but does not decode future coordinates or support counterfactual rollouts. The paper does not test whether prediction error is calibrated or anatomically meaningful.

## Access status

Full text read on 2026-09-03. The official project page and ECVA paper are public. The official project page links the paper but exposes no code or checkpoint link. Searches found no author-linked skeletal S-JEPA checkpoint. This repository's implementation is therefore a reimplementation, not a verified official checkpoint.

## Sources

- Project page: https://sjepa.github.io/
- Full paper: https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04755.pdf
