# SleepFM: A Multimodal Sleep Foundation Model for Disease Prediction

## Problem

Polysomnography records brain and eye activity, cardiac activity, muscle activity, and respiration, but sites use different channel sets and orders. Supervised models usually target one outcome, require expert labels, and transfer poorly. SleepFM asks whether large-scale self-supervision can produce one representation that tolerates heterogeneous sensor configurations and adapts cheaply to many clinical tasks.

## Method in five sentences

1. SleepFM resamples each signal to 128 Hz, standardizes it, and divides it into 5-second tokens grouped as brain activity signals, ECG, EMG, or respiration.
2. A shared six-layer one-dimensional CNN creates 128-dimensional channel tokens, attention pools the unordered available channels, and a three-layer transformer models 5 minutes of temporal context.
3. Leave-one-out contrastive learning aligns each modality embedding with the average embedding of the other temporally matched modalities while treating other batch members as negatives.
4. Pretraining uses 432,000 hours from 48,000 participants for one epoch, after which the encoder supplies frozen 5-second embeddings across a standardized 9-hour record for downstream adaptation.
5. A small two-layer bidirectional LSTM pools the available modality embeddings, and task heads predict sleep stages, apnea, or 1,041 time-to-disease outcomes using a multilabel Cox proportional-hazards loss with age and sex added for disease prediction.

## Headline numbers

The dataset section and Table 1 report more than 585,000 recording hours from about 65,000 participants; Methods, Implementation details reports a 4.44-million-parameter model trained for 15 hours on one A100. In “SleepFM enables comprehensive disease prediction,” Fig. 2 and Supplementary Table 5, 130 of 1,041 future conditions have both C-Index and 6-year AUROC at least 0.75 after correction; C-Indices include mortality 0.84, dementia 0.85, myocardial infarction 0.81, and heart failure 0.80. In the cross-site SHHS transfer experiment, Fig. 3, fine-tuning uses 3,291 participants and testing uses 2,000; C-Indices are 0.81 for stroke, 0.83 for congestive heart failure, and 0.86 for cardiovascular death. Fig. 4 reports 5 to 17 percent AUROC improvement over supervised baselines. Supplementary Tables 7 to 10 report that combined modalities perform best overall, with disease-specific strengths across brain, ECG, and respiratory signals.

## What it makes possible here

The reusable pattern is frozen foundation representations plus a small sequence adapter, set pooling, and task head. For gait, synchronized RGB, 2D pose, lifted 3D, and motion tracks could act as modality groups, with AMASS projections providing paired views and GAVD providing noisy real subsets. Leave-one-out alignment and explicit availability masks could make detector failures inputs rather than zero-valued joints. Any experiment should train with sensor dropout and report every modality subset, not only the complete-input score.

## Limitations

SleepFM itself required large pretraining from scratch, so only its downstream adaptation pattern fits this repository's constraint. The paper states resilience to missing modalities, but reports no controlled inference-time modality-drop curve for one fixed model; separately trained single-modality models and heterogeneous-site transfer are indirect evidence. The referral-heavy sleep-clinic cohort is not population representative, temporal performance degrades, and case-level interpretation remains weak. SHHS external validation covers only six cardiovascular outcomes. Risk prediction is associative, not causal, and PSG physiology is not gait dynamics.

## Access status

Full open-access article and Methods read on 2026-09-03, not abstract only. Code plus base, disease, and sleep-staging checkpoints are public under CC BY-NC 4.0. SHHS, MrOS, and MESA are available through sleep-data portals. Stanford Sleep Bench requires credentials, a signed data-use agreement, and training, and prohibits redistribution. BioSerenity data remain proprietary.

## Sources

- Article and full text: https://www.nature.com/articles/s41591-025-04133-4
- Open PDF: https://www.nature.com/articles/s41591-025-04133-4.pdf
- Official code and checkpoints: https://github.com/zou-group/sleepfm-clinical
- Stanford Sleep Bench: https://bdsp.io/content/08vg8vqv2wdtwonc1ddy/1.0/
- Fetch date: 2026-09-03
