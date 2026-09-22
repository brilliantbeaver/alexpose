# Revision 06 — a focused conference draft

19 September 2026. Follows [revision 05](abstract-v05-encoder-readout.md), incorporating the independent laterality reviewer’s critique.

## Proposed title

**What Does Latent Prediction Add to Paired Synthetic Pose Restoration?**

## Abstract

We study whether paired synthetic supervision improves 2D pose trajectories while preserving movement, and whether latent prediction adds value beyond learning coordinates from the same targets. Earlier controlled laterality summaries show that joint-name swaps distort side-sensitive measurements, while setting naming probabilities to 50/50 matched the proposed model’s gain. Audited synthetic-adaptation outcomes likewise found no added benefit from choosing lessons using a model’s learning response. We pair image-estimator tracks from controlled renders with projected anatomical proxies. A joint-embedding predictive architecture (JEPA) predicts reference features; a separately fitted coordinate readout uses its frozen encoder. Comparisons include coordinate pretraining, direct restoration, a fitted readout on an untrained encoder and shuffled pretraining pairs. Training excludes development people and ViTPose. In a summary-reported pilot with two development people and one seed, paired JEPA improves coordinate error by less than 1% over coordinate pretraining on HRNet and ViTPose, with both reported intervals including zero, and worsens RTMPose by 14.0%. Untrained-encoder readouts yield similar coordinate gains. Preservation remains unresolved: amplitude of signed image-plane ankle separation is supported on 12 of 48 correlated records, and no paired-JEPA record meets the joint reference-and-prediction peak-timing criteria. Planned calibration and reference self-checks will measure how much improvement simple corrections recover and how many records can support timing. Independent real-video evaluation is needed before extending these synthetic coordinate results to preserved movement.

## Changes and critique

The historical comparison now says what the controls did rather than suggesting that the proposed mechanisms had already been established. The draft describes 50/50 naming probabilities and choosing synthetic lessons through a learning response. Calibration will measure what simple corrections recover; it is not presented as a clean causal separation of “correction” and “representation learning.” Peak timing is explicitly tied to the implemented signal.

The title targets ICLR's interest in learned representations, although it gives less prominence to the broader preservation objective. “Untrained-encoder readouts yield similar coordinate gains” is descriptive, not an equivalence result; RTMPose is actually better under the initialized readout. The pilot does not establish that the frozen encoder is unnecessary across recipes or that learning temporal information cannot help.

The final review should challenge whether the bilateral objective is sufficiently visible, whether the predecessor sentence suggests real-video evidence, and whether a fitted temporal readout could be mistaken for a calibration model. Version 07 should use a title that balances the paired-supervision question and movement fidelity, name the current support limitation without diagnosing its cause, and retain the same evidence scores unless an identifiable weakness is repaired. Further stylistic refinement alone does not justify increasing the total.

## Fixed-rubric assessment

| Dimension | Weight | Score / 10 | Reason and remaining weakness |
| --- | ---: | ---: | --- |
| Conference relevance and contribution | 20% | 7.0 | Clearly links the application to what paired latent prediction contributes beyond coordinate supervision; significance remains bounded by the pilot. |
| Claim accuracy and evidence support | 20% | 9.0 | Summary provenance, anatomical proxies, named comparators and unresolved timing are explicit; no causal calibration or preservation claim. |
| Evaluation and statistical rigor | 15% | 3.5 | Unchanged sample and support: one seed, two evaluation people, correlated records, no verified real transfer. |
| Scientific insight and related-work positioning | 15% | 7.5 | Controls support an attribution question rather than an unsupported new-architecture or first-denoising claim. |
| Reproducibility | 10% | 5.5 | Source code and reported aggregates are inspectable, but newer raw predictions and independent replication are still missing. |
| Clarity and narrative | 10% | 8.5 | Predecessor findings are explained with less shorthand, and the final paragraph ties support limits to specific next analyses. |
| Figures | 5% | 7.5 | Common reviewed conceptual figure; no empirical trajectories and limited readability when reduced to manuscript size. |
| Submission fit | 5% | 8.5 | Focused, informative, anonymous draft within the editorial length target; no unverified conference-theme or future-result claim. |

**Weighted total: 70.50 / 100.** Rigor remains 3.5 and reproducibility 5.5. This is the strongest current framing, not evidence of a main-track-ready empirical result.

## Evidence and illustration

The [fixed evidence ledger](abstract-20260919-evidence-and-rubric.md) distinguishes the laterality summary, original synthetic aggregate artifacts, summary-reported restoration pilot, separate real-video diagnostic, fixtures and uncompleted work. New [arithmetic checks](abstract-20260919-exploratory-summary.json) are exploratory calculations from rounded summary values.

The [pipeline illustration](images/abstract-training-pipeline-20260919-v01.svg) is shared across the series. Its readout is trained against projected references using training people; inference excludes reference coordinates and masks. Signed ankle separation is an image-plane proxy, not anatomical-side identification.

