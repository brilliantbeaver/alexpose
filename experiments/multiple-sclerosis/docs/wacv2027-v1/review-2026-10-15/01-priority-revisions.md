# Review of the original WACV draft

Reviewed 15 October 2026. The target is `docs/wacv2027-v1/wacv-v1.pdf`, titled *Evaluating Predictive Skeleton Learning for Video-Based Neurological Gait Analysis*. Page numbers below are the PDF's printed pages. This review preserves the existing manuscript and Overleaf package; the replacement passages are in [02-overleaf-replacements.md](02-overleaf-replacements.md).

## The paper's strongest defensible contribution

The study shows that a compact S-JEPA trained on a small collection of pose sequences produces features that support prediction of labels on recordings excluded from training. Each encoder trained on only 24 source videos, represented by 45–58 clips. With a separate linear classifier, its frozen features achieve source-weighted macro-F1 of 0.457, compared with 0.325 for the corrected handcrafted joint-angle Random Forest. This is a useful empirical result for learning movement representations from limited video data.

The paper can lead with that finding and explain its significance in ordinary research language. The evidence supports useful **trained representations**; an untrained-encoder comparison would be needed to measure the improvement attributable specifically to pretraining. Mean pose remains a strong comparison at 0.452, so the result does not establish that temporal order supplies the predictive information. These distinctions can be stated once, clearly, without making statistical non-significance the paper's organizing theme.

## Corrections that affect the scientific account

| Priority | Exact location in the original PDF | Required correction and reason | Replacement |
|---|---|---|---|
| Essential | Abstract, p. 1; §5.1 RF paragraph, p. 5; Table 3 and §6.2–6.3, p. 6; Fig. 4(a,c) and §7.1, p. 7; §8, p. 8 | The reported RF score, 0.411, belongs to an earlier implementation that duplicated the right-ankle range in the left-ankle field. The corrected saved execution scores 0.325. Replace all affected RF values together, retain the earlier result transparently, and update the figure. | R1, R7, R10–R13 |
| Essential | §4.1, p. 3, sentence beginning “Across 5,000 sampled masks…” | The percentages around 77–78% measure the fraction of landmark–time positions hidden, not the fraction of windows with any hiding. The two quantities are substantially different. Correct the definition or remove these diagnostic percentages from the main text. Explain that the 1.5 multiplier applies to region-selection weights. | R4 |
| Essential | §6.2, p. 6, “exceeding RF by 0.046”; §7.1, p. 7 | Compute differences from unrounded scores. The current difference is 0.131440 → **0.131**; the historical difference is 0.045361 → **0.045**, rather than 0.046 obtained by subtracting displayed rounded values. These are differences on the F1 scale, not percentage-point changes in accuracy. | R10, R12 |
| Essential | §6.1/Fig. 3, pp. 5–6; §6.3, p. 6 | If using the corrected October execution, use its diagnostics and predictions consistently. Continued-stage effective rank is 9.558–11.638, rather than the older 9.65–11.69. S-JEPA exceeds corrected RF in three folds, rather than two. The original RF bootstrap comparison also belongs to the earlier baseline. | R10; supplementary text R14 |
| Essential | §2.1, p. 2, “Stenum et al. have validated such 2D pose-based gait measures…” | Identify that those studies used OpenPose. They establish the potential of video gait measurement, but do not validate the present BlazePose processing or unverified collection labels. | R3a |

The corrected RF is the appropriate main comparison because it implements the stated bilateral feature set. Its lower score is a consequence of repairing that implementation, not a new improvement to S-JEPA. The earlier RF remains useful as a disclosed historical comparison: S-JEPA also exceeds its source-weighted score. Both executions use the same sources, folds and seed; they do not constitute independent replications across datasets or initializations.

## Revisions that make the paper clearer and more relevant

| Location | Editorial change | Why it matters |
|---|---|---|
| Title and abstract, p. 1 | Name the limited-data representation-learning question, state 24 training videos per fold, and lead with the learned-feature comparison. | “88 clips” describes the whole evaluation collection, not the training set of any one encoder. The small training set is a concrete strength. |
| Final introduction paragraph, p. 1 | State what was learned, what the frozen classifier tests, and how this contributes to markerless behavioral analysis. | This is the direct connection to Vision4Neuro. There are no neural recordings or brain-signal decoding experiments in this study. |
| End of §2.2, p. 2 | Add Kaur et al. and GaitForeMer, using R3c and the supplied BibTeX entries. | These are close prior studies of learned neurological gait representations. Their clinical labels, 3D inputs and evaluation tasks differ, so published scores are not direct benchmarks. |
| §2.3, p. 2 | Replace “better preserve semantic structure” for MaskFeat with a direct description of masked feature prediction. | The broader wording is unnecessary; MaskFeat includes handcrafted feature targets. Retain the existing S-JEPA attribution and its cross-entropy training description. |
| §3, pp. 2–3 | Keep the source-label caveat and exact data counts. State that the target cadence is approximately 15 Hz; avoid a fixed physical duration for every window. | Actual integer-stride sampling varies across recordings. Repetition-padding also does not create additional observed motion. |
| §4.3–4.4, pp. 4–5 | Explain hidden-feature prediction, the 800-update stage and restarted 400-update continuation, then frozen-feature classification. Move the softmax equations and optimizer details to supplementary methods. | Readers need to understand the experiment before its numerical implementation. Notebook filenames mentioning normal-only pretraining or VICReg do not describe the executed experiment. |
| §5.2–5.3, p. 5 | Explain source separation and equal-source scoring in words; retain the weighting definition. Move bootstrap mechanics to a compact supplementary note. | The distinction between clips and source recordings is more consequential here than terminology about estimators. |
| §6, pp. 5–6 | Lead with held-out classification; follow with where errors occur and variation across held-out sources; summarize training behavior last. Move the existing large diagnostic Fig. 3 to the supplement. | Feature-prediction loss is useful supporting evidence but is not the main behavioral result. |
| §7.1–7.3, pp. 6–8 | Explain why trained features are useful, retain mean pose and unmatched-input limitations, and consolidate the clinical boundaries into connected paragraphs. | This avoids repeating uncertainty while keeping the scientific interpretation honest. |
| Figure 4, p. 7 | Replace the RF fold scores and confusion matrix, using the supplied LaTeX figure replacement. | A prose-only correction would leave the old baseline in the visual evidence. |

## Claims to preserve, qualify or remove

**Preserve:** S-JEPA's frozen features support held-out label prediction; the encoder was trained without condition labels; only 24 source videos were used to fit each encoder; the corrected handcrafted RF is lower under both scoring schemes; all clips from one identified source stay in one partition.

**Qualify:** “learned useful representations” refers to the trained encoder and its downstream predictions. The comparison changes the representation and classifier family together: RF uses lower-limb angle summaries, while S-JEPA uses all 33 landmarks and visibility with a linear classifier. A claim about superiority over handcrafted methods in general would exceed this comparison. A claim about sample efficiency would require performance at several training-set sizes. MediaPipe itself is pretrained; the entire visual pipeline was not learned from these 24 videos.

**Remove or replace:** “clinically validated representation,” “disease-specific features,” “subject-independent” or “patient-independent,” “motion-aware masking is contraindicated,” and any claim that the anatomical bias, temporal ordering, or pretraining alone caused the observed advantage. The draft's original S-JEPA loss is not an error: the published S-JEPA also uses cross-entropy over feature dimensions with centering and sharpening.

## What belongs in the main paper

Retain the dataset table, the source-partition table and a compact results table. Keep one pipeline illustration and a corrected performance/error figure. The landmark schematic is helpful if it remains legible at column width, but a detailed mask-frequency analysis is supplementary material. The primary story should be readable without understanding effective rank, bootstrap resampling or temperature-scaled feature distributions.

The original §6.3 does not need to remain as a long argument about failure to establish significance. A shorter account can state that S-JEPA's scores vary by held-out fold and that the study establishes an observed result on this collection. Keep the quantitative uncertainty analysis in the supplement, where readers can inspect it. Removing unnecessary jargon should improve emphasis; it should not turn an observed advantage into a claim of reliable clinical superiority.

## Remaining author-supplied information

The available records do not establish independently confirmed diagnoses, unique participant identities across videos, source-media permissions, participant consent for secondary analysis, or an institutional ethics determination. These are missing documentation, not findings that permission or review was denied. Add factual statements supported by the authors' records before submission; do not infer a permissions basis from a repository software license. The review can complete the scientific corrections without inventing these facts.

The supplied replacement sections are intended for an anonymous regular paper. Apply them together, rebuild with the official review template, and verify the final page count and citation keys. [03-evidence-and-submission.md](03-evidence-and-submission.md) records the current venue requirements and numerical evidence.
