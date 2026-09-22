# Revision 01 — assemble the evidence

19 September 2026. First manuscript draft in an append-only revision series. Scores assess this draft and its available evidence, not its probability of acceptance. The fixed rubric and evidence distinctions are specified in the companion evidence record.

## Proposed title

**Does Paired Synthetic Supervision Preserve Movement in 2D Pose Trajectories?**

## Abstract

Pose trajectories support measurements of movement amplitude, timing and left–right joint relationships, but improved coordinate accuracy does not by itself establish that these measurements are preserved. We investigate whether paired synthetic supervision helps a joint-embedding predictive architecture (JEPA), which learns by predicting hidden features, restore estimated 2D trajectories. Two earlier studies motivate the comparison. A reported AMASS laterality experiment found that a uniform-uncertainty control reproduced the gain of a probability-aware model, while a retained synthetic-adaptation experiment found no added benefit from response-based lesson selection over simpler scene choices. We render matching observations with blur and obstruction, pair estimated tracks with projected body-model joints, and compare latent prediction with coordinate pretraining, direct restoration, an initialized encoder with a fitted readout, and shuffled pairing. Training excludes development people and one pose-extractor family. In a reported synthetic pilot with two development people and one seed, paired JEPA reduces normalized coordinate error by 35.6–45.3% relative to unchanged tracks across three extractors. Against coordinate pretraining, it improves two extractors by less than 1% and worsens the third by 14.0%; initialized and shuffled controls are also competitive. Amplitude evaluation is supported on 12 of 48 correlated records, and paired JEPA has no supported timing records under the current peak-matching metric. These results motivate testing calibration and reference observability before attributing improvement to temporal representations. The laterality result and restoration pilot remain available as summaries in this manuscript audit; real-video preservation and a repeatable advantage for paired JEPA require further evidence.

## Critique and revision decision

This draft keeps the unfavorable comparisons visible and connects both predecessor studies to the proposed investigation. It is crowded, however: the first half reads like three separate projects, and the laterality observation does not explain why anatomical side matters. The final sentence combines artifact availability with an empirical limitation, making both harder to follow. “Supported timing records” also needs an operational explanation. Generic synthetic denoising is established in the related literature, so the contribution needs to concern what the controlled comparison can establish about learned representations.

For revision 02, organize the argument around separating coordinate improvement from preservation. Retain the historical findings as motivation, explain left–right naming ambiguity without implying real-video validation, and state that the projected targets are anatomical proxies. This is an editorial improvement. More people, repeated seeds, complete timing references and recovered prediction artifacts require additional evidence and receive no score credit from rewriting.

## Fixed-rubric assessment

| Dimension | Weight | Score / 10 | Reason and remaining weakness |
| --- | ---: | ---: | --- |
| Conference relevance and contribution | 20% | 6.5 | Representation-learning question is relevant, but the scientific contribution is diffuse and the pilot is small. |
| Claim accuracy and evidence support | 20% | 8.0 | Reports unfavorable controls and summary-only availability; proxy anatomy and controlled laterality need clearer boundaries. |
| Evaluation and statistical rigor | 15% | 3.5 | One seed, two development people, low motion support and absent real-transfer evidence remain substantial weaknesses. |
| Scientific insight and related-work positioning | 15% | 5.5 | Mechanism controls are useful, but the abstract does not yet distinguish this study clearly from existing temporal refinement. |
| Reproducibility | 10% | 5.5 | Retained original-study tables and local code support inspection; laterality and restoration predictions are unavailable locally. |
| Clarity and narrative | 10% | 6.0 | Explicit numbers help, but three study histories compete with the main question. |
| Figures | 5% | 7.5 | The shared pipeline schematic explains the design; it is conceptual and cannot replace empirical trajectory plots. |
| Submission fit | 5% | 7.0 | Suitable research scope and a substantive abstract; framing and evidence remain short of a convincing complete main-track paper. |

**Weighted total: 62.0 / 100.** All versions use `sum(weight_percent × score / 10)`. The common figure is assessed once for the series; its score does not increase with version number. Scores are editorial judgments with limited precision.

## Evidence and figure

- Laterality: [historical validation summary](../../latent-laterality/results/validation.md), controlled relabeling of motion-capture data; original run predictions unavailable here.
- Original synthetic training: [retained study and aggregate audit](../../synthetic-training/study/README.md), synthetic development evaluation, not real-video transfer.
- Revised restoration: [reported source pilot](../proposal/README.md), summary supplied from HAIC; no fresh inference or motion analysis performed for this abstract.
- Shared illustration: [training pipeline](images/abstract-training-pipeline-20260919-v01.svg), conceptual design with pending checks marked explicitly.

**Arithmetic correction, 19 September 2026:** the component scores above sum to **61.25 / 100**, not 62.0. The individual scores are unchanged; all cross-version comparisons use 61.25. This note preserves the first draft as written.

**Precision note from final validation:** direct rounding of the reported errors gives a paired-versus-unchanged reduction range of **35.6–45.2%** at one decimal, or **35.63–45.25%** at two decimals. The abstract above used 35.6–45.3% after intermediate rounding; this note corrects that precision issue while preserving the draft. Revisions 05–07 omit this range and emphasize matched controls.
