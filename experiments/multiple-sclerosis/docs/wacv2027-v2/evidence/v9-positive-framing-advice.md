# Independent framing advice for v9

Prepared 2026-10-09 after reading v8, the verified literature and venue record, and the v8 review. This document advises the authoring agent; it does not change the manuscript or its evidence. A separate review should assess the finished v9 and retain the existing rubric without a forced score increase.

## Strongest supported contribution

Lead with the implemented application and the useful comparison. The study connects automatically estimated body landmarks, predictive feature learning, and label classification on held-out source recordings. It compares learned features with inexpensive summaries on identical partitions and shows readers how the evaluation unit affects the result. The numerical findings and their traceable provenance make this an informative pilot for behavioral-vision research.

The learning objective uses no collection labels, while the downstream classifier is fitted to them. Saying that the **encoder** learns without collection labels is accurate. Calling the entire system unsupervised, annotation-free, or clinically validated would be inaccurate. The study demonstrates that this configured pipeline can be trained and evaluated on the retained collection; it does not demonstrate that training improved its features, that its temporal information is useful, or that its prediction mechanism is clinical.

The main observed score is 0.457 under equal-total-source weighting, versus 0.452 for mean pose and 0.259 for the majority-class procedure. These are descriptive comparisons of complete procedures. A higher observed score than the constant baseline can be reported plainly, but should not become an above-chance significance claim or proof of a learned disease signature. The mean-pose result is a constructive finding: an inexpensive summary is a valuable reference for assessing a more complex sequence model in this collection.

## Abstract structure

An accessible positive opening can present the opportunity before the unresolved comparison. For example:

> Automatically estimated body landmarks offer a practical representation for studying walking in video. We apply predictive skeleton learning to a small collection of neurological gait videos and compare its learned features with simple pose summaries on source recordings held apart during fitting and testing.

Follow immediately with the 88 clips, 41 source recordings, supplied Normal/MS/PD labels, and absence of clinical label verification. Explain the method as predicting hidden features from visible body landmarks before introducing its acronym. Report the primary learned/mean-pose scores and their conditional difference interval. Preserve the clip-weighting reversal and four-of-five-fold result; omitting them while emphasizing the small pooled lead would change the balance of the evidence.

A constructive closing can emphasize the evaluation lesson:

> The findings support retaining simple pose summaries and explicit recording-level weighting when evaluating learned gait features, and provide a documented starting point for testing temporal benefit in verified participant cohorts.

Here “support” concerns the value of the comparison, not a claim that the current features are clinical measures. Keep the key implementation and identity limitations in the abstract or a clearly connected limitations sentence. The reflection/mask mismatch remains a material qualification of this particular predictive-learning experiment.

## Introduction and conclusion

The introduction can begin with the opportunity created by video-derived movement measurements, supported by the already verified Stenum and Shin studies. The next paragraph should explain why useful comparisons require separated source recordings and simple pose controls. This makes the experiment's design a response to a scientific question rather than a list of caveats.

A suitable contribution paragraph would describe the implemented adaptation, common evaluation partitions, weighting-sensitive comparison, and auditability. Do not claim novelty for source grouping, JEPA, anatomical masking, or macro-F1; the contribution is their explicit application and analysis in this small collection. The reported outcome is collection-label prediction, with unknown participant independence and no direct neural decoding measurements.

The conclusion can close around the result's practical use, for example:

> This study provides a documented comparison of learned skeleton features and simple pose summaries in 88 gait clips from 41 source recordings. S-JEPA reaches 0.457 source-weighted macro-F1, close to mean pose at 0.452, while the ordering changes under clip weighting and mean pose leads in four of five test groups. These findings make simple pose summaries and explicit evaluation units useful reference points for the next stage of gait-representation research. Testing the added value of training and frame order, and validating on independently verified participants, would establish which parts of the approach can support clinical measurement.

The final sentence states what future evidence would address; it does not promise that the present approach will become a clinical measure. “Would establish which parts” can be softened to “would help determine whether and how” if the final prose risks sounding inevitable.

## Phrases to distinguish

| Defensible positive description | Unsupported stronger interpretation |
|---|---|
| Applies predictive skeleton learning to estimated gait landmarks | Introduces S-JEPA or a clinically validated masking strategy |
| Produces features used to predict collection labels on held-out source IDs | Learns disease-specific gait features or detects neurological disease |
| Has a higher observed score than the majority-class procedure | Is significantly above chance, without a corresponding test |
| Has a score numerically close to mean pose under one weighting | Is equivalent or non-inferior to mean pose |
| Documents how source and clip weighting affect this comparison | Establishes that one weighting universally gives the correct answer |
| Keeps recorded source IDs separate during fitting, selection, and testing | Demonstrates generalization to new participants, sites, or devices |
| Provides an auditable evaluation and a starting point for further tests | Proves that training, temporal order, or anatomical masking caused a gain |

## Live venue recheck

The [workshop call](https://cvpath-vision4neuro.github.io/wacv2027/vision4neuro.html) now extends the regular-paper deadline to **October 16, 2026, 23:59 AoE**, with the earlier October 9 date struck through. The format remains an anonymous WACV 2027 regular paper of up to eight body pages, excluding references; supplementary material is optional. The OpenReview submission must select Vision4Neuro, and accepted work requires an in-person presenter. The central venue record has been updated; the older saved October 8 snapshot remains a historical record.

The [official author guide](https://wacv.thecvf.com/Conferences/2027/AuthorGuides) was also rechecked. The existing official kit and anonymous layout remain appropriate. Preparing an Overleaf source ZIP is distinct from submitting the paper or clearing source-media rights; nothing in this advice records either action as completed.
