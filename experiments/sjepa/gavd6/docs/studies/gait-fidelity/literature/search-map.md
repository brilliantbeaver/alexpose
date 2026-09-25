# Literature scope and source checks

[Current positioning](novelty.md) · [Scientific protocol](../methods/jepa-response.md) · [Historical broad audit](../records/history/novelty-before-response-20260923.md)

The 23 September 2026 revision narrows the question from broad gait restoration to a controlled pretraining comparison. Sources were read from original proceedings, author preprints and publisher records. This is a focused scientific positioning exercise, not a systematic review or proof that no related method exists.

## Sources checked for the focused comparison

| Question | Primary sources | Why they matter |
| --- | --- | --- |
| Is change supervision itself new? | [Sobolev Training](https://proceedings.neurips.cc/paper/2017/file/758a06618c69880a6cee5314ee42d52f-Paper.pdf); [augmentation-aware self-supervision](https://proceedings.nips.cc/paper_files/paper/2021/hash/94130ea17023c4837f0dcdda95034b65-Abstract.html) | These motivate a narrow residual-coupling claim and an endpoint-regression control. |
| What has predictive representation learning already established? | [S-JEPA](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04755.pdf); [V-JEPA 2.1](https://arxiv.org/html/2603.14482v3) | Feature prediction and dense representation learning are established; a specific movement-preservation outcome still needs evidence. |
| Is pose refinement or synthetic gait learning new? | [SmoothNet](https://arxiv.org/abs/2112.13715); [SynSP](https://openaccess.thecvf.com/content/CVPR2024/html/Wang_SynSP_Synergy_of_Smoothness_and_Precision_in_Pose_Sequences_Refinement_CVPR_2024_paper.html); [synthetic musculoskeletal gaits](https://www.nature.com/articles/s41467-025-61292-1) | Position/smoothness refinement and synthetic clinical learning are substantial precedents, not contributions to claim here. |
| What does the source dataset represent? | [AMASS](https://openaccess.thecvf.com/content_ICCV_2019/html/Mahmood_AMASS_Archive_of_Motion_Capture_As_Surface_Shapes_ICCV_2019_paper.html) | Clarifies motion-capture/body-model provenance and the distinction from real-video references. |

Publisher access can be intermittent; the SmoothNet author preprint and CVF repository records were used when a direct PDF request failed. This revision did not acquire any cited dataset, reproduce another paper's experiments or verify every access statement in older candidate lists.

## Scope of the earlier search

The 19 September investigation covered clinical laterality, synthetic biomechanical learning, equivariant representations, uncertainty, temporal refinement and controlled simulation. The 21 September extension added graph/time masking and a direct audit of the local videos and caches. Their detailed hypotheses and citations are retained in the [historical audit](../records/history/novelty-before-response-20260923.md), [local-video review](../data/local-videos.md) and [evidence index](../evidence/README.md).

Those searches motivated alternative directions, including explicit bilateral architectures and anatomical-assignment uncertainty. The current follow-up defers them to keep architecture and deployment inputs fixed. A diagram or promising paper does not add an experiment to the saved matrix.

## Remaining evidential boundaries

The literature does not predict which of the three new variants will perform best. Measured source results, person-level uncertainty and matched controls must determine that. A future first-of-its-kind claim would require a broader forward-citation review than this implementation update. Clinical transfer and practical measurement benefit also require references beyond the current synthetic cohort.
