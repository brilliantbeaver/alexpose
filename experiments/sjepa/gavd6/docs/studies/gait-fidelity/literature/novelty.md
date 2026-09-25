# Scientific positioning of the JEPA response experiment

[Proposal](../README.md) · [Implemented protocol](../methods/jepa-response.md) · [Search scope](search-map.md) · [Earlier hypothesis audit](../records/history/novelty-before-response-20260923.md)

The active question concerns the accessibility of movement-change information after representation pretraining. The experiment compares an auxiliary loss coupling endpoint feature errors with equally supported independent endpoint regression. Its potential contribution is the controlled finding and its explanation. Predicting differences, using synthetic motion or adding a JEPA does not itself establish novelty.

## What the comparison adds

The core study already supervises a bilateral movement change while fitting the coordinate readout. Because the encoder is frozen at that stage, the readout cannot change which information pretraining made accessible. The follow-up moves an additional change constraint into pretraining while keeping the downstream measurement supervision fixed.

The endpoint feature-regression arm receives the same reference examples, support, target-construction rule and coefficient as the paired arm. On common tensors, their loss difference is the cross-endpoint residual inner product, which isolates the coupling term. Their moving-average teachers can subsequently evolve differently, so this is a controlled training-objective comparison rather than fixed teacher features throughout optimization. The coordinate-difference arm asks whether an improvement requires feature-space prediction. These distinctions are more informative than comparing the new objective only with ordinary JEPA.

The added training signal remains privileged synthetic supervision. It uses projected reference poses and registered pair membership. The deployed restorer uses observed trajectories only, but that deployment restriction does not make the pretraining comparison label-free.

## Closest precedents and the claims they rule out

| Primary source | Relevant contribution | Consequence for this study |
| --- | --- | --- |
| [Sobolev Training for Neural Networks, NeurIPS 2017](https://proceedings.neurips.cc/paper/2017/file/758a06618c69880a6cee5314ee42d52f-Paper.pdf) | Trains against derivatives as well as function values. | Extra supervision of change is established. Our finite differences under a chosen normalization are not automatically derivatives of physical movement. |
| [Augmentation-Aware Self-Supervision, NeurIPS 2021](https://proceedings.nips.cc/paper_files/paper/2021/hash/94130ea17023c4837f0dcdda95034b65-Abstract.html) | Uses information about transformations in representation learning. | A difference target alone is insufficient novelty; the matched endpoint control and restored measurement must carry the argument. |
| [S-JEPA, ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04755.pdf) | Learns predictive skeleton representations for action recognition. | This follow-up tests a particular restoration and measurement task, not the first predictive skeleton representation. |
| [V-JEPA 2.1, version 3](https://arxiv.org/html/2603.14482v3) | Examines dense features in video self-supervision. | It motivates checking feature accessibility, but supplies no evidence that the present pose encoder preserves bilateral response. We do not import its architecture or claim its results. |
| [SmoothNet, ECCV 2022](https://arxiv.org/abs/2112.13715) and [SynSP, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Wang_SynSP_Synergy_of_Smoothness_and_Precision_in_Pose_Sequences_Refinement_CVPR_2024_paper.html) | Address temporal pose refinement and precision/smoothness. | A restoration claim needs a movement consequence beyond smoother or more accurate coordinates. The bounded child matrix is not a new exhaustive benchmark of refiners. |
| [Utility of synthetic musculoskeletal gaits, Nature Communications 2025](https://www.nature.com/articles/s41467-025-61292-1) | Studies synthetic gait supervision and self-supervised pretraining for healthcare applications. | Synthetic gait learning and clinical downstream utility already have precedent. The present synthetic response comparison does not independently establish patient validity. |
| [AMASS, ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/html/Mahmood_AMASS_Archive_of_Motion_Capture_As_Surface_Shapes_ICCV_2019_paper.html) | Unifies motion-capture collections in a common body-model representation. | Provides the motion source and rendering basis, not dense independently measured real-video references for this child. |

These sources were checked for this revision. The [historical audit](../records/history/novelty-before-response-20260923.md) retains the broader gait, symmetry, clinical and uncertainty literature with its original qualifications; those candidate branches are not added to the current matrix.

## How the result would change our understanding

A repeatable improvement over endpoint regression, accompanied by improved encoder accessibility and acceptable coordinate/nuisance behavior, would support residual coupling under the registered observation procedure. Predictor-only accessibility would instead motivate a later capacity-matched test of retaining predictor features. This run diagnoses that possibility without adding another restoration architecture.

If endpoint regression obtains a comparable improvement, continuous feature supervision may be sufficient. Establishing practical equivalence requires a scientifically justified equivalence margin fixed before inspecting results and an interval sufficiently contained within that margin; failure to reject a zero difference is insufficient. If coordinate-difference pretraining matches the improvement, the remedy need not be specific to latent prediction. If only the latent loss improves, useful transfer to movement restoration has not been established.

The teacher can suppress movement, and independent endpoint normalization can create feature changes unrelated to the physical edit. Shared endpoint error also cancels in the delta objective. The probes, no-change diagnostics and level-accuracy measurements test these limitations; they do not convert a latent distance into an anatomical unit.

## What remains outside the claim

There is no new pretraining re-pairing arm, so the experiment cannot establish that correct anatomical pairing uniquely causes a gain. Existing shuffled-reference results provide context, not that missing control. Development people do not become an independent confirmation population because a new loss is evaluated. Three training seeds cannot replace independent people.

No new real-video or clinical reference dataset enters this run. Projected knee excursion is an engineering measurement, and an imposed movement state is not a diagnosis or treatment response. Those applications motivate preserving side-specific movement, but require independent measurement validation before clinical use.

Earlier gavd5-drift laterality, gavd6 latent-laterality and synthetic-training-v2 studies have different inputs, grouping units and endpoints. They motivate this design without supplying its findings. The current Gait Fidelity core and child must establish their own results.
