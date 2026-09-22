# Abstract development: conference fit and primary literature

*Version 01 · 19 September 2026 · Research notes for the seven-version abstract series. These notes add no experimental evidence and preserve the existing study record.*

The strongest defensible direction is a controlled study of whether synthetic supervision improves the quantities that movement analysis needs, and whether latent prediction contributes beyond the coordinate targets and fitted readout. The current pilot supports this question and exposes weaknesses in its evaluation. It does not yet show that paired JEPA preserves movement or outperforms simpler restoration methods.

## Conference fit and submission facts

ICLR 2027 welcomes representation learning, computer vision, interpretation of learned representations, and evaluation work. Its call does not announce a narrower annual technical theme. Position this study around what a learned representation retains for quantitative movement measurement, with gait as the motivating application. Calling the work a world model would imply evidence about prediction or dynamics beyond the present offline restoration task. [Official call](https://iclr.cc/Conferences/2027/CallForPapers).

The reviewer guidelines ask whether the work provides meaningful new knowledge, supports its claims, and is rigorous and reproducible. They explicitly allow contributions that do not achieve state of the art. A negative result can therefore fit the conference, but a small inconclusive pilot needs a substantial, generalizable methodological insight to carry that argument. This last judgment is our assessment of the present evidence, not an acceptance prediction. [Official reviewer guidelines](https://iclr.cc/Conferences/2027/ReviewerGuidelines).

The abstract deadline is **18 September 2026, 23:59 AoE**, equivalent to **19 September, 11:59 UTC / 04:59 PDT**. The full-paper deadline is **25 September, 23:59 AoE**, equivalent to **26 September, 11:59 UTC / 04:59 PDT**. Genuine abstracts may be revised before the full-paper deadline, but the submission must remain recognizably the same paper. The author list cannot gain or lose authors after the abstract deadline. The explicit formatting section specifies **nine main-text pages initially** and ten during rebuttal/camera-ready; a later FAQ inconsistently calls ten pages the submission limit. Follow the explicit initial formatting rule. No abstract word limit was stated on the inspected guide; a roughly 200–250-word draft is an editorial target, not a verified rule. [Author guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines).

The final manuscript needs an AI-use statement and corresponding submission-form disclosure. This revision involved AI assistance with framing, literature search, interpretation, writing, and figure preparation; authors should describe the assistance actually used and their verification. [AI policy for authors](https://iclr.cc/Conferences/2027/AIPolicyForAuthors).

## The closest literature and what it permits us to claim

| Primary source | Relevant contribution | Consequence for this manuscript |
| --- | --- | --- |
| Abdelfattah and Alahi, **S-JEPA**, ECCV 2024. [Paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04755.pdf), [author project](https://sjepa.github.io/) | Learns skeleton action representations by predicting masked latent joint representations rather than raw coordinates. The paper distinguishes predictive JEPA objectives from representation-matching objectives that directly impose invariance. | Ask whether a local paired-target adaptation retains measurement-relevant variation. Do not assert that JEPA inherently discards laterality or timing, and do not present the adaptation as an official S-JEPA reproduction. The inspected project page linked the paper, but did not provide an official implementation link. |
| Zeng et al., **SmoothNet**, ECCV 2022. [Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136650615.pdf), [official implementation](https://github.com/cure-lab/SmoothNet) | Uses a temporal refinement network to improve pose smoothness and accuracy, studying transfer across pose estimators, modalities, and datasets. | Temporal pose refinement and estimator transfer already have substantial precedent. Compare against this literature without implying that the local SmoothNet-style adaptation reproduces the published method or defeats its strongest configuration. |
| Baradel et al., **PoseBERT**, TPAMI 2022. [Paper](https://arxiv.org/abs/2208.10211), [author implementation](https://github.com/naver/posebert) | Uses masked modeling of motion-capture data for 3D pose refinement, completion, and prediction. The linked repository provides demo code and pretrained models associated with the earlier MoCap work. | Learning a temporal prior from motion capture is established. Explain why the present question concerns image-estimator errors, matched projected references, and fidelity of 2D motion quantities. Do not describe the repository as a complete reproduction of every TPAMI experiment. |
| Zhu et al., **MotionBERT**, ICCV 2023. [Paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhu_MotionBERT_A_Unified_Perspective_on_Learning_Human_Motion_Representations_ICCV_2023_paper.pdf), [author project](https://motionbert.github.io/), [official implementation](https://github.com/Walter0807/MotionBERT) | Pretrains motion representations by recovering 3D motion from noisy partial 2D observations, then transfers to multiple tasks. | Noisy-to-clean motion pretraining alone is not the novelty. The stronger question is whether aligned latent targets add identifiable value over coordinate learning when accuracy and preservation are evaluated separately. |
| Hedlin, Rhodin, and Yi, **A Simple Method to Boost Human Pose Estimation Accuracy by Correcting the Joint Regressor for the Human3.6m Dataset**, CRV 2022. [Paper](https://arxiv.org/abs/2205.00076), [official implementation](https://github.com/ubc-vision/joint-regressor-refinement) | Shows that an inaccurate SMPL-to-joint regressor can mislead evaluation and that regressor correction can improve measured pose accuracy without retraining the pose model. | Supports joint-convention mismatch as a plausible confound. It studies a different mapping from this experiment's SMPL-H/COCO comparison; it does not establish that calibration explains the current results. |
| Assran et al., **I-JEPA**, CVPR 2023. [Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Assran_Self-Supervised_Learning_From_Images_With_a_Joint-Embedding_Predictive_Architecture_CVPR_2023_paper.pdf), [official implementation](https://github.com/facebookresearch/ijepa) | Predicts representations of target image regions from context to learn semantic representations. | Provides architectural context for latent prediction. Semantic representation quality on recognition tasks is not evidence of accurate coordinate, amplitude, or event-timing measurement. |

The manuscript should use the closest three or four references to define the gap rather than enumerate every motion foundation model. S-JEPA, SmoothNet, and the joint-regressor study explain the objective comparison and calibration control particularly well; PoseBERT and MotionBERT establish that motion-capture-based restoration has prior art. None supplies missing evidence for this study's own movement-preservation claim.

## Seven title directions

| Candidate title | Strength | Main risk |
| --- | --- | --- |
| **Can Paired Synthetic Supervision Restore Pose Trajectories Without Distorting Movement?** | Directly states the scientific question without assuming success. | A question title still requires a meaningful empirical answer in the completed paper. |
| **Paired Synthetic Supervision for 2D Pose Restoration: An Audit of Accuracy and Motion Fidelity** | Best fit for the current mixed evidence and diagnostic contribution. | “Audit” may sound narrow unless the evaluation produces insight beyond this pipeline. |
| **What Do Synthetic Pose Targets Teach? Calibration and Motion in Trajectory Restoration** | Highlights the unresolved mechanism and strong calibration alternative. | Too mechanism-specific until calibration checks are completed. |
| **Testing Latent Prediction for Motion-Preserving Pose Restoration** | Makes the representation-learning comparison visible to ICLR readers. | The body must retain direct and coordinate controls so the study is not architecture-led. |
| **From Pose Accuracy to Movement Fidelity: Evaluating Paired Synthetic Supervision** | Gives the broader measurement motivation while staying cautious about results. | Needs a precise definition of fidelity early in the abstract. |
| **Learning to Correct Pose Trajectories: Accuracy, Timing, and Bilateral Structure** | Accessible and names the motion quantities of interest. | Does not foreground synthetic supervision or the latent-versus-coordinate comparison. |
| **When Pose Coordinates Improve: Auditing Movement Preservation in Synthetic Learning** | Organizes the observed coordinate gains around the unanswered preservation question. | Can sound like a general negative result; limit conclusions to the studied setting. |

For the current evidence, candidates 1, 2, and 5 are the strongest. Avoid titles such as “Motion-Preserving JEPA” or “Reliable Gait Measurement from Synthetic Supervision,” which imply preservation or real-data validity that has not been established.

## Adversarial novelty assessment

The easiest rejection argument is that an established temporal architecture receives synthetic clean targets, is compared on two development people, and does not beat simple controls consistently. A larger raw-pose improvement does not answer this argument when an untrained encoder with a fitted readout obtains nearly the same reduction. Nor does reduced coordinate error establish preserved amplitude or timing. These are limitations of the available evidence, not wording problems.

The potentially stronger contribution is a controlled separation of three explanations: better agreement with the target joint convention, benefits from learning temporal representations, and benefits from correctly paired latent supervision. Same-motion rendered conditions help isolate observation corruption; trained-on-one-family/evaluated-on-another comparisons can test transfer of the correction. However, a held fitting family should not be described as an extractor with certified untouched pretraining exposure. Person-level uncertainty and reference-eligible motion support are central to this argument.

The laterality study can motivate retaining signed left–right relationships, and the original synthetic study can motivate revisiting the supervision target after a lesson selector failed to add useful headroom. Neither provides evidence that the new restorer improves real movement measurement. A real-data finding available only as a summary must remain a summary-backed motivation, separate from locally recomputed data and from synthetic demonstrations. The abstract should not combine their denominators, imply the studies share a validation population, or turn retrospective oracle headroom into an achieved gain.

The most defensible present interpretation is that clean-target coordinate agreement improves substantially, while the incremental contribution of paired JEPA and preservation of movement remain unestablished. It would overreach to conclude that latent prediction fails in general, that synthetic supervision necessarily erases motion, or that systematic calibration explains the gains before the controls are run.

## Improvements that revision can make now

1. Lead with the measurement question, then explain a paired example: one underlying motion yields a projected joint reference and an imperfect image-estimator track. State that the projection is an anatomical proxy.
2. Define a joint-embedding predictive architecture once as predicting hidden representations of matching clean motion, then explain that a separately fitted coordinate readout reconstructs trajectories.
3. Report the informative comparisons: raw tracks, initialized readout, coordinate learning, direct restoration, aligned and shuffled latent targets. A large gain against raw tracks alone cannot establish the role of pretraining.
4. Include the evaluation scale and evidence scope near numerical claims: two development people, one seed, synthetic source screen, and no verified real temporal transfer.
5. State unavailable support as unavailable measurement. Zero supported timing records is not zero timing error and does not, by itself, locate the failure in the reference or prediction.
6. Label recalculations from rounded report values as exploratory summary-derived analyses. Do not describe interval recomputation, fresh training, or access to raw trajectories unless those steps occurred.

These changes can raise accuracy, clarity, and positioning scores. They cannot legitimately raise statistical-rigor scores to the level of a replicated, adequately sampled experiment.

## Improvements requiring completed analysis or new experiments

| Priority | Evidence needed | What the result would settle |
| --- | --- | --- |
| First | Repair source verification and finish the cached-data calibration and reference-against-itself checks. | Whether simple trained corrections explain the coordinate gain; how much timing support the reference could attain. These remain pending until successful artifacts exist. |
| First | Inspect preselected reference and predicted trajectories, separating clean and corrupted conditions, with missed/extra peaks and support counts. | Whether preservation failures reflect signal observability, prediction distortion, or both. |
| Next | More independent people and motions, repeated seeds, converged fits, and a fair compute comparison. | Whether the small objective differences persist beyond the present screen and training recipe. |
| Next | Independent landmark-convention validation and dense real temporal annotations. | Whether improved synthetic agreement transfers to movement measurement with a defensible target. |
| Conditional | New viewpoints or longer windows with sufficient reference events, declared before comparison. | Whether an observable version of the motion task changes the conclusions. This is a new experiment, not a repair of the old result. |

A publishable negative result would need to establish a repeatable failure or confound under meaningful controls. A positive result would need to demonstrate both improved coordinate accuracy and preserved movement on adequate, independently supported trajectories. Neither outcome should be promised in a deadline abstract.

## Pipeline illustration checks

Use one left-to-right data path with a compact training-only branch for the projected reference. Make the shared motion, camera, and timestamps explicit. Keep the frozen image estimator distinct from the trainable temporal encoder, and show the clean-reference teacher only during pretraining. Draw the teacher update separately from the data flow and explain EMA as a running average of encoder weights in the caption. The frozen-encoder coordinate-readout stage needs its own label: otherwise viewers may mistake this for end-to-end coordinate training.

The deployment path must receive only observations, native scores, and missingness information. Reference coordinates and reference validity cannot feed deployment. Label the output “restored 2D trajectory,” and list coordinate error, amplitude, timing support, and signed left–right relations as evaluation targets rather than verified achievements. The diagram should not imply a real-data transfer result or completed calibration check.

## Independent review of the generated pipeline

*Added 19 September 2026 after direct visual inspection of the rendered PNG and inspection of the SVG generator. Reviewed artifact: [pipeline version 01](images/abstract-training-pipeline-20260919-v01.svg). No figure artifact was edited during this review.*

The rendered diagram has no visible clipping, overlapping text, crossed data arrows, or confusing teacher-update direction. Its four stages are easy to follow, and the pale input/target colors remain subordinate to the labels. The diagram correctly isolates projected references from inference, distinguishes fixed image pose estimators from the learned temporal encoder, depicts the teacher as an average of online weights, and labels the reference branch as detached. The frozen-encoder readout stage, held-from-fitting ViTPose family, and pending calibration/real-video checks are explicit.

The remaining improvements are limited but concrete:

1. Explain in the caption that the coordinate readout is supervised using projected **training** references and then fixed for inference. “Fit on training pairs only” is correct, but the supervised target and loss are not drawn in that stage.
2. Expand “L/R separation” to signed ankle separation in the caption. The evaluation covers an image-plane proxy; the abbreviation alone can suggest broader bilateral coordination measurement.
3. Preserve readability at manuscript scale. At a seven-inch full-width placement, the 17-pixel body labels in the 1,400-pixel-wide source become approximately six-point text. The current figure works well as a screen document illustration; a more compact, less wordy version would suit the main paper better than shrinking it into a single column.
4. Add an empirical paired trajectory figure only after valid artifacts are available. This schematic explains processing and cannot establish the claimed restoration-versus-preservation behavior.

**Recommended figure score: 7.5/10 for the present manuscript package.** Layout and scientific separation are strong; print-scale readability and the absence of empirical motion visualization remain material weaknesses. The same unchanged figure should receive the same score in every abstract revision. There is no visual or scientific blocker to using the figure in the draft record.

## Independent critique of abstract revision 03

*Added 19 September 2026. Reviewed [revision 03](abstract-v03-jepa-vs-coordinates.md) without editing it; suggestions are for revision 04.*

The opening now gives ICLR readers a clear comparison: what does predicting learned target features add when coordinate methods receive the same privileged synthetic supervision? This is more specific than generic temporal denoising and is well situated relative to S-JEPA and motion-capture pretraining. Its potential contribution remains an evaluation design and an unresolved mechanism, however. A consistent, general empirical conclusion has not yet emerged from two development people and one seed. Calling the result inconclusive is accurate; calling the experimental design itself a decisive representation-learning advance would be premature.

The current abstract avoids claiming proven preservation, a causal calibration explanation, or a general failure of JEPA. Four changes would sharpen it:

1. **Restore the missing baseline.** The sentence reporting a 35.6–45.3% reduction never states “relative to unchanged tracks.” The percentage needs this qualifier every time it appears.
2. **Lead with the matched comparison.** Report the below-1% HRNet/ViTPose differences and 14.0% RTMPose degradation first. If space permits, say that the reported paired intervals for the two small improvements include zero. Keep the two-person, one-seed scope adjacent so those intervals are not read as persuasive population-level evidence.
3. **Describe the timing failure operationally.** “No eligible timing comparisons” can imply that reference ineligibility is established. What is known is that no paired-JEPA records satisfy the combined reference-and-prediction peak rules. The pending reference-against-itself check is needed to distinguish weak reference support from prediction failure.
4. **Reduce historical overhead.** The predecessor results motivate matched controls and laterality-sensitive measurement; they do not independently validate the new restorer. Keep one compact sentence connecting their findings to this experimental design, and reserve the full evidence categories and provenance for the surrounding record.

The rubric's evaluation score of 3.5/10 and reproducibility score of 5.5/10 should remain unchanged unless evidence changes. A higher positioning score can be justified by the clearer same-supervision comparison, but should not be described as increased novelty of the underlying algorithm. The present 7.5/10 submission-fit score concerns abstract framing and scope; it must not be interpreted as readiness of the completed empirical paper. Fixing the missing baseline and timing ambiguity can improve claim accuracy and clarity without improving statistical rigor.

## Final independent review before revision 07

*Added 19 September 2026. Reviewed [revision 06](abstract-v06-iclr.md) and the [evidence and rubric record](abstract-20260919-evidence-and-rubric.md) without editing either.*

The source interpretation is sound. The evidence ledger correctly separates controlled motion-capture relabeling, synthetic image adaptation, summary-reported restoration, analytic fixtures, and a separate real-video classification probe. Its discussion of Hedlin et al. supports a convention-mismatch hypothesis without diagnosing this experiment. Its account of S-JEPA does not imply that JEPA necessarily suppresses side or timing information. The related-work comparison identifies an attribution question under shared supervision rather than claiming that synthetic denoising or temporal refinement is new.

Revision 06 remains credible about the limited evidence: there is no preservation success claim, no extrapolation to clinical measurement, and no claim that the initialized readout proves learning is unnecessary. The argument is clearer than earlier drafts, but the main empirical outcome still cannot identify why the coordinate gains occur or establish general behavior beyond this pilot. A polished abstract does not remove that contribution limitation.

For revision 07, make only targeted refinements:

1. Restore the explicit three-part scientific scope in the opening: movement amplitude, timing, and left–right relationships. The current phrase “preserving movement” is broader and less informative.
2. Identify the historical laterality evidence as controlled motion-capture summaries, so the reader does not mistake it for naturally occurring real-video tracking errors.
3. Describe the initialized control as a fitted **temporal** readout on an untrained **frozen** encoder. Otherwise a reader may infer that it is a tiny static calibration, which it is not.
4. Name the quoted intervals as reported 95% intervals, retaining the nearby two-person, one-seed limitation.
5. Choose a title balancing the preservation question with the representation comparison. “Paired Synthetic Pose Restoration: Testing Accuracy and Movement Preservation” fits the current scope without announcing a positive preservation finding. The revision 06 title makes latent prediction more central than the user's primary movement-fidelity question.

**Score recommendation: retain 70.50/100; do not increase it for revision 07.** The total is arithmetically correct. Claim clarity and positioning are already credited in revision 06, and these final refinements are too small to warrant another half-point change. Evaluation 3.5/10, reproducibility 5.5/10, and figures 7.5/10 should remain fixed. Scientific-insight and conference-fit scores describe the quality of the research question and its presentation; the manuscript has not yet supplied the generalizable empirical finding that would make it a convincing completed main-track contribution.
