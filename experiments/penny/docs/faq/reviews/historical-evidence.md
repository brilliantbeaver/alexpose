# Fundamentals and anatomical-prior evidence review

Scope: the latest `gavd3` notebooks, especially 00–06. Earlier `gavd` and `gavd2` material is relevant only to terminology and the retained handcrafted reference. This note is supporting evidence for the FAQ, not a fresh experiment or a claim that the notebooks were rerun.

## What can be stated from the implementation

|Reader's question|Supported explanation|Evidence location|
|---|---|---|
|What does JEPA mean?|Joint Embedding Predictive Architecture describes a learning arrangement in which an encoder converts observations into learned numerical representations and a predictor estimates a target representation from a partial view. “Joint” means the representations belong to the same learning arrangement; in the acronym it does not mean an anatomical joint.|S-JEPA paper, Section 2; notebook 00, cells 3 and 8–9.|
|What does the S stand for?|Skeleton. The observations are sequences of estimated body landmarks. The GAVD model follows the paper's core learning graph with documented changes in data, scale, masking and augmentation.|Notebook 00, cells 8–10; `gavd3/README.md`.|
|What is an encoder?|A trainable numerical mapping from an input sequence to feature vectors. Here a linear layer embeds each joint-time patch and Transformer layers combine information across patches. These features are learned; their coordinates do not have assigned clinical meanings.|Notebook 00, cell 15, `SkeletonPatchEncoder`.|
|What is a joint-time patch or token?|One body landmark across four consecutive frames becomes one patch. Its four x/y/relative-z samples, 12 values, pass through a linear layer to become a token. With 64 frames and 33 landmarks there are 16 temporal segments and 528 tokens.|Notebook 00, cells 3 and 15.|
|Why is position included?|Learned temporal and landmark position vectors identify which segment and anatomical landmark a token belongs to. Equal coordinate samples at different positions can therefore enter the Transformer differently.|Notebook 00, cell 15, `positioned_tokens`.|
|What does a Transformer do here?|Self-attention combines information among the available tokens, allowing features for one landmark and moment to depend on other landmarks and moments. Attention weights alone do not establish a physiological explanation or causal relationship.|Notebook 00, cell 15, `nn.TransformerEncoder`.|
|Why two encoders?|The view encoder receives an augmented clip with selected tokens removed. The target encoder receives the complete, unmasked clip and produces contextual representations at every position. Only the selected target positions enter the training loss.|Notebook 00, cells 8 and 15, `SJEPAGait.forward`.|
|What is an EMA target encoder?|Exponential moving average means that each target parameter retains a large fraction of its previous value and adds a small fraction of the updated view parameter. The target is initially copied from the view encoder, receives no gradients, and changes by this averaging rule.|Notebook 00, cell 15, `update_target`; notebook 04, cell 14.|
|What is stop-gradient?|The target branch's output contributes to the numerical loss while its parameters receive no gradient from that loss. This does not mean the target encoder is permanently unchanged: the EMA update still changes it.|Notebook 00, cells 15 and 17.|
|What is the predictor?|A separate Transformer receives visible features plus learned placeholder tokens at the hidden positions, with space-time position information, and outputs latent vectors for the hidden positions.|Notebook 00, cell 15, `SkeletonPredictor`.|
|What does the loss compare?|The target and predictor vectors are converted into distributions across latent feature dimensions using softmax. Cross-entropy penalizes disagreement between those distributions. These are not probabilities of Parkinson's disease, stroke or any other diagnosis.|Notebook 00, cell 9 and `sjepa_cross_entropy` in cell 15.|
|What are centering and sharpening?|Latent centering subtracts a running average from the target feature vector. Sharpening uses a low softmax temperature to make its distribution more concentrated. This target centering is separate from physically centering poses on the hips during preprocessing.|Notebook 00, cells 9 and 15; notebook 04, cell 6.|
|What is a frozen probe?|After pretraining, the target encoder is held fixed and a supervised model is trained on its output. Notebook 06 uses a Random Forest, a collection of decision trees. This is a nonlinear probe, so do not call it the paper's linear evaluation.|Notebook 06, cells 10–17.|
|What is pooling?|A variable collection of token vectors is summarized into one vector per sequence. The implementation concatenates means and standard deviations over all valid tokens and over valid tokens in the ten selected landmarks.|Notebook 05, cells 17–18; notebook 06, cells 15–16.|
|Why 384 numbers?|The real model has 96 latent channels. Four summaries of 96 channels give a 384-dimensional sequence vector. These are 384 learned summary features, not 384 measured gait parameters.|Notebook 04, cell 14; notebook 05, cell 18.|
|Is self-supervised learning free of labels?|The pretraining loss uses no condition labels, because targets come from the full input clip. The normal-only training cohort is selected using folder annotations, and the later classifiers do use condition labels.|Notebook 04, cell 12; notebook 06, classifier cells.|
|Does ordinary S-JEPA predict the future?|The original 00–06 objective estimates hidden representations within the same observed clip. Notebook 10 introduces a separate future-prediction variant, with a prefix-only view. Claims about future prediction must identify that variant.|Notebook 00, cell 18; notebook 10.|

The primary paper directly supports the full-input target encoder, partial view encoder, predictor, EMA update, and latent cross-entropy. It studies skeletal action recognition. Its benchmark performance does not validate GAVD condition classification or the anatomical prior. [S-JEPA, ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04755.pdf), Sections 2–3 and Figure 2.

## What the data preparation preserves and changes

Notebook 01 makes one manifest row per annotated sequence and retains the original annotation fields. The health-condition name comes from the selected dataset folder, and the notebooks do not independently verify a person's diagnosis. Several sequences can be fragments of the same YouTube upload; separate sequence IDs therefore do not establish independent people or recordings.

Notebook 02 reads the GAVD bounding box to crop the intended walker and converts detected landmarks back to full-frame coordinates. It uses `result.pose_landmarks`, with relative z, rather than calibrated ground-truth 3D motion-capture coordinates. A coordinate plot should label depth as relative depth and lengths as normalized or proxy units unless another calibration has been demonstrated. The notebook checks first, middle and last image/skeleton overlays because a wrong walker or wrong frame span can produce plausible numerical tensors.

A detector failure remains on the annotated sequence timeline with missing coordinates and zero visibility. This avoids speeding up motion by dropping failed detections. Preprocessing interpolates only internal coordinate gaps of at most four original frames, preserves the original validity indication before temporal resizing, centers each frame at the hip midpoint or available hip, and scales using body width. It then resizes the sequence to 64 frames in real mode. That final resizing changes the physical relationship between one model step and elapsed time, so an embedding's apparent temporal pattern is not automatically cadence in steps per minute. The validity array itself is resampled and thresholded; avoid describing it as an unchanged mask in the final resized tensor.

Validity-masked pooling excludes invalid patches from the final mean and standard-deviation calculations. A patch is valid only when all four resampled frame indicators are true for its landmark. However, incomplete tokens still participate in Transformer context, and short gaps may contain interpolated coordinates. Masked pooling therefore does not prove immunity to pose-detector artifacts. Notebook 06's missingness-only classifier asks whether patterns of detection failure can themselves predict the condition folders.

Sources: notebook 02, cell 7, `detect_pose_in_crop` and `extract_one_sequence`; notebook 04, cell 6, `interpolate_low_visibility`, `center_and_scale`, `temporal_resize`, `prepare_sequence`; notebook 05, cell 18, `pooled_embeddings`; notebook 06, cells 13–14.

## The ten-landmark anatomical prior

The allowed target set is `[11, 12, 23, 24, 25, 26, 27, 28, 31, 32]`: left and right shoulders, hips, knees, ankles and foot-index landmarks. It is the union of high-priority, region-specific mappings in the four files under `neuroscience/data`. Each feature mapping identifies landmarks needed by an existing anatomical feature calculation or software proxy. It does not assign a disease to a landmark, prove a neural mechanism, or establish diagnostic thresholds.

|Mapping family|Why these regions entered the mapping|Interpretation limit|
|---|---|---|
|Parkinson's|Bilateral hip/knee trajectory symmetry and ankle motion/timing feature mappings.|Some feature names are aliases or proxies. The mapping explains region selection, not condition discrimination.|
|Stroke|Left/right knee and ankle angle chains, plus ankle-derived timing and trajectory summaries.|Different functions can share the same landmarks. Similar anatomical inputs do not make the derived quantities independent.|
|Cerebral palsy|Shoulder–hip–knee, hip–knee–ankle and knee–ankle–foot chains used to compute hip, knee and ankle angles.|Video-estimated angles remain camera-dependent; the pelvic-tilt field includes a software proxy.|
|Myopathic|Proximal hip/knee angle chains, shoulder-based trunk proxy and bilateral ankle-separation proxy.|Myopathy is a broad condition family; the stored trunk and step-length fields are not direct validated measurements of those quantities.|

The exact software caveats matter to explanations:

- `stride_time_cv` aliases `step_regularity_cv` in the earlier feature code. CV means coefficient of variation, generally standard deviation divided by mean, but these field names do not create independent evidence.
- `step_length_cv` is populated from `step_width_std` in the referenced feature vector. Calling it measured step-length variability would be incorrect.
- The prior maps flag legacy COCO ankle indices 15/16 as a format error for BlazePose, where those indices are wrists and the ankles are 27/28. This is a caveat about the legacy feature implementation, not evidence that the S-JEPA sampler uses wrists.
- `trunk_lean_angle` has a shoulder-symmetry fallback, while `pelvic_tilt_mean` has a hip-symmetry fallback. These names cannot establish that actual trunk lean or pelvic tilt was measured.
- The ten landmarks remain available as context when they are not chosen as targets. The full target encoder uses all 33 landmarks; the encoder has not been restricted to a ten-point skeleton.

The core sampler uniformly selects valid eligible joint-time patches without reading motion magnitudes. Its nominal `mask_fraction=0.60` applies to the eligible set, with the same hidden-token count across a batch determined by its smallest eligible count. For a fully valid 64-frame sample, 160 of 528 tokens are eligible and 96 are hidden, around 18.2% globally. This differs materially from the original paper's 90% global masking. A figure should distinguish eligible landmarks from the subset hidden on any particular draw.

The rationale that low-motion regions could remain informative is a hypothesis motivating the prior. A benefit from that prior requires a matched mask comparison. Notebook 09 performs such comparisons, including a motion-aware alternative; therefore “motion-aware masking is absent” should be scoped to the main model in 00–06.

Sources: notebook 03, cells 4–14 and `uniform_neurologic_mask`; `neuroscience/data/PD_keypoint_mapping.md`, `CP_keypoint_mapping.md`, `MYO_keypoint_mapping.md`, `STROKE_keypoint_mapping.md`; notebook 09.

## A defensible intellectual journey

No preregistration or original formal null-hypothesis declaration was found in notebooks 00–06. For an educational story, say explicitly that the following are working null hypotheses used to organize the retained experiments, rather than preregistered statistical tests. A measured performance difference is not automatically a statistically supported rejection, especially with few independent source videos and many comparisons.

|Stage|Question that motivates the next step|Working null or alternative explanation|What the notebook can establish|
|---|---|---|---|
|00|Can a partial skeleton representation predict complete-context target features?|The learning graph is incorrectly wired, or a constant representation minimizes its loss.|Smoke shapes, finite loss and gradient/EMA checks verify implementation paths; real representation usefulness remains open.|
|01–02|What observation does each row represent, and did the detector recover the intended walker?|Source fragments are mistaken for independent observations, or extraction distorts anatomy/timing.|Manifest counts, source-video census, original FPS/frame numbers, overlays and coverage make the input accountable.|
|03|Can an anatomical prior be made into a precise sampling rule?|The chosen regions offer no advantage, or the implementation quietly hides forbidden joints.|Sampler assertions verify the rule. They do not show an anatomical-prior performance benefit.|
|04|Does normal-only latent training avoid a constant representation?|Loss decreases through a degenerate solution.|Feature variation, pairwise similarity and entropy can detect important collapse symptoms, without proving a physiologically useful latent space.|
|05|What similarities emerge before condition labels are fitted?|Retrieval and distances mainly reflect the recording, person or detector pattern.|Latent matching, nearest neighbors and a normal centroid offer descriptive checks. Similarity is not a validated anomaly diagnosis.|
|06|Do frozen representations help distinguish dataset conditions on these splits?|Condition scores can be explained by imbalance, recording overlap or detector missingness.|Classifier scores, confusion matrices, counts and shortcut controls describe performance under the named protocol.|
|07–09|Would results survive source grouping, interpretable probes and a matched mask comparison?|The apparent success depends on source overlap; interpretable gait quantities are not readily recoverable; the prior has no advantage.|These controls narrow the claim and explain failures; the FAQ should report their retained measurements separately.|
|10|Does a prefix-conditioned predictor yield an informative future-target discrepancy?|Errors mainly measure distribution shift, teacher-target effects or generic deviations from the single normal source.|The variant supports a future-latent proof of concept; source/condition confounding still prevents a clinical forecast or diagnosis claim.|

Avoid inventing a chronological account of investigators discovering each result or a series of experiments chosen before results were seen. The notebook order provides a logical educational journey, while the retained artifacts supply the actual measured outcomes.

## Corrections the FAQ should incorporate

1. Notebook 04's Markdown scale table still lists “20 real” epochs, while its executable configuration distinguishes 20 quick epochs from 300 recommended epochs. Use checkpoint configuration and retained training history when describing the completed run.
2. Describe the main model as an adaptation of S-JEPA, not an official model reproduction or a reproduction of the paper's clinical results. The original paper studies action recognition.
3. Distinguish latent target centering from coordinate centering, predicted latent distributions from condition probabilities, and missing-token prediction from notebook 10's future-prediction variant.
4. Explain normal-only pretraining overlap with downstream normal test clips. Holding the Random Forest test rows out does not erase earlier encoder exposure.
5. Treat source-disjoint scores as stress tests when class/source coverage is inadequate. They cannot resolve the absence of independent normal recordings by renaming the evaluation.
6. Keep the handcrafted comparison brief and label it a system comparison. Different pose extraction, preprocessing and feature engineering prevent attributing a score gap to encoder design alone.
7. Label schematic skeletons as schematic and projected/unscaled trajectories as normalized or relative. Avoid drawings of specific disease patterns that appear to be measured representative samples unless they are traced to actual artifacts and carry the relevant source/detector caveats.

## Verification performed

The review read Markdown and executable cells from notebooks 00–06, inspected the 33-point extractor and preprocessing implementation, checked the four final anatomical mapping tables, and opened the final ECCV S-JEPA paper and project page. It did not run model training, download additional gait videos or modify notebook results. Later notebook outcomes are delegated to the separate retained-artifact evidence review.

## Independent notebook 10 methodological review

### The prefix mask is narrower than a strictly causal pipeline

Notebook 10, cells 7 and 10, removes all 33 landmarks in the last four temporal segments before the view Transformer. Its model therefore receives 396 prefix tokens and predicts 132 future token representations, with a full-clip teacher providing the targets. This is a separate whole-body future-mask experiment; it does not preserve the main experiment's ten-landmark-only masking restriction. The outputs for all four future segments are generated in one predictor pass. No recursive use of predicted latents as subsequent context is implemented, so “rollout probe” should be explained as a probe of predicted future segments, not a demonstrated autoregressive rollout.

There is a material qualification to the notebook's “past alone” language. Shared preprocessing in cell 5 computes median shoulder/hip body width from the complete clip before the future tokens are removed. Its fallback pelvis position also uses a full-clip median, and short-gap interpolation can use observed coordinates on either side of the later context/future boundary. Consequently the raw future can affect the prefix's transformed coordinates even though future tokens never enter the view Transformer. A strictly causal evaluation would fit normalization and interpolation rules using only the observed prefix, then apply that fixed transform to future targets.

This dependence was checked with a small deterministic fixture that executed the notebook's preprocessing functions without running a model. Two fully valid 64-frame skeletons had identical first 48 frames and different shoulder widths only in frames 48–63. `prepare_sequence(..., frames=64)` produced different prefixes, with maximum absolute difference 0.6135197 in normalized coordinates. This verifies a preprocessing information path; it does not quantify how much it changes the retained forecast results. Describe the current result as prefix-masked latent prediction with full-clip preprocessing, and list strictly causal preprocessing as a required follow-up.

### What the forecast comparisons test

The normal forecast errors in cell 20 use the same twelve sequences used to train the causal variant. They are training-distribution, in-sample references rather than error on unseen normal gait. The phase-bin baseline excludes the query sequence when building its latent template, but the forecaster itself was trained on that sequence. Their errors therefore compare different exposure conditions. Moreover, the baseline pools teacher features across all joints and broadcasts one averaged vector to each joint, whereas the forecaster retains joint positions. A large error difference against this deliberately simple template does not isolate predictive dynamics or demonstrate generalization.

Abnormal clips in cell 22 were excluded from training, but changes in condition, source recording and person occur together. Larger abnormal error is an observed distribution difference. It cannot establish whether gait pathology, recording identity, detector behavior or the contrast between seen normal clips and unseen abnormal clips is responsible. The teacher itself sees the complete abnormal clip and may encode its source differences in the targets.

Cross-entropy is measured in nats, units based on natural logarithms. It includes the target distribution's own entropy as well as prediction-target disagreement. Consequently this metric is not automatically a calibrated measure of movement abnormality, and error ratios across condition groups should be described as latent cross-entropy ratios. A useful next control would report a prefix-persistence or other equally exposed forecasting baseline, target entropy and divergence, and held-out normal recordings.

### The phase-probe metric needs a plain label

Cell 24 saves `circular_correlation` as the mean of `cos(absolute circular phase error)`. This quantity describes phase alignment: 1 means exact agreement, around 0 means broadly unaligned angles, and negative values indicate systematically opposing phase estimates. It is not the usual centered circular correlation coefficient. The FAQ should call it mean cosine phase agreement, explain that lower angular error is better, and avoid repeating the column name as a statistical correlation interpretation.

The target phase is calculated from the full-clip Hilbert transform of normalized left-ankle horizontal motion. It is a signal-derived proxy, not independently verified foot-contact phase. The persistence reference uses a Hilbert transform of the context prefix only. The Ridge decoder holds each sequence out when fitting that decoder, while the underlying forecaster has already seen all twelve normal sequences. This distinction matters when explaining what leave-one-out testing does and does not hold out.

### Artifact-root provenance

The baseline 00–06 artifacts exist under both `gavd3/cache/artifacts/real` and `gavd3/work/artifacts/real`. Later 07–10 result tables and their checkpoints are present under `gavd3/work/artifacts/real`, not the cache root. The notebook environment cell defaults to `work/artifacts` and allows `GAVD3_ARTIFACT_DIR` to select another root. Each FAQ numerical claim should identify which retained artifact was read; matching copies should be verified before mixing baseline results from one root with later results from the other. This review did not assume the two baseline copies are identical.
