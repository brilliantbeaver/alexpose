# Geometry, symmetry, and learned gait representations: progress through 13 September 2026

This report follows the seven notebooks in the `multiple-sclerosis` folder, checking their retained outputs against the saved evaluation files and the code that implements the methods. The reporting date identifies this review; it does not imply that a new training experiment ran on 13 September. The accompanying [slide tutorial](../slides/slides.md) presents the same evidence visually.

## 1. Why study walking through geometry and symmetry?

Walking involves coordinated movement across a connected body. A knee angle depends on the positions of the hip, knee, and ankle, while the relationship between hip and knee angles changes throughout a stride. These relationships give us a way to describe movement that is more meaningful than a list of pixel locations. Translating a person across an image changes every landmark coordinate but leaves the angles between their limbs unchanged. Geometry therefore helps us decide which changes describe the walk and which describe how it was recorded.

Symmetry adds a second question: how do the two sides of the body relate over time? During walking, the legs generally alternate, so comparing the left and right leg at the same instant can confuse ordinary alternation with an impairment. A useful comparison must specify whether it concerns average posture, movement amplitude, or trajectories aligned to corresponding stages of a stride. It must also allow for ordinary variation between people and for camera projection. A two-dimensional video cannot recover all the three-dimensional relationships of a moving body.

![A schematic comparison of left and right motion across a stride](../images/progress-0913/01_symmetry_cycle.svg)

*Figure 1. Synthetic curves explain why corresponding gait phases must be matched before interpreting a left–right difference. They are not measurements from these videos.*

Clinical studies motivate these measurements without supplying an answer for our collection. Lewek and colleagues found greater arm-swing asymmetry in a small study of early Parkinson’s disease, even though arm-swing magnitude alone did not significantly distinguish their groups. This motivates measuring the relationship between sides alongside the amount of movement. [Lewek et al., 2010](https://pubmed.ncbi.nlm.nih.gov/19945285/)

For multiple sclerosis, Pau and colleagues studied how hip, knee, and ankle angles vary together through the gait cycle. Their clinical cohort showed altered coordination, with differences related to disability level. An angle–angle plot, which traces one joint angle against another over time, can therefore reveal relationships that a single average angle misses. These findings concern a clinical three-dimensional measurement setting; they do not establish the same associations in our internet-video collection. [Pau et al., 2022](https://pubmed.ncbi.nlm.nih.gov/35305428/)

These observations motivate the project’s connection to physical AI: can a learned representation retain useful relationships among body parts and their movement? A **representation** is the vector of numbers an encoder produces from an input sequence. Here, Skeleton Joint Embedding Predictive Architecture (**S-JEPA**) trains an encoder by predicting representations of hidden parts of a skeleton sequence. The original method supplies the learning idea; our implementation applies it to two-dimensional landmarks plus visibility. [Abdelfattah and Alahi, ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04755.pdf)

The current notebook demonstrations and the retained five-fold reference establish a modest but useful boundary. They demonstrate a source-separated learning and evaluation pipeline, and they expose substantial weaknesses in both the representation and its measurement conditions. They have yet to demonstrate a disease-specific symmetry mechanism, a learned physical dynamics model, or clinical diagnostic validity.

## 2. The questions and the evidence available to answer them

The starting scientific hypothesis is that relationships among body parts, including left–right differences and coordination over time, contain information associated with the collection’s gait labels. A corresponding **null hypothesis** is that the proposed motion representation does not improve held-out macro-F1 over the chosen simpler comparison under a source-separated evaluation. This operational test concerns prediction quality; it does not directly measure all information retained by an encoder. For the main comparison, define the difference as:

\[
\Delta = F1_{\mathrm{S\text{-}JEPA}} - F1_{\mathrm{angle\text{-}based\ RF}}.
\]

For this score comparison, the operational null is \(H_0: \Delta \leq 0\). A claim of improvement would need evidence that this difference is positive under a predeclared evaluation, with uncertainty assessed at the appropriate independent unit. A more specific symmetry hypothesis would require an additional comparison: removing or altering symmetry information should change held-out performance in the predicted direction while relevant recording conditions remain controlled.

These are explicit organizing hypotheses for this report. The retained notebooks do not document a completed, preregistered null-hypothesis test, and we should not retrospectively describe the notebook sequence as one. Instead, the sequence progressively asks whether the data can be represented consistently, whether training behaves as intended, whether further training changes prediction, and whether simple alternative explanations remain competitive.

| Question | Evidence available | What follows |
|---|---|---|
| Can the videos yield consistent landmark arrays? | Notebook 01 retains 47 usable sequences, each with 33 landmarks and three channels. | The pipeline produces usable inputs, with exclusions and measurement limits described below. |
| Does masking give all joints opportunities to contribute? | Notebook 02 retains a successful mask-coverage demonstration. | The repaired sampler addresses permanent exclusion of joints from context. |
| Can the self-supervised objective train without an obvious constant representation? | Notebook 03 retains falling loss and effective-rank diagnostics. | Training is numerically productive; useful disease information still needs a separate test. |
| Does additional self-supervised training improve the demonstration’s probe? | Notebook 04 reports macro-F1 of 0.600 before and after another 400 updates. | No improvement is observed on that one held-out fold. |
| Do learned embeddings separate labels visually? | Notebook 05’s saved execution stops with a missing-checkpoint error. | No retained embedding plot or silhouette result answers this question. |
| Did the retained S-JEPA reference outperform the angle-based Random Forest? | Frozen five-fold results, before the later centering fix: 0.438 versus 0.667 pooled macro-F1. | The observed difference is −0.228; a benefit was not demonstrated for that implementation. |
| Could simpler measurements explain similar predictive performance? | Order-insensitive pose summaries and visibility controls perform strongly. | Sequence-specific and symmetry-specific explanations remain unestablished. |

### The saved five-fold result predates one important repair

The reference run called **R1** already included major repairs, but its loss updated the running target center once for each example in a batch. The later correction scores every example against the same center and updates that center once per batch. The [AR-5 entry in the phase ledger](03-0802-PHASE_LEDGER.md) explicitly states that the frozen R1 run was **not rerun after this correction**. Its macro-F1 of 0.438 therefore describes the retained earlier objective, not a complete five-fold evaluation of today's trainer.

The current notebook demonstrations use the subsequently corrected training path and retain separate fold-0 outputs. This distinction matters more than file modification dates: the latest implementation, latest notebook demonstration, and latest complete five-fold reference are different kinds of evidence. A five-fold run of the current objective remains outstanding.

![The questions, comparisons, and revisions in the study](../images/progress-0913/02_hypothesis_loop.svg)

*Figure 2. The sequence uses negative results and simpler controls to refine the next question. It does not represent a completed preregistered testing program.*

### A map of the current notebooks

Several filenames and older overview passages describe an earlier training design. The current executable cells are more reliable guides to the work actually performed.

| Notebook | Latest retained execution and role |
|---|---|
| [00 — overview](../00_overview_and_video_gallery.ipynb) | Counts 49 input clips from 37 source videos and embeds example videos. Its normal-only pretraining and progressive VICReg roadmap is outdated. |
| [01 — pose extraction](../01_pose_extraction_from_raw_video.ipynb) | Demonstrates extraction, cleaning, normalization, and the resulting 47-sequence cache. |
| [02 — masking and tokens](../02_anatomical_mask_and_tokenization.ipynb) | Demonstrates repaired stochastic masks and the 264-token layout. |
| [03 — self-supervised training](../03_sjepa_model_and_pretrain_normal.ipynb) | Trains the repaired model for 800 updates on all three label categories within fold 0’s training sources. Labels are unused by the objective. |
| [04 — further training and probing](../04_progressive_finetune_ms_pd_vicreg.ipynb) | Loads those weights, trains for another 400 updates with restarted training state, and compares frozen supervised probes. No VICReg loss runs in this path. |
| [05 — representations](../05_representation_visualization.ipynb) | Retains a `FileNotFoundError` before embedding extraction; subsequent visualization and silhouette cells are unexecuted. The required checkpoints now exist, but their presence does not replace the missing execution. |
| [06 — capstone](../06_capstone_rf_vs_sjepa.ipynb) | Reads the frozen five-fold 1,000-update run and shortcut controls; separately executes a 500-update, single-fold demonstration. |

## 3. Step 1 — count sources before creating training examples

Notebook 00 begins with the unit that produced the observation: the source video. A long source can contribute several clips, and each clip can later contribute many overlapping windows. These additional examples are useful for training, but they do not create additional independent people.

| Dataset stage | Normal clips / sources | MS clips / sources | PD clips / sources | Total clips / sources |
|---|---:|---:|---:|---:|
| Input video collection | 20 / 17 | 12 / 12 | 17 / 8 | 49 / 37 |
| Usable cached sequences | 19 / 16 | 11 / 11 | 17 / 8 | 47 / 35 |

The cache contains **8,693 frames**: 1,840 normal, 2,918 MS, and 3,935 PD. Under the 32-frame laptop window configuration, these yield **481 overlapping windows**: 92 normal, 167 MS, and 222 PD. These are frame and window counts, not additional independent observations.

Notebook 01 excludes `diCVwltkV5M.mp4` and `JUsbspdRRJA.mp4` because too few frames meet its detection requirement. The usable collection contains one fewer normal source and one fewer MS source. The **normal**, **MS**, and **PD** names are dataset labels; the retained provenance does not verify every participant’s diagnosis, medication, severity, or identity across different uploads.

Source grouping is therefore a meaningful protection with a specific scope. All clips sharing a recorded source identifier stay together. Two different source videos may still contain the same person, and different sources can share a recording setup. The 35 source groups must not be described as 35 verified independent participants.

The acquisition metadata also warrants attention. Earlier provenance work records the MS clips as 60-fps, square-format videos. Such a pattern can correlate with the label even after source separation. The retained evaluation includes several related controls, but its duration control does not directly test frame rate or resolution, and none of these controls establishes how much of the model’s prediction is caused by acquisition conditions.

**Shape and separation at this step.** The source inventory preserves the relationship among sources and their clips. Pose-quality exclusion changes the analyzed cohort and is reported explicitly. No cross-video normalization or learned feature fitting occurs here; the eventual learning splits are fixed at the source level before windows are used for training.

## 4. Step 2 — turn video into geometry, with a precise account of what survives

### Landmarks, missing data, and time

The pose front end estimates 33 landmarks per sampled frame. Each cached sequence has shape `(T, 33, 3)`, where `T` is the number of retained frames and the channels are **x-coordinate, y-coordinate, and visibility**. Visibility is the detector’s estimate of how visible a landmark is. It is not a depth coordinate, so these arrays are not three-dimensional motion capture.

Notebook 01’s example retains a `(155, 33, 3)` array, with 61% of sampled frames fully detected before cleaning. Its cleaned and normalized arrays retain that shape; the displayed normalized x-coordinate range is −9.95 to 2.88, illustrating that normalization does not constrain coordinates to a fixed unit box.

The loader samples approximately 15 frames per second by selecting every `round(source_fps / 15)` frames, with a minimum stride of one. This integer rule does not guarantee a common physical time interval. For example, sampling every other frame of a 24-fps recording produces 12 samples per second, even when the cache stores the requested value of 15. Timing-dependent interpretations therefore require verified effective timestamps.

Cleaning rejects sequences when fewer than 30% of frames have finite x-coordinates for all joints. It trims completely undetected ends and interpolates missing values within each remaining channel. Although the helper’s prose refers to short gaps, its implementation does not cap gap length. Interpolated positions are estimates, and their movement should not be presented as directly observed motion. Visibility is interpolated too; subsequent normalization leaves that cleaned channel unchanged.

### What normalization preserves

For frame \(t\), let \(p_t\) be the midpoint of the hips and \(s_t\) the distance between hip midpoint and shoulder midpoint, bounded below by a small positive constant. A landmark \(x_{t,j}\) becomes

\[
\widetilde{x}_{t,j}=\frac{x_{t,j}-p_t}{s_t}.
\]

Subtracting the same point and dividing by the same positive scalar for every joint preserves angles and ratios of distances **within that frame’s two-dimensional pose**, apart from numerical precision. It also retains joint identity. This is the defensible sense in which normalization preserves shape.

The full dataset’s geometry is not preserved exactly. Pelvis centering removes the root’s travel through the image, and a different scale in each frame can alter changes in normalized limb displacement over time. Absolute pixel distance and calibrated walking speed are not retained in the normalized stream. Camera viewpoint, perspective, occlusion, and estimation error also remain. The cache separately stores cleaned pixel coordinates, so it is more accurate to say that root travel is removed from the normalized model input than that it has disappeared from every stored artifact.

![What pelvis centering and torso scaling preserve and remove](../images/progress-0913/06_normalization.svg)

*Figure 3. A shared translation and scale preserve relative shape within a frame, while root travel is removed from the normalized input.*

### From sequences to tokens

The laptop configuration uses 32-frame windows with a stride of 16 frames. Overlapping windows preserve their internal frame order but reuse observations. A short sequence is padded by repeating its final frame. Three cached clips require this padding: their lengths of 15, 29, and 26 frames require 17, 3, and 6 repeated frames, respectively. These repeated frames have no accompanying validity mask in the current windowing path; the attention mask used to hide JEPA targets serves a different purpose. A trailing segment shorter than the next complete window can be omitted.

Within each window, four consecutive frames of one landmark become one token. Four frames times three channels produce a 12-number input vector. Eight temporal blocks times 33 landmarks give **264 tokens**, which are projected into 96-dimensional learned vectors:

\[
(B,32,33,3)\rightarrow(B,8,33,12)\rightarrow(B,264,96).
\]

Here `B` is the number of windows in a training batch. The first rearrangement preserves the supplied values and their indexing; the learned projection and subsequent attention do not guarantee preservation of distances between examples. Separate joint and time tags identify each token’s location, with token index `time_block × 33 + joint`.

![The arrangement of frames and landmarks into tokens](../images/progress-0913/09_tensor_flow.svg)

*Figure 4. The initial rearrangement keeps values and their indexing; the learned 96-dimensional projection is not a distance-preserving guarantee.*

**Training boundary.** Extraction, per-clip cleaning, and per-frame normalization use no fitted population statistics. Training windows are created from the training clips for the relevant fold. A model can therefore transform a held-out clip at evaluation time without fitting on it. Any later scaler, feature selection rule, or classifier is fitted separately using only training data.

## 5. Step 3 — make hidden-part prediction a meaningful learning task

### The current anatomical masking design

The original fixed mask permanently hid both shoulders and the complete legs: 12 landmarks across every temporal block. That design deprived those positions of direct context exposure. The current repaired sampler instead draws connected anatomical regions over spans of time, separately for each example. Shoulders and legs receive a sampling preference, while every joint can appear in both context and target over training. The preference is an anatomical prior, not a measured finding that these joints are the optimal diagnostic set.

Notebook 02 retains six distinct masks for six examples, each with visible and hidden tokens. In its bank of 512 masks, the mean hidden-token fraction is 0.63. The smallest joint-visible coverage is 0.74 and the smallest joint-target coverage is 0.81. These latter values count whether a joint appears at least once in each role somewhere across a window’s time blocks; they are not complementary fractions of individual tokens. A joint can satisfy both conditions within one mask.

The configured target ratio is 0.60. Sampling whole regions can exceed that requested fraction, so the empirical 0.63 should not be silently relabeled as an exact 60% mask. The coverage demonstration supports the sampler’s intended behavior, without establishing that anatomical masking improves classification relative to a randomized alternative.

![Permanent hidden joints compared with changing anatomical masks](../images/progress-0913/10_mask_repair.svg)

*Figure 5. Rotating context and target roles addresses permanent exclusion from context. These mask patterns illustrate the mechanism.*

### The two encoders and the predictor

The **context encoder** processes visible tokens from an augmented window. A **predictor** combines those features with the joint and time positions of hidden slots. The **target encoder**, often called the teacher, processes the complete reference window. Training compares predictions with the teacher’s representations at the hidden positions; it does not try to classify the health label during this stage.

The teacher is updated by an exponential moving average of the context encoder’s parameters. This means each update combines a small amount of the current context encoder with the previous teacher. Gradients do not optimize the teacher directly. The prediction loss compares distributions across feature dimensions. A **temperature** controls how concentrated each distribution is: lower values put more weight on its largest components. The **running center** is a moving average of target features that is subtracted before making the target distribution, reducing persistent bias toward particular feature dimensions. These distributions concern hidden features, rather than probabilities of the three health labels. These mechanisms stabilize the learning targets, but their presence alone cannot certify useful representations.

The predictor’s joint and time tags repair another consequential defect: hidden slots previously lacked enough positional information to produce distinct position-conditioned predictions. These tags supply the identity of the missing measurement, rather than allowing the predictor to treat every hidden slot as interchangeable. Current code also updates the target center once per optimization batch. The frozen R1 objective used the earlier per-example update, so its score does not measure this correction.

![The context encoder, predictor, and moving-average teacher](../images/progress-0913/12_teacher_student.svg)

*Figure 6. The current objective predicts teacher features at hidden joint-time locations; condition labels enter only in the downstream classifier.*

### Shape-preserving augmentation has a limited meaning

The augmented view uses an in-plane rotation of up to 15 degrees, scale variation between 0.9 and 1.1, translation up to 0.1 normalized units per axis, and a horizontal reflection with probability 0.5. The same sampled transform applies throughout a window; reflection also swaps left and right landmark identities and their associated visibility values. There is no coordinate-noise augmentation in the current `random_view` implementation.

These operations preserve each pose’s Euclidean shape up to the chosen similarity transformation or reflection. They do change the coordinates and the training distribution. Reflection also changes side identity, which matters if a future task predicts the affected side. In the current training path, the reflected context and the reference target are not accompanied by an explicit reflection-alignment objective. Augmentation therefore does not establish **equivariance**, the property that a known transformation of the input produces a corresponding, predictable transformation of the representation.

A gait-symmetry study would need to measure that property, or directly evaluate phase-aligned left–right motion, before attributing successful classification to symmetry. None of the retained runs reports that experiment.

## 6. Step 4 — learn from training sources, then test a separate supervised readout

### What the 800-update demonstration establishes

Notebook 03 uses fold 0’s **37 training clips from 29 sources**, yielding 395 windows. All three label categories are present in this training partition. The filename’s reference to normal-only pretraining describes an earlier design.

The repaired trainer samples windows with probability inversely proportional to the number of windows contributed by their source. Every training source therefore has the same expected sampling probability, while longer sources no longer dominate because they supply more windows. This preserves the available observations and source boundaries, but deliberately changes their optimization weights. Finite random draws do not give every source identical realized exposure.

The label accompanies each dataset item for later evaluation but is ignored by the self-supervised loss. The trainer uses 800 optimizer updates in the retained full laptop demonstration. Mean loss over the first five updates falls from **4.990** to **1.112** over the last five, and the final effective-rank diagnostic is **9.5**.

**Effective rank** summarizes how broadly the batch’s feature variation is distributed across independent directions. Here it is calculated from centered, pooled batch embeddings. With a batch of 32, its mathematical rank ceiling is at most 31, even though each feature vector has 96 dimensions. A value near nine is evidence against a constant representation in that measured batch. It does not prove that the features encode gait pathology or that the network has enough capacity.

### What the additional 400 updates establish

Notebook 04 loads the checkpoint from notebook 03 and trains on the same training sources for another 400 updates. It does not pass the saved training state back through `resume_state`. The weights continue from the checkpoint, but the optimizer, running center, random streams, and learning-rate and teacher schedules restart. This is a **warm start from trained weights**, rather than a bit-for-bit continuation of one uninterrupted 1,200-update schedule.

Both checkpoints are evaluated through the same readout procedure. The frozen target encoder processes full windows; representations are averaged over a seeded, fixed set of sampled target-token positions, then across windows to produce one vector per clip. This readout is not the old fixed set of 12 clinical joints. The seeded mask selects 161 of 264 token positions and touches all 33 joints at some time blocks. A standard scaler and class-balanced logistic regression are fitted on the training embeddings. Logistic regression is the small supervised classifier that converts the frozen vector into one of the three labels; this is where the diagnosis labels enter.

| Notebook 04 comparison | Held-out clips / sources | Macro-F1 | Final batch effective rank |
|---|---:|---:|---:|
| Probe on the 800-update checkpoint | 10 / 6 | 0.600 | 9.5 |
| Probe after another 400 updates with restarted state | 10 / 6 | 0.600 | 10.4 |

![The unchanged fold-0 probe score after further training](../images/progress-0913/15_continuation_result.svg)

*Figure 7. Both frozen probes score 0.600 on ten held-out clips from six sources. The second stage reloads weights and restarts training state.*

Further training changed the representation diagnostics without improving the displayed classification score. The same score does not establish equivalent representations or identical clip predictions. This small, repeatedly inspected fold provides a teaching comparison, not a reliable estimate of the effect of training longer.

### Where VICReg fits in the history

The old progressive pipeline contained a class-aware term described as VICReg, making its objective dependent on diagnosis labels. It operated on within-class residual variation rather than directly separating class centers. The repaired `train_sjepa_v2` path used by these notebooks removes that term. Standard VICReg encourages two views to agree while maintaining feature variation and reducing redundant dimensions; it does not use disease labels to push three condition clusters apart. [Bardes, Ponce, and LeCun, ICLR 2022](https://arxiv.org/abs/2105.04906)

The current notebook-04 comparison is additional label-free training followed by supervised frozen probes. It supplies no empirical comparison of VICReg against the repaired objective, and no evidence for an advantage from progressively introducing normal, MS, and PD examples.

## 7. Step 5 — inspect representations without turning pictures into validation

Notebook 05 plans to compare the two checkpoints with a visibility-based representation, using t-SNE and UMAP to draw two-dimensional scatter plots. These methods compress high-dimensional data into a plane and distort some relationships as a consequence. Their pictures can help identify examples worth investigating, but a visually separated group is not a held-out classification result.

The notebook’s retained execution stopped while loading `artifacts/sjepa_ssl.pt`. No embedding extraction, nuisance comparison, t-SNE/UMAP output, or silhouette score follows that error. Both `sjepa_ssl.pt` and `sjepa_ssl_continued.pt` now exist on disk. This review therefore records an incomplete retained visualization execution, rather than asserting that the prerequisite remains unavailable.

The proposed **silhouette score** compares within-label distances with distances to other label groups. The notebook would calculate it over the whole collection, including training examples, so even a completed result would be descriptive. It would not establish generalization. The planned visibility representation also differs from the capstone’s control: notebook 05 uses per-joint visibility means and standard deviations, whereas the frozen control uses global mean and standard deviation plus per-joint means.

This missing result matters to the research narrative. We cannot show invented clusters as evidence that training discovered normal, MS, and PD structure. The tutorial’s illustrative symmetry drawings are explicitly schematic, while its result plots use retained numerical outputs.

## 8. Step 6 — compare against a transparent geometric baseline

### What the Random Forest actually receives

The classical branch starts with cleaned pixel landmarks, computes lower-limb joint angles, and summarizes each clip. A joint angle is calculated from the vectors joining three relevant landmarks; its value is unchanged by image translation and uniform scale. Projection into two dimensions still affects how well it corresponds to an anatomical angle.

The `GaitFeatureVector.from_joint_angles` path populates **15 core feature slots**: six mean angles, three absolute left–right differences of mean angles, and six angle ranges. Each range is the maximum minus the minimum among valid angles over the clip, so it is sensitive to outliers and to which frames pass the landmark-confidence screen. The much larger feature container also names speed, stride, timing, and extended symmetry fields, but this calling path leaves those fields at their defaults. Constant columns are removed using training-set variance.

For example, the knee asymmetry feature is

\[
A_{\mathrm{knee}}=|\overline{\theta}_{L,\mathrm{knee}}-\overline{\theta}_{R,\mathrm{knee}}|.
\]

It measures a difference in average two-dimensional knee angle. Two sides can have similar averages while differing in movement amplitude, timing, or the shape of their trajectories, so this feature does not exhaust the meaning of gait asymmetry. No retained angle-by-angle ablation establishes which joint contributes most to the classification result.

A current upstream implementation issue further limits interpretation: `left_ankle_range` is assigned from `right_ankle_stats`, duplicating the right-side range. The feature path therefore does not provide six independently computed ranges as its names suggest. The frozen scores below remain the saved historical measurements; this report does not silently replace them with a corrected baseline. A future correction requires a separately versioned, paired reevaluation, and its effect on the historical result cannot be inferred from the code alone.

The Random Forest fits 100 decision trees, each limited to depth five, with class weighting and a fixed random seed. Both its scaler and its constant-column selection are fitted on training clips only. This provides a transparent comparison between a limited set of geometric summaries and a learned vector, without assuming either one measures disease mechanisms directly.

### The five-fold source boundary

The retained main evaluation uses the saved `g1` registry. It divides source groups into five folds, attempting to balance labels while keeping each source intact. For every fold, the self-supervised encoder is newly built and trained exclusively on that fold’s training clips. The scaler and supervised probe are then fitted there, and the test clips are embedded only for prediction. The Random Forest uses the identical partition.

| Fold | Training clips / sources | Test clips / sources | Test normal / MS / PD |
|---|---:|---:|---:|
| 0 | 37 / 29 | 10 / 6 | 3 / 2 / 5 |
| 1 | 37 / 28 | 10 / 7 | 4 / 3 / 3 |
| 2 | 38 / 28 | 9 / 7 | 4 / 2 / 3 |
| 3 | 38 / 28 | 9 / 7 | 4 / 2 / 3 |
| 4 | 38 / 27 | 9 / 8 | 4 / 2 / 3 |

![Source groups kept together in training and evaluation](../images/progress-0913/08_source_split.svg)

*Figure 8. Windows inherit the source assignment. Pretraining and supervised fitting both exclude the held-out source groups.*

The saved predictions contain one **out-of-fold** prediction per clip: each clip is predicted by a model whose fitting excluded that clip’s recorded source. Combining these 47 predictions produces the pooled result. Source separation applies to self-supervised learning as well as to the supervised head; ignoring labels during pretraining would not justify including held-out source videos.

Pooling at the clip level gives sources with multiple clips more weight in the reported score. Source-uniform training does not change that evaluation weighting. Likewise, source grouping does not remove repeated inspection of these development folds or guarantee independence between different uploads of the same person.

## 9. Results — the complete five-fold reference and the current demonstrations

The main results come from [the frozen R1 result file](../artifacts/runs/r1_g1_1k_s42/results.json) and its [47 out-of-fold prediction records](../artifacts/runs/r1_g1_1k_s42/oof.json). The run uses the laptop encoder, 32-frame windows, 1,000 updates per fold, target ratio 0.60, and the recorded seed setting of 42. This R1 run predates the once-per-batch centering correction described in section 2. These settings identify one retained run; they do not constitute repeated-seed replication or a complete evaluation of the current objective.

**Accuracy** is the fraction of clips with the correct label. **Recall** for a condition is the fraction of its true clips retrieved by the classifier. **Precision** is the fraction of predictions of that condition that are correct. F1 combines precision and recall through their harmonic mean; **macro-F1** averages the three condition-specific F1 values equally.

| System | Pooled macro-F1 | Accuracy | Correct clips | PD recall |
|---|---:|---:|---:|---:|
| Angle-based Random Forest | 0.667 | 0.660 | 31 / 47 | 10 / 17 = 0.588 |
| Retained R1 S-JEPA with frozen probe, before centering fix | 0.438 | 0.447 | 21 / 47 | 4 / 17 = 0.235 |

![Retained five-fold macro-F1 and accuracy for the two branches](../images/progress-0913/18_paired_results.svg)

*Figure 9. Paired clip-level results from the retained R1 run, before the centering correction. These point estimates have no accompanying inferential interval.*

The observed S-JEPA-minus-Random-Forest macro-F1 difference is **−0.228**. This retained run provides no empirical support for an advantage from its learned representation; the corresponding five-fold comparison after the centering correction is still unavailable. It also does not establish that JEPA cannot help with more suitable data or a different, prospectively evaluated learning design.

### The errors differ by condition and by model

| True condition | Support | RF precision / recall / F1 | S-JEPA precision / recall / F1 |
|---|---:|---:|---:|
| Normal | 19 | 0.684 / 0.684 / 0.684 | 0.688 / 0.579 / 0.629 |
| MS | 11 | 0.727 / 0.727 / 0.727 | 0.353 / 0.545 / 0.429 |
| PD | 17 | 0.588 / 0.588 / 0.588 | 0.286 / 0.235 / 0.258 |

Confusion matrices below use true labels as rows and predicted labels as columns, both ordered **normal, MS, PD**. Each entry counts clips.

| True label | RF → normal | RF → MS | RF → PD | S-JEPA → normal | S-JEPA → MS | S-JEPA → PD |
|---|---:|---:|---:|---:|---:|---:|
| Normal | 13 | 1 | 5 | 11 | 3 | 5 |
| MS | 1 | 8 | 2 | 0 | 6 | 5 |
| PD | 5 | 2 | 10 | 5 | 8 | 4 |

![True versus predicted label counts for the retained models](../images/progress-0913/19_confusion_matrices.svg)

*Figure 10. Rows are true labels and columns are predictions. The matrices expose different error patterns without attributing their causes to physiology.*

S-JEPA predicts MS for eight PD clips, its largest off-diagonal count. The Random Forest’s PD errors go mainly to normal, with five such errors versus two to MS. The notebook’s prose suggesting that both models share the same dominant PD-to-MS confusion is therefore inaccurate.

These error counts motivate targeted inspection of the corresponding clips, recording conditions, and landmark quality. They do not establish that MS and PD share a particular measured asymmetry, or that the classifier confused them for a physiological reason. No retained case-level analysis connects these errors to a specific joint-angle trajectory.

### Training diagnostics narrow one explanation

Across the five folds, final effective rank ranges from **7.71 to 9.51**, embedding standard deviation from **0.566 to 0.643**, and final loss from **1.135 to 1.584**. The reported teacher–student drift is finite, from **0.0141 to 0.0171**. This diagnostic averages one minus the cosine similarity between corresponding encoder parameter vectors before the teacher update; a larger value means their parameter directions differ more. These measurements argue against the simplest failure in which all measured embeddings become constant. They leave open whether the retained variation describes movement, static appearance, detector behavior, or some mixture.

The older S-JEPA macro-F1 value near 0.570 belongs to the historical pipeline. Several mechanisms changed during the repairs, including masking, positional information, label use, sampling, and schedules. The difference between that score and 0.438 is not a controlled estimate of how much any one defect inflated performance.

### The live notebook demonstration is a separate result

Notebook 06 also retains **0.644 S-JEPA macro-F1 and 0.915 RF macro-F1** for a single fold-0 demonstration using 500 self-supervised updates; the final effective rank is 8.9. This output belongs to a ten-clip test partition, with a different execution, centering implementation, and training budget from the frozen evaluation. Recomputing the frozen fold-0 predictions gives RF macro-F1 of 0.915 and S-JEPA macro-F1 of 0.467. The RF matches the live demonstration, while the two S-JEPA results cannot isolate the effect of training duration because other conditions differ. It must not replace the five-fold scores or be combined with them as an additional independent fold.

## 10. Step 7 — challenge the motion explanation with simpler controls

The [saved control results](../artifacts/eval/g1/E0_results.json) evaluate summaries that discard the order of the frames. This asks whether good label prediction requires modeling the sequence in detail. The following table reports both classifiers for every control, avoiding the capstone scoreboard’s selection of the higher score after inspecting both.

| Control and its actual input | Logistic regression pooled macro-F1 | Random Forest pooled macro-F1 |
|---|---:|---:|
| Mean normalized pose: 66 coordinate means | 0.511 | 0.657 |
| Mean and standard deviation of normalized pose: 132 values | 0.703 | 0.624 |
| Visibility: global mean and standard deviation, plus 33 landmark means | 0.551 | 0.636 |
| Retained-frame count: the same count supplied twice | 0.353 | 0.469 |
| Nine distances in pixel coordinates from the per-joint median pose | 0.406 | 0.440 |

![All logistic regression and Random Forest control results](../images/progress-0913/20_shortcut_controls.svg)

*Figure 11. Both classifiers are retained for each control. The summaries discard frame order but may retain clinically relevant posture or movement amplitude.*

The mean-and-standard-deviation logistic model has the highest listed point estimate, **0.703**. Its difference from the angle-based Random Forest is **0.036**, with no saved uncertainty interval establishing a reliable advantage. The angle-based Random Forest is consequently the stronger of the two principal branches, but it is not the highest-scoring system in the full retained comparison.

These controls require careful interpretation. Coordinate standard deviations can retain movement amplitude even after time order is discarded. Mean pose can contain both clinically relevant posture and camera-dependent projection. Visibility can depend on recording quality, occlusion, and movement itself. Their performance creates a substantial alternative explanation to detailed temporal learning, without proving that every successful prediction is a camera shortcut.

Two inherited control names overstate their scope. `duration_acq` contains duplicated retained-frame count; it measures neither calibrated duration nor the original frame rate and resolution. `body_proportion` contains raw pixel distances, rather than scale-free body ratios. A rigorous follow-up needs controls matched to those intended questions before claiming acquisition invariance or the removal of body-size effects.

## 11. What statistical rigor has been achieved, and what remains to be established?

The strongest safeguards are concrete: recorded sources remain disjoint across each fold; pretraining excludes held-out sources; the repaired self-supervised loss ignores condition labels; scalers, variable-column selection, and supervised heads fit on training clips; paired models share a saved registry; and predictions are retained at the clip level. These choices make the existing result inspectable and prevent several common forms of direct leakage.

They do not by themselves yield confirmatory statistical inference. The collection is small and already inspected during development. The five training folds overlap, clips within a source are related, and source identifiers do not verify participant independence. A nominally fixed seed also does not replace recorded initial states and complete execution provenance.

No confidence interval, permutation-test result, or repeated-seed uncertainty estimate accompanies the frozen reference run. An older roadmap mentions a bootstrap interval, but a supporting retained computation was not located; it is excluded from this report’s findings. The RF fold-mean macro-F1 of **0.651 ± 0.145** in the E0 artifact uses the standard deviation across five folds, with divisor five. It differs from pooled macro-F1 of 0.667, and the standard deviation is not a confidence interval. Scores from unequal test partitions should not be mixed by substituting one summary for the other.

Similarly, **one third is not a universal chance macro-F1** for three imbalanced classes. It is the expected accuracy of a uniform random guess, while expected F1 depends on class frequencies and the prediction rule. A meaningful null comparison should specify that rule or obtain its reference distribution through an appropriate group-level procedure. The updated figures omit the old generic chance-F1 line.

A suitable next analysis would predeclare the score, comparator, and resampling unit. A paired source-level bootstrap could resample source groups and preserve all their clips, recomputing the difference from paired predictions. That would quantify uncertainty conditional on the retained predictions and source grouping; it would not include the full variability of retraining or solve participant overlap. A label-permutation experiment would need group-level permutations, defensible exchangeability assumptions, and refitting of label-dependent steps. A claim about new people ultimately needs verified participant separation, and a claim about new clinical settings needs a separate site or cohort.

These are proposed analyses. Their results are not fabricated here to make the current report appear statistically complete.

## 12. The next experiments follow from the unresolved questions

### First, make measurement and reproduction dependable

A separately versioned five-fold run of the current once-per-batch centering objective is needed to close the main evaluation gap. Before treating subsequent comparisons as reproducible, record the complete code state, initialization, cache identity, and training configuration.

The next data revision should retain effective timestamps, record missing-data provenance, bound interpolation gaps, and mark repeated padding frames. Root-relative joint geometry and root motion should be stored as separate signals, with camera-motion limits made explicit. This would preserve useful shape information while allowing timing and movement extent to be examined rather than assuming that normalized coordinates contain everything needed.

The classical feature path needs its left-ankle range assignment corrected and checked against an independently computed angle summary. The visualization notebook needs a fresh execution with recorded checkpoint identities. These operations should generate new, versioned results without overwriting the frozen evidence used in this report.

### Then, ask which geometric information contributes

An informative sequence would begin with angle means, then add movement ranges, then left–right differences, and finally phase-aware coordination measures. Each addition should be evaluated on the same source or participant partitions with choices confined to training data. An **ablation** removes or adds a defined component so that its contribution can be compared. Without this, a high score from a feature bundle cannot identify the useful angle or symmetry measurement.

![Proposed tests of joint angles, reflection, and coordination](../images/progress-0913/22_proposed_symmetry_tests.svg)

*Figure 12. These are proposed controlled comparisons, not completed symmetry results.*

Phase alignment must be explicit. For example, left and right knee trajectories can be compared after matching corresponding points of a stride; aligning by arbitrary frame number would create a different measurement. Angle–angle trajectories could test coordination within a leg. These are proposed extensions inspired by the geometric motivation, and none has a retained result in the current notebook series.

### Test the learning objective against those measurements

With the measurement pipeline fixed, compare the same frozen-probe protocol before and after pretraining, across declared training budgets and repeated initializations. Test time-shuffled inputs to assess reliance on order, and compare transformation-consistency errors with downstream prediction. A reflected input should be paired with a defined transformation of joint identities and targets before interpreting representation consistency as symmetry.

External motion pretraining and reduced-label experiments remain reasonable research directions, but this collection has not demonstrated their benefit. Additional scale should be evaluated as an intervention with a fixed comparison, rather than presented as an inevitable solution to the current negative result.

## 13. What has been learned so far

The notebook sequence now supports a clearer account of both progress and failure. Pose extraction produces a usable, source-indexed collection; repaired masking and position tags make the hidden-part task better specified; and held-out-source exclusion extends through self-supervised learning. The measured features vary, yet another training stage leaves the demonstration’s probe unchanged, and the retained five-fold learned model underperforms the angle-based baseline. The corrected centering objective still needs a complete five-fold evaluation.

The control experiments sharpen the scientific question. Strong performance from summaries that discard time order means that sequence modeling has not yet established an added benefit. The visualization stage remains incomplete, and no experiment isolates the contribution of anatomical symmetry. Geometry remains a well-motivated way to describe gait, but its role in successful prediction must be measured through targeted comparisons.

For physical AI, the useful outcome is an explicit set of relationships to preserve and test: joint identity, relative pose, timing, coordinated motion, and the behavior of representations under transformations. The next phase can build on those definitions while keeping every claim tied to a measured comparison and every proposed extension visibly separate from the retained evidence.

## 14. Evidence and reproducibility record

| Evidence | Location and use |
|---|---|
| Notebook execution inventory | [Notebook evidence review](../artifacts/reviews/2026-09-13-notebook-evidence.md), with cell-level retained outputs and incomplete execution status. |
| Implementation review | [Method evidence review](../artifacts/reviews/2026-09-13-method-evidence.md), covering transformations, training, labels, masks, splits, and limitations. |
| Main evaluation | [R1 results](../artifacts/runs/r1_g1_1k_s42/results.json), [predictions](../artifacts/runs/r1_g1_1k_s42/oof.json), [run manifest](../artifacts/runs/r1_g1_1k_s42/run_manifest.json), and [completion record](../artifacts/runs/r1_g1_1k_s42/COMPLETED.json). |
| Shared partition | [Fold registry](../artifacts/eval/g1/fold_registry.json), source-grouped, five folds, split seed 42. |
| Baseline and controls | [E0 results](../artifacts/eval/g1/E0_results.json) and [RF predictions](../artifacts/eval/g1/E0_RF_oof.json). |
| Data and transformations | [Data handling](../sjepa/data.py), [tokenizer](../sjepa/tokenizer.py), and [augmentation](../sjepa/augment.py). |
| Model and objective | [Models](../sjepa/models.py), [repaired masks](../sjepa/masking_v2.py), [repaired trainer](../sjepa/train_v2.py), and [losses](../sjepa/losses.py). |
| Evaluation implementation | [Frozen-run driver](../scripts/scripts_r1_repaired.py), [baseline wrapper](../sjepa/classical.py), [control definitions](../scripts/scripts_phase0_provenance.py), and [metric definitions](../sjepa/eval.py). |

Read-only checks recomputed all pooled metrics from the 47 saved prediction records, verified zero source overlap in each fold, and confirmed that all 47 current pose-cache file hashes and the video manifest match the saved E0 provenance record. These checks establish consistency of the retained inputs and predictions; they do not supply an exact historical code snapshot.

This documentation update preserves the notebooks and saved experiment outputs. The evidence review checks their content and internal relationships; it does not claim that all notebooks were rerun or that the frozen training has been reproduced on the present environment. The [adversarial review and revision record](../artifacts/reviews/2026-09-13-adversarial-review.md) accompanies the final documentation.
