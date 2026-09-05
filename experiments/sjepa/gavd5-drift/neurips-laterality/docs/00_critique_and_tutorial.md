# Bilateral Symmetry as a Geometry Audit for a Skeleton World Model

## A plain-language guide to the laterality v2.1 study

> **Read this first.** This guide explains the current experimental design. Earlier notebooks used a transductive design in which the encoder had already seen the evaluated sequences. Those old numbers are useful only as historical motivation and are not results from this study. The current protocol trains a fresh model without each test fold. Its paper run and governance reviews are not complete, so this guide explains what will be measured rather than announcing an outcome.

## 1. What question are we asking?

A video of a walking person can be converted into a sequence of estimated body landmarks. Each landmark represents a point such as a shoulder, knee, ankle, heel, or foot point. Together, these points form a simplified moving skeleton.

The study asks whether a self-supervised model learns something about the skeleton's left–right structure. "Self-supervised" means that the model learns by predicting hidden parts of its input rather than by receiving the final left-versus-right target during training. After training, a small prediction model, called a read-out or probe, tests whether the learned features contain useful laterality information.

The question has two parts. First, can the learned features predict which side moves more on source videos excluded from training? Second, does the prediction reverse correctly when the input skeleton is mirrored? A convincing result needs both predictive usefulness and the correct mirror behavior.

## 2. The essential vocabulary

- **GAVD** means the Gait Abnormality in Video Dataset. The condition names supplied with the dataset are annotations, not diagnoses made by this project.
- A **pose sequence** is a time-ordered series of estimated landmark coordinates derived from part of a source video.
- A **source video** is the original video associated with one or more pose sequences.
- A **cohort** is the set of sequences that pass the pre-written quality checks.
- An **encoder** turns coordinates into learned numeric features.
- A **read-out** or **probe** is a simpler model fitted to the encoder features to predict the study target.
- A **fold** is one portion of a dataset used in rotation for training, validation, or testing.
- **Cross-validation** repeats the analysis while rotating which fold is held out.
- A **seed** fixes a stream of pseudorandom choices. Repeating training with several seeds shows sensitivity to initialization and other randomized steps.
- **Leakage** occurs when information from the test set influences training or model selection. Leakage can make performance look better than it really is.
- **Transductive** evaluation allows the representation learner to see the evaluated inputs. The previous exploratory experiment was transductive.
- **Fold-local** or **inductive** evaluation trains a separate encoder without the test fold. Laterality v2.1 uses this design.

## 3. How do raw files become the audited cohort?

The frozen inventory begins with 666 GAVD annotation files describing 103 source videos. A matching pose archive exists for 642 annotations. The remaining 24 annotations do not have a pose archive and therefore cannot enter the pose analysis.

Quality control then asks whether each available pose sequence contains enough observed landmark information. It also asks whether the laterality target can be computed and whether enough complete four-frame blocks remain for the model. These rules are locked before the model results are inspected. A sequence is not included or excluded because its target is favorable or unfavorable.

After these checks, 625 sequences from 93 source videos remain. Seventeen of the 642 available pose sequences are excluded. The resulting cohort is saved with a content fingerprint so later notebooks can verify that they loaded exactly the audited data.

## 4. What is the laterality target?

The target is a coordinate-derived summary of relative motion on the two sides. It is not a medical score. It uses five left/right landmark pairs: shoulders, knees, ankles, heels, and foot-index points.

For one pair, imagine that the typical left-side speed is 3 units and the typical right-side speed is 2 units. The normalized contrast is

$$
\frac{3-2}{3+2}=0.2.
$$

A positive result means more left-side motion under this formula, a negative result means more right-side motion, and zero means equal motion under the formula. The program calculates one contrast for each of the five pairs and averages them.

The comparison uses only **paired-valid transitions**. Both landmarks must be visible at the start and end of the same transition. This matters because comparing different moments on the two sides could confuse visibility with motion. Short gaps may be filled for model input, but filled values do not contribute to the target.

Mirroring provides an implementation check. A mirror flips the horizontal coordinate and swaps the anatomical left and right landmarks. The target should then have the same magnitude and the opposite sign. Mirroring twice should restore the original pose. These facts are tested before training results are interpreted.

## 5. Why split by source video instead of sequence?

One source video can yield many pose sequences. If those sequences were split independently, the training set might contain one excerpt while the test set contained another excerpt from the same video. Camera position, clothing, background, extraction artifacts, and the person's movement could then appear on both sides of the split. That would be video-level leakage.

The protocol first creates a table with one row per source video. It assigns each source to a fold and only then expands that assignment to its sequences. Every clip, mirrored version, and augmentation from the same source inherits the same fold. Consequently, an outer-test video and all of its derived data remain outside the corresponding training set.

This protection stops at the video boundary. GAVD does not provide a stable person identifier, and the project does not infer identity. The same unidentified person could appear in two different videos. The method therefore supports a held-out-video claim, not a held-out-person claim.

## 6. The five outer train/test folds

The 93 eligible source videos are divided into five folds. The division is stratified, meaning that the dataset annotation counts are kept reasonably balanced across folds. The annotations help balance the split; they are not prediction targets for the encoder.

In each outer fold, about 80% of the source videos train the encoder and fit the eventual read-out. The other roughly 20% form the untouched test set.

| Outer fold | Training sources | Test sources | Training sequences | Test sequences |
|---:|---:|---:|---:|---:|
| 0 | 74 | 19 | 436 | 189 |
| 1 | 74 | 19 | 443 | 182 |
| 2 | 74 | 19 | 553 | 72 |
| 3 | 75 | 18 | 548 | 77 |
| 4 | 75 | 18 | 520 | 105 |

The source counts are almost equal, while the sequence counts vary. This is expected because some videos produce many more sequences than others. Equalizing sequences would require breaking or distorting the source-level split, so the design prioritizes leakage prevention and balances source videos instead.

There is no permanent test set. Fold 0 is tested while folds 1–4 train, then fold 1 is tested while the others train, and so on. Every source video is tested exactly once. The final out-of-fold prediction table therefore contains one held-out prediction for every eligible sequence.

## 7. Where does validation happen?

The word "validation" can be confusing because the encoder and read-out use different procedures.

The encoder does not use a validation set for early stopping or checkpoint selection. Its architecture, learning rate, epoch count, and other settings are fixed in advance. In each outer fold, notebook 03 trains the encoder for the registered 300 epochs on all 74 or 75 outer-training sources.

Validation is used later for the ridge read-out. Within the outer-training sources, the program makes four inner folds. In one inner round, 55 to 57 sources fit a candidate read-out and 18 or 19 sources validate the ridge penalty. The validation role rotates until every outer-training source has been used once for validation. The test sources remain entirely outside this process.

The corresponding sequence counts are not fixed because sources contain different numbers of sequences. Across the registered inner folds, 266 to 450 sequences fit a candidate read-out and 80 to 197 sequences validate it. These sequences all belong to the current outer-training sources; none comes from the outer-test fold.

After the penalty is selected, the read-out is fitted again using all 74 or 75 outer-training sources. It is then applied once to the corresponding 18 or 19 outer-test sources. This arrangement is called **nested cross-validation** because the four-fold validation loop sits inside the five-fold testing loop.

## 8. What exactly happens in notebook 03?

Notebook 03 trains an S-JEPA, short for Skeleton Joint-Embedding Predictive Architecture. It receives 64 time steps with 33 landmarks and organizes time into four-frame patches. During training, some latent content is hidden, and the model learns to predict that hidden content. The target later used for evaluation does not enter this training objective.

Sampling is source-balanced. The program first samples one of the outer-training source videos uniformly and then samples one of that video's sequences. Without this step, a video that produced many sequences would influence training more heavily merely because it was split into more files.

The registered paper design has five outer folds, five seeds, and two variants. The vanilla variant does not add reflections. The reflection-augmented variant reflects a training sample with probability 0.5. This gives

$$
5\text{ folds}\times5\text{ seeds}\times2\text{ variants}=50\text{ encoders}.
$$

The two variants use matched initialization and training randomness wherever possible so their difference isolates the reflection-augmentation recipe more cleanly. Each checkpoint records the sources it was allowed to see and the test sources it was forbidden to see. A mismatch stops the pipeline rather than reusing an incompatible model.

## 9. How is testing performed?

Notebook 04 loads one fold-local encoder at a time. It extracts features for the outer-training and outer-test sequences, but it fits the read-out only with outer-training targets. The four inner folds select the ridge penalty. The fitted read-out then produces predictions for the untouched outer-test sources.

Several comparison lanes prevent an overly simple interpretation. An untrained encoder shows what the architecture provides before learning. A target-component self-consistency check confirms that the saved left/right pair contrasts reconstruct the target; it is not a learned baseline. Missingness and acquisition features test measured shortcuts. Constructed read-outs test what happens when the mirror rule is forced mathematically. Direct token comparisons ask the stricter question of whether the encoder representation itself transforms correctly.

Mirrored test inputs are paired transformations of existing test sequences. They are not counted as new independent test cases. All performance summaries give each source video equal total weight, so a source with many sequences does not dominate a source with only a few.

## 10. How is uncertainty handled?

The five seeds measure variation due to randomized model training. The source bootstrap measures variation associated with the observed collection of source videos. In the bootstrap, entire videos are resampled and all of their sequences travel together. The protocol uses 2,000 such resamples.

These two sources of variation answer different questions and are not blended. Metrics are first computed for each checkpoint. Registered seeds are then summarized. Source-bootstrap intervals remain conditional on the trained cross-validation pipeline; they do not prove that the 93 videos represent every person or every real-world recording condition.

## 11. What can the completed study claim?

If all empirical gates pass, the strongest supported statement concerns post-development, within-GAVD performance on held-out source videos. A useful probe-level symmetry claim requires positive predictive utility, sufficiently small mirror error, and improvement over the paired untrained initialization. A direct encoder-equivariance claim additionally requires a low token-level reflection error and improvement over initialization.

Even a successful result would not establish unseen-person generalization, diagnosis, clinical validity, disease prevalence, or treatment effects. It would also remain conditional on the BlazePose landmark schema, pose preprocessing, architecture, seeds, and measured controls.

## 12. Current status and governance

The cohort and split have been audited. The full paper computation and held-out evaluation are not complete, so the current protocol has no reportable v2.1 performance conclusion. Synthetic smoke runs only check that the software works and are never scientific evidence.

Submission is also blocked by unresolved governance. The institutional ethics determination, data-use review, and derived-pose release review must each have a genuine dated internal reference. Public availability of source links does not replace these reviews. Derived poses, identifiers, embeddings, predictions, and checkpoints must not be redistributed until the completed reviews permit it.

## 13. Reading the notebooks in order

1. Notebook 00 freezes the protocol and reports governance status.
2. Notebook 01 constructs the cohort and audits the coordinate-derived target.
3. Notebook 02 assigns source videos to the five outer and four inner folds.
4. Notebook 03 trains the 50 registered fold-local encoders.
5. Notebook 04 fits read-outs and predicts the held-out sequences.
6. Notebook 05 combines all registered folds and seeds and calculates uncertainty.
7. Notebook 06 checks whether a separately governed external dataset is eligible for a future subject-disjoint evaluation.

The main idea connecting these steps is simple: define the target without filled-in values, keep related sequences together, prevent test information from shaping training or validation, and state only the claim that the available identifiers and completed evidence can support.
