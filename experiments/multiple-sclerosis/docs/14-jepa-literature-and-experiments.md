# What recent JEPA research suggests for better Normal, MS, and PD classification

_Reviewed against primary papers on September 20, 2026. Companion to
[the notebook 03–06 methodology review](13-representation-and-classification.md).
This is a research assessment and experiment plan; it does not change training code._

## 1. The decision that matters

Our goal is an encoder that preserves useful differences between walks, followed by
a classifier that can recognize the Normal, MS, and PD labels in previously unseen
sources. A smaller pretraining loss is useful only if it helps that goal.

There are three separate questions:

1. **Does the representation vary?** If every walk becomes the same vector, the
   encoder has collapsed.
2. **What does that variation describe?** A vector could describe useful movement,
   camera viewpoint, body proportions, pose-estimation errors, or a mixture.
3. **Can a classifier use it on new sources?** The answer requires held-out labels
   and an evaluation that keeps related recordings together.

These questions explain why the newest self-supervised objective is not automatically
the best next change. First establish what the current encoder adds, whether its
training targets line up, and whether the classifier treats sources fairly.

The current full-data path pretrains on **all three conditions within the training
partition**, without putting diagnosis labels into the JEPA loss. The `normal` text
in notebook 03's filename is historical. Notebook 04 continues self-supervised
training; its filename does not mean the active path performs diagnosis-supervised
VICReg adaptation. Only the fold-0 encoder run is currently established by the saved
artifacts. **There is no completed full-v1 cross-validation classification score to
use as evidence of improvement.** See [the split specification](11-full-data-splits.md).

## 2. What the research adds

### Better targets can teach better descriptions

A JEPA predictor practices estimating missing **representations**. The classifier
has a different job: estimating the three condition labels. Making the practice task
harder can help, but only when solving it requires information that the final task
needs. An impossible task or one solved mostly through position identity can waste
the training budget.

The original skeleton S-JEPA provides the closest foundation. More recent GFP and
V-JEPA 2.1 investigate targets at different temporal scales or network depths. Their
results motivate experiments that preserve both local movement and longer patterns.
They do not establish which of those details separate this project's labels.
[S-JEPA](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04755.pdf),
[GFP](https://arxiv.org/abs/2509.03609),
[V-JEPA 2.1](https://arxiv.org/abs/2603.14482).

### Avoiding collapse and learning class boundaries are different jobs

VICReg and SIGReg encourage useful statistical properties of embeddings. Supervised
contrastive learning explicitly uses class labels to shape their relationships.
Those are different sources of training information.
[VICReg](https://arxiv.org/abs/2105.04906),
[LeJEPA](https://arxiv.org/abs/2511.08544),
[Supervised Contrastive Learning](https://arxiv.org/abs/2004.11362).

**Project interpretation:** imagine three colors of dots spread across a page.
Spreading the dots prevents a single pile, but the colors can still be thoroughly
mixed. A useful condition representation must make the relevant differences
available to the classifier. Spread alone does not establish this.

The legacy class-aware VICReg explanation needs particular care. Subtracting each
class's mean removes the class center from the values being regularized. Moving
that entire class elsewhere would leave those residuals unchanged. Thus those
variance/covariance terms cannot directly push class centers apart; a variance
floor encourages within-class spread. This is a mathematical observation about
the [local implementation](../sjepa/losses.py), not a claim made by the VICReg paper.
The active repaired training loop does not use that class-aware path.

### Recent results also show tradeoffs

V-JEPA 2.1 provides a useful warning: adding its visible-context loss alone improved
dense visual tasks but reduced action-classification accuracy in the reported
ablation. Supervision at several network depths recovered much of that loss.
LeVJEPA likewise reports that aggressive token dropping can weaken motion
performance during shorter training schedules.
[V-JEPA 2.1, section 2.3](https://arxiv.org/html/2603.14482v3),
[LeVJEPA, discussion](https://arxiv.org/html/2608.27395v1).

**Our inference:** a method may improve one kind of information while weakening
another. Compare one change at a time. Our laptop configuration has **264 skeleton
tokens**, compared with 132 in the smoke configuration and 528 in the GPU
configuration. A masking ratio successful on dense RGB patches is not a justified
default for these much smaller skeleton sequences.

## 3. Primary-source evidence ledger

Dates below are the paper's submission/revision dates where available, not search
engine crawl dates. A preprint is research shared publicly before a confirmed
peer-reviewed publication. A paper appearing on arXiv may also have a confirmed
conference or journal publication, as marked below. The list focuses on decisions
we can actually test.

| Paper and verified version/status | What it contributes | Limit for this project |
|---|---|---|
| **Mohamed Abdelfattah and Alexandre Alahi. _S-JEPA: A Joint Embedding Predictive Architecture for Skeletal Action Recognition._** ECCV 2024, publisher version. [Primary PDF](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04755.pdf) | Masked skeleton representation prediction with a slowly updated target encoder, using an exponential moving average (EMA), plus target centering/sharpening and latent cross-entropy. The full input reaches the target encoder; selected outputs supply targets. | Large 3D action datasets differ from our small 2D gait collection. Latent softmax entries describe features, not Normal/MS/PD probabilities. |
| **Adrien Bardes, Jean Ponce, and Yann LeCun. _VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning._** ICLR 2022. arXiv v1 May 11, 2021; reviewed v3 January 28, 2022. [Paper](https://arxiv.org/abs/2105.04906) | Combines agreement between views, a variance floor, and a covariance penalty to discourage constant and redundant representations. | Its standard objective has no diagnosis labels. A single-view call has no active view-agreement term. It cannot certify condition separation. |
| **Prannay Khosla et al. _Supervised Contrastive Learning._** NeurIPS 2020. arXiv v1 April 23, 2020; reviewed v5 March 10, 2021. [Paper](https://arxiv.org/abs/2004.11362) | Uses known classes to encourage same-class similarity and different-class separation. | Any application here is supervised adaptation. Class labels can also correlate with recording conditions; a class-shaped embedding can still learn shortcuts. |
| **Shengkai Sun et al. _Towards Efficient General Feature Prediction in Masked Skeleton Modeling._** ICCV 2025. arXiv v1 September 3, 2025. [Paper](https://arxiv.org/abs/2509.03609) | GFP predicts local and global features at several temporal scales, using lightweight target generators and variance/covariance constraints. | Evidence is from action benchmarks. Additional target networks and losses are a substantial experiment, not a proven improvement for this dataset. |
| **Mido Assran et al. _V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning._** Reviewed arXiv v1 June 11, 2025; treated here as a preprint. [Paper](https://arxiv.org/abs/2506.09985) | Scales latent video prediction and evaluates motion understanding and anticipation; action-conditioned robotic prediction is a separate post-training stage. | RGB video, training scale, and task differ greatly. Its scores are not expected scores for our skeleton classifier. |
| **Randall Balestriero and Yann LeCun. _LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics._** Preprint. arXiv v1 November 11, 2025; reviewed v3 November 14, 2025. [Paper](https://arxiv.org/abs/2511.08544) | SIGReg encourages an isotropic Gaussian embedding distribution: roughly, a balanced cloud without privileged directions. The resulting objective removes the usual need for an EMA teacher and stop-gradient. | Its theory has stated assumptions. A favorable distribution does not prove that this dataset's condition information survives. Adding SIGReg to our unchanged architecture would be a hybrid, not a faithful reproduction. |
| **Lorenzo Mur-Labadia et al. _V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised Learning._** Preprint. arXiv v1 March 15, 2026; reviewed v3 June 11, 2026. [Paper](https://arxiv.org/abs/2603.14482) | Supervises visible/masked tokens and several encoder depths. In its component study, visible-context supervision reduced accuracy on SSv2, an action-recognition benchmark, from 72.8% to 62.5%; deep supervision recovered it to 72.0%. | The component tradeoff argues for careful ablations. Dense image/video improvements do not establish a classification benefit for our short gait windows. |
| **Lukas Kuhn et al. _LeVJEPA: Efficient & Scalable Video Pretraining without the Heuristics._** Preprint. arXiv v1 August 27, 2026. [Paper](https://arxiv.org/abs/2608.27395) | Shared-encoder learning aligns global/local clip views with SIGReg. Random token dropping lowers cost; block-causal attention restricts each frame to present/past information. | A recent video result, not a skeleton validation. Its 95% dropping setting must not be copied blindly. Causal attention serves streaming requirements; it is not an established diagnosis improvement. |
| **Quentin Garrido et al. _RankMe: Assessing the Downstream Performance of Pretrained Self-Supervised Representations by Their Rank._** ICML 2023. arXiv v1 October 5, 2022; reviewed v3 June 26, 2023. [Proceedings](https://proceedings.mlr.press/v202/garrido23a.html), [version history](https://arxiv.org/abs/2210.02885) | Effective rank provides a label-free diagnostic and helps select models in the studied settings. | State the exact estimator and sample set. Random variation can also produce high rank. Rank is neither an accuracy percentage nor proof of meaningful condition separation. |
| **Daniel Richards Arputharaj, Daniel Jönsson, and Gabriel Eilertsen. _A Comparative Study of Label-free Representation Quality Metrics in Deep Learning._** TMLR, August 2026. arXiv v1 August 24, 2026. [Paper](https://arxiv.org/abs/2608.23182) | Across 260 vision models and six datasets, the reliability of label-free quality metrics varies with architecture and training objective. | Supports cautious interpretation of geometric diagnostics; its experiments are not a gait-specific validation. Keep held-out classification as the task test. |

The skeleton S-JEPA above should not be confused with papers using the same
abbreviation for EEG or speech. This review does not assume every paper with
“JEPA” in its name shares the same objective.

## 4. Current capabilities versus proposed experiments

| Capability | Present status in the active full-data path |
|---|---|
| Repaired predictor positions, stochastic per-example masks, EMA targets, source-uniform pretraining | Implemented; see [training code](../sjepa/train_v2.py) |
| Frozen target-encoder representation and class-balanced logistic-regression head | Implemented in [full_experiment.py](../sjepa/full_experiment.py) |
| Full-token mean pooling | Available through `model.embed(x, None)`, but not the current shared readout |
| Fixed seeded subset pooling | Current shared readout; samples its token subset once from a fixed seed |
| Weighted scaler and source-plus-class weighted classifier | Proposed; current scaler is unweighted and head balances clip classes |
| Diagnosis-supervised encoder adaptation, SupCon, SIGReg, GFP-style targets, deep supervision | Not implemented in this active workflow |
| Complete full-v1 outer-fold results | Not yet available; code support does not establish completed execution |

The following experiments are **proposals**, not changes already made or results
already observed. Their order reflects this dataset's small number of independent
sources and the value of resolving simpler explanations first.

## 5. Ranked experiment roadmap

### Priority 1 — Measure what pretraining actually adds

Compare a freshly initialized encoder with the trained encoder using the **same
architecture, target-encoder readout, scaler, classifier, partitions, and seeds**.
Fit each classifier only on its own training representations. Include the existing
Random Forest, mean-pose, visibility, and majority controls.

This answers a concrete question: does JEPA training add useful information beyond
the structure already present in an untrained network? If the scores are similar,
we should investigate training objectives and shortcuts before making the network
larger. If the trained encoder is better, we have evidence for a pretraining benefit
under that evaluation. Neither outcome establishes patient-level generalization.

Record source-weighted macro-F1, each class's precision/recall, confusion counts,
and the clip predictions with their source IDs. Macro-F1 averages a precision/recall score across
the three classes. These make a change understandable even when an
average score conceals a decline for one class.

### Priority 2 — Audit the meaning of each prediction target

The augmentation code can mirror the input and swap left/right joint slots, while
the teacher sees the original skeleton. First trace a labeled left/right example
through the tokenizer, mask, student input, and teacher target. Decide explicitly
whether the desired correspondence is anatomical joint identity or image-side
location.

Run a controlled **no-horizontal-flip** comparison. If a flip is retained, test
whether targets and positional identities need corresponding remapping. A deliberate
view-invariance objective can be valid; an unnoticed correspondence mismatch can
make the practice task misleading. The audit determines which interpretation applies.

Also check that augmentations preserve the quantities we want to study. A stronger
augmentation is not automatically better when a subtle movement difference may be
the useful signal. These are project-specific questions, not settled by action
recognition benchmarks.

### Priority 3 — Compare readouts before retraining the encoder

Freeze one checkpoint. Compare the current fixed subset with **full-token mean
pooling**. The subset is repeatable, but its selected joints/time positions come
from a seeded random draw; it is not a verified clinically optimal selection.

Then, only if needed, compare an average over a few predetermined stochastic
subsets or a small temporal summary that preserves early/late-window differences.
Use identical readout rules on training, validation, and eventual testing. Specify
the seeds and number of views in advance. Prediction-time randomness should not
silently change between runs.
For each readout, refit the scaler and classifier using that readout's training
features, with classifier settings fixed. The encoder weights remain frozen.

This experiment is inexpensive because it reuses the encoder. If a better readout
helps, the encoder may already contain useful information that the existing pooling
throws away. A larger learned attention head is a later comparison because it adds
parameters that the small training set must estimate.

### Priority 4 — Give sources a fair influence on the classifier

Source-uniform pretraining does not automatically make the later scaler or
classifier source-uniform. The current head uses one row per clip and
`class_weight="balanced"`. That equalizes total class weight but leaves
multi-clip sources more heavily weighted within their class. Assigned loss weights
are different from measured influence on the fitted coefficients.

For example, in fold 0, one MS source contributes **13 of the 23 MS training
clips**, or **56.5%** of that class's nominal classifier weight. Class balancing
alone does not remove this concentration. Counts come from the frozen
[full-data registry](11-full-data-splits.md).

Proposed training-only weights are:

- For the scaler, give each clip from source `s` weight `1 / m_s`, where `m_s`
  is that source's number of training clips. Each source then contributes the same
  total weight to the feature mean and scale.
- For the classifier, use weights proportional to `1 / (m_s × S_c)`, where
  `S_c` is the number of training sources in that clip's class. Normalize these
  weights to mean 1 and set `class_weight=None` to avoid balancing classes twice.

The classifier formula equalizes total class weight and, within a class, total
source weight. Compute all counts inside the training partition. Fit the scaler
there too. This changes the fitting objective; it must still be compared on the
same source-weighted validation metric. It is not a guaranteed accuracy increase.
Initially keep `C=1`, the readout, and encoder fixed so this comparison isolates
the weighting recipe. Report equivalent weighting choices for the controls when
comparing complete pipelines.

### Priority 5 — Let diagnosis labels adjust a small part of the encoder

After the frozen-head baseline is understood, attach a three-class head and train
that head first. Next, unfreeze only the last encoder block with a small learning
rate and a fixed training budget. Use condition cross-entropy, with the training
weights above. Keep the remaining blocks frozen initially.

Unlike self-supervised continuation, this sends condition-label information back
into the representation. The small number of updated parameters limits one source
of overfitting. Specify whether the trainable encoder is initialized from the
teacher or student checkpoint; do not silently mix the two.

Compare frozen-head and last-block adaptation before attempting full-network
fine-tuning. A training improvement accompanied by poorer validation suggests
memorization or a representation change that does not transfer. “Fine-tuning”
describes an action, not an improvement.
To claim improved representations specifically, freeze the adapted encoder and
fit a fresh copy of the original linear probe. This separates a representation
change from an improvement due only to a different classifier.

### Priority 6 — Test supervised contrast only after the supervised baseline

Add a modest supervised contrastive term to the supervised adaptation experiment.
Construct same-class positive pairs from **different training sources** when
possible; neighboring windows from one recording are too easy to match by recording
identity. Ensure the batches actually contain the intended positive pairs.

Compare this single addition with condition cross-entropy alone. Keep the sampling,
head, budget, and validation rule fixed. Report the source composition of batches.
The goal is to see whether explicit relationships between labeled walks help beyond
the decision-boundary loss. This is a project adaptation of
[SupCon](https://arxiv.org/abs/2004.11362), not evidence that the method will work
with our small number of sources.

### Priority 7 — Test one temporal or JEPA extension

Only after the earlier comparisons, select one branch based on the remaining
failure:

| Remaining question | Bounded proposed experiment | What would make it useful? |
|---|---|---|
| Does the model need more movement information? | Add joint differences over time, keeping coordinates available; or compare one contiguous temporal mask with the current masks | Better held-out classification and sensitivity to meaningful temporal changes |
| Are local and longer movement patterns poorly combined? | Add one GFP-inspired longer-scale target | Improvement beyond the same-budget baseline, without a class-specific decline |
| Is the present anti-collapse recipe fragile? | Compare a faithful small LeJEPA-style objective | Stable training plus better downstream performance, not rank alone |
| Are final-layer features losing useful local detail? | Compare one intermediate-layer prediction loss | A measurable benefit that justifies added computation |
| Is streaming inference actually required? | Evaluate causal attention separately | Acceptable accuracy and reduced need to reprocess past frames |

Label partial adaptations “inspired by” their source method. Do not combine all
branches into one experiment: an improvement would then be difficult to explain,
and a failure would not identify the cause. In particular, a teacher-free objective
is a separate design choice from adding another regularizer to the current teacher.

## 6. External motion data: potentially useful, with conversion work

**GaitForeMer** is directly relevant to the idea of learning motion from a larger
collection before a smaller clinical task. Endo et al.'s _GaitForeMer:
Self-Supervised Pre-Training of Transformers via Human Motion Forecasting for
Few-Shot Gait Impairment Severity Estimation_ was accepted at MICCAI 2022; the
reviewed arXiv v1 is dated June 30, 2022. Despite the title, its pretraining combines
motion forecasting with **supervised NTU activity cross-entropy**. The downstream
task is gait impairment severity, not our three condition labels.
[Primary paper](https://arxiv.org/abs/2207.00106),
[method and pretraining objective](https://arxiv.org/pdf/2207.00106).

**CARE-PD** offers a larger clinical motion resource: 362 participants and 8,477
walking segments in its reported collection. Vida Adeli et al.'s _CARE-PD: A
Multi-Site Anonymized Clinical Dataset for Parkinson's Disease Gait Assessment_
was accepted at NeurIPS 2025; reviewed arXiv v1 is dated October 5, 2025. It provides
3D SMPL-based data, varied annotations, severity benchmarks, and cross-dataset
evaluation. It does not supply a matched Normal/MS/PD benchmark for us.
[Primary paper](https://arxiv.org/abs/2510.04312),
[dataset table and methods](https://arxiv.org/html/2510.04312v1).

**Proposed transfer plan:** first establish a compatible joint map, coordinate
convention, temporal sampling rule, and treatment of missing/confidence channels.
Then compare external initialization against training from scratch under the same
local evaluation. Keep a record of any labels used during external pretraining.
The different representations mean that importing weights is not automatically
equivalent to learning from compatible gait data.
Check external data and checkpoint provenance for overlapping recordings, reposts,
or participants in the local validation/test sources. External initialization is
not automatically free from test-data exposure; document unresolved overlap.

## 7. Keep the experiment plan small enough to trust

Fold 0 has eight validation sources. Selecting the best of dozens of configurations
on those sources can reward chance. A large table of scores would look systematic
without providing reliable evidence.

For the immediate development pass, write down a small sequence of comparisons
before running them: same-head random baseline, no-flip audit, full versus subset
readout, and source/class weighting. Give each comparison a fixed budget, a primary
metric, and a clear tie rule. Use a small predetermined seed set when affordable;
do not keep rerunning until a favorable seed appears. Validation findings remain
development findings.

For broader optimization, use **true grouped nested cross-validation**. In each
outer fold, set aside the test sources. Make several inner training/validation
splits using only the outer development sources. Rebuild every learned component
inside each inner training split, including self-supervised pretraining. Select the
recipe from those inner results, refit using the declared development protocol, and
evaluate the outer test sources after that choice is fixed. The current one-holdout
inner design is not this broader nested search.

Summarize one correctly attributed out-of-fold prediction per usable clip, with
source weights for the main metric. Distinguish weighting clips equally per source
from averaging probabilities into one source prediction; those are different
evaluations. Repeated windows add training examples, but not independent people.
If uncertainty intervals are computed, resample at source level, not window level.
Resampling saved out-of-fold predictions conditions on the fitted models. It
omits retraining and selection variability and does not eliminate dependence from
overlapping training folds.

Finally, source IDs are a practical grouping boundary, not a verified participant
registry. A future participant-level study needs resolved identities and independent
data. For now the defensible claim is narrower: a specified change helped or hurt
classification of these labels under a specified source-grouped protocol. That is
enough to guide the next experiment without promising an accuracy that the current
evidence cannot support.
