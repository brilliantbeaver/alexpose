# Auditing Gait Laterality in Skeleton Predictive Representations

Anonymous working manuscript · Revision 3 · 9 September 2026

## Abstract

Human movement provides a useful test of whether predictive representations preserve the geometry of an articulated body. Left–right reflection supplies a known transformation, while differences between the two sides provide an observable quantity whose sign should change under that transformation. We study this relationship in a skeleton Joint-Embedding Predictive Architecture using pose sequences from gait videos. A source-separated evaluation compares trained and initial encoders, tests reflection behavior, and changes the anatomical and temporal placement of hidden training targets. The available experiments show that predicting a clip's hidden features more accurately than another clip's features can coexist with weaker recovery of a signed movement contrast. Motion-sensitive summaries improve access to information already present at initialization, while motion-weighted and connected-region masks provide no demonstrated advantage over their matched random references. We distinguish this evidence from exact symmetry imposed at the output and from synthetic demonstrations of explicit reflection training. The study contributes an evaluation of articulated-body representations for Physical World AI, identifying a gap between feature prediction and the preservation of a task-relevant physical measurement.

## 1. Motivation and research question

Walking involves coordinated movement of an articulated body with corresponding left and right limbs. This geometry provides both useful regularities and meaningful departures from them. An average description of movement can remain unchanged when anatomical sides are exchanged, whereas a quantity describing which side moves more should reverse sign. A representation intended for physical prediction may need to support both responses.

Gait abnormalities motivate studying that distinction. Post-stroke studies measure spatial and temporal asymmetries, and Parkinson's research examines unequal arm swing [Patterson et al.](https://pubmed.ncbi.nlm.nih.gov/18226655/), [Lewek et al.](https://pubmed.ncbi.nlm.nih.gov/19945285/). These observations motivate our measurement; they do not validate it as a clinical index. Here, laterality means a signed comparison of estimated left- and right-side movement.

We ask whether masked feature prediction makes that comparison easier to recover from a skeleton encoder on held-out source videos. The initial null is that training provides no improvement over the same encoder at initialization under a fixed downstream evaluation. Separate questions concern whether the output changes sign correctly and whether landmark features follow a specified reflection rule. Answering one does not establish the others.

Our contribution is a controlled evaluation of a small predictive representation model. The completed experiments concern whole-clip masked prediction and frozen-feature readouts. Their relevance to world models is the assessment of a geometric property that could matter to a future dynamics model. The study has no demonstrated real-data forecasting, planning, or multimodal-fusion result.

## 2. Data and the geometric target

The local subset of the [Gait Abnormality in Video Dataset](https://arxiv.org/abs/2407.04190) contains 666 annotations, of which 642 have pose archives. Quality control retains 625 sequences from 93 source videos. The annotations cover normal, Parkinson's, stroke, myopathic, and cerebral-palsy gait. They stratify source assignments and describe the dataset; they do not supervise encoder training.

Each archive contains a variable-length sequence of 33 landmarks with estimated coordinates and visibility. Coordinates are derived from video, including an inferred depth component, and are not calibrated metric three-dimensional motion capture.

For each of five bilateral pairs—shoulders, knees, ankles, heels, and foot tips—we calculate left and right speeds using observed timestamps and only transitions for which both landmarks are observed at both ends. Let their median speeds be \(m_{L,k}\) and \(m_{R,k}\). The target is

\[
y(x)=\frac{1}{5}\sum_{k=1}^{5}
\frac{m_{L,k}-m_{R,k}}{m_{L,k}+m_{R,k}+\epsilon}.
\]

All five pairs must provide at least eight shared valid transitions. Hips establish the pelvis reference but are excluded from this target: pelvis centering makes their speed magnitudes identical. The target is calculated before input interpolation and resizing.

Reflection \(M\) negates the centered horizontal coordinate and exchanges all anatomically paired landmarks, together with validity. Consequently, \(M^2x=x\) and \(y(Mx)=-y(x)\). This identity is a property of the coordinate operation and formula, rather than evidence that every reflected recording would occur naturally.

The encoder input is prepared separately. Visibility at least 0.45 and finite coordinates determine observations; up to four missing samples can be interpolated for input only. Pelvis normalization removes translation and applies a body scale. Resizing then samples 64 locations along normalized sequence index, without preserving the original timestamp intervals. Values under invalid entries are zero-filled only alongside an explicit validity mask.

| Stage | Array shape | What remains associated; what changes |
|:--|:--|:--|
| Archived clip | \(T_i\times33\times4\), plus frame numbers and frame rate | Coordinate channels and visibility remain linked to one source and sequence. |
| Target branch | \(T_i\times33\times3\), validity \(T_i\times33\) | Original timestamps and observed-only paired transitions determine the target. |
| Model input | \(64\times33\times3\), validity \(64\times33\) | Joint identities remain; interpolation, normalization and resampling alter coordinate values and timing. |
| Four-frame patches | \(16\times33\times12\) | Each patch concatenates four coordinates for the same landmark; complete validity determines eligibility. |
| Embedded token grid | \(16\times33\times96\) | Time-block and landmark indices identify each 96-dimensional feature. |
| Batch with mask | \(B\times528\times96\), plus target indices and validity | Gathering and padding retain each clip's own targets and common mask count. |

The pipeline preserves the correspondence between recordings, landmarks, masks, and targets. It cannot promise preservation of the exact observed shape or speed after preparation. Cohort preparation occurs before split-file creation in the implementation; it uses deterministic clip-local operations rather than statistics fitted across sources. Source membership is fixed before training sampling, augmentation, and model fitting.

## 3. Training and source separation

All clips and derived views from a source video inherit its fold. Five outer folds contain respectively 19, 19, 19, 18, and 18 test sources; their test sequence counts are 189, 182, 72, 77, and 105. Each encoder trains only on its matching outer-training sources. New encoders are initialized for each fold and seed, and source-balanced sampling limits domination by videos with many clips.

The base model uses a 96-dimensional embedding, four encoder layers, two predictor layers, and four attention heads. The online encoder receives a full token grid with hidden coordinate content suppressed and positional information retained; a predictor estimates teacher features at hidden positions. A target encoder supplies those features and is updated by an exponential moving average (EMA) of online weights. Its parameters receive no gradients.

The real-data objective matches distributions over feature channels using cross-entropy,
\[
\mathcal L=\mathcal L_{\mathrm{CE}}(p_{\bar\theta},p_{\theta,\phi})
+0.05\,\mathcal L_{\mathrm{VICReg}}.
\]
Here \(\theta,\phi,\bar\theta\) denote online-encoder, predictor, and teacher parameters. Teacher centering and separate temperatures form the target and predicted distributions. VICReg penalizes insufficient feature variation and redundant channels, alongside agreement between geometric views. It is calculated from unmasked views pooled over twelve gait landmarks. Consequently, anatomical information remains in the regularizer even when targets are chosen across the body. Mean-squared feature error appears in diagnostics; it is not the real-data pretraining objective.

The historical base profile records 300 epochs, four updates per epoch, AdamW, batch size 20, learning rate \(10^{-3}\), and weight decay 0.05. All completed controlled grids use 1,200 updates per encoder. Mask fraction is applied to eligible gait tokens and reduced to a feasible shared count; the original 0.6 setting does not hide 60% of all 528 tokens.

The study introduces anatomical laterality at different locations that should be kept explicit:

| Entry point | Effect on training or evaluation | Evidence status |
|:--|:--|:--|
| Named left/right landmarks and gait pooling | Preserve anatomical identities and choose locations contributing to regularization | Used in completed real-data training |
| Gait-only target eligibility | Select hidden tokens from six bilateral landmark pairs while retaining all-body context | Historical base and Notebook 12 gait arm |
| Anatomical reflection augmentation | Exchange joint identities, horizontal signs, validity and transformed views consistently | Previously reported original variant |
| Explicit reflection penalty | Penalize token disagreement after anatomical exchange; gradients update the online encoder | Notebook 09 synthetic demonstration |
| Odd readout and signed regression target | Constrain or evaluate a frozen representation | Downstream only |

The latest motion/region trainer fixes reflection probability and explicit symmetry weight to zero. It does not test a combined motion-plus-reflection recipe. None of these encoder objectives receives \(y\), a diagnosis, or an affected-side label.

After pretraining, encoder weights are frozen. Inner source-separated validation selects a ridge-regression penalty; ridge regression is a linear readout whose weight penalty limits fitting unstable feature combinations. The original audit uses four inner folds, while later comparisons use three. Outer-training groups contain 74 or 75 sources. Each original inner fit contains 55–57 sources, and validation contains 18 or 19. No outer-test source enters these fits. Imputation and scaling are fitted within each inner fitting partition and refitted on the outer-training set after selection. These inner folds select the readout only: the encoder has already seen their unlabeled inputs during outer-training pretraining.

## 4. Evaluation through successive questions

### 4.0 Does the prepared input preserve the measurement?

The original audit prompted a check before attributing weak performance to the encoder. Notebook 07 recomputes the movement formula on the prepared input. Repeating that descriptive calculation during this review produces finite pairs for 623 clips from 92 sources. Source-weighted sign agreement with the original target is 70.4%; the direct calculation-agreement \(R^2\) is 0.218. Two clips cannot supply a finite recomputation.

This comparison changes the entire preparation path, including timing, interpolation and valid support. It identifies a measurement mismatch without localizing its cause. Its \(R^2\) is neither a model score nor an information ceiling: a learned mapping from prepared input could behave differently. It motivates evaluating the target after individual preparation stages on the same eligible clips.

### 4.1 Does the original representation satisfy the reflection audit?

The retained original aggregate summary reports source-balanced \(R^2=0.060\), with a 95% source-bootstrap interval of \([-0.025,0.126]\). The learned-minus-initial difference is \(-0.018\), with interval \([-0.039,0.002]\). Reflection augmentation changes \(R^2\) by \(0.004\), with interval \([-0.006,0.013]\). These estimates do not demonstrate a training benefit or establish equivalence between variants.

The direct token test aligns landmark positions under reflection while leaving feature channels unchanged. Its normalized squared discrepancy is

\[
q=\frac{\|Z(Mx)-SZ(x)\|_C^2}
{\|Z(Mx)\|_C^2+\|SZ(x)\|_C^2},
\]

where \(S\) exchanges joint identities and \(C\) selects common valid tokens. The retained summary reports \(q=0.114\) for learned features and \(0.083\) at initialization, with a paired increase of \(0.031\), interval \([0.016,0.048]\). This is unfavorable evidence for the specified identity-channel transformation. An encoder could express reflection through a different transformation of feature channels; the test does not examine that possibility.

An output can be made antisymmetric through an odd feature,
\(z^-(x)=[z(x)-z(Mx)]/\sqrt{2}\), followed by a linear readout without an intercept or feature centering. Its symmetry follows algebraically. Predictive accuracy still requires evidence: the recorded learned-minus-initial difference under this construction is negative.

The original values are available in a retained figure-summary JSON. Its full report, checkpoint, and per-example prediction chain was not found in this checkout during the present review. We retain these results as previously reported context and distinguish them from the newly audited comparisons below.

### 4.2 Does anatomical target selection improve learning?

Notebook 12 compares scattered hidden targets selected from twelve gait landmarks with the same number selected across all 33 landmarks. Both encoders still receive the full landmark set and retain anatomical choices in regularization and readout. Starting weights, sampled clips, geometric views, and 1,200 updates are paired.

The teacher-feature readouts obtain \(R^2=-0.023\) with gait targets and \(-0.010\) with all-landmark targets; the initial encoder obtains \(0.048\). The broader-minus-gait difference has interval \([-0.018,0.050]\), leaving both improvement and harm compatible with these recordings. Larger ridge penalties frequently reach the edge of the search range, limiting conclusions about the best achievable readout. The later grid extends the candidates to 10,000, which is still selected by 39% of teacher motion-sensitive fits and 77% of the corresponding online fits. The reported deficits are results of these readout procedures, rather than a bound on what any decoder could recover.

Notebook 08's earlier values are retained in the tutorial but excluded from the main comparison: implementation and readout choices changed together before Notebook 12. Likewise, Notebook 12's direct-pose \(R^2=0.130\) and the latest \(0.0345\) come from different evaluation recipes. Their change cannot be attributed to a mask or a single regularization choice.

### 4.3 Does targeting motion or connected anatomy help?

Notebooks 15–18 extend the comparison to motion-weighted and connected-region targets. The completed grid contains 125 trained encoders, each with 1,200 updates. Motion comparisons and region comparisons have their own random references, matched to the actual hidden count. Equal nominal mask percentages would not ensure equal tasks because missing observations and eligible landmark pools differ.

| Frozen features or control | Mean \(R^2\) ± seed SD | Mean absolute error |
|:--|--:|--:|
| Training-source target mean | −0.0106 ± 0.0000 | 0.046172 |
| Direct prepared-pose summary | 0.0345 ± 0.0000 | 0.044960 |
| Initial encoder, mean summary | 0.0708 ± 0.0186 | 0.044344 |
| Initial encoder, motion-sensitive summary | 0.2225 ± 0.0268 | 0.041546 |
| Motion-uniform teacher, motion-sensitive | 0.1137 ± 0.0110 | 0.043658 |
| MAMP-style teacher, motion-sensitive | 0.1129 ± 0.0306 | 0.043751 |
| Robust-motion teacher, motion-sensitive | 0.1142 ± 0.0210 | 0.043595 |
| Region-uniform teacher, motion-sensitive | 0.1094 ± 0.0088 | 0.044037 |
| Connected-region teacher, motion-sensitive | 0.1008 ± 0.0092 | 0.043756 |

Each row covers 625 clips and 93 sources. SD describes the five optimization seeds, not sampling uncertainty. The initial encoder, direct-pose, and training-mean controls are reused across arms; their repeated appearance in machine-readable tables is not independent replication. “Initial” refers only to the S-JEPA weights: the pipeline still uses a pretrained pose detector, supplied anatomical identities, feature engineering, and a supervised ridge readout.

The motion-sensitive summary adds temporal variation, feature changes, and observation support. All trained arms remain below their matched initial encoder in each of the five seeds. Its advantage over mean pooling cannot yet be attributed solely to motion because the summary also includes support information. Temporal standard deviation is insensitive to ordering. The initial gain is \(0.1517\) in \(R^2\), compared with \(0.0270\)–\(0.0468\) for trained teachers. The online encoders also underperform initialization, with mean \(R^2\) between 0.0636 and 0.0809, so evaluating the teacher alone does not explain the deficit.

We additionally recomputed trained-minus-initial intervals from the saved predictions for this revision. They are new exploratory analyses, using 2,000 paired source resamples with bootstrap seed 812 and no model retraining. All five intervals lie below zero; the motion-uniform contrast is \(-0.1088\), interval \([-0.1701,-0.0412]\). The full set appears in Appendix A. This strengthens the conditional description of a learning deficit while leaving evaluation choices and adaptive development as limitations.

| Mask minus its matched random reference | \(\Delta R^2\) | 95% source interval |
|:--|--:|:--|
| MAMP-style motion weighting | −0.0008 | [−0.0204, 0.0153] |
| Robust motion mixture | 0.0005 | [−0.0155, 0.0152] |
| Connected region | −0.0087 | [−0.0333, 0.0177] |

The mask audits confirm that the proposed changes took effect. Motion targets average 80.771 valid tokens per clip, approximately 17% of valid all-body tokens. Relative to eligible-token motion, average enrichment is approximately zero for random selection, 0.0175 for MAMP-style weighting, and 0.0378 for the robust mixture. The mixture retains a uniform component and limits extreme motion scores; high displacement may still reflect tracking error.

In the structure audit, immediate visible time brackets surround 69.9% of random-reference targets and none of the connected-region targets. Connected regions also reduce the availability of visible anatomical neighbors. Thus the masks remove different observable clues, although they do not isolate what the contextualized teacher encodes. Whole trajectories and interior gaps are audited for coverage, but are absent from the completed training comparison.

The motion and region families use different hidden budgets and separate random references. We compare each alternative only with its own reference. The largest numerical score among arms is not a validated best mask, and an interval spanning zero does not establish equal performance.

### 4.4 What does successful feature prediction establish?

The trained predictors consistently predict their own clip's hidden teacher features more accurately than targets from another source video, comparing matched and mismatched errors on the same eligible control clips and target positions. Initial predictors show no consistent preference. This establishes useful clip correspondence for that diagnostic. Posture, viewpoint, observation support, or motion could contribute to it; the result does not isolate a physical mechanism.

Thus, predictor training can succeed on its own features while the frozen representation yields weaker laterality readouts. Because each arm supplies its own changing teacher, raw feature-prediction losses are not a common between-model measurement scale. The coordinate-derived outcome supplies a shared evaluation target.

## 5. Statistical interpretation

For \(V\) sources and \(n_v\) clips from source \(v\), each clip has weight \(w_{vi}=1/(Vn_v)\). With \(\bar y_w=\sum w_{vi}y_{vi}\),
\[
R^2_w=1-\frac{\sum_{vi}w_{vi}(y_{vi}-\hat y_{vi})^2}
{\sum_{vi}w_{vi}(y_{vi}-\bar y_w)^2}.
\]
Each source therefore receives equal total weight even when it supplies many clips. Within a seed, predictions from all five held-out folds are pooled before computing \(R^2\); the reported result averages the five seed-specific scores. It is neither an average of fold \(R^2\) values nor a score from an ensemble of seed-averaged predictions. Negative \(R^2\) means squared error exceeds variation around the weighted evaluation mean; a separately fitted training-source mean is the usable prediction baseline.

Paired uncertainty resamples entire videos while retaining their clips and paired seed predictions. For a mask comparison the null is no positive difference in source-balanced readout performance relative to the count-matched random policy; for learning it is no positive difference relative to matched initialization. These are superiority questions, and no equivalence margin or power guarantee is established. The 2,000-resample intervals condition on the fitted models and exclude full-retraining and new-split uncertainty. Repeated seeds and diagnostic mask draws are not independent participants. Subsequent experiments were informed by the same development cohort, so their source separation does not make the overall research trajectory an untouched confirmatory study.

## 6. Implications and limitations

The strongest supported conclusion concerns access to one signed movement measurement under the evaluated readouts. A negative learned-over-initial contrast does not demonstrate that all motion information has disappeared. Mean pooling, feature scaling, regularization, and the difference between the original target and prepared input remain possible contributors.

The study fits the workshop's articulated geometry and evaluation themes. Coordinates derived from RGB do not constitute multimodal sensing, and an inferred depth coordinate does not establish calibrated 3D reconstruction. Clinical interpretation would require independent measurements, affected-side labels, and participant-separated validation. The source-video split cannot exclude the same unidentified person appearing in different videos.

The next experiments should first isolate support features, broaden training-only ridge selection, and measure the target after each preparation stage. A later study can test whether explicit geometric training improves common observable future endpoints, using past-only preprocessing and persistence and velocity baselines. Existing symmetry-training and future-decoding demonstrations are synthetic; they are not evidence of forecasting gait recordings.

Institutional ethics, data-use, and derived-pose release reviews remain unresolved in the recorded governance status. This manuscript is an internal revision, and it does not claim those determinations have been obtained.

## 7. Related work and reproducibility

[S-JEPA](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04755.pdf) already studies masked skeleton-feature prediction, and [MAMP](https://arxiv.org/abs/2308.07092) motivates motion-weighted masking. [SLiM](https://arxiv.org/abs/2603.10648) provides recent structured skeleton-masking precedent. Our contribution is the controlled laterality evaluation, with no claim to invent skeleton JEPA or anatomical masking.

The [tutorial](../TUTORIAL.md), [source-split reference](../TRAIN_TEST_SPLIT.md), [core audit](../README.md#core-evidence-audit), and [extension audit](../README.md#extension-evidence-audit) document the evidence and its limits. Original notebooks and experiment artifacts are preserved. This revision adds no training run or clinical finding.

## Appendix A. Additional exploratory learning contrasts

These contrasts were calculated during the 9 September revision from the complete latest prediction grid. They were not part of the original registered reflection audit. Each compares the final teacher's motion-sensitive summary with its matched initial encoder using the same readout family, clips and seed. Intervals are marginal percentile source-bootstrap intervals conditional on the fitted models; there is no family-wise multiplicity adjustment.

| Training arm | Learned minus initial \(R^2\) | 95% source interval |
|:--|--:|:--|
| Motion uniform | −0.1088 | [−0.1701, −0.0412] |
| MAMP-style motion | −0.1097 | [−0.1651, −0.0514] |
| Robust motion | −0.1083 | [−0.1706, −0.0423] |
| Region uniform | −0.1131 | [−0.1662, −0.0596] |
| Connected region | −0.1218 | [−0.1828, −0.0553] |

The [recomputation script](../verify_physworld_evidence.py) checks the prediction coverage and reproduces the recorded pooled scores before calculating these contrasts. The [extension evidence audit](../README.md#extension-evidence-audit) records the retained artifact paths and full-precision values.
