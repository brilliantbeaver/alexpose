---
title: "Evaluating What Gait Representations Learn: Source-Held-Out Prediction and Explicit Laterality Constraints"
abstract: |
  Video-derived pose can support behavioral representation learning, but downstream prediction and geometric consistency require separate evidence. We examine two source-disjoint evaluation studies on overlapping GAVD-derived cohorts. A classification pilot contains 639 sequences from 97 sources; on its 20 test sources, raw pose features achieve macro-F1 0.441 versus 0.292 for learned features. A separate laterality study evaluates 625 sequences from 93 sources across five outer folds, five seeds, and two training variants. It defines a paired-valid motion contrast whose sign reverses under anatomical reflection, then compares learned features with their recorded initialization and explicit odd readouts. Reflection augmentation reduces strict token-equivariance error by 0.00843 (95% source-bootstrap interval [0.00687, 0.01020]), while its predictive improvement remains uncertain. An odd readout enforces output sign reversal for any encoder, yet its learned features underperform the corresponding initialization by 0.05874 in predictive R-squared. These controls distinguish useful learned information from a supplied geometric constraint. The findings support a bounded evaluation contribution for movement representations, without establishing a pretraining advantage or clinical validity.
---

## 1. Introduction

Movement representations should preserve information that supports useful readouts and respond appropriately to changes in anatomical coordinates. An annotation classifier alone tests neither property completely. In video-derived pose, multiple excerpts also share a recording, so evaluation must separate source uploads throughout encoder fitting and downstream testing.

We study compact models inspired by skeleton joint-embedding predictive architectures [@abdelfattah2024sjepa], using GAVD-derived pose [@ranjan2025gavd]. Study A is a source-held-out classification pilot with temporal and drift diagnostics. Study B adds a completed controlled evaluation of explicit laterality: can learned features predict a signed movement quantity, does reflection augmentation improve the representation's transformation behavior, and what does an imposed sign rule contribute?

Here, explicit laterality enters the target, anatomical pooling, reflection operation, and constrained readout. Encoder pretraining receives no laterality labels. Readout weights are learned from pose-derived targets, while the sign law is imposed analytically. Group-equivariant modeling and symmetry projection are established methods [@cohen2016gcnn; @bronstein2021gdl]; our contribution is their controlled use to test whether pretraining adds information beyond initialization and supplied anatomy. The resulting distinction between geometric consistency and predictive benefit is relevant to evaluating behavioral foundation models, although these experiments are neither foundation-scale nor measurements of neural activity.

## 2. Two studies with separate evidence

Both studies originate from 666 annotated sequences spanning 103 uploads, with normal, Parkinson's, stroke, myopathic, and cerebral-palsy folder annotations. These labels are not diagnoses validated by this project. Study A's September 4, 2026 audit retains 657 sequences/100 sources with public metadata, 655/98 after media-span checks, and 639/97 after pose quality control. Source roles freeze before downstream attrition. Study B freezes 642 available pose archives and retains 625/93 after its own quality and target-computability gates.

| Design | Study A: classification pilot | Study B: laterality |
|---|---|---|
| Evaluation | Fold 0, seed 42; 59/18/20 train/validation/test sources | Five outer source folds; five seeds; two training variants |
| Encoder exposure | Cumulative annotation-defined stages | All outer-training categories together |
| Four-frame patch | Coordinate average; default width 64 | Flattened coordinates; verified width 96 |
| Local evidence | Saved notebook execution; current bundles absent | 50 trained checkpoints and held-out predictions verified |

The cohorts overlap, so Study B is neither an independent replication nor an extension of Study A's classifier experiment. Their sample counts and scores must not be pooled. Source grouping controls shared-upload dependence [@roberts2017crossvalidation], but the same person may occur in several uploads.

For Study B, the local audit loaded the cohort, splits, all 50 trained checkpoints, and 100,000 prediction rows spanning 16 evaluation lanes. Recomputing the two primary bootstrap tables from saved predictions reproduced them to numerical precision. This verifies the retained artifacts and report calculations; it is not independent retraining. The protocol was internally frozen after development, without external preregistration.

## 3. Methods

### 3.1 Representation learning and the classification pilot

Both implementations use 33 landmarks, pelvis centering, scale normalization, four-frame patches, and an exponential-moving-average target encoder. They define 64-frame inputs, verified for Study B and a documented default for Study A. Twelve anatomically specified landmarks are eligible masked targets. Preprocessing, predictor architecture, and objectives differ between studies.

Study A uses coordinate-averaged patches and a pooled-context MLP predictor, with the implemented loss
$$
L_A=L_{\mathrm{SmoothL1}}+0.10L_{\mathrm{variance}}+0.01L_{\mathrm{covariance}}.
$$
The optional condition cross-entropy term is disabled. Annotation labels nevertheless determine the cumulative normal-first exposure order. Training uses 59 sources; 18 validation sources select stage checkpoints. A balanced logistic classifier selects regularization on validation sources and is then refitted, including its scaler, on all 77 development sources. Test probabilities are averaged within each upload before scoring. Appendix A records preprocessing limitations and secondary results.

Study B uses four encoder layers, a two-layer Transformer predictor, four attention heads, and 96-dimensional tokens. Each patch flattens four XYZ observations. Invalid patches are excluded from attention, and up to 60% of eligible valid target positions are masked, limited by the batch minimum. Its centered latent cross-entropy uses student/teacher temperatures 0.10/0.06, with
$$
L_B=L_{\mathrm{latent\,CE}}+0.05(25L_{\mathrm{invariance}}+
25L_{\mathrm{variance}}+L_{\mathrm{covariance}}),
$$
using two-view VICReg-style regularization [@bardes2022vicreg]. Each of 300 epochs draws one clip per training source, then source-uniform padding produces four batches of 20: 1,200 updates per encoder. Vanilla and reflection-augmented training share initialization for each fold/seed; the augmented variant reflects each sample with probability 0.5. Checkpoints follow a fixed schedule, without early stopping.

### 3.2 Define a signed target before fitting a readout

Let $M$ negate the horizontal coordinate and swap the full anatomical left/right landmark map, including validity. Then $M^2=I$. For each of five pairs—shoulders, knees, ankles, heels, and foot tips—we calculate median left/right speeds $m_{L,k},m_{R,k}$ on common observed transitions. Both sides must be valid at both endpoints, with at least eight transitions per pair. Speeds use the original frame timestamps, and targets use uninterpolated normalized coordinates:
$$
y(X)=\frac{1}{5}\sum_{k=1}^{5}
\frac{m_{L,k}-m_{R,k}}{m_{L,k}+m_{R,k}+10^{-8}}.
$$
Hips define pelvis centering but are excluded from this contrast because their centered speed magnitudes coincide. The target satisfies $y(MX)=-y(X)$, including for an asymmetric observed gait. It is a dimensionless coordinate-derived index; clinical asymmetry and neural dominance are unmeasured.

For each pair, the native feature $A(X)$ concatenates the difference and sum of left/right mean tokens on common valid temporal support. Five pairs produce 960 features. This anatomical structure is supplied equally to trained and initial encoders.

### 3.3 Separate learned prediction from the sign constraint

Alongside the native readout, define equal-width odd and even features:
$$
\Phi^-(X)=\frac{A(X)-A(MX)}{\sqrt{2}},\qquad
\Phi^+(X)=\frac{A(X)+A(MX)}{\sqrt{2}}.
$$
The constructed odd readout $h(X)=w^\top\Phi^-(X)$ disables feature centering and the regression intercept. Origin-preserving scaling retains $h(MX)=-h(X)$ for any encoder and any fitted $w$. Its parameters learn to predict $y$; its sign behavior follows from construction. Native/free-intercept, odd/free-intercept, odd/zero-origin, and even/free-intercept lanes are evaluated for both learned and recorded initial features.

A separate test compares target-encoder tokens directly:
$$
q(X)=\frac{\|Z(MX)-SZ(X)\|_C^2}
{\|Z(MX)\|_C^2+\|SZ(X)\|_C^2}.
$$
Here $S$ swaps all anatomical token positions while leaving latent channels unchanged; $C$ selects common-valid positions. No alignment or fitted channel action is allowed. Lower $q$ indicates greater consistency under this particular action. Because the metric is uncentered, shared or input-insensitive features can lower it; a small initial value does not establish useful anatomical understanding.

### 3.4 Source-held-out fitting and uncertainty

Study B uses five stratified outer source folds, with 74–75 training and 18–19 test sources per fold. Four inner source folds select ridge regularization on outer-training embeddings. These inner folds tune the readout; encoder training may include their inputs. Scaling, fitting, and predictive scoring give each clip weight inverse to its source's clip count.

For each seed, all five outer folds' held-out predictions are pooled to compute source-balanced sequence-level $R^2$; the primary estimate averages the five seed scores. It is neither a source-mean-target score nor a seed-ensemble prediction score. The 2,000 paired bootstrap draws resample 93 source clusters, retaining each source's clips and all seeds together. Intervals are pointwise and conditional on the fitted models and fixed split; they do not include full-retraining or split-selection uncertainty. Fifty trained fits are not fifty independent datasets.

## 4. Results

### 4.1 The pilot does not establish a learned-feature advantage

| Study A readout | Macro-F1 | Balanced accuracy |
|---|---:|---:|
| Learned representation | 0.292 | 0.257 |
| Missingness control | 0.251 | 0.248 |
| Raw pose statistics | 0.441 | 0.443 |

On 20 held-out sources, raw pose exceeds learned features by 0.148 macro-F1. All three readouts miss all three stroke-annotated sources. With one fold/seed and no matched initialization control, this is a limited pipeline comparison; Study B's initialization controls do not retroactively isolate the cause of this result.

### 4.2 Laterality reveals different geometric and predictive outcomes

The vanilla native learned readout has $R^2=0.05979$, with interval $[-0.02527,0.12571]$. Its difference from paired initialization is $-0.01798$, $[-0.03851,0.00248]$. Thus a predictive gain from this pretraining recipe is unestablished; the interval does not demonstrate equivalence.

![Laterality controls across five source folds and five seeds. Panel A's first two contrasts use vanilla training; the constructed readout uses odd features and zero intercept. Panel B measures strict token error. Intervals are pointwise source-bootstrap intervals conditional on fitted models. Higher predictive differences and lower token-error differences favor the first condition.](figures/submission_laterality_effects.svg){width=100%}

Reflection augmentation improves strict token consistency relative to vanilla by $\Delta q=-0.00843$, $[-0.01020,-0.00687]$, a reduction of approximately 7.4%. Its predictive difference is only $+0.00408$ in $R^2$, $[-0.00556,0.01277]$. A measurable geometric improvement therefore coexists with uncertain predictive benefit.

Absolute token errors are 0.08322 for initialization, 0.11377 for vanilla training, and 0.10534 for augmented training. Relative to initialization, both trained variants increase $q$: vanilla by 0.03055 $[0.01576,0.04763]$, augmented by 0.02213 $[0.00844,0.03934]$. Both trained absolute intervals cross the protocol's operational threshold of 0.10. They fail its upper-bound criterion, without establishing that the population error exceeds 0.10 or excluding other latent symmetry actions.

The constructed odd readout passes the numerical output sign check for every seed, with recorded original-plus-mirrored predictions equal to zero. Under vanilla training, its learned-feature $R^2$ is 0.04302 $[-0.04356,0.11283]$, and the difference from its matched initial-feature lane is $-0.05874$ $[-0.09549,-0.01740]$. Guaranteed output antisymmetry consequently provides no demonstrated pretraining benefit in this experiment.

## 5. Interpretation, limitations, and data use

The contribution of explicit laterality is an experimentally grounded separation of a supplied anatomical constraint from information acquired during pretraining. Reflection augmentation improves one measured representation property, while the constrained readout shows that correct sign behavior can coexist with poor predictive utility. This is useful for BrainBodyFM's movement and pretraining-evaluation themes: a physically interpretable output should be checked against initialization and downstream error before its behavior is attributed to representation learning.

The study does not introduce symmetry projection, demonstrate successful laterality-supervised encoder training, or establish a universal limitation of JEPA. The two cohorts share underlying recordings; external replication, reliable person grouping, and calibrated clinical targets remain absent. Reflection acts on extracted coordinates rather than independently acquired camera views. Uncentered $q$ tests a specified channel action, and small-sample linear readouts cannot identify all information in a representation.

Public accessibility does not establish research consent or unrestricted reuse [@gavdRepo2026]. Gait trajectories can be identifying without RGB imagery. The laterality project's ethics determination, data-use review, and derived-pose release review are all recorded as unresolved; its explicit submission/release gate remains closed. This document is an internal draft pending those reviews and author approval. No clinical, surveillance, or deployment claim is supported.

## 6. Conclusion

Source-held-out controls make the laterality result more informative than a single accuracy score: reflection augmentation improves token consistency, its predictive gain remains unresolved, and enforced output antisymmetry does not establish useful pretraining. Together with the clearly scoped classification pilot, these findings motivate evaluating movement representations against both simple baselines and explicit transformation laws.

## Appendix A. Study A: implementation and exploratory diagnostics

The current evidence consists of retained fold-0/seed-42 notebook outputs. Post-QC roles contain 377/131/131 sequences from 59/18/20 training/validation/test sources. Reported manifest, split, and final-checkpoint SHA-256 identifiers begin with 7fd559e5105b, ff3518b87b1d, and f510be2a0453. The corresponding current checkpoint and evaluation bundles are absent from this checkout, so their bytes and derived embeddings have not been independently reverified.

Notebook 04 uses finite coordinates and visibility at least 0.45 for validity, pelvis centering, and a scale based on the median per-frame maximum of two-dimensional shoulder and hip widths. It performs no short-gap interpolation. Finite low-visibility coordinates remain present, and invalid positions are not removed through an attention-padding mask, even when excluded from pooling and target selection. Later diagnostic preparation interpolates internal gaps up to four frames and handles invalid coordinates differently. These limitations apply to Study A, not Study B's distinct preparation.

The implemented defaults are width 64, two encoder layers, four heads, batch size 32, 20 epochs of 100 steps per stage, learning rate 0.001, and EMA coefficient 0.996. Without the resolved run bundle, these are code defaults rather than independently confirmed execution settings. The cumulative schedule is normal, then plus Parkinson's, stroke, myopathic, and cerebral-palsy sequences; earlier categories remain available. An optional 0.10-weighted condition cross-entropy term is disabled in the primary record. No controlled condition-order or functional-retention experiment is complete.

Learned classifier features pool means and standard deviations across all landmarks and the 12-landmark subset, producing $4d$ features (256 at default width). Raw pose supplies 144 features: coordinate means/stds, mean absolute first differences, and first-difference stds across 12 landmarks and three coordinates. Missingness supplies 97 features at the defaults. Standardization and balanced logistic regression fit source-mean features; validation selects $C\in\{0.1,1,10\}$, then fitting repeats on all 77 development sources. At test time, sequence probabilities are averaged within each source. Exact macro-F1 values are 0.292424, 0.251111, and 0.440513 for learned, missingness, and raw features; accuracies are 0.30, 0.30, and 0.50.

Temporal probes freeze the encoder and compare equal-width global mean/standard-deviation, signed temporal-moment, and four-bin summaries. Four-bin pooling increases peak-position $R^2$ from 0.173011 to 0.317745 and reduces MAE from 0.092086 to 0.075263 in normalized clip time. A late/early motion ratio improves from $R^2=0.105126$ to 0.176175. All bilateral ankle-height-lag readouts have negative $R^2$. Their unvalidated pose-derived targets and changed preprocessing prevent a physiological interpretation.

Normal-anchor cosine compares each clip's final representation with its normal-only-checkpoint representation, averaging within and then across sources. Final values are 0.701058 on five validation-normal sources and 0.849632 on seven test-normal sources. They measure geometric drift; functional forgetting, Procrustes/CKA comparisons, and improved consolidation remain untested. Forecasting is blocked: its separately trained checkpoint is absent, the producer lacks the required future objective, and the proposed mask leaves 21 future landmarks visible. No forecasting result enters this paper.

## Appendix B. Study B: reproducible target and fitting details

The retained cohort contains 270 normal, 39 Parkinson's, 75 stroke, 183 myopathic, and 58 cerebral-palsy sequences, respectively from 29, 9, 18, 28, and 9 sources. Each source is outer test once per seed. The five test folds contain 189, 182, 72, 77, and 105 clips, illustrating why source-balanced weighting matters. Seeds are 42–46. The five seed-specific initializations are reused across fold-local training runs and paired between variants.

Target preparation uses finite coordinates and visibility at least 0.45, requires both hips for centering, and uses the median of pooled three-dimensional shoulder/hip distances for scale. The speed contrast uses uninterpolated, common-valid transitions and original timestamp differences. Its mean is -0.00607, standard deviation 0.05915, and observed range [-0.19482, 0.21468]. Input preparation separately interpolates internal gaps of at most four frames, resizes to 64 frames, applies a resized-validity threshold of 0.999, and zeroes invalid coordinates. The 0.50 coverage gate uses this prepared validity mask and can include filled gaps.

Sixteen temporal patches across 33 landmarks yield 528 token positions. The target mask samples 60% of eligible valid positions, with the count limited by the batch minimum. Input embeddings at targets are hidden; invalid patches are padded out of attention. The EMA teacher sees the complete valid input. Two views use small rotations (up to eight degrees) and XY translations (up to 0.03 normalized units). Sample-consistent reflection, when enabled, precedes view generation.

AdamW uses learning rate 0.001, betas (0.9, 0.95), weight decay 0.05, cosine learning-rate decay, and gradient-norm clipping at 1. EMA increases from 0.999 toward 1, and target centering uses coefficient 0.9. Each model receives 1,200 updates. The fixed schedule removes validation-based checkpoint selection in this study.

Ridge penalties are selected from $10^{-3},10^{-2},10^{-1},1,10,100,1000$. Outer-test rows enter neither encoder fitting nor readout selection. Weighted fitting, scaling, and scoring give each source total weight one. The constructed odd/zero-origin lane uses no feature centering and no intercept; it needs both original and reflected encoder evaluations. The initialization control receives the same anatomical pooling and readout search.

Strict token scoring requires at least eight common-valid tokens and rejects negligible representation energy. It uses the full 33-landmark permutation, an identity latent-channel action, and no Procrustes alignment. The native, odd/free, odd/zero-origin, and even/free lanes separate parity projection from intercept removal. Additional measured-nuisance and target-component controls are secondary; the target-component oracle is an algebraic self-consistency check, not an independent learned baseline.

Absolute strict-error intervals are 0.08322 [0.07458, 0.09404] for initialization, 0.11377 [0.09513, 0.13843] for vanilla, and 0.10534 [0.08783, 0.12778] for augmented training. The protocol requires an upper interval bound below 0.10 together with improvement over initialization; 0.10 is an operational, uncalibrated threshold. Output antisymmetry uses numerical tolerance $10^{-6}$, separately from token error. Secondary high-coverage and seed-ensemble reports use different estimands and are excluded from the primary comparisons.

## Appendix C. Evidence and release ledger

Study B's protocol SHA-256 begins 6f7baefbda07, cohort 28c164fae903, and split 0dd230e67d5eb. Its artifact root is the protocol-keyed paper run in the laterality project. The primary numerical sources are the checkpoint source-bootstrap report and strict representation-equivariance source-bootstrap report; the figure generator records their exact filenames, full hashes, and plotted values. The local audit checked cohort/split compatibility, checkpoint state and lineage, source weights, saved token-error energy ratios and common-token counts, and prediction coverage. Recalculation of both bootstrap reports differed from saved values by less than $10^{-16}$, without rerunning encoder inference, retraining, or altering artifacts.

| Claim | Evidence boundary |
|---|---|
| Raw features lead the classifier pilot | Retained Study A notebook outputs; one fold/seed |
| Augmentation improves strict token consistency | Verified Study B predictions; paired conditional source-bootstrap interval |
| Pretraining improves laterality prediction | Not established against paired initialization |
| Constructed output reverses sign | Algebra plus saved numerical checks; not evidence of learned encoder equivariance |
| External or clinical generalization | Not evaluated |
| Submission/release authorization | Blocked by three unresolved project reviews |

The Study A temporal, drift, and forecasting analyses remain distinct from Study B. Older transductive laterality narratives and simulated outcomes in Study A notebooks 05a–05d are excluded. No laterality performance is imported from those archived branches.
