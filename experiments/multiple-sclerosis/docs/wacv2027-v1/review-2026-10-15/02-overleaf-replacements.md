# Copy-and-paste revisions for Overleaf

These passages revise the original `docs/wacv2027-v1/wacv-v1.pdf`. They use the corrected saved execution of 9 October 2026; the earlier Random Forest result is retained explicitly. Apply all numerical changes together. Individual `.tex` files are in [latex/](latex/); they are fragments, not a complete manuscript or an already compiled Overleaf project.

The guide deliberately provides full replacements for the results and discussion, so that the positive result develops coherently rather than being inserted between repeated warnings. No neural model was retrained for this review. Numerical verification and the distinction between the two executions are recorded in [the evidence note](03-evidence-and-submission.md).

Before pasting, keep the existing bibliography and official WACV review style. The fragments use `stenum2021` [25], `stenum2024` [26], `filli2018` [1], `plotnik2020` [5], and `dino` [35] for references already in the original paper. Substitute the original project’s keys if they differ. R3c adds the closely related studies `kaur2023` and `gaitforemer`; their verified entries are in [additional-references.bib](additional-references.bib). All seven keys also exist in `docs/wacv2007/references.bib`. Do not duplicate keys when merging bibliography entries. R3b replaces sentence text only and retains the original MaskFeat citation [38]. Add `\label{tab:source_partitions}` immediately after Table 2's caption, or replace that label in R8 with the project's existing key. The other new labels are defined in the supplied fragments. Replacing whole sections also means removing their old floating tables/figures to avoid duplicates.

For Figure 1, box 5 should read “Feature prediction without condition labels / 800 updates; 400-update continuation tested”; box 6 should read “Frozen encoder + linear classifier / Compare with handcrafted and simple pose features”. The continuation starts from learned weights with restarted optimization. Keep the original schematic only after editing these labels. R16 supplies the matching captions.

The statistical note R14 belongs in a supplementary document. Main-text details removed from §4.3, especially its feature-distribution equation, can also go there unchanged. When moving original equation (2), retain its DINO attribution [35], using the project’s existing key or `\cite{dino}`, in the accompanying explanation. The source-partition table and dataset table remain numerically valid. The original Figure 3 is removed from the main text; if retained in the supplement, regenerate it using the current diagnostic values in the evidence JSON rather than the older endpoints.

## R0. Recommended title

Page 1, replace the title.

```latex
\title{Learning Skeleton Representations\\from a Small Collection of Neurological Gait Videos}
```

## R1. Abstract

Page 1, replace the complete abstract. The corrected RF is the main comparison; its earlier implementation is disclosed in the results.

```latex
\begin{abstract}
Learning useful descriptions of movement from limited video data could broaden access to quantitative behavioral analysis. We evaluate a compact Skeleton Joint Embedding Predictive Architecture (S-JEPA) on gait videos labeled Normal, multiple sclerosis, or Parkinson's disease. In each split, the encoder learns from 24 source videos, comprising 45--58 clips, by predicting hidden skeleton features without condition labels. A separate linear classifier uses its frozen features to predict collection labels for recordings excluded from training. Evaluation across five source-grouped folds covers 88 usable clips from 41 videos. S-JEPA achieves source-weighted macro-F1 of 0.457, compared with 0.325 for a Random Forest using handcrafted joint-angle features and 0.452 for mean pose. With equal weight per clip, S-JEPA also exceeds the Random Forest, scoring 0.397 versus 0.321. These results show that a compact predictive skeleton model trained on a small video collection can produce useful features for held-out label prediction. Mean pose remains a strong comparison, and further controls are needed to isolate the contribution of pretraining and temporal information. The study supports representation learning for markerless behavioral analysis, with clinical interpretation requiring recordings with verified diagnoses and participant identities.
\end{abstract}
```

## R2. Introduction: study question and contribution

Page 1, replace the final paragraph beginning “The aim of this study…” with these two paragraphs. Preserve the earlier domain references.

```latex
We investigate whether a compact predictive skeleton model can produce useful representations from a small collection of gait videos. The model learns to predict hidden features of pose sequences without condition labels; a separate linear classifier then tests whether the frozen features contain information associated with the collection labels. Each encoder is trained on only 24 source recordings. Keeping all clips from a source together allows us to evaluate recordings that were excluded from both encoder and classifier fitting.

The main contribution is an empirical demonstration of useful trained features in this limited-data setting. S-JEPA exceeds a corrected Random Forest using handcrafted joint-angle summaries under both source-weighted and clip-weighted scoring. A mean-pose comparison helps assess how much can already be predicted from average pose without frame order, while a visibility-only comparison examines information available in the pose detector's outputs. Together, these experiments provide a grounded case study of representation learning for markerless behavioral analysis and identify the comparisons needed before interpreting the features as measures of neurological gait impairment.
```

## R3a. Related work: pose-validation scope

Page 2, §2.1, replace the sentence beginning “Stenum et al. have validated…” and its immediate generalization about reliability. Attach the original citations [25,26] to the first sentence; use the citation keys in the guide below.

```latex
Stenum et al. evaluated OpenPose-derived gait measures against reference measurements and extended this work across camera perspectives and clinical populations~\cite{stenum2021,stenum2024}. These studies establish the potential of video-based gait measurement while showing why the pose estimator, recording conditions, and intended measurement require validation. Their findings do not independently validate the BlazePose processing used in the present collection.
```

## R3b. Related work: MaskFeat

Page 2, §2.3, replace only the sentence beginning “Wei et al. demonstrated…”; retain the existing citation command for reference [38] after this sentence.

```latex
Wei et al. explored predicting features of masked image or video regions, including handcrafted image descriptors, as a way to learn visual representations without class labels.
```

## R3c. Related work: learned neurological gait analysis

Page 2, append this paragraph at the end of §2.2 before §2.3. Add the two entries in additional-references.bib unless those keys are already in the Overleaf bibliography.

```latex
Kaur et al. used multi-view video and estimated 3D joints to classify gait from people with MS, PD, and matched older adults, including evaluations across participants and walking tasks~\cite{kaur2023}. GaitForeMer combined motion forecasting with activity classification during pretraining and then adapted its representations to PD gait-impairment severity~\cite{gaitforemer}. These studies place the present work within an existing line of learned neurological gait analysis. Our experiment examines a small collection of heterogeneous 2D web videos through a frozen-feature classifier; differences in labels, inputs, and evaluation designs prevent direct comparison of the reported scores.
```

## R4. Anatomical masking

Page 3, replace all of §4.1. The heading can be renamed “Anatomical masking”. This replaces the incorrect “hidden at least once” percentages as well as the excess diagnostic detail.

```latex
\subsection{Anatomical masking}
For each training window, the model predicts hidden features from the visible parts of the skeleton sequence. We sample masks over seven body regions: the head, mouth, left and right arms, trunk, and left and right legs. A selected region is hidden across one to six consecutive time blocks, and regions are added until approximately 60\% of the landmark--time tokens are hidden. Overlap and the final region can increase this proportion; a check of 5,000 sampled masks gave an average of 63.2\%.

Regions containing any of 12 gait-relevant landmarks---both shoulders, hips, knees, ankles, heels, and foot indices---receive a 1.5-fold multiplier in their selection weights, with an adjustment for overlapping region membership. The multiplier also affects the arm regions because they contain shoulders; it does not mean that every landmark in the selected set is hidden 1.5 times as often. All 33 landmarks can serve as visible context or hidden targets across successive masks. At least one token from the selected landmark set remains visible somewhere in each window.

This anatomical preference is motivated by the role of lower-limb movement and body coordination in gait~\cite{filli2018,plotnik2020}. The sampling checks verify how the masks are constructed; a separate comparison would be needed to determine whether the preference improves classification.
```

## R5. Training strategy

Page 4, replace §4.3 in full. Move its original equation (2), optimizer settings and mathematical diagnostic definition to the supplementary methods; preserve their values, except use current-run diagnostics in R14.

```latex
\subsection{Training strategy}
The encoder is trained on the Normal-, MS-, and PD-labeled videos in the training partition, without using their labels in the feature-learning objective. Each batch samples source videos with equal probability and then samples windows within those sources. This prevents recordings with more windows from dominating encoder training. Batches contain 32 windows.

The context encoder receives an augmented view with parts of the skeleton hidden, and the predictor estimates the target encoder's features at the hidden positions. Training compares distributions over feature dimensions using cross-entropy with centering and sharpening~\cite{dino}. These distributions describe learned features, rather than the three collection labels. The target encoder is updated gradually from the context encoder's weights. Spatial augmentation uses rotation within $\pm15^\circ$, scaling from 0.9 to 1.1, translation within $\pm0.1$ normalized units, and horizontal reflection with probability 0.5. A single transformation is applied throughout each window. Reflection exchanges left and right landmark slots; its effect on useful gait asymmetry was not separately evaluated.

We train for 800 updates and evaluate a further 400-update continuation from the learned weights using the same training sources and objective. The continuation restarts the optimizer, running center, random-number generators and schedules. It therefore differs from uninterrupted training for 1,200 updates. For each stage, a linear classifier is fitted to the frozen features. The stage with higher source-weighted validation macro-F1 is selected, retaining the initial stage on a tie. Validation videos are excluded from encoder and classifier fitting, and test predictions are generated only after stage selection. The selected model is not refitted using the validation videos.
```

## R6. Frozen representation and classifier

Page 5, replace §4.4 in full. This preserves the unusual fixed-mask pooling operation and the training-only fitting boundary.

```latex
\subsection{Frozen representation and classifier}
After training, the target encoder's weights are fixed. Each complete, unmasked window is passed through this encoder, and features at positions selected by one fixed sampled mask are averaged into a 96-dimensional vector. The mask selects outputs for averaging; it does not hide input landmarks during classification. Window vectors are then averaged to give one vector per clip. The same readout mask is used for all clips and folds, and sensitivity to that mask was not tested.

A scaler fitted on training clips standardizes the features, and logistic regression with $C=1$ and inverse-frequency class weights predicts the three collection labels. Each training clip contributes one observation. Thus, sources with more clips contribute more observations to classifier fitting, even though source videos are sampled equally during encoder training. Validation and test clips contribute to neither feature scaling nor classifier fitting.
```

## R7. Handcrafted Random Forest baseline

Page 5, §5.1, replace the first paragraph only. Keep the following mean-pose, visibility and majority descriptions.

```latex
The handcrafted Random Forest (RF) uses summaries of left and right hip, knee, and ankle angles, including mean angles, bilateral differences, and ranges of movement. The feature vector has 82 available fields, of which 15 angle statistics vary in this collection; the remaining fields are unpopulated or constant. Columns with no variation in the training partition are removed. A training-fitted scaler precedes an RF with 100 trees, maximum depth five, square-root feature subsampling, and balanced class weights. We use the corrected extractor with separate left- and right-ankle range features. An earlier implementation duplicated the right-ankle range, and its result is reported separately for transparency. This baseline summarizes lower-limb angles, whereas S-JEPA uses all 33 landmarks and visibility values, so the comparison evaluates these particular analysis pipelines.
```

## R8. Source-grouped evaluation

Page 5, replace §5.2. Retain Table 2, whose counts are correct.

```latex
\subsection{Source-grouped evaluation}
We divide the 41 source videos into five test folds, keeping the proportions of the three collection labels approximately balanced. Within the remaining sources, the first split of a stratified four-fold partition defines the training and validation sets. Each encoder and classifier is fitted using 24 source videos, comprising 45--58 clips depending on the fold (Table~\ref{tab:source_partitions}).

All clips and windows from a source remain in the same partition. Across the five test folds, every usable clip receives one prediction from a model that excluded its source from training and validation. This procedure separates identified source recordings; it cannot establish separation of participants who may appear in more than one source video.
```

## R9. Evaluation measures

Page 5, replace §5.3. Move the bootstrap procedure to R14. Assign the source-partition table the label used in R8 or substitute the project’s existing label.

```latex
\subsection{Evaluation measures}
We report macro-F1, which combines precision and recall separately for each collection label and then averages the three scores. Its range is zero to one, with larger values indicating better classification. The main score gives every source video equal total weight: a clip from a source containing $n$ usable clips receives weight $1/n$. Weighted prediction counts are combined across the five test folds before F1 is calculated. This prevents a recording with many clips from dominating the result.

We also report scores giving every clip equal weight and the number of correctly classified clips. These answer a different question because recordings contribute different numbers of clips. Fold results show how performance changes with the held-out recordings. Classes with undefined precision or F1 receive a value of zero. Supplementary material reports the resampling analysis and its scope.
```

## R10. Results: replace the complete section

Pages 5–6, replace all of §6, including the original §6.1–6.3 text and Table 3. Remove the old Figure 3 from the main text, and replace old Figure 4 using R13. Figure numbering will change automatically.

```latex
\section{Results}
\subsection{Classification using learned representations}
S-JEPA's frozen features support prediction of collection labels on recordings excluded from training. The pooled source-weighted macro-F1 is 0.457, compared with 0.325 for the corrected handcrafted RF (Table~\ref{tab:heldout_revised}), a difference of 0.131 on the F1 scale. With equal weight per clip, S-JEPA also exceeds RF, scoring 0.397 versus 0.321 and correctly classifying 36 of 88 clips compared with 30 for RF. Each encoder was trained on only 24 source videos.

Mean pose is a strong comparison, with source-weighted macro-F1 of 0.452. It exceeds S-JEPA with equal clip weighting (0.410 versus 0.397) and correctly classifies 37 clips. The small difference under source weighting therefore provides little basis for attributing S-JEPA's predictions specifically to temporal information. One MS-labeled source contributes 13 clips, illustrating why the two weighting schemes can produce different rankings.

\begin{table}[t]
\centering
\small
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}p{0.43\columnwidth}rrr@{}}
\hline
Method & Source F1 & Clip F1 & Correct clips \\
\hline
S-JEPA + linear classifier & 0.457 & 0.397 & 36/88 \\
Joint angles + corrected RF & 0.325 & 0.321 & 30/88 \\
Mean pose + linear classifier & 0.452 & 0.410 & 37/88 \\
Visibility + linear classifier & 0.318 & 0.318 & 29/88 \\
Majority class & 0.259 & 0.202 & 23/88 \\
\hline
Earlier RF (duplicate) & 0.411 & 0.397 & 37/88 \\
\hline
\end{tabular}
\caption{Held-out classification of the collection labels. Macro-F1 combines predictions across all five test folds, weighting each source equally (Source F1) or each clip equally (Clip F1). Correct clips are unweighted counts. The final row records the earlier RF implementation with a duplicated ankle-range field; it is separate from the corrected main comparison.}
\label{tab:heldout_revised}
\end{table}

The earlier RF implementation scores 0.411 with source weighting. S-JEPA also exceeds that result, by 0.045, although the two clip-weighted scores round to 0.397 and the earlier RF is slightly higher before rounding. Correcting the ankle feature changes 12 RF predictions and lowers its score. The earlier and corrected executions use the same collection, partitions and seed, and produce identical S-JEPA predicted labels. They do not provide independent evidence across training seeds or datasets.

\subsection{Variation across recordings and labels}
S-JEPA exceeds corrected RF in three of five test folds, with fold macro-F1 ranging from 0.217 to 0.780 (Figure~\ref{fig:results_revised}). Mean pose exceeds S-JEPA in four folds. The higher pooled S-JEPA score reflects performance across all held-out predictions, including its strong result in fold 2, rather than a consistent advantage in every partition.

With equal source weighting, S-JEPA correctly identifies 39.6\% of Normal-, 42.9\% of MS-, and 56.9\% of PD-labeled examples, compared with 27.1\%, 23.1\%, and 51.0\% for corrected RF. PD predictions also include many examples from the other labels: their source-weighted precision is 38.6\%. In raw clip counts, S-JEPA correctly classifies 10 of 24 Normal clips, 8 of 29 MS clips and 18 of 35 PD clips; 13 MS clips are assigned to PD. The source-weighted percentages and raw counts use different weights. These errors limit clinical interpretation even though the features support useful prediction within this collection.

\subsection{Learning during training}
The training records show that the model fits the hidden-feature prediction task while retaining variation across inputs. In the saved fold-0 training history, the first and last 50 updates of the initial stage have mean losses of 3.858 and 1.022, respectively. At the end of that stage, effective rank---a measure of how varied a batch of features is---is 12.13; the corresponding values across folds range from 10.90 to 12.13. Together with the held-out classifier results, these observations support the practical usefulness of the trained representation. They do not measure how much classification improved relative to an untrained encoder.

The continuation stage lowers the final training-batch loss in every fold, but validation selects it only in folds 3 and 4. The initial stage is retained in folds 0--2. Further optimization of the training objective therefore does not consistently improve label prediction; the validation step determines which stage is used for each test fold.
```

## R11. Discussion: replace the complete section

Pages 6–8, replace §7.1–7.3 and their heading with this connected discussion. Add an ethics/data-use statement only when its factual basis is documented.

```latex
\section{Discussion}
\subsection{What the trained representation contributes}
The central finding is that a compact S-JEPA trained on a small collection of pose sequences produces features that support label prediction on held-out source recordings. A linear classifier using these features exceeds the evaluated handcrafted joint-angle RF under both source-weighted and clip-weighted scoring. This makes predictive skeleton learning a useful approach to investigate when large, clinically annotated movement datasets are unavailable.

The training history and downstream predictions provide complementary evidence. The model fits its feature-prediction objective without reducing all inputs to a constant representation, while the frozen-feature classifier measures information associated with the collection labels. A matched untrained encoder would establish how much of that information is gained through pretraining. Mean pose's similar performance also leaves open the contribution of temporal order, as S-JEPA differs from that baseline in visibility inputs, feature dimension and the frames included in its window averages.

\subsection{Scope of the evidence}
The collection was assembled from publicly accessible gait videos, and its Normal, MS and PD labels were not independently clinically verified. Grouping by source video prevents clips from one identified recording from crossing partitions, but the same participant could appear in different recordings. Camera viewpoint, occlusion, pose-estimation errors and other recording characteristics may also contribute to prediction. The visibility-only score of 0.318 shows that detector outputs contain label-associated information; it does not identify whether that information arises from gait or recording conditions.

The RF comparison uses a specific set of lower-limb angle summaries, while S-JEPA uses all 33 landmarks and visibility. It therefore supports a comparison of these implemented pipelines, without establishing an advantage over every handcrafted gait analysis. The small number of training recordings also limits how confidently performance can be extended to new settings. Although the S-JEPA encoder is fitted on these recordings, its MediaPipe pose-estimation front end is pretrained; this is a small-data representation-learning experiment rather than a measurement of the full pipeline's data requirements.

Preprocessing further limits biomechanical interpretation. Centering on the hips and scaling by torso length remove absolute position and apparent body size, so the outputs are not calibrated measurements of walking speed or stride length. Missing detections are interpolated without a maximum gap length. Fifteen short clips are padded by repeating their final frame---five Normal and ten PD clips, with none from the MS collection---which can introduce label-associated temporal patterns. The minimum accepted clip length is eight sampled frames, so a clip need not contain a full gait cycle.

\subsection{Next comparisons}
The most informative next comparison would use the same architecture and readout with a randomly initialized encoder, followed by a supervised model trained on the same sources. Training and evaluating a matched model with shuffled frame order, while holding frames and visibility inputs fixed, would test whether temporal structure contributes to prediction. Comparing anatomical, unbiased regional and motion-based masking would assess the masking choice directly. Evaluation on recordings with verified diagnoses, participant identities and new acquisition settings would then test whether the learned features support reliable behavioral or clinical measurements.
```

## R12. Conclusion

Page 8, replace §8 in full.

```latex
\section{Conclusion}
S-JEPA produced useful features for gait-video label prediction after training on 24 source recordings per fold. Its frozen features, paired with a linear classifier, achieved source-weighted macro-F1 of 0.457, exceeding the corrected handcrafted joint-angle RF at 0.325 with performance close to the mean-pose comparison under both reported weighting schemes. This study establishes a practical small-data example of predictive representation learning for markerless behavioral analysis. Comparisons with untrained encoders and clinically verified recordings would clarify the contribution of pretraining and the relevance of these features to neurological movement assessment.
```

## R13. Corrected replacement for Figure 4

Page 7, replace the complete original Figure 4. This LaTeX-native figure is a compact numerical display; it requires no image download or plotting package. In panel (a), fold numbers retain the original 0–4 identifiers. In panel (b), N = Normal, M = MS, P = PD. The grouping changes from separate panels to one comparison table, so remove old panel-specific references.

```latex
\begin{figure*}[t]
\centering
\begin{minipage}[t]{0.36\textwidth}
\centering
\small
\textbf{(a) Held-out macro-F1 by fold}\\[5pt]
\begin{tabular}{rrrr}
\hline
Fold & S-JEPA & RF & Mean pose \\
\hline
0 & 0.515 & 0.316 & 0.554 \\
1 & 0.363 & 0.267 & 0.500 \\
2 & 0.780 & 0.323 & 0.187 \\
3 & 0.316 & 0.364 & 0.510 \\
4 & 0.217 & 0.256 & 0.386 \\
\hline
\end{tabular}
\end{minipage}\hfill
\begin{minipage}[t]{0.62\textwidth}
\centering
\small
\textbf{(b) Predictions within each collection label (\%)}\\[5pt]
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lrrr|rrr|rrr}
\hline
 & \multicolumn{3}{c|}{S-JEPA} & \multicolumn{3}{c|}{Corrected RF} & \multicolumn{3}{c}{Mean pose} \\
Label & N & M & P & N & M & P & N & M & P \\
\hline
Normal & 39.6 & 27.1 & 33.3 & 27.1 & 49.0 & 24.0 & 43.8 & 25.0 & 31.2 \\
MS & 14.5 & 42.9 & 42.6 & 15.4 & 23.1 & 61.5 & 12.1 & 36.4 & 51.5 \\
PD & 24.4 & 18.7 & 56.9 & 24.1 & 24.9 & 51.0 & 21.9 & 20.7 & 57.4 \\
\hline
\end{tabular}
\end{minipage}
\caption{Held-out classification with equal total weight per source video. (a) Each row uses the same test sources for all methods; higher macro-F1 indicates better classification. (b) Each method's three columns show predicted Normal (N), MS (M), and PD (P) labels. Percentages are normalized within the true collection label and may not sum to 100 after rounding. RF uses the corrected bilateral ankle-range features.}
\label{fig:results_revised}
\end{figure*}
```

## R14. Supplementary note: uncertainty and training details

Add to supplementary material, replacing the original bootstrap paragraph and §6.3 interval discussion if those are moved. These intervals were recomputed from the corrected saved predictions for this review; they are not copied from the original paper.

```latex
\section{Variation in the retained predictions}
We resampled source videos within each collection label 10,000 times, retaining 16 Normal, 13 MS and 12 PD source draws per resample. Sampling was with replacement. Each sampled source contributed total weight one across its clips, with repeated draws counted repeatedly. All methods used the same draws. The calculation used NumPy's PCG64 generator with seed 42 and sorted source identifiers within each label. The 2.5th and 97.5th percentiles of the resulting scores define the intervals reported here.

For the corrected execution, source-weighted macro-F1 is 0.457 for S-JEPA (interval 0.331--0.580), 0.325 for RF (0.207--0.445), and 0.452 for mean pose (0.324--0.573). The paired S-JEPA minus RF difference is 0.131 (interval $-0.046$ to $0.309$); the difference from mean pose is 0.005 ($-0.167$ to $0.174$). The range of differences includes either method performing better under resampling. This describes uncertainty around the observed ranking without establishing equivalence or a reliable advantage in new populations.

These calculations keep the trained models and predictions fixed. They do not include changes from retraining, checkpoint selection or alternative source partitions, and they treat source videos as independent units even though participant identities across videos are unknown. Fold variability describes the same collection under overlapping training partitions, rather than five independent replications.

\section{Implementation and training diagnostics}
The encoder has three transformer layers with width 96 and four attention heads; the predictor has two layers of width 96. A 32-frame window is grouped into eight four-frame blocks, yielding 264 landmark--time tokens. AdamW uses peak learning rate $10^{-3}$, weight decay 0.05, coefficients $(0.9,0.95)$ and gradient-norm clipping at 1. The first 10\% of each stage warms up the learning rate, followed by cosine decay. Target-encoder momentum rises from 0.996 toward 1.0. Predictor and target temperatures are 0.10 and 0.06; the running center retains 90\% of its previous value at each update.

Effective rank is the exponential of the entropy of normalized singular values from centered batches of 32 pooled target-encoder vectors. The centered batch has a maximum rank of 31. Initial-stage endpoint ranks are 10.896--12.135; continuation-stage endpoints are 9.558--11.638. These are diagnostics of feature variation, not measurements of clinical separation. Endpoint losses describe single training batches, so averages over updates are more informative about the training trend.

The mask-frequency check used 5,000 independent masks with seed 42. The mean hidden-token fraction was 0.632. For example, the left shoulder was hidden in 78.3\% of landmark--time positions but in at least one time block in 98.6\% of windows. The token fraction and the fraction of windows describe different masking events. Sampling diagnostics do not determine whether the anatomical preference improves classification.
```

## R15. Preprocessing: duration and missing detections

Page 3, §3, replace ONLY the sentence beginning “Each 32-frame window spans…” and ending “…for representation learning.” with the first paragraph below. Append the second paragraph to the existing final windowing/padding paragraph of §3. Preserve the intervening MediaPipe extraction, cleaning description, equation (1), Table 1 and their citations.

```latex
The target sampling cadence is approximately 15 Hz, with the actual cadence depending on the original frame rate and integer sampling stride. Windows contain 32 sampled frames with a stride of 16 frames; their duration therefore varies across recordings.

% Append this sentence to the existing final padding paragraph of Section 3:
Padding short clips by repeating the final frame does not add observed movement.
```

## R16. Figure captions and pipeline labels

Page 3, Figure 1: use the first caption below and replace box 5 text as instructed in the guide. Page 4, Figure 2: use the second caption. Keep the corresponding image assets.

```latex
% Figure 1 replacement caption:
\caption{From gait clips to held-out label prediction. All clips from one source recording remain in the same training, validation or test partition before windows are constructed. Each encoder is fitted on 24 source videos. The frozen-feature classifier uses collection labels, while the encoder's feature-learning objective does not.}

% Figure 2 replacement caption:
\caption{BlazePose landmarks used to define the anatomical sampling preference. Both shoulders, hips, knees, ankles, heels and foot indices belong to the selected set. Regions containing these landmarks receive increased selection weight, with an overlap adjustment; this does not prescribe a fixed hiding probability for individual landmarks. All 33 landmarks remain available to the model.}
```

