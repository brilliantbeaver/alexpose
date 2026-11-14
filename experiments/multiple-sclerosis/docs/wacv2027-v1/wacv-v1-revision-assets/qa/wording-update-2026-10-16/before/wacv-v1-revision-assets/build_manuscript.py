"""Assemble the reviewed passages into complete, editable LaTeX manuscripts.

The generated .tex files are fully expanded and do not depend on this script
or the review folder when uploaded to Overleaf.
"""
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parent.parent
REVIEW = ROOT / "review-2026-10-15"


def fragment(name):
    return (REVIEW / "latex" / f"{name}.tex").read_text().split("\n", 1)[1].strip() + "\n\n"


PREAMBLE = r"""\documentclass[10pt,twocolumn,letterpaper]{article}
\usepackage[T1]{fontenc}
\usepackage[review,applications]{wacv}
\usepackage{graphicx,booktabs,amsmath,amssymb,microtype,placeins}
\definecolor{wacvblue}{rgb}{0.13,0.34,0.54}
\usepackage[pagebackref,breaklinks,colorlinks,allcolors=wacvblue]{hyperref}
% Replace DRAFT with the assigned OpenReview paper identifier before submission.
% The WACV style option is distinct from choosing Vision4Neuro in OpenReview.
\def\wacvPaperID{DRAFT}
\def\confName{WACV}
\def\confYear{2027}
"""
END = r"""
\FloatBarrier
\bibliographystyle{ieeenat_fullname}
\bibliography{wacv-v1-revision}
\end{document}
"""

main = PREAMBLE + fragment("r0") + r"""
\author{Anonymous submission}
\begin{document}
\maketitle
""" + fragment("r1") + r"""
\section{Introduction}
Walking provides an observable expression of motor function, making gait a useful subject for behavioral and clinical neuroscience. Multiple sclerosis (MS) can affect joint movement, stability and coordination in different combinations across individuals~\cite{filli2018,plotnik2020}. Video-based gait measurements have also been investigated in Parkinson's disease (PD)~\cite{shin2021}. These findings motivate the analysis of movement from ordinary recordings, where camera position, visibility and recording conditions accompany the movement of interest.

Markerless pose estimation converts video into trajectories of estimated body landmarks. Comparisons with reference measurements show that video-derived gait measures can be useful under specified recording and analysis conditions~\cite{stenum2021,stenum2024}. A common next step is to summarize these trajectories using quantities chosen in advance, such as joint angles and their ranges. Learning a representation directly from pose sequences offers another way to describe movement, without specifying every potentially useful relationship between landmarks.

Self-supervised methods create a training task from the observations themselves, allowing a model to learn features before condition labels are used. S-JEPA, introduced by Abdelfattah and Alahi, predicts features of hidden skeleton regions from visible context~\cite{sjepa}. Applying this framework to a small, heterogeneous gait-video collection tests whether the resulting features can be useful when few study-specific recordings are available.

""" + fragment("r2") + r"""
\section{Related work}
\subsection{Video-based gait analysis}
Pose estimators such as BlazePose provide body landmarks from ordinary video~\cite{blazepose}.
""" + fragment("r3a") + r"""
Shin et al.\ investigated quantitative gait analysis from single-camera videos of people with PD~\cite{shin2021}. Such studies connect defined video measurements to clinical or biomechanical references. The present collection instead supports a study of label prediction, because independently verified diagnoses and reference gait measurements are unavailable.

\subsection{Learning representations of movement}
SkeletonMAE learns representations by reconstructing masked skeleton information~\cite{skeletonmae}. MotionBERT learns motion representations by recovering 3D motion from partial or corrupted 2D observations~\cite{motionbert}. These methods motivate learning relationships among landmarks rather than relying exclusively on predefined summary features.

""" + fragment("r3c") + r"""
\subsection{Predicting hidden features}
""" + fragment("r3b").replace("class labels.", r"class labels~\cite{maskfeat}.") + r"""
Joint-embedding predictive architectures learn by predicting feature vectors for hidden parts of an input from the visible context, as developed for images~\cite{ijepa}, video~\cite{vjepa} and skeletons~\cite{sjepa}. In S-JEPA, a second encoder supplies target features and is updated gradually from the context encoder. Prediction takes place in feature space, without reconstructing raw joint coordinates.

The original S-JEPA samples higher-motion tokens for hiding. Our compact adaptation uses 2D landmarks and detector visibility, with stochastic masks over anatomical regions. The experiment examines this trained representation in a small neurological gait-video collection; it does not compare the anatomical sampler with the original motion-based sampler.

\section{Dataset and preprocessing}
Figure~\ref{fig:pipeline} summarizes the procedure. We compiled 91 gait clips from 41 YouTube source videos found through searches for normal, MS and PD gait. Labels were assigned from video titles without independent clinical verification. Walking segments were selected manually: forward and return walks, or segments with different walking directions, could form separate clips. Footage with multiple people was cropped where needed to emphasize one person. Source-video identifiers retained in filenames group clips from the same recording.

The pretrained MediaPipe pose landmarker from the BlazePose family~\cite{blazepose} extracts 33 landmarks per sampled frame. Image coordinates are converted to pixels; horizontal and vertical coordinates and detector visibility are retained, while depth is discarded. Frame selection uses integer stride $\max(1,\operatorname{round}(f_{\rm src}/15))$, where $f_{\rm src}$ is the original frame rate. The target cadence is approximately 15 Hz, but the actual cadence depends on the original frame rate and integer sampling stride.

A clip is retained if at least 30\% of its sampled frames contain finite horizontal coordinates for every landmark and at least eight frames remain after cleaning. Empty ends are trimmed. Missing values are interpolated independently for each landmark and channel, boundary gaps receive the nearest observed value, and channels without observations are filled with zero. No maximum interpolation gap is imposed. Three clips are excluded, leaving 88 usable clips and all 41 sources (Table~\ref{tab:data}).

\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{wacv-v1-revision-assets/pipeline.pdf}
\caption{From gait clips to held-out label prediction. All clips from one source recording remain in the same training, validation or test partition before windows are constructed. Each encoder is fitted on 24 source videos. The frozen-feature classifier uses collection labels, while the encoder's feature-learning objective does not. Validation selects between the two executed training stages. Baseline methods use the same source partitions.}
\label{fig:pipeline}
\end{figure*}

\begin{table}[t]
\centering\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lrrrrr}
\toprule
Label & Raw & Usable & Sources & Padded & Windows\\
\midrule
Normal & 26 & 24 & 16 & 5 & 59\\
MS & 30 & 29 & 13 & 0 & 171\\
PD & 35 & 35 & 12 & 10 & 128\\
\midrule
Total & 91 & 88 & 41 & 15 & 358\\
\bottomrule
\end{tabular}
\caption{Dataset composition by collection label. Raw, usable and padded counts refer to clips. Sources are recordings, not verified participants. Windows contain 32 sampled frames and may overlap.}
\label{tab:data}
\end{table}

Let $h_t$ and $s_t$ denote the hip and shoulder midpoints in frame $t$. For S-JEPA and mean pose, landmark position $x_{tj}$ is normalized as
\begin{equation}
\widetilde{x}_{tj}=\frac{x_{tj}-h_t}{\max(\|s_t-h_t\|_2,10^{-3})}.
\label{eq:normalization}
\end{equation}
Visibility is unchanged. The RF branch calculates joint angles from cleaned pixel coordinates before this spatial normalization.

After source partitioning, each cleaned clip is divided into 32-frame windows with a stride of 16 frames; incomplete trailing windows are omitted. Window duration therefore varies across recordings. Clips shorter than 32 frames repeat their final frame to form one window, giving 15 padded clips in total. Padding does not add observed movement.

\section{Methodology}
\subsection{S-JEPA architecture}
An encoder converts the skeleton sequence into feature vectors. Four consecutive frames for one landmark form a token containing 12 values: horizontal position, vertical position and visibility in each frame. The 32-frame window therefore gives eight time blocks and $8\times33=264$ landmark--time tokens. A linear projection maps each token to 96 features, and learned landmark and time-position vectors identify its location in the sequence.

The context encoder combines information across visible tokens using three transformer layers with four attention heads. A two-layer transformer predictor, with separate positional vectors, estimates features at hidden positions. A target encoder with the same structure as the context encoder supplies the features to predict. It starts as a copy of the context encoder and processes the complete, untransformed training window. It receives no loss gradients; a moving average updates its weights from the context encoder. The skeleton model starts from fresh weights in each fold, while the upstream MediaPipe pose estimator is pretrained.

""" + fragment("r4").replace("landmarks---both shoulders", r"landmarks (Figure~\ref{fig:landmarks})---both shoulders") + r"""
\begin{figure}[t]
\centering
\includegraphics[width=0.93\columnwidth]{wacv-v1-revision-assets/landmarks.pdf}
\caption{BlazePose landmarks defining the anatomical sampling preference. Both shoulders, hips, knees, ankles, heels and foot indices belong to the selected set. Regions containing these landmarks receive increased selection weight, with an overlap adjustment; this does not prescribe a fixed hiding probability for individual landmarks. All 33 landmarks remain available. Schematic front view, with connections following BlazePose~\cite{blazepose}.}
\label{fig:landmarks}
\end{figure}

""" + fragment("r5").replace(
    "The target encoder is updated gradually from the context encoder's weights. ",
    ""
).replace(
    "Reflection exchanges left and right landmark slots; its effect on useful gait asymmetry was not separately evaluated.",
    "Reflection exchanges left and right landmark slots and visibility values, while the target window and target-mask positions remain unchanged. Its effect on side correspondence and useful gait asymmetry was not separately evaluated."
) + r"""
The supplementary material gives the loss equation, optimization settings and exact masking procedure.

""" + fragment("r6") + r"""
\section{Experimental setup}
\subsection{Comparison methods}
""" + fragment("r7") + r"""
Mean pose averages the normalized horizontal and vertical coordinates of all 33 landmarks across the full clip, giving 66 features without retaining frame order. The visibility baseline uses the mean and standard deviation of each landmark's visibility, also giving 66 features. Both use the same scaler and logistic-regression settings as the S-JEPA readout, fitted separately on the training clips. A majority-class baseline predicts the most frequent training-clip label. Mean pose uses coordinates alone, whereas S-JEPA also receives visibility.

""" + fragment("r8") + r"""
An additional 8--9 sources per fold are used for validation, so fitting and stage selection together use 32--33 development sources. All methods share the same source partitions. The experiment uses training seed 42; it does not average multiple initializations. Keeping model selection separate from test evaluation follows the distinction emphasized by Varma and Simon~\cite{varma2006}.

\begin{table}[t]
\centering\small
\setlength{\tabcolsep}{5pt}
\begin{tabular}{lrrr}
\toprule
Fold & Training & Validation & Test\\
\midrule
0 & 24 / 51 & 8 / 19 & 9 / 18\\
1 & 24 / 58 & 9 / 17 & 8 / 13\\
2 & 24 / 54 & 9 / 17 & 8 / 17\\
3 & 24 / 53 & 9 / 21 & 8 / 14\\
4 & 24 / 45 & 9 / 17 & 8 / 26\\
\bottomrule
\end{tabular}
\caption{Sources / clips in each partition. Folds are numbered 0--4 throughout the paper and supplement.}
\label{tab:source_partitions}
\end{table}

""" + fragment("r9") + fragment("r10") + r"""
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{wacv-v1-revision-assets/results.pdf}
\caption{Held-out classification with equal total weight per source video. (a) Each fold uses the same test sources for all three methods; higher macro-F1 indicates better classification. The horizontal offsets separate the markers and do not represent additional folds. (b--d) Confusion matrices show the percentage assigned to each predicted label within each collection label. Every panel uses the same 0--100\% color scale; rows may not sum to 100 after rounding. RF uses the corrected bilateral ankle-range features.}
\label{fig:results_revised}
\end{figure*}

""" + fragment("r11") + fragment("r12") + END

supp = PREAMBLE + r"""
\title{Supplementary Material\\Learning Skeleton Representations\\from a Small Collection of Neurological Gait Videos}
\author{Anonymous submission}
\begin{document}
\maketitle
\setcounter{section}{0}
\renewcommand{\thesection}{S\arabic{section}}
\renewcommand{\thetable}{S\arabic{table}}
\renewcommand{\thefigure}{S\arabic{figure}}
\renewcommand{\theequation}{S\arabic{equation}}

\section{Purpose and evaluation scope}
This supplement gives the implementation details and uncertainty analysis for the main paper. It describes the same 88 clips, 41 source recordings and fixed five-fold evaluation. The primary results come from the complete execution recorded on 9 October 2026; the earlier RF comparison is from 21 September 2026. Both use the same collection, source partitions and training seed. No new encoder or classifier was trained during preparation of this revision.

The main paper reports an observed advantage over the corrected handcrafted RF. The resampling analysis below describes variation in that comparison while keeping the fitted models and their predictions fixed. It does not supply a new test collection or an independent replication.

\section{Pose processing and source partitions}
The pose front end uses the pretrained full MediaPipe pose landmarker from the BlazePose family~\cite{blazepose}, in per-frame image mode with one pose and detection and presence thresholds of 0.5. It operates on the supplied clips after any manual cropping. Only image-plane coordinates and visibility are retained. The main paper specifies the frame-selection, cleaning and spatial-normalization rules.

Frames missing all horizontal landmark coordinates are trimmed from the ends. Missing channels are interpolated independently without a maximum interior gap. Complete 32-frame windows begin at frames 0, 16, 32 and so on; a trailing partial window is omitted. A clip shorter than 32 frames repeats its last frame, without a separate padding-validity mask. Retained clips have at least 14 cleaned frames in this collection, although the implemented acceptance threshold is eight.

The outer source splitter uses seed 42. Within fold $k$, the validation splitter uses seed $43+k$, and the first of its four splits defines the validation set. This is one validation holdout per outer fold, not four inner evaluations. The saved assignments determine the reported result. Training uses 207, 226, 238, 224 and 190 windows in folds 0--4, respectively, from 24 sources in each fold.

Encoder training samples a source uniformly with replacement and then a window uniformly within that source. This gives equal source probabilities, although realized draws need not be identical. The downstream classifier instead treats clips as observations and weights classes inversely to their training frequency. Class balancing and source balancing therefore operate at different stages.

\section{Anatomical mask construction}
Table~\ref{tab:regions} gives the seven regions by zero-based BlazePose landmark index. The preferred landmark set is $\{11,12,23,24,25,26,27,28,29,30,31,32\}$. For joint $j$, let $m_j$ be the number of regions containing it. Set $b_r=1.5$ if region $r$ contains a preferred landmark and $b_r=1$ otherwise. Its unnormalized sampling weight is
\begin{equation}
a_r=\frac{b_r}{|r|^{-1}\sum_{j\in r}m_j}.
\end{equation}
Normalizing these weights gives the region-selection probabilities. The overlap adjustment and region membership mean that the 1.5 multiplier is not an individual joint's hiding probability.

\begin{table}[t]
\centering\small
\begin{tabular}{ll}
\toprule
Region & Landmark indices\\
\midrule
Head & 0, 1, 2, 3, 4, 5, 6, 7, 8\\
Mouth & 9, 10\\
Left arm & 11, 13, 15, 17, 19, 21\\
Right arm & 12, 14, 16, 18, 20, 22\\
Trunk & 11, 12, 23, 24\\
Left leg & 23, 25, 27, 29, 31\\
Right leg & 24, 26, 28, 30, 32\\
\bottomrule
\end{tabular}
\caption{Regions used in mask sampling. Arms and trunk share shoulders; legs and trunk share hips.}
\label{tab:regions}
\end{table}

For each example, a region is sampled, followed by a span length chosen uniformly from one to six time blocks and a start position where that span fits. Its tokens are added to the target set until the count reaches $\operatorname{round}(0.6\times264)=158$, or 36 region draws have occurred. Overlap and the last draw can change the realized hidden fraction.

If every token associated with the preferred landmark set is hidden, one randomly selected token in that set is made visible. The implementation also ensures nonempty context and target sets. This rule preserves some preferred-landmark context somewhere in the window; it does not guarantee a visible leg or preferred landmark in every time block.

\section{Loss, optimization and readout}
Let $u$ be the predictor vector, $z$ the target-encoder vector at the corresponding hidden position, and $c$ the running center. Following the centering and sharpening of DINO~\cite{dino} used by S-JEPA~\cite{sjepa},
\begin{equation}
\begin{aligned}
p &= \operatorname{softmax}(u/0.10),\\
q &= \operatorname{softmax}((z-c)/0.06).
\end{aligned}
\end{equation}
Cross-entropy $-\sum_d q_d\log p_d$ is averaged over each window's hidden tokens and then across windows. These distributions are over 96 feature dimensions, not the collection labels. After the loss is formed, the center updates once per batch as $c\leftarrow0.9c+0.1\bar z$, where $\bar z$ averages all masked target vectors in the batch. No loss gradient enters the target encoder.

Both encoders have width 96, three layers, four attention heads, feed-forward expansion two and no dropout; the predictor has two layers with separate learned landmark and time positions. AdamW uses peak learning rate $10^{-3}$, weight decay 0.05, coefficients $(0.9,0.95)$ and gradient-norm clipping at one. The first 10\% of each stage warms up the learning rate, followed by cosine decay. Target-encoder momentum rises from 0.996 toward 1.0.

Each fold starts from fresh skeleton-model weights with seed 42. After 800 updates, the 400-update continuation retains learned model weights but restarts the optimizer, center, random-number generators and schedules. Both stages receive a separately fitted scaler and classifier on training clips. Source-weighted validation F1 selects the stage, retaining the initial stage on a tie. The initial stage is selected in folds 0--2, and continuation in folds 3--4.

Reflection exchanges left/right slots in the context input while the untransformed target and mask indices keep their original slots. The physical joint available at a context position can therefore change sides. The reported experiment retains this implementation; its effect has not been isolated in a controlled training comparison.

For classification, the full window enters the frozen target encoder. A readout mask sampled once with NumPy generator seed zero selects 161 of 264 output positions for averaging. Window vectors are averaged within a clip. Logistic regression uses $C=1$, balanced class weights, random state 42 and a 2,000-iteration limit. Every scaler and classifier is fitted only on its training clips.

\section{Handcrafted and simple comparisons}
The RF computes hip, knee and ankle angles from the shoulder--hip--knee, hip--knee--ankle and knee--ankle--foot-index triplets on each side. An angle is omitted when the geometric mean of its three landmark visibility scores is below 0.3. Six mean angles, six maximum-minus-minimum ranges and three absolute left/right mean differences form the 15 varying fields. Missing angle summaries are filled with zero. The 82-field feature container includes other fields that are unpopulated or constant here; zero-variance columns are removed using the training partition.

The corrected extractor records left- and right-ankle ranges separately. The earlier extractor copied the right-ankle range into both fields. The two saved feature matrices differ only in that field, and the corresponding RF predictions change for 12 of 88 clips. The main results retain the earlier score in a separate historical row.

The RF uses 100 trees, maximum depth five, square-root feature subsampling, balanced class weights and seed 42, after a training-fitted scaler. Mean pose uses 66 full-clip coordinate averages. Visibility uses 33 means and 33 standard deviations. These two comparisons use the same training-only scaling and logistic-regression settings as S-JEPA.

\section{Training behavior}
The first-fold training history reports means over the first and last 50 updates of the initial 800-update stage (Table~\ref{tab:training}). These are saved summaries, not a reconstructed per-update curve. The retained checkpoint's serialized payload agrees with the first-fold initial checkpoint of the primary evaluation.

\begin{table}[t]
\centering\small
\begin{tabular}{lrr}
\toprule
Diagnostic & First 50 & Last 50\\
\midrule
Feature-prediction loss & 3.858 & 1.022\\
Mean feature standard deviation & 0.154 & 0.526\\
Effective rank & 8.13 & 10.94\\
\bottomrule
\end{tabular}
\caption{Training-batch summaries for fold 0. The feature spread and rank use pooled target-encoder outputs.}
\label{tab:training}
\end{table}

For these diagnostics, each window is represented by the average of all target-encoder output tokens, rather than only the positions used by the classifier. Feature standard deviations are calculated across the 32 sampled windows and averaged across the 96 dimensions. For effective rank, the batch is centered, singular values above $10^{-9}$ are normalized as $r_i=s_i/\sum_j s_j$, and the score is $\exp(-\sum_i r_i\log r_i)$. The centered batch has maximum rank 31.

Initial-stage endpoint ranks range from 10.896 to 12.135; continuation-stage endpoints range from 9.558 to 11.638. These diagnostics show variation in learned features and do not measure clinical separation. Endpoint losses describe individual training batches; the averages in Table~\ref{tab:training} summarize a longer portion of training.

The mask-frequency check used 5,000 masks with seed 42. The mean hidden-token fraction was 0.632. The left shoulder was hidden in 78.3\% of landmark--time positions, but in at least one time block in 98.6\% of windows. These are different denominators. The sampling check does not determine whether the anatomical preference improves classification.

\section{Score calculation}
For clip $i$ from source $s$, source weighting uses $w_i=1/n_s$, where $n_s$ is that source's usable clip count; clip weighting uses $w_i=1$. With collection label $y_i$ and predicted label $\widehat y_i$, the confusion count is
\begin{equation}
C_{ab}=\sum_i w_i\,\mathbf{1}[y_i=a,\widehat y_i=b].
\end{equation}
Class F1 is $2C_{kk}/(\sum_b C_{kb}+\sum_a C_{ak})$, and macro-F1 is the equal average across Normal, MS and PD. Undefined scores are zero. The pooled result combines counts from all test folds before calculating F1, so it need not equal the mean of the five fold scores.

"""
uncertainty = fragment("r14").split(r"\section{Implementation and training diagnostics}")[0]
uncertainty = uncertainty.replace(
    "The 2.5th and 97.5th percentiles of the resulting scores define the intervals reported here.",
    "All 10,000 draws for Normal are generated first, followed by all draws for MS and then PD. The 2.5th and 97.5th percentiles, using linear interpolation, define the intervals reported here."
)
supp += uncertainty + END

# Apply assembly-level wording and placement refinements without changing the
# archived copy-and-paste guide from the preceding review.
main = main.replace(
    "Overlap and the final region can increase this proportion;",
    "Adding a whole final region can take the hidden fraction above 60\\%;"
)
main = main.replace(
    "Together, these experiments provide a grounded case study of representation learning for markerless behavioral analysis and identify the comparisons needed before interpreting the features as measures of neurological gait impairment.",
    "These comparisons provide a case study of representation learning for markerless behavioral analysis and identify what remains to be tested before clinical interpretation."
)
results_figure = next(block for block in re.findall(
    r"\\begin\{figure\*\}.*?\\end\{figure\*\}", main, re.S
) if "assets/results.pdf" in block)
main = main.replace(results_figure, "")
main = main.replace(r"\section{Results}", results_figure + "\n\n" + r"\section{Results}")

ROOT.mkdir(exist_ok=True)
(ROOT / "wacv-v1-revision.tex").write_text(main)
(ROOT / "wacv-v1-revision-supplementary.tex").write_text(supp)
for name in ("wacv.sty", "ieeenat_fullname.bst"):
    shutil.copyfile(REVIEW / "working/layout" / name, ROOT / name)
bib = (ROOT.parent / "wacv2007/references.bib").read_text()
extra = r"""

@inproceedings{maskfeat,
  author = {Wei, Chen and Fan, Haoqi and Xie, Saining and Wu, Chao-Yuan and Yuille, Alan and Feichtenhofer, Christoph},
  title = {Masked Feature Prediction for Self-Supervised Visual Pre-Training},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages = {14668--14678},
  year = {2022},
  url = {https://openaccess.thecvf.com/content/CVPR2022/html/Wei_Masked_Feature_Prediction_for_Self-Supervised_Visual_Pre-Training_CVPR_2022_paper.html}
}
"""
(ROOT / "wacv-v1-revision.bib").write_text(bib + extra)
print("Assembled main manuscript, supplementary manuscript, bibliography and official style files.")
