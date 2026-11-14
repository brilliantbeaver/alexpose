from pathlib import Path
import re
r=Path(__file__).resolve().parents[1];s=(r/'wacv2007-v4.tex').read_text()
s=s.replace('Evaluating Predictive Skeleton Representations for Neurological Gait Videos','Evaluating Learned Skeleton Features in Neurological Gait Videos')
a=r'''\begin{abstract}
Video-based gait analysis could support research on neurological movement, but a classifier may use average posture or recording conditions as well as movement over time. We test whether learning from sequences of automatically tracked body landmarks improves on a simple average-pose summary. The collection contains 88 usable clips from 41 source recordings, labeled Normal, multiple sclerosis, or Parkinson's disease; these labels were not clinically verified. A skeleton joint-embedding predictive architecture (S-JEPA) learns numerical descriptions of the clips by predicting hidden features. We keep source recordings separate during training, model selection, and testing, and compare the learned descriptions with simpler summaries. Macro-F1, which averages F1 across the three labels, is 0.457 for S-JEPA and 0.452 for mean pose when every source recording receives equal total weight. Resampling whole sources gives a difference of 0.005 with a conditional 95\% interval of $[-0.160,0.174]$. Mean pose performs better when clips receive equal weight and in four of five test groups. These results leave the advantage over mean pose unresolved. Unverified participant identities and a mismatch between reflection and training masks further limit interpretation. The study illustrates the value of simple pose summaries and explicit evaluation units when assessing learned gait features.
\end{abstract}'''
s=re.sub(r'\\begin\{abstract\}.*?\\end\{abstract\}',lambda _:a,s,flags=re.S)
s=s.replace('An encoder', 'An encoder')
s=s.replace('We investigate whether a compact predictive skeleton representation provides a consistent classification advantage over simple pose summaries', 'We investigate whether a model that learns numerical summaries of skeleton sequences provides a consistent classification advantage over simple pose averages')
s=s.replace('The encoder follows the joint-embedding predictive approach', 'The feature-extraction network, or encoder, follows the joint-embedding predictive approach')
s=s.replace('A linear classifier then predicts the three collection labels from frozen features.', 'After this training, the encoder is held fixed. A separate linear classifier combines its features to predict the three collection labels.')
s=s.replace("The empirical contribution is that the learned representation's small pooled lead over mean pose under source weighting reverses under clip weighting and is concentrated in one fold. The evaluation pairs a temporal representation with an order-invariant mean-pose control and a visibility-only control on identical partitions. The observed ordering changes with the evaluation weights, and fold variation is substantial.", "The empirical contribution is a bounded comparison: the learned model's small overall lead over mean pose reverses when clips, rather than source recordings, receive equal weight. The learned model leads in only one of the five test groups. We compare both methods and a visibility-only summary on the same source partitions, so readers can assess the extent and consistency of the differences.")
s=s.replace('We evaluate a compact implementation with a region-sampling rule motivated by gait anatomy.', 'The original S-JEPA samples target joints with probabilities related to motion. Our compact adaptation uses video-derived 2D coordinates and visibility, with masks sampled from body regions. This sampling choice is motivated by gait anatomy.')
s=s.replace('Thus, 15 frames per second is a nominal target rather than an exact common sampling rate.', 'Thus, 15 frames per second is a nominal target rather than an exact common sampling rate. Windows contain the same number of sampled frames but can cover different lengths of time.')
s=s.replace('Padding and interpolation can change apparent dynamics; their distribution across labels is therefore reported explicitly.', 'Padding and interpolation can change apparent dynamics. Table~\\ref{tab:data} reports the padding counts by label; the retained caches do not provide per-label counts or gap lengths for interpolation.')
s=s.replace('\\subsection{Encoder and training objective}', '\\subsection{Learning to predict hidden features}')
start=s.index('The tokenizer maps four frames')
end=s.index('The student receives a geometrically transformed window.',start)
s=s[:start]+r'''The encoder converts each group of four frames for one landmark into a 96-number feature vector. Learned position codes specify the landmark and time block. Transformer layers combine information across visible positions, and a second network predicts features at the hidden positions. A reference network, called the target encoder, supplies those training targets. It starts as a copy of the context encoder and is updated gradually toward its weights through an exponential moving average (EMA). This makes the targets change more slowly than the network being optimized. Table~\ref{tab:settings} records the architecture and optimization settings.

\begin{table}[t]
\centering
\caption{Settings for the evaluated model. Both training stages use the same architecture and loss; the continuation restarts optimization state.}
\label{tab:settings}
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{p{.43\columnwidth}p{.49\columnwidth}}
\toprule
Setting & Value\\
\midrule
Input window & $32\times33\times3$\\
Frames per landmark token & 4 (12 input values)\\
Encoder / predictor depth & 3 / 2 transformer layers\\
Feature width / attention heads & 96 / 4\\
Feed-forward expansion / dropout & 2 / 0\\
Batch size / training updates & 32 / 800, then 400\\
AdamW peak rate / weight decay & $10^{-3}$ / 0.05\\
AdamW coefficients / norm clip & $(0.9,0.95)$ / 1\\
Learning-rate schedule & 10\% warmup, cosine decay\\
EMA momentum & Cosine, 0.996 toward 1\\
Center momentum & 0.9\\
Prediction / target temperature & 0.10 / 0.06\\
Initialization and training seed & 42\\
\bottomrule
\end{tabular}
\end{table}

'''+s[end:]
s=s.replace('The student receives a geometrically transformed window.', 'The context network receives a geometrically transformed window.')
s=s.replace('Within the context encoder, target positions are excluded as attention keys and values; their output vectors are replaced by a shared learned mask token before the predictor receives them. The per-example implementation retains computations at hidden query positions rather than physically dropping those positions.', 'To withhold the hidden observations, visible positions cannot draw information from target positions in the context encoder. The implementation enforces this through an attention mask and replaces hidden-position outputs with a shared learned placeholder before prediction. The predictor also receives joint and time position codes, so different hidden positions can receive different predictions.')
s=s.replace('Both stages use the same label-free objective across all three labels; neither a normal-only pretraining restriction nor a class-aware VICReg loss is active.', 'Both stages use all training sources and the same prediction task, without condition labels in the encoder loss.')
start=s.index('AdamW uses peak learning rate')
end=s.index('\\subsection{Frozen representation and classifier}',start)
s=s[:start]+r'''The settings in Table~\ref{tab:settings} define the evaluated configuration. They do not describe an independently optimized architecture. The technical implementation, including the feature-distribution loss and its weighting, is recorded in the companion methods audit.

'''+s[end:]
s=s.replace('\\subsection{Frozen representation and classifier}', '\\subsection{From learned features to a clip label}')
s=s.replace('At readout, the frozen EMA target encoder receives every token of the complete window.', 'For classification, we keep the target encoder\'s weights fixed and supply every token of the complete window.')
s=s.replace('A training-fitted standardizer and class-balanced logistic regression ($C=1$, maximum 2,000 iterations, seed 42) form the classifier.', 'Each feature is standardized using training clips only, then a logistic-regression classifier is fitted to their labels ($C=1$, maximum 2,000 iterations, seed 42). This is a linear classifier: it combines the feature values with learned weights. Class weighting gives greater fitting weight to labels with fewer training clips.')
s=s.replace('Label-stratified five-fold splitting operates on one row per source, then expands source assignments to clips.', 'We sort the source identifiers and divide one row per source into five groups using shuffled, label-stratified splitting (\\texttt{StratifiedKFold}). The procedure distributes the three labels across groups, then expands source assignments to clips. Each group serves once as the outer test set, or fold.')
s=s.replace('Within each outer development pool, the first split from a four-fold label-stratified source splitter forms the validation holdout.', 'Among the other sources, the first split from a four-fold version of the same procedure forms a separate validation group for choosing the training stage.')
s=s.replace('S-JEPA procedure additionally selects between two stages on validation data.', 'S-JEPA procedure additionally selects between two stages on validation data. These are comparisons of complete procedures: they do not isolate the effect of temporal information alone.')
s=s.replace('Our primary descriptive measure is pooled source-weighted macro-F1.', 'F1 balances precision (how often a predicted label is correct) and recall (how many examples with that label are found). We calculate F1 separately for Normal, MS, and PD, then average the three values to obtain macro-F1. Higher values are better, with one representing perfect predictions; the score is not the fraction of correctly classified clips. Our primary descriptive measure gives every source recording equal total weight and combines all test predictions before scoring. We call this pooled source-weighted macro-F1.')
s=s.replace('We add a retrospective paired source bootstrap to the notebook-associated outer-test predictions.', 'We assess sensitivity to the sampled sources by repeatedly drawing whole source videos, a procedure called a source bootstrap. It is a retrospective analysis of the notebook-associated test predictions.')
s=s.replace('These are conditional source-resampling intervals with class-source counts and fitted predictions held fixed.', 'The resulting intervals describe variation when sources are resampled while the fitted models, their predictions, and the number of sources per label remain fixed.')
s=s.replace('They exclude retraining, seed, split, stage-selection, and development uncertainty.', 'They do not capture changes caused by retraining, another random initialization, different partitions, or other choices made during development.')
s=s.replace('The top-level notebooks define the reported experiment. The capstone notebook contains a written result table and executable evaluation procedure, but its code cells have no saved outputs. We verify its written results against the corresponding retained evaluation artifacts.', 'The primary comparison follows the protocol and written results in the top-level notebooks. We verified these results against the matching saved predictions; the notebook\'s evaluation code has no saved cell outputs.')
s=s.replace(' The original PDF is not an evidence source when its numbers disagree with those records.', '')
s=s.replace('figures/folds-and-errors.pdf','figures/folds-and-errors-v5.pdf')
s=s.replace('(a) Source-weighted macro-F1 by outer fold; connecting lines identify the methods across fixed folds and imply no time trend.', '(a) Source-weighted macro-F1 in each test fold. A straight line joins the two methods evaluated on the same sources; circles denote S-JEPA and squares mean pose.')
s=s.replace('An untrained-encoder baseline and controlled changes to temporal order would address different parts of that question.', 'An otherwise identical encoder with untrained weights would test the contribution of training. Controlled changes to frame order would address whether the sequence itself provides useful information. Neither comparison was run here.')
(r/'wacv2007-v5.tex').write_text(s)
# Expose the identifier in the WACV bibliography style, which omits eprint.
bib=(r/'references.bib').read_text().replace('eprint = {2006.10204},','eprint = {2006.10204},\n  note = {arXiv:2006.10204},')
(r/'references.bib').write_text(bib)
