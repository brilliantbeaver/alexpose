from pathlib import Path
r=Path(__file__).resolve().parents[1]
s=(r/'wacv2007-v2.tex').read_text()
s=s.replace('\\usepackage[review,applications]{wacv}', '\\usepackage[T1]{fontenc}\n\\usepackage[review,applications]{wacv}')
s=s.replace('\\small\n\\begin{tabular}', '\\small\n\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}')
s=s.replace('The study identifies practical controls needed before attributing classification performance to temporal gait information.', 'A paired source bootstrap gives an S-JEPA-minus-mean-pose difference of 0.005 with a conditional 95\\% percentile interval of $[-0.160,0.174]$. The comparison motivates explicit order-invariant controls before attributing predictive performance to temporal gait information.')
s=s.replace('Our contribution is an empirical evaluation under source-separated testing.', 'The empirical contribution is that the learned representation\'s small pooled lead over mean pose under source weighting reverses under clip weighting and is concentrated in one fold.')
s=s.replace('\\section{Video collection and pose processing}', r'''\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/pipeline.pdf}
\caption{The notebook-defined evaluation. Pose extraction precedes body-relative normalization. Source partitions are fixed before window generation; encoder training and probe fitting use only training sources, and validation chooses the training stage. Test predictions are pooled under two explicitly defined weighting rules.}
\label{fig:pipeline}
\end{figure*}

\section{Video collection and pose processing}''')
s=s.replace('Paired source-resampling uncertainty must be computed from the matched predictions before interpreting small pooled differences.', r'''We add a retrospective paired source bootstrap to the notebook-associated outer-test predictions. In each of 10,000 draws (seed 20261004), we sample 16 Normal, 13 MS, and 12 PD sources with replacement within label strata. Each sampled occurrence contributes total weight one, distributed equally among that source's clips, and every method uses the same source draws. We recompute pooled macro-F1 and paired differences, reporting the 2.5th and 97.5th percentiles.

These are conditional source-resampling intervals with class-source counts and fitted predictions held fixed. They exclude retraining, seed, split, stage-selection, and development uncertainty. Sources are treated as independent clusters even though participant overlap is unresolved, and shared training data induce dependence across fitted outer-fold models. We use the intervals as descriptive sensitivity summaries, not a confirmatory population test or evidence of equivalence. This analysis was added for the manuscript; it is not a saved notebook output.''')
s=s.replace('Mean pose exceeds S-JEPA in four of five outer folds.', r'''The conditional interval for S-JEPA macro-F1 is $[0.329,0.579]$. Its paired difference from mean pose is $0.0047$ with interval $[-0.160,0.174]$; against the notebook RF it is $0.0454$ with interval $[-0.133,0.220]$. Both span zero. The data and this conditional procedure leave the directions uncertain without showing that the methods are equivalent.

\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/folds-and-errors.pdf}
\caption{Notebook-associated outer-test results. (a) Source-weighted macro-F1 by outer fold; connecting lines identify the methods across fixed folds and imply no time trend. (b,c) Row-normalized source-weighted confusion percentages. Rows are collection labels, columns predictions, and each source contributes total weight one before row normalization. These are not percentages of clinically verified patients.}
\label{fig:results}
\end{figure*}

Mean pose exceeds S-JEPA in four of five outer folds (Figure~\ref{fig:results}a).''')
s=s.replace('Mean pose is a demanding control here because it retains body configuration while discarding frame order.', 'Mean pose is a demanding control here because it retains body configuration while discarding frame order. S-JEPA fold scores range from 0.217 to 0.780. Its raw correct counts are 10/24 Normal, 8/29 MS, and 18/35 PD clips; source-weighted recalls are 0.396, 0.429, and 0.569 (Figure~\\ref{fig:results}b). The average score therefore coexists with substantial errors in all three labels.')
s=s.replace('A subsequent saved evaluation changes the RF predictions after an extractor correction; it is separate from the notebook-reported comparison. Repeated records with the same data, split, and seed do not establish independent-seed replication.', r'''The accompanying reproducibility record identifies the notebook-associated run, its registry and prediction hashes, and the script that recomputes the tables and intervals. The original PDF is not an evidence source when its numbers disagree with those records.

\subsection{Separate verification of the RF correction}
A later saved evaluation uses the true left-ankle range in place of the duplicated right-ankle range. As a manuscript-stage verification, we refit the small deterministic RF baseline on the existing partitions. Current extraction reproduces all 88 later predictions, and replacing only that feature with the historical duplicate reproduces all 88 notebook-associated RF predictions. Twelve predictions change between versions. This identifies the feature correction as sufficient to reproduce the RF difference under these settings, while preserving both records.

The corrected RF yields source-weighted macro-F1 0.325 and clip-weighted macro-F1 0.321, compared with 0.411 and 0.397 in the notebook-reported implementation. Its paired difference from S-JEPA is $0.1314$ with conditional interval $[-0.046,0.304]$. A correction can lower performance; the direction alone says nothing about which implementation faithfully computes the intended feature. This verification does not establish superiority over a well-tuned classical gait system. Both saved executions use the same data, split, and training seed, and their S-JEPA hard predictions coincide despite different embedding values. They are not independent-seed replications.''')
(r/'wacv2007-v3.tex').write_text(s)
