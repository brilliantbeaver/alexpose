# Evidence review for the standalone draft02

## Reviewed evidence

The review read the current draft01 manuscript, its complete forecast and anomaly tables, the restart protocol, model contract, interpretation notes, implementation, primary-source ledger, and source-level run outputs. The top-level teaching notebooks intentionally have no saved outputs. Their executed counterparts in `artifacts/executed-restart/` contain the results. In particular, notebook 14 displays all forecast arms, paired contrasts, abnormality rankings, and condition-specific errors. Notebook 15 connects those results to claims and provenance. The manuscript uses the completed `restart-dynamics-v1` results rather than mixing them with historical JEPA outputs.

The builder validates the frozen run manifest and prepared-array hashes. It independently recomputes original forecast means, paired mean differences, all five anomaly AUROCs and average-precision estimates, and source/fold membership. This checks reported evidence without refitting a model.

## Positive interpretation and its boundary

Normal-source forecasting improves over persistence. The current-posture model is especially informative because it predicts changing future outputs from the latest posture and reduces error by 40.9%. Its success supports a compact observation transition, while limiting a claim that long bilateral history is necessary.

The prespecified excess-error score also has a favorable association with abnormal annotations. Its interpretation is clearer when the score is written as predicted displacement squared minus twice predicted/observed displacement alignment. This is an exact squared-error identity only when both errors use the same observed targets, latest-coordinate anchors, and averaging weights. That matched support holds in the evaluated run. The identity is an explanation, not a newly fitted estimator or evidence of disease mechanism.

Negative persistence error provides a necessary post hoc diagnostic. Its AUROC of 0.806 is slightly larger than excess error's 0.793. It was motivated after seeing the observed score directions, and cannot be retroactively called prespecified. It is kept outside the frozen result identity and does not replace any original score or choose a threshold. The manuscript reports it alongside the positive findings because omitting it would make the learned anomaly interpretation misleading.

## Added analysis identity

`draft03-frozen-score-audit-v1` operates only on immutable source-level scores and fold membership. The two scores are compared on identical source resamples. Pooled uncertainty uses 5,000 class-stratified source resamples with seed 2703. Within-fold uncertainty uses 5,000 source resamples within each fold/class cell with seed 2704. Each fold retains its normal and abnormal counts, and within-fold AUROCs are weighted by their number of normal-abnormal source pairs. This excludes cross-fold pairs but does not make the overlapping fitted folds independent.

All added intervals are conditional descriptive intervals. They omit retraining and uncertain participant overlap. Condition slices reuse the same 29 normal sources, are descriptive, and are not multiple independent confirmations. No p-values, score selection, thresholds, new model fitting, or clinical accuracy claims are produced.

## Primary literature checked

The GAVD paper describes the upstream annotation resource. Its reported population does not establish verified participant identity in this local derivative. [GAVD](https://arxiv.org/abs/2407.04190)

Laboratory segment coordination motivates relational modeling, and measured bilateral timing is clinically relevant in Parkinson's disease. These studies do not equate projected video prediction errors with neural measurements. [Borghese and colleagues](https://doi.org/10.1113/jphysiol.1996.sp021539), [Plotnik and colleagues](https://pubmed.ncbi.nlm.nih.gov/17503027/)

The stroke and cerebral-palsy studies concern measured muscle activation. Their claims are inherited from the previously verified primary-source ledger; the direct PubMed stroke URL returned an incomplete shell during the current recheck. The draft uses only the broad relationship between altered muscle activation and impaired walking. [Stroke](https://doi.org/10.1152/jn.00825.2009), [cerebral palsy](https://doi.org/10.1111/dmcn.12826)

Skeleton forecasting, normative reconstruction, and latent skeleton prediction are established precedents. The review checked the primary MoCoDAD and current GenGait abstracts and the S-JEPA conference PDF. Consequently, the manuscript claims an evaluation contribution, not the first skeleton predictor or normative gait model. [MoCoDAD](https://arxiv.org/abs/2307.07205), [GenGait](https://arxiv.org/abs/2604.01997), [S-JEPA](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04755.pdf)

The need for strong simple motion references is also directly supported by the motion-prediction baseline study. [Martinez and colleagues](https://arxiv.org/abs/1705.02445)

## Claims deliberately excluded

The evidence does not establish algorithmic superiority over a movement-only anomaly score, clinically valid sensitivity or specificity, patient-disjoint generalization, disease severity estimation, neural mechanisms, treatment effects, or a complete medical world model. These are boundaries of the evidence, not omissions to be filled by rhetoric.

FSGait is direct gait-specific prior art: it trains separate reconstruction and prediction modules on normal gait and analyzes postural and temporal abnormalities. The standalone manuscript therefore cannot claim the first normal-only gait predictor or the first separation of postural and temporal errors. [Official ACCV paper](https://openaccess.thecvf.com/content/ACCV2024/html/Duan_FSGait_Fine_Grained_Self-Supervised_Gait_Abnormality_Detection_ACCV_2024_paper.html)
