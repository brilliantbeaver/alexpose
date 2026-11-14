# Evidence boundaries for the v9 framing

This pre-draft review checks how the retained experiment can support a clearer, more positive account of learned features and the Random Forest comparison. It reads v8 and the existing numerical audit; it adds no experiment and changes no manuscript. The top-level notebooks and their associated historical execution remain primary.

## Positive claims supported by the experiment

The encoder was trained through prediction of hidden features, without condition labels in that training loss. Its fixed 96-value clip representation then supported collection-label prediction by a fitted linear classifier on held-out source recordings. “Learned features” is therefore an accurate description of the procedure. A direct, defensible statement is:

> Features obtained through label-free encoder training supported prediction of the collection labels on held-out source recordings.

The classifier and training-stage selection do use labels. Describing the entire classification system as unsupervised, or saying that the final encoder was selected without labels, would be incorrect.

The observed primary score is higher than the tested joint-angle RF, visibility-only and majority-label procedures when each source recording receives equal total weight. It is also slightly higher than mean pose. A concise statement suitable for the abstract is:

> With each recording contributing equally, the fixed S-JEPA features and linear classifier achieved macro-F1 0.457, compared with 0.411 for the notebook's joint-angle Random Forest and 0.452 for mean pose. The learned features also scored above the tested visibility-only and majority-label baselines. These results support the feasibility of extracting label-predictive features from this small video collection, while mean pose provides a strong reference for future tests of temporal learning.

The historical RF feature defect must remain visible in the paper and marked in its results table. “Highest observed source-weighted score among the evaluated procedures” is also literally supported, provided the same passage makes the close mean-pose comparison clear. The study does not establish a repeatable ranking across new training runs or collections.

## What “meaningful features” can mean here

An operational definition can be limited to **information useful for predicting these supplied collection labels with a simple classifier**. That is a descriptive property of the evaluated trained representation and classifier. It does not identify the underlying information as movement dynamics, physiology, disease severity, or a clinically meaningful measurement. Posture, viewpoint and collection-specific recording cues remain possible contributors.

The stronger claim that self-supervised training **created or improved** this useful information needs an otherwise identical untrained-encoder comparison. The available experiment does not supply it. “Real representational learning” would be ambiguous between the established fact that the feature extractor was trained and the unestablished causal claim that training improved label information. Prefer the explicit statement above. Lower training-batch prediction error does not resolve this gap, and the reflected-mask correspondence defect still needs its existing disclosure.

## Checked RF comparisons

These values are read from `results-audit.json`, using `notebook_primary_metrics` for the primary experiment and `corrected_verification_metrics` for the later verification. “Difference” means S-JEPA minus comparator.

| Procedure | Source-weighted macro-F1 | Clip-weighted macro-F1 | S-JEPA source difference | S-JEPA clip difference |
|---|---:|---:|---:|---:|
| S-JEPA | 0.456624159 | 0.397033957 | — | — |
| Historical joint-angle RF | 0.411262910 | 0.397315740 | +0.045361249 | −0.000281783 |
| Corrected joint-angle RF, separate verification | 0.325184104 | 0.320940497 | +0.131440055 | +0.076093460 |
| Mean pose | 0.451892876 | 0.409840252 | +0.004731283 | −0.012806295 |
| Visibility summary | 0.317564628 | 0.318057047 | +0.139059531 | +0.078976910 |
| Majority-label procedure | 0.259398496 | 0.202125202 | +0.197225663 | +0.194908755 |

The historical RF and S-JEPA clip scores both round to 0.397; the historical RF is slightly higher before rounding. S-JEPA therefore does not lead that baseline under both weighting rules. It leads historical RF in two of five source-weighted test folds, corrected RF in three, and mean pose in one. Its source-weighted score exceeds the majority-label procedure in all five folds and visibility in four; these are descriptive comparisons, not tests against a general chance distribution.

Correcting the duplicated ankle feature reproduces all 88 later RF predictions, while reinstating the duplicate reproduces all 88 historical RF predictions. Twelve predictions change. The correction's lower observed score does not make the incorrect feature definition preferable. The two saved executions share data, partitions and seed; identical S-JEPA hard predictions do not constitute independent replication. Comparisons are to the configured RF procedure, not every possible classical or tuned gait classifier.

## Minimum uncertainty and scope statements

The abstract need not contain interval endpoints or repeat a claim about statistical significance. Retain the actual numerical differences and their intervals once in the Results, with the procedure explained in Evaluation. The relevant paired source-resampling intervals, from 10,000 draws with seed 20261004, remain:

| Comparator | Difference | Conditional 95% percentile interval |
|---|---:|---|
| Historical RF | +0.045361249 | [−0.133446835, 0.220113928] |
| Corrected RF, separate verification | +0.131440055 | [−0.046114011, 0.303792121] |
| Mean pose | +0.004731283 | [−0.160496452, 0.173998638] |

A plain-language interpretation can say that the observed advantage varies with which recordings are represented, and that the resampling keeps the existing fitted models and predictions fixed. It does not describe the uncertainty from retraining or demonstrate equivalence. Avoid calling these results “above chance” without an appropriate chance analysis, or saying the features are statistically validated merely because their point scores exceed the majority-label baseline.

The body should retain the mean-pose comparison and fold variation, the lack of an untrained encoder and frame-order intervention, the one-seed design, the historical RF defect, and the known reflection mismatch. Source separation remains distinct from verified participant separation. The supplied labels and data-use provenance are not clinically verified. These boundaries can be consolidated into a readable limitations section instead of being repeated after every positive observation.
