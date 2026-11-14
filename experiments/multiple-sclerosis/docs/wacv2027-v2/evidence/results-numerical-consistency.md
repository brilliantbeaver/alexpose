# Numerical consistency and bootstrap audit of the supplied draft

The supplied PDF's main S-JEPA numbers are inconsistent with both retained full-v1 prediction files. A search over all 176 one-clip label changes finds one counterfactual that reproduces every examined rounded S-JEPA quantity: change the prediction for `-oJM2wUUjws_P1_02` from the saved `ms` to `pd`. This is a diagnostic reconstruction only. Both saved runs record `ms`; the manuscript must use the retained predictions, and no artifact was changed.

The reconstruction also reproduces the draft's stated S-JEPA bootstrap interval and its RF-difference interval. The draft repeats the RF interval for mean pose, which fails independent calculation even for the reconstructed rows. The editing history is unavailable, so these checks locate the numerical discrepancy without claiming how it arose.

## Complete scalar table audit

| Model | Quantity | Draft | Notebook/Sep21 saved | Corrected/Oct09 saved | Matches notebook at 3 dp? |
|---|---|---:|---:|---:|---|
| sjepa | pooled_source_macro_f1 | 0.468 | 0.456624 | 0.456624 | NO |
| sjepa | pooled_clip_macro_f1 | 0.406 | 0.397034 | 0.397034 | NO |
| sjepa | fold_mean | 0.452 | 0.438068 | 0.438068 | NO |
| sjepa | fold_population_std | 0.189 | 0.195980 | 0.195980 | NO |
| mean_pose | pooled_source_macro_f1 | 0.452 | 0.451893 | 0.451893 | yes |
| mean_pose | pooled_clip_macro_f1 | 0.410 | 0.409840 | 0.409840 | yes |
| mean_pose | fold_mean | 0.427 | 0.427332 | 0.427332 | yes |
| mean_pose | fold_population_std | 0.132 | 0.132210 | 0.132210 | yes |
| rf | pooled_source_macro_f1 | 0.411 | 0.411263 | 0.325184 | yes |
| rf | pooled_clip_macro_f1 | 0.397 | 0.397316 | 0.320940 | yes |
| rf | fold_mean | 0.395 | 0.395384 | 0.305256 | yes |
| rf | fold_population_std | 0.098 | 0.097960 | 0.039217 | yes |
| visibility | pooled_source_macro_f1 | 0.318 | 0.317565 | 0.317565 | yes |
| visibility | pooled_clip_macro_f1 | 0.318 | 0.318057 | 0.318057 | yes |
| visibility | fold_mean | 0.285 | 0.284918 | 0.284918 | yes |
| visibility | fold_population_std | 0.104 | 0.104069 | 0.104069 | yes |
| majority | pooled_source_macro_f1 | 0.259 | 0.259398 | 0.259398 | yes |
| majority | pooled_clip_macro_f1 | 0.202 | 0.202125 | 0.202125 | yes |
| majority | fold_mean | 0.159 | 0.159394 | 0.159394 | yes |
| majority | fold_population_std | 0.022 | 0.021986 | 0.021986 | yes |

## Per-class and fold checks

For S-JEPA, the saved source-weighted recalls are 0.395833 (normal), 0.428994 (MS), and 0.569048 (PD), and precisions are 0.568189, 0.458888, and 0.385787. The PDF reports 0.611 PD recall, 0.479 MS precision and 0.403 PD precision; each is reproduced by the one-clip counterfactual. Normal and MS correct clip counts are 10/24 and 8/29; PD is 18/35, not 19/35. Consequently S-JEPA has 36 correct clips, while the historical RF and mean pose each have 37. The corrected RF has 30.

The saved S-JEPA fold source-weighted macro-F1 values are 0.515263, 0.362559, 0.779598, 0.315654 and 0.217267. Their mean is 0.438068 and population SD is 0.195980. The counterfactual changes fold 3 to 0.386466, yielding the draft's 0.452230 mean and 0.189059 SD. The draft's range 0.217–0.780 remains correct. S-JEPA exceeds historical RF in two folds (0 and 2), not the draft's three; the counterfactual would add fold 3. It exceeds corrected RF in three folds and mean pose in only fold 2.

All reported counts in dataset Table 1 agree with the registry and caches: 91 raw clips, 88 usable clips, 41 sources, 15 padded clips and 358 windows. The non-S-JEPA historical result rows agree to the shown precision. Stage selections remain ssl/ssl/ssl/continued/continued; original fold-0 and fold-4 validation examples agree with retained results. Those accurate quantities do not validate the inconsistent S-JEPA table entries.

## Bootstrap verification with the draft's exact settings

Every interval below uses 10,000 paired class-stratified source resamples with seed 20261004; repeated draws preserve weight one per occurrence. Label order is normal/MS/PD, sources are sorted, all methods share identical draws, and zero-denominator F1 is zero.

| Prediction basis | Quantity | Point | 95% percentile interval |
|---|---|---:|---:|
| Notebook/Sep21 saved | S-JEPA | 0.456624159 | [0.328714413, 0.578793142] |
| Notebook/Sep21 saved | S-JEPA minus RF | 0.045361249 | [-0.133446835, 0.220113928] |
| Notebook/Sep21 saved | S-JEPA minus mean pose | 0.004731283 | [-0.160496452, 0.173998638] |
| Corrected/Oct09 saved | S-JEPA | 0.456624159 | [0.328714413, 0.578793142] |
| Corrected/Oct09 saved | S-JEPA minus RF | 0.131440055 | [-0.046114011, 0.303792121] |
| Corrected/Oct09 saved | S-JEPA minus mean pose | 0.004731283 | [-0.160496452, 0.173998638] |
| Counterfactual explaining PDF — not data | S-JEPA | 0.468121736 | [0.343264573, 0.588596161] |
| Counterfactual explaining PDF — not data | S-JEPA minus RF | 0.056858826 | [-0.120935610, 0.230974905] |
| Counterfactual explaining PDF — not data | S-JEPA minus mean pose | 0.016228860 | [-0.147895738, 0.184568561] |

The PDF's repeated [−0.121, 0.231] interval equals the counterfactual RF interval after rounding. The counterfactual mean-pose interval is [−0.148, 0.185], so the identical displayed intervals are unsupported. For the actual notebook predictions the correct differences are 0.045361 [−0.133447, 0.220114] versus historical RF, and 0.004731 [−0.160496, 0.173999] versus mean pose.

Retrospective descriptive sensitivity intervals conditional on fixed OOF predictions, models, split and class source counts. Assumes source clusters independent, although participant independence is unknown and training sets overlap. No retraining, model selection, split or seed uncertainty; not a confirmatory population confidence interval or an equivalence test.

## Root causes and disposition

1. S-JEPA draft results are inconsistent with both retained OOF runs. A single unsupported ms-to-pd change on -oJM2wUUjws_P1_02 reconstructs all examined rounded S-JEPA results, including abstract, table, per-class values, fold spread and individual/RF bootstrap intervals.
2. The identical RF and mean-pose paired intervals are erroneous. Even for the counterfactual rows explaining the draft numbers, mean-pose gives [-0.1478957, 0.1845686], whereas RF gives [-0.1209356, 0.2309749]. Equality is consistent with a copied or misassigned interval, but the editing history is unavailable.
3. RF results in the draft describe the historical duplicated-ankle feature implementation. The corrected current implementation produces 12 different predictions and a lower score; all old and new RF predictions were independently reproduced by a one-column feature change.
4. Dataset composition, mean-pose/visibility/majority rows, mean-pose superiority in four folds, and the distinction between pooled metrics and fold spread agree with retained artifacts.

The abstract, Results, Discussion, Conclusion, main table, raw counts, per-class descriptions, fold comparisons and bootstrap statements must all be regenerated from a single declared saved run. The latest corrected RF belongs in a labeled reanalysis if the notebook-anchored study remains the main result. Notebook prose has historical claims about the RF defect; these should be described as historical context rather than as the current extractor's state.

Reproduce with `.venv/bin/python docs/wacv2007/evidence/results_audit.py`; inspect `original_draft_audit` in `results-audit.json` for exact values and the exhaustive one-change search. The counterfactual never overwrites OOF files or enters the study's reported results.
