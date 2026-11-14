# Manuscript claim ledger

This ledger separates recorded results, implementation facts, additional reanalysis, and interpretation. It covers the substantive counts, methods, findings and limitations used in the revised manuscripts. Notebook cell numbers are **zero-based JSON positions, including Markdown cells**; they are not execution counts. The [notebook audit](notebook-audit.md) provides the detailed cell-by-cell review, and [results-audit.json](results-audit.json) contains unrounded numerical values.

## Source hierarchy

The seven top-level notebooks define the intended experiment, as requested by the user. Notebook 06 contains a numerical narrative and runnable evaluation code, but its seven code cells have no saved execution outputs. Its narrative describes the historical full-v1 capstone with 215.981 seconds elapsed, the duplicated-ankle RF implementation, and rounded source-weighted macro-F1 0.457/0.452/0.411 for S-JEPA/mean pose/RF. The matching retained run is used to verify those numerical statements.

| Alias | Evidence source | Role |
|---|---|---|
| NB00–NB06 | Top-level notebooks numbered 00–06 | Experiment definition, saved outputs where present, and narrative context |
| REG | `artifacts/eval/full-v1/fold_registry.json` | Frozen inventory, exclusions, source identities and partitions |
| HIST | `artifacts/runs/full-v1/9496e61b050f/laptop-1d09c8e1eea5/capstone-20260921T011305448893Z/` | Notebook-primary `oof.json`, `results.json`, `provenance.json`, fold artifacts |
| CORR | Same parent directory, `capstone-20261009T060955236580Z/` | Separately labeled corrected-RF execution and verification evidence |
| AUDIT | `results_audit.py`, `results-audit.json`, `results-numerical-consistency.md` in this folder | Added metric recalculation, paired source bootstrap, RF refits and forensic checks |
| MASK | `notebook-audit-checks.json` | Added deterministic mask/readout calculations; separate from actual training traces |
| REFS | `reference-audit.md`, `reference-audit.json`, and cited primary papers | Bibliographic and external-claim checks |

The supplied v1 PDF is the document being corrected. It has no authority over conflicting notebook-associated predictions. Current source code explains active behavior, but the corrected RF code must not be mistaken for the historical implementation behind HIST. Older `g1`/R1 results use another dataset and procedure; they are excluded from the primary study. The standalone Fold-0 teaching run in NB03–NB05 is also distinct from the five capstone folds.

Status labels below mean: **recorded** = retained data or experiment output; **implementation** = directly inspected executable behavior; **reanalysis** = a manuscript-stage calculation on existing artifacts; **inference** = a qualified interpretation; **unrun/unknown** = not established by the available evidence.

## Collection and observation process

| ID | Manuscript claim or quantity | Status | Evidence and precise boundary |
|---|---|---|---|
| D01 | 91 raw clips from 41 source-video IDs; labels Normal/MS/PD | Recorded | NB00 cell 8; REG `inventory.raw_clips`. Labels come from collection folders/metadata. |
| D02 | Raw counts 26/30/35, usable clips 24/29/35, sources 16/13/12 | Recorded and reanalysis | NB01 cell 13; REG inventory; AUDIT `counts`. Source counts are not participant counts. |
| D03 | Three exclusions leave 88 usable clips and all 41 sources | Recorded | NB01 cell 13 and `artifacts/eval/full-v1/exclusions.json`: `JD1AGVpftps_P1_02`, `diCVwltkV5M_P1_01`, `tsOMPBS277Q_P1`. The log does not distinguish which gate failed for each clip. |
| D04 | 7,643 retained frames: 1,427 Normal, 3,450 MS, 2,766 PD | Recorded and reanalysis | NB01 cell 13; AUDIT `counts`; cached-array lengths. |
| D05 | 358 windows: 59/171/128; 15 padded clips: 5/0/10 | Reanalysis of notebook-defined procedure | `sjepa/data.py::sliding_windows`; AUDIT `counts`; MASK `dataset`. Window counts do not increase the independent source count. |
| D06 | One MS source supplies 13 usable clips | Recorded and reanalysis | REG cache entries for `tsOMPBS277Q`; AUDIT source counts. NB00 cell 9's 14 refers to raw clips, including one exclusion. |
| D07 | Source IDs are extracted from filename prefixes | Implementation | `sjepa/data.py::source_id_from_name`; `sjepa/splits.py::inventory`. Prefix grouping cannot detect repeat people or footage across different IDs. |
| D08 | MediaPipe/BlazePose-family full-frame pose extraction, first returned pose, 33 landmarks | Implementation and notebook execution | NB01 cells 6–7; `sjepa/data.py::load_video_sequence`, `_keypointset_to_row`; `ambient/pose/keypoint_extractor.py::extract_from_image`. This is not verified participant tracking. |
| D09 | Arrays contain pixel x, pixel y and visibility; depth excluded | Implementation | NB01 cell 7; `ambient/pose/keypoint_data.py::KeypointSet.from_mediapipe`; `sjepa/data.py::_keypointset_to_row`. MediaPipe image-normalized coordinates are converted to pixels before this array is cached. |
| D10 | Nominal 15 fps uses `max(1, round(source_fps/15))` stride | Implementation | NB01 cell 6; `sjepa/data.py::load_video_sequence`. Realized sampling rates and physical window durations can differ across recordings. |
| D11 | Initial gate requires fully finite x coordinates for all 33 landmarks in at least 30% of sampled frames | Implementation and recorded exclusion context | `sjepa/data.py::clean_sequence`; NB01 cells 9/13; exclusion file. It is not a confidence-threshold gate on each coordinate. |
| D12 | Empty ends trimmed, missing coordinates and visibility interpolated without an interior-gap limit; at least eight frames retained | Implementation | `sjepa/data.py::clean_sequence`; NB01 cell 9 and extraction loop cell 13. “Short gaps only” in older prose is inaccurate. |
| D13 | Per-frame hip-centering and torso-length scaling with lower bound 10⁻³ | Implementation | NB01 cell 9; `sjepa/data.py::normalize_sequence`. The denominator is shoulder-midpoint to hip-midpoint distance. |
| D14 | Spatial normalization leaves already cleaned visibility unchanged | Implementation | Same function. Visibility can have been interpolated during the earlier cleaning step. |
| D15 | Angles within a frame are preserved under the common positive scale, while absolute pelvis translation/apparent torso size are removed | Mathematical consequence | Normalization equation and code. Defined 2D angles are preserved; calibrated 3D quantities, absolute speed and metric stride length are not recovered. |
| D16 | Padding/interpolation/viewpoint may influence apparent dynamics | Inference | Their transformations are known; their effects on classification were not isolated. Do not state that they caused particular errors. |
| D17 | Different source IDs may include the same person, and a source may include multiple people | Unknown possibility | No participant-identity adjudication in REG or notebooks. The study neither demonstrates nor rules out these forms of overlap. |

## Representation, training and baselines

| ID | Manuscript claim or quantity | Status | Evidence and precise boundary |
|---|---|---|---|
| M01 | All 33 landmarks and three channels enter 32-frame windows; four frames/joint become one token; 264 tokens | Notebook and implementation | NB02 cell 9; NB03 cells 8–9; `sjepa/config.py`, `sjepa/tokenizer.py::SkeletonTokenizer`. Three channels are x/y/visibility. |
| M02 | Twelve highlighted joints are shoulders, hips, knees, ankles, heels and foot indices | Notebook and implementation | NB02 cell 11; `sjepa/masking_v2.py::CLINICAL_JOINTS`. They influence the sampler; they are not the entire input or a permanent mask. |
| M03 | Seven overlapping anatomical groups; contiguous time spans of one to six blocks | Implementation | `sjepa/masking_v2.py::JOINT_GROUPS`, `sample_target_mask`; NB02 cells 12–13. |
| M04 | Group weights use a 1.5 factor when selected joints are present, divided by mean membership multiplicity, then normalized | Implementation | `_region_weights`. This is a region-selection factor, not an exact 1.5× marginal masking rate for each selected joint. |
| M05 | Nominal target ratio 0.6 may overshoot; at least one selected-set context token remains somewhere in the window | Implementation | `sample_target_mask`; NB02 cell 13. Context is not guaranteed for each leg or every time block. |
| M06 | Saved 512-mask seed-0 bank averages about 63.3% targets | Recorded diagnostic | NB02 cell 13; exact recalculation in notebook audit: 0.6327607126. This is not a training-stream exposure log. |
| M07 | Added 5,000-mask seed-0 bank averages 63.3161% targets and exhibits nonuniform joint/time coverage | Reanalysis | MASK `mask_diagnostic`. Distinct sample bank from M06; no downstream accuracy ablation. |
| M08 | Encoder: three layers, width 96, four heads, MLP expansion two, zero dropout; predictor: two layers, width 96 | Implementation and retained configuration | NB03 cells 9/14; HIST `results.json.config`; `sjepa/models.py`. The compact evaluated recipe is not claimed as a new general architecture. |
| M09 | Student context is augmented; teacher sees full untransformed input and receives no loss gradient | Implementation | NB03 cell 12; `sjepa/train_v2.py`, `sjepa/models.py`. Target slots are excluded as attention keys/values; the implementation does not physically remove all hidden token computations. |
| M10 | Window-level augmentation: ±15° rotation, [0.9,1.1] scale, ±0.1 translation, reflection probability 0.5 | Implementation | NB03 cell 12; `sjepa/augment.py::random_view`. Reflection permutes left/right landmark and visibility slots. |
| M11 | Softmax temperatures 0.10/0.06; teacher center momentum 0.9; EMA starts 0.996 and approaches 1 | Implementation and configuration | `sjepa/losses.py`, `sjepa/train_v2.py`; HIST config. The 96-dimensional distributions are feature distributions, not disease probabilities. |
| M12 | Cross-entropy averages target tokens within examples, then examples | Implementation | `sjepa/losses.py`; NB03 cell 12. No coordinate reconstruction or label prediction enters this encoder loss. |
| M13 | AdamW peak lr .001, decay .05, betas .9/.95, clip norm 1, 10% stage warmup followed by cosine decay | Implementation | `sjepa/train_v2.py::train_sjepa_v2`; NB03 cells 13–14; HIST config. Legacy epoch fields do not control this loop. |
| M14 | 800 initial + 400 continuation updates per fold, batch 32, seed 42 | Recorded configuration | NB03 cell 14, NB04 cell 10, NB06 cells 9–11; HIST `base_updates`, `continuation_updates`, `config`. This is one declared seed, not multi-seed evidence. |
| M15 | All three training labels supply label-free windows, with equal total sampling probability per source | Implementation | NB06 cell 9; `sjepa/data.py::SequenceWindowDataset`; `sjepa/train_v2.py`. Names “pretrain_normal” and “ms_pd_vicreg” are historical filenames, not active restrictions. |
| M16 | Continuation carries network weights but resets optimizer, center, schedules and seeded sampling streams | Implementation | NB04 cells 8/10; `train_sjepa_v2` called anew. It is not equivalent to one uninterrupted 1,200-update call. |
| M17 | Frozen EMA teacher sees complete windows; a fixed seed-0 mask selects 161 output token features, then windows are averaged to a 96-vector | Implementation and reanalysis | NB04 cell 11; `sjepa/full_experiment.py::embed_records`; `sjepa/models.py::embed`; MASK `readout`. The readout mask does not hide inputs; all 33 joints occur in the selected set. |
| M18 | Probe is training-only StandardScaler + balanced logistic regression, C=1, max_iter=2000, random_state=42 | Implementation | NB04 cell 13; NB06 cell 8; `full_experiment.py::fit_probe`. The iteration cap is not an observed iteration count. |
| M19 | Strictly higher validation source-F1 selects continuation; ties keep initial; no train+validation refit | Implementation and retained selection | NB04 cell 13; NB06 cells 9/11; `full_experiment.py::run_fold`; HIST selections. Test representation extraction follows selection. |
| M20 | Mean pose is 66 mean normalized x/y values; visibility is 66 per-joint mean/std values | Notebook and implementation | NB05 cell 9, NB06 cell 8; `full_experiment.py::nuisance_features`. Both are order-invariant, but can encode posture, phase occupancy or recording effects. |
| M21 | Mean pose and visibility use the same probe recipe, with separately fitted parameters | Implementation | `fit_probe` and `run_fold`; NB06 cell 8. Different dimensions, stage selection and full-clip versus window coverage mean these are not matched causal time-order interventions. |
| M22 | RF has 100 trees, max depth 5, square-root feature subsampling, balanced classes and seed 42 | Implementation | NB06 cell 8; `sjepa/classical.py::train_rf_and_predict`; HIST provenance. Constant columns and scaling are fitted only on training clips. |
| M23 | Feature container has 82 slots, 15 varying columns per training fold; angle means/asymmetries/ranges are actually populated | Reanalysis and implementation | NB06 cell 8; AUDIT `rf_verification`; `ambient/classification/features.py::from_joint_angles`. This route is not a complete 82-measure clinical gait model. |
| M24 | Historical left-ankle range duplicates right-ankle range; current code corrects it | Historical narrative plus verified implementation difference | NB06 cell 8; HIST predictions; current feature code; AUDIT RF refits. Pair each result with its implementation version. |
| M25 | Majority predicts MS in folds 0/2/3 and PD in 1/4 | Recorded and implementation | HIST OOF; NB06 cell 8; `run_fold`. It uses training-clip frequency, not source frequency or test labels. 1/3 is not a universal macro-F1 chance level. |

## Evaluation and verified numerical findings

| ID | Manuscript claim or quantity | Status | Evidence and precise boundary |
|---|---|---|---|
| E01 | Five source-stratified outer folds, seed 42; first split of four-fold inner procedure, seed 43+k | Notebook and implementation | NB02 cell 6, NB06 cells 5–7; REG folds; `sjepa/splits.py::make_registry`. One inner holdout per outer fold, not exhaustive nested CV. |
| E02 | Training/validation/test source and clip counts are 24/51, 8/19, 9/18 in fold 0; 24/58, 9/17, 8/13 in fold 1; 24/54, 9/17, 8/17 in fold 2; 24/53, 9/21, 8/14 in fold 3; 24/45, 9/17, 8/26 in fold 4 | Recorded and reanalysis | REG; AUDIT `folds`; NB06 cell 7. Every training partition has 24 sources. |
| E03 | All 88 usable clips appear exactly once as outer-test rows; same-source clips never cross within-fold partitions | Recorded and validated | `splits.py::validate_registry`; `full_experiment.py::summarize_oof`; independent AUDIT assertions. This checks recorded source identity, not participant independence. |
| E04 | Source-weighted metrics give each clip weight 1/(usable clips in its source), then pool confusion counts and macro-average class F1 | Implementation | NB06 cell 12; `full_experiment.py::source_weights`, `score_records`; AUDIT independent confusion-matrix calculation. It is not a single source-vote classifier or average per-source F1. |
| E05 | Fold spread is population SD (ddof=0); it is not a standard error | Implementation and statistical qualification | `sjepa/eval.py::aggregate_folds`; folds share training data. |
| R01 | S-JEPA source F1 .456624; clip F1 .397034; fold mean/SD .438068/.195980 | Recorded and independently recalculated | NB06 cell 14; HIST OOF; AUDIT `notebook_primary_metrics.sjepa`. |
| R02 | Mean pose source F1 .451893; clip F1 .409840; fold mean/SD .427332/.132210 | Recorded and independently recalculated | Same sources, `mean_pose`. Use JSON for exact precision if rounding changes. |
| R03 | Historical RF source F1 .411263; clip F1 .397316; fold mean/SD .395384/.097960 | Recorded and independently recalculated | Same sources, `rf`; duplicated-ankle implementation. |
| R04 | Visibility source F1 .317565; clip F1 .318057; fold mean/SD .284918/.104069 | Recorded and independently recalculated | Same sources, `visibility`. |
| R05 | Majority source F1 .259398; clip F1 .202125; fold mean/SD .159394/.021986 | Recorded and independently recalculated | Same sources, `majority`. |
| R06 | S-JEPA source lead over mean pose .004731 reverses under clip weighting; mean pose wins four of five folds | Recorded and reanalysis | HIST OOF; NB06 cells 14/18; AUDIT fold scores. This is procedure-specific evidence, not proof that motion is unhelpful. |
| R07 | S-JEPA folds .515263/.362559/.779598/.315654/.217267; range .217–.780 | Recorded and reanalysis | HIST OOF; AUDIT `fold_source_macro_f1`; NB06 cell 14. The v5 results figure uses registry indices 0–4 and matched within-fold connectors; archived v3 uses display labels 1–5. |
| R08 | S-JEPA correct clips: Normal 10/24, MS 8/29, PD 18/35; 36 total | Recorded and reanalysis | HIST OOF; `clip_weighted.confusion`; NB06 cell 17. Mean pose and historical RF have 37 correct clips each. |
| R09 | S-JEPA source-weighted recalls .395833/.428994/.569048; precisions .568189/.458888/.385787 | Recorded and reanalysis | HIST OOF; AUDIT per-class metrics in Normal/MS/PD order. Row-normalized confusion percentages are weighted clips, not verified patients. |
| R10 | Initial stage selected in folds 0–2, continuation in 3–4; sampled final loss decreases in all five while validation improves in two | Recorded | HIST `selections`; NB06 cell 11. Final losses are sampled minibatch endpoints, not common held-out expected-loss measurements. |
| R11 | Fold-0 validation .293979 → .276105 selects initial | Recorded | NB04 cell 13 and HIST fold-0 selections. Standalone teaching execution remains distinct from capstone model provenance. |
| R12 | Training-only silhouette is near zero; no held-out accuracy conclusion follows | Recorded diagnostic with qualified inference | NB05 cells 9/17: 51 training clips/24 sources, original standardized feature space; current saved rounded values −.004/−.012/−.004. Historical more precise values from another run must not be substituted. |
| R13 | 10,000 paired stratified source bootstrap resamples, seed 20261004, source counts 16/13/12 held fixed | Added reanalysis | AUDIT `original_draft_audit.historical_saved_draft_seed_bootstrap`. This is manuscript-stage uncertainty analysis, not notebook output. |
| R14 | Conditional S-JEPA interval [.328714,.578793]; SJ−mean pose .004731 [−.160496,.173999]; SJ−historical RF .045361 [−.133447,.220114] | Added reanalysis | Same JSON object. Same resampled sources for each model; each repeated source draw contributes total weight one. No retraining or seed/split/selection uncertainty is included. |
| R15 | Corrected RF source F1 .325184; clip F1 .320940; 30 correct clips; 12 predictions differ from HIST | Separate verification evidence | CORR OOF; AUDIT `corrected_verification_metrics` and `run_comparison`. Do not replace R03 silently. |
| R16 | Current features reproduce all 88 CORR RF predictions; duplicating only left-ankle range reproduces all 88 HIST RF predictions | Added deterministic RF verification | AUDIT `rf_verification`. This demonstrates sufficiency of the feature change under these saved folds/settings; no neural retraining was performed by this audit. |
| R17 | SJ−corrected RF .131440 [−.046114,.303792] | Added reanalysis | AUDIT `original_draft_audit.latest_saved_draft_seed_bootstrap`, same 10k/20261004 settings. Separate from notebook-primary comparison. |
| R18 | Both runs share data, registry, split, seed and S-JEPA hard predictions; embedding arrays/checkpoint bytes differ | Recorded and reanalysis | HIST/CORR provenance and manifests; AUDIT `run_comparison`. Reports support repeat execution, not literal identical neural-cache reuse or independent-seed/cohort replication. |

## Interpretation, unresolved tests and external claims

| ID | Claim or question | Status | Permitted conclusion |
|---|---|---|---|
| I01 | No consistent advantage over mean pose is established in this evaluated configuration | Inference | Small source difference, reversed clip ordering, four losing folds and broad conditional paired interval support this bounded statement. They do not establish equivalence. |
| I02 | Model accuracy might reflect static posture, viewpoint, visibility or other acquisition effects | Plausible alternatives, not identified causes | Controls show those summaries can predict some labels. They neither locate a shortcut in S-JEPA nor exhaust alternative cues. |
| I03 | Reflection swaps physical joint observations without mapping the mask to the same physical joints | Implementation defect | NB03 training code plus `augment.py` and `train_v2.py`; MASK diagnostic. This creates a within-example correspondence/information path, not held-out-source or label leakage. |
| I04 | Mask-bank exposure quantifies a potential information path, not its effect on classification | Added diagnostic, causal effect unrun | MASK: pooled conditional exposure 21.1694% in a 5,000-mask seed-0 bank; expectation at reflection probability .5 is 10.5847%. Neither is a measured saved-training exposure rate. |
| I05 | Benefit of temporal order, anatomical masking or learned versus untrained weights | Unrun | Requires matched time-order, mask-rule and untrained-encoder controls. No positive or negative causal benefit may be claimed from the present comparison. |
| I06 | Generalization across independently identified participants, sites or verified diagnoses | Unrun/unknown | Source IDs are the available grouping unit; no participant/site-independent validation or adjudicated clinical labels. |
| I07 | Screening performance, clinical biomarker validity, severity prediction, calibration or neural decoding | Unrun | No suitable reference outcomes or neural recordings were analyzed. The task is collection-label classification. |
| I08 | Consent, rights to redistribute source video, and ethics-review status | Unknown in reviewed records | Public availability is not evidence of authorization. This audit supplies no missing consent, rights assessment or review decision. |
| I09 | MS/PD gait background and markerless gait measurement validity | External literature | REFS and verified primary citations. Clinical-study findings motivate the question; they do not validate this collection's labels, normalization or classifier. |
| I10 | S-JEPA/JEPA/MAE methodological relationships | External literature | REFS and original method papers. Do not call all self-supervised methods JEPAs, confuse graph-based SkeletonMAE with a transformer, or cite motion recognition as clinical validation. |

## Discrepancy diagnosis and corrections

The numerical reconstruction is detailed in [results-numerical-consistency.md](results-numerical-consistency.md). The script tested all 176 possible one-clip changes to the three-class S-JEPA predictions. A unique counterfactual—changing `-oJM2wUUjws_P1_02` from recorded `ms` to `pd`—reproduces the supplied PDF's complete rounded S-JEPA table/per-class/fold signature, as well as its individual bootstrap interval and RF difference interval. Both retained runs record `ms`. This identifies the unsupported numerical alteration needed to reproduce the PDF, **not how or by whom the discrepancy arose**. Synthetic rows were never saved as evidence or substituted into study files.

| Discrepancy/root cause | Exact disposition |
|---|---|
| Headline S-JEPA .468, clip F1 .406, fold mean .452 and SD .189 correspond to the unsupported one-clip counterfactual | Replace throughout abstract, table, Results, Discussion and Conclusion with .456624, .397034, .438068 and .195980; round consistently. |
| Inflated PD success count and per-class scores | Replace 19/35 PD with 18/35; total 37 with 36; PD recall .611 with .569; MS/PD precisions .479/.403 with .459/.386. |
| S-JEPA said to beat historical RF in three folds | Actual HIST wins are folds 0 and 2 only. The counterfactual would add fold 3. Corrected RF is a separate comparison with three S-JEPA wins. |
| RF and mean-pose paired intervals both printed [−.121,.231] | That interval matches the counterfactual RF after rounding, but counterfactual mean-pose is [−.147896,.184569]. Actual saved-data intervals are R14. A copied/misassigned interval is consistent with the evidence; editing history is unknown. |
| Current corrected RF and historical notebook RF conflated | Keep HIST primary; disclose CORR separately with its lower .325184 score and explicit feature verification. A correction need not improve predictive performance. |
| Sampling, cleaning and coordinate transformations described out of order | Describe detection → pixel coordinates → cleaning → torso normalization. Distinguish image-normalized, pixel and torso-normalized coordinates. |
| Twelve-joint prior described as all inputs or fixed mask; exact 1.5× per-joint rate implied | Report all 33 input joints, seven weighted overlapping groups and the multiplicity correction. Treat realized mask-bank rates as diagnostics. |
| Old filenames/configuration text imply normal-only pretraining, progressive disease addition, epochs or VICReg | Report the active all-condition, label-free, update-based two-stage objective. |
| Training diagnostic silhouettes/embeddings or repeated execution treated as held-out replication | Identify the relevant standalone/capstone run; keep training-only observations descriptive and same-seed execution separate from independent replication. |
| Broad clinical, participant-level or temporal-causal claims outrun the evidence | Use the boundaries in I01–I10 and identify the needed studies as unrun. |

The reference audit separately corrects bibliographic inaccuracies and claim-to-citation mismatches. It should accompany this ledger rather than treating an authentic reference as proof that every sentence citing it is valid.
