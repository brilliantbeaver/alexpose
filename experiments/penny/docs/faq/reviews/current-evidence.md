# Current GAVD3 evidence ledger

Inspected on 2026-10-04. This ledger checks retained notebook code, outputs, CSVs, and JSONs; no training or evaluation was rerun. The FAQ should cite the retained copies in [evidence](../evidence/provenance-manifest.json). That manifest records the original relative path, SHA-256, byte count, and notebook execution inventory.

## Evidence locations and execution status

The important location distinction is `gavd3/cache/artifacts/real` versus the ignored `gavd3/work/artifacts/real`. The first retains notebooks 04 and 06 results; the second contains real results for 07–10, alongside copies of earlier artifacts. `gavd3/work/smoke_artifacts/smoke` contains synthetic fixtures and must not supply research results. Default `rg --files` omits ignored `work` files: recursive filesystem inspection was required.

Notebook 04 has 14 code cells with execution counts and seven with outputs; notebook 06 has 22 executed code cells and 14 with outputs. Notebooks 05 and 07–10 have no embedded execution counts or outputs. The CSVs and checkpoints in `work/artifacts/real` therefore establish retained evidence for 07–10 despite empty notebook outputs. Execution inventory alone does not establish whether a notebook was ever run. No numerical retrieval, similarity, or normal-distance results from notebook 05 were found in the inspected real artifact roots.

## Cohort and the original fixed-split experiment

The cohort is 96 sequences from 18 source videos: normal 12/one video; Parkinson's 9/two; stroke 12/three; cerebral palsy 16/two; myopathic 47/ten. Source counts come from [source_video_concentration.csv](../evidence/original/source_video_concentration.csv). All normal sequences are from video `3KnFt8bH3tE`.

Notebook 04 pretrains on all 12 normal sequences, including the normal sequences later called classifier test examples. Thus the downstream normal tests are transductive with respect to representation pretraining: their unlabeled inputs were used to train the encoder. This is different from sharing only source identity; both problems occur here.

Notebook 06 reproduces the legacy 68-row, nonstratified 47/21 split using the legacy NumPy permutation and the fixed sequence order. Its exploratory split uses all 96 rows with a fixed stratified 67/29 partition. The classifier is a pipeline containing `StandardScaler` and a Random Forest with 100 trees, maximum depth five, square-root feature sampling, bootstrap, balanced class weights, and random state 42. The encoder is frozen before classifier fitting.

| Lane/system | Correct/test | Accuracy | Balanced accuracy | Macro F1 |
| --- | ---: | ---: | ---: | ---: |
| Exact S-JEPA | 13/21 | 0.619048 | 0.596190 | 0.613333 |
| Exact missingness | 7/21 | 0.333333 | 0.363810 | 0.336061 |
| Exact training-majority constant | 7/21 | 0.333333 | — | — |
| Exact saved handcrafted 82-feature RF | 16/21 | 0.761905 | not supplied | 0.728333 |
| All-96 S-JEPA | 18/29 | 0.620690 | 0.624286 | 0.593651 |
| All-96 missingness | 13/29 | 0.448276 | 0.465714 | 0.428889 |
| All-96 training-majority constant | 14/29 | 0.482759 | — | — |

Sources: [classifier_metrics.csv](../evidence/original/classifier_metrics.csv), [missingness metrics](../evidence/original/missingness_only_classifier_metrics.csv), [comparison](../evidence/original/exp5_comparison.csv), and the [row contract](../evidence/original/classifier_contract.json). The training majority is myopathic in both splits. The old manuscript values 0.294 and 0.490 are not the test accuracies of the majority constants in these lanes; 47/96≈0.490 is the full-cohort myopathic proportion, which happens to equal the grouped comparator below.

The exact test class counts are cerebral palsy five, myopathic seven, normal three, Parkinson's three, and stroke three. Exact recalls are 3/5, 5/7, 2/3, 2/3, and 1/3. Stroke has two errors into cerebral palsy. The [error list](../evidence/original/five_class_errors.csv) records all eight wrong predictions. The two systems' 14.3 percentage-point difference is three predictions; it is a system comparison using differing pose and feature pipelines, rather than an isolated effect of representation learning. The handcrafted result is an imported saved reference, not a fresh rerun here.

| One-versus-normal task | Train/test sequences | Correct/test | Accuracy | ROC AUC |
| --- | ---: | ---: | ---: | ---: |
| Parkinson's | 14/7 | 5/7 | 0.714286 | 0.750000 |
| Stroke | 17/7 | 6/7 | 0.857143 | 1.000000 |
| Cerebral palsy | 19/9 | 8/9 | 0.888889 | 1.000000 |
| Myopathic | 41/18 | 14/18 | 0.777778 | 0.910714 |

These are condition filters applied after the global all-96 split, with four pretraining-exposed normal test clips in every binary task. Perfect ranking on these small samples does not establish independent discrimination.

The [leakage audit](../evidence/original/leakage_audit.csv) records all nine exact test videos in classifier training, and all 16 all-96 test videos in classifier training. Three exact test sequences and four all-96 test sequences were also normal pretraining inputs.

## Training objective and what its diagnostics measure

Notebook 04 prepares each clip using visibility threshold 0.45, interpolation of at most four internal frames, pelvis centering, a median shoulder/hip width scale, and temporal resizing to 64 frames. Interpolation preserves the original observed validity; a filled coordinate does not automatically become an eligible target. A joint-time patch has four frames of one joint, hence 12 coordinate values. The 64×33 grid becomes 16×33=528 tokens.

The encoder width is 96 with four layers/four heads; the predictor has two layers. There are 700,800 trainable parameters and zero trainable target-encoder parameters. The target encoder is an EMA copy receiving the complete clip; the view encoder receives augmented visible tokens. Uniform masking is restricted to ten shoulder/hip/knee/ankle/foot landmarks. The requested fraction 0.60 applies to eligible tokens inside that set. Equal masked-token counts across a batch use the minimum eligible count, so observed coverage makes the achieved eligible fraction slightly below 0.60 and the global fraction about 0.17–0.18, rather than 60% of all 528 tokens.

The teacher output minus its running center is softmaxed with temperature 0.06, and the predictor output with 0.10. Cross-entropy compares distributions across the 96 latent channels; those channels are not the five condition labels. Labels do not supply the pretraining loss, though the normal label selects the pretraining subset and literature supplied the anatomy prior.

The recommended run uses 300 epochs, batch size four, 900 updates, seed 42, AdamW, and cosine EMA from 0.999 toward one. In [training_history.csv](../evidence/original/training_history.csv), loss falls 12.541527→0.571975, feature standard deviation 0.339858→0.411713, and mean pair cosine 0.636178→0.534903. These are descriptive training quantities, not held-out predictive quality. The latter two diagnostics use one globally pooled vector per normal sequence without validity masking, so they check variation across the 12 sequences rather than directly checking each token's semantic meaning. They argue against a constant pooled representation; they do not rule out subtler collapse or identify clinically meaningful dimensions.

Downstream pooling concatenates validity-masked global mean/std and ten-landmark mean/std, yielding 4×96=384 features. Transformer attention can transmit missingness information from invalid tokens to valid ones; masked pooling alone does not remove that shortcut.

## Notebook 07: identity and grouped evaluation

[Identity probe](../evidence/later/07_video_identity_probe.csv) repeats unstratified 70/30 sequence splits with three seeds. All four feature families use the same Random Forest reader. Nominal uniform video guessing is 1/18=0.055556, but unequal source counts make this an incomplete baseline.

| Representation | Mean source-ID accuracy | Recorded standard deviation |
| --- | ---: | ---: |
| Frozen S-JEPA, 384 dimensions | 0.494253 | 0.121099 |
| Untrained same-architecture encoder, 384 dimensions | 0.517241 | 0.158020 |
| Raw patch pooling, 48 dimensions | 0.459770 | 0.039817 |
| Missingness, 100 dimensions | 0.425287 | 0.019909 |

The pretrained reader does not show an advantage over the untrained one in this small descriptive comparison. This does not prove pretraining adds zero identity information, or that equal raw and learned identity scores identify whether the signal is biomechanics or recording artifacts. The reader, feature dimensions, source availability, and finite splits constrain that inference. The untrained standard deviation 0.124 elsewhere in the manuscript conflicts with the retained 0.158020.

Six source-grouped folds each test 16 sequences. [Fold accuracies](../evidence/later/07_video_grouped_five_class_folds.csv) are 7/16, 0/16, 6/16, 5/16, 3/16, and 2/16, totaling 23/96=0.239583; their recorded sample standard deviation is 0.164965. Sequence-grouped folds use the same six-fold machinery and total 63/96=0.656250, standard deviation 0.116927. Training-fold modal-label constants average 0.489583=47/96 in both lanes. The code selects the training mode, not an oracle based on the test labels.

[Gap JSON](../evidence/later/07_generalization_gap.json) records 0.381106 between fixed 70/30 and source-grouped accuracy, and 0.416667 between the two six-fold lanes. These gaps reveal dependence on the evaluation protocol. They do not isolate a numerical fraction of accuracy caused by identity leakage, because folds change class support and clip composition.

A central structural failure remains: one fold holds all 12 normal clips, removing normal entirely from classifier training; its overall accuracy is zero. The frozen encoder was nevertheless already pretrained on those 12 clips. The [pretraining overlap](../evidence/later/07_pretrain_overlap.csv) records this. Thus source-grouped classification here is a stress test, not a valid fully inductive five-class estimate. Notebook 06 and 09 correctly block their stricter five-class grouped lanes until multiple normal sources exist.

The [leave-one-condition-video-out table](../evidence/later/07_one_vs_normal_grouped.csv) averages accuracy across condition videos: Parkinson's 0.000/2 videos, stroke 0.630952/3, cerebral palsy 0.437500/2, myopathic 0.641667/10. Test folds contain only the held condition's positives, and normal remains in fitting throughout. These accuracies are condition recall on held condition videos; they do not estimate normal specificity or balanced binary performance.

## Notebook 08: biomechanical proxies and weak linear readouts

[Parameter scores](../evidence/later/08_gait_parameter_probe_scores.csv) use standardized RidgeCV within leave-one-sequence-out fitting. All targets use the same raw cached monocular poses that supply the model inputs, rather than independent clinical measurements. Cadence uses ankle-x Hilbert phase advance and original FPS; speed uses pelvis displacement times FPS; other quantities are image-derived asymmetry, angle excursion, sway, and a toe-height stance/swing proxy. Cadence is cycles per second of an ankle signal, not validated clinical steps per minute.

| Proxy | N | Latent R² | Raw pooled-coordinate R² | Missingness R² |
| --- | ---: | ---: | ---: | ---: |
| Cadence cycles/second | 96 | 0.020025 | −0.050535 | 0.117935 |
| Step-length asymmetry | 96 | 0.023787 | 0.057378 | −0.127645 |
| Knee excursion | 96 | 0.085701 | −0.469545 | 0.174098 |
| Trunk sway | 96 | 0.036680 | −0.029256 | −0.012193 |
| Speed proxy | 96 | 0.053892 | −0.130557 | 0.028817 |
| Stance/swing proxy | 95 | 0.022257 | −0.022470 | −0.026203 |

These values show weak linear decoding under this reader and protocol. Knee decoding exceeds a limited raw pooling baseline but remains below missingness. Negative R² means predictions perform worse than the score's constant-mean reference; it does not mean information is absent from every nonlinear representation.

[Phase scores](../evidence/later/08_phase_probe_metrics.csv) average cos/sin R² to −0.127358 (cos −0.197508; sin −0.057208). Per-frame features repeat each four-frame segment's pooled vector. Whole sequences are kept together in five folds, avoiding adjacent-frame train/test overlap; shared source videos and pretraining-exposed normal sequences remain. A separate ankle readout has mean R² −0.533194 in [ankle metrics](../evidence/later/08_phase_probe_ankle_metrics.csv). No improvement claim follows from these scores. Beating a raw/missingness baseline alone would still not identify causally pure gait physics or clinical validity.

## Notebook 09: mask ablation and single-seed uncertainty

[Four real arms](../evidence/later/09_ablation_metrics.csv) each use seed 42 and 300 epochs. The neurologic arm is retrained; its 12/21 score differs from the original notebook 06's 13/21, so the latter must not be reused as the ablation's neurologic result.

| Mask arm | Correct/test | Accuracy | Macro F1 | Prepared-clip cadence R² |
| --- | ---: | ---: | ---: | ---: |
| Neurologic-10 | 12/21 | 0.571429 | 0.550476 | −0.059574 |
| Random-10 | 13/21 | 0.619048 | 0.613333 | −0.002372 |
| Motion-aware-10 | 12/21 | 0.571429 | 0.564790 | −0.030425 |
| Full-body-33 | 13/21 | 0.619048 | 0.613333 | −0.003129 |

The mask budget is held equal. `random-10` samples tokens exchangeably over all 33 joints; “10” refers to the budget's derivation, not restriction to ten randomly selected joints. `full-body-33` stratifies that same budget across time. Motion-aware sampling is restricted to the neurologic ten. Neurologic versus full-body changes both joint restriction and temporal sampling; it is not a clean isolated anatomy effect. Random versus full-body isolates temporal stratification, while neurologic versus motion-aware compares weighting within the same anatomy. The [sampler audit](../evidence/later/09_mask_sampler_audit.csv) records 90 masked tokens/sample in the illustrated audit batch, not a fixed count for every training batch.

The real [summary](../evidence/later/09_ablation_summary.csv) has one run per arm and blank uncertainty columns. The maximum accuracy difference is one test prediction. This supports “no demonstrated advantage in this single run,” not equivalence or proof that anatomy never helps. The grouped lane is blocked because normal has one source. The cadence readout uses cycles per frame after temporal resizing; it is a different target and protocol from notebook 08's raw-FPS cycles/second.

## Notebook 10: retained forecast signal and its boundaries

The forecast variant hides the final four temporal patches (16 prepared frames) across all 33 joints, keeping the first 48. The teacher still sees the entire clip. Hiding all future joints is necessary because otherwise non-neurologic future coordinates would expose future motion to the view encoder. This is an explicit variant of the earlier restricted infilling mask.

[Horizon losses](../evidence/later/10_horizon_prediction.csv) are 0.352519, 0.353637, 0.340690, and 0.375992 at four/eight/12/16 prepared frames after context; phase-bin baseline losses are 12.320189, 12.123582, 12.284669, and 12.603931. These are latent cross-entropies in nats, not degrees, centimeters, or coordinate errors. There is no monotonic growth with horizon in these four in-sample averages.

The learned forecaster is evaluated on the same 12 normal clips used to train it. The phase-bin template leaves out the current clip when averaging the other 11 teacher outputs, but the teacher itself was trained on all 12. Furthermore, the baseline averages teacher logits across joints and clips then treats them as predictor logits with temperature 0.10; it discards joint identity and is not trained or calibrated like the predictor. The dramatic loss ratio cannot establish superiority over a strong, equally trained temporal model or amount of motion understanding.

Past-only token availability also does not establish fully causal raw-data processing. `prepare_sequence` uses a scale estimated from median width across the whole clip, complete-clip temporal resizing, and short-gap interpolation that may consult a later observation. These can allow information from future raw observations into prepared context. The teacher's contextual targets also summarize full-clip information. Describe this as prediction of held future latent tokens from prepared context and require context-only preprocessing for an online forecasting claim.

[Condition errors](../evidence/later/10_ood_prediction_errors.csv): normal n=12, mean 0.362021; Parkinson's n=9, 2.541492; stroke n=12, 3.112661; cerebral palsy n=16, 2.965492; myopathic n=47, 3.038386. Ratios of means range from about 7.0 to 8.6. The normal reference is training fit, while abnormal inputs introduce new condition, source/person, camera, and pose distribution simultaneously. This cannot identify a condition-specific biomarker or error on unseen normal videos.

Per-sequence [errors](../evidence/later/10_sequence_forecast_errors_all.csv) show normal 0.162–0.627, Parkinson's 0.661–5.603, stroke 0.735–5.256, cerebral palsy 0.633–5.298, and myopathic 0.666–13.466. In the retained values the ranges do not overlap; the normal maximum and cerebral-palsy minimum are only approximately 0.0055 apart. Rounded to two decimals they both become 0.63, so saying observed ranges “overlap” is inaccurate. Neither separation nor a retrospectively chosen threshold is independently validated.

[Phase rollout probe](../evidence/later/10_phase_probe.csv) mean angular errors: predicted 1.530911 radians, teacher 1.400724, persistence 1.372475. The CSV field named `circular_correlation` is 0.029730, but its code computes mean cosine of angular error, a phase-alignment score rather than the usual circular correlation coefficient. The learned phase reader underperforms simply retaining context phase; its regression is leave-one-clip-out, but representation training uses every normal clip. No autoregressive rollout, action conditioning, planning, intervention test, or uncertainty distribution is retained.

## Statements the FAQ must revise or qualify

- Use accurate majority denominators and distinguish original notebook 06 from retrained mask arms.
- Present notebook 07's grouped number as a source stress test with an absent normal training class and SSL overlap, not independent clinical five-way performance.
- Treat fold spread, three split seeds, and single-seed arms as descriptive variation; none supplies a formal rejection of a clinical null hypothesis or a powered equivalence result.
- Explain a plausible starting null as a retrospective organizing question unless a preregistration is found. No retained preregistration or initial formal significance protocol was found in this inspection.
- Missingness matching or losing to a majority constant does not prove it contains no label information: balanced accuracy and macro F1, source association, model capacity, and uncertainty all matter.
- Weak linear probe scores establish limits of the tested readout, not absence of gait mechanics from every latent dimension.
- Forecast errors support an observed within-corpus input-shift signal with training-fit normal reference. They do not establish independent disease detection, validated gait physics, an online causal world model, or action-guided planning.
- Anatomical masking remains a tested hypothesis with no demonstrated advantage in the one retained matched run. Future normal-source diversity, representation-pretraining isolation, repeated grouped evaluation, and controls using context-only preprocessing remain required for broader claims.
