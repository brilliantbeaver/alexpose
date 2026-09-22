# Laterality evidence audit for the synthetic-restoration abstract

*19 September 2026 · Independent manuscript evidence audit · Version 01*

This is an audit of existing evidence, not a new experiment. It distinguishes the reported AMASS laterality study from the retained real-video reflection probe and from the newer rendered 2D restoration pilot. No existing artifact was overwritten.

## What the laterality result can support

The completed latent-laterality comparison is a controlled study of real motion-capture trajectories with artificially imposed joint-name errors. Its numerical results are available here only as a historical written summary. The original per-window predictions, fitted models, evaluation contract and benchmark gate tables are not installed in this checkout. Consequently, their statistics cannot be independently reconstructed for this manuscript revision.

The strongest supported conclusion is that a probability-aware laterality model improved over a correction-first model in one development seed, but the same architecture supplied with uninformative 50/50 probabilities reproduced the improvement. The gain therefore does not establish the value of informative assignment probabilities. The study stopped confirmation and left the test split sealed.

Sources: [validation summary](../../latent-laterality/results/validation.md), [evidence-availability inventory](../../../repository/evidence-availability.md#latent-laterality), and [experiment registry](../../experiment-registry.json).

## Data, endpoint and units

| Property | Reported design or result | Manuscript consequence |
| --- | --- | --- |
| Source | AMASS motion capture; selected travelling motions with clear pelvis travel direction | Actual recorded movement under controlled synthetic corruption; not observed errors from real-video pose tracking |
| Skeleton | Core11: pelvis plus five bilateral lower-body pairs; sampled at 30 Hz | Different input/target space from the rendered 2D body12 restoration study |
| Conversion | 8,854 motions considered; 3,134 converted; 3,076 sufficiently long | Selection limits breadth; conversion success is not a model result |
| Training | 2,474 motions, 113 people | Fitting and calibration use training identities |
| Validation | 259 motions, 15 people; 1,146 overlapping evaluation windows | The independent outer evaluation unit is the person, not the window |
| Test | 343 motions, 15 people | Sealed and unevaluated |
| Corruption | Full-sequence coherent swaps of all five bilateral pairs, independently added missing-data gaps, no general coordinate noise | Does not cover arbitrary partial-joint errors or establish their real prevalence |
| Window | 64 frames; four-frame blocks; 32-frame stride | Correlated windows must not be treated as independent examples |
| Main targets | Mean magnitude of blockwise right-minus-left distal motion energy; mean total energy | Coordinate-derived unsigned motion summaries; not signed anatomical accuracy, clinical measures or 2D trajectory restoration |
| Main metric | Absolute error divided by the training mean absolute target; then averaged within person and across people | Normalized MAE here is not comparable numerically to v2 reference-box-normalized coordinate error |
| Repetition | One model-training seed, 7 | No model-seed uncertainty or confirmed architecture ranking |

The metric and target meanings were checked against the retained [evaluation implementation](../../../../src/gavd6_sjepa/research_directions/latent_laterality/evaluation.py). This checks the current source contract, not the provenance of the absent historical execution.

## Reported decisive comparison

| Representation | Unsigned side-sensitive normalized MAE | Side-insensitive normalized MAE |
| --- | ---: | ---: |
| Raw, uncorrected | 1.6309 | 1.0820 |
| Raw, continuity-corrected | 1.5727 | 1.0221 |
| Raw, answer-key-corrected | 1.4851 | 1.0097 |
| Correction-first S-JEPA | 0.9300 | 0.7472 |
| SG-JEPA with informative path probabilities | 0.8671 | 0.6696 |
| Otherwise matched uniform-uncertainty control | 0.8668 | 0.6690 |

SG-JEPA's reported reductions against correction-first are 6.8% and 10.4%; the uniform control achieves 6.8% and 10.5%. SG-JEPA is reported to beat uniform for only 5/15 people on the side-sensitive score and 6/15 on the side-insensitive score. Three people contribute 73% of validation sequences, although person-balanced averaging limits their influence on the headline score.

**Exploratory arithmetic check, not fresh empirical inference:** recalculating from the rounded published table gives reductions of 6.7634% and 10.3854% for SG-JEPA, versus 6.7957% and 10.4657% for uniform. The written summary's finer 0.04%/0.08% comparisons between SG-JEPA and uniform cannot be reproduced exactly from four-decimal entries (rounded entries give about 0.0346%/0.0896%). This is compatible with rounding, but the unavailable full-precision artifacts prevent resolution. Prefer “matched the gain” or the displayed values rather than finer relative differences. No new confidence interval is warranted from these summaries.

The difference between model families still changes architecture and input correction. The evidence does not isolate which remaining design element explains the common gain. “Pairing causes the improvement” would be an unsupported mechanistic claim.

## What the benchmark gates establish

The initial sequence benchmark failed: an arbitrary coordinate convention was predictable (reported AUROC 0.680, upper estimate 0.786), and continuity correction already matched the answer key. The repaired design pairs arbitrary conventions and separates missingness from swap boundaries. Its reported convention AUROC is 0.500; missingness-relative improvement is −0.000239. Continuity correction has error 0.114 on temporarily swapped sequences while the answer-key correction has zero, leaving controlled headroom.

Among 177 validation sequences with temporary changes, continuity recovers the whole path in 106 and makes a path error in 71; it worsens the side-sensitive result relative to doing nothing in 36. These figures motivate comparing correction accuracy with preservation, but they are summary-only synthetic-corruption results. The repaired audit contains 2,733 source draws, represented as 5,466 paired views across 128 training/development people. The two views are not independent scientific examples and the training exposure count is not doubled.

The preliminary frozen-encoder swap probe reports an error decrease from 2.79 to 0.096 under continuity correction. It used an artificial root anchor and a legacy encoder, and continuity matched the answer key. It is weaker background evidence than the repaired three-arm comparison and should not occupy the final abstract.

## Anatomical laterality and the proposed 2D paper

A temporary exchange of left/right tensor slots can alter a side-sensitive movement summary without changing the underlying physical movement. This makes preservation of bilateral relationships a scientifically meaningful target alongside coordinate accuracy and movement amplitude. However, a coordinate sequence alone cannot establish absolute anatomical left/right under the laterality study's specific symmetry assumptions: globally swapping latent anatomical names while complementing the hidden assignment path leaves observations unchanged. An independent anchor or retained informative cue is required to name the anatomical side.

This limitation is conditional, not a claim that all monocular video is ambiguous. Pixels, appearance, trusted viewpoint or an independently checked limb marker may provide a cue absent from the controlled model. See the [theory document](../../latent-laterality/protocol/theory.md).

For the new paper, the strongest connection is methodological: preserving the quantity of interest and testing the proposed mechanism require separate controls. The laterality study's uniform-probability control and the v2 study's initialized, coordinate and shuffled-pairing controls all prevent a broad model-versus-baseline gain from being credited prematurely to a specific latent-learning mechanism. The original synthetic-training study's limited estimator-specific headroom supplies a parallel reason to test pooled supervision before adding personalization. These observations motivate the new restoration question; they do not establish its answer.

A suitable abstract-level bridge is: “Motivated by controlled laterality failures and limited gains from estimator-specific synthetic adaptation, we examine whether paired synthetic supervision can improve pose trajectories without changing the movement they describe.” A later sentence can report the current two-person synthetic development result and explicitly leave movement preservation unresolved.

## Retained real-video evidence from a different study

The adjacent fixed-reflection study has retained predictions under [work/artifacts/gavd_core11_frozen_probe](../../../../work/artifacts/gavd_core11_frozen_probe). Its 96-sequence adapter table spans 18 source videos. The strict, no-short-clip-padding prediction cohort contains **90 unique sequences**, not 96, from the same 18 videos.

**Exploratory reconstruction of existing outputs:** I recalculated per-fold, five-class macro-F1 from `strict90_no_short_clip_padding_nested_probe_predictions.csv` and averaged the five folds. The reconstructed values agree with the retained summary to numerical precision:

| Representation | Recomputed mean macro-F1 |
| --- | ---: |
| Raw Core11 | 0.423090569561 |
| EMA paired shared/no-cross | 0.233891100300 |
| EMA reflection-equivariant | 0.245067213488 |
| Randomly initialized controls, range across six recorded controls | 0.333557226399–0.536561445091 |

All folds share 10–12 source videos between fitting and evaluation (mean 11). The full adapter cohort's normal category has only one source video; independent-video grouped generalization is therefore unsupported. This is a real-video **within-corpus condition-classification diagnostic**, not a test of SG-JEPA, real swap repair, or synthetic-to-real restoration. See [the associated report](../../reflection-equivariance/results/gavd-probe.md).

These results may belong in an evidence appendix explaining the progression of the project. Adding them to the final restoration abstract is likely to obscure the task and create a misleading impression of completed real-data restoration validation.

## Claims to keep out of the manuscript abstract

- That SG-JEPA fixes naturally occurring tracking-name errors, estimates their prevalence, or preserves anatomical sign.
- That 5,466 paired views or 1,146 overlapping windows are independent people or independent trials.
- That the laterality result is a confirmed multi-seed or test-set result.
- That informative uncertainty, temporal learning or clean-pair alignment has been established as the source of the reported gains.
- That fixed-reflection GAVD classification demonstrates unseen-video or synthetic-to-real restoration transfer.
- That improved 2D coordinate agreement proves preserved amplitude, timing, bilateral coordination or clinical accuracy.
- That the historical normalized MAE and new normalized coordinate error can be pooled into one improvement statistic.

## Improvements available now and those requiring evidence

Revision can sharpen the scope, distinguish controlled corruption from natural tracking errors, use the two-person/one-seed limitation plainly, explain strong controls, and reserve the historical laterality numbers for the evidence record if abstract space is tight. Those changes improve accuracy, positioning and clarity.

New evidence is required to establish preservation on the v2 trajectories, incremental paired-JEPA benefit, independent real-video transfer, convergence sufficiency, calibration mechanisms and seed/population stability. Restoring the original laterality predictions is required before independently checking that historical result or producing fresh uncertainty intervals. These limitations should keep evaluation and reproducibility scores below a strong-submission level even when later abstract versions read more smoothly.


## Independent critique of abstract revision 02

*Reviewed 19 September 2026; draft left unchanged.*

[Revision 02](abstract-v02-preservation.md) handles the laterality bridge accurately. It explains that exchanging joint names affects side-sensitive measurements and immediately discloses the uniform-control result. The final sentence identifies laterality and restoration as summary-reported evidence. Calling these “controlled joint-name-swap experiments” would make the artificially imposed corruption even clearer without adding a long qualification.

The timing sentence is also appropriately restrained: no paired-JEPA record satisfies the joint reference/prediction requirements. It does not treat absent support as zero error, or infer that all unsupported records were distorted by the model. Keeping reference eligibility separate from prediction peak-count agreement is essential to subsequent revisions.

Three concrete improvements would strengthen revision 03:

1. **Separate the two explanations being tested.** “Systematic correction or weak observability explains these findings” can suggest that weak observability explains coordinate improvement. Calibration tests how much coordinate improvement comes from a simple fitted correction. Reference self-comparison establishes attainable timing support, allowing reference-ineligible records to be distinguished from incomplete predictions or peak-count mismatch. A compact alternative is: “Pending calibration controls will assess the source of coordinate gains, while reference self-comparison will separate limited timing eligibility from prediction failures.”
2. **Avoid implying statistical equivalence.** “The initialized readout and shuffled pairing are competitive” is understandable, but “yield similar coordinate-error point estimates” is more precise with only two evaluation people and one seed. The result does not establish equivalence, exclude small benefits or identify why models are close.
3. **Keep the measurement scope visible.** The intended question is preservation of amplitude, timing and left–right relationships, but current motion metrics are derived from image-plane ankle trajectories. They do not establish anatomical-side recovery, biomechanical coordination or preserved physical movement. The draft's figure caption states this correctly; one short qualification in the abstract would reduce dependence on material readers may not see.

For narrative improvement, lead with what paired latent prediction adds when clean synthetic coordinates and a readout are already available to controls. Condense the earlier studies into their shared justification for strong mechanistic comparisons. Retain their role as motivation rather than treating their different targets as additional replications of the v2 restoration result.

The title is acceptable as a statement of scope and does not assert that preservation succeeded. The reported percentages are consistent with the v2 summary. No higher statistical-rigor or reproducibility score is justified by these revisions: neither new independent people nor missing prediction artifacts become available through wording. Accuracy or clarity can improve if the ambiguous future-check sentence and potential equivalence inference are repaired.


## Independent critique of abstract revision 05

*Reviewed 19 September 2026; draft left unchanged.*

[Revision 05](abstract-v05-encoder-readout.md) interprets the initialized control fairly: the encoder is untrained and fixed, while the coordinate readout is fitted. Substantial coordinate improvement therefore weakens attribution to encoder pretraining. It does not establish that no temporal learning is needed, because the trained readout can still learn from temporal inputs. It also does not establish that a tiny calibration is sufficient. The distinction should remain explicit in revision 06.

No numerical claim error was found. The named extractor comparison, anatomical-proxy qualification and summary-only provenance are consistent with the current evidence. Signed image-plane ankle separation is an appropriate operational bilateral endpoint, provided it remains a measurement in the image coordinate frame. Its amplitude does not certify anatomical asymmetry or general left–right coordination; its detected peaks are not independently labeled gait events. “Ankle-separation peak timing” is clearer than an unqualified “event timing” where the actual measured endpoint is described.

Two phrases deserve correction:

- **“Gains attributed to more elaborate mechanisms.”** This implies that the prior studies had established or endorsed those causal attributions. Their controls instead rejected the intended extra mechanism. Use “matched the more elaborate methods” or describe the control outcome directly.
- **“Distinguish coordinate correction from representation gains.”** These are not disjoint explanations: the learned readout and pretrained representations both participate in coordinate correction. Calibration can quantify how much improvement a simple fitted correction recovers. It cannot by itself identify anatomical-convention mismatch as the cause or separate every representation mechanism.

A readable two-sentence motivation is: “Controlled joint-name swaps can distort side-specific motion summaries, yet a laterality model’s gain survived removal of informative assignment probabilities. An earlier synthetic-adaptation study likewise found no added benefit from response-based personalization.” This names what each study found without asking readers to decode “uniform control” or “response-free selector.” A tighter alternative can merge the common control lesson after first explaining why bilateral measurements matter.

The title “Evaluating Paired Synthetic Supervision for Movement-Preserving Pose Restoration” is qualified as an evaluation goal, so it is not strictly a false success claim. However, it can still be skimmed as naming a demonstrated preserving method. **“Paired Synthetic Pose Restoration: Testing Accuracy and Movement Preservation”** states the scope with less implication of an achieved outcome.

For the final future-work sentence, name the actual pending measurements: simple calibration will assess recoverable coordinate gains; reference self-comparison will locate the timing-support limitation. Further people, seeds and independent real references are needed for broader claims. The statement that preservation remains unresolved is necessary and should survive stylistic tightening.


## Final independent critique of revision 06 and the evidence ledger

*Reviewed 19 September 2026; drafts and ledger left unchanged.*

[Revision 06](abstract-v06-iclr.md) and the [evidence-and-rubric record](abstract-20260919-evidence-and-rubric.md) preserve the essential distinctions. The ledger correctly identifies recorded motion capture with imposed naming errors, empirical synthetic observations, summary-only restoration evidence, unrelated within-corpus real-video classification, analytic fixtures and uncompleted analyses. It does not pool their metrics or use real training images as evidence of real-video transfer. No material claim error was found.

Four refinements would make revision 07 more self-contained:

- Restore the explicit goal of **movement amplitude, timing and left–right relationships** in the opening sentence. This is an objective, and the later unresolved-preservation statement prevents it becoming a success claim.
- Describe the historical input as “reported motion-capture experiments with imposed joint-name swaps.” This is more concrete than “controlled laterality summaries” and distinguishes recorded underlying movement from naturally occurring errors in real-video pose estimates.
- Name the current bilateral endpoint as **signed horizontal ankle separation in the image plane**. This is narrower than the broad relationship-preservation goal and must not imply anatomical-side identification, general coordination, gait events or metric stride measures.
- Keep the temporal learning inside the readout visible. “A fitted temporal readout on an untrained frozen encoder also substantially reduces coordinate error” is accurate without implying that the readout is a tiny calibration, that temporal information is unnecessary or that model errors are statistically equivalent.

If space permits, make the evidence sequence explicit: preservation must first be established on the saved synthetic trajectories and subsequently assessed against independent real-video references. New real-video data would not by itself explain the present synthetic timing-support failure; the reference self-comparison is still needed.

The current **70.50/100 total can remain unchanged in revision 07**. These edits refine scope and accessibility within already-high accuracy and clarity judgments. They do not recover missing predictions, supply independent people or training seeds, demonstrate the proposed latent mechanism, or establish movement preservation. The fixed rigor score of 3.5 and reproducibility score of 5.5 remain defensible. The total is an editorial assessment, not evidence of acceptance readiness or a substitute for the unresolved empirical contribution.
