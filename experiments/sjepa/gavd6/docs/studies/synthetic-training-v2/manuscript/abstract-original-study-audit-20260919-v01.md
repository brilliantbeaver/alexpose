# Original synthetic-training evidence for the abstract revisions

*Independent source audit · 19 September 2026 · Version 01 · Append-only manuscript support*

The original synthetic-training study supports the decision to test synthetic supervision directly before adding a more elaborate learning mechanism. It does not establish movement preservation or real-video transfer. Its outcome table is an empirical synthetic development experiment, distinct from the analytic fixtures used to test the revised study's software.

## What was executed and what can be checked

The original experiment adapted the final prediction heads of pretrained image pose estimators. A common ten-update probe preceded eight alternative synthetic lessons, a pooled lesson, and replay controls. Synthetic branches also used labeled real COCO replay. The selector used summarized prediction changes and other source/context information to choose a lesson; it did not output a restored pose trajectory. Its V-JEPA context encoder was frozen, and no S-JEPA model was trained.

Retained source outcomes cover four estimator checkpoints: HRNet-W32 and RTMPose-M for fitting, and HRNet-W48 and RTMPose-S for validation. All four use seed 17. There are 2,304 aggregate outcome rows, 6,912 selector decisions, and 144 configurations. The reported 75-update budget and selector settings were chosen on the validation panel. Each policy mean averages two validation estimators across 24 recurring synthetic scene conditions, for 48 correlated estimator–condition outcomes. These are not 48 independently sampled people or 48 independent adaptation runs.

On 19 September, `reconstruct_pilot(Path.cwd())` in the [retained-evidence audit implementation](../../../../src/gavd6_sjepa/research_directions/synthetic_training_v2/audit.py) was run again without an output directory. It checked complete keys and joined every selector decision to its source outcome. The largest cross-file numerical discrepancy was 9.71445146547012e-17; the largest recomputed summary discrepancy was 3.469446951953614e-18, both below the declared 1e-12 absolute tolerance. No model was fitted, inference rerun, or historical file changed.

The metric is visible-body-landmark Euclidean error divided by a reference person-box diagonal. It is dimensionless, and lower is better. Missing predictions incur a one-diagonal penalty. The retained CSVs already aggregate across frames and people, so this audit verifies aggregate arithmetic rather than reconstructing each frame's metric.

## Recomputed comparisons

| Policy | Mean normalized error | Relative reduction versus replay |
|---|---:|---:|
| Replay after the common probe | 0.02732053665 | Reference |
| Pooled synthetic lessons | 0.02703671741 | 1.03885% |
| Full response selector | 0.02702648000 | 1.07632% |
| Matched source-progress selector | 0.02702576239 | 1.07895% |
| Fixed front lesson | 0.02689915697 | 1.54236% |
| Scene/domain selector | 0.02678073984 | 1.97579% |

The full response selector is approximately 0.00266% worse than the matched source-progress control in relative error. It changes three of 48 lesson choices: one improves the outcome, one worsens it, and one changes the lesson without changing the outcome. These small, selected development differences do not establish a useful response-personalization mechanism.

The retrospective shared-scene oracle has mean error 0.02667588234; the student-specific oracle reaches 0.02666505946. Their additional relative improvement is 0.0405718%. Both choices use the outcomes after they are observed. The difference describes the remaining personalization opportunity in this specific action library and development panel; it is neither an achieved policy nor a population-wide upper bound.

## Exploratory decomposition by validation estimator

The following newly recomputed decomposition is exploratory. It uses the frozen selected policies and retained source errors, without changing their settings or selecting a new policy. Each cell averages the same 24 scene conditions for one estimator.

| Validation estimator | Replay error | Full response reduction | Matched source-progress reduction | Scene/domain reduction |
|---|---:|---:|---:|---:|
| HRNet-W48 | 0.02723933438 | 0.04859% | 0.13241% | 0.03025% |
| RTMPose-S | 0.02740173892 | 2.09796% | 2.01987% | 3.90980% |

The pooled improvement is concentrated in RTMPose-S. Scene/domain selection is best in the pooled reported comparison, but is not best for each estimator separately. This strengthens the need to retain estimator-specific results and weakens any broad claim that a particular selector uniformly improves synthetic adaptation. The decomposition has no new confidence interval and supplies no independent replication.

## Evidence categories and limits

| Evidence category | What is available here | What the abstract may say |
|---|---|---|
| Empirical synthetic development outcomes | Source error CSVs, selector decisions, selection settings, probe loss records | Synthetic adaptation produced modest selected development gains; response personalization added no demonstrated benefit over the matched control. |
| Real images used during training | Labeled COCO replay in adaptation batches | Real-image replay was used; this does not make the evaluation a real-data transfer result. |
| Independently verified real-video outcomes from the original study | None in the retained source bundle | Real GAVD transfer remains unestablished. |
| Frame-level restoration or movement preservation | No retained per-frame predictions or joint trajectories from the original study | No conclusion about displacement, amplitude, timing, or left–right preservation follows from these aggregate visible-coordinate errors. |
| Analytic software fixtures | Separate revised-study fixture runs | They demonstrate software behavior, not a biological cohort or an empirical transfer result. |
| Planned experiments | Held ViTPose-family transfer, real recording evaluation, confirmation | Describe as future work unless completed, traceable results become available. |

The retained run-02-v1 bundle contains no NPZ prediction caches and no PT checkpoints. Its full realized person and render manifests are absent. The preparation code indicates a small repeated motion/reference library, but the available inventory cannot certify the realized person identities or supply a person-level uncertainty estimate. Model selection on the same validation panel, one seed, two related validation checkpoints, and repeatedly rendered conditions prevent strong generalization claims.

## How this should shape the abstract

The useful bridge to the revised paired-restoration study is: earlier source adaptation showed that synthetic training can change visible coordinate accuracy, while adding response-based personalization failed to improve on a matched simpler control. This motivates testing whether the paired clean-target learning objective contributes beyond coordinate supervision, an initialized encoder with a fitted readout, and simple calibration.

The original study should usually occupy one compact motivating clause in the abstract. Its 2,304 rows, 6,912 decisions, eight lesson names, and retrospective oracle percentage would crowd the central restoration question and invite the wrong interpretation of sample size. If one numerical historical result is needed in a longer introduction, report the roughly 1.98% scene-policy and 1.08% full-response improvements with their synthetic-development scope, not as evidence of real gait restoration.

Do not combine these original image-head adaptation errors numerically with the revised trajectory-restoration errors: they use different data, targets, interventions, and evaluation settings. Do not interpret frozen V-JEPA context features as a trained JEPA ablation. The laterality study can motivate why bilateral information requires direct evaluation, but its real-data summaries remain a separate evidence stream; neither historical study establishes that the revised model preserves left–right movement relationships.

## Traceable retained sources

- [Original study writeup](../../synthetic-training/study/README.md), especially the methodology and results limitations.
- [Detailed retained-evidence audit](../artifact-audit.md) and [machine-readable original pilot audit](../../synthetic-training/study/evidence/pilot-audit.json).
- [HRNet-W32 outcomes](../../../../notebook_runs/synthetic-training/run-02-v1/source/hrnet_w32/outcomes.csv), [HRNet-W48 outcomes](../../../../notebook_runs/synthetic-training/run-02-v1/source/hrnet_w48/outcomes.csv), [RTMPose-M outcomes](../../../../notebook_runs/synthetic-training/run-02-v1/source/rtmpose_m/outcomes.csv), and [RTMPose-S outcomes](../../../../notebook_runs/synthetic-training/run-02-v1/source/rtmpose_s/outcomes.csv).
- [Frozen selector settings](../../../../notebook_runs/synthetic-training/run-03-v1/selectors/selection.json), [validation decisions](../../../../notebook_runs/synthetic-training/run-03-v1/selectors/validation_predictions.csv), and [validation summary](../../../../notebook_runs/synthetic-training/run-03-v1/selectors/validation_summary.csv).

## Independent final review of abstract revision 06

*Added 19 September 2026; the historical audit above is preserved. Review target: [revision 06](abstract-v06-iclr.md). This review supplies suggestions for revision 07 and does not edit either abstract.*

The coordinate comparison's directions and magnitudes are correct. An independent decimal calculation from the rounded reported table gives paired-JEPA error reductions versus coordinate pretraining of 0.476471% for HRNet, 0.717215% for ViTPose, and −14.036532% for RTMPose. Thus “less than 1%” on the first two and “worsens RTMPose by 14.0%” accurately summarize that baseline. The two reported intervals including zero are summary-derived evidence, not newly calculated intervals. These calculations add no observations or replication.

The original-study sentence needs to retain its comparator. “No added benefit from choosing lessons using a model's learning response” should specify the matched selector without that response. The full selector did improve on replay, so omitting the comparator can accidentally suggest that synthetic lesson selection had no benefit at all. The laterality motivation is appropriately called a controlled, summary-reported result; it should remain separate from natural video failure prevalence.

The initialized control has a fitted nonlinear coordinate readout and a frozen, untrained temporal encoder. Its predictions are not the output of a wholly untrained system. In the code, the readout maps contextual encoder tokens to coordinate corrections; cross-frame information is still available through the encoder. Its competitive coordinate-error point estimates therefore challenge the necessity of encoder pretraining under this recipe, not the usefulness of all learning or temporal information. “Similar coordinate gains” would be clearer as “competitive coordinate-error point estimates”: the baseline for “gains” is otherwise implicit, and initialized RTMPose is better than paired JEPA.

Inspection of the training phases also confirms that the shuffled arm changes target pairing during latent pretraining only. Its coordinate-readout phase still receives correctly aligned labels. The phrase “shuffled pretraining pairs” is accurate; a claim that pairing throughout the pipeline is unnecessary would not follow. See the [model implementation](../../../../src/gavd6_sjepa/research_directions/synthetic_training_v2/models.py) and [phase-specific training implementation](../../../../src/gavd6_sjepa/research_directions/synthetic_training_v2/training.py).

The final sentence should distinguish two unresolved questions: preserving movement on the synthetic references, and transferring to real video. Independent video evaluation is necessary for the latter; it is not the sole missing condition for the former. The absent timing self-check and unreported amplitude/displacement errors still prevent a synthetic-preservation conclusion.

The [fixed scoring rubric](abstract-20260919-evidence-and-rubric.md) defines submission fit as formal and topical fit, rather than acceptance probability, so its high submission score can be read consistently. The claim-accuracy score of 9.0 is defensible only for accurately bounded, summary-reported claims and benefits from the wording repairs above. The unchanged evaluation score of 3.5 and reproducibility score of 5.5 correctly retain substantial empirical weaknesses. The scientific-insight score of 7.5 should not rise in the final revision: the objective comparison is better articulated, but the proposed explanation for coordinate gains remains untested. A further stylistic improvement alone warrants no increase in the total.
