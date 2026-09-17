# What the proposed experiments can legitimately claim

> **Scope: September 14 seven-proposal portfolio (P1–P7).** The later [synthetic-training proposal](../proposals/synthetic-training-selection.md) has its own experiment and annotation plan; its requirements supersede the portfolio budget for that direction.

This is a new decision portfolio following notebook 06. No new training, HAIC execution, or video annotation was performed while writing it. Completed results, proposed methods, and planning thresholds are different kinds of evidence.

**The immediate constraint is scientific uncertainty, not GPU count.** Eight H100s provide at most 1,344 GPU-hours in seven days, before queueing, failures, rendering and analysis. The plan spends a small fraction establishing useful information before selecting one experiment. A public architecture can be reliable while its proposed use remains unverified.

**Start with these definitions.**

| Term | Meaning in this portfolio |
| --- | --- |
| JEPA | Joint-Embedding Predictive Architecture: learn by predicting a representation of missing observations. This does not automatically make every released encoder causal or action-conditioned. |
| Representation or latent | Numbers produced by an encoder. A useful latent prediction score need not imply an accurate motion forecast. |
| Frozen model | Its weights stay fixed. Only a small attached head, pooling module, or adapter is trained. |
| Prefix | Every observation available by a declared cutoff time. Future frames are evaluation targets, never preprocessing inputs. |
| Candidate | One proposed correction, future, or extra observation. Its usefulness must be measured for the stated task. |
| Oracle | A diagnostic allowed to inspect hidden reference answers. It measures opportunity, not a deployable result. |
| Proper score | A forecast-distribution score that rewards truthful probabilities in expectation. One observed future per example is sufficient to evaluate its average, but not to identify all sources of uncertainty. |
| Headroom | How much improvement is available before asking a learned method to recover it. |
| Held out | Excluded from the relevant training and selection. Held-out adaptation people are not necessarily absent from foundation-model pretraining. |

**The completed studies constrain the new hypotheses.**

| Observation | Evidence | Consequence |
| --- | --- | --- |
| Expanded future-feature study still stopped | 1,403 windows from 290 recordings; RGB R² 0.662830; aligned skeleton increment 0.0003571; mismatched increment 0.0006029 | Do not promise that scaling the same latent-target objective yields useful motion transfer. This does not rule out skeleton-only learning or video helping a skeleton model. |
| A previous laterality readout favored random initialization | R² 0.222544 versus approximately 0.1008–0.1142 for the trained conditions | One observable became less linearly accessible. Information deletion, all-model failure, and clinical impairment do not follow. |
| Additional bone projection damages accurate observations | Projected raw removal is -78.1% calibration and -53.9% development on the displayed event-plus-noise groups | Restore true identity and evaluate projection independently of any learned verifier. |
| Current MoMask reconstruction exceeds the error budget | Raw MSE 0.27 cm²; unprojected MoMask 170.94 calibration and 20.94 development cm² | Current reconstruction is not a demonstrated useful repairer. Clean-motion damage also occurs. |
| The current repair oracle has little margin | Pointwise unprojected removal 27.4% calibration and 30.0% development | A 25% target requires about 91% and 83% of these oracle gains. A block-level verifier has a smaller, unknown ceiling. |
| Flow responds to the paired video change | Favorable relative contrast in 15/16 pairs; both expected preferences in only 5/16 | This motivates an evidence test. It is not 94% repair accuracy. The summary also uses a privileged event mask. |

Sources: [scaling strategy](../../../docs/studies/future-feature-prediction/scaling/research-strategy.md), [notebook 06 report](../../../docs/studies/motion-preservation/results/pilot-01-diagnostics.md), and [its displayed tables](../../../docs/studies/motion-preservation/results/pilot-01/diagnostics/). The notebook 06 event-plus-noise comparison contains eight people per role. Its 128 cases are constructed variants, not 128 independent people. Full per-case curves remain on HAIC. Means across these different studies must not be pooled.

**AMASS provides the primary quantitative motion reference.** The [manifests](../../../manifests/amass/) identify 8,854 eligible converted motions from 189 unique audited identities: 151 training, 19 validation and 19 test. The 201 split rows include aliases. Eligible source counts are KIT 4,232 motions, BioMotionLab 3,061, Eyes Japan 750, EKUT 349, ACCAD 247 and HDM05 215. They do not provide six equally large replication cohorts.

Use full-body 22-joint positions and compatible SMPL-H rotations, retaining metric scale and real timestamps. The raw inventory includes the required body parameters; the current motion-preservation path already processes the full body. Core11 is an ablation. Fitted joint rotations and rendered material-point trajectories are model-derived references from captured motion. They are not measured clothing motion, contact force or clinical outcomes. Main forecasting results use untouched captured motions. Rendered views and injected observation failures test mechanisms; do not call synthetic video an in-the-wild result.

Preserve the existing subject allocations. The motion-preservation validation roles are nine calibration and ten development people; the 19 final people remain untouched during choice of flagship. Use inner training folds for new method choices. The memory proposal has a smaller activity-eligible subset, so it cannot inherit a 19-person effective sample size automatically. The previous requirement for 20 final people was infeasible under this split and is explicitly retired in the new proposal, not silently satisfied.

**GAVD supplies real video, not hidden clinical ground truth.** Its [manifests](../../../manifests/gavd/) contain 1,874 sequences from 348 recording IDs. They do not contain verified participant identity, measured 3D joints, severity progression or force. Source-separated results cannot be described as participant-separated. The two normal/abnormal annotation fields are not interchangeable. The [GAVD paper](https://arxiv.org/html/2407.04190v1) also qualifies the often quoted 92/94% performance: its GAVD test subset is abnormal-only, and normal-video/view transfer is weaker. We retain the user's exclusion of binary classification without claiming that external gait diagnosis is solved.

For a selected proposal needing a quantitative GAVD motion endpoint, reserve **60 clips from at least 40 eligible development recording IDs**, with four specified frames and two clearly visible landmarks per clip. This is 240 frames and 480 landmark placements, plus visibility labels. Budget 8–12 human hours, including a second review of at least 20% and resolution of disagreements. Use existing RGB only; no clinical collection is involved. These annotations do not exist yet. Preselect sources and frames independently of model errors. Keep the prior feature study's protected confirmation recordings excluded. Derive image displacements in original pixels, normalized by an observed-prefix person-height estimate, and report pixels as well. Timing events need their own annotated endpoint if used.

This small panel is external corroboration, not population-level validation. Any image-coordinate forecasting or utility head must be trained and calibrated outside these recording IDs, then frozen. Prefer a render-trained 2D head for a declared zero-shot test; if separate GAVD pseudo-label training is used, report it and keep its sources disjoint. The annotation panel cannot simultaneously train the head and supply external evaluation. If annotation cannot be completed, GAVD remains a qualitative real-video stress test and the paper claim must narrow. Never substitute the same flow estimator's output as both input and unquestioned truth. Automated tracker outputs may be reported as pseudo-reference sensitivity analyses only.

**The shared model stack is deliberately small.** Use the locally available V-JEPA 2.1 ViT-B checkpoint for frozen dense features, SEA-RAFT for estimated image transport, existing body geometry, and standard small temporal heads. A compatible existing skeleton checkpoint is optional. Unverified S-JEPA author weights, human action-conditioned world models, GaitForeMer clinical data, or GaitDynamics-to-SMPL conversion are not dependencies. See the [access ledger](portfolio-literature.md).

The distilled 2.1 B/L encoders and their predictors can use different output spaces. Do not subtract B-encoder tokens from a predictor trained against the G-size teacher. The proposals use frozen encoder features with explicitly trained small heads, not an unverified native prediction energy. Process the prefix independently: full-clip attention, interpolation, floor estimation, velocities, contact channels, crops, and tracking can otherwise leak future information. Extra evidence in P1 and P5 comes from already observed frames. It is not an imaginary new camera view of GAVD.

**Use comparisons that can defeat the idea.** Match permitted observations, temporal extent, target normalization, trainable capacity and tuning effort. Include coordinates, kinematic extrapolation, static/shape/source controls, raw flow, and the strongest proposal-specific alternative. Query selection cannot inspect an unrevealed crop and claim its cost was avoided. A compact downstream token interface does not eliminate dense upstream extraction cost.

Group all derived windows, views and edits with their original person or recording before fitting. Source, person, trial, and pretraining overlap are different boundaries. Prefix length, motion energy, cadence, body size, centroid displacement, missingness and foreground area are nuisance or simple-state controls when relevant. They are never acceptable substitutes for the intended physical endpoint.

The 48-hour thresholds in the proposals are **new prospective continuation choices**, not effect estimates or power calculations. Keep the old repair target where the old condition is reused. Select one primary endpoint per proposal; secondary plots do not rescue a failed primary claim. Report paired person or recording intervals and three final training seeds separately. If the eligible group count is too small to resolve the effect, say so. Selecting among seven pilots on development data does not turn development into a fresh final test.
