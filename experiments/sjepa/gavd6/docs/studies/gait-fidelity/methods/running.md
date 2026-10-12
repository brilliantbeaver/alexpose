# Running and understanding the implemented experiments

[Proposal](../README.md) · [Follow-up HAIC commands](../../../../slurm/gait-fidelity/JEPA_RESPONSE.md) · [Notebook F](../../../../notebooks/gait_fidelity/experiments/F_jepa_response.ipynb) · [Resource contract](execution.md)

Pose restoration corrects observed joint positions over time by learning from clean reference trajectories. JEPA, a joint-embedding predictive architecture, first trains an encoder to represent observations as numerical features and predict reference features. A coordinate readout then learns position corrections while the encoder is frozen, so its parameters no longer change.

The active workflow is a separate JEPA movement-response follow-up, `jepa-response-01`, whose data and baseline predictions come from the completed `walking-core-01` run. It adds three pretraining objectives and nine final models. CPU tests and a generated-data fixture have checked the workflow; actual H100 compatibility, throughput and source results remain subject to HAIC preflight and profiling.

## Begin with the completed parent

Finish the core and its evaluation before initializing the child. Follow-up setup verifies the parent configuration, source release, admission records, prepared arrays, completion receipts and retained predictions. It also reads the update counts selected by the parent's actual profile. A parent with unresolved workers, changed artifacts or incomplete phases is refused.

Use the separate immutable release described in the HAIC guide. The child references existing arrays read-only and writes its own configuration, ledger and outputs. Its work directory must be outside the parent's directory. Preserve the parent's training-admission amendment, including any singleton-person exclusion; historical pilot counts are not the active cohort size. No preparation jobs, new data admission or confirmation evaluation are added.

The inherited source panel uses body12 tracks: shoulders, elbows, wrists, hips, knees and ankles on both sides. The source configuration uses 128 frames at 25 Hz, grouped into four-frame joint tokens. Physical movement states, cameras, naming errors and observation conditions retain the parent's metadata and references. Read the frozen manifest for actual counts and the retained review status of its intervals. Reusing data preserves its existing strengths and limitations; it does not upgrade an automated screen into a human reference review.

## Understand the three new models

Each row below uses seeds 17, 29 and 43, which set reproducible training randomness. A residual means a prediction minus its target; an auxiliary loss adds a penalty to the existing training objective. One final model requires pretraining and a subsequent coordinate-readout phase, producing nine pretraining phases and nine readouts in total.

| Variant | Added pretraining loss | Scientific purpose |
| --- | --- | --- |
| `jepa_delta_v1` | Difference between the two endpoints' centered student–teacher residuals. | Tests whether coupling errors across movement states improves the eventual restored response. |
| `jepa_endpoint_v1` | Separate squared centered residuals at each endpoint. | Tests whether extra continuous endpoint supervision explains the benefit. |
| `coordinate_delta_v1` | Difference between coordinate errors expressed in one common observation-derived scale. | Tests whether the remedy depends on predicting latent features. |

An endpoint is one member of a baseline/movement pair. The encoder sees its ordinary permitted observations, while pair membership and clean references enter the loss calculation only. All three variants use the inherited graph-time sampler and independently normalize each endpoint from retained observations. The two JEPA auxiliaries use the same coefficient; coordinate calibration has its own coefficient because its output units differ. The [response protocol](jepa-response.md) gives the equations and calibration rule.

After pretraining, the encoder is frozen and a fresh coordinate readout is initialized with matched seed-specific weights. Every child readout uses the existing `paired_change` objective: coordinate error plus error in the measured knee response, with the same short-segment penalty and coefficient. Readout training has no artificial pretraining mask. Ordinary deployment processes one observation sequence without its partner, reference, teacher or auxiliary loss.

Five methods are imported from the parent, each with its original `paired_change` readout or training objective: plain paired JEPA, coordinate pretraining, direct end-to-end training, the initialized encoder and shuffled-reference JEPA. The primary contrast remains delta JEPA against endpoint JEPA. Comparing the direct model with a frozen encoder is a practical benchmark whose trainable parameters also differ.

## Follow the saved dependency graph

Setup produces a child configuration and plan, followed by a combined calibration/profiling worker. Calibration, the fixed procedure for setting auxiliary-loss weights, uses 32 fixed training batches at seed-17 initialization without optimizer steps. It does not select a loss weight using development performance. The timing stage measures each new phase type and either admits all eighteen optimization phases at the parent's actual schedule or records a resource refusal.

An admitted pretraining phase enables only its matching seed/variant readout. After all fits complete, representation diagnostics inspect the parent and child encoders on a fixed training-only panel, then evaluation reads saved development predictions. If profiling refuses the matrix, the implementation can retain parent representation diagnostics and the refusal without creating a partial winning subset.

For exact commands, follow [setup, preflight, plan and launch](../../../../slurm/gait-fidelity/JEPA_RESPONSE.md). In every new HAIC shell, source the child's saved `session.env`. Resume an existing child through its saved session; do not recreate or edit the frozen run. Failed allocations remain charged, and an interrupted checkpoint is a retained attempt rather than a completed fit.

Notebook F opens the mathematics, saved configuration, calibration, feature diagnostics and response tables. Without a child session it runs small examples labelled as software illustrations. The earlier notebooks explain parent preparation, masks and core/full recipes; their availability does not expand the active follow-up. GPU work runs within Slurm allocations, while completed-output inspection and reconstruction can run on CPU.

## Inspect evidence at the right level

The final report links the primary paired comparison to `evaluation/response-comparisons.json` and the person table to `evaluation/per-person.csv`. Read these with `coverage.csv`, the condition-specific tables, individual seed differences and each limb's excursion. Response curves show actual reference changes; their successful-output points and fitted slopes are conditional diagnostics, while the primary absolute error retains failed eligible predictions.

Representation diagnostics preserve joint/time slots in feature vectors and fit a fixed linear probe, a small linear model that tests whether the response can be predicted from those features, using three groups of people that take turns being held out for evaluation. They describe whether the measured response can be read from these features; they neither choose a model nor admit or stop training. Identical baseline inputs under independently sampled masks quantify variation due to masking and normalization. The clean teacher sees fixed references and validity, so its variation in that comparison isolates the changing normalization.

Run the saved verification command after completion. It checks bound artifacts and reconstructs numerical tables in temporary storage without replacing published results. Submission success, a favorable probe or a lower pretraining loss cannot substitute for the completed nine-model comparison.

## Separate the active follow-up from the broader study

The implementation also retains the original core and full experiment sets. The graph-time core has ten recipe cells: coordinate, paired JEPA, direct, initialized and shuffled-reference models, each with base or paired-change output training, across the same three seeds. `walking-core-01` remains the parent; the child does not extend its ledger or replace its primary comparison.

The broader full matrix contains 34 recipes and 102 final models, including time-block and uniform-token masks, topology/duration controls, practical refiners, per-example measurement labels and re-paired change labels. Those controls address mask structure, data exposure and pairing specificity. They remain a separate experimental program, outside the active eighteen-phase child. Clinical events, new real-video references and independent confirmation likewise require their own data admission and analysis plan.
