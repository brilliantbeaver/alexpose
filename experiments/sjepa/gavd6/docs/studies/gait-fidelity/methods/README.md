# Methods and experimental protocols

[Research proposal](../README.md) · [Data specification](../data/README.md) · [Interactive paper](../proposal.html)

A pose records joint positions at one time, and a trajectory follows those positions over time. Restoration corrects noisy observed trajectories using a clean reference for training and evaluation. Here the question is whether the corrected trajectories preserve a known change between two movement states.

The active experiment studies JEPA, a joint-embedding predictive architecture that learns to predict numerical feature vectors of a reference sequence. Its encoder converts observations into features during pretraining, the initial learning stage. A later coordinate readout converts those features into position corrections while the encoder is frozen, meaning its learned parameters no longer change. We ask whether coupling prediction errors across the two movement states during pretraining improves the response preserved by that readout. The separate `jepa-response-01` run reuses the completed `walking-core-01` data and retained comparisons. Three new objectives across seeds 17, 29 and 43 produce nine models through eighteen optimization phases. Source GPU results remain pending; the completed local validation establishes software behavior on CPU.

The [plain-language terms and examples](terms.md) explain the model and measurement vocabulary. Start with the response protocol for the scientific comparison, then read the running and execution guides before using the HAIC commands. The broader masking matrix and clinical validation proposals remain useful context, but they are outside the follow-up's 48 H100-hour allocation and September 24, 2026, 6 PM Pacific cutoff.

| Protocol | What it specifies | Read before |
| --- | --- | --- |
| [JEPA response coupling](jepa-response.md) | Paired latent residuals, independent endpoint regression, the coordinate control, and their limits. | Interpreting the active hypothesis or its controls. |
| [Running the experiments](running.md) | How the completed parent supplies data and predictions to the child, and which outputs to inspect. | Using the [HAIC follow-up guide](../../../../slurm/gait-fidelity/JEPA_RESPONSE.md) and notebook F. |
| [Execution and resource contract](execution.md) | The eighteen phases, inherited schedule, timing admission, budget and cutoff. | Launching or resuming `jepa-response-01`. |
| [Evaluation and measurement](evaluation.md) | The signed knee measurement, person-balanced response error, missing predictions and uncertainty. | Reading comparison tables or making a scientific claim. |
| [Masking and input contract](masking.md) | The implemented body12 sampler, context-only normalization and common auxiliary support. | Inspecting masks, feature diagnostics or loss coverage. |
| [Data and references](../data/README.md) | Which observed tracks, references and identities support the experiment. | Interpreting the inherited population and source limitations. |
| [Method and novelty audit](../literature/novelty.md) | The relationship to prior feature prediction, derivative supervision and anatomical masking. | Describing a contribution beyond the controlled finding. |

The primary comparison is **delta JEPA versus endpoint JEPA on person-balanced `response_error`**. Both keep the original centered cross-entropy and representation regularizer, add a calibrated continuous loss, then freeze the encoder and train the same fresh readout with coordinate and paired-change supervision. The coordinate-delta model tests whether any benefit also appears when pretraining predicts coordinates. Original core predictions supply plain JEPA, coordinate, direct, initialized and shuffled-reference comparisons without refitting them.

The measured response is a change in the right-minus-left difference in projected knee excursion. Each limb's excursion is the 95th minus 5th percentile of its image-plane hip–knee–ankle angle on fixed reference-supported timestamps. This engineering outcome has physical units of projected degrees; latent feature distance does not. Coordinate error, the angle trajectory over time, sensitivity to changes in observation conditions, and failures are reported beside the primary result.

No new rendering, pose extraction, GAVD experiment, clinical annotation or confirmation-set evaluation belongs to this follow-up. Population counts and exclusions come from the bound parent manifest and admission receipts. A completed development comparison would support a finding within that observation procedure; broader real-video or clinical claims require independently referenced data.
