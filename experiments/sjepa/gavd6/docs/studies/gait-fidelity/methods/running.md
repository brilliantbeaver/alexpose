# Running and understanding the implemented experiments

[Proposal](../README.md) · [HAIC commands](../../../../slurm/gait-fidelity/README.md) · [Notebook tutorials](../../../../notebooks/gait_fidelity/README.md) · [Compute plan](execution.md)

The implementation turns the proposed comparisons into a shared data preparation stage, a dependency graph of training phases and a common evaluation. The HAIC guide gives the commands; the notebooks expose the saved configuration, references, masks and tables so that you can inspect what each command does. Completed implementation and software checks do not constitute a completed source experiment.

## Begin with inspectable data

Prepare the source once, then use the same admitted bundle across every method. The available HAIC AMASS data, body models, rendering assets and pose-estimator checkpoints are reused from synthetic training v2. The setup discovers recorded asset paths instead of requiring another manually assembled preparation JSON. Optional external evaluation datasets retain the availability status in the [data inventory](../data/availability.md).

Each record connects an observed trajectory to its projected reference, source person, physical movement state, camera, naming condition and observation condition. Image overlays and synchronized trajectories make those connections inspectable. The `--prepare-only` launch stops before fitting so you can review the resulting viewer and validation records. A machine-generated screen remains labelled automated; the implementation does not manufacture a reviewer or certify clinical references.

The new 128-frame intervals require a fresh overlap check. At 25 Hz, the original starts at five and eight seconds would share source motion. Preparation uses a deterministic nonoverlap rule within the same frozen recordings, selecting starts such as five and 10.12 seconds, and retains the parent audit alongside new subinterval screening. These longer intervals still require visual reference review; inheritance of a shorter-window audit does not certify the entire new interval.

Mirroring checks transformation consistency. Global and temporary left–right label swaps test controlled assignment errors. Occlusion and camera variation test observation sensitivity. Reference-verified movement levels establish whether restoration preserves a difference between physical states. These interventions have different purposes and remain separate in the saved metadata and evaluation.

## Understand the five experiment groups

All groups use seeds 17, 29 and 43. A final model is a recipe/seed output; repeated frames, render variants and model seeds do not increase the number of independent people.

| Group | Exact comparison | Final models | Interpretation |
| --- | --- | ---: | --- |
| Masking and movement supervision | Coordinate-pretrained or paired-JEPA encoder; time-block, uniform-token or graph-time mask; base or paired-change output objective | 36 | Tests how missing-information structure and movement supervision interact. |
| Mask structure controls | The same encoder/output comparisons under shuffled anatomical connections or random-joint intervals | 24 | Tests whether graph connectivity adds value beyond duration and visibility. |
| Practical benchmarks | Direct coordinate training, direct plus paired change, static model and learned temporal refiner | 12 | Establishes practical restoration performance against simpler training choices. |
| Pretraining information | Initialized or shuffled-reference JEPA encoder, each with base or paired-change output objective | 12 | Tests whether useful pretraining and correct input/reference alignment explain a gain. |
| Additional supervision | Coordinate-pretrained, paired-JEPA or direct model with per-example measurement labels or valid re-paired change labels | 18 | Tests whether meaningful pairing adds value beyond additional supervision. |
| Total | 34 configurations across three seeds | **102** | All planned controls run before a scientific conclusion is selected. |

Coordinate pretraining predicts reference joint coordinates at queried positions. Paired JEPA instead predicts numerical features extracted from the matching reference poses. Their encoders are then fixed while independently initialized output networks learn coordinate corrections. Direct training updates the complete network end to end. This distinction matters because comparing a frozen encoder with a fully trainable network changes more than the representation objective.

Identical pretraining is shared across output objectives when its full specification agrees, producing 33 pretraining phases, 84 frozen-encoder output phases and 18 end-to-end phases. Each phase records the source and upstream checkpoint identities needed to reproduce that sharing. Initialized encoders have no pretraining phase. Readout training uses ordinary observed inputs for all arms.

## Compare movement objectives without changing data exposure

The base objective supervises joint coordinates. The paired-change term penalizes a disagreement between the predicted difference across movement states and the corresponding reference difference. The primary measurement is the right-minus-left difference in projected knee excursion, where excursion is the 95th minus 5th percentile of the image-plane hip–knee–ankle angle on admitted time samples.

Every regime receives both endpoints with the same coordinate supervision. The per-example control also supervises the measurement at each endpoint separately. The re-pairing control rearranges partnerships across training source families under the declared strata and recomputes the correct reference differences. Retaining labels from the original pairing would corrupt the experiment. A reference-change distribution mismatch is reported and bounds the interpretation of pairing-specific claims.

Matching exposure applies to each update as well as the full training population. The training-reference-only re-pairing permutation is built from cycles of two or three source families, and each batch contains complete cycles. This gives every compared objective the same multiset of endpoints; re-pairing changes their partnerships. The nominal endpoint batch size is 16, with some three-cycle packings producing 18 or 20 endpoints. Receipts retain the actual counts, so the paired objective cannot gain an unreported increase in examples.

Training coefficients and budgets belong to the saved configuration. The source starting schedule uses 2,000 updates for pretraining and 2,000 for output training, or 4,000 for end-to-end training. Before final fits, the coordinator profiles representative phases on an allocated H100 and retains the measured timing. It chooses the full common schedule or the predefined half-budget schedule from cost feasibility before outcomes are ranked, and stops if neither fits. Profiling is charged to the study, while every configuration and seed remains in the selected matrix.

## Follow the notebooks alongside Slurm

Notebooks 00–03 establish the environment, prepare and inspect data, exercise the actual mask sampler and display every recipe/dependency. Five additional tutorials in `notebooks/gait_fidelity/experiments` explain each experiment group and expose its exact recipes, checkpoints, histories and result rows. Notebook 04 submits or resumes the source study through Slurm. Notebook 05 reads the saved per-window, per-person and movement-response tables, and notebook 06 reconstructs artifacts before manuscript reporting.

The notebooks default to a deliberately small CPU fixture. It demonstrates execution using generated tracks and is labelled as a software check throughout. Selecting an initialized HAIC work directory changes them to the source workflow. Source GPU work always runs inside Slurm allocations, while data inspection and completed-output analysis can run on CPU.

The isolated source release protects a running study from routine edits to the working checkout. Each worker has a unique attempt directory, and the coordinator enforces a combined concurrency of eight and a shared allocation budget. Resuming retains completed phases and the costs of failed attempts. The run's configuration, source and shared bundle remain fixed once execution begins.

## Interpret the completed study at the supported level

Evaluate position error alongside changes over time, signed knee differences, intervention-response errors and coverage. Use the same reference-supported samples in paired measurement cells, and keep failures visible. The person-level table is the basis for population uncertainty; event or track counts are descriptive support, not additional participants.

The existing eight inspected development people support development conclusions. A confirmation claim requires a separately admitted population and a primary comparison frozen before its outcomes are examined. Additional datasets or clinical annotations may broaden that claim later, but extra H100 capacity alone does not create independent references or statistical significance.
