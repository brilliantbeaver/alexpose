# Registered tutorial protocol, version medwm-gait-v1

This protocol defines a new study. Historical ASTAD and multiple-sclerosis results remain separate. The code and notebooks implement a research prototype; a working smoke test does not establish clinical utility.

## Question and estimands

Can a compact model trained to predict normal skeletal motion improve detection of unfamiliar gait patterns compared with a conventional feature classifier under the same source-video partitions? A second, explicitly supervised question is whether adapted representations separate the five GAVD labels: normal, parkinsons, stroke, cerebral palsy, and myopathic. The repeated stroke entry in the brief denotes one class. Myopathic gait is a neuromuscular category and should not be described as a central neurological diagnosis.

The primary anomaly detector uses a normal-only encoder and predictor. Freeze this reference before adding condition-labeled training data. Use normal validation sources to set an operating threshold. Abnormal training examples belong only to a separate adapted representation and supervised classifier. Keeping these identities separate makes the meaning of a normality score stable.

The primary comparison is source-aggregated balanced accuracy for the binary normal-versus-selected-abnormal task, with source bootstrap uncertainty. Also report AUROC, average precision with prevalence, sensitivity and specificity at the frozen validation threshold, macro F1, per-class recall, and a five-class confusion matrix. The five-class task is secondary. Aggregate windows to sequences and sequences to sources so long clips cannot dominate. Report support at all three levels. Compare RF and JEPA on exactly the intersection of eligible observations, never separately optimized cohorts.

## Partition and processing order

Read the current transformed sequence CSVs and record hashes. Exclude upstream source CSVs from counts. Apply recorded media mismatch exclusions before modeling. Split by original YouTube ID before fitting transforms, deriving windows, pretraining, tuning, or adapting. Sources shared across labels stay together. Related uploads or known participants require a stronger grouping before claiming participant generalization. A YouTube ID alone cannot establish participant independence.

Use a fixed train/validation/test partition for tutorials. Before a paper, repeat a predeclared grouped evaluation with multiple training seeds and report the variation separately from within-run source bootstrap intervals. Never select seeds or model variants based on the final test. If a class loses test sources during pose filtering, report the endpoint unavailable rather than changing the split silently.

All pose input is BlazePose-33. The selected joint table is the only masking vocabulary. Use image-plane x and y coordinates, retain confidence and missingness, and compute normalization using observed context only. These coordinates do not measure physical speed, ground contact, muscle activity, or 3D anatomy. Overlapping windows remain within their source partition. Missing joints are not training targets.

## Models and controlled extensions

The baseline is an S-JEPA-inspired compact transformer with a student context encoder, EMA target encoder and masked-token predictor. Uniform random joint-time masks follow the user's requirement. Published MAMP and S-JEPA use motion-aware masking; the uniform policy is a declared change. The loss and target construction are explained in the literature review and model source. VICReg variance and covariance penalties are anti-collapse regularizers. They do not force health conditions apart. Supervised classifiers supply label discrimination.

Train normal-only masked completion first, then train explicit past-to-future prediction with a suffix mask. Bidirectional completion error is not a future forecast. A coordinate prediction head makes motion predictions inspectable; it is an auxiliary extension, not evidence that the original latent-only objective predicts calibrated physical trajectories.

Add parkinsons, stroke, cerebral palsy and myopathic examples from training sources in declared stages, retaining normal replay and copying the initial checkpoint. Keep a stage record. Evaluate any curriculum claim against a single mixed-data adaptation with the same update budget and same data exposure. Class order is a design choice, not a biological progression.

Stochastic position jitter and conditional latent variables are separate optional experiments. Compare four cells: neither, positions only, latent variables only, both. Keep masks, sources, dimensions, updates and seeds matched. Positional jitter does not create a distribution over outcomes. Conditional latent samples can express diverse predictions only if the learned conditional distribution uses the noise. At inference the latent prior may see context only. Measure sample diversity, error versus uncertainty, interval coverage and sample-count sensitivity on validation before making uncertainty claims. A stochastic extension with an unused latent is a negative result.

## Paper evidence gate

A full empirical claim needs an audited eligible cohort, complete grouped comparisons, repeated seeds, confidence intervals, ablations, representative failure examples, and data provenance. Add camera-view and source-style sensitivity checks. GAVD supports an exploratory observational benchmark; its web-video selection, small source counts, unknown participant overlap and annotation labels do not establish clinical diagnosis or deployment generalization. External patient-level evaluation is required for that claim.

All synthetic demonstrations are labeled synthetic. Training duration and exclusions belong in each run record. Incomplete or failed runs remain visible. No result may be copied from an older experiment and relabeled as this method.
