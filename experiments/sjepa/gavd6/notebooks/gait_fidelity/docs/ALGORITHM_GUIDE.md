# Reading the gait-fidelity algorithm

Start with [00](../00_start_here.ipynb), then work through [01–04](../01_data_and_references.ipynb) for the data, model and core training calculations. Continue to [05](../05_evaluate_and_visualize.ipynb) and [06](../06_verify_and_write.ipynb) for scoring and uncertainty, then read [F](../experiments/F_jepa_response.ipynb) for the response-pretraining extension and [G](../experiments/G_readout_repair.ipynb) for readout repair. Keep [07](../07_completed_study_walkthrough.ipynb) alongside this sequence to connect each operation to the completed results.

The directory contains fourteen algorithm/workflow notebooks (00–06 and A–G), plus the completed-evidence walkthrough 07: fifteen notebooks total. Their Python cells contain the worked computations. The `scripts/lessons/lesson_*.py` files generate those visible cells; they are not runtime wrappers hiding the algorithm. Standard attention, automatic differentiation and optimizer primitives remain library operations. Return to the [tutorial index](../README.md) for the folder layout and commands.

## Locate the computation and its check

| Scientific operation | Visible implementation | How the notebook checks it |
| --- | --- | --- |
| Population and paired data | **01:** count people, raw motions, windows and condition records separately; build the training endpoint table; show repeated baselines and no-change pairs. | Split assertions at all identity levels, held-condition exclusion, and exact ordering against `training.paired_indices`. |
| Projection and naming | **01:** camera-coordinate projection and pixel conversion; left/right permutation of coordinates, confidence and availability. References stay anatomically named. | Projection against `project_points`; all naming conditions against `apply_naming`; double-swap and duplicate-baseline checks. |
| Input-only normalization | **01:** context median origin, quantile-span isotropic scale, fixed fallback, and inverse transform to pixels. | Exact agreement with `normalize_batch`, hidden-coordinate perturbation invariance, empty-context fallback and pixel round trip. |
| Masks and tokens | **02–03:** exact-budget NumPy mask samplers; five channels `[x, y, confidence, availability, time]`; explicit reshape/transpose of four frames per joint. | Mask bits against every production policy, sparse-input cases, token ordering and intermediate model tensors. |
| Encoder and prediction | **03:** linear token embedding, joint/time positions, one attention block, predictor queries, missing-context handling, readout layers and residual skip. | Intermediate outputs and full predicted coordinates against the shared model; missing-context and supported-query checks. |
| Sampling and core training | **04:** four-level pair sampling; coordinate and centered feature-prediction losses; representation regularization; paired angle loss; optimizer, clipping, teacher and center updates. | Seeded sampler parity; values and relevant gradients; explicit teacher/center update equations and frozen-encoder checks. |
| Feature-response pretraining | **F:** paired token support, centered temperature-scaled residuals, delta/endpoint losses, coordinate residual control, combined forward loss and gradient calibration. | Values, gradients, unsupported-pair behavior and combined losses against response source functions; optional retained-result inspection. |
| Frozen-readout repair | **G:** angle geometry, linear percentiles, common support, equal-pair scalar/dense reductions, geometry penalty, calibration and two matched optimizer updates. | Loss values and coordinate gradients, collapsed-limb/unsupported cases, readout-gradient energies, complete model states, and six saved calibration coefficients when available. |
| Scores and aggregation | **05–06:** coordinate distance, displacement, knee waveform/excursion, response and nuisance contrasts; explicit failure accounting; condition→window→motion→person aggregation. | Saved example rows, source evaluators, aggregate CSVs and primary comparisons. Failed predictions retain their declared costs. |
| Diagnostics and uncertainty | **05–06:** person-fold ridge regression, training-fold standardization, dual/primal solves, paired crossed bootstrap, and repair's person-level t interval. | Fold identities, every probe prediction, equal-person errors, interval endpoints and rejection of incomplete pairing. |

Source-parity checks establish that an explanation computes the implemented quantity. They do not establish that a model has converged or that a scientific conclusion generalizes.

## Follow the quantities across stages

An input batch begins as coordinates of shape `[B,T,12,2]`, accompanied by scores, observed flags and timestamps. After artificial pretraining queries are excluded, normalization uses only the retained observations:

\[
o=\operatorname{median}(X_C),\quad
s=\|Q_{.95}(X_C)-Q_{.05}(X_C)\|_2,\quad
\widetilde X=(X-o)/s.
\]

The source model has 128 frames, four-frame patches and width 96. Packing five channels yields `[B,384,20]` patch vectors; the encoder produces `[B,384,96]` features. The coordinate readout unpacks two coordinates per frame and adds corrections to usable input positions. Its output returns to pixels through `prediction * s + o`.

During JEPA pretraining, the student sees masked estimated poses and the moving-average teacher sees projected training references in the same input-derived coordinate system. The feature loss compares distributions over learned feature channels, not diagnosis classes. The teacher and references supply training targets. Deployment uses the observed-input encoder and coordinate readout with no artificial query mask; no teacher, person ID, intervention label, reference mask or evaluation scale enters its input dictionary.

For knee angles, excursion is `P95 − P5`; signed asymmetry is right excursion minus left excursion. A response compares this whole-window measurement between paired movement states. Dense repair compares angle differences at corresponding frames and legs between those states. Neither operation forecasts the future or estimates a temporal derivative.

Reference support is fixed before examining predictions. Paired angular training uses frames supported at both endpoints and both legs. Evaluation intersects reference support across the declared variants of a source family. Short or nonfinite predicted limbs cannot improve scores by deleting difficult reference frames.

## Keep the completed experiments distinct

| Experiment | Completed design | What the extension changes |
| --- | --- | --- |
| Walking core | 10 recipes × 3 seeds = **30 final fits**; 9 shared pretraining fits | Representation family and base versus original paired-change output loss. |
| JEPA response | 3 pretraining variants × 2 output objectives × 3 seeds = **18 new final fits**; 9 new pretraining fits | Coupling between paired feature residuals during pretraining; F derives the matched endpoint control. |
| Readout repair | 2 retained encoders × 2 new objectives × 3 seeds = **12 new readouts**; 6 calibrations, no new pretraining | Scalar-loss strength or dense paired angle supervision; G holds the encoder fixed. |

The full implemented matrix has 102 final fits and 33 shared pretraining fits. Tutorials B and E explain controls outside the completed core. Their calculations do not imply those source experiments ran. E's endpoint **re-pairing** differs from G's **readout repair**.

The completed source studies use `person_motion`: choose a person, then one of that person's raw motions, then a window, then a condition/intervention pair. This shares exposure across objectives while preventing participants with more footage from dominating. The optional re-pairing protocol instead uses complete permutation cycles to preserve endpoint exposure.

F and G have different calibration targets. F scales the larger initial JEPA auxiliary gradient to a declared fraction of the base-pretraining gradient. G sets the weighted dense-readout gradient equal to the **weighted low-scalar gradient**, separately for each encoder and seed:

\[
\lambda_s=0.1w,\qquad
\lambda_d=\lambda_s\sqrt{\frac{\sum_b\|\nabla L_s^{(b)}\|^2}
                                      {\sum_b\|\nabla L_d^{(b)}\|^2}}.
\]

Both new repair arms retain coordinate supervision and the original geometry coefficient. Initialization matching does not guarantee matching later in training, nor does it isolate gradient sparsity as a causal mechanism.

## Read results within their population

The three completed experiments reuse fourteen development people, with three fitted seeds. Core and response primary summaries pool the three estimators; repair's declared primary compares delta-JEPA dense versus low-scalar **ViTPose waveform error**. Its t interval first averages the three seed effects within each person. The additional crossed bootstrap resamples people and seeds separately while keeping methods paired.

The local `outputs/gait-fidelity` source report describes walking core. `outputs/iclr` also contains later response and repair evidence, without the raw trajectories or checkpoints. Notebook 07 verifies that packet's hashes. Small constructed arrays and CPU fixtures demonstrate calculations; their outputs are not source-study measurements. Protected confirmation and completed GAVD evaluation are absent from this evidence, and the probe's held folds remain within the encoder's training population.

See [ENVIRONMENT.md](ENVIRONMENT.md) for execution and [VISUAL_WALKTHROUGH.md](VISUAL_WALKTHROUGH.md) for the figure sequence. A successful local execution receipt validates software behavior at its stated scope; HAIC rendering, extraction and GPU fitting require their own retained checks.
