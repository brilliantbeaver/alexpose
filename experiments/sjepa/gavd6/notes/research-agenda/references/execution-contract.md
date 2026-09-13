# Evidence and one-week execution contract

These seven proposals are alternatives. The objective is to identify and complete one strong study with a first result in two or three days. No training has been run for this portfolio. Proposed thresholds, sample sizes and GPU budgets are planning choices, not experimental findings or calibrated probabilities of publication.

## A short reading guide

| Term | Plain meaning here |
| --- | --- |
| World model | A model of how observations or states can change. A plausible generated video alone does not prove that the model understands causes. |
| Motion prior | A pretrained model's learned preference for some movements over others. Preference is not evidence that a particular person moved that way. |
| Skeleton and joint | A skeleton is a time series of body landmarks. A joint is one landmark, such as a knee or wrist. |
| Latent representation | An internal list of numbers that summarizes an input. Its useful contents must be tested, not assumed. |
| S-JEPA | Skeleton Joint-Embedding Predictive Architecture. It learns by predicting hidden skeleton representations from observed context. |
| Frozen backbone and adapter | Keep the large pretrained model's weights fixed; train a small additional component to change how it is used. |
| Teacher and student | A richer model supplies training targets to a smaller model. The deployed student may have less information. |
| Residual | What remains after subtracting a reference prediction. The choice of reference determines what is removed. |
| Nuisance | A recording property, such as camera motion, that can influence a score without answering the intended movement question. |
| Held-out group | A person, recording, event family or other group excluded from the relevant fitting and selection. Its precise role matters. |
| Calibration | Use separate development examples to choose thresholds or adjust predicted probabilities. Calibration on synthetic data need not hold on real videos. |
| Tradeoff or frontier | The best attainable combinations of two goals, such as noise removal and true movement retained. |
| R² | Improvement in squared prediction error relative to predicting a reference mean. It is not a probability that the model is correct. |
| Bootstrap interval | Repeatedly resample independent groups to estimate sampling uncertainty. It does not repair leakage or automatically include training uncertainty. |

Each proposal starts with the concrete question and example, then gives the implementation and the comparison that could disprove it. Read its first two sections before the technical details.

## What we know from this repository

| Evidence | Verified observation | Consequence for new experiments |
| --- | --- | --- |
| Latest source learning curve | 1,403 windows, 290 recordings; largest outer training sets contain 231 to 233 recordings | There is a materially larger development cohort, but windows and sources increase together. |
| RGB-conditioned endpoint | RGB R² 0.662830; real skeleton increment 0.0003571; conditional bootstrap interval [0.0000414, 0.0006697] | A small positive increment exists in the saved output. It is about 140 times below the frozen 0.05 target. |
| Control endpoint | Mismatched skeleton increment 0.0006029 | A positive increment by itself cannot establish useful aligned motion transfer. |
| Frozen outcome | `development_stop` | Do not replace a failed prospective gate with a favorable retrospective story. |
| Prior laterality study | Common expanded readout: random initialization R² 0.222544; five pretrained teachers approximately 0.1008 to 0.1142 | Learned representations can lose access to a motion observable. This does not prove that generative priors erase real events. |
| GAVD manifests | 1,874 annotated sequences, 348 recordings; 347 recordings contain one `gait_pat` value | Label and recording context are strongly entangled. Source holdout is necessary and insufficient for proving motion dependence. |
| Independent people | Local GAVD participant identity is unknown | Never equate recordings or sequences with people. |

The sources are the [run analysis](../../../notebook_runs/haic-run-02/ANALYSIS.md), [ICLR strategy](../../../docs/studies/future-feature-prediction/scaling/research-strategy.md), [manifest files](../../../manifests/gavd/), and [independent audit](reviews/evidence-audit.md). The saved notebook is available locally; raw HAIC caches and checkpoints were not re-executed during this ideation review. Conditional bootstrap intervals do not include training or target-selection uncertainty. The all-source endpoint was fitted once per outer fold, not three independent times.

The earlier small student-input panel also has a confidence-route problem: duplicate confidence information could alter effective regularization. A new panel must place common quality features in the reference once and vary coordinates separately. This is a design correction, not a revised historical result.

## Data roles and the whole-body bridge

**AMASS supplies motion reference trajectories.** Use the user's existing HAIC archive. Extract 22 body joints from the compatible SMPL-family body model, including spine, shoulders, elbows and wrists. Retain metric scale and physical timestamps in the scientific data store. Convert into the selected checkpoint's exact joint ordering, normalization and sampling rate only at its boundary. Fingers, contact forces and pathological labels are not implied by this representation. Core11 is an equal-capacity input ablation, not the default ceiling on information.

AMASS is motion capture fitted to a body model, not error-free physical truth. It supplies an independent reference for **new observation corruptions**, which is much stronger than comparing a denoiser with its own outputs. For controlled rendering, use simple deterministic body meshes and multiple held-out camera profiles first. Expensive photorealistic video generation is unnecessary for the first mechanism test. Audit any overlap between pretrained model training data and evaluated AMASS sources; a held adaptation participant may still have appeared in foundation pretraining. Report this distinction.

**GAVD supplies real-world video and, when needed, presentation classification.** It supplies no precise 3D trajectory, dense affected-side labels, severity progression or treatment counterfactuals. A classification supplement uses multiclass `gait_pat`, with classes and source eligibility frozen before fitting. Exclude classes with too few independent recordings for the declared split and report each exclusion. Do not make binary normal-versus-abnormal accuracy the headline.

The prior learning-curve study reserved 43 GAVD confirmation recordings. Its latest availability report implies 41 have media, with eligibility unknown. Keep this confirmation set sealed during proposal selection and threshold tuning. The user reports full videos on HAIC; the earlier run's 334-video availability reflects that extraction snapshot, not a fresh audit of all scratch files today.

**CARE-PD is optional corroboration.** Its public meshes and identity metadata support inexpensive external checks. Some trajectories are themselves model reconstructions. They cannot prove preservation of clinical truth or exact 3D accuracy. The published participant totals and license text differ across release surfaces, so count actual eligible records and preserve provenance. No primary proposal depends on downloading a new clinical cohort or obtaining a restricted checkpoint.

## Controls that protect the headline

The unit of inference is the independent person where identity is known, otherwise the original recording or motion trial. Split before generating windows, rendered views, edits, neighbors or teacher targets. Keep all derived versions together. Group AMASS identities by source corpus and subject, and check known cross-corpus duplicates where metadata permits. Uncertain identity linkage limits the claim.

Every learned comparison includes raw coordinates, a suitable strong classical baseline, equal-capacity heads and a random-encoder or random-feature placebo. Whenever a model uses RGB, the baselines get the same pixels, detector passes and allowed future visibility. Two pose trackers are not independent sensors simply because their names differ.

Nuisance features include duration, timestamps, centroid drift, box and foreground area, camera motion, image size, static background, missingness and confidence. Motion summaries include cadence, scale, speed, phase and individual-joint spectra when relevant. Use matched pairs or stratified evaluation, not only a weak nuisance classifier. Never remove a genuine part of the question by conditioning on the answer itself.

Synthetic edit families are development interventions on a renderer or kinematic model. They are not disease simulations or patient counterfactuals. Hold out at least one event family and one observation process. Check smooth boundaries and matched support. A method that recognizes the corruption generator has not learned the intended mechanism.

The 48-hour gates use development groups. A family, camera or activity used to choose the flagship is no longer a final untouched test, even if excluded from pilot training. Reserve final groups before the pilots. For a claim about an unseen **family**, reserve an additional family or use a fully nested family split. In P1, use arm-leg timing and foot-clearance edits for development, then reserve a separately constructed trunk-pelvis timing event family for the final test. Natural-motion confirmation is separate again. If the budget cannot support this separation, narrow the claim instead of calling a revisited gate independent evidence.

Three optimization seeds estimate training sensitivity for the final comparison. Group bootstrap estimates sampling sensitivity. Report both, with paired starts where possible. Three seeds, many overlapping windows or 2,000 bootstrap draws do not create more independent people. Do not claim a power guarantee from a nominal window count; use the pilot's group-level variance to judge whether the final effect can be resolved.

For every forecasting study, the information boundary includes preprocessing. HumanML3D velocities and contact channels, floor estimation, heading, resampling and interpolation must not inspect the query future. Compute boundary quantities from permitted past samples or mark them unavailable. A cached feature produced from a full trajectory is not prefix-safe simply because its rows are later sliced. Replace the withheld suffix and require the complete model-visible prefix tensor to remain unchanged, including metadata and masks. Fix stochastic forecast aggregation before evaluation, preferably the sample mean for squared-error endpoints, and never select the sample nearest the test future. Retrospective restoration may use the declared whole observed clip; label that distinction explicitly.

## Compute and scheduling

Eight H100s for seven continuous days provide a theoretical ceiling of **1,344 GPU-hours**. Assume 80 GB devices only for planning and verify actual memory. Storage, model loading, rendering and CPU decoding can consume calendar time even when GPUs are idle. All individual proposal budgets are caps for choosing that alternative, not quantities to add together. Count both initial pilots and any abandoned branch in the overall resource ledger, even though only one full study is selected.

| Time | Work | Decision |
| --- | --- | --- |
| First 6 hours | Verify exact model binary, one forward pass, 22-joint conversion, eligible groups and a 128-window timing sample | A published checkpoint link is not proof of a working local pipeline. Use the verified fallback or stop that branch. |
| Day 1 | Run Proposal 1's prior-erasure assay and Proposal 7's joint-input/flow assay in parallel; count Proposal 6's eligible people on CPU | Establish that the proposed problem and usable signal exist before adapting a model. |
| Day 2 | Test the strongest cheap explanation for each signal; optionally run Proposal 2's geometry-only witness assay | Choose one flagship using development groups only. Keep no more than two small pilots active. |
| Days 3 to 5 | Run the selected adaptation and matched baselines, capped at the proposal budget | Require the method to beat the best simple explanation. |
| Day 6 | Held event, held activity, held observation process or held teacher family, as appropriate | Test the actual generalization claim. |
| Day 7 | Paired seed repeats, uncertainty, failure analysis, figures and result package | Freeze the result even if it is negative. Do not spend the final day searching new hypotheses. |

Reserve roughly 25% of the wall-clock window for failed runs, conversion issues and final verification. Do not reproduce the full Wan adaptation recipe in parallel with the flagship. Masked Visual Actions used 15 hours of **robot data**, but its reported training was four days on eight H200s with rank-256 LoRA. That is a valuable design inspiration and a poor assumption of cheap one-week gait adaptation. See the [world-model reading memo](reviews/world-models.md).

## What would justify an ICLR submission

A promising pilot needs one clear effect. A compelling paper needs the effect, an explanation that survives strong alternatives, and a new capability demonstrated beyond the examples used to design it. For this portfolio, require at least two materially different stress settings, a useful margin over the best baseline, a generalization axis that was genuinely held out, and an honest accounting of public checkpoint/data overlap.

The most plausible flagship is evidence-preserving restoration. The optical-flow state proposal is the second leading assay, sharing infrastructure with the first. Cross-activity motor memory is the independent fallback. The earlier distillation continuation is cheap to audit but now has a much narrower novelty opening. None is honestly a high-confidence ICLR acceptance forecast today. The staged plan increases the chance of spending the week on a real effect rather than making the initial probability look impressive.
