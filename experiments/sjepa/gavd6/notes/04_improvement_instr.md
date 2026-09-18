# Implementation prompt: a stronger synthetic-training and temporal S-JEPA study

You are the lead research engineer and a skeptical AI/ML scientist specializing in joint-embedding predictive architectures, video learning, and gait measurement. Work in this repository to implement a new, scientifically controlled notebook suite from `notes/prompts/03_improvement_plan.md`.

Use the strongest available reasoning model appropriate to the host, including the user's suggested frontier models such as Anthropic Fable 5 or GPT 6 Astra **if those models are actually available**. Resolve installed model/tool identifiers rather than inventing them. Think carefully, compare competing explanations, and check your conclusions. Output evidence, decisions, calculations, and concise rationales; do not expose private chain-of-thought.

Your objective is a reproducible experiment that can establish whether paired synthetic motion improves pose measurement while preserving useful gait dynamics, and whether temporal JEPA adds value beyond ordinary denoising. Do not optimize for a positive result, a fashionable method name, or a larger collection of notebooks. A well-supported negative result is valid.

## 1. Scope and nonnegotiable constraints

Read the improvement plan in full before editing. Inspect applicable repository instructions and the current worktree. Preserve unrelated changes, historical executed notebooks, frozen results, manuscript claims, protected evaluation groups, and all previously recorded STOP decisions.

Implement additively in a new study namespace, provisionally:

- `src/gavd6_sjepa/research_directions/synthetic_training_v2/`
- `scripts/research_directions/synthetic_training_v2/`
- `notebooks/synthetic_training_v2/`
- `tests/synthetic_training_v2/`
- `slurm/synthetic-training-v2/`
- `docs/studies/synthetic-training-v2/`
- `outputs/synthetic-training-v2/<unique-run-id>/`

Check existing layout/registration conventions before adopting these paths. Reuse well-defined existing modules through explicit interfaces. Do not copy entire pipelines or refactor unrelated studies. If a shared change is necessary, preserve old behavior and test the relevant contract. Never silently change a historical configuration to make a new result look like a continuation.

**HAIC must use Torch 2.6.0+cu124.** Preserve the working Torchvision 0.21.0+cu124 and compatible MMCV configuration. Do not use another Torch version, run an unqualified root-project `uv sync`, or copy installation commands from papers that replace this environment. Optional dependencies must fit the required stack or be deferred with a documented reason.

The established HAIC checkout is `/hai/scratch/tedmui/alexpose/experiments/sjepa/gavd6`, and the established interpreter is `/hai/scratch/tedmui/envs/synthetic-training-cu124/bin/python`. Verify configured paths rather than assuming they remain valid. The `hai` partition previously rejected interactive allocation; use the existing Slurm batch conventions. A guessed `CUDA_HOME` path is not a toolkit installation. Verify the actual interpreter, CUDA operators, and allocation before claiming GPU compatibility.

This prompt authorizes implementation, local analysis, fixture validation, and preparation of concrete HAIC batch commands. Execute real GPU stages only within the user's available access and explicitly configured run/budget scope. Never fabricate HAIC access, scheduler results, human annotations, licensed assets, or completed experiments. If access or essential data are missing, complete independent implementation work and produce precise commands and a clearly pending empirical stage.

## 2. Use dynamic, bounded subagent workflows

If running in Claude Code, fan out independent research and implementation tasks using its available subagent mechanism. If running in Codex, use its supported collaboration tools. If an `ultracode` workflow is available, inspect its actual instructions and use it where relevant; if absent, state that once and use native subagents. Never invent tool calls or claim an unavailable workflow ran.

Use up to the environment's available concurrency, with a coordinator plus three workers as a sensible starting point:

1. **Implementation auditor:** inspect data, rendering, training, feature extraction, JEPA objectives, schemas, time contracts, and leakage boundaries.
2. **Results/statistics auditor:** independently reconstruct primary results and oracle opportunity from saved artifacts; inspect dependence, uncertainty, and exposure history.
3. **Literature challenger:** find the closest primary research, verify current artifacts and publication status, and challenge the novelty claim.

Give each worker a bounded question, exact inputs, expected output, and ownership restrictions. Initially all workers are read-only. Require file/line or artifact evidence and clear distinctions among verified facts, hypotheses, and missing information. Do not ask three agents to produce interchangeable generic plans.

After their reports, reassign workers dynamically to disjoint implementation areas: data/contracts, models/training, and notebooks/reporting. The coordinator owns interfaces and integration. Resolve shared-file ownership before edits. Dependencies determine the workflow: data contracts precede training; saved predictions precede claims; a failed scientific gate cancels dependent expansion. Integrate small changes and inspect every worker result.

Use **Codex for independent adversarial review** at the protocol, implementation, and results stages. Prefer a separate Codex session or agent with read-only access and the actual files/artifacts. When the coordinator is Codex, use a separate reviewer context; do not review your own summary and call it independent. Use only documented installed interfaces. If Codex is unavailable, record that limitation and use a separate available reviewer without falsely labeling it Codex.

## 3. Audit before designing changes

Read at least:

- `notebooks/synthetic_training/README.md` and its notebook builder.
- `src/gavd6_sjepa/research_directions/synthetic_training/`, especially `data.py`, `rendering.py`, `estimators.py`, `context_features.py`, `measurements.py`, `trials.py`, `selectors.py`, and `workflow.py`.
- `notebook_runs/synthetic-training/run-01-v2/`, `run-02-v1/`, `run-03-v1/selectors/`, and `run-07-v1/`.
- `docs/studies/temporal-gait/{protocol,development,results}/` and `src/gavd6_sjepa/research_directions/temporal_gait/`.
- `docs/studies/future-feature-prediction/gate/results.md`, `accessibility/results.md`, and relevant provenance/split records.
- The existing HAIC runbook and study-specific dependency manifest.

Reconstruct the pilot table from CSVs, not notebook previews. Expected checkpoints to verify include 2,304 source rows, 6,912 selector decisions, and frozen 75-update mean errors of approximately 0.02702648 for full, 0.02702576 for matched source-progress, 0.02678074 for scene/domain, and 0.02689916 for fixed `front`. Check units, aggregation, duplicate keys, missing observations, matched controls, and selected hyperparameters. Investigate discrepancies instead of forcing these values.

Recompute the retrospective shared-scene and student-specific oracles. The existing validation panel showed only about 0.0406% additional relative error reduction from student-specific choices. Do not confuse that diagnostic with an achieved method, population bound, or proof that all personalization is useless.

Verify these structural concerns against current code:

- Synthetic training uses a frozen video descriptor; it does not train S-JEPA.
- Current response summaries discard temporal order.
- Lessons change underlying motions as well as rendering conditions.
- Existing AMASS eligibility is not a walking annotation.
- Small head-only updates and architecture-dependent learning behavior limit inference.
- Current visible 2D landmark metrics cannot support clinical, 3D, or real hidden-joint claims.
- Previously examined validation artifacts are development evidence for this redesign.

Do not attribute failure to feature scaling, weak gradients, teacher collapse, or insufficient data without measurement. If cached features/checkpoints are absent, identify exactly which explanation cannot be tested. Preserve successful branch-reset, replay-cost, target-isolation, and frozen-artifact controls.

## 4. Research authoritative literature and challenge novelty

Refresh the literature as of the actual execution date; do not assume a remembered paper list is current. Search arXiv, IEEE/CVF/IEEE conference or journal records, ACM proceedings, ECVA, NeurIPS, AAAI, MICCAI, and author-linked repositories. Use search engines for discovery and primary papers/publisher/author sources for claims. Blogs and model-generated summaries are not evidence of technical results.

Start from S-JEPA, V-JEPA/V-JEPA 2/2.1, LeJEPA, seq-JEPA, GaitForeMer, FSGait, GaitPT, skeleton SSL scaling, H-MoRe, SM-SGE, and the emerging GaitJEPA. Check CARE-PD, GAITGen, DiffuseGaitNet, Task2Sim, PoseExaminer, PoseSyn, and adaptive-curriculum work for direct novelty collisions. Check recent factorized video-JEPA and task-relevant-information studies if proposing those mechanisms.

Treat SmoothNet, PoseBERT, and DeciWatch as especially close restoration precedents. Include a source-trained SmoothNet-style temporal MLP as an inexpensive strong baseline, with documented adaptations to this 2D schema. Do not claim that estimator-independent correction or motion-capture-based masked pretraining is new. Do not require a 3D pretrained model to run on a 2D schema without a justified adapter and separate scope.

For the closest five to eight methods, record title, authors, DOI/arXiv ID, exact version/date, publication status, input modality, objective, supervision, datasets, split design, metric, actual supported claim, limitations, official code/checkpoints, license, and access status. Use compact entries for contextual references. Distinguish published proceedings from a preprint or an author announcement. Record inaccessible artifacts; do not assume that a paper's “code available” sentence means weights and data are downloadable.

Read methods and evaluation sections before borrowing an objective. Follow references and forward citations around the closest three methods. Search specifically for disconfirming precedents: paired synthetic pose denoising, preservation of gait dynamics, nuisance-invariant skeleton SSL, latent prediction versus coordinate prediction, and estimator-independent correction. Maintain a compact claim-to-source ledger.

Keep task distinctions explicit: gait identity recognition, pose measurement, anomaly detection, clinical severity, and motion forecasting are different problems. For example, GaitForeMer pretraining mixes forecasting with action labels; identity-oriented pace or mirroring augmentation may erase functional signals. CARE-PD offers released derived motion, not an assumed raw clinical RGB collection, and some related papers share its cohorts. Check these details again before use.

Select a narrow hypothesis after this review. Do not claim “first JEPA for gait,” new synthetic hard-example generation in general, or a clinical world model. Explain what new result would exceed the nearest existing work, what simpler explanation must be excluded, and what negative outcome would refute the proposed advantage.

## 5. Freeze the first experiment's question and information boundary

Implement the smallest informative core first:

> Does paired synthetic supervision improve restoration of imperfect 2D pose tracks across unseen motions and pose extractors without erasing motion timing or side-specific differences, and does latent JEPA pretraining improve that tradeoff beyond matched coordinate denoising?

This is offline restoration over a declared observed window, not forecasting. All restoration methods receive the same frames and latency. The first target is the simulator's **clean 2D projected body-12 motion in the same camera and at the same timestamps**. It is neither an error-free real annotation nor a canonical 3D skeleton. Document the approximate SMPL-H-to-COCO landmark convention.

Model inputs may contain only deployment-available estimated coordinates, estimator confidence, observed/missing flags, physical timestamps, and input-derived normalization. Exact renderer visibility, true errors, clean coordinates, intervention labels, and privileged camera parameters belong in separate training/evaluation records, not model inputs. Never equate estimator confidence with ground-truth visibility. Calibrate cross-estimator confidence only on permitted source data.

Record the source of person boxes. The existing synthetic foreground boxes are privileged. Use a consistent supplied-box protocol, test development box perturbations, and require detector/tracker-box evidence before claiming an automatic pipeline. Annotation boxes used for evaluation normalization must not enter inference unannounced. Extend track extraction to retain actual keypoint scores and missing detections: the current coordinate-only `MMPoseEstimator.predict` drops those scores. Prefer a new structured API over breaking the historical return type. Never invent confidence from reference error.

Freeze an initial protocol before large fitting: hypotheses, primary contrasts, grouping/splits, target and input schema, controls, tuning allowance, metrics, practical margins, compute cap, permitted exposure, and branch stop rules. Use the plan's sample sizes and margins as proposals to assess with fresh development evidence, not established power calculations. Record any later change as a new development protocol; never retrofit it to confirmation results.

## 6. Build paired data and meaningful controls

Construct a locomotion eligibility manifest using metadata plus a documented audit. Retain anatomical sides and real elapsed time. Use fixed physical-time windows, initially 64 samples at 25 Hz, without stretching arbitrary sequences. All overlapping windows and all rendered variants of an original motion remain grouped. Preserve person identities, dataset aliases, existing reservations, and exposure history.

Start with matched appearance/degradation pairs at a fixed camera: same motion, shape, time, background, lighting, and seed except for the declared intervention. Record actual person pixel height, camera fitting, clipping, visibility, and depth. Hold out nuisance combinations and motion identities. Viewpoint changes require their correct projected targets; do not force 2D coordinates to be invariant to camera changes. Cross-view canonicalization or 3D lifting is a separate experiment.

Use explicit arrays and schemas for `xy`, `confidence`, `observed`, `timestamps`, normalization, and separately stored targets. Adapt the reusable temporal encoder's 33-joint assumption to the named 12-joint schema with tests; do not silently zero-pad or invent joints. Distinguish missing inputs from invalid targets and structural padding. Fit preprocessing and normalization rules on allowed inputs/training data only.

Build contact sheets, schema/time/pairing audits, and manifest hashes before training. Cache identities must include source content, checkpoint, schema, configuration, preprocessing, and code hashes. Reject incompatible caches and checkpoint resumes. Unsupported or invalid data must produce explicit counts and status, not zero-error successes.

Audit the simulator landmark convention before large fitting. Inspect fixed overlays, anatomical mappings, and systematic joint offsets against independent source/development annotations. Estimator-versus-proxy disagreement cannot establish which is correct. If simulator joint centers systematically differ from the intended COCO locations, resolve the mapping or narrow the target claim; do not train a corrector to erase an accurate real landmark in favor of a biased proxy. When annotations are absent, report synthetic-proxy accuracy and leave real anatomical accuracy pending.

Preserve the old pose-adaptation workflow as a baseline and asset source. Add the essential augmented-COCO and matched-synthetic comparisons when evaluating whether rendering improves the pose estimator. Do not make a broad backbone/loss/learning-rate sweep a dependency of the temporal experiment. First verify actual gradients and output changes; expand adaptation scope only when its own development gate warrants it.

## 7. Implement a fair temporal model comparison

Reuse the small temporal-gait components where suitable: width 96, four encoder layers, two predictor layers, four heads, four-frame patches as starting settings. Keep temporal/joint tokens available. Separate offline restoration interfaces from existing past-only forecasting interfaces; do not weaken the latter's guarantees.

Use the plan's bounded development defaults as initial hypotheses: AdamW at 3e-4 with weight decay 0.01, batch 64 if memory permits, 5% warmup, gradient cap 1.0, 200-update smoke test and at most 2,000 screening updates. Log and validate the EMA/temperature schedule. Use float32 loss diagnostics even if H100 forwards use bfloat16. Give coordinate comparators equally competent tuning and extend training only with a measured learning-curve reason inside the cap. These defaults are not confirmed optima.

The candidate uses a masked online encoder of observed tracks, an EMA target encoder of aligned clean synthetic projected tracks, and a latent predictor. Stop gradients through target embeddings. Begin with the existing centered/sharpened objective and identical stability machinery in the ordinary/paired JEPA controls. Log realized masking, effective teacher lag, online/teacher/initialized state, entropy, variance/rank, and dependence on input. EMA alone is not evidence against collapse or information loss.

Separate input-observed, artificial pretraining-hidden, and target-valid masks. The existing encoder hides only observed tokens and zeros invalid outputs; it cannot be reused unchanged for missing-joint restoration. Add a fixed time×joint output query grid that can decode absent inputs without knowing target validity. A missing input may have a valid synthetic target. Use residual output where an input coordinate exists and a defined absolute prediction when it does not; never add a residual to NaN. Test this with a missing-input/valid-target fixture. Mutating target masks may change supervised loss/support counts, but must not change inference inputs or predictions.

Describe clean-target training as privileged synthetic supervision. Do not call the whole pipeline wholly self-supervised or label-free. A later coordinate/displacement auxiliary loss or different anti-collapse formulation must be separately named and compared.

Treat the proposed paired objective as an assembly of established components. Its empirical advantage must distinguish it from existing temporal refiners. Do not imply an official extension of published S-JEPA by calling this local implementation “S-JEPA v2.”

Include:

1. Unchanged pose estimates and a tuned simple temporal filter/interpolator with the same latency.
2. A lightweight direct temporal denoiser trained on the same noisy/clean pairs.
3. Same-backbone coordinate reconstruction pretraining followed by a frozen-encoder readout.
4. Initialized frozen encoder with the same readout architecture.
5. Ordinary masked JEPA with observed-track targets.
6. Paired clean-target JEPA.
7. Essential alignment/temporal controls, including shuffled pairing and a static/support-matched coordinate control.

Include the SmoothNet-style temporal MLP alongside the same-backbone denoiser: the former is a strong practical baseline; the latter isolates architecture and objective. These answer different questions and should be labeled accordingly.

The decisive latent-versus-coordinate comparison receives the same clean target information, pairs, masks, encoder/readout capacity, and tuning opportunities. Apply the resource matching specified for each contrast below. Ordinary noisy-target JEPA is a supervision-source control, not sufficient evidence that latent prediction beats coordinate learning.

Specify two distinct resource comparisons: a matched-data/steps contrast for objective attribution, reporting the extra teacher/predictor cost, and an equal-total-compute contrast for practical utility, allowing different step counts. Do not assert simultaneous exact equality of data exposure, steps, parameters, and GPU time when the architectures differ. Match what each contrast requires and disclose the rest.

Fit readout weights separately for each frozen representation, using the same architecture and permitted labeled training subset. Literally sharing weights across incompatible latent bases is not a fair comparison. Also report the end-to-end denoiser separately with a budget including the candidate's pretraining plus readout fitting. Count extraction, probes, feature caching, and training in costs.

Keep confidence, validity, and timing inputs identical and included once when isolating coordinate history. Define a per-frame coordinate comparator with the same auxiliary channels, removing only neighboring coordinate history. Do not alter missingness while attributing an effect entirely to temporal coordinates. Shuffled clean-target pairing is a pretraining control; evaluate it on correctly paired natural examples. Diagnostic test-time shuffles are separate and are not biological interventions. Never use time warping during scoring to conceal timing error.

Do not introduce a large video teacher, diffusion generator, model-personalized policy, multiple probe library, new clinical endpoint, and JEPA loss in the same first comparison. If the smallest experiment fails, analyze that result before adding components.

## 8. Evaluate useful accuracy and preservation together

Choose one primary coordinate endpoint: visible body-12 landmark error with declared independent normalization, missing-prediction penalties, and person/recording-balanced aggregation. Report lower-limb, large-error, clean-condition, and nuisance strata separately. Keep synthetic occluded/all-joint results separate from real visible-joint results.

Add motion measures only where references support them: signed ankle separation, displacement over fixed elapsed times, trajectory amplitude, and event timing/phase. Cadence requires enough observed cycles. Uncalibrated 2D tracks do not support metric stride length or true 3D joint angles. Smoothness alone is not a gait-quality metric.

Compare error reduction against attenuation or distortion of actual movement. Plot the accuracy–preservation tradeoff over a predeclared small range of filtering/regularization strengths, selected on development data. Include initially accurate tracks and unusual-motion strata. Synthetic time changes or asymmetry perturbations calibrate sensitivity but do not manufacture clinical diagnoses.

Use person-level paired uncertainty for AMASS, with motions nested within people; use recording/verified-person groups for real video according to actual metadata. Render variants, adjacent frames, and scores from a shared adapted checkpoint are correlated. Report training-seed variability and each held extractor separately. Do not turn row counts into sample sizes or a few architectures into a population-level significance claim.

Independent real temporal references are required before claiming preservation on real video. Build annotation manifests/tools and measure annotation effort on a small development sample. Do not use model predictions as independent ground truth. Preserve protected confirmation recordings. If only sparse frames exist, report coordinate evidence at those frames and explicitly leave temporal confirmation pending.

Freeze the selected method, inputs, metrics, margins, and manifests before opening confirmation labels. Human annotators should not see candidate outputs. Repeatedly inspected GAVD/source outcomes remain development data, even under a new directory or random seed.

## 9. Enforce staged decisions and cost limits

Execute in this dependency order:

`artifact audit → paired-data smoke test → direct temporal baseline → matched JEPA screen → repeated finalists → real development → protocol lock → independent confirmation`.

Keep pose-adaptation calibration as a separately reported supporting branch. Start with one or two source estimators, one seed for feasibility, then three seeds for the decisive comparison. Establish an excluded estimator family before training. Existing held ViTPose assets are not automatically untouched; check exposure records.

Gate A for image-estimator adaptation and Gate B for temporal restoration are independent after the common paired-data checks. A failed image-finetuning gate must not automatically cancel temporal restoration, and a JEPA failure must not suppress a useful direct-denoising result.

The optional personalization and video-feature branches also have separate gates. Do not use a gain in one to claim success in the other.

Use measured throughput/memory to produce a run ledger and job array bounds. The plan's suggested 48 H100-hour development ceiling is a provisional cap, not permission for automatic spending or a runtime guarantee. Count any GPU body-model execution, EGL rendering, pose extraction, feature caching, and retries inside the GPU cap; separately report CPU-only preparation, storage, and annotation. Before expanding, show projected cost and the gate that justifies the expansion. Do not evade a cap through uncounted preprocessing or increasingly broad sweeps.

Return explicit `pass`, `fail`, or `insufficient_evidence` with supporting artifacts:

- Synthetic pose adaptation must beat augmented-real replay or establish a separate motion-coverage benefit.
- Paired JEPA must improve a common useful endpoint beyond matched coordinate training without unacceptable motion distortion.
- Real transfer requires independent real references and held recording evidence.
- Personalization requires meaningful development oracle headroom and then a gain beyond matched scene/state/source-progress controls at equal total cost.
- Optional video tokens require a measured incremental benefit beyond temporal skeleton and scene inputs.

A failed gate ends its dependent branch. Produce a clear skipped notebook section and negative report. Do not silently lower thresholds, replace the primary with a favorable subgroup, or force an action when a no-addition baseline wins.

## 10. Build readable notebooks and auditable artifacts

Use the plan's proposed notebooks 00–08, or a simpler documented arrangement preserving the same gates. Notebook 06 is conditional; notebook 07 must separate freezing from explicit confirmation execution. Generate canonical output-free notebooks from a builder and retain executed copies under unique run identities.

Each notebook must state its question, required inputs, computation, expected outputs, checks, interpretation, and next gate in plain language. Put research logic in tested modules. Show actual units and independent sample counts. Avoid success-colored boilerplate or technical setup details that obscure the scientific result.

Retain configuration and hashes; split/exposure manifests; pairs and rendering metadata; environment and checkpoint provenance; training curves; initialized/online/teacher checkpoints; per-window predictions/targets/masks/times; per-person/recording metrics; uncertainty draws; costs; and gate decisions. A report must be reconstructible from predictions rather than only saved means.

Use meaningful tests: schema permutations and invalid mappings, physical-time resampling, support/target separation, paired-factor identity, forbidden test-label access, future-input boundaries where applicable, stale cache rejection, checkpoint/buffer reset, resume identity, and arithmetic reconstruction. Tests should challenge contracts rather than mirror implementation. Execute fixture notebooks in fresh kernels. Fixture success is software evidence, not real scientific validation.

Prepare concrete HAIC batch commands with explicit environment, output root, stage dependencies, resources, and resume behavior. Perform syntax and dry-run checks locally. Claim GPU operator or runtime checks only when actually performed in an allocated compatible environment.

## 11. Required Codex adversarial-review handoff

Give the independent Codex reviewer this task, together with the actual protocol, code diff, manifests, predictions, and report:

> Try to falsify this study's main conclusion. Inspect evidence directly. Identify leakage, hidden supervision, mismatched data/compute/tuning, camera-coordinate inconsistencies, information lost by preprocessing, correlated sampling, invalid uncertainty, unsupported temporal/clinical claims, and overlap with the closest prior art. Ask whether a simple denoiser, augmentation baseline, scene policy, or initialized readout explains the result. Check that negative gates remain visible and that old results were preserved. Do not accept low JEPA loss, smooth output, or code execution as proof of useful gait representation. Return prioritized findings with exact evidence, consequences, and the smallest adequate correction. Include one strongest competing explanation and a verdict for each proposed claim.

Run a separate readability review: can a general technical reader identify the question, comparison, result, uncertainty, and next step without deciphering configuration jargon? Require the reviewer to flag ambiguous uses of “self-supervised,” “gait,” “world model,” “ground truth,” “independent,” and “significant.”

Record findings and dispositions. Fix supported issues, rerun the checks affected by fixes, and request review of material unresolved changes. Distinguish actual independent review from the coordinator's own follow-up verification. Do not claim reviewers endorsed the research simply because they found no software defect.

## 12. Completion and final response

Deliver the new source modules, builder-generated notebooks, configuration, tests, runbook, literature/evidence ledger, protocol, review dispositions, and artifact-derived report appropriate to the stages actually executed. Preserve all historical results and unrelated edits.

Before finishing, verify local links, notebook schemas, stage dependencies, effective configuration, meaningful tests, fixture execution, report arithmetic, and the final diff. Recheck that Torch 2.6.0+cu124 remains required. Ensure empirical statuses distinguish planned, fixture-tested, source-run, real-development, and confirmed.

In the final response, lead with the concrete outcome. Link the notebook guide, report, and review dispositions. State what changed, what was actually tested, the result of each scientific gate, and the exact next runnable step. If empirical work is blocked, identify the missing input/access and provide the prepared command or annotation requirement. Do not call the research stronger, novel, clinically useful, or complete beyond what the evidence supports.
