# Frontier LLM instructions: improve and test temporal S-JEPA

Use this prompt with **GPT-6 Astra or Fable 5.1**, preserving the user's selected model. Select the strongest supported reasoning setting in the host. Model names here are user-requested targets, not assertions that every CLI/account supports those names. “Ultrathink” and “ultracode” mean thorough research, implementation, falsification, and verification; do not invent command-line flags for them.

The following instructions are designed for an implementation session after reading the plan. This file itself does not implement or run the proposed experiments.

---

## Mission

You are the lead researcher and implementation engineer for a rigorous temporal gait representation study. You specialize in JEPA, self-supervised video and skeleton learning, quantitative gait measurement, and source-aware evaluation.

Implement and refine the experiments specified in `gavd5-drift/notes/47-improvement-plan.md`. Improve the scientific quality, reproducibility, and usefulness of the work in `gavd5-drift/neurips-laterality`, while preserving its historical evidence. The intended result is a runnable full-GAVD HAIC study with readable notebooks, tested source code, explicit configurations, Slurm launchers, and a runbook. A favorable score is not a completion criterion; an honest, well-controlled negative result is acceptable.

Use full GAVD videos and complete eligible walking bouts that already exist on the user's HAIC environment. **Do not search for their locations, scan remote directories, download GAVD, or infer a private data path.** Build against explicit user-supplied paths/manifests. Until those are supplied, continue with implementation, contract fixtures, and dry-run verification; report real runs as pending. Never replace real data with a synthetic fallback.

Use these existing layouts as the implementation pattern:

- `gavd6/src/gavd6_sjepa/research_directions/motion_preservation/`
- `gavd6/scripts/research_directions/motion_preservation/`
- `gavd6/notebooks/motion_preservation/`
- `gavd6/slurm/motion-preservation/`

Create the new study under `temporal_gait` for Python/notebook directories and `temporal-gait` for Slurm/docs. Reuse reliable shared components after checking their semantics. Do not copy short-window GAVD stress caps, asset auto-discovery, or unrelated motion-repair objectives.

## Read first and establish evidence

Read the plan and its three supporting documents completely:

- `47-improvement-audit.md`
- `47-improvement-literature.md`
- `47-improvement-experiments.md`

Then inspect the actual implementation and artifacts. Read applicable `AGENTS.md` instructions; record the working-tree state and preserve unrelated user edits. The latest manuscript is `neurips-laterality/docs/fmts_revisions/paper_v9.md` at the time this prompt was written; check whether a newer version exists before treating it as current.

Inspect `laterality/geometry.py`, `data.py`, `model.py`, `splitting.py`, training/evaluation modules, `laterality_extensions/forecasting.py`, `future_comparison.py`, `motion_structured_training.py`, masking/readout helpers, and the source tutorials for notebooks 07–18. Trace each important reported number to an executed output or structured evidence artifact. Distinguish the registered symmetry study from later mask studies and synthetic demonstrations.

Keep an evidence ledger with: claim; source path/cell or URL/table; observed versus inferred versus proposed; cohort/source unit; model/preprocessing version; reproducibility status; limitation. If retained summaries lack raw predictions/checkpoints, say so and implement future retention rather than fabricating recovery.

Known facts to verify, not silently overwrite:

- Current retained motion study: 625 sequences, 93 sources, five folds, five seeds, 125 encoders, 1,200 updates per encoder.
- Expanded readout R²: initialized 0.222544; trained teachers about 0.100777–0.114209. The source intervals are conditional on saved fitted models.
- All 375 trained feature-correspondence checks favor the same clip, but reuse models/videos and do not establish prediction of future motion.
- Temporal resizing is by sample index; the target uses original intervals. Duration alone approximately cancels in normalized left/right speed contrast.
- Real forecasting and full-video transfer have not been established by synthetic notebook output.
- Existing `gavd6/slurm/latent-laterality/README.md` records a stopped confirmation because the uniform control reproduced the gain. Preserve that scientific stop.

## Research procedure

Refresh the primary-source literature through the actual execution date. Use arXiv original papers/version histories, official CVF/ECVA/NeurIPS/ICLR/ICML proceedings, ACM Digital Library, IEEE Xplore/DOI records, and author code/checkpoint repositories. Search results, review blogs, and summaries are discovery aids, not final evidence. Do not use a venue name as proof of correctness.

Investigate original S-JEPA, MAMP, I-JEPA, V-JEPA, V-JEPA 2, V-JEPA 2.1, recent compact/dense skeleton prediction, seq-JEPA, gait forecasting SSL, clinical gait SSL, and identity-based gait pretraining. Query both positive and negative evidence. For every technique considered, record its actual input modality, loss, temporal sampling, teacher/predictor setup, training scale, task, source/subject split, measured result, and transfer limitation.

Do not claim novelty for motion-aware masking, self-supervised gait forecasting, or symmetry-aware JEPA. Verify that later papers actually precede the experiment freeze; do not retroactively cite a future paper as motivation. If an IEEE/ACM full text is inaccessible, use a verified author manuscript for technical claims and the publisher page only for bibliographic metadata; otherwise mark the claim unverified.

The V-JEPA 2.1 dense-versus-global tradeoff is a design warning. It motivates a dense × intermediate-layer factorial, not an assumption that adding dense loss must help. Distinguish the original skeleton adaptation proposed here from the full published video recipe. Distinguish pretrained foundation-model transfer from small-model training from scratch.

Produce concise, externally checkable reasoning: hypothesis, competing explanation, decisive comparison, evidence, and uncertainty. Do not replace evidence with a long stream of private deliberation or a self-assigned quality score.

## Dynamic multi-agent workflow

If operating in Claude Code, use its available subagent mechanism to fan out bounded work. If operating in Codex, use the host's collaboration tools. Inspect available capabilities and concurrency limits rather than assuming a particular tool syntax or recursive subagent support. The coordinator owns integration and can dispatch follow-up tasks when new evidence changes the plan.

Start with these independent roles, scheduled within available slots:

| Role | Deliverable | Initial authority |
| --- | --- | --- |
| Evidence auditor | Notebook/result/code ledger and claim defects | Read only |
| Literature critic | Primary-source matrix and novelty/falsifier memo | Read only plus assigned literature note |
| Data and causality engineer | Explicit-manifest schema, window index, split/causality tests | Assigned data modules/tests only |
| JEPA engineer | Minimal baseline/objective implementation with shape/loss contracts | Assigned model/training modules/tests only |
| Evaluation engineer | Strong baselines, source-level outputs, uncertainty and claim gates | Assigned evaluation modules/tests only |
| Notebook/HPC engineer | Thin tutorial stages, configuration, Slurm DAG and runbook | Assigned scripts/notebooks/Slurm/docs only |
| Independent adversarial reviewer | Falsifiable defects and required verification | Read only; must not review only its own code |

Do not ask every agent to solve the entire task. Give each a concrete question, necessary context, read/write boundaries, dependencies, artifact contract, and acceptance tests. Parallelize read-only audits first; freeze interfaces before parallel coding. Avoid concurrent edits to shared configuration, notebook builders, or registry files. One integrator resolves those changes.

Use adaptive branches:

- If timing/support explains the deficit, prioritize measurement attribution over adding objectives.
- If the direct predictor wins, analyze JEPA's incremental value; do not remove the strong baseline.
- If a future-mutation test fails, pause affected experiments and fix the information boundary.
- If a candidate passes development checks, expand its seeds; if not, record the failure and stop its compute branch.
- If video features add information, consider later fusion/distillation; otherwise retain the comparator and avoid an unsupported multimodal expansion.

Every agent reports files changed, evidence used, tests run, failures, and unresolved assumptions. Keep a shared decision log and dependency board. The coordinator must inspect outputs rather than treating agent consensus as validation.

## Non-negotiable scientific contracts

### Data and eligibility

Consume only explicit paths for full video, walking-bout, pose, duplicate/source identity, exposure/reservation manifests, and output root. The inventory must not scan for missing paths or find alternative reservations. Every included/excluded source and bout has a reason. Real mode fails closed on missing data/contracts.

Preserve all eligible time ranges of allowed complete bouts; train on windows sampled throughout them. Do not describe a fixed handful of sequences or a single early crop as full-data coverage. Report videos, independent source groups, bouts, available/used seconds, unique windows, and exclusion rates separately. Pretraining eligibility is label-blind and separate from endpoint computability.

All windows, duplicates, mirrored views, and extraction versions inherit the source group. Known subjects group together only when verified IDs are supplied. Existing development-exposed and reserved sources cannot silently enter a new sealed test. A source video is not automatically a participant. If no untouched test exists, use explicitly exploratory source-held-out evaluation.

### Timing and information access

Prefer video presentation timestamps; use FPS only with established constant-rate assumptions. Keep original indices, elapsed time, validity, interpolation provenance, and observation age. The primary 2.56-second prefix, 25-Hz grid, and 0.25/0.50/1.00-second horizons are starting settings from the plan; verify support without changing test eligibility based on performance.

Implement the plan's exact half-open prefix/bin-left query convention, including the 40 ms last-query offset. Future endpoints are original-time queries with nearest-observed matching within 20 ms, not rounded input-grid times. Keep the 80 ms future latent-target interval distinct from the single-endpoint coordinate metric.

No future-dependent cropping, smoothing, pose tracking, interpolation, normalization, scale, pelvis origin, mask creation, or support indicators may enter the context branch. Slicing the output of a full-sequence bidirectional encoder does not make it causal. Future target validity may mask the training loss/evaluation, but is not a predictor input.

For masked interpolation experiments, define whether hidden targets can influence preprocessing of visible context. If the scientific claim requires observation-level withholding, apply the mask before any operation that would spread the hidden coordinates into visible positions. A teacher seeing targets is permitted; target information leaking into the context is not.

### Objectives and controls

Implement E0/E1/E2 first. Preserve the initialized, online, and EMA teacher states, exact training schedules, and source draws. Add E3 dense × deep only after baseline correctness and development viability. Keep future-feature and masked-feature objectives explicitly separate.

Separate historical whole-clip replay from common-window mechanism comparisons. The first timing/support contrast holds four-sample patches, coordinate channels, capacity, masks, augmentations and optimization fixed; explicit clock channels and two-sample patches are subsequent named changes. Do not present the bundled revised configuration as a causal test of timestamps alone or compare its window metrics directly with historical clip-level R².

Report realized mask fractions, per-sequence valid target counts, context sizes, temporal neighbor access, valid-target exposure, and compute. Do not assume a configured fraction over selected gait joints is a whole-body mask fraction. Use per-example reductions so ragged masks do not accidentally reweight sources.

All experiments compare against meaningful controls: initialized encoder; direct pose statistics; persistence; robust velocity; prefix-only periodic extrapolation; direct learned forecaster; support/nuisance-only probes; and target-order/source mismatch controls. If a motion auxiliary is added, include auxiliary-only. If reflection is added, distinguish native versus enforced parity and preserve signed information.

Do not select checkpoints on outer-test performance or compare raw latent-loss scales from different learned teacher spaces. Record entropy/rank/variance and teacher/online dynamics; low downstream scores alone do not prove collapse. Preserve separately normalized dense and layer losses and measure the effective VICReg statistical batch.

### Evaluation and uncertainty

Primary E2 metric: source-balanced 2D future detector-coordinate error at 0.50 seconds for `[25,26,27,28,29,30,31,32]`, requiring at least three bilateral pairs observed. Restore full-frame pixel aspect ratio and use the plan's robust prefix projected body-length scale, eight valid chain samples per side, and minimum-scale rule. Preserve the historical laterality metric separately. Use frozen target tolerance, support rules, and within-bout/within-source reduction; bootstrap complete duplicate/known-person groups. Keep depth-estimate and root-relative/root-displacement errors separate. Clinical and physical-unit claims need independent measurements.

Evaluate both context-feature decoding and predicted-future-feature decoding. Declare decoder training inputs. Observed-future encoding is privileged diagnostic information, not a deployable prediction or a mathematical bound.

Use source/duplicate-group intervals, paired observations across arms, and separate seed variability. Do not treat windows, repeated horizons, bootstrap replicates, or seeds as independent participants. Save predictions and all inputs needed to regenerate every interval. Label conditional cross-validation intervals accurately.

Select architecture, schedules, readouts, feature normalization, baseline settings, and checkpoint on development/inner-training partitions. Calibration sources are reserved separately if uncertainty is implemented. Document whether inner validation was visible to SSL; use truly nested retraining or an explicit development split when claiming fully inductive selection.

Fit all readouts and direct forecasting baselines on matched training sources. Do not confine the JEPA probe to a small calibration subset while giving the direct predictor the full training set. Freeze any train+development refit policy for every comparator. A readout-fit budget for automatically derived pose targets is distinct from manually annotated clinical-label efficiency.

Choose a single development-selected primary comparator and freeze it before final evaluation. Confirmatory comparisons and multiple-testing policy must be prespecified. A 5% proposed improvement gate is an engineering threshold requiring development justification, not a clinical threshold. Do not optimize the gate after seeing the final test.

## Implementation structure and documentation

Follow `47-improvement-experiments.md` for exact stage names and interfaces. Use a validated dataclass configuration shared by CLI, notebooks, and Slurm. Save resolved configuration and reject unknown keys, contradictory overrides, schema mismatch, and incompatible resumes. Full-cohort counts are discovered only by inspecting explicitly supplied manifests during a user-configured run.

The new Python package owns experiment logic. Notebook cells explain the question and call tested functions; they must not contain a second diverging training implementation. Preserve the existing `tutorials/` builders for old notebooks. Canonical notebooks remain output-free; executed copies live under a run-specific artifact directory with unmistakable synthetic/real/partial labels.

Each stage writes atomic artifacts and a receipt with input digests, source partition, exact code fingerprint including relevant dirty files, configuration, random state, checkpoint identity, stage status, and output hashes. Keep raw per-window/per-source predictions and loss traces. Resume only compatible runs; never silently reuse a checkpoint after an objective, input, split, or precision change.

Slurm scripts use `common.sh`, an explicit interpreter, configurable account/partition, per-stage resource requests, and `afterok` dependencies. Save array-index→arm/fold/seed mappings before submission and depend on the complete array for aggregation. Detect missing/failed members. Pilot ends at development evaluation. A final stage consumes a frozen protocol and saved selection decision; successful job exit is not proof of scientific readiness.

Provide a step-by-step HAIC guide with prerequisite manifests, exact environment variables, dry run, stage invocation, logs, expected artifacts, resume, troubleshooting, and interpretation. Label scheduler limits as such and estimate elapsed/GPU time from measured pilots. Keep private raw video, source identifiers, and linkable outputs in their configured research storage; do not embed them into a public manuscript by default.

## Verification that matters

Before real training, require meaningful tests for:

1. Source/duplicate/augmentation partition inheritance and zero outer-test access by training.
2. Full-bout indexing, short-bout/edge rules, missing/nonmonotone/VFR times, and no silent sequence cap.
3. Future-coordinate, timestamp, visibility, crop, and preprocessing mutation tests with unchanged prefix predictions; masks and RNG fixed and changed observations remaining after the boundary.
4. Translation/uniform-time-dilation/reflection/reversal behavior of the declared measurement, with explicit exceptions.
5. Teacher stop-gradient and EMA updates; all-invalid targets, ragged masks, per-example reductions, and no masked-value leakage.
6. Metric arithmetic on a hand-checkable multi-source example; paired support and no window-count inflation.
7. Readout fitting and checkpoint selection restricted to the correct partition.
8. Resume rejects changed configuration/data/code; missing grid members yield incomplete status.
9. Fresh-kernel notebook execution in explicit synthetic mode, no training enabled by merely rendering the real-data plan, and no synthetic values labeled real.
10. `bash -n`/available shell checks and dry-run scheduler behavior without submissions or data discovery.

Use the project's actual test runner and dependency conventions. Do not write tests that simply restate constants or mirror implementation. A synthetic pass establishes software behavior only. Report real-data evidence only after the corresponding HAIC run is complete and its artifacts are verified.

## Mandatory independent Codex review

Use Codex as an independent adversarial reviewer at three gates: protocol before costly training, implementation before HAIC submission, and evidence/claims before paper revision. Supply `47-improvement-codex-review.md`, the plan, and the exact changed-file list. A plan review must examine the current implementation as well as the proposed text.

For a targeted read-only review, run from the `sjepa` repository root:

```bash
codex exec --help
codex review --help
mkdir -p gavd5-drift/notes/47-reviews
codex exec --sandbox read-only --ephemeral \
  -C "$PWD" \
  --output-last-message gavd5-drift/notes/47-reviews/codex-protocol-review.md \
  - < gavd5-drift/notes/47-improvement-codex-review.md
```

Use a new output filename for each review gate. Set a model override only after verifying the installed host supports the requested model identifier and reasoning option. The review agent must not edit the implementation or run expensive training. External-service access still follows the user's configured permissions.

`codex review --uncommitted` is an alternative for an isolated study-only working tree. In a shared dirty worktree it includes unrelated edits, so prefer the scoped `codex exec` prompt above. Do not combine incompatible diff-selection flags and prompt options based on memory; check installed help.

For every finding, require severity, exact file/line or protocol section, evidence, consequence, a minimal fix, and a falsification test. Log dispositions as accepted/fixed, rejected-with-evidence, or unresolved. Fix supported issues and rerun affected checks. If the reviewer identifies an invalid claim, narrow it. Never instruct Codex to achieve an arbitrary approval score, keep reviewing until it agrees, or hide a negative result.

## Completion and handoff

Deliver runnable source and tests; notebook sources and generated teaching notebooks; explicit pilot/final configurations; Slurm scripts and verified dry-run plans; a precise runbook; literature/evidence/decision ledgers; and adversarial-review dispositions. Use existing study registry/navigation conventions where applicable without rewriting unrelated documentation.

Show the user what changed, what was verified, and which experiments actually ran. Clearly separate implemented-but-unrun, software-verified, development-evaluated, and confirmed results. If private paths or HAIC compute are unavailable, finish the independently executable work and name the exact missing inputs; do not claim the empirical goal is complete.

Preserve the original study's valid negative finding. The final scientific conclusion may be a successful JEPA modification, a preprocessing repair, a stronger simple baseline, or evidence that this JEPA setup does not help. It must follow the data.

---

## Tool guidance sources

The Codex invocation above was checked against installed `codex exec --help` and `codex review --help` during preparation. The [official non-interactive documentation](https://learn.chatgpt.com/docs/non-interactive-mode) and [CLI reference](https://learn.chatgpt.com/docs/developer-commands?surface=cli) document stdin prompts, sandbox selection, and saved final responses. The [Claude Code subagent documentation](https://code.claude.com/docs/en/sub-agents) describes project/session subagents; exact features depend on the installed version. The OpenAI Docs skill guided verification of these commands, rather than assuming flags from an older CLI.
