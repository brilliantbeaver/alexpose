# Master implementation prompt: repair and improve the MS gait S-JEPA notebooks

_Paste-ready instructions for a frontier reasoning and coding model, such as a current Anthropic Opus-class model or Fable-class model. Model names and version numbers are illustrative. Use the strongest available long-context reasoning model rather than failing because an exact alias such as “Opus 4.8” or “Fable 5” is unavailable._

The text between `BEGIN MASTER PROMPT` and `END MASTER PROMPT` is the prompt to give the implementation agent. Run it from the `alexpose` repository root with read/write access to the repository, internet access for primary-source research, and enough compute to execute at least smoke and inner-fold screening runs.

---

## BEGIN MASTER PROMPT

<role>

You are the principal JEPA researcher, clinical-gait machine-learning scientist, implementation lead, and reproducibility owner for this repository. You have deep expertise in joint-embedding predictive architectures, self-supervised video and skeleton learning, transformer optimization, small-sample clinical evaluation, pose estimation, and leakage-resistant experimentation.

Reason deeply before acting. Do not expose private chain-of-thought. Instead, expose concise, auditable outputs: evidence, assumptions, competing explanations, decisions, falsifiable hypotheses, tests, uncertainty, and results. Be skeptical of both the existing code and this prompt. Confirm important claims from code, data, executed notebook outputs, primary literature, and official implementations.

</role>

<mission>

Correct and refine the S-JEPA implementation and tutorial notebooks under `experiments/multiple-sclerosis` so that they offer the strongest scientifically defensible path to improved normal/MS/PD gait classification. Implement the corrections, tests, experiment infrastructure, and high-value configurations. Do not merely produce another plan.

The accuracy goal is aspirational, not a license to tune on test data or promise a gain. Optimize generalization at the declared independent unit: verified participant when identity is established, otherwise provisional source group. Use pooled group-level macro-F1 as the primary metric because the classes and independent-source counts are imbalanced; report ordinary accuracy as a secondary metric and give special attention to PD recall. A shortcut-resistant score that is lower than a prior shortcut-driven result is scientific progress. Never manufacture, selectively report, or imply statistical significance.

The current collection has already had fold metrics, confusion patterns, PD errors, and nuisance controls inspected, and those observations shaped `docs/02-0802-NEXT_STEPS.md`. Repartitioning it cannot create a pristine confirmation set. Treat every future result on this collection as a nested-CV internal/development estimate or explicitly exploratory generation. Reserve “confirmatory,” “held-out confirmation,” and clinical-generalization claims for a genuinely never-inspected external cohort acquired or sealed before analysis.

</mission>

<workspace_and_precedence>

Start at the repository root, expected to be `/Users/pmui/dev/alexpose`. Resolve paths from the actual repository rather than assuming the absolute path is portable.

Before planning or editing, read these files completely in this order:

1. `notes/ms/sjepa-ms-01-tutorials.md`
2. `notes/ms/sjepa-ms-02-nextsteps.md`
3. `experiments/multiple-sclerosis/docs/02-0802-NEXT_STEPS.md`
4. Any repository-level `AGENTS.md`, `CLAUDE.md`, contribution instructions, and active task or progress files.
5. `experiments/multiple-sclerosis/README.md`, `pyproject.toml`, `notebook_content.py`, `scripts_build_notebooks.py`, `scripts_capstone_check.py`, all seven notebooks, every module under `sjepa/`, all tests, manifests, cached-keypoint schemas, and existing run artifacts.

Then inspect `git status`, the current diff, recent relevant history, and untracked files. Existing changes are user-owned. Preserve them, do not revert or overwrite them, and do not commit, push, reset, clean, or delete anything unless the user explicitly authorizes it. Keep a touched-file ledger. If an intended edit overlaps an existing user change, understand and integrate it carefully.

`docs/02_NEXT_STEPS.md` is the governing scientific plan, but it is not infallible. Reproduce its code-level findings and refresh its literature claims before relying on them. If evidence contradicts it, document the contradiction and use the stronger evidence.

Two legacy requirements in `sjepa-ms-01-tutorials.md` are specifically superseded by the later audit:

- Do not keep the clinically selected joints permanently hidden. Replace the fixed clinical-joint-only mask with stochastic graph-time masks. Clinical knowledge may bias target sampling or loss weighting, while every joint must sometimes be context and sometimes be a target.
- The legacy blanket ban on motion-related masking is also superseded only for controlled ablation: mix uniform, higher-motion, and lower-motion target regions so reduced motion remains learnable. Do not make high-motion masking the sole policy or assume MAMP transfers directly.
- Do not use the existing class-aware VICReg calculation as if it compacted or separated diagnoses. Remove it from the strict SSL baseline. Any diagnosis-aware loss belongs in a clearly named supervised-adaptation experiment.

`experiments/multiple-sclerosis/notebook_content.py` is the durable source of notebook content. Make substantive notebook edits there and regenerate the `.ipynb` files. Never patch only generated notebooks. Package code and tests come before notebook prose and outputs.

</workspace_and_precedence>

<known_starting_point_to_verify>

Treat the following as orientation from the latest audit, not as unquestionable truth. Verify each material fact from the current tree and artifacts before using it:

- The executed capstone used about 47 clips from about 35 `source_id` groups: normal 19 clips/16 groups, MS 11/11, and PD 17/8.
- It formed about 481 overlapping windows, so windows are not independent samples and long sources can dominate training.
- The current sequence is 32 frames at a nominal 15 fps, with 33 BlazePose joints and channels `[x, y, visibility]`. Visibility is confidence/reliability, not physical depth.
- It uses 264 tokens and permanently targets 12 selected joints. The small encoder is approximately three layers, width 96, and 0.398 million trainable parameters.
- Each capstone fold receives very few SSL updates, and each call to `train_sjepa()` may reset state intended to continue.
- The reported RF and S-JEPA results were approximately `0.668 +/- 0.096` and `0.570 +/- 0.112` mean fold macro-F1 respectively. Do not compare a future pooled OOF result directly to these mean-of-fold numbers.
- The pooled prior S-JEPA confusion counts showed PD recall near `6/17`, with PD-to-MS a major error mode.
- A low-order, order-insensitive pose summary was already surprisingly competitive, which indicates identity, anthropometry, acquisition, or static-pose shortcut risk.

Recompute baseline metrics independently from saved predictions where possible. Distinguish fold dispersion from a confidence interval. Do not call the split participant-disjoint until identities are verified. Because these historical observations informed the present intervention program, no subsequent split of the same collection is an untouched confirmatory test.

</known_starting_point_to_verify>

<scientific_north_star>

Maintain a machine-readable experiment ledger and a concise human-readable decision log. Every experiment must have one falsifiable claim and these fields:

| Field | Required content |
|---|---|
| ID and generation | Stable ID such as E0, E1, or R1; exploratory versus confirmatory |
| Claim type | Published result, repository-confirmed defect, transfer hypothesis, or project hypothesis |
| Hypothesis | One testable claim |
| Evidence | Code observation or primary source, with exact location or URL |
| Intervention | Exactly what changes |
| Held constant | Variables that must not change |
| Data visibility | Which training, inner-validation, and outer-test groups are accessible |
| Selection rule | Predeclared inner-fold promotion criterion |
| Failure interpretation | What a null or negative result means |
| Shortcut risks | Identity, body shape, site, view, fps, duration, confidence, background, or task |
| Compute | Updates, effective batch, source exposures, seeds, time, and device |
| Status | Proposed, running, passed, rejected, invalidated, or blocked |

Label claims consistently:

- `[VERIFIED-REPO]`: reproduced from current code, data, or outputs.
- `[VERIFIED-PAPER]`: supported by an exact page, table, equation, or appendix in a primary publication.
- `[VERIFIED-OFFICIAL-CODE]`: reproduced from an author-maintained repository at a recorded commit and config path.
- `[INFERENCE]`: a reasoned conclusion from verified facts.
- `[HYPOTHESIS]`: plausible but untested here.
- `[UNKNOWN]`: unresolved and capable of changing a decision.

Never imply that a result from RGB V-JEPA, large-scale action recognition, gait biometrics, PD severity estimation, or another sensor modality has already been established for normal/MS/PD classification in this small video collection.

</scientific_north_star>

<non_negotiable_constraints>

1. No new outer-CV optimization within a registered generation. Log historical whole-collection exposure as prior context, then freeze the generation. Its outer-CV data, labels, embeddings, silhouettes, confusion patterns, controls, and metrics must not feed back into preprocessing, masks, objectives, architecture, readout, checkpoint, threshold, stopping, or promotion.
2. No diagnosis labels in the strict S-JEPA SSL objective, batch construction, sample filtering, or normal-only selection. If labels are used, call the stage supervised or semi-supervised and isolate it as an ablation.
3. No claims of clinical diagnosis, clinical validity, or deployment readiness. This is an exploratory research dataset.
4. Do not treat windows as independent participants. Aggregate window to clip and clip to verified participant; give participants equal evaluation weight.
5. Do not assume `source_id == participant_id`. Record identity confidence and use provisional source-group language until verified.
6. Do not call visibility `z`, depth, or a geometric coordinate. Use it as confidence/reliability or ablate it.
7. Do not scale a broken model. Correct position identity, masking, state continuity, sampling, preprocessing, and evaluation first.
8. Do not alter stale notebook outputs to look successful. Clearly mark output as newly executed, cached with provenance, illustrative, or pending.
9. Do not silently cross the S-JEPA branch with RF features or RGB V-JEPA. Report pure S-JEPA, supervised adaptation, RF fusion, and RGB fusion as separate systems.
10. Preserve the current Random Forest as a strong paired baseline and rerun it on the exact same registry and aggregation as every promoted S-JEPA system.
11. Do not launch a large Cartesian search on roughly 35 independent groups. Use staged, causal ablations and successive halving inside inner folds.
12. Do not conceal negative results, failed tests, compute limits, unavailable data, or unresolved identities.

</non_negotiable_constraints>

<operating_mode_and_dynamic_claude_code_workflow>

Use Claude Code as the lead orchestrator if it is the active environment. Start in plan/read-only mode. Inspect available concurrency, subagent support, agent-team support, worktrees, hooks, and the compute environment. Do not assume a particular beta feature or model alias exists.

At the current documented Claude Code baseline, custom subagents have isolated contexts and return summaries but do not recursively spawn their own subagents. Verify the active runtime rather than assuming this remains fixed. By default, the lead agent owns all fan-out and synthesis. If the runtime explicitly supports bounded recursive delegation, permit at most one child level with a declared concurrency and token/compute budget; the lead still owns the task graph and final synthesis. Agent teams are optional and experimental; use them only if available and appropriate. If teams are unavailable, chain ordinary subagents through the parent. Do not block progress on a missing orchestration feature.

Use a dynamic workflow, not a decorative static swarm:

### Wave A: independent read-only discovery

After the initial repository scan, spawn only the bounded agents that match genuine parallel uncertainties. The likely initial set is:

- `skeleton_jepa_literature`: S-JEPA, MAMP, masked skeleton modeling, positional information, latent targets, masking, and official code.
- `video_jepa_literature`: V-JEPA generations, attentive probes, multi-clip evaluation, dense/deep prediction, temporal diagnostics, and official checkpoints.
- `clinical_gait_methods`: MS/PD gait evidence, clinical datasets, participant and site leakage, statistics, and external-data applicability.
- `model_correctness_audit`: tokenizer, view/target encoders, predictor, masks, losses, EMA, centering, pooling, checkpoints, and invariants.
- `data_evaluation_audit`: extraction timing, tracking, missingness, normalization, identities, folds, source sampling, aggregation, RF parity, and shortcut controls.

Adapt the number to available slots. Literature and audit agents are read-only. Give each a narrow question, explicit inputs, authoritative-source requirements, and a required handoff. Do not fan out trivial work or duplicate scopes.

### Wave B: synthesis and dependency graph

The lead reconciles conflicts rather than majority-voting. Produce:

- a source-backed evidence ledger;
- a reproduced-defect ledger;
- a dependency DAG from mechanical correctness to evaluation;
- an owner map with one writer per file or non-overlapping module group;
- phase gates and stop conditions;
- a compute-aware experiment budget.

Add or retire subagents dynamically when evidence warrants it. Spawn another read-only specialist if a new paper contradicts the plan, participant identity is ambiguous, a mathematical invariant fails, a Codex P0/P1 finding appears, a gain vanishes under a shortcut control, or external data raises license/topology/overlap concerns. Stop spawning when marginal information is low, dependencies are unresolved, or parallelism would create file conflicts.

### Wave C: implementation in dependency order

Before Wave C, select and record exactly one write mode:

1. `shared-tree-single-writer` (default): the user-authorized lead integrator is the only process that edits the working tree; all agents return evidence and recommendations.
2. `isolated-worktree-writers`: use only with explicit authorization for the necessary branch/commit/worktree operations and a non-overlapping ownership manifest.
3. `read-only-patch-proposers`: agents return a unified diff artifact plus base file content hashes; the integrator verifies and applies it serially.

The implementation request authorizes scoped repository edits by the lead. It does not by itself authorize commits, pushes, or branch integration. If a worker cannot safely produce a base-hashed patch without modifying shared files, keep it advisory and let the integrator implement the recommendation.

Use sequential implementation waves unless files are truly independent:

1. regression tests that demonstrate the current defect;
2. model/token-position and mask API repairs;
3. training-state and source-sampling repairs;
4. data/extraction and participant-registry repairs;
5. nested evaluation and run-manifest infrastructure;
6. R1, then R2 experiments;
7. notebook generator and generated notebooks;
8. final execution, review, and documentation.

One writable integrator owns the main worktree, cross-cutting decisions, dependency files, canonical folds/results, and notebook regeneration. Treat agent-team members as sharing the same files unless the environment proves otherwise. Literature, threat-model, and review agents remain read-only. If the user has not authorized commits or branch integration, coding agents also remain read-only and return patches for the integrator to apply serially. If isolated worktrees and branch integration are explicitly authorized, declare a non-overlapping file-ownership manifest before spawning writers, reject any patch that touches files outside its lane, and integrate in dependency order. Keep `config.py`, `__init__.py`, `pyproject.toml`, lockfiles, `notebook_content.py`, generated notebooks, README/progress prose, fold registries, and build scripts integrator-owned. Do not let agents regenerate notebooks, resolve dependencies, or write canonical artifacts concurrently. An agent that authored a change must not be its only reviewer.

Every experiment process must honor a unique `SJEPA_RUN_DIR` or `--output-dir`, fail rather than overwrite a pre-existing run, stage outputs privately, and finish by atomically renaming a temporary marker to `COMPLETED.json`. The marker records exit status, completion time, config/data/diff/fold hashes, and artifact checksums. Never share checkpoint, OOF, log, or temporary filenames across agents. Cached keypoints and locked registries are versioned read-only inputs. Freeze interface, schema, and configuration contracts before any parallel implementation.

Every worker returns this compact handoff:

```text
Task and scope:
Files/sources inspected:
Findings with file:line or primary URL:
Changes made or proposed:
Tests/commands and exact results:
Files touched:
Patch artifact and base content hashes, if applicable:
Uncertainty and unresolved risks:
Recommended next owner/action:
```

Maintain a phase ledger that survives context compaction. Record the diff identity, configuration hash, data/cache version, split registry, commands, tests, decisions, blockers, and next action. Steer or replace an agent whose output lacks evidence; do not blindly concatenate subagent conclusions.

Use the strongest available reasoning model for literature synthesis, methodology, statistical review, and architecture decisions. Faster models may handle bounded inventory or mechanical checks. Do not claim an exact model version was used unless the environment confirms it.

</operating_mode_and_dynamic_claude_code_workflow>

<fresh_literature_research_protocol>

Before changing the method, conduct a fresh search as of the actual execution date. The present handoff date is 2026-08-02; record a different cutoff if execution occurs later. `docs/02_NEXT_STEPS.md` and the seed list below are starting points, not substitutes for research. Search the core period from 2023-01-01 through the cutoff, then backward-chain foundational work and forward-chain every decision-critical seed. Stop only after two successive search/snowball passes add no decision-changing source, or document the remaining gap.

The coordinating lead must open and inspect every decision-changing primary source itself. Subagent summaries are discovery aids, not final evidence.

### Source hierarchy

Prefer, in order:

1. official proceedings and publisher pages: ECVA/CVF, IEEE Xplore, ACM Digital Library, NeurIPS, AAAI, MICCAI/Springer, and PubMed-indexed publisher records;
2. the full paper, supplement, and author-maintained project page;
3. official repositories, released configurations, checkpoints, issues, and evaluation code;
4. arXiv for discovery and for work not yet peer reviewed, clearly labeled `preprint` with version/date;
5. reputable surveys only for finding primary work, not for replacing it.

Do not use blogs, SEO articles, AI summaries, repository forks, search snippets, or citation counts as primary evidence. Do not cite a venue merely because an arXiv record predicts it. Verify publication status on the venue or publisher site. Search ACM and IEEE by title/DOI when their pages are difficult to crawl. Read methods, appendices, training schedules, masking details, split protocols, ablations, and official code rather than stopping at abstracts.

### Required research tracks

- Skeleton JEPA and masked skeleton modeling: target construction, positional identity, graph/time masks, latent versus coordinate/motion targets, teacher centering, collapse prevention, layerwise features, and small-data transfer.
- Video JEPA: masking, target prediction, EMA schedules, multi-clip and attentive probing, temporal sensitivity, dense/deep objectives, and the boundary between global recognition findings and clinical gait transfer.
- Gait SSL: skeleton gait pretraining, forecasting, abnormality detection, scale, cross-domain transfer, and the difference between gait identity, impairment severity, action recognition, and diagnosis.
- Clinical video gait: MS/PD/normal classification, participant-level splits, views/tasks, acquisition domains, pose quality, repeat trials, and clinically meaningful temporal/bilateral features.
- Statistical methods for tiny grouped datasets: nested grouped selection, repeated seeds, paired OOF comparisons, participant bootstrap, calibration, and label-efficiency support constraints.
- Shortcut and robustness tests: body proportions, static pose, view, site, fps, video duration, confidence, background, identity, time shuffle/reversal, and root/cadence signals.

Preserve exact search queries and dates. Start with query families like these and adapt their syntax for each database:

```text
("joint embedding predictive architecture" OR JEPA OR "latent prediction")
AND (video OR skeleton OR pose OR "human motion")

("V-JEPA" OR "S-JEPA")
AND (masking OR predictor OR positional OR pooling OR "intermediate layer"
     OR "deep self-supervision")

("self-supervised" OR "masked modeling" OR "feature prediction" OR data2vec)
AND (skeleton OR "3D pose" OR gait)
AND (motion OR temporal OR graph OR collapse)

("multiple sclerosis" OR Parkinson* OR neurological OR "pathological gait")
AND (video OR markerless OR pose)
AND (classification OR severity OR diagnosis OR generalization)

("clinical gait" OR "pathological motion")
AND ("cross-domain" OR multi-site OR participant-disjoint OR leakage
     OR shortcut OR calibration OR uncertainty)
```

Also search target-output versus target-input masking, target-token positional identity, centered/sharpened latent cross-entropy versus regression losses, EMA half-life and update scaling, multi-clip/attentive/MIL pooling, intermediate-layer features, motion/bone/root streams, timestamp uncertainty, participant/site leakage, and external pathological-motion transfer.

Classify evidence before it enters the plan:

- **T1 direct peer-reviewed:** S-JEPA, clinical video gait, pathological gait, or directly applicable clinical-motion transfer.
- **T2 peer-reviewed mechanistic:** adjacent skeleton SSL, video JEPA, masked motion, pooling, positional encoding, or domain generalization.
- **T3 preprint or official implementation:** useful for current mechanisms and exploratory hypotheses, never represented as peer-reviewed.
- **T4 secondary/discovery:** surveys, blogs, snippets, and third-party summaries; never sufficient for an implementation claim.

Down-rank or exclude a clinical claim without a participant/group split; gait-identity work when anthropometry is its intended signal; a non-comparable modality or task without an explicit transfer mechanism; inaccessible private-data claims that cannot guide an actionable experiment; unverifiable full text; and duplicate arXiv/proceedings records. A source may motivate a mechanism without predicting an accuracy gain. Never compare absolute scores across datasets or convert action recognition, gait identity, severity estimation, or anomaly detection into an expected normal/MS/PD classification gain.

### Evidence record

For every source that affects an implementation or experiment decision, save a structured record:

```yaml
source_id: ""
citation: ""
authors: ""
venue_and_status: "peer-reviewed | accepted | preprint | dataset"
publication_or_revision_date: ""
doi_or_arxiv_version: ""
primary_url: ""
official_code_or_checkpoint: ""
code_commit_and_config_path: ""
access_date_and_content_hash: ""
task_and_modality: ""
input_representation: "RGB | 2D pose | 3D pose | point cloud | other"
dataset_scale_and_independent_unit: ""
pretraining_data_and_label_use: ""
split_and_evaluation: ""
method_detail_used_here: ""
optimizer_updates_batch_mask_and_ema: ""
reported_ablation_or_result: ""
exact_page_table_equation_or_config: ""
comparator_metric_seeds_and_uncertainty: ""
limitation_or_domain_gap: ""
contrary_or_null_evidence: ""
project_translation: ""
translation_strength: "direct | plausible | exploratory"
confidence: ""
verified_by: "agent name and date"
```

Link this source table to a separate claim ledger containing the precise claim, claim type, source IDs and locators, exact conditions, scope limit, counterevidence, repository translation, prerequisites, and falsification test. Also log each database, search date, query, inclusion/exclusion decision, and URL. Paraphrase sources; do not copy lengthy text. If two sources conflict, describe the conflict and resolve it through method/code inspection or keep it explicitly unresolved.

### Authoritative seed set to verify and extend

At minimum, verify the latest versions and official implementations of these works. Do not assume every later technique should be transferred to this project.

Core skeleton and latent prediction:

- I-JEPA, CVPR 2023, for foundational target-block and positional-conditioning mechanics: <https://openaccess.thecvf.com/content/CVPR2023/html/Assran_Self-Supervised_Learning_From_Images_With_a_Joint-Embedding_Predictive_Architecture_CVPR_2023_paper.html>
- S-JEPA, ECCV 2024 proceedings paper and author project page: <https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04755.pdf> and <https://sjepa.github.io/>
- MAMP, ICCV 2023: <https://openaccess.thecvf.com/content/ICCV2023/html/Mao_Masked_Motion_Predictors_are_Strong_3D_Action_Representation_Learners_ICCV_2023_paper.html>
- MotionBERT, ICCV 2023, for 2-D-to-3-D motion pretraining and a possible later external-motion branch: <https://openaccess.thecvf.com/content/ICCV2023/html/Zhu_MotionBERT_A_Unified_Perspective_on_Learning_Human_Motion_Representations_ICCV_2023_paper.html>
- General Feature Prediction, ICCV 2025: <https://openaccess.thecvf.com/content/ICCV2025/html/Sun_Towards_Efficient_General_Feature_Prediction_in_Masked_Skeleton_Modeling_ICCV_2025_paper.html>
- Adaptive Masked Reconstruction, CVPR 2026: <https://openaccess.thecvf.com/content/CVPR2026/html/Sun_Exploring_Adaptive_Masked_Reconstruction_for_Self-Supervised_Skeleton-Based_Action_Recognition_CVPR_2026_paper.html>
- Self-Supervised Representation Learning for Skeleton-Based Group Activity Recognition, ACM Multimedia 2022, retained as task-distant T2 evidence: <https://doi.org/10.1145/3503161.3547822>

Video JEPA and cross-modal latent targets:

- V-JEPA paper and official repository: <https://arxiv.org/abs/2404.08471> and <https://github.com/facebookresearch/jepa>
- V-JEPA 2 paper and official repository: <https://arxiv.org/abs/2506.09985> and <https://github.com/facebookresearch/vjepa2>
- V-JEPA 2.1 preprint and official code: <https://arxiv.org/abs/2603.14482> and the V-JEPA 2 repository above
- SV-data2vec, WACV 2025: <https://openaccess.thecvf.com/content/WACV2025/html/Dozdor_SV-data2vec_Guiding_Video_Representation_Learning_with_Latent_Skeleton_Targets_WACV_2025_paper.html>
- “Latent Video Prediction Learns Better World Models,” 2026 preprint: <https://arxiv.org/abs/2605.15618>. Treat its robustness claims as preprint evidence until venue status is verified.

Gait and clinical transfer:

- GaitForeMer, MICCAI 2022: <https://conferences.miccai.org/2022/papers/230-Paper0398.html> and <https://doi.org/10.1007/978-3-031-16452-1_13>
- FSGait, ACCV 2024: <https://openaccess.thecvf.com/content/ACCV2024/html/Duan_FSGait_Fine_Grained_Self-Supervised_Gait_Abnormality_Detection_ACCV_2024_paper.html>
- GaitPT scaling, AAAI 2026: <https://ojs.aaai.org/index.php/AAAI/article/view/37340>
- CARE-PD: A Multi-Site Anonymized Clinical Dataset for Parkinson's Disease Gait Assessment, NeurIPS 2025: <https://proceedings.neurips.cc/paper_files/paper/2025/file/bedc73979a95be7727af0c9a99c675ce-Paper-Datasets_and_Benchmarks_Track.pdf>; dataset DOI <https://doi.org/10.5683/SP3/TWIKMK>. Verify release/version, research-use terms, cohort overlap, and skeleton-mapping provenance before use.
- A Vision-Based Framework for Predicting Multiple Sclerosis and Parkinson's Disease Gait Dysfunctions, IEEE JBHI 2023: <https://doi.org/10.1109/JBHI.2022.3208077>
- Cross-Domain Self-Supervised Complete Geometric Representation Learning for Real-Scanned Point Cloud Based Pathological Gait Analysis, IEEE JBHI 2022: <https://doi.org/10.1109/JBHI.2021.3107532>. Treat it as contextual T2 evidence from depth/RGB-D-derived point-cloud pose estimation, not direct RGB-video JEPA classification evidence.
- Markerless Video-Based Gait Analysis in People With Multiple Sclerosis, IEEE TNSRE 2025: <https://pubmed.ncbi.nlm.nih.gov/40668714/> and <https://doi.org/10.1109/TNSRE.2025.3589765>
- Video-based 2D markerless gait analysis in people with multiple sclerosis, MSARD 2026: <https://doi.org/10.1016/j.msard.2026.107285>
- Self-Supervised Learning of Gait-Based Biomarkers, preprint: <https://arxiv.org/abs/2307.16321>

Unless a later official venue record is found, label V-JEPA, V-JEPA 2, V-JEPA 2.1, and the gait-biomarker work as preprints. Use newer papers only after confirming their date, status, code, task, data scale, and relevance. In particular, V-JEPA 2.1 dense/deep objectives, motion-adaptive skeleton masking, and million-scale gait results are later-stage hypotheses here. They do not outrank the immediate correctness repairs or justify copying large-data hyperparameters into a 35-group experiment.

Create `experiments/multiple-sclerosis/docs/LITERATURE_UPDATE.md` and a machine-readable evidence ledger under `experiments/multiple-sclerosis/artifacts/research/`. Include a “what transfers / what does not transfer” synthesis. If the user has not authorized downloading large datasets or checkpoints, research and document them without initiating a large download.

</fresh_literature_research_protocol>

<repository_audit_protocol>

Before editing, trace one batch end-to-end from cached sequence through tokenization, mask sampling, view encoding, target encoding, predictor inputs, loss, EMA update, embedding aggregation, probe, and metric. Record shapes and semantic meaning at each boundary. Inspect actual arrays and manifests, not just type hints or prose.

Reproduce or refute these suspected blocking defects with minimal deterministic tests:

1. **Missing target-position identity.** Hidden predictor slots may receive an identical mask token without predictor-space joint/time positions, making target predictions identical across hidden locations. Measure target-position standard deviation and test target-ID permutation sensitivity.
2. **Permanent anatomical-mask starvation.** The online encoder may never see the selected lower-body/shoulder joints, leaving their spatial-position rows without context gradients while downstream pooling focuses on them. Audit per-joint target/context exposure and gradients.
3. **Misdescribed class-aware VICReg.** Subtracting class means before a variance floor can encourage residual within-class spread and is invariant to class-center location. Write a small numerical test and correct notebook claims.
4. **Inadequate update and EMA scale.** Count optimizer updates and source exposures, derive teacher half-life, and measure target drift rather than quoting epochs.
5. **Continuation-state reset.** Confirm whether center, optimizer, schedule, EMA position, scaler, RNG, sampler, and global step reset across calls advertised as continuation.
6. **Source imbalance.** Quantify how overlapping windows and source duration affect sampling probability.
7. **Temporal and acquisition shortcuts.** Reproduce static/mean-std, confidence-only, root-only, single-frame, fps/domain, duration, time-shuffle, and time-reversal controls.
8. **Frame-rate and pose semantics.** Compare nominal versus actual sampled timestamps, source fps, MediaPipe image/video mode, missingness/interpolation, padding, and whether scale/root normalization erases speed or amplifies jitter. Search every augmentation for 3-D rotations or transforms that mix `[x,y,visibility]`; confidence must never be rotated as a spatial `z` coordinate.
9. **Grouping validity.** Audit duplicates, repeated clips/trials, source-to-person mapping, and any cross-notebook pretraining leakage.
10. **Metric parity.** Verify that RF and S-JEPA use identical group registries, training-only transformations, aggregation, labels, and OOF accounting.
11. **S-JEPA method fidelity.** The ECCV proceedings method starts with a 3-D skeleton sequence, forms geometric views, predicts the missing joints' latent representations from the same sequence, masks target-encoder outputs rather than inputs, and uses centered/sharpened target distributions with cross-entropy. Do not confuse the paper's discussion of MotionBERT prior art with its own objective; if an abstract or project page conflicts, privilege and document the full proceedings method. This repository uses cached `[x,y,visibility]`, may feed the same view to both lanes, and may use a different loss. Compare paper, official code if available, and repository tensor semantics line by line. State whether the repaired system is faithful S-JEPA, a 2-D same-modality skeleton JEPA variant, or S-JEPA-inspired. Do not introduce unreliable pseudo-3-D targets merely for naming fidelity; make target modality and loss a controlled, documented decision.

For every reproduced defect, first add a regression test that fails for the right reason. Avoid tests that merely assert loss decreases or teacher/student weights differ; collapsed and shortcut representations can satisfy both.

Treat existing tests as code under audit. In the current tree, smoke coverage may positively enforce the obsolete fixed 12-joint mask and merely exercise the class-aware VICReg path. Replace those expectations with the new invariants; do not keep them simply because they are green. Likewise, trace `scripts_capstone_check.py` and notebooks 03–06 for the old method after every API repair.

</repository_audit_protocol>

<phased_implementation_plan>

Implement in the following order. Do not retain a scientific choice because it improves an outer score. Mechanical correctness fixes are retained because they are necessary invariants.

### Phase 0: freeze provenance and reproduce E0

- Inventory videos, cached arrays, metadata, clips, provisional sources, and candidate participants.
- Hash or version the data manifest and caches. Record extraction code/config and actual timestamps.
- Export the current grouped fold registry and saved OOF probabilities as a versioned registry generation. Call it source-grouped unless participant identity is verified. If identity work later changes the grouping unit, create a new registry generation and rerun E0, RF, and every comparator used in a paired claim; never pair scores across registry generations.
- Recompute RF and current S-JEPA metrics from OOF predictions and record whether the historical result is exactly reproducible.
- Add run manifests with git revision/diff, cache version, group IDs, config, seed, updates, effective batch, examples and sources exposed, mask coverage, selection rule, environment, and output paths.
- Establish static-pose/body-proportion, order-insensitive moment, confidence-only, root-only, domain/fps/duration, time-shuffle, and time-reversal controls using inner-development partitions for all decisions. After the candidate is frozen, compute outer-CV control estimates once for reporting; do not feed them back into the current generation.

Gate: E0 provenance is explicit; data overlap is checked; baseline predictions are reproducible or discrepancies are explained; no result is mislabeled participant-disjoint.

### Phase 1: repair mathematical correctness

Implement predictor-dimensional factorized joint and temporal positional embeddings. A hidden predictor token should include the learned mask token plus the correct joint/time identity. Preserve explicit target indices across masks. Ensure context and padding masks reach attention correctly.

Replace the one-dimensional fixed mask with per-example multi-masks shaped conceptually as `(batch, masks, tokens)`. Start with fixed target counts if necessary for vectorization, expand batch across masks, pad variable context lengths, and use a key-padding mask. Build connected anatomical graph-time regions with contiguous temporal intervals. Mix uniform, higher-motion, and lower-motion regions so low-motion pathological signals are not ignored. Use clinical lower-body knowledge only as a modest target-sampling bias, initially about `1.5x`, while maintaining full context coverage. Sample two masks per sequence where memory permits.

Required tests:

- `test_hidden_target_predictions_have_position_identity`
- target-ID permutation changes the corresponding predictions;
- `test_mask_bank_meets_context_and_target_coverage`
- `test_every_joint_receives_context_gradient_over_mask_bank`
- `test_per_example_multimasks_and_context_padding_are_respected`
- output shapes and gather/scatter ordering remain correct for multiple masks;
- masks are deterministic under a saved seed and diverse across examples.

Initial mask gates over an epoch or sufficiently large deterministic bank:

- each joint is visible in at least 20 percent of masks;
- each joint is targeted in at least 10 percent;
- every joint-position row receives nonzero gradient over the bank;
- hidden-target prediction variance is finite and nonzero;
- permuting position IDs changes the mapped predictions.

Save coverage histograms and representative graph-time masks with each run.

### Phase 2: repair objectives, schedules, sampling, and state

- Remove class-aware VICReg from the strict SSL baseline and correct all notebook prose. If standard VICReg is retained, apply it to a disposable projection head across two views and compare it with no auxiliary loss.
- Pretrain strict S-JEPA on every source in the current training partition without diagnosis labels. Preserve normal-only SSL only as a separately named one-class/anomaly branch.
- Implement participant- or source-uniform sampling so long clips do not dominate; record exposure counts.
- Express warmup, total training, EMA, logging, and checkpointing in optimizer updates, not nominal epochs. Derive and report EMA half-life in steps. Choose a responsive teacher for the actual update budget.
- Persist and restore the online and target models, predictor, center, optimizer, scheduler, scaler, global step, EMA schedule position, RNG states, sampler state, and relevant running diagnostics.
- Support gradient accumulation and an explicit effective batch without changing source balance.

Required tests:

- `test_source_sampler_is_uniform_despite_window_count`
- `test_save_resume_preserves_center_optimizer_schedule_and_predictions`
- uninterrupted and save/resume runs match within documented deterministic tolerance;
- strict SSL consumes no diagnosis labels;
- teacher drift and student-teacher distance are finite and nontrivial;
- embedding per-dimension standard deviation, covariance, singular spectrum, and effective rank are logged.

### Phase 3: build and validate the corrected data lineage for R2

Keep the legacy cache immutable for the R1 mechanical comparison. Write corrected extraction outputs under a new cache/version ID; do not silently replace inputs beneath E0 or R1.

- Sample target times exactly from video timestamps rather than using an integer frame stride that makes 24 fps and 25 fps sources land at different effective rates while being labeled 15 fps.
- Prefer temporal/video tracking mode where supported. Record pose-model version and mode.
- Retain raw coordinates, confidence, and validity separately. Interpolate only short gaps and preserve missingness. Pass validity and padding masks through attention and pooling.
- Use geometry (`x,y`) as the primary input. Treat visibility as reliability or a controlled auxiliary stream, never depth.
- Compare robust sequence-level scale with the current per-frame normalization. Preserve a controlled root/translation/cadence stream because aggressive root centering can erase speed, while testing that it is not merely a camera shortcut.
- Add quality plots by group and domain: missingness, confidence, scale, fps, duration, view, task, and cadence proxies.
- Create explicit fields for participant, trial, source, site/domain, task, view, fps, duration, and identity confidence. Block participant-level labeling and aggregation if identity cannot be established; continue provisional source-grouped development work with honest naming.

Required tests:

- `test_timestamp_resampling_hits_requested_times`
- `test_padding_is_excluded_from_attention_and_pooling`
- `test_declared_evaluation_groups_are_disjoint`, using verified participants when available and otherwise provisional sources;
- a separate identity-status assertion that prevents a synthetic or unverified `participant_id` from being presented as verified;
- `test_visibility_is_not_named_or_used_as_depth`
- interpolation never bridges a gap longer than configured;
- all train-fitted transforms remain inside their partition.

### Phase 4: run `R1_repaired32`

This is a cumulative correctness baseline, not a causal attribution of every repair. Keep the architecture and simple readout close to the current system:

| Setting | Initial R1 value |
|---|---|
| Data/cache | versioned legacy cache for a mechanical comparison; original sampling/normalization retained and its limitations logged |
| Window/input | 32 frames and existing `[x,y,visibility-confidence]`; the third channel keeps its legacy numeric role but is never called depth or mixed into spatial rotations |
| Encoder | current 3 layers, width 96, 4 heads |
| Predictor | current size plus factorized predictor positions |
| Masks | stochastic graph-time, ratio 0.60, two masks/sample, clinical target bias about 1.5x |
| SSL data | all sources in the current training partition, labels hidden |
| Sampling | source-uniform |
| Auxiliary loss | none initially |
| Optimizer | AdamW, initial LR `3e-4`, weight decay about `0.04`, 10 percent warmup, cosine decay |
| EMA | responsive schedule, initially explore approximately `0.99 -> 0.9995`, justified by step half-life |
| Budget | learning curves at 300, 1,000, and 3,000 optimizer updates |
| Readout | frozen balanced linear probe with current mean aggregation |

Run a smoke seed first. Select update budget and checkpoint only on grouped inner data. Record train loss, inner performance, teacher drift, collapse diagnostics, source exposure, and mask coverage. Freeze the cumulative R1 configuration before any outer evaluation.

R1 is the cumulative E1-E6 repair. If a causal breakdown of E1-E5 is useful, replay it only as an optional inner-development ablation; do not create a second implementation path. If mechanically valid R1 fails the preregistered inner promotion threshold, stop local architecture scaling. Prioritize the corrected data lineage, participant/provenance work, and externally pretrained clinical motion rather than automatically proceeding through larger E7-E17 models.

### Phase 5: run `R2_local_gait_jepa` through sequential ablations

After R1 passes mechanical gates, test one factor family at a time. Avoid a crossed grid.

Initial search envelope:

| Area | Controlled choices |
|---|---|
| Resampling/input | exact 15 fps; `x,y` primary; confidence as reliability/ablation |
| Window | 64 frames, stride 32, validity/padding masks |
| Clips at inference | four distributed clips where duration allows |
| Tokenization | four-frame temporal groups initially |
| Encoder | 4 layers, width 128, 4 heads |
| Predictor | 3 layers, width 96 |
| Positions | factorized joint/time in encoder and predictor |
| Mask ratios | 0.40, 0.60, 0.75 inside inner folds; 0.90 only as a literature-faithfulness ablation |
| Masks | two per sequence; modest clinical-region oversampling |
| SSL budget | 1k, 3k, 5k; consider 10k only if inner learning curves justify it |
| Effective batch | target about 128 through accumulation if feasible |
| LR | small inner search among `1e-4`, `3e-4`, `5e-4` |
| Readout | balanced logistic regression; tune C and any PCA inside the inner folds only |

Continue the E1-E17 logic from `docs/02_NEXT_STEPS.md` at E7 because R1 already implements cumulative E1-E6:

7. 64-frame context;
8. multi-clip mean aggregation;
9. mean+standard-deviation and regional pooling;
10. learned-query/attentive pooling;
11. early, middle, and final-layer probes;
12. last-block supervised adaptation;
13. motion, bone/angle, confidence, and controlled root streams;
14. world/lifted-3D input if its quality is verified;
15. external gait/pathological-motion pretraining;
16. separately reported S-JEPA plus RF late fusion;
17. separately reported frozen RGB V-JEPA branch.

For screening, promote only a mechanically valid candidate with a predeclared, stable inner-fold gain, initially about `+0.05` pooled inner macro-F1 over rerun E0 and the strongest reproducible nuisance control, without material calibration or shortcut regression. Treat that value as a practical gate, not a p-value. Before seeing relevant results, also define a dynamics-claim threshold; use a `0.03` pooled inner macro-F1 degradation under temporal shuffle and a `0.03` advantage over the best static/nuisance control as the default heuristic. A model may still be useful if it misses this gate, but it cannot support a temporal-dynamics claim. Use at least three training seeds for early screening and five for the final promoted system when compute permits; prefer improvement in at least four of five preregistered final seeds. Log every attempted configuration, not only winners, and give RF and S-JEPA transparent, predeclared inner-search budgets with no post hoc budget expansion for the lagging branch. Report seed variance separately from fold-composition variance.

### Phase 6: genuine supervised adaptation

Compare clearly named regimes:

1. frozen encoder plus balanced linear probe;
2. last one or two encoder blocks plus readout trained with balanced cross-entropy;
3. full fine-tuning only after credible external pretraining and adequate regularization.

Use balanced cross-entropy first. If warranted, add exactly one supervised contrastive, center, or prototype objective as a separate ablation. Labels may influence only the labeled subset and inner-selection process. Use layerwise LR decay, early stopping inside inner folds, and small parameter-efficient heads where appropriate. Do not call this stage SSL or claim VICReg alone constitutes diagnosis fine-tuning.

### Phase 7: dynamics, external scale, and optional fusion

Only after the local baseline is valid:

- add velocity/motion, bone/angle, and controlled root/cadence streams one at a time;
- examine laterality and bilateral asymmetry without encoding camera/view identity;
- test reliable world landmarks or lifted 3-D only after quality, coordinate frame, and missingness audits;
- evaluate external CARE-PD or other clinical-motion adaptation only after license, consent, topology, label, participant-overlap, and domain audits;
- scale the encoder only when external data or inner learning curves justify it;
- evaluate RF fusion and RGB V-JEPA as separately labeled systems, with fusion parameters fitted inside training folds.

V-JEPA 2/2.1 inference may have heavy GPU and video-decoder requirements, especially on macOS. Detect compatibility before downloading checkpoints. A frozen feature branch with multi-clip attentive probing is preferable to attempting foundation-model pretraining on this dataset. Dense/deep self-supervision is an exploratory transfer idea, not part of the minimum corrected S-JEPA baseline.

</phased_implementation_plan>

<evaluation_firewall>

Create and enforce a locked, versioned nested-CV development protocol. It reduces new leakage but does not retroactively make this previously inspected collection confirmatory:

1. Verify participants and repeated trials. Use participant groups when verified; otherwise label all results provisional source-grouped.
2. Create a locked grouped outer-fold registry and grouped inner folds for each registry generation. Within a generation, RF and every S-JEPA comparator use the identical raw cohort, declared evaluation unit, fold assignments, and aggregation policy. Branch-specific features may differ, but every fitted transform remains training-only. If participant mapping changes, version a new registry and rerun all paired comparators.
   Before scores are visible, record each branch's maximum candidate configurations, inner fits, seeds, and compute allowance. The budgets need not be numerically identical because training costs differ, but both must be explicit, defensible, and fixed rather than expanded only for the lagging branch.
3. Within inner selection, SSL sees only inner-training sources. It must not see inner validation or outer test clips even unlabeled for the inductive result.
4. After selecting a configuration, refit it on all outer-training groups and evaluate the frozen preregistered configuration once on each outer-CV fold. Call the pooled result an internal/development estimate, not confirmation.
5. Fit normalization, imputation, PCA, feature selection, calibration, classifier hyperparameters, thresholds, pooling choices, and checkpoints using training/inner data only.
6. Aggregate window to clip and clip to the declared independent unit. Use verified participants and equal participant weight when identity is established; otherwise aggregate and weight provisional sources and say so in every table. Document how multiple views and trials are combined.
7. Save raw OOF rows with run/config ID, seed, outer fold, evaluation-unit ID, source and participant IDs, true label, ordered class probabilities, and prediction. There must be exactly one probability vector per evaluation unit after the declared aggregation.
8. Report declared-unit pooled macro-F1 as primary. Also report balanced accuracy, ordinary accuracy, per-class precision/recall/F1, PD recall, count and normalized confusion matrices, one-vs-rest AUROC/AUPRC where estimable, Brier score/calibration, and paired group-bootstrap intervals and deltas to RF and rerun-current S-JEPA.
9. For the final system, average or ensemble the preregistered seed probabilities before the paired group comparison; do not count seeds as copies of an evaluation unit. Report seed dispersion separately. Use a fixed-seed paired, class-stratified bootstrap of whole verified participants, or whole provisional sources while identity is unresolved, with at least 10,000 replicates, a declared interval estimator, and explicit handling/sensitivity reporting for class-missing ordinary-bootstrap replicates. Never resample windows, clips, folds, or seeds. State that an interval over fixed OOF predictions conditions on the fitted models, selected configuration, and fold registry; it is not full retraining or model-selection uncertainty. For accuracy, use a preregistered paired discordant-case analysis where its assumptions hold. Avoid unsupported claims of significance on this development collection.
10. For label efficiency, draw several group-stratified labeled subsets at 10, 25, 50, and 100 percent. Unlabeled outer-training clips may support SSL; labels and diagnosis-aware losses may use only selected groups. Require at least two labeled training groups per class and enough groups for the inner split; omit infeasible fractions rather than silently rounding upward.
11. Never use test-label silhouette or a visually attractive UMAP to select a model. Visualizations are diagnostic only.
12. Any change made after viewing an outer-CV result starts a new explicitly exploratory generation with a logged reason. A discovered implementation bug invalidates affected results and outputs. Only a never-inspected external cohort can support a future confirmatory assessment.

Call a system “internally competitive with RF” only if its frozen, paired pooled OOF macro-F1 is at least as high as the RF rerun on the same raw cohort, registry generation, declared evaluation unit, and aggregation, with both branches following predeclared train-only tuning policies and reported search budgets. Do not require identical feature preprocessing, and never compare with the historical mean-of-fold RF `0.668`. Treat PD recall of at least `0.60` as a useful target, not a guaranteed criterion. Apply the preregistered nuisance and temporal thresholds from inner development unchanged. Reversal is diagnostic rather than mandatory because periodic gait can be reversal-insensitive. Static pose, body proportions, confidence, source, view, fps, duration, or root-only controls must not fully explain a clinical-representation claim.

</evaluation_firewall>

<codex_adversarial_review_contract>

Use actual Codex as an independent adversarial reviewer. Do not simulate a Codex review or allow the authoring agent to self-certify.

First detect the available integration. If a repository-installed `codex:adversarial-review` mechanism exists, inspect its help and use it. From a shell, run `codex review --help` and invoke only a supported noninteractive review form, such as an available uncommitted or task-patch review. Inside the Codex app, `/review` is an interactive alternative; do not present it as a universal shell command. If needed, run a separate Codex session in a fresh context.

Codex is read-only with respect to the user's working tree. Materialize each task-only candidate and all mutating checks in a throwaway copy or explicitly supported `--check`/`--output-dir`; Codex returns a report and the integrator archives it. Do not aim an uncommitted-tree review blindly at the dirty repository because it can mix unrelated user changes into the packet. If Codex is unavailable, an independent frontier reviewer may provide interim value, but final Codex signoff is `BLOCKED: Codex unavailable` unless the user explicitly changes the requirement. Never impersonate Codex or claim that fallback review is Codex.

At preflight, record the base SHA, initial status, and content hashes. A reconstructable task-only snapshot consists of the base SHA, a binary tracked-file patch, and relative path/mode/content hashes plus archived contents for every task-owned untracked file. Materialize that snapshot into a temporary review copy. A bare `git diff` or diff hash is insufficient because it omits untracked content.

Codex review references:

- OpenAI Codex best practices: <https://learn.chatgpt.com/guides/best-practices.md>
- Codex subagents: <https://learn.chatgpt.com/docs/agent-configuration/subagents.md>
- Codex code review: <https://learn.chatgpt.com/docs/code-review.md>

Use a fresh Codex context for each round so earlier author explanations do not anchor the review:

1. **AR-0, pre-implementation threat model.** Assemble its review packet in memory or a unique temporary directory, not tracked project files. Review the literature/claim ledger, experiment registry, stage-by-stage data and label access table, proposed APIs, acceptance thresholds, base snapshot, causal ablations, compute feasibility, and leakage firewall. Try to find outer-test reuse, labels entering SSL, participant/source ambiguity, multiple-comparison leakage, paper-to-project overclaims, and unbalanced RF/S-JEPA search budgets. AR-0 closure means each design P0/P1 has a corrected specification, an explicit blocked decision, or an owned remediation plus failing acceptance test assigned to AR-1/AR-2; it does not require implementation evidence before implementation begins.
2. **AR-1, core JEPA mechanics.** After model, mask, and state code lands, run the author tests plus independent ephemeral micro-tests. Verify predictor joint/time identity and aligned target ordering; position-ID permutation sensitivity; stop-gradient target encoder excluded from the optimizer; the EMA formula and half-life; truly per-example `(B,M,N)` masks with promised disjointness/connectivity/temporal contiguity; all-joint exposure and gradients; label-permutation invariance of strict SSL after the registry/sample order is held fixed; padding-value invariance of valid features/readout; source-uniform exposure; and uninterrupted versus save/resume next-loss and prediction equivalence.
3. **AR-2, data, split, and evaluation audit.** Independently parse manifests, folds, and outputs without importing the evaluation helpers under review. Test timestamp sampling at representative 24, 25, 29.97, and 60 fps; persisted actual timing; short-gap-only interpolation; validity/padding propagation; confidence naming; inner sets as strict subsets of outer training; group disjointness; train-only fitting of scalers/PCA/pooling/stopping/probes; valid probability class order/sums; one OOF record per evaluation unit; paired group bootstrap; and exact RF/S-JEPA registry parity. Replace forbidden outer labels with sentinels and confirm they cannot change checkpoints or probabilities.
4. **AR-3, notebook and claim parity.** In the throwaway candidate copy, regenerate all notebooks twice to unique output directories and require the second generation to be byte-stable. Parse/execute smoke cells with isolated run directories and trace notebooks 03–06 plus `scripts_capstone_check.py` into the repaired APIs. Flag any active fixed-mask, normal-only classification pretraining, class-aware VICReg, test-label silhouette, invalid label-efficiency sweep, stale README/slides/progress claim, visibility-as-depth language, participant-disjoint overclaim, or uncited literature statement. The global tutorial checkpoint from notebook 03 must never enter a scored holdout.
5. **AR-4, result red team.** Only after inner selection is frozen, recompute every metric from raw OOF rows with an independent implementation. Verify data/fold/config/diff hashes, all preregistered seeds rather than a favorable subset, probability ordering, count and normalized confusion matrices, per-class metrics, calibration, seed ensembling, and paired uncertainty. Audit single-frame/body-proportion, pose moments, duration/fps/domain, confidence-only, root-only, temporal shuffle/reversal, same-source-nearest-neighbor, and random-label controls. A missing temporal-shuffle degradation blocks a dynamics claim, while a nuisance control matching the model blocks a clinical-representation claim; neither may be hidden behind UMAP.
6. **AR-5, clean-room signoff.** Freeze the candidate boundary: code, configuration, dependency lock, data/cache hashes, registry, generated notebooks, claim-bearing prose, and claimed OOF/result hashes. Materialize the reconstructable snapshot in a clean temporary environment without altering user work. Install from the lockfile, run complete tests, verify generator idempotence, run all smoke notebooks/pipeline with isolated outputs, exercise interruption/resume, independently reproduce at least one locked fold/seed, and recompute stored OOF metrics. Use a detached worktree or candidate commit only if the user has authorized that operation. Emit `PASS` or `BLOCKED` with an issue ledger. Append-only review and replication evidence does not invalidate the candidate; a change to the frozen boundary, selection, or claim does.

Codex must receive independent evidence, not only author summaries: base and candidate revision plus diff hash; `git diff --name-status`; config, lockfile, data-cache, manifest, and fold-registry hashes; raw OOF rows; per-run update/effective-batch and source-exposure counts; mask/gradient/position diagnostics; effective rank/singular values; teacher drift and EMA half-life; selection chronology; hardware/software provenance; and the primary-source claim ledger. The metric/split audit must not call the code under audit. Codex should rerun smoke plus one locked fold/seed even when the implementer supplies full expensive logs.

Give Codex this review instruction, adapted with the current phase packet:

```text
Act as a hostile but evidence-driven ML methods and code reviewer. Review only; do not modify files. Try to falsify the claimed correctness and result. Prioritize:

P0: leakage, invalid independent unit, mathematical defect, incorrect target/mask alignment, fabricated or stale output, outer-test-driven choice, destructive loss of user work, or a metric that cannot be reproduced. It blocks all claims and cannot be waived.
P1: failed required invariant, active obsolete method, state-continuation/data/evaluation defect, notebook/code divergence, or major reproducibility failure. It blocks promotion and cannot be waived.
P2: missing control, incomplete uncertainty/reporting, stale or overstated claim, or important maintainability issue. A P2 touching the central scientific claim blocks that claim until resolved.
P3: low-risk naming, efficiency, maintainability, or optional diagnostics that may enter a visible backlog.

For every finding return: severity; exact file:line or artifact; evidence/reproduction; scientific impact; minimal corrective direction; and the regression test or independent calculation that should catch it. Check for contrary evidence and avoid stylistic findings unless they affect comprehension or reproducibility. Do not recommend a method because it performed better on outer-test data. Explicitly state what you inspected and what you could not verify.
```

Each review packet must include the phase goal, hypothesis ledger, primary-source links, relevant diff, tests with exact output, configuration and split registry, run manifest and OOF predictions when applicable, known risks, and unresolved questions.

Each issue must include an ID, review round, severity, violated invariant/claim, candidate hash, exact file:line or artifact, minimal reproducer, expected versus actual behavior, scientific impact, required remediation, and regression test.

Triage every finding as `accepted`, `rejected with evidence`, `duplicate`, or `deferred with owner/rationale`. “Disagree” is not closure. P0 and P1 cannot be waived. Assign a narrow fix owner, add a test that fails before and passes after the fix where feasible, and request a new blind Codex context for the affected round. Before candidate freeze, normal inner experiments do not trigger AR-4/AR-5. After freeze, a core change reruns AR-1 onward; a data/evaluation change reruns AR-2 onward; a notebook claim change reruns AR-3 and AR-5; a protocol or scientific-claim change reruns AR-0; and a frozen config, threshold, selection, or claimed-result change reruns AR-4 and AR-5. Repeat until there are zero P0/P1 findings and no claim-relevant P2. Codex returns reports; only the integrator writes the archived prompts, findings, dispositions, and rerun evidence under `artifacts/reviews/`.

</codex_adversarial_review_contract>

<notebook_refinement_contract>

Keep the seven-notebook learning progression, but make the narrative match the repaired implementation:

- `00_overview_and_video_gallery.ipynb`: dataset card; provenance and license status; video/gallery inspection; participant/source/trial distinction; domain, view, fps, duration, quality, and imbalance tables; explicit exploratory-use warning.
- `01_pose_extraction_from_raw_video.ipynb`: exact timestamp resampling; tracking mode; confidence and missingness semantics; short-gap interpolation; validity masks; robust normalization alternatives; cache version and quality control.
- `02_anatomical_mask_and_tokenization.ipynb`: retain the de-duplicated clinically relevant-joint table as domain context, but teach stochastic graph-time masking, clinical target bias, mask coverage, context/target rotation, padding, and position identity. Explain why permanent hiding was harmful.
- `03_sjepa_model_and_pretrain_normal.ipynb`: show the two encoder lanes, stop-gradient, predictor positions, per-example masks, centering/EMA by optimizer step, collapse diagnostics, and full state resume. Present all-training-source label-free SSL as the classification baseline; label normal-only pretraining as a separate one-class tutorial.
- `04_progressive_finetune_ms_pd_vicreg.ipynb`: rename or clearly retitle the conceptual stage if filename stability is important. Remove claims that current class-aware VICReg compacts diagnoses. Compare strict SSL continuation, balanced supervised adaptation, and any auxiliary loss honestly.
- `05_representation_visualization.ipynb`: layerwise and regional features; train/inner-only model selection; static, shuffled, reversed, confidence, root, domain, and source probes; effective rank and covariance; UMAP/silhouette as diagnostics rather than evidence.
- `06_capstone_rf_vs_sjepa.ipynb`: locked nested grouped protocol; identical RF/S-JEPA fold registry generation and aggregation; R0/R1/R2 configs; OOF probabilities; paired bootstrap at the declared group unit; calibration; confusion matrices; PD recall; correct label-efficiency study; pure S-JEPA separate from supervised/fusion/RGB branches.

Each notebook must be understandable, executable locally with `uv`, and honest about Colab requirements. Do not repeatedly install dependencies in a way that breaks the active environment. Keep cells reasonably small and deterministic. Add a fast smoke mode, make full-run compute explicit, and route all writable outputs through the unique run directory. Every table/figure must be generated from a versioned artifact or visibly marked illustrative. Keep vector diagrams uncluttered and subject them to visual review; do not spend implementation time on decorative graphics before scientific gates pass.

Update package code and tests first, then `notebook_content.py`. Before touching canonical notebooks, add or use generator `--output-dir`/`--check` support and generate into a temporary directory. Compare each current `.ipynb` with generated content, identify user-owned manual divergence and meaningful outputs, and port intentional source changes into `notebook_content.py`. Only the integrator then performs one canonical content generation, followed by one controlled no-change regeneration/idempotence check. Verify the notebooks contain the intended cells and no stale claims. Do not hand-edit generated JSON to fix source bugs or overwrite unreconciled user changes.

</notebook_refinement_contract>

<validation_and_compute_policy>

Discover the exact project commands, then use the canonical equivalents of:

```bash
cd experiments/multiple-sclerosis
# Only after an intentional dependency change: update the lock once and review its diff.
uv lock
uv sync --frozen
SJEPA_SMOKE=1 uv run pytest sjepa/tests -q
uv run python scripts_build_notebooks.py
SJEPA_SMOKE=1 uv run python scripts_capstone_check.py
```

If dependencies did not change, do not rewrite the lockfile; begin with `uv sync --frozen`. If they did, only the integrator runs `uv lock`, reviews the `pyproject.toml`/`uv.lock` diff, and then runs the frozen sync. Add focused unit tests for every repaired invariant and run them before broader tests. Add development dependencies such as `pytest`, `nbformat`, or `nbclient` to the project configuration/lockfile before relying on their commands. Validate notebook JSON, imports, links, and cell-source parity. Execute notebooks in smoke mode in dependency order if the environment supports it. Run the capstone end-to-end on real cached data at least in smoke/R1 mode; do not report a full R2 result if it was not actually executed.

Create stable validation interfaces if they do not exist, such as a notebook smoke validator, a run/fold/OOF auditor, and bounded `--fold-limit`, `--step-limit`, and unique `--output-dir` options for the capstone checker. The current script may not expose those flags, so implement and test them before using them in evidence. Do not pretend a command exists. Make notebook generation idempotence testable by hashing outputs, generating a second time, and asserting no second-pass change; use `git diff --exit-code` only against a deliberately clean candidate where intended notebook changes have already been captured.

As a targeted stale-path audit, inspect active matches of:

```bash
rg -n 'class_aware_vicreg|pretrain_epochs // 4|finetune_epochs // 4|silhouette\(np\.vstack|normal_tr' \
  notebook_content.py scripts_capstone_check.py sjepa
```

A match is a review target rather than an automatic failure when it appears in a regression test or clearly archived history. It is a failure if an active scored path still uses the obsolete behavior.

Before expensive work, inventory device, memory, disk, expected wall time, and checkpoint/data sizes, then record hard limits for wall hours, accelerator-hours, disk growth, training trials, seeds, and external-download size. Use the operator's limits when supplied. If none are supplied and the operator does not answer a budget request, default to 8 wall hours, 8 available accelerator-hours (zero when no accelerator exists), 20 GB disk growth, 24 training trials, and no single external download over 1 GB. The mandatory minimum is focused tests, generator/notebook smoke validation, and one bounded real-cache fold/seed run; a full R1/R2 evaluation may be handed off as pending when it cannot fit.

Use a small deterministic smoke run, then one fold/seed, then inner-fold successive halving. Before launching a job, estimate whether it fits the remaining budget. Parallelize independent read-only work and safe fold/seed runs with unique output directories, not overlapping writes. Checkpoint long jobs and communicate progress at least once per hour. When the budget is exhausted, stop launching work, preserve valid checkpoints/manifests, provide exact continuation commands and estimates, and label every unexecuted result pending. Do not silently reduce updates while retaining a full-run label.

Do not mutate raw videos. Treat caches as versioned derivatives. Do not download large external datasets or checkpoints, accept licenses, upload data, or send external messages without authorization. Small paper and metadata downloads for research are allowed if the environment policy permits them.

</validation_and_compute_policy>

<required_artifacts>

Create or update, using names consistent with the repository:

- corrected package source and focused tests under `experiments/multiple-sclerosis/sjepa/`;
- updated `notebook_content.py` and regenerated notebooks;
- `docs/LITERATURE_UPDATE.md` with verified source status and transfer analysis;
- a concise implementation/decision log linked to `docs/02_NEXT_STEPS.md` rather than duplicating it blindly;
- a versioned source/participant/trial/domain manifest and versioned fold-registry generations;
- machine-readable research evidence, hypothesis/experiment ledgers, run manifests, and OOF probabilities;
- mask coverage, source exposure, teacher drift, collapse/effective-rank, shortcut-control, and evaluation artifacts;
- Codex review packets, findings, dispositions, and rerun evidence;
- exact commands and environment metadata needed to reproduce each reported result.

Choose stable paths under `artifacts/` and avoid checking in unnecessarily large binary checkpoints unless repository policy calls for them. Use config/run IDs that reveal generation, fold, seed, and configuration hash.

</required_artifacts>

<done_when>

Report the highest achieved completion tier rather than treating every expensive branch as mandatory:

1. **Core-correction complete:** critical mechanics, state, data/evaluation interfaces, focused tests, run provenance, and notebook source parity are repaired and smoke-validated.
2. **R1 evaluation complete:** cumulative E1-E6 R1 has inner selection plus one frozen nested-CV development evaluation on the current collection, with paired RF and controls.
3. **Accuracy-program complete:** every budget-approved R2/supervised/fusion branch has either passed its gate and been evaluated or is explicitly rejected/deferred with evidence. E13-E17 are not mandatory when gates, license, data, or compute block them.
4. **Confirmatory evaluation complete:** a frozen candidate has been evaluated once on a genuinely never-inspected external participant cohort. This tier cannot be achieved by repartitioning the current collection.

Do not declare the claimed tier until all applicable items are evidenced:

- the current baseline and critical defects were independently reproduced or explicitly refuted;
- predictor target positions have identity and their invariant tests pass;
- stochastic per-example multi-masks meet context/target coverage and gradient gates;
- strict SSL is diagnosis-label-free and class-aware VICReg is removed from that baseline;
- source exposure is balanced and schedules/EMA are justified in optimizer steps;
- save/resume preserves complete training state and reproduces predictions within tolerance;
- timestamp, confidence, missingness, padding, and normalization semantics are tested;
- participant identity status and split terminology are honest, with no overlap at the declared group unit;
- for the R1 tier, E0 and `R1_repaired32` have reproducible manifests and OOF probabilities, or the handoff is explicitly blocked/pending rather than called R1-complete;
- every promoted choice was made using inner data before the once-per-generation outer-CV development evaluation;
- RF and S-JEPA use the same registry generation, raw cohort, declared evaluation unit, and aggregation;
- shortcut and temporal controls are reported alongside headline metrics;
- notebooks were regenerated from `notebook_content.py`, smoke-tested, and checked for stale or fabricated output;
- independent metric recomputation matches the notebook/report;
- actual Codex completed the applicable rounds with no unresolved P0/P1 findings and no claim-relevant P2; Codex unavailability is a blocked signoff, while an alternate review remains clearly interim;
- no unrelated user changes were reverted or overwritten;
- the final report distinguishes implemented, executed, supported, negative, pending, and deferred work.

On the already inspected collection, say “numerically improved internal estimate,” “promising,” or “no demonstrated gain” as warranted, never “significantly improved” in a confirmatory sense. Reserve that wording for a preregistered frozen comparison on a never-inspected external participant cohort with an appropriate paired analysis and uncertainty. If the internal result does not improve, complete the mechanically correct tier, report the negative result, identify the best-supported next bottleneck, and do not tune against the outer-CV folds.

</done_when>

<final_report_contract>

Return a concise executive summary followed by auditable evidence:

1. completion tier, outcome, and whether the gain is only internal/exploratory or externally confirmatory;
2. files changed and why;
3. literature decisions, including techniques rejected or deferred;
4. defects reproduced and fixed, with regression tests;
5. data/split status and leakage firewall;
6. configurations actually run, compute, and exact commands;
7. RF, current S-JEPA, R1, R2, supervised, and fusion results in separate rows with paired uncertainty;
8. shortcut/temporal controls and limitations;
9. Codex findings and dispositions;
10. pending expensive runs and the single highest-value next action.

Use direct links to local files and primary sources. Never present an unexecuted configuration as a measured result. Never hide that a result is source-grouped rather than participant-grouped. End with a reproducibility command sequence and a list of artifact paths.

</final_report_contract>

Begin now with read-only preflight, primary-source research, and dynamic subagent fan-out. Use read-only or ephemeral checks to reproduce the critical failures. Do not edit tracked files until the lead has synthesized the evidence, defined file ownership, and passed Codex AR-0. If Codex is unavailable, return a blocked preflight and request direction rather than substituting another model silently.

## END MASTER PROMPT

---

## Tooling notes for the operator

The master prompt is intentionally capability-adaptive. Claude Code documents custom subagents as isolated workers that report back to the parent, while agent teams and worktrees are separate mechanisms with different coordination and file-collision tradeoffs. The prompt therefore makes the parent own fan-out and uses worktrees only for truly independent writers. See the official [Claude Code subagent documentation](https://code.claude.com/docs/en/sub-agents), [agent comparison](https://code.claude.com/docs/en/agents), [agent-team documentation](https://code.claude.com/docs/en/agent-teams), and [hooks documentation](https://code.claude.com/docs/en/hooks).

Likewise, `codex:adversarial-review` may be a local integration rather than a portable built-in command. The prompt requires `codex review --help` capability detection for shell use, recognizes `/review` as an interactive Codex-app surface, and otherwise uses a separate read-only Codex session. This prevents a model from claiming that a review occurred when no Codex surface was available.
