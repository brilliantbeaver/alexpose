# Three-week GAVD5 research portfolio

**Decision date:** 15 August 2026
**Experiment window:** 16 August to 5 September 2026
**ICLR 2027 deadlines:** abstract 18 September 2026, paper 25 September 2026, both Anywhere on Earth

## Recommendation

Run proposal 05, the **temporal readout diagnostic**, as the primary study. Build proposal 01's source-video-disjoint evaluation harness first and include proposal 06's missingness and provenance controls. This produces one coherent paper question:

> Does the current mean-and-standard-deviation readout discard temporal order that is still present in frozen S-JEPA tokens?

This is the strongest three-week direction because the motivating fact is exact, not speculative. A mean and a standard deviation do not change when time segments are permuted. The current pooled readout therefore cannot retain temporal order by construction. The experiment can hold the encoder and token tensor fixed, vary only the readout, match parameter counts, and test timing-sensitive gait scalars under a strict source-video split.

Proposal 04, the position-versus-motion target ablation, is the best backup. It asks a clean method question but requires controlled retraining. Proposals 03, 06, and 07 are valuable mechanism and validity audits. Proposal 02 is useful supporting analysis, not a lead machine-learning paper. Proposal 01 is mandatory infrastructure but cannot support a broad clinical anomaly-detection claim because canonical normal gait comes from one source video.

Do not run all seven as one paper. Choose one primary mechanism, use the shared validity checks, and stop expanding scope after the Day 5 gate.

![How the seven proposals fit together](01-honest-video-disjoint-anomaly-screening/images/04_portfolio_hub.svg)

## What notebooks 00 to 06 establish

The table separates saved evidence from interpretation. Full citations and artifact cautions are in [`_shared/evidence-ledger.md`](_shared/evidence-ledger.md).

| Notebook | What was implemented or measured | What the evidence supports | What it does not support |
|---|---|---|---|
| 00 | A compact S-JEPA with 33 joints, 16 four-frame time patches, 528 tokens, an online encoder, EMA target encoder, predictor, latent cross-entropy, and VICReg | The learning graph runs and avoids obvious numerical failure in the tutorial checks | A clinical model or a future-predictive world model |
| 01 | A canonical cohort of 96 sequences from 18 source videos, plus 63 accepted added-normal sequences from 17 videos | Exact source and provenance accounting | Treating 96 clips as 96 independent people |
| 02 | MediaPipe Pose Landmarker Lite extraction with 33 landmarks and explicit validity masks | Pose coverage and failure patterns can be audited | Markerless coordinates are not motion-capture ground truth |
| 03 | Uniform masking over 12 selected landmarks with a nominal 0.60 eligible-token target | The masking contract is explicit and testable | Motion-aware masking, since displacement and velocity are not used |
| 04 | A five-stage curriculum with 159 sequences, 600 epochs, and 11,400 updates | Final feature standard deviation 0.414 argues against total constant collapse; normal-anchor cosine falling to 0.594 shows drift | Pure self-supervision after Stage 0, since Stages 1 to 4 use condition labels |
| 05 | A 384-dimensional mean/std pooled representation | Cosine silhouette 0.009 and minimum centroid distance 0.0367 do not show clean five-condition geometry | Temporal-order preservation or unseen-source separation |
| 06 | Random Forest readouts and missingness-only controls | Labels are decodable inside the known corpus | Inductive generalization, since the encoder saw every evaluation row and sequence splits reused source videos |

### The five claim boundaries

1. The independent unit is the **source video**, not the extracted clip.
2. A held-out probe split is still transductive if the encoder saw that video's clips.
3. Variation across training seeds does not measure variation across source videos.
4. The label-aware group loss makes Stages 1 to 4 supervised representation fine-tuning.
5. Folder names such as `stroke` and `parkinsons` are dataset annotations, not diagnoses made by this project.

## Ranked directions

The proposal number is a stable folder identifier. The scientific rank below reflects expected value under a three-week deadline. Ratings are prospective reviewer readiness on the current evidence, not predicted acceptance scores.

| Rank | Proposal | Core question | Three-week feasibility | Main reviewer risk | Role |
|---:|---|---|---|---|---|
| 1 | [05: Temporal readout diagnostic](05-representation-vs-readout-diagnostic/README.md) | Does a capacity-matched temporal readout recover timing and asymmetry information that mean/std pooling removes? | High | A larger temporal head would be an unfair comparison | Primary paper |
| 2 | [04: Position versus motion targets](04-motion-vs-position-target-ablation/README.md) | Does predicting motion-derived targets improve held-out-source recovery of pre-registered gait scalars? | Medium-high | Target scale or difficulty, not semantics, could explain the effect | Best backup |
| 3 | [01: Strict inductive evaluation repair](01-honest-video-disjoint-anomaly-screening/README.md) | Which current findings survive when source videos are excluded before encoder training? | High for infrastructure, medium for a paper | Only one canonical normal source prevents broad anomaly claims | Required foundation |
| 4 | [03: Readout-aligned collapse audit](03-sigreg-effective-rank-audit/README.md) | Do token-level health metrics hide low sequence-level effective rank after pooling? | High for audit, medium for remedy | A repository bug report alone is not significant | Mechanism study |
| 5 | [06: Missingness and provenance audit](06-visibility-channel-confound-audit/README.md) | Can detector validity or extraction pathway predict source and condition without gait geometry? | High | A near-chance null is too narrow by itself | Mandatory control |
| 6 | [07: Selective invariance stress test](07-viewpoint-sensitivity-audit/README.md) | Is the representation stable to camera-like transforms while remaining sensitive to anatomical left-right changes? | Medium-high | Synthetic transforms are not real camera viewpoints | Stretch mechanism |
| 7 | [02: Literature-grounded scalar audit](02-clinical-threshold-calibration-audit/README.md) | Can decoded scalars be mapped to defensible clinical constructs without claiming population validity? | High | One normal source and transductive training block inferential clinical claims | Supporting analysis |

## Why these seven survived the idea audit

The audit covered the complete local trees under `/Users/theodoremui/dev/worldmodels/gait/` and `/Users/theodoremui/dev/worldmodels/wiki/`. Repeated ideas were merged by the experiment that would actually distinguish them. The final set favors questions that use existing pose tensors, checkpoints, and evaluation code; isolate one factor; have a useful null result; and can finish by 5 September.

The source collections contributed several attractive longer-term themes: predictive-error anomaly detection, cross-view invariance, hierarchical phase, concept bottlenecks, paired transformations, multimodal teachers, personal baselines, physics constraints, and planning. They were not copied directly. Three-week feasibility and identifiability changed the selection:

- Generic anomaly detection became a strict inductive evaluation repair because rarity is not pathology.
- Cross-view learning became a bounded transform stress test because no paired multi-view cohort is available locally.
- Concept bottlenecks became a descriptive construct audit because the current scalars and labels are not clinically validated.
- World-model, control, and intervention proposals were deferred because the notebooks have no action-conditioned transition model, rollout test, reward, or planning task.
- Unsupervised disentanglement and invariant risk minimization were deferred because the required factors and independent environments are not identifiable in this cohort.
- Personal normative modeling was deferred because repeated longitudinal sessions per person are missing.

## Reviewer rubric used for selection

Current official guides converge on the same core dimensions:

- ICLR asks whether the problem is specific, the method is motivated in prior work, the claims are supported rigorously, and the work contributes useful new knowledge. State-of-the-art performance is not required. See the [ICLR 2026 Reviewer Guide](https://iclr.cc/Conferences/2026/ReviewerGuide).
- ICML scores soundness, presentation, significance, and originality separately. It explicitly recognizes insight from evaluating existing methods as a valid form of originality. See the [ICML 2026 Reviewer Instructions](https://icml.cc/Conferences/2026/ReviewerInstructions).
- NeurIPS scores quality, clarity, significance, and originality. A negative result must change understanding through careful analysis, not merely report a failed run. See the [NeurIPS 2026 Reviewer Guidelines](https://neurips.cc/Conferences/2026/ReviewerGuidelines).

Each proposal was screened against seven questions:

1. Is the question falsifiable in one sentence?
2. Is the source video treated as the independent unit before all fitting?
3. Does the experiment change only the named factor, or include a control for every extra change?
4. Is there a simple non-neural or nuisance baseline?
5. Would a null result rule out a plausible belief?
6. Can the decisive figure be produced by Day 14?
7. Does the claim matter beyond this repository?

[`_shared/reviewer-scorecard.md`](_shared/reviewer-scorecard.md) records the prospective scores, fatal concerns, and required repairs.

## What outstanding papers teach this project

The useful pattern is not model size. It is a crisp assumption, a decisive test, a mechanism, and clear claim limits.

- The ICLR 2026 Outstanding Paper *LLMs Get Lost In Multi-Turn Conversation* was recognized for exposing a gap between common evaluation and real deployment with scalable, careful experimental design. The analogous move here is to expose the gap between sequence-level evaluation and unseen-source use. See the [official ICLR announcement](https://blog.iclr.cc/2026/04/23/announcing-the-iclr-2026-outstanding-papers/).
- The ICML 2019 Outstanding Paper *Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations* paired a precise assumption challenge with controlled evidence. Here, the exact assumption is that a global mean/std readout adequately measures temporal representation quality. See [Locatello et al.](https://proceedings.mlr.press/v97/locatello19a.html).
- ICML 2026 selected papers partly for longevity beyond a narrow subcommunity. A GAVD5 result must therefore produce a reusable evaluation lesson for small, grouped skeleton datasets. See the [official ICML awards process](https://blog.icml.cc/2026/07/05/announcing-the-icml-2026-awards/).
- NeurIPS 2025 recognized systematic negative results and evaluation work that changed practice. A null here matters only if it survives strict splits, matched controls, source-level sensitivity, and mechanistic probes. See the [official NeurIPS awards announcement](https://blog.neurips.cc/2025/11/26/announcing-the-neurips-2025-best-paper-awards/).

## Shared three-week execution

### Week 1, 16 to 22 August: lock validity and run the cheapest falsification

- Freeze the source-video manifest, provenance labels, primary endpoint, and exact held-out-source enumeration.
- Recover one internally consistent checkpoint lineage or retrain the small strict-inductive baseline.
- Reproduce the existing transductive number before changing the split.
- Run the missingness-only, provenance-only, raw-kinematic, and untrained-encoder controls.
- Implement the primary proposal's smallest decisive comparison.

**Day 5 gate:** continue only if all compared systems share the same split, preprocessing, target, and tuning boundary, and the primary measurement is defined for every required held-out source. A scientifically honest stop still leaves a reusable evaluation artifact.

### Week 2, 23 to 29 August: obtain the decisive figure

- Run three screening seeds on the frozen comparison.
- Report every held-out source before a pooled summary.
- Add the mechanistic perturbation that distinguishes the intended explanation from a nuisance explanation.
- Freeze the final model and analysis choices by 29 August.

**Day 14 gate:** if the central effect is smaller than the pre-registered practical margin, changes sign across feasible source holdouts, or disappears against the simple baseline, stop method expansion and write the qualified or negative result.

### Week 3, 30 August to 5 September: confirm and package

- Run five confirmation seeds only for the frozen compact comparison.
- Compute source-level sensitivity and effect sizes. Do not use seed spread as the only interval.
- Package the split manifest, configs, hashes, environment, and seed-level results.
- Draft one main claim, one mechanism, one limitation table, and one decisive figure.
- Complete the required ICLR 2027 AI-use and reproducibility statements. The [ICLR 2027 Author Guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines) require an AI-use statement and recommend a reproducibility statement.

## Shared evaluation and stopping rules

Every proposal follows [`_shared/evaluation-contract.md`](_shared/evaluation-contract.md). In particular:

- Split by source video before preprocessing fit, encoder training, checkpoint selection, probe fitting, or calibration.
- Label any reused checkpoint as transductive when it saw held-out sources.
- Report exact source membership and extraction pathway.
- Use raw kinematics, missingness-only, provenance-only, untrained encoder, and current exposed S-JEPA baselines where applicable.
- Report per-source results, effect size, and source-level sensitivity.
- Enumerate every feasible held-out choice for classes with fewer than four sources.
- Use at least three seeds for screening and five only after the analysis is frozen.
- Never convert a dataset folder label into a diagnostic claim.

## Final choice rule

Choose proposal 05 if a same-token, capacity-matched temporal readout beats mean/std pooling on the pre-registered timing or asymmetry endpoint and the sign is stable across feasible held-out sources. Choose proposal 04 if proposal 05 shows that the token tensor itself lacks the needed information. Use proposal 03 to determine whether collapse or low effective rank explains either null. Always ship proposal 01's split and proposals 06's controls with the result.

The strongest submission is a single coherent story:

> Honest source-level evaluation reveals a limitation, one controlled intervention identifies its mechanism, and a simple baseline shows whether the intervention recovers useful temporal gait information.
