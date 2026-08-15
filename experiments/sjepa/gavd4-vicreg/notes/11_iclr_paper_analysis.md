# From `gavd4-vicreg` to an ICLR paper

**A systematic analysis of the repository, an audit of what it actually established, and three concrete paper designs ranked by novelty, feasibility, and value to the ICLR community.**

Prepared 12 August 2026. All repository claims below are traced to specific files, cells, or saved artifacts.

---

## Part 0. Executive summary

### What the repository is

`gavd4-vicreg` is a carefully documented, honesty-first reimplementation of a skeleton Joint-Embedding Predictive Architecture (S-JEPA) applied to monocular pathological gait video. It is unusually rigorous about provenance: checkpoint fingerprints, artifact contracts, adversarial review logs, a leakage ledger, and a documented refusal to over-claim. That discipline is a genuine asset.

### What it established

Three defensible things, and one that is over-read:

1. **The pipeline runs and does not fully collapse.** 600 epochs, 11,400 optimizer updates, final feature std 0.414, mean pairwise cosine 0.609.
2. **The learned five-class geometry is weak.** Cosine silhouette 0.009; minimum centroid distance 0.037 versus mean within-condition distance 0.120.
3. **Every downstream score is confounded.** The encoder saw all evaluation rows, was fine-tuned with their labels after Stage 0, and source videos cross every split.
4. **Over-read:** the paper says the missingness-only control at 0.448 accuracy "shows detector behavior alone carries condition-related information." At 0.448 it is *below* the 0.490 majority baseline. The balanced-accuracy figure (0.466 vs 0.20 chance) supports a weaker version of the claim; the accuracy figure does not.

### Why it is not yet an ICLR paper

ICLR rewards a generalizable claim about representation learning, supported by controlled experiments with baselines and ablations. The current work is a single-configuration engineering audit on 159 sequences from 35 videos, with no controlled comparison, no ablation, no seed variance, and no held-out estimate. Reframed as-is, it reads as a well-executed negative result on a very small dataset — a good MICCAI/CHIL/workshop paper, not an ICLR main-track paper.

### What is genuinely ICLR-shaped inside it

One design choice in this repo is non-standard and, as far as I can find, unstudied as a general mechanism:

> **The prediction-target set is restricted to a fixed anatomical subset (12 of 33 landmarks), while the context encoder still sees all 33.**

Every JEPA/MAE variant I checked either masks uniformly at random, masks by learned/heuristic motion salience (MAMP, MaskSem, Skeleton2vec), or masks by spatial blocks (I-JEPA, V-JEPA). None of them asks the question this repo accidentally set up: **does the choice of *what must be predicted* — independent of mask ratio, architecture, and data — determine *what the representation encodes*?** That question is testable, cheap on this hardware, generalizes far beyond gait, and the repo already owns the two things needed to answer it well: a working JEPA and (via the sibling 82-feature project) a clinically graded battery of interpretable probe targets.

**Recommended paper (P1):** *Target vocabulary as an inductive bias — what a JEPA is asked to predict determines what it encodes.* Details in Part 4.

Two credible alternates are in Part 5: an evaluation-bias measurement paper (P2), and a Datasets & Benchmarks submission (P3).

---

## Part 1. Systematic analysis of the repository

### 1.1 Structure

| Area | Contents | Notes |
|---|---|---|
| Notebooks 00–06 | 7 executable tutorials | Self-contained; each repeats the code it needs |
| `docs/` | 3 staged manuscripts (`staged_sjepa_gait`, `staged_details`, `staged_evolution`), figure generators, `references.bib`, `result_history.csv` | Manuscript-grade, IEEE/URTC-targeted |
| `notes/` | 10 research notes incl. improvement plan, literature verification, adversarial review log | The improvement plan (`04`) is a strong prior analysis |
| `cache/artifacts/real/` | Checkpoints, embeddings (parquet), metrics, confusion matrices, missingness audits, `classifier_contract.json` | Reproducible artifact chain |
| `images/`, `slides/` | 12 tutorial SVGs, a built HTML/PPTX deck | Pedagogical layer |

### 1.2 Data

| Layer | Sequences | Videos | Provenance |
|---|---:|---:|---|
| Canonical GAVD cohort | 96 | 18 | GAVD annotations |
| Added normal (accepted) | 63 | 17 | Self-annotated YouTube, MediaPipe auto-bbox |
| Stage 0 normal total | 75 | 18 | Mixed |
| Full curriculum | 159 | 35 | Mixed |

Canonical class counts: normal 12, Parkinson's 9, stroke 12, myopathic 47, cerebral palsy 16.

**The critical structural fact:** GAVD contains **1,874 sequences from 450+ videos and 400+ subjects** ([Ranjan et al., IEEE Access 2025](https://arxiv.org/html/2407.04190v1)). This project used **5%** of it. Almost every limitation in the current results — no video-disjoint five-class split, two-fold ceiling on grouped CV, one canonical normal video — dissolves at full GAVD scale. This is the single highest-leverage change available and it costs pose-extraction compute, not research risk.

### 1.3 Method, precisely

**Input.** MediaPipe Pose Lite, 33 landmarks, visibility threshold 0.45. Internal gaps ≤4 frames interpolated; longer gaps and end gaps left missing. Pelvis-centred, body-scale normalized (max of shoulder/hip width, median over frames), temporally resized to 64 frames. Missing coordinates become **0**, and only `[..., :3]` is passed forward — the visibility channel is dropped.

**Tokenization.** 4-frame patches × 33 joints = 16 × 33 = **528 tokens**. Learned separate time and joint positional embeddings, added.

**Architecture (real profile).** `embed_dim=96`, encoder depth 4, predictor depth 2, 4 heads, pre-norm Transformer, FFN 4×. Target encoder is a gradient-free EMA copy. Under a million parameters.

**Masking.** Eligible tokens = valid × {11,12,23,24,25,26,27,28,29,30,31,32} (shoulders, hips, knees, ankles, heels, foot indices). `n_mask = floor(0.60 × min_over_batch(eligible_count))`, sampled uniformly without replacement, same count for every sample.

**Objective.**

```
L = L_JEPA + 0.05 · L_VICReg + 0.25 · L_group
```

- `L_JEPA`: DINO-style centred/sharpened cross-entropy, `softmax` taken **over the 96 embedding dimensions**, teacher temp 0.06, student temp 0.10, centre EMA β=0.9.
- `L_VICReg`: invariance 25 / variance-hinge 25 / covariance 1, on a 2-layer projector over authorized-pooled tokens from two geometric views.
- `L_group`: **label-aware**, active only in Stages 1–4. Within-condition compactness + hinge on centroid distances below margin 1.0.

**Curriculum.** One continuing model. Stage 0: 75 normal, 300 epochs. Stages 1–4: add Parkinson's → stroke → myopathic → cerebral palsy, 75 epochs each, condition-balanced replay, optimizer restarted at lr 3e-4 (Stage 0 used 1e-3), model state never reinitialized.

**Downstream.** Frozen target encoder → validity-masked mean/std over all tokens ⊕ validity-masked mean/std over the 12 authorized landmarks = **384-d** vector → `StandardScaler` + `RandomForestClassifier(100, depth 5, sqrt, balanced)`.

### 1.4 Technical findings from reading the code

These are things the manuscripts do not currently say, and several of them materially affect how the existing results should be interpreted.

**F1 — The mask ratio is confounded with curriculum stage.** `n_mask` is derived from `counts.min()` over the batch. As stages add conditions with poorer landmark coverage, the batch minimum falls, so the realized eligible-mask fraction drops from 0.551 (end of Stage 0) to 0.423 (end of Stage 4). The *pretext task gets easier as the curriculum progresses*. The reported "normal-anchor cosine drift" from 0.954 → 0.594 is therefore **not cleanly attributable to forgetting** — task difficulty changed at the same time. This confound needs an explicit control (fixed absolute `n_mask`) before any drift claim can stand.

**F2 — The effective global mask ratio is ~15–20%, not 60%.** 0.60 applies to eligible tokens (≤192 of 528). Realized global masking is ≈0.20 at Stage 0 and ≈0.15 at Stage 4. I-JEPA and V-JEPA operate at 75–90%. The context encoder always sees a near-complete skeleton, so the pretext task is closer to spatial interpolation than motion modelling. Their own `notes/04_improvement_plan.md` flagged this (R1); it remains unaddressed.

**F3 — The JEPA loss takes softmax over feature dimensions, not over prototypes.** DINO's cross-entropy is over a prototype/projection head's output logits. Here `softmax` is applied directly to the 96-d encoder output. This makes the objective sensitive to the arbitrary basis of the embedding space, couples dimensions in a way that has no clear geometric meaning, and is a deviation from both DINO and the S-JEPA paper it claims alignment with. A smooth-L1 or cosine regression target (V-JEPA's choice) is the standard, better-understood alternative and should at minimum be an ablation arm.

**F4 — The teacher freezes at the end of every stage.** `cosine_ema(step, total_steps, start=EMA_START, end=1.0)` reaches momentum 1.0 at `step == total_steps - 1`, and `global_step` resets at each stage boundary. So the target encoder stops updating near the end of Stage 0, restarts, freezes again at the end of Stage 1, and so on — five separate freeze-thaw cycles. Standard practice caps momentum below 1 (e.g. 0.9995).

**F5 — The zero sentinel collides with the pelvis origin.** After `center_and_scale`, the pelvis sits at (0,0,0) and every missing landmark is also written to (0,0,0). The model cannot distinguish "at the hip" from "not detected", and the visibility channel is discarded before the encoder. Feet and ankles — the least reliable landmarks — are exactly the prediction targets.

**F6 — VICReg's geometric view rotates in the x–z plane.** MediaPipe's `z` is a weakly-supervised relative depth estimate from a monocular image. Rotating x against z injects estimator noise into the invariance branch rather than a meaningful viewpoint change. A 2D-only affine/scale/temporal-jitter view family would be better motivated.

**F7 — Cadence is destroyed and never restored.** `temporal_resize` to a fixed 64 frames removes absolute speed and cadence, which are the strongest clinical discriminators for all four pathologies. Downstream pooling is order-invariant mean/std, which removes rhythm and phase as well. The representation is being asked to carry gait information that the preprocessing already deleted.

**F8 — Provenance is almost perfectly correlated with the label.** 63 of 75 normal sequences come from the self-annotated pipeline; every abnormal sequence comes from the canonical pipeline. A normal-vs-abnormal classifier can win by detecting extraction pipeline. The Lane C binary result (0.849 accuracy, 0.966 AUC) is the score most exposed to this, and it is the headline binary number.

**F9 — Stages 1–4 are supervised.** The group loss uses condition labels. Calling the pipeline self-supervised, or comparing it to self-supervised baselines, is not accurate for anything after Stage 0. The repo says this; the framing needs to stay this careful in any future paper.

### 1.5 Results ledger, with validity annotation

| Readout | Acc | Bal. acc | Macro-F1 | Validity |
|---|---:|---:|---:|---|
| All-96 stratified | 0.793 | 0.889 | 0.821 | ✗ row + video + label exposure |
| All-96 missingness-only | 0.448 | 0.466 | 0.429 | control; below 0.490 majority |
| Exact exp5 split | 0.714 | 0.730 | 0.742 | ✗ same exposure |
| Historical 82-feature RF | 0.762 | — | 0.728 | ✗ same split; different features |
| Lane C binary (5 folds) | 0.849 | 0.874 | 0.826 | partial; encoder saw all rows; provenance confound |
| Lane C five-class (2 folds) | 0.653 | 0.603 | 0.625 | partial; 2 folds is not an estimate |
| Silhouette (canonical 96) | 0.009 | | | valid, and the most informative number in the repo |

**Reading:** the only number that supports a claim is the silhouette. Everything else is a description of a fitted system on a corpus it was fitted to.

---

## Part 2. Assets inventory — what is reusable for a real paper

| Asset | Value for an ICLR paper |
|---|---|
| Working JEPA + curriculum + VICReg + group loss, in a few hundred lines of plain PyTorch | Full experimental control; no framework friction |
| Sub-million-parameter model, 11.4k updates | **Nested/fold-local pretraining is affordable.** This is rare and is the enabling condition for the strongest protocols |
| Artifact contract + fingerprinting + exposure ledger | Reproducibility appendix writes itself; unusual reviewer goodwill |
| Missingness-only shortcut control already implemented | A confound control most papers lack |
| 82 clinically graded handcrafted gait features (sibling project) | **A probe battery with ground truth and clinical importance weights.** This is the single most valuable and least-exploited asset |
| GAVD access + extraction pipeline | 20× data expansion available for extraction cost only |
| Verified citation ledger (`notes/06`) | Related work is pre-vetted |
| Adversarial review discipline | Sound methodology habits |

---

## Part 3. Novelty landscape — where the gap actually is

I surveyed the adjacent literature to locate an unoccupied, defensible claim.

**JEPA family.** [I-JEPA](https://ai.meta.com/blog/yann-lecun-ai-model-i-jepa/) (CVPR 2023) predicts multi-block image targets. V-JEPA and V-JEPA 2 extend to video with spatiotemporal tube masks and smooth-L1 feature regression. [S-JEPA](https://link.springer.com/chapter/10.1007/978-3-031-73411-3_21) (ECCV 2024) applies the design to skeleton action recognition. [LeJEPA](https://arxiv.org/html/2511.08544v2) addresses training stability and loss-based model selection. All treat the target set as a *sampling* detail, not as a design variable that determines representation content.

**Masking-strategy work in skeleton SSL.** [MaskSem](https://arxiv.org/html/2508.12948) (2025) guides masking by motion semantics; MAMP predicts masked *motion*; Skeleton2vec uses contextualized targets; a recent [topology-masked prediction](https://www.nature.com/articles/s41598-026-39330-9) work uses skeletal topology. These change *how* tokens are chosen, and evaluate by downstream accuracy. **None measures whether the representation preferentially encodes information about the target set versus the context set.** That asymmetry is the gap.

**Frozen-feature auditing.** [Frozen Brain-MRI Foundation Models Are Site Fingerprints](https://arxiv.org/abs/2608.10295) (Aug 2026) shows acquisition site is linearly decodable at ~0.9 balanced accuracy from frozen features — above every clinical variable — and is intrinsic rather than learned. A related [EEG foundation-model cross-cohort audit](https://arxiv.org/html/2607.24834) makes a parallel point. This establishes that the community *is receptive to rigorous audit papers*, and gives a strong methodological template. It also means a plain "we audited our gait encoder" paper would now be derivative.

**Evaluation bias.** [Moscovich & Rosset (JRSS-B 2022)](https://academic.oup.com/jrsssb/article/84/4/1474/7073256) prove that *unsupervised preprocessing* applied to the full dataset before cross-validation biases the CV estimate. Deep SSL pretraining is unsupervised preprocessing with 10⁶–10⁹ parameters, and is almost universally applied to the full corpus before probe-level grouped CV. **The deep-SSL analogue of that theorem has not been measured.** That is the gap P2 targets.

**JEPA anomaly detection.** Already crowded — T-SAR-JEPA, MTS-JEPA/SC-JEPA, V-JEPA-2-as-process-model. A "normal-first JEPA as gait abnormality score" framing would land in a busy field with a weaker dataset than the incumbents. **Do not choose this.**

**Continual/curriculum SSL.** Well-trodden (continual representation learning, drift compensation, exemplar-free methods). The normal-first curriculum is not, by itself, a novel contribution.

---

## Part 4. Recommended paper (P1)

### Title

**Target Vocabulary Is an Inductive Bias: What a Joint-Embedding Predictive Architecture Is Asked to Predict Determines What It Encodes**

### One-sentence claim

In JEPAs, the *target set* — the subset of tokens the predictor must reconstruct in latent space — is a first-class design variable that steers representation content more strongly than mask ratio or model width in the small-data regime, and it does so through a measurable **encoding asymmetry**: variables that depend on target tokens become substantially more linearly decodable than variables that depend on context-only tokens, even though the encoder sees both.

### Why this is novel

- Prior masking work varies *how tokens are chosen* and evaluates by end-task accuracy. This paper varies *which semantic set must be predicted* and measures *what information ends up in the representation*, with an interpretable probe battery.
- It reframes a hyperparameter as a controllable prior. That is the kind of reusable knob ICLR likes: architecture-free, cost-free, applies to I-JEPA/V-JEPA/S-JEPA/MAE alike.
- The asymmetry claim is falsifiable and, if it holds, is a mechanistic explanation for why semantic masking methods (MaskSem, MAMP) work — currently justified only post hoc by accuracy.

### Why this is feasible here

Every arm is a re-run of notebook 04 with a different `MASK_KEYPOINTS` set. The model is under a million parameters. On a single modern GPU one pretraining run of 50k updates on full-GAVD-scale data is roughly 20–40 minutes. The full matrix below is on the order of **40–80 GPU-hours**, i.e. a few days on one rented GPU.

### Experimental design

**Datasets.**

| Role | Dataset | Why |
|---|---|---|
| Primary | GAVD, full extraction (~1,874 seq / 450+ videos / 400+ subjects) | The clinically graded probe battery lives here |
| Generality | NTU-RGB+D 60 (cross-subject) or PKU-MMD | Standard skeleton SSL benchmark; shows the effect is not gait-specific |
| Optional third | A wearable-IMU HAR set (PAMAP2 / UCI-HAR) | Shows the effect survives a change of sensing modality |

**Target-set arms** (all with *identical* absolute token counts masked, to decouple set choice from difficulty — this is the control the current repo lacks, see F1):

| Arm | Target set | Purpose |
|---|---|---|
| T-random | uniform over all 33 joints | standard baseline |
| T-anatomical | the 12-landmark clinical whitelist | the repo's current choice |
| T-random-12 | random fixed subset of 12 joints, resampled per seed | **matched-cardinality control** — isolates *which* joints from *how many* |
| T-distal | ankles, heels, foot indices only | tests distal→temporal-variable hypothesis |
| T-proximal | shoulders, hips, trunk only | tests proximal→postural-variable hypothesis |
| T-unilateral | one full side across time | tests asymmetry encoding |
| T-motion | MAMP-style motion-salience selection | strong published baseline |
| T-block | V-JEPA-style spatiotemporal tube | strong published baseline |

**Sweeps.** Global mask ratio ∈ {0.2, 0.4, 0.6, 0.75, 0.9} × target arm — this directly tests "target set matters more than ratio," and fixes F2. Width `D` ∈ {96, 192, 384}. 3 seeds everywhere.

**Measurements.**

1. **Probe battery (the core result).** Ridge/linear probe from frozen embedding → each of the 82 clinically graded gait features, reporting R² per feature, grouped by (a) the joints the feature is computed from, and (b) its clinical importance grade. The headline plot is R² for *target-dependent* features versus *context-only* features, per arm.
2. **Encoding asymmetry index.** A single scalar: mean R² over target-dependent probes minus mean R² over context-only probes, normalized. Report it as a curve against mask ratio and target-set size.
3. **Downstream task.** Five-class and binary gait-condition classification, linear probe and Random Forest, under the fold-local protocol below.
4. **Confound controls, always reported alongside.** Majority class; missingness-only probe (already implemented); **video-identity decodability** and **subject-identity decodability** from the frozen embedding — the direct analogue of the brain-MRI site-fingerprint metric, and a strong addition to this literature; extraction-pipeline decodability (fixes F8).
5. **Standard SSL benchmark transfer.** NTU-60 linear-probe accuracy per arm, so the effect is legible to the skeleton SSL community.

**Evaluation protocol (non-negotiable, and a selling point).** Fold-local everything: outer split on **subject**, then within each outer training partition — choose preprocessing rules, pretrain all stages from scratch, fit the probe — and open held-out subjects once. Report the same numbers under the conventional (pretrain-on-all) protocol as well; the delta is a free secondary contribution and directly instantiates the Moscovich–Rosset bias for deep SSL.

**Fixes to apply before running anything** (from Part 1.4): fixed absolute mask count (F1); explicit visibility channel as a 4th input dimension instead of the 0-sentinel (F5); EMA momentum capped at 0.9995, no per-stage reset (F4); smooth-L1/cosine latent regression as the default objective with the DINO-style CE kept as an ablation arm (F3); 2D-only view family (F6); scalar rate features (native fps, duration, autocorrelation cadence) restored from the pre-resize track as an explicitly-labelled auxiliary input (F7).

### Predicted results and what each means

| Outcome | Interpretation | Still publishable? |
|---|---|---|
| Asymmetry index > 0 and grows with target-set specificity | Main claim confirmed; target sets are a design knob | Yes — headline |
| Asymmetry index ≈ 0 | Full-context encoders equalize information regardless of target; semantic masking works for other reasons | Yes — a clean negative result that contradicts the implicit assumption behind MaskSem/MAMP, with a mechanism to explain why |
| Asymmetry present but downstream accuracy unchanged | Representation content and task accuracy decouple | Yes — an important caution about accuracy-only evaluation of masking strategies |

Every branch yields a paper. That property is what makes this a good bet.

### Structure

1. **Intro** — masking strategy is treated as a nuisance hyperparameter; we show the target set is a prior.
2. **Related work** — JEPA family; masking strategies in skeleton SSL; probing/interpretability; frozen-feature audits; CV bias from unsupervised preprocessing.
3. **Formalism** — JEPA with an explicit target vocabulary T ⊆ tokens; define the encoding asymmetry index; state the hypothesis.
4. **Experimental protocol** — datasets, fold-local pretraining, probe battery, confound controls.
5. **Results** — asymmetry curves; target-set × mask-ratio interaction; downstream; NTU-60 transfer; confound panel.
6. **Analysis** — which clinical variables are recoverable at all; where the information actually lives; the conventional-vs-fold-local delta.
7. **Limitations** — monocular pose noise, YouTube provenance, dataset scale, no lab-grade validation.
8. **Reproducibility appendix** — fingerprints, contracts, exposure ledger (already built).

### Figures

- F1: target vocabulary schematic — context vs target sets across arms.
- F2: **the money plot** — R² for target-dependent vs context-only probes, per arm, with error bars over seeds.
- F3: asymmetry index vs global mask ratio, one line per arm.
- F4: per-feature R² heatmap (82 features × 8 arms), rows grouped by joint dependency and clinical grade.
- F5: confound panel — majority / missingness / video-ID / subject-ID / pipeline-ID decodability alongside every reported score.
- F6: conventional vs fold-local protocol delta.
- F7: NTU-60 linear-probe accuracy per arm.

---

## Part 5. Alternates

### P2 — *How much does self-supervised pretraining leak? Quantifying the fold-local gap in small-data representation learning*

**Claim.** The standard evaluation protocol in small-data SSL (pretrain on the full corpus → freeze → grouped CV on a probe) is optimistically biased, and the bias is large and predictable in exactly the regime where SSL is most promoted.

**Contributions.**
1. A leakage taxonomy for SSL pipelines: row exposure, group exposure, **representation exposure**, **label exposure** (semi-supervised or label-aware pretraining objectives), preprocessing-statistic exposure, and selection exposure.
2. A measurement of the fold-local gap Δ across ≥3 small datasets × ≥3 SSL objectives (JEPA / MAE / contrastive) × corpus sizes, as a function of N, group count G, and capacity.
3. A **cheap estimator of Δ** so practitioners do not have to nest: candidates are group-identity decodability of the frozen features (following the site-fingerprint metric), or leave-one-group-out embedding displacement between two encoders differing only in that group's presence during pretraining.
4. A protocol and audit toolkit.

**Track.** Main track, or Datasets & Benchmarks.

**Novelty risk — the main one.** Reviewers may say "everyone knows pretraining on the test set is bad." The rebuttal must be empirical and blunt: survey N recent small-data SSL papers, report what fraction nest pretraining inside the fold (my expectation: near zero), and show Δ is often larger than the margins those papers claim as improvements. Without that survey the paper is vulnerable; with it, it is strong.

**Feasibility.** High. This repo's compute profile is precisely what makes nesting affordable. It is arguably the more *certain* paper, and the less exciting one.

### P3 — *GAVD-SSL: a leakage-audited benchmark for representation learning on pathological gait video*

**Claim.** Provide the community with the full-GAVD pose extraction, subject- and video-disjoint splits, a fold-local pretraining protocol, a confound control suite (missingness, video ID, subject ID, extraction pipeline), and reference implementations of 4–5 SSL objectives with reported baselines.

**Track.** Datasets & Benchmarks.

**Value.** Real. Clinical gait video SSL has no standard leakage-audited benchmark, and the confound suite would be a differentiator against generic action-recognition benchmarks.

**Risk.** GAVD is URL-linked YouTube content — link rot and redistribution constraints are a genuine review objection. Mitigation: release extracted pose tensors and manifests (derived, non-redistributive), plus the extraction code, and document a link-health check.

### P4 — Normal-first JEPA as a continuous gait abnormality score

**Do not choose this.** JEPA-for-anomaly-detection is already crowded (T-SAR-JEPA, MTS-JEPA/SC-JEPA, V-JEPA-2 process models), and this dataset is weaker than the incumbents'. It is a fine *section* inside P1 or P3, not a paper.

### Ranking

| | Novelty | Feasibility | ICLR fit | Risk |
|---|---|---|---|---|
| **P1 target vocabulary** | High | High | High (representation learning) | Medium — depends on effect existing, but all branches publish |
| P2 leakage gap | Medium-High | Very high | High (evaluation) | Medium — "already known" objection |
| P3 benchmark | Medium | Medium | Medium-High (D&B) | Medium — data redistribution |
| P4 anomaly scoring | Low | High | Low | High |

**Recommendation: P1, with P2's fold-local protocol as its methodological backbone and a compressed version of P2's delta measurement as a secondary result.** If the schedule slips or the asymmetry effect proves null and uninteresting, P2 is a clean fallback using the same runs.

---

## Part 6. Feasibility, sequencing, and prerequisites

### Hard prerequisites (do these before any modelling)

| # | Item | Why it blocks | Effort |
|---|---|---|---|
| 1 | **Extract full GAVD** (~1,874 seq / 450+ videos), preserving `subject_id` | Without ≥100 videos and subject IDs, no honest split and no seed-stable estimate exists | Days of extraction compute; some link rot expected — budget for it and report the yield |
| 2 | **Recover the 82-feature battery** with per-feature joint dependencies and clinical grades | It *is* the probe battery; P1's core plot needs it | Sibling repo; mostly bookkeeping |
| 3 | **Confirm subject-level identifiers** in GAVD, not just video IDs | Subject-disjoint is the correct split; video-disjoint is the fallback | Dataset inspection |
| 4 | **Apply the F1–F7 fixes** | F1 and F5 are confounds that would invalidate the central comparison | ~1 week of careful edits to nb02/nb04 |
| 5 | **Move to a GPU** | 100+ pretraining runs is not a CPU workload | Rental |

### Compute estimate

Full matrix for P1: 8 target arms × 5 mask ratios × 3 seeds × 5 outer folds = 600 pretraining runs at full scope. That is too many. **Reduce with a staged design:** run the 8 arms × 3 seeds × 5 folds at one fixed mask ratio (120 runs) for the main result, then the 5-ratio sweep on 3 representative arms × 3 seeds at a single fold (45 runs). ≈165 runs × ~25 min ≈ **70 GPU-hours**, plus NTU-60 arms. Comfortably within a few hundred dollars of rented compute.

### Timeline (aggressive but realistic for the next ICLR cycle)

| Weeks | Milestone |
|---|---|
| 1–3 | Full GAVD extraction; subject metadata; extraction yield report |
| 2–4 | F1–F7 fixes; unit tests; re-verify the 159-sequence run reproduces under the fixed code (a regression anchor) |
| 4–5 | Probe battery wired up; confound suite (video/subject/pipeline decodability) implemented |
| 5–6 | Fold-local harness; conventional-vs-fold-local delta on the existing small corpus as a smoke test |
| 6–10 | Main matrix: 8 arms × 3 seeds × 5 folds |
| 10–12 | Mask-ratio sweep; width sweep; NTU-60 transfer |
| 12–14 | Analysis, figures, writing |
| 14–16 | Internal adversarial review (the repo already has this habit), rebuttal-proofing, appendix |

### Where the plan can fail, and the mitigation

| Failure | Probability | Mitigation |
|---|---|---|
| GAVD link rot yields far fewer than 1,874 sequences | Medium-high | Report yield honestly; supplement with a public pathological-gait set; the paper's claim does not depend on absolute scale, only on having enough subjects for real splits |
| Encoder too weak → all probes near-zero R², no signal | Medium | Scale width/data first; validate on NTU-60 where SSL is known to work; if gait probes are uniformly null, that itself is a reportable finding about monocular pose SSL |
| Asymmetry effect is null | Medium | Publish it as a negative result contradicting the implicit assumption of semantic-masking papers; strengthen with the mask-ratio interaction |
| Reviewers reject the clinical framing | Low-medium | Lead with the representation-learning claim; gait is the testbed, not the subject; NTU-60 arm proves generality |
| Compute overrun | Low | The staged design above; the model is tiny by ICLR standards |

---

## Part 7. What to stop doing

1. **Stop reporting the confounded lanes as results.** The all-96 (0.793) and exact-exp5 (0.714) numbers should appear only inside a leakage demonstration, never as performance.
2. **Stop calling the pipeline self-supervised after Stage 0.** The group loss is supervised. Comparisons to SSL baselines must be like-for-like.
3. **Stop adding stages, weights, and terms without ablation.** Three loss terms and five curriculum stages were added with no controlled comparison; nothing in the current design is attributable.
4. **Do not target URTC and ICLR with the same content.** URTC is a fine home for the current audit. An ICLR paper must be a different, larger contribution — the venues will not both accept the same work, and dual submission is a policy problem.
5. **Do not run one seed.** Nothing at n=159 with one seed is distinguishable from noise. Three seeds minimum, everywhere, with variance reported.

---

## Part 8. What is unusually good here and should be preserved

The honesty infrastructure. Checkpoint fingerprints bound to data payloads, artifact contracts, an exposure ledger printed beside every score, three reference lines (majority / shortcut floor / handcrafted ceiling) on every readout, an adversarial review log, and a `result_history.csv` that records whether *the model* changed or only *the evaluation* changed. Most ICLR submissions have none of this. Ported into P1 as a reproducibility appendix and a confound-control panel that appears in the main results table, it is a differentiator — the site-fingerprint and EEG-audit papers succeeded on precisely this kind of rigor.

---

## Sources

- [Ranjan et al., Computer Vision for Clinical Gait Analysis: A Gait Abnormality Video Dataset (GAVD)](https://arxiv.org/html/2407.04190v1)
- [Abdelfattah & Alahi, S-JEPA: A Joint Embedding Predictive Architecture for Skeletal Action Recognition, ECCV 2024](https://link.springer.com/chapter/10.1007/978-3-031-73411-3_21)
- [Assran et al., I-JEPA (CVPR 2023)](https://ai.meta.com/blog/yann-lecun-ai-model-i-jepa/)
- [LeJEPA: stability and loss-based model selection for JEPAs](https://arxiv.org/html/2511.08544v2)
- [MaskSem: Semantic-Guided Masking for 3D Hybrid High-Order Motion Representation](https://arxiv.org/html/2508.12948)
- [Skeleton motion topology-masked prediction and contrastive learning](https://www.nature.com/articles/s41598-026-39330-9)
- [Self-Supervised Skeleton-Based Action Representation Learning: A Benchmark and Beyond (IJCV 2025)](https://link.springer.com/article/10.1007/s11263-025-02644-8)
- [Frozen Brain-MRI Foundation Models Are Site Fingerprints (arXiv 2608.10295)](https://arxiv.org/abs/2608.10295)
- [Cross-Cohort Spectral–Temporal Dissociation in Frozen EEG Foundation-Model Representations](https://arxiv.org/html/2607.24834)
- [Moscovich & Rosset, On the Cross-Validation Bias due to Unsupervised Preprocessing (JRSS-B 2022)](https://academic.oup.com/jrsssb/article/84/4/1474/7073256)
- [Roberts et al., Cross-validation strategies for data with temporal, spatial, hierarchical or phylogenetic structure (Ecography 2017)](https://doi.org/10.1111/ecog.02881)
- [T-SAR-JEPA: Self-Supervised Temporal Anomaly Detection via Latent Prediction](https://arxiv.org/abs/2606.05700)
- [SC-JEPA / MTS-JEPA: latent predictive learning for time-series anomaly prediction](https://arxiv.org/html/2602.04643)
- [Shortcut learning in medical AI hinders generalization](https://pmc.ncbi.nlm.nih.gov/articles/PMC11094145/)
- Repository files: `README.md`, `docs/staged_sjepa_gait.md`, `docs/staged_evolution.md`, `docs/result_history.csv`, `notes/04_improvement_plan.md`, `notes/06_literature_findings.md`, `notes/08_codex_review_log.md`, notebooks `00`–`06`
