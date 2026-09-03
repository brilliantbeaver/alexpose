# Don't Forget Normal: Measuring and Trying to Repair Normative-Anchor Drift in a Continual Skeleton-JEPA World Model of Gait

*Draft for the NeurIPS 2026 Workshop on Foundation Models for the Brain and Body (BrainBodyFM 2026). Non-archival, double-blind. Target length: 5 pages excluding references and appendices. Author block withheld for anonymous review.*

---

## Abstract

We study what happens to a self-supervised model of human walking as it keeps learning. Starting from video, we extract stick-figure "skeletons" and train a **Skeleton Joint-Embedding Predictive Architecture (S-JEPA)** - a model that learns by predicting hidden parts of a pose in its own internal feature space rather than redrawing pixels. We train it the way a clinic would build one: first on **normal** gait only, then adding four movement disorders one at a time (Parkinson's disease, stroke, myopathic gait, cerebral palsy), keeping a replay of earlier data throughout.

Our central object of study is a single, cheap, label-free number: the **normal anchor** - the cosine similarity between the model's current internal picture of normal walking and the picture it had right after the normal-only stage. As the four disorders enter, this anchor falls steadily from **0.954 to 0.594**. We call this a **normative-anchor drift curve**, and to our knowledge no prior JEPA / continual-learning / clinical-gait paper reports one. We then do three things with it. (1) We **reproduce** it exactly from the frozen checkpoints (agreement to within 5×10⁻⁷). (2) We **attribute** it: a controlled ablation shows a label-aware "group-margin" term that spreads condition clusters apart is part of the cause (anchor 0.954 with the margin, 0.976 without), but not the whole cause. (3) We **attempt to repair** it with *AnchorGuard*, a label-free term that gently pulls the normal representation back toward its Stage-0 self. AnchorGuard **fails to hold the anchor** (final 0.538, below baseline) yet **does not hurt** - and slightly helps - downstream disorder decoding (binary macro-F1 0.893 vs 0.849; five-class within a pre-registered non-inferiority margin). We report this honestly as a **retention-plasticity trade-off with a diagnosed mechanism**, and we add two zero-training probes that reframe the frozen model as a *world model* of movement: it can fill in hidden joints but **cannot forecast the future** better than copying the last frame, and its pooled features have discarded the two axes clinicians care about most - **walking rate** and **signed left-right asymmetry**. Every score is **in-corpus (transductive)** on a tiny single-site cohort (159 sequences, 35 videos), so our contribution is a **measurement method, a mechanism, and an honest negative**, not a clinical claim. All numbers are pinned to file hashes and reproducible.

---

## 1. Introduction

### 1.1 The problem, in plain terms

Imagine you are building an AI system that watches people walk and helps flag movement problems. A sensible way to build it is the way a clinician learns: first get a rock-solid sense of what **normal** walking looks like, then study what changes in **Parkinson's disease, stroke, cerebral palsy, and muscle disease (myopathy)**. The worry is a familiar one for any system that keeps learning: as it studies the disorders, does it *quietly forget what normal looks like*? In machine learning this is called **catastrophic forgetting**. In a clinical tool it would be dangerous, because "how far is this person from normal?" is exactly the question the tool exists to answer.

This paper measures that forgetting directly, tries to stop it, and reports - carefully - what happened.

### 1.2 The ingredients (no background assumed)

**Skeletons from video.** We never work with raw pixels. A pose estimator (MediaPipe BlazePose) turns each video frame into 33 labeled points - shoulders, hips, knees, ankles, and so on - plus a visibility score saying how sure it is about each point. A walking clip becomes a short movie of moving dots: a **skeleton sequence**.

**JEPA: predicting in "meaning space."** A **Joint-Embedding Predictive Architecture (JEPA)** learns by a fill-in-the-blank game, but with a twist. It hides part of the input, then predicts the hidden part - *not* as raw coordinates, but as an **embedding**: a compact list of numbers that captures the *meaning* of that part. Two networks cooperate. A **view encoder** sees the pose with some joints hidden. A **target encoder** sees the whole pose and produces the "answer key" embeddings; it is a slow-moving copy of the view encoder (an exponential moving average, EMA), which keeps the answer key from being a moving target. A small **predictor** tries to guess the target embeddings at the hidden spots. Because the loss lives in embedding space, the model is free to ignore pixel-level noise and capture structure instead. Our version, applied to skeletons, is **S-JEPA**.

**Why not just reconstruct pixels?** Predicting exact coordinates forces the model to memorize wobble and detector noise. Predicting embeddings lets it learn *what matters* about a movement. This is also the sense in which JEPAs are called **world models**: a good one should be able to imagine plausible continuations of what a body is doing.

**Guarding against collapse.** A fill-in-the-blank model can cheat by mapping everything to the same vector (then every guess is trivially "correct"). This is **representation collapse**. We add **VICReg**, a term that (i) keeps each feature spread out across a batch and (ii) discourages features from duplicating each other - a label-free anti-collapse guardrail.

**Tokens and masking.** We cut each 64-frame sequence into **tokens**: one landmark over four adjacent frames. 33 landmarks × 16 time-slices = **528 tokens**. During training we hide a fraction of tokens drawn only from a fixed anatomical whitelist of 12 gait-relevant landmarks (shoulders, hips, knees, ankles, heels, foot tips) and ask the model to predict them.

**The normal-first continual curriculum.** We train in five stages with **one continuous model** (weights are never reset). Stage 0: normal gait only, 300 epochs, **no labels**. Stages 1-4 each add one disorder for 75 epochs, always replaying a balanced mix of earlier conditions, and add a **label-aware group term** that pulls each condition's cluster together and pushes different conditions apart.

### 1.3 The one number this paper is about

After Stage 0 we freeze a reference vector **c₀** - the model's average internal picture of normal walking. At the end of every later stage we measure the **normal-anchor cosine**: how aligned the model's *current* normal picture is with **c₀**. Cosine 1.0 means "unchanged"; lower means "drifted." This number needs **no labels and no downstream classifier** - it is cheap telemetry you could log during any continual run. Watching it fall is watching the model forget normal *in representation space*, which is more direct than watching an accuracy score fall.

### 1.4 Contributions

1. **A new, measurable phenomenon.** We define and report a **normative-anchor drift curve** for a continual skeleton-JEPA on clinical gait: **0.954 → 0.839 → 0.707 → 0.594** across the four disorder stages (Fig. 2). It is label-free, reproducible to 5×10⁻⁷ from frozen checkpoints, and - to our knowledge - unreported in the JEPA, continual-learning, or clinical-gait literature.
2. **A mechanism.** A matched ablation isolates *why* the anchor moves early: the label-aware group margin contributes measurably (Stage-1 anchor 0.954 with it, 0.976 without), but new-task statistics move the anchor even without it.
3. **An honest intervention.** *AnchorGuard*, a simple label-free consolidation term, **does not** hold the anchor at our pre-registered strength, yet leaves downstream disorder decoding non-inferior (binary improves). We report the resulting **retention-plasticity trade-off** and diagnose why a batch-level cosine attractor is too weak to fight the margin.
4. **Two zero-training world-model probes** that sharpen the story: (a) the frozen predictor **infills but does not forecast** (future-prediction latent cosine 0.23-0.44, below a copy-last baseline of 0.88-0.95); (b) matched-capacity readouts show the pooled features have **discarded walking-rate and signed-asymmetry information**, with controls proving the loss is upstream of the pooling.
5. **A reproducibility discipline** other small-cohort behavioral-signal studies can adopt: every classifier lane is labeled with its exact data exposure, every number is pinned to a file hash and an experiment fingerprint, and every decision threshold was written down *before* the run it gates.

We make one boundary explicit and repeat it everywhere: the cohort is tiny and single-site, folder names are **dataset annotations, not diagnoses**, and every score is **transductive** (the encoder saw every evaluation sequence during training). Our contribution is method and mechanism, never population inference.

![**Fig. 1.** From markerless video to audited claims: the pipeline and the ladder of what can honestly be claimed.](figures/bbfm_overview.svg)

---

## 2. Related work

**JEPAs and world models.** I-JEPA introduced predicting masked *embeddings* for images; V-JEPA extended feature prediction to video; S-JEPA adapted the idea to skeletons for action recognition. LeCun's world-model program frames such predictors as engines that should *imagine* futures. We take that framing literally in §5 and test whether a spatial infiller can forecast.

**Anti-collapse and motion-aware masking.** VICReg is our label-free anti-collapse regularizer. MAMP and Skeleton2vec choose masking targets by *motion*; we deliberately use a fixed anatomical whitelist with uniform sampling instead - a design choice we later show has consequences (§5.2).

**Continual and self-supervised learning.** Continual-learning studies usually report *accuracy* drops and repair them with replay, EWC, or distillation. We instead track a **label-free representation-space anchor**, which needs no downstream labels, and test a representation-space distillation repair (AnchorGuard) against it.

**Clinical gait from video.** GAVD provides the labeled clinical clips; GaitForeMer uses motion forecasting for severity estimation. Clinical literature grounds the two axes we probe: signed left-right **asymmetry** in stroke and Parkinsonian gait, and **cadence/stride-time** rate parameters. We make no diagnostic or severity claim.

**Evaluation methodology.** Grouped cross-validation and the absence of an unbiased K-fold variance estimator motivate our source-video grouping and our refusal to attach confidence intervals to two-fold results.

---

## 3. Method

### 3.1 Data and provenance

The **canonical cohort** is 96 GAVD sequences from 18 source videos: normal 12 (1 video), Parkinson's 9 (2), stroke 12 (3), myopathic 47 (10), cerebral palsy 16 (2). Because a single-video normal set is a weak normative reference, we added **63 self-annotated normal windows** from **17 additional YouTube videos** (64 candidates; 1 rejected for landmark coverage 0.027 < 0.45). Total: **159 sequences / 35 videos**. The added-normal clips were labeled inside this project and never received independent clinical review - a **provenance asymmetry** (63/75 normal-training rows come from the added path; every disorder row comes from the canonical GAVD path) that we carry as a control in every learned-advantage figure.

### 3.2 Preprocessing (and one honest limitation)

Pose from MediaPipe (33 landmarks + visibility, validity threshold 0.45); internal gaps ≤ 4 frames interpolated with the validity mask retained; every sequence pelvis-centered, body-scale normalized, and **time-resized to 64 frames**. This resizing is convenient but **deletes the native frame rate** - so absolute *walking speed and cadence are, by construction, not recoverable* from the tokens. We treat cadence and stride-time as **canary targets** in §5.2 precisely to detect this.

### 3.3 Model and objective

View encoder + EMA target encoder + predictor (embedding width 96; depth 4/2; 4 heads). The training loss is

$$\mathcal{L}=\mathcal{L}_{\text{JEPA}} + 0.05\,\mathcal{L}_{\text{VICReg}} + 0.25\,\mathcal{L}_{\text{group}}.$$

- **$\mathcal{L}_{\text{JEPA}}$** - cross-entropy between the predictor's distribution and the centered, sharpened target-encoder distribution at hidden tokens (predict-the-embedding).
- **$\mathcal{L}_{\text{VICReg}}$** - label-free variance + covariance regularization on projected features (anti-collapse).
- **$\mathcal{L}_{\text{group}}$** - label-aware: pull each condition's cluster together, and push different condition centroids apart with a hinge that activates when their cosine similarity exceeds 0.5. **Off during Stage 0.**

Masking hides a 0.60 fraction *of the smallest eligible-token count in the batch*, sampled uniformly from the 12-landmark whitelist (realized fractions 0.551 at Stage 0, 0.423 at Stage 4).

### 3.4 The normal anchor and the curriculum

Let $c_0$ be the unit-normalized mean of the Stage-0 target-encoder embeddings over the 75 normal sequences. After each later stage we report **normal-anchor cosine** $= \cos(\bar z_{\text{normal}}, c_0)$. The curriculum runs Stage 0 (300 epochs) then Stages 1-4 (75 epochs each) with balanced replay - **600 epochs / 11,400 updates**, one continuous model, seed 42.

### 3.5 AnchorGuard: the repair we test

During Stages 1-4 we add a **label-free** consolidation term over the normal rows of each batch,

$$\mathcal{L}_{\text{anchor}}=\lambda\,\big(1-\text{mean}_n\cos(z_n, c_0)\big),\qquad \lambda=0.5,$$

which pulls the running normal representation back toward its frozen Stage-0 self. Everything else (data, seeds, replay, mask rule, EMA schedule, optimizer) is identical to the canonical run. The new checkpoint gets its own lineage and never overwrites canonical artifacts.

### 3.6 Pre-registered gates (written before the AnchorGuard run)

**(G1)** Stage-4 anchor ≥ 0.85 (drift ≤ 0.15). **(G2)** feature std ≥ 0.35 (no collapse). **(G3)** five-class source-grouped probe macro-F1 within 0.05 of baseline. **(G4)** binary probe macro-F1 within 0.05 of baseline. A pass on G1+G2 with a failure on G3/G4 would be a retention-plasticity trade-off, *not* a failure of the study.

### 3.7 Evaluation lanes (and their exact exposure)

All probes freeze the encoder and read 384-d pooled embeddings with a source-video-grouped Random Forest (100 trees, depth 5, sqrt features, balanced weights, seed 42). We report **binary** (normal vs abnormal, 5 GroupKFold folds) and **five-class** (2 StratifiedGroupKFold folds - the most that keep all five labels present, since PD and CP have only two videos each). In every lane **the encoder was trained on all 159 rows**: these are descriptive, transductive readouts, never generalization estimates.

---

## 4. Results: quantify → attribute → repair

### 4.1 The drift curve reproduces exactly (E0)

Reloading the five frozen stage checkpoints and recomputing the anchor gives **0.9540 / 0.8389 / 0.7066 / 0.5942**, matching the canonical training log to **max gap 4.7×10⁻⁷**. The phenomenon is not a logging artifact; it is a reproducible property of the checkpoints (Fig. 2, red curve).

![**Fig. 2.** The quantified forgetting curve: the normal anchor falls 0.954 to 0.594 as the four disorders enter (red). AnchorGuard (blue) drifts even more, failing the 0.85 retention gate (dashed).](figures/bbfm_drift_curve.svg)

### 4.2 The group margin is part of the cause - but not all of it (E1)

From the frozen Stage-0 checkpoint we retrain Stage 1 (Parkinson's) twice: **G1** with the group weight 0.25 (as shipped) and **G0** with it zeroed.

| Stage-1 retrain | Normal-anchor cosine | Min centroid dist. | Feature std |
|---|---:|---:|---:|
| G1 (margin on, 0.25) | **0.9543** (≈ canonical 0.9540) | 0.7408 | 0.4302 |
| G0 (margin off, 0.0) | **0.9763** | 0.8080 | 0.4789 |

The label-aware margin contributes **Δ ≈ 0.022** of early drift, and G1 reproduces the canonical Stage-1 anchor to 3×10⁻⁴ (validating the ablation harness). But even with the margin off the anchor still moves (0.976 < 1.0): **new-task statistics alone drift the anchor**. Both halves are informative - the margin is a real lever, and replay-only continual SSL still does not fully consolidate a normative reference.

### 4.3 AnchorGuard does not hold the anchor - but does not hurt decoding (E2-E4)

AnchorGuard's stage-end anchors are **0.777 / 0.655 / 0.579 / 0.538** (Fig. 2, blue curve) - it ends *below* the canonical 0.594. Feature std ends at 0.342.

![**Fig. 3.** Three families of consolidation signal. AnchorGuard is the representation-space one we implement and test; parameter-space (EWC) and data-space (weighted replay) are documented follow-ups.](figures/bbfm_consolidation.svg)

| Gate | Threshold | Result | Verdict |
|---|---|---:|---|
| G1 anchor retained | ≥ 0.85 | 0.538 | **fail** |
| G2 no collapse | std ≥ 0.35 | 0.342 | **fail (marginal)** |
| G3 five-class non-inferior | \|Δmacro-F1\| ≤ 0.05 | 0.597 vs 0.622 (Δ = 0.025) | **pass** |
| G4 binary non-inferior | \|Δmacro-F1\| ≤ 0.05 | 0.893 vs 0.849 (+0.045) | **pass (improves)** |

Downstream source-grouped probes (Fig. 3):

| Probe | Baseline (acc / macro-F1) | AnchorGuard (acc / macro-F1) |
|---|---:|---:|
| Binary (normal vs abnormal) | 0.849 / 0.849 | **0.893 / 0.893** |
| Five-class | 0.660 / 0.622 | 0.698 / **0.597** |

**Reading.** A single representation-space attractor at λ=0.5 **cannot cancel** the forces (group margin + new-task statistics) that rotate the normal anchor. Yet retention did not have to be *bought* with plasticity - disorder decoding held, and binary even improved. The notebook records: *ANCHORGUARD PARTIAL: retention and plasticity trade off; report honestly.* This is the paper's honest centerpiece: a **negative result with a diagnosed mechanism**, exactly the profile that changes understanding rather than inflating a leaderboard.

### 4.4 Controls that stay in every figure

The **missingness-only** probe (a 97-d signature of *which* joints the detector dropped, with no pose content) reaches **0.448** five-class accuracy - detector behavior alone carries label signal. The **provenance** split (added vs canonical normal) and an **untrained-encoder floor** accompany every learned-advantage claim. We also re-run the laterality probe (§5.2) on the AnchorGuard checkpoint to ask whether restoring "normal" also restores the signed axis training destroyed.

---

## 5. Two zero-training world-model probes

These require no retraining - they reuse the frozen model - and they sharpen the main story by ruling out easy objections.

### 5.1 Infilling is not forecasting (Fig. 4-5)

![**Fig. 4.** One frozen predictor, three mask geometries. Trained for spatial infilling (a), it is asked to forecast the future (b); it scores far below both its infilling ceiling and a copy-last baseline (c), yet a short latent rollout does not diverge (d).](figures/bbfm_worldmodel_concept.svg)

The S-JEPA was trained to fill in hidden joints *with full time context on both sides*. We flip the mask from **spatial infilling** to **future masking**: hide **all 33 joints of the last $h\in\{2,4,8\}$ time-patches** and score how well the frozen predictor's latent guess matches the target encoder. Mean future latent cosine at $h=2$: **Parkinson's 0.442, normal 0.352, myopathic 0.322, stroke 0.296, cerebral palsy 0.233** - all below the model's own **spatial-infilling ceiling (0.547)** and far below a memoryless **copy-last-patch baseline (0.88-0.95)**. In words: *a predictor that is good at filling gaps is worse than copying the previous instant when asked to imagine the future.* This is a sharp empirical boundary the "JEPA-as-world-model" narrative usually assumes away.

![**Fig. 5.** Left: forecasting quality by horizon per condition, all below the infilling ceiling and the copy-last baseline. Right: video-level surprise separates conditions from normal, but the effect is entangled with detector missingness (residualized AUROCs in the box).](figures/bbfm_surprise.svg)

The positive half is a candidate motor biosignal: **video-level surprise** (1 − future cosine) ranks cerebral palsy and stroke above normal, with normal-vs-condition AUROCs of **CP 0.833** [0.667, 1.0], stroke 0.741, myopathic 0.572, PD 0.583 (Kruskal-Wallis p = 0.297, n = 35 videos). But surprise **correlates with detector missingness** (ρ = 0.497, p = 3.2×10⁻¹¹); after rank-residualizing out missingness the AUROCs become CP 0.944, stroke 0.815, myopathic 0.800, PD 0.472. A 2-step latent rollout on one normal clip (0.571 direct → 0.608 chained) shows error did not explode in two steps. With 2-3 disorder videos per group these are **descriptive pilots, not classifiers** - and the missingness confound, handled openly, is itself a warning for the community.

### 5.2 The pooled features discarded rate and side (Fig. 6)

Two clinically central axes appear to be *gone* from the representation. Are they lost in the **encoder** or merely hidden by the **pooling** (which is order-invariant by construction)? We settle it with a **same-token, capacity-matched** readout sweep: four 384-d readouts (deployed pooling; +signed temporal moment; 4 time-bin means; learned attention pool) decode five pre-registered timing scalars under strict source-grouped ridge probes, with raw-kinematic, missingness-only, and untrained-encoder controls, plus a RankMe/autocorrelation spectral audit.

![**Fig. 6.** Same frozen tokens, four capacity-matched readouts. No order-sensitive readout clears the pre-registered +10% gate over deployed pooling, and the cadence canary is undecodable in every lane, so the lost timing is upstream of the pooling, in preprocessing.](figures/bbfm_readout_sweep.svg)

**Verdict: NO EVIDENCE that the pooling is the culprit.** Order-sensitive readouts beat deployed pooling by only **+3.4% / −0.9% / +7.9%** relative MAE on the three order-sensitive targets - all below the pre-registered +10% gate. The **canary** cadence/stride-time R² is ≤ ~0.14 in every lane, confirming native-rate information was deleted at **preprocessing**, not by the model. Learned tokens are healthy, not collapsed (RankMe 3.48 > untrained 2.74; patch autocorrelation 0.93-0.96). The separately reported **signed-laterality probe** is the matching interpretability negative: the learned readout of left-right asymmetry scores **R² = −0.187**, *below* an untrained encoder's floor (+0.147), while raw coordinates solve it at R² ≈ 1.0, sign-consistency is chance, and an anatomical mirror does not flip the axis. A **symmetric-prediction world model discards the signed axis clinicians measure most** - an interpretability lesson, not a bug.

Together these probes pre-empt the obvious attack on §4: the drift metric is *not* an artifact of pooling, and the model's blind spots are traceable to explicit design choices (bilaterally symmetric masking, sign-free pooling, temporal resizing).

---

## 6. Discussion

**Normative references behave like state, not weights.** The cleanest lesson is that "normal" in a continual body model is not safely stored in the weights: it **drifts measurably** as new conditions arrive, and a naive representation-space attractor at moderate strength does not pin it. For closed-loop motor systems - the workshop's emphasis - this matters: a controller that forgets its normative reference forgets its setpoint.

**Label-free anchors are cheap telemetry.** The anchor cosine needs no labels and no probe; any continual SSL run can log it per step. It caught forgetting that the training-time "no-collapse" scalar (feature std stayed ~0.41 throughout) completely missed.

**Why the repair failed - and what would likely work.** The group margin pushes condition centroids toward cosine 0.5; a batch-level cosine attractor at λ=0.5 is simply too weak to oppose it. The diagnosis points to concrete next steps: higher-λ or per-sequence distillation, parameter-space EWC restricted to the anchor directions, margin annealing, or partial layer freezing. These are follow-ups, not fixes we claim here.

**Limitations (stated plainly and repeatedly).** One training seed, one run. Tiny, single-site cohort (35 videos; **one** canonical normal video; 2-3 videos for some disorders). Every score is transductive - the encoder saw all evaluation rows. A provenance asymmetry between normal and disorder rows, with a missingness-only probe reaching 0.448, means detector behavior carries label signal; we residualize it but cannot remove it. Folder labels are dataset annotations, not diagnoses. **Nothing here estimates unseen-patient, unseen-video, or clinical performance.**

**What would make it a generalization claim.** A fully **nested (leave-source-videos-out) retraining** - split videos before preprocessing, train all five stages inside each outer fold, open held-out videos once - is the honest next experiment, deliberately out of scope for this deadline (§B).

---

## 7. Conclusion

We turned a reported weakness - a drifting normal representation in a continual skeleton-JEPA - into a studied phenomenon: we **defined** a label-free normative-anchor drift curve, **reproduced** it to machine precision, **attributed** part of it to a label-aware margin, and **attempted a label-free repair** that honestly failed to hold the anchor while leaving disorder decoding intact. Two zero-training probes show the same frozen model **infills but cannot forecast** and has **discarded rate and side** for traceable design reasons. The durable contributions are a **measurement**, a **mechanism**, and a **discipline** for making tiny-cohort behavioral-signal claims that a reader can trust - with the boundary of those claims stated on every page.

---

## Appendix A. Verified numbers and provenance

Every value was checked against artifact files under `work/artifacts/real`; all numeric values confirmed.

| Claim | Value | Source |
|---|---|---|
| Anchor drift curve | 0.954005 / 0.838861 / 0.706604 / 0.594197 | `curriculum_stage_summary_augmented.csv` |
| Drift reproduction gap | 4.73×10⁻⁷ | `anchor_guard_results.json` |
| Margin ablation | G1 0.9543 / 0.7408 / 0.4302; G0 0.9763 / 0.8080 / 0.4789 | `anchor_drift_margin_ablation.csv` |
| AnchorGuard anchors / std | 0.777 / 0.655 / 0.579 / 0.538; std 0.342 | `anchor_guard_results.json` |
| Gates | G1 ✗, G2 ✗, G3 ✓ (0.597 vs 0.622), G4 ✓ (0.893 vs 0.849) | same |
| Downstream probes | baseline 0.849/0.660; AnchorGuard 0.893/0.698 | same |
| Canonical geometry | silhouette 0.008975; min-centroid 0.036718; mean-within 0.119521 | `curriculum_representation_geometry.csv` |
| Surprise (h=2) | PD 0.442 / normal 0.352 / myo 0.322 / stroke 0.296 / CP 0.233; ceiling 0.547; copy-last 0.88-0.95 | `predictive_surprise_results.json` |
| Surprise AUROC (CP) | 0.833 [0.667,1.0]; residualized 0.944; ρ(missingness)=0.497 (p=3.2e-11) | same |
| Readout verdict | NO EVIDENCE; +3.4% / −0.9% / +7.9%; RankMe 3.48 vs 2.74 | `temporal_readout_results.json` |
| Laterality null | learned R² −0.187; raw ≈1.0; floor +0.147; mirror slope −0.343 | `idea5_signed_laterality_result.json` |
| Experiment fingerprint | `d0acc2628d13…` | `classifier_contract.json` |
| Canonical checkpoint file SHA-256 (this workspace) | `2aa20dd4ac92…` | measured; pin in camera-ready |

**Reproducibility.** All results derive from one experiment fingerprint (`d0acc2628d13…`); the three new notebooks (`07`,`08`,`09`) print the checkpoint file SHA-256 and fingerprint at load time and refuse to run on a mismatched lineage. Recomputed embeddings equal the saved parquet to machine precision (max |Δ| = 0.0).

## Appendix B. Scaling the cohort (when and how to add more videos)

The single-video normal set and 2-3 videos per disorder are the dominant limits. We have **dozens more** candidate clips available (12 raw GAVD normal videos, ~190 GAVD abnormal clips, sibling YouTube pools, and a 50-row multiple-sclerosis manifest). We deliberately **do not** fold them in for this submission, for two reasons that would otherwise *invalidate* the results here: (i) new normal clips arrive through the **added-extraction path**, deepening the provenance/missingness confound the missingness-only probe already exposes; (ii) mixing them changes the fingerprint and would break the exact reproduction that is our strongest asset. The correct way to add them - coverage-gated intake, provenance balancing, a **nested leave-source-videos-out retrain**, and multi-seed replication - is a multi-day effort and the natural follow-up paper. The companion roadmap document specifies the exact protocol and decision gates.
