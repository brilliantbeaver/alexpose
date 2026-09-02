# Archived planning note

> Historical only. This plan predates the completed 12-target, five-stage augmented-normal run and its corrected evaluation. Its 10-target, 61.9% accuracy, data-size, and pending-training assumptions are not current results. Use `README.md`, `docs/progressive_training.md`, and `docs/staged_details.md` for the current experiment and revision ledger.

# S-JEPA Gait Classifier: Improvement Plan

**Goal.** Significantly and *honestly* improve five-class gait-condition classification
(normal, Parkinson's, stroke, cerebral palsy, myopathic) in
`06_capstone_health_condition_classifiers.ipynb`, where a frozen Skeleton-JEPA
representation currently reaches **61.9 % accuracy / 0.613 macro-F1** on the exact
47/21 split — **14 points below** the 82-feature handcrafted baseline (76.2 % / 0.728),
and only ~17 points above a **pose-missingness-only shortcut** (45 %).

This plan is grounded in a systematic review of notebooks `00`–`06` plus a fanned-out
literature study (I-JEPA, V-JEPA / V-JEPA 2, VideoMAE, DINO/DINOv2, VICReg,
MAMP, MotionBERT, 2s-AGCN, logit-adjustment, and the GAVD dataset paper). Every lever
below was adversarially stress-tested against this project's hard constraints
(96 sequences / 18 videos, normal = **1** video, CPU/laptop compute, undergraduate scope,
URTC honesty-first framing).

---

## 1. Context — why the current number is what it is

Six mutually-reinforcing root causes, all confirmed against the code:

| # | Root cause | Evidence in the notebooks |
|---|------------|---------------------------|
| **R1** | **Trivial pretext task.** Only 10 of 33 joints are ever maskable and 60 % of *those* → **~18 % global mask ratio** (I-JEPA/V-JEPA use 75–90 %). The context encoder always sees a near-complete skeleton, so the predictor solves spatial infilling, not motion modelling. | `03` `uniform_neurologic_mask`, `MASK_KEYPOINTS` (10 joints); `04` cell 16 `mask_fraction=0.60`. |
| **R2** | **Single-video, single-pattern pretraining.** 12 normal clips from **one** YouTube video, ~900 optimizer updates. The encoder never saw Parkinsonian / hemiparetic / CP / myopathic dynamics. | `04` cell 12 "all normal sequences share one source video… transductive". |
| **R3** | **Dynamics are deleted twice.** (a) `temporal_resize` to a fixed 64 frames erases absolute cadence/speed — the top clinical discriminator; (b) downstream pooling is **order-invariant global mean/std**, discarding rhythm, phase, and asymmetry timing. | `04`/`06` `prepare_sequence` resize; `06` cell 16 `masked_mean_std` + `pooled_embeddings` (384-d = mean/std over all tokens + over 10 neuro tokens). |
| **R4** | **Tracking-artifact shortcut.** MediaPipe **LITE**, no smoothing, conf 0.45; missing joints → **0 sentinel that collides with the pelvis origin (0,0,0)**. Feet/ankles (the masked *targets*) are the least reliable joints. Missingness alone predicts 45 % of the label. | `02` LITE model + conf 0.45, no filter; `center_and_scale` 0-sentinel; `06` cell 14 missingness-only baseline. |
| **R5** | **Leaky, single-split evaluation.** *Every* test video also appears in training (`assert set(test_videos).issubset(train_videos)`), so 61.9 %/62.1 % are **optimistically biased**. No cross-validation, no confidence intervals. A video-disjoint 5-class split is **structurally impossible** while normal = 1 video. | `06` cell 25 leakage audit + assert; cell 30 "BLOCKED". |
| **R6** | **Fragile objective + weak collapse guard at tiny N.** DINO centering+sharpening only, **no active variance/covariance term**; EMA momentum runs `0.999 → 1.0`, so the teacher is *literally frozen* by the last steps; collapse is only *monitored* post-hoc. | `00`/`04` `sjepa_cross_entropy`, `cosine_ema(..., end=1.0)`, `collapse_diagnostics`. |

**Consequence:** on its own, honestly evaluated, the frozen S-JEPA embedding is unlikely
to beat the handcrafted baseline in this program. Its credible role is to **add** a few
points on top of clinical features, or to **match** them — presented as an honest,
leakage-resistant result. The single most valuable scientific move is to convert an
inflated 62 % into a *defensible* number with a proper protocol, then push that number up.

---

## 2. Prioritized levers (ranked by honest gain ÷ effort)

Ranking reflects the adversarial verdicts, not raw enthusiasm. Effort is relative to an
undergraduate on a laptop. "Re-pretrain" = must rerun notebook 04.

### Tier A — do first (high ROI, mostly no re-pretraining)

**A1. Dynamics-preserving downstream pooling (parameter-free).** — *nb06, effort low, +5–12 pts*
Replace the global `masked_mean_std` with, per the frozen encoder's `[16 time × 33 joint × D]`
grid: (a) **per-time-segment** mean/std (keeps the 16-step temporal profile);
(b) **parameter-free cadence** features — dominant frequency via FFT / autocorrelation of the
per-segment token trajectory, plus cycle-to-cycle coefficient of variation;
(c) **signed left-vs-right homologous-joint differences** over time (shoulders/hips/knees/
ankles/feet) — the asymmetry axis that separates hemiparetic stroke (F1 = 0.33 today) and CP.
*Why parameter-free, not a GRU/attention head:* a trainable head on ~67 leaky training rows
memorizes video identity and inflates the confounded split. All five research streams
converge here; it reuses the frozen encoder with zero retraining.
*Evidence:* MotionFormer / attentive-probing literature; the winning 82-feature RF is built
almost entirely from exactly these spatiotemporal/asymmetry parameters.

> FORWARD POINTER (added 2026-08-19). Item (c), the signed left-versus-right axis, is no longer an open
> question. It was later measured directly in Idea 5 and Idea 9, so treat this plan bullet as history
> rather than as a live hypothesis. What we did: fit a linear read of a signed left-minus-right target
> from the frozen `ea59fea0` encoder on source-video-disjoint folds, against a raw-coordinate null, an
> untrained-encoder floor, and a side-blind pooled control. What we observed: the learned lane scored
> R-squared -0.602 while the untrained floor scored -0.156 and the side-blind control -0.131, so two
> controls beat the treatment; sign consistency was 44.4 percent against a required 75 percent. What we
> may conclude: the frozen representation does not make this axis linearly available, so a signed
> left-versus-right pooling feature should not be expected to recover stroke or cerebral palsy on its own
> at this cohort size. Verdict: INFORMATIVE NULL. Two follow-ups closed the obvious escape routes and
> neither rescued the idea. See
> [ideas-claude/09-reflection-equivariant-symmetry-axis/IMPLEMENTATION.md](./ideas-claude/09-reflection-equivariant-symmetry-axis/IMPLEMENTATION.md)
> section 9a.

**A2. Honest evaluation protocol + reference lines.** — *nb01+nb06, effort medium, accuracy-neutral→negative but essential*
Replace the single 70/30 `exp5_exact_split` with **GroupKFold / leave-one-video-out keyed on
`video_id`**, reporting **mean ± bootstrap 95 % CI** for accuracy and macro-F1. On *every*
reported split, always print the three reference lines: **majority-class (49 %)**,
**missingness-only RF** (already computed), and **82-feature handcrafted RF (76.2 %)** — a
JEPA number is only interpretable relative to these. This is the credibility backbone of an
URTC honesty-first paper; honest numbers will *drop* from 62 %, but every other lever's gain
is unverifiable without it.

**A3. De-confound pose-missingness + input quality.** — *nb02/nb04/nb06, effort medium, small (possibly-negative) but correct*
Replace the 0-sentinel with an **explicit per-joint visibility channel** so "missing" and
"at-origin (0,0,0)" are distinguishable; add **tremor-band-preserving temporal smoothing**
(One-Euro / Savitzky-Golay) and short-gap interpolation. Report classifier accuracy **with and
without** missingness covariates to *prove* the embedding adds signal beyond the detector
artifact. Optionally upgrade MediaPipe **LITE → FULL** (defer HEAVY; CPU cost). This is a
prerequisite for harder masking (A5/A6) and for any velocity stream (derivatives of noisy feet
= noise). *Caveat:* if pathology *causally* occludes feet (CP toe-drag), residualizing
missingness may remove real signal — hence report both variants.

**A4. Reintroduce the rate axis + swap/augment the head.** — *nb06, effort low*
Concatenate **scalar rate features computed from the RAW pre-resize track** (native fps,
duration, FFT/autocorrelation cadence, stride time) into the downstream vector — cheaply
recovering the clinical cadence axis that `temporal_resize` destroys. Add a **regularized
logistic-regression / linear-SVM linear probe** (the standard low-variance SSL-quality metric)
*alongside* the RF, and apply **logit adjustment** (Menon et al., ICLR 2021) or class-balanced
weighting for the myopathic-heavy (~49 %) imbalance and stroke collapse.

### Tier B — the accuracy floor and honest attribution

**B1. Hybrid feature model (embedding ⊕ clinical) with explicit ablation.** — *nb06, effort medium, raises floor to ~mid-70s honest*
Build the downstream vector as **[temporally-pooled S-JEPA embedding] ⊕ [~30–50 clinical
spatiotemporal features from the raw track: cadence, stride/step time, step-time asymmetry /
symmetry index, double-support fraction, arm-swing amplitude & asymmetry, trunk sway, knee ROM,
foot clearance]**. **Report handcrafted-alone, embedding-alone, and fused separately** so
S-JEPA's *marginal* contribution is visible, not laundered behind the handcrafted win. This is
the single most reliable accuracy lever, but it does not by itself answer "does SSL learn gait"
— hence the mandatory three-way ablation. Depends on A3 (clean feet) for trustworthy
cadence/clearance features.

### Tier C — representation surgery (requires re-pretraining; gate behind Tier A)

**C1. Broaden SSL pretraining to GAVD scale + EMA/temperature hygiene.** — *nb01/nb02/nb04, effort medium, +3–8 pts contingent*
Stop pretraining on 12 clips / 1 video. Rerun `04` on **hundreds of unlabeled GAVD skeletons**
(SSL uses no labels, so *all* conditions are fair game) and add **5–10 independent normal
videos** so a video-disjoint 5-class split becomes possible. Scale ~900 → **≥20k updates**.
Hygiene fixes: **cap `cosine_ema` end at 0.9995** (not 1.0, which freezes the teacher) and
**warm the teacher temperature 0.10 → 0.06** over the first ~30 % of steps.
*Gate:* log the train/test *video* partition **before** pretraining, or you manufacture encoder
leakage. Pays off only if paired with C2 (else more data flows into a still-trivial mask task).

**C2. Non-trivial SSL objective: all-joint high-ratio block masking + active variance floor.** — *nb03/nb04, effort medium, moderate but strictly gated*
Make **all 33 joints maskable** (currently capped at 10/33 ≈ 30 % ceiling); raise the global
ratio to **40 → 60 %** with **contiguous space-time block/tube masks** (mask a whole limb or one
full side across time), keeping the neurologic set as a **sampling *bias*, not a hard gate**.
**Exclude detector-invalid tokens from the *target* set** (never predict a zero-sentinel foot).
Add an **active VICReg-style variance-hinge** term on the L2-normalized target embeddings,
pooled over tokens and accumulated over steps, as a collapse guardrail *before* raising
difficulty. *Highest collapse/overfit risk* — only safe on the broadened corpus (C1), after
the visibility fix (A3), validated under the honest protocol (A2).

**C3. Embedding-dimension `D` — sweep, don't bump; gate width on data breadth.** — *nb04, effort low-medium, contingent*
`D` is currently **96**, chosen (per `docs/staged_details.md` §24) only because it is compact
and divisible by the 4 heads — **there is no width ablation**. Widening it is tempting but is a
**capacity** change, and here the binding constraint is **data scale, not capacity**: pretraining
sees only **12 correlated sequences from 1 video** (~12 × 528 ≈ 6.3k highly-correlated tokens).
Transformer parameters are dominated by the `D²` terms (attention projections ≈ 4·D² + FFN at
`dim_feedforward=4·D` ≈ 8·D², so ≈ 12·D² per encoder layer). Relative to the current 700,800
params:

| `D` | width vs 96 | `D²`-term param scale | verdict on the *current* 1-video corpus |
|-----|-------------|-----------------------|------------------------------------------|
| 96 (now) | 1.0× | 1.0× | baseline |
| 128 | 1.33× | ≈1.8× | mild; safe to try |
| 192 | 2.0× | ≈4× | only with broadened corpus |
| **256** | 2.67× | **≈7×** | **only after C1**; else memorizes the one source video |
| 512 | 5.3× | **≈28×** | **do not** on the present corpus — bottleneck argument does not apply when *data*, not width, is the limiter |

**Recommendation:** treat `D` as an **ablation axis**, not a fixed jump. Sweep
`D ∈ {96, 128, 192, 256}` (all divisible by 4 heads), **gate any move to ≥256 on first
broadening the normal-video cohort (C1)**, keep the existing collapse diagnostics (feature std,
pairwise cosine) as the guardrail, and report the sweep under grouped source-disjoint evaluation
(A2) at matched compute — fold it into the same ablation program as the mask/objective changes.
**Do not jump straight to 512.** Widening `D` before the data is broadened will very likely
increase single-video memorization, not representation quality. (Note: if you adopt velocity/bone
input streams from D1, the *token input* dim grows from 12 — that is orthogonal to the *embedding*
`D`; keep them as separate ablation knobs.)

### Tier D — optional / ablation-only (evaluate only if C shows the encoder learns dynamics)

- **D1. Velocity/acceleration (+ bone) input streams and a *motion* target** (MAMP / MotionBERT /
  2s-AGCN): augment the 12-d token with first/second differences and predict masked **velocity**
  latents rather than positions. Requires A3 smoothing first. Re-pretrain.
- **D2. Skeleton topology bias** — low-effort variant only: a **static adjacency + symmetric-bone
  additive bias** on attention logits (reject full ST-GCN rewrite as over-scope). Re-pretrain.
- **D3. A small attentive-probe / covariance (diagonal, top-k) pooling head** — only under
  GroupKFold, <10k params, heavy weight decay. Never report a trainable-head number on the
  current leaky split.
- **D4. External upper-bound baseline** — adapt a released **MotionBERT** (AMASS/H3.6M) encoder to
  MediaPipe-33 as a *context* comparison, **not** the headline result (off-mission; weights &
  topology mismatch).

---

## 3. Sequenced experiment plan

```
STEP 0  Honesty foundation (parallel with Step 1) ...... A2 + A3
        GroupKFold-by-video + bootstrap CIs; visibility channel replacing the (0,0,0)
        collision; One-Euro smoothing + short-gap interpolation; always print
        majority / missingness / handcrafted baselines. Raises no accuracy; makes
        every later number trustworthy.

STEP 1  Highest ROI, no re-pretrain ................... A1 (+ A4)
        Per-time-segment + parameter-free dynamics pooling (cadence, cycle CV,
        signed L-R asymmetry) + scalar rate features + linear-probe & logit-adjust.
        Measure on the honest split vs baselines. Biggest single honest lever.

STEP 2  Accuracy floor + honest attribution ........... B1
        Hybrid [embedding ⊕ clinical] RF/linear. Report handcrafted-alone,
        embedding-alone, fused separately.

STEP 3  Unblock honest eval + representation .......... C1
        Broaden nb04 pretraining to hundreds of GAVD videos; +5–10 normal videos;
        ≥20k updates; EMA end 0.9995; teacher-temp warmup. Log video partition first.

STEP 4  Non-trivial objective (only after Step 3) ..... C2 (+ C3)
        All-joint 40–60 % block/tube masking + active VICReg variance floor.
        Sweep embedding dim D ∈ {96,128,192,256} — width move to ≥256 gated on the
        broadened corpus. Re-pretrain on the broadened corpus only.

STEP 5  Optional, gated behind 0+3 .................... D1–D4
        Velocity streams / motion target / adjacency bias / attentive probe /
        MotionBERT upper-bound. Trainable heads only under GroupKFold.

STEP 6  Ablations ..................................... mask-geometry, target-type,
        pooling — the controlled comparisons the paper currently lacks.
```

**Mandatory ablations for the paper** (each on the honest split, with CIs):
mask geometry (neurologic-10 vs all-joint-block vs motion-aware), prediction target
(position vs velocity latent), pooling (global mean/std vs per-segment+dynamics vs attentive),
**embedding width `D` ∈ {96,128,192,256}**, head (RF vs linear probe), and features
(handcrafted vs embedding vs fused).

---

## 4. Realistic honest ceiling

A video-disjoint 5-class ceiling is roughly **70–80 % accuracy (macro-F1 ≈ 0.65–0.75)**, set
**almost entirely by the handcrafted clinical spatiotemporal features** (the 82-feature RF
already hits 76.2 % *with* leakage; video-disjoint it likely lands high-60s to mid-70s). The
frozen S-JEPA embedding, alone and honestly evaluated, is **unlikely to independently exceed**
that baseline here; its credible honest role is to **add a few points in a fused model or match
it**. Three things dominate the ceiling, in order:

1. **Data / leakage** — 96 seq / 18 videos, normal = 1 video → no honest 5-class split exists
   today; addressable only by adding GAVD videos.
2. **Input-signal destruction** — `temporal_resize` erases cadence; LITE + unsmoothed feet +
   (0,0,0) sentinel inject a tracking-artifact shortcut.
3. **Representation** — trivial ~18 % infilling on one subject over ~900 updates with a frozen
   teacher cannot have learned transferable gait dynamics.

**The strongest deliverable** is therefore an honest, leakage-resistant hybrid result in the
**low-to-mid 70s with proper CIs**, explicitly reframing the prior 61.9 % as inflated — a more
credible scientific contribution than a fragile within-corpus SOTA claim.

---

## 5. Key references (grade each before citing)

| Paper | Venue / year | Relevance | Citation status |
|-------|--------------|-----------|-----------------|
| **I-JEPA** — Self-Supervised Learning from Images with a JEPA | CVPR 2023 | JEPA needs large, informative, high-ratio target blocks — indicts the ~18 % mask. | `arXiv:2301.08243` ✔ verified |
| **V-JEPA** — Revisiting Feature Prediction for Video | 2024 | Spatiotemporal multiblock/tube masking + smooth-L1 feature regression + **frozen-backbone / attentive-probe** eval. | `arXiv:2404.08471` ✔ verified (Something-Something-v2 72.2 %) |
| **V-JEPA 2** — Self-Supervised Video Models… | 2025 | Scale + latent prediction; frozen-feature probing (SSv2 77.3 %). Template for data-scale argument. | `arXiv:2506.09985` ✔ verified |
| **VICReg** — Variance-Invariance-Covariance Regularization | ICLR 2022 | Active variance-hinge collapse guardrail to bolt onto the DINO loss at tiny N. | `arXiv:2105.04906` ✔ verified |
| **DINO** — Emerging Properties in SS ViTs | ICCV 2021 | Source of the centering+sharpening objective; motivates teacher-temp warmup & momentum hygiene. | `arXiv:2104.14294` ✔ verified |
| **VideoMAE** — Data-Efficient Masked Video Pretraining | NeurIPS 2022 | High-ratio tube masking as the driver of temporal SSL, data-efficient on small video sets. | `arXiv:2203.12602` — recall, verify |
| **MAMP** — Masked Motion Prediction for Skeleton SSL | ICCV 2023 | Predicting **velocity** beats masked-coordinate reconstruction for skeletons. | arXiv id **unverified** (`2308.07092?`) — confirm before citing |
| **MotionBERT** — Unified Human Motion Representations | ICCV 2023 | Masked 2D→3D motion pretraining on noisy monocular poses; candidate external upper-bound. | arXiv id **unverified** (`2210.06551?`) — confirm before citing |
| **2s-AGCN** — Two-Stream Adaptive GCN | CVPR 2019 | Bone + motion streams beat positions alone → velocity/bone channels. | `arXiv:1805.07694` — recall, verify |
| **Logit Adjustment** — Long-Tailed Recognition via Logit Adjustment | ICLR 2021 | Principled label-frequency correction beyond RF `class_weight`. | `arXiv:2007.07314` — recall, verify |
| **GAVD** — Gait Abnormality in Video Dataset | IEEE Access 2025 | 1,874 seq / >450 videos — enables broader pretraining, more normal videos, honest disjoint eval. | DOI **unverified**; `arXiv:2309.01480` did **not** resolve to GAVD on fetch — confirm the exact IEEE Access citation before using |

> **Citation hygiene.** Items marked "unverified" came from model recall, not a resolved
> fetch. Two arXiv IDs I probed during this study (a guessed MAMP id and `2309.01480` for GAVD)
> resolved to *unrelated* papers. **Confirm every id/DOI against the actual abstract before it
> enters the manuscript.**

---

## 6. What this plan deliberately does *not* re-propose

The paper's own "Next Steps" already list: independent-video cohort, source/person-grouped
splits, grouped CV with uncertainty, mask-geometry ablations, probing embeddings for gait
variables, objective comparisons, and robustness testing. This plan **operationalizes** those
(A2, C1, mask ablations) *and* goes beyond them with concrete, code-anchored levers the docs
never specify: dynamics-preserving parameter-free pooling (A1), the missingness-shortcut
de-confound with a with/without ablation (A3), rate-feature reintroduction (A4), the
three-way hybrid attribution (B1), an **active** variance floor + EMA/temperature hygiene (C2),
and motion-target / topology-bias / attentive-probe options (D).
