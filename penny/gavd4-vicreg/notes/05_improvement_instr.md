# Archived planning instruction

> Historical only. This instruction predates the completed 12-target, five-stage augmented-normal run and its corrected evaluation. Its old checkpoint, 10-target, and 61.9% accuracy assumptions must not be reported as current. Use `README.md`, `docs/progressive_training.md`, and `docs/urtc2026_details.md` for the current experiment and revision ledger.

# Master Instruction: Refine the S-JEPA Gait Notebooks to Significantly Improve Classification Accuracy

> **What this file is.** A ready-to-paste prompt for a frontier reasoning model
> (**Anthropic Opus 4.8** or **Fable 5**) running inside **Claude Code**, directing it to
> correct and refine the notebooks in `penny/gavd3/` so that five-class gait-condition
> classification in `06_capstone_health_condition_classifiers.ipynb` improves in an
> *honest, leakage-resistant* way. It encodes best-practice prompting: an explicit role, a
> hard "ultrathink" reasoning directive, deep authoritative-source research (arXiv / ACM /
> IEEE), **Codex-driven adversarial review**, and **Claude Code dynamic-workflow subagent
> fan-out**. Copy everything between the `═══` rules into the model.
>
> Companion: `notes/04_improvement_plan.md` (the research-backed lever analysis this prompt
> operationalizes). The model should read it first.

═══════════════════════════════════════════════════════════════════════════════

## ROLE

You are a **principal ML research engineer** specializing in self-supervised learning
(JEPA family), skeleton/video representation learning, and clinical gait analysis, working
inside **Claude Code** with subagent orchestration, a Codex adversarial-reviewer agent, and
`WebFetch`. You are refining a real undergraduate research project targeting **MIT URTC 2026**.
Your north star is a **defensible, leakage-resistant improvement** in five-class gait
classification — *not* a fragile within-corpus SOTA number. An honest 74 % with confidence
intervals beats an inflated 85 % that dies on an unseen video.

## REASONING DIRECTIVE — ultrathink

**Ultrathink before you act.** Think step by step, at length, and expose your reasoning in a
`<thinking>` block before every irreversible action (editing a notebook, launching a
workflow, committing). Explicitly weigh: *does this change raise a genuine, leakage-resistant
number, or does it only inflate the confounded split?* Enumerate at least two alternatives and
say why you rejected them. When you are uncertain about a research fact, **do not guess — verify
it** (see RESEARCH below). Prefer being slow and correct to fast and wrong.

## PRIME DIRECTIVE — honesty first

This corpus is **96 sequences / 18 YouTube videos**, and the **normal** class from GAVD comes 
from a **single** video.  To augment the "normal" class, we will use the "normal" videos in
folder "data-videos/normal".  For these augmenting "normal" videos, use your best video pose 
estimation capabilities (such as MediaPipe) to first annotate individual gait sequences with 
bounding box on the moving subject.  Use these results & information to write to the GAVD "csv"
file for processing by the notebooks.

Therefore:

1. **Never report or optimize a video-leaky metric.** All headline numbers must come from
   **GroupKFold / leave-one-video-out keyed on `video_id`**, with **bootstrap 95 % CIs**.
2. **Always print three reference lines** on the same split: majority-class (~49 %),
   **missingness-only RF**, and the **82-feature handcrafted RF (76.2 %)**.
3. A **pose-missingness-only baseline already reaches 45 %** — treat any gain as suspect until
   you have shown it survives *with-and-without* missingness covariates. Beating the shortcut,
   not beating chance, is the bar.
4. If a change would only help the leaky split, **say so and do not present it as progress.**

Document thoroughly what is meant by "missingness-only RF" in the noteboo
ks, in README.md, 
as well as those documents in "docs".

## GROUND TRUTH — read before touching anything

Read, in order: `notes/04_improvement_plan.md` (the plan you are executing), then
`README.md`, `docs/urtc2026_sjepa_gait.md`, and notebooks `00`, `02`, `03`, `04`, `06`.
Confirm these code facts *from the source* (do not trust this summary blindly):

- **Objective** (`00`/`04` `sjepa_cross_entropy`): DINO-style soft cross-entropy in latent
  space, centering (β=0.9) + sharpening (teacher τ=0.06, predictor τ=0.10). **No variance /
  covariance term.** EMA momentum `cosine_ema(..., start=0.999, end=1.0)` → teacher frozen by
  the last steps. Collapse only *monitored* (`collapse_diagnostics`), never penalized.
- **Masking** (`03`/`04` `uniform_neurologic_mask`, `MASK_KEYPOINTS`): only **10 joints**
  eligible, 60 % of those → **~18 % global** (I-JEPA/V-JEPA/VideoMAE use 75–90 %). Random, not
  block. Context encoder sees a near-complete skeleton → trivial infilling.
- **Input** (`04` cell 12 `prepare_sequence`): MediaPipe **LITE**, 33 joints, (x,y,rel-z),
  **temporally resized to fixed 64 frames** (erases cadence/speed), pelvis-centred; **missing
  joints → 0 sentinel colliding with the (0,0,0) origin**; **no temporal smoothing**; feet/ankles
  least reliable and they are the masked *targets*.
- **Pretraining** (`04` cells 12–16): **12 normal clips, 1 video**, ~900 updates, AdamW lr 1e-3.
- **Downstream** (`06` cell 16 `pooled_embeddings`/`masked_mean_std`, cell 18 `make_rf`):
  frozen target encoder → **384-d = mean/std over all tokens ⊕ mean/std over the 10 neuro
  tokens** (order-invariant; dynamics discarded) → RandomForest(100, depth 5, balanced).
- **Eval** (`06` cells 20–30): single stratified split + `exp5_exact_split`; leakage audit
  asserts test-videos ⊆ train-videos; 5-class video-disjoint eval is **BLOCKED** (normal=1).

## SUCCESS CRITERIA

- A reproducible **GroupKFold-by-video** protocol with mean ± 95 % CI, and the three baselines
  printed on every split.
- Measurable, honest improvement in **macro-F1** (especially **stroke**, currently 0.33) on the
  disjoint protocol, *or* a rigorously argued explanation of the data-bound ceiling if the
  encoder cannot deliver it.
- Every representation change accompanied by the **ablation** that isolates its effect
  (mask geometry, target type, pooling, head, features).
- No non-finite artifacts; `smoke` mode still runs end-to-end; checkpoints keep their audit
  fingerprints; the smoke/real artifact separation is preserved.

═══════════════════════════════════════════════════════════════════════════════

## WORK PLAN — execute in staged order (from `notes/04_improvement_plan.md`)

Do **not** do everything at once. Land each stage, measure on the honest protocol, and only
then proceed. Re-pretraining stages (C1/C2) are gated behind the honesty foundation.

- **STEP 0 — Honesty foundation** *(nb01/nb02/nb06; no re-pretrain).*
  GroupKFold/leave-one-video-out by `video_id` + bootstrap CIs; replace the 0-sentinel with an
  explicit **visibility channel** (distinguish "missing" from "at origin"); add One-Euro /
  Savitzky-Golay smoothing + short-gap interpolation; print majority / missingness / handcrafted
  baselines everywhere.
- **STEP 1 — Dynamics-preserving pooling** *(nb06; no re-pretrain; biggest honest lever).*
  Replace global `masked_mean_std` with **per-time-segment mean/std** + **parameter-free
  dynamics** (FFT/autocorrelation cadence, cycle-to-cycle CV, **signed left-vs-right homologous-
  joint differences**). Add scalar **rate features from the raw pre-resize track**. Add a
  regularized **linear probe** alongside RF and **logit-adjustment** for imbalance. *No trainable
  temporal head on the leaky split.*
- **STEP 2 — Hybrid + honest attribution** *(nb06).*
  Downstream vector = **[pooled embedding] ⊕ [~30–50 clinical spatiotemporal features]**. Report
  **handcrafted-alone / embedding-alone / fused** separately so S-JEPA's marginal value is
  explicit, never laundered.
- **STEP 3 — Broaden pretraining + hygiene** *(nb01/nb02/nb04; re-pretrain).*
  Pretrain on **hundreds of unlabeled GAVD skeletons** (all conditions — SSL uses no labels);
  add **5–10 independent normal videos**; scale ~900 → **≥20k updates**; **cap EMA end at
  0.9995**; **warm teacher τ 0.10→0.06**. **Log the train/test video partition BEFORE
  pretraining** so you don't manufacture encoder leakage.
- **STEP 4 — Non-trivial objective** *(nb03/nb04; re-pretrain; only on the broadened corpus).*
  Make **all 33 joints maskable**; raise global ratio to **40–60 %** with **space-time
  block/tube** masks; keep the neurologic set as a **sampling bias, not a hard gate**; **exclude
  detector-invalid tokens from the target set**; add an **active VICReg variance-hinge** term as a
  collapse guardrail. **Sweep the embedding dimension `D` ∈ {96,128,192,256}** (all divisible by
  the 4 heads) as an ablation axis — **do not jump to 512, and gate any move to ≥256 on the
  broadened corpus (Step 3)**: transformer params scale ≈`D²` (256 ≈ 7× params, 512 ≈ 28× vs the
  current 96), so widening before the data is broadened memorizes the single source video rather
  than learning transferable gait. Keep the token *input* dim (velocity/bone streams, Step 5)
  separate from the embedding `D` knob.
- **STEP 5/6 — Optional & ablations.** Velocity/bone streams + **motion-target** prediction
  (MAMP/MotionBERT/2s-AGCN), static **adjacency/bone attention bias**, small **attentive probe**
  (GroupKFold-only), optional **MotionBERT** external upper-bound. Then the full ablation grid.

═══════════════════════════════════════════════════════════════════════════════

## RESEARCH — deep, authoritative, verified (arXiv / ACM / IEEE)

Before Steps 3–4 (and to ground the paper), **research the latest JEPA / skeleton-SSL / gait
literature from authoritative sources** and confirm every fact you rely on.

**Method.** You have `WebFetch` (fetch a *known* URL and query it) but **no open web search**.
So: (1) reason from your own knowledge to name candidate papers and their claims; (2)
**verify each candidate by fetching its canonical URL** — prefer `arxiv.org/abs/<id>`, then
ACM Digital Library / IEEE Xplore DOIs; (3) **mark every citation's confidence** and **never
cite an id you have not resolved** — two ids probed during the review (a guessed MAMP id and
`arXiv:2309.01480` for GAVD) resolved to *unrelated* papers. When in doubt, downgrade to
"unverified — confirm before citing."

**Anchor set to verify and mine** (confirm ids; extract the specific mechanism you will reuse):

| Topic | Paper | Verify |
|-------|-------|--------|
| JEPA + high-ratio target blocks | I-JEPA | `arXiv:2301.08243` |
| Video feature-prediction, tube masking, smooth-L1, frozen/attentive probe | V-JEPA | `arXiv:2404.08471` |
| Scale + latent prediction, frozen-feature probing | V-JEPA 2 | `arXiv:2506.09985` |
| Active variance/covariance anti-collapse | VICReg | `arXiv:2105.04906` |
| Centering+sharpening; teacher-temp warmup; momentum schedule | DINO | `arXiv:2104.14294` |
| High-ratio tube masking, data-efficient small-video SSL | VideoMAE | verify id |
| **Predict velocity, not coordinates**, for skeleton SSL | MAMP | **id unverified — resolve** |
| Masked 2D→3D motion pretraining on noisy monocular pose | MotionBERT | **id unverified — resolve** |
| Bone+motion multi-stream beats positions | 2s-AGCN | verify id |
| Principled long-tail correction | Logit Adjustment | verify id |
| The dataset (1,874 seq / >450 videos) | GAVD (IEEE Access 2025) | **DOI unverified — resolve** |

**Deliverable of research:** a short `notes/06_literature_findings.md` — per paper: verified
citation, the *one mechanism* you will port, and how it maps to a specific notebook cell.

═══════════════════════════════════════════════════════════════════════════════

## ORCHESTRATION — Claude Code dynamic-workflow subagent fan-out

Do not do this linearly in one context. **Fan out with the `Workflow` tool (dynamic
orchestration)** and independent subagents. Recommended structure — adapt as findings dictate:

1. **Understand (parallel readers).** One subagent per notebook (`00`,`02`,`03`,`04`,`06`) →
   each returns a structured map of the exact functions/variables to change and the current
   printed numbers. Barrier; merge into a shared change-list.

2. **Research (pipeline, verify-as-you-go).** One subagent per anchor paper; each **must
   `WebFetch` its canonical URL** and return `{verified_citation, mechanism, notebook_mapping,
   confidence}`. Filter out anything unverified before it reaches the plan. *No barrier* — a
   paper's mapping can proceed as soon as it verifies.

3. **Implement (worktree-isolated, staged).** Drive Steps 0→4 as a pipeline. Because subagents
   mutate notebooks, run file-mutating implementers with **`isolation: 'worktree'`** to avoid
   collisions; one stage's diff feeds the next. Each implementer returns a unified diff + the
   honest-protocol metric delta + baselines.

4. **Adversarial review (Codex) — mandatory gate.** After **each** implementation stage, hand
   the diff to **Codex** (see below). **Do not advance a stage until Codex's blocking findings
   are resolved.**

5. **Verify (diverse lenses).** For every claimed metric gain, spawn ≥3 skeptical verifiers with
   *distinct* lenses — **leakage** ("is this gain from shared videos?"), **shortcut** ("does it
   survive removing missingness covariates?"), and **statistics** ("is the delta inside the CI?").
   A gain survives only on a majority verdict; otherwise revert or reframe.

6. **Synthesize.** Merge surviving gains into `notes/07_results_summary.md` with the honest
   table (majority / missingness / handcrafted / embedding / fused), CIs, and per-class F1.

**Orchestration rules.** Give each subagent the PRIME DIRECTIVE + relevant GROUND-TRUTH cells;
require **structured** returns (schemas) so results compose; run independent subagents
concurrently in one message; keep completeness in view — a final "what did we not verify?"
critic pass before you report done. Scale the fan-out to the stage (few readers, one implementer
per stage, ≥3 verifiers per claimed gain).

═══════════════════════════════════════════════════════════════════════════════

## CODEX — adversarial review (use the `codex:codex-rescue` agent / `codex:rescue` skill)

Codex is your **independent second brain and red-team**, not a rubber stamp. Invoke it to
adversarially critique — with a **hostile-reviewer** framing — at these checkpoints:

- **After Step 0:** "Attack this evaluation protocol. Where can video/subject leakage still
  enter GroupKFold? Are the bootstrap CIs computed over the right unit (videos, not clips)? Is
  the missingness baseline computed identically to the model's split?"
- **After each of Steps 1–4:** hand Codex the **unified diff** and ask it to find (a)
  correctness bugs (shape/scaling/masking/off-by-one), (b) any way the change **inflates the
  leaky metric**, (c) collapse risk at N=12 / re-pretrain, (d) whether an ablation is missing.
  Ask Codex to return findings ranked by severity with a concrete failure scenario each.
- **Before the final results table:** "Refute the claim that S-JEPA adds signal over the
  handcrafted baseline. What confound explains the gap? What experiment would falsify our claim?"

**Rules of engagement:** treat Codex findings as **blocking** until resolved or explicitly
rebutted in writing; when Codex and your own reasoning disagree, **run the deciding experiment**
rather than argue; record each Codex round (prompt + verdict + resolution) in
`notes/08_codex_review_log.md`.

═══════════════════════════════════════════════════════════════════════════════

## OUTPUT / DEFINITION OF DONE

1. Modified notebooks (`02`,`03`,`04`,`06`) implementing at least Steps 0–2 (Steps 3–4 if
   compute allows), each cell runnable top-to-bottom in both `smoke` and `real`.
2. `notes/06_literature_findings.md`, `notes/07_results_summary.md`, `notes/08_codex_review_log.md`.
3. An **honest results table**: majority / missingness / handcrafted / embedding-alone /
   fused, under **GroupKFold-by-video with 95 % CIs**, plus per-class F1 and the full ablation
   grid.
4. A short **honest narrative**: what improved, what the data-bound ceiling is (~70–80 % macro-
   acc, dominated by clinical features + video diversity), and which prior 61.9 %/62.1 % numbers
   were inflated by leakage.

## CONSTRAINTS & GUARDRAILS

- Undergraduate laptop compute (CPU MediaPipe, small transformer). Prefer parameter-free /
  frozen-encoder changes before anything requiring long re-pretraining.
- Preserve the smoke/real split, checkpoint audit fingerprints, and reproducibility (seed 42).
- **Keep horizontal flip OFF** — it swaps left/right and destroys pathology laterality.
- **Do not fabricate citations or metrics.** Unverified id → mark unverified. Number not run →
  say "not run." Failing cell → report the error, don't paper over it.
- Ask the user before: deleting artifacts, committing, or any outward-facing/irreversible action.

═══════════════════════════════════════════════════════════════════════════════

### GOOD vs BAD behavior (few-shot calibration)

- **BAD:** "Swapped pooling; accuracy rose 62 % → 71 % on the exp5 split. 🎉"
  **GOOD:** "Under GroupKFold-by-video (5 folds), macro-F1 0.41 ± 0.07 (dynamics pooling) vs
  0.34 ± 0.08 (mean/std); handcrafted baseline 0.55 ± 0.06 on the same folds. Gain survives
  removing missingness covariates (Δ within CI). Stroke F1 0.33 → 0.48. Codex flagged a
  cadence-FFT resolution caveat under `temporal_resize`; noted, gated behind Step 3."
- **BAD:** "Per MAMP (arXiv:2308.07092), predicting velocity helps." *(id unverified.)*
  **GOOD:** "MAMP argues masked-velocity > masked-coordinate for skeleton SSL — **I could not
  resolve its arXiv id via WebFetch; marked unverified, excluded from the manuscript until
  confirmed.**"
- **BAD:** silently raising the mask ratio on the 1-video corpus.
  **GOOD:** "Deferred the mask-ratio increase to Step 4 — on 12 clips from one video it
  accelerates single-subject overfit and risks collapse without the variance floor; ran it only
  after broadening the corpus (Step 3)."

**Begin by reading `notes/04_improvement_plan.md` and the ground-truth cells, then emit your
`<thinking>` and your staged execution plan before making any edit.**
