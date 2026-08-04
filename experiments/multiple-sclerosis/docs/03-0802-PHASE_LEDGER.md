# Phase ledger — S-JEPA correctness + R1 program

_Lead-owned working ledger. Survives context compaction. Append-only decisions; update status inline._

- **Branch:** `feat/ms-sjepa-correctness-r1` (created from `main` @ `f21aa60`, carrying the user's uncommitted doc/diagram work).
- **Scope (user-approved 2026-08-02):** Core-correction + one bounded R1 fold/seed. Edit freely, may commit to branch, no push/PR without asking. Verify decision-critical literature only.
- **Write mode:** `shared-tree-single-writer`. The lead is the only writer of tracked files. Subagents are read-only / advisory and return evidence.
- **Compute:** macOS, 18 CPU, 68 GB RAM, Apple **MPS** (no CUDA). Budget: <= 8 wall hours, 0 dedicated accelerator-hours (MPS best-effort), <= 20 GB disk growth, <= 24 training trials, no external download > 1 GB.
- **Codex:** `codex-cli 0.146.0`, authenticated (api_key, model `gpt-5.6-sol`). `codex review --uncommitted|--base` and `codex exec` available. AR contract can be honored.

## Preserved user-owned changes (do NOT revert/overwrite)
- `docs/01-0802-PROGRESS.md` / `.pdf` (renamed from PROGRESS.md, edited) — the user's dated progress snapshot.
- Modified SVGs in `images/` + `scripts_make_diagrams.py` (prior-session diagram polish).
- `git rm`-staged `notes/ms/sjepa-ms-tutorials.md` (superseded by `sjepa-ms-01/02/03`).

## Confirmed defects (reproduced from source during preflight; agents re-verify with micro-tests)
| ID | Defect | Evidence (file:line) | Status |
|---|---|---|---|
| D1 | Predictor has no target-position identity; all hidden slots identical | `models.py:122-125` (no pos emb), `models.py:105` (shared mask_token) | CONFIRMED-inspection |
| D2 | Fixed 12-joint mask starves online encoder; those joints never context, yet pooled downstream | `masking.py:42-99`, `models.py:101-106`, `models.py:206-218` | CONFIRMED-inspection |
| D3 | Class-aware VICReg consumes diagnosis labels in "SSL"; description != behaviour | `train.py:103`, `losses.py:114-120` | CONFIRMED-inspection |
| D4 | `train_sjepa` resets optimizer/center/step each call; checkpoint omits them | `train.py:63-73`, `train.py:129-140` | CONFIRMED-inspection |
| D5 | Frame-rate/domain leakage: ALL 12 MS = 60fps & square 1080x1080; stride mislabels 24/25fps as 15fps | `data.py:73-74`; ffprobe sweep 2026-08-02 | CONFIRMED-empirical |
| D6 | Per-frame norm erases speed; arbitrary-gap interp; padding w/o mask; source_id != participant_id | `data.py:130-182,258-274`; manifest | CONFIRMED-inspection |
| D7 | Single `(N,)` mask reused across batch, not per-example `(B,M,N)` | `masking.py`, `train.py:58-61` | CONFIRMED-inspection |

## Data facts (Phase 0, generation g1 = source-grouped)
- 47 usable clips, 35 source groups. normal 19/16, MS 11/11, PD 17/8. 481 sliding windows (W32/S16).
- MS: 12/12 sources @ 60fps, 1080x1080 square. normal/PD: 24-30fps, varied res/aspect. => acquisition nearly separates MS.
- Mean visibility by class: normal 0.862, MS 0.920, PD 0.870.
- `source_id` is a YouTube id; NOT verified `participant_id`. All splits/results labelled **provisional source-grouped**.

## Decision log
- 2026-08-02: Legacy cache (`artifacts/keypoints/*.npz`) is immutable read-only input `cache_v0`. R1 uses it. Corrected data lineage (timestamp resample) is R2/deferred, out of this session's scope.
- 2026-08-02: R1 keeps 32f + 3L/96d encoder; only mechanical fixes + label-free all-source SSL + source-uniform + predictor positions + stochastic masks. No architecture scaling.
- 2026-08-02: Because the collection was already inspected (informed NEXT_STEPS), every result here is an **internal/development estimate**, never confirmatory.

## Registry generations
- `g1`: source-grouped, cache_v0, legacy 5-fold StratifiedGroupKFold(seed=42). RF + current S-JEPA + R1 all paired on g1.

## Wave A results (2026-08-02/03)
### Model-correctness audit (agent, ephemeral micro-tests) — all CONFIRMED
- D1: `predicted[:,target_idx,:]` std across token axis = **0.0 exactly**; permuting target-position ids changes predictions by 0.0. Root: `models.py:105` shared mask_token + `models.py:122-125` predictor has no pos emb.
- D2: `view_encoder.tokenizer.spatial_emb.grad` norm = **0.0** for exactly joints {11,12,23,24,25,26,27,28,29,30,31,32}.
- D3: class-aware VICReg variance term = 0.986 (near-max) for maximally-separated tight clusters; float64-invariant to class-center location. Rewards within-class spread. `losses.py:114-120`, label leak `train.py:103-104`.
- D4: fresh CE-center(zeros)/AdamW/step每 call; checkpoint dict = {config,extra,mask_token,predictor,target_encoder,view_encoder} only.
- D7: mask is global `(N,)` reused across batch & steps; no `(B,M,N)`.
- BONUS: `EMA.update` HARD-COPIES buffers (`models.py:141-142 tb.copy_(sb)`), not EMA. Inert now (LayerNorm, 0 buffers) but latent bug if BatchNorm added. Keep param-EMA; leave buffer copy but note it.

### Literature (decision-critical, verified from primary PDFs; project page unreliable)
- **S-JEPA ECCV'24**: target-OUTPUT masking mandatory (+7.2pt, Table 6; repo already does this ✓). Factorized spatial+temporal PEs (Sec 4.2). Loss = centered(target-only, β=0.9)+sharpened latent-CE, τ_p=0.1 > τ_t=0.06 (beats MSE). Input is **3D** (SO(3) views) — our 2D+visibility is a documented VARIANT. Scale knobs (0.9 mask/1200ep/0.9999 EMA/batch256) are for ~57K NTU, NOT 35 groups — retune down.
- **I-JEPA CVPR'23**: AUTHORITATIVE spec for the D1 fix — predictor = context tokens + "shared learnable mask token + **positional embedding of the target slot**", explicitly ablated. Port this pattern directly.
- **MAMP ICCV'23**: predicting temporal motion (frame differences) beats coordinates by +10.1/+6.1 (Table 7) — 2D-safe R2 win. CAUTION: motion-aware (high-motion) masking assumes low-motion is redundant — **clinically wrong** for MS/PD hypokinesia. Use uniform / low-motion-inclusive masks. Validates NOT copying high-motion masking.
- Implication for R1 masks: keep mixed/uniform stochastic graph-time masks; clinical target bias ~1.5x only; do NOT bias toward high-motion.

### Data/evaluation audit (agent, quantified on real cache) — CONFIRMED
- fps mismatch exact: 24->12.0fps(-20%), 25->12.5fps(-16.7%), 29.97->14.985, 30->15.0, 60->15.0. All cached label "fps=15".
- fps x class: MS = 11/11 @ 60fps 1080x1080 square. normal/PD 24-30fps varied. Rule "MS iff fps>=50" => 100% acc, 0 false pos. "MS iff square" => 11/11 MS + 2 PD false pos.
- **Shortcut controls (same g1 folds) EXCEED headline S-JEPA 0.570**: visibility-only RF **0.666**, mean+std-xy logreg **0.694**, mean-xy 0.602, duration+fps/res 0.614, body-proportion 0.470. NEXT_STEPS (b)~0.623 verified (estimator-dependent). Visibility means: MS 0.920 / normal 0.862 / PD 0.870 (exact match).
- Normalization: pelvis-at-origin every frame (drift 692px raw -> 4e-7 norm => speed destroyed). Max |norm xy| = **38.2** (near-zero torso blowup; ratio not clamped, only numerator at 1e-3). clean_sequence interpolates ARBITRARY gaps (80-frame gap linearly filled); no validity mask kept. sliding_windows pads by repeating last frame; no padding mask.
- **Identity worse than assumed**: source_id contains up to **5 distinct labeled patients** (Patient_N in original names: pFLC9C-xH8E has 5, _Wn9oYGpRdM 3, bmi1hYOnTHs 3, M-_cogKwXK4 2, EHymg4AGMJs 2, nXuJIs25z1U 2). MS test-retest evidence => one person likely spans multiple source_ids landing in DIFFERENT folds => train/test can share a person. Split is source-disjoint but NOT participant-disjoint, and groups are not participants. manifest "high" confidence = YouTube-ID recovery, not identity.
- RF/S-JEPA parity CLEAN: same fold loop, same test_recs, same label order ['normal','ms','pd'], per-CLIP aggregation for both, deterministic folds. No test-label leakage into selection. silhouette on vstack(train+test) is descriptive-only (over-optimistic separability, not leaked into F1).

### Clinical-gait literature (verified; IEEE/Elsevier via PubMed/S2 fallback, labeled)
- FG-2024 (arXiv 2405.17817): PD severity 3-class, LOSO **F1 0.62** vs random **0.75** (+0.13 inflation). => participant-disjoint mandatory.
- Kaur JBHI 2023 (PMID 36126031): closest 3-class MS/PD/control; new-subject acc 78.1%/AUC 0.87; uses accuracy+AUC not F1; lab multi-view 3D (not our 2D in-the-wild).
- GaitForeMer MICCAI'22 (2207.00106): NTU pretrain -> PD severity LOSO macro-F1 0.60->0.76 (+0.16). SEVERITY not diagnosis. Top scaling bet (external pretraining).
- CARE-PD NeurIPS'25: 362 subj/8477 walks/9 cohorts, SMPL topology, CC-BY-NC(-ND) research-only, **NO S-JEPA checkpoint**. LODO macro-F1 ~0.50 (cohort-disjoint degradation). External work, not a free win; topology remap needed.
- MS markerless (TNSRE'25 25+25; MSARD'26 n=20): distal shank/foot elevation-angle ROM, tandem heel-to-toe, reduced stride length/time scale with EDSS. Clinically meaningful feature story.

### KEY DECISION-CHANGING TAKEAWAYS
1. Current S-JEPA (0.570) is BELOW nuisance controls (0.666-0.694). Any "gain" must be checked against these controls, reported alongside every headline.
2. Identity: cannot claim participant-disjoint. Report everything as **provisional source-grouped**; add a data-manifest field capturing Patient_N multiplicity + MS test-retest risk. A participant registry is future work (out of this session's core-correction+R1 scope) — flag loudly.
3. Masks: stochastic + uniform/clinical-target-bias, NEVER high-motion (MAMP caution + MS hypokinesia).
4. Predictor fix spec = I-JEPA (mask token + target-slot PE).

## Phase 0 EXECUTED (2026-08-03) — artifacts/eval/g1/
- cache_v0 combined sha256 = d12e042980d5e135... (47 npz). Registry g1 locked to fold_registry.json.
- **E0-RF**: fold-mean macro-F1 **0.708 +/- 0.131**, pooled **0.725**. Pooled confusion: normal[13,0,6] ms[1,8,2] pd[5,0,12]; recalls normal .68 / ms .73 / pd .71.
- **E0 reproducibility note**: NOT an exact match to historical RF 0.668 +/- 0.096. Cause: `classical.train_rf_and_predict` uses max_depth=5 + drops zero-variance cols; the notebook's exact RF config/fold composition differ. This is a paired baseline on g1 (the honest comparator for R1), NOT a claim to reproduce the historical mean-of-fold number. Documented, not hidden.
- **Shortcut controls on g1** (best of logreg/rf, fold-mean macro-F1): mean_std_pose **0.694**, mean_pose 0.599, visibility_only 0.602, duration_acq 0.430, body_proportion 0.397. => nuisance controls are near/above the OLD S-JEPA 0.570. R1 must be checked against these.
- Files: fold_registry.json, E0_results.json, E0_RF_oof.json, run_manifest.json, COMPLETED.json.
- Script: `scripts_phase0_provenance.py` (deterministic, refuses to overwrite a completed run).

## Phase 1 EXECUTED (2026-08-03) — model correctness
- New `sjepa/masking_v2.py`: per-example stochastic graph-time masks; connected JOINT_GROUPS (head/mouth/arms/trunk/legs, every joint targetable); region weights down-weight multi-region joints so no joint monopolized; clinical target bias 1.5x (NOT high-motion). `sample_target_mask`, `sample_mask_batch`, `mask_bank_stats`.
- `models.py`: TransformerBlock/Transformer now accept `key_padding_mask`. `SJEPAEncoder.forward_context_per_example` (per-example (B,N) masks via key-padding attention). New `PredictorV2` = in_proj + factorized joint/time position embeddings (I-JEPA fix) + transformer + out_proj. `SJEPA(repaired=True)` swaps in PredictorV2 + `forward_repaired`. `build_model(..., repaired=True)`. Legacy `Predictor`/`forward`/`forward_context` kept for E0.
- Coverage (mask bank, both smoke T=4 & full T=8, ratio 0.5/0.6): min_visible >=0.535, min_target >=0.648, mean_target ~0.54-0.64. Gates (vis>=0.20, tgt>=0.10) PASS with margin.
- Tests `sjepa/tests/test_correctness.py` (8): D1 legacy defect reproduced (std<1e-5) + repaired position identity (std>1e-3) + permutation sensitivity; D2 every-joint context gradient; mask coverage gates; D7 per-example diversity + determinism + clinical bias. ALL PASS.
- Full suite `pytest sjepa/tests/` = 11 passed (smoke + correctness), on BOTH repo-root .venv (3.14) and experiment .venv (3.12). CANONICAL env = experiment-local `.venv` (Python 3.12, pinned for UMAP/numba). Use `experiments/multiple-sclerosis/.venv/bin/python` for all runs.

## Phase 2 EXECUTED (2026-08-03) — objectives, sampling, state
- New `sjepa/train_v2.py`: `train_sjepa_v2` (label-free; NO class-aware VICReg), source-uniform resumable data stream (`_source_uniform_weights` + `_draw_batch_indices` off a saved `data_rng`), schedules in optimizer UPDATES (`total_updates`), decoupled `schedule_updates` horizon so save/resume is exact, EMA half-life reported, collapse diagnostics (emb_std, effective_rank, teacher_drift). `save_checkpoint_v2`/`load_checkpoint_v2` persist ce-center/opt/step/torch_rng/mask_rng/data_rng.
- Bug found+fixed during testing: schedule (LR/EMA/warmup) was keyed to `total_updates`, so a 4-step and 8-step run diverged at step 3. Fix: `schedule_updates` decouples horizon from stopping point. Now save/resume reproduces uninterrupted run to <1e-3 every step.
- Tests `sjepa/tests/test_train_v2.py` (4): source-uniform equalizes exposure (long 75/87-window source no longer dominates); strict SSL loss invariant to label permutation (D3 no-leak); save/resume matches uninterrupted <1e-3 all 8 steps (D4); diagnostics finite/non-trivial. ALL PASS.
- Full suite = **15 passed** (3 smoke + 8 correctness + 4 train_v2).

## Phase 5 + 4 EXECUTED (2026-08-03) — firewall + bounded R1
- `scripts_r1_repaired.py`: R1 runner + eval firewall. Loads locked g1 registry; per fold trains repaired label-free source-uniform SSL on ALL train sources, freezes, embeds train+test (per-clip mean over windows of a FIXED target-token pool), fits balanced logistic probe on TRAIN ONLY, predicts test -> OOF probabilities (one row per clip). Paired RF on identical folds. Flags: --fold-limit --total-updates --mask-ratio --seed --device --output-dir. Refuses to overwrite COMPLETED.json. Saves oof.json/results.json/run_manifest.json/COMPLETED.json + per-fold checkpoints.
- Bug found+fixed: `_effective_rank` used `torch.linalg.svdvals` which is NOT implemented on MPS -> silently returned 0.0 (masked collapse). Fixed to compute SVD on CPU. Now eff_rank real (7.7-9.5 => NO collapse).
- **R1_repaired32 result** (g1, 1000 updates, seed 42, MPS, 12.5 min, artifacts/runs/r1_g1_1k_s42/):
  - **S-JEPA pooled macro-F1 0.438, acc 0.447, PD-recall 0.235**. Confusion normal[11,3,5] ms[0,6,5] pd[5,8,4] (PD->MS is the main error, as before).
  - **RF paired pooled macro-F1 0.667, acc 0.660, PD-recall 0.588** (matches historical ~0.668 -> sound comparator).
  - eff_rank per fold [8.8,9.5,8.2,8.8,7.7] => representation did NOT collapse.
- **INTERPRETATION (honest)**: repaired S-JEPA (0.438) is BELOW old broken S-JEPA (0.570), BELOW RF (0.667), and BELOW nuisance controls (0.694). This is the EXPECTED, plan-anticipated outcome: removing shortcuts + label leak + fixing masks lowered a partly-shortcut-driven score. Per NEXT_STEPS promotion gate, a mechanically-valid R1 that fails the inner threshold => STOP local architecture scaling; the next bottleneck is the DATA PIPELINE (timestamp resample, speed-preserving norm, domain de-confound) + external clinical-motion pretraining, NOT a bigger model. Reported as a NEGATIVE result, not hidden.
- Note: single seed, 1000 updates (bounded per session scope). NOT a learning-curve sweep or multi-seed. Full R1 (300/1k/3k x 3-5 seeds, inner-fold selection) is runnable via the same script and handed off as PENDING.

## Codex AR-1/AR-2 (2026-08-03) — archived artifacts/reviews/AR1_codex_review_2026-08-03.md
Codex `gpt-5.6-sol`, read-only sandbox, model_reasoning_effort medium. (Note: Codex could NOT run pytest — no writable tmp in its sandbox — so the "2 failed" lines in an earlier run were sandbox artifacts, not real failures; our 15/15 pass reproducibly.) Verified by Codex: D1 token layout t*V+v + target alignment; production masks always keep context; 1-visible-token attention finite, no all-masked production row; folds source-disjoint; all 47 clips once in OOF; probe/scaler train-only; embed mask fixed pre-test; controls train-only scaling. Findings + dispositions:
- **P0-1 RF not reproducible across runners (0.725 vs 0.667)** — ACCEPTED. Root cause: Phase 0 ran under Python 3.14/sklearn-X, R1 under 3.12. FIXED: re-ran Phase 0 under pinned 3.12 venv -> E0-RF pooled now **0.667**, exactly matching R1 paired RF. Manifests now record python/numpy/sklearn versions + git_dirty.
- **P0-2 result doesn't identify producing code (dirty tree, manifest HEAD=f21aa60)** — ACCEPTED. FIX: commit the code to the feature branch (user-authorized) so future manifests reference a real SHA; manifests now stamp git_dirty. The 1000-update R1 result was produced pre-commit on this exact working tree (documented).
- **P1-3 MPS/CUDA RNG not restored on resume** — ACCEPTED+FIXED. `random_view` draws on x.device; added `_device_rng_state`/`_restore_device_rng` (torch.mps/torch.cuda get/set_rng_state) to export/resume. (CPU test already passed; MPS resume now covered.)
- **P1-4 schedule_updates not in checkpoint (resume test supplied it externally)** — ACCEPTED+FIXED. Checkpoint now carries `schedule` {schedule_updates, seed, mask_ratio, masks_per_sequence, clinical_bias}; resume defaults from it (no external args needed).
- **P2 strict path calls label-producing `dataset[i]`** — ACCEPTED (cosmetic; Codex confirmed NO mathematical label leak). Left as-is with note; window is taken as `[0]`, label discarded. Regression `test_strict_ssl_ignores_labels` proves invariance. (Future: label-free dataset view.)
- **P2 D1 permutation test used all-visible tokens + possibly-identity randperm** — ACCEPTED+FIXED. Hardened to inspect HIDDEN slots only, deterministic roll perm, spatial AND temporal tags independently. Codex independently confirmed hidden-slot std 0.053-0.059 (real position identity).
- **P2 NaN diagnostic didn't block COMPLETED** — ACCEPTED+FIXED. R1 now writes FAILED.json + raises if any fold's eff_rank/emb_std/loss is non-finite, before COMPLETED.
- All 15 tests still pass after fixes. Zero unresolved P0/P1. AR-1 dispositions closed; a re-run of AR post-commit is the clean-room AR-5 (handed off).

## Codex AR-3 (2026-08-04) — uncommitted notebooks + docs — archived artifacts/reviews/AR_uncommitted_codex_2026-08-04.md
Codex `gpt-5.6-sol`, `codex review --uncommitted` on the working tree (notebook parity + naming edits + new docs/07 roadmap). Two REAL P1 defects found in `scripts/notebook_content.py` (the notebook SOURCE), both reproduced and FIXED this session. Note these were tutorial-notebook-only; the frozen R1 result from `scripts_r1_repaired.py` is NOT affected (that runner already trains per-fold on `train_recs` only and uses `get_config(smoke=False)`).
- **P1-A smoke flag tested for truthiness, not parsed** — `UPDATES = 60 if os.environ.get('SJEPA_SMOKE') else 800` (also 40/400 and 60/500). With `.env` `SJEPA_SMOKE=0`, `get()` returns the string `"0"`, which is truthy, so a normal run silently used the 60-update SMOKE budget. **FIXED:** key the budget off `cfg.profile.endswith('smoke')` (config's `_env_flag` parses `0/1/true` correctly). Verified: SMOKE=1 -> profile `laptop+smoke` -> True; SMOKE=0 -> `laptop` -> False. Sites: notebook_content.py:395-398, 510-511, 845-846.
- **P1-B nb_03 SSL checkpoint trained on ALL sources, then nb_04 reports held-out probe on it** — even though SSL ignores labels, the encoder saw fold-0 test clips' motion, so nb_04's "held-out" probe score was covariate-leaked. **FIXED:** nb_03 now loads the locked `g1` fold registry and trains `sjepa_ssl.pt` on **fold 0 training sources only**, with an assertion that no test source leaks in; checkpoint stage relabeled `ssl_fold0_train`. Verified by smoke run: nb_03 "fold 0 training half: 37 videos" (not 47); nb_04 loads 37 train / 10 test, sources disjoint, probe scores the 10 unseen videos.
- Post-fix validation: `py_compile` OK; `scripts_build_notebooks.py --check` all 7 MATCH (idempotent); smoke-run nb_03->nb_04->nb_06 all cells OK; frozen nb_06 scoreboard unchanged (S-JEPA 0.438 / RF 0.667 / eff_rank [8.8,9.5,8.2,8.8,7.7]). Codex could not itself run the notebooks (sandbox); fixes verified by the lead on the pinned 3.12 venv.
- Roadmap doc (docs/07) load-bearing claims independently re-verified by the lead against code/artifacts: confusion/per-class numbers, eff_rank, confident-wrong 0.76-vs-0.81, control scores, "3/481 padded windows none MS", npz stores constant fps=15 (true source fps discarded), raw keypoints in pixels. Two imprecise claims in an earlier draft (frame-size "not saved"; fps framing) were corrected.
- Clean-room AR-5 on the committed tree remains handed off (requires a commit; not authorized this session).
