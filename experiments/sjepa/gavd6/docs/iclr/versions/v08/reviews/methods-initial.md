# v08 initial independent scientific contribution and methods audit

Scope: the frozen v07 manuscript, completed configuration/ledger/calibration exports, matching source implementation, prior audit, and primary literature. This reviewer did not edit the manuscript or run new training. All source pointers below are relative to `src/gavd6_sjepa/research_directions/`. The v08 manuscript is not yet frozen; this record gives evidence and concrete revision requirements rather than a final score.

## Central claim and contribution

**Proposed claim:** In the completed paired synthetic restoration protocol, reference-feature prediction does not establish more faithful recovery of movement change than the tested controls. Response, trajectory, and anatomical naming evaluations rank fitted procedures differently; downstream supervision and explicit failure costs materially alter the interpretation of apparent gains.

The informative contribution is an empirical evaluation of the connection between feature prediction and movement restoration, with three parts: (1) a controlled pairing of physical movement edits, observation corruption, and projected reference targets; (2) explicit comparisons that separate learned features, frozen coordinate readouts, and joint coordinate adaptation; (3) a follow-up and reliability analysis showing which apparent gains survive zero-response, waveform, naming, and failure-accounting checks. Scalar cancellation is a familiar mathematical reason to examine several outcomes, not a novel result. The evidence supports neither general ineffectiveness of predictive representations nor a mechanism explaining the observed ordering.

The existing `CONTRIBUTION.md` is consistent with this position. Prefer “does not establish” to “fails to recover”: a positive direction score and clear-observation results contain useful information despite the pooled zero-response ranking. Keep the naming diagnostic explicitly post hoc and the feature-difference benefit unresolved.

## What each fitted comparison actually tests

| Family | Pretraining / privileged targets | Frozen during final fitting | Final supervision and updates | Question addressed / limitation |
|---|---|---|---|---|
| Initialized encoder | None; seeded random encoder | Encoder | Fresh coordinate readout, 2,000 updates; coordinate-only or original change package | Whether trained features improve this frozen-readout system over initialized features; residual input path still exists. |
| Coordinate-pretrained | Masked coordinate reconstruction against projected references; encoder and its first readout learn for 2,000 updates | Pretrained encoder; pretraining head is replaced | Fresh readout, 2,000 updates; either final objective | Coordinate reconstruction pretraining versus latent targets under the local procedure. |
| Core reference JEPA | Masked observed-input tokens predict EMA-encoder features of projected references, with CE and VICReg; 2,000 updates | Online encoder; teacher/predictor discarded | Fresh readout, 2,000 updates; either final objective | Reference-feature pretraining under privileged synthetic supervision, not unsupervised learning from natural observations. |
| Shuffled-reference JEPA | Same core loss, but a different accepted window from the same person supplies the teacher reference, retaining movement role/magnitude and nuisance condition | Online encoder | Same fresh-readout alternatives | Observation–reference alignment control. It does **not** randomize the paired movement states in the new delta auxiliary. |
| Delta / endpoint JEPA | Same base feature loss plus one calibrated auxiliary; 2,000 updates | Respective online encoder | Same fresh-readout alternatives, 2,000 updates | Finite-procedure feature-difference versus endpoint-residual comparison, with separate evolving teachers and unequal initial auxiliary strength. |
| Coordinate-delta | Masked coordinate loss plus common-scale paired coordinate-residual auxiliary; 2,000 updates | Encoder | Same fresh-readout alternatives, 2,000 updates | Whether paired coordinate supervision provides an alternative to feature differences. |
| Direct coordinate | None | Nothing in encoder/readout | Encoder and readout jointly learn for 4,000 updates; either final objective | Practical adapted restoration comparator. Equal update totals do not match coordinate-target exposure, optimized parameters, or adaptation. |
| Repair: low scalar / dense | No new pretraining; reuse the delta and endpoint encoders | Retained encoder | New readout, 2,000 updates; coordinate+geometry+low-scalar or dense change | How downstream supervision changes trajectories, holding the encoder and geometry term fixed. |

Source: `gait_fidelity/training.py:43–72,192–204,488–510,578–651`; `gait_fidelity/repair_training.py:261–305`. The completed core has five families, not the larger planned matrix. SmoothNet, PoseBERT and MotionBERT were not external baselines run here. Avoid a table that calls every method “pretrained,” or groups direct training into a frozen representation ranking.

## Verified calibration and optimization limitation

The saved receipt `outputs/iclr/jepa-response/diagnostics/loss-calibration.json` reports 32 fixed training batches, seed 17, and all 702,624 trainable JEPA parameter coordinates with unused gradients counted as zero. For component j,

\[
G_j=\sum_{b=1}^{32}\|\nabla_{\psi}L_j^{(b)}\|_2^2,
\quad R_j=\sqrt{G_j/(32P)},
\quad \lambda_J=0.1\sqrt{G_{\mathrm{base}}/\max(G_\Delta,G_E)}.
\]

| Component | Saved gradient energy G | Saved RMS R | Weighted auxiliary as % of base |
|---|---:|---:|---:|
| Base feature loss |90,192.74877929688|0.06333581045789884|Reference magnitude|
| Endpoint auxiliary |5,639,096.859375|0.5008045868129164|10.000000000000002%|
| Delta auxiliary |0.09584604314295575|0.00006529059327844553|0.001303713963443308%|

The common coefficient is **0.012646811176583523**. The percentages are `100 λJ Rauxiliary/Rbase`. Calibration performs no optimization step. The auxiliary derivatives are measured before global clipping. They concern aggregate initial derivative magnitude, not gradient alignment, persistent influence, or component-specific AdamW update magnitude. The common coefficient deliberately makes the larger auxiliary 10% of base; it does not make both auxiliaries 10%.

I independently recomputed the ratios and verified that all eight source hashes in the calibration receipt match the reviewed checkout: local `training.py`, `response_calibration.py`, `response_objectives.py`, `measurements.py`, `masking.py`, and inherited `models.py`, `training.py`, `objectives.py`. Source: `gait_fidelity/response_calibration.py:50–58,124–136,184–231`.

The response ledger independently records **11,999 / 12,000 clipped pretraining updates** across the six newly fitted delta/endpoint JEPA encoders: all 2,000 in every fit except delta seed 43, which has 1,999. The three coordinate-delta pretraining fits record 0 / 6,000. The exact ledger path is `completed/<pretrain-response-variant-graph_time-seed-N>/result/gradient_clipping`. The implementation clips the combined gradient to norm one before AdamW: `gait_fidelity/training.py:659–675,744–746`. This count does not apply to all readouts or every earlier JEPA fit. It establishes frequent rescaling, not an optimization failure or a causal explanation. The compact diagnostics contain representation/probe exports, not the complete gradient time series; do not infer persistent auxiliary ratios from the calibration receipt.

**Required placement:** directly beside the delta–endpoint comparison and its uncertain 0.37° effect. A reader should know before interpreting that contrast that the compared procedures share a coefficient without matching initial auxiliary influence.

## Notation and exact equations to preserve in the appendix

Let a and b be full, time-aligned movement states from the same source window under the same orientation, camera, estimator, naming and observation condition. Y is projected reference xy; X contains estimated xy, confidence, availability and timestamps. Naming corruption permutes bilateral observation channels, including confidence/availability, while retaining time and references. Physical mirroring reflects the 3D body about a pelvis-defined plane, exchanges body-model side indices, and then reprojects under the fixed fitted camera. Merely exchanging already computed excursions `(qL,qR)` changes `A=qR−qL` into `−A`. Physical mirroring and reprojection need not exchange those projected excursions exactly.

For each endpoint, context excludes unavailable and artificially hidden coordinates. Set o to per-axis context medians, s to the Euclidean norm of the per-axis 95th–5th percentile span, and normalize `(X−o)/s`; fallback is `(o,s)=(0,1)` with fewer than two context points, nonfinite span, or span below 10⁻⁶. The same input-derived transform is applied to teacher references; neither references nor hidden coordinates determine it. All 384 four-frame joint slots remain in the encoder. Masking zeros xy/confidence/availability, retaining timestamp and positional channels; hidden slots are not deleted. Sources: `gait_fidelity/training.py:288–321`; `synthetic_training_v2/models.py:94–125`.

Use predictor logits u, detached teacher logits t, running center c, and `H(v)=v−mean_channels(v)`:

\[
q=\operatorname{softmax}((t-c)/.06),\quad
p=\operatorname{softmax}(u/.1),\quad
L_{\mathrm{base}}=\operatorname{mean}_{i}\operatorname{mean}_{k\in Q_i}[-q_{ik}^{\top}\log p_{ik}]+.05L_{\mathrm{VICReg}},
\]
\[
e_{ik}=H\{u_{ik}/.1-\operatorname{sg}[(t_{ik}-c)/.06]\},\qquad
L_\Delta=\mathbb E_{(a,b)}\operatorname{mean}_{k\in Q_{ab}}\frac{\|e_{bk}-e_{ak}\|^2}{2D},
\]
\[
L_E=\mathbb E_{(a,b)}\operatorname{mean}_{k\in Q_{ab}}\frac{\|e_{ak}\|^2+\|e_{bk}\|^2}{2D},\quad D=96.
\]

Base queries are artificial queries or patches with any missing input frame, intersected with any valid teacher reference frame. Auxiliary support Qab requires queries at both endpoints and all four reference frames valid in both. Reduce supported tokens within each pair, then pairs equally. Unsupported auxiliaries do not erase supported base terms. On the same tensors, `LΔ,k−LE,k=−ea,kᵀeb,k/D`; fitted teachers differ over training. VICReg uses 25 invariance + 25 variance + covariance on projected mean encoder features from independent translations in normalized xy ±.02. Sources: `response_objectives.py:18–57`, `response_calibration.py:80–110`, `temporal_gait/objectives.py:29–76`.

For coordinate-delta let `sab=(sa+sb)/2` and `ri=(si/sab)(Xhatnorm_i−Ynorm_i)`. Its per-frame term is `||rb−ra||²/4`, averaged over four-frame patches, supported tokens and pairs. Its coefficient is separately calibrated as `.1 sqrt(Gcoordinate/Gcoordinate-delta)`, not λJ (`response_objectives.py:60–74`; `response_calibration.py:219–220`).

Define `θ=atan2(|ux vy−uy vx|,uᵀv)180/π` from knee-to-hip and knee-to-ankle vectors; `qℓ=P95(θℓ)−P5(θℓ)` and `A=qR−qL`. Write Δ for b−a, never a time derivative. Then

\[
L_s=\mathbb E_{(a,b)}[(\Delta\widehat A-\Delta A)^2/180^2],\qquad
L_d=\mathbb E_{(a,b)}\operatorname{mean}_{t,\ell\in S_{ab}}
[(\Delta\widehat\theta_{t\ell}-\Delta\theta_{t\ell})^2/180^2].
\]

Original change is `Lx+Ls+Lg`; coordinate-only is Lx. Repair low-scalar is `Lx+.1Ls+Lg`; dense is `Lx+λdLd+Lg`. Lg is squared relative segment-length shortfall below 2 px, averaged on reference-fixed common support. For each retained encoder/seed, `λd=.1 sqrt(Σ||∇φLs||²/Σ||∇φLd||²)` over 32 fresh-readout training batches. φ is the readout parameter vector; this matches dense initial gradient magnitude to **low-scalar**, not coordinate loss. Common angular training support requires both legs/endpoints reference-valid, all segments at least 2 px, at least 16 frames and 80% of the 128-frame window. Sources: `measurements.py:96–183`; `repair_objectives.py:20–81`; `repair_training.py:160,168–182`.

Appendix also needs: exact coordinate-loss endpoint reductions versus patch reductions, input/residual readout path, architecture 4 encoder/2 predictor blocks at width 96 with 4 heads, AdamW 3e−4/.01/default betas/eps, 5% warmup/cosine, norm-one clipping, teacher EMA .99→.999 and center momentum .9, equal hierarchical person→motion→window→pair sampling, fresh readout seed+100003, and inverse normalization before angular measurement. The existing `docs/iclr/evidence/reproducibility-methods.md` provides the full verified numerical contract. A complete rerun additionally requires raw assets, renders, extraction, predictions and checkpoints absent from the compact packet; local reaggregation is not end-to-end replication.

## Interpretation and provenance requirements

| Objection | Severity | Evidence and required correction | Residual limitation |
|---|---|---|---|
| Apparent feature benefit interpreted as isolated coupling effect | Major | Pair the uncertain estimate with readout reversal, initial gradient asymmetry and near-universal new-pretraining clipping. | No coefficient sweep, matched-influence refit, convergence study or movement re-pairing control. |
| Zero-response treated as an incidental baseline | Major | Put 5.811° beating all 16 pooled neural variants in the central result; distinguish noisy-input improvement from response recovery. | Small true responses and failure costs can reward zero; observation/magnitude strata need complete exploratory reporting. |
| “Scalar loss damages waveforms” causal language | Major | Original addition is Ls+Lg. Attribute deterioration to the tested package; show original/low/dense absolute means and primary interval. | Repair changes information and gradient locations; no unique sparse-gradient mechanism. |
| Direct-versus-frozen scores read as representation quality ranking | Major | Show adaptation/supervision differences in the table and grouping. | No equal-coordinate-exposure or full-finetuning factorial study. |
| “74% of gain” read as failure probability or mechanism | Major | State unconditional failure-cost contribution separately from weighted failure rate and conditional successful error; include prospective interpretation of penalty sensitivity as exploratory. | Penalty decomposition is arithmetic, not causal. |
| Declared equated with preregistered | Major | Saved configs/ledgers establish local declarations and execution identity, not external preregistration. No registry receipt was found. | All three studies reuse 14 development people; protected confirmation remains unrun. |
| Broad novelty or external benchmark implication | Moderate | Attribute S-JEPA directly, distinguish adaptations, move closest work early, state external methods were not run. | Contribution is bounded empirical evaluation, not a new generic architecture or world model. |
| Geometric assignment called clinical laterality or chance test | Moderate | Mark post hoc, define wrong+ambiguous+missing and its separate denominator. No 50% chance reference. | 2D geometry and projected body-model joints are not independent clinical truth. |

The recorded creation times put local configs before their respective ledgers: core 2026-09-23 05:07 UTC → 06:12, response 2026-09-24 18:16 → 18:30, repair 2026-09-25 06:07 → 07:20. Those timestamps and saved hashes alone do not prove independent prospective registration or that analyses were never informed by previous development inspection. In repair, read the nested repair statistical contract: the inherited top-level evaluation block is not its operative primary contrast.

## Closest primary literature, verified 26 September 2026

- **S-JEPA** is the closest feature-pretraining precedent: partial skeleton sequences predict missing-joint latent representations with centered teacher targets, evaluated for skeletal action recognition. The local model changes the task to estimated 2D input/reference-target restoration and uses an online-encoder frozen coordinate readout; do not imply these downstream conclusions reproduce its original recognition setting. [Official ECCV 2024 paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04755.pdf).
- **PoseBERT** trains a temporal transformer from 3D motion-capture data through masked modeling, with 3D keypoint or body/hand parameter variants for refinement, completion and prediction. It is relevant temporal restoration precedent, not equivalent to a feature-only objective. [Authors’ primary paper](https://arxiv.org/abs/2208.10211).
- **MotionBERT** learns motion representations by recovering 3D motion from noisy, partial 2D observations and transfers by finetuning an encoder with simple heads. Its reconstruction and adaptation regime differs from the present frozen 2D readouts. [Official ICCV 2023 paper page](https://openaccess.thecvf.com/content/ICCV2023/html/Zhu_MotionBERT_A_Unified_Perspective_on_Learning_Human_Motion_Representations_ICCV_2023_paper.html).
- **SmoothNet** is a temporal-only pose refinement network; its temporal fully connected processing addresses jitter and difficult frames across estimators and 2D/3D modalities. It motivates a missing learned refinement comparator; the source's available “SmoothNet-style” arm does not mean its published implementation was evaluated in the completed matrix. [Official ECCV 2022 paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136650615.pdf).

The defensible positioning is that these papers establish masked skeletal representation and temporal refinement capabilities, while this study examines whether its particular representation/readout procedures preserve paired projected movement measurements and anatomical naming. Do not claim prior work never evaluates fidelity, biomechanics, or laterality as a broad literature-wide negative without a dedicated systematic review.

Final v08 review will score the frozen source/PDF with the unchanged 20/20/15/15/10/10/5/5 rubric. v07's independent score was 79.25/100; a stronger narrative or figure set can improve clarity/positioning, but the same 14 people, three seeds, unequal objectives and missing confirmation cannot earn a research-evidence bonus.
