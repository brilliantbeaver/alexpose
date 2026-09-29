# Reproduction contract for the ICLR study

This supporting record states the implemented numerical procedure; it is separate from the nine-page manuscript. Source pointers refer to the checkout reviewed on 25 September 2026. Frozen experiment configurations, receipts, and hashes govern what ran; defaults in a current source file alone do not prove an executed setting. This document adds no experiment or outcome.

## Data and model identity

The completed source configuration is in `outputs/iclr/walking-core/config.json` (also copied to `outputs/gait-fidelity/config.json`). It uses 128 frames at 25 Hz, 12 joints, four-frame patches, width 96, four encoder blocks, two predictor blocks, four attention heads, batch size 16, and seeds 17, 29, 43. The 128 samples span 5.08 seconds from first to last timestamp. Joint order is left/right shoulder, elbow, wrist, hip, knee, ankle.

Training-only pairs comprise one baseline and one intervention/no-change endpoint under the same source window, orientation, camera, observation, naming and extractor condition. Baseline reuse is shared by all objectives. The executed sampler chooses person, motion, window and pair uniformly in succession. The 15° edit and ViTPose family are excluded from training. Source: `gait_fidelity/training.py:43–72,172–189,557–563`; `gait_fidelity/preparation.py:466–528`, relative to `src/gavd6_sjepa/research_directions/`.

## Input-only normalization and tokens

For endpoint/window `i`, let `O_i(t,j)` be supplied observation availability and `M_i(t,j)` the artificial query mask repeated over each four-frame patch. Context is `C_i=O_i AND NOT M_i`. Let `X_C` contain its available `(x,y)` coordinates. The transform is

\[
o_i=\operatorname{median}(X_{C_i}),\qquad
s_i=\left\|Q_{.95}(X_{C_i})-Q_{.05}(X_{C_i})\right\|_2,
\qquad \widetilde X_i=(X_i-o_i)/s_i.
\]

Median and quantiles act independently on the two coordinate axes. If fewer than two context points remain, or the span is nonfinite or less than `1e−6`, the transform is fixed to `o_i=(0,0), s_i=1` and the event is counted. Hidden input coordinates and references cannot affect this fallback. Inference sets the artificial mask to zero. The teacher reference uses the same endpoint's observation-derived transform. Predictions return to pixels by `Xhat_i=s_i Xhat_i_norm+o_i`. Source: `gait_fidelity/training.py:288–321,569–573,600–608`.

Each four-frame joint patch packs five channels per frame: safe coordinates, confidence, availability, and seconds since the first timestamp. Linear 20→96 projection, a learned joint embedding and sinusoidal temporal position form 384 tokens. **All 384 slots remain in the encoder sequence.** Artificial masking zeros coordinate/confidence/availability channels rather than deleting slots. A token with no usable frame additionally gets the learned missing-query embedding. Time and position identifiers remain available. The bidirectional encoder can use the whole observed window; it is not causal forecasting. The predictor returns all slots, with losses selecting supported queries. Source: `synthetic_training_v2/models.py:94–125,211–217`.

Graph-time masking selects connected named joint regions and cyclic time intervals until reaching the exact token budget `min(max(1,round(.5 n)),n−1)` for `n>1` available tokens; otherwise no artificial masking is added. Overlap and final-region truncation are retained in its audit. A token is available if any of its frames is observed. Other implemented policies were not completed source comparisons. Source: `gait_fidelity/masking.py:21–27,47–125`.

## Core coordinate and feature objectives

For final coordinate fitting, with reference validity `V_i(t,j)`, the endpoint-balanced loss is

\[
L_x=\frac{1}{|I|}\sum_{i\in I}
\frac{\sum_{t,j}V_i(t,j)\|\widehat X^{\rm norm}_{i,t,j}-Y^{\rm norm}_{i,t,j}\|_2^2}
{2\sum_{t,j}V_i(t,j)}.
\]

`I` contains supported endpoints. For masked coordinate pretraining, first average valid frames within each queried patch/joint token, then supported tokens within each endpoint, then endpoints equally. This differs from evaluator NLE: evaluation takes unsquared Euclidean pixel distance and divides by the reference rendering-box diagonal. Source: `synthetic_training_v2/training.py:209–217,254–267`.

Let `u_ik` be predictor logits, `t_ik` detached teacher logits, and `c` the running center. Base feature prediction uses

\[
q_{ik}=\operatorname{softmax}((t_{ik}-c)/.06),\quad
\log p_{ik}=\operatorname{logsoftmax}(u_{ik}/.1),\qquad
L_{\rm CE}=\operatorname{mean}_{i}\operatorname{mean}_{k\in Q_i}[-q_{ik}^{\top}\log p_{ik}].
\]

The query condition is `(artificially hidden OR any input frame missing)` and `any reference frame valid` in that token. Teacher inputs are projected reference coordinates, reference validity as availability/confidence, and the physical timestamps. They are training-only. Source: `gait_fidelity/training.py:578–615`; `gait_fidelity/response_calibration.py:68–99`; `temporal_gait/objectives.py:8–54`.

The full base feature loss is `L_CE+0.05 L_VICReg`. For two independent translated input views, each coordinate shift is uniform in `[-.02,.02]` normalized units. The projector acts on the mean of student encoder token features. It supplies

\[
L_{\rm VICReg}=25L_{\rm invariance}+25L_{\rm variance}+L_{\rm covariance}.
\]

Invariance is mean squared feature difference. Variance averages `relu(1−sqrt(var+1e−4))` over channels/views, with population variance. Covariance penalizes squared off-diagonal entries of each view's sample covariance, averaging over the two views and dividing by feature dimension. A singleton supported batch cannot supply this regularizer. Source: `temporal_gait/objectives.py:57–76`; `gait_fidelity/training.py:617–625`.

## Feature-response extension

For each common queried token and endpoint `i`, define channel-mean removal `H(v)=v−1 sum(v)/D` and

\[
e_i=H[u_i/.1-\operatorname{sg}((t_i-c)/.06)],\quad
L_{\Delta,k}=\frac{\|e_b-e_a\|^2}{2D},\quad
L_{E,k}=\frac{\|e_a\|^2+\|e_b\|^2}{2D}.
\]

The auxiliary support requires the token to be queried at **both** endpoints and all four reference frames valid in **both** patches. Average supported tokens within each pair, then supported pairs equally. Unsupported auxiliary pairs do not erase valid base CE supervision. The same-tensor algebraic difference is `L_delta−L_endpoint=−e_a^T e_b/D`; independently fitted teachers later evolve separately. Independent endpoint masks also change endpoint normalizations, so feature change is not a calibrated physical unit. Source: `gait_fidelity/response_objectives.py:18–57`.

The coordinate-delta control puts normalized coordinate errors into a common scale: `s_ab=(s_a+s_b)/2`, `r_i=(s_i/s_ab)(Xhat_i_norm−Y_i_norm)`. Its per-frame term is `||r_b−r_a||²/4`, followed by four-frame averaging, then the same token/pair reduction. Source: `gait_fidelity/response_objectives.py:60–74`.

At a fresh initialization and 32 fixed training batches with calibration seed 17, let `G_j=sum_b ||grad_P L_j^(b)||²`, including all trainable parameters with unused gradients counted as zero. Set

\[
\lambda_J=.1\sqrt{G_{\rm base}/\max(G_\Delta,G_E)},\qquad
\lambda_C=.1\sqrt{G_{\rm coordinate\ base}/G_{\rm coordinate\ delta}}.
\]

Both JEPA auxiliaries use the same `lambda_J` across seeds. The larger initial auxiliary RMS is 10% of base RMS; both are not individually guaranteed to be 10%. Final fitting initializes afresh. Calibration performs no parameter update. Source: `gait_fidelity/response_calibration.py:24–27,139–174,195–231`.

### Retained feature-calibration strength, independently recomputed

The completed calibration receipt, `outputs/iclr/jepa-response/diagnostics/loss-calibration.json`, records `lambda_J=0.012646811176583523`, 32 batches, and 702,624 trainable parameters for all three JEPA gradient calculations. Direct arithmetic from its `gradients` and `coefficients` fields gives:

| Initial quantity | Unweighted gradient RMS | Weighted RMS / base RMS | Percentage of base RMS |
|---|---:|---:|---:|
| Base feature objective |0.06333581045789884|1|100%|
| Endpoint auxiliary |0.5008045868129164|0.10000000000000002|10%|
| Delta auxiliary |0.00006529059327844553|0.000013037139634433077|0.001303713963443308%|

The last two ratios are `lambda_J × auxiliary_RMS / base_RMS`. This confirms that the shared coefficient equalizes neither the two initial auxiliary gradient magnitudes nor their directions. The calibration deliberately sets the larger auxiliary to 10% of base. These initial derivative measurements do not establish later gradient influence, clipped update contributions or a causal explanation of final performance. They are a retained protocol limitation, not evidence that the coupling has no effect during training.

## Scalar and dense angular supervision

For knee-to-hip and knee-to-ankle vectors `u,v`,

\[
\theta=\operatorname{atan2}(|u_xv_y-u_yv_x|,u^\top v)180/\pi,\quad
A_e=(P_{95}-P_5)(\theta_{e,R})-(P_{95}-P_5)(\theta_{e,L}).
\]

Percentiles use linear interpolation. `atan2(0,0)` is numerically guarded; exactly straight knees have a zero subgradient through the absolute cross product. The coordinate term is retained. Training angular support is fixed from valid reference hip/knee/ankle coordinates and reference segment lengths at least 2 px, common to both legs and both endpoints. A pair must have at least 16 supported frames and at least 80% of its window. Predictions cannot delete this support. Source: `gait_fidelity/measurements.py:96–178`; `gait_fidelity/repair_objectives.py:40–81`.

Writing `S_ab` for common frame/leg support,

\[
L_s=\mathbb E_{(a,b)}[((\widehat A_b-\widehat A_a)-(A_b-A_a))^2/180^2],
\]

\[
L_d=\mathbb E_{(a,b)}\frac{1}{|S_{ab}|}\sum_{(t,\ell)\in S_{ab}}
[((\widehat\theta_b-\widehat\theta_a)-(\theta_b-\theta_a))/180]^2.
\]

The geometry term `L_g` averages squared relative length shortfall `relu((2−length)/2)^2` over segments, legs, endpoints and common frames, then pairs equally. The original paired objective is `L_x+beta L_s+beta L_g`, `beta=1`, whereas base is `L_x`. Source: `gait_fidelity/measurements.py:137–183`; `gait_fidelity/training.py:638–651`.

Repair uses `L_x+0.1 beta L_s+beta L_g` or `L_x+lambda_d L_d+beta L_g`. For each retained encoder/seed, 32 training-only batches at the identical fresh readout set

\[
\lambda_d=.1\beta\sqrt{\frac{\sum_b\|\nabla_\phi L_s^{(b)}\|_2^2}
{\sum_b\|\nabla_\phi L_d^{(b)}\|_2^2}},
\]

where `phi` contains all readout parameters, including zero gradients. Thus weighted dense RMS equals weighted **low-scalar** angular RMS at initialization. Coordinate gradient is only a diagnostic. The encoder and teacher remain frozen; no teacher EMA occurs during readout fitting. Source: `gait_fidelity/repair_training.py:160,168–182,261–305`.

## Optimization and moving averages

The executed phases use 2,000 pretraining updates, 2,000 readout updates, or 4,000 end-to-end direct updates, with AdamW, learning rate `eta0=3e−4`, default betas `(0.9,0.999)`, epsilon `1e−8`, weight decay `.01`, and combined gradient norm clipped at 1 before each update. Let `U` be update count, `k=0,...,U−1`, and `W=max(1,round(.05U))`. The learning rate is

\[
\eta_k=\begin{cases}\eta_0(k+1)/W & k<W,\\
\frac{\eta_0}{2}[1+\cos(\pi(k-W)/(U-W))] & k\ge W.\end{cases}
\]

Feature pretraining then updates the teacher using

\[
m_k=.999-\frac{.009}{2}[1+\cos(\pi k/(U-1))],\qquad
\bar\theta\leftarrow m_k\bar\theta+(1-m_k)\theta.
\]

After the student optimizer step, teacher weights update, then `c←.9c+.1 mean_i mean_(queried valid k) t_ik` using the already computed pre-update teacher tokens and equal supported-endpoint weighting. Source: `gait_fidelity/training.py:512,654–683`.

Readouts share seed-dependent initialization (`seed+100003`) and endpoint sampling. Their optimizers evolve independently. The readout is LayerNorm→linear(96,96)→GELU→linear(96,8), last layer initialized to zero. It predicts normalized coordinate residuals, added to usable input positions; missing positions receive absolute output. Source: `synthetic_training_v2/models.py:128–141,227–236`; `gait_fidelity/training.py:488–510`.

## Scoring and local reproduction boundary

Angular evaluation intersects reference support across **every variant in a source family**, rather than only the two training endpoints. Failed eligible predictions remain with costs: excursion asymmetry 360°, waveform 180°, movement/nuisance response 720°, interaction 1,440°. These are scoring costs. Geometric side assignment uses separately reference-valid bilateral joint pairs at least 4 px apart and a 2 px named-versus-swapped margin; wrong, ambiguous and nonfinite predictions count as failures. Source: `gait_fidelity/evaluation.py:15–149`.

Conditions average within windows, windows within raw motions, motions within people, then seeds. Core/response crossed bootstraps draw people and seeds independently, preserving paired methods, using 2,000 draws with seed 731. Repair's primary is the paired-person t interval after seed averaging, for ViTPose waveform error; its crossed interval is descriptive sensitivity. Three seeds do not create 42 independent participants. Source: `gait_fidelity/evaluation.py:153–172`; `gait_fidelity/repair_evaluation.py`.

The compact packet in `outputs/iclr` supports hash checks, reaggregation of exported participant/condition summaries, published interval reconstruction, calibration-receipt arithmetic and regeneration of manuscript figures. `docs/iclr/evidence/independent_evidence_audit.py` and its JSON result record the local numerical checks. This is not a complete rerun of rendering, extraction, model training or inference: raw motion/body-model assets, rendered videos, per-example coordinate predictions and complete checkpoints are not all in the transferred packet. Existing local analysis CSVs are derived evidence, not newly acquired data.

The fifteen notebooks in `notebooks/gait_fidelity` expose source-parity examples and retained-result reconstruction. Generated arrays and small CPU fits establish software calculations only. They do not contribute scientific participants, runs, confidence intervals or source-study outcomes. The full 102-fit recipe matrix, mask-structure/re-pairing tutorials, protected-person plans and GAVD inventories must not be counted as completed evidence.
