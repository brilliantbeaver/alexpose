# Independent methods and world-model framing audit

25 September 2026. Reviewer did not author manuscript prose or modify experiment code, notebooks, retained outputs, or previous documents. This initial review establishes the implementation contract for subsequent independent reviews of versions 01–07. The evidence auditor separately checks numerical results and participant statistics.

## Defensible scientific question and contribution

The strongest question is whether predicting privileged reference-pose features supplies a useful representation for restoring a defined, side-sensitive movement measurement, and how the choice of output objective changes that conclusion. The contribution is a controlled empirical evaluation: it separates coordinate fidelity, image-plane knee-waveform fidelity, signed left/right excursion response, and the reliability of computing those measurements. The paired synthetic procedure provides temporal and anatomical reference alignment; it does not supply clinical ground truth.

The completed results can support a bounded negative finding about the tested reference-paired feature-prediction recipe and a positive methodological finding about measurement-aware evaluation and loss controls. They do not establish a generally inferior JEPA family, an improved world model, learned physical dynamics, future-state prediction, planning, or clinically valid laterality restoration. The encoder attends bidirectionally across a fully supplied time window, and the intervention label is absent from its inputs. “Response” means the difference between separately restored whole movement states at matched times, not an action-conditioned prediction or a temporal derivative.

## Exact data and laterality contract

1. A recorded AMASS motion drives the body model. Source windows contain 128 samples at 25 Hz, spanning 5.08 seconds between first and last samples. The local source configuration uses width 96, four encoder blocks, two predictor blocks, four attention heads, and four-frame patches. Source: `outputs/gait-fidelity/config.json` and `outputs/iclr/walking-core/config.json`; implementation defaults alone are not sufficient evidence of the executed configuration.
2. A swing-gated right-knee local rotation is applied before mirroring. Its gate combines the ankle-height 40th/80th percentiles, smoothstep, and a squared-sine taper at the interval boundaries. Requested magnitudes are body-model parameter edits, not measured changes in image-plane excursion. Technical geometry checks constrain low-foot displacement, penetration, extreme flexion and discontinuities; they do not validate dynamics or pathology. Source: `src/gavd6_sjepa/research_directions/gait_fidelity/preparation.py:135` and `:161`.
3. Physical mirroring reflects the body in a pelvis-defined plane and exchanges anatomical joint sides. Because the right-knee edit precedes this operation, mirrored examples carry the side-exchanged effect. Image-plane excursion need not simply change sign under camera projection. Source: `preparation.py:118`, `:466`, `:480`.
4. Each source family crosses two physical orientations, two cameras (45° and 90°), clear/occluded rendering, three estimator families, correct/global/temporary naming, and movement-state records. Cameras are fitted once to the union of geometries and frozen across paired states. The occluder is fixed in image coordinates. Foreground rendering boxes supply privileged detector boxes. Source: `preparation.py:207`, `:234`, `:482`, `:505`.
5. Global and middle-third temporary swaps permute coordinates, native estimator confidence, and observation flags together. Reference coordinates remain anatomically named and time is unchanged. Source: `src/gavd6_sjepa/research_directions/gait_fidelity/data.py:302`, `:163`.
6. The baseline is paired separately with its exact no-change duplicate and each admitted intervention endpoint. Movement pairs share source, orientation, camera, corruption, observation, and extractor. Baseline repetition is common to every downstream objective, so paired training does not acquire extra endpoints. The no-change duplicate remains a diagnostic and is excluded from the primary nonzero response average. Source: `training.py:43`; `preparation.py:510`; `evaluation.py:329`.
7. Training selects person, raw motion, window, then condition/intervention pair uniformly at successive levels, with replacement. Original person splits separate training and development; original test people are protected. ViTPose and the 15° edit are excluded from source training. No completed protected-person or GAVD result is in the retained packet. Source: `training.py:172`, `:557`; `cohort.py:198`; `preparation.py:466`, `:506`.

“Laterality” here means anatomical side-sensitive trajectory and measurement preservation, with controlled label swaps. The geometric assignment diagnostic is not a calibrated probability of anatomical identity, and some projected bilateral landmarks are unresolved. A paper must not imply that synthetic global swaps reproduce the full ambiguity of real-world laterality.

## Representation, training and inference

The inference dictionary contains only estimated `xy`, `confidence`, `observed`, and `timestamps`. Neither person identity, camera metadata, intervention labels, target validity, nor evaluation scale enters it. Within a window, normalization uses the median origin and Euclidean length of the per-axis 95th–5th-percentile span of available context points. Artificially hidden coordinates are excluded before computing that transform. Fewer than two points or a degenerate span trigger a fixed zero origin/unit scale. The normalization is isotropic and constant over the window. Source: `training.py:288`.

For each joint, four frames of `[x,y,confidence,availability,time]` form a 20-number patch. The 32 temporal patches and 12 joints produce 384 tokens, linearly embedded to 96 channels with joint embeddings and sinusoidal patch positions. Missing tokens remain output queries; their coordinates are zeroed and a missing-query embedding is added. Four noncausal Transformer blocks form the encoder. Source: `src/gavd6_sjepa/research_directions/synthetic_training_v2/models.py:94`.

The coordinate readout is LayerNorm → linear 96→96 → GELU → linear 96→8, unpacked to four `(x,y)` corrections. Its last layer starts at zero. Corrections are added to usable normalized input coordinates; absent coordinates receive absolute predictions. Pixel coordinates are recovered with the input-derived inverse transform. Source: `models.py:128`, `:227`.

During feature pretraining, the student sees masked estimated trajectories. An exponential-moving-average teacher sees projected synthetic references in the student's input-derived coordinate system, with confidence set to reference validity. The predictor maps student encoder tokens to teacher-sized feature logits. Teacher targets are detached. This is privileged paired supervision, not label-free self-supervision on estimated poses alone. Source: `training.py:594`.

The implemented feature loss is cross-entropy over learned feature channels:

\[
q_k=\operatorname{softmax}((t_k-c)/\tau_t),\quad
p_k=\operatorname{softmax}(u_k/\tau_s),\quad
L_{\mathrm{CE}}=\operatorname{mean}_{i}\operatorname{mean}_{k\in Q_i}[-q_k^\top\log p_k],
\quad \tau_s=0.1,\ \tau_t=0.06.
\]

Here each endpoint is equally weighted after its valid queried tokens are averaged. The original JEPA recipe adds `0.05 × VICReg`, with 25 invariance + 25 variance + 1 covariance weighting on two translated student views. The teacher momentum increases from .99 to .999; the center follows .9 old center + .1 current teacher mean. These are implementation choices and should not be implied to reproduce I-JEPA or V-JEPA exactly. Source: `src/gavd6_sjepa/research_directions/temporal_gait/objectives.py:29`, `:57`; `training.py:617`, `:676`.

Core and response frozen encoders are pretrained for 2,000 updates and receive newly initialized coordinate readouts fitted for another 2,000. Direct training jointly updates its encoder and readout for 4,000 coordinate-supervised updates. Matching total updates does not match coordinate exposure, trainable parameters, effective compute, or convergence. Same-seed downstream objectives share initial readout weights and endpoint draws. Source: `training.py:488`, `:499`, `:512` and executed configuration.

## Exact scalar, dense and feature-response objectives

Let `a,b` denote the two complete movement states, `t` time, and `ell` the left/right leg. The projected knee angle is

\[
\theta=\operatorname{atan2}(|u_xv_y-u_yv_x|,u^\top v)180/\pi,
\quad u=\text{hip}-\text{knee},\quad v=\text{ankle}-\text{knee}.
\]

With linear percentiles on fixed reference support,

\[
A_e=(P_{95}-P_5)(\theta_{e,\cdot,R})-(P_{95}-P_5)(\theta_{e,\cdot,L}),
\quad \Delta A=A_b-A_a,
\]

\[
L_s=\operatorname{mean}_{\mathrm{pairs}}\left[\left(\frac{\Delta\widehat A-\Delta A}{180}\right)^2\right],
\quad
L_d=\operatorname{mean}_{\mathrm{pairs}}\operatorname{mean}_{t,\ell}
\left[\left(\frac{(\widehat\theta_b-\widehat\theta_a)-(\theta_b-\theta_a)}{180}\right)^2\right].
\]

The angular support includes both legs and both endpoints, with reference segments at least 2 pixels, at least 16 frames and at least 80% of the window. Prediction failures cannot delete supervised frames. Geometry loss `L_g` is the squared relative deficit below the segment-length threshold, averaged over this same support. Source: `measurements.py:96`, `:137`; `repair_objectives.py:20`, `:40`.

The original base objective is coordinate loss `L_x`. The original paired objective is `L_x + w L_s + w L_g`, with `w=1`. Consequently the original base/paired contrast is a compound objective change, not an isolated scalar-loss intervention. Repair uses

\[
L_{\mathrm{low}}=L_x+0.1wL_s+wL_g,\qquad
L_{\mathrm{dense}}=L_x+\lambda_dL_d+wL_g,
\]

\[
\lambda_d=0.1w\sqrt{\frac{\sum_{b=1}^{32}\|\nabla_w L_s^{(b)}\|_2^2}
{\sum_{b=1}^{32}\|\nabla_w L_d^{(b)}\|_2^2}}.
\]

The subscript `w` in the gradients means readout parameters, whereas the scalar `w=1` above is the inherited coefficient; the manuscript should use different symbols to avoid this collision. Calibration is training-only, once per retained encoder/seed, at the common initialization. It matches the dense gradient magnitude to the *weighted low-scalar angular gradient*, not the coordinate gradient. All new arms retain coordinate supervision and the original geometry coefficient. Matching is at initialization before clipping and does not identify sparse percentile gradients as the unique cause of later performance. Source: `repair_training.py:160`, `:261`, `:295`.

The response-pretraining auxiliary uses a different residual and should never be conflated with angle response:

\[
H(v)=v-D^{-1}\!\sum_d v_d,\quad
e_i=H[u_i/\tau_s-\operatorname{sg}((t_i-c)/\tau_t)],
\]

\[
L_\Delta=\mathbb E\frac{\|e_b-e_a\|^2}{2D},\qquad
L_E=\mathbb E\frac{\|e_a\|^2+\|e_b\|^2}{2D}.
\]

Their difference is `−E[e_aᵀe_b]/D`; shared endpoint error can cancel in the delta auxiliary. Both retain the CE/VICReg anchoring objective, so cancellation does not imply the entire training loss vanishes. Paired auxiliary support requires a queried token at both endpoints and all reference frames valid in both patches; base CE support requires only any reference frame valid at the queried endpoint. A shared JEPA coefficient is set by the larger initial auxiliary gradient energy to 10% of base gradient RMS; it does not make both auxiliaries individually 10%. Source: `response_objectives.py:18`, `:38`; `response_calibration.py:80`, `:219`.

The coordinate-delta control converts endpoint errors to a common pair scale before differencing: `r_i=(s_i/mean(s_a,s_b)) (xhat_i_norm−y_i_norm)`. Its per-frame loss is `||r_b−r_a||²/4`, averaged within a patch and then over supported tokens/pairs. Source: `response_objectives.py:60`.

## Completed experiments and unexecuted controls

| Evidence | Completed new fitting | Interpretation boundary |
|---|---:|---|
| Walking core | 30 final fits; 9 pretraining fits | Five encoder/training families × two output objectives × three seeds |
| JEPA response | 18 new final readouts; 9 pretraining fits | Delta-feature, endpoint-feature, coordinate-delta pretraining; two output objectives |
| Readout repair | 12 new readouts; six initial-gradient calibrations | Two retained encoders × low-scalar/dense objectives × three seeds; no new pretraining |

The 102-final-fit, 33-pretraining full matrix is implemented, not completed. All source pretraining in these completed experiments uses graph-time masks. Topology-shuffled/random-joint mask comparisons, time-block/uniform masks, and label re-pairing are tutorial or planned controls, not results supporting a mask superiority or pairing-specific causal claim. “Readout repair” and “re-pairing” are unrelated operations. Source: `spec.py:6`; `outputs/iclr/*/plan.json`; notebooks 00, B, E, F and G.

Graph-time masking hides approximately half of observed joint-patch tokens under an exact available-token budget. Anatomical region/time intervals use cyclic starts; overlap and final truncation alter realized distributions. These artificial queries are not physical image occluders, and the graph is in the mask sampler, not an anatomically constrained attention architecture. Source: `masking.py:47`.

## Evaluation and diagnostic boundaries

Evaluation fixes angular support by intersecting reference support across every variant of the whole source family, stricter than pairwise training support. It measures waveform mean absolute error in degrees; excursion/response errors are also absolute errors. Failed eligible endpoint excursions cost 360°, waveforms 180°, response/nuisance contrasts 720°, and interactions 1,440°. These are scoring conventions rather than measured angles. Do not remove failed outputs or compare only the successful contribution to a full-score baseline. Source: `evaluation.py:26`, `:43`, `:90`, `:113`, `:133`.

Coordinate training is mean squared error in input-normalized coordinates, whereas coordinate NLE evaluation is unsquared Euclidean distance divided by the reference rendering-box diagonal. Neither is measured in degrees. A low coordinate error alone does not establish a preserved signed excursion response.

The source population is 112 training people and 14 development people; the same development people are reused adaptively across all stages. The protected confirmation plan is not evidence of completed confirmation. Person-separated probe folds hold a person out of ridge fitting only, because all 112 people were available during encoder pretraining. Teacher probes receive reference inputs and cannot demonstrate deployable encoder transfer. Nonzero variance rules out a fully constant representation on the measured panel but does not rule out partial collapse. Gradient clipping is documented on almost every update of the six *new response JEPA* fits; that fact does not prove an explanation of performance or apply automatically to every historical run.

## Review priorities for every manuscript version

**Material if misstated:** separate projected synthetic measurements from anatomical/clinical truth; label the task as restoration; disclose privileged references and teacher inputs; give the original scalar-plus-geometry package accurately; identify the repair's low-scalar calibration target; keep base/paired and delta/endpoint comparisons distinct; retain all three unresolved primary intervals; report the zero-response baseline and failure decomposition; identify ViTPose-only repair versus pooled core/response populations; do not imply completed mask/re-pairing/confirmation/GAVD experiments.

**Important weaknesses requiring new evidence:** small repeatedly reused development population; two-dataset development concentration; no natural-video/clinical references; no convergence or compute-matched feature-learning sweep; no latent re-pairing control; no independent protected confirmation; no raw restored example available locally in the compact packet. Revision should bound these limitations, not erase or declare them solved.

**Revision-level opportunities:** articulate one representation-learning hypothesis, use the compound loss finding to motivate the frozen-encoder repair, explain the null delta/endpoint comparison through benchmark and probe limits, and use a laterality diagram that distinguishes physical side exchange from erroneous channel naming. Source-derived diagrams should show reference/teacher/loss edges only during training and one observed-input restoration path during inference.
