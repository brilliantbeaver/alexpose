# Method: gauge transport before prediction

For each temporal window, the encoder produces an invariant content state, a
parity-sensitive state, and a posterior `q(g_t | x_{1:T})` over the local
semantic gauge. A temporal graph supplies noisy relative-gauge edges. The
model marginalizes or transports predictive latent targets over the posterior
rather than forcing every window into a silently chosen global convention.

## Required components

1. **Typed transformations.** Encode sensor-frame and anatomical actions
   separately; a known synthetic mirror is a calibration action, not evidence
   that the observed gauge is known.
2. **Gauge posterior.** Predict `q(g_t | x)` and expose its calibration.
3. **Temporal synchronization.** Use overlapping windows and relative evidence
   to infer consistent transports, with cycle-consistency diagnostics.
4. **Gauge-marginal predictive objective.** Predict masked latent targets after
   transport and marginalize unresolved gauge states.
5. **Anchored-or-orbit readout.** For odd targets, use an anchor to emit a
   signed value; otherwise emit a two-sign distribution/set. Even outcomes use
   the gauge-invariant representation.

The existing fixed-reflection models are baselines, not submodules to which
these semantics can be retroactively attributed.
