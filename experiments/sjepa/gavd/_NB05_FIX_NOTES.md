# Notebook 05 (frozen-probe) correctness pass: approach, issues, and fixes

This documents the July 2026 correctness pass on `05-frozen-probe-full-eval.ipynb`
(and the changes it forced in `04-pretrain-jepa-at-scale.ipynb`). It is an internal
engineering note, not a user-facing page.

## The reported symptom

The frozen-probe results "continued to not be right." Running the notebook as
committed (REAL mode) showed the smoking gun:

- The real `labeled_holdout.npz` held only 26 clips, not 68.
- On a single stratified 70/30 split of 26 clips (about 8 test clips), the probes
  gave Linear 0.25, MLP 0.00, RF 0.00.
- RQ3 gave negative test R-squared for BOTH clinical scalars.

## Root-cause analysis (three separate problems)

1. Encoder was a permutation-invariant bag of tokens. The `ContextEncoder` was a
   plain `nn.TransformerEncoder` with NO positional encoding, and the clip embedding
   was `encoder(x).mean(dim=1)` over all T*33 = 1056 tokens. A transformer with no
   positional signal is permutation-equivariant, so a mean over its tokens is a
   permutation-INVARIANT function of the bag of (x, y, z) coordinates. It discards
   frame order AND joint identity. Verified directly: permuting the 1056 tokens
   changed the pooled embedding by ~2e-7. Gait IS temporal and left/right structure,
   so this architecture literally cannot represent the thing being classified.

2. RQ3 probed an UNDECODABLE target. The old RQ3 scalar `stride_time_cv` is a
   coefficient of variation of frame-to-frame ankle displacement: a ratio of
   statistics of differences, which is NONLINEAR in the coordinates. A linear
   (ridge) probe cannot recover it from ANY representation. Verified the linear-probe
   CEILING from the raw flattened coordinates (no encoder bottleneck at all):
   `stride_time_cv` R-squared ~0.02, while `asymmetry_index` ~0.70. So the negative
   R-squared was a property of the target, not evidence about the encoder. The prose
   nonetheless claimed the encoder "learned rhythm variability from stride timing."

3. Headline was a single noisy split, and config had stale VICReg weights. One 70/30
   split of a tiny labeled set is high-variance (single-split 0.81 vs 20-split mean
   0.74 on the synthetic set). And nb05's CONFIG still carried the OLD VICReg weights
   (VAR 25, COV 1, gamma 1) while nb04's corrected loss uses VAR 0.5, COV 0.04,
   gamma 0.5 with a LayerNorm target.

## Fixes

### Architecture (nb04 and nb05, kept in lockstep)

Added learned positional embeddings to `ContextEncoder`: a time embedding (T, D) and
a joint embedding (33, D), added to each projected token as
`pos[t,j] = time_embed[t] + joint_embed[j]`, with tokens in row-major (t, j) order
(token n = t*33 + j). Init std 0.1 so they are on the scale of the projected
coordinates (smaller and the transformer LayerNorm washes them out). This is the
standard I-JEPA / V-JEPA fix and is the ONE intentional divergence from tutorials/03.

nb04 now constructs the encoder with `T` and `N_JOINTS`, and the saved checkpoint
config carries them so nb05 can rebuild the encoder identically. nb05 rebuilds from
the SAVED config and, on any `load_state_dict` shape mismatch (e.g. a stale
checkpoint from the old nb04), prints a WARNING to re-run nb04 and falls back to a
fresh encoder instead of crashing or silently using mismatched weights.

IMPORTANT for the real path: the checkpoint currently on disk was trained by the old
(no-pos-embed) nb04, so it will NOT load into the new architecture. nb04 must be
re-run to regenerate `jepa_encoder_gavd.pt`. The mismatch guard makes this safe.

### Pooling (nb05)

Kept the plain `mean(dim=1)` clip embedding. With positional embeddings this is no
longer a permutation-invariant bag: each token embedding already carries where-in-time
and which-joint context. Head-to-head on the small labeled set, plain mean beat a
mean+temporal-std concat (higher classification accuracy, and the extra 64 dims hurt
the tiny-sample classifier), so we kept the simpler recipe the README specifies.

### RQ3 scalars (nb05)

Replaced `stride_time_cv` with `step_amplitude` (mean of the L and R ankle swing
ranges), and kept `asymmetry_index`. Both are roughly LINEAR in the coordinates, so a
positive R-squared is meaningful. Added an explicit note in code and prose that a
cycle-to-cycle timing CV is nonlinear and not linearly decodable from any embedding,
so we do not probe it (that is a property of the target, not a failure of the
encoder). On the real path you would read the documented H-priority scalars from the
82-feature pipeline instead.

### Stability (nb05)

RQ1, RQ2, RQ3 now report the MEAN (and std) over N_SPLITS = 20 stratified 70/30
splits (StratifiedShuffleSplit), refitting the StandardScaler on each split's TRAIN
fold only. One representative seed split is kept for the confusion matrix. The holdout
loader reports the ACTUAL clip count and falls back to synthetic only when there are
fewer than 10 clips or fewer than 2 per class.

### VICReg / RQ4 (nb05)

nb05 CONFIG now matches nb04's corrected loss (VAR 0.5, COV 0.04, gamma 0.5), and its
`vicreg_loss` is the corrected LayerNorm-target, online-only version. The RQ4 collapse
demo was rebuilt as a faithful miniature of the nb04 loop (block masking + EMA target +
LayerNorm-target L2) with the var/cov terms toggled. The old two-view MSE demo could
not survive the pos-embed encoder (both ON and OFF collapsed, because a mean-pool over
a pos-embed transformer makes the two views trivially match). The JEPA-style demo uses
a faster EMA (0.99) and slightly stronger DEMO-ONLY var weights (1.0, gamma 1.0) over
200 steps so the effect is visible on a fresh encoder. HONEST framing: on easy smoke
data the EMA target does most of the anti-collapse work and VICReg ADDS a margin (ON
std climbs above OFF); we do NOT claim the OFF run crashes to zero.

### Also fixed

nb04's final cell was a markdown cell mistyped as `code` (it would SyntaxError in
Jupyter). Converted to markdown.

## Verified smoke numbers (fresh encoder, SMOKE_TEST=True)

- nb04: loss 32 -> 3.7 over 50 steps, embedding std ~0.58, no collapse. Runs clean.
- nb05: Linear 0.781 +/- 0.086 (beats 76), MLP 0.619, RF 0.405. Label-efficiency
  0.38 -> 0.55 -> 0.71 -> 0.78. RQ3 asymmetry 0.82, step_amplitude 0.95. RQ4
  ON 0.892 vs OFF 0.788. Runs 0-error top to bottom.

## What is NOT claimed

The smoke numbers only prove the machinery is sound. The scientific claim (match or
beat 76% with a frozen probe) lives on the real path, which requires re-running nb04
to regenerate the encoder under the new architecture and then nb05 in REAL mode.
