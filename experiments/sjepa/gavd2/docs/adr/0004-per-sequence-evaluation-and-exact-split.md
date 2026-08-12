# ADR 0004: Evaluate per sequence, and reproduce exp5's exact split for a like-for-like point

**Status:** accepted

## Context

iteration 1's probe classified per window with a stratified shuffle split over the 296
overlapping windows. Windows from one sequence therefore appeared in both train and
test, so the classifier could match a test window to a near-duplicate training window.
Measured directly, this inflated accuracy by roughly 39 points: the per-clip number read
around 0.88 while the honest per-sequence number is around 0.49. The prior Random Forest
is per sequence, so a per-window number is not comparable to it.

Separately, exp5's single split is order-sensitive: `get_train_test` does
`np.random.seed(42); np.random.permutation(len(all_features))` over the pickle's native
feature-list ORDER and takes the first 47 as train, last 21 as test. Applying seed 42 to
a sorted id set yields a different partition, so a "like-for-like" point requires exp5's
exact ordering, not just its id set.

## Decision

Make the honest per-sequence number the headline. In notebook 05, mean-pool each
sequence's window embeddings into one vector per sequence (exp5's unit), then report a
repeated-split band (mean plus/minus standard deviation) for a linear probe, an MLP, and
a Random Forest matched to exp5's classifier family (100 trees, max_depth 5,
class_weight balanced, seed 42) with a StandardScaler refit per split. Additionally
persist exp5's exact seed-42 47/21 partition over exp5's native list order
(`exp5_split.csv`, written by notebook 00) and report the per-sequence accuracy on that
exact partition as a direct like-for-like point beside 0.76. When coverage is below 68,
the exact-split point is reported restricted to the available ids, never silently
skipped. The per-clip number is kept only as an explicitly labelled leaky diagnostic and
is never headlined.

## Consequences

- The headline is honest and comparable; the per-sequence number is high-variance
  because the test fold is small (about 13 to 21 sequences, uneven class balance), so it
  is reported as a band and with the realized N.
- Reproducing exp5's exact split depends on preserving its feature-list order; the
  resolver captures that order from the pickle (tier 1) or reuses a checked-in copy
  (tiers 2 and 3). If the curated 68 ever change, the checked-in order must be
  regenerated.
