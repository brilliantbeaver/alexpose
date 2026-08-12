# ADR 0001: Build iteration 2 in a separate gavd2 directory, keep gavd as a checkpoint

**Status:** accepted

## Context

The iteration-1 Gait-JEPA full-dataset series in `gait/skeleton-jepa/gavd/` produced a
complete, runnable pipeline and a set of executed results. A systematic review found
that its headline comparison against the prior Random Forest was not truly controlled
(see ADR 0003 and 0004), so the numbers needed correcting. We wanted to fix the rigor
without losing the iteration-1 artifacts, which are a valid record of where the work
stood.

## Decision

Create a new directory `gait/skeleton-jepa/gavd2/` for iteration 2 and leave `gavd/`
untouched as the iteration-1 checkpoint. iteration 2 is copied from iteration 1 and then
edited, so the changes read as a legible delta and the hard-won iteration-1 fixes
(positional embeddings, the corrected VICReg loss, ffprobe video validation, the
state_dict shape-mismatch guard) are preserved by construction. `gavd2` is fully
self-contained: it keeps its own derived-artifact cache and regenerates every artifact,
sharing only the external downloaded-video cache so it does not re-download.

## Consequences

- `gavd/` remains a citable snapshot; anyone can diff `gavd2` against it to see exactly
  what the rigor pass changed.
- Some duplication (notebooks, helpers, images) between the two directories. Acceptable:
  the two are meant to be independently openable and runnable.
- iteration 2 must not read any `gavd/` derived artifact, or the independence claim
  breaks. This is enforced by pointing its cache at `gavd2/cache` (blank
  `GAVD_CACHE_DIR`) and shipping its own `.env`.
