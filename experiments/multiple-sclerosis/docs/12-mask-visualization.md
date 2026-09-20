# Interpreting stochastic masks and the hip visualization

Audit date: 2026-09-20. Scope: the current `masking_v2` sampler, notebook 02,
the renderer, tokenizer alignment, and the repaired training path.

## What is expected

Sampling is stochastic across examples and training updates. It does not promise
that every joint alternates roles within every window. Each chosen body region
covers a contiguous span, but the union of several overlapping regions can cover
all time blocks of one joint. The `max_time_span_frac=0.75` limit applies to one
region, not to the final union. Hips belong to trunk and leg regions; shoulders
belong to trunk and arm regions. The multiplicity correction only approximately
balances their sampling frequency.

The sampler guarantees context somewhere in the window and at least one visible
token in the clinical set (which includes shoulders). It does not guarantee a
visible hip, a visible leg, or context at every instant. The encoder attends
across the token sequence, so an all-target time block can use context from other
blocks. Bank coverage tests check whether every joint receives both roles over
many draws; they are empirical checks rather than a finite-sample guarantee.

This repository's graph-region sampler is a project-specific choice.
The [S-JEPA paper, Section 3](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04755.pdf)
describes masking joint-time patch embeddings using motion-based sampling. It
does not impose a requirement to expose each joint once in each window.

## Reproduced observation and measured coverage

The old demonstration sampled **once** with `default_rng(0)`, then repeated its
32-frame GIF indefinitely. Rerunning the cell recreated that same RNG seed.
For the laptop profile (33 joints, 8 time blocks), draw 1 masks joints 23–32 for
the entire window. Both hips therefore really were always red in that saved
example. The renderer was reporting the mask accurately.

Drawing 10,000 masks from one advancing seed-0 generator gives:

| Joint | Windows with some visible context | Windows fully targeted for this joint | Targeted joint-time tokens |
|---|---:|---:|---:|
| Left hip (23) | 75.72% | 24.28% | 77.09% |
| Right hip (24) | 75.87% | 24.13% | 76.81% |

These measurements used the default target ratio 0.6, clinical region bias 1.5,
and maximum region span 0.75. Among all 33 joints, the minimum window-context
coverage was 75.25%. The smoke (4 blocks) and GPU (16 blocks) configurations were
also checked: minimum window-context coverage was 55.08% and 89.05%, respectively.
The existing test that every spatial joint embedding receives a context gradient
also passes. No permanent hip-target rule or RNG reset inside the training loop
was found.

Window coverage and token frequency measure different things. A joint visible
in only one of eight blocks counts as a visible window but contributes only
1/8 to that window's visible-token fraction. Similarly, "visible at least once"
and "targeted at least once" can both be true for the same joint and window.

### A real imbalance that window coverage concealed

Uniformly choosing the start of a span that fits entirely inside the window
does not give uniform per-time-block masking. Central blocks belong to more
possible intervals than endpoints do. Overlapping regions compound this effect.
An additional 20,000-mask seed-0 bank gives these **context** frequencies:

| Joint | Block 1 | Block 2 | Block 3 | Block 4 | Block 5 | Block 6 | Block 7 | Block 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Left hip (23) | 47.3% | 23.8% | 12.7% | 9.2% | 9.0% | 12.6% | 23.4% | 46.9% |
| Right hip (24) | 47.2% | 23.7% | 12.5% | 9.1% | 9.3% | 12.6% | 23.8% | 46.9% |

There is no permanently hidden joint/time slot in this bank, but this is clearly
not balanced masking. The minimum context frequency over all joint/time slots
is about 8%. The diagnostics now retain the time axis (`token_visible_frac`),
and regression tests check both roles at every exact slot for 4/8/16-block
profiles. The earlier window-level coverage check could miss such starvation.

These results establish the sampler's behavior, not its scientific optimality.
The training policy is unchanged: forcing each hip to be visible within every
window or equalizing time coverage would define a different masking experiment
and require evaluation. Neither rule is an invariant of the current sampler.

### Why the old animation kept coming back

The on-disk artifacts were inconsistent during the follow-up audit. The new
`mask_demo_samples.gif` was written at 16:30:26 and contained 256 frames, but
`mask_demo_temporal.gif` was regenerated at 16:32:58 with only 32 frames and
"Mask sample 1/1." It still hid both hips throughout its single saved sample.
The timestamp and content establish that the old output was generated again;
an older in-memory notebook cell is a plausible source, but the editor state
cannot be established from those files alone. Notebook source on disk already
contained the eight-sample demo. Merely changing the output filename had left
the user able to view or regenerate a contradictory older artifact.
The user subsequently confirmed their displayed counter was "Mask sample 1/1,"
identifying the one-sample output as the version they were viewing.

## Corrections

- The animation now displays eight consecutive draws from one seeded RNG using
  the same motion window. Samples are not filtered or recolored. It separates
  the mask-sample counter from the frame/time-block counter.
- Explicit left/right hip status stays readable when a hand overlaps a hip.
  An all-target time block is labeled as having context at other times.
- Full-figure redraw preserves the sample counters and hip labels in the saved
  GIF. Visual inspection caught figure-level animated text being omitted when
  blitting was enabled; a regression test now checks the exported label pixels.
- A supplied `frame_group` enforces exact model-window alignment, including
  otherwise-divisible lengths that could silently stretch token durations.
- The notebook reports all-joint token frequency, window-context coverage, and
  fully targeted windows, plus hip context frequencies at each exact time block.
  Its earlier per-block-context assertion was removed
  because that is not a sampler invariant.
- Notebook and CLI now call one `write_mask_demo` helper. All three GIF names
  receive the same animation bytes, and `mask_demo_timeline.png` shows the exact
  sample/time/joint bits as a static grid, with both hip rows labeled.
- A manifest records the seed, full mask bank, hip visibility counts, and artifact
  SHA-256 hashes. `verify_mask_demo` detects old cells overwriting any GIF later.
- Sampler documentation now states the actual guarantees. The sampling code,
  weights, random draw order, and training procedure are unchanged.

The first two laptop-profile draws have left/right visible hip block counts
`[0, 0]` and `[4, 3]` (out of eight). Across the eight displayed draws all 33
joints appear as both context and target. The GIF still repeats a finite saved
sequence; change `DEMO_SEED` to generate another batch.

Across the eight samples the left hip is context in 17 of 64 blocks (68 of 256
frames), and the right hip in 10 blocks (40 frames). No masks were filtered to
obtain this result.

The current artifacts are `artifacts/mask_demo_samples.gif` and
`artifacts/mask_demo_timeline.png`. Both older GIF names are refreshed as aliases.
From the experiment directory:

```sh
python scripts/scripts_mask_demo.py
python scripts/scripts_mask_demo.py --check
```

Reopen notebook 02 from disk if an editor still shows the older one-sample cell;
updating files cannot replace an already-open editor buffer or already-embedded
notebook output. Generated notebook source lives in `scripts/notebook_content.py`.

## Verification

Regression checks cover coverage for 4/8/16 blocks, the real seed-0 hip example,
the distinction between token and window statistics, sample/frame alignment,
exact marker assignments and hip labels, all-target time blocks, GIF export,
compatibility with static/unmasked callers, exact timeline bits, legacy-filename
refresh, and detection of later stale overwrites. The generated mask-demo cell
and coverage cell were executed against a cached training clip, and the static
timeline and an exported frame with both hips visible were visually checked.
All 47 package tests pass. Notebook 02 matches its generator. All three GIF
aliases contain 256 frames and match the manifest hashes. Sampling function
bodies were compared by AST with the repository version and are unchanged
(only their documentation changed).
