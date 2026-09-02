# Gait-JEPA on all of GAVD, iteration 2: the controlled-comparison series

> **Current evidence status, reviewed 2026-07-27.** This directory is a historical
> engineering study. Its strongest result is the demonstration that random splits of
> overlapping windows greatly inflate accuracy. The sequence-level model scores remain
> same-video-confounded because 68 sequences come from 12 source videos. The clinical
> scalar regressions use window-level splits, the VICReg statistics flatten token
> positions into the sample axis, and the two enhanced predictor lanes share one
> checkpoint identity. Treat the stored model results as diagnostic and
> hypothesis-generating. See
> [`notes/research-experiment-review.md`](notes/research-experiment-review.md) and the
> [project-level evidence review](../README.md) for the current interpretation.

This is iteration 2 of the full-dataset Skeleton-JEPA series. Its core notebooks 00
through 05 keep the iteration-1 method unchanged and make the comparison against the
prior Random Forest genuinely controlled: same 68 labelled sequences, same label
taxonomy, same pose-extraction contract, and an apples-to-apples per-sequence
classification unit and split. Later notebooks 06 through 09 are separate model-scaling
and predictor follow-ups. Iteration 1 lives one folder over in
[`../gavd/`](../gavd/) and is preserved as a checkpoint.

You do not need a machine-learning background. Every notebook runs on a plain laptop CPU
in seconds in its default smoke mode.

![The full-dataset pipeline](images/pipeline-overview.svg)

## What iteration 2 changes (and why)

A review of iteration 1 found that its "vs 76 percent Random Forest" claim was not a
controlled comparison. Iteration 2 fixes four things while leaving the method identical:

1. **Exact-68 lock.** The labelled probe set is resolved to the exact 68 sequences the
   prior study trained on, by sequence id, with a REAL-mode fail-stop. (`nb00`)
2. **Per-sequence evaluation.** The probe classifies one vector per sequence, not per
   overlapping window, so window leakage cannot inflate the number. exp5's exact seed-42
   split is reproduced for a like-for-like point. (`nb03`, `nb05`)
3. **Video-level leakage control.** Windows whose source video also backs a held-out
   labelled sequence are excluded from pretraining. (`nb03`)
4. **Matched probe + artifact fingerprinting.** A Random Forest matched to exp5's family
   (100 trees, depth 5, balanced), and a canonical-id hash stamped on every artifact.

The honest per-sequence frozen probe is well above the 0.20 chance level and, on this
small labelled set, below the tuned 0.76 baseline. The point of iteration 2 is the
controlled harness and the honest reading it produces, not beating the baseline (that is
future work). See [`docs/paper.md`](docs/paper.md).

## The ten notebooks

Notebooks 00 through 05 define the controlled baseline pipeline. Notebooks 06 through 09
are two paired extensions that reuse the locked data artifacts from 00 through 03.

0. [`00-scan-all-gavd-csvs.ipynb`](00-scan-all-gavd-csvs.ipynb) - scan every sequence,
   lock the labelled set to the exact exp5 68, persist exp5's exact split.
1. [`01-bulk-download-youtube.ipynb`](01-bulk-download-youtube.ipynb) - download the
   unique videos once, labelled-backing videos first, resumable, ffprobe-validated.
2. [`02-batch-extract-skeletons.ipynb`](02-batch-extract-skeletons.ipynb) - run MediaPipe
   BlazePose over every sequence, report coverage of the 68.
3. [`03-build-pretraining-corpus.ipynb`](03-build-pretraining-corpus.ipynb) - normalize,
   window, hold out the 68, exclude co-occurring videos from pretraining.
4. [`04-pretrain-jepa-at-scale.ipynb`](04-pretrain-jepa-at-scale.ipynb) - pretrain the
   JEPA on the unlabelled corpus, watch for collapse.
5. [`05-frozen-probe-full-eval.ipynb`](05-frozen-probe-full-eval.ipynb) - freeze, embed
   per sequence, and compare honestly to the 0.76 baseline.
6. [`06-pretrain-enhanced-jepa.ipynb`](06-pretrain-enhanced-jepa.ipynb) - train a
   four-layer, 128-wide encoder with the original per-token MLP predictor.
7. [`07-enhanced-probe-full-eval.ipynb`](07-enhanced-probe-full-eval.ipynb) - strictly
   load, freeze, and evaluate the Notebook 06 encoder.
8. [`08-pretrain-enhanced-predictor.ipynb`](08-pretrain-enhanced-predictor.ipynb) - keep
   the enhanced encoder and replace the MLP predictor with a two-layer transformer that
   can gather context across tokens.
9. [`09-enhanced-predictor-full-eval.ipynb`](09-enhanced-predictor-full-eval.ipynb) -
   freeze and evaluate the Notebook 08 encoder with the same sequence-level probe harness.

Notebook 08 reduces the stored pretraining loss from 0.542 to 0.372 relative to Notebook
06. Downstream results are mixed: the exact-split Random Forest improves from 0.667 to
0.714, while repeated-split linear and MLP accuracies move from 0.621 and 0.660 to 0.595
and 0.629. The correct conclusion is that cross-token prediction improves objective fit
and one evaluation lane, but is not uniformly superior. See
[`docs/enhanced-predictor.md`](docs/enhanced-predictor.md).

## How to run

### Smoke mode (runs anywhere in seconds)

Every notebook's `CONFIG` has `SMOKE_TEST` as its first key. These committed notebooks
ship with `SMOKE_TEST = False` (the real run that produced the numbers in the docs); set
it to `True` for smoke mode, which builds tiny synthetic data, needs no network and no
pose model, and runs top to bottom in seconds. Open any notebook's Colab badge, or
locally:

```bash
uv sync
uv run jupyter lab 00-scan-all-gavd-csvs.ipynb
```

### Real mode for the controlled 00 to 05 pipeline

Copy `.env.example` to `.env` (a filled `.env` is already provided for this machine) and
set `SMOKE_TEST = False` in each notebook, then run `00` through `05` in order. See
[`RUNBOOK.md`](RUNBOOK.md) for the exact commands and the coverage-chase step. The numbers
in the docs and slides are from this real run on the exact exp5 68 sequences (chased to
68-of-68 coverage): the honest per-sequence probe reads 0.49 (linear) to 0.63 (MLP) and
the exp5 exact-split matched Random Forest reads 0.62, against the 0.762 baseline.

### Real mode for the enhanced experiment pairs

After 00 through 03 have produced the locked corpus and holdout, choose one paired lane:

- run 06 then 07 for the enhanced encoder with the MLP predictor; or
- run 08 then 09 for the enhanced encoder with the transformer predictor.

Notebooks 06 and 08 currently write the same model ID and checkpoint filename. Use a
different `GAVD_CACHE_DIR` for each lane, or preserve a copy before switching. Otherwise
the later pretraining run silently replaces the earlier encoder and the evaluation
notebook cannot identify which discarded predictor produced it.

## Slides and docs

- [`slides/research.html`](slides/research.html) - the honest controlled-comparison talk.
- [`slides/teaching.html`](slides/teaching.html) - the pipeline walkthrough for a newcomer.
  Both are reveal.js decks with reveal.js vendored locally (offline). See
  [`slides/README.md`](slides/README.md).
- [`docs/paper.md`](docs/paper.md) / `docs/paper.html` - the iteration-2 paper.
- [`docs/enhanced-predictor.md`](docs/enhanced-predictor.md) - the complete Notebook 08
  and 09 architecture, training, evaluation, results, limitations, and reproduction guide.
- [`docs/learning/learning-journey.md`](docs/learning/learning-journey.md) -
  the plain-language, high-school-level story of the whole `gavd` to `gavd2` journey: the
  four bugs, the four fixes, why per-clip scoring leaks and per-sequence is the honest
  unit, where Penny's neuroscience maps into the code, and an honest look at whether the
  approach is promising. No machine-learning background needed.
- [`docs/pipeline.md`](docs/pipeline.md) - one page per notebook.
- [`docs/glossary.md`](docs/glossary.md) and [`docs/adr/`](docs/adr/) - the ubiquitous
  language and the architecture decision records.

## Independence from iteration 1

`gavd2/` regenerates all of its own derived artifacts into `cache/` and never reads
`../gavd/`'s cache. It shares only the external downloaded-video cache (`YOUTUBE_CACHE_DIR`
in `.env`) so it does not re-download.
