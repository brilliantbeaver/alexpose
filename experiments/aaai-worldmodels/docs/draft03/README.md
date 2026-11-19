# Draft03: A sequence-led gait observation benchmark

Read the [PDF](manuscript.pdf), [manuscript](manuscript.md), and [reviewer assessment](REVIEWER.md). Editable prose is [manuscript_source.md](manuscript_source.md); its generator is [build_next_draft03.py](../../scripts/build_next_draft03.py).

GAVD was introduced as the largest publicly accessible collection of abnormal-gait videos annotated by expert clinicians. Its full local catalog contains **1,874 walking sequences**. This study evaluates **456 unique annotated gait sequences**, including **141 normal sequences**, represented by **2,815 forecast windows**. The [primary-source review](reviewer_support/literature_scale.md) verifies the attribution and video-specific scope.

Sequences describe movement observations. Their **91 original recordings**, including **29 normal recording groups**, remain intact during validation and uncertainty estimation. Single-direction boundaries do not establish independence. The [sequence audit](reviewer_support/sequence_audit.md) records shared frames and annotation limits; known overlaps stay within recording groups.

The original primary forecasting result is a **40.8% reduction** relative to persistence, with recordings given equal weight. A separately declared post hoc sensitivity gives sequences equal weight and finds **41.6% [35.9%, 46.9%]**, while resampling whole recording groups. See [results](sequence_weighting_results.json) and [per-sequence errors](sequence_level_forecast_errors.csv). Anomaly and masking results remain unchanged and do not establish superiority over their strongest simple controls. External transfer separately evaluates 773 windows from 168 recordings and 90 coded participants.

The main ranking result uses one annotated comparison figure with two plainly explained scores and one metric. The complete eleven-rule score table and pooled metrics remain in the appendix. Its previous layout and all values are preserved in the [pre-figure-revision snapshot](revisions/before_score_figure_v3/snapshot_manifest.json).

## Identities and reproduction

The reporting revision uses `medwm-reference-v2-draft03-evidence-v3` and publication identity `medwm-reference-v2-draft03-publication-v3`. The method remains `medwm-reference-masking-v2`. No model, cohort gate, fold, pose cache, frozen prediction, or primary score is replaced. The new weighting analysis is `reference-v2-sequence-scale-posthoc-v2`; its first metadata-check failure is preserved separately.

The [pre-revision snapshot](revisions/sequence_scale_before_v2/snapshot_manifest.json) preserves the previous paper, figures, generator, build evidence, and acceptance record. Additional snapshots preserve notebook26 and its generator. Draft01, draft02, and notebooks00–16 remain intact. The old `scripts/build_draft03.py` still writes preserved draft02.

The [evidence](evidence.json), [claim ledger](claim_ledger.json), and [publication manifest](publication_manifest.json) identify completed inputs and displayed values. Original source and participant scores remain beside sequence errors. The primary internal AUROC still ranks recording groups; each calibration component still uses seven or eight disjoint normal groups. Those endpoints are not relabeled as independent-sequence or patient-level results.

With the experiment's Python 3.12 environment and declared local runs, execute from `experiments/aaai-worldmodels`:

```bash
.venv/bin/python scripts/build_next_draft03.py --compile --render
```

The builder verifies manifests and writes Markdown, LaTeX, tables, evidence, and editable SVG/vector PDF figures without fitting models. The [build record](build_record.json) retains compiler and render commands. Flags `--pdftex` and `--render-python` support equivalent tools. Notebook26 independently rebuilds six tables and five SVGs. Example selection rules remain fixed. Page renders, text, contact sheet, and boundary checks are under `rendered/`.

Earlier full-study acceptance remains tied to its original hashes. A separate [reporting acceptance](reporting_acceptance.json) identifies this exact revised paper and assessment. Material changes reopen checks. Author information and submission choices remain to be completed; no submission is performed.

## Generalizability

**GAVD supports a heterogeneous public-video benchmark, but GAVD alone does not establish patient-disjoint, site-disjoint, or prospective clinical generalization.** Sequences describe observations, recordings define available dependence groups, and verified participants would support stronger clinical inference. Expert annotation does not mean this study independently reassessed diagnoses.

External coded-participant transfer adds a different acquisition setting. Age, native frame rate, view, markers, person selection, retention, and the change from video to participant-averaged calibration units remain possible influences. This work tests movement observations and short-term prediction, rather than disease progression or treatment simulation.

The [official Med-WM call](https://med-wm.github.io/aaai27-mwm/) includes dynamics, datasets, benchmarks, uncertainty, and validation. [REVIEWER.md](REVIEWER.md) explains the contribution and remaining novelty questions without assigning acceptance odds.
