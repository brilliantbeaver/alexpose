# Standalone draft02: reference-aware gait anomaly detection

Read [the manuscript](manuscript.md) or [the compiled PDF](manuscript.pdf). This standalone draft explains the positive forecasting and anomaly-ranking findings in plain language while retaining the controls that determine their meaning. It makes no state-of-the-art or clinical-validation claim.

Bilateral prediction reduces normal-source error by 33.5%, and the prespecified excess-error score has source AUROC 0.793. The current-posture transition reduces forecast error by 40.9%. A newly documented post hoc low-movement diagnostic has AUROC 0.806. This comparison is essential to understanding the anomaly result. It does not change the frozen protocol, score directions, or outputs, and the diagnostic is not a clinically validated score.

## Generalizability and dataset size

GAVD supports exploratory generalization across heterogeneous source videos. It does not establish participant-level or clinical generalizability. The evaluated cohort contains 2,815 windows, 456 sequences, and 91 source videos, including only 29 retained normal sources. All 32 normal source IDs in the pinned inventory were already within the selected labels. Additional GAVD conditions can broaden a separately declared transfer benchmark, but cannot add normal sources. Dataset acquisition must follow a declared coverage goal rather than continue until a significance threshold is crossed.

Source IDs are not verified participant IDs, and more overlapping windows do not increase the independent participant count. Confidence, duration, geometry, and identity gates create unequal annotation retention. External verified participants, sites, clinical labels, and longitudinal observations are needed for stronger medical claims.

## Reproduction and artifacts

From the repository root:

```bash
experiments/aaai-worldmodels/.venv/bin/python experiments/aaai-worldmodels/scripts/build_draft03.py --compile --render
```

The build uses immutable `restart-dynamics-v1` and `restart-full10hz-v1` artifacts, verifies their hashes, and independently recomputes every original forecast mean and anomaly ranking. It never refits a model or overwrites a prior draft. Edit [the prose source](manuscript_source.md) and [the generator](../../scripts/build_draft03.py), then rebuild. The generator reuses the existing evidence-validator and rendering helpers with all output redirected to draft02. Local defaults use portable TinyTeX and a PDF-rendering interpreter under `/tmp`; use `--pdftex` and `--render-python` for equivalent tools elsewhere.

[The claim ledger](claim_ledger.json) separates supported findings from unestablished stronger claims. [The post hoc audit](posthoc_score_audit.json) records exact inputs, score directions, paired bootstrap settings, condition results, and within-fold rankings. [The build record](build_record.json) and [render checks](rendered/checks.json) identify what actually ran. Figures are editable SVG and vector PDF in `figures/`.

The PDF is an internal manuscript with placeholder author details. The official workshop accepts a short-paper body of 2 to 4 pages or a full-paper body of 6 to 8 pages, excluding references and appendices: [official call](https://med-wm.github.io/aaai27-mwm/). The renderer reports actual page counts. No submission has been performed.
