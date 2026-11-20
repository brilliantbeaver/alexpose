# Representational Learning with Neurological Gait Videos

Open the [editable poster](SCCUR-JEPA.pptx) or print the [poster PDF](SCCUR-JEPA.pdf) at actual size on a 48 × 34 inch landscape sheet. The [section-by-section design review](poster-review.md) documents the changes from the [previous edition](SCCUR-JEPA-v6.pptx). Earlier PowerPoint editions are preserved as v1 through v5.

## Student tutorial

Open the [interactive tutorial](Gait-Video-Tutorial.html) in a browser or download the [Word handbook](Gait-Video-Tutorial.docx). Both editions explain all 22 notebooks. The research notes specify the questions. The numbered notebooks, protocols and saved results in [the experiment folder](../../experiments/aaai-worldmodels/) document what ran.

| Notebook series | Student investigation |
| --- | --- |
| 00–07 | Hide joint observations and test a classifier on new recordings. |
| 10–16 | Forecast future leg positions and compare them with copying the last posture. |
| 20–26 | Test whether the amount of movement explains a promising anomaly score, then evaluate the frozen model on a second dataset. |

The browser edition includes four interactive worked examples, per-lesson quizzes, a chart comparing measured ranking scores, and an embedded Word download. Its lesson text, figures and exercises work offline. Repository links in the HTML use the local checkout and links in the Word edition point to GitHub.

The two editions share the [lesson sources and original vector illustrations](tutorial/). From the repository root, run the following commands in order. Word generation uses Python with `python-docx`, along with Pandoc and `rsvg-convert`. The HTML build requires Python and Pandoc.

```bash
python3 docs/sccur-jepa/source/build-tutorial-word.py
python3 docs/sccur-jepa/source/build-tutorial-html.py
```

## Conference poster

The navy masthead, Cambria title, Calibri text, side logos, section bars and wide central results column follow [PACS-algebra.pptx](../sccur/PACS-algebra.pptx). The charts and diagrams are editable in PowerPoint.

## Evidence and review

- Study A uses the [illustrated FAQ](../sccur/gait-with-jepa.docx), [archived classifier and overlap tables](../../experiments/penny/docs/faq/evidence/original/) and [video identification results](../../experiments/penny/docs/faq/evidence/later/07_video_identity_probe.csv). The editable dot chart shows totals on the same 21 clips.
- Study B uses the [world-model results release](../../experiments/aaai-worldmodels/docs/research_results.json), [classification protocol](../../experiments/aaai-worldmodels/docs/revision_protocol.md) and [scientific review](../../experiments/aaai-worldmodels/docs/final_scientific_review.md). The classifier comparison displays source-level balanced accuracy and paired intervals on the 17 test recordings.
- The independent coordinate benchmark uses the [cohort table](../../experiments/aaai-worldmodels/docs/draft03/cohort.csv), [ankle trajectory](../../experiments/aaai-worldmodels/docs/draft03/figures/forecast_example_data.csv), [recording-weighted forecast result](../../experiments/aaai-worldmodels/docs/draft03/sequence_weighting_results.json) and [paired ranking comparisons](../../experiments/aaai-worldmodels/docs/draft03/figures/ranking_comparison_data.csv).

To rebuild the current slide, run `node docs/sccur-jepa/source/build.cjs` with pptxgenjs 4.0.1 available. Export it to PDF at its original slide size.
