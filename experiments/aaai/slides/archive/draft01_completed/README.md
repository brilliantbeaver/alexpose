# Neuromechanics research slides

Open [index.html](index.html) locally, read the [editable Markdown](tutorial.md), or use the [vector PDF](tutorial.pdf). This deck contains 40 vector slides, with expandable speaker notes and scientific sources. It works offline; external source links require internet access. The earlier 16-slide tutorial remains in [the archive](../pre_neuromechanics/index.html).

The deck teaches the matched own-joint, patch-local and full-context experiment, then presents the completed real 512-head-update primary study, the normal-diagnostic-motivated 4,096-head-update sensitivity, source controls, category descriptions, supervised classifiers and development diagnostics. Primary results come from run `3044566091261819`; the sensitivity is `eba2bd5a6e05efa1`. Motion and nuisance controls are `7407af320461a092` under the primary run. Historical three-seed results remain explicitly labeled and do not substitute for the new study.

The primary AUROC difference has a conditional interval that contains zero. Neither the primary nor longer schedule establishes a learned JEPA advantage. The population has 251 source videos but only 23 normal sources, all previously explored. These are not verified independent patients. The real teacher diagnostic concerns raw KL on eight unchanged joints and does not decompose the full nonlinear anomaly profile.

Rebuild from `experiments/aaai`:

```bash
uv run python scripts/build_research_slides.py --pdf --render
```

The defaults point to the exact completed runs above. To use reviewed replacement artifacts, pass `--run-dir` and `--sensitivity-run-dir`. The builder requires a passed artifact-verification report, a completed real summary, four folds, matching common-source coverage and all declared trained, random-backbone and ridge target/budget result cells. It reads the full-precision artifact values and only rounds for display. `result_input.json` preserves the loaded evidence and source-file hashes, while `deck_spec.json` records all slide descriptions and source links. The primary comparison is never automatically selected from the favorable schedule.

Generation uses Python's standard library. PDF export needs `rsvg-convert` and `pdfunite`; inspection also needs `pdftoppm` and Pillow. On macOS, `brew install librsvg poppler` supplies the command-line renderers. Omit `--pdf --render` to regenerate editable SVG, HTML and Markdown alone. Every page is an editable SVG in `images/`; eight detailed source vectors remain under `source_figures/` and in the expandable notes.

All changed pages must be rendered and visually inspected. The complete rendered deck and contact sheets are in `review/`; [visual_review.md](visual_review.md) records the actual inspection and evidence checks. A successful build is not a scientific validation of the findings. See the [results report](../../../docs/context_study_results.md), [protocol](../../../docs/context_study_protocol.md) and [manuscript evidence](../../../docs/draft01/evidence.json) for the full interpretation.
