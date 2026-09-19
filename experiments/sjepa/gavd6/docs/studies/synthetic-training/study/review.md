# Original-study writeup review

**Independent verdict: approve; no material findings.** The installed `codex:adversarial-review` plugin, version 1.0.6, reviewed the writeup and workflow against the original implementation and retained source/selector artifacts. Its [complete output](review-final.md) is preserved verbatim. The review ran in read-only mode and edited no files.

The reviewer visually inspected all six PNGs at native size and 900-pixel display width. It found no material clipping, panel overflow, text or arrow overlap, ambiguous arrows, clutter or readability failures. It confirmed the distinction between conceptual diagrams, achieved validation results, retrospective oracles and unconfirmed real transfer. It also found the methodology and reported arithmetic consistent with the implementation and retained evidence.

Before independent review, the author shortened and wrapped Figure 1 labels to leave clear space around the arrows. All six final figures were rendered and visually inspected. Figures 1–4 and 6 are diagrams; Figure 5 is an empirical chart generated with Matplotlib from the reconstructed policy errors. The SVG originals contain no embedded raster images; PNGs are review previews.

Local checks verified six accessible SVG titles/descriptions, six HTML image descriptions, all 12 writeup links, all five document sections, seven table values and seven plotted percentages. The [artifact checks](evidence/artifact-checks.json) record those checks; the [layout report](images/layout-check.json) covers text/text intersections and canvas bounds for the five diagrams. Neither replaces visual inspection. The [asset manifest](reviewed-assets.json) records SHA256 hashes of the delivered files and their shared generator/audit dependencies.

The [aggregate audit](evidence/pilot-audit.json) includes input-file hashes and source/decision joins. Three selected-policy means were also independently averaged from the decision CSV. This verifies retained aggregate arithmetic, not per-frame inference, experimental independence, statistical significance or real-world effectiveness.

## Reproduce the assets

From the `gavd6` repository directory in the current macOS environment:

```bash
DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib \
MPLCONFIGDIR=/tmp/st-original-writeup-mpl \
.venv/bin/python docs/studies/synthetic-training/study/build_writeup.py
```

This requires the retained `notebook_runs/synthetic-training/run-02-v1/source` and `run-03-v1/selectors` files listed in the aggregate audit, plus CairoSVG, Pillow, Matplotlib, Mistune, Arial and Homebrew Cairo. The builder reuses the companion proposal's figure style and the existing aggregate-audit module. On another platform, adapt the Cairo library and font paths. It regenerates the SVGs, PNG previews, HTML, audit JSON and reported-results CSV; it does not rerun training or automatically renew this independent review or the asset manifest.
