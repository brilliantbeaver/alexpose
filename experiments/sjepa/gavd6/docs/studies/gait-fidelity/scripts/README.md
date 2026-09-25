# Rebuild the research documents and inspect the source audits

[Current proposal](../README.md) · [Figure gallery](../images/gallery.html) · [Evidence inventory](../evidence/README.md)

These scripts generate documentation and inspect retained audit sources. They do not train models or submit HAIC jobs. For the amended 27-phase response follow-up, use the [HAIC guide](../../../../slurm/gait-fidelity/START_RESPONSE_02.md); the original core run supplies its prepared inputs and comparison artifacts.

## Rebuild the reading experience

Run from the GAVD6 repository root using the existing `.venv`:

```bash
DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib \
  .venv/bin/python docs/studies/gait-fidelity/scripts/build_figures.py
DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib \
  .venv/bin/python docs/studies/gait-fidelity/scripts/build_readout_figure.py --preview
DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib \
  .venv/bin/python docs/studies/gait-fidelity/scripts/build_research_visuals.py --preview
.venv/bin/python docs/studies/gait-fidelity/scripts/build_paper.py
DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib \
  .venv/bin/python docs/studies/gait-fidelity/scripts/build_brief.py
```

The figure builder regenerates all 18 concept SVGs, 36 raster previews and layout receipts. The older `build_proposal_figures.py` and `build_execution_figure.py` names are compatibility wrappers for this same complete build; they need not be run separately. The amended readout figure has its own builder and does not modify the older 18-phase diagrams. The paper builder uses the root README as its scientific source, renders all dollar-delimited LaTeX through Matplotlib MathText into SVGs under `images/equations`, and adds section navigation, figure enlargement and two interactive examples: feature-error coupling and signed/absolute movement-response error. The page works offline, with no external JavaScript or model/data requests. The controls illustrate arithmetic; they cannot change experiment settings.

Open [proposal.html](../proposal.html) for the complete article, [proposal-brief.html](../proposal-brief.html) for the focused overview, and [proposal-brief.pdf](../proposal-brief.pdf) for its two-page print version. The [gallery](../images/gallery.html) includes the original diagrams, amended readout design, new method diagrams, and actual core results. Scope labels distinguish the active follow-up, inherited core design, diagnostics and deferred extensions. Interactive numerical examples are marked as illustrative. The proposal separately reports the completed core’s development means and links their reconstruction; follow-up results remain unreported.

The brief builder reads `proposal-brief.md` for both outputs, requires exactly the five requested sections, and checks that the PDF has two pages. ReportLab lays out 10.5-point text and credited photographs; CairoSVG and pypdf preserve the two research diagrams as vector graphics. It uses Georgia/Arial on macOS or DejaVu on Linux. Dependencies also include BeautifulSoup. The builder writes `records/brief-build.json`; inspect the final pages after changing either prose or layout:

```bash
pdftoppm -scale-to 1500 -png \
  docs/studies/gait-fidelity/proposal-brief.pdf /tmp/gait-proposal-brief
```

Photo crops and reuse licenses are documented in [image provenance](../images/context/README.md). The full proposal's Markdown and the brief are separate authored documents; a change to scientific conclusions should be reflected in both before rebuilding.

### Rebuild only the results graphs

```bash
.venv/bin/python docs/studies/gait-fidelity/scripts/build_results_figure.py
```

This Matplotlib builder reads `method-means.csv` and `paired-contrasts.csv` from the retained core analysis. It writes full and compact SVGs plus 220-dpi PNG previews. Both mean panels use 0–25° axes; labels use one decimal while the source values remain unchanged. The full graph adds a separate panel for the declared paired comparison and its descriptive crossed person/seed intervals. These intervals describe differences, not uncertainty around individual method means.

The builder checks rendered text bounds, text overlaps, and connector/reference intersections with numeric labels, and writes `records/results-figure-build.json`. Its SVGs contain accessible titles/descriptions and editable text. The complete `build_research_visuals.py` command also invokes this builder, so regenerating the method diagrams preserves the Matplotlib result graphs. After rebuilding, regenerate both proposals using the commands above. See the [independent graph review](../reviews/results-figures-matplotlib-20260924.md).

CairoSVG uses the Homebrew Cairo library on this Mac. The local environment also supplies Pillow, Mistune and Matplotlib. Equation SVGs include source LaTeX as accessible alternative text and require no external font or JavaScript service. Use braced font macros (for example `\mathcal{L}`) and avoid `\operatorname`. The SVG sources remain the editable vector artifacts, while 900-pixel previews are convenient for visual review. See the [figure index](../images/README.md) for per-image meaning and rebuild details.

## Local-video viewer

The local viewer remains a deferred feasibility tool. It uses the 91 audited clips in the sibling `experiments/multiple-sclerosis` checkout; none enters the response follow-up. It links original videos without copying them, and optionally creates local thumbnails:

```bash
.venv/bin/python docs/studies/gait-fidelity/scripts/build_video_gallery.py --thumbnails
```

Thumbnail generation requires FFmpeg. Review notes can be exported as JSON; folder-label hiding is not a blinded clinical annotation protocol. The viewer has no independently referenced clinical overlays. A copy of the page alone cannot supply the sibling videos or their redistribution rights.

## Reproduce dated preparation audits

Write new audit output separately so that original evidence is preserved:

```bash
.venv/bin/python docs/studies/gait-fidelity/scripts/audit_masking.py \
  --root "$PWD" --output /tmp/gait-fidelity-mask-check.json
.venv/bin/python docs/studies/gait-fidelity/scripts/audit_video_inventory.py \
  --output /tmp/gait-fidelity-video-check
.venv/bin/python docs/studies/gait-fidelity/scripts/audit_video_cache.py \
  --output /tmp/gait-fidelity-video-check
.venv/bin/python docs/studies/gait-fidelity/scripts/audit_video_timestamps.py \
  --output /tmp/gait-fidelity-video-check
```

The timestamp audit reads the preceding inventory and requires ffprobe. These commands inspect historical source assumptions; rerunning them does not expand the child's population or replace its frozen parent records. Dated review and validation files retain the hashes valid at their original revision.
