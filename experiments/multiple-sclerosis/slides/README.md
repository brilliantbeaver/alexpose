# Gait tutorial slides

[slides.md](slides.md) is the editable Marp source for a 33-slide visual tutorial through the seven top-level gait notebooks. The September 13, 2026 update explains geometry, training and evaluation, retains unsuccessful results, and separates current notebook demonstrations from the frozen five-fold reference. Fuller explanations and evidence paths appear in speaker notes.

The deck uses 24 SVG diagrams in [images/progress-0913](../images/progress-0913/). Synthetic gait examples are marked schematic. Comparison and control plots read the saved JSON artifacts; notebook demonstration values are transcribed from retained cells and documented in the notes. These exports are documentation updates, not new training runs. The detailed companion report is available as [Markdown](../docs/08-0913-PROGRESS.MD) and [Microsoft Word](../docs/08-0913-PROGRESS.docx).

## Evidence distinctions

- Notebooks 03 and 04 contain a fold-zero 800-update demonstration followed by 400 updates from saved weights, with restarted optimizer, center and schedule. Probe macro F1 stays at 0.600.
- Notebook 06's separate fresh 500-update fold-zero demonstration gives S-JEPA macro F1 0.644 and Random Forest macro F1 0.915.
- The frozen five-fold reference gives pooled macro F1 0.438 for S-JEPA and 0.667 for Random Forest. It predates the final once-per-batch centering repair and is not a rerun of the final corrected trainer.
- Notebook 05's retained visualization run failed. Checkpoints now exist, but a successful retained visualization execution is still absent.
- Source grouping is implemented; participant verification, direct symmetry measurement and confirmatory clinical inference remain unfinished.

## View or export

Open `slides.md` in the Marp for VS Code extension for an editable preview. The exported [HTML](slides.html) supports presentation mode and speaker notes. The [PDF](slides.pdf) is suitable for reading or sharing, and [PowerPoint](slides.pptx) provides an additional presentation format.

Run these commands from the experiment's top-level directory:

```bash
# Regenerate vector diagrams; Python standard library only.
python3 scripts/scripts_make_progress_diagrams.py

# Export self-contained HTML and PDF with pinned Marp CLI 4.5.0.
# Requires Node.js/npm and a supported Chromium browser.
python3 scripts/scripts_build_slides.py

# Export PowerPoint, including explanatory speaker notes.
uv run --script scripts/scripts_build_slides_pptx.py
```

The HTML exporter embeds the SVGs so the HTML can be shared as one file. PDF also preserves vector diagrams and slide outlines; speaker notes remain available in HTML presentation mode and PowerPoint. The PowerPoint exporter keeps text editable and includes SVG diagrams with raster fallbacks for compatibility. Its Python package dependencies are described in the exporter source.

## Updating the evidence

The diagram generator reads `artifacts/runs/r1_g1_1k_s42/results.json` for the frozen comparison and `artifacts/eval/g1/E0_results.json` for controls. The notebook training summaries are documented constants from retained outputs. When results change, review the source notebooks and update the summaries, slide captions and notes; rebuilding by itself does not establish that the interpretation is current.

Keep old frozen runs separate whenever preprocessing, training logic, readout, features or split definitions change. Counts refer to clips or source groups unless explicitly stated otherwise. Charts without intervals do not imply uncertainty estimates.


## Export the progress report to Word

```bash
# Requires uv, pandoc, and rsvg-convert (librsvg).
uv run --script scripts/scripts_build_progress_docx.py
```

The Word export preserves the complete report, editable tables and equations, source links, and SVG figures with PNG fallbacks. Rebuild it after changing the Markdown or its diagrams.
