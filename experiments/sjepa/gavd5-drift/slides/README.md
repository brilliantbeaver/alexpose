# GAVD3 S-JEPA tutorial slides

This folder contains two runnable versions of the August 2026 presentation:

- `GAVD3_SJEPA_Gait_Tutorial.pptx` for PowerPoint, Keynote, Google Slides, or LibreOffice Impress
- `GAVD3_SJEPA_Gait_Tutorial.html` for any modern browser

Both outputs come from `GAVD3_SJEPA_Gait_Tutorial.md`. The presentation has 15 main slides, including the title, followed by 16 appendix slides. The main talk is designed for about 10 minutes. Speaker notes provide the fuller tutorial explanation.

## What changed in this version

The main talk now follows the complete method evolution instead of describing only the final pipeline. It explains:

- the legacy 12-sequence, one-video normal-only prototype;
- the failure audit that found video overlap, predictive missingness, and lost temporal order;
- the expansion to 75 normal sequences from 18 videos;
- the provenance risk created by a separate added-normal extraction path;
- the corrected 12-identity, batch-safe target rule;
- the three-part JEPA, VICReg, and group objective;
- the five-stage continuing model lineage and fresh stage optimizers;
- evidence against total collapse together with normal-feature drift;
- the weak canonical five-group geometry;
- the exact meaning of A1, A2, and Lane C;
- previous and current results without overwriting the old values;
- the Lane C evaluation repair with the same checkpoint;
- the full outer-video experiment that still must be run.

The main result chart keeps two kinds of change separate:

- Exact A2 model revision: 0.619 to 0.714 accuracy and 0.613 to 0.742 macro-F1.
- Lane C evaluation repair: 0.604 to 0.653 mean accuracy and 0.407 to 0.625 fixed-label macro-F1, with the checkpoint and embeddings unchanged.

The appendix preserves class-level F1, concrete confusion examples, missingness controls, majority baselines, all legacy and current ledger rows, artifact lineage, reproducibility instructions, and authoritative references.

## Build and validate

Requirements:

- Python 3
- Pandoc 3.10 or a compatible recent version
- Node, used only to check the browser runtime syntax

From this folder, build and validate both outputs:

```sh
python3 build_slides.py
```

Build only one format:

```sh
python3 build_slides.py --format html
python3 build_slides.py --format pptx
```

Validate existing outputs without rebuilding:

```sh
python3 build_slides.py --check-only
```

The builder also applies a portable Arial-based GAVD3 color theme to PowerPoint. The validator checks the 15-slide main-talk boundary, all local image paths, alternative text, JavaScript syntax, PowerPoint archive integrity, 16:9 shape bounds, slide counts, unique HTML slide IDs, and offline resource embedding.

## Run the HTML presentation

Open `GAVD3_SJEPA_Gait_Tutorial.html` directly in a browser. It is a self-contained file. All 23 images, the style sheet, and the runtime are embedded, so a network connection is not required. The reference links remain optional external links.

You can also serve the folder locally:

```sh
python3 -m http.server 8000 --directory .
```

Then open `http://localhost:8000/GAVD3_SJEPA_Gait_Tutorial.html`.

### Browser controls

- Right arrow, Page Down, or Space: next slide
- Left arrow or Page Up: previous slide
- Home or End: first or last slide
- `O`: slide overview
- `N`: speaker notes
- `F`: fullscreen
- `?`: keyboard help
- Escape: close the top panel
- Swipe left or right: touch navigation

The toolbar also provides navigation, overview, notes, fullscreen, print, and help buttons. Hash links open a specific slide directly. The Print button creates one 16:9 slide per page, which can be saved as a PDF.

## Source files

- `GAVD3_SJEPA_Gait_Tutorial.md`: shared presentation content and speaker notes
- `build_slides.py`: reproducible builder and validator
- `html/template.html`: accessible browser shell
- `html/slides.css`: 16:9 screen and print layouts
- `html/slides.js`: navigation, overview, notes, fullscreen, touch, and deep links
- `figures/`: slide-specific vector result summaries

Most other visuals come from `../docs/figures` and `../images`. The generated PowerPoint embeds 22 SVG figures and one PNG confusion matrix. The generated HTML embeds the same assets as data resources.

All reported classifier values remain descriptive. The final encoder had already seen every evaluated row, and Stages 1 through 4 used folder labels. A valid unseen-video estimate requires a source-video outer split and fresh five-stage representation training inside every outer fold.
