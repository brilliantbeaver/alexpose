# GAVD S-JEPA tutorial slides

This folder contains three runnable versions of the August 2026 presentation:

- `GAVD_SJEPA_Gait_Tutorial.pptx` for PowerPoint, Keynote, Google Slides, or LibreOffice Impress
- `GAVD_SJEPA_Gait_Tutorial.html` for any modern browser
- `GAVD_SJEPA_Gait_Tutorial.pdf` for reading, printing, and sharing

All three come from `GAVD_SJEPA_Gait_Tutorial.md`. The presentation has 17 main slides, including the title, followed by 18 appendix slides. Speaker notes provide the fuller tutorial explanation. See the pace section below before presenting: 17 main slides do not fit a ten-minute slot that also has to hold questions.

The file names carry no generation number on purpose. This deck describes how the study evolved across several GAVD generations, so naming it after any single one would both misdescribe the content and go stale at the next iteration. Earlier copies were named `GAVD3_SJEPA_Gait_Tutorial`.

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
- the three reflection-symmetry experiments and why their three verdicts mean three different things, an informative null, an artifact, and no credit;
- the two separate blockers behind those verdicts: on the labelled laterality task the cohort, namely 18 source videos and 7.5 percent between-source variance against a preregistered 30 percent, and for arm 2 the lost feature spread, 0.400 down to 0.371 against a control spread of 0.008, which the cohort does not explain;
- the full outer-video experiment that still must be run.

The main result chart keeps two kinds of change separate:

- Exact A2 model revision: 0.619 to 0.857 accuracy and 0.613 to 0.881 macro-F1.
- Lane C evaluation repair: 0.604 to 0.614 mean accuracy and 0.407 to 0.615 fixed-label macro-F1, with the checkpoint and embeddings unchanged.

The appendix preserves class-level F1, concrete confusion examples, missingness controls, majority baselines, all legacy and current ledger rows, the resolved symmetry gate table, the equivariance loss form that satisfied itself instead of the encoder, artifact lineage, reproducibility instructions, and authoritative references.

## Presentation pace

The booked slot is ten minutes including questions, so about eight speaking minutes. Seventeen main slides is roughly 28 seconds each, which the builder reports as over pace against a 40-second guide. The symmetry block has already been tightened from three main slides to two, with the preregistered gate table moved to the appendix, so the audience gets the distinction between the three verdicts rather than ten numbers.

Closing the rest of the gap means cutting main slides, which changes what the talk claims and therefore needs a decision rather than an edit. A ten-slide main talk fits comfortably. The four candidates to move to the appendix, in the order they cost the least, are the masking-math slide, the checkpoint-lineage slide, the all-96 stratification slide, and the Lane C repair slide, whose point survives inside the results-changed slide. Moving a slide to the appendix keeps its speaker notes, so nothing is lost from the written record. If the main slide count changes, update `EXPECTED_MAIN_SLIDES` in `build_slides.py` in the same commit, or the validator will fail the build.

## Build and validate

Requirements:

- Python 3
- Pandoc 3.10 or a compatible recent version
- Node, used only to check the browser runtime syntax
- Tectonic and rsvg-convert, used only for the PDF
- Poppler, optional, used only to check the PDF page count

From this folder, build and validate all three outputs:

```sh
python3 build_slides.py
```

Build only one format:

```sh
python3 build_slides.py --format html
python3 build_slides.py --format pptx
python3 build_slides.py --format pdf
```

Validate existing outputs without rebuilding:

```sh
python3 build_slides.py --check-only
```

The builder also applies a portable Arial-based GAVD color theme to PowerPoint. The validator checks the 17-slide main-talk boundary, the ban on em dash characters, all local image paths, alternative text, JavaScript syntax, PowerPoint archive integrity, 16:9 shape bounds, slide counts, unique HTML slide IDs, offline resource embedding, and one PDF page per slide. It also prints the pace the main-slide count implies for the ten-minute slot. That line is advisory: pace depends on delivery, so it never fails a build.

### How the PDF is made

The PDF goes through Beamer at 16:9, so it is a deck with one page per slide rather than a run of prose. Speaker notes stay hidden, which is Beamer's default, so the PDF shows what an audience sees rather than what the presenter reads. Two small files exist only for this path and are not used by the HTML or PowerPoint builds:

- `beamer_preamble.tex` sizes code blocks, tables, and captions to fit a frame, and lets long artifact names and the 64-character fingerprint wrap instead of running past the edge.
- `beamer_slide_fit.lua` marks every slide `shrink`, which asks Beamer to scale a frame down only when its content would otherwise overflow. Slides that already fit are untouched.

Keeping both out of the Markdown means the same source still produces clean HTML and PowerPoint.

## Run the HTML presentation

Open `GAVD_SJEPA_Gait_Tutorial.html` directly in a browser. It is a self-contained file. All 27 images, the style sheet, and the runtime are embedded, so a network connection is not required. The reference links remain optional external links.

You can also serve the folder locally:

```sh
python3 -m http.server 8000 --directory .
```

Then open `http://localhost:8000/GAVD_SJEPA_Gait_Tutorial.html`.

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

- `GAVD_SJEPA_Gait_Tutorial.md`: shared presentation content and speaker notes
- `build_slides.py`: reproducible builder and validator
- `html/template.html`: accessible browser shell
- `html/slides.css`: 16:9 screen and print layouts
- `html/slides.js`: navigation, overview, notes, fullscreen, touch, and deep links
- `beamer_preamble.tex`: PDF-only typesetting so frames fit and long strings wrap
- `beamer_slide_fit.lua`: PDF-only filter that scales an overflowing frame down
- `figures/`: slide-specific vector result summaries

Most other visuals come from `../docs/figures` and `../images`. The generated HTML embeds 27 images as data resources, and the PowerPoint and PDF embed the same assets.

All reported classifier values remain descriptive. The final encoder had already seen every evaluated row, and Stages 1 through 4 used folder labels. A valid unseen-video estimate requires a source-video outer split and fresh five-stage representation training inside every outer fold.
