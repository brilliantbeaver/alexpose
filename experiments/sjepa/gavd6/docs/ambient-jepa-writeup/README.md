# JEPA gait research writeup

Start with the **[research overview and proposed continuation (PDF)](output/pdf/jepa-gait-overview.pdf)** or the [readable HTML version](writeup/overview.html). The seven-page document preserves the compact overview on pages 1–3 and appends a researched recommendation on pages 4–7, including references. The overview was not shortened to accommodate the extension.

The proposed focus is preserving independently measured changes in each leg while identifying comparisons that admit conflicting explanations under occlusion. Its literature review includes September 2026 work and distinguishes the proposed contribution from existing gait-change measurement, biomechanical uncertainty, and JEPA-based gait estimation. A literature search cannot certify that an idea has never appeared before.

Editable sources are the [three-page overview](writeup/OVERVIEW.md) and the [appended research direction](writeup/RESEARCH_DIRECTION.md). The builder combines them in that order. Supporting records include the [search and prioritization record](reviews/RESEARCH-SEARCH.md), [independent novelty audit](reviews/research-direction-novelty-audit.md), [draft critique](reviews/research-direction-draft-review.md), [revision dispositions](reviews/RESEARCH-DISPOSITION.md), and [PDF validation](qa/overview/VALIDATION.md).

The [overview figure notes](figures/OVERVIEW-FIGURES.md) link to editable vector assets, alt text, and provenance for its result graph and proposed-study schematic.

The earlier **[17-page full writeup](output/pdf/jepa-gait-research-writeup.pdf)** remains available with its [HTML](writeup/writeup.html) and [Markdown](writeup/WRITEUP.md). *Can a motion model preserve how someone walks?* develops the methodology and seven proposed research extensions at greater length. It includes six main figures, a technical companion with two further figures, and 22 references. Its earlier future-work discussion is supplemented by the newer literature assessment in the appended continuation.

The [writing plan](plan.html) ([source](PLAN.md)) and the figure package remain available as supporting material.

- [Figure gallery with captions and alt text](figures.html) · [caption source](FIGURES.md)
- [Eight-page vector figure atlas](output/pdf/jepa-gait-figure-atlas.pdf)
- [Independent-review dispositions](reviews/DISPOSITION.md), [initial critique](reviews/initial-adversarial-review.md), [draft review](reviews/draft-adversarial-review.md), and [final review](reviews/final-adversarial-review.md)
- [Figure source hashes and checks](figures/provenance.json) · [248 plotted values](figures/plotted-values.csv)
- [Visual and numerical QA](qa/VALIDATION.md)
- [Full-writeup adversarial review](reviews/writeup-adversarial-review.md), [final closeout](reviews/writeup-final-review.md), and [revision dispositions](reviews/WRITEUP-DISPOSITION.md)
- [Full-writeup validation](qa/writeup/VALIDATION.md) and [build/source manifest](qa/writeup/build-manifest.json)

The central recommendation is a measurement-preservation narrative: pose restoration, asymmetry-change recovery, and anatomical naming require separate evidence. A focused subsequent study should test preservation of side-specific atypical movement under occlusion, with independently measured 3D references. Physics-guided prediction, human-object generation, and learned tool routing follow that validation.

The full narrative reports the existing v08 study; it adds no new model training, statistical inference, or empirical results. It changes no paper-v08 source files. Six figures reproduce existing numerical evidence; two diagrams are explicitly conceptual. Independent adversarial review of the full prose found no unresolved substantive blockers after revision.

## Rebuild

To rebuild the overview and continuation using the environment below:

```sh
/Users/theodoremui/dev/alexpose/.venv/bin/python docs/ambient-jepa-writeup/scripts/build_overview_figure.py
/Users/theodoremui/dev/alexpose/.venv/bin/python docs/ambient-jepa-writeup/scripts/build_research_figure.py
/private/tmp/jepa-writeup-env/bin/python docs/ambient-jepa-writeup/scripts/build_overview.py
pdftoppm -r 125 -png docs/ambient-jepa-writeup/output/pdf/jepa-gait-overview.pdf docs/ambient-jepa-writeup/qa/overview/page
```

The first figure reproduces the three primary comparisons from v08. The second illustrates the proposed experiment and contains no empirical results. These two overview figures are separate from the original eight-page atlas. The overview builder checks that the continuation starts on page 4 and the combined document has seven pages; it does not shrink typography to force the length.

The full report uses one Markdown source for both HTML and PDF. On this macOS workspace, the build used an isolated Python 3.12 environment:

```sh
uv venv --python /Users/theodoremui/dev/alexpose/.venv/bin/python /private/tmp/jepa-writeup-env
uv pip install --python /private/tmp/jepa-writeup-env/bin/python -r docs/ambient-jepa-writeup/scripts/requirements-writeup.txt
/private/tmp/jepa-writeup-env/bin/python docs/ambient-jepa-writeup/scripts/build_writeup.py
pdftoppm -r 110 -png docs/ambient-jepa-writeup/output/pdf/jepa-gait-research-writeup.pdf docs/ambient-jepa-writeup/qa/writeup/page
```

If the environment already exists, skip the first command. The builder also requires Pandoc and the macOS Georgia/Arial font files. Rendering requires Poppler. Vector figures are embedded at their original seven-inch width; the builder never recomputes their numerical estimates. To rebuild the earlier figure package and plan views:

```sh
/Users/theodoremui/dev/alexpose/.venv/bin/python docs/ambient-jepa-writeup/scripts/build_figures.py
/Users/theodoremui/dev/alexpose/.venv/bin/python docs/ambient-jepa-writeup/scripts/build_documents.py
```

These are the exact workspace commands used, with Python 3.12.9; the inherited bare `python` selection is not assumed to work. For another machine, substitute an explicitly configured Python environment. The first command requires NumPy, pandas, Matplotlib, and Pillow. The second requires Pandoc and Poppler (`pdfunite`). Both run from the repository workspace. Version details and all source hashes are retained in the figure provenance. SVGs preserve editable text; PDFs preserve vector artwork and embedded fonts; PNGs are 300 dpi. The final atlas is assembled from the individual vector PDFs.

The HTML pages use local CSS, SVG images and browser-native MathML; they need no external scripts. Relative links to figures and the source manuscript assume this directory stays in its repository location. The PDF embeds its figures and can be shared alone; its optional link to the local source manuscript requires the repository layout. The supplied HAI report was read in its Downloads location and is not copied into this package.
