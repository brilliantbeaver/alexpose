# FAQ assets and document build

The reader-facing documents are [the FAQ Markdown](../faq.md) and [the downloadable Word document](../faq.docx). The illustrations in `figures/` are original, editable SVG artwork with a restrained palette, large text, descriptive metadata and explicit caveats. Conceptual illustrations are labeled separately from measured results in their captions and metadata.

## Build

From the repository's `experiments/penny` directory:

```sh
uv venv docs/faq/.venv
uv pip install --python docs/faq/.venv/bin/python -r docs/faq/scripts/requirements.txt
docs/faq/.venv/bin/python docs/faq/scripts/build_figures.py
docs/faq/.venv/bin/python docs/faq/scripts/build_docx.py
```

On macOS, CairoSVG also needs the Cairo shared library, available through `brew install cairo`. The Python environment and temporary render output are ignored by Git.

The Word build preserves headings, numbered FAQ questions, tables, lists, captions and external links. Relative source links retain their original hyperlink targets, resolving from the document's `docs/` directory in a local checkout. Downloaded copies need the accompanying source files for those links; web references remain independently clickable. An optional `--base-url` converts relative links when the corresponding files are published at that repository URL.

Every SVG is embedded in the DOCX through Office's `svgBlip` extension with a high-resolution PNG fallback. Word uses the vector media; viewers without SVG support can display the fallback. The renderer verifies media relationships, figure alt text and the page-number field. Both media formats remain in the DOCX archive.

## Evidence and reproducibility

The conceptual diagrams explain the current implementation, including four-frame tokens, ten eligible landmarks, a 60% mask fraction among eligible valid tokens, EMA target updates and four 96-value pooled summaries. Quantitative illustrations use retained GAVD3 CSV artifacts in `gavd3/work/artifacts/real/`, interpreted alongside their notebooks. See the evidence ledger and adversarial review records in `reviews/` for findings and corrections.

The plotting script prints exact retained values in figure descriptions; table-like numeric figures use rounded reader-facing values. Drawn point clouds and mask grids explain concepts and are explicitly illustrative. Saved grouped results have different test partitions from the exact 47/21 classifier comparison. Future latent-loss results are not presented as a valid superiority estimate because the model and reference evaluations use different exposure rules.

## Optional visual check

```sh
mkdir -p docs/faq/render
/Applications/LibreOffice.app/Contents/MacOS/soffice --headless \
  --convert-to pdf --outdir docs/faq/render docs/faq.docx
```

The PDF is a local inspection artifact; the Markdown, Word file and original SVG illustrations are the delivery formats.
