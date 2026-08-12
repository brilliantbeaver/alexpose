#!/usr/bin/env bash
# Build docs/PROGRESS.pdf from docs/PROGRESS.md, embedding the SVG diagrams.
#
# Needs: pandoc, tectonic (LaTeX engine), rsvg-convert (from librsvg).
#   brew install pandoc tectonic librsvg
#
# The SVG figures are converted to PDF first so LaTeX embeds them as crisp
# vectors, then pandoc + tectonic renders the document.

set -euo pipefail
# This script lives in scripts/; run everything from the experiment root so the
# docs/ and images/ relative paths below resolve correctly.
cd "$(dirname "$0")/.."

BUILD="docs/_pdfbuild"
mkdir -p "$BUILD"

# 1. Convert each referenced SVG to a vector PDF.
for svg in images/*.svg; do
  base="$(basename "$svg" .svg)"
  rsvg-convert -f pdf -o "$BUILD/${base}.pdf" "$svg"
done

# 2. Rewrite ../images/NAME.svg image links to the built PDFs.
python3 - <<'PY'
import re, pathlib
src = pathlib.Path("docs/PROGRESS.md").read_text()
def repl(m):
    alt, path = m.group(1), m.group(2)
    name = pathlib.Path(path).stem
    return f'![{alt}](_pdfbuild/{name}.pdf){{width=85%}}'
out = re.sub(r'!\[([^\]]*)\]\(\.\./images/([^)]+)\)', repl, src)
pathlib.Path("docs/_pdfbuild/PROGRESS_build.md").write_text(out)
PY

# 3. Render to PDF.
pandoc docs/_pdfbuild/PROGRESS_build.md -o docs/PROGRESS.pdf \
  --pdf-engine=tectonic --resource-path=docs \
  -V geometry:margin=1in -V linkcolor:blue -V fontsize=11pt

# 4. Clean up.
rm -rf "$BUILD"
echo "wrote docs/PROGRESS.pdf"
