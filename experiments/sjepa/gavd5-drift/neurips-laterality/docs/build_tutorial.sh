#!/usr/bin/env bash
# Rebuild the tutorial PDF; does not train models or regenerate result assets.
set -euo pipefail
TUTORIAL_DOCS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
pandoc "$TUTORIAL_DOCS_DIR/TUTORIAL.md" \
  --from=gfm --standalone --pdf-engine=tectonic \
  --resource-path="$TUTORIAL_DOCS_DIR" \
  --lua-filter="$TUTORIAL_DOCS_DIR/tutorial_pdf.lua" \
  --variable geometry:margin=0.8in \
  --variable fontsize=11pt \
  --variable colorlinks=true \
  --output="$TUTORIAL_DOCS_DIR/TUTORIAL.pdf"
