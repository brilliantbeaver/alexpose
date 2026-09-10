#!/usr/bin/env bash
# Builds revision 08 only. Earlier revisions and their renderers are unchanged.
set -euo pipefail
risex_dir="$(cd "$(dirname "$0")" && pwd)"
risex_stem="08_risex_bilateral_motion_evaluation"
awk '/^## INTRODUCTION/{body=1} body' "$risex_dir/$risex_stem.md" | \
  pandoc --standalone --from markdown --to latex \
  --template="$risex_dir/risex_v08_template.tex" \
  --lua-filter="$risex_dir/risex_v08_layout.lua" \
  --metadata title="Neurologically Motivated Self-Supervised Learning for Bilateral Gait Geometry" \
  --output="$risex_dir/$risex_stem.tex"
(cd "$risex_dir" && tectonic --keep-logs "$risex_stem.tex")
risex_pages="$(pdfinfo "$risex_dir/$risex_stem.pdf" | awk '/^Pages:/{print $2}')"
if [[ "$risex_pages" != 1 ]]; then
  printf 'Revision 08 has %s pages; shorten the text before packaging.\n' "$risex_pages" >&2
  exit 1
fi
if rg -q 'Overfull \\hbox|Overfull \\vbox|Missing character' "$risex_dir/$risex_stem.log"; then
  printf 'Check the revision 08 LaTeX log for layout or missing-glyph errors.\n' >&2
  exit 1
fi
(cd "$risex_dir" && zip -q -j "$risex_stem"_overleaf.zip \
  "$risex_stem.tex" "$risex_stem.pdf" "$risex_stem.md" \
  08_overleaf_instructions.md RISEx-2026-1-page-template.docx)
pdfinfo "$risex_dir/$risex_stem.pdf"
