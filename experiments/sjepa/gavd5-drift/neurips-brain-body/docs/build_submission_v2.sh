#!/usr/bin/env bash
# Markdown is canonical. This build does not train or submit anything.
# Builds the V2 workshop draft (laterality-led) from bbfm2026_paper_V2.md.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
submission_python="${BRAIN_BODY_PYTHON:-../../.venv/bin/python}"
"${submission_python}" figures/submission_laterality_generate.py
pandoc bbfm2026_paper_V2.md \
  --from=markdown --to=latex --standalone --natbib \
  --shift-heading-level-by=-1 --template=submission_template.tex \
  --lua-filter=submission_vector_figures.lua \
  --output=bbfm2026_paper_V2.tex
tectonic --keep-logs bbfm2026_paper_V2.tex
