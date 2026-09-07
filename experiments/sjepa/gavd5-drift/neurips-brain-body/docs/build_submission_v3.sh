#!/usr/bin/env bash
# Build V3 only; preserve V2 and its shared figure assets.
# Regenerate the V3 figures from verified results and methods; no training or inference.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
submission_v3_python="${BRAIN_BODY_PYTHON:-../../.venv/bin/python}"
"${submission_v3_python}" -B figures/submission_laterality_v3_generate.py
"${submission_v3_python}" -B figures/submission_pipeline_v3_generate.py
test -f figures/submission_laterality_effects_v3.pdf
test -f figures/submission_pipeline_v3.pdf
pandoc bbfm2026_paper_V3.md \
  --from=markdown --to=latex --standalone --natbib \
  --shift-heading-level-by=-1 --template=submission_v3_template.tex \
  --lua-filter=submission_v3_figures.lua \
  --output=bbfm2026_paper_V3.tex
tectonic --keep-logs bbfm2026_paper_V3.tex
