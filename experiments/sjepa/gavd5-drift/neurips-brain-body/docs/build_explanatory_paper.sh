#!/usr/bin/env bash
# Rebuild figures from retained evidence, then the standalone tutorial.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
tutorial_python="${BRAIN_BODY_PYTHON:-../../.venv/bin/python}"
"${tutorial_python}" figures/generate_brain_body_tutorial_figures.py
"${tutorial_python}" figures/submission_laterality_generate.py
pandoc explanatory_paper.md \
  --from=markdown --to=latex --standalone --shift-heading-level-by=-1 \
  --template=tutorial_template.tex --lua-filter=tutorial_vector_figures.lua \
  --output=explanatory_paper.tex
tectonic --keep-logs explanatory_paper.tex
