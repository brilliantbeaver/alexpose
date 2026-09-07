#!/usr/bin/env bash
# Markdown is the source of truth for the two NeurIPS manuscripts.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
for manuscript in paper extended_abstract; do
  pandoc "${manuscript}.md" \
    --from=markdown --to=latex --standalone --natbib --number-sections \
    --template=manuscript_template.tex --output="${manuscript}.tex"
  tectonic --keep-logs "${manuscript}.tex"
done
