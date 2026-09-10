#!/usr/bin/env bash
set -euo pipefail

risex_dir="$(cd "$(dirname "$0")" && pwd)"
for markdown_file in "$risex_dir"/0[1-7]_*.md; do
  stem="${markdown_file%.md}"
  tex_file="${stem}.tex"
  pdf_file="${stem}.pdf"
  zip_file="${stem}_overleaf.zip"
  tail -n +4 "$markdown_file" | pandoc --standalone \
    --from markdown \
    --to latex \
    --resource-path="$risex_dir" \
    --template="$risex_dir/risex_onepage_template.tex" \
    --metadata title="Neurologically Motivated Self-Supervised Learning for Bilateral Gait Geometry" \
    --output="$tex_file"
  perl -0pi -e 's/\\begin\{longtable\}/\\begin{tabular}/g; s/\\end\{longtable\}/\\end{tabular}/g; s/\\endhead\n\\bottomrule\\noalign\{\}\n\\endlastfoot\n//g' "$tex_file"
  if [[ "$(basename "$markdown_file")" == 05_* ]]; then
    perl -0pi -e 's!\\begin\{tabular\}\[\]\{@\{\}\n.*?@\{\}\}!\\begin{tabular}{\@{}p{0.30\\columnwidth}p{0.25\\columnwidth}p{0.35\\columnwidth}\@{}}!s' "$tex_file"
  elif [[ "$(basename "$markdown_file")" == 03_* || "$(basename "$markdown_file")" == 06_* ]]; then
    perl -0pi -e 's!\\begin\{tabular\}\[\]\{@\{\}lrr@\{\}\}!\\begin{tabular}{\@{}p{0.40\\columnwidth}p{0.22\\columnwidth}p{0.28\\columnwidth}\@{}}!g' "$tex_file"
  else
    perl -0pi -e 's!\\begin\{tabular\}\[\]\{@\{\}\n.*?@\{\}\}!\\begin{tabular}{\@{}p{0.43\\columnwidth}p{0.47\\columnwidth}\@{}}!s' "$tex_file"
  fi
  (cd "$risex_dir" && tectonic "$(basename "$tex_file")")
  (cd "$risex_dir" && zip -q -j "$(basename "$zip_file")" "$(basename "$tex_file")" "$(basename "$pdf_file")" "risex_onepage_evidence.png" "risex_onepage_template.tex")
done
