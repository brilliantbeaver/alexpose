#!/usr/bin/env bash
# Rebuild local review artifacts from retained outputs; no training or inference.
# Packaging does not authorize submission, ethics approval, or public data release.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
genai_submission_python="${GENAI4HEALTH_PYTHON:-../../.venv/bin/python}"
test -x "${genai_submission_python}"
for genai_submission_command in pandoc tectonic pdfinfo pdftotext pdftoppm pdffonts; do
  command -v "${genai_submission_command}" >/dev/null
done
"${genai_submission_python}" -B numerical_supplement/verify.py
"${genai_submission_python}" -B reproducibility/build_evidence.py
"${genai_submission_python}" -B reproducibility/generate_submission_figures.py
tectonic --keep-logs genai4health_paper_draft.tex
tectonic --keep-logs genai4health_extended_abstract.tex
"${genai_submission_python}" -B reproducibility/make_reader_copies.py
"${genai_submission_python}" -B reproducibility/render_and_check.py
"${genai_submission_python}" -B reproducibility/package_submission.py
