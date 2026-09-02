# MIT URTC 2026 paper package

The canonical manuscript is `urtc2026_sjepa_gait.tex`. It uses the standard
IEEE conference class and stays below the MIT URTC five-page limit. The
current compiled manuscript is four US-letter pages.

## Files

- `urtc2026_sjepa_gait.tex`: IEEE-style manuscript
- `urtc2026_sjepa_gait.pdf`: compiled review copy
- `urtc2026_sjepa_gait.md`: high-fidelity readable manuscript with inline PNG figures and IEEE-numbered references
- `urtc2026_details.md`: illustrated, step-by-step tutorial for the S-JEPA loss
- `references.bib`: primary and review literature used by the paper
- `figures/`: PDF publication figures and PNG previews
- `make_figures.py`: reproducible figure generator using locked real-run artifacts

## Build

From this directory, use an IEEE-compatible LaTeX installation:

```sh
latexmk -pdf urtc2026_sjepa_gait.tex
```

The LaTeX source is canonical for submission. The Markdown copy mirrors its
content for browser and repository review, but it does not reproduce IEEE
two-column pagination.

Regenerate figures from the real-run artifacts with:

```sh
MPLCONFIGDIR=.matplotlib uv run python docs/make_figures.py
```

## Required checks before submission

1. Verify the listed author names, affiliation, and contact emails.
2. Confirm author order, student eligibility, mentor role, funding, and data-use statements.
3. Compile with the current URTC paper template and verify a maximum of five pages.
4. Review the final PDF for figure readability in two-column print.
5. Recheck every reference against Crossref or its publisher record.
6. Do not describe the reported classifier values as clinical validation or independent-video performance.

The manuscript was compiled with Tectonic 0.16.9. It produces a four-page PDF
with only non-blocking underfull-box warnings. The official 2026 submission
page states a strict five-page maximum:
<https://urtc.mit.edu/submission>.
