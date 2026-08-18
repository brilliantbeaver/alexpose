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

The manuscript was compiled with Tectonic 0.17.0. It produces a five-page PDF
with only non-blocking underfull-box warnings, right at the official 2026
submission page's strict five-page maximum:
<https://urtc.mit.edu/submission>.

## The missingness-only control behind the paper

The manuscript reports a missingness-only control in the abstract, in the
evaluation protocol as the first of two confounding audits, and in the results
for both splits. This section documents how that control is built and how to
read it, so a reviewer can trace each reported number back to the notebook.

### Why the control exists

The pose detector does not always find every landmark. When a joint is
occluded, leaves the frame, or is returned with low confidence, notebook 02
marks it invalid. Each condition in the 96-sequence cohort comes from a small
set of source videos, and different videos are easier or harder to track. If a
condition is systematically harder to track, then the pattern of detector
failures becomes correlated with the label. A classifier could then appear to
read gait while actually reading which landmarks tend to disappear. The control
measures how much of the result that shortcut alone could explain.

### How the control is constructed

The construction lives in notebook `06_capstone_health_condition_classifiers.ipynb`
and follows these steps.

1. Collect the per-sequence validity mask shaped [frames, 33 joints], which
   records only whether each joint was seen at each frame.
2. Average over frames to obtain the visible fraction of each of the 33 joints,
   giving 33 numbers per sequence.
3. Average over joints to obtain the visible fraction within each of the 64
   frames, giving 64 numbers per sequence.
4. Concatenate the two into a 97-number signature that holds no coordinates and
   therefore describes only which landmarks were missing and when.
5. Fit the same Random Forest, on the same train and test split, using this
   signature in place of the frozen S-JEPA embedding, so the input is the only
   thing that changes.

The signature is written to `pose_missingness_features.csv` and the scores to
`missingness_only_classifier_metrics.csv`, both under the real artifact folder.

### What the reported numbers mean

On the exact 47/21 split the missingness-only control reaches 0.333 accuracy
with 0.336 macro F1, above the 0.294 majority baseline but well below the 0.619
of the frozen S-JEPA embeddings. On the all-96 split it reaches 0.448 accuracy,
which falls below the 0.490 majority baseline while the embeddings still reach
0.621. Together these support the paper's careful statement that missing pose
patterns carry some label-associated information but do not explain the full
S-JEPA result.

### The limit of the control

This audit rules out only the detector-failure shortcut. It does not rule out
source-video leakage, which is the paper's central limitation and which the
second audit reports separately. Every test video also appears in training, so
a clean missingness result must never be read as evidence of independent-video
or clinical generalization.
