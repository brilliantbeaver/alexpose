# V9 figure redesign

The two figures were redesigned in place at the author's request. The scientific text and numerical evidence are unchanged; the captions now describe the new panel layout and marker conventions. The manuscript retains four main-text pages, one reference page and one appendix page.

## References and alternatives considered

Visual references were inspected in the authors' PDFs of [Joint Embeddings Go Temporal](https://arxiv.org/pdf/2509.25449), Figure 1 (NeurIPS 2024 Time Series in the Age of Large Models workshop), and [Mamba4Cast](https://arxiv.org/pdf/2410.09385), Figures 1–4 (the same workshop). Their compact architecture modules, explicit data paths, aligned statistical panels and conventional axes informed this redesign. No reference artwork was copied.

Three arrangements were considered for Figure 1:

- One continuous architecture diagram would retain long routing arrows and force small labels for preparation, training and evaluation.
- Two side-by-side preparation/training panels would shorten some arrows but give the teacher path too little width at 5.5 inches.
- Three full-width panels give each stage a clear beginning and end. This was selected, rendered, and refined at actual manuscript size.

For Figure 2, adding another panel of paired bootstrap effects would compress both existing questions and repeat Table 1. The selected layout preserves the seed-level prediction scores and same-clip diagnostic while giving each an explicit axis. The diagnostic is a point-and-line plot with direct counts and percentages.

## Figure 1: separate the three questions

Panel (a) distinguishes timestamped target computation from prepared model input. Panel (b) separates visible-patch prediction, full-clip teacher targets and the shared-encoder full-view regularizer. A single short dashed arrow indicates EMA weight updates; solid arrows carry data. Panel (c) isolates the frozen ridge evaluation and states where fitting and scoring occur. Patch dimensions, masking details and normalization definitions remain in the methods.

The design removes stick-figure poses, decorative token tiles, tinted cards and rounded containers. Only trainable/teacher modules receive thin rectangular outlines. STIX serif type integrates with the manuscript; one muted blue accent emphasizes outputs. Layout and line style carry meaning without requiring color.

## Figure 2: make the evidence directly readable

All 30 seed values and six means are taken from the existing evidence JSON. Open dark circles identify initialization; filled blue circles identify trained teachers. Mean ticks and a right-aligned numeric column support direct comparisons. A bounded vertical placement algorithm separates nearby circles without moving any score horizontally.

The diagnostic panel retains the recorded denominators and adds their exact proportions: 33/75 = 44% and 375/375 = 100%. Its axis is labeled as a percentage of diagnostic comparisons. The caption retains the warning that videos and models are reused and that the comparisons are not independent trials. No new interval, model inference or experiment is introduced.

## Production and verification

Figures are generated at the official 396 pt text width. Figure 1 is 202 pt high; Figure 2 is 136 pt high. Primary text is 9–10 pt and secondary labels are 8.5 pt. Matplotlib writes PDFs with embedded TrueType fonts and SVGs with editable text; PNG files support review. The retained numbers and manuscript text outside figure captions are checked for preservation. Layout checks accompany the figures; the final rendered paper and the anonymous source package are checked separately.
