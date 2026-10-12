# Figure design review

The design-critique skill was applied to publication figures for an educated biomedical research audience. The figures use vector PDF artwork, simple straight lines, aligned panels, and a restrained color palette. No decorative icons, curved routes, or gradients are used in the workflow.

## Findings and corrections

| Figure/version | Finding | Correction and verification |
|---|---|---|
| Pipeline in v3 | Long labels exceeded boxes and the training/test descriptions collided. | Replaced with four equal process boxes and three equal partition boxes; short centered labels and straight arrows. Bounding-box assertions require all text to remain within its panel. |
| Pipeline v4 | “Probes” was unexplained for readers outside ML. | v5 onward says “label classifiers”; the caption is also revised in v6. |
| Results in v3/v4 | Displayed folds 1–5 did not match registry/table 0–4. | v5 onward consistently uses 0–4. |
| Results v5 | Connecting successive folds could look like a time trend. | Straight horizontal segments pair the two methods within each fold. Circles and squares distinguish methods independently of color. |
| Results prototype | Left labels and legend were outside the initial canvas. | Increased left/bottom margins, anchored the legend inside the figure, and reinspected the full rendered image before including it in v5. |
| Confusion matrices | Light text on medium blue made several cells less clear. | Use dark text throughout the observed 0–57.4% range, with a common 0–100% scale for both methods. Exact values are printed in every cell. |

The final workflow uses 8.2–8.5pt labels at its intended 7-inch width and 0.9–1.05pt lines. Results labels use 8–9pt text; marker shape and printed values carry the information without color. The caption states the weighting rule, panel meanings and label direction. Matrix panels have equal dimensions and matching typography.

Read the PDFs at final page size as well as inspecting the source images. Automated panel-bound checks supplement, and do not replace, visual inspection. Final per-page verification is recorded in `delivery-verification.json` and the final review report.
