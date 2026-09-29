# Optional full technical reference

The default manuscript uses the concise five-section appendix in `appendix.tex`.
`technical-details.tex` retains the complete technical specification and
numerical inventories. Workspace paths and notebook-specific descriptions have
been removed from its reader-facing text. It supplies exact implementation
reductions and all configuration tables; these are reference material rather
than additional independent experiments. The data and numerical tables have not
been changed or pooled into broader method averages.

To typeset the full reference, make a copy of the project and replace the final
`\input{appendix.tex}` in `main.tex` (Overleaf) or `paper-v08.tex` (portable source)
with `\input{supplement/technical-details.tex}`. Compile normally. Include one
appendix or the other, not both: they intentionally reuse section labels. This
alternative is for detailed auditing; the default concise manuscript is the
submission-facing version.

The full reference contains:

| Sections | Material |
|---|---|
| A–B | Movement generation, normalization, losses, optimization, calibration, completed fits, measurement, and inference |
| C–D | All 16 neural variants; all four observation/edit and six observation/estimator groups |
| E | Success/failure accounting and all four fixed failure costs for every neural variant |
| F | All retained readout-repair means and their interpretation |
| G–I | Reproduction scope, experimental sequence, hypotheses, shape and leakage controls |
| J | Every development participant's response profile |

All required table sources and vector figures are in both bundles. The portable
source bundle additionally contains the complete CSVs, editable SVG figures,
and Python rebuild scripts. `evidence/method_summary.csv`,
`condition_summary.csv`, `condition_effects.csv`, `penalty_sensitivity.csv`,
`penalty_effects.csv`, `repair_means.csv`, and `repair_effects.csv` retain the
unabridged numerical results. The supplied per-person response export supports
rebuilding the main comparison graphics. The available materials do not contain
complete training assets or checkpoints.
