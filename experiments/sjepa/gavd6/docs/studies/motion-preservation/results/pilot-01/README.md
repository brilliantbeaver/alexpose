# Pilot 01 extracted evidence

These artifacts support the [pilot analysis](../pilot-01-analysis.md). The source is the five executed notebooks copied into `notebook_runs/motion-preservation`. They are not replacements for the original HAIC per-case files.

| File | Origin and limit |
| --- | --- |
| [full_strength_baselines.csv](full_strength_baselines.csv) | Notebook 02, cell 12; all 14 displayed baseline rows |
| [calibration_points.csv](calibration_points.csv) | Notebook 03, cell 10; all 20 displayed operating points |
| [development_summary.csv](development_summary.csv) | Notebook 04, cell 9; all 20 displayed methods |
| [development_decision.json](development_decision.json) | Notebook 04, cell 14; exact displayed decision and interval |
| [displayed_flow_preview.csv](displayed_flow_preview.csv) | Notebook 02, cell 8; only the displayed first 30 distinct diagnostic rows, not all cases |
| [notebook_execution.json](notebook_execution.json) | Saved metadata for notebooks 00 through 04 |
| [author_example_projection_probe.csv](author_example_projection_probe.csv) | Separate local diagnostic on the author HumanML3D example; not the pilot sample |
| [outcome.svg](outcome.svg) | Development retention and mean per-case error ratio |
| [calibration.svg](calibration.svg) | Selected calibration methods relative to the fixed target |

CSV values retain the notebook's displayed precision, generally six decimal places. The first unnamed column is the displayed DataFrame index. No hidden rows or columns were reconstructed from rounded summaries. Error ratios in the figure are `1 - noise_removal`, not ratios of the aggregate MSE columns.

The separate projection probe used the first 64 frames recovered from the official MoMask checkout's `example_data/000612.npy`, without pretrained inference. For seeds 17 through 24, it applied the current `tracking_noise` function to joints 7 and 10, using burst for the first four settings, oscillation for the last four, and amplitudes `0.025 * (1 + 0.3 * (i % 3))` meters. It evaluated observed positions and compared projection with median observed lengths against projection with reference-derived lengths. Reference lengths are a privileged diagnostic. This probe establishes a possible failure mechanism and does not explain the actual development cases by itself.
