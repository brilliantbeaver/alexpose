# Evidence supporting the two-page writeup

This directory contains reproducible calculations behind [the writeup](../README.md), rather than additional manuscript pages. Source outputs were read only. The analysis does not train models, rerun inference, modify scientific decisions or certify remote scheduler completion.

## Evidence available on 20 September 2026

| Seed | Local source below `outputs/full-runs/` | Verification performed |
| --- | --- | --- |
| 17 | `updates-2000-seed-17/updates-2000-seed-17/` | Verified all 134 files in the expansion manifest; independently reaggregated 6,720 expanded metric cells, retaining unsupported values. |
| 29 | `updates-2000-seed-29/` | Reconstructed 4,608 per-window rows from 12 methods' prediction/reference arrays, matching every saved column; checked all ten stage receipts and 288 summary cells. |
| 43 | `updates-2000-seed-43/` | Same checks as seed 29, with another 4,608 reconstructed rows and 288 summary cells. |

The three source-bundle hashes (input arrays, target arrays and manifest) agree across all seeds, including the hashes recorded in seed 17's diagnostic provenance. Seeds 29/43 contain 24 training and eight development identities, with disjoint person groups and the declared extractor/condition exclusions. Their deterministic unchanged/filter outcomes match seed 17. The per-joint offset and affine controls are retained seed-17 calibrations on that same bundle and are shared across seeds; they are not three independent fits. This analysis does not refit calibration.

All 24 learned fits have complete recorded budgets and finite loss histories: 30,000 optimizer updates per seed, 90,000 altogether. Seed 17's evidence is its retained history/summary; seeds 29/43 additionally retain model files and receipts. The audit verifies saved predictions, not fresh inference from those model files. Seed-17 raw neural arrays, the planned 200-update runs and the suite's scheduler accounting are absent locally. Existing pilot and fixture folders were not counted as additional full runs.

All 431 source files have recorded SHA-256 hashes before analysis and were checked unchanged afterward. The maximum aggregate discrepancy is 8.88e-16 for the seed-17 diagnostic table and below 1e-16 for seeds 29/43 primary summaries. All available files named by their stage receipts match their saved hashes. Details are in [verification.json](verification.json).

## Interpretation of the tables

Position and displacement use the original visible-reference masks and framewise reference-box scale. Each condition receives equal weight within a physical window; windows are averaged within motion, motions within person, and people equally. There is one motion per person here. Unsupported measurements invalidate their aggregate instead of disappearing from the denominator. The same eight people appear in every seed and extractor. Training repetitions do not create additional independent participants.

The writeup uses means of the three fitted seeds. Figure 2 separates extractors into six labeled columns and prints the mean percentage error reduction in every cell. A common blue scale from 0 to 50% applies to both endpoints; light orange marks negative reductions. Outlined bold cells identify the highest mean in each column, without implying a significant difference. Values are rounded to one decimal; zero can therefore include very small changes. The [detailed companion figure](../images/03-seed-ranges.svg) shows means and minimum–maximum ranges to two decimals. Those ranges measure observed training variation rather than population uncertainty. The paired intervals average seeds within each person, resample eight people with replacement 50,000 times using seed `20260920`, and take the 2.5th/97.5th percentiles of the relative error reduction. These are exploratory pointwise intervals, conditional on the three fitted models, without multiplicity or model-selection adjustment. The models are compared at matched update counts, not equal compute.

The all-valid/fixed-scale analysis changes only evaluation support and scale: it includes valid synthetic reference joints even when hidden, and uses one median reference-box diagonal per window. Its newly reconstructed seed-29/43 results remain exploratory. An amplitude ratio is computed within each window as predicted/reference root-mean-square horizontal ankle-separation variation, then balanced across the panel. The reported range 1.28–1.54 is the range of nine seed/extractor means, not the range of individual windows.

Timing summaries keep seeds separate and count repeated record-level events descriptively. They match positive local peaks in demeaned horizontal ankle separation within 0.12 seconds, using the saved diagnostic algorithm. Recall is matched/reference peaks; precision is matched/predicted peaks. Neither metric measures clinical heel strikes. Conditional timing errors exclude unmatched events and must be read alongside coverage. The fixed-scale synthetic reference makes 128/128 records eligible per extractor; original visible-reference eligibility is 32/128. These are correlated track records, not participants. One incomplete RTMPose baseline record leaves some all-valid amplitude aggregates unsupported; missing values remain missing.

| File | Contents |
| --- | --- |
| [per-seed-primary.csv](per-seed-primary.csv) | Every method/seed/extractor primary aggregate, with missing values retained. |
| [three-seed-summary.csv](three-seed-summary.csv) | Mean and observed seed range by method, extractor and metric. |
| [per-person-primary.csv](per-person-primary.csv) | Person-level balanced measurements underlying paired comparisons. |
| [paired-contrasts.csv](paired-contrasts.csv) | Relative reductions, exploratory person-bootstrap intervals and counts of people improved. |
| [per-seed-conditions.csv](per-seed-conditions.csv) | Separate corruption conditions, including hidden synthetic joint errors. |
| [all-valid-fixed-scale.csv](all-valid-fixed-scale.csv) | Expanded synthetic endpoints, including motion amplitude, separated by seed. Calibration rows are available only for seed 17 in this table. |
| [timing.csv](timing.csv) | Reference eligibility, matches, extra/missed events, precision, recall and conditional timing errors. |
| [training.csv](training.csv) | Completion, update counts, recorded time and final feature diagnostics. |
| [verification.json](verification.json) | Input hashes, checks and evidence limitations. |

## Reproduction

From the `gavd6` directory, use an environment with NumPy, pandas, matplotlib and the repository modules:

```sh
MPLCONFIGDIR=/tmp/stv2-mpl .venv/bin/python docs/studies/synthetic-training-v2/writeups/analyze_runs.py
.venv/bin/python docs/studies/synthetic-training-v2/writeups/build_writeup.py
```

The analysis reuses the saved evaluator for raw-array reconstruction and independently reaggregates tables with the earlier audit helper. It imports that helper without running its old entry point; no historical report is rewritten. The PDF builder uses ReportLab and pypdf, with Times New Roman and Arial from the macOS supplemental font directory. It exports two main SVG illustrations and a detailed seed-range companion under `../images/`, all with opaque white backgrounds. The main figures' vector primitives are embedded directly in the PDF, which must contain exactly two pages and no raster images. SVG exports include descriptive titles and use standard font-family/weight declarations. The final PDF was rendered with Poppler and both pages inspected for clipping, overlap and legibility.
