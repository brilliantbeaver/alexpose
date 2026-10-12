# Participant illustrations: evidence and design review

The new `figures/cases.pdf` is an exploratory illustration of saved participant-level response scores. It uses every one of the 14 development people in a fixed canonical-ID order, represented as P01–P14. It does not select favorable cases or present reconstructed poses that are absent from the compact packet. The original source identifiers and all plotted values are retained in the version-local CSVs.

The three panels answer related but different questions. Panel A compares direct coordinate restoration with unchanged noisy observations after pooling clear and occluded images. Panels B and C compare the same direct procedure with zero-response prediction within clear and occluded images. Positive comparator-minus-direct error differences favor direct. This distinction matters because an improvement over the noisy input need not establish useful recovery relative to zero response.

| Panel | Comparator and scope | Mean gain, degrees | People whose seed-averaged gain favors direct |
|---|---|---:|---:|
| A | Unchanged observations, pooled conditions | 5.143928 | 10/14 |
| B | Zero response, clear images | 1.913510 | 14/14 |
| C | Zero response, occluded images | -5.380161 | 1/14 |

“Success” and “failure” in this illustration describe a relative response-error improvement or deterioration. They do not mean binary measurement validity, successful anatomical side recovery, clinical benefit, or disease classification. The manuscript should use “lower/higher error” or “favors” when discussing these points, so they are not confused with the separately reported measurement-failure rate.

Small gray points are the three fitted seeds, with slight vertical displacement to expose overlapping values; thin horizontal lines show their observed minimum-to-maximum range. Blue diamonds are arithmetic seed means for each person. These ranges are not confidence intervals. They describe three retained fits, not three new participants. All original 720-degree response-failure costs remain in the plotted scores. Every person has all three seed entries in every panel.

Clear/occluded scores are recomputed from the original condition export using the established nonheld:held response weights of 2:1. The results agree to `1e-10` with the prior independent condition audit, and equal pooling of the two observation groups reproduces each direct person/seed score in the core export. Cameras, physical orientations, naming conditions, nominal edit levels, and the three pose estimators remain pooled within each specified observation group. This is an illustration of the same development population, not new confirmation.

The x-axis ranges differ across panels to make the individual values and seed variation legible. Each is explicitly tick-labeled in degrees and includes zero. The figure should therefore be read from the displayed values, not by comparing the physical lengths of effects between panels. The compiled width should remain 5.5 inches; its height is 4.0 inches, with the existing paper style and 8–9 point labels. The color and grayscale exports were inspected, including the lower annotation margin. The PDF retains embedded vector fonts and line art.

## Suggested self-contained caption

Participant-level examples of response improvement and deterioration, showing all 14 development people in fixed canonical order without selection. A: unchanged-minus-direct response error, pooled over observation conditions. B–C: zero-minus-direct response error for clear and occluded images. Positive values favor jointly fitted direct coordinate restoration. Diamonds average seeds 17, 29, and 43; gray points and horizontal ranges show those three fitted values, not confidence intervals. Original 720-degree measurement-failure scoring and person/motion/window/condition aggregation are retained; response strata use 2:1 seen/held edit weighting. All estimators, cameras, naming conditions, and physical orientations are pooled. P01–P14 identify the same people in every panel; source IDs are retained in the audit CSV. The x-axis ranges differ. These exploratory participant summaries are not reconstructed poses or clinical examples.

## Reproduction and artifacts

- `scripts/build_case_figure.py` reconstructs the rows from the core per-person and response condition exports, checks pairing and aggregation, and plots them using `paper_style.py`.
- `evidence/case-figure-seeds.csv` retains all 126 plotted person/seed/panel rows, including both comparator and candidate errors.
- `evidence/case-figure-person.csv` retains all 42 participant means, seed ranges, and comparator directions.
- `evidence/case-figure-provenance.json` records source hashes, definitions, validation results, totals, axis ranges, and figure/script hashes.
- `figures/cases.{pdf,svg,png}` and `figures/cases-gray.png` provide the editable/vector and review outputs.

The full repository command is `.venv/bin/python docs/iclr/versions/v08/scripts/build_case_figure.py`. In a source bundle, `python scripts/build_case_figure.py --from-audit` verifies and plots the retained analysis-ready seed CSV without requiring the upstream experiment exports. Both modes completed successfully. Existing manuscript, figures, evidence, and scripts were not changed by this task.

## Hazards in healthy/disease and world-model motivation

The completed cohort was admitted using source metadata, walking filenames, and projected-geometry checks. This does not verify that every movement is healthy, supply disease labels, or create an evaluated healthy-to-diseased pair. Right-knee rotations are controlled kinematic edits, not validated simulations of stroke, Parkinson's disease, multiple sclerosis, or treatment response. Their nominal amplitude is not the resulting projected response, and neither is an established clinical dose.

The current encoder receives a complete bidirectional observation window and learns from privileged projected reference targets. The task does not test prediction of future physical states, action-conditioned dynamics, causal intervention generalization, multistep rollouts, or planning. The relation to world-model research is a motivation about whether predictive feature learning retains useful movement information, not an empirical demonstration of a learned biomechanical world model.

Anatomical naming, measured side-specific kinematics, and a clinically affected side are different quantities. The geometric assignment diagnostic can detect disagreement with named projected references under its stated separation/margin rule; it does not independently identify a patient's affected side or validate a clinical endpoint. Natural clinical examples elsewhere in the repository were not part of these completed experiments and must not appear as evaluated evidence. A stronger clinical motivation can explain why these distinctions matter while explicitly locating such evaluation in future work.

These boundaries follow the executed configurations and exports and are consistent with `docs/studies/gait-fidelity/proposal.html`, `data/README.md`, `data/clinical-candidates.md`, and the earlier evidence audit. They do not require weakening the empirical statement that response, trajectory, and naming fidelity can disagree in the fitted procedures.
