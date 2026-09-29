# Visualizing the completed experiments

Start with [07 · Completed-study walkthrough](../07_completed_study_walkthrough.ipynb). Its visual sequence follows what the experiment does to data, representations and measurements. Each result panel answers the question introduced by the preceding diagram. This is a local review of `outputs/iclr`; notebooks 00–06 retain the source-run and software-fixture workflow. The [algorithm guide](ALGORITHM_GUIDE.md) connects the figures to visible computations, including [F's response pretraining](../experiments/F_jepa_response.ipynb) and [G's frozen-readout repair](../experiments/G_readout_repair.ipynb).

The central distinction is where each intervention occurs. The core compares representation families and output objectives. The response follow-up changes pretraining. The repair changes the coordinate readout while keeping its encoder fixed. Their development participants are the same fourteen people, so the diagrams connect the runs without presenting them as independent replications.

## Visual sequence

| Step | Visual or calculation | What the reader should understand | Connection to existing tutorials |
| --- | --- | --- | --- |
| Locate the experiment | Three-stage lineage diagram with counts read from saved plans | Core: 30 final fits; response: 18 new final fits; repair: 12 new readouts and no new pretraining. Shared predictions and encoders are reused. | 00, 03, F, G |
| Construct paired data | Two paths from an original/edited AMASS movement: estimated observations and projected references | The pair holds nuisance conditions matched. A difference between two motion states is distinct from future prediction. | 01 |
| Count the population | Admitted train/development table and explicit factorization of 55,800 records | Repeated cameras, estimators, naming conditions and edits improve coverage; they do not create independent people. | 01, 06 |
| Define the measurement | Constructed reference and circularly shifted knee-angle waveforms; excursion and response calculations | A whole-window scalar can be preserved while time-aligned angles change. This is an illustration, not a fitted-model example. | 05 |
| Build features | Joint-by-time-token heatmap from the production graph-time sampler on constructed availability | Four frames per joint become one token; natural missing inputs and artificial query masks have separate meanings. | 02 |
| Train and deploy JEPA | Student, predictor, reference teacher and loss flow; separate frozen-encoder deployment path | Teacher features have reference-pose access. They cannot be treated as deployment features. | 03, 04, F |
| Read the core result | Connected base/original-change points for five families, across coordinate, response and waveform error | Coordinate accuracy and movement metrics can differ; the original added-loss package worsens waveform error across these families. | A, C, D, 05 |
| Isolate paired coupling | Visible delta/endpoint residual formulas, cancellation example and six-variant result table | The endpoint control is necessary, and the uncertain primary difference cannot be replaced by a more favorable secondary outcome. | F |
| Repair the readout | Loss-coefficient table, measured initialization gradient plots, and four-readout comparison | Dense is calibrated to the weighted low-scalar gradient strength. The primary repair compares dense with low scalar, on ViTPose. | [G](../experiments/G_readout_repair.ipynb), 06 |
| Account for failure | Stacked successful-error and invalid-pair-cost contributions with zero-response baseline | Published scores include failure costs. Removing failures changes the evaluated population. | 05, 06 |
| Quantify uncertainty | Three primary contrasts with intervals; fourteen repair participants with three seed values each | People and seeds are paired units; thousands of frames cannot substitute for independent participants. | 06 |
| Interpret feature probes | Observed-input encoder and reference-input teacher bars against a zero-response probe baseline | Probe branch, input access and training-population scope determine what a positive diagnostic establishes. | D, F, 05 |

## Read and rerun

Open notebook 07 with the checkout's `.venv` kernel and run all cells from top to bottom. It needs NumPy, pandas, SciPy, Matplotlib, and the usual Jupyter packages. It uses no GPU, body-model assets, pose-estimator installation, network connection, or Slurm command. It reads the source mask sampler and the downloaded evidence; it does not initialize a fixture or use `GF_WORK`.

The loader checks the latest transfer inventory, verifies every selected file's size and SHA-256 hash, rejects a fixture packet, and checks the shared people and seeds. Missing or altered evidence stops execution. You can point `GF_EVIDENCE_ROOT` at another compact packet with the same study structure, but assertions deliberately stop if its scientific scope differs.

Two teaching controls can be edited and rerun: `SHIFT_FRAMES` changes the constructed waveform example; `MASK_SEED` changes the illustrative mask draw. They never change result tables, participant selection, primary outcomes or evaluation costs. No optional widget dependency is required.

To retain a populated notebook, a browser-readable HTML copy, and the figures:

```bash
.venv/bin/python notebooks/gait_fidelity/scripts/execute_walkthrough.py --export-figures
```

The runner uses the invoking Python interpreter in a temporary Jupyter kernel specification. Its outputs go to `outputs/gait-fidelity/notebook-walkthrough-20260925/`: `index.html`, an executed notebook, `execution.json`, and PNG/SVG/PDF figures. `--output` selects another output directory. It does not modify `outputs/iclr`. Source notebooks remain cleared of outputs and are regenerated from `scripts/lessons/lesson_walkthrough.py` by `scripts/build_notebooks.py`. Return to the [tutorial index](../README.md) for the complete folder layout.

The algorithm/fixture runner executes 00–06 and A–G and intentionally excludes 07. Its successful execution would establish software behavior on generated data; it cannot validate the downloaded research evidence. G exposes the differentiable geometry, percentile gradients, equal-pair reductions and matched readout updates underlying the repair panels. Its small constructed examples remain separate from source measurements.

## Visual and scientific conventions

Every chart title distinguishes **SOURCE RESULTS** from **ILLUSTRATION**. Blue generally denotes the base/observed-input branch, orange an added loss or reference-side branch, and teal a distinct control; each figure carries its own explicit labels. Different metrics have separate axes. Error scores are labeled with their units, and the probe's squared-degree units are kept separate from absolute error in degrees.

The core and response primary results pool three pose estimators. The repair's primary result is ViTPose-only. Notebook 07 preserves that distinction rather than connecting their absolute scores as a time series. It reconstructs the saved primary differences and crossed people/seed intervals, plus the repair's declared person-level t interval. Comparisons are descriptive development estimates; a confidence interval that crosses zero does not establish equivalence.

Read the two calibration panels separately. Scalar/coordinate gradient RMS is a diagnostic ratio describing the initial update signals. It does not set the dense coefficient: G shows that the weighted dense gradient is matched to the **weighted low-scalar gradient** across 32 training batches, separately for each encoder and seed. The angle-support panel counts eligible angle entries receiving a nonzero derivative. Neither panel establishes matching throughout optimization or identifies sparse derivatives as the sole cause of an observed difference.

The compact packet contains no raw reference/prediction trajectories or videos. Consequently, it cannot support an empirical animation of restoration, a learned attention map, a before/after skeleton movie, or an epoch-by-epoch learning curve. Such panels would require additional matched source artifacts and an explicit example-selection rule. A synthetic illustration may explain a calculation but cannot supply those missing observations.

For the paper, the strongest compact visual sequence is the experiment lineage, the core objective comparison, and the primary-contrast plot. The repair calibration and failure accounting explain the main competing interpretations. The detailed [results analysis and adversarial review](../../../docs/studies/gait-fidelity/results/iclr-analysis-20260925/README.md) supplies the wider statistical and mechanistic context.
