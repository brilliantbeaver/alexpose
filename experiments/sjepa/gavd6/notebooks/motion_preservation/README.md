# Preserve real movement while repairing tracking failures

These six notebooks implement the staged experiment in [Proposal 01](../../docs/studies/motion-preservation/protocol/proposal.md). The question is whether a small gate can preserve real motion while removing tracking noise, beyond a calibrated simple optical-flow baseline.

The default uses real data and configured pretrained models. The explicit `demo` mode uses generated motions and stand-in estimators to teach and check the pipeline. A demo result cannot support a claim about AMASS, GAVD, or a released motion prior.

| Notebook | What you do | What you inspect |
| --- | --- | --- |
| [00 · Data and question](00_data_and_question.ipynb) | Read the existing AMASS/GAVD manifests and runtime availability | People, sources, model configuration and data roles |
| [01 · Controlled pairs](01_make_controlled_pairs.ipynb) | Generate events and tracking errors on development people | Identical skeleton pairs, mixed event/error cases and event timing |
| [02 · Prior, flow and baselines](02_prior_flow_and_baselines.ipynb) | Cache frozen prior repairs and image-motion evidence | Actual backend provenance and full-strength erasure diagnostics before training |
| [03 · Train and calibrate](03_train_and_calibrate.ipynb) | Fit the small gate and choose strengths on calibration people | Optimization history and locked operating points |
| [04 · Preservation and repair](04_preservation_and_repair.ipynb) | Evaluate saved models on development or an explicitly opened final set | Retention versus achieved noise removal and the continue/stop decision |
| [05 · GAVD visual stress](05_gavd_visual_stress.ipynb) | Inspect real-video flow and optional projected trajectory exports | Flow galleries, unresolved ambiguity and transfer limitations |

Run each notebook from top to bottom in a fresh kernel. Every notebook contains the same short setup cell, explains the experiment step, executes it, and displays its outputs. Python implementation lives in `src/gavd6_sjepa/research_directions/motion_preservation`; the notebook cells expose the research sequence without duplicating the numerical code.

## Start on HAIC

Follow the [Slurm and environment guide](../../slurm/motion-preservation/README.md). Use a small pilot before expanding the number of motions. The launcher submits notebooks 00 through 04 with `afterok` dependencies. It does not submit the final event test or GAVD stress test automatically.

For interactive use, set the same environment variables and start Jupyter:

```bash
export GAVD6_ROOT="$PWD"
export MP_RUN_ROOT="/hai/scratch/$USER/motion-preservation/pilot-01"
export MP_CONFIG="$GAVD6_ROOT/slurm/motion-preservation/pilot.example.json"
# Set AMASS, model and video paths as shown in the launch guide first.
"$GAVD6_ROOT/.venv/bin/python" -m jupyter lab notebooks/motion_preservation
```

The example configuration selects a small workload. It does not contain personal scratch paths or licensed body assets. Notebook 00 inventories data without claiming that a listed checkpoint has successfully loaded; notebook 02 performs the actual model load.

## The experiment boundary

Train, calibration, development and final people are separate. Event and corruption variants inherit their person's role. Training uses an arm-leg timing event; development checks foot-clearance changes. The trunk-pelvis timing family remains reserved for final evaluation. Adaptation holdout does not establish pretraining holdout.

The implemented final condition changes the event family, camera angle and corruption mechanism together. This is a combined stress test. It does not isolate event transfer; that attribution needs additional comparisons with the nuisance conditions held fixed.

The same observation tensor can correspond to a true event or a tracking failure. The paired video supplies the distinguishing evidence. Independent tracking noise is crossed with event presence, including overlapping event and noise. The observation-equivalent occlusion fixture tests whether the method remains uncertain when no input can resolve the explanation.

The primary comparison chooses each method's strength on calibration people, then reports both retention and achieved noise removal on new people. Test-set curve points are descriptive. They cannot replace the calibration-locked comparison. Confidence intervals group all trials and variants by person or the explicitly reported weaker identity group.

These are offline restoration experiments using the declared complete clip. No forecasting result is claimed. Flow, pose and image features represent the same RGB observations and are not independent measurements.

## Read results in this order

1. Check that a real frozen prior deletes a supported event often enough to matter.
2. Compare simple confidence and calibrated flow rules before interpreting the learned gate.
3. Check event-plus-noise cases and the reference-informed mixture diagnostic. Its weights use truth before projection, so its post-projection score is not a certified upper bound.
4. Compare retention at locked operating points with comparable achieved noise removal.
5. Inspect ambiguous cases, nuisance controls and whole-body versus reduced-observation limitations.
6. Expand only after the development gain is useful and robust; then open the final event family once.

The provisional development target is a 15 percentage point retention gain over the strongest baseline, at least 25% error removal, and a person-grouped interval supporting improvement. These thresholds are decision rules, not expected outcomes. A strong simple flow baseline is a scientifically useful null result for the learned-method claim.

## Scope and limitations

The implementation is a research prototype, not a claim that all comparisons in the proposal have already been executed. Released motion and flow checkpoints must load in the user's GPU environment. Exact MFTIQ, H-MoRe, HTD-Refine and robust-prior-update results require their actual compatible implementations or explicitly labelled external predictions. An approximation must be named as such.

| Comparison | Implementation status |
| --- | --- |
| Raw, projected raw, Gaussian/median smoothing, robust Kalman | Implemented locally |
| Frozen MoMask and representation-only round trip | Direct author-code bridge; real assets must load |
| SEA-RAFT, alternate torchvision RAFT | Frozen local checkpoint loaders; no automatic download |
| Confidence gate, local/clip flow gate, flow propagation | Implemented locally |
| Small temporal gate, coordinate-only, shuffled-flow, random-feature arms | Implemented locally; no pretrained S-JEPA claim |
| Learned linear flow baseline | Implemented locally to challenge the need for the temporal network |
| Derivative-alignment objective | Bounded local comparator; not HTD-Refine/PVA-Net reproduction |
| MDM or another motion prior | Explicit prediction interchange; generation runs separately |
| Exact MFTIQ, H-MoRe, HTD-Refine, robust-prior updates, second tracker | Compatible external predictions required; not counted as executed by default |
| V-JEPA or pretrained S-JEPA feature branch | Further adaptation experiment; not the initial gate implementation |

The real path renders the actual SMPL-H triangle mesh and computes material-point transport. Appearance is deliberately simple: a fixed pinhole camera, vertex texture and controlled shadow or occlusion fixtures. The demo path uses procedural articulated tubes. Neither establishes photorealistic clothing, clinical pathology, physical gait validity, or calibration on natural videos. GAVD trajectory estimates are not 3D truth. GAVD is used for visible evidence and failure inspection, not a binary classification headline.

The GAVD stage requires the previous study's source-reservation CSV so its held sources remain unopened. It estimates flow and checks optional 2D projected trajectories; it does not automatically apply the trained 3D gate. Extending that gate to real reconstructed motion requires its actual 3D/camera bridge and an explicit transfer experiment.

S-JEPA or V-JEPA features are not automatically validated by a successful temporal gate. Such a branch must be compared against an equally sized direct coordinate model and the flow-only evidence. Any missing external comparison or representation ablation remains part of the research work before a paper-level claim.

## Editing and rerunning

The source notebooks are generated by [build_notebooks.py](../../scripts/research_directions/motion_preservation/build_notebooks.py). Edit that file when changing the tutorial, then regenerate:

```bash
.venv/bin/python scripts/research_directions/motion_preservation/build_notebooks.py
```

Executed copies stay under the run's `notebook_runs` directory, with cell outputs and failures retained. Source notebooks remain output-free. Reuse a run only with the same data and model settings; use a new run directory for a different experimental condition.

The [validation record](VALIDATION.md) separates completed CPU and notebook checks from the real checkpoint and HAIC experiments still to run.
