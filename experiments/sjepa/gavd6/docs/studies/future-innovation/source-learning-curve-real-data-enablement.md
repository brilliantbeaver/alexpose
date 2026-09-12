# Connecting the source learning curve to real videos and notebook results

The real-video pipeline already contained preparation, teacher encoding, nested
fitting, scoring and notebook execution. This revision closes a missing-media
failure path and makes the stages' evidence visible in notebook 23. It preserves
the repaired predictor and the scientific comparison. **The expanded GAVD curve
has not been measured in this session.** HAIC is not accessible, and the local
checkout lacks the full videos, annotations and teacher checkpoint.

## What the current evidence establishes

The retained development inventory contains **1,874 annotated sequences from
348 recordings**. Its reservation assigns **304 recordings / 1,696 sequences**
to development and **44 recordings / 178 sequences** to confirmation. These
are metadata counts, not counts of successfully decoded or pose-eligible clips.
Only the original **50 clips from 43 recordings** have the inherited completed
teacher evidence. Participant identities remain unknown without an explicit
registry. See the [frozen cohort audit](../../../outputs/future-innovation-source-curve-dev-20260911-v2/reports/cohort-audit.json).

Additional videos need their own teacher features. The existing 50-window cache
cannot supply those arrays. The expanded path preserves input frames 0–31, the
projected person-region target at frames 38–39 encoded in the full 64-frame clip,
and the parent projection. It therefore continues to predict contextual teacher
features. Confirmation recordings remain outside pose extraction, encoding,
training, donor matching and evaluation.

## Repairs and their evidence

| Finding | Change | Verification |
|---|---|---|
| An existing directory did not establish that the development videos were present. | Submission discovers exact recording IDs, including nested files or declared paths, and rejects missing, ambiguous and empty development files. `submit.sh check` performs these checks without submission or output creation. | Shell tests cover success and failure before a run is created; reserved files may remain unavailable. |
| The shared candidate builder could seal a reduced cohort after recording missing media as exclusions. | Expanded `prepare` repeats the availability check against the frozen reservation before calling the unchanged historical builder. Missing media cannot redefine the development cohort. | A fixture with 25 available recordings and a missing 26th reproduces the old sealed 50-candidate result. The repaired path blocks before sealing and permits retry after restoration. |
| Model files could exist at the wrong paths or contain different bytes. | Preflight reuses the same parent-bound annotation, pose-model and checkpoint digest checks as preparation. | Replacing a model behind an existing path is rejected. Teacher loading retains its existing clean-checkout, strict-weight and geometry checks. |
| Successful notebook execution could obscure which upstream stage failed. | Jobs 19–23 invoke the existing stages through a small logging wrapper. Each attempt retains its command, timestamps, fold, exit status and output. | Real subprocess tests cover failure, retry, startup failure, historical-root protection, escaped output paths and CLI error propagation. Existing production locks continue to reject duplicate stage writers. |
| The initial 50-clip eligibility count could be mistaken for an expanded processed count. | Notebook 23 separates the initial audit from completed preparation counts, fold counts, cache reuse/new encoding, model receipt counts and stage attempts. | Notebook tests distinguish these tables and preserve failed attempts after a retry. |
| A saved report alone did not show that numerical verification had finished. | The report job's separate verifier receives a stage record bound to unchanged study, plan and report bytes. The notebook labels a matching record as a previous verification job. | Missing/changed bindings fail; altering report bytes makes the notebook reject a previously matching verification record. The notebook itself does not reconstruct models or recheck cache integrity. |

The generated-video test writes a short video, uses the actual OpenCV decoder,
checks the requested 64 source frames and the saved 32-frame pose indices, and
produces an alignment image. Its pixels and poses are synthetic. It validates
the decoder-to-artifact connection, not real GAVD pose quality or V-JEPA outputs.

Ordinary annotation/pose eligibility exclusions remain visible under the existing
rules. File discovery alone does not prove successful decoding. If required media
cannot be restored, a smaller cohort requires a separate documented amendment;
the software must not quietly choose that smaller experiment.

## Scientific and compatibility boundaries

The protocol remains `source-learning-curve-v1`. The joint model, preprocessing,
feature construction, controls, penalty grid, nested source partitions, source
weights, target units and paired bootstrap calculations are unchanged. Penalties
remain `lambda = training_window_count × rho`, so increasing the number of
training windows does not silently weaken a fixed numerical penalty.

The availability guard and Slurm entry-point changes alter the recorded software
fingerprint. Use a **fresh run root** for this implementation. A new root changes
neither the exposure history nor the development status of these recordings.
Older frozen studies remain readable; their saved software hashes are not edited.
The historical FI implementation and scaling numerical modules are unchanged.
Stage logs use `source-curve-stage-execution-v1` and belong to the child run.

## What to run on HAIC

Use the path block in the [short execution guide](../../../slurm/future-innovation/SOURCE_LEARNING_CURVE.md).
`FI_PARENT_ROOT` points to the completed gate-v2 directory;
`FI_RUN_ROOT` points to a new sibling study directory. With those variables set:

```bash
bash slurm/future-innovation-scaling/launch/submit.sh all
```

The dependency chain is:

1. Validate the original parent, run the fixed synthetic calibration and freeze
   the source reservation using metadata and known exposure records.
2. Discover the required full recordings, construct aligned windows and extract
   prefix poses. Save actual eligibility and exclusion records.
3. Encode newly eligible development windows with V-JEPA on the GPU. Reuse
   original cached windows with their original bindings, run the new teacher
   audits and freeze the training-subset plan.
4. Fit the repeated nested subsets on CPUs, comparing real history, shuffle,
   mismatch and validity-preserving no-skeleton with the shared RGB reference.
5. Save scores, paired uncertainty and the development decision; independently
   reconstruct the numerical results.
6. Execute notebook 23, embedding the saved tables and curve. This final job also
   runs after failed predecessors finish, so it can explain incomplete evidence.

New arrays go under `$FI_RUN_ROOT/data/teacher-cache/`; reused original arrays stay
under `$FI_PARENT_ROOT/teacher-cache/`. The new cache index records their origins
and bindings. Results go under `$FI_RUN_ROOT/reports/`, stage attempts under
`logs/stages/`, and each executed notebook under a new `notebook_runs/<batch>/`.
The older Experiment 0 notebooks and bridge notebooks 19–22 describe separate
experiments; notebook 23 is the source-learning-curve report.

Use `submit.sh status` to inspect progress and `submit.sh notebooks` to regenerate
only the inspection notebook. Restore missing input files and rerun `all` with
the same frozen code and paths to resume. Use `fit` when preparation, encoding,
audits and the plan are already complete. A killed process can leave a `running`
attempt record; Slurm's final state resolves that case. Logs are operational
evidence and do not replace artifact checks or a scientific decision.

## Local validation and measured limits

The fixed calibration ran **80 distinct synthetic source-subset fits**, including
production nested selection, checkpoint reload and paired source scoring. All
six prospective criteria passed. The values below are software calibration,
not a real-data learning curve:

| Synthetic comparison | 40 training recordings | All 64 training recordings |
|---|---:|---:|
| Mean real-minus-no-skeleton, three RGB-only draws | −0.00018659 R² | −0.00010219 R² |
| Planted temporal real-minus-no-skeleton | +0.742955 R² | +0.749573 R² |
| Planted real-minus-shuffle | +0.751649 R² | +0.750505 R² |

The seeds, fixtures, thresholds and full-precision results are in the
[calibration record](../../../work/artifacts/source-learning-curve-real-data-enablement-20260912/calibration/calibration.json).
The full regression run discovered 171 tests: **169 passed and two were skipped**
because the reviewed V-JEPA source checkout was not supplied through
`FI_TEST_VJEPA_ROOT`. A subsequent run of the six stage-execution tests also
passed, including an added test of every job's command routing and failure
propagation. Python parsing, shell syntax and local documentation links passed.
The [validation record](../../../work/artifacts/source-learning-curve-real-data-enablement-20260912/validation.json)
links the command logs and confirms unchanged parent file hashes, sizes and
modification times. The retained study configuration and cohort audit still
pass their original integrity checks.

The new [executed notebook 23](../../../outputs/future-innovation-source-curve-dev-20260911-v2/notebook_runs/manual-20260912T193406-e6c6f7ba/23_source_learning_curves.ipynb)
ran all five code cells against the retained local inventory and correctly shows
that expanded processing and the learning-curve report are absent. It left its
displayed inputs unchanged and performed no numerical reconstruction.

Commands exercised locally:

```bash
.venv/bin/python scripts/research_directions/future_innovation/calibrate_source_learning_curve.py \
  --output-root work/artifacts/source-learning-curve-real-data-enablement-20260912/calibration

FI_LAUNCH_TEST_CALIBRATION=work/artifacts/source-learning-curve-real-data-enablement-20260912/calibration/calibration.json \
  .venv/bin/python -m unittest discover -s tests -p 'test_future_innovation_*.py' -v

.venv/bin/python slurm/future-innovation-scaling/launch/notebooks.py \
  --run-root outputs/future-innovation-source-curve-dev-20260911-v2
```

The local preflight was also exercised with the real copied parent and a proposed
new output directory. It returned exit code **2** at the missing
`data/gavd_full/annotations/GAVD/data/GAVD_Clinical_Annotations_1.csv` and created
no study directory. The [saved output](../../../work/artifacts/source-learning-curve-real-data-enablement-20260912/local-input-check.log)
is an input failure, not a negative scientific result. Initial sandboxed notebook
tests could not bind localhost ports; the permitted local-kernel rerun passed.
No HAIC job was submitted, and GPU runtime, expanded pose eligibility and actual
learning-curve effects remain unmeasured in this session.

The intended conclusion still depends on **real-minus-no-skeleton across training
sizes**, alongside RGB, shuffle and mismatch comparisons. Better absolute scores
alone would not establish an additional skeleton contribution. A growing matched
increment could motivate independent-source confirmation; a flat increment would
motivate a separate target or representation study. Neither outcome can be
inferred from the calibration or from successful notebook execution.
