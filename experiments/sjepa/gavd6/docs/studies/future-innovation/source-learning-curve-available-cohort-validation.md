# Available-video source learning curve: implementation and validation

The source learning-curve launcher now freezes a processing cohort from **GAVD
manifest recordings whose files are available**. It lists missing, ambiguous,
empty or unreadable recordings as exclusions and continues with available
development recordings. Confirmation recordings remain reserved. The change is
implemented and locally tested; no job was submitted to HAIC in this session.

The [execution guide](../../../slurm/future-innovation/SOURCE_LEARNING_CURVE.md)
contains the normal path setup and submission commands. The
[cohort amendment](source-learning-curve-available-cohort-protocol.md) defines
`available-development-v1`, including its effect on scientific interpretation.
The existing `source-learning-curve-v1` model and numerical protocol remain in
place. No expanded real-data result is claimed by this repair.

## Why the previous command stopped

The full manifests describe annotated recordings, including recordings that
were not successfully downloaded. The previous launcher required every planned
development recording to be present. Its error arose in the media preflight,
before pose extraction or teacher encoding. Having 334 files in a directory
does not establish that all 305 planned development IDs are among those files;
some available recordings may be reserved for confirmation.

The new policy separates the complete reservation from the processing inputs.
It assigns development/confirmation roles and outer folds using the complete
metadata first, then selects available development recordings. Consequently,
availability cannot move a reserved recording into development or reassign a
recording to another fold. All sequences belonging to an included recording
remain candidates for the existing alignment and pose checks.

## What changed and how it was checked

| Failure or risk | Implementation | Verification |
|---|---|---|
| A missing development recording aborts the whole launch | Resolve manifest IDs before processing and freeze explicit inclusion/exclusion flags | A fixed fixture with 348 manifest recordings, 334 available files and 305 planned development recordings includes 293 development recordings and excludes the missing 12 |
| Unrelated files or confirmation recordings enter processing | Derive processing manifests from the intersection of available files and development IDs | An extra unregistered file is ignored; all 43 confirmation IDs in the fixture remain excluded |
| Copying a manifest breaks its relative video paths | Resolve paths relative to the original manifest, then save one absolute path per included recording | Relative explicit paths and duplicate links are tested; adding a different export after freezing does not change the chosen file |
| An unavailable source returns through the old candidate or cache manifest | Filter inherited candidate comparisons and cache reuse to included sequences; replace inherited display video paths with the selected current path | Production candidate construction, preparation, cache writing and reload process 54 fixture windows: 53 new stand-in encodings and one included parent-cache reuse; the unavailable parent's window is excluded |
| A retry silently changes the experiment | Seal processing manifests and source availability; check selected file size and modification time before decoding | Later downloads do not add members; disappearance or modification of an included file is rejected; completed preparation and cache stages reuse their outputs |
| Editing checksums hides a wrong cohort | Recompute the relationship between reservation, inclusion masks, processing manifests and cache cohort identities | Rehashed confirmation inclusion and reintroduction of an omitted source are rejected |
| Initialization crashes halfway through publishing its files | Retain the existing staged publication, locks and sealed recovery | An injected crash after publishing configuration recovers successfully against the verified real parent cache, without changing the reservation or parent files |
| Notebook output conceals exclusions or implies a new result | Display the cohort policy, availability counts and excluded IDs separately from processing and model results | Notebook tests cover availability displays, incomplete results, failure tracebacks and immutable execution batches |

The 348/334/305 counts above form a deliberately constructed regression fixture,
not a measurement of the current HAIC filesystem. The separate integration test
uses the retained local manifests and exposure inventory: their reservation has
304 development recordings and 44 confirmation recordings. Reproducing the
parent's saved availability with presence-only fixture files includes 292
development recordings. That test checks reservation preservation and recovery;
its placeholder files are never decoded or treated as real videos. HAIC's actual
count is determined by its manifests, exposure inventories and files when the
new study is frozen.

`config/processing-sequences.csv` and `config/processing-videos.csv` are the
executable processing inputs. `config/full-*.csv`, the original reservation and
`config/media-availability.csv` retain excluded identities for explanation and
provenance. Every missing ID remains visible in the audit; it is absent from the
active candidate and expanded cache cohort. The original parent cache remains
read-only and retains its original embedded bindings.

## Commands actually exercised locally

All commands below were run from the repository root on September 12, 2026.
The [machine-readable validation record](../../../work/artifacts/source-learning-curve-available-cohort-20260912/validation.json)
links their artifacts and records the code fingerprint.

```bash
.venv/bin/python scripts/research_directions/future_innovation/calibrate_source_learning_curve.py \
  --output-root work/artifacts/source-learning-curve-available-cohort-20260912/calibration

FI_LAUNCH_TEST_CALIBRATION=work/artifacts/source-learning-curve-available-cohort-20260912/calibration/calibration.json \
  .venv/bin/python -m unittest discover -s tests -p 'test_future_innovation_*.py' -v

.venv/bin/python slurm/future-innovation-scaling/launch/notebooks.py \
  --run-root outputs/future-innovation-source-curve-dev-20260911-v2
```

| Check | Measured outcome |
|---|---|
| Complete future-innovation regression suite | 180 tests discovered: **178 passed, 2 skipped**, in 64.489 seconds |
| Available-cohort tests | All seven passed, including the production preparation/cache integration fixture |
| Synthetic source learning-curve calibration | All six criteria passed over 80 distinct fitted subsets; fixed RGB-only and planted temporal fixtures retain the previous behavior |
| Public shell launcher | Read-only preview, six-job submission chain with a scheduler stand-in, environment propagation and dependency routing passed |
| Syntax | All 12 changed/new Python files parsed; `bash -n` passed for all nine scaling shell and Slurm files |
| Retained-study notebook inspection | Five code cells executed; no error outputs; displayed artifact snapshots unchanged |
| Parent preservation | All 337 parent files retained identical SHA-256 hashes, sizes, modification times and inventory compared with the prior saved snapshot |
| Historical study compatibility | Existing study configuration and cohort-audit seals remain readable and valid |

The [full test log](../../../work/artifacts/source-learning-curve-available-cohort-20260912/all-future-innovation-tests.log)
records the individual results. The two skipped tests require the reviewed
V-JEPA source through `FI_TEST_VJEPA_ROOT`; that source was unavailable locally.
The [calibration result](../../../work/artifacts/source-learning-curve-available-cohort-20260912/calibration/calibration.json)
is synthetic software evidence, not a scientific result on GAVD.

The separately executed
[notebook 23](../../../outputs/future-innovation-source-curve-dev-20260911-v2/notebook_runs/manual-20260912T221427-a0d77ebc/23_source_learning_curves.ipynb)
inspects the retained historical study. It correctly identifies its historical
cohort policy and the absence of expanded results. It does not retrofit the new
policy into that study or claim numerical verification. Initial notebook tests
could not start kernels because the local sandbox denied localhost ports; the
permitted rerun and full suite passed. This was a local execution restriction,
not a HAIC result.

## Running the amended study on HAIC

After transferring the updated code, use the environment setup in the execution
guide. Keep `FI_PARENT_ROOT` pointing to the completed gate-v2 run. Set
`FI_RUN_ROOT` to a separate study directory and run:

```bash
export FI_VIDEO_ROOT="$GAVD_FULL_ROOT/youtube/all"
bash slurm/future-innovation-scaling/launch/submit.sh check
bash slurm/future-innovation-scaling/launch/submit.sh all
```

`check` previews counts without writing a run or submitting jobs. `all` performs
initialization, preparation, teacher encoding, fitting, reporting and notebook
execution. Missing files at initial selection no longer block it. If the earlier
attempt stopped at preflight before freezing anything, its intended run path can
be reused. If a study was already frozen under the previous implementation,
choose a fresh directory. Subsequent retries use the same directory and code.

This change does not suppress errors for an incorrect model, broken annotation
binding, failed frame decoding, insufficient eligible sources, corrupt cache or
a selected file that later disappears. Those checks protect the interpretation
of the experiment. Availability filtering also changes the population being
studied: conclusions apply to eligible available development recordings.
HAIC execution, actual video decoding and expanded teacher encoding remain the
required next checks before claiming an expanded learning-curve result.
