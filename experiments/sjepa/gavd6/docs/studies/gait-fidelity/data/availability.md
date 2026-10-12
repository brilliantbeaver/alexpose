# Dataset availability and the active run

[Proposal](../README.md#3-data-and-reference-measurements) · [Data specification](README.md) · [HAIC instructions](../../../../slurm/gait-fidelity/JEPA_RESPONSE.md)

**Revised 23 September 2026.** HAIC holdings below are based on the user's reported environment and run history. The local-video inventory was directly audited on the Mac. This documentation update performed no new remote filesystem audit.

The active follow-up requires the **completed prepared bundle and baseline artifacts from `walking-core-01`**. It does not require a new AMASS download, new GAVD extraction or an optional clinical dataset. `setup-followup` reads the actual bundle path from the parent ledger; do not guess it from a raw-data variable.

## Already provisioned in your HAIC setup

`$GAVD6_ROOT` is `/hai/scratch/$USER/alexpose/experiments/sjepa/gavd6` in the supplied environment.

| Data or asset | Existing location | Role now |
| --- | --- | --- |
| Prepared Gait Fidelity cohort | Bundle named in `$GAVD6_ROOT/outputs/gait-fidelity/walking-core-01/ledger.json` | Required read-only child dependency; includes retained observations, references and split metadata |
| Parent checkpoints, predictions and receipts | Under the same `walking-core-01` directory | Required completed comparisons; setup verifies their hashes and actual schedule |
| AMASS extracted motion files | `$ST_AMASS_ROOT` or `$GAVD6_ROOT/data/amass/extracted` | Raw source used by core preparation; the child consumes prepared arrays instead |
| Full GAVD videos, approximately 1,800 sequences | `$ST_GAVD_VIDEO_ROOT` or `$GAVD6_ROOT/data/gavd_full/youtube/all` | Reported available; unused in this follow-up |
| COCO images and annotations | `/hai/scratch/$USER/coco/train2017` and `/hai/scratch/$USER/coco/annotations/person_keypoints_train2017.json` | Available for optional image adaptation, outside the current comparison |
| Synthetic-training-v2 data/results | `$GAVD6_ROOT/outputs/synthetic-training-v2/` | Historical motivation and provenance; not the child training-bundle interface |
| Body models and DMPLs | `/hai/scratch/$USER/body_models`, with `dmpls` below it | Existing rendering assets used by the parent; no new child rendering |
| Textures, UV map and backgrounds | `$ST_MODEL_ROOT/synthetic-rendering/` | Existing rendering assets; availability of backgrounds does not certify completeness of COCO |
| Pose-estimator and other checkpoints | `$ST_MODEL_ROOT`, ordinarily `/hai/scratch/$USER/models` | Exact estimator identities come from the parent configuration/receipts |

A root path establishes an intended location; the saved manifest and file checks establish the inputs actually consumed. AMASS_ROOT commonly names the dataset directory, whereas the motion-reader path points to its `extracted` files. This distinction is already resolved in the parent's saved configuration and is not reset for the child.

## Verified locally; exact HAIC copy unconfirmed

The 91 MS/PD/Normal clips are at `/Users/theodoremui/dev/alexpose/experiments/multiple-sclerosis/video-data-full`. The [dated audit](local-videos.md) checked files, timestamps and 88 historical pose caches. The [gallery](video-gallery.html) opens the local clips for review.

Shared source IDs with GAVD do not establish identical cuts, annotations or cache availability on HAIC. This collection is deferred development material, not an input to `jepa-response-01` and not independent external confirmation.

## Optional acquisitions, outside this run

| Candidate | Possible later use | Unresolved prerequisites |
| --- | --- | --- |
| Stroke full-body motion capture | Recorded asymmetric motion for a separate rendering/reference study | Acquire and inspect raw trajectories; validate retargeting and anatomical metadata |
| LIVE-GaitNeuroKids | Clinical video compared with independent movement measurements | Acquire data; check same-trial matching, synchronization, coverage and permitted use |
| Full MoVi synchronized video/reference release | Healthy real-image trajectory evaluation | Acquire appropriate modalities; verify calibration and AMASS identity overlap |
| ProGait and Parkinson turning/freezing collections | Separately defined clinical stress tests | Verify access and endpoint-specific references; do not treat task labels as dense trajectories |

No usable HAIC copy of these releases has been established for this study. Their descriptions in the [candidate audit](clinical-candidates.md) are dated feasibility assessments, not evidence of acquisition. AMASS motion derived from a collection does not establish possession of its full synchronized video release.

## What must finish before the child starts

The core must complete preparation, profiling, all registered fitting phases and evaluation. Child setup verifies its configuration, code, array hashes, admission amendment and completed receipts. The child then produces its own calibration receipt, profile decision, nine final models, diagnostics and paired evaluation. Local CPU validation has completed; actual H100 compatibility and cost remain subject to HAIC profiling.
