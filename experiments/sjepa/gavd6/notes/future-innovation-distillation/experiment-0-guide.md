# Experiment 0: does skeleton history help predict the future?

**Historical protocol:** this document describes the original five-arm gate.
For new runs, use the [direct-v2 50-clip gate specification](../../docs/studies/future-innovation/direct-gate-protocol.md).
Direct-v2 removes background-selectivity prerequisites and explicitly tests the
increment over a matched no-skeleton head. Existing legacy runs retain their
frozen rules. Full GAVD is reserved for the subsequent real experiment.

**Status reconciled 10 September 2026:** the feasibility pipeline is implemented.
This checkout has local alignment overlays but no completed gate report; it
does not establish the current remote job state. Start with the
[study overview and five executable tutorials](../../docs/studies/future-innovation/).
Use the [HAIC guide](../../slurm/future-innovation/README.md) for real execution
and recovery. This document preserves the detailed design and implementation
sketches; saved run contracts and production code define actual execution.

Before training a skeleton model to learn from a large video model, we need to know whether the skeleton contains a useful part of what we want it to learn. This guide turns the [48-hour gate in Proposal 2](../world-model-extensions/proposals-03/02-future-innovation-distillation.md#the-48-hour-decision-gate) into that first test:

> After a predictor has seen the recent video and the recording details, does adding the person's skeleton history help it predict the video model's description of a future moment?

We test 50 short video windows. A useful result must survive two simple challenges: scramble the skeleton's timing, and give the predictor a skeleton from the wrong video. If these alternatives work just as well, we have not established the value of correctly paired motion.

### How to read this guide

Start with [the comparison and a worked example](#1-understand-the-comparison). Then follow the build sequence. Each stage explains its purpose, the implementation, and what must be true before moving on.

| Stage | Question it answers | What you produce |
| --- | --- | --- |
| [2. Freeze the setup](#2-freeze-the-setup) | Will every comparison use the same rules? | Versioned configuration and teacher |
| [3. Build 50 aligned examples](#3-build-50-aligned-examples) | Do the video, boxes, and skeleton describe the same frames? | A fixed cohort and alignment overlays |
| [4. Create inputs and targets](#4-create-inputs-and-targets) | What may the predictor see, and what must it predict? | Saved video features |
| [5. Check validity](#5-check-validity-before-training) | Is the future hidden, and does the target respond to the person? | Leakage and target-sensitivity reports |
| [6. Fit the comparisons](#6-fit-the-comparisons) | Does real skeleton history help on unseen sources? | Predictions for every model and control |
| [7. Score and decide](#7-score-the-predictions-and-decide) | Is the gain large enough and stable enough to continue? | `ADVANCE`, `STOP`, or `INCONCLUSIVE` |

The [48-hour schedule](#appendix-b-the-48-hour-schedule) is an allocation of work, not a reason to skip a failed check. The appendices collect [implementation and tests](#appendix-a-implementation-and-tests), [failure interpretations](#appendix-c-what-a-failure-means), and [the files to save](#appendix-d-final-artifact-checklist).

## 1. Understand the comparison

### 1.1 Follow one walking clip

Imagine a clip of a person walking. We reveal frames 0–31 and ask the predictor about a later moment, at frames 38–39. The target ends eight frames after the last observed frame.

There are three pieces of information:

| Piece | What it contains | Role |
| --- | --- | --- |
| Recent video and recording details, `x` | Features of RGB frames 0–31, plus camera, box, timing, and tracking-quality information | Input to both predictors |
| Skeleton history, `s` | Body-landmark positions, confidence, and validity over frames 0–31 | Extra input to the skeleton correction |
| Future description, `y` | A fixed video model's numerical description at the future person region, reduced to 256 numbers | The answer used for training and scoring |

RGB is the ordinary color video. A **representation**, also called a feature vector or latent, is an array of numbers a model uses to describe its input. Here, `y` is such an array. We predict those numbers; we do not generate future pixels or predict a diagnosis.

The fixed video model is **V-JEPA 2.1**, the teacher. Its encoder turns video into features. **Frozen** means its weights never change during this experiment. Only the small predictors described below are trained.

Throughout this guide, “current RGB” means the **whole observed 32-frame context**, including visible motion. The baseline already sees that context. The experiment measures the added usefulness of an explicit skeleton representation under these predictors; it does not claim that a skeleton reveals something absent from the underlying RGB.

### 1.2 Predict first, then learn a correction

First, fit the **baseline**: a predictor that uses `x` alone. Recording details in `x` are called **nuisance variables** because they might explain a result without requiring the body-motion signal we want to study. For example, the background might identify a familiar source video.

Next, calculate what the baseline missed. This error is the **residual**, called **future innovation** in the proposal. “Innovation” here means the part left unexplained by the baseline.

Finally, train a small **residual head** to predict a correction using skeleton history and the same baseline inputs. A head is simply a small trainable model. In notation and code:

```python
# Conceptual pseudocode: each example has a 256-number target y.
baseline_prediction = baseline(x)
residual_to_learn = y - baseline_prediction       # During training only.
predicted_correction = head(s, x)
full_prediction = baseline_prediction + predicted_correction
```

For one imaginary feature, if the true value is `0.8` and the baseline predicts `0.5`, the residual is `0.3`. A skeleton correction of `0.2` moves the prediction to `0.7`. The real experiment performs this comparison across 256 features and all held-out windows. These numbers are illustrative, not results.

The distinction between the answer and the inputs is crucial:

```text
ALLOWED WHEN MAKING A PREDICTION

past RGB + recording details ──> baseline ────────────────┐
                                                        + ──> full prediction
past RGB features + recording details + skeleton ──> head┘

ANSWER USED TO TRAIN OR SCORE THE PREDICTION

full clip ──> frozen teacher ──> future target y
                                Compare with predictions;
                                never feed y into either predictor.
```

We can read an answer to compute a training loss or a test score. Feeding that answer into the predictor would be **leakage**. Later checks verify that no future pixels enter even indirectly through feature extraction or preprocessing.

### 1.3 What counts as an improvement?

We evaluate on entire source videos excluded from fitting. A **source video** is the original uploaded video; a **window** is one 64-frame excerpt. Holding out sources prevents adjacent excerpts from the same recording from appearing on both sides of the test.

Three quantities summarize prediction quality:

| Quantity | Plain-language meaning | Illustrative value |
| --- | --- | --- |
| \(R^2_{\mathrm{base}}\) | How well the baseline predicts, compared with always predicting the training mean | `0.60` |
| \(\Delta R^2 = R^2_{\mathrm{full}} - R^2_{\mathrm{base}}\) | How much the skeleton correction improves that score | `0.66 - 0.60 = 0.06` |
| \(F_8 = \Delta R^2 / (1 - R^2_{\mathrm{base}})\) | The fraction of the baseline's remaining error recovered at the 8-frame horizon | `0.06 / 0.40 = 0.15`, or 15% |

The required gain of `0.05` is an absolute increase in \(R^2\), not a 5% relative improvement and not classification accuracy. Higher \(R^2\) is better; zero matches the training-mean reference, and negative values mean worse predictions than that reference. Report negative gains too.

The residual can contain body motion, appearance changes, camera effects, and noise. Calling it “innovation” does not establish that all of it is useful or skeleton-predictable. That is why we need controls.

### 1.4 What the controls ask

An **arm** is one version of the experiment. A control changes one relevant part while keeping the comparison otherwise matched.

| Comparison | What changes? | Why it matters |
| --- | --- | --- |
| Real skeleton | Nothing: correct history for the video | Measures the proposed gain |
| Time shuffle | Reorder short blocks of the same skeleton history | Tests whether the original order matters |
| Clip mismatch | Substitute a similar skeleton history from another source | Tests whether the skeleton belongs to this example |
| Background target | Predict teacher features outside the person region | Tests whether the gain is concentrated in the person region |
| No-skeleton capacity control | Remove coordinates but retain validity information in the same-size head | Tests whether extra model capacity or tracking quality explains the gain |

Separately, before any fitting, edit the future pixels and check the teacher. Future edits must leave the past input unchanged; person edits should affect the future target more than background edits. These are tests of the measurement itself.

### 1.5 The pass rules, in one place

**Preregister** means write down the rules before inspecting results. The first two numerical rules below come from the proposal. The other values are this guide's recommended operational definitions of its qualitative checks. Freeze them in `config/thresholds.json` before feature extraction; if you choose different values, update the proposal contract at the same time.

| Check | Required result |
| --- | --- |
| Real gain | \(\Delta R^2_{\mathrm{real}} \geq 0.05\) |
| Timing matters | Real gain is at least \(2\max(\Delta R^2_{\mathrm{shuffle}}, 0)\) |
| Correct pairing matters | Mismatched gain is at most `0.01` |
| Gain is concentrated in person tokens | Using the background target reduces the positive real-skeleton gain by at least 50% |
| Target responds preferentially to the person | Median target change from person edits is at least twice that from background replacement; person edits have the larger effect in at least 8 of 10 audit windows |
| Measurement is valid | Complete cohort and controls; aligned inputs; stable teacher; no future leakage; valid source splits and metrics |
| Result is stable | Real gain is positive in at least 90% of source-bootstrap resamples and every planned initialization; a threshold reached only for a lucky initialization is inconclusive |

The implementation uses the arithmetic mean of seed scores for point checks
and also reports every seed. The no-skeleton arm is a required diagnostic with
paired contrasts and uncertainty, but the implemented gate has no separate
cutoff for it. If it reproduces the apparent gain, state that motion attribution
is unresolved even if the automatic gate returns `ADVANCE`; do not rewrite the
saved decision or invent a cutoff after seeing results. The pseudocode later
in this design document is not a replacement for the production gate.

`ADVANCE` means the valid, stable gate supports the next measurement experiment.
`STOP` means a required validity or point-threshold check failed, or measurement
is incomplete; inspect `measurement_complete` and the failed checks.
`INCONCLUSIVE` means the point checks pass but source-bootstrap or seed
stability fails. Interpret the reported capacity control alongside the saved
automatic decision without assigning it an undeclared cutoff.

This first gate uses 50 windows, the 8-frame horizon, and raw whole-body skeletons. It does not train S-JEPA, adapt V-JEPA, compare all three horizons, or train a skeleton-only student. An advance permits those next measurement steps; adapter training still depends on the later frozen-S-JEPA result. No outcome here establishes a publication or clinical claim.

## 2. Freeze the setup

The purpose of this stage is to make one reproducible experiment: the same data selection, teacher, preprocessing, and decision rules for every arm. A **contract** below is simply a saved configuration recording those choices. A **cache** stores already-computed features so all arms reuse the same numbers.

### 2.1 Check what exists before planning jobs

The repository already contains:

- `manifests/gavd/gavd_full_sequences.csv`: 1,874 annotated sequences from 348 source videos;
- `manifests/gavd/gavd_full_videos.csv`: the one-row-per-source inventory;
- `gavd6 gavd download`: a resumable full-source video downloader and decoder audit;
- historical MediaPipe pose files for a much smaller 96-sequence cohort; and
- frozen local S-JEPA checkpoints.

At the time this protocol was drafted, the repository did **not** yet contain:

- a maintained full-GAVD pose-extraction command;
- a V-JEPA dependency or checkpoint;
- future-innovation feature extraction;
- the temporal residual head;
- the gate metrics and decision code; or
- `gavd6 future-innovation ...` CLI routes.

These components are now implemented under `research_directions/future_innovation/`, with lazy `gavd6 future-innovation` commands and jobs under `slurm/future-innovation/`. The sketches below describe the scientific contract; use the linked run guide and command-specific `--help` for the implemented interfaces.

The 96-sequence local pose cache is useful for testing array schemas, but it must not be described as the full GAVD cohort. The 48-hour gate should use the full-GAVD source manifest and the videos cached on HAIC.

The Python blocks below are implementation sketches. Some define complete helper functions; others depend on arrays or adapters introduced in the surrounding text. They are not a single script to run from top to bottom. The shell setup assumes HAIC, the compute environment holding the source videos.

### 2.2 Create a versioned HAIC run root

Keep checkpoint downloads and teacher tensors outside the Git checkout. The full-GAVD source cache and annotation checkout live under `gavd6/data/gavd_full`. Use explicit HAIC paths:

```bash
export SJEPA_ROOT="/hai/scratch/$USER/alexpose/experiments/sjepa"
export GAVD6_ROOT="$SJEPA_ROOT/gavd6"
export GAVD_FULL_ROOT="$GAVD6_ROOT/data/gavd_full"
export VJEPA2_ROOT="/hai/scratch/$USER/vendor/vjepa2"
export FI_TEACHER_CHECKPOINT="/hai/scratch/$USER/models/vjepa2_1_vitb_dist_vitG_384.pt"
export FI_POSE_MODEL="/hai/scratch/$USER/models/pose_landmarker_lite.task"
export FI_ANNOTATION_ROOT="$GAVD_FULL_ROOT/annotations/GAVD/data"
export FUTURE_INNOVATION_ROOT="/hai/scratch/$USER/experiments/future-innovation"
export FI_RUN_ID="gate-v1"
export FI_RUN_ROOT="$FUTURE_INNOVATION_ROOT/$FI_RUN_ID"
```

Validate inputs before creating model outputs:

```bash
test -d "$GAVD6_ROOT"
test -d "$GAVD_FULL_ROOT/youtube/all"
test -f "$GAVD_FULL_ROOT/manifests/gavd_full_sequences.csv"
test -f "$GAVD_FULL_ROOT/manifests/gavd_full_videos.csv"
mkdir -p "$FI_RUN_ROOT"
cd "$GAVD6_ROOT"
```

Use this layout:

```text
$FI_RUN_ROOT/
├── config/
├── manifests/
├── poses/
├── teacher-cache/
├── models/
├── predictions/
├── qc/
├── reports/
└── logs/
```

Never overwrite `gate-v1` after inspecting its aggregate metrics. A code or protocol repair produces `gate-v2` with a written change reason.

### 2.3 Pin the teacher code

Use the V-JEPA 2.1 ViT-B/16 384-pixel model exposed through PyTorch Hub by the [official repository](https://github.com/facebookresearch/vjepa2). ViT-B is the base-size video transformer. The selected [model builder](https://github.com/facebookresearch/vjepa2/blob/main/src/hub/backbones.py) uses 64 frames, 16-pixel spatial patches, and two-frame temporal tubelets; [Stage 4](#4-create-inputs-and-targets) explains those units. Verify these settings against the commit you pin. We will enforce the past/future boundary explicitly rather than assume that a pretrained video model is causal.

Clone once, select a reviewed commit, and record it:

```bash
git clone https://github.com/facebookresearch/vjepa2.git "$VJEPA2_ROOT"
git -C "$VJEPA2_ROOT" rev-parse HEAD
```

Place the 40-character result in the study configuration and check it out explicitly on all jobs:

```bash
export VJEPA2_COMMIT="<reviewed-40-character-commit>"
git -C "$VJEPA2_ROOT" checkout "$VJEPA2_COMMIT"
test "$(git -C "$VJEPA2_ROOT" rev-parse HEAD)" = "$VJEPA2_COMMIT"
```

Do not use `main` as a reproducibility identifier.

### 2.4 Prepare the runtime environment

The GAVD6 project already pins PyTorch. The official V-JEPA README additionally requires `timm` and `einops`; its demonstration route uses a video decoder. Install these into the same locked environment only after reviewing the pinned repository requirements:

```bash
cd "$GAVD6_ROOT"
uv sync --frozen
uv pip install timm einops decord2
uv run --no-sync python -c "import cv2, einops, timm, torch; print(torch.__version__)"
```

Record `uv.lock`, `python --version`, the CUDA version, GPU model, and `pip freeze` in `config/environment.txt`. If the official repository requires a conflicting PyTorch version, create a separate V-JEPA feature-extraction environment and exchange only immutable `.npz` or Parquet artifacts. Do not silently change the S-JEPA project's PyTorch pin.

### 2.5 Load the teacher and record its fingerprint

Use the local pinned repository as the code source:

```python
from pathlib import Path
from hashlib import sha256
import torch


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


vjepa_root = Path("/absolute/path/to/pinned/vjepa2")
processor = torch.hub.load(
    str(vjepa_root),
    "vjepa2_preprocessor",
    source="local",
    crop_size=384,
)
encoder, predictor = torch.hub.load(
    str(vjepa_root),
    "vjepa2_1_vit_base_384",
    source="local",
    pretrained=True,
)
encoder.eval().requires_grad_(False)
predictor.eval().requires_grad_(False)
```

`eval()` turns off training-specific behavior, and `requires_grad_(False)` prevents weight updates. The hub returns both an encoder and a predictor; the recommended gate route uses the frozen encoder and trains its own small baseline. We load the predictor here to match the hub interface, not because the gate trains or needs it.

PyTorch Hub may download the checkpoint into its cache. Resolve the actual file path after the first load and store its SHA-256. This hash is a fingerprint of the file: it lets later jobs verify that they used exactly the same weights.

Wrap the hub interface in a project-owned **adapter**: a small wrapper that gives our code stable method names even if the external library changes. Test it against the pinned commit.

Before moving on, record the teacher commit and checkpoint hash, frame layout, crop policy, token layout, projection seed, thresholds, model settings, initialization seeds, and aggregation rule. These files are listed in [Appendix D](#appendix-d-final-artifact-checklist).

## 3. Build 50 aligned examples

The purpose of this stage is to make every training example mean the same thing: a known past, a specific future location, and a skeleton aligned to the same person and frames.

### 3.1 Mark the past and future on one timeline

Frame numbers in the table are **local, zero-based indices within a window**. Each window also has an absolute starting frame in its source video. Use one layout throughout:

| Frames | Role |
| --- | --- |
| `0..31` | Observed past: RGB and skeleton inputs |
| `32..37` | Unobserved gap: no predictor input |
| `38..39` | Future location whose person-region features we predict |
| `40..63` | Also hidden from the predictors; available only to the teacher's target calculation and pixel-edit audits |

The last observed frame is 31. The target ends at frame 39, so the **horizon**, or prediction distance, is `39 - 31 = 8` frames. The teacher groups pairs of frames into **tubelets**, so frames 38 and 39 share one time position in its output.

Keep all 64 frames to match the teacher layout and the later experiment's 16- and 32-frame horizons, ending at frames 47 and 63. The teacher computes the target from the full clip, so the target at frames 38–39 can incorporate surrounding frames, including later ones. The horizon names the target's location; it does **not** mean the target representation sees only through frame 39. The predictors, however, must see only through frame 31.

Represent the contract in code rather than repeating numbers:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class FrameContract:
    frames_per_clip: int = 64
    context_stop_exclusive: int = 32
    target_horizon_frames: int = 8
    target_tubelet_start: int = 38
    target_tubelet_stop_exclusive: int = 40
    resolution: int = 384
    patch_size: int = 16
    tubelet_size: int = 2


FRAME = FrameContract()
assert FRAME.context_stop_exclusive % FRAME.tubelet_size == 0
assert FRAME.target_tubelet_start % FRAME.tubelet_size == 0
assert FRAME.target_tubelet_stop_exclusive - FRAME.target_tubelet_start == 2
assert (
    FRAME.target_tubelet_stop_exclusive - 1
    - (FRAME.context_stop_exclusive - 1)
) == FRAME.target_horizon_frames
```

Frame rate changes the elapsed time: eight frames is about `0.27` seconds at 30 frames per second and `0.32` seconds at 25. Preserve decoded FPS in the nuisance vector and report `8 / FPS` for each source. Resampling to a common rate would be a different protocol decision and must be frozen explicitly.

### 3.2 Select 50 usable windows without presentation labels

For this gate, one **window** is one contiguous 64-frame excerpt drawn from one annotated GAVD sequence. The cohort contains exactly 50 windows.

Use these eligibility rules:

- the source video decodes through the window's final frame;
- the annotated sequence contains at least 64 consecutive source frames;
- a usable person box exists throughout the 32-frame context and target tubelet;
- a whole-body pose can be extracted for the context with explicit missingness;
- the person remains substantially inside the frozen V-JEPA crop; and
- no source contributes more than two windows.

The last rule forces at least 25 distinct source videos. Fifty windows from one recording would give us far less evidence about generalization than fifty windows spread across sources.

The pixel-edit audit later needs person boxes throughout frames `32..63`. Select its 10 windows before feature inspection and verify those boxes too. Save all exclusions and their reasons so a failed decode or missing track cannot silently change the cohort.

Do not use `gait_pattern_annotation`, `dataset_annotation`, or any presentation label to select folds or tune the model. Those columns may be retained only for a post-freeze coverage table.

### 3.3 Keep complete source videos together

A **fold** is one group of sources. Divide sources into five folds; eventually, each group takes a turn as the test set. Every window from a source stays in that source's fold. This tests new recordings, not necessarily new people: GAVD does not provide reliable participant identities.

Use a fixed seed and hashes to make the selection and fold assignments reproducible. A hash turns the same identifier into the same sortable value on every rerun:

```python
from hashlib import sha256
import pandas as pd


SELECTION_SEED = 260905


def stable_key(*values: object) -> str:
    payload = ":".join(map(str, (SELECTION_SEED, *values)))
    return sha256(payload.encode("utf-8")).hexdigest()


def assign_source_folds(video_ids, n_folds=5):
    ordered = sorted(set(map(str, video_ids)), key=lambda value: stable_key("fold", value))
    return {video_id: index % n_folds for index, video_id in enumerate(ordered)}


def deterministic_window_start(first_frame: int, last_frame: int, sequence_id: str) -> int:
    latest = last_frame - 63
    if latest < first_frame:
        raise ValueError("sequence is shorter than 64 frames")
    choices = latest - first_frame + 1
    offset = int(stable_key("window", sequence_id)[:16], 16) % choices
    return first_frame + offset
```

Fix the candidate ordering by stable hashes, then establish eligibility through the decoding and alignment checks below. From eligible candidates, take at most two windows per source until 50 are selected. Assign all windows from the same `video_id` to the same fold. The final manifest is frozen at the end of this stage.

In each of the five rounds, roughly 40 windows are available for fitting and 10 for testing. Every window eventually receives one prediction from a model that never fitted its source. This is **out-of-fold**, or OOF, prediction. Fifty windows still make a small pilot: its purpose is to decide whether a larger experiment is justified.

### 3.4 Decode the frames once

Decode by absolute source-frame index because GAVD annotations refer to the source video. The array shape `[64, H, W, 3]` means 64 frames, image height, image width, and three color channels. Keep raw frames outside Git.

```python
import cv2
import numpy as np


def decode_exact_window(video_path, first_frame, count=64):
    capture = cv2.VideoCapture(str(video_path))
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(first_frame))
    frames = []
    for expected in range(count):
        ok, bgr = capture.read()
        if not ok:
            capture.release()
            raise ValueError(f"decode failed at local frame {expected}")
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.release()
    if fps <= 0:
        raise ValueError("video reports non-positive FPS")
    return np.stack(frames), fps
```

Confirm whether the manifest uses one-based or zero-based frame numbers before calling this function. Perform that conversion once in the manifest builder and record it; do not scatter `-1` corrections through the pipeline.

### 3.5 Keep the person box aligned after resizing

A **person box** is the rectangle marking the person in an image. Resizing or cropping the image moves that rectangle. Apply exactly the same geometry to the box; otherwise we could accidentally ask the teacher for features from the wall beside the person. Mirror the pinned preprocessor's geometry in a small project-owned function and verify pixel equivalence on a test clip.

For each frame, save the model-space box as normalized `(x0, y0, x1, y1)` coordinates. Reject or flag a window if less than 90% of the annotated person-box area survives the frozen crop in any target frame. This avoids calling a background-only crop a person-region target.

Do not adopt an unmerged “preserve border” preprocessing change halfway through the run. Crop policy is part of the teacher contract.

### 3.6 Represent the skeleton and its missing points

Use MediaPipe's 33 body landmarks for frames `0..31`. Save at least:

```python
@dataclass(frozen=True)
class SkeletonHistory:
    xy_box_normalized: np.ndarray  # [32, 33, 2]
    confidence: np.ndarray         # [32, 33, 1]
    valid: np.ndarray              # [32, 33, 1], bool
```

Normalize `x` and `y` relative to the person box, then center on the mid-hip and divide by a robust body scale estimated from visible context frames. Keep the original box position, scale, and centroid motion in the nuisance vector.

This keeps body shape and movement in the skeleton while placing image location and overall scale in the baseline. The comparison can then ask what skeleton structure adds after those easier clues have been supplied to both models.

The four channels for each landmark are `x`, `y`, confidence, and validity. A validity value tells the head whether a point was actually observed. Missing coordinates may be filled with zero only after centering and scaling, with confidence and validity retained; otherwise “missing” becomes indistinguishable from a real point at the origin. Never fill a past landmark using future frames.

### 3.7 Check the alignment with overlays

Before running V-JEPA, render at least one window from every source fold with:

- source-frame number;
- annotated person box;
- MediaPipe landmarks;
- the observed/future boundary after frame 31; and
- the target frames 38 and 39 highlighted.

A one-frame RGB–pose offset can hide a real effect, while an offset that varies by source can create a misleading one. The pipeline checks exact decoding, crop retention, and landmark validity before freezing the cohort, and saves labeled overlays and pose arrays for optional diagnosis.

### 3.8 Freeze the final cohort manifest

A **manifest** is the table linking each example to its source files and fold. Once eligibility and alignment are established, save these fields:

```text
window_id, sequence_id, video_id, video_path,
source_first_frame, source_last_frame, decoded_fps,
source_width, source_height, outer_fold,
box_path, pose_path, eligibility_reason
```

Check the final selection:

```python
assert len(cohort) == 50
assert cohort["window_id"].is_unique
assert cohort["video_id"].nunique() >= 25
assert cohort.groupby("video_id").size().max() <= 2
assert cohort.groupby("video_id")["outer_fold"].nunique().max() == 1
assert set(cohort["outer_fold"]) == set(range(5))
```

Write `manifests/gate-windows.csv`, hash it, and place the hash in `config/run-contract.json`. Before moving on, all 50 windows must satisfy the data contract. Never regenerate a more favorable sample after viewing features or scores.

## 4. Create inputs and targets

The purpose of this stage is to turn each video into two separate objects: features the predictors are allowed to use and an answer they must predict. The expensive frozen-teacher computations happen once and are saved for all model arms.

### 4.1 Understand patches, tubelets, and tokens

The teacher divides a video into small blocks. A **spatial patch** covers a small rectangle of pixels; a **tubelet** extends a patch across two frames. The numerical feature vector representing one such block is a **token**. Later transformer layers let tokens exchange information with one another.

At 384-pixel resolution with 16-pixel patches, there are `384 / 16 = 24` patches along each image axis. With two-frame tubelets, there are `64 / 2 = 32` time positions: `32 × 24 × 24 = 18,432` tokens in a complete clip. The observed frames occupy time positions 0–15; target frames 38–39 occupy position 19.

Flatten tokens in temporal-major order:

```python
def token_index(tubelet: int, row: int, col: int, grid: int = 24) -> int:
    return tubelet * grid * grid + row * grid + col


def all_spatial_tokens(tubelets, grid=24):
    return [
        token_index(tubelet, row, col, grid)
        for tubelet in tubelets
        for row in range(grid)
        for col in range(grid)
    ]


context_indices = all_spatial_tokens(range(0, 16))
target_tubelet = 19  # source frames 38 and 39
```

### 4.2 Find the tokens covering the person

Take the rectangle covering the person boxes in both target frames. Select patches whose centers lie inside that rectangle, then expand the selection by one patch on each side. This **dilation** reduces abrupt region changes from small tracking errors.

```python
def patches_in_box(box_xyxy, grid=24, dilation=1):
    x0, y0, x1, y1 = box_xyxy
    cols = [col for col in range(grid) if x0 <= (col + 0.5) / grid <= x1]
    rows = [row for row in range(grid) if y0 <= (row + 0.5) / grid <= y1]
    if not rows or not cols:
        raise ValueError("person box contains no patch centers")
    cols = range(max(0, min(cols) - dilation), min(grid, max(cols) + dilation + 1))
    rows = range(max(0, min(rows) - dilation), min(grid, max(rows) + dilation + 1))
    return [(row, col) for row in rows for col in cols]
```

The target person-token indices are those `(row, col)` positions at tubelet 19. Background indices come from the same tubelet, outside the person selection and its frozen one-patch guard band. Apply the same region rule to observed boxes when pooling past person and background features.

Write one unit test against a synthetic tensor whose token values equal their flattened indices. Do not trust an assumed flattening order merely because tensor shapes match.

### 4.3 Use two separate teacher calls

The proposal permits two routes:

1. use the official predictor with explicit past-context and future-target masks; or
2. use frozen encoder targets with a separately trained equal-capacity baseline if the predictor route is not clean and stable.

For this gate, use route 2: the frozen encoder creates features and targets, and a small fitted baseline makes predictions. This directly supports the conditional comparison and keeps the teacher integration manageable within 48 hours. Keep the predictor capacities matched across the skeleton controls.

The adapter should perform two different operations:

```python
class FrozenVJEPAAdapter:
    def encode_past_context(self, video_64, context_indices):
        """Return tokens computed with future tokens removed before transformer blocks."""
        ...

    def encode_full_target(self, video_64):
        """Return frozen tokens from the complete 64-frame clip for target creation."""
        ...
```

`encode_past_context` must remove future tokens **before transformer attention**. In the [official encoder](https://github.com/facebookresearch/vjepa2/blob/main/src/models/vision_transformer.py), token selection precedes the transformer blocks; verify that the pinned adapter preserves this order and that the indices denote the tokens to keep.

Why not encode the full clip and keep only the first half afterward? Because past tokens may already have read future tokens through attention. Cutting off their output positions would not erase the information they absorbed. The pixel-edit test in Stage 5 checks this directly.

`encode_full_target` sees the complete clip to create the answer. Its output never enters either predictor. As established in the frame contract, the answer is a **contextual target**: the token at frames 38–39 can summarize other parts of the clip too. Record that property explicitly in the report.

**Pooling** means averaging several token vectors into one summary vector. Construct these five pools:

- `context_global`: mean of all past context tokens;
- `context_person_last`: mean of past person-region tokens at the last observed tubelet;
- `context_background`: mean of past tokens outside the person region;
- `future_person`: mean of future person-region tokens at tubelet 19; and
- `future_background`: mean of target-tubelet background tokens.

The first three are past inputs; the last two are targets. Use the final normalized encoder layer and record its exact output and normalization rule in `teacher-contract.json`. If the adapter returns several layers, select the final layer explicitly. Apply the checkpoint's documented target normalization before pooling, without duplicating a normalization already performed by the adapter. This fixed model operation is separate from the data-fitted standardization in Stage 6.

### 4.4 Reduce each target to 256 numbers

The pooled feature may contain more numbers than a small pilot can model efficiently. Multiply it by one fixed random matrix to obtain a 256-number target. This **projection** chooses random combinations of the original features. It is not learned from the videos and must not be selected for a favorable result.

For a pooled dimension \(D \geq 256\), use orthogonal matrix columns, then scale them so squared distances are preserved in expectation over random projections:

```python
import numpy as np


def fixed_projection(input_dim: int, output_dim: int = 256, seed: int = 260905):
    if output_dim > input_dim:
        raise ValueError("output dimension cannot exceed input dimension")
    rng = np.random.default_rng(seed)
    matrix = rng.standard_normal((input_dim, output_dim))
    q, _ = np.linalg.qr(matrix)
    scale = np.sqrt(input_dim / output_dim)
    return (q * scale).astype(np.float32)


projection = fixed_projection(input_dim=future_person.shape[-1])
target_256 = future_person @ projection
```

Save the matrix and its SHA-256 before model fitting. Apply it to both person and background targets. Each example now has past video features, a `[32, 33, 4]` skeleton history, a `[256]` person target, and a `[256]` background target. Save one checksum-bound cache file per window, tied to the cohort, teacher, and preprocessing contracts. Refuse missing, duplicated, or mismatched cache entries.

## 5. Check validity before training

These checks ask whether the experiment is measuring what it claims to measure. Run them before fitting the baseline or any skeleton head. A high score cannot rescue a predictor that has already seen the answer.

### 5.1 Change the future; the past input must stay the same

Copy each audit clip, replace frames `32..63` with random pixels, and encode the past again. If the past features change, some future information crossed the boundary. The sketch assumes the adapter returns NumPy arrays:

```python
import numpy as np


def future_pixel_leakage_test(adapter, video, context_indices, rng):
    changed = video.copy()
    changed[32:] = rng.integers(0, 256, size=changed[32:].shape, dtype=np.uint8)

    original = adapter.encode_past_context(video, context_indices)
    modified = adapter.encode_past_context(changed, context_indices)

    difference = np.abs(original - modified)
    return {
        "max_abs_difference": float(difference.max()),
        "mean_abs_difference": float(difference.mean()),
    }
```

Run once in float32 with deterministic kernels. Require exact equality when practical; otherwise freeze a tolerance justified by repeated identical-input runs. A suggested ceiling is the larger of:

- `1e-6`; or
- twice the maximum difference observed when the unchanged clip is encoded twice.

If future edits exceed that numerical floor, stop. Do not train a model and “control for” direct target leakage later.

Also test three common failure modes:

- the frame-31/32 boundary splits no temporal tubelet;
- no input mean, scale, crop choice, or other content-dependent preprocessing statistic was computed from future frames; and
- camera-motion, scale, or skeleton interpolation uses only frames `0..31`.

### 5.2 Change the person and background separately

A target could pass the leakage check while mostly describing scenery. On the 10 audit windows chosen before feature inspection, test what changes its value. If using more than 10, freeze an equivalent direction-consistency rule in advance.

Create two variants for each window:

1. **Future-person edit:** replace pixels inside the person box throughout unobserved frames `32..63` with motion-different person regions, while keeping the background fixed.
2. **Static-background replacement:** replace pixels outside the person box throughout unobserved frames `32..63` with a background from another clip, while keeping the person region fixed.

Use the same **feathered** boundary—a gradual blend at the edge of the edited region—for both edits, so a sharp cut is not unique to one condition. Keep the target-pooling indices and projection fixed to those of the original window. Encode each edited clip and measure change in the same projected target space:

```python
def normalized_change(original, edited, epsilon=1e-8):
    numerator = np.linalg.norm(edited - original, axis=-1)
    denominator = np.linalg.norm(original, axis=-1)
    return numerator / np.maximum(denominator, epsilon)


motion_change = normalized_change(target_original, target_person_edited)
background_change = normalized_change(target_original, target_background_edited)
ratio = np.median(motion_change) / max(np.median(background_change), 1e-8)
```

For example, median normalized changes of `0.12` for person edits and `0.04` for background edits give a ratio of `3.0`. Pass the recommended audit when the ratio is at least `2.0` and the person edit causes a larger change in at least 8 of 10 windows. Save both changes, the ratio, and contact sheets of all edits. Check finite, nondegenerate targets; the small epsilon prevents division by zero but cannot make a constant target valid.

A large ratio supports relative person sensitivity. It does not prove that the teacher understands motion: person replacements can also change appearance or introduce editing artifacts. Describe the actual edit, and avoid interpreting the ratio as proof that background has no influence.

Before moving on, the repeated-input stability check, future-pixel leakage check, crop-retention check, and person-versus-background audit must pass. Save their results in `qc/` even if the run stops here.

## 6. Fit the comparisons

The purpose of this stage is to give the baseline and each skeleton arm a fair test on unseen source videos. The teacher stays frozen. We train only the baseline and small correction heads, using the cached features.

### 6.1 Assemble the baseline inputs

Build one row per window from frames `0..31` and the permitted recording metadata below. All image, pose, and camera-motion features use the observed context only.

| Group | Example features | Why it is included |
| --- | --- | --- |
| Current video | global and last-person V-JEPA context pools | Accounts for appearance, pose, and motion already represented in past RGB |
| Timing | annotated duration, relative window position, decoded FPS | Removes easy progress and horizon-duration clues |
| Framing | first/last box center, width, height, area, scale change | Removes crop and person-location shortcuts |
| Motion nuisance | box-centroid velocity and background optical flow | Removes camera and global translation clues |
| Pose quality | confidence mean/quantiles and joint missingness | Removes detector-quality shortcuts |
| Source properties | width, height, aspect ratio, view estimate | Removes recording-format clues |
| Background | past background V-JEPA pool and RGB mean/std | Removes static-scene identity |

One possible typed contract is:

```python
@dataclass(frozen=True)
class GateExample:
    window_id: str
    video_id: str
    outer_fold: int
    baseline_features: np.ndarray
    skeleton_history: np.ndarray   # [32, 33, channels]
    target_person_256: np.ndarray  # [256]
    target_background_256: np.ndarray
```

An array of baseline features might combine pixel-derived embeddings, frame rates, box sizes, and missingness rates. Their numerical scales differ. **Standardization** subtracts a training-set mean and divides by a training-set standard deviation so one unit system does not dominate the fit.

Every preprocessing operation learned from data must be **fold-local**: fit it using the training sources for that particular fit, then apply it unchanged to validation or test sources. This applies to:

- continuous-feature means and scales;
- missing-value imputers;
- view-category encoders; and
- target means and scales.

During inner validation, “training” means only the inner-training sources. During the final outer fit, it means all outer-training sources. Do not compute a global normalizer before splitting. Give every source equal total weight, as shown below.

Annotated duration and relative window position are permitted recording metadata in this offline comparison. They may require knowing the clip's endpoints; record that assumption explicitly. They do not permit future pixels, future pose measurements, or future camera motion to enter `x`. A later live-deployment experiment would need its own input-availability contract.

### 6.2 Fit the baseline

Use **ridge regression**: a linear predictor with a penalty that discourages very large weights. This helps when there are many features and few windows. “Multi-output” means it predicts all 256 target numbers. Its `alpha` parameter controls the strength of the penalty.

```python
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def make_baseline(alpha: float):
    return make_pipeline(
        StandardScaler(),
        Ridge(alpha=alpha),
    )


baseline = make_baseline(alpha=10.0)
baseline.fit(
    x_train,
    y_train,
    standardscaler__sample_weight=train_source_weights,
    ridge__sample_weight=train_source_weights,
)
```

The example uses `10.0` to show the API. Select the actual `alpha` from a frozen grid such as `(0.1, 1.0, 10.0, 100.0, 1000.0)` using inner source folds. The outer test sources never select it. [Section 6.5](#65-run-the-whole-fit-inside-source-folds) shows how those two levels fit together.

### 6.3 Fit a small skeleton correction

The **temporal head** looks for patterns across skeleton frames. Flatten the 33 landmarks and their four channels into 132 values per frame, giving `[batch, 32, 132]`. Two one-dimensional convolutions scan neighboring frames; averaging over time gives one summary. Concatenate that summary with the standardized baseline inputs to predict the 256-number correction:

```python
import torch
from torch import nn


class SkeletonResidualHead(nn.Module):
    def __init__(self, input_channels: int, baseline_dim: int, width: int = 64):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv1d(input_channels, width, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(width, width, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.output = nn.Linear(width + baseline_dim, 256)

    def forward(self, skeleton, baseline_features):
        # skeleton: [batch, time, joints * channels]
        encoded = self.temporal(skeleton.transpose(1, 2)).mean(dim=-1)
        return self.output(torch.cat([encoded, baseline_features], dim=-1))
```

Train against the baseline's error. This sketch makes the NumPy-to-PyTorch boundary explicit; `x_baseline_scaled` has already been transformed using the fitted baseline scaler, and `y_future` is in the same target units as the baseline prediction:

```python
baseline_prediction = torch.as_tensor(
    baseline.predict(x_baseline), dtype=torch.float32, device=device
)
residual_target = y_future - baseline_prediction     # Training target only.
residual_prediction = head(x_skeleton, x_baseline_scaled)
full_prediction = baseline_prediction + residual_prediction
```

Minimize source-weighted squared error between `residual_prediction` and `residual_target`. At test time, compute `full_prediction` without using `y_future`; compare with the target afterward.

Keep the head width, optimizer, maximum update budget, stopping rule, and parameter count identical across skeleton controls. **Weight decay** penalizes large neural-network weights. **Early stopping** chooses a training duration from validation performance before the head overfits. With only 50 windows, both need conservative, frozen settings.

Repeat the matched comparisons with a fixed initialization schedule; three seeds are a practical choice. An **initialization** is the random starting set of head weights. Report each seed's results rather than selecting the best. If the aggregate point checks pass but only one initialization reaches the real-gain threshold, the result is `INCONCLUSIVE`.

### 6.4 Change one part for each control

The real, time-shuffled, clip-mismatched, and no-skeleton arms share the person target and its baseline predictions. All arms share source folds, head architecture, maximum training budget, stopping policy, and seed schedule. The background-target arm changes the target, so it requires its own baseline fit.

**Time-shuffled skeleton: does order matter?**

Split the 32-frame context into eight four-frame blocks and permute the blocks with a frozen window-specific seed. This destroys temporal order without changing the values, missingness fraction, or within-block smoothness.

```python
def block_shuffle(history, rng, block=4):
    blocks = history.reshape(len(history) // block, block, *history.shape[1:])
    order = rng.permutation(len(blocks))
    return blocks[order].reshape(history.shape)
```

For example, blocks `[A, B, C, D, E, F, G, H]` might become `[C, A, H, E, B, G, D, F]`. Shuffle coordinates, confidence, and validity together. Do not shuffle individual coordinates independently; that would make anatomically incoherent inputs. This control disrupts order across blocks while retaining motion within each four-frame block.

**Clip-mismatched skeleton: does it belong to this video?**

Within each outer split, match every RGB example to a skeleton from a different source with the closest available decoded FPS, view, duration, box scale, centroid speed, and pose-quality summary. Solve the matching without replacement when possible.

Make separate donor assignments within each training, validation, or test partition. For an outer fold, training examples may borrow only training skeletons, and test examples may borrow only test skeletons. Repeat that separation inside inner validation. A donor must come from a different source, and matching uses context metadata, never future targets. This keeps a held-out skeleton out of model fitting.

GAVD does not provide participant IDs. Call this a clip or source mismatch unless identity is independently verified.

**Background target: is the gain concentrated near the person?**

Repeat the real-skeleton evaluation using `target_background_256`. Refit the baseline for that target, then calculate the skeleton gain relative to that background baseline. Reusing the person-target baseline would mix two different prediction problems.

For example, a person-target gain of `0.06` and a background-target gain of `0.02` gives a reduction of `1 - 0.02 / 0.06 ≈ 67%`, passing the recommended 50% rule. Negative background gains count as zero for this reduction calculation; still report their original values.

“Background-only” describes which output tokens are pooled. It does not remove the person from the teacher's input or undo attention between person and background tokens. Interpret it together with the separate pixel-edit audit.

**No-skeleton capacity control: is motion actually needed?**

Keep the input shape and head capacity identical, but zero the coordinate and confidence channels and retain the validity channel. The head still sees baseline inputs and which landmarks were detected. This is a validity-only version of the no-skeleton control; freeze this choice in `control-contract.json`.

If this arm achieves a similar gain to the real arm, nonlinear use of baseline features or missingness may explain the result. Report the paired gain difference and its uncertainty. The experiment has not isolated a motion contribution merely because the real head beats ridge regression.

### 6.5 Run the whole fit inside source folds

There are two levels of separation. **Outer folds** estimate performance on unseen sources. **Inner folds** choose settings using only sources available for training that outer model.

For example, when outer fold 0 is the test set, folds 1–4 are available for development. Split those sources again to choose `alpha` and training duration. Fold 0 participates only after all choices for that fit are fixed.

For each outer fold:

1. Set aside every window from its source videos.
2. Within the remaining sources, create inner training/validation splits. Fit preprocessing and the baseline on inner training only; construct residuals with that baseline, fit the head, and evaluate on inner validation. Use these comparisons to choose regularization and training duration.
3. With those settings fixed, refit preprocessing and the baseline on all outer-training sources. Fit each correction head on the corresponding training residuals for its selected duration.
4. Predict the untouched outer-test windows for every arm and seed.
5. Save the predictions and training-mean references before moving to the next fold.

An inner validation example must not influence the baseline used to evaluate its correction head. Fitting the baseline on all outer-training examples *before* inner validation would contaminate that validation step. The same rule applies to learned preprocessing and mismatch donors.

Do not tune on the pooled 50-window out-of-fold score.

Give every source equal total influence. A source with two windows gives each window half the weight of a source with one:

```python
def equal_source_weights(video_ids):
    counts = pd.Series(video_ids).value_counts()
    raw = np.array([1.0 / counts[value] for value in video_ids])
    return raw * (len(raw) / raw.sum())
```

Use these weights in ridge fitting, neural loss, and aggregate squared-error calculations.

Save one row per window, arm, initialization, and target feature:

```text
window_id, video_id, outer_fold, arm, seed,
target_feature, y_true, y_pred_baseline, y_pred_full
```

This **long-form** table uses 256 rows per window/arm/seed. It makes every score independently recomputable. Also save the fold's target means, scales, and valid-feature flags so raw and standardized predictions can be reconciled.

Before moving on, every eligible window must have one outer-test prediction per planned arm and seed, all controls must be complete, and the split and preprocessing audits must pass.

## 7. Score the predictions and decide

The purpose of this stage is to turn saved predictions into a decision using rules fixed before fitting. The metric and report code must not retrain models or select a more favorable cohort, projection, or seed.

### 7.1 Score against the training-mean reference

For each projected feature, compare two squared errors:

$$
R^2 = 1 - \frac{\text{weighted squared error of the predictor}}
{\text{weighted squared error of predicting the training mean}}
$$

A perfect prediction scores 1. Matching the reference scores 0. Doubling its error scores −1. Calculate this separately for each feature, then average valid feature scores with equal weight.

Standardize each target feature using outer-training statistics, then transform both outer-test targets and predictions with those same statistics. If fitting already used standardized targets, do not standardize the predictions twice. The reference prediction for each outer-test row is zero in these units: that fold's outer-training mean. Pool OOF values only after all folds finish.

Mark a feature invalid if its outer-training variance is essentially zero in any fold, using a frozen tolerance. Use the same valid-feature mask for every arm sharing a target. Person and background targets have separate variance checks.

```python
import numpy as np


def featurewise_predictive_r2(
    y_true, y_pred, y_reference, sample_weight, training_variance_valid
):
    if not all(np.isfinite(a).all() for a in (y_true, y_pred, y_reference)):
        raise ValueError("non-finite targets or predictions")
    weights = np.asarray(sample_weight, dtype=np.float64)[:, None]
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("source weights must be finite and positive")
    error = np.sum(weights * (y_true - y_pred) ** 2, axis=0)
    reference_error = np.sum(weights * (y_true - y_reference) ** 2, axis=0)
    valid = np.asarray(training_variance_valid, dtype=bool) & (reference_error > 1e-12)
    if not valid.any():
        raise ValueError("no valid target features")
    score = np.full(y_true.shape[1], np.nan, dtype=np.float64)
    score[valid] = 1.0 - error[valid] / reference_error[valid]
    return score, valid


y_reference_oof = np.zeros_like(y_true_oof)
training_variance_valid = training_variance_valid_by_fold.all(axis=0)
r2_base_features, valid = featurewise_predictive_r2(
    y_true_oof, y_base_oof, y_reference_oof, source_weights, training_variance_valid
)
r2_real_features, valid_real = featurewise_predictive_r2(
    y_true_oof, y_real_oof, y_reference_oof, source_weights, training_variance_valid
)
assert np.array_equal(valid, valid_real)

r2_base = float(np.mean(r2_base_features[valid]))
r2_real = float(np.mean(r2_real_features[valid]))
delta_r2 = r2_real - r2_base

denominator = 1.0 - r2_base
f8 = delta_r2 / denominator if denominator > 1e-8 else np.nan
```

This is **predictive \(R^2\)**: the reference is the mean available during training, rather than a mean computed from held-out answers. Equal feature weights prevent a few large-scale dimensions from dominating. Report the valid-feature count, median, quartiles, fraction of positive feature gains, and complete distribution alongside the headline mean.

Record both ways of aggregating \(\Delta R^2\):

1. subtract the two aggregate \(R^2\) values; and
2. average the 256 feature-level \(R^2\) differences.

They are algebraically equal under a uniform mean; asserting equality catches bookkeeping errors.

If baseline \(R^2 > 0.95\), even perfect prediction cannot add `0.05`; at exactly `0.95`, passing would require perfection. Report that the target leaves little or no room for the required gain. Do not weaken the baseline to manufacture more innovation. If \(1 - R^2_{\mathrm{base}}\) is effectively zero, report \(F_8\) as undefined rather than forcing a finite value.

### 7.2 Ask how sensitive the result is to sources and seeds

With a small cohort, a few unusually easy or difficult sources can change the result. A **bootstrap** approximates this uncertainty by repeatedly drawing a new collection from the observed sources, with replacement. A source may appear twice or not at all in a draw. Carry all of its windows together because they are related:

```python
def source_bootstrap_indices(video_ids, repetitions=2000, seed=260905):
    rng = np.random.default_rng(seed)
    unique = np.array(sorted(set(video_ids)))
    video_ids = np.asarray(video_ids)
    for _ in range(repetitions):
        draw = rng.choice(unique, size=len(unique), replace=True)
        yield np.concatenate([np.flatnonzero(video_ids == source) for source in draw])
```

For every draw, recompute baseline \(R^2\), each arm's \(R^2\), \(\Delta R^2\), and \(F_8\) from saved OOF predictions. Use the same draw for all arms so their comparisons stay paired. Carry the original per-window source weights with each sampled occurrence; do not collapse repeated source draws or recompute weights that cancel their multiplicity. Each occurrence of a source must retain equal total weight.

Save percentile intervals, the fraction of draws with positive real gain, and the fraction in which the real arm beats each control. When combining seeds, recompute each seed's score within the draw and apply the frozen seed-aggregation rule. Resampling saved predictions estimates source sensitivity; it does not refit the networks or replace the separate initialization check.

The 48-hour contract uses point thresholds, but a wide interval is evidence that the pilot is unstable. Recommended reporting is:

- `ADVANCE`: all validity, point, and stability checks pass, including positive real gain in at least 90% of source bootstraps and all planned initializations;
- `STOP`: a required condition clearly fails; or
- `INCONCLUSIVE`: point thresholds pass but uncertainty or seed sensitivity is too large.

An `INCONCLUSIVE` result supports planning a larger preregistered gate before proceeding to the full study. A wide interval should be shown even if it does not trigger the frozen decision rule; do not invent a new width cutoff after seeing it.

### 7.3 Compute the decision from the saved checks

Keep the decision in three layers: validity, point thresholds, then stability. This prevents a passing average from overriding a failed leakage test, and gives unstable results an explicit outcome.

The example below assumes the point metrics use the arithmetic mean across the planned seeds. Its audit flags summarize saved reports:

| Flag | Report must establish |
| --- | --- |
| `data_contract_valid` | Exactly 50 eligible windows, at least 25 sources, source caps, correct alignment and crop retention |
| `evaluation_contract_valid` | Correct folds, training-only preprocessing, isolated control donors, and consistent cache/prediction records |
| `controls_complete` | Every planned arm and seed has all required predictions |
| `target_audit_complete` | All 10 prespecified edited windows were evaluated and saved |
| `target_variance_valid` | Finite targets and a nonempty, consistently applied valid-feature set |

The no-skeleton comparison is included in the paired-control report for every run, but does not require a separate manual interpretation to complete the experiment. For seed stability, require positive gain in every planned seed and at least two seeds reaching `0.05`, implementing the rule against advancing on one lucky initialization.

```python
from dataclasses import asdict, dataclass
from math import isfinite
from numbers import Real


@dataclass(frozen=True)
class GateThresholds:
    real_delta_r2_min: float = 0.05
    real_to_shuffle_min: float = 2.0
    mismatch_delta_r2_max: float = 0.01
    person_ablation_reduction_min: float = 0.50
    motion_to_background_change_min: float = 2.0
    person_edit_direction_fraction_min: float = 0.80
    bootstrap_positive_fraction_min: float = 0.90


def decide_gate(metrics, thresholds=GateThresholds()):
    numeric_keys = (
        "delta_r2_real", "delta_r2_time_shuffle", "delta_r2_clip_mismatch",
        "delta_r2_background_target", "delta_r2_no_skeleton",
        "motion_to_background_change_ratio", "person_edit_direction_fraction",
        "bootstrap_positive_fraction",
    )
    seed_gains = metrics.get("seed_real_gains", [])
    numbers = [metrics.get(key) for key in numeric_keys]
    valid_numbers = (
        isinstance(seed_gains, list)
        and len(seed_gains) > 0
        and all(
            isinstance(value, Real) and not isinstance(value, bool) and isfinite(value)
            for value in numbers + seed_gains
        )
    )
    if valid_numbers:
        valid_numbers = (
            0 <= metrics["person_edit_direction_fraction"] <= 1
            and 0 <= metrics["bootstrap_positive_fraction"] <= 1
            and metrics["motion_to_background_change_ratio"] >= 0
        )
    if not valid_numbers:
        return {
            "decision": "STOP",
            "allow_full_experiment": False,
            "allow_adapter_training": False,
            "checks": {"complete_finite_metrics": False},
            "thresholds": asdict(thresholds),
            "metrics": None,  # Save malformed input separately; emit valid JSON here.
        }

    real = metrics["delta_r2_real"]
    shuffled = max(metrics["delta_r2_time_shuffle"], 0.0)
    mismatch = metrics["delta_r2_clip_mismatch"]
    background_target = max(metrics["delta_r2_background_target"], 0.0)

    shuffle_pass = real >= thresholds.real_to_shuffle_min * shuffled
    person_reduction = 1.0 - background_target / max(real, 1e-8)

    validity = {
        name: metrics.get(name) is True
        for name in (
            "data_contract_valid", "evaluation_contract_valid", "controls_complete",
            "target_audit_complete", "target_variance_valid",
            "teacher_stable", "causal_leakage_absent",
        )
    }
    point_checks = {
        "real_gain": real >= thresholds.real_delta_r2_min,
        "time_shuffle": shuffle_pass,
        "clip_mismatch": mismatch <= thresholds.mismatch_delta_r2_max,
        "person_region": person_reduction >= thresholds.person_ablation_reduction_min,
        "target_sensitivity": (
            metrics["motion_to_background_change_ratio"]
            >= thresholds.motion_to_background_change_min
        ),
        "person_edit_consistency": (
            metrics["person_edit_direction_fraction"]
            >= thresholds.person_edit_direction_fraction_min
        ),
    }
    stability = {
        "bootstrap_positive": (
            metrics["bootstrap_positive_fraction"]
            >= thresholds.bootstrap_positive_fraction_min
        ),
        "seeds_stable": (
            all(gain > 0 for gain in seed_gains)
            and sum(gain >= thresholds.real_delta_r2_min for gain in seed_gains) >= 2
        ),
    }
    if not all(validity.values()) or not all(point_checks.values()):
        decision = "STOP"
    elif not all(stability.values()):
        decision = "INCONCLUSIVE"
    else:
        decision = "ADVANCE"

    return {
        "decision": decision,
        "allow_full_experiment": decision == "ADVANCE",
        "allow_adapter_training": False,  # Requires the later frozen-S-JEPA result.
        "checks": {**validity, **point_checks, **stability},
        "thresholds": asdict(thresholds),
        "metrics": metrics,
    }
```

The input-validation layer must also check array shapes, expected seed counts, and that aggregate values match the saved per-seed scores. A missing required report cannot default to a pass. Add provenance fields when writing `reports/gate-decision.json`, and serialize with non-finite JSON values disallowed.

Example output schema:

```json
{
  "experiment": "future-innovation-48-hour-gate",
  "run_id": "gate-v1",
  "decision": "STOP",
  "allow_full_experiment": false,
  "allow_adapter_training": false,
  "manifest_sha256": "...",
  "vjepa_commit": "...",
  "checkpoint_sha256": "...",
  "checks": {},
  "thresholds": {},
  "metrics": {}
}
```

The JSON above is an output schema, not an observed result. `allow_adapter_training` stays `false` even when Experiment 0 advances: raw skeletons passing this gate do not establish the later frozen-S-JEPA criterion.

### 7.4 Write a report that makes the decision checkable

Lead `gate-report.md` with the decision and the reason. Then show the data count, validity audits, and a table with each arm's baseline \(R^2\), full \(R^2\), \(\Delta R^2\), and \(F_8\). Identify the background arm's separate target and baseline. Include every seed, source-bootstrap intervals, and the person-edit audit; do not show only the best scores.

For orientation, these are hypothetical outcomes, not experimental evidence:

| Saved evidence | Decision | Why |
| --- | --- | --- |
| Real gain `0.06`, shuffle `0.02`, mismatch `0.005`, background gain `0.02`; all audits and stability checks pass | `ADVANCE` | Size, timing, pairing, region, and stability requirements all pass |
| Real gain `0.06`, shuffle `0.04` | `STOP` | Real gain is less than twice the shuffled gain |
| Point checks pass, but only 70% of bootstrap draws have positive real gain | `INCONCLUSIVE` | The source-stability requirement fails |
| Future edits change the context representation | `STOP` | The inputs contain future information, so prediction scores cannot answer the question |

State which stage ran, which checks failed, and what the next experiment would need to repair. A completed valid null result is useful evidence. A run that stops on invalid data or leakage has not tested the scientific hypothesis.

## Appendix A. Implementation and tests

### Required tests

Implement synthetic tests before HAIC feature extraction:

| Test | Required behavior |
| --- | --- |
| `test_future_innovation_cohort.py` | deterministic 50-window selection, source caps, and fold isolation |
| `test_future_innovation_frames.py` | exact 64-frame decode and correct observed/target indices |
| `test_future_innovation_boxes.py` | RGB and boxes undergo identical resize/crop geometry |
| `test_future_innovation_tokens.py` | temporal-major token indexing and correct person/background masks |
| `test_future_innovation_causality.py` | arbitrary future-pixel edits cannot alter context features |
| `test_future_innovation_projection.py` | deterministic scaled-orthogonal projection with fixed checksum |
| `test_future_innovation_controls.py` | shuffles preserve shape and never cross outer folds |
| `test_future_innovation_metrics.py` | analytical \(R^2\), \(\Delta R^2\), \(F_8\), weighting, and bootstrap cases |
| `test_future_innovation_gate.py` | validity/point failures force `STOP`; instability gives `INCONCLUSIVE`; only a complete stable pass advances |

Mandatory failure cases include:

- a source video appearing in two outer folds;
- fewer than 50 eligible windows;
- an empty or severely cropped target person region;
- a target tubelet overlapping observed frames;
- future pixels changing past context features;
- normalization fitted on all sources;
- a clip-mismatch pairing crossing an outer fold;
- a constant target dimension being treated as valid evidence;
- NaN or infinite \(R^2\) being converted to zero;
- a gate pass when any required check is false;
- a target audit passing its ratio but failing the 8-of-10 direction rule;
- an unstable point pass being labeled `ADVANCE`; and
- adapter training being allowed by this raw-skeleton gate alone.

Run focused tests with:

```bash
cd "$GAVD6_ROOT"
uv run --no-sync python -m unittest discover -s tests -p 'test_future_innovation_*.py'
```

### Suggested module and job layout

Follow the current `src` package structure:

```text
src/gavd6_sjepa/research_directions/future_innovation/
├── cohort.py
├── video_and_pose.py
├── vjepa_adapter.py
├── token_regions.py
├── feature_cache.py
├── nuisance_features.py
├── residual_models.py
├── controls.py
├── metrics.py
├── gate_decision.py
└── entrypoint.py

slurm/future-innovation/
├── 01-build-gate-cohort.sbatch
├── 02-extract-poses.sbatch
├── 03-cache-vjepa-features.sbatch
├── 04-run-causal-and-target-audits.sbatch
├── 05-fit-gate-models.sbatch
└── 06-build-gate-report.sbatch
```

After unit tests pass, add lazy CLI routes resembling:

```bash
uv run --no-sync gavd6 future-innovation build-cohort --help
uv run --no-sync gavd6 future-innovation extract-poses --help
uv run --no-sync gavd6 future-innovation cache-teacher --help
uv run --no-sync gavd6 future-innovation run-gate --help
```

These interfaces are proposed and do not exist yet.

Use CPU jobs for cohort creation, pose extraction, metrics, and reports. Use an H100 for V-JEPA caching. The small residual heads may use the same GPU after caching or a separate short GPU allocation. Teacher jobs must write one unique cache file per `window_id`; aggregation should refuse missing, duplicated, or checksum-mismatched files.

## Appendix B. The 48-hour schedule

| Hours | Work | Exit condition |
| --- | --- | --- |
| 0–4 | Freeze configuration; load the pinned teacher; identify candidate windows | Environment works, settings are saved, and one teacher forward pass succeeds |
| 4–10 | Decode candidates, transform boxes, extract poses, inspect overlays, and freeze the final cohort | Exactly 50 eligible aligned windows with fixed source folds |
| 10–18 | Encode past and target features; pool, project, and cache them | One complete, checksum-bound feature artifact per window |
| 18–24 | Test repeated inference, leakage, crop retention, target variance, and pixel-edit sensitivity | Validity checks pass before fitting |
| 24–38 | Fit nested source-fold baselines and all correction/control arms | Complete OOF predictions for every planned seed |
| 38–46 | Compute feature and aggregate scores, source bootstraps, and seed comparisons | Uncertainty and control comparisons are recorded |
| 46–48 | Run the decision function and write the report | Decision, reasons, and reproducible artifacts are saved |

Run synthetic tests while building the relevant modules. Check teacher stability and leakage on a small smoke-test clip as soon as the adapter works, then complete the formal audits on the frozen cohort. This catches integration failures before paying for the full cache.

The final cohort is frozen after eligibility is established but before inspecting teacher features or scores. If implementation, data repair, or a failed audit consumes the budget, report where the run stopped. The 48-hour time limit never turns an unfinished comparison into an advance.

## Appendix C. What a failure means

| Failure | Meaning | Correct next action |
| --- | --- | --- |
| Fewer than 50 aligned windows | Data route is not ready | Repair decoding, box, or pose extraction before modeling |
| Context changes after a future edit | Direct temporal leakage | Repair masking or abandon the teacher route |
| Background edits change the target as much as person motion | Target is not person-selective | Redesign region pooling; do not fit the skeleton head |
| Baseline \(R^2\) is near 1 | Almost no innovation remains | Report a low ceiling; do not weaken the baseline |
| Real \(\Delta R^2<0.05\) | Skeleton signal is too small in this gate | Stop or run one preregistered larger confirmation gate |
| Time shuffle performs similarly | Correct block order is not needed for the gain | Investigate static pose, within-block motion, or source shortcuts; the timing criterion fails |
| Clip mismatch performs similarly | Correct pairing is not needed for the gain | Investigate source, view, phase, or model-capacity explanations |
| Person and background targets show similar gains | Effect is not localized to the person | Stop the claimed mechanism |
| No-skeleton head explains the apparent gain | Motion attribution remains unresolved | Preserve the automatic decision, report the paired contrast, and investigate capacity before claiming a motion contribution |
| Seeds or bootstrap fail the frozen stability rule | Pilot does not provide a stable pass | Mark `INCONCLUSIVE` if point checks pass; plan a larger gate |

A valid null result says that this gate does not justify skeleton distillation from the chosen target under these controls. It does not establish that skeleton motion contains no useful information under every target, horizon, or model.

## Appendix D. Final artifact checklist

```text
$FI_RUN_ROOT/
├── config/
│   ├── environment.txt
│   ├── run-contract.json
│   ├── frame-contract.json
│   ├── teacher-contract.json
│   ├── nuisance-contract.json
│   ├── model-contract.json
│   ├── control-contract.json
│   ├── thresholds.json
│   └── projection-256.npy
├── manifests/
│   ├── gate-windows.csv
│   ├── exclusions.csv
│   └── cache-index.csv
├── poses/
│   └── <window_id>.npz
├── teacher-cache/
│   └── <window_id>.npz
├── models/
│   └── <fold-and-arm>/  # Fitted heads, baseline, and preprocessing statistics.
├── predictions/
│   ├── baseline.parquet
│   ├── real-skeleton.parquet
│   ├── time-shuffle.parquet
│   ├── clip-mismatch.parquet
│   ├── background-target.parquet
│   └── no-skeleton.parquet
├── qc/
│   ├── alignment-overlays/
│   ├── pixel-edit-contact-sheets/
│   ├── teacher-stability.csv
│   ├── causal-leakage.csv
│   └── target-sensitivity.csv
└── reports/
    ├── featurewise-r2.csv
    ├── aggregate-metrics.csv
    ├── source-bootstrap.csv
    ├── gate-report.md
    └── gate-decision.json
```

The report must state what was planned, what ran, how many sources and windows were valid, all thresholds, every control result, uncertainty across sources and seeds, and whether the full experiment is authorized. A failed gate is a completed experiment; a missing control is not.

## Official technical references

- [V-JEPA 2 and V-JEPA 2.1 official repository](https://github.com/facebookresearch/vjepa2)
- [Official V-JEPA 2.1 hub model builders](https://github.com/facebookresearch/vjepa2/blob/main/src/hub/backbones.py)
- [Official V-JEPA 2.1 predictor masking interface](https://github.com/facebookresearch/vjepa2/blob/main/app/vjepa_2_1/models/predictor.py)
- [V-JEPA 2.1 paper](https://arxiv.org/abs/2603.14482)
