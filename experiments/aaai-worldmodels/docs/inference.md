# Classify one local walking sequence

Use `worldmodels.binary_inference` for the study's **direct normal-versus-abnormal classifiers**. It loads the sealed Random Forest and protocol-declared mixed-adaptation JEPA probe from the clean cohort. It applies their existing fitted transforms and weights, averages window probabilities within the input sequence, and uses the declared abnormal-probability threshold of 0.5. It performs no training, model selection or threshold tuning.

These outputs predict pooled GAVD annotation labels. They do not establish a person's clinical diagnosis. The four abnormal labels in this study are parkinsons, stroke, cerebral palsy and myopathic; an unfamiliar condition or activity is outside that supervised label definition.

## Prerequisites and command

Complete the ordered preparation and clean classifier workflow in [the execution guide](execution.md). The adapter needs both `artifacts/runs/binary-development-clean-v1` and its referenced `pilot-comparison-clean-v1`, along with the recorded preparation files and pose-model weights. It checks those identities before loading the classifiers.

Use artifacts generated locally by this project or otherwise independently trusted. Joblib can execute code during deserialization. Hash checks establish consistency with the selected run; they do not establish trust in an unknown artifact's author.

From `experiments/aaai-worldmodels`, replace the example video and frame interval with a walking sequence you have reviewed:

```bash
uv run python -m worldmodels.binary_inference \
  --video "/absolute/path/walking.mp4" \
  --development artifacts/runs/binary-development-clean-v1 \
  --start-frame 1 \
  --end-frame 121 \
  --pose-model ../../data/models/pose_landmarker_lite.task \
  --max-windows 4 \
  --output artifacts/input-sequence-binary-prediction.json
```

`--development` defaults to the clean binary run shown above. Both frame arguments are **one-based and inclusive**. Source frame 1 corresponds to decoded OpenCV index zero, and the shared extraction adapter performs that conversion once. No truth label is supplied.

The Python interface is:

```python
from worldmodels.binary_inference import classify_video

report = classify_video(
    "/absolute/path/walking.mp4",
    "artifacts/runs/binary-development-clean-v1",
    "../../data/models/pose_landmarker_lite.task",
    start_frame=1,
    end_frame=121,
    max_windows=4,
)
report["classifiers"]["jepa_mixed"]
```

The classification portion of a successful report contains:

| Field | Meaning |
| --- | --- |
| `classifiers.rf.probabilities` | Direct-binary RF probabilities for `normal` and `abnormal`. |
| `classifiers.jepa_mixed.probabilities` | Direct-binary probabilities from the fixed mixed JEPA probe. |
| `predicted_annotation` within each classifier | `abnormal` when its probability is at least 0.5, otherwise `normal`. |
| `decision_threshold` | The existing 0.5 rule, including the abnormal decision at a tie. |
| `selection_sha256` and classifier hashes | The sealed selection and exact classifier artifacts used. |
| `quality`, `input` and `n_windows` | Observation coverage, frame range, sampled rate, input/model identities and retained window count. |
| `person_review_status` | Always `required`, because numerical confidence does not establish identity continuity. |

Probability columns are aligned by class name before aggregation. The mixed JEPA primary stays fixed even when another encoder has a higher validation score. Reported probabilities are not a clinical confidence interval or a guarantee of probability calibration.

## Shared video and pose contract

The wrapper reuses the existing RGB image extraction, feature construction and windowing interfaces. It samples every second source frame. A 30 FPS video therefore supplies a 15 Hz sampled sequence; temporal RF features use that actual sampled rate. Variable-frame-rate input is represented using the decoder's nominal FPS, matching this training pipeline. It is not timestamp-based temporal resampling.

Each window contains 16 sampled frames and advances by eight sampled frames. The first eight frames establish the torso scale. Coordinates use the declared 16 BlazePose joints, centered at each frame's pelvis and divided by that prefix scale. A confidence threshold of 0.5 and minimum valid-joint fraction of 0.7 apply. Missing detections preserve their frame positions and validity flags. They do not become interpolated observations.

`--max-windows` bounds the first temporal region processed. Four windows require at most 40 sampled frames, covering source frames 1 through 79 when starting at frame 1. The report records the requested and processed endpoints. Failed windows are excluded instead of being replaced by a later, easier interval. Without this option, extraction processes the requested range.

The RF consumes the shared 37-feature contract. The JEPA path applies the saved mixed-adaptation EMA encoder and its fitted binary probe. An input label, new training fit, or alternative encoder is not needed.

## Review the person and scene

Play the selected interval and confirm that the intended person walks continuously through it. The image API selects the first detected person in the full frame. This adapter has no identity tracker or GAVD annotation box for an arbitrary input video. A different person, a scene cut, seated activity or an occlusion can invalidate interpretation despite high pose confidence.

The clean training cohort also uses annotation-box containment and documented whole-source exclusions. Arbitrary unlabeled input lacks that annotation box, so the adapter cannot reproduce that part of the training eligibility review. It applies the same numerical pose/window rules and reports the selection mechanism, while retaining the manual person-review requirement. Training-source exclusions do not automatically recognize a corresponding new clip.

A quality failure returns no classifiers. Input, compatibility and quality failures produce an explanatory JSON status and exit code 2. Successfully computed predictions return exit code 0 while still requiring person review.

## Optional five-class baseline and compact forecast

The original `worldmodels.inference` command remains available for the five-class baseline and its compact 16-frame forecast:

```bash
uv run python -m worldmodels.inference \
  --video "/absolute/path/walking.mp4" \
  --run artifacts/runs/pilot-comparison-clean-v1 \
  --start-frame 1 --end-frame 121 \
  --pose-model ../../data/models/pose_landmarker_lite.task \
  --max-windows 4
```

Its `abnormal_probability` is the sum of four abnormal **five-class** probabilities. That quantity is not the direct-binary classifier's output and must not be presented as the direct-binary study result. The two commands have different artifact contracts: do not pass a binary-development directory as the five-class command's `--run`.

The five-class adapter also reports its saved normal-forecast error and threshold when available. A missing threshold remains `null`. This compact forecast is separate from the 32-frame, three-seed residual forecasting study. The direct-binary wrapper does not claim to serve that separate forecasting experiment.

## Verification and evidence

Before prediction, the binary adapter checks the sealed selection, refit lineage, primary encoder identity, recorded method/dependency/runtime versions, preparation hashes, classifier and encoder bytes, baseline compatibility and exact pose-model weights. Its report includes the new wrapper's source hash. A mismatch stops execution instead of changing or retraining the model. Restore the recorded environment or create a new declared run when inputs or methods change.

Five focused binary-adapter tests cover class-column alignment and sequence averaging, the exact 0.5 decision boundary, quality refusal, pose-weight mismatch and changed-selection refusal before loading. The underlying extraction adapter has separate tests for one-based indexing, RGB channel order and sampled FPS.

A real clean **training-source** interval completed video decoding, MediaPipe extraction and both classifier calls. The [smoke record](../artifacts/binary-inference-train-smoke.json) pins the source, frames and model identities. For that single window, RF returned abnormal probability 0.22765 and mixed JEPA returned 0.51516, producing different annotation decisions at the fixed threshold. The clip was selected from the training partition before inference; this is execution evidence, not new-person or held-out accuracy. An independent reviewer repeated extraction and inference on the same actual training frames and reproduced both probabilities exactly. The [execution guide](execution.md) and sealed research artifacts distinguish software checks, training demonstrations and final source-level evaluation.
