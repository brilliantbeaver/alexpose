# Experiment 0 practical guide: availability and harmonization

This guide turns Experiment 0 of the cross-protocol perturbation-response study into an executable data-validation workflow. It covers the complete Georgia Tech ground-translation release and an initial development subset of at least two Stanford Dryad participants. It does **not** train S-JEPA, fit a recovery predictor, tune a probabilistic head, or evaluate the study hypothesis.

The objective is narrower and more important:

> Establish that the two releases can be converted into one leakage-safe pelvis-and-feet representation and that the same four recovery outcomes can be computed reliably in both protocols.

Experiment 0 ends with a versioned `PASS` or `FAIL` decision. Modeling begins only after a pass.

## 1. Definition of done

The gate is complete when every expected perturbation trial has a manifest row and the pipeline has produced:

1. an immutable raw-file inventory with checksums;
2. a protocol-specific trial inventory tied to documented participant and intervention metadata;
3. harmonized global and pelvis-relative kinematics;
4. recorded-onset alignment and a strictly pre-onset input window;
5. common foot-contact and recovery-window events;
6. the four primary recovery outcomes or an explicit failure reason;
7. stratified manual-audit figures; and
8. availability summaries overall, by protocol, participant, condition, direction, and magnitude.

Use the following decision rule:

- at least 90% of expected trials in **each protocol** have a valid pre-onset window;
- at least 90% of expected trials in **each protocol** yield each of the four outcomes;
- every prespecified manual-audit trial has the correct perturbation direction, stance side, and recovery contacts, or any discrepancy is resolved and the complete audit is rerun; and
- no failure is silently converted to a zero, mean value, or copied neighboring trial.

Requiring 90% within each protocol is slightly stricter than pooling the two releases. It prevents a large, clean source protocol from hiding unusable target-protocol data.

## 2. What Experiment 0 does not do

Do not load an S-JEPA checkpoint during this experiment. Do not use recovery labels to choose a model. Do not create participant-level train/test performance estimates.

The data flow is:

```text
immutable raw files
        |
        +--> protocol inventory and recorded intervention metadata
        |
        +--> protocol-specific kinematics
                     |
                     v
             common physical schema
                     |
          +----------+-----------+
          |                      |
          v                      v
 strict pre-onset input    post-onset outcomes
          |                      |
          +----------+-----------+
                     v
          availability + manual audit
                     |
                  PASS/FAIL
```

The later S-JEPA bridge consumes a derived view of the successful pre-onset data. It must not dictate whether the underlying physical conversion is correct.

## 3. Configure HAIC paths

Keep raw data and large derived arrays on HAIC rather than inside the Git checkout. Export explicit absolute paths in the interactive HAIC shell:

```bash
export GAVD6_ROOT=/absolute/path/to/gavd6
export GEORGIA_TECH_ROOT=/absolute/path/to/extracted/georgia-tech
export DRYAD_ROOT=/absolute/path/to/extracted/dryad
export CROSS_PROTOCOL_RUN_ROOT=/absolute/path/to/cross-protocol-response
```

Validate them before running any parser:

```bash
test -d "$GAVD6_ROOT"
test -d "$GEORGIA_TECH_ROOT"
test -d "$DRYAD_ROOT"
mkdir -p "$CROSS_PROTOCOL_RUN_ROOT"
cd "$GAVD6_ROOT"
```

Use this run layout:

```text
$CROSS_PROTOCOL_RUN_ROOT/
├── raw-inventory/
├── experiment-0-v1/
│   ├── config/
│   ├── manifests/
│   ├── canonical-trials/
│   ├── qc/
│   ├── reports/
│   └── logs/
└── experiment-0-v2/       # only if a versioned repair is needed
```

Never overwrite `experiment-0-v1` after inspecting its aggregate outcome availability. A repaired event detector or outcome definition belongs in `v2`, with the reason recorded.

Verify the software environment:

```bash
cd "$GAVD6_ROOT"
uv run --no-sync python -c "import ezc3d, numpy, pandas, scipy; print('environment ready')"
```

OpenSim is not required for the first pass. Dryad provides Vicon marker trajectories and redundant processed representations. Start from marker data; reserve OpenSim inverse-kinematics outputs for secondary validation.

## 4. Preserve an immutable raw-file inventory

Before interpreting filenames, inventory exactly what exists. The inventory should contain the dataset version, relative path, size, modification time, suffix, and SHA-256 hash.

The reusable implementation belongs in `src/gavd6_sjepa/data_foundations/`, not in a notebook. A minimal file record is:

```python
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path


@dataclass(frozen=True)
class RawFileRecord:
    protocol: str
    relative_path: str
    size_bytes: int
    sha256: str


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory_tree(protocol: str, root: Path) -> list[dict]:
    root = root.resolve()
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        rows.append(
            asdict(
                RawFileRecord(
                    protocol=protocol,
                    relative_path=path.relative_to(root).as_posix(),
                    size_bytes=path.stat().st_size,
                    sha256=file_sha256(path),
                )
            )
        )
    return rows
```

Run the expensive hashing once, write it atomically, and reuse it. The repository already provides atomic CSV and JSON writers in `shared_infrastructure/artifact_io_operations.py`.

Before writing dataset parsers, summarize suffixes and sample paths:

```python
from collections import Counter
from pathlib import Path

for name, root in {
    "georgia_tech": Path("/absolute/path/to/georgia-tech"),
    "dryad": Path("/absolute/path/to/dryad"),
}.items():
    files = [path for path in root.rglob("*") if path.is_file()]
    print(name, len(files), Counter(path.suffix.lower() for path in files))
    for path in files[:20]:
        print(" ", path.relative_to(root))
```

This discovery step is intentional. The parser should be based on actual labels and metadata, not an assumed archive layout.

## 5. Build the expected-trial ledger

Availability uses documented expected perturbation trials as its denominator, not only files that happen to parse.

Create one row for every expected trial with these minimum fields:

```python
TRIAL_COLUMNS = (
    "protocol",
    "participant_id",
    "session_id",
    "trial_id",
    "condition",
    "trial_order",
    "repeat_number",
    "direction_ap_raw",
    "direction_ml_raw",
    "magnitude_raw",
    "magnitude_units",
    "target_phase",
    "kinematics_relative_path",
    "intervention_relative_path",
    "expected",
)
```

For Dryad, the official release describes 10 participants, four conditions per participant, and 16 perturbations per condition: four directions, two magnitudes, and two repetitions. Each session's `Perturbation Order` table maps row number to the identically numbered Vicon/Speedgoat trial suffix. Check that mapping rather than interpreting the trial suffix as a perturbation class.

Dryad session names encode trial type and condition. Preserve the original session string, then add a separate decoded condition field. Also preserve the release subject identifier exactly; do not silently renumber it to the manuscript identifier.

For Georgia Tech, construct the ledger from the supplied participant/trial metadata and documentation. Confirm the participant count and expected intervention combinations from parsed metadata before fixing a denominator. A missing trial must remain in the ledger with `expected=True` and a failure reason.

Required integrity checks include:

```python
def assert_unique_trials(frame):
    key = ["protocol", "participant_id", "session_id", "trial_id"]
    if frame.duplicated(key).any():
        duplicates = frame.loc[frame.duplicated(key, keep=False), key]
        raise ValueError(f"duplicate trial identities:\n{duplicates.head()}")


def assert_dryad_order_mapping(frame):
    dryad = frame.loc[frame.protocol == "dryad"]
    counts = dryad.groupby(["participant_id", "session_id"]).size()
    if not (counts == 16).all():
        raise ValueError(f"Dryad sessions without 16 expected trials:\n{counts[counts != 16]}")
```

If subject-specific notes document missing or invalid trials, keep those rows and store the documented exclusion reason. Do not reduce the denominator without recording why.

## 6. Define one common physical schema

Use a richer physical intermediate than the eventual model input. A practical common skeleton is:

```python
COMMON7 = (
    "pelvis",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_forefoot",
    "right_forefoot",
)

CHANNELS = ("forward", "vertical_up", "mediolateral_left")
```

Each canonical trial should contain:

```python
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class CanonicalTrial:
    time_s: np.ndarray                 # [T], relative to recorded onset
    global_position_m: np.ndarray      # [T, 7, 3]
    pelvis_relative_m: np.ndarray      # [T, 7, 3]
    valid: np.ndarray                  # [T, 7], bool
    intervention_direction: np.ndarray # [2], forward and ML-left
    intervention_magnitude_raw: float
    leg_length_m: float
    stride_time_s: float
```

Preserve both views:

- `global_position_m` retains pelvis translation needed for pelvis-speed and excursion outcomes;
- `pelvis_relative_m` retains limb configuration and later supplies the S-JEPA/Core11 bridge.

Do not replace missing hips or knees with fabricated coordinates during Experiment 0.

### Landmark construction

Construct pelvis and foot landmarks from documented marker labels. Use dataset-specific alias tables, but make the resulting definitions identical:

- pelvis: center of the available anterior and posterior iliac markers under a locked minimum-marker rule;
- ankle: midpoint of medial and lateral malleolus markers when both exist;
- heel: heel marker;
- forefoot: midpoint of the available medial and lateral forefoot/metatarsal markers;
- foot center for contact and placement: a locked combination of heel and forefoot.

Do not silently fall back from one definition to another. Emit `valid=False` and a reason when the required markers are absent.

## 7. Parse Dryad without assuming MATLAB field positions

Dryad Vicon motion capture is recorded at 100 Hz, and analog signals are recorded at 1000 Hz. The release includes C3D files, MATLAB structures, perturbation-order tables, and OpenSim outputs. Prefer label-based access over numeric columns.

A C3D inspection helper is:

```python
from pathlib import Path
import ezc3d
import numpy as np


def inspect_c3d(path: Path) -> dict:
    archive = ezc3d.c3d(str(path))
    labels = [str(value).strip() for value in archive["parameters"]["POINT"]["LABELS"]["value"]]
    points = np.asarray(archive["data"]["points"])
    analogs = np.asarray(archive["data"]["analogs"])
    return {
        "labels": labels,
        "point_shape": tuple(points.shape),
        "analog_shape": tuple(analogs.shape),
        "point_rate_hz": float(archive["parameters"]["POINT"]["RATE"]["value"][0]),
        "analog_rate_hz": float(archive["parameters"]["ANALOG"]["RATE"]["value"][0]),
    }
```

For `ezc3d`, point arrays are conventionally shaped `[4, marker, frame]`. Convert the first three channels to `[frame, marker, xyz]`, and determine invalid samples from the C3D residual/validity convention plus finite coordinates. Confirm the position units from `POINT:UNITS` and convert once to metres.

Perturbation order supplies direction, body-weight-normalized magnitude, target phase, perturbation ID, and repetition. The primary onset must come from the synchronized processed perturbation signal. The nominal event occurs near 10 seconds and lasts about 300 ms, but `10.0` seconds is a QC expectation—not an acceptable primary onset substitute.

Use the official processing scripts and labels to identify the perturbation-force channel, then implement a documented onset detector. For example:

```python
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class OnsetConfig:
    baseline_seconds: float
    threshold_sigma: float
    minimum_duration_seconds: float


def detect_recorded_onset(
    time_s: np.ndarray,
    force: np.ndarray,
    config: OnsetConfig,
) -> float:
    baseline_stop = time_s[0] + config.baseline_seconds
    baseline = force[time_s < baseline_stop]
    center = np.median(baseline)
    scale = 1.4826 * np.median(np.abs(baseline - center))
    active = np.abs(force - center) > config.threshold_sigma * max(scale, 1e-9)
    # Require a sustained run; do not accept a single threshold crossing.
    required = max(1, round(config.minimum_duration_seconds / np.median(np.diff(time_s))))
    run = np.convolve(active.astype(int), np.ones(required, dtype=int), mode="valid")
    candidates = np.flatnonzero(run == required)
    if not len(candidates):
        raise ValueError("no sustained recorded perturbation onset")
    return float(time_s[candidates[0]])
```

The numbers in `OnsetConfig` are not filled in here deliberately. Select them on prespecified development sessions, compare the detected onset with the processed data and official scripts, freeze the configuration, and then rerun all trials.

## 8. Parse Georgia Tech through the same adapter contract

Georgia Tech supplies marker tables and intervention metadata in CSV/MAT-based files. Begin by printing actual headers and shapes for one file of each type:

```python
from pathlib import Path
import pandas as pd
from scipy.io import whosmat

root = Path("/absolute/path/to/georgia-tech")

for path in sorted(root.rglob("*.csv"))[:10]:
    print(path.relative_to(root))
    print(pd.read_csv(path, nrows=3).columns.tolist())

for path in sorted(root.rglob("*.mat"))[:10]:
    print(path.relative_to(root), whosmat(path))
```

Create a Georgia Tech adapter that returns the same fields as the Dryad adapter:

```python
class ProtocolAdapter:
    def expected_trials(self): ...
    def load_marker_trajectory(self, trial): ...
    def load_recorded_intervention(self, trial): ...
    def load_reference_walking(self, participant): ...
```

Use marker labels rather than fixed column offsets. The released marker subset includes pelvic and foot landmarks sufficient to construct pelvis, ankles, heels, and forefeet. Record which raw labels produced each common landmark in the trial manifest.

As with Dryad, use the recorded ground-translation command or measured platform signal for onset. The intervention label and nominal schedule may validate the result, but motion response must never define onset.

## 9. Separate input preprocessing from outcome preprocessing

This separation is the central leakage defense:

```python
@dataclass(frozen=True)
class AlignedTrial:
    pre_time_s: np.ndarray
    pre_positions_m: np.ndarray
    pre_valid: np.ndarray
    recovery_time_s: np.ndarray
    recovery_positions_m: np.ndarray
    recovery_valid: np.ndarray


def split_at_recorded_onset(time_s, positions_m, valid, onset_s):
    relative = time_s - onset_s
    pre = (relative >= -1.0) & (relative < 0.0)
    recovery = relative >= 0.0
    if not pre.any():
        raise ValueError("no samples in the required pre-onset window")
    return AlignedTrial(
        pre_time_s=relative[pre],
        pre_positions_m=positions_m[pre],
        pre_valid=valid[pre],
        recovery_time_s=relative[recovery],
        recovery_positions_m=positions_m[recovery],
        recovery_valid=valid[recovery],
    )
```

Dryad documentation states that processed marker data were commonly filtered with a zero-lag Butterworth filter. Zero-lag filtering the complete trial can blend post-onset motion into samples immediately before onset. For the model input, either start from unfiltered marker data or truncate at onset before applying any acausal filter. A simple safe default is a causal second-order-sections filter:

```python
from scipy.signal import butter, sosfilt


def causal_lowpass_pre_onset(values, sample_rate_hz, cutoff_hz=10.0):
    sos = butter(4, cutoff_hz, btype="low", fs=sample_rate_hz, output="sos")
    return sosfilt(sos, values, axis=0)
```

Do not automatically accept `10 Hz` as final. Inspect spectra and marker noise on development participants, then freeze one cutoff supported by both protocols. Outcome trajectories may use a separately documented post-onset filter because they never become model inputs.

Add this mandatory regression test:

```python
def test_post_onset_changes_cannot_change_model_input():
    original = make_synthetic_trial()
    modified = original.copy()
    modified.positions_m[modified.time_s >= modified.onset_s] += 1000.0

    first = build_pre_onset_input(original)
    second = build_pre_onset_input(modified)

    np.testing.assert_array_equal(first.coordinates, second.coordinates)
    np.testing.assert_array_equal(first.valid, second.valid)
```

## 10. Construct a common coordinate frame

Do not infer treadmill heading from net pelvis displacement: on a treadmill, the pelvis remains approximately stationary in the laboratory. Use the documented laboratory axes and pelvis orientation markers, estimated only from pre-onset data.

Choose a physical right-handed basis with positive forward, positive left, and positive up:

```python
import numpy as np


def normalize(vector, epsilon=1e-8):
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm < epsilon:
        raise ValueError("undefined coordinate axis")
    return vector / norm


def make_body_axes(forward_world, up_world):
    up = normalize(up_world)
    forward_horizontal = forward_world - np.dot(forward_world, up) * up
    forward = normalize(forward_horizontal)
    left = normalize(np.cross(up, forward))
    # Physical order [forward, left, up] is right-handed because forward x left = up.
    return forward, up, left


def project_positions(position_world, forward, up, left):
    channels = np.stack([forward, up, left], axis=1)
    return position_world @ channels
```

The stored channel order `[forward, vertical_up, mediolateral_left]` matches the existing Core11 convention. Convert the intervention direction through the same basis. Dryad's raw mediolateral sign convention must therefore be transformed, not copied.

Store two translations:

```python
def make_global_and_relative_views(position_body, pre_onset, pelvis_index=0):
    pelvis = position_body[:, pelvis_index]
    pre_indices = np.flatnonzero(pre_onset)
    if not len(pre_indices):
        raise ValueError("no pre-onset pelvis sample is available as the origin")
    origin = pelvis[pre_indices[-1]].copy()
    global_view = position_body - origin[None, None, :]
    relative_view = position_body - pelvis[:, None, :]
    return global_view, relative_view
```

For a complete trial spanning pre- and post-onset time, use one origin and one orientation fixed from the pre-onset segment. Do not recompute the frame after perturbation.

### Synthetic sign audit

Construct a toy trajectory, apply a known mediolateral reflection, and require:

- forward and vertical values remain unchanged;
- mediolateral coordinates change sign;
- left/right landmark identities swap;
- the intervention direction changes consistently; and
- projected foot-placement outcome changes according to the same transformation.

This catches a class of failures that can produce plausible plots but reverse the scientific interpretation.

## 11. Resample only after the native-rate audit

First validate each protocol at its native sampling rate. Then select the highest rate reliably supported by both releases. Dryad motion capture is documented at 100 Hz; confirm the Georgia Tech rate from parsed metadata before choosing 100 Hz as the common analysis rate.

Use anti-alias filtering when downsampling. A validity value at a new timestamp is true only if the required source brackets are valid. Never interpolate across onset or across a long marker gap.

Record two independent eligibility flags:

```text
analysis_pre_window_valid    # the protocol's [-1.0, 0.0) window
sjepa_64x30_window_valid     # later 64-frame, 30-Hz frozen-encoder input
```

The second flag does not control the Experiment 0 pass. It identifies whether the current S-JEPA temporal contract will require a later protocol decision.

## 12. Detect contacts and define the recovery window

Use marker-based events as the shared primary detector. Ground-reaction force may serve as an independent validation reference when available, but the primary event definition should not change silently by protocol.

Represent detector settings explicitly:

```python
@dataclass(frozen=True)
class ContactConfig:
    maximum_foot_height_m: float
    maximum_vertical_speed_mps: float
    minimum_contact_seconds: float
    minimum_swing_seconds: float
```

The detector should return events, confidence, and a failure reason—not only a frame index:

```python
@dataclass(frozen=True)
class ContactEvent:
    time_s: float
    side: str
    confidence: float


@dataclass(frozen=True)
class RecoveryEvents:
    stance_side_at_onset: str
    first_contact: ContactEvent
    second_contact: ContactEvent
    recovery_stop_s: float
```

Fit marker thresholds on prespecified development trials by comparison with force-derived contacts. Freeze one configuration, then evaluate agreement on a separate manual-audit sample. Report timing error and side agreement rather than only the number of detected contacts.

Define the recovery window as `[0, second recovery contact]`, subject to a locked maximum observation duration. If the second contact is outside the recording or low-confidence, mark outcomes requiring that window unavailable.

## 13. Build participant reference templates

Peak speed deviation and stabilization require an expected unperturbed pelvis-speed trajectory. Prefer each participant's explicitly recorded unperturbed or pre-perturbation walking trial. Dryad includes approximately three-minute pre- and post-perturbation walks; use the pre-perturbation recording only for the prediction-time reference.

The template builder should:

1. detect valid strides;
2. exclude perturbation intervals;
3. phase-normalize strides;
4. use a robust median template;
5. store the number of contributing strides; and
6. reject templates below a locked minimum stride count.

This participant-specific reference is allowed because it is observable before the perturbation outcome. Post-perturbation walking must not contribute to a test participant's baseline.

If Georgia Tech does not provide an equivalent reference walk, test whether sufficient clean pre-onset strides exist across that participant's trials. Do not silently use a different semantic baseline; record the difference and determine whether one shared outcome definition remains defensible.

## 14. Compute the four outcomes

Keep each outcome implementation pure: arrays and frozen configuration enter; a value, validity flag, and reason leave.

```python
@dataclass(frozen=True)
class OutcomeValue:
    value: float
    valid: bool
    reason: str | None


@dataclass(frozen=True)
class RecoveryOutcomeVector:
    foot_placement: OutcomeValue
    peak_speed_deviation: OutcomeValue
    stabilization_time: OutcomeValue
    maximum_ml_excursion: OutcomeValue
```

### 14.1 First recovery-foot placement

At the first valid post-onset contact, compute foot center relative to pelvis, project the horizontal displacement onto the harmonized perturbation direction, and divide by leg length:

```python
def projected_foot_placement(foot, pelvis, direction_ap_ml, leg_length_m):
    displacement = np.array([foot[0] - pelvis[0], foot[2] - pelvis[2]])
    direction = direction_ap_ml / np.linalg.norm(direction_ap_ml)
    return float(np.dot(displacement, direction) / leg_length_m)
```

With channel order `[forward, up, ML-left]`, horizontal components are indices `0` and `2`.

### 14.2 Peak pelvis-speed deviation

Differentiate the heading-aligned global pelvis trajectory, evaluate its horizontal speed against the participant's phase-matched pre-perturbation template, and take the largest absolute deviation within the recovery window. Divide by baseline walking speed, using a frozen numerical floor.

### 14.3 Time to stabilization

Find the earliest post-onset time at which speed enters the frozen tolerance band around the phase template and remains there for the complete dwell interval. If the dwell interval extends past valid recording time, return invalid/censored; do not use the last recorded time as if stabilization occurred.

### 14.4 Maximum mediolateral pelvis excursion

Fit or otherwise lock an extrapolated pre-onset pelvis path using only pre-onset samples. Within the recovery window, take the maximum absolute mediolateral deviation from that path and divide by leg length.

Store unnormalized diagnostic quantities alongside the normalized primary values. They make unit and scale mistakes much easier to detect.

## 15. Pre-register the manual audit sample

Choose the audit rows before examining aggregate outcome success. Use a fixed seed and stratify across protocol, direction, magnitude, condition, and participant.

```python
def stratified_audit_sample(manifest, seed=1701):
    strata = ["protocol", "direction_label", "magnitude_label", "condition"]
    eligible = manifest.loc[manifest.expected].copy()
    return (
        eligible.groupby(strata, dropna=False, group_keys=False)
        .sample(n=1, random_state=seed)
        .sort_values(strata + ["participant_id", "trial_id"])
        .reset_index(drop=True)
    )
```

If a stratum contains too few trials, include all of them. Do not replace an inconvenient failed audit trial with a cleaner example.

For every selected trial, generate a multi-panel figure containing:

1. pelvis and bilateral foot paths in the horizontal plane;
2. the positive forward and mediolateral axes plus perturbation arrow;
3. foot heights and detected contacts;
4. pelvis speed, reference template, tolerance band, and stabilization event;
5. mediolateral pelvis trajectory and extrapolated path; and
6. textual metadata, outcome values, validity flags, and warnings.

The reviewer should record direction, stance, first contact, second contact, and overall acceptability in a CSV. Free-text corrections are comments, not hidden data edits.

## 16. Compute availability and the gate decision

Use expected trials as the denominator and report each outcome independently:

```python
OUTCOME_FLAGS = (
    "foot_placement_valid",
    "peak_speed_deviation_valid",
    "stabilization_time_valid",
    "maximum_ml_excursion_valid",
)


def availability_table(frame):
    expected = frame.loc[frame.expected].copy()
    rows = []
    for protocol, group in expected.groupby("protocol"):
        row = {
            "protocol": protocol,
            "expected_trials": len(group),
            "parsed_fraction": float(group.parsed.mean()),
            "pre_window_fraction": float(group.pre_window_valid.mean()),
        }
        row.update({name: float(group[name].mean()) for name in OUTCOME_FLAGS})
        rows.append(row)
    return pd.DataFrame(rows)


def gate_passes(summary, audit):
    fractions = ["pre_window_fraction", *OUTCOME_FLAGS]
    numerical_pass = bool((summary[fractions] >= 0.90).all().all())
    manual_pass = bool(
        audit[["direction_ok", "stance_ok", "contacts_ok", "overall_ok"]]
        .fillna(False)
        .all()
        .all()
    )
    return numerical_pass and manual_pass
```

Also calculate the same fractions by participant, condition, direction, and magnitude. The formal threshold is protocol-level, but a concentrated failure can still invalidate interpretation. For example, 95% overall availability is not reassuring if nearly every posterior perturbation fails.

The final decision JSON should contain:

```json
{
  "experiment": "availability-and-harmonization",
  "version": "v1",
  "passed": false,
  "threshold": 0.9,
  "protocol_level_requirement": true,
  "manual_audit_passed": false,
  "repair_allowed_before_modeling": true,
  "modeling_allowed": false
}
```

Set these fields from computed results; the example is deliberately a safe default.

## 17. Required tests

Create focused tests under `tests/`:

| Test file | Required behavior |
|---|---|
| `test_perturbation_raw_inventory.py` | Deterministic paths and checksums; missing files remain visible |
| `test_georgia_tech_perturbation_conversion.py` | Metadata/marker parsing, units, identities, onset mapping |
| `test_dryad_perturbation_conversion.py` | Session/order mapping, C3D labels, units, recorded onset |
| `test_perturbation_common_schema.py` | Shapes, channel order, right-handed axes, validity semantics |
| `test_perturbation_recovery_events.py` | Stance and contact events on analytical trajectories |
| `test_perturbation_recovery_outcomes.py` | Exact values and censoring for all four outcomes |
| `test_perturbation_experiment_zero.py` | End-to-end synthetic smoke test and gate logic |

Mandatory failure cases include:

- duplicated participant/session/trial IDs;
- missing intervention-order row;
- nonmonotonic timestamps;
- unknown length or force units;
- missing pelvis or foot markers;
- onset outside the recording;
- fewer than one second of valid pre-onset data;
- changes after onset altering the input;
- ambiguous coordinate orientation;
- first or second recovery contact missing;
- stabilization censored by recording end; and
- invalid outcomes being serialized as ordinary numeric values.

Run locally against synthetic fixtures before submitting HAIC jobs:

```bash
cd "$GAVD6_ROOT"
uv run --no-sync python -m unittest discover -s tests -p 'test_perturbation_*.py'
```

## 18. Suggested implementation ownership and commands

Follow the repository's current ownership structure:

```text
src/gavd6_sjepa/data_foundations/
├── perturbation_raw_inventory.py
├── georgia_tech_perturbation_conversion.py
├── dryad_perturbation_conversion.py
└── perturbation_common_schema.py

src/gavd6_sjepa/research_directions/cross_protocol_response/
├── perturbation_recovery_events.py
├── perturbation_recovery_outcomes.py
├── perturbation_availability_audit.py
└── experiment_zero_entrypoint.py

slurm/cross-protocol-response/
├── 01-inventory-and-validate.sbatch
├── 02-convert-trials.sbatch
└── 03-build-experiment-zero-report.sbatch
```

Add lazy CLI routes only after the functions are tested. A useful intended interface is:

```bash
uv run --no-sync gavd6 response inventory --help
uv run --no-sync gavd6 response convert --help
uv run --no-sync gavd6 response experiment-zero --help
```

These commands are proposed interfaces; they do not exist at the time this guide is written.

Use a CPU Slurm array indexed by participant or session for conversion. Experiment 0 does not require an H100. Each task should write to a unique output subtree, and the final aggregation job should refuse to run until every expected task has either a success artifact or an explicit failure artifact.

## 19. Recommended execution order on HAIC

1. Inventory and hash both raw roots.
2. Parse documentation, subject notes, trial-order tables, and marker labels.
3. Build the complete Georgia Tech expected-trial ledger.
4. Build the ledger for two prespecified Dryad development participants.
5. Implement and test each protocol adapter on one trial.
6. Confirm recorded-onset alignment on a small stratified set.
7. Freeze landmark definitions, coordinate signs, filtering, and event parameters.
8. Convert all Georgia Tech trials and the two Dryad participants.
9. Generate the four outcomes and every exclusion reason.
10. Generate and manually score the prespecified QC sample.
11. Compute protocol- and stratum-level availability.
12. Write the immutable `v1` report and gate decision.
13. If the gate fails, classify the cause before changing anything:
    - missing/unavailable source data: stop or narrow the claim;
    - parser or synchronization defect: repair, version, and rerun;
    - incompatible common landmark: revise the schema before modeling;
    - unreliable outcome definition: repair using development data or stop;
    - only the S-JEPA 64-frame flag fails: resolve the encoder bridge later without invalidating the physical harmonization result.

## 20. Final artifact checklist

The completed run should contain:

```text
experiment-0-v1/
├── config/
│   ├── landmark-contract.json
│   ├── coordinate-contract.json
│   ├── onset-config.json
│   ├── contact-config.json
│   └── outcome-config.json
├── manifests/
│   ├── raw-files.csv
│   ├── expected-trials.csv
│   ├── canonical-trials.csv
│   └── exclusions.csv
├── canonical-trials/
│   └── <protocol>/<participant>/<trial>.npz
├── qc/
│   ├── audit-sample.csv
│   ├── manual-review.csv
│   └── figures/
└── reports/
    ├── availability-by-protocol.csv
    ├── availability-by-stratum.csv
    ├── contact-agreement.csv
    ├── experiment-zero-report.md
    └── gate-decision.json
```

The report must state what was expected, what was found, what failed, which definitions were frozen, whether the 90% requirements passed, and whether Experiment 1 is authorized. A failed gate is a valid scientific result; an undocumented repair is not.

## Official release references

- [Georgia Tech: Biomechanics of locomotion during ground translation perturbations](https://repository.gatech.edu/entities/publication/73a7c133-6535-4a88-b81e-5c39df5efb3e)
- [Dryad: Detecting artificially impaired balance in human locomotion](https://datadryad.org/dataset/doi:10.5061/dryad.cnp5hqch3)
- [Dryad processing code](https://github.com/mraitor/balanceMetric)
