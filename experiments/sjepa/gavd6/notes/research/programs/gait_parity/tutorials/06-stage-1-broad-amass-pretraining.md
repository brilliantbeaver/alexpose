# Stage 1 tutorial: broad AMASS pretraining

Stage 1 will eventually teach each GaitParity S-JEPA variant general human-motion structure before the training distribution is narrowed to walking. This tutorial deliberately uses only files that are present in the current `gavd6` checkout.

## Read this first: the current repository boundary

| Component | Present now? | What it supports |
| --- | --- | --- |
| `amass_raw_inventory.csv` | Yes, locally | Auditing and selecting raw AMASS sequences |
| `scripts/make_amass_inventory.py` | Yes | Rebuilding the raw inventory |
| `scripts/convert_amass_core11.py` | Yes | Converting AMASS SMPL+H/DMPL sequences to the frozen core-11 representation |
| `tests/test_convert_amass_core11.py` | Yes | Testing converter contracts that do not require licensed body-model files |
| `src/gavd6_sjepa/gait_parity_jepa.py` | Yes | The existing 33-landmark GAVD feasibility experiment only |
| Extracted AMASS NPZ corpus | Not stored in Git | Must be staged separately and match the inventory |
| Licensed SMPL-H and DMPL model files | Not stored in Git | Must be obtained and staged under the applicable licenses |
| Production subject-split helper | No | Use the auditable inline procedure in section 4 |
| Core-11 window dataset and loader | No | Must be implemented before AMASS training |
| Core-11 JEPA trainer and configs | No | Must be implemented before AMASS training |
| Stage 1 checkpoint audit tool | No | Must be implemented before a Stage 1 checkpoint can be accepted |

The runnable code path in this tutorial therefore ends with an audited core-11 conversion, provided the licensed AMASS and body-model inputs are already available. If they are absent, the current checkout can still stage the inventory, create the subject-registry scaffold, and run converter unit tests, but it cannot synthesize those licensed inputs. This tutorial does **not** claim that pretraining can start from the current checkout. Commands shown through section 5 use software that exists now; section 6 marks the hard stop and specifies the remaining implementation work without inventing command-line interfaces.

Do not feed raw AMASS or converted core-11 arrays directly to `src/gavd6_sjepa/gait_parity_jepa.py`. Use `src/gavd6_sjepa/amass_core11_jepa.py`, which defines the Core11 joint and reflection contract.

## Fastest valid path with the files present

1. Verify the HAIC environment and licensed body-model files.
2. Select one uncomplicated female walk and one uncomplicated male walk from the existing inventory.
3. Create a temporary two-row smoke split with the inline code in section 5.5.
4. Convert those two sequences with `scripts/convert_amass_core11.py` and complete the numerical, anatomical, replay, and visual checks.
5. Manually audit subject identities, then generate the production split and eligible inventory with the inline code in section 4.
6. Submit full conversion to the `hai` batch partition.
7. Stop at the section 6 handoff until a core-11 loader, model, trainer, and audit path actually exist.

Do not wait for `hai-interactive` if its node is busy or reserved. The environment check and conversion can run as `sbatch` jobs in `hai`; interactive access is convenient for debugging, not a prerequisite.

The current inventory snapshot contains 10,941 sequences from 308 candidate subjects. Resampling it to 30 Hz is expected to yield approximately 4.15 million frames. A future length-64, stride-32 index would contain roughly 124,000 windows, but no such index is created by the current repository.

## 1. Decide where data and artifacts live

HAIC provides a small home directory and larger scratch storage. Keep the repository and lightweight configuration under your user area; keep AMASS, converted tensors, and future checkpoints under `/hai/scratch`.

After connecting through the Stanford VPN when required:

```bash
ssh YOUR_SUNETID@haic.stanford.edu
```

The HAIC head node is for lightweight setup and job submission. Do not preprocess AMASS or train a model there. Check the [official HAIC page](https://www-dev.cs.stanford.edu/haic) before a large launch because partitions, wall-time limits, and quotas can change.

Set project-specific paths without repurposing `$HOME`:

```bash
export GAVD6_ROOT=/absolute/path/to/alexpose/experiments/sjepa/gavd6
export AMASS_EXTRACTED_ROOT=/hai/scratch/YOUR_SUNETID/amass/extracted
export AMASS_RUN_ROOT=/hai/scratch/YOUR_SUNETID/gait-parity/amass-v1
```

Create the manifest directory and copy the inventory from the checkout if it is present there:

```bash
mkdir -p "$AMASS_RUN_ROOT/manifests"
if test -f "$GAVD6_ROOT/amass_raw_inventory.csv"; then
  cp "$GAVD6_ROOT/amass_raw_inventory.csv" \
    "$AMASS_RUN_ROOT/manifests/amass_raw_inventory.csv"
fi
test -f "$AMASS_RUN_ROOT/manifests/amass_raw_inventory.csv"
```

If the final `test` fails because the manifest exists only on your workstation, upload that existing file before continuing. From the workstation, run:

```bash
scp amass_raw_inventory.csv \
  YOUR_SUNETID@haic.stanford.edu:/hai/scratch/YOUR_SUNETID/gait-parity/amass-v1/manifests/
```

Do not rebuild the inventory merely to change its location.

## 2. Discover your live HAIC account

HAIC requires `--account` for `srun` and `sbatch`. Do not copy another research group's account name.

Use the cluster's account-reporting command if available, or inspect Slurm associations:

```bash
sacctmgr show associations where user="$USER" format=User,Account,Partition
sinfo -s
```

Save the correct value:

```bash
export HAIC_ACCOUNT=YOUR_APPROVED_ACCOUNT
```

The normal batch partition is `hai`. Use `hai-interactive` only for short debugging sessions. `hai-lo` is lower priority and may be preempted.

## 3. Create and verify the software environment without waiting unnecessarily

If `hai-interactive` can start promptly, request one short session:

```bash
srun \
  --account="$HAIC_ACCOUNT" \
  --partition=hai-interactive \
  --gres=gpu:1 \
  --cpus-per-task=8 \
  --mem=64G \
  --time=02:00:00 \
  --pty bash
```

Inside that allocation:

```bash
cd "$GAVD6_ROOT"
uv sync
uv run --no-sync python -m unittest tests.test_convert_amass_core11
uv run python -c 'import torch; print(torch.__version__); print(torch.cuda.is_available())'
nvidia-smi
```

Do not submit a full run until `torch.cuda.is_available()` prints `True` in the allocated GPU shell.

If the interactive request remains pending because its requested resources are unavailable or reserved, cancel only that unused request and submit the same checks to the normal batch partition. Create the job file in the run directory, not in the source tree:

```bash
mkdir -p "$AMASS_RUN_ROOT/jobs" "$GAVD6_ROOT/data/amass/outputs/logs"

cat > "$AMASS_RUN_ROOT/jobs/check-gpu-env.sbatch" <<'SBATCH'
#!/usr/bin/env bash
#SBATCH --job-name=gp-env
#SBATCH --partition=hai
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/hai/scratch/%u/alexpose/experiments/sjepa/gavd6/data/amass/outputs/logs/gp-env-%j.out
#SBATCH --error=/hai/scratch/%u/alexpose/experiments/sjepa/gavd6/data/amass/outputs/logs/gp-env-%j.err

set -euo pipefail
: "${GAVD6_ROOT:?Set GAVD6_ROOT when submitting}"

cd "$GAVD6_ROOT"
uv sync
uv run --no-sync python -m unittest tests.test_convert_amass_core11
uv run --no-sync python - <<'PY'
import torch

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
assert torch.cuda.is_available()
PY
nvidia-smi
SBATCH

GPU_ENV_JOB=$(sbatch \
  --parsable \
  --account="$HAIC_ACCOUNT" \
  --export=ALL \
  "$AMASS_RUN_ROOT/jobs/check-gpu-env.sbatch")
echo "Submitted GPU environment check: $GPU_ENV_JOB"
```

Monitor the job from the head node:

```bash
squeue -j "$GPU_ENV_JOB"
sacct -j "$GPU_ENV_JOB" --format=JobID,State,Elapsed,AllocTRES,ExitCode
```

`human-body-prior` is declared in the current `pyproject.toml` and lockfile, so `uv sync` should install it. Never run conversion until section 5.2 confirms that the exact Python environment used by the job can import it.

## 4. Freeze subject identities before cutting windows

The raw inventory contains `subject_id_candidate`. Audit it for each AMASS source and produce an immutable subject table:

```text
$AMASS_RUN_ROOT/manifests/amass_subject_registry.csv
```

Required columns:

```text
source_dataset
subject_id_candidate
audited_subject_id
identity
identity_audit_status
known_downstream_overlap
excluded
exclusion_reason
```

Define:

```python
identity = source_dataset + "::" + audited_subject_id
```

Create a pending registry scaffold from the existing inventory. This command refuses to overwrite a registry so that a later rerun cannot erase manual decisions:

```bash
uv run --no-sync python - <<'PY'
import os
from pathlib import Path

import pandas as pd

manifest_root = Path(os.environ["AMASS_RUN_ROOT"]) / "manifests"
inventory = pd.read_csv(
    manifest_root / "amass_raw_inventory.csv", keep_default_na=False
)
output_path = manifest_root / "amass_subject_registry.csv"
assert not output_path.exists(), f"refusing to overwrite {output_path}"

registry = (
    inventory.loc[inventory["status"].eq("ok"), ["source_dataset", "subject_id_candidate"]]
    .drop_duplicates()
    .sort_values(["source_dataset", "subject_id_candidate"])
    .reset_index(drop=True)
)
registry["audited_subject_id"] = ""
registry["identity"] = ""
registry["identity_audit_status"] = "pending"
registry["known_downstream_overlap"] = ""
registry["excluded"] = ""
registry["exclusion_reason"] = ""

temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
registry.to_csv(temporary_path, index=False)
os.replace(temporary_path, output_path)
print(f"Wrote {len(registry)} pending subject rows to {output_path}")
PY
```

Open that CSV and complete every pending field from source documentation and downstream-overlap records. Do not run the split procedure below while any boolean or audit field remains blank.

Exclude every known downstream person or overlapping source dataset before splitting. Do not infer these decisions from filenames alone: inspect every source's subject convention and record the decision in the registry. Use `identity_audit_status=approved` only after that inspection. Any row that is not approved, or is known to overlap downstream data, must have `excluded=true` and a nonempty reason.

There is no subject-split helper in the current checkout. After manually completing the registry, run the following self-contained procedure from `gavd6`. It creates both:

- `amass_subject_splits.csv`, the deterministic identity-level split; and
- `amass_raw_inventory_eligible.csv`, the structurally valid, audited subset accepted by full conversion.

Keeping a separate eligible inventory matters because the converter correctly refuses inventory subjects that have no split assignment.

```bash
uv run --no-sync python - <<'PY'
import hashlib
import os
from pathlib import Path

import pandas as pd

run_root = Path(os.environ["AMASS_RUN_ROOT"])
manifest_root = run_root / "manifests"
inventory_path = manifest_root / "amass_raw_inventory.csv"
registry_path = manifest_root / "amass_subject_registry.csv"
split_path = manifest_root / "amass_subject_splits.csv"
eligible_inventory_path = manifest_root / "amass_raw_inventory_eligible.csv"
seed = "gait-parity-amass-v1"

inventory = pd.read_csv(inventory_path, keep_default_na=False)
registry = pd.read_csv(registry_path, keep_default_na=False)

required_inventory = {"source_dataset", "subject_id_candidate", "status"}
required_registry = {
    "source_dataset",
    "subject_id_candidate",
    "audited_subject_id",
    "identity",
    "identity_audit_status",
    "known_downstream_overlap",
    "excluded",
    "exclusion_reason",
}
assert required_inventory <= set(inventory.columns), (
    f"inventory missing {sorted(required_inventory - set(inventory.columns))}"
)
assert required_registry <= set(registry.columns), (
    f"registry missing {sorted(required_registry - set(registry.columns))}"
)


def parse_bool(series: pd.Series, name: str) -> pd.Series:
    normalized = series.astype(str).str.strip().str.lower()
    allowed = {"true", "false", "1", "0", "yes", "no"}
    unknown = sorted(set(normalized) - allowed)
    assert not unknown, f"{name} contains unsupported values: {unknown}"
    return normalized.isin({"true", "1", "yes"})


registry = registry.copy()
registry["subject_id_candidate"] = registry["subject_id_candidate"].astype(str).str.strip()
registry["audited_subject_id"] = registry["audited_subject_id"].astype(str).str.strip()
registry["identity"] = registry["identity"].astype(str).str.strip()
registry["identity_audit_status"] = (
    registry["identity_audit_status"].astype(str).str.strip().str.lower()
)
registry["is_excluded"] = parse_bool(registry["excluded"], "excluded")
registry["has_downstream_overlap"] = parse_bool(
    registry["known_downstream_overlap"], "known_downstream_overlap"
)

assert registry["subject_id_candidate"].ne("").all(), "blank subject candidate"
assert registry["audited_subject_id"].ne("").all(), "blank audited subject ID"
assert registry["identity"].ne("").all(), "blank audited identity"
expected_identity = (
    registry["source_dataset"].astype(str).str.strip()
    + "::"
    + registry["audited_subject_id"]
)
assert registry["identity"].eq(expected_identity).all(), (
    "identity must equal source_dataset + '::' + audited_subject_id"
)
assert not registry.duplicated(["source_dataset", "subject_id_candidate"]).any(), (
    "registry has duplicate source/subject rows"
)
candidate_sources = registry.groupby("subject_id_candidate")["source_dataset"].nunique()
assert candidate_sources.le(1).all(), (
    "subject candidates reused across sources are ambiguous to the current converter"
)
identity_counts = registry.groupby("subject_id_candidate")["identity"].nunique()
assert identity_counts.le(1).all(), "one candidate maps to multiple identities"
assert not (registry["has_downstream_overlap"] & ~registry["is_excluded"]).any(), (
    "every downstream overlap must be excluded"
)
assert not (
    registry["is_excluded"] & registry["exclusion_reason"].astype(str).str.strip().eq("")
).any(), "every exclusion needs a reason"
assert not (
    registry["identity_audit_status"].ne("approved") & ~registry["is_excluded"]
).any(), "every non-approved identity must be excluded"

ok_inventory = inventory.loc[inventory["status"].eq("ok")].copy()
inventory_candidates = set(ok_inventory["subject_id_candidate"].astype(str))
registry_candidates = set(registry["subject_id_candidate"])
missing_registry = sorted(inventory_candidates - registry_candidates)
assert not missing_registry, (
    f"{len(missing_registry)} inventory candidates lack registry decisions; "
    f"first values: {missing_registry[:10]}"
)

eligible_registry = registry.loc[
    registry["identity_audit_status"].eq("approved") & ~registry["is_excluded"]
].copy()
eligible_candidates = set(eligible_registry["subject_id_candidate"])

identities = sorted(set(eligible_registry["identity"]))
assert len(identities) >= 10, "too few eligible identities for an 80/10/10 split"
ranked = sorted(
    identities,
    key=lambda identity: hashlib.sha256(
        f"{seed}\0{identity}".encode("utf-8")
    ).hexdigest(),
)
n_validation = max(1, round(0.10 * len(ranked)))
n_test = max(1, round(0.10 * len(ranked)))
assert len(ranked) - n_validation - n_test >= 1, "split has no training identities"

split_by_identity = {identity: "train" for identity in ranked}
for identity in ranked[:n_validation]:
    split_by_identity[identity] = "validation"
for identity in ranked[n_validation : n_validation + n_test]:
    split_by_identity[identity] = "test"

splits = eligible_registry[
    ["source_dataset", "subject_id_candidate", "audited_subject_id", "identity"]
].copy()
splits["split"] = splits["identity"].map(split_by_identity)
splits["split_seed"] = seed
assert splits.groupby("identity")["split"].nunique().max() == 1

eligible_inventory = ok_inventory.loc[
    ok_inventory["subject_id_candidate"].astype(str).isin(eligible_candidates)
].copy()
assert set(eligible_inventory["subject_id_candidate"].astype(str)) <= set(
    splits["subject_id_candidate"]
)

manifest_root.mkdir(parents=True, exist_ok=True)
for frame, output_path in (
    (splits.sort_values(["split", "identity", "subject_id_candidate"]), split_path),
    (eligible_inventory, eligible_inventory_path),
):
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    frame.to_csv(temporary_path, index=False)
    os.replace(temporary_path, output_path)

print("eligible sequences:", len(eligible_inventory))
print("eligible identities:", len(ranked))
print(splits.groupby("split")["identity"].nunique().sort_index())
print("wrote:", split_path)
print("wrote:", eligible_inventory_path)
PY
```

Review the printed counts and version or hash the registry, split, and eligible inventory together. If the registry changes, regenerate both outputs and treat the result as a new dataset lineage.

### 4.1 Conversion-only exception: two temporary train-only identities

The two-sequence converter smoke test does not estimate model performance and supports no model comparison. It may therefore use a separate, clearly named two-subject split containing only the female and male smoke subjects, both assigned to `train`. This tests converter mechanics while the full identity audit proceeds.

The temporary split must satisfy all of these rules:

- its filename contains `smoke`;
- it contains only the two selected smoke subjects;
- both rows use `split=train`;
- its identities remain source-qualified;
- it is never supplied to a full conversion, window index, training run, validation run, model-selection run, or reported experiment.

Section 5.5 creates this temporary split together with the smoke inventory. Point the converter at it explicitly:

```bash
export AMASS_SUBJECT_SPLITS="$AMASS_RUN_ROOT/manifests/amass_subject_splits_smoke.csv"
```

For production conversion, replace that value with the frozen audited split:

```bash
export AMASS_SUBJECT_SPLITS="$AMASS_RUN_ROOT/manifests/amass_subject_splits.csv"
test -f "$AMASS_SUBJECT_SPLITS"
```

The temporary split is converter test data, not a shortcut around the production identity audit.

## 5. Smoke-test the AMASS-to-core-11 conversion on HAIC

Raw AMASS files contain Extended SMPL+H parameters rather than GaitParity joint coordinates. `scripts/convert_amass_core11.py` applies the licensed gender-matched body model and DMPL model, extracts the ordered core-11 skeleton, constructs the frozen body frame, normalizes by robust leg length, and resamples the entire sequence to 30 Hz.

Do not begin with the full inventory. First convert exactly one female walking sequence and one male walking sequence. This exercises both licensed body models and both DMPL models while keeping the output small enough to inspect manually.

### 5.1 Confirm that you are on a GPU compute node

Run these checks from the `gavd6` directory inside either the interactive allocation or a short batch allocation created in section 3:

```bash
pwd
test "$(basename "$PWD")" = gavd6
nvidia-smi

uv run --no-sync python - <<'PY'
import torch

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
assert torch.cuda.is_available(), "Run this smoke test inside a HAIC GPU allocation."
PY
```

`uv run --no-sync` is intentional after the environment check: the compute job uses the already-synchronized lockfile environment instead of resolving dependencies again.

### 5.2 Verify the converter backend

The converter uses AMASS's reference `human_body_prior.body_model.BodyModel`, including 16 beta coefficients and 8 DMPL coefficients. Confirm that the package imports from the same environment that will run the converter:

```bash
uv run --no-sync python - <<'PY'
import inspect
import human_body_prior
from human_body_prior.body_model.body_model import BodyModel

print("human_body_prior package:", inspect.getfile(human_body_prior))
print("BodyModel implementation:", inspect.getfile(BodyModel))
PY
```

If that import fails, stop and inspect the preceding `uv sync` result; do not silently switch package sources inside a conversion job. Record the locked package version in the run ledger. The converter also records hashes of the imported BodyModel and linear-blend-skinning source files.

### 5.3 Verify all four licensed model files

Keep licensed model files outside Git. Set their root and require this exact layout:

```text
body_models/
├── smplh/
│   ├── female/model.npz
│   └── male/model.npz
└── dmpls/
    ├── female/model.npz
    └── male/model.npz
```

On HAIC:

```bash
export AMASS_BODY_MODEL_ROOT="/hai/scratch/$USER/body_models"

for gender in female male; do
  test -f "$AMASS_BODY_MODEL_ROOT/smplh/$gender/model.npz"
  test -f "$AMASS_BODY_MODEL_ROOT/dmpls/$gender/model.npz"
done

find "$AMASS_BODY_MODEL_ROOT" -type f -name model.npz -print
```

Stop if any `test -f` command fails. Do not substitute neutral or SMPL-X files and do not rename an incompatible model to make the check pass.

### 5.4 Verify the input manifests

The smoke test uses either the audited production split or the isolated smoke split so identity and split are written into each output's provenance:

```bash
test -d "$AMASS_EXTRACTED_ROOT"
test -f "$AMASS_RUN_ROOT/manifests/amass_raw_inventory.csv"

if test -f "$AMASS_RUN_ROOT/manifests/amass_subject_splits.csv"; then
  export AMASS_SUBJECT_SPLITS="$AMASS_RUN_ROOT/manifests/amass_subject_splits.csv"
else
  export AMASS_SUBJECT_SPLITS="$AMASS_RUN_ROOT/manifests/amass_subject_splits_smoke.csv"
fi

mkdir -p "$AMASS_RUN_ROOT/manifests"
mkdir -p "$AMASS_RUN_ROOT/reports"
mkdir -p "$AMASS_RUN_ROOT/core11-smoke"
```

The smoke split path may not exist until section 5.5 creates it. Do not run the converter until `test -f "$AMASS_SUBJECT_SPLITS"` succeeds there.

Do not omit `--subject-splits`. Without it, conversion can run, but the saved identity and model split are marked unassigned. For this conversion smoke test, `AMASS_SUBJECT_SPLITS` may name the isolated two-row smoke split from section 4.1. It must name the audited production split before full conversion.

### 5.5 Create a deterministic male/female smoke inventory

Do not use `--limit 10` on the raw inventory. It selects the first ten rows, which does not guarantee that both genders are present. Instead, create a two-row inventory containing one uncomplicated walking sequence for each gender. If the audited production split already exists, use it. Otherwise, this command also creates the isolated train-only smoke split permitted by section 4.1:

```bash
uv run --no-sync python - <<'PY'
import os
from pathlib import Path

import pandas as pd

run_root = Path(os.environ["AMASS_RUN_ROOT"])
inventory_path = run_root / "manifests/amass_raw_inventory.csv"
output_path = run_root / "manifests/amass_male_female_smoke.csv"
production_splits_path = run_root / "manifests/amass_subject_splits.csv"
smoke_splits_path = run_root / "manifests/amass_subject_splits_smoke.csv"

inventory = pd.read_csv(inventory_path)

required_inventory = {
    "relative_path",
    "subject_id_candidate",
    "motion_id",
    "gender",
    "pose_width",
    "num_frames",
    "mocap_framerate",
    "status",
}
assert required_inventory <= set(inventory.columns), "Raw inventory is missing required columns."
eligible = inventory[
    inventory["status"].eq("ok")
    & inventory["gender"].isin(["female", "male"])
    & inventory["motion_id"].str.contains(r"(?:^|[^a-z])walk", case=False, na=False)
    & inventory["pose_width"].eq(156)
    & inventory["num_frames"].ge(2)
].copy()

# Prefer explicitly named normal walks; fall back deterministically only if the
# audited production split contains none for a gender.
complex_motion = r"run|hop|jump|leap|turn|back|stair|side|skip|crouch|box|pick"
eligible["_smoke_priority"] = 1
eligible.loc[
    eligible["motion_id"].str.contains(r"normal[_ -]?walk", case=False, na=False),
    "_smoke_priority",
] = 0
eligible.loc[
    eligible["motion_id"].str.contains(complex_motion, case=False, na=False),
    "_smoke_priority",
] = 2
eligible = eligible.sort_values(["_smoke_priority", "relative_path"])

if production_splits_path.is_file():
    splits = pd.read_csv(production_splits_path)
    required_splits = {"subject_id_candidate", "identity", "split"}
    assert required_splits <= set(splits.columns), "Production split manifest is malformed."
    eligible = eligible[
        eligible["subject_id_candidate"].isin(set(splits["subject_id_candidate"].dropna()))
    ]

selected = pd.concat(
    [
        eligible[eligible["gender"].eq("female")].head(1),
        eligible[eligible["gender"].eq("male")].head(1),
    ],
    ignore_index=True,
).drop(columns="_smoke_priority")

assert len(selected) == 2, "Could not find one assigned walking sequence per gender."
assert set(selected["gender"]) == {"female", "male"}
assert selected["relative_path"].nunique() == 2

selected.to_csv(output_path, index=False)

if production_splits_path.is_file():
    chosen_splits = splits[
        splits["subject_id_candidate"].isin(selected["subject_id_candidate"])
    ][["subject_id_candidate", "identity", "split"]].drop_duplicates()
    assert chosen_splits["subject_id_candidate"].nunique() == 2
    split_path = production_splits_path
else:
    chosen_splits = selected[["source_dataset", "subject_id_candidate"]].drop_duplicates()
    chosen_splits["identity"] = chosen_splits.apply(
        lambda row: (
            str(row["subject_id_candidate"])
            if str(row["subject_id_candidate"]).startswith(
                f"{row['source_dataset']}::"
            )
            else f"{row['source_dataset']}::{row['subject_id_candidate']}"
        ),
        axis=1,
    )
    chosen_splits["split"] = "train"
    chosen_splits[["subject_id_candidate", "identity", "split"]].to_csv(
        smoke_splits_path, index=False
    )
    split_path = smoke_splits_path

print(selected[
    [
        "gender",
        "relative_path",
        "subject_id_candidate",
        "num_frames",
        "mocap_framerate",
    ]
].to_string(index=False))
print(f"\nWrote: {output_path}")
print(f"Use subject split: {split_path}")
PY
```

Read the printed table. It must contain exactly one `female` row and one `male` row. Prefer uncomplicated walking motions. If either selected filename describes a transition, turn, backward walk, or compound action, inspect `amass_male_female_smoke.csv` and deliberately replace it with a cleaner eligible row before continuing.

If you replace either row while using the temporary smoke split, regenerate the corresponding split row as well. The converter must never receive an inventory subject absent from its split manifest.

Set the exact split path printed by the command:

```bash
if test -f "$AMASS_RUN_ROOT/manifests/amass_subject_splits.csv"; then
  export AMASS_SUBJECT_SPLITS="$AMASS_RUN_ROOT/manifests/amass_subject_splits.csv"
else
  export AMASS_SUBJECT_SPLITS="$AMASS_RUN_ROOT/manifests/amass_subject_splits_smoke.csv"
fi
test -f "$AMASS_SUBJECT_SPLITS"
```

Confirm the resulting file:

```bash
test -f "$AMASS_RUN_ROOT/manifests/amass_male_female_smoke.csv"
wc -l "$AMASS_RUN_ROOT/manifests/amass_male_female_smoke.csv"
```

`wc -l` should report `3`: one header plus two data rows.

### 5.6 Run the two-sequence GPU conversion

Use an isolated smoke output directory so a failed experiment cannot be mistaken for the production corpus:

```bash
set -o pipefail

uv run --no-sync python scripts/convert_amass_core11.py \
  --amass-root "$AMASS_EXTRACTED_ROOT" \
  --inventory "$AMASS_RUN_ROOT/manifests/amass_male_female_smoke.csv" \
  --subject-splits "$AMASS_SUBJECT_SPLITS" \
  --body-model-root "$AMASS_BODY_MODEL_ROOT" \
  --canonical-fps 30 \
  --device cuda \
  --batch-size 256 \
  --verify-source-sha256 \
  --output-root "$AMASS_RUN_ROOT/core11-smoke" \
  --output-manifest "$AMASS_RUN_ROOT/manifests/amass_core11_conversion_smoke.csv" \
  --rejects "$AMASS_RUN_ROOT/manifests/amass_core11_conversion_smoke_rejects.csv" \
  2>&1 | tee "$AMASS_RUN_ROOT/reports/amass_core11_conversion_smoke.log"
```

If no interactive GPU is allocated, put that exact converter command in a short batch job and submit it now. The environment variables exported in sections 1 through 5 must be included through `--export=ALL`:

```bash
mkdir -p "$AMASS_RUN_ROOT/jobs" "$AMASS_RUN_ROOT/logs"

cat > "$AMASS_RUN_ROOT/jobs/convert-core11-smoke.sbatch" <<'SBATCH'
#!/usr/bin/env bash
#SBATCH --job-name=gp-convert-smoke
#SBATCH --partition=hai
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00

set -euo pipefail
: "${GAVD6_ROOT:?}"
: "${AMASS_EXTRACTED_ROOT:?}"
: "${AMASS_RUN_ROOT:?}"
: "${AMASS_BODY_MODEL_ROOT:?}"
: "${AMASS_SUBJECT_SPLITS:?}"

cd "$GAVD6_ROOT"
nvidia-smi
set -o pipefail
uv run --no-sync python scripts/convert_amass_core11.py \
  --amass-root "$AMASS_EXTRACTED_ROOT" \
  --inventory "$AMASS_RUN_ROOT/manifests/amass_male_female_smoke.csv" \
  --subject-splits "$AMASS_SUBJECT_SPLITS" \
  --body-model-root "$AMASS_BODY_MODEL_ROOT" \
  --canonical-fps 30 \
  --device cuda \
  --batch-size 256 \
  --verify-source-sha256 \
  --output-root "$AMASS_RUN_ROOT/core11-smoke" \
  --output-manifest "$AMASS_RUN_ROOT/manifests/amass_core11_conversion_smoke.csv" \
  --rejects "$AMASS_RUN_ROOT/manifests/amass_core11_conversion_smoke_rejects.csv" \
  2>&1 | tee "$AMASS_RUN_ROOT/reports/amass_core11_conversion_smoke.log"
SBATCH

SMOKE_CONVERT_JOB=$(sbatch \
  --parsable \
  --account="$HAIC_ACCOUNT" \
  --export=ALL \
  --output="$AMASS_RUN_ROOT/logs/gp-convert-smoke-%j.out" \
  --error="$AMASS_RUN_ROOT/logs/gp-convert-smoke-%j.err" \
  "$AMASS_RUN_ROOT/jobs/convert-core11-smoke.sbatch")
echo "Submitted conversion smoke job: $SMOKE_CONVERT_JOB"
```

While this job waits or runs, continue the manual subject audit in section 4. Do not start the full conversion from the temporary smoke split.

The command should exit successfully and print a final count equivalent to:

```text
[2/2] converted=2 skipped=0 rejected=0
```

On a safe rerun with unchanged inputs and configuration, `skipped=2` is expected because the converter verifies and reuses matching outputs. Do not add `--overwrite` merely to suppress a provenance error.

### 5.7 Audit the smoke outputs programmatically

Run this check before opening or plotting either result:

```bash
uv run --no-sync python - <<'PY'
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

run_root = Path(os.environ["AMASS_RUN_ROOT"])
output_root = run_root / "core11-smoke"
model_root = Path(os.environ["AMASS_BODY_MODEL_ROOT"])
manifest_path = run_root / "manifests/amass_core11_conversion_smoke.csv"
rejects_path = run_root / "manifests/amass_core11_conversion_smoke_rejects.csv"

def sha256_file(path, block_size=1024 * 1024):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()

expected_model_hashes = {
    gender: {
        "body_model_sha256": sha256_file(model_root / "smplh" / gender / "model.npz"),
        "dmpl_model_sha256": sha256_file(model_root / "dmpls" / gender / "model.npz"),
    }
    for gender in ("female", "male")
}

manifest = pd.read_csv(manifest_path)
rejects = pd.read_csv(rejects_path)

assert len(manifest) == 2, f"Expected 2 manifest rows, found {len(manifest)}."
assert rejects.empty, rejects.to_string(index=False)
assert set(manifest["gender"]) == {"female", "male"}
assert set(manifest["status"]) <= {"converted", "skipped_valid_existing"}
assert manifest["source_sha256_verified"].astype(str).str.lower().eq("true").all()
assert manifest["canonical_fps"].eq(30.0).all()
assert manifest["valid_fraction"].between(0.0, 1.0).all()
assert manifest["identity"].notna().all()
assert manifest["split"].isin(["train", "validation", "test"]).all()

expected_joints = [
    "pelvis",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_forefoot", "right_forefoot",
]
expected_channels = ["forward", "vertical_up", "mediolateral"]

for row in manifest.itertuples(index=False):
    tensor_path = output_root / row.tensor_relative_path
    assert tensor_path.is_file(), tensor_path

    with np.load(tensor_path, allow_pickle=False) as archive:
        coordinates = archive["coordinates"]
        coordinates_m = archive["coordinates_m"]
        valid = archive["valid"]
        times = archive["canonical_times_s"]
        pelvis_world = archive["pelvis_world_m"]
        world_to_body = archive["world_to_body_transform"]
        body_to_world = archive["body_to_world_transform"]
        physical_basis = archive["physical_basis_world"]
        joint_names = archive["joint_names"].tolist()
        channel_names = archive["channel_names"].tolist()
        provenance = json.loads(str(archive["provenance_json"].item()))

    frames = coordinates.shape[0]
    assert coordinates.shape == (frames, 11, 3)
    assert coordinates_m.shape == coordinates.shape
    assert valid.shape == (frames, 11)
    assert times.shape == (frames,)
    assert pelvis_world.shape == (frames, 3)
    assert coordinates.dtype == np.float32
    assert coordinates_m.dtype == np.float32
    assert valid.dtype == np.bool_
    assert np.isfinite(coordinates).all()
    assert np.isfinite(coordinates_m).all()
    assert np.all(coordinates[~valid] == 0)
    assert np.all(coordinates_m[~valid] == 0)
    assert joint_names == expected_joints
    assert channel_names == expected_channels
    assert np.allclose(body_to_world, world_to_body.T, atol=1e-8)
    assert np.allclose(world_to_body @ world_to_body.T, np.eye(3), atol=1e-8)
    assert np.isclose(np.linalg.det(physical_basis), 1.0, atol=1e-8)

    reconstructed_world = coordinates_m @ world_to_body + pelvis_world[:, None, :]
    assert reconstructed_world.shape == coordinates.shape
    assert np.isfinite(reconstructed_world).all()

    assert provenance["source"]["gender"] == row.gender
    assert provenance["source"]["identity"] == row.identity
    assert provenance["source"]["split"] == row.split
    assert provenance["source"]["subject_assignment_status"] == "assigned"
    assert provenance["source"]["sha256_verified_against_current_file"] is True
    assert provenance["body_model"]["body_model_sha256"] == expected_model_hashes[row.gender]["body_model_sha256"]
    assert provenance["body_model"]["dmpl_model_sha256"] == expected_model_hashes[row.gender]["dmpl_model_sha256"]
    assert provenance["body_model"]["dmpls_used"] is True
    assert provenance["body_model"]["num_betas"] == 16
    assert provenance["body_model"]["num_dmpls"] == 8
    assert provenance["schema"]["name"] == "core11-v1"
    assert provenance["coordinate_frame"]["name"] == "gait-parity-body-v1"
    assert provenance["resampling"]["canonical_fps"] == 30.0

    print(
        row.gender,
        tensor_path,
        coordinates.shape,
        "valid_fraction=", f"{valid.mean():.4f}",
        "forward_method=", provenance["coordinate_frame"]["forward_method"],
        "leg_length_m=", f"{provenance['coordinate_frame']['leg_length_m']:.4f}",
    )

print("\nPASS: male and female core-11 smoke outputs satisfy the structural contract.")
PY
```

This audit verifies file integrity, tensor shapes, dtypes, masks, schema ordering, transforms, replay state, source hashes, DMPL use, subject assignment, and the frozen frame rate. It does not prove anatomical left/right correctness by itself.

### 5.8 Perform the required human checks

For both sequences, inspect the recorded `forward_method`, `travel_straightness`, `travel_linearity`, `lateral_hip_alignment`, and `leg_length_m` in `amass_core11_conversion_smoke.csv`.

Require all of the following before the full conversion:

- the female row used the female SMPL-H and DMPL checksums;
- the male row used the male SMPL-H and DMPL checksums;
- leg length is plausible and nonzero;
- the pelvis is centred at zero in the body-frame coordinates;
- forward motion is not visibly reversed for an ordinary forward walk;
- left hip, knee, ankle, heel, and forefoot remain on the anatomical left chain;
- right joints remain on the anatomical right chain;
- the reconstructed world-space skeleton is finite and visually coherent;
- any hip-facing fallback is understood from the selected motion rather than ignored.

Numerical reflection tests cannot detect a dataset that was mirrored consistently from beginning to end. Compare the labelled converted skeleton with an official AMASS rendering or another trusted rendering of the same source sequence. Preserve at least one labelled screenshot for each gender in the smoke-test report.

### 5.9 Diagnose failures instead of forcing the run

Read both:

```text
$AMASS_RUN_ROOT/reports/amass_core11_conversion_smoke.log
$AMASS_RUN_ROOT/manifests/amass_core11_conversion_smoke_rejects.csv
```

Common failures have specific meanings:

| Failure | Required response |
|---|---|
| `human_body_prior is required` | Inspect and repair `uv sync` from the lockfile; repeat the import check. |
| Missing licensed model file | Correct `AMASS_BODY_MODEL_ROOT`; do not substitute another model family. |
| Source SHA-256 differs from inventory | Stop and determine whether AMASS changed; rebuild and re-audit the inventory if appropriate. |
| Subject absent from split manifest | Repair the audited subject registry/split; do not assign a split ad hoc. |
| Stale or unreadable output provenance | Inspect why source, model, mapping, or configuration changed; use a new smoke output root or deliberate `--overwrite` only after resolving it. |
| CUDA unavailable or out of memory | Confirm the allocation; reduce `--batch-size` if necessary. Batch size may change memory use but not the schema. |
| Implausible forward diagnostics | Select a simpler walking smoke sequence and investigate the original motion. |

Keep failed logs and reject rows. A rejected smoke record is evidence that must be understood, not a row to remove silently.

### 5.10 Promote the exact contract to the full inventory

Run the full conversion only after both genders pass the programmatic and visual checks. Use the eligible inventory produced in section 4, the same body-model root, the audited subject split, and the same frozen 30 Hz contract:

```bash
export AMASS_SUBJECT_SPLITS="$AMASS_RUN_ROOT/manifests/amass_subject_splits.csv"
test -f "$AMASS_SUBJECT_SPLITS"
test -f "$AMASS_RUN_ROOT/manifests/amass_raw_inventory_eligible.csv"
```

The full identity audit is a hard gate here. The two-row smoke split is not accepted for production conversion.

```bash
uv run --no-sync python scripts/convert_amass_core11.py \
  --amass-root "$AMASS_EXTRACTED_ROOT" \
  --inventory "$AMASS_RUN_ROOT/manifests/amass_raw_inventory_eligible.csv" \
  --subject-splits "$AMASS_SUBJECT_SPLITS" \
  --body-model-root "$AMASS_BODY_MODEL_ROOT" \
  --canonical-fps 30 \
  --device cuda \
  --batch-size 256 \
  --verify-source-sha256 \
  --output-root "$AMASS_RUN_ROOT/core11" \
  --output-manifest "$AMASS_RUN_ROOT/manifests/amass_core11_conversion.csv" \
  --rejects "$AMASS_RUN_ROOT/manifests/amass_core11_conversion_rejects.csv"
```

Run this as an `sbatch` job in `hai`, not on the head node and not by waiting for `hai-interactive`. Reuse the section 5.6 job structure, change the inventory and output paths to the production values above, and use a realistic time limit based on the measured two-sequence smoke throughput. Record that estimate in the submission log rather than guessing a duration in this tutorial.

Do not pass `--limit` for production. Keep body-model files outside Git. Every output stores source, model, parameter-layout, schema, coordinate-frame, scaling, resampling, validity, replay, and converter provenance. Existing outputs are reused only when their verified source and complete conversion fingerprint match; use `--overwrite` deliberately when changing a frozen input or contract.

## 6. Hard stop: the current checkout cannot pretrain on AMASS

After section 5, the repository can produce these real artifacts:

```text
amass_subject_registry.csv
amass_subject_splits.csv
amass_raw_inventory_eligible.csv
amass_core11_conversion.csv
amass_core11_conversion_rejects.csv
core11/**/*.npz
```

Confirm the production manifest before declaring the data handoff complete:

```bash
uv run --no-sync python - <<'PY'
import os
from pathlib import Path

import pandas as pd

run_root = Path(os.environ["AMASS_RUN_ROOT"])
manifest = pd.read_csv(run_root / "manifests/amass_core11_conversion.csv")
rejects = pd.read_csv(run_root / "manifests/amass_core11_conversion_rejects.csv")
output_root = run_root / "core11"

assert not manifest.empty, "conversion manifest is empty"
assert set(manifest["status"]) <= {"converted", "skipped_valid_existing"}
assert manifest["identity"].astype(str).str.strip().ne("").all()
assert manifest["split"].isin({"train", "validation", "test"}).all()
assert manifest["canonical_fps"].eq(30.0).all()
assert manifest["source_sha256_verified"].astype(str).str.lower().eq("true").all()

missing = [
    path
    for path in manifest["tensor_relative_path"]
    if not (output_root / path).is_file()
]
assert not missing, f"missing converted files; first values: {missing[:10]}"

print("successful sequences:", len(manifest))
print("successful identities:", manifest["identity"].nunique())
print(manifest.groupby("split")["identity"].nunique().sort_index())
print("reject rows requiring review:", len(rejects))
if not rejects.empty:
    print(rejects.groupby("error").size().sort_values(ascending=False).head(20))
PY
```

Review every reject category against the frozen inclusion rules. A reject is not automatically an exclusion: fix recoverable environmental or mapping failures and rerun. If a source sequence is irrecoverably invalid, record the reason and regenerate a final manifest whose counts are reconciled with the eligible inventory.

There is no valid next training command in this checkout. In particular:

- there is no code that turns converted sequences into deterministic windows;
- there is no dataset that loads core-11 windows and validity masks;
- the current JEPA implementation requires 33 landmarks;
- the current reflection swaps MediaPipe pairs and negates channel 0, whereas core-11 reflection must swap five core-11 pairs and negate channel 2;
- there is no AMASS training configuration, checkpoint lineage, or Stage 1 audit path.

Stop here rather than manufacturing a tensor shape that the existing model happens to accept.

## 7. Minimum implementation needed to cross the stop line

This section is a handoff specification, not a set of runnable commands. Add and test the capabilities in this order; document actual filenames and invocations only after they exist and have been exercised from a clean checkout.

### 7.1 Parameterize the skeleton contract

Preserve the existing GAVD behavior while making the reusable JEPA components accept an explicit skeleton specification. The AMASS specification is:

```python
CORE11_JOINTS = 11
CORE11_LEFT_RIGHT_PAIRS = (
    (1, 2),   # hip
    (3, 4),   # knee
    (5, 6),   # ankle
    (7, 8),   # heel
    (9, 10),  # forefoot
)
CORE11_MEDIOLATERAL_CHANNEL = 2
```

The stored channel order is `[forward, vertical_up, mediolateral]`. Core-11 mirroring must negate channel 2 and swap the five pairs above. Parameterize tokenizer and predictor joint counts, joint positional embeddings, maskable joints, swap pairs, reflected channel, token-count calculations, input validation, and checkpoint metadata.

Tests must prove that mirroring twice returns the original, forward and vertical channels do not change sign, bilateral coordinates and validity masks swap correctly, and every claimed equivariant layer commutes with reflection. Existing GAVD tests must continue to pass.

### 7.2 Build a manifest-only window index

Do not materialize duplicate tensors for overlapping windows. Index slices of the converted sequence files with this frozen initial policy:

```yaml
schema: core11-v1
canonical_fps: 30
window_frames: 64
stride_frames: 32
time_patch_frames: 4
minimum_valid_joint_fraction: 0.95
window_storage: index-only
mirror_policy: online-orbit
```

For `T >= 64`, emit starts `0, 32, 64, ...` and append `T - 64` once if the regular stride misses the final complete window. Reject shorter sequences rather than stretching them. Derive each stable window ID from immutable source provenance and the start frame. Write deterministic, atomic output and carry identity, split, source SHA-256, and conversion fingerprint into every row.

“Broad motion” means every nonclinical AMASS motion that passes the prespecified conversion and structural QC rules; it does not require a BABEL action label. Do not filter activities after looking at downstream clinical results.

### 7.3 Build the loader

Each sample must provide:

```text
coordinates: [64, 11, 3] float32
valid:       [16, 11] bool, aggregated over four-frame patches
window_id:   stable string
```

The loader must reject the wrong schema, shape, joint order, channel order, split, or conversion fingerprint. Invalid coordinates remain zero and invalid targets remain ineligible for masking. Generate the mirrored orbit online; never write the mirror as a separately split sample.

Because compressed NPZ files are not cheap random-access containers, start with a bounded per-worker sequence cache. Measure loader wait time before introducing another storage representation.

### 7.4 Add a matched trainer and acceptance audit

The trainer must run `standard`, `paired_unconstrained`, and `reflection_equivariant` variants against the same original-window order, mask schedule, and update budget for each seed. Derive schedules from the frozen seed and stable window IDs so variant-specific code paths cannot consume randomness differently.

Before any long run, require one real-CUDA, 20-update smoke run per variant. Each run must save configuration, code and manifest hashes, optimizer state, loss history, GPU metadata, and checkpoint lineage; reload the checkpoint in a fresh process; and report finite losses. The equivariant variant must also pass layerwise commutation checks in training and evaluation modes.

The production audit must additionally report validation loss, representation variance, covariance effective rank, cosine similarity, even/odd channel energy, commutation residuals, parameters, updates, original windows seen, masked tokens, wall time, GPU type, and peak memory. It must end in an explicit `pass`, `fail`, or `incomplete` state.

## 8. Handoff checklist

Complete now with the current repository:

```text
[ ] HAIC environment imports torch and human_body_prior on a GPU node
[ ] Female and male licensed SMPL-H and DMPL files verified
[ ] Two-sequence converter smoke test passed
[ ] Numerical, anatomical, replay, and visual smoke audits passed
[ ] Subject registry manually audited
[ ] Downstream overlaps explicitly excluded
[ ] Deterministic subject split and eligible inventory frozen together
[ ] Full core-11 conversion completed
[ ] Every reject reviewed and final counts reconciled
[ ] Conversion manifest and provenance preserved
```

Blocked on implementation, not on another shell command:

```text
[ ] Skeleton-parameterized JEPA with a tested core-11 reflection
[ ] Deterministic manifest-only window index
[ ] Validity-aware, cached core-11 dataset
[ ] Matched three-variant trainer with checkpoint lineage
[ ] Real-CUDA 20-update smoke gate for all variants
[ ] Production data and checkpoint audits
```

Do not continue to Stage 2 until the second checklist exists as tested code and Stage 1 checkpoints have passed its audit.
