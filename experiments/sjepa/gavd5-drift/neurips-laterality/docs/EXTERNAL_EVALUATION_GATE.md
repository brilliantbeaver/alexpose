# How to prepare the optional external subject gate

Notebook 06 reports `not configured / not run` in a fresh checkout by design.
The repository cannot contain or invent another dataset's subject identifiers,
permission record, or local file paths. This neutral state does not block the
internal workflow in notebooks 00–05. External contract validation becomes
available only when the operator supplies two real records: a subject-indexed
pose manifest and governance scoped to that exact external dataset.

Passing Notebook 06 means **the prerequisites were validated**. It does not
mean an external model evaluation ran. The current suite contains no external
training, checkpoint-transfer, prediction, or metric stage.

## Step 1: confirm that the proposed use is actually authorized

Ask the responsible data custodian and institutional reviewers about the
specific planned use: subject-disjoint pose evaluation. Public availability,
permission to view a video, or the internal GAVD review does not authorize a
different dataset.

Start from `governance/external-status.template.json` and keep the completed
copy outside version control. Use a stable `dataset_reference` that identifies
the governed dataset without putting personal information in the file. Keep
`authorization_scope` exactly `subject_disjoint_pose_evaluation`.

Each of these three entries must have `status: "resolved"`, a non-empty approved
internal `reference`, and a `date`:

1. `ethics_determination` records the applicable ethics or human-subjects
   determination.
2. `data_use_review` records that the planned computation is permitted by the
   license, agreement, and custodian restrictions.
3. `derived_pose_release_review` records the permitted handling and release
   scope for poses and downstream artifacts.

Do not mark a review resolved merely to satisfy the validator. An environment
variable points to a decision record; it does not create the decision.

## Step 2: obtain custodian-supplied subject identifiers

The manifest must use a persistent, pseudonymous `subject_id` supplied by the
custodian. Do not infer identity from faces, filenames, URLs, or gait. A subject
may have several sequences, but every sequence for that subject must belong to
the same partition.

Choose partitions before inspecting test outcomes. `train` is for fitting,
`validation` is optional and is for model or hyperparameter choices, and `test`
is for the final untouched assessment. At least one train subject and one test
subject are required. No subject identifier may occur in more than one of
these partitions.

## Step 3: build the CSV manifest

Copy the header from `governance/external-manifest.template.csv`. Add one row
per pose sequence with all six values populated:

| Column | Requirement |
|---|---|
| `dataset_reference` | Exactly matches the governance record |
| `sequence_id` | Unique across all rows |
| `subject_id` | Custodian-supplied and confined to one split |
| `pose_path` | Unique existing file beneath the pose root |
| `split` | `train`, `validation`, or `test` |
| `joint_schema` | Exactly `BlazePose33` |

Prefer relative `pose_path` values beneath one dedicated pose root. Notebook 06
checks that files exist and that paths cannot escape that root. It does not
currently open every archive and validate its numeric contents, so the
custodian's extraction and format audit remains a separate prerequisite.

## Step 4: configure the notebook without publishing local paths

Copy the three names from `governance/external.env.template` into the root
`.env` file and replace the examples with the real paths. The pose root is
optional, but setting it explicitly makes the allowed file boundary clear.
Do not commit the completed governance record, subject manifest, or pose files.

Notebook 06 reads only these three keys. A process environment value takes
precedence; otherwise the notebook checks `neurips-laterality/.env` and then the
repository-root `.env`. Relative values loaded from a `.env` file are resolved
relative to that file, not relative to Jupyter's launch directory.

## Step 5: run and interpret the gate

Restart the kernel or run all cells from the top. The configuration diagnostic
reports which settings were found and where they came from while redacting the
actual paths. The progress display ends at 100% when it has accounted for both
preflight checks, including a deliberately skipped contract check when no
external study was configured. Progress completion describes control flow, not
scientific evidence.

Interpret `gate_state` as follows:

| State | Meaning |
|---|---|
| `not_configured` | Neither required path was supplied; optional study skipped normally |
| `configuration_blocked` | Exactly one required path was supplied; configuration is incomplete |
| `contract_blocked` | Both paths were supplied, but the fail-closed validator rejected a rule |
| `contract_validated` | Governance and manifest prerequisites passed; evaluation still did not run |

If validation is blocked, `reason` identifies the first failed contract rule.
Fix the source record or data layout; do not weaken the rule.

The validated-contract status implemented today is:

```text
manifest contract validated; evaluation not run
```

The output then reports sequence and unique-subject counts for the three
partitions. `evidence_created` deliberately remains `False`.

## Step 6: recognize the remaining implementation boundary

A real external result requires a separately reviewed and pre-specified stage.
Before test access, it must define which checkpoint is transferred or which
external-train subjects are used, what preprocessing is allowed, what choices
may use validation data, the primary metric, uncertainty unit, missing-data
rules, and the exact claim boundary. Test subjects must remain unavailable to
training and model selection. Only after that code and protocol exist can the
project produce an external metric or support an unseen-person statement.
