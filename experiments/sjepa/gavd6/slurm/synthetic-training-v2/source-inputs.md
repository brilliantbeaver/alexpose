# Prepare the reviewed AMASS inputs

Use this page when `locomotion-audit.csv` and `person-reservations.csv` have not yet been prepared. These are required research inputs. Creating `preparation.json` records their paths but does not create the files or perform the reviews.

The existing AMASS manifests provide motion paths, identities, and original splits. They do not establish that a particular 64-frame window is locomotion, or supply this study's complete reservation and exposure history. The commands below prepare worksheets from those manifests; a reviewer must complete the decisions before source preparation can proceed.

**1. Open the generated worksheets.**

The managed `init` command creates them under `$STV2_WORK/inputs/review-drafts/`. If worksheet creation reported a missing manifest, correct the input paths and rerun:

```bash
bash "$STV2_ROOT/slurm/synthetic-training-v2/submit.sh" inputs
```

This command reads manifests and checks motion-file availability. It does not load motion arrays or request a GPU, and it preserves existing worksheets. The generated locomotion worksheet initially contains only column names; reservation decisions remain blank.

The three files serve different purposes:

| Draft | What to do with it |
| --- | --- |
| `motion-candidates.csv` | Look up real motion paths, their registry identities, original splits, and durations. Availability and duration alone do not establish locomotion or permission to use a person. |
| `person-reservations.draft.csv` | Complete reservation, alias, and exposure decisions for the intended panel and its known aliases. |
| `locomotion-audit.draft.csv` | Add one row for each window that you actually review. Initially this file contains only column names. |

Original test identities are omitted from the motion candidate list. The person worksheet retains the identities returned by the manifest join, including test identities, to help preserve original assignments. Neither list replaces historical reservation records or discovers aliases that are absent from the registry.

**2. Resolve person reservations before reviewing source motions.**

Use the existing identity registry, split manifest, and previous experiment records. For a first engineering check, aim for two eligible training people and two eligible validation people. Actual availability and reservations determine whether that panel is possible.

Complete these fields for each selected person and every known alias needed to preserve their reservations:

| Field | What to enter |
| --- | --- |
| `person_id` | The exact registry identity shown in the worksheet. |
| `canonical_person_id` | The established identity shared by known aliases. If the registry identity is already canonical and has no separate known aliases, use that same identity. |
| `original_split` | Preserve `train`, `validation`, or `test` exactly as recorded. Do not move people between splits to reach the desired counts. |
| `reserved` | `true` or `false`, based on the existing reservation history. Blank or unknown reservation decisions are not ready for use. |
| `exposure` | The documented exposure status. Use `unknown` if exposure is unresolved; that cannot support an unexposed confirmation claim. |

The completed file must have one row per `person_id`. Include the people referenced by the final locomotion audit and their relevant known aliases; omit unrelated unfinished worksheet rows. A reservation on one alias applies to the whole canonical identity. Original test people and reserved people are excluded from source fitting.

The old study's GAVD `source-reservation.csv` concerns video recordings. It is not a substitute for this AMASS person file.

**3. Review the motion windows.**

Choose motions belonging to the admitted people and inspect source-motion playback or existing documented review evidence for each selected interval. A filename containing `walk`, or an identity audit, does not by itself establish that the chosen interval is locomotion.

Add a row to the locomotion worksheet only when its review is complete:

| Field | What to enter |
| --- | --- |
| `relative_path` | Copy the exact value from `motion-candidates.csv`. |
| `start_s` | The reviewed window's start time in seconds. |
| `locomotion_status` | `audited_locomotion` after confirming locomotion in that window. |
| `audit_reviewer` | The person who performed the review. |
| `audit_evidence` | A traceable reference to the reviewed playback or other evidence, including the interval and relevant observations. |
| `audit_date` | The actual review date, preferably `YYYY-MM-DD`. |
| `exposure` | Exactly the same status recorded for this person in the reservation file. |
| `canonical_person_id` | Exactly the canonical identity in the reservation file. |

Each window has 64 samples at 25 Hz. The last sample occurs at `start_s + 2.52`, which must fit within the motion's duration. Starts in the same motion must be at least 2.56 seconds apart. Choose at least two distinct eligible windows per training person; two per person in both splits is a useful first engineering panel. Review each window separately.

Keep the locomotion file to the eight listed columns. In particular, do not copy `person_id`, `original_split`, `raw_path`, or `duration_s` into it; the loader obtains those from the existing manifests and rejects attempts to override them.

**4. Save the completed files at the configured paths.**

For the managed run, save your completed records as:

```text
$STV2_WORK/inputs/locomotion-audit.csv
$STV2_WORK/inputs/person-reservations.csv
```

These are sibling files of the `review-drafts` directory, not files inside it. Do not point the preparation configuration to an unfinished worksheet. If completed files already live elsewhere, use their real absolute paths in `preparation.json` instead.

**5. Run the metadata check again.**

Return to Step 2 of the [HAIC guide](README.md) and run `submit.sh check`. Require `STV2_INPUTS_PASSED`, and inspect the printed person/window counts before rendering. The checks validate file contents and joins; they cannot verify that a human review was actually performed. The first engineering panel establishes pipeline feasibility, not a scientific result.
