# Complete local GAVD annotations

This folder contains the complete five-part GAVD CSV annotation dataset,
reorganized as **1,874 sequence CSVs, 458,116 annotation rows, 12 gait patterns,
and 348 distinct YouTube video IDs**. All ten parsed field values are preserved,
including empty annotations. The local split has **100% CSV row coverage** of
the pinned upstream snapshot, with no missing, extra, or duplicate records.

The CSVs associate gait patterns and annotated video frames with YouTube IDs
and URLs. They contain annotations and video references; video files, extracted
frames, and pose coordinates are retrieved or computed separately. URL
availability checks are summarized below; full media playback was not tested.

Completed November 9, 2026 (America/Los_Angeles; November 10 UTC), against
upstream `main` at commit
[`a87859c881603443f200bcd640663d2c3d7a8136`](https://github.com/Rahmyyy/GAVD/tree/a87859c881603443f200bcd640663d2c3d7a8136/data).
The immutable source files and provenance are retained in [source/](source/).

## Coverage and organization

Each transformed annotation file is `<gait_pat>/<seq>.csv`. Directory names
match the literal source label, including the space in `cerebral palsy/`.
Counts below exclude headers.

| Directory / `gait_pat` | Sequence CSVs | Annotation rows |
| --- | ---: | ---: |
| `abnormal/` | 767 | 216,492 |
| `antalgic/` | 35 | 6,556 |
| `cerebral palsy/` | 64 | 20,346 |
| `exercise/` | 234 | 54,941 |
| `inebriated/` | 23 | 7,007 |
| `myopathic/` | 188 | 34,595 |
| `normal/` | 291 | 41,340 |
| `parkinsons/` | 47 | 10,426 |
| `pregnant/` | 6 | 909 |
| `prosthetic/` | 39 | 17,999 |
| `stroke/` | 76 | 33,934 |
| `style/` | 104 | 13,571 |
| **Total** | **1,874** | **458,116** |

| Downloaded source in `source/` | Rows | Distinct sequences within part | Rows represented in split |
| --- | ---: | ---: | ---: |
| `GAVD_Clinical_Annotations_1.csv` | 91,624 | 374 | 91,624 |
| `GAVD_Clinical_Annotations_2.csv` | 91,623 | 353 | 91,623 |
| `GAVD_Clinical_Annotations_3.csv` | 91,623 | 369 | 91,623 |
| `GAVD_Clinical_Annotations_4.csv` | 91,623 | 313 | 91,623 |
| `GAVD_Clinical_Annotations_5.csv` | 91,623 | 469 | 91,623 |
| **Union** | **458,116** | **1,874** | **458,116** |

Four sequences cross adjacent source-part boundaries; the per-part sequence
counts therefore sum to 1,878 rather than the union's 1,874. Each now has one
complete output file:

| Sequence | Parts | Combined rows | Source frame range |
| --- | --- | ---: | --- |
| `cljrrgtez000c3n6lkj3o1mlv` | 1 + 2 | 1,236 | 809–2044 |
| `cll44qpg400493o6lfcd9cdqa` | 2 + 3 | 89 | 1783–1871 |
| `cll8ebce5000u3o6lj55n38oe` | 3 + 4 | 1,201 | 4823–6023 |
| `clld3ef9b003p3o6l1f3mrlez` | 4 + 5 | 178 | 2639–2816 |

The original local dataset covered part 1 only (374 files, 91,624 rows, 69
video IDs). Completion added 1,500 sequence files and 366,492 rows, including
747 rows for the previously partial prosthetic sequence above. All original
parsed annotation records remain represented.

## Folder contents

```text
data/gavd/
├── source/
│   ├── GAVD_Clinical_Annotations_1.csv  # through _5.csv
│   ├── manifest.json                  # URLs, commit, bytes, SHA256, Git blob IDs
│   ├── README.md                      # source-cache usage
│   ├── UPSTREAM_README.md              # verbatim upstream attribution/schema
│   └── LICENSE                         # verbatim upstream MIT license
├── abnormal/ ... style/                # 12 literal gait_pat directories
├── download_gavd.py                    # restore/check pinned CSV source cache
├── process_gavd.py                     # merge all five parts, split by sequence
├── verify_processing.py                # independent exhaustive comparison
├── test_processing.py                  # regression and fault-injection tests
├── check_video_availability.py         # dated public YouTube response audit
├── test_video_availability.py          # availability classification tests
├── video_availability.json             # per-video HTTP/player evidence
├── verification_report.json            # full-dataset verification result
├── REVIEW.md                           # adversarial review and fixes
├── INSTRUCTIONS.md                     # operational workflow
└── README.md
```

`ssl-authorization.json` and `.cursor/` are existing ancillary configuration
and editor metadata. They are outside the annotation workflow.

The upstream `data/` inventory contains five CSVs, five corresponding PKL
representations, and a one-newline `data.txt`. All five requested CSVs are
cached here. PKLs are alternative table representations and are not needed
for this CSV derivative. The full upstream file inventory and Git blob IDs
are recorded in [source/manifest.json](source/manifest.json).

The sibling `../GAVD_Clinical_Annotations_1.csv` and `../GAVD_Clinical_Annotations_1.1.csv`,
`../GAVD_Clinical_Annotations_1.2.csv`, `../GAVD_Clinical_Annotations_1.3.csv`
are existing project inputs outside this derivative. They are not read or
changed by the new workflow. The sibling part 1 has equivalent parsed rows but
different newline bytes; `source/` holds the original upstream bytes.

## Reproduce and verify

Use Python 3.9 or later; all tools use the standard library. From the project
root:

```bash
python3 data/gavd/download_gavd.py
python3 data/gavd/process_gavd.py
python3 data/gavd/verify_processing.py --report data/gavd/verification_report.json
python3 -m unittest discover -s data/gavd -p 'test_*.py'
```

Script defaults resolve relative to the script directory, independent of the
working directory. Explicit relative CLI paths resolve from the current
working directory. The download command reuses verified cached files and
restores missing or corrupt ones from pinned URLs. It does not track future
changes to upstream `main`.

To reproduce into a separate directory:

```bash
python3 data/gavd/process_gavd.py --output-dir /tmp/gavd-reproduced
python3 data/gavd/verify_processing.py --output-dir /tmp/gavd-reproduced --source-dir "$PWD/data/gavd/source"
```

Processing reads all five parts in numeric order into staged files with at
most 32 open CSV writers. It retains sequence/frame keys in memory to reject
duplicates, rather than loading every record into memory. Source headers,
checksums, sizes, row counts, labels, and sequence consistency are validated
before publishing annotation files. Existing CSV rows must be a subset of the
complete source. Unexpected/stale CSVs, including a second `cerebralpalsy/`
copy, cause refusal. Reconcile those files outside this folder and rerun;
the processor does not silently delete them.

Output replacement is atomic per file. An interrupted publication can leave a
mixture of complete older and newer files; rerun processing, then verify to
establish complete coverage. An unchanged rerun writes zero CSVs. CSV quoting
and newline bytes may differ from the source, while parsed field strings and
row multiplicities remain exact. Rows within a sequence follow the source
parts' order; frame numbers retain their original values.

Verification uses an independent on-disk SQLite comparison of every parsed
row across all five source parts and every transformed CSV. It checks source
byte hashes, all headers, literal directory labels, filename/sequence
agreement, complete sequence inventory, duplicate full records and
`(seq, frame_num)` keys. Success requires zero missing and extra records.
The source cache is explicitly excluded from transformed-record counts; a
recursive count of every CSV under this folder would count source and split
rows twice. Reports exit nonzero on failure.

The saved [verification_report.json](verification_report.json) records
458,116 source and output rows, 1,874 output files, zero missing/extra rows,
zero duplicate records or keys, and an empty issue list. See
[REVIEW.md](REVIEW.md) for independent checks and adversarial review.

## YouTube source-video availability by gait pattern

Checked **November 9, 2026, 23:16:57–23:18:38 PST**
(November 10, 07:16:57–07:18:38 UTC), using unauthenticated requests from this
environment. Of **348 distinct YouTube IDs**, **338 (97.1%) had positive
public availability evidence**, **5 were explicitly private or removed**,
and **5 remained inconclusive**. No fallback response was classified as
access restricted in this run. These are video-ID counts, not sequence or frame counts.

Here, “positive public response” means either valid direct YouTube oEmbed
metadata (331 IDs), or a direct watch-page player status of `OK` after oEmbed
failed (7 IDs, all with embedding disabled). It is a practical check that the
video reference is still valid publicly; **media playback and downloadability
were not tested**. Metadata can remain accessible even when playback is
restricted, so 338 must not be read as a verified playable-video count.

| `gait_pat` | Unique source videos | Positive public response | Private / removed | Inconclusive |
| --- | ---: | ---: | ---: | ---: |
| `abnormal` | 117 | 113 | 1 | 3 |
| `antalgic` | 10 | 10 | 0 | 0 |
| `cerebral palsy` | 11 | 10 | 1 | 0 |
| `exercise` | 98 | 96 | 2 | 0 |
| `inebriated` | 8 | 8 | 0 | 0 |
| `myopathic` | 30 | 29 | 0 | 1 |
| `normal` | 32 | 32 | 0 | 0 |
| `parkinsons` | 11 | 11 | 0 | 0 |
| `pregnant` | 2 | 2 | 0 | 0 |
| `prosthetic` | 8 | 8 | 0 | 0 |
| `stroke` | 19 | 18 | 1 | 0 |
| `style` | 3 | 2 | 0 | 1 |
| **Distinct union** | **348** | **338** | **5** | **5** |

One video ID occurs in multiple gait patterns, so the category rows must not
be summed to obtain the distinct union. The five explicit unavailability
responses were four private videos and one video from a terminated account.
The other five watch pages returned a generic “video unavailable” message;
the cause could not be established, so they are not labeled deleted.

The audit first queried `https://www.youtube.com/oembed` for every unique ID.
A successful result required HTTP 200 JSON identifying a YouTube video with
a nonempty title. For all 17 failures, it inspected the direct watch page’s
`ytInitialPlayerResponse.playabilityStatus`. HTTP 401/403/404 alone was never
used as proof of an invalid video. Explicit private/removal reasons, access
restrictions, and ambiguous errors are classified separately. The checker
does not sign in or fetch media streams. Results can vary by time, location,
account, and YouTube response behavior.

[video_availability.json](video_availability.json) retains all 348 IDs, their
gait-pattern memberships, request URLs, per-attempt HTTP responses, timestamps,
and fallback player evidence. The per-video timestamps identify when the
evidence was obtained; the final pass reused the 331 successful first-pass
oEmbed responses and refreshed the other 17 cases. To refresh all IDs:

```bash
python3 data/gavd/check_video_availability.py --cache-hours 0
```

The default cache lasts 24 hours. Rerunning writes the JSON report; update
this dated Markdown table from the new `by_gait_pattern` summary as needed.

No annotation records or video references were dropped because of current
YouTube availability; the complete transformed dataset remains lossless.

## CSV schema and interpretation

All source and sequence CSVs have this ordered header:

```text
seq,frame_num,cam_view,gait_event,dataset,gait_pat,bbox,vid_info,id,url
```

| Field | Representation |
| --- | --- |
| `seq` | Sequence ID for a section of gait in a single direction; matches the output filename stem. |
| `frame_num` | Frame number in the source video, preserved without renumbering. |
| `cam_view` | Person-centric view: `front`, `back`, `left side`, `right side`, or empty. |
| `gait_event` | Sparse clinical event annotation; often empty. |
| `dataset` | Broad upstream grouping: 422,141 rows say `Abnormal Gait` and 35,975 say `Normal Gait`. Use `gait_pat` for pattern grouping. |
| `gait_pat` | Observed source gait label; exactly matches the output directory. |
| `bbox` | Python dictionary literal containing `top`, `left`, `height`, `width`. |
| `vid_info` | Python dictionary literal containing video `height`, `width`, `mime_type`. |
| `id` | YouTube video ID; multiple sequences can reference the same video. |
| `url` | Source video URL, preserved verbatim. |

Use a CSV parser because quoted fields contain commas. Use `ast.literal_eval`
for `bbox` and `vid_info` if dictionary objects are needed; their single-quoted
Python syntax is not JSON. Keep empty annotation strings intact when comparing
records (for pandas, use `keep_default_na=False`). Sequence count is distinct
from video count. Source gait labels have not been clinically revalidated by
this organization step.

For attribution, citation, and access details, see the
[pinned upstream README](https://github.com/Rahmyyy/GAVD/blob/a87859c881603443f200bcd640663d2c3d7a8136/README.md),
its [local verbatim copy](source/UPSTREAM_README.md), and the
[upstream license copy](source/LICENSE). Relative figure links inside the
verbatim README resolve in the upstream repository.
