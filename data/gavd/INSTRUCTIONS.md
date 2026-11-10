# GAVD download, organization, and verification

The complete snapshot contains **458,116 rows in 1,874 sequence CSVs across
12 gait-pattern directories**, referencing 348 YouTube video IDs. It was
completed November 9, 2026 (America/Los_Angeles), from upstream commit
`a87859c881603443f200bcd640663d2c3d7a8136`.
See [README.md](README.md) for coverage tables, schema, provenance, and citation.

## Standard workflow

Run with Python 3.9 or later from the project root. No third-party packages
are required.

```bash
python3 data/gavd/download_gavd.py
python3 data/gavd/process_gavd.py
python3 data/gavd/verify_processing.py --report data/gavd/verification_report.json
python3 -m unittest discover -s data/gavd -p 'test_*.py'
```

1. **Download:** validate the committed `source/manifest.json` and restore any
   missing/corrupt source CSVs using pinned raw GitHub URLs. Already valid
   files are reused. Downloads use temporary files and replace their targets
   only after successful verification. The source cache is about 127 MB in
   decimal units. The transformed CSVs require another approximately 127 MB.
2. **Process:** read all five source parts together and write exactly one
   `<gait_pat>/<seq>.csv` per sequence. Four sequences span two parts and must
   be merged. Every parsed field is preserved as a string, including empties.
3. **Verify:** compare all ten fields and row multiplicities against all five
   sources. Success requires valid headers, filenames, labels, sequence
   inventory, hashes, no duplicate records/keys, and zero missing/extra rows.
   `verification_report.json` captures the result; a failure exits nonzero.
4. **Test:** run the temporary-fixture regression suite, including cross-part
   sequences and damaged-data cases. Tests leave the real dataset unchanged.

Defaults are relative to each script, so invoking the scripts by absolute
path also works from another working directory. `--help` lists options.

## Reproduce elsewhere

```bash
python3 data/gavd/process_gavd.py --source-dir "$PWD/data/gavd/source" --output-dir /tmp/gavd-copy
python3 data/gavd/verify_processing.py --source-dir "$PWD/data/gavd/source" --output-dir /tmp/gavd-copy
```

`download_gavd.py --source-dir <cache>` expects the pinned manifest already
present in that cache. Copy `source/manifest.json` there first to create a
fresh download cache. The downloader intentionally requires the recorded
snapshot; updating to another upstream revision requires reviewing the new
inventory and changing the pinned metadata and downloader identities.

## Failure recovery

- **Missing/corrupt source:** rerun the downloader. Keep the committed manifest
  intact; do not change its hashes to accommodate a damaged file.
- **Unexpected output CSV / stale directory:** move it outside the output
  tree, reconcile it, and rerun. In particular, use `cerebral palsy/` exactly;
  retaining a duplicate `cerebralpalsy/` folder fails verification.
- **Existing rows absent from source:** preserve that file outside the output
  tree and investigate before replacing it. Processing refuses to overwrite
  such records.
- **Interrupted processing:** rerun, then verify. Files are staged before
  publication and replaced atomically individually. Publication across all
  sequence files is not one atomic operation.
- **Verification failure:** inspect its JSON issue list. Counts alone do not
  prove equality; do not accept a dataset until the exhaustive comparison
  succeeds.

Processing uses staged CSVs and a limited writer cache; it keeps all
sequence/frame keys in memory for duplicate detection. Verification stores
full rows in a temporary SQLite database and requires additional temporary
disk space. Neither command downloads videos.

## Working with the CSVs

Header:

```text
seq,frame_num,cam_view,gait_event,dataset,gait_pat,bbox,vid_info,id,url
```

Use `csv.DictReader` or pandas with `keep_default_na=False` to retain empty
fields. Nested `bbox` and `vid_info` values are Python dictionary literals;
parse them with `ast.literal_eval` when necessary. Frame numbers refer to
original video frames. Video IDs and URLs associate the annotations with
YouTube sources; a sequence filename is not a video ID.

Only count category CSVs as the transformed dataset. `source/` contains the
same annotation records in five original tables, so including that cache in
a recursive union duplicates the row count. The verifier excludes it.

The original one-part workflow and advice to run each part separately have
been replaced: separate runs could truncate shared sequences. Existing
sibling `data/GAVD_Clinical_Annotations_1*.csv` inputs are not modified.

## Refresh YouTube availability

```bash
python3 data/gavd/check_video_availability.py --cache-hours 0
```

This checks every distinct video ID from the five source CSVs, using direct
YouTube oEmbed metadata and watch-page player responses for failed metadata
checks. It saves per-video evidence, timestamps, category membership, and
category summaries in `video_availability.json`. No media files are fetched.
Without `--cache-hours 0`, results less than 24 hours old are reused.

Public metadata or a player `OK` response is evidence of a valid public
reference at check time, rather than confirmation of actual playback. Explicit
private/removed-video responses are distinguished from access restrictions
and inconclusive errors. Several gait patterns can share a video ID, so do
not sum category counts to compute a global video total. See the dated table
and method limitations in [README.md](README.md). Refresh that table when
saving a later audit; rerunning the checker updates JSON only.
