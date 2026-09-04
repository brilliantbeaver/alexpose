# GAVD data audit

Audit date: 2026-09-03. Counts below come from the five public annotation CSVs and files inspected on this machine. They do not reuse dataset-scale numbers from the paper.

## Verdict

The configured HAIC root is `/hai/scratch/tedmui/datasets/gavd_full`. The expected source cache is `youtube/all`, manifests are in `manifests`, original annotations are in `annotations`, and downloader reports are `reports/video_download_audit_shard-<i>-of-<n>.csv`. On 2026-09-03, the user confirmed that all 348 source videos are present and decodable on HAIC and cover all 1,874 sequences. These operational counts reconcile exactly with the checked-in manifest. This session could not independently enumerate the cache because one read-only SSH attempt timed out and a second failed during host-name resolution.

The checked-in sequence manifest has 1,874 unique sequences and 348 unique YouTube IDs. All 1,874 rows map to a syntactically valid 11-character ID, every URL encodes the recorded ID, and there are 348 distinct ID/URL pairs. The manifest proves resolution. The user confirmation above supplies the current HAIC availability status.

The only cache directly inspected from this environment is the older fixed-cohort cache at `/Users/theodoremui/dev/alexpose/experiments/sjepa/gavd5-tm/work/youtube`. It contains 18 decodable source videos. All 18 reach the maximum annotated frame in the full manifest and collectively cover 121 of the 1,874 manifest sequences. This remains a local smoke-test cache, not the full HAIC corpus.

## Annotation inventory

The five public CSVs contain 458,116 frame rows. They reconcile exactly to all 1,874 sequence IDs and 348 source IDs in `manifests/gavd/gavd_full_sequences.csv` (SHA-256 `0ee5c91e7f4a1947c7a7a138716392fa46d73d31b0bdea3f21a03a153eef2721`). The binary `dataset` label has 1,620 abnormal sequences from 321 videos and 254 normal sequences from 27 videos.

The observational `gait_pat` taxonomy is:

| Label | Sequences | Source videos |
| --- | ---: | ---: |
| abnormal | 767 | 117 |
| normal | 291 | 32 |
| exercise | 234 | 98 |
| myopathic | 188 | 30 |
| style | 104 | 3 |
| stroke | 76 | 19 |
| cerebral palsy | 64 | 11 |
| parkinsons | 47 | 11 |
| prosthetic | 39 | 8 |
| antalgic | 35 | 10 |
| inebriated | 23 | 8 |
| pregnant | 6 | 2 |

These are dataset annotations, not independently verified diagnoses. The label sets are not nested cleanly: 37 sequences from 5 videos have `gait_pat=normal` while `dataset=Abnormal Gait`. One source, `wRntYsztIEY`, has eight sequences labeled `cerebral palsy` and eight labeled `abnormal`.

`gait_event` is absent in 457,358 of 458,116 rows. The 758 populated rows contain Right initial contact (383), Left initial contact (368), Left tibia vertical (3), Left toe off (2), and Left foot flat (2). Camera view is front, back, left side, right side, or other. Five sequences have no view on 929 total rows, and 11 sequences change view within a sequence. The sequence manifest stores one view and therefore hides those transitions.

## Subjects and splits

Neither the public schema nor either checked-in manifest contains a participant, subject, person, or split field. `seq` identifies a tracked gait section in one direction. `id` identifies a YouTube source. Neither identifies a person. A video can contain several people, and the same person can recur across sequences or videos. The real subject count is therefore not recoverable, and no subject-disjoint train, validation, or test split exists in the released data.

Source-video grouping is the strongest available leakage control, but it is only a surrogate. The later Core11 converter requires a separately built pose manifest with `split` and enforces one split per video. The user confirms that a complete full-video pose manifest has not yet been created. The full sequence manifest is not a valid substitute.

## Duplicates and ambiguous records

There are no exact duplicate rows, duplicate `(seq, frame_num)` keys, duplicate sequence IDs, or frame gaps within a sequence. Four sequences cross adjacent CSV-part boundaries and reconstruct correctly after concatenation.

Nine source IDs carry two incompatible annotated geometries. They account for 135 sequence records and produce 68 cross-geometry temporal overlaps, including 60 pairs with interval intersection-over-union at least 0.8. These look like repeated annotation passes, but the release provides no version or subject key, so deterministic deduplication is unsafe. Across the corpus, 91 sequence pairs overlap within 17 videos. Three pairs have exactly the same video and frame interval, but none has identical bounding boxes. Treat sequences as correlated within video and audit the nine dual-geometry sources before modeling.

## Reproduction

Fetch the public source used here:

```bash
git clone --depth 1 https://github.com/Rahmyyy/GAVD /tmp/GAVD-audit
cp /tmp/GAVD-audit/data/GAVD_Clinical_Annotations_{1,2,3,4,5}.csv /tmp/
```

Core count and taxonomy:

```python
import glob, pandas as pd
d = pd.concat([pd.read_csv(p, low_memory=False) for p in sorted(glob.glob('/tmp/GAVD_Clinical_Annotations_*.csv'))])
s = d.groupby('seq').agg(video=('id','first'), label=('gait_pat','first'))
print(len(d), s.index.nunique(), s.video.nunique())
print(s.groupby('label').agg(sequences=('video','size'), videos=('video','nunique')))
```

On a network with HAIC access, validate the cache without substituting manifest counts:

```bash
ssh tedmui@haic.stanford.edu
cd /hai/scratch/tedmui/alexpose/experiments/sjepa/gavd6
export GAVD_FULL_ROOT=/hai/scratch/tedmui/datasets/gavd_full
find "$GAVD_FULL_ROOT/youtube/all" -maxdepth 1 -type f
ls "$GAVD_FULL_ROOT/reports"/video_download_audit_shard-*-of-*.csv
```

Count a source only when the report has `ok=True`, OpenCV decodes a frame, FPS and frame count are positive, and frame count reaches that video's `required_last_frame`. Join valid `video_id` values back to the sequence manifest to count covered sequences. Keep `failed`, missing, empty, and truncated files separate.

## Open questions

- Which pose extractor, schema, confidence fields, and source-video split builder should define the new full-video pose manifest?
- Are the nine dual-geometry records repeated annotation versions, distinct people, or both?
- Can the dataset authors supply stable participant IDs? Without them, subject counts and participant-held-out evaluation are impossible.
