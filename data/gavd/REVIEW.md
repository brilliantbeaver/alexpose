# Adversarial review and verification

Completed November 9, 2026 (America/Los_Angeles), against upstream commit
`a87859c881603443f200bcd640663d2c3d7a8136`. Review used separate subagent
contexts for source acquisition, exhaustive verification, and adversarial
challenge. These were independent checks by the same model family, rather
than a claim of cross-model review.

## Dataset evidence

- Source reviewer obtained the authoritative upstream Git inventory and
  checked all five downloaded CSVs against Git blob IDs. SHA256, byte sizes,
  row counts, pinned URLs, and revision are recorded in `source/manifest.json`.
- The standalone verifier compared every ten-field parsed record and row
  multiplicity in an on-disk SQLite database. Result: 458,116 source and local
  rows, 1,874 files, 348 video IDs, 12 labels, no missing or extra rows, no
  duplicate records or `(seq, frame_num)` keys, and no header/path issues.
- The adversarial reviewer independently compared per-sequence row counters
  and exact row order against all five source parts, using a different
  comparison implementation. Every output path and parsed record matched.
- All four cross-part sequences were independently checked: 1,236, 89, 1,201,
  and 178 combined rows, respectively. The initially partial prosthetic file
  gained its missing 747 rows.
- An unchanged full processing rerun wrote zero CSVs. LF output line endings
  match the existing files; the 373 previously complete sequences retain
  their annotation contents. Source files remain byte-identical upstream
  originals.

The machine-readable full comparison is in
[verification_report.json](verification_report.json).

## Findings and resolutions

1. **Part-by-part processing would truncate shared sequences.** Replaced the
   one-part splitter with a combined five-part staged process. Cross-part
   fixtures and full-data checks establish complete boundary sequences.
2. **Nested duplicate source CSVs could evade a basename-only inventory.**
   Both processor and verifier now compare relative paths against exactly the
   five root source filenames. A persistent regression test inserts a nested
   same-name source and requires refusal without changing existing outputs.
3. **Old documentation described partial coverage and unsafe reruns.**
   Replaced local README and instructions, corrected literal `cerebral palsy`
   directory naming, distinguished Python dictionary fields from JSON, and
   documented duplicate source/output counting and per-file publication.
4. **Default CSV writer changed existing newline bytes unnecessarily.**
   Set LF explicitly to match existing output conventions. Only the formerly
   partial tracked prosthetic CSV needs added annotation rows; the pre-existing
   `cerebralpalsy` → `cerebral palsy` working-tree rename was preserved.

## Regression checks

Run `python3 -m unittest discover -s data/gavd -p 'test_*.py'`.
All 10 dataset temporary-fixture tests passed. They exercise all-part sequence merging,
empty and quoted dictionary strings, missing rows, duplicate rows/keys,
altered empty fields, header damage, stale category aliases, corrupted source
refusal, unexpected local row preservation, nested duplicate source refusal,
and unchanged rerun bytes/modification time.

The download reviewer also exercised actual missing-file restoration from
GitHub, verified cache reuse, a mocked outage with no target replacement or
temporary-file remnants, and manifest path traversal rejection.

## YouTube availability review

The audit checked all 348 distinct IDs. Positive public response evidence was
found for 338 (331 oEmbed metadata responses and seven watch player `OK`
responses). Four were explicitly private, one referenced a terminated account,
and five generic unavailable errors remained inconclusive. The README reports
unique videos by gait label; overlapping category memberships are counted in
each relevant category and once in the global union.

Adversarial inspection found two classification hazards before finalization:
searching serialized JSON could match field names/tracking strings, and
reading only the top-level reason could miss explicit account termination in
a nested visible message. The final classifier uses displayed text, ignores
tracking fields, gives location/age/login restrictions their own category,
and leaves generic unavailability inconclusive. Cached fallback results carry
a method version, so changed classification semantics invalidate older checks.
The 17 fallback cases were refreshed after these corrections.

Eleven additional mocked-response tests passed, covering private versus restricted
login, player `OK`, generic errors, nested termination messages, ignored
tracking fields, regional copyright restrictions, nested JSON parsing, missing
or malformed player data, invalid video-details objects, and mismatched video IDs. The complete suite has
**21 passing tests**. Actual media playback was not tested; the README states
this boundary explicitly.

## Practical limits

Publication is atomic per sequence file, not across the whole output tree.
An interruption can require rerunning processing and verification. New upstream
revisions require deliberate manifest and pinned downloader updates.
Verification proves agreement with the stored snapshot and recorded hashes;
it does not clinically revalidate the source labels.

YouTube checks and their per-video evidence are kept separately in
`video_availability.json`; their method, dated category counts, and limitations
are documented in [README.md](README.md). Public metadata or player responses
are not a guarantee of uninterrupted playback in every location or account.
