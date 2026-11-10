# Pinned upstream source cache

These five original CSVs were retrieved from
[upstream GAVD](https://github.com/Rahmyyy/GAVD/tree/a87859c881603443f200bcd640663d2c3d7a8136/data)
at commit `a87859c881603443f200bcd640663d2c3d7a8136` on November 10, 2026 UTC
(November 9 America/Los_Angeles). They contain 458,116 rows in total.

[manifest.json](manifest.json) records each pinned URL, original byte size,
SHA256 digest, Git blob SHA1, row count, and the full upstream data inventory.
The five CSVs occupy 126,723,609 bytes. PKLs are alternative representations
and are not part of this CSV cache. `data.txt` contains a newline.

`UPSTREAM_README.md` and `LICENSE` are verbatim copies for attribution. The
upstream README's relative figure links resolve in its original repository;
see its [online copy](https://github.com/Rahmyyy/GAVD/blob/a87859c881603443f200bcd640663d2c3d7a8136/README.md).

From the project root, run `python3 data/gavd/download_gavd.py` to check or
restore the pinned CSVs. Preserve the manifest when restoring the cache.
See [../INSTRUCTIONS.md](../INSTRUCTIONS.md) for custom locations and the
complete download, split, and verification workflow.

This cache is excluded when verifying transformed category CSVs. Including
both original and transformed tables in a row union would count annotations
twice. The source tables are kept byte-identical; the split preserves parsed
field values while CSV quoting/newline formatting may differ.
