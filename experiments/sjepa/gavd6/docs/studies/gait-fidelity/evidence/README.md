# Audit evidence and reproduction

These files record the 21 September 2026 development audit. They contain metadata and code checks, not new restoration-performance results.

| Artifact | Contents |
| --- | --- |
| [masking-bank.json](masking-bank.json) | 512 masks, full joint/time frequencies, current v2 example masks, ten inspected-source hashes and package versions |
| [masking-source-review.md](masking-source-review.md) | Independent notebook/source inspection with cell and line references |
| [masking-checks.json](masking-checks.json) | Small hidden-value and interpolation checks; descriptive implementation probes |
| [video-inventory.csv](video-inventory.csv) / [JSON](video-inventory.json) | All 91 paths, hashes, container properties and source IDs |
| [video-summary.json](video-summary.json) | Collection totals and distributions |
| [timestamps-audit.json](timestamps-audit.json) | Decoded presentation-time spacing and sampling consequences |
| [cache-inventory.json](cache-inventory.json) | All 88 pose-cache schemas, hashes and support summaries |
| [gavd-source-overlap.json](gavd-source-overlap.json) | Exact source-ID intersection with local GAVD manifests |
| [visual-selection.json](visual-selection.json) / [temporal-selection.json](temporal-selection.json) | Metadata-selected visual-review scope |

For reproduction, use the [scripts guide](../scripts/README.md). It writes fresh audit outputs to a separate directory and rebuilds both the interactive paper and the complete figure gallery. The underlying source audits remain development evidence; reference annotations and new model comparisons are still pending.

The original visual samples are specified in the selection JSON files. Open the [local gallery](../data/video-gallery.html) for the current footage review, and read the [data specification](../data/README.md) before treating a source as suitable for a measurement.
