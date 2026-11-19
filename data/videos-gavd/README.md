# GAVD local video storage

This directory contains source recordings, extraction/download ledgers and local previews. New downloads are organized by source recording ID, with previews in `<source_id>/previews/`. Legacy root-level cached videos remain at their recorded paths; the ingestion pipeline does not duplicate recordings into condition folders.

The current annotation inventory is [data/gavd/README.md](../gavd/README.md). The executed [iteration02 protocol](../../experiments/aaai/docs/iteration02/protocol.md) uses fresh CSV labels and verifies annotation, video, extraction and pose hashes. Its [case-review ledger](../../experiments/aaai/artifacts/iteration02/d66e1b126c9d9a9adf1b/video_review/review_ledger.csv) records ten selected previews, original intervals and source paths. Previews retain original playback timing. The post-outcome boundary audit is descriptive and did not change model fits or eligibility.

Source video redistribution permissions have not been cleared for the manuscript bundle. Keep previews local. The paper uses measured vector skeleton traces rather than source photographs.
