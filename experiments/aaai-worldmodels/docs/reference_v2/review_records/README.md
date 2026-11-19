# Independent review evidence

These are exact copies of the independent reviewer's small source scripts and result records. `export_manifest.json` retains each original local path and SHA256. The reviewer authored the checks and review records, but none of the study protocol, data, model, evaluation, notebook, manuscript, or slide implementations. The final acceptance file identifies the exact publication artifacts accepted after these checks.

The original scratch directory remains local under `artifacts/reference-v2-independent-review`. The copied scripts retain their original relative-path assumptions so their bytes and recorded hashes stay unchanged. The launcher restores those same bytes to their original scratch locations before executing them. It refuses to overwrite a different scratch script.

From the repository root, use the experiment's Python 3.12 environment. For example:

```bash
uv run --project experiments/aaai-worldmodels python experiments/aaai-worldmodels/docs/reference_v2/review_records/replay_review_checks.py core_results_checks.py
uv run --project experiments/aaai-worldmodels python experiments/aaai-worldmodels/docs/reference_v2/review_records/replay_review_checks.py external_results_checks.py
```

Real data, pinned pose caches, and the corresponding complete run artifacts must already exist. No script substitutes synthetic efficacy for absent real outputs. Core, transfer, robustness, and post hoc result checks replay predictions or small saved normal references and perform no new forecast training. `prefit_checks.py` additionally fits three explicitly synthetic tiny fixtures. The initial GAVD and external input checks verify local media and pose artifacts without downloading or extracting new data.

Replays write fresh numerical check records to the ignored scratch directory. Their timestamps and execution times can differ from these reviewed snapshots. Manual image inspection is a separate human review event. The external-input launcher verifies the exact previously inspected image hashes rather than claiming fresh inspection of changed images.

The final delivery checks include all 145 fitted outer checkpoint trajectories, every source/executed notebook cell, the publication rebuild, and slide structure. `slide_delivery_checks.py` additionally requires PyMuPDF and the `node` executable; use the publishing environment recorded in the manuscript build record. Its assertions verify exact identities, vectors, fonts, query geometry, links, and syntax. They do not repeat the human full-size page review or exercise browser navigation. A reviewer text-checker failure caused by a wrapped PDF phrase is preserved separately from artifact findings.

The retained diagnostics explain three numerical details. Torch replay matches the producer's one-thread setting. The external routine uses native float32 error arithmetic for learned predictions, with no change to component tail scores in a double-precision sensitivity. Finally, discrete empirical ranks require the producer's declared CSV transport path to reproduce exact ties and AUROC. The failed first floor replay is preserved under its own identity; its corrected successor does not replace any original primary result.
