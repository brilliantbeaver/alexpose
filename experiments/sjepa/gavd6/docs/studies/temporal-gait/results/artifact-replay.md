# Independent retained-artifact replay — 2026-09-15

**Passed, synthetic software evidence only.** This audit covers
`gavd6/outputs/temporal-gait/software-20260915-03`, after the receipt-integrity and
unsupported-secondary-horizon fixes, including the earlier Slurm GPU-allocation
and governing-decision changes. It does not demonstrate GPU/Slurm execution,
real GAVD validity, generalization, or clinical usefulness.

The machine-readable [report](artifact-replay.json) records exact paths, anchor
SHA-256 digests, counts, numerical differences and all four paired contrasts.
The self-contained [diagnostic](artifact_replay.py) imports only NumPy and the
Python standard library, not the experiment's statistics/evaluation package.
It reads the explicitly named synthetic run and prints JSON; it does not write
files, train, open media, discover private sources, or print source identifiers.

## Reproduce

From the repository root, using the existing environment:

```sh
gavd6/.venv/bin/python \
  gavd6/docs/studies/temporal-gait/results/artifact_replay.py \
  --run-root "$PWD/gavd6/outputs/temporal-gait/software-20260915-03"
```

Success prints `"status": "passed"`; an assertion, missing file or hash mismatch
exits nonzero. Do not use `python -O` or `PYTHONOPTIMIZE`: the diagnostic rejects
optimized execution so its assertions cannot be silently disabled. The retained
JSON is the diagnostic's semantic output; compare parsed JSON, not whitespace
or alternate spellings of floating-point zero. Environment: Python 3.12.10,
NumPy 2.5.2. No new dependencies are required.

Verification executed: a fresh replay matched the retained report exactly as
parsed JSON. Six negative checks correctly rejected changed scores, support,
missingness, array shape, a relative run-root path, and optimized execution.
There were no unexpected failures; those checks did not alter any run artifact.

## Independent calculation

1. Read the frozen configuration and explicitly referenced E0, five pilot-arm,
   and locked-test evaluation artifacts. Check every artifact fingerprint,
   fitted-array hash, checkpoint hash, receipt output, and the canonical frozen
   identity. All referenced paths must remain inside the supplied run root.
2. Reconstruct lower-limb target support from endpoint-valid pairs 25/26, 27/28,
   29/30 and 31/32, requiring at least three pairs and valid prefix scale. Compute
   Euclidean endpoint errors from saved joint coordinates; average joints,
   windows within a bout, bouts within a video, and finally videos equally.
   An unavailable prediction on eligible support remains unavailable, not a
   reason to silently drop an observation. Compare all source scores, summaries,
   coverage counts and missing-prediction counts.
3. Independently subtract each skeleton's future hip-midpoint (23/24) for the
   root-relative secondary. Recompute root displacement and articulation errors,
   their extra observed-hip support, and their separate source summaries. Neither
   secondary changes the primary mask or enters baseline/model selection.
4. Recreate all saved bootstrap draws from seed 812 by expanding sampled connected
   groups into their complete video lists. This independently checks the package's
   multiplicity-weight implementation. Keep all seeds together in each group draw,
   average seed errors per source and then sources equally, and compute relative
   gain as a ratio of means, not a coordinate ensemble or average of ratios.
   Recompute both percentile endpoints, point effects and separate seed statistics.
5. Verify the test condition's exact frozen checkpoint/readout hashes and train-only
   fit provenance, train/evaluation identity separation, the complete requested pilot
   grid, and software-only claim/advancement flags. This replays predictions, not
   their training process; checkpoint bytes are hashed without deserialization.

Numerical acceptance is `abs(a-b) <= 1e-12 + 1e-12*abs(b)`. Array shapes,
boolean support and unavailable-value patterns must match exactly.

## Verified coverage and agreement

| Check | Coverage | Largest absolute discrepancy |
|---|---:|---:|
| Primary source scores / summaries | 882 / 294 | 0 |
| Secondary source scores / summaries | 1,764 / 588 | 2.78e-17 |
| Saved secondary window arrays | 784 | 5.56e-17 |
| Paired contrasts / draw pairs | 4 / 200 | 9.55e-18 for error-difference draws |
| Error-difference 95% CI endpoints | All four contrasts | 6.94e-18 |
| Relative-gain draws / CI endpoints | All four contrasts | 3.64e-12 / 4.55e-13 |

All relative-gain differences satisfy the stated combined tolerance. Their large
absolute values arise because this periodic fixture's baseline error is nearly
zero; they are not meaningful estimates of real gait improvement.

There are seven evaluation artifacts and 98 method-evaluations: eight E0 methods,
15 methods for each of five trained arms, and 15 methods on the locked synthetic
test. The development and test roles each contain 48 windows from three videos
in three groups; all comparisons use one seed and 50 bootstrap draws. Repeated
arms/methods reuse observations: the score counts are not independent sample sizes.
One seed does not estimate training-seed variability.

Integrity checks covered 28 evaluation-artifact references, seven fitted-array
references, six checkpoint references, and 116 receipt-output references, spanning
121 unique verified files. The JSON contains the canonical digest of the complete
verified-file path→SHA-256 mapping, plus individual anchor and evaluation digests.

## Outcome and evidence boundary

| Role | Selected masked/online-context error | Periodic baseline error | Baseline comparison |
|---|---:|---:|---|
| Development | 0.005149632973505037 | 0.0000023208839440733224 | Failure |
| Synthetic test | 0.03755226941240341 | 0.0000026124127986972324 | Failure |

Errors are projected-image Euclidean distance divided by prefix-only projected
body length. The selected model improves over its initialized counterpart in this
fixture but loses to the strongest baseline. The synthetic test exercises frozen
software paths; it is not permission to bypass failed real-data advancement gates.

Retained claims are `software_verified_only` and `software_only`.
`confirmatory_claim`, `real_runs_complete` and `ready_for_expansion` are all false.
No numerical or provenance defect was found. This final refresh changed only the
JSON and Markdown reports. The diagnostic, earlier `software-20260915-01` and
`software-20260915-02` outputs, final run outputs, package code, decisions and
governing protocol documents were not changed by the replay.

Frozen run identity:
`dcd02f5ead7c638498387ced10b10ae437c93d9b31f7987c33b26bd2446a5892`.
Diagnostic SHA-256:
`4de85ce24cd659e7e56cc988043684e4b0d7d3e64c6b2e5bf89543dbf89969f9`.
Verified-file manifest SHA-256:
`82f3c6a1d7a59a5850566cd1de70802790cc83756a3ab3a1c785ef8fb0d25b7a`.
