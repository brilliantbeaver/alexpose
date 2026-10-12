# Full-data pipeline validation

This record covers the full-manifest Gait Fidelity revision. The [HAIC guide](../README.md) explains execution, and the [notebook guide](../../../notebooks/gait_fidelity/README.md) explains the twelve tutorials. Local checks exercise the software; they do not certify the remote CUDA environment or produce research findings.

**Final local result:** 85 automated tests passed. All 12 notebooks completed their 98 code cells without an error, fitting 102 fixture models across 135 optimization phases. Verification checked 1,071 artifacts and reconstructed 39,168 metric rows. Independent visual review found the evaluated figures readable and clearly labelled as software fixtures.

## Population and experimental scope

The AMASS planner reads all 8,854 eligible manifest motions and joins their audited identities to the original person splits. It records selection decisions for the entire inventory. The default `named_walking` preset selects filename-based walking candidates long enough for complete, nonoverlapping 128-frame windows at 25 Hz:

| Original role | People | Motions | Windows |
| --- | ---: | ---: | ---: |
| Training | 113 | 934 | 2,171 |
| Validation, used for development | 15 | 102 | 214 |
| Test, locked for confirmation | 14 | 97 | 199 |

These are local manifest counts before remote file verification, technical screening and any additional reservation or interval-review exclusions. The [manifest census](manifest-census.json) retains the exact input hashes, alternative presets and storage projections. Its missing-on-disk counts describe this local audit environment, not HAIC. `all_eligible` admits other actions and must be described as a heterogeneous motion experiment rather than a walking cohort.

The GAVD planner inventories all 1,874 sequences from 348 source videos. It uses the original annotated person boxes, records missing or unsuitable media, and groups related recordings before partitioning. Exact duplicate media, supplied person links and historical reservation groups cannot cross partitions. A missing cross-video identity mapping limits independence claims to known recording groups.

AMASS provides projected joint references for restoration training and controlled measurement tests. GAVD provides a separate real-video gait-label probe; its annotations do not supply joint-coordinate or affected-side ground truth. Probe preprocessing and fitting use training recordings only. Camera/height and observation-availability controls help detect acquisition confounding. Short sequences, low frame rates, failed decoding and missing boxes remain visible in coverage reports.

The source default retains ten core recipes across three seeds: 30 final models and 39 optimization phases. The full option retains 34 recipes, 102 final models and 135 phases. Core training samples people, motions and windows hierarchically. The full matrix uses matched endpoint cycles for its pairing controls. Both record actual sampling coverage; making the complete cohort available does not imply that a fixed update budget visits every derived example.

## Validation paths

The checks include:

- Full-manifest joins, canonical identities, content duplication, original splits, reviewed intervals, historical reservations and locked confirmation populations.
- Preparation with small motion data and substituted expensive body/render/pose backends, including the same chunk publication, restart and merge paths used by source jobs.
- Memory-mapped source arrays, both training samplers, frozen encoders, every registered recipe, training-only calibration, fixed reference support and hierarchical metric aggregation.
- Real MP4 decoding and SQLite annotation lookup for GAVD, with a substituted pose extractor; actual feature generation, linear probes, recording-group uncertainty and confirmation checks run on the resulting tracks.
- Slurm script syntax, execution from relocated scheduler paths, shared concurrency and budget handling, ambiguous submissions, and receipt tampering.
- Fresh Jupyter execution of all twelve notebooks, with retained outputs, complete fixture fitting, metric reconstruction and artifact verification.

The machine-readable [checks record](checks.json) and [notebook execution receipt](tutorial-execution.json) identify the final tested revision. Fixtures validate execution and algebra, not anatomy, generalization, convergence or statistical power.

## Adversarial findings addressed

Independent reviewers examined data, learning and workflow code written by other agents. The [workflow review](workflow-adversarial-review.md) records its findings. Material corrections include:

1. Full-manifest preparation consumes frozen reviewed windows directly; it cannot silently shift them through the legacy interval extender.
2. All source prediction exports use disk-backed arrays, including the full matched-cycle protocol.
3. Runtime projections separate optimization from fixed data verification, loading and output hashing. Actual allocation accounting remains the spending authority.
4. Evaluation receipts bind calibration, baseline predictions, numerical tables and reports; mutation checks cover those outputs.
5. Missing declared reservation files stop setup. Reinitialization rejects conflicting explicit protocol settings.
6. GAVD preserves historical linked groups and protection flags. When a reservation file contains both `role` and `split`, protection in either field takes precedence; previously exposed development recordings cannot become fresh confirmation data.

## Checks still required on HAIC

Use the saved source configuration for CPU preflight, then run preparation on allocated GPUs and inspect its reference videos/contact sheets before fitting. The source worker checks CUDA/MMCV execution and rendering dependencies; measured profiling determines the common training budget. Local validation cannot establish current HAIC asset availability, EGL behavior, CUDA operator compatibility, H100 throughput or whole-cohort peak resource use.

Review walking candidates and reference quality without selecting examples by model performance. Keep failed and reference-ineligible cases in coverage accounting. Confirmation requires reviewed prior-exposure evidence and frozen methods before opening the protected data. Report person-level uncertainty for AMASS and recording-group uncertainty for GAVD, together with training-seed variation and supported class counts. A larger cohort and a correct implementation make stronger tests possible; they do not guarantee a favorable or statistically significant finding.
