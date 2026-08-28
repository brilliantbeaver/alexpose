# Data contract

## Allowed roles

| Data source | Permitted role | Boundary |
| --- | --- | --- |
| AMASS | broad motion pretraining; controlled gauge-corruption ground truth | synthetic gauge injection is not evidence that a real pose pipeline fails |
| GAVD | video-grouped pose-transfer and natural-laterality-ambiguity audit | no participant-disjoint, diagnostic, force, or balance claim |
| Public force/stability cohorts | participant-grouped biomechanical validation | verify download terms, measured quantities, and participant IDs before use |

Do not replace a failed biomechanical endpoint with a GAVD label or a
coordinate-derived proxy. Do not add a dataset requiring a license agreement
without recording that it is outside the program's permitted data scope.

## Immutable manifests

Every completed run records the input manifest, retrieval/extraction version,
joint conversion, quality exclusions, split manifest, seeds, code revision,
and checkpoint selection rule. The source video is the independent group in
GAVD; the participant is the independent group in force/stability cohorts.

## Present evidence boundary

`outputs/repaired-jepa-seed7-v2` is AMASS pretraining evidence. It is not a
GAVD downstream result, a real-gauge audit, or a biomechanics result.
