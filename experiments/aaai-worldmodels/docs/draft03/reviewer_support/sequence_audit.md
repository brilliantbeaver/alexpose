# GAVD sequence scale, overlap, and interpretation audit

Analysis identity: `draft03-sequence-contract-audit-v1`.

This audit verifies the scale and relationship of the movement observations already used in the study. It reads the transformed annotation CSVs and immutable common-support cohort. It does not change the method, eligibility rules, windows, source folds, pose caches, predictions, or study identities. No forecast outcomes were consulted. The accompanying [machine-readable audit](sequence_audit.json) contains input hashes, every sequence's identity and annotation summary, complete overlapping sequence-pair lists, example overlapping windows, and independent replay checks.

## Sequence-first scale is supported

The local catalog contains **1,874 unique sequence IDs, 458,116 annotation rows, and 348 source-video IDs**. The five requested gait annotations contain **666 sequences from 103 sources**. The completed study retains **456 sequences and 2,815 forecast windows from 91 source videos**. Its normal population contains **141 sequences and 465 windows from 29 sources**. Each sequence has exactly one source ID and one source URL across all its rows. These counts exclude the pinned `data/gavd/source/` provenance copies.

| Gait annotation | Catalog sequences | Catalog source IDs | Retained sequences | Retained windows | Retained source IDs |
| --- | ---: | ---: | ---: | ---: | ---: |
| normal | 291 | 32 | 141 | 465 | 29 |
| parkinsons | 47 | 11 | 37 | 292 | 11 |
| stroke | 76 | 19 | 58 | 695 | 16 |
| cerebral palsy | 64 | 11 | 52 | 518 | 8 |
| myopathic | 188 | 30 | 168 | 845 | 27 |
| **Five-label total** | **666** | **103** | **456** | **2,815** | **91** |

The full catalog has the following coverage. The labels describe observed gait annotations rather than independently verified patient diagnoses. Per-label source counts sum to 349 because `wRntYsztIEY` has both `abnormal` and `cerebral palsy` sequences; their union contains 348 source IDs.

| Gait annotation | Catalog sequences | Annotation rows | Source IDs within label |
| --- | ---: | ---: | ---: |
| abnormal | 767 | 216,492 | 117 |
| antalgic | 35 | 6,556 | 10 |
| cerebral palsy | 64 | 20,346 | 11 |
| exercise | 234 | 54,941 | 98 |
| inebriated | 23 | 7,007 | 8 |
| myopathic | 188 | 34,595 | 30 |
| normal | 291 | 41,340 | 32 |
| parkinsons | 47 | 10,426 | 11 |
| pregnant | 6 | 909 | 2 |
| prosthetic | 39 | 17,999 | 8 |
| stroke | 76 | 33,934 | 19 |
| style | 104 | 13,571 | 3 |
| **Union** | **1,874** | **458,116** | **348** |

The 456 retained sequence IDs account for 111,013 rows in their original complete annotation spans. This is a raw annotation count for contributing sequences. Some of those rows do not contribute to a retained forecast window, and the count is not the number of eligible model observations or independent examples.

## What single-direction segmentation means

The primary GAVD paper describes manually chosen sequence boundaries intended to capture at least two gait cycles, limited occlusion, and one direction relative to the camera. The local schema and pinned upstream README also define a sequence as a section of gait in one direction. This explains why one source recording can supply several useful walking observations. It does not constitute a new verification of the actual path or turning behavior in every retained sequence. [GAVD paper, Section 3.2.1](https://arxiv.org/html/2407.04190)

The CSV fields contain camera-view labels and sparse gait-event annotations. They contain no measured heading, straightness score, participant key, or explicit turn-free certification. All 1,874 sequences have contiguous raw annotation frames, but temporal continuity does not prove a straight path. Preparation splits sampling at annotation gaps and applies observation-quality gates. It does not detect a turn or filter a sequence according to estimated walking direction. A 24-observation forecasting window also does not inherit a verified two-cycle guarantee from the description of a complete source sequence.

Literal view annotations change within **11 catalog sequences**, **seven sequences in the five-label scope**, and **three retained sequences**. The retained sequences contain seven transitions between `front` and `right side`:

| Sequence ID | Source ID | Annotation | Raw frame range | Literal view transitions |
| --- | --- | --- | --- | ---: |
| `clk7vopzf003b3n6ls09yk91c` | `nXuJIs25z1U` | normal | 3,261 to 3,540 | 3 |
| `clk7vr8n7003n3n6l1mm7nwq5` | `nXuJIs25z1U` | normal | 4,486 to 4,722 | 2 |
| `cll7nljgk000d3o6l3psndypi` | `JUMhhwFANKE` | parkinsons | 678 to 902 | 2 |

Three retained windows sample frames with more than one nonempty view label. The JSON gives their window IDs and raw frame ranges. This count uses actual sampled frame lists and per-frame labels. In contrast, the existing `view` column copies the complete sequence's view summary, so a mixed sequence-level value does not imply that every window crosses a label boundary.

A view-label change could reflect actual reorientation, camera movement, or annotation inconsistency. Some changes last only one frame. These are therefore not verified turns. Conversely, a constant coarse view label does not establish that the person never turns. The evidence supports describing the intended segmentation. It does not support saying that all retained clips have been independently verified to contain no turns.

## Different sequence IDs can share source frames

For each population, we compared every pair of different sequence IDs from the same source. An overlap means that their original one-based `frame_num` sets intersect. A second calculation using inclusive span intersections reproduced the frame-set results exactly because all raw sequences are contiguous.

| Population | Sequences | Overlapping sequence pairs | Sequences in those pairs | Sources in those pairs | Distinct source/frame keys assigned to two sequences |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full catalog | 1,874 | 91 | 180 | 17 | 31,675 |
| Five-label scope | 666 | 17 | 34 | 3 | 2,722 |
| Retained sequence IDs | 456 | 17 | 34 | 3 | 2,722 |

The 17 retained pairs occur in `R8LRCiTvUz8` (12 pairs), `jzJIpY6vRLo` (four pairs), and `v1SoZ_S31pk` (one pair). For example, myopathic sequences `cljargosy00cx3n6lm6atozrz` and `cll8h0akb00683o6ls74ez7x9` share frames 119 through 205 in `R8LRCiTvUz8`. The JSON contains every pair and its frame intersection.

Shared source imagery does not by itself prove that two annotation boxes describe the same person, or that every normalized joint coordinate is identical. This audit did not resolve person identity from those boxes. The direct finding is that different sequence IDs do not guarantee disjoint imagery. Even without shared frames, repeated participant and acquisition conditions could create statistical dependence.

## Forecast windows are overlapping observations

Preparation uses 24 observations at nominal 10 Hz, with 16 prefix observations, eight future observations, and stride eight. Consecutive accepted windows can reuse 16 of their 24 observations. Deduplication removes a repeated complete `(source_id, frame_numbers)` tuple, while retaining partial overlaps.

The cohort has **2,815 unique window IDs and zero duplicate complete source/frame tuples**. However, **2,712 windows share at least one exact native source frame with another retained window**.

| Overlapping unordered window pairs | Pair count |
| --- | ---: |
| Within one sequence ID | 4,272 |
| Across different sequence IDs | 155 |
| **Total** | **4,427** |

The across-sequence pairs involve seven sequence pairs. Sixty-five across-sequence window pairs share a future target frame; 267 distinct source/frame keys appear as future targets in two retained windows. Within-sequence future target sets are disjoint under the declared stride, but prefix support and future/prefix support overlap. There are 29,949 unique source/native-frame keys across complete windows and 22,253 across future targets. Frame membership does not imply that all joints at that frame are valid, and these are not counts of independent participants.

All overlapping observations remain grouped by source. The audit confirms zero source IDs, sequence IDs, or relation components assigned to multiple outer folds. These descriptive checks require no change to frozen splits or outcomes.

## Distinct passes and statistical independence answer different questions

A separate straight walking pass provides another movement observation. It can improve coverage of posture, phase, view, and variation within a recording. It does not become a new independent participant or recording simply because it has another sequence ID or excludes a turn.

Repeated passes can share the person, impairment, habitual movement, clothing, camera projection, background, pose-estimation behavior, and recording conditions. Disjoint frames eliminate literal frame reuse while retaining those common causes of correlation. Here **73 of the 91 retained sources contribute multiple sequences**, with a median of three and a maximum of 30 sequences per source. The annotations do not provide a verified participant key establishing the number of people in these sources or any repeated people across sources.

The existing source-component folds and uncertainty calculations should remain intact. Presenting 456 sequences first communicates movement-observation scale. Presenting 91 source groups alongside that number explains evaluation grouping. Neither count should be renamed patient count. A different sequence-level statistical analysis would require its own declared identity and dependence model; manuscript wording cannot substitute it into a frozen evaluation.

## Accurate positive framing for draft03

The GAVD paper introduces the dataset as the largest online-accessible collection of abnormal-gait videos annotated by expert clinicians. This supports an attributed description of its contribution. This audit has not performed an exhaustive current comparison of every clinically annotated gait dataset. Avoid an unrestricted superlative spanning all sensor modalities or implying confirmed individual diagnoses. [Original GAVD abstract](https://arxiv.org/abs/2407.04190)

Suggested sequence-first manuscript wording:

> We evaluate on GAVD, a public collection of clinician-annotated gait videos introduced as the largest online-accessible collection of its kind. Its annotation catalog contains 1,874 walking sequences. The five gait categories considered here provide 666 sequences, and the fixed observation-quality and target-support rules retain 456 sequences that yield 2,815 short forecasting windows. These include 141 normal sequences. The sequences come from 91 source videos, including 29 with normal-gait annotations. GAVD's sequence boundaries are intended to describe walking in one camera-relative direction, so several sequences can provide distinct movement observations from one recording. We keep all sequences and windows from the same source together during evaluation and summarize uncertainty at that group level.

A compact caption or dataset-table entry can read: “The evaluated cohort contains 456 annotated walking sequences and 2,815 forecast windows from 91 source videos. Related observations remain grouped by source.” The sequence scale is useful evidence of movement coverage without assuming independent repeated observations.

## Evidence locations and replay

Paths below beginning with `worldmodels/`, `docs/`, or `artifacts/` are relative to `experiments/aaai-worldmodels`. Other paths are repository-relative. The JSON records the exact named input hashes and a separate CSV hash for each sequence.

| Evidence | Location |
| --- | --- |
| Local sequence and source definitions | `data/gavd/README.md:233` and `:241` |
| Pinned upstream direction definition | `data/gavd/source/UPSTREAM_README.md:57` |
| Contiguous-span sampling | `worldmodels/restart_data.py:36` |
| Window stepping and exact tuple deduplication | `worldmodels/restart_data.py:242` and `:259` |
| Sequence-wide view summary stored on a window | `worldmodels/restart_data.py:273` |
| Window, sequence, and source aggregation | `worldmodels/restart_experiment.py:151` |
| Source-level outer folds | `worldmodels/restart_experiment.py:157` |
| Frozen grouping and dependence contract | `docs/restart/data_contract.md:47` |
| Exact retained observations | `artifacts/runs/reference-v2-cohort-v1/cohort.csv` |

Run the following read-only replay from the repository root. It verifies input hashes and independently reconstructs the principal overlap counts with the Python standard library. It writes no files.

```bash
experiments/aaai-worldmodels/.venv/bin/python - <<'PY'
from pathlib import Path
from collections import defaultdict, Counter
from itertools import combinations
import csv, hashlib, json

root = Path.cwd()
project = root / "experiments/aaai-worldmodels"
audit = json.loads((project / "docs/draft03/reviewer_support/sequence_audit.json").read_text())
for name, expected in audit["input_hashes"].items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name

sequences = audit["sequence_inventory"]
for sequence in sequences:
    path = root / sequence["csv_path"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == sequence["csv_sha256"]
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    frames = sorted(int(row["frame_num"]) for row in rows)
    assert frames == list(range(frames[0], frames[-1] + 1))
    assert len({row["seq"] for row in rows}) == 1
    assert len({row["id"] for row in rows}) == 1

for population in ("catalog", "five_label_scope", "retained_sequences"):
    selected = [s for s in sequences if population == "catalog"
                or (population == "five_label_scope" and s["in_five_label_scope"])
                or (population == "retained_sequences" and s["retained_in_common_cohort"])]
    sources = defaultdict(list)
    for sequence in selected:
        sources[sequence["source_id"]].append(sequence)
    pairs = sum(max(x["first_frame"], y["first_frame"]) <= min(x["last_frame"], y["last_frame"])
                for group in sources.values() for x, y in combinations(group, 2))
    expected = audit["populations"][population]
    assert len(selected) == expected["sequences"]
    assert pairs == expected["raw_annotation_overlap"]["unordered_same_source_sequence_pairs_sharing_exact_annotation_frames"]
    print(population, len(selected), "sequences,", pairs, "overlapping sequence pairs")

with (project / "artifacts/runs/reference-v2-cohort-v1/cohort.csv").open(newline="") as stream:
    windows = list(csv.DictReader(stream))
sources = defaultdict(list)
for row in windows:
    sources[row["source_id"]].append((row["sequence_id"], set(json.loads(row["frame_numbers"]))))
counts = Counter()
for group in sources.values():
    for (a, frames_a), (b, frames_b) in combinations(group, 2):
        if frames_a & frames_b:
            counts["same_sequence" if a == b else "different_sequences"] += 1
assert len(windows) == 2815
assert counts == {"same_sequence": 4272, "different_sequences": 155}
print("Window overlap replay passed:", dict(counts))
PY
```

Both a frame-set method and a second interval calculation passed for raw annotations. Direct brute-force pair comparison reproduced the window counts. The audit rechecked all 1,874 annotation hashes and every named input hash after calculation. This is a descriptive manuscript-support audit, not an independent visual certification of each sequence or a new efficacy experiment.
