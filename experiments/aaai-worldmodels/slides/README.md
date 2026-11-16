# Reference-aware gait anomaly detection

Open [the current illustrated tutorial](iteration02.html) or [the 36-page vector PDF](iteration02.pdf). The deck connects neurological reasoning, the actual GAVD ingestion pipeline, completed results, and the next controlled experiments. Each page marks measured evidence, post hoc diagnostics, or proposed work. The HTML includes full-sentence explanations and links to evidence beneath every slide. [tutorial.html](tutorial.html) opens this deck.

The strongest anomaly result is the declared excess-error score with source AUROC 0.793. The deck also shows that a post hoc negated-persistence control reaches 0.806. Therefore, these results motivate a reference-aware evaluation and stronger motion controls; they do not establish superiority to a movement-only explanation. The current-pose transition also forecasts better than bilateral history. Both findings remain visible beside the favorable results.

## Follow the reasoning

| Slides | Purpose |
| --- | --- |
| 1–4 | Establish the observation-layer contribution and explain what neurological reasoning permits us to test. |
| 5–12 | Trace the complete inventory, cohort attrition, Ambient ingestion, frame conversion, time grid, joint reference and missingness. |
| 13–16 | Explain the comparison arms, matched timing control, nested source folds and aggregation. |
| 17–21 | Show frozen forecast results, paired controls, declared anomaly scores and the post hoc low-motion diagnostic. |
| 22–30 | Specify confound controls, matched masking alternatives, reliable-cycle requirements, objective checks and normal-only calibration. These experiments are proposed. |
| 31–36 | Connect the notebook roadmap, expanded GAVD use, generalization requirements and adversarial acceptance checks. |

The masking examples preserve the latest target anchor, hide the same number of valid prefix cells, and keep every future target hidden. The anatomical example uses a connected foot region. Its cells are schematic rather than sampled poses. Anatomical masking itself is not claimed as novel; the notes cite MAMP, S-JEPA, ASMa and SLiM where relevant.

## What GAVD establishes

The completed cohort has 2,815 windows from 456 sequences and 91 source videos, including 29 normal sources. The full inventory contains 1,874 sequences and 348 source IDs. The five requested annotations already include every annotated normal source. Additional labels can support a separate transfer study, but exercise, pregnancy and style must not automatically become disease labels.

GAVD supports exploratory internal generalization across heterogeneous source videos. It does not establish verified participant separation, independent clinical diagnoses, site-disjoint clinical transportability, treatment response, or safe deployment. More overlapping windows cannot supply those missing forms of evidence. The next study needs a frozen comparison menu and, where accessible, compatible external data with verified participant identities. These limits are explained on the slides and in the [experiment README](../README.md).

## Regenerate and inspect

Editable source is [build_iteration02_slides.py](../scripts/build_iteration02_slides.py). It reads the frozen `restart-dynamics-v1/results.json`, `restart-full10hz-v1/preparation.json`, and `restart-full10hz-v1/inventory.json`. It does not train models, download videos, alter those runs, or overwrite the preserved restart deck.

From the repository root:

```bash
experiments/aaai-worldmodels/.venv/bin/python \
  experiments/aaai-worldmodels/scripts/build_iteration02_slides.py
```

The generator exports SVG, PNG and single-page PDF files in [iteration02_images](iteration02_images/), the combined vector PDF, HTML, [narrative source](iteration02_source.json), [input and output manifest](iteration02_manifest.json), and [layout checks](iteration02_layout_checks.json). SVG text remains editable. All 36 pages have been rendered and visually inspected at full size. The validation record describes numerical replay, PDF checks, review corrections and remaining limitations in [iteration02_visual_review.md](iteration02_visual_review.md).

The earlier [restart deck](restart.html), [restart PDF](restart.pdf), and [original JEPA tutorial](tutorial_before_restart.html) remain preserved. Read [draft02](../docs/draft02/README.md), the [research instruction](../../../notes/aaai-worldmodels/05-iterate-02-INSTR.md), and the [literature review](../docs/iteration02/literature_and_masking.md) for the current manuscript and next execution contract.
