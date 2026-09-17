# JEPA research agenda

Start with the [revised assessment of synthetic training selection](proposals/synthetic-training-selection.md#9-final-research-judgment). The proposal is scientifically plausible but speculative, suitable at most for a bounded feasibility test rather than the primary one-week route to a significant, novel paper result. This replaces the earlier endorsement recorded in the [independent review](reviews/synthetic-teaching.md). No new experimental result prompted the reassessment, and no numerical success estimate is justified.

The [motion-preservation study](../../docs/studies/motion-preservation/README.md) remains the home of the implemented experiments and their results. Proposals here describe possible next research directions; they do not change that study's execution protocol.

## Proposals by question

For a compact paper-style overview of the teaching hypothesis, read the [two-page summary of synthetic training selection](proposals/synthetic-training-selection-summary.md). The full proposal retains the detailed protocol and research assessment.

All eight proposals live in one folder. The topics below provide a reading order without adding small nested directories. P1–P7 are the identifiers from the earlier September 14 comparison, not experiment numbers or the current priority order.

| Topic | Proposal | Question |
| --- | --- | --- |
| Teaching | [Synthetic training selection](proposals/synthetic-training-selection.md), feasibility hypothesis | Which labeled synthetic examples help an unfamiliar pose estimator on real video? |
| Teaching | [Distillation target selection](proposals/distillation-target-selection.md) — P7 | Which representation targets actually help a skeleton forecasting student? |
| Forecasting | [Observation selection](proposals/observation-selection.md) — P1 | Which additional observation improves a motion forecast? |
| Forecasting | [Motion beyond joints](proposals/motion-beyond-joints.md) — P2 | What useful movement information do joint positions leave out? |
| Forecasting | [Cross-activity prediction](proposals/cross-activity-prediction.md) — P4 | Can a person's walk help predict another activity? |
| Forecasting | [Uncertainty and evidence](proposals/uncertainty-and-evidence.md) — P5 | Does forecast uncertainty respond appropriately to additional evidence? |
| Measurement | [Repair verification](proposals/repair-verification.md) — P3 | Can a model judge whether a proposed tracking correction helps? |
| Measurement | [Motion ambiguity](proposals/motion-ambiguity.md) — P6 | Can different valid motions explain the same observations? |

Synthetic training selection and distillation target selection ask different questions and remain separate proposals.

## Evidence and references

| Reading | Scope |
| --- | --- |
| [Portfolio evidence](references/portfolio-evidence.md) | Completed results, data limits and the earlier P1–P7 experiment plan. Its annotation budget does not apply to the later synthetic-training proposal. |
| [Portfolio literature](references/portfolio-literature.md) | Prior work, checkpoints and novelty limits for P1–P7. The later teaching proposal and its review contain their additional sources. |
| [Experiment guidelines](references/experiment-guidelines.md) | Earlier shared evidence and execution constraints, still cited by the motion-preservation study. |
| [Papers and checkpoints](references/papers-and-checkpoints.md) | Earlier literature and model-access ledger, still used as a study reference. |
| [Optical-flow evidence](references/optical-flow-evidence.md) | Image-motion measurements, their limitations and the original motion-beyond-joints argument. |
| [September 13 review summary](references/review-summary-2026-09-13.md) | Scientific and figure corrections to the earlier portfolio. |

## Decisions and independent reviews

Read the decision stages in order when tracing how the recommendation changed:

1. [September 12–13 portfolio](archive/2026-09-13/README.md): the original motion-preservation recommendation and deferred alternatives.
2. [September 14 proposal comparison](proposal-comparison.md): the subsequent seven-option comparison, initially prioritizing observation selection and motion beyond joints.
3. [Direction-search review](reviews/direction-search.md): a later search that initially found no new flagship; its conclusion was superseded by the next stage.
4. [Synthetic-teaching review](reviews/synthetic-teaching.md): the historical independent assessment supporting a qualified endorsement.
5. [Revised research judgment](proposals/synthetic-training-selection.md#9-final-research-judgment): the current assessment limits this direction to a bounded feasibility test. The earlier endorsement overstated the evidence for transferable teaching under the one-week objective.
6. [Further significance-focused reassessment](reviews/significance-first-reassessment.md): a more focused question about preserving predictive history beyond current kinematic state, with the independent objections and decisive comparisons. It does not establish a new result or endorse a high-probability deadline outcome.

The other September 14 reviews cover the seven-option portfolio: [research options](reviews/research-options.md), [gait data](reviews/gait-data.md), [world models](reviews/world-models.md), [methods](reviews/portfolio-methods.md), [draft figures](reviews/portfolio-figures-draft.md), and [completed revisions](reviews/portfolio-revisions.md). The draft figure review records issues before correction; the revision record does not claim to review the later teaching proposal.

## Figures and historical material

[Figures](figures/) contains the seven-option portfolio's conceptual diagrams, named after their proposal topics. Each has a `-mechanism.svg` and `-experiment.svg`; PNG previews and contact sheets are in `figures/previews/`. The synthetic-training tutorial also has a [teaching workflow](figures/synthetic-teaching-workflow.svg) and a [before/after comparison](figures/synthetic-teaching-comparison.svg), with matching PNG previews. These are method illustrations, not measured results.

Regenerate the original seven-option set from the repository root with:

```bash
.venv/bin/python notes/research-agenda/figures/generate_figures.py
```

The generator requires CairoSVG, Pillow and Matplotlib. On macOS with Homebrew Cairo, set `DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib` if Cairo cannot be found. The [layout checks](figures/layout-checks.json) record text collisions and canvas overflow.

[Archive: September 12–13](archive/2026-09-13/README.md) keeps the earlier proposals, reviews, diagrams, original research brief and mathematical check together. Archiving this decision stage does not invalidate its evidence or the original motion-preservation diagram still used by the active study. Dates identify historical rounds; topic names identify the current reading material.
