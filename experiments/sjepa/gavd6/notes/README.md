# Notes index

The notes tree separates current references, active research, and historical
inputs. This prevents an old prompt or draft from looking like an implemented
protocol.

## Current references

- [S-JEPA evolution tutorial](current/methodology/sjepa_evolution_tutorial.md): implementation and methodology evolution
- [Citation verification](current/literature/citation_verification.md): literature and citation audit
- [Codex review log](current/reviews/codex_review_log.md): adversarial implementation review
- [Diagram design system](current/design/diagram_design_system.md): SVG conventions

Polished project documentation remains under [`docs/`](../docs/README.md).

## Active research

- [Research portfolio](research/README.md): index of the twelve proposals
- [Research scorecard](research/SCORECARD.md): proposal evaluation rubric and scores
- [Shared facts](research/_shared_facts.md): common quantitative and provenance constraints
- [Neuroscience facts](research/_neuro_facts.md): verified domain anchors
- [GaitParity program](research/programs/gait_parity/README.md): joined Idea 05/09 research program and AMASS tutorials
- [`research/ideas/`](research/ideas/): one directory per proposal

Research notebook source lives under [`notebooks/experiments/`](../notebooks/experiments/),
and its deterministic builders live under
[`scripts/notebook_builders/`](../scripts/notebook_builders/).

## Archive

- [`archive/prompts/`](archive/prompts/): original requests and execution prompts
- [`archive/early_plans/`](archive/early_plans/): superseded planning documents
- [`archive/paper_drafts/`](archive/paper_drafts/): early manuscript framing

Archive files are preserved as historical records. Their commands and paths may
refer to the repository layout that existed when they were written and should
not be treated as current instructions. Verify implementation claims against
the executable notebooks, builders, and artifact manifests.
