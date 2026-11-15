# Illustrations for the research proposal

Open the [visual guide](index.html) for a large, readable view. Each new diagram is saved as SVG, vector PDF, and PNG. All curves, values, and mechanism pairs are conceptual illustrations, not experimental evidence. Clinical citations and qualifications are in the [research direction](../neurological_research_direction.md).

| Figure | Purpose |
| --- | --- |
| [Biological chain](neurological_chain.svg) | Distinguishes hidden origins, movement effects, and observed pose. |
| [Condition routes](condition_routes.svg) | Explains overlapping effects of the four condition families. |
| [Predictable patterns](predictable_patterns.svg) | Shows why regularity alone does not establish normality. |
| [Support profile](support_profile.svg) | Explains departures in either direction from a normal reference. |
| [Immediate experiment](neuromechanics_experiment.svg) | Shows the normal-only, matched-prediction workflow. |
| [Intervention controls](intervention_controls.svg) | Separates relative timing, local context, and measurement changes. |
| [Physical world model](physical_world_model.svg) | Separates controller, body, feedback, and observations. |
| [Mechanism ambiguity](mechanism_ambiguity.svg) | Introduces compatible explanations and measurement selection. |
| [Original prediction comparison](predictive_benefit.svg) | Illustrates the hidden-ankle example in the README. |

The eight new diagrams are original Matplotlib drawings. Regenerate them from the repository root with the existing tutorial environment:

```bash
MPLCONFIGDIR=/tmp/aaai-figures experiments/aaai/.venv/bin/python \
  experiments/aaai/docs/draft00/images/build_neuromechanics_figures.py
```

The script also creates a PNG contact sheet for layout review. Individual images were inspected, and independent review identified and corrected legend overlaps, an ambiguous measurement arrow, and wording that overstated compensation. The visual guide uses relative links and can be opened locally without a server.
