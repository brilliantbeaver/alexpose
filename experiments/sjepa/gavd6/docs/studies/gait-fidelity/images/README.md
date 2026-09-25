# Gait Fidelity figures

[Current study](../README.md) · [Interactive paper](../proposal.html) · [All figures](gallery.html) · [Response protocol](../methods/jepa-response.md)

The current proposal uses the [complete method diagram](research-method.svg), its [compact overview](research-method-compact.svg), and two Matplotlib results graphs: [full](research-core-results.svg) and [compact](research-core-results-compact.svg). The graphs read person-balanced means and declared paired contrasts from `results/core-analysis-20260924`; the other diagrams are explanatory schematics. The [photographic context](context/README.md) contains two licensed crops from prior research, with original figures and source provenance. Neither photograph is data from this study.

The original eighteen editable SVGs explain the initial JEPA response follow-up. The additional [two-readout design](proposal-readout-design.svg) illustrates the amended `jepa-response-02` experiment; its [builder](../scripts/build_readout_figure.py) creates a separate PNG preview. The proposal’s mathematical SVGs live in `equations/` and are generated from its Markdown source. Examples, heatmaps and traces in the older concept diagrams are illustrative rather than experimental results. Each SVG has a descriptive title and alternative text; the original eighteen also have 1200-pixel and 900-pixel PNG previews.

Rebuild the four current research figures and their previews with:

```sh
DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib .venv/bin/python \
  docs/studies/gait-fidelity/scripts/build_research_visuals.py --preview
```

The full results graph adds a separate uncertainty panel for the declared JEPA-minus-direct comparison; the compact graph shows means, with uncertainty in the surrounding text. Both use common scales, objective-specific colors and shapes, unchanged-pose references, and one-decimal labels. [The Matplotlib builder](../scripts/build_results_figure.py) checks rendered text and line intersections and also exports PNG previews. [Independent adversarial review](../reviews/results-figures-matplotlib-20260924.md) checked numerical meaning and final visual clarity.

The [two-page PDF](../proposal-brief.pdf) embeds the compact method and results figures as vectors, preserving readable text at print resolution. Its 10.5-point body type and figure labels were inspected in rendered pages. See the [document review](../reviews/proposal-overview-20260924.md).

The scope ribbon tells the reader how to use a figure:

| Ribbon | Meaning |
| --- | --- |
| **FOLLOW-UP** | The current comparison: errors across two movement states versus separate endpoint errors, with a coordinate-error control. |
| **INHERITED CORE** | Data, geometry, splits, transformations and masking reused from the completed parent run. |
| **INHERITED DESIGN / SCHEMATIC** | The crossed movement/observation design, illustrated without implying pending core results. |
| **DIAGNOSTIC** | Checks that help explain learned features and available training evidence. |
| **DEFERRED** | Real-video or bilateral interpretation extensions outside this run. |

Features are learned numbers. A residual is a prediction error. Coupling compares the errors from two movement states. Freezing an encoder keeps its weights fixed while a new coordinate output network is trained. Masking hides context during training. The pictures introduce these terms where needed; exact implementation details remain in the linked protocol.

| Concept | Figures |
| --- | --- |
| Current question, controls and loss | [01 · Movement response](01-research-question.svg), [06 · Three training variants](06-matched-mask-experiment.svg), [14 · Training stages](14-paired-jepa-method.svg), [JEPA response · Difference of errors](jepa-response.svg) |
| Inherited data and geometry | [02 · Movement and observation](02-crossed-design.svg), [09 · Parent population and splits](09-source-splits.svg), [11 · Transformation rules](11-transformation-contracts.svg) |
| Hidden input and available references | [03 · Existing graph masks](03-changing-graph-masks.svg), [04 · Three availability roles](04-mask-contract.svg), [15 · Joint and time support](15-graph-time-mask.svg) |
| Calibration, diagnostics and evaluation | [05 · No-change and coverage checks](05-coverage-audit.svg), [12 · Fixed calibration and comparison](12-evidence-workflow.svg), [16 · Projected movement response](16-response-estimand.svg) |
| Claim boundaries and deferred work | [07 · Real-video evidence](07-video-evidence.svg), [08 · Future inspection viewer](08-data-review-view.svg), [10 · Anatomical side ambiguity](10-assignment-uncertainty.svg), [13 · Data and claims](13-data-to-claims.svg) |
| Current resources | [17 · Nine models and 48 H100-hours](17-parallel-execution.svg) |

The original `jepa-response-01` design illustrated by the earlier figures has three variants, seeds 17/29/43 and eighteen fitting phases. The amended `jepa-response-02` adds coordinate-only readouts, for 27 phases and 18 final models, with a September 25, 2026, 8 AM Pacific deadline. The following budget description belongs to the original design. It reuses the parent's actual update budget and reference geometry. Calibration uses 32 fixed training batches and a shared coefficient for the two JEPA additions. The resource allowance is 48 additional H100-hours, with results due September 24, 2026 at 18:00 Pacific. These drawings do not add datasets, GAVD evaluation or confirmation people.

Rebuild all SVGs, both preview sizes and the layout reports from the repository root:

```sh
DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib .venv/bin/python \
  docs/studies/gait-fidelity/scripts/build_figures.py
```

The command requires Pillow and CairoSVG. The earlier `build_proposal_figures.py` and `build_execution_figure.py` entry points call this same builder, so all figures remain consistent. The paper builder owns the HTML gallery.

`figure-layout.json` registers all eighteen filenames, titles, descriptions, scope labels and measured geometry. `layout-check.json` records asset hashes and checks for text bounds, panel containment, text overlap and connector intersections. The proposal/execution JSON files retain corresponding subsets. These automated checks supplement inspection of the rendered images and scientific review. Dated reviews describe the versions and hashes recorded in those reviews.

`previews/` contains the current renders. `local-video-review/` contains the separate thumbnails used by the [video viewer](../data/video-gallery.html); they are inspection frames, not reference annotations.
