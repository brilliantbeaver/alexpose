# Archived iteration02 research plan

This planning deck was preserved before new model execution. Its proposal status is historical and must not be confused with the completed-results deck.

# Proposed second-iteration research slides

Open the [40-page vector PDF](tutorial.pdf), the [offline HTML viewer](index.html), or the [Markdown notes](tutorial.md). The HTML viewer includes expandable explanations and source links. Use the left and right arrow keys to move between slides. Each page also has an editable SVG, vector PDF and PNG in [images](images/).

This is a research plan, not a report of new model results. Slides 2 and 3 preserve the completed draft01 findings. Slides 10 through 12 report read-only support and annotation checks. The remaining methodological diagrams explain proposed comparisons. Notebooks 12 through 17 and the proposed residual predictor have not been implemented or run. The [canonical instructions](../../../../../notes/aaai/05-iterate-02-INSTR.md) define the experiment in full.

The masking extension compares U, J and A with exactly 64 hidden tokens and a one-half marginal hiding probability per token. U samples uniformly. J hides a complete single-joint tube, while A hides a width-weighted anatomical group over an aligned temporal block. J and A each hide an eight-token core and then uniformly fill 56 other positions. All training arms use the same frozen U evaluation bank; the side-balanced distal-leg bank is a separate diagnostic. The coordinate ablation retains one U-trained baseline, and any JEPA pretraining-mask ablation keeps downstream head masks fixed to U. No new masking method has been trained or evaluated.

The core recommendation is to retain a competent own-leg predictor, then test whether observations of the other leg add information beyond local geometry and extra model capacity. The plan preserves movement size, projected posture and raw timing. It tests a compound distal timing-and-geometry input change using visible hip-relative offsets from a 48-frame parent, fixed targets, pair-specific common support and unchanged normalization anchors. The primary timing effect is the frozen both-leg prediction error with shifted context minus aligned-context error. Prediction adequacy is rechecked on the exact sparse paired inputs. It does not infer a neurological lesion from a video.

The new exploratory primary endpoint is normal versus the four requested condition families. The read-only candidate subset has 90 sources. The mandatory secondary endpoint uses all 12 folders, with 304 candidate sources, and applies the same frozen scores. These are eligibility counts, not accuracy results. All 28 candidate normal sources have appeared in prior exploration, so the redesign is not an independent confirmatory study. The historical all-folder primary endpoint remains unchanged. The new primary AUROC is a fixed pair-count-weighted mean of within-fold AUROCs. Paired bootstrap comparisons preserve fold and binary-label strata; average precision is reported per fold with prevalence. A failed relational gate sets R to zero for that fold, preserving the local-motion ranking within the fold.

## Read the deck by topic

| Slides | Topic |
|---|---|
| 1–4 | The question, completed evidence and evidence status. |
| 5–8 | Neurological examples and the limits of the novelty claim. |
| 9–13 | Ingestion, source coverage, duration costs, contact labels and allowed joints. |
| 14–19 | Matched U/J/A masking recipes, prior art, fixed evaluation banks and masking-specific gates. |
| 20–27 | Nested predictors, capacity controls, phase leakage, timing probes and separate score channels. |
| 28–31 | Source roles, paired evaluation, baselines and the new endpoint decision. |
| 32–35 | Six planned notebooks, local execution and scientific gates. |
| 36–40 | The week plan, compute discipline and evidence-dependent manuscript outcomes. |

Six shared diagrams are available as SVG, PDF and PNG in [images/iteration02](../../../images/iteration02): [rationale](source_figures/iteration02/rationale.svg), [residual predictor](source_figures/iteration02/residual_predictor.svg), [timing controls](source_figures/iteration02/timing_controls.svg), [execution gates](source_figures/iteration02/execution_gates.svg), [mask families](source_figures/iteration02/mask_families.svg) and [mask comparison](source_figures/iteration02/mask_comparison.svg). Each visibly identifies itself as an illustration of a proposal. None contains an invented empirical curve.

## Rebuild locally

From `experiments/aaai`, use the existing project environment:

```bash
uv sync
uv run python scripts/build_iteration02_slides.py
```

The builder uses Matplotlib and Pillow from the existing `pyproject.toml`. It also needs Poppler for PDF assembly, rendering, text-bound checks and font inspection. On macOS, install them with `brew install poppler` if they are unavailable. The script does not download videos, fit models or change experiment artifacts.

The builder reads [draft01/evidence.json](../../../docs/draft01/evidence.json), the two completed runs' verification reports, and [data_feasibility_probe.json](../../../docs/iteration02/data_feasibility_probe.json). It writes rounded display values from those records. [result_input.json](result_input.json) preserves the loaded values and input hashes. [build_manifest.json](build_manifest.json) records the generated PDF hash and the exact archived PDF hash. [deck_spec.json](deck_spec.json) contains the slide titles, explanations and scientific references.

The final PDF uses embedded TrueType fonts and vector shapes. All 40 PDF pages have been rendered in [review](review/). The [visual review](visual_review.md) records layout inspection and the scope of the checks. A successful build verifies the presentation artifacts, not the proposed scientific claims.

## Historical completed deck

The exact previous 40-page PDF is preserved in [archive/draft01_completed/tutorial.pdf](../draft01_completed/tutorial.pdf), together with its HTML, Markdown, images, source figures and review artifacts. Its SHA-256 is `bb0502c207305ac9a5f2746c3af5dfaac7eda8c5d7805aa6007726f86fc317d8`. Relative source links in the archived HTML and Markdown were repaired after relocation; the PDF bytes were not changed. The [earlier tutorial archive](../pre_neuromechanics/index.html) also remains available.

The historical builder, `scripts/build_research_slides.py`, generates the completed-study presentation. Running it with its default output directory would replace the active planning deck, so use the new builder above for this iteration.
