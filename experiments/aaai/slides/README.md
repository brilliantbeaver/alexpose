# Understanding gait prediction

This 43-slide deck explains the study for general AI/ML researchers. It begins with a hidden-ankle example, introduces the data and three predictors, and then examines prediction quality, anomaly scores and independent laboratory measurements. The experiments and results are unchanged.

Open the [offline viewer](index.html), [slide PDF](tutorial.pdf), or [detailed speaker notes](tutorial.md). The viewer has a slide selector, keyboard navigation, a text version of each slide and a button to open all notes. The viewer offers Light and Dark buttons on every slide, a device-aware deck default, and a control that returns all slides to that default. Theme choices persist in this browser. The PDF retains the light theme. Both SVG themes embed their fonts, so they work offline without a system-font dependency. Score-focused slides also include a [plain-language guide to M, R and S](score_guide.md). Every slide has five note sections: purpose, explanation, reading the visual, evidence and limits, and transition.

The [manuscript](../docs/draft03/README.md) uses the same terminology. Random, whole-joint and anatomical masks describe training. Scattered and grouped hiding describe evaluation. The movement score describes posture and movement size; the cross-leg score compares the benefit of seeing the other leg with separate normal reference recordings. Benefits that are unusually high or unusually low increase that score. The combined score adds these two numerical scores and divides by two. This fixed average is tested against movement alone; none of the scores is a probability of disease.

The [input record](result_input.json), [build manifest](build_manifest.json), [render checks](review/validation.json), [browser checks](review/viewer_validation.json), and [visual review](visual_review.md) document the evidence and verification. The [previous completed deck](archive/before_accessibility/tutorial.pdf), [planning deck](archive/iteration02_plan/tutorial.pdf) and [draft01 deck](archive/draft01_completed/tutorial.pdf) remain archived.

From the repository root, rebuild with:

```bash
uv run --project experiments/aaai python experiments/aaai/scripts/build_iteration02_result_slides.py --run-dir experiments/aaai/artifacts/iteration02/d66e1b126c9d9a9adf1b --external-dir experiments/aaai/artifacts/iteration02/external_fb9d0d23b56ce29fd846
```

After rebuilding, run the browser checks with `uv run --project experiments/aaai --with playwright python experiments/aaai/scripts/validate_gait_slide_viewer.py`. The report is marked pending until these checks finish.

Visible content is maintained in `scripts/accessible_gait_deck.py`. Structured note sources live in `slides/source/notes_*.json`. The offline viewer and its explicit dark SVG palette are maintained in `scripts/gait_slide_viewer.py`. The builder exports light and dark SVG pages, light PDF and PNG pages, assembles the deck, renders every PDF page and checks links, bounds and fonts. It reads saved evidence without training models or changing scores.
