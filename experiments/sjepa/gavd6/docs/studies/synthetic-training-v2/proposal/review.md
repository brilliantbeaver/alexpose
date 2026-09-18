# Proposal figure and workflow review

The installed `codex:adversarial-review` plugin reviewed the proposal, workflow and all five rendered PNGs, including at 900-pixel display width. Its [initial output](review-initial.md) is preserved verbatim. It found no material overflow, illegibility, clutter or arrow intersections, but identified one medium-severity semantic defect in Figure 4.

**Correction:** evaluation references previously appeared inside a shared model-input box. Figure 4 now separates matched training data (including training-only clean labels) from scoring references, which feed only the common evaluation box. The generator, SVG, PNG and HTML were regenerated; the author visually rechecked Figure 4. No text collisions or canvas overflows were reported by the automated bounds check.

**Follow-up verdict: approve.** The [second review output](review-final.md) is preserved verbatim. The reviewer visually inspected all five PNGs at native size and 900-pixel width and found no material overflow, clutter, intersecting arrows or illegibility. It confirmed the clean-coordinate controls, training/deployment separation, scoring-reference isolation and development/confirmation boundaries. It reported no material findings. The request was run through the installed `codex:adversarial-review` companion runtime, version 1.0.6, in read-only review mode; the author made the correction between review passes.

Local validation also checked SVG XML, accessible titles/descriptions, absence of embedded raster images, four requested HTML sections, five figure references with alt text and all local proposal links. The [layout report](images/layout-check.json) covers text/text intersections and canvas bounds; it is not a substitute for the visual review. The [asset manifest](reviewed-assets.json) records SHA256 hashes of the delivered proposal, generator and images.

## Reproduce the assets

From the repository root on the current macOS environment (CairoSVG, Pillow, Mistune, Arial and Homebrew Cairo):

```bash
DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib .venv/bin/python \
  docs/studies/synthetic-training-v2/proposal/generate_figures.py
```

The SVGs are editable vector originals in `images/`; PNGs in `images/previews/` are visual-review copies. `proposal.html` embeds the original SVGs. Rendering on another platform requires its local Cairo library and a corresponding font-path adjustment in the generator.

These reviews assess communication and workflow consistency, not empirical model effectiveness. All trajectories are conceptual and all proposed research benefits remain unmeasured.
