# Build the proposal and reproduce source audits

[Research proposal](../README.md) · [Evidence inventory](../evidence/README.md) · [Visual gallery](../images/gallery.html)

Run these commands from the gavd6 repository root. They rebuild documentation or inspect source files; they do not train a restoration model. The existing `.venv` supplies Pillow, CairoSVG, Mistune and the audit dependencies. On this Mac, CairoSVG also uses the Homebrew Cairo library and local Arial fonts.

For GPU experiments, read the [eight-H100 execution specification](../methods/execution.md). Its 102-fit manifest is a plan; the builders below do not implement the parallel training coordinator or certify the one-hour setup assumption.

## Rebuild the complete reading experience

```bash
DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib \
  .venv/bin/python docs/studies/gait-fidelity/scripts/build_figures.py
DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib \
  .venv/bin/python docs/studies/gait-fidelity/scripts/build_proposal_figures.py
DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib \
  .venv/bin/python docs/studies/gait-fidelity/scripts/build_execution_figure.py
.venv/bin/python docs/studies/gait-fidelity/scripts/build_paper.py
.venv/bin/python docs/studies/gait-fidelity/scripts/build_video_gallery.py --thumbnails
```

Open [proposal.html](../proposal.html) for the navigable paper and illustrative response explorer. The Markdown README is its scientific source; the paper builder adds navigation, figure enlargement and the interactive explanation. The illustrative numbers are unrelated to experiment results. The [figure gallery](../images/gallery.html) includes all editable SVGs and smaller review previews.

The [local-video viewer](../data/video-gallery.html) needs the sibling `experiments/multiple-sclerosis` checkout. It links original MP4s and stores optional JPEG thumbnails under `images/local-video-review/`, where they are ignored by Git. Export review notes to JSON to retain them outside browser storage. The viewer is a feasibility-review tool; reference overlays and a blinded clinical annotation system remain separate implementation work.

## Reproduce the completed preparation audits

Write new audit output to a separate directory so that dated evidence remains intact:

```bash
.venv/bin/python docs/studies/gait-fidelity/scripts/audit_masking.py \
  --root "$PWD" --output /tmp/gait-fidelity-mask-check.json
.venv/bin/python docs/studies/gait-fidelity/scripts/audit_video_inventory.py \
  --output /tmp/gait-fidelity-video-check
.venv/bin/python docs/studies/gait-fidelity/scripts/audit_video_cache.py \
  --output /tmp/gait-fidelity-video-check
.venv/bin/python docs/studies/gait-fidelity/scripts/audit_video_timestamps.py \
  --output /tmp/gait-fidelity-video-check
```

The timestamp audit reads the inventory produced by the preceding command. Video audits require ffprobe; thumbnail generation uses ffmpeg. These commands read the sibling videos and caches without modifying them. [Evidence](../evidence/README.md) records their scope, source hashes and limits.
