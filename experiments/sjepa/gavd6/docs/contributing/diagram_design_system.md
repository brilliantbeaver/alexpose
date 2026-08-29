# Diagram design system (images/*.svg)

Shared visual language for the nine tutorial SVGs. Every redesigned figure MUST
follow these tokens so the set reads as one coherent system. The goal is modern,
clean, uncluttered: generous whitespace, aligned edges, non-overlapping text,
breathing room around every arrow and box.

## Canvas
- `viewBox="0 0 1200 720"` (fixed for all nine; was 675 — the extra height buys
  vertical breathing room and prevents footer crowding). `role="img"` + `<title>`
  + `<desc>` retained for accessibility (keep the existing wording).
- Background fill: `#f7f5ef` (warm paper). One flat rect, no gradients.
- Outer safe margin: nothing closer than **48px** to any canvas edge.

## Type ramp (Arial / Helvetica, system sans)
- `.h`  title:        `700 32px`  fill `#0f2440`  — one per figure, top-left.
- `.t`  card title:   `700 20px`  fill `#0f2440`.
- `.b`  body:         `16px`      fill `#33475e`.
- `.s`  caption/note: `14px`      fill `#5c6f84`.
- `.w`  on-dark text: `700 16px`  fill `#ffffff`.
- `.ws` on-dark note: `14px`      fill `#cdd8e4` — muted secondary line inside a
  dark banner (e.g. loss sub-terms), a light-slate step down from white.
- Line-height: leave **>=26px** between stacked text baselines (was ~24 and felt
  tight). Never place two text baselines closer than 22px.

## Color palette (muted, few hues)
- Ink / stroke:      `#20364e`
- Rule / divider:    `#c4cdd8` (1.5px)
- Accent (primary):  `#2f6f99` (blue)
- Accent (positive): `#5f9e7e` (green) — teacher / target path
- Accent (warm):     `#e07a4b` (orange) — highlights, "hot" nodes, student path
- Accent (violet):   `#8a6fb3` — predictor
- Card fills (pastel, low saturation):
  - blue  `#e7f0f8`
  - green `#e6f2ea`
  - warm  `#f7e7d8` (warm sand)
  - violet`#ece6f6`
  - neutral card `#ffffff`
- Dark banner fill: `#20364e`, text white.
- **Auxiliary tokens** (derived from the core hues; permitted only where noted):
  - Skeleton neutrals (fig. 03 only): bone `#8a9bab`, joint node fill `#d8e0e7`
    with `#20364e` stroke — a desaturated grey ladder so the *highlighted* target
    joints (warm `#e07a4b`) stand out against unselected structure.
  - Darkened accent strokes: each pastel/accent fill may carry a hand-darkened
    border for contrast against the paper — e.g. hot-node stroke `#a44c26`
    (darker `#e07a4b`), risk-pill stroke `#c0483a`. Use only the darker shade of
    an already-sanctioned hue; do not introduce a new hue.
  - Tints: a lighter wash of a core hue is allowed for emphasis fills — e.g.
    `#f6e2df` (warm tint) on the "blocked / risk" lane in fig. 08. Keep it within
    the same hue family as the core accent it derives from.

## Boxes / cards
- Corner radius `rx="16"`.
- Stroke `#20364e` width `1.75`.
- **Text inset >= 22px** from every card edge (top text baseline sits >=30px
  below the card's top edge; left edge of text >=22px from card left). This is
  the single biggest fix — old figures had text touching borders.
- Minimum gap **between adjacent cards: 56px** so arrows have room.

## Arrows / connectors
- One shared marker: `<marker id="arrow" ...>` filled `#20364e`, `markerWidth=8`.
- Stroke `#20364e` width `2.5`, `fill:none`, `marker-end:url(#arrow)`.
- **Prefer straight horizontal/vertical runs.** If a dogleg is unavoidable, use a
  single right-angle bend with a >=20px landing segment before the arrowhead;
  never route an arrow through or under a box.
- EMA / secondary path: dashed `stroke-dasharray:7 6`, color `#5f9e7e`, and route
  it in clear space with a short text label beside it (not overlapping).
- Leave >=16px clear space between an arrowhead and the box it points at.

## Layout discipline
- Establish a horizontal baseline grid; align card tops and text left-edges.
- Default flow is left->right, top->bottom. A multi-row *sequence* (e.g. the
  seven-step roadmap in #09) may use a controlled boustrophedon: read a row
  left->right, then drop straight down and read the next row right->left, so the
  reading path is continuous and the return bend is a single clean right-angle
  elbow. This is permitted only when (a) each row is itself strictly
  left->right or right->left, (b) the direction reversal is signalled by the
  connector geometry, and (c) there is exactly one bend per row transition. The
  old #09 was banned for tangled, ambiguous routing — not for having rows.
- One idea per card. If a card needs >4 body lines, split it or shorten copy.
- Footer note (if any): single line, `.s`, on its own row with >=24px above it,
  never overlapping the last row of cards.

## Invariants (do not change)
- Keep each file's semantic content and the existing `<title>`/`<desc>` meaning.
- Keep the same filename and the same conceptual message; this is a visual
  refresh, not a content rewrite.
- Valid standalone SVG 1.1, no external fonts, no scripts, no raster images.
- Must render identically in `IPython.display.SVG` (Jupyter) and via
  `rsvg-convert` (the pandoc->LaTeX PDF path).
