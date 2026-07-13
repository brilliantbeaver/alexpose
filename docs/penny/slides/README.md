# Gait-JEPA interactive slides

A single self-contained slide deck that explains the Gait-JEPA
proposal: the problem it solves, the JEPA approach from scratch, the
specific skeleton-and-block-masking method, the research questions with
their metrics and targets, and the neuroscience grounding.

It is the presentation companion to the concept proposal one folder up
([`../README.md`](../README.md)) and the hands-on tutorial series in
[`../tutorials/`](../tutorials/).

## How to view

Open [`slides.html`](slides.html) in any modern browser. Nothing to
install and no network needed. The deck is one HTML file with inline CSS
and JavaScript, and it loads the vector illustrations from
[`images/`](images/).

## Controls

- Right arrow, space, or Page Down: next slide.
- Left arrow or Page Up: previous slide.
- Home and End: jump to the first or last slide.
- Click the right third of the screen to advance, the left third to go back.
- Swipe left or right on a touch screen.
- `O`: open the overview grid, then click any slide to jump to it.
- `F`: toggle fullscreen.
- The address bar carries the slide number (for example `slides.html#7`),
  so you can deep-link to any slide.

## What is inside

Eighteen slides, following the parts of the proposal:

1. Title
2. The label bottleneck (only 68 labeled clips)
3. Why 82 hand features throw away the walk
4. The two-step big idea: learn first, label last
5. Self-supervised learning as fill-in-the-blank
6. Why JEPA predicts meaning, not pixels
7. The four JEPA pieces
8. The collapse trap and the VICReg fix
9. This proposal's twist: JEPA on skeletons
10. The two block-masking styles
11. The 33-joint pose graph reference
12. The neuroscience connection
13. Penny's feature-to-condition mapping, the neuroscience bridge
14. The four research questions with metrics and targets
15. The end-to-end pipeline versus the prior path
16. How you will know it worked, including the null-result framing
17. Risks and mitigations
18. Summary and grading

## House style

The deck follows the same rules as the rest of this series: a light,
uncluttered, pastel theme, vector illustrations rather than photos, plain
language, and no em-dashes or en-dashes anywhere. Every figure in
`images/` is a hand-authored SVG that matches the visual style of the
proposal figures in `../images/`.
