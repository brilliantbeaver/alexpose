# Independent review of the interactive proposal

21 September 2026. The reviewer implemented the separate new SVG figures but did not author the interactive paper, its JavaScript, stylesheet or HTML builder. This review covers those root-authored components: [paper.js](../scripts/paper.js), [paper.css](../scripts/paper.css), [build_paper.py](../scripts/build_paper.py) and the generated [proposal](../proposal.html).

## Scope and verdict

The arithmetic and scientific interpretation pass the source review. The explorer clearly identifies its numbers as illustrative, keeps the camera fixed, and separates a change in the movement measurement from an observation-induced change in its error. It also explains that correct changes can coexist with biased individual measurements.

No browser session was available. The checks below use source inspection, Node execution and a minimal DOM simulation; they do **not** establish rendered layout, browser keyboard behavior or assistive-technology compatibility. The separate PNG review establishes diagram legibility, not the enclosing web page's layout.

## Arithmetic and document checks

The cell order is `[motion A / clean, motion A / degraded, motion B / clean, motion B / degraded]`, matching the visible table. With `e = estimate − reference`, the implementation uses:

```text
R(clean)    = e10 − e00
R(degraded) = e11 − e01
N(A)        = e01 − e00
N(B)        = e11 − e10
Interaction = R(degraded) − R(clean) = N(B) − N(A)
```

An independent Node sweep checked **158,565 parameter combinations**: true change from −8° through +8° in 0.5° steps, both retained-change multipliers from 0 through 1.5 in 0.05 steps, and offsets of −3°, −0.25°, 0°, +0.25° and +3°. Every contrast agreed with its independently derived expression to `1e-10`; all outputs were finite. Negative changes retain the appropriate sign, zero true change gives zero response errors and interaction, and a common observation offset cancels from the response contrast while remaining visible in the observation contrasts. Amplification above 100% is represented correctly.

A minimal DOM simulation checked the initial outputs, all three preset handlers and negative/zero slider updates. The generated HTML contained no duplicate IDs or missing same-document fragment targets. Its static default table and metric outputs agree with the initial sliders. The figure links remain ordinary SVG links when modal support or JavaScript is unavailable; native buttons, ranges, details elements and a visible-focus rule provide a reasonable keyboard-oriented foundation.

## Findings sent to the author

1. **Mobile navigation needs a source-level correction.** Selecting a section does not close the sticky contents panel, and the 24px scroll padding is smaller than its collapsed height. An expanded panel can obscure the target heading. Close it after selection and reserve adequate mobile anchor space. Actual viewport behavior still needs browser verification.
2. **The no-JavaScript fallback should disable or hide interactive controls.** Its explanatory notice is accurate, but active-looking sliders can move while all calculations remain at the default values. Default-disabled controls enabled after successful initialization would make this boundary clear.
3. **Small clarity/accessibility improvements:** disclose that values are rounded, since the 0.25° offset step is displayed to one decimal; give each figure link a descriptive accessible name rather than the repeated “Enlarge diagram”; and bound the sticky desktop contents panel's height with scrolling so its resource links remain reachable on short screens.

These findings concern presentation and navigation rather than the proposed scientific conclusions. Any subsequent corrections require a source recheck; browser interaction and layout remain unverified until a browser is available.

## Final source recheck

The author addressed all findings above, and the revised source and generated page pass the independent recheck. All seven explorer controls are disabled in the static HTML and become enabled only after the initial calculation succeeds. Every mobile contents link closes the panel; the mobile stylesheet now reserves 80px above fragment targets. The desktop sidebar has a viewport-bounded height and vertical scrolling. All six figure links have distinct descriptive accessible names. Quarter-degree offsets are displayed exactly to two decimal places.

The full 158,565-case arithmetic sweep passed again. A fresh minimal DOM simulation passed the default outputs, all presets, negative and zero changes, the +0.25° offset, enabling all seven controls, and closing the panel through all ten mobile links. The generated page embeds the reviewed JavaScript and CSS, has no duplicate IDs, and has no missing same-document fragment targets.

**Final disposition:** the identified source-level defects are resolved. Browser rendering, actual viewport scrolling, native dialog focus behavior and assistive-technology interaction remain unverified; this review makes no claim that they were exercised.

SHA-256 values identify the exact files inspected during this recheck:

```text
21802a39de804c224613b0c12a806cec87bfd825582e10d69c87c7bd749a257a  scripts/paper.js
28c057ccd92f2c5f8aa2c6fcec25230eaf8c003eb74d469161f43231a65854e0  scripts/paper.css
8981ea9c91a605600a49683f6b541e094b805e94045809e3cca5d35f297c5f04  scripts/build_paper.py
87a023d0f2ba62556e148390c3b76093b09cdec097be15b21909638fb42135a7  proposal.html
```
