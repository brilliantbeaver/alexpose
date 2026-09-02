export const meta = {
  name: 'sjepa-portfolio-figures',
  description: 'Draw fig1/fig2 for items 08-12, fix their README Figures sections, harsh UI/UX QA+repair over all 24 SVGs, refresh the 12-item portfolio hub',
  phases: [
    { title: 'Draft', detail: 'draw fig1+fig2 for 08-12, patch README figures, delete stray placeholders' },
    { title: 'QA', detail: 'harsh UI/UX review of every SVG, then repair flagged files in place' },
    { title: 'Hub', detail: 'refresh 00_portfolio_hub.svg to 12 items / 4 themes' },
  ],
}

const DIR = '/Users/pmui/dev/alexpose/experiments/sjepa/gavd5-draft/notes/ideas-claude'
const DESIGN = '/Users/pmui/dev/alexpose/experiments/sjepa/gavd5-draft/notes/09_diagram_design_system.md'

// ---- The SVG design contract, inlined so every agent obeys the SAME rules ----
const SVG_RULES = `
You are drawing clean, uncluttered, publication-quality vector figures for an ICLR/ICML-style research
portfolio. You MUST read ${DESIGN} in full and obey it EXACTLY. The single reference implementation you
must match for structure, spacing, and voice is:
  ${DIR}/05-signed-laterality-decodability/images/fig1.svg   (READ IT FIRST; copy its skeleton)

NON-NEGOTIABLE DESIGN TOKENS (a reviewer rejects on any violation):
- Standalone SVG 1.1. First line: <?xml version="1.0" encoding="UTF-8"?>. Root:
  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 720" role="img" aria-labelledby="tX dX"> with a
  <title id="tX"> and <desc id="dX"> that describe THIS figure. No scripts, no external fonts, no <image>,
  no raster, no gradients.
- Background: one flat <rect x="0" y="0" width="1200" height="720" fill="#f7f5ef"/>.
- Outer safe margin: NOTHING closer than 48px to any canvas edge (so x in [48,1152], y in [48,672]).
- Type ramp (Arial/Helvetica): .h 700/32px #0f2440 (ONE title, top-left at x=48,y=60); .t 700/20px #0f2440;
  .b 16px #33475e; .s 14px #5c6f84; .w 700/16px #fff; .ws 14px #cdd8e4 (on dark). Define these in a <style>.
- NEVER place two text baselines closer than 22px; prefer >=26px between stacked lines.
- Palette ONLY: ink/stroke #20364e; rule #c4cdd8; blue #2f6f99; green #5f9e7e (target/ceiling); warm #e07a4b
  (student/primary); violet #8a6fb3 (predictor); pastel card fills blue #e7f0f8, green #e6f2ea, warm #f7e7d8,
  violet #ece6f6, white #ffffff; dark banner #20364e. Darker accent strokes allowed (e.g. #a44c26 on warm).
  Introduce NO new hues.
- Cards: rx="16", stroke #20364e width 1.75. Text inset >=22px from every card edge; the top text baseline
  sits >=30px below the card's top edge. Minimum gap between adjacent cards: 56px.
- Arrows: exactly ONE shared <marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4"
  orient="auto" markerUnits="userSpaceOnUse"><path d="M0,0 L8,4 L0,8 z" fill="#20364e"/></marker> in <defs>.
  Arrow lines: stroke #20364e width 2.5, fill:none, marker-end="url(#arrow)". Prefer straight H/V runs; at
  most one right-angle bend with a >=20px landing segment; NEVER route an arrow through or under a box; leave
  >=16px clear space between an arrowhead and its target box. Do NOT put marker-end on axis lines or legend
  swatches (those are not arrows).
- EMA/secondary/identity lines: dashed stroke-dasharray "7 6", green #5f9e7e, labeled in clear space.

CLARITY DISCIPLINE (this is the user's explicit ask):
- One idea per card; <=4 body lines per card or split it.
- Every plotted point, line, and label must sit in clear space. NO overlapping text. NO line crossing through
  text or through another figure element unless it is a legitimate data point on a plotted trend. NO duplicate
  or malformed arrowheads. Axis ticks and their number labels must not collide.
- These are ILLUSTRATIVE EXPECTED-SHAPE figures, not measured results. Say so in the .s subtitle and footer.
- No em-dashes anywhere (use commas, periods, or parentheses). Every number consistent with
  ${DIR}/_shared_facts.md. Folder labels are dataset annotations, not diagnoses; all current results transductive.
`

// ---- Per-item figure jobs (08-12): specs pulled verbatim from _selection.json figure_specs ----
const ITEMS = [
  {
    slug: '08-concept-bottleneck-disentangled',
    title: 'Concept-bottleneck disentangled S-JEPA',
    fig1: `STEERABILITY MATRIX. A 3x3 heat-style matrix as the hero: rows = the subspace you INTERVENE on
(z_asym, z_rhythm, z_posture); columns = the CHANGE observed in each biomarker (symmetry ratio, stride-time CV,
anterior pelvic tilt). The diagonal cells are strong (large filled marker / dark fill) and annotated "above raw-
coordinate ceiling"; the off-diagonal cells are faint (small/pale) and annotated "leak below pre-registered
bound". Add a compact legend card explaining diagonal=steer, off-diagonal=leak, plus the pass rule
(steerability ratio >= 3, leak <= 0.2). This is the causal-intervention hero figure.`,
    fig2: `PER-SUBSPACE BIOMARKER-RECOVERY scatter, in the SAME style as 05/fig1: x = held-out source (or ground-
truth biomarker), y = recovered R-squared, one dot per held-out source, three colored series (one per subspace).
Overlay the raw-coordinate probe CEILING (green dashed) and the untrained-encoder FLOOR (grey dashed). Flag
z_rhythm as the highest-risk subspace with a small note (stride-time CV is not linearly decodable from ~2s
windows). Right side: a compact lane legend + pre-registered pass card (dark banner).`,
  },
  {
    slug: '09-reflection-equivariant-symmetry-axis',
    title: 'Reflection-equivariant symmetry axis',
    fig1: `PAIRED-DOTS comparison, hero. x = held-out source, y = signed-decodability R-squared on item 05's
instrument; two paired dots per source connected by a thin connector: the by-construction reflection-equivariant
encoder (warm #e07a4b) vs the standard d0acc262 encoder (blue #2f6f99). Draw the pre-registered +0.05 R-squared
advantage as a shaded band / annotation. Right: lane legend + pass card (dark banner) stating "by-construction
must beat d0acc262 by >= 0.05 held-out-source R-squared".`,
    fig2: `MIRROR-SLOPE panel, in the 05/fig1 scatter style: x = readout on ORIGINAL input, y = readout on the
anatomically MIRRORED input, with the y = -x reflection line (green dashed). The by-construction encoder lands
EXACTLY on y = -x (guaranteed negation; warm dots tight on the line, label "guaranteed"); the learned d0acc262
encoder scatters near but off the line (blue dots, a fitted slope annotated "measured, approximate"). Footer: a
mirror flips the sign of the lateralized biomarker (Patterson 2010).`,
  },
  {
    slug: '10-prediction-error-severity',
    title: 'Prediction-error-as-severity',
    fig1: `INJECTION DIRECTION TEST, hero. A line/scatter plot: x = injection magnitude; two response curves,
asymmetry-channel error (warm) and posture-channel error (blue). Show TWO injection conditions with clear
separation: under one-sided knee-flexion injection the ASYMMETRY channel rises steeply while posture stays flat;
under symmetric proximal-deficit injection the POSTURE channel rises while asymmetry stays flat (use two small
stacked sub-panels or two clearly-labeled curve pairs). Draw the random-encoder CONTROL lane (grey dashed, flat).
Right: pass card (dark banner) with the pre-registered margin over the random-encoder control.`,
    fig2: `MECHANISM-GROUPED ERROR DECOMPOSITION. Left card: a simple lower-body stick skeleton (desaturated grey
bones #8a9bab, joint nodes #d8e0e7 stroke #20364e) with per-token relative error shown as warm halos, grouped
by three labeled channels: asymmetry (left-vs-right lower limb), rhythm (temporal), posture (pelvis/hip). Right
card: a 3-bar chart of per-channel error on a held-out abnormal source, with a near-tautology CONTROL bar
("error beyond the injected coordinate") to show error is not merely echoing the perturbed coordinate.`,
  },
  {
    slug: '11-target-isolation-substrate',
    title: 'Target-isolation substrate',
    fig1: `GROUPED BARS, hero. Three column-groups (one per mechanism: symmetry ratio, stride-time CV, anterior
pelvic tilt); within each group, four bars for the four target families T1 raw coordinates, T2 one-frame motion,
T3 centered-sharpened latent, T4 normalized latent regression (four distinct palette hues). Draw the raw-input
CEILING as a green dashed horizontal line and the untrained-encoder FLOOR as a lower grey dashed line PER group;
shade the pre-registered +0.05 margin band above the ceiling. Legend maps T1..T4 to colors. Shows at a glance
which family, if any, clears the ceiling on each mechanism.`,
    fig2: `MATCHED-SUBSTRATE AUDIT panel. A clean 4-row table/card grid, one row per target arm (T1..T4), columns:
Encoder, Compute budget, Updates, Mask, showing IDENTICAL values across all four rows (e.g. same embed 64 /
depth 2, same 11,400 updates, same 12-joint mask) with a checkmark strip confirming "only the target differs".
This is the internal-validity control figure: everything fixed except the prediction target.`,
  },
  {
    slug: '12-cross-view-invariance',
    title: 'Cross-view gait invariance',
    fig1: `LEAVE-ONE-VIEW-OUT DRIFT, hero. x = held-out camera view (across public multi-view cohorts CASIA-B,
OU-MVLP-Pose, GREW, Gait3D); y = feature drift (lower is better). Three series: Lane A view-as-action + no-flip
(warm, lowest drift), Lane B flip-augmented baseline (blue, higher), Lane C Procrustes raw-coordinate baseline
(grey). Mark the pre-registered >=10 percent relative drift-reduction margin between A and B. Right: lane legend
+ pass card (dark banner). Footer: cohorts are non-clinical multi-view pose; clinical claim is reach-tier.`,
    fig2: `MIRROR PROBE, in the 05/fig1 scatter style: x = signed lateralized biomarker on ORIGINAL input, y =
same biomarker on LEFT-RIGHT-FLIPPED input, y = -x reflection line (green dashed). Flip-augmented baseline
(blue) collapses the sign toward y = +x (a mirror does NOT invert it, so asymmetry is destroyed); the no-flip
view-conditioned predictor (warm) preserves the sign on y = -x (a mirror correctly inverts it). Label the two
behaviors clearly. This separates the invariance benefit from the asymmetry-protection benefit.`,
  },
]

const DRAW_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['slug', 'fig1_written', 'fig2_written', 'readme_patched', 'placeholder_removed'],
  properties: {
    slug: { type: 'string' },
    fig1_written: { type: 'boolean' },
    fig2_written: { type: 'boolean' },
    readme_patched: { type: 'boolean' },
    placeholder_removed: { type: 'boolean' },
    notes: { type: 'string' },
  },
}

const REVIEW_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['path', 'verdict', 'defects'],
  properties: {
    path: { type: 'string' },
    verdict: { type: 'string', enum: ['CLEAN', 'REPAIR'] },
    defects: {
      type: 'array',
      items: {
        type: 'object', additionalProperties: false,
        required: ['kind', 'detail'],
        properties: {
          kind: {
            type: 'string',
            enum: [
              'malformed-or-duplicate-arrowhead', 'arrow-through-or-under-box', 'missing-arrowhead-clearance',
              'overlapping-text', 'text-touching-or-outside-card', 'line-crosses-text-or-figure',
              'element-past-48px-margin', 'baselines-too-close', 'card-gap-too-small', 'off-palette-color',
              'em-dash', 'number-inconsistent-with-facts', 'not-well-formed', 'other',
            ],
          },
          detail: { type: 'string' },
        },
      },
    },
  },
}

const REPAIR_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['path', 'repaired', 'summary'],
  properties: {
    path: { type: 'string' }, repaired: { type: 'boolean' }, summary: { type: 'string' },
  },
}

// ============ Phase Draft: one agent per item, draws BOTH figs + patches README ============
phase('Draft')
const drawn = await parallel(ITEMS.map((it) => () => agent(
  `${SVG_RULES}

TASK for portfolio item ${it.slug} (${it.title}). Do ALL of the following:

1. Read ${DIR}/${it.slug}/README.md in full so the figures match its actual claim, its blockquote question,
   its pre-registered margins, and its lane structure. Read ${DIR}/05-signed-laterality-decodability/images/fig1.svg
   as the structural template. Read ${DESIGN} in full.

2. WRITE ${DIR}/${it.slug}/images/fig1.svg to this spec:
${it.fig1}

3. WRITE ${DIR}/${it.slug}/images/fig2.svg to this spec:
${it.fig2}

   Both files must be standalone valid SVG 1.1 obeying every design token above. Lay out coordinates carefully:
   compute card rectangles first, then place text baselines >=30px below each card top and >=22px inside each
   edge, then draw plotted points/lines inside the plot frame, then legends in clear space. Double-check that no
   two text elements overlap, no arrow passes through a box, arrowheads have >=16px clearance, and nothing sits
   within 48px of the canvas edge.

4. PATCH ${DIR}/${it.slug}/README.md "## Figures" section IN PLACE: REMOVE every "DEFERRED"/"(deferred)"/
   "No SVGs are drawn"/"see ./images/README.md" phrase. Keep the two ![...](./images/fig1.svg) and
   ![...](./images/fig2.svg) image embeds (fix the alt text if needed). Rewrite the two captions ("Fig 1:" and
   "Fig 2:") as real present-tense captions of what the drawn figure shows (drop the "(DEFERRED)"/"(deferred)"
   parenthetical). Do not touch any other section. No em-dashes.

5. DELETE the stray placeholder file ${DIR}/${it.slug}/images/README.md (use Bash: rm -f). The images folder
   should contain only fig1.svg and fig2.svg afterward.

Return the structured status. Set placeholder_removed true only after the rm succeeded.`,
  { label: `draw:${it.slug.slice(0, 2)}`, phase: 'Draft', schema: DRAW_SCHEMA, effort: 'high' }
)))
const okDraw = drawn.filter(Boolean)
log(`Draft complete: ${okDraw.length}/5 items drew both figures`)

// Build the full list of 24 SVGs to QA (14 existing + 10 new). Portfolio hub handled in Phase Hub.
const EXISTING = [
  '01-provenance-pathway-attribution', '02-surprise-tomography', '03-inference-time-motion-energy',
  '04-resize-timing-tax', '05-signed-laterality-decodability', '06-mask-geometry-as-object',
  '07-group-loss-supervision-isolation',
]
const NEW = ITEMS.map((it) => it.slug)
const ALL_SLUGS = [...EXISTING, ...NEW]
const ALL_SVGS = []
for (const s of ALL_SLUGS) { ALL_SVGS.push(`${DIR}/${s}/images/fig1.svg`); ALL_SVGS.push(`${DIR}/${s}/images/fig2.svg`) }

// ============ Phase QA: harsh review -> conditional repair, pipelined per SVG ============
phase('QA')
const qaResults = await pipeline(
  ALL_SVGS,
  // Stage 1: harsh UI/UX + correctness review of ONE svg
  (path) => agent(
    `${SVG_RULES}

You are a HARSH UI/UX design reviewer plus a correctness checker. Review EXACTLY ONE file: ${path}.
Read the file in full. Also read ${DESIGN} and ${DIR}/_shared_facts.md.

Inspect at the COORDINATE level and report every real defect (empty array if genuinely clean):
- malformed-or-duplicate-arrowhead: a marker-end on a non-arrow (axis/legend), two arrowheads on one line, a
  marker defined but arrowhead visually detached from the line end, or an arrow with no visible shaft.
- arrow-through-or-under-box: an arrow line segment whose path crosses the rectangle of any card/box, or ends
  with <16px clearance from its target box.
- missing-arrowhead-clearance: arrowhead within 16px of the box it points to.
- overlapping-text: two <text> elements whose bounding boxes overlap, or a text baseline within 22px of another.
- text-touching-or-outside-card: text within 22px of its card edge, or outside the card, or top baseline <30px
  below the card top.
- line-crosses-text-or-figure: any line/plotted trend passing through a text bounding box or through an
  unrelated figure element (a legitimate data point ON its own trend line is fine).
- element-past-48px-margin: any drawn element with x<48, x>1152, y<48, or y>672.
- baselines-too-close / card-gap-too-small (<56px between adjacent cards).
- off-palette-color: any fill/stroke hue not in the sanctioned palette (report the offending hex).
- em-dash anywhere; number-inconsistent-with-facts vs _shared_facts.md; not-well-formed SVG.

To estimate text bounding boxes, use ~0.55*fontsize per character for width and the font-size for height above
the baseline. Be precise about coordinates, be strict, but do NOT invent defects that are not in the file.
verdict = CLEAN only if defects is empty; otherwise REPAIR. Return the review for THIS path.`,
    { label: `qa:${path.split('/').slice(-2).join('/')}`, phase: 'QA', schema: REVIEW_SCHEMA, effort: 'high' }
  ),
  // Stage 2: repair IN PLACE only if the review said REPAIR (else pass through)
  (review, path) => {
    if (!review || review.verdict === 'CLEAN' || !review.defects || review.defects.length === 0) {
      return { path, repaired: false, summary: 'CLEAN, no repair needed' }
    }
    const defectList = JSON.stringify(review.defects, null, 2)
    return agent(
      `${SVG_RULES}

Repair EXACTLY ONE file IN PLACE: ${path}. Read it in full first. A harsh reviewer found these defects; fix
EVERY one while preserving the figure's semantic content, its <title>/<desc> meaning, its filename, and its
conceptual message (this is a visual repair, not a content rewrite):

${defectList}

Move/resize elements as needed so that: no text overlaps, no arrow passes through or under a box, every
arrowhead has >=16px clearance, nothing sits within 48px of the canvas edge, adjacent cards are >=56px apart,
text baselines are >=22px apart and >=30px below their card top and >=22px inside every card edge, every color
is on-palette, there are no em-dashes, and every number matches _shared_facts.md. Keep it valid standalone
SVG 1.1 (xmllint must parse it). Save the file. Return the repair summary.`,
      { label: `fix:${path.split('/').slice(-2).join('/')}`, phase: 'QA', schema: REPAIR_SCHEMA, effort: 'high' }
    )
  }
)
const repaired = qaResults.filter((r) => r && r.repaired)
log(`QA complete: ${qaResults.filter(Boolean).length}/${ALL_SVGS.length} reviewed, ${repaired.length} repaired`)

// ============ Phase Hub: refresh the portfolio hub to 12 items / 4 themes ============
phase('Hub')
const hubResult = await agent(
  `${SVG_RULES}

TASK: Refresh ${DIR}/images/00_portfolio_hub.svg so it presents ALL TWELVE proposals grouped by FOUR themes,
replacing the current seven-item / three-theme version. Read the current 00_portfolio_hub.svg first (keep its
title voice and overall composition), read ${DIR}/README.md for the authoritative 12-row grouping, the four
themes, and the bands, and read ${DESIGN}.

The four themes and their items (group them into four labeled panels/columns):
- Evaluation validity: 01 provenance-pathway-attribution, 04 resize-timing-tax.
- World-model / predictive: 02 surprise-tomography, 03 inference-time-motion-energy.
- Mechanism / design: 05 signed-laterality-decodability, 06 mask-geometry-as-object,
  07 group-loss-supervision-isolation.
- Neuro-grounded world model: 08 concept-bottleneck-disentangled, 09 reflection-equivariant-symmetry-axis,
  10 prediction-error-severity, 11 target-isolation-substrate, 12 cross-view-invariance.
Note in a small caption that 05 and 09 also carry a mechanism reading. Each item shows its number, a very short
label, and its ICLR/ICML band as a tiny pill (Strong main-track candidate vs Competitive with revision). Add a
single top caption line stating the unifying symmetry-axis thesis (lateralized vs rhythm/variability vs
symmetric-proximal). Keep it clean and uncluttered: twelve compact cards in four themed columns, generous gaps,
no overlaps, no arrows through boxes, everything inside the 48px margin. No em-dashes.

Save the file. Return a one-line summary.`,
  {
    label: 'hub:portfolio', phase: 'Hub', effort: 'high',
    schema: {
      type: 'object', additionalProperties: false,
      required: ['written', 'summary'],
      properties: { written: { type: 'boolean' }, summary: { type: 'string' } },
    },
  }
).catch(() => null)
log('Hub complete: portfolio hub refreshed to 12 items / 4 themes')

return {
  drawn: okDraw,
  qa_reviewed: qaResults.filter(Boolean).length,
  qa_repaired: repaired.length,
  hub: hubResult && hubResult.written ? 'refreshed' : 'FAILED',
}
