export const meta = {
  name: 'sjepa-write-proposals',
  description: 'Draft, illustrate (clean SVGs), adversarially review, and revise the 7 Skeleton-JEPA research proposals',
  phases: [
    { title: 'Draft', detail: 'write first-principles README per proposal' },
    { title: 'Illustrate', detail: 'author 2 design-system SVGs per proposal' },
    { title: 'Review', detail: 'adversarial UI/UX + content review' },
    { title: 'Revise', detail: 'fix figures and text to final' },
  ],
}

const ROOT = '/Users/pmui/dev/alexpose/experiments/sjepa/gavd5'
const IDEAS = `${ROOT}/notes/ideas`
const FACTS = `${IDEAS}/_shared_facts.md`
const DESIGN = `${ROOT}/notes/09_diagram_design_system.md`
const SELECTION = `${IDEAS}/_selection.json`

// The 7 proposals with their hardened specs (mirrors _selection.json; inlined so agents need not parse JSON).
const PROPOSALS = args && Array.isArray(args) && args.length ? args : [
  { slug: '01-provenance-pathway-attribution', theme: 'evaluation-validity' },
  { slug: '02-surprise-tomography', theme: 'world-model / predictive' },
  { slug: '03-inference-time-motion-energy', theme: 'world-model / predictive (zero-retrain)' },
  { slug: '04-resize-timing-tax', theme: 'evaluation-validity / preprocessing' },
  { slug: '05-signed-laterality-decodability', theme: 'mechanism' },
  { slug: '06-mask-geometry-as-object', theme: 'mechanism / design' },
  { slug: '07-group-loss-supervision-isolation', theme: 'supervision isolation (highest novelty)' },
]

const COMMON = `You are an expert JEPA / world-model research scientist and a clear science writer. You are writing ONE proposal in a 7-proposal portfolio at ${IDEAS}/.

MANDATORY READING before you write (use the Read tool):
1. ${FACTS} - the single source of truth for every number. Keep all quantitative claims consistent with it.
2. ${SELECTION} - find YOUR proposal object by its "slug" for the exact one_line_question, why_it_wins, incorporated_repairs, and figure_specs. You MUST fold in every repair listed there.
3. ${DESIGN} - the SVG design system (only needed for the illustrate/revise phases).
4. One existing plan README for house style and depth, e.g. ${ROOT}/plan/05-representation-vs-readout-diagnostic/README.md.

HARD WRITING RULES:
- Plain language, first principles. Explain every concept from scratch (what a token is, what "transductive" means, what a source-video-disjoint split is, what an EMA teacher is, etc.) and how concepts interrelate. Assume a smart reader who is new to JEPA.
- NO em-dashes anywhere. Use commas, colons, or short sentences. (This is a hard user requirement.)
- Ground every claim in ${FACTS}. Never invent numbers. If you reference a metric, it must match the facts file.
- The research question must be SMART: specific, measurable, achievable, relevant, time-bound.
- State explicitly how this proposal is DISTINCT from its nearest neighbor in the existing plan/ portfolio (the distinctness note is in ${FACTS}).
- Use only the verified citations in ${FACTS}. Cite inline with author + venue/arXiv id.
- Responsible-use note: folder labels (stroke, parkinsons) are dataset annotations, not diagnoses.
`

// ---- Phase 1: Draft README ----
phase('Draft')

const drafts = await parallel(PROPOSALS.map(p => () =>
  agent(`${COMMON}

YOUR TASK: Write ${IDEAS}/${p.slug}/README.md (theme: ${p.theme}).

Use this section structure (matching the plan/ house style), all in Markdown:
1. Title (H1) and a one-plain-sentence research question in a blockquote.
2. "The question in plain words" - restate for a newcomer, 2-4 short paragraphs, first principles.
3. "Why this matters" - what belief a positive result confirms, what a null result rules out.
4. "Background and related work" - explain the JEPA machinery this uses (encoder, EMA teacher, predictor,
   masking, VICReg as needed) from scratch, with the verified citations.
5. "Method" - concrete steps; reuse existing 528-token tensors / checkpoints / code where possible; name the
   exact frozen artifact fingerprint where relevant.
6. "The decisive experiment" - the exact source-video-disjoint split stated BEFORE any fitting, the primary
   endpoint, the pre-registered margin, and the simple non-neural / nuisance baseline. Include a small table.
7. "Controls and incorporated repairs" - list every repair from incorporated_repairs in ${SELECTION} and how
   you address it. This section is what makes the proposal reviewer-defensible.
8. "How this differs from the existing plan" - one short paragraph naming the nearest plan/ proposal.
9. "Three-week timeline" - Week 1 (16-22 Aug), Week 2 (23-29 Aug), Week 3 (30 Aug-5 Sep) with a Day-5 and a
   Day-14 gate. Convert relative dates to these absolute ones.
10. "Figures" - reference ./images/fig1.svg and ./images/fig2.svg with one-line captions matching the two
    figure_specs for your slug.
11. "Responsible use" note.
12. "References" - the citations you used, from ${FACTS}.

Write the file with the Write tool. Return a 3-sentence summary of what you wrote and the two figure captions.`,
    { label: `draft:${p.slug.slice(0,22)}`, phase: 'Draft', effort: 'high' })
    .then(txt => ({ ...p, draftSummary: txt }))
    .catch(e => ({ ...p, draftSummary: null, error: String(e) }))
))

log(`Drafted ${drafts.filter(d => d.draftSummary).length}/${PROPOSALS.length} READMEs.`)

// ---- Phase 2: Illustrate (author 2 SVGs) ----
phase('Illustrate')

const SVG_RULES = `SVG DESIGN SYSTEM (from ${DESIGN}) - follow EXACTLY:
- viewBox="0 0 1200 720". Background: one flat rect fill #f7f5ef (warm paper). role="img" + <title> + <desc>.
- Nothing closer than 48px to any canvas edge. Minimum 56px gap between adjacent cards.
- Type: title .h 700 32px #0f2440 (one, top-left); card title .t 700 20px #0f2440; body .b 16px #33475e;
  caption .s 14px #5c6f84; on-dark .w 700 16px #fff. Never place two text baselines closer than 22px.
- Palette (few hues): ink/stroke #20364e; rule #c4cdd8; blue #2f6f99; green #5f9e7e (teacher/target);
  orange #e07a4b (student/hot); violet #8a6fb3 (predictor). Pastel card fills: blue #e7f0f8, green #e6f2ea,
  warm #f7e7d8, violet #ece6f6, white #ffffff. Dark banner #20364e with white text.
- Cards: rx=16, stroke #20364e width 1.75, text inset >=22px from every edge (top baseline >=30px below top).
- Arrows: ONE shared <marker id="arrow"> fill #20364e markerWidth 8; connector stroke #20364e width 2.5,
  fill:none, marker-end. Prefer straight horizontal/vertical runs; if a dogleg is needed use a SINGLE
  right-angle bend with a >=20px landing segment; NEVER route an arrow through or under a box; leave >=16px
  between an arrowhead and its target. EMA/secondary path: dashed 7 6, color #5f9e7e, in clear space with a label.
- One idea per card; if a card needs more than 4 body lines, split it or shorten. Footer note: single .s line,
  >=24px above it, never overlapping the last row of cards.
- Valid standalone SVG 1.1, no external fonts, no scripts, no raster images.

UI/UX QUALITY BAR (hard user requirement): each image must be clean, uncluttered, with NO overlapping text,
NO lines crossing through boxes or labels, and arrows that are intuitive and visually appealing. Lay out on a
clear grid, align card tops and text left-edges, and leave generous whitespace. Compute positions so no two
text/box bounding boxes overlap. Prefer fewer, larger, well-spaced elements over many small crowded ones.`

const illustrated = await pipeline(
  drafts,
  (d) => agent(`${COMMON}

${SVG_RULES}

YOUR TASK: Author TWO standalone SVG files for proposal ${d.slug}:
- ${IDEAS}/${d.slug}/images/fig1.svg
- ${IDEAS}/${d.slug}/images/fig2.svg
Their content must match the two figure_specs for slug "${d.slug}" in ${SELECTION} (read it). Also read your
own README at ${IDEAS}/${d.slug}/README.md so the figures match the text. Fig1 is usually the concept/method
flow; fig2 is usually the results-shape / experiment-design schematic described in the spec. Since results do
not exist yet, draw results-shape figures as clearly-labeled ILLUSTRATIVE schematics (axes, expected ordering,
baseline/null reference lines, per-source dots as a scatter shape) and mark them "illustrative expected shape".

Write both files with the Write tool. Keep them clean and uncluttered per the UI/UX bar. Return the two file
paths and a one-line description of each.`,
    { label: `svg:${d.slug.slice(0,24)}`, phase: 'Illustrate', effort: 'high' })
    .then(txt => ({ ...d, svgSummary: txt }))
    .catch(e => ({ ...d, svgSummary: null, error: String(e) }))
)

log(`Illustrated ${illustrated.filter(d => d.svgSummary).length}/${PROPOSALS.length} proposals (2 SVGs each).`)

// ---- Phase 3: Adversarial UI/UX + content review ----
phase('Review')

const REVIEW_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['slug', 'readme_issues', 'fig1_issues', 'fig2_issues', 'em_dash_found', 'number_inconsistencies', 'overall_verdict'],
  properties: {
    slug: { type: 'string' },
    readme_issues: { type: 'array', items: { type: 'string' }, description: 'Clarity, first-principles gaps, missing repairs, non-distinctness, missing SMART elements.' },
    fig1_issues: { type: 'array', items: { type: 'string' }, description: 'Concrete UI/UX defects: overlapping text/boxes (give coordinates), arrows through boxes, off-palette colors, clutter, edge violations, unreadable labels.' },
    fig2_issues: { type: 'array', items: { type: 'string' } },
    em_dash_found: { type: 'boolean' },
    number_inconsistencies: { type: 'array', items: { type: 'string' }, description: 'Any number in README/figures that contradicts _shared_facts.md.' },
    overall_verdict: { type: 'string', enum: ['CLEAN', 'MINOR_FIXES', 'MAJOR_FIXES'] },
  },
}

const reviewed = await pipeline(
  illustrated,
  (d) => agent(`You are a HARSH combined reviewer: (a) a Staff UI/UX designer auditing vector graphics for clutter and overlaps, and (b) an ICLR/ICML/NeurIPS reviewer auditing the proposal text.

Read ALL of these with the Read tool:
- ${IDEAS}/${d.slug}/README.md
- ${IDEAS}/${d.slug}/images/fig1.svg
- ${IDEAS}/${d.slug}/images/fig2.svg
- ${FACTS} (to catch any number that contradicts the source of truth)
- the "${d.slug}" entry in ${SELECTION} (to confirm every incorporated_repair is actually addressed)

For the SVGs, PARSE the coordinates and detect: any two text or box bounding boxes that overlap; any arrow/line
that passes through or under a box or over text; any element within 48px of the canvas edge; adjacent cards
closer than 56px; text baselines closer than 22px; any color outside the sanctioned palette; any label that is
too long for its card. Report each defect concretely with the element and its coordinates so it can be fixed.
For the README, flag: em-dashes (hard fail), numbers inconsistent with the facts file, missing SMART elements,
any incorporated_repair not addressed, weak first-principles explanation, and missing distinctness-from-plan.

Be specific and actionable. A figure with ANY overlap or arrow-through-box is at best MINOR_FIXES.`,
    { label: `review:${d.slug.slice(0,20)}`, phase: 'Review', schema: REVIEW_SCHEMA, effort: 'high' })
    .then(v => ({ ...d, review: v }))
    .catch(e => ({ ...d, review: null, error: String(e) })),
  // ---- Phase 4: Revise based on review ----
  (d) => {
    if (!d.review) return d
    const r = d.review
    const clean = r.overall_verdict === 'CLEAN' &&
      !r.em_dash_found &&
      (r.number_inconsistencies || []).length === 0 &&
      (r.readme_issues || []).length === 0 &&
      (r.fig1_issues || []).length === 0 &&
      (r.fig2_issues || []).length === 0
    if (clean) { log(`${d.slug}: clean, no revision needed.`); return { ...d, revised: 'CLEAN' } }
    return agent(`${COMMON}

${SVG_RULES}

YOUR TASK: Revise proposal ${d.slug} to fix EVERY issue below. Edit the files in place (Read then Edit/Write):
- ${IDEAS}/${d.slug}/README.md
- ${IDEAS}/${d.slug}/images/fig1.svg
- ${IDEAS}/${d.slug}/images/fig2.svg

REVIEW FINDINGS TO FIX:
README issues: ${JSON.stringify(r.readme_issues)}
Em-dash present (must be removed): ${r.em_dash_found}
Number inconsistencies vs facts: ${JSON.stringify(r.number_inconsistencies)}
FIG1 issues: ${JSON.stringify(r.fig1_issues)}
FIG2 issues: ${JSON.stringify(r.fig2_issues)}

For figure fixes, RECOMPUTE element positions so nothing overlaps: move boxes apart (>=56px gap), reroute
arrows as straight runs or single right-angle bends that never cross a box, keep everything >=48px from edges,
and ensure every label fits inside its card with >=22px inset. When in doubt, simplify: fewer, larger, better
spaced elements. After editing, re-verify mentally that no two bounding boxes overlap and no arrow crosses a box.

Return a short list of what you changed.`,
      { label: `revise:${d.slug.slice(0,20)}`, phase: 'Revise', effort: 'high' })
      .then(txt => ({ ...d, revised: txt }))
      .catch(e => ({ ...d, revised: null, error: String(e) }))
  }
)

const summary = reviewed.filter(Boolean).map(d => ({
  slug: d.slug,
  verdict: d.review ? d.review.overall_verdict : 'NO_REVIEW',
  em_dash: d.review ? d.review.em_dash_found : null,
  num_issues: d.review ? ((d.review.readme_issues||[]).length + (d.review.fig1_issues||[]).length + (d.review.fig2_issues||[]).length) : null,
  revised: d.revised === 'CLEAN' ? 'clean' : (d.revised ? 'revised' : 'FAILED'),
}))

return { summary }
