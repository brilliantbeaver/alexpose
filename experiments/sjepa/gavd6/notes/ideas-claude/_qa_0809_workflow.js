export const meta = {
  name: 'qa-0809-figures',
  description: 'Harsh UI/UX QA + in-place repair of the four SVGs (items 08 and 09) that missed the main QA pass',
  phases: [{ title: 'QA', detail: 'review then repair each of the four 08/09 SVGs' }],
}

const DIR = '/Users/pmui/dev/alexpose/experiments/sjepa/gavd5/notes/ideas-claude'
const DESIGN = '/Users/pmui/dev/alexpose/experiments/sjepa/gavd5/notes/09_diagram_design_system.md'

const SVG_RULES = `
Clean, uncluttered, publication-quality vector figures for an ICLR/ICML research portfolio. Obey ${DESIGN}
EXACTLY. Reference implementation: ${DIR}/05-signed-laterality-decodability/images/fig1.svg.
Tokens: standalone SVG 1.1; viewBox 0 0 1200 720; role="img" + <title>/<desc>; flat #f7f5ef bg; nothing within
48px of any edge; type ramp .h 700/32px #0f2440 (one title x=48 y=60), .t 700/20px, .b 16px #33475e, .s 14px
#5c6f84, .w 700/16px #fff, .ws 14px #cdd8e4; baselines >=22px apart (prefer >=26); palette ONLY ink #20364e,
rule #c4cdd8, blue #2f6f99, green #5f9e7e, warm #e07a4b, violet #8a6fb3, pastel fills #e7f0f8/#e6f2ea/#f7e7d8/
#ece6f6/#fff, dark banner #20364e, darker accent strokes #a44c26 allowed; cards rx=16 stroke #20364e 1.75, text
inset >=22px, top baseline >=30px below card top, adjacent cards >=56px apart; ONE shared <marker id="arrow"
markerWidth=8 markerHeight=8 refX=7 refY=4 orient=auto markerUnits=userSpaceOnUse><path d="M0,0 L8,4 L0,8 z"
fill="#20364e"/></marker>; arrow lines stroke #20364e 2.5 fill:none marker-end only on real arrows (never axes/
legend swatches), prefer straight H/V, at most one right-angle bend, never through/under a box, >=16px arrowhead
clearance; dashed identity/reflection lines "7 6" green #5f9e7e. No overlapping text, no line through text/figure,
no duplicate/malformed arrowheads, no off-palette hue, no em-dash, numbers consistent with ${DIR}/_shared_facts.md,
illustrative expected-shape (say so in subtitle+footer), transductive, labels are dataset annotations not diagnoses.
`

const FILES = [
  `${DIR}/08-concept-bottleneck-disentangled/images/fig1.svg`,
  `${DIR}/08-concept-bottleneck-disentangled/images/fig2.svg`,
  `${DIR}/09-reflection-equivariant-symmetry-axis/images/fig1.svg`,
  `${DIR}/09-reflection-equivariant-symmetry-axis/images/fig2.svg`,
]

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
  properties: { path: { type: 'string' }, repaired: { type: 'boolean' }, summary: { type: 'string' } },
}

phase('QA')
const results = await pipeline(
  FILES,
  (path) => agent(
    `${SVG_RULES}

You are a HARSH UI/UX design reviewer plus correctness checker. Review EXACTLY ONE file: ${path}. Read it in
full, plus ${DESIGN} and ${DIR}/_shared_facts.md. Inspect at the COORDINATE level: estimate text bounding boxes
as ~0.55*fontsize per character wide and font-size tall above the baseline. Report every real defect (empty if
genuinely clean) using the enum kinds: malformed/duplicate arrowheads, arrow through/under a box, arrowhead
clearance < 16px, overlapping text or baselines < 22px, text within 22px of a card edge or top baseline < 30px
below card top, a line crossing text/an unrelated figure element, any element with x<48 / x>1152 / y<48 / y>672,
adjacent cards < 56px apart, off-palette hex, em-dash, number inconsistent with _shared_facts.md, not-well-formed.
Be strict but do NOT invent defects absent from the file. verdict CLEAN only if defects is empty.`,
    { label: `qa:${path.split('/').slice(-2).join('-')}`, phase: 'QA', schema: REVIEW_SCHEMA, effort: 'high' }
  ),
  (review, path) => {
    if (!review || review.verdict === 'CLEAN' || !review.defects || review.defects.length === 0) {
      return { path, repaired: false, summary: 'CLEAN, no repair needed' }
    }
    return agent(
      `${SVG_RULES}

Repair EXACTLY ONE file IN PLACE: ${path}. Read it in full first. Fix EVERY defect below while preserving the
figure's semantic content, its <title>/<desc> meaning, filename, and conceptual message (visual repair, not a
content rewrite). Ensure no text overlaps, no arrow through/under a box, >=16px arrowhead clearance, nothing
within 48px of the canvas edge, adjacent cards >=56px apart, baselines >=22px apart and >=30px below card top and
>=22px inside every edge, on-palette colors only, no em-dashes, numbers matching _shared_facts.md. Keep it valid
standalone SVG 1.1. Save the file.

DEFECTS:
${JSON.stringify(review.defects, null, 2)}`,
      { label: `fix:${path.split('/').slice(-2).join('-')}`, phase: 'QA', schema: REPAIR_SCHEMA, effort: 'high' }
    )
  }
)
const repaired = results.filter((r) => r && r.repaired)
log(`QA complete: ${results.filter(Boolean).length}/4 reviewed, ${repaired.length} repaired`)
return { reviewed: results.filter(Boolean).length, repaired: repaired.length, detail: results }
