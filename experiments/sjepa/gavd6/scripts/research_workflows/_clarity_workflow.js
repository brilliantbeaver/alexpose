export const meta = {
  name: 'sjepa-clarity-pass',
  description: 'Rewrite the 7 Skeleton-JEPA proposals to high-school reading level, fully annotate every equation, and add worked numeric examples plus short illustrative code',
  phases: [
    { title: 'Simplify', detail: 'rewrite each README simpler + add math annotations, worked example, code' },
    { title: 'Critique', detail: 'adversarial clarity + fidelity review' },
    { title: 'Polish', detail: 'fix flagged issues in place' },
  ],
}

const ROOT = new URL('../..', import.meta.url).pathname.replace(/\/$/, '')
const IDEAS = `${ROOT}/notes/archive/portfolio-ideas/ideas`
const FACTS = `${ROOT}/notes/research/_shared_facts.md`

const SLUGS = args && Array.isArray(args) && args.length ? args : [
  '01-provenance-pathway-attribution',
  '02-surprise-tomography',
  '03-inference-time-motion-energy',
  '04-resize-timing-tax',
  '05-signed-laterality-decodability',
  '06-mask-geometry-as-object',
  '07-group-loss-supervision-isolation',
]

const COMMON = `You are an expert JEPA / world-model research scientist AND a gifted science teacher who makes hard ideas feel obvious to a curious 10th-grade student. You are improving ONE existing proposal README in the portfolio at ${IDEAS}/.

You are NOT rewriting the science or changing any claim, number, split, endpoint, margin, citation, or figure reference. You are making the SAME proposal much easier to read and adding the concrete scaffolding the user asked for. Preserve every section, every grounded number, every citation, and both figure references exactly.

HARD RULES (all are non-negotiable user requirements):
- NO em-dashes and NO en-dashes anywhere. Use commas, colons, periods, or short sentences. Use "to" for ranges (e.g. "16 to 22 Aug", "0.75 to 0.90").
- Every number must still match ${FACTS}. Never invent, round differently, or drop a grounded number. If the current text has a number, keep it identical.
- Keep both figure references (./images/fig1.svg and ./images/fig2.svg) and their captions. Do not rename files.
- Keep all citations exactly as written (author + venue/arXiv id or DOI).
- Keep the responsible-use note (folder labels are dataset annotations, not diagnoses).`

// ---- Phase 1: Simplify + annotate + example + code ----
phase('Simplify')

const simplified = await parallel(SLUGS.map(slug => () =>
  agent(`${COMMON}

MANDATORY READING (use the Read tool) before editing:
1. ${IDEAS}/${slug}/README.md  (the file you will improve)
2. ${FACTS}  (source of truth for every number)

YOUR TASK: rewrite ${IDEAS}/${slug}/README.md in place so it satisfies ALL of the following, then Write the whole improved file back.

1) HIGH-SCHOOL READING LEVEL. Go sentence by sentence. Shorten long sentences. Replace jargon with plain words on first use, then keep the technical term in parentheses so a reviewer still sees it. Prefer concrete verbs and everyday analogies (e.g. "an EMA teacher is like a slow-moving average that ignores day-to-day noise"). Aim so a motivated 10th grader could follow it. Do NOT dumb down the science or remove rigor; just make the wording direct and the logic explicit. Keep every section heading.

2) FULLY EXPLAIN EVERY EQUATION. For EVERY math expression in the file (at minimum the loss \`L = L_JEPA + 0.05 * L_VICReg + 0.25 * L_group\` if present, plus any temperature, fraction, ratio, R^2, cosine, F1, or margin that appears), add a short "Reading the math" explanation immediately after it that:
   - names the whole expression in one plain sentence ("this says the total training loss is a weighted sum of three parts"),
   - defines EVERY symbol and constant on its own bullet line (what L is, what each subscript means, what the numbers 0.05 and 0.25 are and why they are small vs large, what "*" and "+" are doing),
   - states the units or range where it matters (e.g. "a cosine is between -1 and 1", "a fraction is between 0 and 1", "macro-F1 is between 0 and 1, higher is better"),
   - explains in one line what would change if a term were removed or a weight set to zero.
   Use an indented block or a short bullet list. Do not leave any symbol undefined.

3) ADD ONE CONCRETE WORKED NUMERIC EXAMPLE. Insert a clearly labeled "Worked example" (a short subsection or bolded lead-in) that plugs small, explicit made-up-but-plausible numbers into the proposal's central computation and walks the arithmetic step by step, then says in one sentence how to read the result against this proposal's pre-registered margin or baseline. Mark invented numbers as illustrative so they are never confused with the grounded facts. Keep it short (roughly 6 to 12 lines).

4) ADD ONE SHORT ILLUSTRATIVE CODE SNIPPET. Insert a fenced \`\`\`python code block (10 to 25 lines) that shows the KEY operation of this proposal concretely (for example: building the source-video-disjoint split, computing the pooled-vs-image surprise, toggling the group-loss weight, the temporal-resize invariance check, the signed left-minus-right axis, the mask-geometry swap, or the transductive-vs-inductive probe, whichever is this proposal's core). Use plain readable numpy / pseudo-torch with comments a beginner can follow. It illustrates the idea; it does not need to run against real files. Keep variable names self-explanatory. No em-dashes in comments.

Place the worked example and the code near the Method or Decisive-experiment section, wherever they clarify the core mechanic best. Keep the file coherent and well ordered.

After writing, return: (a) a 2-sentence note on what you simplified, (b) confirmation that every number still matches the facts file, (c) confirmation of zero em/en-dashes.`,
    { label: `simplify:${slug.slice(0,20)}`, phase: 'Simplify', effort: 'high' })
    .then(txt => ({ slug, note: txt }))
    .catch(e => ({ slug, note: null, error: String(e) }))
))

log(`Simplified ${simplified.filter(s => s.note).length}/${SLUGS.length} READMEs.`)

// ---- Phase 2: adversarial clarity critique -> Phase 3: polish (pipelined) ----
const CRITIQUE_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['slug', 'still_too_hard', 'unexplained_symbols', 'missing_pieces', 'number_drift', 'dash_found', 'verdict'],
  properties: {
    slug: { type: 'string' },
    still_too_hard: { type: 'array', items: { type: 'string' }, description: 'Specific sentences/phrases still too dense for a 10th grader; quote them.' },
    unexplained_symbols: { type: 'array', items: { type: 'string' }, description: 'Any math symbol/constant/metric that appears without a plain definition.' },
    missing_pieces: { type: 'array', items: { type: 'string' }, description: 'Is a "Reading the math" block, a worked numeric example, or a python code snippet missing or weak? Name which.' },
    number_drift: { type: 'array', items: { type: 'string' }, description: 'Any number that now contradicts _shared_facts.md.' },
    dash_found: { type: 'boolean', description: 'True if any em-dash or en-dash is present.' },
    verdict: { type: 'string', enum: ['CLEAN', 'MINOR_FIXES', 'MAJOR_FIXES'] },
  },
}

const reviewed = await pipeline(
  simplified,
  (s) => agent(`You are a HARSH clarity-and-fidelity reviewer with two hats: (a) a high-school teacher who will flag ANY sentence a motivated 10th grader could not follow, and (b) a fact checker guarding the numbers.

Read with the Read tool:
- ${IDEAS}/${s.slug}/README.md
- ${FACTS}

Check and report concretely:
1. Reading level: quote any sentence/phrase still too dense or jargon-heavy without a plain gloss.
2. Math coverage: list any symbol, constant, ratio, temperature, R^2, cosine, F1, fraction, or margin that appears WITHOUT a plain-language definition nearby. The loss expression especially must have every symbol and both weights (0.05, 0.25) explained.
3. Required additions: confirm the file contains a "Reading the math" style annotation, ONE worked numeric example with step-by-step arithmetic, and ONE fenced python code snippet illustrating the core idea. Flag any that are missing, trivial, or not actually illustrating this proposal's core mechanic.
4. Number fidelity: flag any number that contradicts the facts file.
5. Dashes: report true if ANY em-dash or en-dash is present (hard fail).

Any missing required addition, any unexplained symbol, any dash, or any number drift is at best MINOR_FIXES. Be specific and actionable so it can be fixed.`,
    { label: `critique:${s.slug.slice(0,20)}`, phase: 'Critique', schema: CRITIQUE_SCHEMA, effort: 'high' })
    .then(v => ({ ...s, review: v }))
    .catch(e => ({ ...s, review: null, error: String(e) })),
  (s) => {
    if (!s.review) return s
    const r = s.review
    const clean = r.verdict === 'CLEAN' && !r.dash_found &&
      (r.still_too_hard || []).length === 0 &&
      (r.unexplained_symbols || []).length === 0 &&
      (r.missing_pieces || []).length === 0 &&
      (r.number_drift || []).length === 0
    if (clean) { log(`${s.slug}: clarity clean.`); return { ...s, polished: 'CLEAN' } }
    return agent(`${COMMON}

MANDATORY READING: ${IDEAS}/${s.slug}/README.md and ${FACTS}.

A clarity reviewer flagged the issues below. Fix EVERY one by editing ${IDEAS}/${s.slug}/README.md in place (Read then Edit/Write). Do not regress any other part.

Sentences still too hard (rewrite simpler): ${JSON.stringify(r.still_too_hard)}
Unexplained math symbols/metrics (add plain definitions): ${JSON.stringify(r.unexplained_symbols)}
Missing/weak required pieces (add or strengthen: Reading-the-math block, worked numeric example, python snippet): ${JSON.stringify(r.missing_pieces)}
Number drift vs facts (must match ${FACTS}): ${JSON.stringify(r.number_drift)}
Dash present (remove all em/en-dashes): ${r.dash_found}

Keep every section, every grounded number, every citation, and both figure references. Return a short list of what you changed.`,
      { label: `polish:${s.slug.slice(0,20)}`, phase: 'Polish', effort: 'high' })
      .then(txt => ({ ...s, polished: txt }))
      .catch(e => ({ ...s, polished: null, error: String(e) }))
  }
)

const summary = reviewed.filter(Boolean).map(s => ({
  slug: s.slug,
  verdict: s.review ? s.review.verdict : 'NO_REVIEW',
  dash: s.review ? s.review.dash_found : null,
  issues: s.review ? ((s.review.still_too_hard||[]).length + (s.review.unexplained_symbols||[]).length + (s.review.missing_pieces||[]).length + (s.review.number_drift||[]).length) : null,
  polished: s.polished === 'CLEAN' ? 'clean' : (s.polished ? 'polished' : 'FAILED'),
}))

return { summary }
