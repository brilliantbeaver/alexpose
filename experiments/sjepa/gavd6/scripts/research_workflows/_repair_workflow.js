export const meta = {
  name: 'repair-sjepa-portfolio',
  description: 'Augment+review+revise item 06 (draft failure), re-grade 06 in SCORECARD, patch _selection.json to 12 and rebuild README index',
  phases: [
    { title: 'Fix06', detail: 'augment 06 in place, review, revise' },
    { title: 'Regrade', detail: 're-grade 06 and patch SCORECARD.md' },
    { title: 'Index', detail: 'patch _selection.json to 12 + rebuild README index' },
  ],
}

const RESEARCH_DIR = new URL('../../notes/research', import.meta.url).pathname.replace(/\/$/, '')
const DIR = `${RESEARCH_DIR}/ideas`

const SHARED = `
You are working on Skeleton-JEPA (S-JEPA) gait research proposals inside ${DIR}. You MUST read and obey:
- ${RESEARCH_DIR}/_shared_facts.md   (single source of truth for every number and citation; NEVER contradict it)
- ${RESEARCH_DIR}/_neuro_facts.md    (discriminative symmetric-vs-lateralized axis, condition mechanisms, world-model levers,
                              AND verified authoritative neuroscience citations with real PMIDs/DOIs)
- ${DIR}/05-signed-laterality-decodability/README.md is the STRUCTURAL AND VOICE TEMPLATE.

HARD RULES (a reviewer rejects on any violation):
1. NO em-dashes anywhere. Use commas, periods, or parentheses.
2. Every quantitative claim consistent with _shared_facts.md. Do not invent numbers.
3. Cite AUTHORITATIVE sources for every idea and prior experiment. Neuroscience claims cite the verified PMIDs in
   _neuro_facts.md; method claims cite the arXiv/DOI anchors in the facts files. Never invent a citation/PMID/DOI.
   GAVD is Ranjan et al., IEEE Access 2025, DOI 10.1109/ACCESS.2025.3545787 (never "Nasir & Sederberg").
4. Neuroscience DEFINES the target and the falsifiable prediction. It NEVER upgrades n=18 sources into a clinical-
   accuracy claim. Any clinical-accuracy claim is external-cohort reach-tier ONLY, stated as such.
5. Source video is the independent unit before all fitting. All current results transductive. Folder labels are
   dataset annotations, not diagnoses.
THE UNIFYING THESIS: LATERALIZED (L-R asymmetry) = stroke (corticospinal decussation PMID 30571044), hemiplegic CP
(unilateral PVL PMID 19081519), early PD (contralateral nigrostriatal onset PMID 22367437); biomarker = symmetry
ratio (Patterson 2010 PMID 19932621). RHYTHM/VARIABILITY = PD basal-ganglia loss of automaticity (Redgrave 2010
PMID 20944662; Wu 2015 PMID 26102020); biomarker = stride-time CV (Hausdorff 1998 PMID 9613733; Schaafsma 2003
8.8% vs 4.2% PMID 12809998). SYMMETRIC + rhythm-preserved + posturally abnormal = myopathy (Barohn 2014
PMID 25037080); discriminator = low L-R asymmetry (Xiong 2023 PMID 37525241) + preserved cadence + anterior pelvic
tilt 16.4 vs 11.6 deg (Vandekerckhove 2022 PMID 35721358). CP crouch = min stance knee flexion >=30 deg (de Morais
Filho 2010 PMID 20300011). Skeleton validity: Stenum 2021 PMID 33891585 temporal MAE 0.02 s/step. Skeletons CANNOT
recover kinetics/propulsion (Bowden 2006), EMG/spasticity (Ropars 2016), transverse rotation, or etiologic diagnosis.
`

const DRAFT_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['id', 'status', 'summary'],
  properties: {
    id: { type: 'string' }, status: { type: 'string', enum: ['written', 'failed'] }, summary: { type: 'string' },
  },
}
const REVIEW_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['id', 'verdict', 'overclaim_flags', 'number_inconsistencies', 'rigor_gaps', 'required_fixes'],
  properties: {
    id: { type: 'string' }, verdict: { type: 'string', enum: ['CLEAN', 'REVISE', 'BLOCK'] },
    overclaim_flags: { type: 'array', items: { type: 'string' } },
    number_inconsistencies: { type: 'array', items: { type: 'string' } },
    rigor_gaps: { type: 'array', items: { type: 'string' } },
    required_fixes: { type: 'array', items: { type: 'string' } },
  },
}

// ---- Phase Fix06: augment 06 in place ----
phase('Fix06')
const LIFT_06 = `Promote 06 from "does one structured mask beat uniform on gavd5" to the MASK-FAMILY-AT-EQUAL-COVERAGE
PRINCIPLE. Pre-register mask families by mechanism: contralateral-pair masking (forces asymmetry infill -> stroke/CP,
Patterson 2010 PMID 19932621), half-cycle / future-phase masking (forces rhythm infill -> PD, Hausdorff PMID 9613733),
proximal-segment masking (forces symmetric proximal infill -> myopathy, Vandekerckhove PMID 35721358). Generalizable
claim: gait-specific mask geometry, at matched difficulty, shapes which clinical axis becomes linearly recoverable,
transferable to any skeleton-JEPA. Keep the equal-coverage / marginal-matched-scramble difficulty control and the
provenance-probe HARD KILL GATE. Feasibility: medium (fold-local retrain, 4-6 weeks). Mark honestly. External-cohort:
PD rhythm family maps to PhysioNet gaitpdb (cross-modal label level, DOI 10.13026/C24H3N); asymmetry and symmetric-
proximal families have NO external skeleton cohort (honest limitation).`

await agent(
  `${SHARED}

TASK: AUGMENT the existing proposal IN PLACE at ${DIR}/06-mask-geometry-as-object/README.md. Read the whole file
first. Add a new section titled "## Conference-level augmentation" placed immediately AFTER the "## Why this matters"
section. Do NOT move or delete any existing content, and PRESERVE the original one-line question in the blockquote
verbatim. The new section must contain, in the plain-language template voice with "Reading the math" style clarity
where numbers appear:
- The neuroscience source -> mechanism -> skeleton-measurable-feature chain with real PMIDs, framed as the three
  pre-registered mechanism-defined mask families (contralateral-pair -> asymmetry; half-cycle/future-phase -> rhythm;
  proximal-segment -> symmetric proximal), each tied to its validated biomarker.
- The generalizable-claim restatement (mask geometry at matched difficulty shapes which clinical axis is recoverable;
  transferable to any skeleton-JEPA).
- The biomarker-specific external-cohort note (PD rhythm -> PhysioNet gaitpdb cross-modal label level; asymmetry and
  symmetric-proximal families have no external skeleton cohort, honest limitation).
- An honest feasibility delta vs the original (fold-local retrain, 4 to 6 weeks core).
Then EXTEND the "## References" list with any newly cited neuroscience PMIDs and world-model anchors not already
present (no duplicates). Keep everything consistent with _shared_facts.md. No em-dashes. Save the file.

INTENDED LIFT:
${LIFT_06}

Return the structured draft summary.`,
  { label: 'draft:06', phase: 'Fix06', schema: DRAFT_SCHEMA, effort: 'high' }
)

const review06 = await agent(
  `${SHARED}

You are a HARSH ICLR/ICML area chair reviewing ${DIR}/06-mask-geometry-as-object/README.md. Read it in full plus the
two facts files. Return the review. HARD flag any clinical-accuracy claim on gavd5 (n=18) not hedged as external-
cohort reach-tier; HARD fail any number drifting from _shared_facts.md; check the source-video unit, transductive
labeling, the provenance kill gate, the equal-coverage difficulty control, an informative null, and that the original
one-line question is preserved verbatim. Flag any em-dash in rigor_gaps. verdict CLEAN only if no HARD flags and no
material rigor gaps; REVISE if fixable; BLOCK only if unsalvageable. List concrete required_fixes.`,
  { label: 'review:06', phase: 'Fix06', schema: REVIEW_SCHEMA, effort: 'high' }
)

if (!review06 || review06.verdict !== 'CLEAN') {
  const fixes = review06 ? JSON.stringify({
    overclaim_flags: review06.overclaim_flags, number_inconsistencies: review06.number_inconsistencies,
    rigor_gaps: review06.rigor_gaps, required_fixes: review06.required_fixes,
  }, null, 2) : '(no review; self-review against the hard rules)'
  await agent(
    `${SHARED}

Revise ${DIR}/06-mask-geometry-as-object/README.md IN PLACE to fix the reviewer findings below. Read the current file
first. Apply every required fix, remove any HARD flag (clinical overclaim not hedged to reach-tier, any number drift
from _shared_facts.md, any em-dash). PRESERVE the original one-line question verbatim. Keep the plain-language voice.
Save the file. Return the draft summary.

REVIEWER FINDINGS:
${fixes}`,
    { label: 'revise:06', phase: 'Fix06', schema: DRAFT_SCHEMA, effort: 'high' }
  )
}
log('Fix06 complete: 06 augmented, reviewed, revised')

// ---- Phase Regrade: re-grade 06 against its now-augmented content and patch SCORECARD.md ----
phase('Regrade')
const RUBRIC = `ICLR/ICML likelihood: score five axes 1-5 (Novelty, Mechanism-grounding, Generalizability-of-claim,
Rigor/evaluation-validity, Feasibility-given-ambition). Composite = 0.20*Nov + 0.20*Mech + 0.25*Gen + 0.25*Rig +
0.10*Feas on a 1.0-5.0 scale. Bands: >=4.3 Strong main-track candidate; 3.6-4.2 Competitive with revision; 2.8-3.5
Workshop/borderline; <2.8 Not yet conference-level. Generalizability stays capped near 3 for CP/myopathy-specific
claims (no public skeleton cohort); the PD-rhythm family has only a cross-modal label-level external anchor
(PhysioNet gaitpdb), so it does not lift Gen to 5. Do NOT let effort bleed into likelihood. Effort tuple:
(core weeks / +reach weeks); retrain scale; data/compute needs.`

const regrade06 = await agent(
  `${SHARED}

Re-grade portfolio item 06 (mask-geometry-as-object) against its NOW-AUGMENTED content. Read
${DIR}/06-mask-geometry-as-object/README.md in full (it now carries a Conference-level augmentation section with three
mechanism-defined mask families and the mask-family-at-equal-coverage principle) plus _shared_facts.md and
_neuro_facts.md. Apply this rubric and return the grade.

${RUBRIC}

Then PATCH ${RESEARCH_DIR}/SCORECARD.md IN PLACE: (a) update the item-06 row in the table with the new axes, composite, band,
and effort, and RE-SORT the whole table by composite descending so 06 sits in its correct rank; (b) rewrite the "**06
mask-geometry-as-object ...**" per-item paragraph to describe the mask-family-at-equal-coverage principle and the three
mechanism-defined families (contralateral-pair -> asymmetry, half-cycle -> rhythm, proximal-segment -> symmetric
proximal), stating what a positive result confirms and what an informative null overturns, plus the effort rationale.
Do not touch any other item's row or paragraph. No em-dashes. Keep every number consistent with _shared_facts.md.
Save the file.`,
  {
    label: 'regrade:06', phase: 'Regrade', effort: 'high',
    schema: {
      type: 'object', additionalProperties: false,
      required: ['id', 'band', 'composite', 'axes', 'effort'],
      properties: {
        id: { type: 'string' }, band: { type: 'string' }, composite: { type: 'number' },
        axes: {
          type: 'object', additionalProperties: false,
          required: ['novelty', 'mechanism', 'generalizability', 'rigor', 'feasibility'],
          properties: {
            novelty: { type: 'number' }, mechanism: { type: 'number' }, generalizability: { type: 'number' },
            rigor: { type: 'number' }, feasibility: { type: 'number' },
          },
        },
        effort: { type: 'string' }, quality_summary: { type: 'string' },
      },
    },
  }
)
log(`Regrade complete: 06 -> composite ${regrade06 ? regrade06.composite : 'FAILED'} (${regrade06 ? regrade06.band : ''})`)

// ---- Phase Index: patch _selection.json to 12 items + rebuild README.md index ----
phase('Index')
const grades = args && args.grades ? args.grades : {}
if (regrade06) grades['06'] = { slug: 'mask-geometry-as-object', band: regrade06.band, composite: regrade06.composite, axes: regrade06.axes, effort: regrade06.effort }
const newItems = args && args.newItems ? args.newItems : {}

await parallel([
  // Patch _selection.json to 12 items with augmentation + scorecard blocks
  () => agent(
    `${SHARED}

TASK: Rewrite ${RESEARCH_DIR}/_selection.json so its "selected" array holds all 12 items (currently only 7). Read the current
_selection.json first to preserve the existing schema of each item object (fields rank, slug, title,
one_line_question, why_it_wins, incorporated_repairs, figure_specs). For all 12 items:
- Keep the existing 7 items' original fields intact but RE-RANK all 12 by the scorecard composite descending (rank 1
  = highest composite). Add the 5 new items (08-12) with the same field schema; derive title and one_line_question by
  reading the FIRST heading and the blockquote of each new README at ${DIR}/<slug>/README.md.
- Add to EVERY item two new blocks: "augmentation" = { neuro_chain, generalizable_claim, external_cohort,
  feasibility_tier } (read each README's Conference-level augmentation section, or Background/Method for new items, to
  fill these succinctly and truthfully), and "scorecard" = { band, composite, axes:{novelty,mechanism,
  generalizability,rigor,feasibility}, effort } taken EXACTLY from the grades map below.
- Preserve the top-level "portfolio_rationale" but update it to reflect 12 items and the symmetry-axis thesis.
Write valid JSON (no trailing commas, no comments, no em-dashes in any string). Save the file.

GRADES (authoritative scorecard numbers, use verbatim for the scorecard blocks):
${JSON.stringify(grades, null, 2)}

NEW-ITEM TITLES (for the 5 new items, use as the title field; derive one_line_question from the README blockquote):
${JSON.stringify(newItems, null, 2)}

Return the JSON you wrote as a string.`,
    { label: 'patch:selection', phase: 'Index', effort: 'high' }
  ),
  // Rebuild README.md index to 12 rows + symmetry-axis thesis + fourth theme
  () => agent(
    `${SHARED}

TASK: Rewrite ${RESEARCH_DIR}/README.md as the portfolio index for all 12 items (currently it describes only seven). Read the
current README.md first to keep its plain-language voice, the shared-setup section, the transductive-vs-inductive
rule, the reviewer-rubric section, and the responsible-use and shared-references sections. Changes to make:
- Retitle from "Seven ... ideas" to reflect 12 proposals (seven augmented audits plus five neuroscience-grounded
  world-model proposals). Update the intro paragraph count from seven to twelve and note the ambition-first framing
  (some items exceed three weeks; feasibility is marked honestly).
- Add a short paragraph stating the unifying discriminative SYMMETRY-AXIS thesis (lateralized vs rhythm/variability vs
  symmetric-proximal, each tied to a validated biomarker with a PMID: Patterson 2010 PMID 19932621; Schaafsma 2003
  PMID 12809998; Xiong 2023 PMID 37525241 / Vandekerckhove 2022 PMID 35721358).
- Replace the "seven proposals at a glance" table with a 12-row table. Columns: # | Proposal (linked to
  ./<slug>/README.md) | Plain question (the README blockquote, condensed) | Theme | ICLR/ICML band | Effort. Sort by
  scorecard composite descending. Add a FOURTH theme "Neuro-grounded world model" for items 08-12 (some 05/09 may
  share Mechanism). Use the bands and effort from the grades map below.
- Add one sentence pointing readers to SCORECARD.md for the full graded rubric.
- Keep the "How these differ from plan/" section but extend it with one line each for 08-12 (distinctness).
- Keep GAVD cited as Ranjan et al. 2025 (DOI 10.1109/ACCESS.2025.3545787). No em-dashes anywhere. Save the file.

GRADES (bands + effort per item id; also gives the composite for sort order):
${JSON.stringify(grades, null, 2)}

NEW-ITEM TITLES:
${JSON.stringify(newItems, null, 2)}

Return a short summary of the rewritten index.`,
    { label: 'patch:readme', phase: 'Index', effort: 'high' }
  ),
])
log('Index complete: _selection.json patched to 12, README.md rebuilt')

return { regrade06, grades }
