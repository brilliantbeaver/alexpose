export const meta = {
  name: 'augment-sjepa-portfolio',
  description: 'Augment 7 S-JEPA proposals in place + draft 5 new + adversarial ICLR/ICML review + graded scorecard',
  phases: [
    { title: 'Ground', detail: 'emit per-item augmentation contracts' },
    { title: 'Draft', detail: 'augment 01-07 in place + draft 08-12' },
    { title: 'Review', detail: 'adversarial ICLR/ICML reviewer per item' },
    { title: 'Revise', detail: 'apply reviewer repairs in place' },
    { title: 'Scorecard', detail: 'grade all 12, update index + selection json' },
  ],
}

const RESEARCH_DIR = new URL('../../notes/research', import.meta.url).pathname.replace(/\/$/, '')
const DIR = `${RESEARCH_DIR}/ideas`

// Shared context every agent must read and obey.
const SHARED = `
You are writing/reviewing research proposals for a Skeleton-JEPA (S-JEPA) gait project. Work inside
${DIR}. You MUST read these files before writing and obey them exactly:
- ${RESEARCH_DIR}/_shared_facts.md   (single source of truth for every number and citation; NEVER contradict it)
- ${RESEARCH_DIR}/_neuro_facts.md    (the discriminative symmetric-vs-lateralized axis, condition mechanisms, world-model
                              levers, AND verified authoritative neuroscience citations with real PMIDs/DOIs)
- The existing proposal ${DIR}/05-signed-laterality-decodability/README.md is the STRUCTURAL AND VOICE TEMPLATE:
  plain-language "Reading the math" boxes that explain every symbol/variable/range, a worked numeric example
  (labelled illustrative, not grounded), lane tables, pre-registered margins, a distinctness-from-plan section,
  a three-week (or feasibility-tiered) timeline with Day-5/Day-14 gates, a responsible-use note, and a
  references list. Match that voice and section structure.

HARD RULES (a reviewer will reject on any violation):
1. NO em-dashes anywhere. Use commas, periods, or parentheses.
2. Every quantitative claim must be consistent with _shared_facts.md. Do not invent numbers.
3. Cite AUTHORITATIVE sources for every idea and every prior experiment. Neuroscience claims cite the verified
   PMIDs in _neuro_facts.md; method claims cite the arXiv/DOI anchors in _shared_facts.md / _neuro_facts.md.
   Never invent a citation, PMID, or DOI. GAVD is Ranjan et al., IEEE Access 2025, DOI 10.1109/ACCESS.2025.3545787
   (never "Nasir & Sederberg").
4. The neuroscience DEFINES THE TARGET and the falsifiable prediction. It NEVER upgrades n=18 sources into a
   clinical-accuracy claim. Any clinical-accuracy claim is external-cohort reach-tier ONLY, and stated as such.
5. Source video is the independent unit before all fitting. All current results are transductive; label every
   number transductive where the encoder saw the rows. Folder labels are dataset annotations, not diagnoses.
6. External-cohort reality (do not overclaim skeleton-level clinical transfer):
   - PD stride-time-variability biomarker -> PhysioNet Gait-in-PD (gaitpdb, 93 PD + 73 controls, Hausdorff,
     DOI 10.13026/C24H3N): force/IMU, so a LABEL-LEVEL cross-modal confirmation of the variability biomarker.
   - Cross-view/viewpoint-invariance -> CASIA-B (Yu 2006), OU-MVLP-Pose (Takemura 2018), GREW (arXiv:2205.02692),
     Gait3D (arXiv:2204.02569): multi-view pose, NON-clinical.
   - Pose validity -> Human3.6M (Ionescu 2014, DOI 10.1109/tpami.2013.248) vs mocap.
   - CP and myopathy skeleton participant-disjoint public cohorts do NOT exist: state as an honest limitation.

THE UNIFYING THESIS (spine of every item): conditions separate along a mechanism-defined axis, each tied to a
VALIDATED biomarker skeletons can measure. LATERALIZED (left-right asymmetry): stroke (corticospinal
decussation -> contralateral hemiparesis, PMID 30571044), hemiplegic CP (unilateral PVL, PMID 19081519), early
PD (contralateral nigrostriatal onset, PMID 22367437); validated biomarker = symmetry ratio (Patterson 2010,
PMID 19932621). RHYTHM/VARIABILITY (high stride-time CV): PD basal-ganglia loss of automaticity (Redgrave 2010
PMID 20944662; Wu 2015 PMID 26102020); validated biomarker = stride-time CV (Hausdorff 1998 PMID 9613733;
Schaafsma 2003 fallers 8.8% vs non-fallers 4.2%, PMID 12809998). SYMMETRIC + rhythm-preserved + posturally
abnormal: myopathy primary muscle disease (Barohn 2014 PMID 25037080); discriminator = low L-R asymmetry
(Xiong 2023 PMID 37525241) + preserved cadence + anterior pelvic tilt 16.4 vs 11.6 deg (Vandekerckhove 2022
PMID 35721358). CP crouch = min stance knee flexion >=30 deg (de Morais Filho 2010 PMID 20300011). Skeleton
validity: Stenum 2021 (PMID 33891585) temporal MAE 0.02 s/step, sagittal joints 4-7 deg. Skeletons CANNOT
recover kinetics/propulsion (Bowden 2006), EMG/spasticity (Ropars 2016), transverse rotation, or etiologic
muscle diagnosis: say so in limitations.
`

// The 12 items. type: 'augment' edits an existing README in place; type: 'new' writes a new folder README.
const ITEMS = [
  {
    id: '01', type: 'augment', slug: '01-provenance-pathway-attribution',
    lift: `Lift from single-cohort audit to a NAMED, TRANSFERABLE leakage diagnostic for grouped skeleton SSL:
      define a Provenance-Decodability Index (fraction of a headline separability recovered by a same-label
      acquisition-pathway probe, reported against the source-identity upper bound), framed under Kapoor &
      Narayanan (arXiv:2207.07048) as a new leakage subtype for self-supervised skeleton pipelines. Neuro is
      light here, but add one sentence: myopathy is the only near-symmetric class (Xiong 2023 PMID 37525241)
      and dominates the canonical path (10/18 sources), so a provenance leak disproportionately corrupts the
      symmetric-vs-lateralized axis every other proposal depends on. This item stays fast (near-zero-retrain);
      it is the bulletproof foundation, do not chase ambition here.`,
  },
  {
    id: '02', type: 'augment', slug: '02-surprise-tomography',
    lift: `PRE-REGISTER the [12 joint x 16 segment] surprise topography from the neuro priors BEFORE looking:
      stroke/hemiplegic-CP -> asymmetric knee/ankle cells (Patterson 2010 PMID 19932621; Chen 2005 stiff-knee
      PMID 15996592); PD -> temporally dispersed variability signature (Hausdorff/Schaafsma PMID 12809998);
      myopathy -> symmetric proximal hip/pelvis cells with preserved temporal regularity (Xiong 2023
      PMID 37525241; Vandekerckhove PMID 35721358). Generalizable claim: masked-prediction error is spatially
      organized by lesion mechanism, and the topography (not the magnitude) is the discriminative object. Add a
      synthetic-injection sanity check (amplify one-sided knee flexion -> error mass must move to the matching
      cells). Fast. This is a headline-candidate promotion.`,
  },
  {
    id: '03', type: 'augment', slug: '03-inference-time-motion-energy',
    lift: `Tie motion-energy recoverability to the PD RHYTHM mechanism (loss of automaticity -> variability is
      the hallmark: Hausdorff PMID 9613733; Wu 2015 PMID 26102020). Reframed endpoint: does frozen-predictor
      residual velocity structure recover the RELATIVE within-window rhythm regularity axis that distinguishes
      PD from the symmetric myopathy baseline, above a raw-coordinate velocity ceiling? Keep the shuffled-motion
      control. HONEST BOUNDARY: 03 cannot recover ABSOLUTE cadence (steps/min); that is item 04's algebraic
      finding (the 64-frame resize erases it), so 03's rhythm claim is restricted to within-window relative
      regularity. State this cross-link to 04 explicitly. Stays cheap, near-zero-retrain.`,
  },
  {
    id: '04', type: 'augment', slug: '04-resize-timing-tax',
    lift: `Promote to a NAMED preprocessing-invariance theorem: duration-warping to a fixed 64-frame token grid
      is a cadence-erasing group action, so absolute temporal biomarkers are non-identifiable by construction.
      Quantify the CLINICAL cost with the two biomarkers that are literally steps/min: post-stroke slowing /
      reduced cadence (Patterson-family symmetry+speed, PMID 18226655) vs PD reduced-stride + compensatory
      cadence (Morris 1994 PMID 7953597; Morris 1996 PMID 8800948). Claim generalizes to ANY fixed-frame
      skeleton pipeline (V-JEPA tube-length normalization included, arXiv:2404.08471). Keep the Week-1
      fps-regeneration kill-gate (29.97 fallback contamination risk). Highest ceiling of the cheap four.`,
  },
  {
    id: '05', type: 'augment', slug: '05-signed-laterality-decodability',
    lift: `Recast this item's ROLE as the EMPIRICAL PROBE LAYER of new item 09 (reflection-equivariance): its
      signed-decodability-vs-raw-null probe and LEFT_RIGHT_PAIRS mirror test are the MEASUREMENT INSTRUMENT that
      09's architectural claim is evaluated with. Make the mechanism framing central: the signed L-R axis is the
      literature-validated discriminant (Patterson 2010 symmetry ratio PMID 19932621; PD asymmetric onset
      Riederer PMID 22367437; hemiplegic vs diplegic CP via PVL laterality Volpe PMID 19081519) that separates
      lateralized from symmetric conditions, and myopathy's ABSENCE of L-R asymmetry (Xiong 2023 PMID 37525241)
      is the built-in negative control. Add a short cross-reference to item 09. Keep everything else; stays fast.`,
  },
  {
    id: '06', type: 'augment', slug: '06-mask-geometry-as-object',
    lift: `Promote from "does one structured mask beat uniform on gavd5" to the MASK-FAMILY-AT-EQUAL-COVERAGE
      PRINCIPLE. Pre-register mask families by mechanism: contralateral-pair masking (forces asymmetry infill ->
      stroke/CP, Patterson 2010 PMID 19932621), half-cycle / future-phase masking (forces rhythm infill -> PD,
      Hausdorff PMID 9613733), proximal-segment masking (forces symmetric proximal infill -> myopathy,
      Vandekerckhove PMID 35721358). Generalizable claim: gait-specific mask geometry, at matched difficulty,
      shapes which clinical axis becomes linearly recoverable, transferable to any skeleton-JEPA. Keep the
      equal-coverage / marginal-matched-scramble difficulty control and the provenance-probe HARD KILL GATE.
      Feasibility: medium (fold-local retrain, 4-6 weeks). Mark honestly.`,
  },
  {
    id: '07', type: 'augment', slug: '07-group-loss-supervision-isolation',
    lift: `Reframe from "does the group loss buy generalization here" to a general claim about
      SUPERVISED-FINE-TUNING-INSIDE-SSL: does label-aware centroid supervision in a JEPA curriculum produce
      transferable structure or transductive memorization, measured by the transductive-minus-inductive gap and
      per-stage V-usable information (Xu 2020, arXiv:2002.10689)? This is a live methods question vs V-JEPA-style
      frozen-probe orthodoxy (arXiv:2404.08471). Neuro anchor: ask whether the forgotten axis behind the
      normal-anchor drift 0.954->0.594 is the clinically load-bearing symmetric baseline (normal/myopathy)
      geometry (myopathy symmetric, Xiong 2023 PMID 37525241). Keep every compute-scoping repair (small fixed
      holdout, matched-step fold-local finetune, sampler-leakage third arm, collapse gate as hard reject).
      Feasibility: medium-high.`,
  },
  {
    id: '08', type: 'new', slug: '08-concept-bottleneck-disentangled',
    title: 'Concept-bottleneck disentangled S-JEPA: named z_asym / z_rhythm / z_posture subspaces tied to validated biomarkers',
    lift: `HEADLINE. Three named latent subspaces, each tied to a validated biomarker with a raw-input probe
      ceiling: z_asym <- symmetry ratio (Patterson 2010 PMID 19932621), z_rhythm <- stride-time CV (Schaafsma
      2003 8.8 vs 4.2%, PMID 12809998; Hausdorff PMID 9613733), z_posture <- anterior pelvic tilt / trunk lean
      (Vandekerckhove 2022 16.4 vs 11.6 deg, PMID 35721358; de Morais Filho crouch >=30 deg, PMID 20300011).
      Auxiliary per-subspace heads + VICReg (arXiv:2105.04906) + biased masking enforce the split. Falsifiable
      claim (steerability test): intervening on one subspace moves ONLY its biomarker, credited only above the
      raw-coordinate ceiling. Cite Locatello 2019 ICML (disentanglement needs inductive bias / supervision) as
      the honest caveat naming WHY the biomarker heads are required. World-model anchor: V-JEPA (arXiv:2404.08471).
      Effort HIGH: full curriculum retrain + new heads, 6-8 weeks; external arm = PhysioNet gaitpdb (PD rhythm,
      cross-modal) + honest limitation that CP/myopathy skeleton cohorts do not exist publicly.`,
  },
  {
    id: '09', type: 'new', slug: '09-reflection-equivariant-symmetry-axis',
    title: 'Reflection-equivariant representation: separating lateralized from symmetric gait by construction',
    lift: `HEADLINE. Constrain/build the encoder so the signed left-minus-right axis is ANTISYMMETRIC BY
      CONSTRUCTION (reflection-equivariant): mirroring the input negates the axis. Then show it mechanistically
      separates LATERALIZED (stroke corticospinal decussation PMID 30571044; hemiplegic CP unilateral PVL
      PMID 19081519; early PD contralateral onset PMID 22367437) from SYMMETRIC (myopathy, Xiong 2023
      PMID 37525241 as the negative class). The no-flip rule (flip_probability stays 0.0) protects lateralized
      asymmetry and is the whole point. Generalizable claim: reflection-equivariance is the correct inductive
      bias for lateralized-vs-symmetric pathology. Use item 05 as the EVALUATION INSTRUMENT (signed-decodability
      + mirror test). Effort HIGH: architecture change + retrain. Cite VICReg (arXiv:2105.04906), the LEFT_RIGHT_
      PAIRS operation, and equivariance/inductive-bias literature honestly.`,
  },
  {
    id: '10', type: 'new', slug: '10-prediction-error-severity',
    title: 'Prediction-error-as-severity: a normal-only gait world model with mechanism-grouped error decomposition',
    lift: `Train on NORMAL ONLY; use relative masked-prediction error as a continuous anomaly/severity score;
      validate with a mechanism-grouped error decomposition (asymmetry cells vs rhythm cells vs posture cells)
      and a SYNTHETIC-INJECTION DIRECTION TEST: inject one-sided knee flexion -> error rises in the asymmetry
      channel and tracks injection magnitude; inject symmetric proximal deficit -> rises in posture channel.
      Random-encoder control. Anchors: V-JEPA-2 (arXiv:2506.09985), Garrido intuitive-physics violation-of-
      expectation (arXiv:2502.11831). Elegantly side-steps the label-collinearity problem because it NEVER
      trains on abnormal labels. Scope to NOT overlap 02: 02 is a frozen-encoder readout comparison on the
      existing checkpoint; 10 is a normal-only model trained from scratch as a severity world model. Biomarkers:
      symmetry ratio (PMID 19932621), stride-time CV (PMID 12809998), anterior pelvic tilt (PMID 35721358).
      Effort medium-high: from-scratch normal-only train. External: PhysioNet gaitpdb cross-modal for PD.`,
  },
  {
    id: '11', type: 'new', slug: '11-target-isolation-substrate',
    title: 'What should a gait world model predict? A matched-substrate target-isolation study',
    lift: `Hold encoder / compute / steps / mask FIXED and vary ONLY the prediction target across {raw
      coordinates, one-frame motion (MAMP-style), S-JEPA centered-sharpened latent, normalized latent
      regression}; measure which target yields the best MECHANISM-PROBE recovery (symmetry ratio PMID 19932621;
      stride-time CV PMID 12809998; anterior pelvic tilt PMID 35721358) at a RAW-INPUT CEILING. Purest
      generalizable-principle proposal ("what to predict for gait"), engaging the JEPA target-design frontier
      (V-JEPA arXiv:2404.08471; V-JEPA-2 arXiv:2506.09985). MUST state sharp distinctness from plan/04 (a 2-way
      position-vs-motion ablation): 11 is a 4-way target-family study with a mechanism-probe endpoint and a raw
      ceiling. Effort HIGH: multiple retrains.`,
  },
  {
    id: '12', type: 'new', slug: '12-cross-view-invariance',
    title: 'Cross-view gait invariance with a no-flip rule that protects lateralized asymmetry',
    lift: `Viewpoint-conditioned predictor with view_delta as an "action" (V-JEPA-2 action-conditioning framing,
      arXiv:2506.09985); leave-one-view-out evaluation; NO-FLIP rule protecting lateralized asymmetry
      (flip would erase the stroke/CP signed axis, Patterson 2010 PMID 19932621). HONEST SCOPE (load-bearing):
      gavd5 is monocular YouTube with no controlled multi-view data, so the CORE invariance arm runs on public
      multi-view POSE cohorts (CASIA-B Yu 2006; OU-MVLP-Pose Takemura 2018; GREW arXiv:2205.02692; Gait3D
      arXiv:2204.02569), all NON-clinical, with gavd5 view-labels only as a secondary probe; the clinical claim
      stays reach-tier. Generalizable claim: view is an action a gait world model can condition on, and
      invariance must be direction-preserving (no L-R flip) for lateralized pathology. Effort HIGH; the
      honest-scope statement is what keeps this reviewable rather than overclaimed.`,
  },
]

const CONTRACT_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['id', 'neuro_chain', 'generalizable_claim', 'external_cohort', 'feasibility_tier', 'key_citations'],
  properties: {
    id: { type: 'string' },
    neuro_chain: { type: 'string', description: 'neuro source -> mechanism -> skeleton-measurable feature, with PMIDs' },
    generalizable_claim: { type: 'string', description: 'the transferable method/principle claim, one sentence' },
    external_cohort: { type: 'string', description: 'named external cohort + honest scope (or honest limitation)' },
    feasibility_tier: { type: 'string', description: 'core weeks / +reach weeks; retrain scale; data needs' },
    key_citations: { type: 'array', items: { type: 'string' }, description: 'PMIDs/arXiv/DOIs this item leans on' },
  },
}

const DRAFT_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['id', 'path', 'status', 'summary', 'neuro_chain_used', 'generalizable_claim', 'external_cohort', 'feasibility_tuple'],
  properties: {
    id: { type: 'string' },
    path: { type: 'string' },
    status: { type: 'string', enum: ['written', 'failed'] },
    summary: { type: 'string' },
    neuro_chain_used: { type: 'string' },
    generalizable_claim: { type: 'string' },
    external_cohort: { type: 'string' },
    feasibility_tuple: { type: 'string' },
  },
}

const REVIEW_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['id', 'verdict', 'mechanism_chain_valid', 'overclaim_flags', 'number_inconsistencies',
    'generalizability_verdict', 'rigor_gaps', 'distinctness_ok', 'required_fixes'],
  properties: {
    id: { type: 'string' },
    verdict: { type: 'string', enum: ['CLEAN', 'REVISE', 'BLOCK'] },
    mechanism_chain_valid: { type: 'boolean', description: 'every cited PMID load-bearing and not invented' },
    overclaim_flags: { type: 'array', items: { type: 'string' }, description: 'any clinical claim exceeding n=18 without reach-tier hedge (HARD)' },
    number_inconsistencies: { type: 'array', items: { type: 'string' }, description: 'any drift from _shared_facts.md (HARD fail)' },
    generalizability_verdict: { type: 'string' },
    rigor_gaps: { type: 'array', items: { type: 'string' } },
    distinctness_ok: { type: 'boolean' },
    required_fixes: { type: 'array', items: { type: 'string' } },
  },
}

// ---- Phase 0: Ground -> per-item augmentation contracts ----
phase('Ground')
const contracts = await parallel(ITEMS.map(it => () =>
  agent(
    `${SHARED}\n\nYou are the grounding pass for portfolio item ${it.id} (${it.slug}). Read _shared_facts.md and
     _neuro_facts.md. Given the intended lift below, emit the augmentation contract: the exact neuro source ->
     mechanism -> skeleton-feature chain with real PMIDs, the one-sentence generalizable claim, the named external
     cohort with honest scope (or an explicit honest limitation if none exists), the feasibility tier, and the key
     citations. Be precise and consistent with the two facts files. Do not write any files.\n\nINTENDED LIFT:\n${it.lift}`,
    { label: `ground:${it.id}`, phase: 'Ground', schema: CONTRACT_SCHEMA, effort: 'high' }
  ).then(c => ({ ...(c || {}), id: it.id }))
))
const contractById = {}
for (const c of contracts.filter(Boolean)) contractById[c.id] = c
log(`Ground: ${Object.keys(contractById).length}/${ITEMS.length} augmentation contracts emitted`)

// ---- Phases 1-3: Draft -> Review -> Revise, pipelined per item ----
function draftPrompt(it, contract) {
  const contractStr = contract ? JSON.stringify(contract, null, 2) : '(no contract emitted; derive it yourself from the facts files)'
  if (it.type === 'augment') {
    return `${SHARED}

TASK: AUGMENT the existing proposal IN PLACE at ${DIR}/${it.slug}/README.md. First Read the whole file. Then add a
new section titled "## Conference-level augmentation" placed immediately AFTER the "## Why this matters" section
(create it there; do not move or delete any existing content, and PRESERVE the original one-line question in the
blockquote verbatim). The new section must contain, in the plain-language voice of the template, with "Reading the
math" style clarity where numbers appear:
- The neuroscience source -> mechanism -> skeleton-measurable-feature chain, with the real PMIDs.
- The generalizable-claim restatement (what transfers beyond gavd5: the method/principle).
- The biomarker-specific external-cohort note (honest scope; or honest limitation).
- An honest feasibility delta vs the original (core weeks / reach weeks; retrain scale).
Then EXTEND the "## References" list at the bottom with any newly cited neuroscience PMIDs and world-model anchors
(do not duplicate existing entries). Keep everything consistent with _shared_facts.md. No em-dashes.

AUGMENTATION CONTRACT for this item:
${contractStr}

INTENDED LIFT:
${it.lift}

When done, return the structured draft summary. The file must be saved before you return.`
  }
  return `${SHARED}

TASK: WRITE A NEW proposal at ${DIR}/${it.slug}/README.md (create the folder if needed; also create an empty-ish
${DIR}/${it.slug}/images/ by noting figures are deferred, do NOT draw SVGs). Title: "${it.title}".
Match the FULL section structure and plain-language voice of the template
(${DIR}/05-signed-laterality-decodability/README.md): a level-1 title, a one-sentence falsifiable question in a
blockquote, "## The question in plain words", "## Why this matters" (including what an informative null rules out),
"## Background and related work" (cite authoritative sources for every idea and prior experiment), "## Method"
(reuse the 528-token tensors / d0acc262 checkpoint / existing code where possible; include a short readable
pseudo-code block and "Reading the math" boxes for any formula), "## The decisive experiment" (source-video-disjoint
split stated BEFORE fitting, a pre-registered numeric margin, a simple non-neural/nuisance baseline, and a lane
table), "## Controls" , "## How this differs from the existing plan" (state distinctness sharply), a
feasibility-tiered timeline with Day-5 and Day-14 gates (ambition-first: core tier can exceed 3 weeks; mark reach
tier honestly), "## Figures" (reference ./images/fig1.svg and ./images/fig2.svg as DEFERRED, describe what they will
show), "## Responsible use", and "## References". Keep every number consistent with _shared_facts.md. No em-dashes.

AUGMENTATION CONTRACT for this item:
${contractStr}

INTENDED LIFT / SCOPE:
${it.lift}

When done, return the structured draft summary. The file must be saved before you return.`
}

phase('Draft')
const results = await pipeline(
  ITEMS,
  // Stage 1: draft/augment in place
  (it) => agent(draftPrompt(it, contractById[it.id]),
    { label: `draft:${it.id}`, phase: 'Draft', schema: DRAFT_SCHEMA, effort: 'high' }),
  // Stage 2: adversarial ICLR/ICML review (uses originalItem for context)
  (draft, it) => agent(
    `${SHARED}

You are a HARSH ICLR/ICML area chair reviewing the proposal at ${DIR}/${it.slug}/README.md. Read the file in full,
plus _shared_facts.md and _neuro_facts.md. Judge it against these gates and return the review:
- mechanism_chain_valid: is every cited PMID/DOI load-bearing and plausibly real (not invented), and does the
  neuro source -> mechanism -> skeleton-feature chain actually support the proposal's target?
- overclaim_flags: HARD flag any clinical-accuracy claim on gavd5 (n=18 sources) that is not explicitly hedged as
  external-cohort reach-tier. Also flag any claim that skeletons recover kinetics/EMG/transverse-rotation/etiologic
  diagnosis without stating the limitation.
- number_inconsistencies: HARD fail any number that drifts from _shared_facts.md.
- generalizability_verdict: does the result transfer beyond gavd5 (a method/principle), or is it only a gavd5 caveat?
- rigor_gaps: source-video unit before fitting? transductive labeling? provenance gate/kill-gate where relevant?
  raw-input ceiling? informative null? single-factor control?
- distinctness_ok: is it clearly distinct from the existing plan/ portfolio and from its sibling items?
- em-dashes present anywhere? (put in rigor_gaps if so.)
Set verdict CLEAN only if there are no HARD flags and no material rigor gaps; REVISE if fixable; BLOCK only if the
core idea is unsalvageable. List concrete required_fixes.`,
    { label: `review:${it.id}`, phase: 'Review', schema: REVIEW_SCHEMA, effort: 'high' }
  ).then(r => ({ ...(r || { id: it.id, verdict: 'REVISE', required_fixes: ['review agent returned null'] }), _draft: draft })),
  // Stage 3: revise in place if not CLEAN
  (review, it) => {
    if (review && review.verdict === 'CLEAN') {
      return { id: it.id, revised: false, verdict: 'CLEAN', draft: review._draft }
    }
    const fixes = review ? JSON.stringify({
      overclaim_flags: review.overclaim_flags, number_inconsistencies: review.number_inconsistencies,
      rigor_gaps: review.rigor_gaps, required_fixes: review.required_fixes,
      mechanism_chain_valid: review.mechanism_chain_valid, distinctness_ok: review.distinctness_ok,
    }, null, 2) : '(no review; do a self-review against the hard rules)'
    return agent(
      `${SHARED}

Revise the proposal at ${DIR}/${it.slug}/README.md IN PLACE to fix the reviewer findings below. Read the current
file first. Apply every required fix, remove any HARD flag (especially clinical overclaims not hedged to reach-tier,
any number that drifts from _shared_facts.md, and any em-dash). Preserve the original one-line question verbatim if
this is an augmented item (01-07). Keep the plain-language voice. Save the file. Return the draft summary.

REVIEWER FINDINGS TO FIX:
${fixes}`,
      { label: `revise:${it.id}`, phase: 'Revise', schema: DRAFT_SCHEMA, effort: 'high' }
    ).then(d => ({ id: it.id, revised: true, verdict: review ? review.verdict : 'REVISE', draft: d, review }))
  }
)

const finalById = {}
for (const r of results.filter(Boolean)) finalById[r.id] = r
log(`Draft/Review/Revise complete for ${Object.keys(finalById).length}/${ITEMS.length} items`)

// ---- Phase 4: Scorecard synthesis (single agent, sees all 12 at once) ----
phase('Scorecard')
const rollup = ITEMS.map(it => {
  const c = contractById[it.id] || {}
  const r = finalById[it.id] || {}
  return {
    id: it.id, slug: it.slug, type: it.type, title: it.title || it.slug,
    neuro_chain: c.neuro_chain, generalizable_claim: c.generalizable_claim,
    external_cohort: c.external_cohort, feasibility_tier: c.feasibility_tier,
    final_verdict: r.verdict, revised: r.revised,
  }
}).filter(Boolean)

const scorecard = await agent(
  `${SHARED}

You are synthesizing the graded SCORECARD for all 12 portfolio items at once (so the ranking uses ONE scale).
Read every ${DIR}/NN-*/README.md (01 through 12), plus _shared_facts.md and _neuro_facts.md. For each item apply
this rubric:

ICLR/ICML likelihood: score five axes 1-5 (Novelty, Mechanism-grounding, Generalizability-of-claim,
Rigor/evaluation-validity, Feasibility-given-ambition). Composite = 0.20*Novelty + 0.20*Mechanism +
0.25*Generalizability + 0.25*Rigor + 0.10*Feasibility, on a 1.0-5.0 scale. Map to a verbal band:
>=4.3 Strong main-track candidate; 3.6-4.2 Competitive with revision; 2.8-3.5 Workshop/borderline;
<2.8 Not yet conference-level. Remember: Generalizability can reach 5 for the PD-variability and cross-view claims
(real external cohorts) but stays capped for CP/myopathy-specific claims (no public skeleton cohort). Do NOT let
effort bleed into likelihood.
Effort level: a 3-field tuple: (core weeks / +reach weeks); retrain scale in {zero-retrain, test-time pass,
fold-local finetune, full curriculum retrain, from-scratch normal-only}; data/compute needs in {reuses cached
528-token tensors, needs NPZ+fps regen, needs new architecture, needs external cohort}.

WRITE two things:
1. ${RESEARCH_DIR}/SCORECARD.md : a markdown file with (a) a short intro paragraph stating the rubric and the unifying
   symmetry-axis thesis; (b) ONE table with columns: # | slug | ICLR/ICML band | composite | Nov | Mech | Gen | Rig |
   Feas | effort (weeks) | retrain | data; rows sorted by composite descending; (c) one short paragraph per item
   giving the scientific-quality summary (what a positive result confirms, what an informative null overturns) and
   the effort rationale. No em-dashes. Every number consistent with _shared_facts.md.
2. Return the structured scorecard so the parent can patch _selection.json.

Here is the rollup of contracts and final verdicts to reference (do not blindly trust it; verify against the actual
README files you read):
${JSON.stringify(rollup, null, 2)}`,
  {
    label: 'scorecard', phase: 'Scorecard', effort: 'high',
    schema: {
      type: 'object', additionalProperties: false,
      required: ['scorecard_path', 'items'],
      properties: {
        scorecard_path: { type: 'string' },
        items: {
          type: 'array',
          items: {
            type: 'object', additionalProperties: false,
            required: ['id', 'slug', 'band', 'composite', 'axes', 'effort'],
            properties: {
              id: { type: 'string' }, slug: { type: 'string' }, band: { type: 'string' },
              composite: { type: 'number' },
              axes: {
                type: 'object', additionalProperties: false,
                required: ['novelty', 'mechanism', 'generalizability', 'rigor', 'feasibility'],
                properties: {
                  novelty: { type: 'number' }, mechanism: { type: 'number' },
                  generalizability: { type: 'number' }, rigor: { type: 'number' }, feasibility: { type: 'number' },
                },
              },
              effort: { type: 'string' },
              quality_summary: { type: 'string' },
            },
          },
        },
      },
    },
  }
)

log(`Scorecard written: ${scorecard ? scorecard.scorecard_path : 'FAILED'}`)

return {
  contracts: contractById,
  results: rollup,
  scorecard,
}
