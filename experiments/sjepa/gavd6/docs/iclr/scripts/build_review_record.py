"""Assemble immutable review rounds and fixed-rubric synthesis into one record."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
d=json.loads((ROOT/'reviews/scores.json').read_text());weights=d['weights']
changes={
1:'Initial nine-page scientific argument built from audited exports; reviewers identify the erroneous mirror/sign implication, compound-loss attribution, incomplete closest-work positioning, and figure collisions.',
2:'Separate idealized exchange of measurements from executed 3D mirroring; correct scalar-plus-geometry attribution and equal-pair dense reduction; define geometric assignment. Residual diagram overflow and missing close-work positioning remain.',
3:'Add S-JEPA, recent skeleton feature prediction, PoseBERT and MotionBERT distinctions; source primary intervals directly from hashed JSON; make laterality explicitly post hoc; retain direct-model per-leg counterexample; repair diagram lanes.',
4:'Specify observation-only normalization, regularizer weights, optimizer schedule, calibration, support and 4:1/2:1 condition weights; distinguish single-state inference from paired training. All main-text numerical/method audits pass, with research limits unchanged.',
5:'Foreground laterality with the audited naming-condition figure and exploratory delta-minus-endpoint interval; retain repair numbers and primary interval; remove weaker training-population probe discussion from main text but preserve it in evidence; disclose rendered person boxes and camera setup.',
6:'Align abstract and introduction with the primary response test and subsequent naming audit; state the shared feature coefficient and exact query/pair reductions; qualify headings. New audit identifies the calibration-gradient asymmetry for explicit final reporting.',
7:'Clarify primary versus secondary readouts, define EMA and D, make primary-contrast caption self-contained, confine the 93% repair ratio to its actual ViTPose setting, and disclose unequal initial auxiliary-gradient influence. Final pass retains all experiment-level limits.'}
lines=['# ICLR manuscript review record','',
'All seven versions keep the requested title. Each round received an independent skeptical representation/world-model review, an evidence audit, a statistical review, and a scope/natural-writing edit. Three independent agents supplied these four perspectives; the evidence/statistics agent recorded them separately. This is adversarial AI review, not human peer review or a conference acceptance prediction.','',
'The rubric is fixed: relevance/contribution20%, accuracy/evidence20%, evaluation/statistics15%, insight/positioning15%, reproducibility10%, clarity10%, figures5%, submission fit5%. Each dimension is scored0–10. A 5 denotes a substantial unresolved weakness; a 10 would mean no material weakness within scope. Weighted total = sum(weight × dimension score /10). The synthesis averages the three panel scores per dimension; it does not select the most favorable reviewer.','',
'## Fixed-rubric comparison','',
'|Version|Relevance20%|Accuracy20%|Evaluation15%|Insight15%|Reproducibility10%|Clarity10%|Figures5%|Fit5%|Total/100|','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for r in d['versions']:lines.append('|v%02d|'%r['version']+'|'.join(f'{x:.2f}' for x in r['dimension_scores'])+f"|{r['weighted_total']:.2f}|")
lines+=['','Per-dimension reasons and remaining weaknesses are retained verbatim in each independent review below. Reviewer disagreement is visible rather than averaged away in the commentary. Later versions can improve precision or presentation without crossing a scoring band; plateaued scores are intentional. The evidence/statistical reviewer keeps evaluation at 5/10 throughout because the underlying small, reused development cohort does not improve through revision.','',
'## Revision decisions','']
for v in range(1,8):lines+= [f'### Version {v:02d}', '',changes[v],'']
lines+=['## Final disposition of objection families','',
'|Objection family|Severity|Evidence|Correction/disposition|Residual limitation|',
'|---|---|---|---|---|',
'|Physical mirror falsely treated as exact projected sign reversal|Major/material|3D reflection plus fixed-camera projection; empirical original/mirror counterexample in evidence review|v02 onward makes Figure 1 an algebraic side-value exchange; actual mirrors are reprojected and remeasured|No direct learned-equivariance validation|',
'|Scalar response treated as changed-side identity|Major|Signed bilateral excursion can cancel bilateral changes; assignment has separate geometric denominator|Operational definition and independent assignment diagnostic; post hoc result explicitly labeled|No anatomical or clinical affected-side classifier|',
'|Compound change objective attributed solely to scalar term|Moderate/material|Core adds scalar and geometry together|Correct abstract/headings; low-scalar repair retains geometry|No complete factorial geometry/scalar or convergence sweep|',
'|Uncertain means promoted to advantage, preservation, or equivalence|Major|All three primary intervals span zero; interaction uncertain|Retain primary comparators/estimands, point-estimate wording, zero-response control and no noninferiority claim|More independent people/seeds and prespecified margins needed|',
'|Failures concealed by successful-only error or causal decomposition|Major|720° response cost and180° waveform/per-leg costs|Common-weight additive accounting, precise successful-contribution definition; 74% is score accounting|Reliability and transfer remain unvalidated|',
'|Post hoc naming result promoted to primary or naive stratum averaging|Major|Reused completed exports; endpoint4:1 vs response2:1 weighting|Explicit exploratory origin, correct weights and paired interval direction|Unadjusted, selected development comparisons need confirmation|',
'|Direct model described as universally best or controlled pretraining ablation|Major|Per-leg excursion counterexample; direct encoder adapts with more coordinate exposure|Report named outcomes and practical-comparison boundary|No matched tuning, architecture or supervision-budget isolation|',
'|Shared feature coefficient mistaken for matched gradient influence|Moderate, scientifically important|Initial endpoint10%base vs weighted delta≈0.0013%base|v07 explicitly reports asymmetry and initial-only interpretation|Optimization mechanism and later influence remain unestablished|',
'|Missing exact loss/support/normalization/calibration details|Moderate|Production code and receipts|Main formulas and supporting reproduction contract; separate teacher/reference/inference paths|Compact packet cannot rerun raw training without absent assets|',
'|Missing closest work or implying general JEPA/world-model novelty|Major|S-JEPA/GFP/PoseBERT/MotionBERT precedents|Target/task/deployment distinctions; narrow empirical evaluation contribution|External learned baselines and generalization are still missing|',
'|Figure overlaps, imprecise interval values, incomplete captions|Moderate|Initial rendered figures and hardcoded core interval|New versioned assets, exact JSON reads/hashes, readable naming plot, full comparators/units/populations|Raw reconstructed example unavailable locally|',
'|Incomplete submission rendering/citations|Moderate/minor|Initial v01 render predates three bibliography entries; later builds resolve them|Final typeset companions use verified entries and Times-family fonts, official style, 9 main pages and AI disclosure; initial preflight renders remain preserved|Human verification, abstract eligibility, metadata and submission remain author tasks|',
'',
'## Limitations requiring new evidence','',
'Fourteen repeatedly used development people, three seeds, two source collections, missing protected-person confirmation, no natural-video anatomical reference, no external learned-refiner comparison, no broad weight/budget/convergence study, and no new latent-pair randomization control remain unresolved research limitations. The frozen teacher/student and direct/frozen comparisons do not isolate a unique causal mechanism. Revision bounds these claims; it does not solve the experimental limitations.','',
'## Typesetting preflight correction','', 'The reviewed source/PDF files without the -final suffix remain untouched. The seven delivered -final companions add T1 font encoding so Tectonic embeds the Times-family fonts required by the official template, suppress a title-word hyphenation, complete the shared bibliography, and compact the Table 3 caption/placement without changing its scientific meaning. This resolved a one-page table float overflow after the font correction. Each delivered companion retains nine main pages. The original style files are unchanged. Final file hashes and font/page checks are recorded in qa/final-validation.json; scientific-review scores are not retrospectively inflated by this packaging correction.','', '## Independent review records','',
'Each complete record below preserves the reviewer’s substantive IDs, severity, evidence, correction or reasoned disposition, and residual limits. Initial source audits are in `reviews/evidence-initial.md`, `reviews/methods-initial.md`, and `reviews/literature-initial.md`.','']
for v in range(1,8):
 for role in ['world-models','evidence-statistics','editor']:
  p=ROOT/f'reviews/v{v:02d}-{role}.md'
  if not p.exists():raise SystemExit(f'Missing final review {p}')
  lines+=['---','',f'### v{v:02d} — {role}','',f'Source record: [{p.name}](reviews/{p.name}).','',p.read_text().strip().replace('](../evidence/', '](evidence/'),'']
(ROOT/'REVIEW-RECORD.md').write_text('\n'.join(lines)+'\n')
print('Wrote integrated review record with',len(d['versions']),'rubric rounds and21 independent panel records.')
