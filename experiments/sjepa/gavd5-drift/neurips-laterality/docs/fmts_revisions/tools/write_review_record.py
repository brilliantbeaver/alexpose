"""Append/update only the new FMTS section, preserving the prior README bytes."""
from pathlib import Path
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]
DIMENSIONS=["Temporal relevance", "Claim accuracy", "Evaluation/statistics", "Insight/positioning", "Reproducibility", "Clarity/narrative", "Figures", "Submission fit"]
WEIGHTS=[20,20,15,15,10,10,5,5]
CARDS=[
 ("Original paper", [4,3,5.5,5,4.5,5.5,6,2], [
  "The gait setting is temporal, but the symmetry/world-model verdict exceeds a whole-clip observable test. Reframe the contribution around observable recovery.",
  "The fully powered null, only-route claim, augmentation verdict and oracle localization overreach the evidence. Keep separate recipes and inferential limits.",
  "Source folds, matched starts and conditional bootstrap are strengths. Failed superiority is not equivalence; the development cohort and participant identities remain unresolved.",
  "Constructed parity is mathematically informative, but it displaces the more revealing comparison between predictive matching and movement readout.",
  "Protocol paths and code are useful; missing raw artifacts and an obsolete recipe for the proposed central grid prevent a reproducible latest-results account.",
  "The exposition explains terminology, but broad verdicts and execution history interrupt the empirical story. Rewrite around the specific measured contrast.",
  "Vector illustrations help, but they need a clearer time boundary, target branch and actual full-view training recipe at workshop size.",
  "No four-page FMTS main-text artifact; working metadata, local operational detail and submission/governance history need separation from anonymous files." ]),
 ("Physical World AI V8", [6.5,8,7,7.5,6.5,7,7,3], [
  "A sound observed-system framing is present, but the temporal-order limit and completion-versus-forecasting distinction deserve earlier, sharper treatment.",
  "Most claims are carefully bounded and current settings are recoverable. The new revision must distinguish saved raw-data audits from present artifact access.",
  "Matched starts, source weighting and nested readout selection are clear. Put all paired deficit intervals in the main text and explain coupled summary/capacity changes.",
  "The loss-versus-observable gap is valuable. Related work and the future test need a focused temporal evaluation synthesis rather than broader physical-world coverage.",
  "Detailed methods and retained audits help. Raw artifacts are absent here, and shared asset paths could let later edits change earlier versions.",
  "Generally careful prose is burdened by repeated limits and lengthy supporting material. Reduce the claim count and foreground the decisive comparison.",
  "Pipeline and results figures are useful; short arrow tails, small labels and training/evaluation boundaries need a dedicated FMTS redesign.",
  "Its eight main pages and eleven total pages exceed FMTS's main-text limit. Official 2026 workshop formatting and a much shorter appendix are required." ]),
 ("FMTS V1", [8,8,7,7.5,6.5,7.5,6.5,8.5], [
  "The paper now evaluates a concrete movement observable within a temporal system. Temporal-direction sensitivity and the limits of correspondence still need sharper placement.",
  "Obsolete recipes and general symmetry verdicts are removed. The headline is supported; the detailed correspondence denominator and all five paired intervals need prominence.",
  "Source weighting, matched starts and train-only regression selection are explicit. One main-text interval leaves the central comparison less assessable than it could be.",
  "The latent-matching/movement distinction drives the paper. Related Work remains somewhat descriptive; static and support-based explanations need stronger integration.",
  "Each version has private sources, figures and aggregate records. Raw predictions/checkpoints and complete acquisition provenance remain unavailable.",
  "A four-page motivation-to-result arc replaces the longer draft. The opening is generic and the decisive uncertainty is unnecessarily in the appendix.",
  "Schematic poses and separate target/training paths are useful, but labels near 6.7 pt and an omitted projector in the graphic need correction.",
  "Official anonymous four-page main text compiles. Two supporting sections, working Markdown references and small figure labels still reduce usability." ]),
 ("FMTS V2", [8,8.5,7.5,7.5,6.5,7.5,6.5,8.5], [
  "The gait evaluation remains focused. The speed contrast still needs a direct explanation of why its recovery is insufficient to validate temporal dynamics.",
  "All five paired contrasts and the 125-model/25-initial-control diagnostic counts are now explicit. Broader contamination and development claims need tighter qualification.",
  "The full main-text interval table makes the primary inference assessable. Conditional selection/split uncertainty and hash-versus-deduplication distinctions still need clearer wording.",
  "The decisive comparison is stronger on the page, but the same unisolated readout and nuisance explanations remain. No new empirical evidence is added.",
  "Private assets and aggregate checks are retained. Missing raw rows still prevent bootstraps or inference from being independently repeated here.",
  "The abstract names the movement estimate and source cross-validation more clearly. Repeated explanatory headings and generic related-work prose remain.",
  "Main-text table improves uncertainty display; pipeline label size and projector omission are unchanged, so figure score does not rise.",
  "Four-page official anonymous layout remains intact with an ordinary-size table. Supporting material and Markdown citation presentation still need refinement." ]),
 ("FMTS V3", [8,9,7.5,7.5,6.5,8,6.5,8.5], [
  "Source holdout is now distinguished from chronology and person independence. The target's temporal-direction weakness remains the next substantive positioning issue.",
  "Archive integrity is separated from content/person deduplication; registered reflection tests are separated from post-inspection extensions. Material scope claims are now bounded.",
  "Omitted retraining, split and selection uncertainty is explicit. This improves interpretation, while the underlying cohort/control limitations keep the score unchanged.",
  "Possible nuisance and capacity explanations are visible but not tested. A synthesized account of task-dependent representation quality would strengthen the contribution.",
  "The provenance narrative is more honest about missing records. It cannot replace raw predictions, acquisition versions or participant IDs.",
  "Cleaner inferential boundaries and clearer registered/exploratory distinctions improve readability. Temporal and related-work explanations still need consolidation.",
  "Plots and vector pipeline remain as in V2, including small labels and the projector omission. No cosmetic chronology credit is given.",
  "Four main pages, anonymous style and consistent PDF references pass. The appendix and Markdown references are still less polished than the eventual final version." ]),
 ("FMTS V4", [8.5,9,7.5,8,6.5,8,6.5,8.5], [
  "Time-reversal invariance and the necessary-but-limited observable-recovery test now anchor the temporal contribution. Genuine dynamics and forecasting still require experiments.",
  "Correct-clip preference is distinguished from comparing losses across changing teachers. Coupled readout changes are stated without an unsupported order-learning claim.",
  "The inference remains conditional and source paired. No new ablation, penalty search, independent cohort or multiplicity adjustment has been added.",
  "Related Work synthesizes which variation prediction retains and which behaviors downstream evaluation tests. It remains conceptual rather than a benchmark comparison.",
  "Private inputs and reproducible typesetting persist. Aggregate-only access and incomplete acquisition/person records remain material limitations.",
  "The scientific argument is more direct; compressed labels and repetitive caveats still add friction, motivating the design and editorial passes.",
  "The figure's full-view projector omission and minimum label size remain unresolved here, despite improved surrounding explanation.",
  "Four-page main text and anonymous sources meet the structural requirement. Appendix length, figure usability and Markdown reference completeness still need work." ]),
 ("FMTS V5", [8.5,9,7.5,8,6.5,8.5,8.5,8.5], [
  "The redesigned time axis, index-resizing label and two-sided context make the temporal boundary easier to see. No future-gait prediction is visually implied.",
  "The graphic now includes the full-view projector and distinguishes target supervision from encoder training. Evidence support otherwise remains unchanged.",
  "Seed dots, means and repeated diagnostic checks are labeled by their actual units. This improves communication, not the underlying statistical design.",
  "The figure makes the two evaluations easier to compare. A causal explanation for their disagreement remains unresolved and is not inferred from the visual design.",
  "Private vector generators reproduce each version's own assets. They do not restore raw training or source-bootstrap artifacts.",
  "Shorter labels and clearer branches improve the main narrative. Appendix execution detail and the working Markdown reference section still interrupt the reading.",
  "About 8.1-pt minimum labels, continuous arrow approaches, a separate y branch and the regularizer projector correct the material V4 diagram weaknesses.",
  "Main text remains four pages after redesign. Anonymous citation presentation and supporting-section economy still benefit from revision." ]),
 ("FMTS V6", [8.5,9,7.5,8,7,8.5,8.5,9], [
  "Suitability for the chosen gait measurement is stated without demanding preservation of every possible observable. The temporal empirical scope is unchanged.",
  "Reflection augmentation and explicit reflection loss retain separate recipes and evidence types even after execution chronology is removed.",
  "Conditional paired intervals and alternative readout explanations remain clear. New statistical evidence is still absent, so rigor does not rise with the version number.",
  "The task-dependent evaluation synthesis remains strong. Additional gait observables, order controls and future-motion tests would be needed for broader insight.",
  "Complete numbered Markdown references, private BibTeX and an explicit aggregate-only availability statement make the deliverable easier to inspect and rebuild.",
  "Local execution history is removed from manuscript prose; repeated caveats and audit-like phrasing still warrant a final editorial pass.",
  "The V5 vector design is retained unchanged. Its print-size legibility and accurate branching remain strengths, without further score inflation.",
  "Official four-page PDF, complete Markdown references, anonymous packages and disclosure are in place. Two supporting sections remain longer than necessary." ]),
 ("FMTS V7", [8.5,9,7.5,8,7,8.5,8.5,9], [
  "The measured temporal-neighbor mask effect is now explicit. This clarifies context availability but does not add a new temporal capability.",
  "Precision is reconciled across prose, table and plots; reflection error's fixed joint/channel action is clearer. No new scientific-evidence credit is awarded for rounding.",
  "MAE increases have defensible magnitudes, and signed tiny mask effects retain uncertainty. Source intervals remain fixed-fit and exploratory.",
  "The same empirical evaluation lesson and unresolved capacity/nuisance controls remain. Numerical polish does not change the scientific evidence.",
  "Full-precision aggregates and private build assets remain available; the paper now has one supporting section. Missing primary artifacts are unchanged.",
  "Consolidation reduces headings, but explanatory rhythm and repeated qualifications remain formulaic enough to justify a fluent rewrite.",
  "The V5 design remains accurate and readable. Further anatomical identity and encoder-role labeling is reserved for V8.",
  "Four main pages with a single appendix section; seven pages total. Reduced headings improve organization but do not materially change the overall score." ]),
 ("FMTS V8", [8.5,9,7.5,8,7,9,9,9.5], [
  "The abstract and introduction lead with an observed gait-speed contrast and its relationship to predictive features. A future-only test is concise and explicitly proposed.",
  "Final claims trace to the matched grid or labeled summary/synthetic evidence. No unresolved material misstatement was found; raw-row verification remains unavailable.",
  "All five paired effects, source conditioning, readout selection and possible nuisance/capacity explanations are explicit. New data and ablations remain necessary for stronger inference.",
  "A coherent evaluation contribution connects latent matching with a specified observable. Mechanism, temporal order, broad transfer and true forecasting remain untested.",
  "Eight isolated source packages reproduce their PDFs; private numerical verifiers reproduce seed means and differences. Raw inference and source intervals remain unreproducible from aggregates alone.",
  "Connected prose replaces repeated audit statements; the actual measurement appears first, terms are explained and the appendix is one page. Some technical density is unavoidable.",
  "Explicit anatomical identities, left/right colors, missing-landmark key and frozen encoder label complete the print-size diagram; teacher readout values and repeated diagnostic units are clear.",
  "Four main pages, one reference page and one appendix page in the official anonymous style, with verified citations and AI disclosure. Actual submission still depends on the project's unresolved governance determinations." ])
]


def build():
    cards=[]
    for name, scores, reasons in CARDS:
        cards.append({"version":name,"scores":dict(zip(DIMENSIONS,scores)),"weighted_total":sum(w*s/10 for w,s in zip(WEIGHTS,scores)),"reasons":dict(zip(DIMENSIONS,reasons))})
    (ROOT/'review/scores.json').write_text(json.dumps({"weights_percent":dict(zip(DIMENSIONS,WEIGHTS)),"cards":cards},indent=2)+'\n')
    header='| Manuscript | Temporal | Claims | Statistics | Insight | Repro. | Clarity | Figures | Fit | /100 |\n|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|\n'
    header+='\n'.join('| '+c['version']+' | '+' | '.join(f'{s:g}' for s in c['scores'].values())+f" | **{c['weighted_total']:.2f}** |" for c in cards)
    header+='\n\nV6 and V7 intentionally receive the same total. V7 improves precision and organization without removing an additional material scientific weakness. V8 gains for clearer prose, diagram labeling and a shorter appendix; its scientific-evidence scores remain unchanged.\n'
    for c in cards:
        header+='\n#### '+c['version']+f" — {c['weighted_total']:.2f}/100\n\n| Dimension | Score | Concrete reason, remaining weakness or next correction |\n|:--|--:|:--|\n"
        header+='\n'.join(f'| {d} | {c["scores"][d]:g} | {c["reasons"][d]} |' for d in DIMENSIONS)+'\n'
    reviews=[]
    for v in range(1,9):
        path=ROOT/f'review/review_v{v}.md' if v<8 else ROOT/'review/review_v8_final.md'
        value=path.read_text()
        value=value.replace('# FMTS', '#### FMTS',1)
        reviews.append(value)
    validation=(ROOT/'review/validation_summary.md').read_text()
    text=(ROOT/'review/record_template.md').read_text().replace('<!-- SCORECARDS -->',header).replace('<!-- REVISION_REVIEWS -->','\n\n'.join(reviews)).replace('<!-- FINAL_VALIDATION -->',validation)
    anchor='<a id="fmts-2026-review"></a>'
    readme=ROOT.parent/'README.md'
    before=readme.read_bytes()
    if anchor.encode() in before:
        before=before.split(anchor.encode(),1)[0]
        state=json.loads((ROOT/'review/readme_preservation.json').read_text())
        original=before[:state['original_bytes']]
        assert hashlib.sha256(original).hexdigest()==state['original_sha256']
        before=original
    else:
        (ROOT/'review/readme_preservation.json').write_text(json.dumps({'original_bytes':len(before),'original_sha256':hashlib.sha256(before).hexdigest()},indent=2)+'\n')
    readme.write_bytes(before+b'\n\n'+anchor.encode()+b'\n\n'+text.rstrip().encode()+b'\n')
    state=json.loads((ROOT/'review/readme_preservation.json').read_text())
    assert hashlib.sha256(readme.read_bytes()[:state['original_bytes']]).hexdigest()==state['original_sha256']
    print(json.dumps({c['version']:c['weighted_total'] for c in cards},indent=2))


if __name__=='__main__':build()
