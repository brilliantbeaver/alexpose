# Final independent adversarial review

**28 September 2026.** Reviewed the revised PLAN.md, review disposition, FIGURES.md, plotting source, retained numerical exports, and rendered figure images. No manuscript, plan, or builder edits were made by this reviewer.

**Verdict: no unresolved blocker.** All four prior P1 research-design objections and both final minor corrections are resolved. No blocker was found in the empirical figure selection, contrast signs, populations, uncertainty procedures, or central scientific framing. Final assembled-document layout was handled in the producing agent's separate QA pass; the scope of this independent review is recorded below.

## Prior findings: confirmed resolutions

- **P1.1, paired protocol:** Section 8 now distinguishes a locked-person 2D confirmation from a subsequent 3D study. Same-motion observation corruption and genuine within-person movement contrasts are separate. The proposed per-leg preservation endpoint, reference conventions, pairing, support, and testing hierarchy make the new outcome explicit.
- **P1.2, causal leakage:** Section 7.1 applies prefix-only access to all preprocessing and tools and requires an input/forecast truncation check. Full-window restoration remains a separate task.
- **P1.3, adaptive test reuse:** Section 8 separates fitting, reward/threshold selection, and final evaluation, with fresh confirmation or an untouched final cohort after adaptive changes.
- **P1.4, object generation:** Section 7.7 distinguishes stationary-chair contact conditioning from joint human/rigid-box generation, with object-persistence and uncoupled baselines.
- **P2, restoration and inference:** Section 4.4 and Figure 1 state that development windows are restored separately. Figure code imports retained interval fields without bootstrap resampling or new interval estimation.

## Independent numerical and provenance audit

An independent standard-library audit, without importing the plotting builder, checked **all 248 exported empirical marks** against their source records. This includes 80 rows containing confidence limits or observed seed ranges. All estimates and limits matched within numerical tolerance; all 14 recorded input hashes matched when checked. The machine-readable record is [final-numerical-audit.json](final-numerical-audit.json).

Checked counts: Figure 2, 41 marks; Figure 3, 3; Figure 4, 13; Figure 5, 14; Figure 6, 9; Figure 7, 168. The repeated direct reference in Figure 5 correctly uses the ViTPose population rather than its pooled mean. Figures 2/4 use the stated comparator-minus-candidate signs; Figure 5 uses the first-named candidate's gain. Figure 6 correctly converts proportions and interval limits to percentages. Figure 7's 42 participant summaries and 126 seed points match the exported comparator-minus-direct differences.

Where source records contain population fields, they retain 14 people and three seeds. The repair plots select the person-t fields; core, response and naming plots select crossed-bootstrap fields. Figure 3 preserves the different outcomes and interval methods. These checks verify faithful reuse of the retained packet; they do not recompute the original model outputs or independently establish the bootstrap's validity.

## Visual and caption inspection

Inspected all eight figures in color and grayscale contact sheets, plus enlarged Figures 4–6. Figure 2 keeps all sixteen configurations and limits the zero-response line to the scalar panel. Figure 3 does not pool its three effects. Figure 4 distinguishes successful-error contributions over all cases from conditional success-only error and retains all four costs. Figure 6 uses the full percentage axis without a chance line and identifies combined wrong/ambiguous/missing outcomes. Figure 7 retains every participant in a fixed order and labels seed ranges as non-inferential. Method labels, shapes, or hatching preserve meaning in grayscale. The two diagrams identify conceptual or proposed content and do not invent reconstructed participant poses.

No substantive misleading encoding was found. The already-identified Figure 7 heading spacing and Figure 8 text-layout work were excluded from this pass, as requested. This was not a pixel-by-pixel audit of every individual PDF/SVG or the final document atlas; font embedding, page breaks and final-size readability require the separate production QA record.

## Minor final corrections

1. **Section 8, primary preservation outcome — resolved:** the revised sentence now identifies cancellation of equal per-leg errors. For an asymmetry error of e_R − e_L, equal/shared errors can cancel; opposite-signed errors add. The final wording was rechecked. This does not change the proposed per-leg endpoint, which is appropriate for its stated purpose.
2. **Rebuild command:** the original bare `python` command failed in this workspace because inherited pyenv selection requested an unavailable Python 3.10. FIGURES.md now names the actual Python 3.12.9 environment; its version was independently confirmed. This operational issue is resolved.

No proposed training, physics validation, generation, agent policy, or clinical assessment was executed or certified by this review. The deliverable is a defensible evidence-led writing/research plan with faithful regenerated figures, not independent confirmation of the underlying experiments.

**Closeout:** the producing agent reports completing the Figure 7/8 layout fixes, inspection of all eight atlas pages, native MathML conversion, local-link checks, and eight figure alt-text checks. Those are production QA results reported to this reviewer, not additional independently repeated checks. No further review gate is requested.
