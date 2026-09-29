# Figure specification

All figures are built at the manuscript's 5.5-inch text width using explicit Matplotlib layouts. White backgrounds; STIX serif text and math; 8–9 pt labels; 9–10 pt panel headings; 0.6–0.9 pt axes; 1.0–1.3 pt interval strokes; 3.5–4.5 pt markers. PDF text and line art remain vector with embedded TrueType fonts; SVG preserves editable text. PNG and grayscale PNG are inspection outputs.

Semantic colors: jointly trained direct coordinate model blue (#0072B2), frozen endpoint-feature model purple (#7B5AA6), frozen delta-feature model vermilion (#D55E00), other controls neutral gray. Supervision is additionally encoded with circle (coordinate), open triangle (original change package), diamond (low scalar), square (dense). Reference lines are neutral and labeled. Anatomical left/right are expressed by subscripts L/R, never by method colors.

1. **Experimental workflow:** 5.5 by3.15inches. Original, hand-designed12-keypoint glyphs and stacked frame outlines illustrate the controlled original/edited motion windows, with the knee edit highlighted in green. They are schematic, not data or reconstructed predictions. The training panel uses112people, exposes projected-reference teacher targets, separates endpoint OR delta objectives, and freezes the encoder before readout fitting. The development panel uses14different people and3seeds; each window is restored independently, and reference poses enter only scoring. The outcomes are asymmetry-change error, knee-trajectory error, and posthoc anatomical assignment. Detailed EMA, score transforms, and query-support definitions remain in the methods; the caption is deliberately short.
2. **Restoration:** aligned response and waveform axes; identical method ordering; direct and frozen groups visually separated. Zero-response reference appears only on the response axis. Uncertainty/participant evidence uses the 14 people and paired seeds, never windows as independent observations.
3. **Reliability:** separate panels for failure probability, additive score contributions, and paired response effects across fixed penalty costs. Distinct labels and units prevent conditional successful-output error from being mistaken for unconditional contribution.
4. **Repair:** original, low-scalar, and dense readouts shown as aligned estimates with participant-level uncertainty; direct-coordinate reference explicitly identified as jointly trained. Paired dense-minus-low effect has its own interval and primary/secondary labels.
5. **Naming:** categorical small multiples or grouped points without connecting lines; all three naming conditions; full 0–100% failure scale; no chance baseline. Wrong/ambiguous/missing combined unless separately recoverable.

Prototype figures 1 and 2 are inspected in the actual official manuscript template before extending the style. Final QA inspects every figure in color and grayscale and every rendered manuscript page. User-specified precedents and substantive differences from v07 are recorded in reviews/visual-precedents.md and REVISION-RECORD.md.

## Participant figure added in the framing revision

Figure 8 (originally Figure 6) appears in the appendix at the full 5.5-inch column width. Its three
aligned panels show every one of the 14 people in fixed canonical order, with
three seed points and a mean per person. The comparisons are unchanged-minus-
direct pooled response error and zero-minus-direct response error under clear
and occluded observations. All original costs and weighting are retained.
Observed seed ranges are explicitly distinguished from confidence intervals;
x-axis ranges differ and are labeled. Source IDs, values and provenance are
retained in evidence/case-figure-*.csv/json; build_case_figure.py regenerates the
vector artwork or plots the audited CSV using --from-audit. These profiles
illustrate lower and higher errors, with no invented pose or clinical examples.

## Comparison simplification figures

Figure 6, `protocol`, summarizes the separate fitting/development people, fixed
endpoint shape and token slots, reference-defined support, and aggregation order
at 5.5 by 2.25 inches. Figure 7, `study-questions`, replaces the 16-row inventory
with the three saved primary paired gains at 5.5 by 2.8 inches. It separates the
repair waveform outcome/person-t interval from the first two response/crossed
intervals and supplies no pooled estimate. Exact selections and hashes are in
`evidence/summary-figure-provenance.json`. `build_summary_figures.py --from-audit`
rebuilds both from that portable record. Their labels remain explicit in
grayscale, and both PDFs embed STIX fonts. The six earlier figure sets and all
existing numerical tables are unchanged.
