# Codex Adversarial Review

Target: working tree diff
Verdict: needs-attention

Block on Figure 4’s misleading evaluation-reference access. Visually inspected all five PNGs, including at 900px width: no material overflow, illegibility, clutter or arrow intersections. README local links resolve; no fabricated results or restoration/forecasting conflation found.

Findings:
- [medium] Remove evaluation references from the model-input path (experiments/sjepa/gavd6/docs/studies/synthetic-training-v2/proposal/images/04-fair-comparison.svg:8-12)
  Figure 4 puts “Same evaluation references” inside “Same allowed inputs” and routes that box into all three training arms. This depicts evaluation labels as permissible model inputs, contradicting protocol.md:9 and its protected-label boundary at line 43. The concern is diagram semantics, not evidence of an implemented leak: following this dataflow could contaminate fitting and invalidate the comparison.
  Recommendation: Move evaluation references to a separate scoring-only input feeding Common evaluation. Label clean-label access as training-only supervision. Update generate_figures.py and regenerate the SVG and PNG.

Next steps:
- Correct Figure 4’s information boundary and visually recheck its regenerated preview.
