# Codex Adversarial Review

Target: working tree diff
Verdict: needs-attention

Two scientific figure errors should be corrected before shipping. Retained numerical results match the evidence; slides.md remains unchanged.

Findings:
- [medium] Phase alignment artificially introduces asymmetry (experiments/multiple-sclerosis/scripts/scripts_make_progress_diagrams.py:91-94)
  The initial curves both have amplitude 74 and differ only by half a cycle. The supposedly aligned right curve then has amplitude 41. Correct phase alignment would make these curves coincide. Figure 1 therefore manufactures the difference it purports to reveal, teaching an incorrect relationship between phase alignment and asymmetry. Calling the curves schematic does not resolve this mathematical inconsistency.
  Recommendation: Preserve each curve’s amplitude across alignment. Either show coincident aligned curves, introduce the amplitude difference in both panels, or clearly separate the asymmetric example. Regenerate 01_symmetry_cycle.svg.
- [medium] Control plot misidentifies the measured quantities (experiments/multiple-sclerosis/scripts/scripts_make_progress_diagrams.py:369)
  The plotted 'Duration / acquisition' input is actually retained-frame count supplied twice, and 'Body proportions' consists of nine unnormalized pixel distances from median landmark positions (scripts_phase0_provenance.py:158–173). Neither measures calibrated duration, frame rate/resolution, or scale-free proportions. Figure 11 thus reinstates interpretations that the report explicitly rejects, overstating which nuisance variables were tested.
  Recommendation: Label these rows 'Retained-frame count' and 'Median-pose pixel distances'; clarify their scope in the figure description and regenerate 20_shortcut_controls.svg.

Next steps:
- Correct the figure generator, regenerate the affected SVGs, and propagate corrections to any slide exports.
