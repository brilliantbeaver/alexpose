# Evaluation plan

## Primary representation tests

1. **AMASS-Gauge:** inject known, blockwise `Z₂` label gauges, occlusion, and
   noise. Measure gauge posterior calibration, transport accuracy, masked
   prediction, odd-target sign risk, and orbit coverage as corruption changes.
2. **Natural-pose audit on GAVD:** quantify disagreement or ambiguity from
   independently configured pose extraction before training a gauge model. Use
   video-grouped splits and report prevalence by source and view.
3. **Cross-domain mechanics/stability validation:** on a public
   participant-identified cohort with measured side-specific force or stability
   quantities, test anchored signed targets and invariant stability outcomes.

## Decisive baselines

- ordinary and mirror-augmented S-JEPA;
- the capacity-matched fixed-reflection and paired-unconstrained models;
- output odd/even symmetrization;
- a temporal left/right swap-corrector;
- discrete synchronization plus ordinary predictive representation; and
- a probabilistic symmetry-breaking/uncertainty baseline.

Success requires more than synthetic sign accuracy: calibrated orbit coverage
when unanchored, improved signed prediction when anchored, and an advantage
that survives naturally occurring ambiguity or a clearly stated robustness
setting.
