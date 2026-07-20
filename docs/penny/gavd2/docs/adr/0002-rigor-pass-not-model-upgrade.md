# ADR 0002: iteration 2 is a rigor pass, not a model upgrade

**Status:** accepted

## Context

"More robust and enhanced" could mean either (a) keep the method and make the comparison
honest, or (b) strengthen the model to try to beat the 0.76 baseline. The iteration-1
finding is that the honest per-sequence probe accuracy is around 0.49 on a small
labelled set, well above the 0.20 chance level but below the tuned 82-feature Random
Forest. Chasing a higher number on 42 to 68 sequences, where variance is large, risks
overfitting the evaluation rather than producing a credible result.

## Decision

iteration 2 keeps the iteration-1 method unchanged (pose-sequence JEPA, EMA target,
LayerNorm-target L2 plus online-only VICReg, block masking, frozen probe) and invests
entirely in a genuinely controlled comparison and clearer presentation. Model upgrades
(deeper or wider encoder, longer pretraining, richer inputs, stronger probe) are
recorded as future work, not done here.

## Consequences

- The one intended difference from exp5 is the representation (learned embedding versus
  hand features); everything else is held constant. This makes the result interpretable.
- The honest headline number may sit below 0.76. That is reported plainly; the scientific
  claim is about the controlled comparison and the label-scarcity thesis, not about
  beating the baseline.
- A future iteration can build on `gavd2` to pursue the model-upgrade branch with the
  controlled harness already in place.
