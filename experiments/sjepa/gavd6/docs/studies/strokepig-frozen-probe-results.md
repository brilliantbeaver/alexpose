# StrokePIG frozen-probe — current negative feasibility result

**Artifact:** `work/artifacts/strokepig_frozen_jepa_probe/`

The 24-participant frozen probe does not support force prediction with the
available representations. Every evaluated arm has negative held-out R-squared:
the least negative result is -0.051 for the random paired-unconstrained control,
while the frozen reflection-equivariant arm is -0.058. Correlations are also
negative and calibration slopes are strongly negative.

This is a useful null result and a boundary on the program: the current frozen
Core11 representations should not be described as predicting force, balance, or
clinical outcomes. Retain the fold, prediction, participant-target, and contact
tables together with this interpretation.
