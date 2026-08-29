# Latent Laterality (Study 02)

**Predictive motion representations under unknown left/right correspondence.**

**Status:** active, gated controlled-mechanism study.

This study asks whether a representation can infer and retain uncertainty over
time-local bilateral token-correspondence errors, rather than applying a
post-hoc correction. The immediate action is to run and evaluate the
validation-only swap probe. Do not start SG-JEPA
training unless the probe and observation gates justify it.

## Read in this order

1. [Overview](../../docs/studies/latent-laterality/) for the claim, evidence
   boundary, and definition of done.
2. [Swap probe](../../docs/studies/latent-laterality/swap-probe.md) for the
   executable frozen-encoder gate.
3. [Experiment guide](experiments.md) for data, controls, budget, and decision
   branches.
4. [Theory](theory.md) for identifiability, transport, and estimands.
5. [Proposal](proposal.md) for the scientific framing and related-work audit.

The fixed-reflection experiment is retained as a
[control](../../docs/studies/fixed-reflection-baselines/); it is not Latent Laterality's
representation-learning claim.
