# Evaluation-integrity contract

## Common rules

1. Pre-register or freeze the manifest, endpoint, model-selection rule, and
   grouping unit before examining held-out outcomes.
2. Keep every reflected copy, crop, window, augmentation, and target-domain SSL
   exposure in the same outer group as its source.
3. Tune only within outer-training groups; aggregate windows before scoring a
   source video or participant.
4. Report all seeds, all completed folds, exclusions, failures, and nuisance
   controls. Seeds are training replicates, not extra people.

## Direction-specific rules

The fixed-reflection baseline is credited only for an advantage over ordinary
S-JEPA, output symmetrization, and matched unconstrained paired fusion. The
semantic-gauge study is credited only if it also beats a temporal swap-corrector
and a synchronization-plus-ordinary-predictor baseline, is calibrated when
unanchored, and transfers beyond injected corruption.

Mask closure, commutation, state-dict loading, and non-collapse are eligibility
checks. They establish that an experiment can be interpreted; they are not
downstream evidence or novelty claims.
