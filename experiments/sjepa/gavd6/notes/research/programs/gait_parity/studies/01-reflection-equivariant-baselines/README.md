# Study 01: fixed-reflection JEPA baselines

**Status:** active engineering and baseline study; not the program's proposed
representation-learning novelty claim.

This study asks whether a skeleton encoder that is equivariant to a *known*
anatomical reflection improves video-grouped GAVD transfer or later
participant-grouped force prediction over ordinary S-JEPA, mirror augmentation,
output symmetrization, and matched paired fusion.

It is valuable because it supplies necessary controls, exposes leakage, and
provides the strongest ablations for Study 02. It must not claim that fixed
chirality, even/odd channels, or skeleton JEPA is newly invented.

- [Protocol and implementation status](./protocol.md)
- [Current evidence](./results.md)
- [Novelty and interpretation limits](./limitations.md)

The retained compatibility documents are [GAVD proposal](../../proposals/README_GAVD_ICLR.md),
[GAVD method](../../methods/METHODS_GAVD_ICLR.md), [force proposal](../../proposals/README_FORCE_FUTURE.md),
and [force method](../../methods/METHODS_FORCE_FUTURE.md). They remain useful
runbooks until their contents are migrated, but do not define the program's
main paper claim.
