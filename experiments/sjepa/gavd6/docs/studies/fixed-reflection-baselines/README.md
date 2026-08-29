# Study 01: fixed-reflection JEPA baselines

**Status:** active engineering and baseline study; not the program's proposed
representation-learning novelty claim.

This study asks whether a skeleton encoder that is equivariant to a *known*
anatomical reflection improves video-grouped GAVD transfer or later
participant-grouped force prediction over ordinary S-JEPA, mirror augmentation,
output symmetrization, and matched paired fusion.

It is valuable because it supplies necessary controls, exposes leakage, and
provides the strongest ablations for Latent Laterality. It must not claim that fixed
chirality, even/odd channels, or skeleton JEPA is newly invented.

- [Protocol and implementation status](./protocol.md)
- [Current evidence](../../../notes/reflection-baselines/results.md)
- [Novelty and interpretation limits](./limitations.md)

Earlier GAVD and force planning material is retained in the repository's
historical notes. It is useful context, but does not define the current claim.
