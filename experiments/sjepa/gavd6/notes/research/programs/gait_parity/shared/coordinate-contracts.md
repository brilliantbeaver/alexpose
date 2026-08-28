# Coordinate and laterality contract

## Fixed skeleton contract

All studies freeze the joint schema, joint order, temporal patching, body-frame
normalization, frame rate, missing-value representation, and confidence
semantics before measuring a downstream outcome. A storage zero is never a
coordinate observation; validity must travel separately through the model.

## Two transformations, not one

The legacy fixed-reflection study uses an **anatomical reflection** `A`: swap
left/right joint identities and negate the mediolateral coordinate in a frozen
body frame. A quantity such as right-minus-left propulsion is odd under `A`;
walking speed is even.

The semantic-gauge study also distinguishes a **sensor-frame transformation**
`C`. `C` changes a camera or coordinate convention without exchanging which
anatomical limb produced a physical measurement. It must not be silently
identified with `A`.

## Semantic gauge

For the new study, `g_t` denotes a latent side-label convention for a temporal
window. It represents uncertain or inconsistent anatomical naming in an
observation pipeline; it is not a claim that anatomy itself changes. When `g_t`
is unanchored, a signed anatomical target can be identifiable only up to its
orbit `{y, -y}`. An anatomical landmark convention, force-plate side label, or
other documented measurement may act as an anchor.

Any experiment must say which of `A`, `C`, and `g_t` it manipulates and which
are observed, inferred, or held fixed.
