# Theory: identifiable parity under a semantic gauge

Let `s_t` be an anatomical motion state and `g_t ∈ Z₂` an unobserved convention
that determines which observed limb label names the anatomical left side. Let
`x_t = O(s_t, g_t)` be the observed pose window. An anatomical exchange `A`
reverses a signed target: `y(A s) = -y(s)`. A sensor-frame transformation `C`
does not exchange the anatomical measurement and therefore has a separately
specified target action.

## Identifiability boundary

If the observation model admits

```text
O(s, g) = O(A s, gA)
```

and no anatomical anchor is observed, no deterministic function of `x` can
recover the signed `y` for both explanations. The identifiable object is its
orbit `{y, -y}` (or an invariant such as `|y|`). A predictor that returns one
unqualified sign is making an unstated convention choice.

## Recovery with evidence

Temporal overlap, kinematic continuity, and pose evidence can provide noisy
relative-gauge measurements `g_i⁻¹ g_j` between windows. An anchor fixes the
otherwise global sign ambiguity within a connected component. The paper must
prove a recovery/risk result in which signed-prediction error is controlled by
posterior gauge error; unanchored components retain their irreducible orbit
ambiguity.

This theorem must be written for the stated observational model and must not
be presented as a generic new result about group actions.
