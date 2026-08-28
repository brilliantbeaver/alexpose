# Fixed-reflection baseline protocol

## Contract

For a known permutation `p`, the fixed anatomical reflection swaps bilateral
joint IDs and validity values and negates the frozen mediolateral coordinate.
The fixed-reflection model must commute with this operation through online
encoder, EMA target encoder, and predictor.

## Required arms

- `standard_sjepa`
- `standard_mirror_aug`
- `paired_shared_no_cross`
- `paired_unconstrained`
- `reflection_equivariant`

Match trainable capacity, source windows, update budget, seeds, tuning
opportunities, and downstream output wrapper. For an even label, all arms use
an even readout. For a signed anatomical label, all arms receive the matching
odd-output control.

## Implementation gate

The canonical proposal requires branch-specific orbit-closed masks:

```text
mask_B[t, p(j)] = mask_A[t, j]
valid_B[t, p(j)] = valid_A[t, j]
```

The paired implementation now requires an orbit-closed mask with shape
`[B, 2, S, J]`, rather than reusing one target mask at the same joint index in
both branches. Its regression tests reject a mask that violates the rule and
verify that a target's reflected physical counterpart is hidden from the other
branch. Existing checkpoints trained before this change remain ineligible for a
repaired-leakage claim; they must be rerun under the new contract.
