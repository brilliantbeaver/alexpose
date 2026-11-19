# Independent review of score language

Reviewed on 15 November 2026. This is a read-only review of the active source notes, `accessible_gait_deck.py` and `gait_slide_viewer.GLOSSARY`. It focuses on content IDs 21, 23–28, 35–36 and 40, while checking score-related wording elsewhere and the earlier corrections. No presentation content was edited.

## Verdict

The score explanations are scientifically consistent and understandable without specialist gait or JEPA knowledge. No material score-language correction remains. All 43 sets of source notes match the corresponding entries in the current `deck_spec.json`.

The three important findings in `review_16_30.md` are resolved. Content 22 now averages errors and benefits rather than coordinate predictions. Content 24 describes the two displayed combination contrasts and the separate masking contrast without introducing an unplotted laboratory row. Content 41 explicitly fixes the predictor design and evaluation targets while refitting the correction weights.

## What the revised language gets right

- **Movement M describes observed movement.** The notes distinguish the 16 video measurements from the 12 laboratory angle measurements. They locate M's reference in the fitting-normal group. The glossary's description of an unusual combination of visible measurements is consistent with the covariance-based distance used in the code.
- **Raw benefit measures a change in prediction error.** The explanations compare the same hidden answer with and without opposite-leg context. They preserve the distinction between improved completion and improved ranking of abnormal recordings.
- **Cross-leg R measures an unusual benefit, rather than benefit alone.** Content 21 and the glossary explain that both unusually large and unusually small benefits can raise R. They correctly use separate reserved normal references, with five recordings per video fold and 20 participants per laboratory fold. They do not describe R as raw prediction error, neurological severity or disease probability.
- **Combined S is a fixed average.** Adding M and R and dividing by two is explained directly. The weights are not selected on test labels, and equal numerical weights are not presented as equal reliability or equal influence on ranking.
- **Results remain bounded.** Content 23–28 separates the fixed combination's performance from the usefulness of all possible opposite-leg information. The baseline-error comparison uses a reference-based distance rather than raw loss. Content 35–36 retains the laboratory and matched-subset contrasts without suggesting diagnosis or causal adjustment. Content 40 describes improved references and capacity controls as proposals, without claiming that either explains or repairs the observed ranking loss.

The score implementation provides the same definitions: R averages `(benefit - normal_mean)^2 / stabilized_normal_variance` across the two sides; S averages M and R. The notes appropriately leave covariance estimation, variance shrinkage and the numerical formulas in the technical documentation.

## Optional precision and presentation polish

1. The glossary says to divide by an “estimate of normal variation” and then square. For a reader implementing this literally, “estimated normal spread, or standard deviation” would remove possible confusion with dividing by variance before squaring. The present plain-language explanation is not a material factual error, but this change would make the arithmetic unambiguous.
2. In the standalone Cross-leg glossary entry, naming “Own-leg correction” and “Both-leg correction” once would reinforce that the comparator is the corrected own-leg model, rather than the retained baseline. The surrounding slides already establish the correct comparator.
3. Two optional visual-note comments from the earlier review remain: content 16 describes connections while displaying a table, and content 17 describes branches while displaying cards. These do not affect the score definitions or numerical conclusions, but matching the spoken directions to the displayed layout would improve presentation flow.

## Reviewed artifact identities

| Artifact | SHA256 |
| --- | --- |
| `scripts/accessible_gait_deck.py` | `f4e411b29b998c416a6d1aa57b68048c4537411650dc354a716a5cee82fa443d` |
| `scripts/gait_slide_viewer.py` | `fe6eb9484a3abfa591c7b99209917b3a22e291d4631d74ebc7d3d203c12a2faa` |
| `slides/source/notes_01_15.json` | `21c4bbaedd62734a9bac2dee8998983fa2fba912d0a62a80d49e59938a2a3a75` |
| `slides/source/notes_16_30.json` | `106635f590c37cbabb1dccfb9b1336c81b2240372bab2390dfb022e9d33bb2a3` |
| `slides/source/notes_31_43.json` | `ccb2f0b92dc9822906c867206419c500a807c613b67d0f55e18125530b530abc` |
| `slides/deck_spec.json` | `292a1422595db6dc803a277b6c7a6756ed327e58c11adbe7d8a5efddef0a93e7` |
| `slides/tutorial.pdf` | `eb4e8558cc3d053a56b5cfc3a42727746a85b8e98b9bcb568f3d14bc89ea9f9a` |

The PDF hash identifies the contemporaneous build. This follow-up reviews source language and its inclusion in the deck specification; it does not replace the separate rendered-layout review.
