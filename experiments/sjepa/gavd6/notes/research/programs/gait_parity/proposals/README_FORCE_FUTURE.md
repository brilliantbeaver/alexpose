# Study 03 legacy entry: fixed-reflection signed-force validation

> **Status:** retained biomechanics-validation route for the fixed-reflection
> baseline. It is not the primary representation-learning claim; see
> [Study 02](../studies/02-semantic-gauge-predictive-representations/).
>
> **Method:** [METHODS_FORCE_FUTURE.md](../methods/METHODS_FORCE_FUTURE.md)

## The question

> After the fixed-reflection JEPA and GAVD baseline protocol are fixed, does encoder-level bilateral equivariance improve held-out participant prediction of signed propulsive-force asymmetry beyond both output repair and equally capable unconstrained paired fusion?

The target is a measured, signed right-minus-left quantity, for example

$$
y_{prop} = \log(J_R / J_L),
$$

where $J_R$ and $J_L$ are independently measured right and left propulsive impulses. This is an odd target: reflection must reverse its sign. It is not a diagnosis, prognosis, or treatment recommendation.

## Why this study is separate

GAVD provides the active, immediately reproducible test of video-disjoint gait-pattern transfer. Its normal/abnormal label is reflection-**invariant**, so it uses `even_output` and source-video groups.

The force study asks the stronger, different question: whether an equivariant interior preserves information useful for a reflection-**odd** biomechanical target. It needs measured force, reliable participant IDs, and participant-grouped evaluation. Do not merge the two results or call GAVD a clinical validation.

## The fixed architecture and decisive controls

The future study inherits the repaired JEPA from the GAVD work:

- branch-specific orbit-closed training masks, including validity and motion masks;
- validity-aware attention and invalid-target exclusion;
- copyability, commutation, rank, entropy, and collapse tests; and
- a frozen architecture/objective choice selected before force outcomes are opened.

Its primary comparisons are deliberately hard:

| Comparison | What it rules out |
| --- | --- |
| `reflection_equivariant` vs `paired_unconstrained` | a gain from generic two-branch cross-attention |
| `reflection_equivariant` vs `odd_output` | a gain obtainable by the cheap final-output sign rule |

All compared models receive the same mirrored examples, paired seeds, participant exposure, tuning budget, update count, and output construction. Report exposure-matched and compute-matched results separately. `standard_sjepa`, mirror augmentation, raw kinematics, random encoder, side-agnostic, nuisance-only, and established skeleton baselines provide context, not substitutes for the two primary controls.

## Evidence design

| Role | Dataset requirement | Question |
| --- | --- | --- |
| Pretraining | AMASS, source-subject-disjoint from downstream cohorts | Does broad non-clinical motion support the representation? |
| Primary test | Stroke cohort with audited bilateral force and participant IDs | Does the architecture improve held-out signed-force prediction? |
| Geometry | MoVi walking data with actor-safe splits | Does signed behaviour survive real camera changes? |
| Replication | Parkinson's cohort, outcome sealed until the protocol is fixed | Does the result replicate under a new cohort? |
| Sanity check | GaitRec or compatible force data | Does the force target behave sensibly at scale? |

The independent unit is a participant. Every cycle, trial, visit, view, condition, and reflected copy from a participant stays in the same outer split. Aggregate cycle predictions to trial and participant before inference. A labelled-person learning curve at 4, 8, 16, and all eligible participants tests the proposed sample-efficiency mechanism.

## What counts as a result

The future study supports an encoder-level claim only when the repaired equivariant encoder has a preregistered practically meaningful advantage over **both** `odd_output` and `paired_unconstrained`, on held-out participants, across paired seeds and simultaneous uncertainty intervals. It must also pass force reliability, representation-health, corruption, coordinate-frame, and real-camera checks.

Possible valid conclusions include that output repair is sufficient, unconstrained fusion is sufficient, benefits occur only with few labels or missing joints, or the constraint harms prediction. A null result is informative. It is not permissible to replace a failed force result with GAVD classification or a geometry residual.
