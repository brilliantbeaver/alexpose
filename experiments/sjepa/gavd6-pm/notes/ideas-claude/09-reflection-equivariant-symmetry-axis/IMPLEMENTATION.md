# Idea 9: Reflection-Equivariant Symmetry Axis, the reified implementation

This is the build log and the decision rulebook for the reflection-equivariant idea, written
so an advanced high-school student can follow it from the ground up. The big-picture story is in
[README.md](./README.md). The hands-on proposal walk-through is in [METHODOLOGY.md](./METHODOLOGY.md).
This file is different from both: it describes the code that now EXISTS in this folder, the exact
numbers each notebook writes to disk, and the decision rule that decides what any run is allowed to
claim.

Two rules hold everywhere in this project, and they matter more than any result.

- Folder labels like "stroke" or "myopathic" are dataset annotations from the GAVD dataset (Ranjan
  et al., IEEE Access 2025, DOI 10.1109/ACCESS.2025.3545787). They are not diagnoses made here.
- Every number is transductive. The model was trained on the very videos it is later scored on, so
  a good score can be memory, not skill. Nothing in this file is a claim about diagnosing a person.

The code below is built so that a negative result is a REAL negative result (it changes what we
believe) and not an accident of a broken measuring stick. That design paid off, because the three
experiments in this family returned three negative results that mean three DIFFERENT things.

**Results at a glance, so nobody has to guess.** All three ran; none produced a positive claim; and the
differences between them are the whole point.

| Experiment | What it changed | Endpoint | Verdict | Where |
|---|---|---|---|---|
| Idea 05, `nb_05a` | nothing | ridge R-squared on a signed target | **INFORMATIVE NULL** | section 2 |
| Idea 9 Arm 1, `nb_09a` | the readout's shape only | ridge R-squared on a signed target | **ARTIFACT (side-agnostic nuisance control fired)** | section 7a |
| Idea 9 Arm 2, `new_nb_09_00..03` | the encoder itself | label-free mirror residual rho | **NO CREDIT** | section 8 |

R-squared runs from 1.000, a perfect linear read, down through 0.000, no better than predicting the
mean, and can go NEGATIVE when a fit is worse than that. rho, which is Arm 2's endpoint, is a
label-free mirror residual on a scale where **0 is mirror equivariant, the best value, and 4 is mirror
blind, the worst value**. Both scales are restated where they are used.

Read section 9a before quoting any of them, because "informative null", "artifact", and "no credit"
are three different epistemic states and are routinely confused. Section 9b lists every claim in this
folder that is now SUPERSEDED, so nobody repeats one.

## 1. What we set out to build, in one paragraph

Walking is nearly a mirror image of itself: the left leg does about what the right leg does, half a
step later. Some walking problems break that mirror on one side only (stroke, hemiplegic cerebral
palsy, early Parkinson's). One does not (myopathy, a muscle disease that weakens both sides about
equally). So there is a natural "left minus right" measuring stick that should tell most of these
apart. Idea 05 asked whether the standard trained model ALREADY carries an honest signed version of
that stick, and found it does not. Idea 9 takes the next step: BUILD a readout whose sign is
guaranteed to flip in a mirror, then test whether building it in helps. This file is the reified
version of that plan: a hardened decision rule, three original notebooks (`nb_09a`, `nb_09b`,
`nb_09c`), the four-notebook `new_nb_09` series that carried out the real Arm-2 run, and smoke tests
that prove the plumbing runs end to end.

## 2. What Idea 05 already told us (the anchor we build on)

Idea 05 is the sister proposal in the folder next door. Here is the tutorial version of what it did,
why, what it observed, and what may be concluded.

WHAT WE DID. We fit a simple linear ridge readout on the frozen curriculum-final target encoder and
measured, on folds where whole source videos are held out, how well a signed left-minus-right
laterality axis could be decoded. Nothing was retrained.

WHY. If the signed axis is already sitting in the representation, we should be able to read it out
with a deliberately plain rule, and we should be able to check it against two honest reference
lanes: a non-neural raw-coordinate ceiling, and an untrained random encoder of identical shape.

CHECKPOINT BINDING. Every number below is bound to the CURRENT authoritative checkpoint, fingerprint
prefix `ea59fea0` (full value
`ea59fea055f0230bcf236deb1d1e8bbf08033766e7cd95a98f28210b3042c4e4`). This supersedes the earlier
`d0acc262` checkpoint, which used the same configuration on a differently extracted pose cache. Any
document still quoting A = -0.187, C = +0.147, D = -0.014, or a mirror slope of -0.343 as a current
Idea 05 result is reading the superseded `d0acc262` bundle and is WRONG. Those stale values also
survive in an unused in-tree copy of the result file that the artifact resolver never selects, so
never read Idea 05 numbers out of a local `gavd6-pm/work/artifacts` path.

WHAT WE OBSERVED. The authoritative bundle, read from the configured artifact root resolved by
`docs/artifact_paths.py`, is:

| Lane | What it is | R-squared |
|---|---|---|
| B, raw-coordinate null | hand-built signed left-minus-right coordinates, no network | 1.000 |
| C, untrained floor | the same probe on a random, untrained encoder | -0.156 |
| A, learned (standard `ea59fea0`) | the trained encoder's own features | -0.602 |
| D, mean/std pooled control | a side-blind pooled readout | -0.131 |

Read those four numbers together. The trained encoder (A = -0.602) scored WORSE than the untrained
random encoder (C = -0.156), by 0.446 R-squared, and worse than the side-blind pooled control
(D = -0.131), by 0.471. Sign consistency across held-out sources was 0.444, that is 44.4 percent,
against a required 75 percent. The measured anatomical-mirror slope was -0.741, which is outside the
preregistered flip band of -1.25 to -0.8, so the decoded scalar did not count as flipping.

WHAT WE MAY CONCLUDE. The verdict was an INFORMATIVE NULL: on this cohort and under these gates, the
standard encoder does not make a signed laterality axis linearly available above a raw-coordinate
baseline, and does not even reach an untrained-encoder floor. That negative result is exactly what
makes Idea 9 worth doing, and it is also the honest yardstick Idea 9 must beat. What it does NOT
show is that no side information exists anywhere in the representation, because nonlinear readouts
were never tested, and it is not a clinical statement of any kind.

"R-squared" is a 0-to-1 score for how well a prediction matches the truth: 1 is perfect, 0 is no
better than always guessing the average, and a negative number means the prediction is worse than
guessing the average. Note that every learned lane above is negative, so all of them predict the
signed target worse than a constant would.

## 3. The one idea that makes it work: the antisymmetric head

Here is the whole trick in one picture. Take the six left-right joint pairs (shoulder 11 and 12, hip
23 and 24, knee 25 and 26, ankle 27 and 28, heel 29 and 30, foot index 31 and 32). For each pair,
run the LEFT joint and the RIGHT joint through the SAME small shared box `f`, subtract right from
left, and add up the differences:

```
s  =  sum over the 6 pairs of  ( f(left joint)  -  f(right joint) )
```

The shared box `f` is a tiny neural network: `Linear(embed_dim, 32) -> GELU -> Linear(32, m)`, with
`m = 4` output numbers by default. The same `f` is used on every joint and on both sides. Only the
DIFFERENCE goes into `s`. There is deliberately no `f(left) + f(right)` term, because a
`left + right` sum does not change when you swap the two sides (it is mirror-invariant), so it would
water down the guaranteed sign-flip.

**Why the sign is guaranteed to flip.** Swap the left and right joints at the head's input. Every
term `f(left) - f(right)` becomes `f(right) - f(left)`, which is its exact negative. So the total
flips sign: `s(swap of x) = -s(x)`, exactly, to floating-point precision. This is pure algebra about
the head. It is true for every input, even ones the model never saw, and it does not depend on how
the encoder was trained.

**The one thing this guarantee is NOT.** The exact `-1` belongs ONLY to that input swap (we call it
the WIRING check). The full anatomical mirror is a different operation: it negates the sideways
coordinate and swaps the landmarks on the RAW skeleton, then runs the mirrored skeleton back through
the encoder before the head ever sees it. The encoder mixes all tokens with attention and adds a
learned position code to each joint, so the mirrored features are not a clean swap of the originals.
The slope through the whole encoder is therefore a MEASURED number, close to but not exactly -1, and
different for every model. We never call that measured slope -1. Keep the two apart:

- WIRING check: swap at the head input, slope is exactly -1, always, by construction.
- ANATOMICAL mirror: mirror the raw skeleton, run it through the encoder, slope is measured (Idea 05
  measured -0.741 for the standard `ea59fea0` model).

## 4. Two arms, one measuring stick

The idea is tested as two experiments (two "arms") that are judged by the SAME frozen instrument
from Idea 05. Only one of them ever changes the encoder.

### Arm 1: zero-retrain readout (notebook nb_09a)

Arm 1 does not retrain anything. It takes the EXISTING standard encoder `ea59fea0`, unchanged, caches
its 528-token features, and asks a narrow question: does the antisymmetry-CONSTRAINED pure-difference
head decode the signed axis better than the unconstrained probe did? The shared box `f` here is a
FIXED, seeded, random map (we do not train it), because the sign-flip guarantee holds for ANY `f`.
Only the final linear ridge probe is fit, and it is fit inside each fold on training sources only. So
Arm 1 stays honestly zero-retrain: nothing about the encoder or the head is learned from the test
videos.

Arm 1 is the fast, cheap arm. It was expected to return a null matching Idea 05's finding. What it
actually returned was one step weaker than a null: an ARTIFACT verdict, because the side-blind control
outscored the treatment. The full record is in section 7a, and section 9a explains why "artifact" and
"null" are not interchangeable words. Either way it is not a failure; it is the clean motivation for
Arm 2.

### Arm 2: equivariance-coupled retrain (notebook nb_09b)

Arm 2 is the only arm that changes the encoder. It adds a small, LABEL-FREE penalty to the training
loss that rewards the model for being mirror-honest, then retrains the five-stage curriculum. The
penalty runs the anatomical mirror all the way THROUGH the encoder:

```
L_equiv  =  mean( ( s(encoder(Mx))  +  s(encoder(x)) )^2 )
```

where `Mx` is the anatomically mirrored RAW skeleton (negate the sideways coordinate and swap the
left and right landmarks) and `encoder(.)` is the trainable student encoder. If the encoder is
perfectly mirror-honest, then `s(encoder(Mx)) = -s(encoder(x))`, so the two cancel and the penalty is
zero. Any departure makes the penalty positive, so the optimizer is pushed toward honesty, and the
gradient flows into the encoder weights (which is the whole point of Arm 2).

**Why the mirror has to go through the encoder (a fixed bug).** An earlier build wrote the penalty as
`mean( ( s(swap of x) + s(x) )^2 )` using the head-input token swap as the mirror. That version is a
mathematical no-op: the head is antisymmetric under the input swap BY CONSTRUCTION, so
`s(swap of x)` is exactly `-s(x)` for every input, the two always cancel, the penalty is identically
zero, and its gradient is zero. It could never teach the encoder anything. The fix is to mirror the
raw skeleton and re-encode it, so `L_equiv` measures whether the ENCODER carries the mirror behavior
(which is not guaranteed and is what we want to train), not whether the head is wired correctly (which
is already guaranteed and is checked separately by gate 7, the wiring identity, in section 5). The notebook proves
the fix with a non-equivariant encoder fixture: `L_equiv` comes back strictly positive and both the
head and encoder gradient norms are strictly positive.

It is label-free on purpose: a label-supervised axis term would be a hollow transductive win on only
about seven lateralized source videos. The full training loss becomes:

```
total_loss  =  jepa_loss  +  0.05 * vicreg_loss  +  0.25 * group_loss  +  EQUIV_WEIGHT * L_equiv
```

with `EQUIV_WEIGHT` about 0.02 (set by the `IDEA9_EQUIV_WEIGHT` environment variable), safely below
the main terms. The target (teacher) encoder stays frozen throughout, and the head's parameters join
the trainable set. Every retrained checkpoint records its OWN fingerprint. It is never the frozen
baseline's `ea59fea0`.

**The ablation ladder.** Adding a loss term nudges the whole training path in a nonlinear way, so a
single before-and-after comparison is not trustworthy. Instead Arm 2 runs a ladder across several
seeds (`IDEA9_SEEDS`, default `0,1,2`):

- **D** = the standard baseline (from Idea 05).
- **D0** = the baseline reproduced, same recipe, flip off, `EQUIV_WEIGHT = 0`. This controls for the
  training lineage itself.
- **E1** = D0 plus the `L_equiv` penalty.

nb_09b's own credit rule credited the equivariance penalty only if the average `E1 - D0` gap was both
bigger than D0's own seed-to-seed wobble AND cleared a fixed 0.05 floor. SUPERSEDED: that rule was
replaced for the real run by the three-condition preregistered rule in section 8, which requires the
seed-spread comparison, a paired-by-source bootstrap that excludes zero, and no guardrail regression,
with `all_three_required` set to true. Do not apply nb_09b's two-part rule to any real number.

**The real run was deferred here and has since been carried out elsewhere.** The expensive part of Arm 2
is the multi-seed full-curriculum retrain, so this notebook ships a runnable SMOKE version that proves
every moving part works. The real run was then done by the `new_nb_09_00` through `new_nb_09_03` series
rather than by following section 8's recipe, because that recipe's absolute `L_equiv` is satisfiable by
shrinking the trainable head. On SYNTHETIC FIXTURES (30 synthetic sequences, 30 epochs, w = 0.02, not
gait data) the term falls about 184-fold while the head's output scale shrinks about 4.8-fold and a
parameter-free mirror residual barely moves. The series replaces it with a scale-invariant term, reports
the parameter-free residual as the endpoint, and returns NO CREDIT under its own preregistered rule. See
section 8 for the superseded recipe and where the real numbers live.

## 5. The hardened decision rule (this supersedes the proposal worked-example)

The proposal in README.md and METHODOLOGY.md sketched a first-draft rule (beat the standard encoder
by 0.05, clear 80 percent of the raw-coordinate null, 75 percent sign consistency). Two adversarial
review passes found real holes in that draft. The gates below are the hardened replacements, and
they are what the code actually enforces. Where this file and the proposal worked-example disagree,
THIS FILE WINS.

Every gate is fixed before any fitting, so we cannot move the goalposts later.

1. **Binding bar (the primary gate).** The new head must beat `max(D, C)` by at least 0.05
   R-squared, where D is the standard encoder and C is the untrained floor. Why the max, and not just
   D? Because in Idea 05 the untrained floor C (-0.156) already beat the trained encoder D (-0.602)
   by 0.446. Beating only D would clear a bar that sits below an untrained network, and that would
   overclaim. The floor C is the real constraint.

2. **Beat the floor.** The new head must also clear C by at least 0.05 on its own. This is the hard
   one, and it is the same bar Idea 05 set.

3. **Permutation null on the new head (A prime).** Instead of a fixed "sign correct on 75 percent of
   sources" threshold (which is meaningless when per-condition source counts are 1, 2, 3, 10, 2), we
   shuffle the source labels many times, refit, and check that the real score sits in the top 5
   percent of the shuffled scores. A real signal should be hard to fake by shuffling.

4. **Attribution to antisymmetry (capacity-matched control, Lane Ac).** The new head must also beat a
   capacity-matched control `Ac` by at least 0.05. `Ac` uses the EXACT same shared box `f`, the same
   width, and the same per-pair aggregation as the antisymmetric head, and differs in only one way: it
   ADDS the symmetric `f(left) + f(right)` path next to the `f(left) - f(right)` path. Because
   everything except the antisymmetry constraint is held identical, a gap between A prime and Ac is
   attributable to the CONSTRAINT itself and not to the head's nonlinearity, width, initialization, or
   which joints it looks at. Without this gate, an A-prime win could just be a generic
   head-architecture win. This gate is what catches the "not attributable" future (F5 below).

5. **Absolute nuisance control (Lane E), on a genuinely side-blind feature.** Lane E is a pooled
   readout that MUST NOT recover a signed axis. We require `abs(E) < 0.05` absolutely, and if E fires,
   the signed claim is withdrawn as an artifact. But raw pooling alone is NOT side-blind: the encoder
   adds a learned position code to each joint, so a pooled token still carries which landmark slot it
   came from. So Lane E is SYMMETRIZED over the anatomical mirror, `E(x) = 0.5 * (pool(encoder(x)) +
   pool(encoder(Mx)))`, which makes `E(x)` exactly equal to `E(Mx)` by construction. Only a feature
   that truly cannot tell a body from its reflection can validly withdraw a signed claim, so the code
   asserts this invariance numerically (`max|E(x) - E(Mx)|` must be about 0) before the gate is
   trusted. (This is also stricter than Idea 05's original OR-clause, which was vacuous when the
   raw-null ceiling is near 1.0.)

6. **Y-quality gate.** Before trusting any R-squared, we split the target's variance into a
   between-source part and a within-source part. If the between-source fraction is below 0.30, the
   target is noise-dominated and R-squared is not interpretable; we report that fact instead of a
   score.

7. **Wiring identity check.** The head's token-swap slope must be -1 to floating-point tolerance.
   This confirms the head is built correctly before any comparison is trusted. It is separate from the
   measured anatomical-mirror slope, which is never expected to be -1.

**What the C permutation is, and is NOT.** We still RUN and REPORT a source-label permutation null on
the untrained floor C, but only as a CHARACTERIZATION of how strong the floor is, not as a
claim-withdrawing gate. Lane C is a deterministic transform of the raw coordinates, so its random
features CAN preserve genuine laterality; a significant C null just means the floor is strong, which
the binding bar `max(D, C)` already handles by raising the bar A prime must clear. It is NOT evidence
of source-identity leakage, and tripping the whole verdict on it could wrongly reject a real A-prime
improvement. So C's permutation is descriptive only. (An earlier draft treated a `null_trustworthy`
check on C as a hard gate; that was replaced by the capacity-matched attribution gate above, which
targets the actual confound.)

We dropped the proposal's "reach 80 percent of the raw-coordinate null" gate entirely. The raw-null
ceiling B is near 1.0 by near-circular construction (it fits raw side-differences to a target built
from the same side-differences), so it is descriptive only and never a gate.

### The six lanes, side by side

| Lane | Feature source | Retrain? | Role | What we expect |
|---|---|---|---|---|
| A prime | new antisymmetric head (difference only) on encoder tokens | no (Arm 1) / yes (Arm 2) | primary | beat `max(D, C)` and Ac by >= 0.05; wiring slope -1 exact; anatomical slope measured |
| Ac | same head, adds the symmetric left-plus-right path | no | capacity-matched control | A prime must beat it by >= 0.05 (attribution to antisymmetry) |
| B | hand-built signed left-minus-right coordinates | no | descriptive ceiling | near 1.0, near-circular, NOT a gate |
| C | random-init encoder, same head | no | floor; permutation reported | near chance; a significant null is floor characterization, not a gate |
| D | standard `ea59fea0` features | no | learned-behavior comparator | Idea 05's -0.602 |
| E | pooled tokens, symmetrized over the anatomical mirror | no | side-blind nuisance control | `E(x) == E(Mx)` exactly; must stay `abs(E) < 0.05` |

## 6. The five possible futures (from nb_09c)

Notebook nb_09c has no neural network in it at all (pure numpy, scikit-learn, and matplotlib). It
feeds hand-chosen numbers through the exact same decision rule as nb_09a and prints what each outcome
would be allowed to claim. This is a rehearsal of the verdict logic, not data. The five futures are
saved in `idea9_futures_bundle.json` and drawn in `images/idea9_possible_futures.png`.

- **F1, head beats the bar (the win).** The new head scores 0.42, well above the binding bar, beats
  the capacity-matched control Ac, clears its permutation null, the nuisance control stays quiet, and
  the measured anatomical mirror flips (slope about -1.02, inside the flip band). Verdict: ANTISYMMETRY
  BEATS BINDING BAR AND CAPACITY-MATCHED CONTROL. This is the only future that LICENSES the claim that
  building the mirror rule in helped.

- **F2, decodable but not flipping.** The head scores 0.40 and clears the bar and Ac, but the measured
  anatomical mirror slope is only about -0.3 (not in the flip band). Verdict: the signed axis is
  decodable, but the through-encoder mirror equivariance is WITHHELD. The exact wiring -1 still holds
  by construction; the encoder just does not carry the mirror behavior all the way through.

- **F3, informative null (nb_09c's predicted Arm-1 result).** The head scores -0.10, below a binding
  bar of 0.147. Verdict: INFORMATIVE NULL. Antisymmetry by construction alone does not lift the
  frozen encoder above the bar. In this scenario the untrained floor C also clears its own
  permutation null (p = 0.01), which is REPORTED as floor characterization (the floor is strong,
  which is why the binding bar uses `max(D, C)`) and does NOT change the verdict.

  Two warnings about F3, both important. First, F3's lane values (A prime -0.10, Ac -0.05, C +0.147,
  D -0.187, E -0.014, mirror slope -0.34) are hand-chosen SCENARIO INPUTS. They were originally
  sketched from the superseded `d0acc262` bundle, so they are NOT the authoritative Idea 05 result
  and must never be quoted as one. They are left unchanged on purpose, because substituting the
  authoritative Idea 05 values would move the binding bar `max(D, C)` from +0.147 to -0.156, at which
  point A prime -0.10 would CLEAR the bar by 0.056 and the scenario would stop exercising the null
  branch it exists to exercise. Second, F3's prediction is SUPERSEDED by the real Arm-1 run:
  nb_09c predicted an informative null for Arm 1, but the actual Arm-1 verdict was
  `ARTIFACT (side-agnostic nuisance control fired)`, which is a weaker epistemic state than a null.
  See section 7a.

- **F4, artifact (the nuisance control fires).** The head scores 0.47, but the side-blind symmetrized
  control also scores 0.46. Verdict: ARTIFACT, signed claim withdrawn. A genuinely signed quantity
  cannot be recovered by a side-blind readout, so a high A prime here is really a magnitude or
  acquisition artifact.

- **F5, not attributable to antisymmetry.** The head scores 0.30 and clears the binding bar and its
  null, but the capacity-matched control Ac scores 0.28, so A prime does NOT beat Ac by the 0.05
  margin. Verdict: NOT ATTRIBUTABLE TO ANTISYMMETRY. The win rides on the head's generic capacity
  (nonlinearity, width, pair information), which Ac holds identical, rather than on the antisymmetry
  constraint. This future exists to prove the attribution gate actually bites; nb_09c asserts F5 comes
  back not-attributable and that a significant C-null never withdraws a clean positive.

## 7a. What the real Arm 1 run measured (nb_09a, the authoritative Arm-1 record)

This section is the real result, not a smoke check and not a future. It is bound to the same frozen
`ea59fea0` encoder Idea 05 used, with zero retraining, 5 source-disjoint folds, head output dimension
4, on the same cohort of 96 canonical sequences from 18 source videos.

WHAT WE DID. We kept the encoder frozen and changed only the READOUT'S SHAPE, replacing Idea 05's
unconstrained probe with a head that is antisymmetric by construction.

WHY. Idea 05's null had an obvious escape route: perhaps the signed quantity was there all along and
the probe was simply the wrong shape to see it. A signed quantity ought to be read by a head that a
left-right swap necessarily negates. Arm 1 closes that escape route.

WHAT WE OBSERVED. The lane ladder, all R-squared:

| Lane | Role | R-squared |
|---|---|---|
| B_raw_null | raw-coordinate ceiling, descriptive only | 1.000 |
| A_prime | treatment, antisymmetric-by-construction head | -0.206 |
| Ac_capacity_matched | same capacity, antisymmetry constraint removed | -0.184 |
| C_floor | untrained-encoder floor | -0.027 |
| D_standard | Idea 05's lane A feature, carried forward | -0.602 |
| E_pooled | mirror-symmetrized, mathematically blind to left and right | -0.066 |

Gate outcomes, every one of them fixed before fitting. The binding bar `max(D, C)` is -0.027, and the
binding delta is -0.179, so the binding bar FAILED. The attribution delta `A prime - Ac` is -0.022,
so the capacity-matched gate FAILED. The permutation null on A prime returned p = 0.970 over 200
permutations against a null mean of -0.108, so the permutation gate FAILED. The absolute nuisance
gate FAILED, because `abs(E)` is 0.066 and the bar is 0.05. The y-quality gate FAILED, on the number
in the next subsection. Two things PASSED: the wiring identity, with a swap slope of -1.0000000000000002,
and lane E's anatomical-invariance self-check, which confirms lane E really cannot tell a body from
its reflection. The measured anatomical-mirror slope was -0.223 and did not flip.

THE DECISIVE NUMBER. The side-blind lane E (-0.066) outscored the antisymmetric treatment A prime
(-0.206) by 0.140. A lane that is mathematically incapable of telling left from right did better than
the lane built specifically to read left from right.

WHAT WE MAY CONCLUDE. **Verdict: `ARTIFACT (side-agnostic nuisance control fired)`.** The
readout-shape objection is answered and rejected: constraining the head to be antisymmetric did not
rescue the null. The wiring slope of exactly -1 proves the head really was antisymmetric, so this is
not an implementation bug. Read the epistemic status carefully. "Artifact" is a WITHDRAWAL, not an
answer. It says this lane is not admissible evidence about sides at all, because a control blind to
sides scored higher. That is a different and WEAKER state than Idea 05's clean informative null, not
a stronger one.

### 7a.1 Arm 1's most valuable output is a fact about the COHORT, not about the encoder

The single most useful number the whole symmetry family produced is measured here, in `nb_09a`, and
nowhere else. It is not an Idea 05 result and it is not an `nb_05b` result; any document attributing
it to either is misattributing it.

`y_between_source_fraction` = **0.075** against a preregistered threshold of **0.30**.

WHAT THIS MEANS, step by step. We split the signed-laterality target's total variance into the part
that lies BETWEEN source videos and the part that lies WITHIN them. Only about 7.5 percent lies
between sources; the other roughly 92.5 percent lies within them. Our folds hold out whole source
videos by design, for the good reason that the source video is the independent unit of evidence. But
that design choice therefore holds out almost all of the usable between-source signal by
construction. So on this cohort and this target, a held-out-source R-squared cannot support a
positive laterality claim no matter how good the encoder is and no matter how well the head is
shaped. The binding constraint is the number of independent source videos, which is 18.

This one measurement explains Idea 05's null and Arm 1's artifact at the same time, and it is the
reason Arm 2 abandons R-squared as its primary endpoint.

## 7b. What the smoke runs proved (PLUMBING ONLY, no evidence about any checkpoint)

Smoke mode runs the entire pipeline on tiny hand-authored motions and a tiny model, so it checks that
every moving part works without paying the 600-epoch cost. Every number in this section is a plumbing
check on synthetic data. None of it is evidence about any checkpoint, any cohort, or any gait
question, and none of it may be quoted as a result. Here is what the smoke runs confirmed, with the
exact values they wrote to disk.

From `nb_09a` (`work/artifacts/smoke/idea9_antisymmetric_readout_result.json`):

- Wiring identity holds: token-swap slope = -1.0000000000000002 (exact to floating point). Good.
- The anatomical-mirror slope is a genuinely different, measured number (+1.15 on the toy data) and
  does not flip. This is the expected separation between the two mirror checks.
- The capacity-matched control Lane Ac builds and fits: on the toy data A prime = -0.321 and
  Ac = -0.369, so the attribution delta is +0.048. This delta being computed and reported at all is
  the plumbing check; the value itself is toy noise, not a result.
- The Lane E anatomical-invariance self-check is exact: `max|E(x) - E(Mx)| = 0`, so the symmetrized
  control is genuinely side-blind, and `E_anatomically_invariant_ok = true`.
- The y-quality gate passes (between-source fraction 0.99 on the planted toy signal).
- The nuisance control FIRES on the toy data (E = -0.40, and the |E| < 0.05 gate fails), so the
  verdict correctly reads ARTIFACT (side-agnostic nuisance control fired). That is the guardrail doing
  its job on noise; it is not a result.
- All six lanes (A prime, Ac, B, C, D, E) fit through the whole-video-held-out folds with at least
  two disjoint sources per fold (30 sequences, 10 sources, 5 folds), and a fold manifest is recorded.

From `nb_09b` (`work/artifacts/<mode>/idea9_equivariant_retrain_result.json`; the contents are synthetic
data at smoke scale wherever the file is written, which is what `data_source` and `training_scale` record):

- The five-stage curriculum completes for every seed (`0, 1, 2`) and for both `equiv_on = false`
  and `equiv_on = true`, and the target encoder stays frozen throughout.
- Six distinct checkpoint fingerprints are recorded, one per (seed, equiv_on) run, and none is the
  baseline's. The exact values are not quotable here: the fingerprint payload includes the artifact mode,
  so the same data and the same training produce different fingerprints when the notebook is pointed at a
  different artifact directory. What the check establishes is distinctness and separation from the
  baseline lineage, both of which hold in the emitted bundle.
- `L_equiv` gives finite, strictly positive values through the encoder (for example 0.0082, 0.322,
  0.191 across stages) and finite gradients into both the head and the encoder. A guardrail asserts
  `L_equiv > 0` and both gradient norms `> 0` on a non-equivariant encoder fixture, so the fixed loss
  can never silently return to the identically-zero no-op it had before.
- The credit rule runs and returns `equiv_credited = false` on the toy data: the effect E1 - D0 is
  +0.025, which neither exceeds D0's seed spread (0.036) nor clears the 0.05 floor. That is the correct
  smoke outcome (a real effect must clear both bars). SUPERSEDED AND NOT EVIDENCE: this whole ladder
  ran on synthetic data at smoke scale, so +0.025, 0.036, and the `equiv_credited = false` verdict are
  statements about whether the code paths execute, not about any checkpoint. They must never be cited
  as an Arm-2 finding. The real Arm-2 ladder is the `new_nb_09_00` through `new_nb_09_03` series in
  section 8, and its rule is a different rule.
- The absolute `L_equiv` written here is nonetheless the wrong objective, which only became visible once
  the endpoint was measured separately from the loss. See section 8.

From `nb_09c` (`idea9_futures_bundle.json` plus `images/idea9_possible_futures.png`):

- All five futures score through the hardened rule and the self-check assertion passes: F1 and F2 are
  positive, F3 is an informative null whose significant C-permutation is reported but does not change
  the verdict, F4 is an artifact, and F5 correctly returns NOT ATTRIBUTABLE TO ANTISYMMETRY because it
  fails the capacity-matched gate. The self-check also asserts that a significant C-null does not
  withdraw a clean positive.

## 8. How to run the real thing

The three notebooks live at the gavd root as `nb_09a_antisymmetric_readout_probe.ipynb`,
`nb_09b_equivariant_retrain.ipynb`, and `nb_09c_futures_and_reach.ipynb`. Each is emitted by a
committed builder script in this folder (`_build_nb_09a.py`, `_build_nb_09b.py`, `_build_nb_09c.py`);
run the builder to regenerate the notebook.

**Arm 1, the real zero-retrain readout on `ea59fea0`:**

```
GAVD_MODE=real  jupyter nbconvert --to notebook --execute nb_09a_antisymmetric_readout_probe.ipynb
```

Then read `work/artifacts/real/idea9_antisymmetric_readout_result.json`. Check that the fingerprint
starts with `ea59fea0`, and that the wiring slope is -1. This has already been run, and the recorded
outcome is in section 7a: the y-quality gate did NOT pass (between-source fraction 0.075 against 0.30)
and the nuisance control did NOT stay quiet (`abs(E)` = 0.066 against a bar of 0.05), giving the verdict
`ARTIFACT (side-agnostic nuisance control fired)`.

**Arm 2, the real retrain: superseded recipe, and what replaced it.**

The recipe below is SUPERSEDED. It is preserved because the mistake in it is instructive, but it must not
be followed. Step 1 installs an absolute squared mirror residual on a TRAINABLE head, which the head can
satisfy by shrinking its own output while the encoder stays exactly as mirror-blind as it was. On SYNTHETIC
FIXTURES, and these are fixture numbers rather than gait results, the term falls about 184-fold and the
head's output scale shrinks about 4.8-fold, while a parameter-free residual improves by only 0.010 against
a gate of 0.049, measured against a control that has no equivariance term at all. The lesson generalizes:
any absolute equivariance penalty sitting on a trainable readout is satisfiable by shrinking that readout,
so the penalty must be normalized by its own magnitude or carry no trainable parameters at all.

<details>
<summary>The superseded recipe, kept for the record</summary>

1. Paste the antisymmetric head and `equivariance_loss` into nb04's training-step cell, add the
   head's parameters to the trainable set, and add
   `EQUIV_WEIGHT * equivariance_loss(head, model.view_encoder, coords, SEGMENTS, EMBED_DIM)` to
   `total_loss` behind an `equiv_on` switch. Note the signature takes the raw `coords` and the
   trainable `view_encoder`, because the loss mirrors the raw skeleton and re-encodes it; passing
   pre-encoded `tokens` would recreate the head-only no-op.
2. Add `equiv_weight` and `equiv_on` to the fingerprint payload so each checkpoint gets its own
   fingerprint (never the baseline's).
3. Train D0: `GAVD_MODE=real SJEPA_RUN_PROFILE=recommended IDEA9_EQUIV_WEIGHT=0` across seeds
   `0,1,2` (add `3,4` for a tighter spread estimate).
4. Train E1: the same command with `IDEA9_EQUIV_WEIGHT=0.02` across the same seeds.
5. Re-score every checkpoint through nb_09a with `SJEPA_INSPECT_CHECKPOINT=<file>`, then apply the
   section-5 credit rule: `E1 - D0` must exceed D0's seed spread AND clear the 0.05 floor.

</details>

What replaced it is a four-notebook series that leaves notebook 04 untouched, normalizes the residual by
its own magnitude so shrinking the head buys nothing, and reports a parameter-free residual (`rho`) that no
trainable weight can influence. Run in this order:

```
new_nb_09_00_methodology_and_contract                 # preregister endpoints, guardrails, credit rule
new_nb_09_01_mechanism_and_smoke_validation           # mechanism checks, endpoint calibration, variant bake-off
new_nb_09_02_real_multiseed_equivariant_training      # real D0-vs-E1 ladder, resumable per rung
new_nb_09_03_evaluation_results_discussion            # recompute endpoints and guardrails, apply the rule
```

Outcome, in `work/artifacts/<mode>/idea9_arm2/`: **NO CREDIT**.

The endpoint is rho, a label-free mirror residual. Its scale: **0 is mirror equivariant, which is the best
value, and 4 is mirror blind, which is the worst value**. rho has no fitted parameters and needs no labels,
which is exactly why Arm 2 can use it after the labelled target failed Arm 1's y-quality gate.

WHAT WE OBSERVED. rho on the target encoder fell from a D0 control mean of 0.462 to an E1 mean of 0.059, an
improvement of 0.403 against a control seed spread of 0.057, which is about 7.1 times that spread. The
paired-by-source bootstrap over 18 sources with 4000 draws gave a 95 percent interval of [1.118, 2.291] on
the per-source ratio, which excludes zero, and **18 of 18** source videos improved. Do not conflate the
per-source ratio with the cohort-level improvement of 0.403; condition 1 uses cohort means of summed terms
while condition 2 averages per-source ratios, and the bundle itself carries that warning. The head's output
scale GREW, from 0.748 to 1.059, so the shrink-the-head degenerate solution is ruled out on real data. The
measured anatomical-mirror slope moved from -0.648 toward the ideal -1, reaching -0.937.

WHY IT IS STILL NO CREDIT. Condition 3 failed. The `feature_std` guardrail fell from 0.400 to 0.371, a
regression of 0.0288 against a D0 seed spread of 0.0082, which is about 3.51 times the spread.
`mean_pair_cosine` passed by a hair, regressing 0.011114 against a spread of 0.011435. The registered
`source_grouped_five_class_balanced_accuracy` guardrail was **not evaluable** on this cohort, because
source videos per condition are normal 1, parkinsons 2, stroke 3, myopathic 10, cerebralpalsy 2, and a
condition with one video leaves a source-grouped fold with nothing to learn that condition from. Never
write that this guardrail "failed"; it was not evaluable. A substitute leaky probe improved, but it leaks
video identity, so it supports no condition claim.

WHAT WE MAY CONCLUDE. The credit rule sets `all_three_required: true`, so a condition-3 failure means no
credit, and that is the honest reading rather than a technicality. A term that asks the encoder to respond
identically to a body and its reflection is ALSO a term that removes variance. So variance loss is a live
competing explanation for the endpoint gain, and this experiment cannot separate "the encoder learned
mirror structure" from "the encoder lost variance in a way that happens to reduce rho". That is precisely
why the guardrail was registered in advance. Note also that rho is a symmetry property of the
representation: it is not accuracy, not class separation, and not clinical value, and the antisymmetric
lane R-squared did not improve, moving from -0.027 to -0.030, so there is no downstream gain to report.

Two protocol deviations are recorded with the result. First, **3 seeds run against 5 registered**, because
measured per-rung cost on MPS exceeded the estimate the wall-clock budget was approved against; the effect
is that the D0 seed-spread yardstick is estimated from fewer samples, so a marginal effect could not have
been adjudicated, and this effect was not marginal. Second, the source-grouped five-class guardrail was
**not evaluable**, for the reason given above.

The next step named by the notebooks is a sweep on `equiv_weight` to separate variance removal from
genuine mirror structure, plus a task with an interpretable endpoint and enough independent sources.

**Futures, any time (no GPU, no torch):**

```
jupyter nbconvert --to notebook --execute nb_09c_futures_and_reach.ipynb
```

## 9. External reach (honest, non-clinical, scaffold only)

There is no honest clinical transfer test available, because no participant-disjoint skeleton cohort
for hemiplegic cerebral palsy or myopathy exists. So the only external step is a method check on
public, non-clinical multi-view pose cohorts (CASIA-B, Yu 2006; OU-MVLP-Pose, Takemura 2018). The
question they can answer is narrow: does the built-in mirror property stay stable when the camera
moves, and does a genuine left-versus-right camera swap flip the sign? nb_09c ships this as a
scaffold with the real loaders marked TODO. The only numbers it produces come from a SYNTHETIC FIXTURE
with the answer planted in it: the view-stability correlation is near 1.0 and the genuine-mirror slope
is near -1.0 BECAUSE the fixture was constructed that way, so those two numbers confirm that the
comparison is wired correctly and confirm nothing else. They are not empirical results, not external
validation, and not evidence about any encoder. The same caution applies to the identical fixture in
`nb_05b`. When the real loaders land, these are still architecture checks on healthy, non-clinical
people and say nothing about diagnosis.

## 9a. The joint conclusion: what Idea 05 and Idea 9 together do and do not say

This is the passage a reader must not be able to miss. Idea 05 and Idea 9 do NOT say the same thing.
Three experiments asked one shared question, and they returned three verdicts that mean three
DIFFERENT things. Never collapse them into "it did not work".

| Experiment | What it changes | Endpoint | Verdict |
|---|---|---|---|
| Idea 05, `nb_05a` | nothing; reads out of the frozen encoder | ridge R-squared on a signed target | **INFORMATIVE NULL** |
| Idea 9 Arm 1, `nb_09a` | the readout's shape only; encoder still frozen | ridge R-squared on a signed target | **ARTIFACT (side-agnostic nuisance control fired)** |
| Idea 9 Arm 2, `new_nb_09_00..03` | the encoder itself, during the full curriculum | label-free mirror residual rho | **NO CREDIT** |

**1. Three verdicts, three distinct epistemic meanings.**

- Idea 05's **informative null** means the measurement was VALID and the answer was NO. There is no
  linearly decodable signed laterality axis above a raw-coordinate baseline or an untrained floor.
  A question was asked and answered.
- Arm 1's **artifact** means the measurement was NOT ADMISSIBLE EVIDENCE about sides at all, because
  a side-blind control (lane E, -0.066) outscored the treatment (A prime, -0.206) by 0.140. The claim
  is WITHDRAWN rather than answered. This is a WEAKER epistemic state than a null, not a stronger one:
  a null tells you the answer is no, whereas an artifact tells you the instrument was not measuring
  what its name says.
- Arm 2's **no credit** means the effect is REAL, LARGE, and CONSISTENT, on 18 of 18 source videos and
  at about 7.1 times the control's seed spread, but a preregistered guardrail failed and that failure
  supplies a competing explanation for the effect, so the effect is not credited. "No credit" is not
  "no effect".

**2. Each experiment closes a specific escape route from the previous one.** Idea 05 could have failed
because the readout was the wrong SHAPE, so Arm 1 built an antisymmetric-by-construction head, verified
its wiring at exactly -1, and the null did not survive as a null; it degraded into an artifact. Arm 1
could have failed because the encoder was never ASKED to respect the mirror, so Arm 2 asked it directly
and drove rho down to 0.059, which proves incapacity was never the explanation. Each result is
informative precisely because it removes one story about why the previous result came out the way it
did.

**3. The binding constraint is the COHORT, not the model and not the readout.** There are 18
independent source videos, and only 7.5 percent of the labelled target's variance lies between them
against a preregistered 30 percent (measured in `nb_09a`, section 7a.1). Source-disjoint folds
therefore hold out nearly all of the usable signal by construction. Arm 2's label-free rho sidesteps
that limit for the symmetry PROPERTY, but nothing in this package sidesteps it for a labelled clinical
target.

**4. What none of the three licenses.** No clinical claim. No statement about unseen videos or unseen
people. No equating of rho with performance or with condition separation. Every score in this family is
TRANSDUCTIVE: the encoder saw every evaluation sequence during training.

**5. The controls carry the findings.** In all three experiments the informative element is a CONTROL,
not the treatment: the untrained floor in Idea 05, the side-blind lane E in Arm 1, the feature-spread
guardrail in Arm 2. That is the single most transferable lesson here. Report the ladders, not just the
headline.

## 9b. Superseded claims register (do not repeat these)

Each row is a claim that was once written down in this folder and is now retired. If you find one in
any document, it is a defect.

| Superseded claim | Why it is retired | What is true instead |
|---|---|---|
| Idea 05 measured A = -0.187, C = +0.147, D = -0.014, mirror slope about -0.343 | those come from the superseded `d0acc262` checkpoint bundle | on the authoritative `ea59fea0` bundle, A = -0.602, C = -0.156, D = -0.131, B = 1.000, mirror slope = -0.741 |
| the section-8 recipe, which pastes an absolute equivariance term into notebook 04's training step | an absolute residual on a TRAINABLE head is satisfiable by shrinking that head, so it can be driven to near zero without changing the encoder at all | the scale-invariant normalized term, and the `parameter_free` variant with no head parameters to shrink, used by the `new_nb_09` series |
| nb_09b's smoke ladder numbers, for example E1 - D0 = +0.025 against a seed spread of 0.036 | they were produced on synthetic data at smoke scale | they are plumbing checks only and are not evidence about any checkpoint; the real ladder is `new_nb_09_02` and `new_nb_09_03` |
| nb_09b's two-part credit rule (beat the seed spread AND clear a fixed 0.05 floor) | replaced before the real run | the three-condition rule with `all_three_required: true`: seed-spread comparison, paired-by-source bootstrap excluding zero, and no guardrail regression |
| the proposal-level Idea 9 gates (beat the standard encoder by 0.05, reach 80 percent of the raw null, sign correct on 75 percent of sources) | the 80-percent gate is unreachable because the raw null is near 1.0 by near-circular construction, and a fixed 75-percent sign threshold is meaningless at per-condition source counts of 1, 2, 3, 10, 2 | the seven hardened gates in section 5: binding bar against `max(D, C)`, beat floor, capacity-matched attribution, permutation null, absolute nuisance bound on the side-blind lane E, y-quality gate, wiring identity |
| nb_09c's future F3, which predicted an INFORMATIVE NULL for Arm 1 | the real Arm-1 run fired the side-blind nuisance control | the actual Arm-1 verdict is `ARTIFACT (side-agnostic nuisance control fired)`, a weaker state than a null |
| the 7.5 percent between-source variance fraction attributed to Idea 05 or to `nb_05b` | it was never measured there | it is measured in `nb_09a`, Idea 9 Arm 1, as `y_between_source_fraction` = 0.075 against a threshold of 0.30 |
| `nb_05b` producing an empirical verdict about the checkpoint | it contains no measurement of any checkpoint | `nb_05b` is a possible-futures simulator plus an external-reach scaffold whose real loaders are still TODO; its simulated futures and its planted near-unity fixture correlation are not empirical results |
| the fixture magnitudes near rho 3.9 forecasting real magnitudes | the real D0 control lands near 0.462 on the target encoder | fixture magnitudes do not forecast real magnitudes, and `new_nb_09_01` explicitly retracts any such reading |
| the SCORECARD composites 4.60 for Idea 05 and 4.55 for Idea 9 as evidence of outcome | they are proposal-quality scores on a five-axis rubric, assigned before any of these experiments ran | the measured outcomes are informative null, artifact, and no credit; a proposal score and a result are different kinds of thing |

## 10. What this cannot tell us

- Transductive. The model saw every evaluation video during training, so no number here is an
  out-of-sample estimate. Seed variation is not source variation.
- Tiny sample. Only 18 canonical source videos, with per-condition source counts as low as 1, so we
  pool across conditions and plot every source as one dot, and make no per-class asymmetry claim.
- Provenance confound. Normal is one video on a mostly-augmented path while abnormal rows are on the
  canonical path, so the primary comparison runs on the provenance-matched canonical subset
  (myopathy versus lateralized, with normal dropped) to avoid reading how a video was processed
  instead of how a person walked.
- Skeleton limits. Skeletons cannot recover forces or push-off, muscle-electrical activity or
  stiffness, twisting rotation, or a muscle-disease diagnosis, and this idea claims none of them.
- One-fingerprint binding. Every number is bound to a single checkpoint before comparing. The frozen
  baseline for Idea 05 and Arm 1 is `ea59fea0`; the superseded `d0acc262` lineage and the `dba24a`
  canonical lineage are never mixed into the same comparison.

## 11. Files this implementation touches

- `_build_nb_09a.py` -> `nb_09a_antisymmetric_readout_probe.ipynb` (Arm 1). Writes
  `work/artifacts/<mode>/idea9_antisymmetric_readout_result.json` and a PNG.
- `_build_nb_09b.py` -> `nb_09b_equivariant_retrain.ipynb` (Arm 2, smoke-first). Writes
  `work/artifacts/<mode>/idea9_equivariant_retrain_result.json`.
- `_build_nb_09c.py` -> `nb_09c_futures_and_reach.ipynb` (futures and reach). Writes
  `idea9_futures_bundle.json` and `images/idea9_possible_futures.png`. Its five futures are hand-chosen
  scenario inputs, not measurements, and its F3 prediction for Arm 1 is superseded (section 9b).
- `_build_new_nb_09_series.py` -> `new_nb_09_00` through `new_nb_09_03` (the real Arm 2). Writes
  `work/artifacts/<mode>/idea9_arm2/`. This is where Arm 2's authoritative numbers live.
- Evaluation machinery (folds, ridge probe, mirror, raw-null, pooled control, JSON and verdict
  shape) is reused verbatim from Idea 05's `_build_nb_05a.py`, so both ideas are judged on the same
  ruler.

## References

Same source list as [README.md](./README.md) and [METHODOLOGY.md](./METHODOLOGY.md); every number in
this file traces to [../_shared_facts.md](../_shared_facts.md) (numbers) and
[../_neuro_facts.md](../_neuro_facts.md) (biology). If any number here disagrees with those files,
trust those files.
