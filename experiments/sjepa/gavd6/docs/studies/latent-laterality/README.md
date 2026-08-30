# Latent Laterality

**Study 02: predictive motion representations under unknown left/right
correspondence.**

> **Status:** the gauge-neutral/validity-aware local implementation, persistent
> sequence benchmark, duration-model gate, source-route screen, and gated
> SG-JEPA objective are implemented and CPU-smoke-tested. Real AMASS/GAVD gates
> and GPU training remain to run on HAIC. The core deliverable is an AMASS
> benchmark and a predictive representation evaluated for relative-gauge and
> output calibration. A GAVD
> natural-event result is included only if a blinded audit passes its evidence
> gate. This study does not claim force prediction or clinical validity.

## Overview

A skeleton sequence is not the body itself. It is a table whose rows have
names such as `left_ankle` and `right_ankle`. A monocular pose pipeline can
retain plausible coordinates while temporarily assigning those two names to
the wrong limbs. The physical motion has not been mirrored; the observation's
semantic convention has changed.

Latent Laterality asks:

> **When bilateral token identity changes locally and is not directly
> observed, can a predictive motion model recover the identifiable relative
> convention, preserve side-sensitive information for later anchored use, and
> avoid inventing the absolute sign it cannot identify?**

The proposed Semantic-Gauge JEPA (SG-JEPA) evaluates both possible token
assignments, estimates a temporal posterior over their relative relation, and
uses that posterior while predicting masked latent motion. It carries two
representation types:

- an **even** channel that is invariant to bilateral token relabeling; and
- an **odd** channel that changes sign and can be transported between blocks.

With an independent anatomical anchor, the odd channel can produce a signed
right-minus-left kinematic estimate. Without one, the honest output is a
symmetric distribution or the quotient $\{y,-y\}$, not a confident arbitrary
side.

```text
canonical motion z ── semantic corruption P^g ── observed token sequence x
                                                    │
                             relative-gauge posterior q(g_i XOR g_j | x)
                                                    │
                                  ┌─────────────────┴────────────────┐
                                  │                                  │
                         even representation h+            transported odd h-
                                  │                                  │
                         side-agnostic tasks             anchor? ── yes ── signed y
                                                               └── no  ── {y, -y}
```

The core result is deliberately narrower than the original seven-document
plan. AMASS-Gauge supplies exact controlled token-corruption paths and
corpus-qualified identity-disjoint evaluation. A small GAVD audit asks whether coherent local
events occur naturally, but GAVD has no anatomical-side ground truth, kinetics,
or reliable participant identifiers. A future side-anchored force cohort is a
separate study, not a two-week dependency.

## Why this is different from the previous Ideas 05 + 09 attempt

The earlier GaitParity direction joined two useful but different questions:

1. [Idea 05](../../../notes/archive/portfolio-ideas/ideas/05-signed-laterality-decodability/) probed a frozen
   encoder for a coordinate-derived signed laterality axis and tested whether a
   known anatomical mirror flipped its decoded sign.
2. [Idea 09](../../../notes/archive/portfolio-ideas/ideas/09-reflection-equivariant-symmetry-axis/) built a
   known, fixed anatomical reflection into paired JEPA layers and separated
   exact even and odd features.

Both assumed the transformation was supplied: construct the mirrored input
$Mx$, test or enforce the expected output, and compare encoder equivariance
with an output-level correction. The repaired-AMASS program retained that same
known-reflection question.

Latent Laterality addresses a different inference problem. The action $P^{g_k}$ is a
latent, potentially time-local *token correspondence*, not a physical mirror.
The model must infer relative correspondence, represent uncertainty, and obey
an anchor-or-quotient output contract. The old fixed-reflection architectures
are therefore decisive controls rather than earlier versions of SG-JEPA.

This revision is designed to be more directly relevant to representation learning because it tests
whether structured uncertainty belongs inside the predictive objective, beyond
preprocessing or a swap corrector. It is designed to be more directly relevant
to biomechanics because
it separates preservation of a side-sensitive quantity from permission to
name its anatomical side. That distinction prevents a plausible magnitude
from being reported with an unjustified sign. The two-week experiment uses
interpretable kinematic functionals; measured force and clinical usefulness
remain future validation.

## What would be scientifically significant

A broad real-world representation claim requires all three layers of evidence
to align:

1. **Phenomenon:** local coherent token-convention errors are consequential in
   the controlled benchmark, and a probability-sampled video audit supports a
   nontrivial occurrence rate.
2. **Method:** SG-JEPA improves a common-target representation metric and
   calibrated relative-gauge inference over transparent temporal correction,
   ordinary S-JEPA, fixed reflection equivariance, and generic paired models.
3. **Decision:** anchored outputs recover signed kinematics while unanchored
   outputs retain the correct global ambiguity.

If a simple corrector matches SG-JEPA, the simpler method is the result. If
events exist only under severe synthetic corruption, the conclusion is a
controlled robustness benchmark. If no natural events are verified, no
real-world pose claim is made.

## Current evidence boundary

`outputs/repaired-jepa-seed7-v2` contains four seed-7, 100-epoch histories and
checkpoints at roughly 822k trainable parameters. The saved features have
nonzero variance, and the tied models pass their programmed commutation audit.
However, the checkpoints predate the branch-specific orbit-closed mask repair;
the directory's `run_config.json` and `summary.csv` were overwritten by the
last standard-only invocation; test evaluation was disabled; and no semantic
permutation, calibration, or downstream biomechanics endpoint was run.

These artifacts establish scaffold and architecture readiness only. They do
not rank the models and must not be presented as repaired-baseline results.
The [working proposal](../../../notes/latent-laterality/proposal.md) gives the
exact audit.

## Immediate execution gate

Run the [frozen-encoder swap probe](swap-probe.md) before beginning SG-JEPA
training. It is the smallest validation-only experiment that can establish
whether temporary bilateral token-name swaps are consequential and recoverable
under the declared corruption contract.

## Documents

- [proposal.md](../../../notes/latent-laterality/proposal.md) is the
  general-audience scientific case. It
  states the question and hypotheses, situates the work, audits current
  evidence, defines the datasets/method/experiments, and limits the claims.
- [theory.md](../../../notes/latent-laterality/theory.md) gives the typed
  transformation model, the corrected
  identifiability result, temporal synchronization, parity projections,
  predictive losses, statistical estimands, failure assumptions, and a full
  technical glossary.
- [experiments.md](../../../notes/latent-laterality/experiments.md) is the
  research experiment guide. It maps data and code,
  specifies corruption artifacts and model arms, separates available commands
  from interfaces still to implement, budgets cluster time, and explains how
  to interpret each result pattern.
- [implementation-tutorial.md](implementation-tutorial.md) gives the ordered
  code-level migration from the legacy AMASS artifacts to gauge-neutral,
  validity-aware baselines and the harder sequence-level AMASS-Gauge benchmark.
- [haic-run-guide.md](haic-run-guide.md) gives the exact benchmark gate,
  three-route source screen, common readout, and gated SG-JEPA commands.

## Definition of done

At day 14, a complete study has:

- immutable data/code/corruption manifests and corpus-qualified identity-disjoint splits;
- no-copy mask, gauge-neutrality, oracle, and checkpoint-reload tests;
- three training seeds for SG-JEPA and its strongest prespecified baseline;
- one sealed test evaluation with common corruption draws;
- identity-clustered uncertainty, per-seed results, and calibration plots;
- an explicit simpler-method/null-result branch; and
- GAVD claims only if the preregistered human-audit gate passes.
