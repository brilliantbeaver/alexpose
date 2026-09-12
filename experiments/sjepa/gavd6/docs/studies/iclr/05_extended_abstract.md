# When Predictive Features Miss Motion: Measuring Temporal Accessibility in Skeleton JEPA

**Extended abstract — evidence-grounded research draft.** The proposed
distillation method has no completed real-data student result.

Predictive representations should be evaluated against observables with explicit
meaning. We connect two gait studies and develop a measurement path toward
selective future distillation.

In a retained skeleton-JEPA study with 625 clips from 93 source videos, trained
predictors favor correct-clip hidden features over mismatched targets in all
375 diagnostic checks. Yet a common expanded readout of signed left–right
coordinate-speed contrast falls from 0.222544 R² at initialization to
0.100777–0.114209 across five training conditions. Every condition loses in every
seed. Separately, a repaired RGB-conditioned experiment on 50 clips from 43
sources finds a skeleton increment of −0.00024242 R², with conditional 95%
source-bootstrap interval [−0.00147402, +0.00091980]. The studies use different targets and R² references. Neither establishes
whether a skeleton-only student benefits from future video supervision.

We implement a source-held panel comparing a skeleton history block with
current posture and observation support. Jointly regularized linear models, training-only preprocessing,
matched temporal controls and exact baseline fallback constrain the comparison.
The named history-block increment is +0.001994 R², interval [−0.006673, +0.012608],
with 68.35% positive paired draws: no supported temporal lead.
This block also changes valid-conditioned confidence features and their
regularization, so the estimate does not isolate motion from observation quality.
The operational finite-model comparison does not estimate conditional mutual
information.

Reflection and time reversal clarify the required observables. The existing
laterality target changes sign under anatomical reflection but is unchanged by
time reversal. Exact parity projections can enforce these rules even on zero
features. A controlled 48-source fixture therefore tests utility separately:
current posture and order-even summaries give R² = 0 for a signed-velocity
target, while ordered velocities recover it nearly exactly. This is a deliberately
constructed readout calibration, not JEPA training or human-motion forecasting.

We propose selecting a small set of teacher directions by held-source temporal
value, retaining an exact no-transfer option, then testing their benefit to a
matched S-JEPA student. Future privileged distillation and equivariant features
already have close precedents, notably SGDD and seq-JEPA. The prospective
contribution is evidence that a student-aligned temporal selection criterion
improves transfer or reliably rejects unsuitable supervision. That claim requires
a complete real student comparison, common observable outcomes and independent
source or participant confirmation. The current work preserves the negative findings as empirical motivation.
