# Laterality v2.1 evaluation contract

This folder is a clean-room replacement for the transductive laterality notebooks. It does not alter or import the old experiment scripts. The protocol supports one narrow empirical statement: post-development, within-GAVD cross-validated performance on source videos excluded from representation training and read-out fitting.

It does **not** establish unseen-person, diagnostic, or clinical generalization. GAVD does not provide a persistent subject identifier, and this project does not infer identity. Condition names are dataset annotations used to balance source-level folds and in one explicitly named confounding control; they never enter the primary training objective.

## Non-negotiable invariants

1. Split source videos before representation training.
2. Keep every sequence, mirror, and augmentation from a source in one fold.
3. Train a new encoder for every outer fold and optimization seed.
4. Use only outer-training sources to fit preprocessing statistics, select ridge regularization, and fit a read-out.
5. Open each outer-test fold only for its final evaluation.
6. Weight sequences so every source contributes equal total weight.
7. Resample source videos—not sequences or folds—for data-uncertainty intervals.
8. Keep optimization-seed variation separate from source-sampling uncertainty.
9. Treat the target-component oracle as self-consistency, not a learned baseline.
10. A paper-mode artifact is not submission-ready until the governance gate is resolved.
11. Square mirror residuals within each checkpoint before aggregating seeds; signed errors may never cancel across seeds.
12. Require absolute predictive utility in addition to relative improvement over a paired initialization.
13. Keep analytic output oddness, probe-level odd behavior, and direct token equivariance as distinct claims.

## Target contract

The target is computed only from paired-valid left/right transitions. For each anatomical pair, it calculates median normalized-coordinate velocity on frames where both sides are observed at both endpoints, then forms

$$
c_k = \frac{m_{L,k}-m_{R,k}}{m_{L,k}+m_{R,k}+\epsilon},
\qquad y=\operatorname{mean}_k c_k.
$$

All five registered motion pairs must each supply at least eight common-valid transitions. The hip landmarks define the pelvis reference and remain available to the encoder, but are excluded from the target: pelvis centering makes their two velocity norms algebraically identical. Eligibility depends on computability and pose coverage, never on the target's sign or magnitude.

This prevents zero sentinels from becoming target signal. The implementation must pass three tests before a real run:

- changing coordinates under invalid entries cannot change `y`;
- mirroring coordinates and validity masks negates `y`;
- mirroring twice restores coordinates and masks.

The quantity is a coordinate-derived representation probe, not a validated clinical gait measure.

## Input inventory contract

Paper mode reconciles the official annotation inventory to the local derived-pose cache before QC. Counts, privacy-preserving inventory digests, pose-model identity, visibility threshold, and the extraction-version census are frozen in `config/protocol.json`; added, removed, or provenance-mismatched archives fail closed. The attrition report therefore starts at 666 annotation files, records the 642 available pose archives, and then applies the locked pose/target QC rules. The suite never reads, displays, or redistributes raw videos, public URLs, or identity-bearing frames.

## Split contract

The split builder first collapses the cohort to one row per `video_id`, verifies that each source has one dataset annotation, and applies stratified K-fold to that source table. This avoids the sequence-balanced failure mode where one long video can stand in for an entire condition within a fold.

Inner source folds are used only to select read-out regularization. The encoder uses a fixed, predeclared update budget on every outer-training source; neither laterality labels nor outer-test performance selects its stopping point.

## Training contract

The primary encoder is label-blind S-JEPA + VICReg. One training epoch visits every outer-training source once in random order, samples one sequence within that source, and uses source-uniform padding only when required to keep the optimizer-update count identical across folds. Sampling, masking, and reflection use separate deterministic RNG streams so the two variants remain paired. The variants are:

- `vanilla`: no reflection augmentation;
- `reflection_augmented`: sample-consistent anatomical reflection with probability 0.5.

The exact frame-averaged read-out and encoder use the same trained checkpoints and require no additional training.

## Evaluation contract

Each fold fits read-outs using only outer-training rows. The predeclared primary is `vanilla` training with `learned_single_free`. It uses one original-view feature vector and a centered linear read-out with an intercept; no output-odd constraint is imposed. Its features still use registered BlazePose left/right pairs, so success characterizes an anatomy-aware encoder-plus-probe pipeline rather than symmetry discovered from raw pixels.

The capacity-matched parity factorial uses (z^-(x)=(z(x)-z(Mx))/\sqrt{2}) and (z^+(x)=(z(x)+z(Mx))/\sqrt{2}). Learned and paired-initial encoders each receive single/free, two-pass odd/free, two-pass odd/zero-origin, and two-pass even/free lanes with equal feature dimension. `learned_two_pass_odd_zero` is the constructed repair. Its feature parity and output oddness are imposed; they are never evidence of emergence. Comparing odd/zero-origin with the identical odd features and free read-out isolates the complete origin constraint. Odd-versus-even and learned-versus-initial contrasts identify parity utility conditional on their wrappers, not intrinsic encoder equivariance. The Reynolds-framed odd lane remains explicitly exploratory.

Measured controls are visibility/missingness, pose acquisition and extraction version, dataset annotation, their concatenation, and learned-single plus measured nuisances. The incremental combined lane tests whether learned features add held-out utility beyond measured nuisances. It cannot rule out unmeasured or nonlinear shortcuts. Global contextual pooling is descriptive learned content, not a pure nuisance control. The dataset-annotation lane is not a diagnostic model, and the target-component oracle is not empirical evidence.

Primary predictive inference computes source-balanced R² separately for every registered checkpoint and only then averages across seeds. Source-cluster bootstrap draws preserve all sequences and registered seeds belonging to each sampled source. The mean-prediction ensemble is reported separately as a secondary operational estimand. Intervals are conditional on the fixed cross-fitted checkpoints; they do not propagate new-seed, split-allocation, or full retraining uncertainty.

## Direct representation-equivariance contract

For every held-out sequence and checkpoint, let (Z(x)) and (Z(Mx)) be target-encoder tokens. Let (S) apply the full 33-joint anatomical permutation while leaving latent channels unchanged. On exactly the tokens valid in both aligned views, the strict error is

\[
q=\frac{\lVert Z(Mx)-S Z(x)\rVert_C^2}
        {\lVert Z(Mx)\rVert_C^2+\lVert S Z(x)\rVert_C^2}.
\]

No channel fitting, Procrustes rotation, sign search, centering, or read-out is allowed. Zero is exact strict equivariance, unrelated equal-energy representations are near one, and a zero-energy collapse is rejected. Errors are formed per sequence and seed before source-balanced aggregation. The learned checkpoint is paired with its exact initial target encoder; vanilla and reflection-augmented variants must share that initialization for each fold and seed.

Probe-level native behavior requires all of: a lower confidence bound above the registered absolute R² minimum, an upper bound below the native output-error margin, and a positive learned-single versus random-single R² lower bound. Strict checkpoint representation equivariance additionally requires the learned token error to improve over paired initialization and its absolute upper bound to fall below the separately registered representation-error margin. Only their conjunction supports the narrow phrase “training-induced symmetry conditional on the pose schema and architecture.”

Constructed learned content requires exact oddness for every prediction and seed, positive absolute constructed R², and improvement over the identically constructed initial-encoder floor. Exactness alone proves only the implementation. Reflection-versus-vanilla contrasts are paired ablations of this recipe; nonsignificance is not evidence of equivalence. All thresholds and null-result rules are stored in `config/protocol.json` and may not be tuned after paper results are observed.

## Governance contract

`governance/status.json` is deliberately unresolved. Notebook 00 checks it and produces a hard submission-readiness verdict. A statistical redesign cannot substitute for institutional ethics and data-use determinations. Derived poses and embeddings must not be presented as inherently anonymous or redistributed until their release status is reviewed. Any external dataset requires its own dataset-scoped governance record; the GAVD review cannot authorize it.
