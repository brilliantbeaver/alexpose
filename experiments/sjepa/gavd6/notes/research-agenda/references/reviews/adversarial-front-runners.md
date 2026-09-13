# Adversarial review of Proposals 1 and 2

Reviewed drafts: `01-preserve-real-movement.md`, `02-ambiguity-witnesses.md`, and `00-evidence-and-execution.md`. Line references refer to the versions initially read during this review. Following explicit orchestration, this reviewer applied the revisions recorded below to Proposals 1 and 2.

The concepts are worth keeping, but the drafts currently contain several gaps that could turn a visually persuasive result into an unsupported scientific claim. These are required substantive revisions, not copy edits.

## Proposal 1: preservation and repair

### 1. The uncertainty output is promised but not specified

**Where:** Lines 13 and 47.

**Problem:** The method predicts a residual-retention gate. That produces a trajectory, but not an uncertainty statement. On two examples with identical available inputs and different underlying truth, the same gate output is inevitable. This establishes non-identifiability; it does not automatically produce the claimed uncertainty flag. An interpolated mean can appear confidently plausible while matching neither explanation.

**Required edit:** Add an explicit event-evidence probability or abstention output. Define the abstention threshold using a separate calibration partition, and score both errors and coverage. Construct exactly identical full-input pairs, including RGB, timing, quality and metadata, with a balanced assignment of the hidden explanation. The best attainable evidence probability on that controlled balanced fixture is one half. Do not claim that its calibration transfers to natural GAVD event prevalence. Separate uncertainty arising from missing evidence from ordinary model misclassification.

### 2. The paired construction needs an exact mathematical contract

**Where:** Lines 11 and 35–37.

**Problem:** “Same coordinate perturbation” is insufficient when joint-angle edits pass through forward kinematics, projection, a detector and a representation converter. The resulting raw trajectories or confidence values could differ in detectable ways. “Identical wherever construction permits” leaves the central information boundary loose.

**Required edit:** Define a clean trajectory x, its edited trajectory x_e, a camera c and an observation tensor z. Assign exactly the same z and metadata to two cases. Case A uses rendered video of x_e and reference truth x_e. Case B uses video of x and reference truth x. A declared common corruption can be added to z in both cases. Verify tensor identity directly. The two RGB videos should have the same camera and rendering settings; their disagreement is the intended extra evidence. Call this an independent observation channel, not statistically independent ground truth.

### 3. Real movement and tracking noise must also occur together

**Where:** Lines 35–45.

**Problem:** The binary fixture may permit a clip-level choice between raw input and the prior output. This does not demonstrate simultaneous preservation and repair. It could succeed when every real-event example is otherwise clean and every non-event example is corrupted.

**Required edit:** Add a factorial design: event absent/present crossed with independent observation noise absent/present. Include cases where noise and the true event overlap in joint and time. Evaluate a clip-level gate as a baseline. A per-joint method must preserve the true event while reducing independent error in the same clip, including overlap cases. Balance the factors within person and rendering profile.

### 4. The matched-error frontier cannot select operating points on test truth

**Where:** Lines 41 and 53.

**Problem:** Tuning each method's restoration strength against test reconstruction error can select favorable operating points using the answer. The retention denominator is also ambiguous unless the event descriptor and its reference value are specified.

**Required edit:** Define the event magnitude as the descriptor change from the paired unedited reference, and specify how output error is normalized by that nonzero change. Report signed error and overshoot as well as normalized retention; do not silently clip negative scores. Choose fixed operating points on calibration data for a target amount of noise removal, then report both achieved error removal and retention on test data. A complete prespecified strength curve can be descriptive, but do not retrospectively pick its best test point as the primary method result.

### 5. The claimed held-event and held-prior tests may be used twice

**Where:** Lines 39, 53 and 64.

**Problem:** The 48-hour go/no-go decision requires improvement on the held-out event family. If that same family guides method changes throughout the week, it is no longer untouched evidence of event-family generalization. Likewise, “holds beyond the prior used to train the adapter” is stronger than independently retraining a gate for each prior.

**Required edit:** Either use an in-family held-person pilot and preserve a second event family for the final test, or use three families with distinct training, development and final roles. Freeze the gate and all normalization when transferring from the first prior to the second if claiming transfer across priors. Otherwise describe it as replicated adaptation across two priors.

### 6. Raw/repaired interpolation has a limited correction range

**Where:** Lines 29–31.

**Problem:** A gate mixing raw and prior positions cannot produce a correct position outside their interpolation segment. Missing coordinates cannot be retained. Jointwise mixing can also break articulated consistency before a repair projection changes the event again.

**Required edit:** State whether the gate is limited to [0,1], how absent raw observations are handled, and whether a final kinematic projection is applied. Score the final projected trajectory, including its event attenuation. Include the best possible gate between raw and prior as an oracle ceiling; if that ceiling cannot satisfy the target, this gate architecture is the wrong intervention.

## Proposal 2: constructive ambiguity

### 7. A new observation can reject both candidates

**Where:** Lines 57 and 83.

**Problem:** Neither found witness trajectory needs to equal the true trajectory. An additional camera may reject both. Even if it rejects exactly one, a third compatible motion could still support the opposite conclusion. Rejecting one witness does not establish that the movement finding is now identifiable.

**Required edit:** Score elimination of the specific competing pair, separately from remaining ambiguity. Select the reveal using current observations and candidate disagreement only. Then reveal evaluator-held evidence and rerun search within a fixed additional budget. The result is “this witness was rejected” or “no new competing witness found,” never proof of resolution from search failure. For constructed fixtures containing the reference trajectory, explicitly label that narrower setting when measuring preservation of the correct candidate.

### 8. Fully observed 3D controls need a descriptor margin

**Where:** Lines 37, 41 and 49.

**Problem:** With nonzero positional tolerance, a near-threshold descriptor can change while all coordinates remain within their admissible bounds. The statement that an opposite answer must violate a constraint is true only when the descriptor is fixed exactly or bounded away from the threshold. SMPL axial rotations also need not be determined by joint positions.

**Required edit:** Define both descriptors directly from declared 3D joint geometry, including a trunk and pelvis axis construction and handling of degenerate configurations. Use exact-coordinate cases only as unit checks. Add harder negative controls with partial observations whose independently computed descriptor bounds remain entirely on one side of the threshold. Require a prespecified margin. Interval bounds or an exact geometric argument, rather than a prior's failure to find an alternative, must establish those control labels.

### 9. Benchmark construction can favor the proposed search

**Where:** Lines 39 and 49–51.

**Problem:** If ambiguous cases are produced with the same prior, optimizer or learned initializer later being evaluated, high recovery can reflect matched construction artifacts. Two similar optimization procedures do not necessarily solve this.

**Required edit:** Generate the primary known-ambiguity set using an articulated geometric construction independent of the proposed prior and search head. Use a separately held construction family and natural-motion-derived cases for transfer. Exclude global reflection, scale changes and left-right relabeling by contract rather than merely noting them after scoring. Report results by construction family and descriptor margin.

### 10. “Wrong on a compatible alternative” is not a natural error prevalence

**Where:** Line 51.

**Problem:** A confident answer may match the actual motion while being unsupported by the observation because another motion is compatible. That is an evidence-validity failure, but not automatically an observed physical error. Constructed alternative cases do not estimate how often either world occurs naturally.

**Required edit:** Distinguish actual-answer error on a labeled fixture from overclaiming uniqueness under the admissible set. Score false certainty on balanced exact-observation pairs with a declared probability or coverage metric. Do not claim GAVD calibration or population ambiguity prevalence from the constructed set. The witness itself proves only the existence of two admissible answers.

### 11. The equal-cost comparison is overconstrained

**Where:** Line 55.

**Problem:** Iteration count, candidate count and elapsed runtime cannot generally all be matched across a neural generator and a geometric optimizer. Equal iteration count can favor the more expensive method; equal GPU time alone can penalize a CPU-based geometry method unfairly.

**Required edit:** Choose a primary common wall-clock budget on declared hardware and CPU-thread allocations, including prior sampling, conversion and verification. Report candidate count and GPU/CPU cost as secondary measures. Include search-head training cost and show the number of queries needed to amortize it. A 15-point recovery advantage must hold against the strongest geometric method under this common budget.

### 12. Kinematic admissibility must remain distinct from human feasibility

**Where:** Lines 41, 65 and 83.

**Problem:** Bone lengths, angle bounds and acceleration limits do not guarantee balance, contact validity or dynamically executable human motion. The draft mostly states this correctly, but “actual valid motions” and “possible” can invite a stronger interpretation.

**Required edit:** Use “kinematically admissible under the declared constraints” consistently. Publish each constraint residual. If balance or contacts are added, distinguish geometric checks from a force-based physical guarantee. Do not let a learned normality score reject unusual motions from the admissible set by definition.

## Common execution contract

### 13. Use people whenever they are known

The common contract correctly prioritizes independent people, but Proposal 1 allows a “person or motion bootstrap” and Proposal 2 specifies a motion-level bootstrap. AMASS variants from the same person are dependent. Amend both to person-level inference whenever identity is available, grouping all trials and variants within person. Where identity linkage is incomplete, name the weaker recording/trial claim explicitly. A second motion from the same person is not a second person.

### 14. Distinguish adaptation holdout from untouched evaluation

The common contract should say that a family can be absent from gradient training yet still be used for development selection. Report both boundaries. Day-2 branch selection may inspect development families, but final event, observation, activity or teacher-family claims require a separately preserved evaluation boundary. All seven proposals are already alternatives; this addition prevents reuse of the one successful pilot as confirmation.

### 15. The uncertainty target needs a declared sampling distribution

Witness existence needs no prevalence model. Calibration does. Whenever a proposal claims calibrated probabilities, coverage or false certainty, state the distribution being sampled, the source/person units, and whether the result concerns synthetic construction, held mocap people or natural video. Agreement across two estimators does not provide calibration labels. Keep GAVD demonstrations outside physical calibration claims unless a directly observable reference has been specified.

## Recommendation after revision

Keep Proposal 1 as the strongest first pilot, provided the same-clip event-plus-noise factorial and calibration-only operating points are added. Keep Proposal 2 as a distinct low-training alternative, but narrow its clarification claim to eliminating witnessed alternatives unless an independent geometric bound proves more. These changes make each study harder; they also make a positive result considerably more meaningful.

## Applied revision log

| Review items | Substantive change applied |
| --- | --- |
| 1 | Added event-evidence probability, separate calibration threshold, errors plus decision coverage, and a balanced exact-input uncertainty fixture with no natural-prevalence claim. |
| 2–3 | Replaced approximate skeleton matching with identical observation-tensor pairs. Added the event-by-noise factorial, overlapping event/noise, and a clip-level gate baseline. |
| 4 | Defined the event descriptor normalization explicitly. Locked operating points on calibration people and separated achieved test noise removal from retention. |
| 5 | Moved the 48-hour test to held-person development cases. Reserved the second event family for final evaluation and required a frozen gate for cross-prior transfer claims. |
| 6 | Bounded the retention gate, specified missing raw coordinates and final kinematic projection, and added an oracle-mixture ceiling. |
| 7 | Limited clarification to rejecting a specific pair, allowed both candidates to be rejected, prohibited evaluator-truth-based reveal selection, and required a budgeted repeat search without uniqueness claims. |
| 8 | Defined knee and trunk-pelvis descriptors from joint geometry, fixed a physical-time window and margin, and added partial-observation controls with independent geometric bounds. |
| 9–10 | Required construction independent of the evaluated prior, held construction families, explicit gauge exclusions, and separate false-certainty versus physical-error reporting. |
| 11–12 | Made wall-clock cost primary and kept candidate counts and hardware costs secondary. Replaced physical-validity implications with explicit kinematic admissibility. |
| 13 | Required person-level bootstrap in both proposals whenever identities are available. |
| 14–15 | Applied untouched-final-evaluation and construction-specific calibration boundaries locally. Corresponding common-contract wording remains for the lead to incorporate. |

These edits improve the protocols, but do not demonstrate that the proposed methods work. No training or scientific evaluation was performed during review.
