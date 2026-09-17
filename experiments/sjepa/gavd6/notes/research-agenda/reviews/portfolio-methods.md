# Adversarial methods review and second-revision suggestions

Reviewed 14 September 2026 against all seven proposal drafts and `evidence.md`. This is a scientific and readability review, not new experimental evidence. The lead author revised P1, P3, and P6 while the review was in progress; those corrections are distinguished below from remaining suggestions.

## Overall judgment

The portfolio is substantially more defensible than another training run around MoMask. It separates useful information, practical methods, and significance. No proposal yet has demonstrated enough empirical headroom to justify calling an ICLR-level result highly probable. P1 and P2 are reasonable cheap pilots. P3 must remain conditional on a viable joint repair-preservation frontier. P4 has an independent hypothesis but a small final cohort. P5, P6, and P7 have especially close methodological precedents and should stay below the leading pilots.

The shared evidence document correctly preserves the old STOP, distinguishes 15/16 paired flow contrasts from classification accuracy, names the privileged event mask, preserves subject allocations, and does not invent GAVD 3D truth or existing annotations. These are material scientific strengths.

## P1: observation value

**Resolved during review:** The first draft emphasized novel weighted questions alongside forecaster transfer. Because the head predicts every coordinate's error reduction, a new weighted combination is obtained by linear addition. It is a compositional check, not a strong zero-shot task result. The revised Step 5 now correctly requires a frozen selector to transfer to a new forecaster and an observation failure excluded from selector training. Update the short opening question to match this stronger claim.

**Remaining revision:** Specify a materially different second forecaster, not another seed or width of the same head. A temporal transformer versus a DCT/MLP forecasting head is sufficient to make the intended test concrete. Both should be competent on the query interface before freezing the selection policy for comparison. The same observation-condition shift must apply to all competing selectors.

**Information fairness:** Pose uncertainty, motion energy, and flow disagreement used before selection must come from the same coarse initial observations. Computing high-resolution flow everywhere in an acquisition baseline spends the query budget just as surely as computing every crop embedding. Compare a fixed image-encoder-plus-temporal-head baseline as well as coordinates and random features before attributing transfer to JEPA.

**Novelty boundary:** Expected observation utility and target-dependent acquisition predate this proposal. [EDDI](https://arxiv.org/abs/1809.11142), [GSM-AFA](https://arxiv.org/abs/2010.02433), [Video Active Perception](https://arxiv.org/abs/2605.01662), and [sensing clocks](https://arxiv.org/abs/2607.01537) should remain explicit. A small in-distribution frame-selection gain is not the headline.

## P2: movement beyond joint positions

No fatal algebraic flaw was found. The natural-motion-first ordering is better than starting from synthetic twists. The main unresolved problem is attribution: surface locations themselves supply pose geometry that joint centers omit, while their histories supply motion. An improvement cannot automatically be attributed to temporal flow.

**Exact revision:** Define the target as the future change in a material point's pelvis-relative projected offset:

`target(h) = [point(t+h) - pelvis(t+h)] - [point(t) - pelvis(t)]`.

Normalize by the same prefix person-height value. This removes ambiguity between forecasting future surface position and forecasting surface transport.

**Add a decisive control:** Give a static-point baseline the same point identities and cutoff locations that the flow representation uses, without their temporal histories. If correspondence locations are privileged in the controlled render, label the baseline accordingly. The existing off-axis-landmark and last-image controls are helpful, but this matched construction specifically tests extra pose geometry versus temporal evidence.

**Keep the external claim narrow:** The GAVD landmark panel can corroborate video-assisted future landmark prediction. It cannot verify missing segment twist or material-point transport between joints. The draft already acknowledges this, and the final figure captions should preserve that distinction.

**Novelty gate:** [MC-JEPA](https://arxiv.org/abs/2307.12698), [JOPAT](https://arxiv.org/abs/2605.23856), [H-MoRe](https://arxiv.org/abs/2504.10676), and [H-Flow](https://arxiv.org/abs/2605.22629) rule out a broad claim based on flow fusion. The compact remedy must outperform equal-budget direct compression. A gain only against joints is an information result, not a new adapter result.

## P3: transferable repair verification

**Resolved during review:** The first draft transferred the edit-based retention metric to untouched natural clips without defining an event-free counterfactual. Revised Step 4 now preserves the old 25% and 15-point condition on the original controlled experiment and separately requires natural-motion MSE and descriptor-error confirmation. This is the correct distinction.

**Remaining revision:** The candidate screen needs a joint repair-preservation oracle frontier, not only a minimum-MSE oracle. A candidate may have substantial MSE headroom while every useful setting erases the event. Conversely, a strong baseline may already preserve so much movement that a further 15-point gain is impossible under the bounded retention definition. Check both axes before launching a verifier. Preserve the declared target when the original condition is reused.

**Exact suggestion:** Add to the first-day decision: “Require feasible oracle settings that jointly exceed the declared repair and preservation comparison, using the actual blocks, candidate menu, and final output rule.”

The corrected true identity, separate completion, fixed masks, observable regions, held repair family, and flow-only comparator are scientifically appropriate. The qualifier about MoMask implementation fidelity versus task usefulness must remain.

## P4: cross-activity memory

**Remaining information confound:** Rendered support videos depend on full SMPL pose rotations and body parameters. A raw 22-joint support baseline omits some of that information, as P2 itself emphasizes. If JEPA wins, the difference could be extra input information rather than better representation of personal motion.

**Exact revision:** Add a strong raw support baseline with full available rotations and body parameters, clearly labeled as a privileged support comparator. Also compare a frozen image encoder with the same temporal pooling and rendered support. The original correct-person versus matched-donor test still establishes personal usefulness, while these additions test why the JEPA memory helps.

The revised eight-person feasibility floor is honestly labeled and cannot be treated as statistical power. About eleven potentially eligible test people remains a real ranking penalty. Showing all person effects and two activity results is essential. [Personalized Pose Forecasting](https://arxiv.org/abs/2312.03528) and the cited motion-style methods make generic personalization an insufficient novelty claim.

## P5: response to added evidence

The proper-score endpoint and caution that useful evidence can increase uncertainty are strong. Residual prediction is a hypothesis rather than a claimed independence operation.

**Exact revision:** Apply the shared exact-duplicate rejection rule before the preliminary failure gate. Otherwise, the pilot can pass by discovering a problem already solved by a rule given to every final comparator. The actual opportunity must survive ordinary duplicate handling and concern overlapping or partially informative observations.

Where possible, let update methods start from the identical frozen initial predictive distribution. This is clearer than merely allowing initial CRPS to differ by one percent. Keep standard end-to-end fusion as a separately identified comparator if it needs its own initial prediction.

If calibrated concatenation or redundancy weighting matches performance, the residual correction remains an implementation variation. The low novelty rating is appropriate.

## P6: ambiguity witnesses

**Resolved during review:** The numerical checker establishes compatibility with joint tracks and geometry, not the entire RGB recording. Revised Step 2 makes this explicit. A video feature can guide search without making its pixels certified constraints.

**Remaining revision:** Give every image-aware ranking baseline the same RGB prefix. Treat geometry-only search as a separate information baseline, and require the JEPA ranker to beat the strongest same-input retrieval/ranking alternative. Otherwise an advantage could come simply from seeing more than projected joints.

The distinction between “no competing motion found” and uniqueness is correctly stated. Likewise, rejecting one witness with an additional controlled view does not reject all possible alternatives. These qualifications should be visible in figure labels, not buried in prose.

## P7: teaching useful targets

The lower-priority positioning is justified. Physical student benefit, held-family decision regret, and comparison with a short actual training pilot are much stronger than reporting latent R-squared. Selecting no teacher everywhere cannot establish beneficial transfer, which the proposal correctly says.

**Mathematical revision:** A difference between residual-error covariance matrices need not be positive semidefinite. Its eigenvectors are not all useful predictive directions. Keep only directions with positive cross-validated error reduction and consider ranks 0, 1, 2, 4, 8, and 16. A forced four-direction minimum can include harmful or unsupported directions when only one or two survive.

**Held-family revision:** Lock the criterion's scoring rule and rank/loss-strength selection procedure before testing the held teacher family. It is legitimate to estimate that new teacher's transform from permitted training data, but this is not the same as never accessing that teacher's features. Do not tune the criterion against held-family physical outcomes and then call its decision regret out of sample.

[LogME](https://proceedings.mlr.press/v139/you21b.html), [VAMP](https://arxiv.org/abs/1707.04659), and the cited complementarity criterion leave little room for residualization itself to be the invention. Novelty should remain around 2/5 until held-configuration decisions deliver substantive new evidence.

## Shared GAVD endpoint and clearer second revision

The annotation budget is now concrete. One remaining requirement is to state how predictions become comparable with those annotations: use a separately identified image-coordinate head trained on permitted AMASS renders or unannotated training recordings, then freeze it before evaluating the small manual panel. Do not fit a new GAVD forecaster using the same 60 clips and describe its errors as external confirmation. Human annotation availability remains a dependency, not completed work.

For readability, make every opening question name the actual decisive test. In P1 that is transfer to a new forecaster and observation failure, not new linear weights. In P2 say “future surface movement” before introducing token budgets. In P3 distinguish the controlled retention result from natural descriptor error at first mention. In P7 explain decision regret as “how much worse our chosen lesson is than the best lesson we later tested” before giving the formula.

Use one concrete example per proposal, followed by the observation, target, and falsifying comparison. Preserve the 48-hour gates but identify them as continuation choices. A final reader should be able to state what would make each idea fail without reading its reference memo.
