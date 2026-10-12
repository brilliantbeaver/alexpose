# Independent review of the appended research direction

Reviewed 28 September 2026. Draft: `writeup/RESEARCH_DIRECTION.md`, 1,840 words when inspected. Line references below identify this version. The overview, draft and assets were not edited.

**Verdict:** The narrowed direction is scientifically defensible and the account of prior work is fair. The draft avoids claiming a JEPA advantage, clinical benefit, certified bounds or exhaustive novelty clearance. Before delivery, fix the observation-model claim, reporting decisions and comparative testing rule. Clarify the pilot's role and the JEPA initialization sufficiently to keep the experiment credible. These changes need a few replacement sentences, not a larger section.

## Required revisions

### 1. State exactly which evidence an alternative must fit — lines 15, 39 and 43

“Fit the visible evidence” is broader than the proposed implementation. A skeleton may agree with detected 2D landmarks while contradicting the silhouette, local image appearance or another visible body part. The search can establish ambiguity under a specified landmark-and-kinematics observation model; it cannot establish that all information in the original video is ambiguous. Likewise, occluded landmarks predicted by the initial model must not silently become measured constraints.

Suggested replacement at line 39:

> The first search would match detected 2D landmarks judged visible, within separately calibrated error tolerances; estimated hidden landmarks would not be treated as observations. Its alternatives demonstrate ambiguity under this landmark-and-kinematics model, rather than equivalence of all the information in the video.

Use “fit the landmark observations” at line 15. A later silhouette constraint is a possible extension, not necessary to the first experiment. The existing distinction between kinematic admissibility and dynamics should remain.

### 2. Define changed, unchanged and indeterminate — lines 5, 47 and 49

The motivating HCI distinction is unsupported comparison versus unchanged movement, but the protocol currently defines retention without explaining when either conclusion is allowed. An interval crossing zero is not evidence of no change. The observation-only null also makes this distinction central to the proposed false-report metric.

Add a concise decision definition:

> Using a pilot-defined range of changes too small to resolve, report a change only when the measurement interval lies wholly beyond that range, and no resolvable change only when it lies wholly inside it; otherwise the comparison remains indeterminate.

These are measurement decisions, not clinical thresholds. For claims about which leg changed, evaluate joint left/right interval coverage or prespecify the multiplicity adjustment. Retention must count the complete trial pair, as already proposed. Keep eligibility failures, declined comparisons and incorrect retained reports as separate outcomes.

### 3. Resolve the testing-order ambiguity and unit of uncertainty — lines 31, 47 and 49

Line 49 announces change error as the primary comparison, then describes preservation within a margin as a gate for trajectory superiority. It is unclear whether success requires improved change error, noninferior change error plus improved trajectories, or both. Those are different claims. “Matched pair retention” also needs a retention target/rule fixed without final reference labels.

Choose one hierarchy explicitly. A version consistent with the current text is:

> At a retention target fixed on development data, first require the upper confidence limit on the increase in per-leg change error to remain below a prespecified preservation margin. Only then test whether trajectory error improves over the locked baseline; a claim of better change measurement would require a separate superiority result.

Alternatively, make improved change error the sole primary success criterion and leave trajectory accuracy as a secondary safeguard. Do not leave both interpretations implicit. State that calibration accounts for repeated pairs within people, and confidence intervals resample people while retaining all their pairs and rendering variants. Evaluating post-selection coverage, as proposed, is essential; it does not by itself supply a coverage guarantee for every subgroup.

### 4. Make the pilot limitation operational — lines 23, 33 and 55

The [OpenCap Monocular primary manuscript](https://arxiv.org/html/2603.24733v1) supports public-data feasibility, but §2.4 describes only ten healthy adults, six walking trials each, with trunk-sway modifications. Those modifications need not produce enough knee-excursion change to defeat an always-zero result. Section 2.2.2 also states that refinement weights were tuned using the prior OpenCap data. The same cohort therefore cannot establish an independent comparison for that existing pipeline.

Suggested addition:

> This ten-person dataset would serve only as a feasibility pilot: first verify enough reference-resolvable knee changes, and reserve a new cohort or independently held-out dataset for the final comparison.

The final design should prespecify the distribution of reference change magnitudes and the share of null versus changed pairs. Otherwise a pooled score can again favor zero simply because changes are usually small. The proposed sample-size calculation and actual atypical-gait evaluation remain necessary after this gate.

### 5. Make the JEPA comparison implementable — line 45

“Matched JEPA-initialized version” leaves the checkpoint, input representation and pretraining source unspecified. v08's student consumes projected 2D skeleton tokens; it is not directly a 3D/SMPL temporal refiner. Recovering its checkpoints does not remove that interface mismatch.

Suggested clarification:

> Both refiners would use the same pose-sequence architecture and input convention; the comparison would vary task-specific JEPA pretraining, with its data and computation reported separately from supervised adaptation.

If actual v08 weights are intended, state that a compatible 2D encoder would supply features alongside the fixed 3D reconstruction, with the same feature inputs available to its control. Do not imply a direct transfer into an unspecified 3D architecture. The simpler alternative is to keep JEPA as a later ablation once the fixed-pipeline measurement study succeeds.

## Smaller clarity and delivery edits

- Line 17's “did not identify” claim appropriately limits novelty. Keep the supporting audit link, but avoid strengthening this to “first” or “highly novel” in the title or final response. CLOSURE and SLUE are fairly used as methodological precedents; CLOSURE explicitly distinguishes its sampled approximation from outer bounds.
- Line 49 is currently the densest paragraph for the CS/HCI audience. The concrete reporting rule above should help. Introduce “pair retention” as the proportion of complete comparisons the system agrees to report.
- Line 55's “within reach” is acceptable as a hypothesis, provided the known-camera, short-sequence restriction and data dependency remain nearby. Runtime and optimizer success must be measured, not inferred from the small refiner.
- Line 57: replace “would become useful only after” with “would be justified in this project after.” The current wording makes an unnecessarily universal claim about Qwen/RL.
- Line 59: “A well-controlled negative result could identify failure conditions…” is more accurate than promising that any negative result will do so. Keep the explicit absence of demonstrated fall-risk or clinical benefit.
- Figure 2's written specification correctly distinguishes observation-only and real-movement comparisons. The image should show two different movement states with the same independent reference for clear/occluded copies, and label every arrow's target. I did not inspect the unfinished image or rendered PDF. Confirm the original three-page overview is unchanged, the extension fits four legible pages including references, and the search-record link resolves when generated.

## Scope of checks

I checked the complete appended prose against the earlier novelty audit and established v08 evidence. I additionally inspected the primary OpenCap Monocular methods/data statements and CLOSURE's approximation statements. SynthGait-19K's specified appendices and DeepGaitLab's baseline adjustment were previously confirmed in directly downloaded primary HTML. No new numerical results are reported in the draft. I did not run optimizers, validate dataset downloads, estimate sample size, or inspect final pagination; those remain distinct implementation and production checks.
