# Skeptical ASTAD Review: Positive Completion Framing

## Verdict

**The positive framing is honest and scientifically useful, with two required corrections.** A defensible positive writeup exists. The manuscript already does the hard separation work most such papers skip: it distinguishes completion gains from detection, isolates them from physiology and causal inference, and refuses to let effective-capacity confounds masquerade as cross-leg information. The arithmetic checks. The required fixes are tightening two phrases, not retracting any result.

## What the positive claim actually is, and why it holds

The core claim is narrow and supported: *under the declared fitted predictor families, both-leg access reduces held-out normal completion error.* Two instances support it.

The laboratory result is the strongest leg to stand on because it has genuine participant separation, a different measurement system (sagittal angles, not image coordinates), and a majority-of-units effect. The own-to-both reduction of 10.454 squared degrees is reproduced exactly in the table (201.038 − 190.584 = 10.454), the relative figure recomputes correctly (10.454 / 201.038 = 5.200%), and the per-fold means in the numerical summary match the table to the digit. The interval [7.154, 13.387] excludes zero, every fold improves, and 109/128 participants improve. This is a real completion result and should be acknowledged as one — I do not require a positive *detection* finding to grant it.

The GAVD masking result (A vs U on E_leg: 0.065020 → 0.060009, 7.71%, 27/28 sources, interval [6.36, 9.00]%) is also arithmetically faithful and internally consistent across the manuscript, audit, and CSV. The 19.5% own-to-both figure on E_leg likewise reconciles: (0.080769 − 0.065020) / 0.080769 = 19.5%.

## Where the manuscript correctly polices itself

The draft pre-empts most of my objections, which is what makes it credible rather than promotional:

- It names the **effective-capacity confound** explicitly and repeatedly (ranks 44 vs 62 external; 111 vs 141 GAVD at equal 72/154 width), and refuses the cross-leg-information interpretation that a naive reader would reach for.
- It declines to **cherry-pick the best policy**: it leads with the U-trained predictor "rather than selecting the largest gain among the three policies," and reports the inconclusive U-bank A-vs-U result ([−0.95, 2.58]%, 16/28) right beside the favorable E_leg one.
- It keeps the **negative primary visible and primary**: S minus M is −0.118 (focal) and −0.076 (broad), both intervals excluding zero, and the fusion loss replicates externally (−0.066). The paper does not bury this or relabel the completion diagnostic as the original endpoint — it states outright that emphasis shifted to a *predeclared secondary* after inspection.
- It blocks the **physiology overreach** at every turn: shared body reference, pose estimates rather than true anatomy, mechanics-not-synergies caveat, and the timing diagnostic's honest self-destruction (all 24 outer adequacy gates fail; the one zero-excluding offset "compares two predictions from a model that is inadequate on the aligned paired inputs").

## The two phrases that need correcting

**1. "changed-primary" language in the abstract/introduction.** The abstract says the analysis "gives greater emphasis to a predeclared secondary completion diagnostic after outcomes were inspected." The audit is cleaner: these are "descriptive intervals for comparisons examined after outcomes" that "do not retrospectively become a primary endpoint." The manuscript's own introduction line — "This draft gives greater emphasis..." — is defensible only if *every* completion interval is flagged post-outcome *at the point of claim*, not only in the abstract. The E_leg 19.5% result in §Results and the A-vs-U 7.7% result carry the correct "post-outcome" / "conditional" flags inline; keep them. The minimal fix is to ensure the laboratory 5.2% claim in the abstract inherits the same "post-outcome descriptive" qualifier the body gives it, so the abstract cannot be read as a pre-registered primary win. This is a wording tightening, not a result change.

**2. Unsupported masks/generalization claim.** The sentence "Anatomical masking improves challenge-task completion by 7.7%" is fine. The risk is the reader generalizing it. The audit already supplies the correct hedge the manuscript should mirror verbatim: "E_leg resembles part of A's training design, so this is also evidence of task alignment rather than a surprising universal generalization result." The manuscript gestures at this ("not a claim that one task's larger loss or gain is intrinsically better") but should state the train/eval alignment explicitly where A-vs-U is first claimed, so "anatomical masking helps" cannot be read as a generalization result. Again: a sentence, not a refit.

## Actual contradictions versus future experiments

I find **no actual contradiction** between the positive framing and the frozen numbers. The things a hostile reviewer would flag — fusion loss, timing failure, previously-explored normals, no participant-level independence in GAVD, STG-NF adapter weakness — are all disclosed as *limits of the current result*, not as contradictions of the completion claim. The calibration question (why does good completion not improve detection?) is correctly marked as a *future* normal-only calibration study requiring a freshly frozen protocol, not something the present data resolves. Do not let a reviewer conflate "the paper cannot yet explain the fusion loss" with "the paper's positive claim is unsupported" — those are different, and the draft keeps them different.

## Table check

Table \ref{tab:detection} is consistent with the prose: M dominates S on both endpoints, no score beats M, and the masking rows (S under U/J/A = 0.576/0.620/0.631 focal) support "no demonstrated fusion advantage." The attrition table's framing ("GAVD units are upload IDs, not verified participants") is the right honesty and is load-bearing for the generalization caveat.

## Recommendation

Accept the positive reframing with the two inline wording corrections above. Preserve all frozen results, intervals, and the negative primary exactly as reported. The paper's value is precisely that it demonstrates useful completion *while* refusing the four tempting overclaims (detection, physiology, cross-leg causal information, generalization) — that restraint is the scientific contribution, not a weakness to be edited away.