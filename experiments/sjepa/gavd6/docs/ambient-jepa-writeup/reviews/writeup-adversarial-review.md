# Independent adversarial review of the full writeup

**Reviewed 28 September 2026.** Scope: `writeup/WRITEUP.md`, the approved plan and figure captions, and the v08 manuscript/technical supplement. I checked selected bibliography details against primary records. The producing agent is separately validating PDF/HTML rendering. The draft was being refined during review, so findings below use section names and quoted phrases rather than unstable line numbers.

**Verdict:** The writeup's central claim and numerical interpretation are defensible. All seven requested extensions are present and separated from completed work. No fabricated result, clinical-risk claim, or claim of established JEPA superiority was found. Correct the methodological/audit wording and small factual errors below before delivery. These require local edits, not new experiments or a restructuring of the article.

## P1 — Make the proposed forecasting audit capable of detecting leakage

**Location:** Technical companion C, “Prevent information leakage and circular rewards.”

The sentence “processing its prefix gives the same available state as processing that prefix in isolation” can describe the same operation twice. It does not explicitly test whether an offline tool used future frames before returning its prefix outputs.

**Replace with:** “Hold the observed prefix fixed, then alter or remove every future frame. All inputs available at forecast time, and the resulting forecast, must remain unchanged.” If stochastic processing is involved, fix the random state for this check. Retain the existing requirement that pose extraction, normalization, camera estimation, SLAM and every downstream tool obey the cutoff. The plan specifies this stronger test; the final prose should preserve it.

## P2 — Describe the actual VICReg augmentation, not apparent coordinate jitter

**Location:** Technical companion A, “Base objective.”

“Its translated views perturb each coordinate by a value drawn uniformly from [-0.02, 0.02]” could mean an independent perturbation at every joint/time coordinate. The technical supplement instead specifies independently translated input views, with each **translation coordinate** uniform over that interval in **normalized units**.

**Revision:** Say that each view receives a sampled two-dimensional translation, whose x and y components are uniform in [-0.02, 0.02] normalized units. This preserves within-view geometry and avoids implying per-joint jitter. Source: `supplement/technical-details.tex`, paragraph beginning “For VICReg.”

## P2 — Correct two bibliography details and remove an operational source note

**Location:** References 4, 21 and 2.

- CHOIS is **ECCV 2024**, not CVPR 2024; the linked primary record explicitly says ECCV. [CHOIS primary record](https://arxiv.org/abs/2312.03913)
- The stroke paper lists **Purnima Padmanabhan**, cited as “Padmanabhan, P.” The extra N in “P. N.” is unsupported by the article. The nine-participant description and biomechanical interpretation in the overview are accurate. [Publisher record](https://link.springer.com/article/10.1186/s12984-020-00732-z)
- Delete “its instructions are not task instructions” from the prior-report reference. Distinguishing document content from user instructions is correct handling by the agent, but it is irrelevant process language in a reader-facing scientific writeup. “Used as narrative background and a style reference” is sufficient.

## P2 — Tighten three precise descriptions

1. **Figure 4 caption:** “fourteen people, and three seed-averaged fits” reverses the averaging description. Use “fourteen people and three fitted seeds, averaged within each person.” The person-t interval is over fourteen person summaries, not three averaged fits.
2. **“Adding biomechanics or simulation during training”:** the OpenCap Monocular sentence compresses different operations. More precise: “The 2026 OpenCap Monocular preprint refines WHAM estimates through optimization, then estimates biomechanical quantities using constrained models, simulation and learning.” This maintains the important prior-work boundary without implying that every stage directly refines WHAM via simulation.
3. **Technical companion B, “Support”:** clarify that reference support is shared across variants of a source window. State explicitly that nonfinite predictions or predicted segments shorter than two pixels count as failures. The current text gives the costs but leaves the failure trigger implicit.

## P2 — State the remaining external-validity and reproducibility limits once

**Location:** End of completed results, or opening of the technical companion.

The prose already explains synthetic preparation, cohort reuse and absent independent confirmation. Add one compact sentence stating that **no GAVD or natural-video evaluation was completed**, and that the retained numerical packet supports summary/figure checks but lacks the complete reconstructed trajectories and checkpoints needed to repeat the full pipeline. “Recover the required assets” later in the proposed-work section is less clear than stating what is currently available. This matters because the earlier report discusses natural videos and the new document's polished figures can otherwise suggest a more complete empirical package.

## Numerical and methodological checks that passed

The headline input/direct/zero means, 5.14° paired response gain, all three uncertain primary gains, 4.39°–7.21° trajectory increases, original/low/dense ViTPose means, 93% descriptive ratio, endpoint secondary interval, global-swap rates, and pooled naming difference agree with v08. Failure accounting correctly distinguishes unconditional successful contributions from success-only error and interprets 74% as part of a score difference. The article does not claim preserved response accuracy from better trajectories.

The 112/14-person split, 1,645/155 windows, 25 Hz/128 samples, 384 tokens, teacher/student temperatures, shared feature coefficient, gradient imbalance, clipping count, readout packages and update budgets are consistent with the sources. The added control paragraph explains all eight families rather than leaving readers to decode the figure labels. Full-window restoration and independent per-window evaluation are clear.

## Coverage, usability and scope

The main narrative follows the earlier report's accessible progression while improving its evidential discipline. The hypothetical chair example is clearly distinguished from an observed case, and the arithmetic example is labeled illustrative. The six main figures and two companion figures are cross-referenced consistently despite using different asset numbers.

The discussion covers physics learning, physical groundedness, discriminative/generative JEPA, simulation/OpenSim supervision, metric depth, Qwen tool routing with RL, and coupled human-object generation. Independent reference measurements, prospective versus completed work, simple controls, abstention, moving-object progression and separate clinical validation remain visible. The next two studies are identifiable: untouched-person 2D confirmation followed by independently referenced 3D movement contrasts.

This review did not rerun training, reproduce source inference, reestimate uncertainty or visually certify the finished PDF. It relies on the earlier independent 248-mark figure audit for unchanged assets and on the present source comparison for prose. Final disposition should record the edits above and the producing agent's separate render checks.
