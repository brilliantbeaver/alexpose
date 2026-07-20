# Shared brief: aligning all Gait-JEPA docs to the honest, latest framing

Internal. NOT user-facing. Every editor reads this plus the two source-of-truth files:
- The learning paper: `gait/skeleton-jepa/gavd2/docs/learning/learning-journey.md`
  (the newest, most careful framing, especially the per-clip vs per-sequence section
  and the "Is the JEPA approach promising / what are we learning / what frontiers"
  sections).
- The ground-truth numbers: `gait/skeleton-jepa/gavd2/docs/learning/_FACTS.md`.

## Writing rules (apply to every edit)
- Reading level: a serious high school student. Plain, natural, fluent words. Short
  sentences. Explain a term the first time it appears. No machine-learning background
  assumed.
- NO em-dashes and NO en-dashes ANYWHERE. Use commas, periods, or "to" for ranges.
- No fancy words (no leverage, utilize, delve, robust, seamless, demonstrate, paradigm).
- Be fair and objective. Do not oversell. Do not undersell. Report both the honest and
  the leaky number where relevant, always labeling the leaky one as leaky.
- Preserve each file's existing voice, structure, and formatting. Make TARGETED edits,
  not full rewrites, unless this brief says "rewrite".

## The honest story in one paragraph (the through-line every doc should reflect)
The first attempt (`gavd`, iteration 1) built the full pipeline but its flashy headline
near 0.88 came from a per-CLIP split that leaked overlapping windows of the same walk
across train and test. The honest per-SEQUENCE number on iteration 1 was about 0.49 on
the 42 sequences that survived. The second attempt (`gavd2`, iteration 2) fixed the
measurement and made the comparison controlled (same exact 68 sequences, same unit, same
split, matched classifier), and chased coverage to all 68 of 68. Its honest per-sequence
numbers are linear 0.486, MLP 0.626, matched Random Forest 0.579, and 0.619 on the
baseline's exact seed-42 split, all well above the 0.20 chance level and below the tuned
0.762 hand-feature baseline. The method looks promising, but this is a controlled
comparison of representations, not a clinical validation, and it does not beat the
baseline. The binding constraint is sample size (68 labeled walks), not the quality of
what the encoder learned.

## THE key clarification to propagate (per-clip vs per-sequence)
- A "sequence" is one continuous walk. The encoder reads it as short 32-frame WINDOWS
  that slide forward and OVERLAP heavily, so neighboring windows are near-duplicates.
  Each walk yields about 7 windows on average.
- PER-CLIP scoring throws all windows in one pile and splits at random, so a window and
  its near-twin from the SAME walk can land on both sides. The model wins by remembering,
  not learning. This is WINDOW LEAKAGE. It reads about 0.87 to 0.92 and is NOT comparable.
- PER-SEQUENCE scoring pools a walk's windows into ONE vector and keeps whole walks on
  one side of the split. The test walk was never seen. It reads the honest 0.49 to 0.63.
- Per-sequence is the ONLY fair unit because the baseline was scored per sequence. Keep
  per-clip only as a labeled diagnostic that MEASURES the leak, never as a result.

## Canonical numbers (from _FACTS.md; do not drift)
- Baseline (exp5 RF, 82 hand features, 100 trees): 0.762. Chance (5 classes): 0.20.
- gavd2 REAL, 68 of 68 coverage, per-SEQUENCE over 20 splits:
  linear 0.486 +/- 0.102, MLP 0.626 +/- 0.083, matched RF 0.579 +/- 0.114.
  exp5 EXACT seed-42 47/21 split, matched RF: 0.619.
- gavd2 per-CLIP DIAGNOSTIC (leaky, label it): linear 0.866, MLP 0.920, RF 0.883.
- RQ2 (per-seq linear): 25% 0.393, 50% 0.417, 75% 0.457, 100% 0.486.
- RQ3 (Ridge R-squared): step_amplitude 0.682 (captured label-free), asymmetry_index 0.081.
- RQ4: embedding std ON 0.904 vs OFF 0.743 (OFF does NOT collapse to zero).
- gavd (iteration 1) REAL: 42 of 68 sequences survived, 296 clips, mean 7 windows/seq.
  per-clip linear 0.880 / MLP 0.915 / RF 0.881; per-sequence linear 0.494 +/- 0.172;
  leak inflation about 39 points. RQ3 iter-1: step_amplitude 0.719, asymmetry 0.154.
  RQ4 iter-1: std ON 0.889 vs OFF 0.766. Training: total 32.0 to 5.5, MSE 1.28 to 0.22,
  emb std 0.38 to 0.76. These iteration-1 numbers stay as iteration-1's own record.

## Per-file instructions

### gavd (iteration 1) docs: sharpen to fully honest framing
The gavd paper and tutorial ALREADY report both numbers honestly. The job is to remove
any residual phrasing that presents the leaky clip-level number as a WIN, and to add a
forward pointer to gavd2's controlled 68/68 result.
- gavd/docs/paper.md: In the Conclusion (Section VIII), the sentence "learns a gait
  representation that is well above chance at the sequence level and beats the 0.76
  baseline at the clip level" must be rephrased so the clip-level number is NOT framed as
  a win (it is leaky). Say instead that the honest per-sequence signal is above chance and
  below the baseline, and that the clip-level number is inflated by leakage. Add one
  sentence pointing to iteration 2 (gavd2), which locked the exact 68, chased coverage to
  68 of 68, and reports the controlled per-sequence numbers above. Keep everything else.
- gavd/README.md: near the top and in the loss/results discussion, add a short note that
  a follow-up iteration (`../gavd2/`) turns this into a controlled comparison and reports
  the honest per-sequence numbers; make sure no line claims a clean "beat 76 percent" win.
- gavd/docs/tutorial.md: it is already honest; add a short forward pointer to gavd2 and
  its learning paper in the intro and in the "Learnings from the real run" area, and make
  sure the RQ1 subsection frames per-clip as leaky, not as a win.

### gavd2 docs: cross-link the learning paper and add the promising/frontiers framing
- gavd2/README.md: add a bullet pointing to docs/learning/learning-journey.md
  as the plain-language, high-school-level story of the whole journey.
- gavd2/docs/paper.md: add one sentence in the Discussion or Conclusion that states the
  honest verdict in plain terms: the method looks promising but this is a controlled
  representation comparison, not a clinical validation, and sample size is the ceiling.
  Cross-link the learning paper. Keep the IEEE numbers exactly as they are.
- gavd2/docs/pipeline.md: in the nb05 section, sharpen the one-line reading of per-clip
  vs per-sequence to match the learning paper's clarity (leaky diagnostic vs honest unit).
  Add a "Related pages" link to ../learning/learning-journey.md.
- gavd2/docs/glossary.md: refresh the "Window leakage" and "Window versus sequence unit"
  entries to the clearer wording, and remove the stale "(pending iteration-2 refresh)" now
  that the real numbers exist (0.88 leaky vs 0.49/0.486 honest). Add a "Related pages"
  link to the learning paper.
- gavd2/docs/tutorial.md (the notebooks 06/07 "enhanced encoder" tutorial): DIFFERENT
  topic. Do NOT merge. Only (a) verify it cites the baseline 0.762 and chance 0.20
  correctly, (b) add a one-line pointer to the learning paper for the gavd->gavd2 back
  story. Do not change its 06/07 content.

### root gait/skeleton-jepa: light touch only
- README.md (the original forward-looking proposal): do NOT rewrite the body. Add a short
  "Results to date" note (a few sentences) near the top or right after Part 7, that says:
  the full-dataset build ran (see gavd and gavd2); the honest controlled result is a
  per-sequence 0.486 to 0.626 versus the 0.762 baseline, above chance, not yet beating the
  tuned baseline, with sample size the main limit; and RQ1's "beat 76 percent" target was
  not met on this small set, which is an honest, useful outcome. Point to
  gavd2/docs/learning/learning-journey.md. Keep the proposal's optimistic,
  forward-looking Part 3 success criteria but make clear they are TARGETS, and that the
  measured outcome to date is the honest per-sequence number.
- Do NOT edit proposal/PROPOSAL.md, proposal/proposal.html, slides/, notebook.ipynb, or
  CONTEXT.md (CONTEXT.md is a gavd3 doc, out of scope).

### slides
- gavd/slides/slides.html: replace any phrasing that says the model "beats the baseline"
  as a plain win with honest phrasing (clip-level is leaky and NOT comparable; the honest
  per-sequence number is above chance and below the baseline). Keep the deck structure.
- gavd2/slides/research.html and teaching.html: already carry the honest numbers. Only
  fix any leftover phrasing that overstates, and (optional) add a pointer to the learning
  paper on a closing slide. Light touch.

## After editing
- Re-render the two markdown papers to HTML with pandoc (gavd/docs/paper.md ->
  gavd/docs/paper.html; gavd2/docs/paper.md -> gavd2/docs/paper.html) using the same
  flags already used for each (gfm -> html5, standalone, embed-resources).
- Verify zero em/en dashes in every edited markdown file.
