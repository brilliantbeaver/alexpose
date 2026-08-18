# How to actually run Idea 7: does the label-aware group loss teach, or just memorize?

This is the plain-language, roll-up-your-sleeves guide to Idea 7. The full proposal narrative lives in
[`./README.md`](./README.md). Every number here is bound to the two facts files, and nothing here may
contradict them: [`../_shared_facts.md`](../_shared_facts.md) holds every number, and
[`../_neuro_facts.md`](../_neuro_facts.md) holds the biology. If a number appears below, it came from one
of those files.

A reminder before we start: the folder labels (normal, parkinsons, stroke, myopathic, cerebral_palsy) are
just tags that came with the GAVD dataset (Ranjan et al., IEEE Access 2025, DOI 10.1109/ACCESS.2025.3545787).
They are not diagnoses made by this project, and nothing here is a medical test.

## The big idea in plain words

A model watches short clips of people walking and turns each clip into a short list of numbers, a kind of
"fingerprint" for how that person moved. We want clips of people who walk in similar ways to end up with
similar fingerprints.

While training this model, we added a special ingredient. On top of the model's normal, self-taught job, we
also told it the condition label for each clip and asked it to pull clips with the same label close together.
That extra ingredient is called the group loss. It clearly makes the model look better on the exact videos it
trained on. The honest worry is simple: did that ingredient teach the model a real rule about what
Parkinsonian or stroke-like walking looks like, so it works on a brand-new video too? Or did it just
memorize the handful of videos it saw, so it falls apart the moment you show it a new one?

This whole guide is about how to tell those two stories apart, using real data, step by step.

Here is the everyday version of the trick. Imagine a student who aces the practice test because they
memorized the answer key, but flops on the real exam with new questions. To catch that student, you do not
just look at the practice score. You give them a fresh test. In our project, the "practice test" is scoring
the model on videos it trained on (we call that transductive), and the "fresh test" is scoring it on videos
it never saw (we call that inductive). If the group loss lifts the practice score but not the fresh-test
score, it was memorizing, not learning.

## A small glossary (the words this guide uses)

- Token: the smallest chunk the model reads. Here one token is one body joint at one short slice of time.
- Embedding: the short list of numbers (a "fingerprint") the model produces for a clip. Similar walks should
  get similar embeddings.
- Encoder: the part of the model that turns tokens into embeddings.
- JEPA: Joint-Embedding Predictive Architecture. The self-taught trick of hiding part of the input and
  predicting the hidden part in fingerprint space, not pixel space.
- Group loss: the label-aware ingredient we are studying. It reads the condition label and pulls
  same-condition clips together while pushing different conditions apart. This is the one knob we turn.
- Probe: a simple classifier (a straight-line rule) trained on top of the frozen fingerprints to guess the
  condition. It is a small "reader" that tells us how much the fingerprints already know.
- Transductive: the model was trained on the very videos you later test it on. A high transductive score can
  just mean the model memorized those videos.
- Inductive: you test on videos the model never saw. Only an inductive score shows real transfer.
- Source-video-disjoint: when we split the data, all clips from one video stay on one side of the split.
  Clips from the same video are not independent evidence.
- Missingness-only: a baseline that sees only which joints were found (visible) versus missing, and none of
  the actual positions. If a model cannot beat this, it is reading holes, not walking.
- Macro-F1: a fairness-minded score from 0 to 1, higher is better. It averages how well each condition is
  found, so one big class cannot hide poor work on the rare ones.
- V-usable information: a careful measure of how much a simple, fixed reader can actually pull out of the
  fingerprints about the label (Xu et al., ICLR 2020, arXiv:2002.10689). We use it because a fancier measure
  can say the answer is "in there" even when a plain probe cannot reach it.

## 1. The question in one sentence

On a small feasible set of source-video holdouts, does the explicit Stage-1 through Stage-4 group-loss term
improve condition separation on held-out (new) source videos over a matched arm with that term switched off,
or does it only improve separation on videos the encoder already saw while leaving new-video separation
unchanged?

In plainer words: turn the group loss on, then off, keep everything else the same, and check whether the
group loss helps on brand-new videos or only on old ones.

## 2. Why this idea, in plain words

The project's headline number is a transductive macro-F1 of 0.821 on the five-class readout. Here 0.821 is a
score between 0 and 1, and higher is better, so 0.821 looks strong. But every one of the 16 test videos and
all 29 test rows also trained the encoder. So 0.821 is a seen-video score. If the group loss is the reason
that number is high, and if it does not survive a fresh test on new videos, then the honest way to describe
the whole project changes: 0.821 becomes a statement about fitting videos we already saw, not about learning
how gait is structured.

There is a second clue that the group loss is doing something forceful. During training, the "normal"
fingerprint drifted a lot. Its cosine similarity to itself across stages fell from 0.954 after Stage 1 down
to 0.594 after Stage 4. Cosine similarity runs from -1 (opposite) through 0 (unrelated) to 1 (identical
direction), so 0.954 means the normal fingerprint barely moved early on, and 0.594 means it had shifted a
great deal by the end. Something pushed the normal fingerprint around even though the normal clips never got
their own active group term after Stage 0.

Here is the plain mechanism, and the biology that makes it interesting. The group loss has a "push apart"
part: every time a new condition is added (Stage 2 stroke, Stage 3 myopathic, Stage 4 cerebral palsy), it
shoves that new condition's center away from the others, including away from normal. Because one shared
encoder makes all the fingerprints, the same shove that separates the new centers also reshapes the features
under the normal clips. So the normal fingerprint drifts.

Why care which geometry got deformed? The neuroscience gives one region special status. Myopathy is a
primary muscle disease: the problem is in the muscle itself, not in one brain hemisphere or the basal
ganglia (Barohn et al. 2014, PMID 25037080). Its weakness hits both sides roughly equally, so at the
skeleton level myopathy walks symmetrically, with no significant left-right asymmetry versus controls
(Xiong et al. 2023, PMID 37525241) and a preserved rhythm (cadence 2.25 versus 2.21 steps per second, not a
significant difference; Vandekerckhove et al. 2022, PMID 35721358). What does stand out for myopathy is
posture: an anterior pelvic tilt of 16.4 degrees versus 11.6 degrees in controls (Vandekerckhove et al.
2022, PMID 35721358). Normal walking is also symmetric and rhythm-preserved, so normal and myopathy together
form one symmetric baseline. Every lateralized condition (stroke, hemiplegic cerebral palsy, early
Parkinson's) is defined as a departure from that baseline. So the sharper question is not just "does the
group loss help" but "when it compacts and separates centers, does it preserve or wreck the symmetric
baseline geometry that every other diagnosis is measured against?"

Reading those numbers plainly:
- 2.25 versus 2.21 steps per second is the walking rhythm. The two are almost the same, and the study calls
  the gap not significant, meaning it is within noise. Myopathy keeps a normal rhythm.
- 16.4 versus 11.6 degrees is how far the pelvis tilts forward. 16.4 is larger, so the myopathy group tilts
  forward more. That posture is what sets myopathy apart even when rhythm and symmetry look normal.

## 3. What data you need

### The internal work (this is the whole core study)

The core runs entirely on the gavd5 GAVD cohort (Ranjan et al., IEEE Access 2025). No new data is collected.
The data comes as skeletons: for each walking clip, a pose detector (MediaPipe BlazePose) finds 33 body
joints in every frame, so a clip becomes a moving stick figure. Each clip is stretched or squeezed to exactly
64 frames, then 4 frames in a row are grouped into one time patch, giving 16 time positions. With 33 joints,
that is 33 x 16 = 528 possible joint-time tokens per clip. One token packs 4 frames times 3 coordinates
(x, y, and a relative z) into a 12-number vector.

Two facts about this cohort bind the whole design, and you must respect them:

- The independent unit is the source video, not the clip. The canonical cohort is 96 sequences from only 18
  YouTube source videos, split as normal 1, Parkinson's 2, stroke 3, myopathic 10, cerebral palsy 2. All 12
  normal sequences come from a single video. So "learn the condition" and "memorize the video" are almost
  the same thing here unless you test on held-out videos. The full curriculum grows this to 159 sequences
  from 35 source videos, but more clips do not remove the source dependence.
- There is a provenance confound. Most normal rows were built through the augmented extraction path, while
  every abnormal row used the canonical path. So a "normal versus abnormal" signal could be about how the
  clip was processed, not how the person walked. We deal with this head-on in Step 2 below.

### The reach step (optional, and non-clinical)

No participant-disjoint skeleton dataset exists for myopathy or cerebral palsy, so there is no honest
skeleton-level clinical transfer test. That is a real limitation, not a footnote. If you want to check
whether the method (the transductive-versus-inductive gap plus the V-information trend) reproduces on a
larger, participant-disjoint pose distribution, use the verified non-clinical, reach-tier pose cohorts:
CASIA-B, OU-MVLP-Pose, GREW, and Gait3D. These are non-clinical gait/pose datasets, so they can only check
whether the method behaves the same way on more data, never make a clinical claim. Pose validity for the
posture and symmetry features is anchored to Human3.6M against motion capture (Ionescu et al. 2014,
DOI 10.1109/tpami.2013.248). For a label-level (not skeleton-level) rhythm check on the Parkinson's side, the
PhysioNet Gait-in-PD cohort exists, but it is force and IMU data, not skeletons. All of these are reach-tier
and add no new clinical data at any point.

How a real team obtains and cleans the data: for the internal work you re-extract the token tensors from the
source CSVs (Step 0 below), which are already produced by the project's own extraction notebooks. For any
reach cohort, you download the public release, run the same 33-joint BlazePose extraction, apply the same
64-frame resize and pelvis-centering, and keep failed pose rows on the timeline with zero visibility so gait
timing is preserved.

## 4. Step by step, how to do it

This is a recipe a motivated student could follow. Note carefully which steps need retraining and which do
not.

1. Regenerate the artifacts inside this clone (needs light recompute, gating). This clone has no saved
   checkpoints (`.pt` files) and no missingness features file. So first re-extract the `.npz` token caches
   from the source CSVs, so the 528-token tensors exist, then regenerate the transductive reference
   checkpoint and the missingness features. Nothing else starts until these reproduce and verify.

2. Bind everything to one fingerprint (no retraining). The reference encoder that saw everything is the
   five-stage, curriculum-final checkpoint with fingerprint prefix `d0acc262` (600 curriculum epochs, 11,400
   optimizer updates). A second lineage prefix, `dba24a`, has also been seen locally; mixing them would be a
   confound, so bind every result to `d0acc262` and print the fingerprint next to every number.

3. Harmonize provenance before training (analysis choice, no retraining yet). Because normal mostly used the
   augmented path and abnormal used the canonical path, either harmonize everything to a single extraction
   pathway before training, or make the primary held-out task a binary contrast with provenance regressed
   out. "Regressed out" means: first measure how much of the score the processing pathway alone can explain,
   then subtract that part off, so what remains is not just the pathway leaking through. Five-class center
   geometry is demoted to a secondary result.

4. Define the three matched arms precisely (this is where retraining happens, but small). The treatment is
   the explicit group-loss term, not all label information. The balanced sampler and VICReg's rule of at
   least 2 samples per condition per batch both sneak label information in even when the group term is
   zeroed. So run three arms:
   - Arm ON: the deployed recipe, `L = L_JEPA + 0.05 * L_VICReg + 0.25 * L_group`. The 0.25 weight is what
     makes the group term active.
   - Arm OFF-matched: set that 0.25 weight to 0, but keep the same balanced sampler. This is the primary
     comparison against ON.
   - Arm OFF-random: set the weight to 0 and also swap the balanced sampler for a plain random one. This
     bounds how much of any ON advantage the sampler alone reproduces.

5. Run a compact, fold-local fine-tune, not a full retrain (small retraining). Do NOT redo 11,400 updates
   per fold. On a small fixed feasible holdout set, each arm gets a short, matched-step fine-tune from the
   same Stage-0 starting point, with identical steps, batch schedule, and seeds across arms. Only the group
   weight (and, for OFF-random, the sampler) is allowed to differ. This retrains only at the small gavd5
   scale and needs no fresh pretraining.

6. Measure separation two ways, on seen and on new videos (no retraining, just reading). For every fold and
   arm, compute condition separation twice: transductive (on videos that fold's encoder trained on) and
   inductive (on the held-out source videos). Measure separation with (a) a linear probe's held-out-source
   macro-F1 and (b) a geometry number, the minimum centroid cosine distance (the gap between the two closest
   condition centers; the seen-video reference is 0.036718, which is small, matching the finding that
   conditions barely separate). Report both so any claim of "better separation" must show up in a probe score
   and in the geometry.

7. Add the secondary panels (no retraining). Compute per-stage V-usable information on the three mechanism
   targets (the symmetry ratio, cadence, and anterior pelvic tilt) against a frozen Stage-0 reference, so
   predictive competence has one fixed ruler. These say whether the symmetric baseline geometry was
   preserved or deformed. See [`./images/fig2.svg`](./images/fig2.svg). How to read that picture: each line
   is a mechanism target across the curriculum stages; a line that sags on the symmetric-baseline targets
   after Stage 2 is the visible sign that centroid separation deformed the normal-plus-myopathy geometry.

The overall design (one knob, three arms, two scores each) is drawn in
[`./images/fig3.svg`](./images/fig3.svg). How to read that picture: read left to right. One shared start
splits into three arms; the only differences are the group-loss weight and, for the third arm, the sampler.
Each arm is scored on seen videos and on new videos, and the box at the bottom is the pre-registered rule
that turns the new-video gap into a verdict.

## 5. The decision rule, decided in advance

Everything below is fixed before any fitting, so we cannot move the goalposts after seeing results.

Because the per-condition source counts are tiny (normal 1, Parkinson's 2, stroke 3, myopathic 10, cerebral
palsy 2), no five-class leave-one-source-out fold can even hold out a normal source. So the primary endpoint
is binary and provenance-controlled.

- Primary held-out task: binary held-out-abnormal separation on the canonical-pathway conditions only
  (myopathic with 10 sources and cerebral palsy with 2 sources), with provenance regressed out of the probe.
  Use leave-one-source-out for every condition with fewer than 4 sources, and show per-source dots plus a
  bootstrap confidence interval (CI). A CI is an honest error bar: a range where the true score most likely
  sits given how little data we have. Bootstrap is how we build it: resample the data many times, recompute
  the score each time, and see how much it wobbles. A wide CI is the warning that one score alone can mislead
  when you only have a few videos.
- Primary endpoint: held-out-source macro-F1 (inductive), reported next to the transductive macro-F1 for the
  same arm and fold.
- Pre-registered margin: ON must beat OFF-matched on the inductive endpoint by at least +0.05 macro-F1, with
  the sign consistent across held-out sources. A gap smaller than +0.05, or one that flips sign across
  sources, is scored as a NULL (memorization: a benefit only on seen videos).
- Minimum source count for the sign rule: the sign-consistency check is only meaningful when a condition
  gives enough folds to reveal a pattern. Myopathic has 10 sources, so leave-one-source-out gives 10 folds
  and a real sign distribution. Cerebral palsy has only 2 sources, giving 2 thin folds, so it contributes to
  the pooled magnitude and per-source dots but is excluded from the sign-consistency test. The sign rule is
  evaluated only on conditions with at least 4 sources (here, myopathic).
- Every arm must clear the missingness-only floor of macro-F1 0.429. An arm below that floor is reading
  visibility holes, not walking.
- Hard collapse gate for both arms: the feature standard deviation and effective rank must stay healthy
  (reference feature std 0.413745). A collapsed arm is rejected no matter how good its macro-F1 looks.

### A worked example (illustrative numbers only, not measured facts)

Suppose you run the myopathic leave-one-source-out folds and, averaged across them, you read these inductive
(new-video) macro-F1 scores. All four numbers here are made up for teaching, not grounded facts.

- ON inductive macro-F1 = 0.62 (illustrative).
- OFF-matched inductive macro-F1 = 0.60 (illustrative).
- Step 1, compute the gap: 0.62 - 0.60 = 0.02.
- Step 2, compare the gap to the +0.05 margin: 0.02 is smaller than 0.05.
- Step 3, so the margin is not crossed, even before checking sign consistency.
- Step 4, also confirm both arms clear the missingness floor of 0.429: both 0.62 and 0.60 do, so both are
  reading more than visibility.
- How to read it: because 0.02 is below +0.05, this illustrative outcome is scored as a NULL, the
  memorization signature, especially if ON showed a large lead only on seen videos. A grounded positive would
  instead need the inductive gap to reach at least +0.05 with the sign consistent across the myopathic folds.

This is exactly the shape drawn in [`./images/fig1.svg`](./images/fig1.svg). How to read that picture: the
left panel is seen videos (transductive) and the right panel is new videos (inductive). If ON towers over OFF
on the left but the two bars sit level on the right, below the +0.05 margin, that is the memorization null.

## 6. Controls that keep us honest

- Missingness-only floor: a probe that sees only which joints are visible, not their positions. Its seen-video
  reference is macro-F1 0.429 (the same control scores 0.448 accuracy and 0.466 balanced accuracy). Any arm
  must beat 0.429 to count as reading gait.
- Untrained-encoder floor: score features from a random, untrained encoder of the same shape. This is the
  "no learning at all" floor; a real result must sit well above it.
- Raw-coordinate ceiling: a handcrafted score straight from the raw skeleton coordinates, with no network.
  This is the non-neural reference, so we can see whether the learned features are even competitive.
- Source-video-disjoint splits: all clips from one video stay on one side of every split, so a seen-video
  score can never masquerade as a new-video score.
- OFF-random arm: bounds sampler leakage. If OFF-random is about equal to OFF-matched, the balanced sampler
  alone did little, and the contrast really is about the explicit group term.
- One-fingerprint binding: bind every number to `d0acc262` and print the fingerprint beside it, so no result
  quietly mixes the `d0acc262` and `dba24a` lineages.
- Provenance regression: subtract off what the processing pathway alone can explain, so acquisition
  differences cannot pose as gait.
- Collapse gate: reject any arm whose features collapse (feature std far below 0.413745, or a low effective
  rank), regardless of its score.
- Fresh seeds only after freezing: seed variation is not source variation, so confirmation uses fresh seeds
  only after the split and margins are locked.

## 7. What could happen, and what each outcome would mean

- Positive: ON beats OFF-matched on new videos by at least +0.05 macro-F1 with a consistent sign, and both
  clear the floor and pass the collapse gate. This licenses the claim that label-aware fine-tuning at this
  scale teaches transferable condition structure, not just video identity. It justifies the supervised
  curriculum as a design choice.
- Null (the memorization signature): ON and OFF-matched are indistinguishable on new videos (gap under +0.05
  or sign-flipping) while ON still wins on seen videos. This rules out the "transferable structure" belief
  and shows the group term buys separation that does not transfer. This null is genuinely informative: it
  changes how the project must report every transductive number, including the 0.821, and it warns other
  small grouped-cohort skeleton projects that a label-aware compactness loss can inflate seen-video metrics
  without improving generalization.
- Sampler-leakage finding: if OFF-random is about equal to OFF-matched, the balanced sampler alone is not
  driving the effect, which sharpens whatever the ON-versus-OFF contrast shows.
- Rejected arm: if any arm fails the collapse gate or cannot beat the missingness floor, that arm is thrown
  out before any verdict, no matter how good it looked.

## 8. What this cannot tell us

- Transductive by nature. The reference encoder saw every evaluation row during the curriculum, so its scores
  are representation diagnostics, not out-of-sample estimates. Only the held-out-source (inductive) numbers
  from the fold-local fine-tune speak to generalization.
- Tiny, unequal sample. With as few as 1 or 2 source videos per condition, some folds are a single point.
  This is why the sign rule is limited to myopathic and why every score gets a bootstrap CI and per-source
  dots rather than a single confident number.
- Provenance confound. Normal is one video on a mostly-augmented path, and abnormal uses the canonical path.
  The provenance regression and the binary primary task reduce this, but at this sample size they cannot
  fully erase it.
- Monocular capture. gavd5 is single-view video, so it cannot answer view-stability questions; those belong
  to the external multi-view reach cohorts.
- Skeleton limits. Skeletons cannot recover kinetics or propulsion, EMG or spasticity, transverse-plane
  rotation, or an etiologic muscle diagnosis. So no outcome here becomes a clinical-accuracy claim, and the
  symmetric-baseline reading stays a mechanism-motivated hypothesis, not a diagnosis.

## 9. How to make it reproducible

- Bind to one checkpoint: `d0acc262`, printed beside every number, never mixed with the `dba24a` lineage.
- Regenerate first, then gate: reproduce the `.npz` token caches and the reference checkpoint and missingness
  features before anything else, and stop if they do not reproduce exactly.
- Fix the seeds. All three arms share identical steps, batch schedule, and seeds; only the named factor
  (group weight, and for OFF-random the sampler) changes. Confirmation uses fresh seeds only after the split
  and margins are frozen.
- Save the split manifest. Write down exactly which source video went to which fold, so anyone can rebuild
  the source-video-disjoint splits.
- Save the arm configs, the per-seed and per-fold results (transductive and inductive macro-F1, minimum
  centroid cosine distance, the missingness floor, the provenance regression, the per-stage V-information),
  and the fingerprint next to each. Package these together so the two-panel figure ([`./images/fig1.svg`](./images/fig1.svg))
  and the secondary panel ([`./images/fig2.svg`](./images/fig2.svg)) can be redrawn from saved numbers.

## References

- Abdelfattah and Alahi, S-JEPA, ECCV 2024, DOI 10.1007/978-3-031-73411-3_21.
- Assran et al., I-JEPA, CVPR 2023, arXiv:2301.08243.
- Bardes et al., V-JEPA, 2024, arXiv:2404.08471.
- Bardes, Ponce, LeCun, VICReg, ICLR 2022, arXiv:2105.04906.
- Grishchenko et al., BlazePose GHUM, 2022, arXiv:2206.11678.
- Kapoor and Narayanan, "Leakage and the Reproducibility Crisis in ML-based Science", 2022, arXiv:2207.07048.
- Xu et al., "A Theory of Usable Information Under Computational Constraints", ICLR 2020, arXiv:2002.10689.
- Ranjan et al., GAVD, IEEE Access 2025, DOI 10.1109/ACCESS.2025.3545787.
- Barohn et al., "Approach to Peripheral Neuropathy and Myopathy", Neurol Clin 2014, PMID 25037080.
- Xiong et al., DMD spatiotemporal symmetry, Biomed Eng Online 2023, PMID 37525241.
- Vandekerckhove et al., DMD versus typically-developing gait, Front Hum Neurosci 2022, PMID 35721358.
- Patterson et al., gait symmetry ratio, Gait Posture 2010, PMID 19932621.
- Stenum et al., pose-estimation gait accuracy versus mocap, PLoS Comput Biol 2021, PMID 33891585.
- Ionescu et al., Human3.6M, IEEE TPAMI 2014, DOI 10.1109/tpami.2013.248.
- Yu et al., CASIA-B multi-view gait database, 2006.
- Takemura et al., OU-MVLP-Pose multi-view pose gait dataset, 2018.
- Zhu et al., GREW, 2022, arXiv:2205.02692.
- Zheng et al., Gait3D, 2022, arXiv:2204.02569.
- Goldberger et al., PhysioNet Gait-in-PD (gaitpdb), DOI 10.13026/C24H3N.
