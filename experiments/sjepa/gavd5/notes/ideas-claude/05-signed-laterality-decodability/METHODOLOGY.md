# Idea 5, in plain words: can we read a signed left-minus-right axis out of the model, and does a mirror flip it?

This is the "how to actually do it" guide for Idea 5, written so a motivated high-school student can follow every
step. The science comes from the folder's [`README.md`](./README.md). Every number here is kept true to
[`../_shared_facts.md`](../_shared_facts.md) (the single source of truth for numbers) and
[`../_neuro_facts.md`](../_neuro_facts.md) (the biology). If a number is not in those files, it is not in here, and any
made-up example is labelled "illustrative numbers only".

One honest note up front, repeated at the end because it matters: the folder labels (normal, parkinsons, stroke,
myopathic, cerebral_palsy) are just tags that came with the GAVD dataset. They are not diagnoses made by this project.
And every result here is transductive, which means the model was trained on the very clips we later test it on. More on
what that costs us below.

## The big idea in plain words (60-second version)

Healthy walking is almost the same on both sides. Your left leg and your right leg do nearly the same thing, just half a
step apart. A lot of gait problems break that balance and hit one side harder than the other. So a very useful thing to
ask about a walk is not just "how uneven is it" but "how uneven, and which side leans." That "which side" part is what we
call the SIGNED difference: left minus right. A plus number means it leans left, a minus number means it leans right.

This project already trained a computer model that watches stick-figure skeletons of people walking (an S-JEPA, defined
below). An earlier check found that this model was surprisingly bad at coughing up the asymmetry number. Idea 5 zooms in
on exactly that. It asks two questions:

1. Is the signed left-minus-right value actually sitting inside the model's learned numbers, in a way a simple
   straight-line rule can read out, and can it hold its own against what you could already get from the raw joint
   positions? (This is DECODABILITY.)
2. If we physically mirror the body, swapping left and right, does the read-out number flip its sign, plus becoming minus
   of the same size? (This is the MIRROR test, a test of EQUIVARIANCE.)

Here is a homey analogy for the mirror test. Imagine you write a number on a piece of glass and hold it up to a mirror.
If the reflection shows the same number with the opposite sign, the writing "respects" the mirror. We are asking whether
the model's signed-asymmetry reading behaves like that: mirror the walker, and the number should cleanly flip.

These two questions are separate on purpose. The model could carry the signed information (question 1 passes) and still
not flip cleanly under the mirror (question 2 fails). That is not a contradiction. It is a real, interesting result: the
side is in there, but it is not stored in the tidy mirror-flipping way we hoped.

## Mini-glossary (the words this idea actually uses)

- SKELETON: a moving stick figure. A pose detector finds body joints in each video frame, so a clip becomes a set of
  moving dots instead of pixels. Smaller than video, and it hides the face and clothing.
- TOKEN: the smallest chunk the model reads. Here, one joint watched over a short 4-frame window.
- EMBEDDING: a short "fingerprint of numbers" the model uses to describe a token. Here each token becomes a list of 64
  numbers.
- S-JEPA (Skeleton Joint-Embedding Predictive Architecture): a model that learns by hiding part of a skeleton and
  predicting the hidden part in fingerprint-space, with no human labels (Abdelfattah and Alahi, S-JEPA, ECCV 2024,
  DOI 10.1007/978-3-031-73411-3_21).
- ENCODER: the part of the model that turns a skeleton into embeddings. "Frozen" means we do not change its weights.
- PROBE: a small, simple readout (here a straight-line rule) that we fit on top of the frozen model to ask "is this
  number in there?"
- SIGNED LEFT-MINUS-RIGHT: a value that keeps direction, not just size. Positive leans left, negative leans right.
- TRANSDUCTIVE: the model was trained on the very clips we test on. A high score can mean it memorized them, not that it
  learned something that transfers to new people.
- SOURCE-VIDEO-DISJOINT: when we split data for testing, no clip from a held-out YouTube source video is used to fit the
  probe. The source video, not the single clip, is the real unit of evidence.
- R-SQUARED: the fraction of a target's up-and-down variation that a readout explains. It runs from 0 to 1, higher is
  better. 0 means "no better than always guessing the average"; 1 means "explains everything."
- ANTISYMMETRIC / EQUIVARIANT under the mirror: mirroring the input cleanly negates the output. Feed in a mirrored body,
  the read-out number changes sign but keeps its size.

## 1. The question in one sentence

On source-video-disjoint folds, is a signed left-minus-right laterality axis linearly readable from the frozen S-JEPA
token tensor, clearing a raw-coordinate baseline by a margin we fix in advance, and does the anatomical mirror that swaps
left and right landmarks flip the sign of the read-out number?

There are two endpoints, kept apart on purpose:

- The PRIMARY endpoint is decodability: is the signed axis linearly present in the frozen tokens, and is it competitive
  with what raw coordinates already give you?
- The SECONDARY endpoint is equivariance: does the reading flip its sign under the mirror?

Decodability can pass while the mirror fails. That is a real and useful outcome, not a mistake.

## 2. Why this idea, in plain words

The clinical literature sorts these gait conditions along a SYMMETRY axis (the one-sided-versus-both-sides split), and
this is the heart of the motivation.

- Conditions with a ONE-SIDED cause tend to make walking lean to one side. Stroke does this because the main motor nerve
  pathway (the corticospinal tract) crosses over to the opposite side of the body, so a stroke in one half of the brain
  weakens the other half of the body (Natali and Javed, StatPearls, PMID 30571044). Parkinson's disease often starts on
  one side (the underlying dopamine loss is one-sided at onset, Riederer and Sian-Hulsmann 2012, PMID 22367437). And
  hemiplegic cerebral palsy comes from a one-sided brain-tissue injury (Volpe 2009, PMID 19081519). All three raise a
  signed left-minus-right difference.
- Conditions with a BOTH-SIDES cause do not lean. Myopathy is a muscle disease that weakens both sides fairly evenly, so
  its walk stays roughly symmetric (Barohn et al. 2014, PMID 25037080). In a Duchenne muscular dystrophy group, the
  step-timing and step-size measures showed no meaningful left-right asymmetry versus controls (Xiong et al. 2023,
  PMID 37525241).

The validated clinical number behind all this is the gait Symmetry Ratio (Patterson et al. 2010, PMID 19932621), built
from step length and swing and stance times. That is the biomarker our signed axis is standing in for.

Now the sharp part. In [`../_shared_facts.md`](../_shared_facts.md), when the earlier analysis (notebook 05) tried to read
simple gait numbers out of the pooled model features, asymmetry came out WORST. Step amplitude (how big each step is) came
out around R-squared 0.719, but asymmetry sat at only about 0.154.

What that means in plain words: the features explained about 72 percent of the up-and-down in step size, but only about
15 percent (about a sixth) of the up-and-down in asymmetry. That is a big gap for the one quantity that most separates
these conditions. The most likely reason is technical: the earlier readout pooled the tokens with a mean and a standard
deviation, and a mean and a standard deviation are "side-blind." They throw away the very left-versus-right information
the signed axis needs. So Idea 5 asks the sharper question the old pooled readout could not: with a side-aware target and
a side-aware probe, is the signed axis actually there, or did the model never build one?

The biology also hands us a free negative control. Myopathy should sit near zero on the signed axis. So if a signed axis
is really there, myopathy sequences should read near zero while stroke, early Parkinson's, and hemiplegic cerebral palsy
should read away from zero. That "myopathy near zero" prediction is a mechanism test the dataset supplies for free.

One thing that is deliberately OUT of scope: Parkinson's rhythm (its jerky-timing signature, measured by stride-time
coefficient of variation, 8.8 percent in fallers versus 4.2 percent in non-fallers, Schaafsma et al. 2003, PMID 12809998;
Hausdorff et al. 1998, PMID 9613733). Our clips get stretched to a fixed 64 frames, which erases absolute cadence, so
that rhythm axis belongs to Idea 4, not here.

## 3. What data you need

### 3.1 The main data: the gavd5 GAVD cohort (internal only)

The main work runs on the canonical GAVD cohort (Ranjan et al., IEEE Access 2025, DOI 10.1109/ACCESS.2025.3545787):
96 sequences from 18 unique YouTube source videos. The per-condition source-video counts are tiny and lopsided: normal 1,
Parkinson's 2, stroke 3, myopathic 10, cerebral palsy 2. All 12 normal sequences come from a single video
(`3KnFt8bH3tE`).

Two consequences fall out of those small numbers, and they shape the whole design.

1. The condition label is almost the same thing as "which video did this come from," especially for normal. So the honest
   unit of evidence is the SOURCE VIDEO, not the individual clip. Two clips from one video are not two independent facts,
   any more than two frames of one movie are two different movies (Kapoor and Narayanan, arXiv:2207.07048; and Varoquaux,
   NeuroImage 2018, on why tiny samples give big error bars).
2. Because some conditions have as few as one source video, you cannot report a per-condition "hold out one source"
   R-squared. One held-out point is not a distribution. So we pool the signed axis across all conditions and plot every
   source video as its own dot.

There is also a PROVENANCE catch. Most normal rows were built through an "augmented" extraction path, while every abnormal
row used the "canonical" path (see [`../_shared_facts.md`](../_shared_facts.md)). If we are not careful, a comparison
could accidentally learn the difference between those two processing pipelines instead of the difference in walking. To
avoid that, the main comparison runs only on the provenance-matched (canonical-path) subset. Provenance is not a stored
column; it is tagged from which loader path produced the row.

### 3.2 The reach data: public multi-view pose cohorts (non-clinical, reach-tier)

There is no public skeleton dataset that pairs left-right clinical labels (stroke, cerebral palsy) with pose and keeps
different people in train and test. So there is no honest skeleton-level clinical transfer test. What does exist is a
NON-CLINICAL reach test for the mirror property itself, on public multi-view pose cohorts: CASIA-B (Yu, Tan, Tan 2006;
124 subjects across 11 camera views) and OU-MVLP-Pose (Takemura et al. 2018; about 10,000 subjects, released keypoints).
GREW (arXiv:2205.02692) and Gait3D (arXiv:2204.02569) are also non-clinical multi-view options.

These are non-clinical and reach-tier. They can only stress the mirror instrument, never make a clinical claim. The reach
questions are about the camera, not the disease: does the signed axis read from one view stay stable at a nearby view (a
property of the gait, not the camera), and does a real left-versus-right camera swap flip it (a genuine physical
reflection, using CASIA-B's symmetric view angles around 90 degrees)? In this pass, that arm is only scaffolded on a
synthetic multi-view fixture in notebook 05b; wiring the real, large, licensed downloads is a marked TODO and is not done
here.

### 3.3 What the data looks like, and how a team would get it

Each sequence is a skeleton: 33 body joints tracked over time, each joint carrying an x, a y, and a relative z coordinate
(MediaPipe BlazePose, Grishchenko et al. 2022, arXiv:2206.11678). A real team obtains it by running the pose detector on
walking videos, then cleaning the result: fill short gaps where the detector blinked, center the skeleton on the pelvis,
scale it to a standard body size, and stretch or squeeze the clip to exactly 64 frames so every sequence is the same
length. Heels are the weak link for the detector (left heel visible about 70 percent of the time, right heel about
67 percent, versus about 99 percent for shoulders and hips), which is one more reason to lean on the pelvis, hips, knees,
and ankles.

## 4. Step by step, how to do it

Good news first: the main arm needs NO retraining. Every step below is a test-time-only linear read of features that are
already frozen and cached. Retrain count for the core: zero.

1. Bind to ONE checkpoint (zero-retrain). Use the curriculum-final target encoder with fingerprint prefix `d0acc262`.
   There is a second lineage prefix `dba24a` floating around locally, and mixing the two would be a confound, so pin every
   number to `d0acc262` before any comparison. Load the checkpoint under the same guards as notebook 05 (correct mode, the
   12-point mask whitelist, curriculum finished, condition order right), and always build the model with
   `SJEPAGait(**checkpoint["config"])` so it reads whatever width and depth the checkpoint actually stored (the project
   default is a small Transformer, embed_dim on the order of 64, depth 2, 4 heads). That way the weights match key for key
   and you can never accidentally load the wrong lineage.

2. Reuse the cached 528-token tensors (zero-retrain). For each of the 96 canonical sequences, run the frozen target
   encoder over all 528 joint-time tokens (the target encoder always sees the full set) and cache the 64-number embedding
   per token. Tag each cached row with its source-video id, condition label, provenance (canonical vs augmented),
   checkpoint fingerprint, and an encoder-exposure flag. These frozen tensors are what every lane reads.

   Reading the math (why 528): 64 frames split into groups of 4 gives 16 time positions, and 33 joints times 16 time
   positions equals 528 tokens. The "x" means multiply: 33 x 16 = 528.

3. Freeze the signed target BEFORE any fitting (zero-retrain). From the RAW cached coordinates (not the model), define the
   thing you are trying to read: a signed left-minus-right value over the anatomical pairs. Freeze this function before
   you compute any result, so you cannot tune it to make the answer look good.

   ```python
   LEFT_RIGHT_PAIRS = [(11, 12), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32)]

   def signed_left_minus_right(coords):          # coords: (frames, joints, 3)
       total = 0.0
       for left_idx, right_idx in LEFT_RIGHT_PAIRS:
           left_excursion = coords[:, left_idx, :].std(axis=0).sum()
           right_excursion = coords[:, right_idx, :].std(axis=0).sum()
           total += left_excursion - right_excursion   # signed: left minus right
       return total                                     # positive leans left, negative leans right
   ```

   In plain words: for each left-right joint pair (shoulders, hips, knees, ankles, heels, foot indices), measure how much
   the left joint moved and how much the right joint moved, then subtract right from left. Add those up. Because it uses
   the wobble (standard deviation) of each joint, it does not care where the person is standing, so it works the same on
   raw or pelvis-centered coordinates. It is also exactly antisymmetric on raw coordinates by construction: mirror the body
   and this number flips sign perfectly. The notebook checks that as a self-test before any modelling.

4. Freeze the anatomical mirror BEFORE any fitting (zero-retrain). The mirror negates the x coordinate and swaps each left
   landmark with its right partner. For the encoder pass it uses the full 16-pair whole-body mirror (face, arm, and lower
   body) so the mirrored input is still a valid, realistic skeleton, then runs it through the identical preprocessing
   (short-gap fill, pelvis-centering and body-scale normalization, resize to 64 frames). This matters because laterality
   flip is OFF during training (flip_probability 0.0). The model was never taught to treat left and right as the same, so
   testing the mirror is a genuine probe of whether it LEARNED the flip behavior on its own.

5. Fit the four lanes (zero-retrain, test-time only). Fit a ridge linear regression to predict the signed target. A linear
   regression finds the best straight-line rule from features to the target; "ridge" adds a gentle penalty that keeps the
   rule's weights small so it does not overreact with so few sequences. Pick the penalty size using only the training
   sources of each fold, so held-out sources never influence it. Run four lanes:

   | Lane | What it reads | Retrain | Role | What we expect in advance |
   |---|---|---|---|---|
   | A learned probe | Frozen `d0acc262` per-token features, side-structured | No | Primary | Beat the floor by at least 0.05 R-squared and reach at least 80 percent of the null |
   | B raw-coordinate null | Handcrafted signed left-minus-right coordinate features, no network | No | Non-neural ceiling | Reference target |
   | C untrained-encoder floor | Same features from a random-init encoder of the same shape | No | Floor | Near chance |
   | D mean/std-pooled control | Side-blind pooled tokens | No | Nuisance | Must NOT recover a signed axis |

   Lane A gives the probe explicit access to a per-side contrast: for each pair it uses the time-mean left token minus the
   time-mean right token (the signed part) alongside their sum (the symmetric context). Lane D is the crucial trap-catcher:
   a mean and a standard deviation are side-blind by construction, so a genuinely signed value should NOT be recoverable
   from them. If Lane D somehow "recovers" the axis anyway, the signed claim is an artifact (this is future F4 below).

6. Run the mirror-equivariance test (zero-retrain). Apply the frozen mirror to each input, re-embed through the same
   frozen `d0acc262` encoder, decode with the already-fit Lane A probe, and compare the decoded number on the original
   input versus the mirrored input. Plot mirrored (y) against original (x) and compare to the line y = -x.

   Reading the math (the line y = -x): x is the reading on the original body, y is the reading on the mirrored body. The
   minus sign means a clean flip: same size, opposite sign. If mirroring changed nothing, the dots would fall on y = x (no
   flip) instead of y = -x (a clean flip).

7. Apply the pre-registered decision rule (Section 5) and write out the results. That is it for the core: everything above
   is a linear read of frozen features. No encoder was retrained anywhere.

## 5. The decision rule, decided in advance

We fix the bar BEFORE looking at results so we cannot move the goalposts. The PRIMARY verdict is a positive ("signed axis
present above raw coordinates") only if all three of these hold at once:

1. Lane A beats Lane C (the untrained floor) by at least 0.05 R-squared.
2. Lane A reaches at least 80 percent of Lane B (the raw-coordinate null) R-squared.
3. The decoded sign is correct on at least 75 percent of held-out sources.

Miss any one of the three, and the run is scored as an informative null. That is not a failure; it is a real answer.

Reading the math (the three numbers):
- 0.05 R-squared is the smallest gap over the floor that counts as beating chance; below it the learned features add
  nothing real.
- 80 percent (0.80) is the share of the raw-coordinate ceiling the probe must reach to count as competitive. Note this is
  a "keep up with" bar, not a "beat it" bar. Lane A can pass while still sitting below the raw-coordinate ceiling. The
  point is whether the learned features are competitive with the non-neural ceiling, not whether they beat it.
- 75 percent (0.75) is the share of held-out sources whose decoded sign must point the right way; below it the sign is not
  reliable.

The SECONDARY mirror verdict is separate. Fit a line to mirrored-versus-original. For a "flips" verdict, the slope must be
negative and inside the band from -1.25 to -0.8 (a band placed around the ideal -1). A near-zero or positive slope is a
"does not flip" verdict, which licenses only the weaker statement that the encoding is non-antisymmetric (mirroring does
not cleanly negate the reading).

Reading the math (the slope band): the slope is how much the mirrored reading changes when the original changes by 1. A
perfect flip has slope -1. The band -1.25 to -0.8 means "clearly negative and reasonably close to -1." A slope near 0
(mirror barely changes the output) or positive (mirror does not reverse it) fails the flip test.

### A worked example (illustrative numbers only, not measured facts)

Say we hold out 4 source videos. On them, Lane A (learned) scores R-squared 0.42, Lane C (untrained floor) scores 0.05,
and Lane B (raw-coordinate null) scores 0.50. Walk the three checks:

- Beat the floor by at least 0.05: 0.42 minus 0.05 = 0.37, which is far above 0.05. Pass.
- Reach at least 80 percent of the null: 0.80 times 0.50 = 0.40, and 0.42 is above 0.40. Pass.
- Sign correct on at least 75 percent of sources: say the decoded sign matched the true side on 3 of the 4 held-out
  sources, so 3 / 4 = 0.75, which meets the bar. Pass.

All three pass, so this illustrative run would clear the margin and support "the signed axis is linearly present above raw
coordinates." Notice it does NOT mean the learned encoder beat the raw ceiling: Lane A (0.42) is still below Lane B (0.50),
so raw coordinates decode the axis better. The pass only reflects the "keep up with 80 percent of the null" bar. If Lane A
had instead scored 0.39, then 0.39 is below the 0.40 needed for 80 percent of the null, and the whole run would be scored
as an informative null.

The two decisive pictures are drawn from real or smoke data by notebook 05a: [`./images/fig1.svg`](./images/fig1.svg) is
decodability against the raw ceiling and the untrained floor, and [`./images/fig2.svg`](./images/fig2.svg) is the mirror
test against the y = -x line.

How to read fig1: each dot is one held-out source video, plotting the decoded signed value against the true signed value.
Dots hugging the diagonal mean good decoding. The raw-coordinate ceiling (Lane B) and the untrained floor (Lane C) are
drawn as reference lines, so you can see at a glance whether the learned probe (Lane A) is up near the ceiling or down
near the floor.

How to read fig2: each dot is one source, plotting the mirrored reading (up) against the original reading (across). If the
dots line up on y = -x (a line sloping down through the origin), the mirror flips the sign cleanly. If they sit flat or on
y = x, the mirror did not flip it. The side-blind Lane D cloud is shown for reference; it should NOT line up as a clean
signed flip.

## 6. Controls that keep us honest

- ONE fingerprint. Bind everything to `d0acc262` before any comparison, so the `dba24a`-versus-`d0acc262` lineage
  difference cannot sneak in as a fake effect.
- The missingness-only spirit and the raw-coordinate NULL (Lane B). We only credit the model if its learned features can
  keep up with what you already get from raw joint coordinates with no network at all. If the learned features cannot even
  reach 80 percent of that, the model added nothing on this axis.
- The untrained-encoder FLOOR (Lane C). A random, untrained encoder of the same shape sets the "pure chance" level. Lane A
  must clearly beat it.
- The side-blind nuisance control (Lane D). A mean-and-standard-deviation pooling throws away side and order, so it must
  NOT recover a signed axis. If it does, the signed claim is an artifact and gets withdrawn.
- Source-video-disjoint splits. Hold out whole source videos, never single clips, using GroupKFold on the video id. The
  probe's penalty is chosen with an inner split on the training sources only, so held-out sources never leak into the
  penalty choice.
- Provenance-matched subset. Run the main comparison only on the canonical-path rows, so a decoded axis cannot be an
  augmented-versus-canonical acquisition artifact.
- No per-class hold-one-source R-squared on n=1 sources. Pool across conditions, one dot per source, and use source-level
  permutation only where the number of held-out sources makes it meaningful.
- Drop the rotation-invariance arm as a finding. The training used a small y-axis rotation (max 8 degrees), so rotation
  invariance is expected by construction and can appear only as a manipulation check, never as a falsifiable result.
- Say "transductive" next to every number. A held-out probe split is still transductive because the frozen encoder saw
  every row during the curriculum.

## 7. What could happen, and what each outcome would mean

Notebook 05b simulates four canonical futures against the exact margins above and lays them out as two yes-or-no
questions crossed into a two-by-two grid in [`./images/fig4.svg`](./images/fig4.svg) (rows = does it read, columns =
does it flip). The decision rule is total: every outcome maps to one clear, pre-registered claim.

| Future | Shape | Primary verdict | Mirror verdict | What it licenses |
|---|---|---|---|---|
| F1 clean-flip positive | Lane A near the ceiling, mirror on the y = -x line | Signed axis present above raw | Flips | The signed axis is carried competitively AND the encoding cleanly flips under the mirror (confirms the reflection-equivariant reading; the substrate for Idea 9) |
| F2 decodable but non-flipping | Lane A near the ceiling, mirror shallow | Signed axis present above raw | Does not flip | Decodability is licensed; equivariance is WITHHELD (the encoding reads side but does not respect the mirror) |
| F3 informative null | Lane A a cloud far from the ceiling | Informative null | Does not flip | Negative result: the frozen tokens do NOT add the signed axis above raw coordinates; overturns the belief that the checkpoint built a laterality axis |
| F4 artifact | Lane A strong BUT Lane D also fires | (withdrawn) | (moot) | The signed claim is WITHDRAWN: a side-blind pooled control cannot carry a signed quantity, so Lane A was reading a magnitude or acquisition artifact |

Given the project's prior that asymmetry is the weakest-decoded scalar (R-squared about 0.154), F2 or F3 are the more
likely futures ahead of time, and both are worth publishing. F2 sharpens the earlier pooled result by showing the side is
present but the symmetry is not respected. F3 is a clean negative for a representation audit, which reviewers value under
the ICLR/ICML framing that rewards informative nulls (see [`../_shared_facts.md`](../_shared_facts.md), reviewer framing).
F1 is the strong positive that would promote the reflection-equivariant idea (Idea 9). F4 is the trap the Lane D control
exists to catch.

How to read fig4: these are drawn expected shapes, not real data. It is a two-by-two grid built from two yes-or-no
questions. Pick the row for your READ answer (does the signed number come out and reach 80 percent of the raw-coordinate
baseline) and the column for your FLIP answer (does a mirror negate it). The box you land on names the one honest claim
that outcome allows. It is a picture of "here is what each possible answer licenses," committed to before we see the real
data.

## 8. What this cannot tell us

- Transductive, so no out-of-sample claim. The encoder saw every evaluation row during training, so no number here is a
  fresh-people performance estimate. These are diagnostics on a frozen encoder. A truly held-out estimate would need
  retraining the whole curriculum inside each outer source split, which this study does not do.
- Tiny, unequal source counts. With as few as one source per class, the pooled endpoint and the source-as-a-dot plots are
  the only defensible readouts. Any per-class asymmetry number would be one point dressed up as a distribution.
- Provenance and label overlap. Normal is one video on a mostly-augmented path; the canonical-path subset softens this but
  cannot fully remove it at this sample size.
- Duration warping erases absolute cadence. The 64-frame resize means this notebook can speak to signed spatial asymmetry
  but not to the Parkinson's rhythm biomarker (stride-time CV); that boundary belongs to Idea 4.
- Monocular capture. gavd5 is single-view, so the view-stability and genuine-mirror questions can only be answered on the
  external, non-clinical multi-view cohorts, and even there the claim is about the reflection property, not diagnosis.
- Skeleton limits. Skeletons cannot recover forces, muscle-electrical activity, spasticity, out-of-plane rotation, or a
  muscle diagnosis, so no clinical-accuracy claim is made on gavd5 at any outcome.

## 9. How to make it reproducible

- Two notebooks carry the work.
  [`../../../nb_05a_signed_laterality_probe.ipynb`](../../../nb_05a_signed_laterality_probe.ipynb) is the decisive probe: it
  copies the S-JEPA model classes verbatim so `load_state_dict` matches key for key, loads the `d0acc262` checkpoint under
  the notebook-05 guards, caches the frozen target-encoder features, fits Lanes A through D with source-video-disjoint
  ridge probes, runs the mirror pass, applies the pre-registered margins, and writes
  `idea5_signed_laterality_result.json`. It runs in `GAVD_MODE=real` (reads the real checkpoint and pose cache) and
  degrades gracefully to `GAVD_MODE=smoke`, which reuses synthetic fixtures plus one clearly-labelled signed lean overlay
  so the plumbing runs end to end; smoke numbers are illustrative only.
- [`../../../nb_05b_reflection_reach_and_futures.ipynb`](../../../nb_05b_reflection_reach_and_futures.ipynb) is the
  possible-futures simulator (writes `idea5_futures_bundle.json` and the futures figure) and the honestly-stubbed external
  multi-view reach scaffold.
- Determinism. Fix the seeds. The smoke lean overlay is deterministic. Save the frozen target function, the
  source-video-disjoint split manifest, and the per-source results. A future real run diffs its
  `idea5_signed_laterality_result.json` against the four canonical futures in `idea5_futures_bundle.json`, so anyone can
  check which future actually happened.

## Responsible use

The folder labels (normal, parkinsons, stroke, myopathic, cerebral_palsy) are dataset annotations from GAVD (Ranjan et al.,
IEEE Access 2025, DOI 10.1109/ACCESS.2025.3545787), not diagnoses made by this project. The signed laterality value is a
representation diagnostic computed from cached skeleton coordinates. It is not a validated clinical measurement of any
individual's health. Every result here is transductive and small-sample, with the source video as the independent unit.

## References

- Abdelfattah and Alahi, S-JEPA, ECCV 2024, DOI 10.1007/978-3-031-73411-3_21.
- Assran et al., I-JEPA, CVPR 2023, arXiv:2301.08243; Bardes et al., V-JEPA, 2024, arXiv:2404.08471.
- Grishchenko et al., BlazePose GHUM, 2022, arXiv:2206.11678.
- Ranjan et al., GAVD, IEEE Access 2025, DOI 10.1109/ACCESS.2025.3545787.
- Kapoor and Narayanan, arXiv:2207.07048 (leakage taxonomy; source video as the independent unit).
- Varoquaux, NeuroImage 2018 (small-sample error bars).
- Patterson et al., Gait and Posture 2010, PMID 19932621 (gait Symmetry Ratio biomarker).
- Natali and Javed, StatPearls, corticospinal tract anatomy, PMID 30571044.
- Riederer and Sian-Hulsmann, J Neural Transm 2012, PMID 22367437 (asymmetric nigrostriatal onset).
- Volpe, Lancet Neurol 2009, PMID 19081519 (periventricular injury and corticospinal fibers).
- Barohn et al., Neurol Clin 2014, PMID 25037080 (symmetric proximal weakness distribution).
- Xiong et al., Biomed Eng Online 2023, PMID 37525241 (Duchenne: no significant left-right asymmetry; myopathy negative
  class).
- Hausdorff et al. 1998, PMID 9613733; Schaafsma et al. 2003, PMID 12809998 (stride-time variability; out of scope here,
  owned by Idea 4).
- Yu, Tan, Tan, CASIA-B, 2006; Takemura et al., OU-MVLP-Pose, 2018 (non-clinical multi-view pose cohorts); Zhu et al.,
  GREW, arXiv:2205.02692; Zheng et al., Gait3D, arXiv:2204.02569.
