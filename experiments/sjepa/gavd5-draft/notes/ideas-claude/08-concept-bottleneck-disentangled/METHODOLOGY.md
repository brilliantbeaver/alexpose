# Idea 8, in plain words: can we build three named "meaning slots" in the model and steer them one at a time?

This is the "how to actually do it" guide for Idea 8, written so a motivated high-school student can follow every
step. The science comes from the folder's [`README.md`](./README.md). Every number here is kept true to
[`../_shared_facts.md`](../_shared_facts.md) (the single source of truth for numbers) and
[`../_neuro_facts.md`](../_neuro_facts.md) (the biology). If a number is not in those files, it is not in here, and any
made-up example is clearly labelled "illustrative numbers only".

One honest note up front, repeated at the end because it matters: the folder labels (normal, parkinsons, stroke,
myopathic, cerebral_palsy) are just tags that came with the GAVD dataset. They are not diagnoses made by this project.
And every core result here is transductive, which means the model was trained on the very clips we later test it on.
More on what that costs us below.

## The big idea in plain words (60-second version)

This project already taught a computer to describe how a person walks, using a list of numbers. Right now that list is a
jumble: every number mixes together many different things about the walk at once. This idea is about tidying that up.

We want to retrain the model so that three small, clearly labeled chunks of that number list each stand for exactly one
real, doctor-trusted measurement of walking:

- one chunk for how ONE-SIDED the walk is (does it lean left or right),
- one chunk for how SHAKY the walking rhythm is (does the timing wobble step to step),
- one chunk for POSTURE (does the pelvis tip forward).

Then we test the model like a row of light switches. Flip the "one-sidedness" switch and watch: if only the one-sidedness
reading changes, and the "rhythm" and "posture" readings stay put, the switches are clean. If flipping one switch drags
the other two readings around, the meanings are still tangled together.

Here is a homey analogy. Think of an old stereo with three knobs: bass, treble, and volume. A well-built stereo lets you
turn up the bass without touching the treble or the volume. A badly-built one has knobs that fight each other, so turning
"bass" also changes the volume. We are asking whether we can build a walking-description whose three knobs each do one job
and leave the others alone. A description like that is not just readable, it is steerable, one meaning at a time.

## Mini-glossary (the words this idea actually uses)

- SKELETON: a moving stick figure. A pose detector finds body joints in each video frame, so a clip becomes a set of
  moving dots instead of pixels. Smaller than video, and it hides the face and clothing.
- TOKEN: the smallest chunk the model reads. Here, one joint watched over a short 4-frame window.
- EMBEDDING / FINGERPRINT: a short list of numbers the model uses to describe an input. Here each token becomes a list of
  64 numbers.
- LATENT / SUBSPACE: a latent is one of the numbers in that fingerprint. A subspace is a named group of them. Our named
  groups are `z_asym`, `z_rhythm`, and `z_posture`, plus a leftover group `z_free`.
- BOTTLENECK: a deliberately narrow passage information has to squeeze through, like a funnel. By forcing the walk's
  meaning through a small set of named blocks, we push the model to pack each real mechanism into its own block instead
  of smearing it across everything. "Concept-bottleneck" just means that narrow passage is organized into human-named
  concepts (one-sidedness, rhythm, posture).
- S-JEPA (Skeleton Joint-Embedding Predictive Architecture): a model that learns by hiding part of a skeleton and
  predicting the hidden part in fingerprint-space, with no human labels needed for the core learning step (Abdelfattah and
  Alahi, S-JEPA, ECCV 2024, DOI 10.1007/978-3-031-73411-3_21).
- ENCODER: the part of the model that turns a skeleton into embeddings.
- EMA TEACHER: a slow, steady copy of the encoder that provides the answer key the main encoder tries to match. EMA means
  "exponential moving average", a running average that changes slowly. It is not trained by the usual learning step.
- PREDICTOR: a small part of the model that guesses the hidden tokens' fingerprints.
- MASKING: hiding some tokens from the model and asking it to guess what was there, like covering part of a photo with
  your hand and guessing what is behind it.
- HEAD / READOUT / PROBE: a tiny, simple rule attached on top of the fingerprint to predict one specific number (here, one
  biomarker).
- BIOMARKER: a measurable body signal a doctor trusts. Our three are the Symmetry Ratio, stride-time CV, and anterior
  pelvic tilt.
- STEERING / INTERVENTION: nudging one named block of numbers and watching what changes.
- DISENTANGLED: the good outcome where each named block carries one meaning only, so nudging one does not move the others.
- VICReg: an anti-collapse tool. It keeps each number spread out and stops different numbers from copying each other. We
  reuse its "stop copying" rule to keep our three named blocks from leaking into each other.
- R-SQUARED: a score from 0 to 1 for how well a readout tracks the true value. 1 is perfect, 0 is no better than always
  guessing the average.
- RIDGE READOUT: ordinary line-fitting with a small brake added so it does not chase noise. Picture the steadiest
  straight-line relationship between the joint positions and the biomarker. We use it as the no-neural-network baseline.
- EFFECTIVE RANK: a health check on the fingerprint. It counts how many of the 64 numbers are really pulling their own
  weight instead of just copying each other. A high number means the fingerprint uses its full range; a low number warns
  that the model has quietly collapsed to a few repeated values.
- RAW-COORDINATE CEILING: how well you can predict a biomarker straight from the plain joint positions, with no neural
  network. The learned block has to keep up with this to earn credit.
- TRANSDUCTIVE: the model was trained on the very clips we later test on. A high score can mean it memorized them, not
  that it learned something that transfers to new people.
- SOURCE-VIDEO-DISJOINT: when we split data for testing, we hold out whole YouTube source videos, never single clips. The
  source video, not the single clip, is the real unit of evidence.

## 1. The question in one sentence

After a full curriculum retrain with three biomarker-supervised latent subspaces, does intervening on one named subspace
move only its mechanism-linked biomarker (the symmetry ratio, the stride-time CV, or the anterior pelvic tilt) and leave
the other two biomarkers unmoved, by a margin we fix in advance, and only where that steerability beats a raw-coordinate
probe ceiling?

There are really two things being asked at once, and it helps to keep them apart:

- Does each named block actually CARRY its own biomarker (can a simple readout pull the right number out of it, better
  than plain joint positions would)?
- When we NUDGE one block, does only its own biomarker move, while the other two stay put?

A block only "passes" if both of those hold at the same time.

## 2. Why this idea, in plain words

The motivation is a piece of neuroscience: different walking problems break walking in different ways, and doctors already
have a separate, trusted number for each way. The full mechanism chains live in
[`../_neuro_facts.md`](../_neuro_facts.md); here is the plain version.

- Some problems are ONE-SIDED (lateralized), meaning they hurt one side of the body more than the other. A stroke is
  damage on one side of the brain, and because the main motor nerve wires cross over on their way down (a crossing point
  called the pyramidal decussation), damage on one side of the brain weakens the opposite side of the body (Natali and
  Javed, StatPearls, PMID 30571044). One kind of cerebral palsy comes from a one-sided brain injury early in life that
  hits the leg wiring (Volpe 2009, PMID 19081519). Early Parkinson's often starts on one side too, because the brain
  chemical loss starts on one side (Riederer and Sian-Hulsmann 2012, PMID 22367437). The trusted way to measure "how
  one-sided is this walk" is the Symmetry Ratio (Patterson et al. 2010, PMID 19932621). This is what `z_asym` is standing
  in for.
- Parkinson's has a second, different fingerprint: the walking rhythm gets shaky. A brain region called the basal ganglia
  loses its grip on automatic movement (Redgrave et al. 2010, PMID 20944662; Wu, Hallett, Chan 2015, PMID 26102020), so
  the time between steps wobbles from one step to the next. The trusted way to measure that wobble is the stride-time
  coefficient of variation (stride-time CV, a "how bumpy is the timing" percentage). People who fall sit near 8.8 percent;
  people who do not fall sit near 4.2 percent (Schaafsma et al. 2003, PMID 12809998; Hausdorff et al. 1998, PMID
  9613733). This is what `z_rhythm` is standing in for.
- Myopathy is different again. It is a muscle disease that weakens both sides fairly evenly, mostly the muscles close to
  the body's center (Barohn et al. 2014, PMID 25037080). It does NOT make walking one-sided (Xiong et al. 2023, PMID
  37525241) and it does NOT wreck the rhythm; the step rate stays roughly normal. What it does is tip the posture: weak
  hip muscles let the pelvis tilt forward, so the anterior pelvic tilt (the forward lean of the pelvis, in degrees) goes
  up, 16.4 versus 11.6 degrees in one Duchenne muscular dystrophy group (Vandekerckhove et al. 2022, PMID 35721358; weak
  hip extensors drive that forward tilt, Vandekerckhove et al. 2025, PMID 41034979). A related crouched-knee posture in
  cerebral palsy is defined as a minimum stance knee flexion of at least 30 degrees (de Morais Filho et al. 2010, PMID
  20300011). This is what `z_posture` is standing in for.

So there are three separate ways walking breaks: one-sidedness, rhythm wobble, and posture. The problem is that the
features this model learned mash all three of those things into one undifferentiated list of 64 numbers per token. An
earlier in-project check (notebook 05) could read one thing (step size) back out pretty well (R-squared about 0.719,
meaning the readout tracks the real value well), but read one-sidedness back poorly (R-squared about 0.154, a weak match),
and the rhythm wobble could not be read out at all from the short (roughly 2-second) clips.

Now the honest catch, which is a proven fact, not just an opinion. Locatello et al. 2019 (ICML) proved that you cannot get
cleanly separated meanings for free from unsupervised learning alone. You have to add either a built-in bias or some
supervision (some form of answer key). That is exactly why we bolt on the biomarker readouts and the biased masking: they
supply that needed push. We say this up front so no one thinks we expect the clean split to fall out of self-supervision
by magic. See [`./images/fig3.svg`](./images/fig3.svg) for the plain picture of the whole idea (the 64-number fingerprint
drawn as a row of boxes, with three colored, labeled blocks each wired to one light switch). How to read that picture:
imagine flipping the `z_asym` switch and watching only the "one-sidedness" bulb brighten while the other two stay off;
that clean one-switch-one-bulb wiring is exactly what we are testing for.

## 3. What data you need

### 3.1 The main data: the gavd5-draft GAVD cohort (internal only)

The core work runs on the canonical GAVD cohort (Ranjan et al., IEEE Access 2025, DOI 10.1109/ACCESS.2025.3545787):
96 sequences from 18 unique YouTube source videos. The per-condition source-video counts are tiny and lopsided: normal 1,
Parkinson's 2, stroke 3, myopathic 10, cerebral palsy 2. All 12 normal sequences come from a single video (`3KnFt8bH3tE`).

Two consequences fall out of those small numbers, and they shape the whole design.

1. The condition label is almost the same thing as "which video did this come from," especially for normal. So the honest
   unit of evidence is the SOURCE VIDEO, not the individual clip. Two clips from one video are not two independent facts,
   any more than two frames of one movie are two different movies (Kapoor and Narayanan, arXiv:2207.07048; and Varoquaux,
   NeuroImage 2018, on why tiny samples give big error bars).
2. Because some conditions have as few as one source video, you cannot report a per-condition "hold out one source"
   number. One held-out point is not a distribution. So the steering result is pooled across all conditions and every
   source video is plotted as its own dot.

There is also a PROVENANCE catch. Most normal rows were built through an "augmented" extraction path, while every abnormal
row used the "canonical" path (see [`../_shared_facts.md`](../_shared_facts.md)). If we are not careful, a decoded axis
could accidentally be reading the difference between those two processing pipelines instead of the difference in walking.
To avoid that, the primary comparison runs only on the provenance-matched (canonical-path) subset.

### 3.2 The reach data: PhysioNet Gait-in-PD, cross-modal and label-level only

There is no public skeleton dataset that pairs clinical labels for these conditions with pose while keeping different
people in train and test. So there is no honest skeleton-level clinical transfer test for the named blocks. The one reach
check that is honest is for `z_rhythm`: download PhysioNet Gait-in-PD (gaitpdb, 93 people with Parkinson's plus 73
controls, Hausdorff, DOI 10.13026/C24H3N) and confirm the stride-time-CV biomarker at the LABEL level only. gaitpdb is
force and inertial-sensor data, not skeleton, so this is a cross-modal check that the wobble biomarker separates
Parkinson's from controls; it is NOT a claim of skeleton-level clinical transfer of `z_rhythm`.

For `z_asym` and `z_posture` there is no participant-disjoint public SKELETON cohort (no cerebral palsy crouch or myopathy
anterior-pelvic-tilt skeleton set), so those two blocks cannot be externally confirmed at the skeleton level. We say so
plainly. If you want non-clinical multi-view pose data to stress unrelated properties, the verified reach-tier
(non-clinical) cohorts in [`../_shared_facts.md`](../_shared_facts.md) are CASIA-B, OU-MVLP-Pose, GREW, Gait3D, and
Human3.6M, but none of them carries these clinical biomarkers, so they do not confirm any named block here.

### 3.3 What the data looks like, and how a team would get it

Each sequence is a skeleton: 33 body joints tracked over time, each joint carrying an x, a y, and a relative z coordinate
(MediaPipe BlazePose, Grishchenko et al. 2022, arXiv:2206.11678). A real team obtains it by running the pose detector on
walking videos, then cleaning the result: fill short gaps where the detector blinked, center the skeleton on the pelvis,
scale it to a standard body size, and stretch or squeeze the clip to exactly 64 frames so every sequence is the same
length. Then 4 frames in a row are grouped into one time patch, giving 16 time positions; with 33 joints that is
33 times 16 = 528 possible joint-time tokens. Heels are the weak link for the detector (left heel visible about 70 percent
of the time, right heel about 67 percent, versus about 99 percent for shoulders and hips), which is one more reason to
lean on the pelvis, hips, knees, and ankles.

## 4. Step by step, how to do it

Honest heads-up first: unlike the frozen-encoder ideas in this folder, THIS one is a RETRAIN. The whole point is to change
what the model learns so that three named blocks carry three meanings, and you cannot do that by only reading a frozen
model. So the core is a full five-stage curriculum retrain (roughly 6 to 8 weeks of effort, the most expensive item in the
portfolio). We keep the S-JEPA shape the same (33 joints times 16 time positions, embed_dim 64, depth 2, 4 heads) and the
same five-stage curriculum, and we ADD three small readouts plus a "keep the blocks apart" term. Everything else reuses
the existing code and the cached 528-token tensors.

Here is the recipe, marking each step as zero-retrain or part of the retrain.

1. Fix the block layout (zero-retrain, do it first and write it down). Set aside three back-to-back slices of the 64-number
   per-token fingerprint as `z_asym`, `z_rhythm`, and `z_posture`, plus a fourth unnamed leftover block `z_free` that
   soaks up everything else, so the named blocks are not forced to explain all the variation. The slice boundaries are
   fixed and logged BEFORE training so they cannot be tuned after the fact.

   ```python
   # Fixed block layout inside the 64-number embedding (logged before training).
   BLOCKS = {"asym": slice(0, 12), "rhythm": slice(12, 24),
             "posture": slice(24, 36), "free": slice(36, 64)}
   ```

2. Fix three biomarker "answer key" functions from raw joint positions (zero-retrain, before training). Each answer key is
   a fixed recipe computed from the cached BlazePose positions in the standard 64-frame time base. It is a MEASUREMENT,
   never a diagnosis.
   - `y_asym`: the signed Symmetry Ratio style contrast on left-versus-right step length and swing and stance timing
     (Patterson 2010, PMID 19932621), using the exact left-right pair anatomy.
   - `y_rhythm`: a step-to-step timing-wobble stand-in in the standard time base (the stride-time CV idea of Hausdorff
     1998, PMID 9613733). We flag up front, per [`../_shared_facts.md`](../_shared_facts.md), that stride-time CV is not
     readable from roughly 2-second windows, so `z_rhythm` is the RISKIEST block and its raw-coordinate ceiling may itself
     be low.
   - `y_posture`: side-view anterior pelvic tilt and trunk-lean angle plus minimum stance knee flexion (Vandekerckhove
     2022, PMID 35721358; de Morais Filho 2010, PMID 20300011).

3. Compute the raw-coordinate ceiling for all three biomarkers FIRST (zero-retrain). Before any retraining, fit a simple
   ridge readout on handcrafted coordinate features (no neural network) for each biomarker. This is Lane B below, the bar
   each named block must keep up with. Doing it first tells you which biomarkers are even worth chasing (especially for
   `z_rhythm`).

4. Retrain the model with the three named readouts (RETRAIN). Attach a small LINEAR readout on each named block: the
   `z_asym` readout predicts `y_asym`, the `z_rhythm` readout predicts `y_rhythm`, the `z_posture` readout predicts
   `y_posture`. Each readout is deliberately simple (a straight-line rule), so the demand is "the biomarker must show up
   in this block in a plain, straight-line way", which is the exact claim we can test.

5. Add block supervision and a "keep the blocks apart" term to the loss (RETRAIN). A loss is a score of how wrong the
   model is; smaller is better. The existing loss already has three parts:

   `L = L_JEPA + 0.05 * L_VICReg + 0.25 * L_group`

   We extend it with a prediction term per block and a between-block decorrelation penalty (the same VICReg covariance
   rule, arXiv:2105.04906, now applied BETWEEN blocks):

   `L = L_JEPA + 0.05 * L_VICReg + 0.25 * L_group + a * (L_asym + L_rhythm + L_posture) + b * L_decorr`

   Reading the math (the augmented loss). This says we add two new pieces to the existing three-part loss.
   - `L_JEPA` is the main prediction error: how badly the predictor guessed the hidden fingerprints. Its weight is 1.
   - `L_VICReg` (weight 0.05) is an anti-collapse penalty; `L_group` (weight 0.25) pulls same-condition examples together.
   - `L_asym`, `L_rhythm`, `L_posture` are how wrong each named readout is at predicting its own biomarker; smaller means
     the biomarker is well carried by its block.
   - `L_decorr` punishes overlap BETWEEN the three named blocks; smaller means the blocks share less information, which is
     what "disentangled" means.
   - `a` is the weight on the biomarker readouts and `b` is the weight on the keep-apart term. Both are new knobs, chosen
     ONLY using the training videos. If `a` is 0 the blocks are not named at all (back to the original model). If `b` is 0
     the blocks can freely overlap, so steering one will drag the others.
   - We do NOT let the new terms overpower `L_JEPA`; the prediction task stays the main goal, so `a` and `b` are set ahead
     of time to keep the sum of the new terms below the `L_JEPA` size on the training videos.

6. Bias the masking toward each block's landmarks per stage, WITHOUT ever reading motion (RETRAIN). The safe sampler still
   hides a fixed count of tokens and never looks at coordinate size, displacement, velocity, acceleration, or any learned
   motion score (the motion-aware tricks called MAMP and MTM stay forbidden). The only allowed bias is anatomical:
   over-weight the left and right paired joints when the asymmetry readout is active, ankle timing tokens for rhythm, and
   pelvis and knee side-view tokens for posture. The global hide cap stays 12 out of 33 = 0.364 (about 36 percent, far
   below the 75 to 90 percent that image and video JEPAs hide), and at least one allowed token always stays visible.

7. Do the steering test on held-out sources (part of the evaluation, after retrain). Split into named blocks, push ONLY
   one block along its readout direction, hold the rest fixed, and read all three biomarkers before and after.

   ```python
   import numpy as np

   def decode_biomarkers(z, heads):
       # heads[name] is the trained linear head for that named block.
       return {name: heads[name] @ z[BLOCKS[name]] for name in ("asym", "rhythm", "posture")}

   def intervene(z, target_block, delta, heads):
       # Push ONLY the target block along its head direction, hold the rest fixed.
       z_new = z.copy()
       z_new[BLOCKS[target_block]] = z[BLOCKS[target_block]] + delta
       before = decode_biomarkers(z, heads)
       after = decode_biomarkers(z_new, heads)
       # Disentangled iff only the target biomarker moved.
       return {name: after[name] - before[name] for name in before}
   ```

   Reading the math (the steerability ratio). For each nudge we measure how much the OWN biomarker moved versus how much
   the OTHER two moved. Let d_own be the change in the biomarker of the block we pushed, and d_other be the biggest change
   among the two blocks we did NOT push (both measured in each biomarker's own standardized units, meaning rescaled so
   they are comparable). The steerability ratio is d_own divided by d_other. A big ratio (own moves, others do not) means
   the axes are cleanly separated. A ratio near 1 means pushing one axis drags the others, which is tangling.

8. Apply the pre-registered decision rule (Section 5), run the controls (Section 6), assemble per-source dots, and write
   transductive caveats next to every number.

## 5. The decision rule, decided in advance

We fix the bar BEFORE looking at results so we cannot move the goalposts. The split is stated first, and folds are
SOURCE-VIDEO-DISJOINT: we hold out whole YouTube source videos, never single clips, because a held-out clip from a video
the model already saw is still transductive (it can just be memorized). The steering endpoint is pooled across conditions
with every source video shown as its own dot, and the comparison runs on the provenance-matched (canonical-path) subset.

For a block to count as DISENTANGLED and STEERABLE, all three of these must hold at the same time on held-out sources:

1. Its readout recovers its own biomarker with a held-out-source R-squared at least 80 percent of that biomarker's
   raw-coordinate ceiling (Lane B).
2. Its steerability ratio (own change over biggest other change) is at least 3.
3. The biggest cross-biomarker leak is no more than 0.2 in standardized units per unit of own-biomarker change.

Any block missing any one of the three is scored as an INFORMATIVE NULL for that block: the biomarker readouts plus VICReg
plus biased masking did not separate that axis. That is a real answer, not a failure.

Reading the math (the three margin numbers):
- 80 percent (a fraction of 0.80) is the share of the raw-coordinate ceiling the readout must reach, so the learned block
  is at least as useful as plain joint positions.
- 3 is the smallest steerability ratio that counts as "mostly the own axis moved": the own biomarker must move at least
  three times as much as the worst-case other biomarker.
- 0.2 (standardized units of leak per unit of own change) caps the spillover; above it, pushing one axis meaningfully moves
  another.

### A worked example (illustrative numbers only, not measured facts)

Suppose we push the `z_asym` block and, after rescaling into standardized units, we see these changes:

- one-sidedness reading (own) changes by 1.0
- rhythm reading changes by 0.15
- posture reading changes by 0.10

Step 1: d_own = 1.0. Step 2: d_other = the bigger of the two other changes = 0.15. Step 3: steerability ratio =
d_own / d_other = 1.0 / 0.15 = about 6.7. Step 4: the biggest leak into another axis is 0.15 per unit of own change.

Now check the three thresholds. The ratio 6.7 is above the required 3, good. The biggest leak 0.15 is below the required
0.2, good. If, in addition, the `z_asym` readout reaches at least 80 percent of the raw-coordinate ceiling for
one-sidedness, then `z_asym` PASSES as a clean, steerable axis.

Now a failing case. If instead the rhythm reading had changed by 0.5 when we pushed `z_asym`, the ratio would be
1.0 / 0.5 = 2.0, which is below 3, and the leak 0.5 is above 0.2, so `z_asym` would FAIL and be scored as an informative
null. (These are illustrative numbers only, not measured facts.)

## 6. Controls that keep us honest

- The RAW-COORDINATE CEILING (Lane B). This is the non-neural baseline for each biomarker: a simple ridge readout on
  handcrafted coordinate features, no neural network. A named block gets credit only if it keeps up with what the plain
  joint positions already give you for free. Steering that does not clear the ceiling gets no credit.
- The UNTRAINED-ENCODER FLOOR (Lane C). A random, untrained encoder of the same shape and the same block layout sets the
  "pure chance" level. A named block must clearly beat this.
- The MEAN/STD-POOLED NUISANCE CONTROL (Lane D). Averaging and spreading the tokens throws away time order and side
  identity, so it must NOT recover a signed one-sidedness axis. If it does, the `z_asym` claim is an artifact and gets
  withdrawn.
- The SHUFFLED-BIOMARKER control. Scramble each biomarker answer across sources; this must knock every readout back down
  to its raw-coordinate floor. If a readout still works with scrambled answers, it was reading the source's identity, not
  the biomarker.
- ABLATE the keep-apart term. Train once with `b > 0` and once with `b = 0`. The separation (steerability ratio) must
  improve with `b > 0`, otherwise the keep-apart term is not doing the work we claim.
- SOURCE-VIDEO-DISJOINT splits. Hold out whole source videos, never single clips. The knobs `a` and `b` and any readout
  penalty are chosen using training sources only, so held-out sources never leak in.
- ONE fingerprint. The new run gets its own written-down fingerprint. The original curriculum-final checkpoint prefix
  `d0acc262` (and the observed canonical lineage prefix `dba24a`) are used only as the Lane E ablation (the original model
  with no named heads, which should look entangled: steering one axis drags the others). Every number is tied to one
  fingerprint before any comparison.
- The `z_rhythm` HONESTY control. Because stride-time CV is not readable from roughly 2-second windows, the `z_rhythm`
  raw-coordinate ceiling is reported first. If that ceiling is near chance, `z_rhythm` cannot pass, and we report it as a
  limit of the short window length, not a failure of the bottleneck idea.
- No per-class hold-one-source margins on a single held-out source. Steering is pooled across conditions, every source is
  a dot, and source-level permutation is used only where the number of held-out sources makes it meaningful.

The five lanes, at a glance:

| Lane | What it reads | Retrain? | Role | What we expect in advance |
|---|---|---|---|---|
| A named-subspace heads | Retrained `z_asym`/`z_rhythm`/`z_posture` blocks | Yes | Primary | Each head at least 80% of its biomarker ceiling; steerability ratio at least 3; leak at most 0.2 |
| B raw-coordinate ceiling | Handcrafted per-biomarker coordinate features | No | Non-neural ceiling | Reference target per biomarker |
| C untrained-encoder floor | Random-init encoder, same block layout | No | Floor | Near chance |
| D mean/std-pooled control | Side-blind pooled tokens | No | Nuisance | Must NOT recover signed asymmetry |
| E original d0acc262 (no named heads) | Frozen curriculum-final features | No | Ablation | Entangled: steering one axis drags the others |

## 7. What could happen, and what each outcome would mean

The decision rule is total: every outcome maps to one clear, pre-registered claim. The two decisive pictures are
[`./images/fig1.svg`](./images/fig1.svg) (the steerability matrix) and [`./images/fig2.svg`](./images/fig2.svg)
(per-block biomarker recovery against the ceiling and floor).

How to read fig1 (the steerability matrix): rows are the block we nudge (`z_asym`, `z_rhythm`, `z_posture`); columns are
the change seen in each biomarker (symmetry ratio, stride-time CV, anterior pelvic tilt). A clean grid lights up only on
the diagonal (top-left to bottom-right), meaning each block moves its own biomarker and leaves the others near blank. A
grid that lights up all over is the "no" result: pushing one axis drags the others.

How to read fig2: each dot is one held-out source video. A block "wins" when its dots sit above the raw-coordinate ceiling
line (Lane B). Dots stuck near the untrained floor line (Lane C) mean the learned block adds nothing over plain input.
`z_rhythm` is flagged as the riskiest block, because stride-time CV is not readable from roughly 2-second windows.

The possible futures:

| Future | Shape | What it licenses |
|---|---|---|
| All three blocks pass | Each head clears 80% of its ceiling; steering ratio at least 3; leak at most 0.2 | The strong positive: biomarker heads plus VICReg plus biased masking build named, causally steerable subspaces |
| Some blocks pass, others null | For example `z_asym` and `z_posture` pass, `z_rhythm` misses because its ceiling is near chance | A partial, honest result: the design separates some mechanisms but not the rhythm one from short windows |
| No block passes the steering test | Nudging one axis drags the others just as much | Rules out the belief that these tools separate these three mechanisms in a small skeleton JEPA |
| No block beats its ceiling | Steered blocks never clear the raw-coordinate ceiling | Rules out the belief that the learned description adds steerable structure beyond plain joint positions |

All of these are worth reporting. ICLR 2026, ICML 2026, and NeurIPS 2026 reward informative negative results that change
understanding (see [`../_shared_facts.md`](../_shared_facts.md), reviewer framing), because a clear "no" tells the next
builder that the bottleneck DESIGN, not the idea of named axes, is what needs to change.

## 8. What this cannot tell us

- Transductive, so no out-of-sample claim. The encoder saw the evaluation rows during the curriculum, so no core number
  here is a fresh-people performance estimate. Where a fold-local encoder was trained without a fold's videos, we mark the
  number as transductive only for that fold; otherwise it is fully transductive.
- Tiny, unequal source counts. With as few as one source per class (normal 1, Parkinson's 2, stroke 3, myopathic 10,
  cerebral palsy 2), the pooled endpoint and the source-as-a-dot plots are the only defensible readouts. Any per-class
  number would be one point dressed up as a distribution.
- Provenance and label overlap. Normal is one video on a mostly-augmented path; the canonical-path subset softens this but
  cannot fully remove it at this sample size.
- The riskiest block. Stride-time CV is not readable from roughly 2-second windows, so `z_rhythm` may never clear its own
  ceiling. That is a limit of the short window length, not proof the bottleneck idea is wrong.
- Monocular capture. gavd5-draft is single-view (one camera), which bounds how well side-view posture angles can be recovered.
- Skeleton limits. Skeletons cannot recover forces or push-off (Bowden), muscle-electrical activity or spasticity
  (Ropars), twisting rotation, or any underlying muscle-disease diagnosis (Stenum et al. 2021, PMID 33891585, sets what
  skeletons CAN recover: timing to about 0.02 seconds per step and side-view joint angles to within 4 to 7 degrees). No
  claim here depends on the things skeletons cannot recover.
- No external skeleton confirmation for two blocks. `z_asym` and `z_posture` have no participant-disjoint public skeleton
  cohort, so they cannot be confirmed on new people at the skeleton level. The gaitpdb reach arm only confirms the wobble
  biomarker at the label level in a force and inertial-sensor cohort, not skeleton-level clinical transfer of `z_rhythm`.

## 9. How to make it reproducible

- Bind to ONE fingerprint. The augmented retrain logs its own written-down fingerprint. The original `d0acc262` (and the
  canonical lineage prefix `dba24a`) appear only as the Lane E ablation, and every number is tied to one fingerprint
  before any comparison.
- Fix the seeds, and keep the same shape as the original run (embed_dim 64, depth 2, 4 heads) so the retrain differs from
  the baseline only in the named heads and the keep-apart term.
- Freeze BEFORE fitting and save: the fixed block layout (the slice boundaries), the three biomarker answer-key functions,
  and the raw-coordinate-ceiling readouts. These are logged before any result so they cannot be tuned after the fact.
- Save the source-video-disjoint split manifest (which videos are held out in each fold), the health metrics of the
  retrain (per-dimension spread, effective rank, between-block covariance, so you can prove there was no collapse), and the
  per-source results with a transductive caveat attached to every number.
- Save the ablation pair (`b > 0` versus `b = 0`) and the shuffled-biomarker control results alongside the main run, so a
  future reader can check that the keep-apart term earned its place and that the readouts learned the biomarker and not the
  source's identity.

## Responsible use

The folder labels (normal, parkinsons, stroke, myopathic, cerebral_palsy) are dataset annotations from GAVD (Ranjan et
al., IEEE Access 2025, DOI 10.1109/ACCESS.2025.3545787), not diagnoses made by this project. The three biomarkers are
representation diagnostics computed from cached skeleton coordinates; they are not validated clinical measurements of any
individual and must not be read as such. All core results are transductive and small-sample, with the source video as the
independent unit. The gaitpdb reach arm confirms the wobble biomarker's clinical signal at the label level in a force and
inertial-sensor cohort; it does NOT establish skeleton-level clinical transfer, and no public skeleton cohort exists to
externally confirm `z_posture`.

## References

- Abdelfattah and Alahi, S-JEPA, ECCV 2024, DOI 10.1007/978-3-031-73411-3_21.
- Assran et al., I-JEPA, CVPR 2023, arXiv:2301.08243; Bardes et al., V-JEPA, 2024, arXiv:2404.08471.
- Bardes, Ponce, LeCun, VICReg, ICLR 2022, arXiv:2105.04906.
- Grishchenko et al., BlazePose GHUM, 2022, arXiv:2206.11678.
- Locatello et al., "Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations", ICML
  2019.
- Ranjan et al., GAVD, IEEE Access 2025, DOI 10.1109/ACCESS.2025.3545787.
- Kapoor and Narayanan, arXiv:2207.07048 (leakage taxonomy; source video as the independent unit).
- Varoquaux, NeuroImage 2018 (small-sample error bars).
- Patterson et al., Gait Posture 2010, PMID 19932621 (gait Symmetry Ratio biomarker).
- Hausdorff et al. 1998, PMID 9613733; Schaafsma et al. 2003, PMID 12809998 (stride-time variability; fallers 8.8 vs
  non-fallers 4.2 percent).
- Vandekerckhove et al. 2022, PMID 35721358 (DMD anterior pelvic tilt 16.4 vs 11.6 degrees); Vandekerckhove et al. 2025,
  PMID 41034979 (hip-extensor weakness drives anterior pelvic tilt).
- de Morais Filho et al. 2010, PMID 20300011 (crouch minimum stance knee flexion at least 30 degrees).
- Xiong et al. 2023, PMID 37525241 (Duchenne: no significant left-right asymmetry, myopathy negative class); Barohn et al.
  2014, PMID 25037080 (symmetric proximal weakness distribution).
- Natali and Javed, StatPearls, PMID 30571044 (corticospinal decussation); Volpe 2009, PMID 19081519 (periventricular
  injury); Riederer and Sian-Hulsmann 2012, PMID 22367437 (asymmetric nigrostriatal onset); Redgrave et al. 2010, PMID
  20944662 and Wu, Hallett, Chan 2015, PMID 26102020 (loss of automatic control).
- Stenum et al., PLoS Comput Biol 2021, PMID 33891585 (skeleton validity: timing MAE about 0.02 s/step, sagittal joints 4
  to 7 degrees).
- PhysioNet Gait-in-PD (gaitpdb), Hausdorff, DOI 10.13026/C24H3N.
