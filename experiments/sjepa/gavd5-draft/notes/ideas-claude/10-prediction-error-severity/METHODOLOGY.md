# Idea 10, in plain words: train a model on normal walking only, then read its surprise as a severity score

This is the "how to actually do it" guide for Idea 10, written so a motivated high-school student can follow every
step. The science comes from the folder's [`README.md`](./README.md). Every number here is kept true to
[`../_shared_facts.md`](../_shared_facts.md) (the single source of truth for numbers) and
[`../_neuro_facts.md`](../_neuro_facts.md) (the biology). If a number is not in those files, it is not in here, and any
made-up example is labelled "illustrative numbers only".

One honest note up front, repeated at the end because it matters: the folder labels (normal, parkinsons, stroke,
myopathic, cerebral_palsy) are just tags that came with the GAVD dataset. They are not diagnoses made by this project.
And every result here is transductive, which means the model was trained on the very source video we later test near.
Worse than usual, in fact: all our normal walking comes from a single YouTube video, so the model is tied to that one
source. More on what that costs us below.

## The big idea in plain words (60-second version)

Healthy walking is very predictable. The two legs take turns in a steady rhythm, the hips stay close to level, and one
stride looks a lot like the next. So here is the plan. Show a model a lot of clips of NORMAL walking and give it a
guessing game: cover up part of the body and ask it to guess the hidden part. After enough practice it gets good at that
guess, but only for normal walking, because that is all it ever saw.

Now freeze the model so it stops learning, and show it a NEW clip. If the new clip is also normal, its guess is close and
we say its "surprise" is small. If the new clip walks in some odd way, the model guesses wrong and its surprise goes up.
That growing surprise is the severity signal we want to read. This is the "world model" trick: learn what is expected,
then measure how much reality breaks that expectation. Video world models like V-JEPA 2 (Assran et al., 2025,
arXiv:2506.09985) and the physics work of Garrido et al. (2025, arXiv:2502.11831) do exactly this: a model trained on
ordinary scenes is more surprised by impossible ones.

The clever part is what the model NEVER sees. It never sees a stroke clip, a Parkinson's clip, a myopathy clip, or a
cerebral-palsy clip during training. It only sees normal. So its surprise cannot be a memorized "this folder is stroke"
shortcut, because there is no abnormal label anywhere in training. That side-steps a trap that haunts the rest of this
project: the condition label is almost the same thing as "which video did this come from," so an ordinary classifier can
cheat by recognizing the video instead of the walking. A normal-only world model has no abnormal videos to memorize, so
it cannot cheat that way.

But a single "total surprise" number is blunt. Two different problems could give the same total for different reasons. So
we do not stop at one number. We split the surprise into three MECHANISM CHANNELS, each tied to a body region and to a
real clinical measure:

- The ASYMMETRY channel watches the left-versus-right leg joints. One-sided problems (stroke, hemiplegic cerebral palsy,
  early Parkinson's) should light it up.
- The RHYTHM channel watches cycle-to-cycle timing. Parkinson's, which loses its automatic rhythm, should load here. This
  channel is the weakest and we treat it as exploratory only.
- The POSTURE channel watches the pelvis and trunk. Myopathy, a muscle disease that tips the pelvis forward on both sides
  while staying left-right symmetric, should load here.

The test is a DIRECTION test. We take a held-out normal clip and deliberately corrupt it in a known way. If we bend one
knee more than the other (a one-sided deficit), the asymmetry channel should rise and the posture channel should stay
quiet, and the rise should grow as we bend harder. If instead we tip the pelvis forward on both sides evenly, the posture
channel should rise and the asymmetry channel should stay quiet. If the channels do not respond to the matched corruption,
our decomposition is not measuring what we claim.

Here is a homey analogy. Imagine a music teacher who has only ever heard one song played perfectly, thousands of times.
Play it correctly and the teacher notices nothing. Play a wrong note in the left hand and the teacher flinches at the
left hand; play the whole piece too slowly and the teacher flinches at the tempo. The flinch is the surprise, and WHERE
the teacher flinches tells you what went wrong. Our three channels are just three places to watch for the flinch.

[`./images/fig3.svg`](./images/fig3.svg) is the gentlest picture of the whole plan. How to read fig3: follow the five
cards left to right, from "show only normal walking" through "turn video into a skeleton," "hide and guess," "freeze it
and show a brand-new clip," to "guessing error equals surprise." A small surprise means the new clip looks normal, a big
surprise means it looks different, and how much bigger becomes the label-free severity score.

## Mini-glossary (the words this idea actually uses)

- SKELETON: a moving stick figure. A pose detector finds body joints in each video frame, so a clip becomes a set of
  moving dots instead of pixels. Smaller than video, and it hides the face and clothing.
- TOKEN: the smallest chunk the model reads. Here, one joint watched over a short 4-frame window.
- EMBEDDING (or FEATURE): a short "fingerprint of numbers" the model uses to describe a token. Here each token becomes a
  list of 64 numbers.
- JEPA (Joint-Embedding Predictive Architecture): a model that learns by hiding part of its input and predicting the
  hidden part in fingerprint-space, not in raw coordinates (Assran et al., I-JEPA, CVPR 2023, arXiv:2301.08243; Bardes et
  al., V-JEPA, 2024, arXiv:2404.08471). The skeleton version is S-JEPA (Abdelfattah and Alahi, ECCV 2024,
  DOI 10.1007/978-3-031-73411-3_21).
- WORLD MODEL: a model that learns what is normal and expected, then reports how much a new input breaks that
  expectation. The break is called prediction error, or surprise.
- MASKING: hiding some tokens and asking the model to guess them. Like covering part of a photo with your hand and
  guessing what is behind it.
- VIEW ENCODER, TARGET ENCODER, PREDICTOR: the view encoder sees only the visible tokens and does the learning. The
  target encoder sees all the tokens and provides the answer key; it is not trained directly, its weights slowly trail
  the view encoder (an EMA, a slowly trailing running average). The predictor is the part that guesses the hidden target
  features.
- RELATIVE ERROR: raw surprise at a token divided by how surprising that same token usually is on normal clips. A ratio,
  so about 1.0 means "normal amount of surprise" and 2.0 means "twice normal."
- MECHANISM CHANNEL: a group of tokens (a body region) whose surprise we read together, tied to a clinical measure.
- INJECTION: a deliberate, controlled corruption we add to a normal clip so we know exactly what went wrong and can check
  whether the right channel notices.
- TRANSDUCTIVE: the model was trained on the very source (or clips) we test near. A high score can mean memorizing, not
  learning something that transfers to new people.
- FLOOR and CEILING: a floor is the easy bar an untrained model would also clear (the random-encoder control). A ceiling
  is a strong non-neural baseline the model must keep up with (the raw-coordinate detector).

## 1. The question in one sentence

When a JEPA world model is trained ONLY on normal gait and its relative masked-prediction error is read as a continuous
severity score, does a controlled one-sided knee-flexion injection raise error specifically in the asymmetry channel (and
track its magnitude), while a symmetric proximal-deficit injection raises error specifically in the posture channel, each
by a pre-registered margin over a random-encoder control?

There is one PRIMARY test (the two confirmatory injections: one-sided knee flexion for the asymmetry channel, symmetric
proximal deficit for the posture channel) and one EXPLORATORY extra (a timing-jitter injection for the rhythm channel).
The rhythm channel is exploratory on purpose, because cycle-to-cycle variability is not cleanly readable from roughly
2-second windows (see [`../_shared_facts.md`](../_shared_facts.md)). The primary verdict does not depend on it.

## 2. Why this idea, in plain words

The clinical literature sorts these gait conditions along a SYMMETRY axis (one-sided versus both-sides), and that split is
the heart of the motivation (see [`../_neuro_facts.md`](../_neuro_facts.md)).

- ONE-SIDED problems make walking lean to one side, so they load the ASYMMETRY channel. Stroke does this because the main
  motor pathway (the corticospinal tract) crosses to the opposite side of the body, so a stroke in one half of the brain
  weakens the other half (Natali and Javed, StatPearls, PMID 30571044). Hemiplegic cerebral palsy comes from a one-sided
  brain-tissue injury (Volpe 2009, PMID 19081519). Early Parkinson's often starts on one side (Riederer and
  Sian-Hulsmann 2012, PMID 22367437). The validated clinical measure here is the gait Symmetry Ratio (Patterson et al.
  2010, PMID 19932621).
- A BROKEN INTERNAL CLOCK loads the RHYTHM channel. Parkinson's loses its automatic rhythm because the basal ganglia lose
  the habit machinery (Redgrave et al. 2010, PMID 20944662; Wu et al. 2015, PMID 26102020), so timing gets more variable.
  The validated measure is stride-time variability: Schaafsma et al. 2003 (PMID 12809998) reported a stride-time
  coefficient of variation of 8.8 percent in fallers versus 4.2 percent in non-fallers. In plain words, the fallers'
  stride times bounced around about twice as much.
- BOTH-SIDES weakness loads the POSTURE channel. Myopathy is a muscle disease that weakens both sides fairly evenly
  (Barohn et al. 2014, PMID 25037080), so its walk stays roughly symmetric (Xiong et al. 2023, PMID 37525241) but the
  weak hip muscles tip the pelvis forward. In a Duchenne group the anterior pelvic tilt was 16.4 degrees versus 11.6
  degrees in typically developing controls (Vandekerckhove et al. 2022, PMID 35721358; mechanism in Vandekerckhove et al.
  2025, PMID 41034979). In plain words, the pelvis sat clearly more forward-tipped.

So the three channels are not a cosmetic grouping. Each one is anchored to a real, measured clinical difference. The
neuroscience defines the target (which channel should move for which mechanism) and the falsifiable direction (bigger
deficit, bigger surprise, in the matching channel). It does NOT license a clinical-accuracy claim, and we hold that line
hard: this is a within-cohort, synthetic-injection direction test, not a diagnosis.

## 3. What data you need

### 3.1 The main data: the gavd5-draft GAVD cohort (internal only)

The training data is the normal-only slice of the canonical GAVD cohort (Ranjan et al., IEEE Access 2025,
DOI 10.1109/ACCESS.2025.3545787). The full canonical cohort is 96 sequences from 18 unique YouTube source videos, but for
this idea we use only the normal rows: the 12 canonical normal sequences (all from the single source video `3KnFt8bH3tE`)
plus up to 63 accepted augmented-normal windows (one of 64 candidates was rejected at neurologic coverage 0.027). That is
the only normal available.

State the severe data bound up front and never hide it: normal comes from ONE source video. So the world model is
transductive to that single source and we cannot separate "the normal world model" from "that one video's provenance."
This is why the honest unit of evidence is the SOURCE VIDEO, not the clip (Kapoor and Narayanan, arXiv:2207.07048;
Varoquaux, NeuroImage 2018, on why tiny samples give big error bars). We hold out whole augmented-normal windows for the
injection tests and label every number transductive to source `3KnFt8bH3tE`.

There is also a PROVENANCE catch. Most normal rows were built through an "augmented" extraction path, and different
extraction paths can leave systematic differences. So we check whether the normal surprise baselines differ by extraction
path and, if they do, compute the baseline separately per path, so a pipeline difference is never mistaken for severity.

### 3.2 The reach data: PhysioNet gaitpdb (non-clinical to us, reach-tier, rhythm channel only)

There is no public skeleton dataset that pairs clinical labels with pose and keeps different people in train and test, so
there is no honest skeleton-level clinical transfer test here. The one external check we can honestly make is at the
LABEL level, for the RHYTHM channel only: PhysioNet Gait-in-PD (gaitpdb, 93 Parkinson's plus 73 controls, Hausdorff,
DOI 10.13026/C24H3N). gaitpdb is force and IMU data, not skeleton, so it can corroborate the stride-time-variability
biomarker at the label level but cannot confirm skeleton-level clinical transfer. It needs the external dataset and a
separate variability pipeline, so it is a REACH tier, not core. The asymmetry and posture channels have NO external
skeleton confirmation, because participant-disjoint public skeleton cohorts for cerebral palsy and myopathy do not exist.

The other verified public cohorts (CASIA-B, OU-MVLP-Pose, GREW, Gait3D, Human3.6M from
[`../_shared_facts.md`](../_shared_facts.md)) are non-clinical multi-view or mocap-validity sets; they do not carry the
clinical labels this idea's channels are about, so they are not part of this idea's plan beyond the general pose-validity
anchor (Human3.6M, Ionescu et al. 2014, DOI 10.1109/tpami.2013.248; Stenum et al. 2021, PMID 33891585, temporal mean
absolute error 0.02 s/step, sagittal joint errors 4 to 7 degrees).

### 3.3 What the data looks like, and how a team would get it

Each sequence is a skeleton: 33 body joints tracked over time, each joint carrying an x, a y, and a relative z coordinate
(MediaPipe BlazePose, Grishchenko et al. 2022, arXiv:2206.11678). A real team obtains it by running the pose detector on
walking videos, then cleaning the result: fill short gaps where the detector blinked, center the skeleton on the pelvis,
scale it to a standard body size, and stretch or squeeze the clip to exactly 64 frames so every sequence is the same
length. Then 4 frames in a row are grouped into one time patch, giving 16 time positions.

Reading the math (why 528 tokens): 64 frames split into groups of 4 gives 16 time positions, and 33 joints times 16 time
positions equals 528 tokens. The "x" means multiply: 33 x 16 = 528. A token is the smallest chunk the model reads.

Heels are the weak link for the detector (left heel visible about 70 percent of the time, right heel about 67 percent,
versus about 99 percent for shoulders and hips). That matters here because heels look surprising even when walking is
normal, so we must not let that baseline difficulty masquerade as severity. The relative-error step in the recipe fixes
exactly this.

## 4. Step by step, how to do it

Honest note first: unlike Idea 5, this idea DOES retrain. It trains a new world model from scratch on normal only. That
is the whole point, and it is the one expensive step. Everything after training is a frozen, test-time read. Retrain
count for the core: one (the normal-only world model). This idea does NOT reuse the project's `d0acc262` curriculum
checkpoint for scoring, because that checkpoint saw all conditions; reusing it is a different idea (item 02).

1. Assemble the normal-only training set (data prep, no training yet). Gather the 12 canonical normal sequences plus up to
   63 accepted augmented-normal windows from the cached skeletons. Then check, row by row, that NO abnormal clip has
   sneaked in. This check is not a formality: the entire argument of this idea is that the model never saw an abnormal
   label, so one leaked abnormal row would break it.

2. Wire the training loss to normal-only (setup). The project's usual loss has three parts:

   `L = L_JEPA + 0.05 * L_VICReg + 0.25 * L_group`

   Reading the math (the training loss): "loss" is the error training tries to make small. `L_JEPA` is the main
   prediction error (how badly the predictor guessed the hidden features), with weight 1. `L_VICReg` is an anti-collapse
   penalty that keeps the features spread out instead of collapsing to one vector (Bardes, Ponce, LeCun, ICLR 2022,
   arXiv:2105.04906), with weight 0.05. `L_group` is a LABEL-AWARE term with weight 0.25. Because we train on normal only,
   there are no condition labels to group, so we DROP `L_group` entirely (weight 0). We train with
   `L = L_JEPA + 0.05 * L_VICReg`. This is what makes our training purely self-supervised, unlike the project's Stages 1
   to 4, which are supervised because `L_group` is active there.

3. Train the world model at project scale (the one retrain). Reuse the existing 33-joint tokenization, 528 tokens,
   embedding dimension 64, depth 2, 4 heads, and the same masking pipeline (configured target 0.60 of the eligible
   tokens, global cap 12/33 = 0.364, laterality flip OFF). Train for roughly the Stage-0 budget of 300 epochs, not the
   full 600-epoch curriculum.

   Reading the math (the 0.364 mask cap): only 12 of the 33 joints (shoulders, hips, knees, ankles, heels, foot indices)
   can ever be hidden as prediction targets; face and arm joints are context but never targets. So the biggest fraction we
   can hide is 12 divided by 33 = 0.364, far below the 75 to 90 percent that image and video JEPAs hide.

4. Bind the new checkpoint to its own fingerprint (hygiene). Give the resulting model its own fingerprint and keep it
   strictly separate from the project's `d0acc262` curriculum checkpoint and the observed `dba24a` lineage. Mixing
   lineages would be a confound. This idea scores only with the new normal-only model.

5. Define relative masked-prediction error per token (frozen read). For a held-out clip, run the frozen view encoder on
   the visible tokens, run the predictor, and compare each guessed masked-position feature to the target encoder's feature
   at that position. That mismatch is the raw error. Then normalize:

   Reading the math (relative error): raw error is how far the guess sits from the answer (bigger is more surprising).
   Relative error divides that by the average raw error the SAME token shows on held-out normal clips. It is a ratio, so
   it is unitless: about 1.0 means "normal amount of surprise," 2.0 means "twice normal," below 1.0 means "less
   surprising than normal." Normalizing per token stops harder-to-predict joints (like the low-visibility heels) from
   looking severe just because they are always a bit surprising.

6. Freeze the three mechanism channels BEFORE any results (frozen read). Fix these token groups now, so you cannot tune
   them to make the answer look good.

   ```python
   ASYM    = [(25, 26), (27, 28), (23, 24)]   # left-right leg pairs: knees, ankles, hips
   POSTURE = [23, 24, 11, 12]                 # pelvis (hips) + trunk (shoulders)
   # RHYTHM = the temporal layout of surprise across the 16 time positions (exploratory)
   ```

   The ASYMMETRY channel is the SIGNED left-minus-right relative error on the paired leg joints, so a one-sided deficit
   makes one side more surprising than the other and the sign tells you which side. The POSTURE channel is the relative
   error on pelvis and trunk tokens, the skeleton stand-in for anterior pelvic tilt and trunk lean. The RHYTHM channel
   reads the timing structure of surprise across the 16 time positions; it is the weakest and pre-registered as
   exploratory only.

7. Build the injection generator (frozen read, applied to held-out normal clips only). Three corruptions, each with a
   magnitude dial:
   - One-sided knee flexion (confirmatory): bend one knee's trajectory by the magnitude, leave the other side alone. A
     lateralized, asymmetry-loading corruption.
   - Symmetric proximal deficit (confirmatory): tip the pelvis forward and reduce hip extension on BOTH sides equally by
     the magnitude. "Proximal" means near the body's center (the hips and pelvis), as opposed to "distal," which means far
     out toward the hands and feet. So this is a both-sides weakness at the hips. It is a posture-loading corruption that
     keeps left-right symmetry (Xiong 2023, PMID 37525241) and cadence (cadence means step rate, how many steps per minute,
     that is, the walking rhythm).
   - Timing jitter (exploratory): stretch and squeeze alternate sub-cycles so the average cadence (average step rate) is
     held but cycle-to-cycle regularity drops. The skeleton stand-in for elevated stride-time variability. Exploratory only,
     because variability is not cleanly readable from roughly 2-second windows.

8. Score the lanes and sweep magnitude (frozen read). For each injection type and each magnitude, run the corrupted clip
   through the trained normal-only world model (Lane A), the random-encoder floor (Lane B), the visibility-only nuisance
   (Lane C), the pooled-scalar control (Lane D), and the non-neural raw-coordinate ceiling (Lane E). Record each channel's
   response as a function of magnitude. The core scoring loop, in short readable pseudo-code:

   ```python
   # Frozen normal-only world model: view encoder, target encoder (EMA), predictor.
   # baseline_err[t] = mean raw error at token t over held-out NORMAL clips.

   def channel_errors(clip):
       visible, masked = mask_pipeline(clip)          # same 0.60 eligible target
       feats = view_encoder(visible)
       pred  = predictor(feats, masked_positions)     # only at masked positions
       tgt   = target_encoder(clip)                   # sees all 528 tokens
       raw   = feature_mismatch(pred, tgt[masked])    # per-token raw error
       rel   = raw / baseline_err[masked]             # relative error, ~1 is normal

       asym = sum(rel_at(rel, l) - rel_at(rel, r) for l, r in ASYM)  # SIGNED
       posture = sum(rel_at(rel, j) for j in POSTURE)
       rhythm  = temporal_variability(rel)            # exploratory channel
       return asym, rhythm, posture

   for m in magnitudes:
       a_knee = channel_errors(inject_one_sided_knee(normal_clip, m))
       a_prox = channel_errors(inject_symmetric_proximal(normal_clip, m))
       a_time = channel_errors(inject_timing_jitter(normal_clip, m))  # exploratory
       # expect a_knee.asym rises with m, a_knee.posture stays flat;
       # expect a_prox.posture rises with m, a_prox.asym stays flat;
       # exploratory: expect a_time.rhythm rises with m (weak, not gated).
   ```

9. Apply the pre-registered decision rule (Section 5), then package the checkpoint, the frozen decomposition function,
   the injection generator, the raw-coordinate ceiling detector, and the results, with a transductive caveat next to
   every number.

[`./images/fig1.svg`](./images/fig1.svg) is the direction-test picture. How to read fig1: two stacked plots share a
left-to-right axis of injection magnitude. In the top plot (one-sided knee flexion) the warm asymmetry curve climbs while
the blue posture curve stays flat; in the bottom plot (symmetric proximal deficit) the blue posture curve climbs while the
warm asymmetry curve stays flat. The grey dashed line is the random-encoder floor and should barely move in either plot.
The dark card on the right lists the pass margins.

[`./images/fig2.svg`](./images/fig2.svg) is the decomposition itself. How to read fig2: the left card shows a faded
lower-body stick figure with warm halos whose size marks how surprising each joint is, bracketed into the posture and
asymmetry channels (rhythm noted as temporal and exploratory). The right card is a bar chart of per-channel surprise, with
a low "beyond the injected coordinate" control bar to show the signal is not merely echoing the exact joint we corrupted.

## 5. The decision rule, decided in advance

We fix the bar BEFORE looking at results so we cannot move the goalposts. For the PRIMARY test (the two confirmatory
injections), a POSITIVE result ("the world model earns the mechanism-decomposition") needs ALL FOUR of these to hold at
once, at the largest injected magnitude on the matched channel:

1. Beat the FLOOR: the matched channel's relative error exceeds the random-encoder floor's matched-channel relative error
   by at least 0.05 (at least 5 percent more surprise than an untrained encoder produces on the same corrupted input).
2. Be SELECTIVE: the matched channel rises by at least 0.05 more than the UNMATCHED channel rises under the same
   injection.
3. Reach the CEILING: the matched channel reaches at least 80 percent (0.80) of the raw-coordinate ceiling's channel-
   detection effect on the SAME injection. This is the decisive bar: beating a random encoder is trivial, but keeping up
   with a handcrafted reader of the corrupted coordinates is not.
4. Be MONOTONE: the response is monotone non-decreasing across at least 3 of the injected magnitudes (bigger deficit,
   never less surprise), so the score tracks magnitude and is not a threshold accident.

Reading the math (the four numbers): 0.05 over the floor is the smallest excess surprise that counts as the trained model
doing something an untrained one does not. 0.05 selectivity is the smallest gap between the matched and unmatched channel
rises. 0.80 (80 percent) of the raw-coordinate ceiling is the share of the non-neural detector's effect the model must
reach to count as competitive with simply reading the coordinates. Monotone across at least 3 magnitudes means the score
climbs (or holds) as the deficit grows.

Miss ANY ONE of the four and the run is scored as an informative null, not a positive. In particular, if the model clears
the floor and is selective but falls below 80 percent of the ceiling, we report that the decomposition adds nothing over
reading the coordinates directly. There is also a SIGN check for the asymmetry channel: a left-knee injection and a
right-knee injection must move the signed asymmetry channel in OPPOSITE directions.

The RHYTHM channel (timing jitter) is EXPLORATORY only. We report it, we expect it to rise weakly, but it does NOT gate
the primary verdict, because variability is not cleanly readable from roughly 2-second windows.

### A worked example (illustrative numbers only, not measured facts)

Say we run the one-sided knee-flexion injection at its largest magnitude and read the ASYMMETRY channel (the matched
channel). Suppose relative error comes out as: Lane A (trained model) 1.60, Lane B (random-encoder floor) 1.10, the
posture channel under the same injection rises to 1.20 (an unmatched-channel rise of 0.20 above its own baseline of 1.00),
and Lane E (raw-coordinate ceiling) shows a matched-channel effect of 0.75 above 1.0. Walk the four checks:

- Beat the floor by at least 0.05: Lane A minus Lane B = 1.60 minus 1.10 = 0.50, far above 0.05. Pass.
- Selectivity at least 0.05: the asymmetry channel rose 0.60 above its baseline of 1.00 (1.60 minus 1.00), the posture
  channel rose 0.20, so the gap is 0.60 minus 0.20 = 0.40, above 0.05. Pass.
- Reach at least 80 percent of the ceiling: 80 percent of the ceiling effect 0.75 is 0.80 times 0.75 = 0.60. The model's
  matched rise above 1.0 is 0.60, which meets 0.60. Pass.
- Monotone across at least 3 magnitudes: suppose the asymmetry channel read 1.15, 1.35, 1.60 across three growing
  magnitudes, which never goes down. Pass.

All four pass, so this illustrative run would support "the world model earns the mechanism-decomposition." But flip one
number: if Lane A's matched rise had been only 0.45 above 1.0, then it is below the 0.60 the ceiling check needs, so the
whole run is scored an informative null (the model is only localizing the corruption, not adding a learned expectation).
That single number decides it, which is exactly why we fix the bar in advance.

## 6. Controls that keep us honest

- RANDOM-ENCODER FLOOR (Lane B). A world model with the identical architecture but randomly initialized (untrained)
  weights, scored the same way. Beating it is necessary but NOT sufficient, because a trivial reader of the injected
  coordinate would also beat it.
- RAW-COORDINATE CEILING (Lane E). A non-neural handcrafted detector read straight from the (possibly corrupted)
  coordinates with no encoder: a signed left-minus-right knee-and-hip angle for the asymmetry channel, and a hip-to-
  shoulder pelvic-tilt angle for the posture channel. This is the decisive bar (per [`../_neuro_facts.md`](../_neuro_facts.md)
  lever 4): the trained model must reach at least 80 percent of this detector's effect before the decomposition claim
  earns credit. This control exists because the injection changes the very coordinates the channel reads, so "error rises
  where we broke the input" is nearly tautological; only beating or matching the ceiling shows the model added a learned
  normal expectation.
- VISIBILITY-ONLY NUISANCE (Lane C). Confirm the channel selectivity is not an artifact of the injection changing which
  joints are visible. Re-run injections that preserve every visibility flag; Lane C must NOT reproduce the selectivity.
- POOLED-SCALAR CONTROL (Lane D). A single pooled surprise number, no channels. If it separates the two injection types as
  well as the three channels do, the decomposition adds nothing and we say so.
- SIGN CONTROL. Because the asymmetry channel is signed (left minus right), a left-knee injection and a right-knee
  injection must push it in OPPOSITE directions. A magnitude-only response would be side-agnostic and weaker than claimed.
- MONOTONICITY CONTROL. Verify the magnitude response climbs steadily rather than jumping at a single threshold.
- SOURCE-VIDEO-DISJOINT SPIRIT and the provenance check. Because normal is one source video, we cannot get true source-
  disjoint holdout for normal; we hold out whole augmented-normal windows and label everything transductive to
  `3KnFt8bH3tE`. We also report whether the normal surprise baselines differ by extraction path and, if so, compute the
  baseline per path so a pipeline difference is not read as severity.
- ONE FINGERPRINT. Bind every number to the new normal-only checkpoint's own fingerprint, kept separate from `d0acc262`
  and the observed `dba24a` lineage.

[`./images/fig4.svg`](./images/fig4.svg) draws all five lanes and the four possible outcomes side by side. How to read
fig4: the left column is the five scoring lanes (A trained model, B floor, C visibility-only, D pooled scalar, E raw-
coordinate ceiling), and the right column is a ladder of four outcomes with the one honest claim each allows. Green F1 is
the clean positive; amber F2 is the "only localizing" null; grey F3 is the "unstructured surprise" null; red F4 is the
"visibility artifact, claim withdrawn" outcome.

## 7. What could happen, and what each outcome would mean

Every outcome maps to one clear, pre-registered claim.

| Future | Shape | Verdict | What it licenses |
|---|---|---|---|
| F1 clean positive | Matched channel beats the floor, is selective, reaches at least 80 percent of the ceiling, is monotone, and Lane C does not copy it | Positive | The world model earns the mechanism-decomposition: a label-free, biomarker-aligned severity axis whose channel-specific rises track deficit direction and magnitude |
| F2 localizing null | Matched channel beats the floor and is selective but falls BELOW 80 percent of the raw-coordinate ceiling | Informative null | The model only LOCALIZES the corruption and adds nothing over reading the coordinates directly; the decomposition claim is not earned |
| F3 diffuse null | Matched channel fails to beat the floor, or shows no selectivity | Informative null | The normal-only world model has no spatially structured expectation of gait; the channel framing adds nothing over one pooled scalar |
| F4 artifact | Lane C (visibility only) reproduces the channel selectivity | Withdrawn | The effect is a missingness artifact, not learned motion; the claim is withdrawn |

There is also a training-health outcome worth naming: if the model COLLAPSES on such a small, single-source normal set
(the fingerprints of numbers lose their spread and start looking almost the same for every token, so the model is barely
describing anything), we report THAT as the finding, that a single-source normal set is insufficient to train a stable
world model. All four F-outcomes above assume training was healthy enough to score.

Given how small and single-source the normal set is, a null (F2 or F3) is a very plausible outcome ahead of time, and it
is publishable: reviewers at ICLR/ICML/NeurIPS 2026 value a well-motivated study that contributes new knowledge, including
a careful negative result (see [`../_shared_facts.md`](../_shared_facts.md), reviewer framing).

## 8. What this cannot tell us

- Transductive to ONE source. Normal comes from a single video (`3KnFt8bH3tE`), so the world model cannot be separated
  from that one source's provenance. No number here is a fresh-people performance estimate.
- Tiny sample. With one normal source video and up to 63 augmented windows, error bars are large and the injection test is
  a controlled internal falsifier, not external validation (Varoquaux, NeuroImage 2018).
- The near-tautology risk. The injection corrupts the very coordinates the channel reads, so "error rises where we broke
  the input" is close to automatic. The floor, selectivity, sign, monotonicity, and above all the raw-coordinate ceiling
  are what guard against this; without the ceiling check we would be over-claiming.
- Provenance confound. Normal mostly uses the augmented extraction path; the per-path baseline check softens this but
  cannot fully remove it at this sample size.
- Monocular capture. gavd5-draft is single-view, so no genuine multi-view or out-of-plane claim is possible.
- Skeleton limits. Skeletons cannot recover forces or propulsion (force plates), muscle-electrical activity or spasticity
  (EMG), out-of-plane rotation, or any etiologic muscle diagnosis (biopsy, blood markers, genetics), and none of those is
  claimed here. The gaitpdb reach check is a label-level cross-modal corroboration of the RHYTHM biomarker only.

## 9. How to make it reproducible

- One checkpoint, its own fingerprint. Train the normal-only world model once, bind it to a fresh fingerprint, and keep it
  strictly separate from `d0acc262` and `dba24a`. Every reported number cites that one fingerprint.
- Fix the seeds. Set and record the random seeds for the training run, the masking sampler, and the injection generator,
  so the same run reproduces.
- Save the split manifest. Record exactly which normal windows trained the model and which were held out for injection, so
  anyone can confirm no abnormal row entered training and no training window was scored as held-out.
- Freeze the artifacts. Save the frozen three-channel decomposition function, the per-token normal baselines
  (`baseline_err`), the injection generator with its magnitude grid, and the non-neural raw-coordinate ceiling detector.
- Save the per-magnitude, per-lane results. For each injection, each magnitude, and each of Lanes A through E, save the
  channel responses, then evaluate the four pre-registered checks from Section 5 against them, so the F1 to F4 verdict is
  recomputable from the saved numbers.
- Write "transductive to `3KnFt8bH3tE`" next to every number, so no seen-source score is mistaken for evidence about new
  people.

## Responsible use

The folder labels (normal, parkinsons, stroke, myopathic, cerebral_palsy) are dataset annotations from GAVD (Ranjan et
al., IEEE Access 2025, DOI 10.1109/ACCESS.2025.3545787), not diagnoses made by this project. The relative prediction error
and its three mechanism channels are representation diagnostics computed from cached skeleton coordinates and synthetic
injections; they are not validated clinical biomarkers and must not be read as a measurement of any individual's health or
severity. The neuroscience literature defines the target and the falsifiable direction of each channel; it does NOT turn
this n=18-source, single-source-normal, transductive study into a clinical-accuracy claim. The gaitpdb reach check is a
label-level cross-modal corroboration of the rhythm biomarker only. Skeletons cannot recover kinetics or propulsion, EMG
or spasticity, out-of-plane rotation, or an etiologic muscle diagnosis, and none of those is claimed here.

## References

- Assran et al., V-JEPA 2, 2025, arXiv:2506.09985.
- Garrido et al., intuitive physics from V-JEPA (violation of expectation), 2025, arXiv:2502.11831.
- Bardes et al., V-JEPA, 2024, arXiv:2404.08471.
- Bardes, Ponce, LeCun, VICReg, ICLR 2022, arXiv:2105.04906.
- Assran et al., I-JEPA, CVPR 2023, arXiv:2301.08243.
- Abdelfattah and Alahi, S-JEPA, ECCV 2024, DOI 10.1007/978-3-031-73411-3_21.
- Grishchenko et al., BlazePose GHUM, 2022, arXiv:2206.11678.
- Ranjan et al., GAVD, IEEE Access 2025, DOI 10.1109/ACCESS.2025.3545787.
- Kapoor and Narayanan, "Leakage and the Reproducibility Crisis in ML-based Science", 2022, arXiv:2207.07048.
- Varoquaux, "Cross-validation failure: small sample sizes lead to large error bars", NeuroImage 2018.
- Patterson et al., symmetry-index methods (Symmetry Ratio), Gait Posture 2010, PMID 19932621.
- Schaafsma et al., stride-time CV fallers 8.8 percent vs non-fallers 4.2 percent, J Neurol Sci 2003, PMID 12809998.
- Redgrave et al., loss of automaticity, Nat Rev Neurosci 2010, PMID 20944662.
- Wu, Hallett, Chan, loss of automaticity in PD, Neurobiol Dis 2015, PMID 26102020.
- Riederer and Sian-Hulsmann, asymmetric nigrostriatal onset in PD, J Neural Transm 2012, PMID 22367437.
- Natali/Javed StatPearls, corticospinal-tract pyramidal decussation, PMID 30571044.
- Volpe, periventricular leukomalacia and CP, Lancet Neurol 2009, PMID 19081519.
- Barohn et al., symmetric proximal weakness in myopathy, Neurol Clin 2014, PMID 25037080.
- Xiong et al., DMD shows no significant left-right asymmetry, Biomed Eng Online 2023, PMID 37525241.
- Vandekerckhove et al., anterior pelvic tilt 16.4 vs 11.6 deg in DMD, Front Hum Neurosci 2022, PMID 35721358.
- Vandekerckhove et al., hip-extensor weakness drives anterior pelvic tilt, J Neuroeng Rehabil 2025, PMID 41034979.
- Stenum et al., markerless gait validity (temporal MAE 0.02 s/step, sagittal joints 4 to 7 deg), PLoS Comput Biol 2021, PMID 33891585.
- Ionescu et al., Human3.6M, IEEE TPAMI 2014, DOI 10.1109/tpami.2013.248.
- Goldberger et al., PhysioNet Gait-in-PD (gaitpdb), DOI 10.13026/C24H3N.
