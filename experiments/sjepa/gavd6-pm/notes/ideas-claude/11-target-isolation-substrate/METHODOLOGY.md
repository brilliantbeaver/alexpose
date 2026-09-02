# Idea 11, in plain words: which prediction question teaches a gait model the most?

This is the "how to actually do it" guide for Idea 11, written so a motivated high-school student can follow every
step. The science comes from the folder's [`README.md`](./README.md). Every number here is kept true to
[`../_shared_facts.md`](../_shared_facts.md) (the single source of truth for numbers) and
[`../_neuro_facts.md`](../_neuro_facts.md) (the biology). If a number is not in those files, it is not in here, and any
made-up example is labelled "illustrative numbers only".

One honest note up front, repeated at the end because it matters: the folder labels (normal, parkinsons, stroke,
myopathic, cerebral_palsy) are just tags that came with the GAVD dataset. They are not diagnoses made by this project.
And every result here is transductive, which means each model was trained on the very clips we later test it on. More on
what that costs us below.

## The big idea in plain words (60-second version)

When you teach a model by hiding part of something and asking it to guess what is hidden, you get to choose WHAT you make
it guess. That choice is not small. It quietly decides what the model bothers to learn.

Think of covering part of a photo with your hand and asking a friend to describe what is behind your hand. You could ask
three different questions. "Guess the exact colors of every hidden pixel." Or "guess how those pixels are moving." Or
"just describe the hidden part in a few words, like a caption." Each question pushes your friend to pay attention to
different things. Same photo, same hand, but a very different lesson.

This project trains a model on walking. It hides some body joints and asks the model to predict them. The one thing we
change here is the question we ask: predict the exact joint positions, predict how the joints are moving, or predict a
short numeric "fingerprint" of the hidden joints. Everything else stays exactly the same. Then we check which question
taught the model the most useful things about walking. That is the whole study.

To keep the comparison fair, we do this like a race where every runner gets the identical shoes, track, and weather, and
the only thing that changes is the runner. Here everything that is not the prediction question is held identical across
the four trained models: same model shape, same amount of training, same mask, same data, same random seeds. Only the
question changes. That is the only way a difference in what the models learned can be blamed on the question and nothing
else.

## Mini-glossary (the words this idea actually uses)

- SKELETON: a moving stick figure. A pose detector finds body joints in each video frame, so a clip becomes a set of
  moving dots instead of pixels. Smaller than video, and it hides the face and clothing.
- TOKEN: the smallest chunk the model reads at once. Here, one joint watched over a short 4-frame window. There are 528
  of them (the math is below).
- EMBEDDING (also "latent" or "fingerprint" or "feature"): a short list of numbers the model uses to describe something.
  Instead of "hip is at these coordinates," the model can carry "hip summary = these 96 numbers." A latent is an internal
  fingerprint the model made up on its own.
- JEPA (Joint-Embedding Predictive Architecture): a model that hides part of the input and predicts the hidden part's
  fingerprint (its latent), not the raw pixels or coordinates. The skeleton version we build on is S-JEPA (Abdelfattah
  and Alahi, ECCV 2024, DOI 10.1007/978-3-031-73411-3_21).
- ENCODER: the part of the model that turns a skeleton into fingerprints (embeddings).
- PREDICTOR: the part that guesses the hidden joints' answer, whichever answer we asked for.
- EMA TEACHER: a slow-moving second copy of the encoder. "EMA" means it is a running average of the main encoder that
  changes gently over time, like a coach who updates their advice slowly. For the fingerprint questions, the teacher
  looks at the hidden joints and computes the fingerprint the predictor is trying to match.
- MASKING: covering some joints on purpose so the model must guess them. This is the "hand over the photo" step.
- PROBE: a small, simple readout (here a straight-line rule) placed on top of a frozen model to test what its features
  already contain. Like one quiz question that reveals whether you understood the lesson.
- RAW-INPUT CEILING: the score you can already get by reading the plain joint coordinates directly, with no neural
  network at all. A trained model only earns credit if it beats this.
- R-SQUARED: a number from 0 to 1 telling you how much of a quantity's ups and downs the probe explains. 0 means "no
  better than guessing the average"; 1 means "perfect." Higher is better.
- TRANSDUCTIVE vs INDUCTIVE: transductive means the model was trained on the very videos you later test it on (it may
  just be memorizing them). Inductive means you test on videos it has never seen. Only inductive tells you it truly
  generalizes.
- SOURCE-VIDEO-DISJOINT: when you split data for testing, you keep whole source videos on one side or the other, never
  mixing clips from the same video across the split.
- COLLAPSE: the cheating shortcut where a model makes every fingerprint identical so it always "matches." A special
  anti-collapse penalty forbids this.

## 1. The question in one sentence

Holding the encoder, the compute budget, the number of updates, and the mask fixed, and varying ONLY the prediction
target across four families (raw coordinates, one-frame motion, S-JEPA centered-sharpened latent, normalized latent
regression), does any target family recover the three validated gait mechanisms (left-right symmetry ratio, stride-time
CV, anterior pelvic tilt) above a raw-input probe ceiling on source-video-disjoint folds by a pre-registered margin?

In plainer words: change only the question we make the model answer, keep everything else identical, and see which
question, if any, teaches the model to carry the three clinically meaningful walking signals better than plain
coordinates already do.

## 2. Why this idea, in plain words

The whole JEPA family of models is built on one bold claim: WHAT you predict matters more than how hard you predict it.
V-JEPA predicts a latent (a fingerprint) of hidden video instead of the raw pixels, and argues this is why it learns
useful structure (Bardes et al. 2024, arXiv:2404.08471); V-JEPA 2 extends the idea to predicting the effect of actions
(Assran et al. 2025, arXiv:2506.09985); S-JEPA brings the same fingerprint-prediction recipe to skeletons (Abdelfattah
and Alahi, ECCV 2024, DOI 10.1007/978-3-031-73411-3_21). But there is a competing school for skeleton motion that says
you should predict MOTION (how far each joint moved since the last frame), not position, because motion is the more
useful lesson for bodies. These are opposite bets, and nobody has tested them cleanly for gait, where "useful" has a
concrete, clinically validated meaning. This project supplies that meaning.

"Useful" here is not the training loss. A model can drive its own training loss to zero and still have learned nothing a
clinician would care about, the same way you can ace a spelling test by memorizing the answer sheet without learning to
spell. So we score each trained model by whether its features let a simple straight-line readout recover three gait
quantities that the clinical literature has already validated. Here is the plain chain from a brain or muscle problem to
something you can actually see in the stick figure, drawn from [`../_neuro_facts.md`](../_neuro_facts.md).

- LEFT-RIGHT SYMMETRY RATIO (how lopsided the two legs are). Some conditions have a one-sided cause and make walking lean
  to one side. Stroke does this because the main motor nerve pathway (the corticospinal tract) crosses over to the
  opposite side of the body, so a stroke in one half of the brain weakens the other half of the body (Natali and Javed,
  StatPearls, PMID 30571044). Hemiplegic cerebral palsy comes from a one-sided brain-tissue injury (Volpe 2009,
  PMID 19081519). Early Parkinson's often starts on one side (Riederer and Sian-Hulsmann 2012, PMID 22367437). The
  validated number is the gait Symmetry Ratio (Patterson et al. 2010, PMID 19932621).
- STRIDE-TIME CV (how much the stride timing wobbles cycle to cycle). Parkinson's slowly loses the brain's dopamine-making
  cells, which weakens the automatic "walking on autopilot" system, so the timing gets jittery (Redgrave et al. 2010,
  PMID 20944662; Wu, Hallett, Chan 2015, PMID 26102020). The concrete anchor is 8.8 percent for Parkinsonian fallers
  versus 4.2 percent for non-fallers, so jittery timing is roughly twice as wobbly (Schaafsma et al. 2003, PMID 12809998;
  foundational biomarker, Hausdorff et al. 1998, PMID 9613733).
- ANTERIOR PELVIC TILT (how far the pelvis tips forward). Myopathy is a muscle disease that weakens both sides fairly
  evenly (Barohn et al. 2014, PMID 25037080), so the walk stays roughly symmetric but the posture changes. The concrete
  anchor is 16.4 degrees for Duchenne muscular dystrophy versus 11.6 degrees for typically developing children, so the
  affected pelvis tips noticeably farther forward (Vandekerckhove et al. 2022, PMID 35721358).

These three are not equally open, and that is important for reading the outcome honestly. The project's earlier work
(notebook 05, summarized in [`../_shared_facts.md`](../_shared_facts.md)) already measured, on the frozen `ea59fea0`
encoder, that pooled features decode step amplitude well (R-squared about 0.719, so the probe explains most of the ups
and downs of step size), but LEFT-RIGHT ASYMMETRY is the weakest-decoded scalar (R-squared about 0.154, so the probe
explains only a small slice of it), and STRIDE-TIME CV is not linearly decodable at all from the roughly two-second
windows. So the symmetry-ratio endpoint starts from a known-weak baseline, and the stride-time-CV endpoint starts from a
known negative. The sharp question is whether ANY of the four questions lifts asymmetry or CV above the raw-input ceiling
where the old pooled readout could not.

A positive result names a winner: one target family recovers the mechanisms above the raw ceiling by a margin, and it
does so at fixed encoder, compute, steps, and mask, so the win is attributable to the target alone. That is a transferable
rule ("for gait, predict X") that someone else could reuse on their own gait model. A null is equally useful: if no
question clears the ceiling, that says the choice of question is not the bottleneck at this scale, and the real limit is
that we only have 18 source videos. Given the known negatives above, a CV or asymmetry null here would confirm the
project's prior finding rather than contradict it, which is exactly the kind of informative negative result ICLR/ICML/
NeurIPS 2026 reviewers value (see [`../_shared_facts.md`](../_shared_facts.md), reviewer framing).

## 3. What data you need

### 3.1 The main data: the gavd5 GAVD cohort (internal only)

The main work runs on the canonical GAVD cohort (Ranjan et al., IEEE Access 2025, DOI 10.1109/ACCESS.2025.3545787):
96 sequences from 18 unique YouTube source videos. The per-condition source-video counts are tiny and lopsided: normal 1,
Parkinson's 2, stroke 3, myopathic 10, cerebral palsy 2. All 12 normal sequences come from a single video (`3KnFt8bH3tE`).

Two consequences fall out of those small numbers, and they shape the whole design.

1. The condition label is almost the same thing as "which video did this come from," especially for normal. So the honest
   unit of evidence is the SOURCE VIDEO, not the individual clip. Two clips from one video are not two independent facts,
   any more than two frames of one movie are two different movies (Kapoor and Narayanan, arXiv:2207.07048; and Varoquaux,
   NeuroImage 2018, on why tiny samples give big error bars).
2. Because some conditions have as few as one source video, you cannot report a per-condition "hold out one source"
   R-squared. One held-out point is not a distribution. So we pool the mechanism probes across all conditions and plot
   every source video as its own dot.

There is also a PROVENANCE catch. Most normal rows were built through an "augmented" extraction path, while every abnormal
row used the "canonical" path (see [`../_shared_facts.md`](../_shared_facts.md)). If we are not careful, a recovered
mechanism could accidentally be an artifact of how the clips were processed instead of how the people walked. To avoid
that, the main comparison runs only on the provenance-matched (canonical-path) subset.

### 3.2 The reach data: public multi-view pose cohorts (non-clinical, reach-tier)

There is no public skeleton dataset that validates the target-design choice itself, and none that pairs left-right or
posture clinical labels (stroke, cerebral palsy, myopathy) with pose while keeping different people in train and test. So
there is no honest skeleton-level clinical transfer test for this idea. What DOES exist is limited and marked reach-tier:

- For the stride-time-CV biomarker only, a LABEL-LEVEL, cross-modal anchor in PhysioNet Gait-in-PD (gaitpdb, 93
  Parkinson's plus 73 controls, force and IMU sensors, DOI 10.13026/C24H3N). This is not a skeleton test; it only confirms
  that stride-time CV is a real Parkinson's signal in another modality.
- Pose validity, meaning "can a single-camera stick figure recover these quantities at all," rests on Human3.6M
  (Ionescu et al. 2014, DOI 10.1109/tpami.2013.248) and Stenum et al. 2021 (PMID 33891585, temporal error about 0.02
  seconds per step, side-view hip/knee/ankle angle errors about 4.0 / 5.6 / 7.4 degrees).
- The non-clinical multi-view pose cohorts named in [`../_shared_facts.md`](../_shared_facts.md), CASIA-B, OU-MVLP-Pose,
  GREW, and Gait3D, exist but are non-clinical and cannot make any clinical claim.

CP and myopathy skeleton cohorts with participant-disjoint splits do not exist publicly. That is an honest limitation, not
an oversight, and any clinical-accuracy statement stays external-cohort reach-tier and out of scope for this
n=18-source transductive study.

### 3.3 What the data looks like, and how a team would get it

Each sequence is a skeleton: 33 body joints tracked over time, each joint carrying an x, a y, and a relative z coordinate
(MediaPipe BlazePose, Grishchenko et al. 2022, arXiv:2206.11678). A real team obtains it by running the pose detector on
walking videos, then cleaning the result: fill short gaps where the detector blinked, center the skeleton on the pelvis,
scale it to a standard body size, and stretch or squeeze the clip to exactly 64 frames so every sequence is the same
length. Heels are the weak link for the detector (left heel visible about 70 percent of the time, right heel about
67 percent, versus about 99 percent for shoulders and hips), which is one more reason to lean on the pelvis, hips, knees,
and ankles.

## 4. Step by step, how to do it

Honest warning first: unlike some sibling ideas, this one is NOT zero-retrain. The heart of it is training four models
from scratch, one per question. The good news is that everything AFTER training reuses the project's existing plumbing:
the cached 528-token feature format and the same probe machinery. Retrain count for the core: four (one per target
family). Fig 3 shows, before any math, the three kinds of question we are choosing between:
[`./images/fig3.svg`](./images/fig3.svg).

1. Build the fixed substrate harness (setup, no retrain yet). Write ONE training config where the only field allowed to
   change between runs is `prediction_target`. Everything else is pinned to the project's own settings: 96 canonical
   sequences from 18 source videos, each sequence resized to 64 frames, 4 adjacent frames per time patch giving 16 time
   positions, 33 joints times 16 positions = 528 tokens, embed dim 96, encoder depth 4, 4 heads, GELU, pre-norm; only 12
   lower-body-and-shoulder landmarks are ever maskable (global mask cap 12/33 = 0.364, meaning at most about 36 percent of
   joints can ever be hidden), configured target 0.60 of eligible tokens; five-stage curriculum totalling 600 epochs and
   11,400 optimizer updates matching the `ea59fea0` curriculum; laterality flip OFF (flip probability 0.0); fixed seeds.

   Reading the math (why 528): 64 frames split into groups of 4 gives 16 time positions, and 33 joints times 16 positions
   equals 528 tokens. The "x" means multiply: 33 x 16 = 528.

2. Define the four questions (the ONLY thing that changes). Retrain the encoder once per question.
   - T1 RAW COORDINATES: predict the hidden joints' exact (x, y, relative z). The plainest bet: "tell me exactly where the
     hidden joints are."
   - T2 ONE-FRAME MOTION: predict the one-frame displacement (how far each hidden joint moved since the last frame) instead
     of its position. The "predict motion, not location" bet.
   - T3 CENTERED-SHARPENED LATENT: the project's current S-JEPA objective. Predict the EMA teacher's hidden fingerprint
     under a centered (running EMA center, beta 0.9) then sharpened (temperature 0.06, stop-gradient) latent cross-entropy,
     prediction at temperature 0.10. Two plain-word notes on the jargon. "Cross-entropy" is just a score of how well one
     probability guess matches another: small when the student's guessed fingerprint lands on the same choices the teacher
     made, large when it does not. "Stop-gradient" means the teacher's fingerprint is treated as a fixed answer key: the
     student is graded against it, but the teacher is not nudged to make the student's job easier (which would let both
     copies agree on a lazy, useless answer). So in plain words, predict a cleaned-up, confident version of the teacher's
     fingerprint of the hidden joints.
   - T4 NORMALIZED LATENT REGRESSION: predict the EMA teacher's hidden fingerprint by gently regressing onto normalized
     (unit-variance) latents instead of a sharpened cross-entropy. The smoother "V-JEPA-style feature regression" bet.

3. Keep the anti-collapse structure identical (part of every retrain). All four keep the project loss. The reference form
   is `L = L_target + 0.05 * L_VICReg + 0.25 * L_group`. A "loss" is just a score of how wrong the model is; training
   makes it small. Only `L_target` changes across T1 to T4.
   - `L_target` is the prediction error against whichever question is active; its weight is 1, so it dominates.
   - `L_VICReg` is the anti-collapse penalty, weight 0.05, which stops the cheating shortcut of making every fingerprint
     identical (Bardes et al. 2022, arXiv:2105.04906). It matters most for T2, T3, T4, which predict latents and can
     collapse. Its weight is held fixed so no question gets a collapse advantage.
   - `L_group` is a label-aware condition-centroid term, weight 0.25, active only in Stages 1 to 4, which makes those
     stages supervised fine-tuning (the model is nudged by the condition labels, not purely self-taught).
   - Because `L_VICReg` and `L_group` are identical across all four retrains, any difference in mechanism recovery is
     attributable to `L_target` alone.

4. Freeze the three mechanism targets BEFORE any probing (deterministic, from raw coordinates). From the RAW cached
   coordinates (not from any model), compute the ground-truth value of each mechanism once, using the validated
   definitions: the signed left-right Symmetry Ratio (Patterson 2010, PMID 19932621), the stride-time CV (Schaafsma 2003,
   PMID 12809998), and the anterior pelvic tilt as a pelvis-segment side-view angle (Vandekerckhove 2022, PMID 35721358).
   Freeze these functions before computing any result so you cannot tune them to make the answer look good. Note one care
   point: stride-time CV is a dimensionless ratio (a standard deviation divided by a mean), so uniformly stretching time
   should leave it unchanged in principle. Its validity on the 64-frame-resized sequences holds only if the resize is a
   uniform rescale that preserves this ratio, so any credit on the CV probe is reported only under that condition, and the
   resize is not allowed to silently confound it.

5. Cache the frozen features for all four encoders (reuses existing plumbing). For each trained encoder, run it over all
   528 tokens of every canonical sequence and cache the per-token embedding, in the exact same cached-tensor format the
   project already uses. Tag each cached row with its source-video id, condition label, provenance (canonical vs
   augmented), the encoder's fingerprint, and an encoder-exposure flag. Also cache the features for an untrained-encoder
   floor (a random-init encoder of the identical shape) and build the handcrafted raw-coordinate features for the ceiling.

6. Fit the shared mechanism probes (no retrain; reuses the probe machinery). For each of the three mechanisms and each of
   the four encoders, fit ONE shared ridge linear probe. A ridge linear probe is just a straight-line rule from features
   to a mechanism value, with a gentle brake on its weights so it does not overreact to any single feature. Pick the brake
   strength using only the training sources of each fold, so held-out sources never leak in. Fit and score the probes on
   SOURCE-VIDEO-DISJOINT folds (whole videos held out), pooled across conditions, one dot per source. Fit the same probe
   on the raw-coordinate ceiling features and on the untrained-encoder floor features for reference.

   Reading the math (R-squared): R-squared runs 0 to 1 and is the fraction of a mechanism's variation the probe explains.
   0 is "no better than always guessing the average," 1 is "perfect." Higher is better.

7. Monitor collapse per family (a health check, no retrain). For each of T1 to T4, report the final feature standard
   deviation and mean pairwise cosine. The reference healthy `ea59fea0` checkpoint sat at feature std 0.362567 and mean
   pairwise cosine 0.659870, so a family whose std drifts far below that has quietly collapsed and its numbers are
   suspect.

8. Apply the pre-registered decision rule (Section 5) and write out the per-mechanism-by-target recovery table with
   per-source dots. Fig 1 is the decisive picture ([`./images/fig1.svg`](./images/fig1.svg)) and Fig 2 is the
   fairness audit ([`./images/fig2.svg`](./images/fig2.svg)).

## 5. The decision rule, decided in advance

We fix the bar BEFORE looking at results so nobody can move the goalposts after seeing the numbers. For each mechanism,
the primary number is the held-out-source R-squared of the shared ridge probe, per target family, pooled across
conditions.

A target family is CREDITED on a mechanism only if BOTH of these hold at once:

1. Its held-out-source R-squared exceeds the raw-input ceiling by at least 0.05 R-squared on that mechanism.
2. Its held-out-source R-squared exceeds the untrained-encoder floor by at least 0.05 R-squared.

Reading the math (the two margin numbers):
- Exceed the raw-input ceiling by at least 0.05 R-squared: the learned features must add at least 0.05 of explained
  variation beyond what handcrafted raw coordinates already give. Below that, the representation added nothing over raw
  input on that mechanism.
- Exceed the untrained-encoder floor by at least 0.05 R-squared: this rules out a random, untrained network scoring well
  by luck.
- Both thresholds are on the 0-to-1 R-squared scale, so 0.05 is a five-percentage-point gap.

The study's headline verdict names the target family credited on the MOST mechanisms. A tie, or a zero-credit outcome, is
scored as an informative null: at this scale and cohort, the choice of prediction target is not the bottleneck.

### A worked example (illustrative numbers only, not measured facts)

Suppose on the anterior-pelvic-tilt mechanism we measure (these match the shape drawn in Fig 1):
- Raw-input ceiling: R-squared 0.62.
- Untrained-encoder floor: R-squared 0.22.
- T3 (centered-sharpened latent): R-squared 0.69.
- T1 (raw coordinates): R-squared 0.58.

Step 1: check T3 against the ceiling. 0.69 minus 0.62 = 0.07, which is at least 0.05. Pass. Step 2: check T3 against the
floor. 0.69 minus 0.22 = 0.47, which is at least 0.05. Pass. So T3 is credited on anterior pelvic tilt.

Now T1: 0.58 minus 0.62 = negative 0.04, so T1 does not even reach the ceiling. Fail. T1 is NOT credited, because its
features add nothing beyond raw coordinates on this mechanism.

If T3 turns out to be the only family credited across all three mechanisms, the headline verdict is "for gait, predict a
centered-sharpened latent." Again, these four numbers are made up to show how the rule works; they are not measured
results.

## 6. Controls that keep us honest

- MATCHED SUBSTRATE (the primary control). The encoder shape, compute, number of updates, mask sampler, mask fraction,
  VICReg weight, group-loss weight, EMA schedule, data, and seeds are identical across T1 to T4. A config diff must show
  `prediction_target` as the sole difference before any results are read. This is what Fig 2 audits.
- RAW-INPUT CEILING. The same probe fit on handcrafted raw-coordinate features, no network at all. It guards against
  crediting a target for information already sitting in the plain coordinates. This is Idea 11's version of the
  missingness-only spirit: a learned representation earns credit only for what it adds beyond a simple non-neural
  baseline.
- UNTRAINED-ENCODER FLOOR. A random-init encoder of the identical shape sets the "pure chance" level, so we never credit a
  target for a lucky random network.
- MEAN/STD-POOLED NUISANCE CONTROL. A mean and a standard deviation over tokens are permutation-invariant: they give the
  same answer no matter how you shuffle the tokens, so they throw away time order. Stride-time CV is defined by the ORDER
  of stride events in time, so this pooled control MUST NOT recover stride-time CV above the ceiling. If it somehow does,
  the CV signal is leaking from something other than timing (for example how the clip was processed), and that mechanism's
  recovery is rejected as an artifact.
- SOURCE-VIDEO-DISJOINT SPLITS. Hold out whole source videos, never single clips, so a held-out score is not just two
  clips of the same video quizzing each other.
- PROVENANCE-MATCHED SUBSET. Run the main comparison only on the canonical-path rows, so a recovered mechanism cannot be
  an augmented-versus-canonical acquisition artifact.
- ONE-FINGERPRINT BINDING. Every retrain's budget matches the `ea59fea0` curriculum and is bound to one lineage before
  comparison; the `dba24a` canonical lineage is never mixed in.
- COLLAPSE MONITORING PER FAMILY. Report feature std and mean pairwise cosine for each of T1 to T4 (reference healthy std
  0.362567, mean cosine 0.659870), since the latent-target families can collapse.
- POOLED PROBING WITH PER-SOURCE DOTS. No per-class hold-one-source R-squared on n=1 sources; pool across conditions and
  use a source-level permutation null only where the number of held-out sources makes it meaningful.
- TRANSDUCTIVE CAVEAT next to every number, so no seen-video score is mistaken for evidence about new people.

## 7. What could happen, and what each outcome would mean

The decision rule is total: every outcome maps to one clear, pre-registered claim. Fig 1 shows the expected shape of the
result.

| Future | Shape | Verdict | What it licenses |
|---|---|---|---|
| A single clear winner | One target family clears both the ceiling and the floor by 0.05 on the most mechanisms | Positive, names the winner | The transferable rule "for gait, predict X" (for example, predict a centered-sharpened latent), attributable to the target alone because the substrate was matched |
| A split decision | Different families win on different mechanisms | Positive but nuanced | The best question depends on the mechanism (for example, latent for posture, motion for something else); still a transferable design note |
| No family clears the ceiling | Every bar sits at or below the raw-input ceiling on every mechanism | Informative null | The choice of prediction target is NOT the bottleneck at this scale; the limit is data breadth (18 source videos), not target engineering. On CV or asymmetry this would confirm the project's known negatives |
| A family collapsed | A latent-target family's feature std drifted far below 0.362567 | Result withdrawn for that family | Its recovery numbers are not trustworthy; the anti-collapse control caught a cheating shortcut |
| Nuisance control fires | The mean/std-pooled control "recovers" stride-time CV above the ceiling | CV recovery rejected as artifact | The CV signal was leaking from token order-free information (for example processing), not real timing |

How to read Fig 1 ([`./images/fig1.svg`](./images/fig1.svg)): it is a grouped bar chart with one column group per
mechanism (symmetry ratio, stride-time CV, anterior pelvic tilt) and four bars per group (T1 to T4). Each group draws the
raw-input ceiling as a green dashed line and the untrained-encoder floor as a lower grey dashed line, with the
pre-registered 0.05 margin shaded as a band above the ceiling. Taller bars are better. A bar only "wins" if its top clears
the green line AND lands inside or above the shaded band. If every bar sits below the green line for a mechanism, that
mechanism was a null. The illustrative shape has the latent-target families lifting anterior pelvic tilt above the ceiling
while stride-time CV stays below the ceiling for every family, an informative null. All values are illustrative and
transductive.

How to read Fig 2 ([`./images/fig2.svg`](./images/fig2.svg)): it is a four-row audit grid, one row per target arm, with
columns for Encoder, Compute budget, Updates, and Mask. Read down each column. If a whole column shows the same value in
all four rows, that ingredient was held fixed and is fair. The only thing allowed to differ between rows is the prediction
target, so any recovery gap in Fig 1 is attributable to the target alone.

Given the project's prior that asymmetry is the weakest-decoded scalar (R-squared about 0.154) and stride-time CV was not
linearly decodable at all, the informative-null and split-decision futures are the more likely ones ahead of time, and
both are worth publishing under the reviewer framing that rewards careful negative results.

## 8. What this cannot tell us

- TRANSDUCTIVE, so no out-of-sample claim. Each encoder saw every evaluation row during training, so no number here is a
  fresh-people performance estimate. A truly held-out estimate would need retraining the whole curriculum inside each
  outer source split, which even this four-retrain study does not do.
- TINY, UNEQUAL SOURCE COUNTS. With as few as one source per class, the pooled endpoint and the source-as-a-dot plots are
  the only defensible readouts. Any per-class margin would be one point dressed up as a distribution.
- PROVENANCE AND LABEL OVERLAP. Normal is one video on a mostly-augmented path; the canonical-path subset softens this but
  cannot fully remove it at this sample size.
- MONOCULAR CAPTURE. gavd5 is single-view, so out-of-plane (twisting) motion is not recoverable, and there is no
  participant-disjoint skeleton cohort to confirm the target-design choice on new people.
- SKELETON LIMITS. Skeletons cannot recover forces or push-off power, muscle-electrical activity or spasticity,
  twisting rotation seen from above, or a specific muscle-disease diagnosis. No claim here touches those, and the three
  mechanism scalars are representation diagnostics, not clinical measurements of any individual.

## 9. How to make it reproducible

- ONE config drives all four runs. Store a single substrate config where only `prediction_target` differs across T1 to T4,
  and save the config diff that proves it. Anyone can rerun the four training jobs from that one file.
- ONE checkpoint lineage. Bind every retrain's budget to the `ea59fea0` curriculum (5 stages, 600 epochs, 11,400 updates)
  and never mix in the `dba24a` lineage. Record each encoder's fingerprint next to its results.
- FROZEN, DETERMINISTIC targets. Compute the three mechanism ground-truths once from the raw coordinates using the frozen
  Patterson 2010, Schaafsma 2003, and Vandekerckhove 2022 definitions, and save those functions. Run a small-noise
  reliability check on them before probing.
- FIX THE SEEDS. Use the same seeds across all four runs so the only difference is the target. Save the seeds.
- SAVE THE SPLIT MANIFEST. Write out the source-video-disjoint fold manifest (which video is in which fold) so the exact
  splits are reproducible, and confirm no held-out source's clips leaked into any probe-fitting fold.
- SAVE THE RESULTS. Cache the frozen 528-token features for all four encoders plus the untrained floor, and write the
  per-mechanism-by-target recovery table with per-source dots, the raw-input ceiling, the untrained floor, the collapse
  health numbers per family, and a transductive caveat next to every number. A future run can then diff its table against
  yours cell by cell.

## Responsible use

The folder labels (normal, parkinsons, stroke, myopathic, cerebral_palsy) are dataset annotations from GAVD (Ranjan et al.,
IEEE Access 2025, DOI 10.1109/ACCESS.2025.3545787), not diagnoses made by this project. The three mechanism scalars
(symmetry ratio, stride-time CV, anterior pelvic tilt) are representation diagnostics computed from cached skeleton
coordinates. They are not validated clinical measurements of any individual's health. Every result here is transductive
and small-sample, with the source video as the independent unit before any fitting.

## References

- Abdelfattah and Alahi, S-JEPA, ECCV 2024, DOI 10.1007/978-3-031-73411-3_21.
- Assran et al., I-JEPA, CVPR 2023, arXiv:2301.08243.
- Bardes et al., V-JEPA "Revisiting Feature Prediction for Learning Visual Representations from Video", 2024,
  arXiv:2404.08471.
- Assran et al., V-JEPA 2, 2025, arXiv:2506.09985.
- Bardes, Ponce, LeCun, VICReg, ICLR 2022, arXiv:2105.04906.
- Grishchenko et al., BlazePose GHUM, 2022, arXiv:2206.11678.
- Ranjan et al., GAVD, IEEE Access 2025, DOI 10.1109/ACCESS.2025.3545787.
- Kapoor and Narayanan, arXiv:2207.07048 (leakage taxonomy; source video as the independent unit).
- Varoquaux, NeuroImage 2018 (small-sample error bars).
- Patterson et al., Gait and Posture 2010, PMID 19932621 (gait Symmetry Ratio biomarker).
- Natali and Javed, StatPearls, corticospinal tract anatomy, PMID 30571044.
- Riederer and Sian-Hulsmann, J Neural Transm 2012, PMID 22367437 (asymmetric nigrostriatal onset).
- Volpe, Lancet Neurol 2009, PMID 19081519 (periventricular injury and corticospinal fibers).
- Redgrave et al., Nat Rev Neurosci 2010, PMID 20944662; Wu, Hallett, Chan, Neurobiol Dis 2015, PMID 26102020 (loss of
  automaticity).
- Schaafsma et al., J Neurol Sci 2003, PMID 12809998 (stride-time CV 8.8 vs 4.2 percent); Hausdorff et al., Mov Disord
  1998, PMID 9613733 (variability biomarker).
- Barohn et al., Neurol Clin 2014, PMID 25037080 (symmetric proximal weakness distribution).
- Vandekerckhove et al., Front Hum Neurosci 2022, PMID 35721358 (anterior pelvic tilt 16.4 vs 11.6 degrees).
- Stenum et al., PLoS Comput Biol 2021, PMID 33891585 (markerless pose gait validity).
- Ionescu et al., Human3.6M, IEEE TPAMI 2014, DOI 10.1109/tpami.2013.248 (pose validity against motion capture).
- Goldberger et al., PhysioNet Gait-in-PD (gaitpdb), DOI 10.13026/C24H3N (cross-modal stride-time-CV anchor, reach-tier).
