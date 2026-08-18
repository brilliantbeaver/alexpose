# Idea 9: Reflection-Equivariant Symmetry

This is the hands-on guide for the reflection-equivariant symmetry-axis idea. The big-picture
story lives in [README.md](./README.md). Every number in this guide comes from
[../_shared_facts.md](../_shared_facts.md) (the single source of truth for numbers) and
[../_neuro_facts.md](../_neuro_facts.md) (the biology behind the idea). If a number here ever seems
to disagree with those two files, trust those files, not this one. Folder labels like "stroke" or
"parkinsons" are dataset annotations from GAVD, not diagnoses made by this project, and every result
we get is transductive (explained below), so nothing here is a claim about diagnosing a real person.

## The big idea in plain words

Walking is almost a mirror image of itself. Your left leg does about the same thing your right leg
does, just half a step later. Some walking problems break that mirror on one side only, like a
stroke, and some do not, like a muscle disease that weakens both sides equally.

We want a computer model whose "left minus right" reading is an honest mirror. Honest means: if you
flip the walking person left-to-right (like looking at them in a mirror), the reading must flip its
sign too, from plus to minus. A standard model can only "hope" to behave this way if it saw enough
examples. This idea instead wires the model so it can do nothing else. Then we test, on a fixed
ruler, whether building the rule in beats hoping the model learns it.

## 1. The question in one sentence

On folds where whole videos are held out, does building the signed left-minus-right axis to be
antisymmetric by construction (a model whose reading a mirror is guaranteed to flip) separate
one-sided gait (stroke, hemiplegic cerebral palsy, early Parkinson's) from both-sides gait
(myopathy) better than the standard `d0acc262` model that was only allowed to learn that behavior,
judged on item 05's frozen ruler at a margin we fix in advance?

"Signed" just means the number can be positive or negative: the size says how different the two
sides are, and the sign says which side leans. "Antisymmetric by construction" means the mirror-flip
of the sign is guaranteed by the wiring, not learned from data. See the glossary at the bottom for
any word that is new.

## 2. Why this idea, in plain words

Here is the biology, kept simple. The brain wires that carry movement commands cross over to the
opposite side of the body (this crossover point is called the pyramidal decussation, Natali and
Javed StatPearls, PMID 30571044, 30521239). So a one-sided brain problem shows up as a one-sided
body problem. That is why these conditions leave a left-versus-right mark on a walking person:

- Stroke weakens one side, because the motor wires crossed over (PMID 30571044, 30521239). The
  validated measuring stick is the Symmetry Ratio on step length, swing time, and stance time
  (Patterson et al. 2010, PMID 19932621).
- Hemiplegic cerebral palsy comes from a one-sided injury to the developing brain's leg fibers, so
  one leg is stiff (Volpe 2009, PMID 19081519; Back 2007, PMID 17261726).
- Parkinson's usually starts on one side, because the brain loses its dopamine cells on one side
  first (Riederer and Sian-Hulsmann 2012, PMID 22367437).

Myopathy is the odd one out. It is a muscle disease that weakens both sides about equally (Barohn
2014, PMID 25037080), and on a stick figure it shows no real left-right difference compared to
healthy walkers (Xiong 2023, PMID 37525241), with preserved cadence and an abnormal anterior pelvic
tilt of 16.4 degrees versus 11.6 degrees in typically-developing controls (Vandekerckhove 2022, PMID
35721358). In plain words, that 16.4 versus 11.6 means the pelvis tips forward more, but the two
sides still match each other.

So there is one measuring stick that tells most of these apart: how different the two sides are, and
which side leans. The everyday picture: imagine a scale that weighs "left leg activity minus right
leg activity." Swap the person's two sides in a mirror, and an honest scale must read the exact
opposite: same size, opposite sign. A both-sides-equal walker (myopathy) parks at zero, because zero
is the only number that stays the same when you flip its sign.

The project already found a clue that makes this worth doing. In the pooled readout the project used
before, asymmetry was the WEAKEST thing it could decode (R-squared about 0.154), far below step
amplitude (R-squared about 0.719). "R-squared" is a 0-to-1 score for how well a prediction matches
the truth: 1 is perfect, 0 is no better than guessing the average. So the axis the biology says
should matter most is the one the old readout did worst on. A likely reason is that the old readout
averaged tokens together in a way that is side-blind by construction, throwing away the very
left-right information the axis needs. This idea attacks that head-on by wiring the side-awareness in.

Item 05 (a sister proposal) asked whether the standard trained model HAPPENED to learn this honest
mirror behavior. This idea takes the next step: BUILD the mirror behavior into the model, then check
whether building it in helps. See ./images/fig3.svg for the friendly version of the whole idea: a
walker with a stronger left side reads positive, the same walker in a mirror reads the exact
negative, and a both-sides-equal walker sits at zero. How to read that picture: follow the arrow
through the mirror and watch the sign flip from plus to minus.

## 3. What data you need

### The internal data (this is where the real work happens)

The core work uses the gavd5 GAVD cohort (Ranjan et al., IEEE Access 2025, DOI
10.1109/ACCESS.2025.3545787). In plain terms it is a set of short YouTube clips of people walking,
already turned into skeletons. A skeleton is a moving stick figure: for each video frame a pose
detector (MediaPipe BlazePose) marks 33 body joints, so the video becomes 33 dots that move over
time. This hides the person's face and clothes and keeps only how the body moves.

The exact shape of the canonical cohort:

- 96 sequences from 18 unique source videos.
- Per condition, the number of source VIDEOS is tiny and uneven: normal 1, Parkinson's 2, stroke 3,
  myopathic 10, cerebral palsy 2. All 12 normal clips come from a single video (`3KnFt8bH3tE`).
- A wider curriculum adds accepted augmented-normal windows, reaching 159 sequences from 35 source
  videos.

Two facts about the shape of one sequence, because the wiring change lands right here:

- Each clip is stretched or squeezed to exactly 64 frames. Then 4 frames in a row form one time
  patch, giving 16 time positions. With 33 joints that is 33 times 16 = 528 possible joint-time
  tokens. A token is the smallest chunk the model reads: one joint watched over a short slice of
  time.
- Only 12 of the 33 joints can ever be hidden and predicted: left/right shoulder (11,12), hip
  (23,24), knee (25,26), ankle (27,28), heel (29,30), foot index (31,32). Those same 12 joints are
  the ones the left-right axis is built from.

How a real team would obtain and clean this: the GAVD videos are public; the team runs MediaPipe
pose_landmarker_lite in video mode (single pose, confidence 0.45) to get the joints, keeps failed
pose rows on the timeline with zero visibility so gait timing is preserved, interpolates short gaps,
centers on the pelvis and scales by body size, then resizes each clip to 64 frames. Heels are the
weak link (left heel visibility about 0.699, right heel about 0.673, versus shoulders and hips near
0.988), so expect the feet to be the noisiest joints. In practice you do not redo any of this: the
project already cached the 528-token tensors, and you reuse them.

### The external data (reach-tier, and non-clinical only)

There is no honest clinical transfer test available, because no participant-disjoint skeleton cohort
for hemiplegic cerebral palsy or myopathy exists. So any external step is a reach-tier method check,
NOT a clinical test, using only these verified non-clinical multi-view pose cohorts already listed in
[../_shared_facts.md](../_shared_facts.md):

- CASIA-B (Yu, Tan, Tan 2006): 124 people across 11 camera views.
- OU-MVLP-Pose (Takemura 2018): about 10,000 people, many views, released keypoints.
- GREW (arXiv:2205.02692) and Gait3D (arXiv:2204.02569): large in-the-wild gait benchmarks.
- Human3.6M (Ionescu 2014, DOI 10.1109/tpami.2013.248): pose checked against motion capture.

The only questions these can answer are about the mirror property itself: does the built-in
left-right behavior stay stable when the camera moves, and does a genuine left-versus-right camera
swap flip the sign? Those are architecture checks on healthy, non-clinical people. They say nothing
about diagnosis.

## 4. Step by step, how to do it

This idea is effort HIGH because it RETRAINS the model. Everything after training reuses item 05's
code with no changes, so the evaluation is zero-retrain and pre-committed.

1. Build the antisymmetric head (needs new code, no training yet). Take the six left-right joint
   pairs. For each pair, run the LEFT joint and the RIGHT joint through the SAME small shared box,
   then subtract right from left, then add up those differences into one signed number. Because both
   sides go through the identical box and only their difference survives, a mirror (which swaps left
   and right and negates the sideways coordinate) is forced to flip the total's sign. In symbols,
   with `f` the shared box and `L_k`, `R_k` the two joints of pair `k`, the axis is
   `s = sum over k of ( f(L_k) - f(R_k) )`, and mirroring turns every `f(L_k) - f(R_k)` into
   `f(R_k) - f(L_k)`, its negative.

2. Prove the guarantee before trusting anything (zero training). Feed in any input, feed in its
   mirror, and check that the head's reading flipped sign to floating-point precision. On a graph of
   mirrored reading versus original reading this must land exactly on the line y = -x, which means
   slope exactly -1. See ./images/fig2.svg. How to read that picture: if the model is an honest
   left-minus-right scale, mirroring the person flips the reading, so all dots should sit on the
   diagonal running top-left to bottom-right; the built-in head sits exactly on it, and the old model
   only near it. If this check fails, stop and fix the wiring before doing anything else.

3. Keep the backbone the same (no change). The S-JEPA backbone stays as it is: one small Transformer,
   embed_dim 64, depth 2, 4 heads, GELU, pre-norm. Only the readout head is new.

4. Retrain the full curriculum (this is the expensive step, it DOES need retraining). Match the
   `d0acc262` lineage exactly: Stage 0 trains on normal for 300 epochs, then Stages 1 to 4 add
   Parkinson's, stroke, myopathic, and cerebral palsy at 75 epochs each. That is 600 curriculum
   epochs and 11,400 optimizer updates, on the 96 canonical (up to 159 curriculum) sequences.
   Single-GPU feasible. Keep the left-right FLIP OFF the whole time (flip_probability 0.0). This is
   the whole point: flipping during training would teach the model to treat left and right as the
   same, erasing the very asymmetry we care about. Keep flip off for BOTH the new model and the old
   `d0acc262` model, so the only difference between them is the wiring, not the training data.

5. Watch for collapse during training (a health check, no extra retrain). Track feature spread
   (standard deviation), effective rank, and mean pairwise cosine against the `d0acc262` reference
   figures. The old model finished at feature standard deviation 0.413745 and mean pairwise cosine
   0.609342, so use those as sanity landmarks. "Collapse" means the model got lazy and made every
   fingerprint nearly the same; if that happens the run is not usable.

6. Cache the fingerprints (zero retrain from here on). For each sequence, save the 528-token features
   from the retrained equivariant model, and separately from the standard `d0acc262` model. From now
   on nothing is retrained; you only read frozen features.

7. Reuse item 05's frozen ruler exactly (zero retrain). Fit item 05's ridge probe (a simple linear
   rule with a small penalty that keeps its weights modest) to read the signed axis from the cached
   features, choosing that penalty on training sources only. Report held-out-source R-squared and
   mean absolute error (the average size of the miss). Do this for both models, against item 05's two
   fixed reference bounds: the raw-coordinate null (the same fit on handcrafted signed left-minus-right
   coordinates, no network at all) and the untrained-encoder floor (the same probe on a random,
   untrained model of the same shape).

8. Run item 05's mirror-slope check on every model (zero retrain). Apply the exact left-right mirror,
   re-embed, decode, and fit the slope of decoded-mirrored versus decoded-original against the ideal
   line y = -x. For the built-in head this slope is -1 by construction (it just confirms the wiring
   and cache are correct); for the standard `d0acc262` model it is a measured, approximate number.

9. Make the two decisive pictures. ./images/fig1.svg pairs each held-out video dot for dot, new model
   against old, with the 0.05 R-squared advantage drawn as a pass band. How to read it: each pair of
   dots is one held-out video, higher is better, and the shaded band shows the head start the new
   model must win by. ./images/fig2.svg is the mirror-slope check from step 8.

## 5. The decision rule, decided in advance

We fix the rule before we fit anything, so we cannot move the goalposts later.

The split is source-video-disjoint: whole videos are held out, never single clips, and the signed
axis is pooled across all conditions with every source video as one dot. We do NOT report per-class
scores on held-out sets of size one, because a single point is not a distribution. The main
comparison runs on a provenance-matched subset (all canonical-path sequences), because most normal
clips came through the augmented processing path while every abnormal clip came through the canonical
path, and we do not want the model reading how the video was processed instead of how the person
walked.

The primary endpoint is: held-out-source signed-decodability R-squared of the retrained equivariant
model MINUS that of the standard `d0acc262` model, both read with item 05's identical frozen probe.

A positive result needs ALL of these at once:

- The equivariant model beats the standard model by at least 0.05 R-squared. Below that, the wiring
  bought nothing measurable.
- The equivariant model clears item 05's own bar: at least 0.05 R-squared above the untrained floor,
  and at least 80 percent (a fraction of 0.80) of the raw-coordinate null.
- The decoded sign points the correct way on at least 75 percent (a fraction of 0.75) of held-out
  sources.

If any one of these is missed, the run is scored as an informative null: at this scale, building the
rule in did not beat letting the model learn it.

### Worked example (illustrative numbers only, not measured facts)

Suppose on 4 held-out source videos the standard `d0acc262` probe scores R-squared 0.36, the
equivariant model scores 0.44, the untrained floor scores 0.05, and the raw-coordinate null scores
0.50.

- Beat the standard model by at least 0.05: 0.44 minus 0.36 = 0.08, which is above 0.05. Pass.
- Beat the floor by at least 0.05: 0.44 minus 0.05 = 0.39, far above 0.05. Pass.
- Reach at least 80 percent of the null: 0.80 times 0.50 = 0.40, and 0.44 is above 0.40. Pass.
- Sign correct on at least 75 percent of sources: correct on 3 of 4, and 3 / 4 = 0.75, which meets
  the bar. Pass.

All four pass, so this illustrative run would support the claim that reflection-equivariance is the
right built-in rule and that building it in beats learning it. But if the equivariant model had
instead scored 0.39, then 0.39 minus 0.36 = 0.03 is below the 0.05 margin, and the run would be an
informative null even though 0.39 still clears the floor. (These numbers are made up to show the
arithmetic; they are not measured results.)

## 6. Controls that keep us honest

- Missingness-only baseline. Across the portfolio, the simplest trap-catcher is the missingness-only
  control: a classifier given only which joints were visible, and no coordinates. The full model
  scores 0.793 accuracy across the five conditions, missingness-only scores 0.448, and pure guessing
  across five classes is 0.20. Missingness sits above chance but well below the full model, so some
  apparent signal is really just gaps, which is why any real finding must beat it.
- Raw-coordinate null (the ceiling). The same probe fit on handcrafted signed left-minus-right
  coordinates, no network at all. If the model cannot get near this, the network is not adding value.
- Untrained-encoder floor. The same probe on a random, untrained model of identical shape. The model
  is credited only when it clears this floor by the fixed margin.
- Mean/std-pooled negative control. A readout that averages tokens is side-blind and order-blind by
  construction, so it MUST NOT recover a signed axis. If it does, the "signed" claim is an artifact
  and gets withdrawn.
- Source-video-disjoint splits. Whole videos are held out, so a held-out video is genuinely new to
  the probe. Every source video is one dot; no per-class scores on held-out sets of size one.
- One-fingerprint binding. Bind every number to a single checkpoint before comparing. The standard
  baseline is the `d0acc262` lineage; the equivariant model gets its own recorded fingerprint. Do not
  mix in the `dba24a` canonical lineage.
- Flip stays off for both models (flip_probability 0.0), so any advantage cannot be an "it saw
  mirrored data" artifact.
- Equivariance manipulation check. Confirm numerically that the built-in head's mirror slope is -1 to
  floating-point tolerance before any comparison is trusted.

## 7. What could happen, and what each outcome would mean

- The equivariant model clears all the margins. This licenses the claim that reflection-equivariance
  is the correct built-in rule for separating one-sided from both-sides gait, and that building it in
  beats learning it, on a fixed ruler, at a pre-registered margin. That is a claim about how to BUILD
  gait models, not a claim about diagnosing anyone.
- The equivariant model does NOT clear the margins. This is an informative null and is genuinely
  useful. Either the standard model already learned the mirror behavior well enough on its own
  (remember flip was off the whole time), or the signed axis is simply not the bottleneck for this
  separation at n=18 sources. Both readings retire the easy intuition that "just add the
  equivariance" is a free win at this scale.
- The mean/std-pooled control fires anyway. Then the signed claim is withdrawn, because a side-blind
  readout should not be able to recover a signed number; if it does, we were reading a magnitude or
  acquisition artifact, not real side information.
- Secondary mechanism check. On the provenance-matched canonical subset, one-sided-labelled sources
  (stroke, hemiplegic-labelled CP, PD) should sit away from zero with a consistent sign, and
  both-sides-labelled sources (myopathic) should sit near zero (Xiong 2023, PMID 37525241; Patterson
  2010, PMID 19932621). This is reported with every source as a dot, never as a per-person diagnosis.

## 8. What this cannot tell us

- Transductive. The model saw every evaluation video during training, so no number here is an
  out-of-sample performance estimate. They are representation diagnostics on a frozen model. A held-
  out probe split is still transductive if the model saw that video's clips (Kapoor and Narayanan,
  arXiv:2207.07048; Varoquaux, NeuroImage 2018). Seed variation is not source variation.
- Tiny sample. There are only 18 canonical source videos, and per-condition source counts go as low
  as 1. That is why we pool across conditions and plot every source as one dot, and why we make no
  per-class asymmetry claim.
- Provenance confound. Normal is one video on a mostly-augmented path while abnormal rows are on the
  canonical path, so a naive contrast could learn processing differences (embedding-level
  normal-versus-abnormal separability is about 0.96 AUC, which is exactly what the confound puts at
  risk). The provenance-matched subset reduces this but cannot fully remove it at this size.
- Monocular capture. gavd5 is single-view, so the view-stability and genuine-mirror questions can
  only be probed on the external, non-clinical multi-view cohorts, and even there the claim is about
  the mirror property, not diagnosis.
- Skeleton limits. Skeletons cannot recover forces or push-off, muscle-electrical activity or
  stiffness, twisting (transverse-plane) rotation, or a muscle-disease diagnosis. This proposal claims
  none of those.

## 9. How to make it reproducible

- Fix all seeds so runs repeat, and remember that changing the seed is not the same as changing the
  source video.
- Bind every result to ONE checkpoint fingerprint before comparing: `d0acc262` for the standard
  baseline, and the equivariant model's own recorded fingerprint for the new one. Never mix lineages.
- Save the source-video-disjoint fold manifest (which video was held out in which fold) so the exact
  split can be rebuilt.
- Save the results: the per-source R-squared and mean absolute error for every lane, the mirror-slope
  numbers, and the mechanism-check dots, in a results file next to the figures.
- Keep item 05's signed target function and fold manifest frozen and unchanged, so the equivariant
  model is judged on exactly the same ruler item 05 used.

## Glossary

- Skeleton: a moving stick figure of 33 joints traced over a walking person; it keeps how the body
  moves and hides the face and clothes.
- Token: the smallest chunk the model reads, one joint watched over a short slice of time. There are
  33 joints times 16 time positions = 528 tokens.
- Embedding: a short list of numbers that acts like a fingerprint for a piece of input; similar
  inputs get similar fingerprints.
- Encoder: the part of the model that turns raw joint positions into embeddings.
- JEPA: Joint-Embedding Predictive Architecture; a learning trick that hides part of the input and
  predicts the hidden part as a fingerprint, not as exact coordinates.
- Signed axis: one number that can be positive or negative; the sign says which side leans, the size
  says how much.
- Reflection-equivariant (antisymmetric by construction): mirroring the input is guaranteed by the
  wiring to flip the output's sign, so `s(mirror of x) = -s(x)` for every input, trained or not.
- Probe: a small, simple (here linear) rule trained on top of the frozen model's fingerprints to read
  out one quantity, like the signed axis.
- R-squared: a 0-to-1 score for how well a prediction matches the truth; 1 is perfect, 0 is no better
  than always guessing the average.
- Transductive: the model was trained on the very videos you later test it on, so a good score can be
  memorizing rather than learning something that transfers.
- Source-video-disjoint: no clip from a held-out video is used to fit the probe, so the test video is
  genuinely new to the probe.
- Missingness-only: a baseline that sees only which joints were visible and none of the coordinates,
  used to catch a model that is reading gaps instead of gait.

## References

- Abdelfattah and Alahi, S-JEPA, ECCV 2024, DOI 10.1007/978-3-031-73411-3_21.
- Assran et al., I-JEPA, CVPR 2023, arXiv:2301.08243.
- Bardes et al., V-JEPA, 2024, arXiv:2404.08471.
- Bardes, Ponce, LeCun, VICReg, ICLR 2022, arXiv:2105.04906.
- Locatello et al., "Challenging Common Assumptions in the Unsupervised Learning of Disentangled
  Representations", ICML 2019.
- Grishchenko et al., BlazePose GHUM, 2022, arXiv:2206.11678.
- Ranjan et al., GAVD, IEEE Access 2025, DOI 10.1109/ACCESS.2025.3545787.
- Kapoor and Narayanan, "Leakage and the Reproducibility Crisis in ML-based Science", 2022,
  arXiv:2207.07048.
- Varoquaux, "Cross-validation failure: small sample sizes lead to large error bars", NeuroImage
  2018.
- Natali and Javed, StatPearls, corticospinal tract anatomy, PMID 30571044, 30521239.
- Volpe, Lancet Neurology 2009, periventricular leukomalacia, PMID 19081519.
- Back et al., Stroke 2007, periventricular white-matter injury, PMID 17261726.
- Riederer and Sian-Hulsmann, J Neural Transm 2012, asymmetric nigrostriatal onset in PD, PMID
  22367437.
- Patterson et al., Gait Posture 2010, symmetry-index methods (Symmetry Ratio), PMID 19932621.
- Schaafsma et al., J Neurol Sci 2003, stride-time CV fallers vs non-fallers, PMID 12809998.
- Barohn et al., Neurol Clin 2014, symmetric proximal (limb-girdle) distribution, PMID 25037080.
- Xiong et al., Biomed Eng Online 2023, DMD shows no significant left-right asymmetry, PMID 37525241.
- Vandekerckhove et al., Front Hum Neurosci 2022, DMD anterior pelvic tilt 16.4 vs 11.6 deg, PMID
  35721358.
- Stenum et al., PLoS Comput Biol 2021, markerless pose validity, PMID 33891585.
- Yu et al., CASIA-B multi-view gait dataset, 2006.
- Takemura et al., OU-MVLP-Pose multi-view gait dataset, 2018.
- Zhu et al., GREW gait recognition in the wild, arXiv:2205.02692.
- Zheng et al., Gait3D, arXiv:2204.02569.
- Ionescu et al., Human3.6M, DOI 10.1109/tpami.2013.248.
