# How to actually run the 64-frame resize tax study

This is the plain-language, roll-up-your-sleeves guide to Idea 4. The README next door
([README.md](./README.md)) tells the science story. This file tells you HOW to do it on real
data, step by step, in words a motivated high-school student could follow. Every number here
must match the two fact files: [`../_shared_facts.md`](../_shared_facts.md) for the numbers, and
[`../_neuro_facts.md`](../_neuro_facts.md) for the biology. If anything here disagrees with those
files, those files win.

One thing to keep in mind the whole way through: the folder labels (normal, parkinsons, stroke,
myopathic, cerebral_palsy) are just tags that came with the dataset. They are NOT diagnoses made
by this project. We use them only as motivating context.

## The big idea in plain words (read this first)

Before any walking clip reaches this project's model, the clip gets stretched or squeezed in time
so it is exactly 64 frames long. Think about two songs: a slow one that lasts 4 minutes and a fast
one that lasts 2 minutes. Now imagine a machine that forces both songs into exactly 64 beats, no
matter how long they really were. Afterward you can still count 64 beats in each song, but you can
no longer tell which one was fast and which was slow in real time. Beats-per-song survive.
Beats-per-minute are gone.

That is exactly what the 64-frame step does to walking clips. How fast someone really walks (their
steps per minute) gets flattened. This study asks one plain thing: how much real-time walking-rhythm
information does that 64-frame step throw away? We answer with a little math (a proof that the
information cannot survive) plus a hands-on measurement on the raw skeleton data (showing how much
really is lost).

### A small glossary (only the words this study actually uses)

- **Skeleton.** A stick figure traced over a person. A pose detector finds 33 body joints in each
  video frame, so a walking clip becomes a moving stick figure. Smaller than video, and it hides the
  face and clothing.
- **Frame.** One still picture in a video. A video is just many frames shown quickly in a row.
- **fps (frames per second).** How many frames the camera recorded each second, around 30 here. If
  you know fps, you can turn a frame number into real seconds. If you throw fps away, you cannot.
- **temporal_resize.** The mandatory step that stretches or squeezes any clip to exactly 64 frames.
  "Temporal" just means "in time". It is not optional: the model was built with 64 input frames
  wired in, so every clip must be forced to that length.
- **Cadence.** How many steps a person takes per minute of real time (steps per minute). It only
  means something once you know how much real time passed.
- **Walking speed.** How far a person moves per second of real time. Also a real-time quantity.
- **Double support.** The short moments in walking when both feet touch the ground at once. Its
  real-time length (in seconds) is partly a real-time quantity too.
- **R-squared (written R^2).** A score from 0 to 1 saying how much of the real spread in one quantity
  you can explain from another. 1 means "explains everything", 0 means "explains nothing".
- **Resize tax.** Our name for the lost fraction: `tax = 1 - R^2`. A tax near 1 means the information
  is gone; near 0 means it survived.
- **Source-video-disjoint.** A fair-testing rule: never tune anything on a video and then also test
  on clips from that same video. Two clips from one video are not two pieces of evidence, the same
  way two frames of one movie are not two different movies.

## 1. The question in one sentence

By how much does the mandatory fixed-64-frame temporal_resize destroy recoverable absolute cadence
and walking-speed information, measured as the fraction of clinically relevant timing variance lost,
using only handcrafted kinematics on raw coordinates plus an algebraic proof that the information
cannot survive?

In even plainer words: how much of "how fast this person really walks" gets erased when we squash
every clip to 64 frames?

## 2. Why this idea, in plain words

Cadence (steps per minute) is one of the sharpest ways to tell the folder labels apart in this
cohort. Here is the biology behind why, taken from [`../_neuro_facts.md`](../_neuro_facts.md), told
simply.

- **Stroke: a slower walk with fewer steps per minute.** The main wiring from the movement part of
  the brain down to the muscles crosses to the other side of the body low in the brainstem (the
  crossing is called the pyramidal decussation). Because it crosses, damage in one half of the brain
  causes weakness on the opposite side, the one-sided weakness that marks stroke gait (Natali and
  Javed, StatPearls, PMID 30571044). What a camera sees is a slower walk with a reduced number of
  steps per minute. Patterson et al. 2008 (PMID 18226655) report that most community stroke survivors
  carry a timing asymmetry, and that this timing asymmetry tracks walking speed and recovery. The
  grounded stroke cadence band sits near 50 to 65 steps per minute.

- **Parkinson's: short strides plus a raised step rate to make up for them.** Loss of the
  dopamine-making cells in a brain region called the substantia nigra leads to a loss of automaticity
  (the "run it in the background" control of well-learned movement like walking without thinking;
  Redgrave et al. 2010, PMID 20944662; Wu, Hallett, Chan 2015, PMID 26102020). Morris et al. 1994
  and 1996 (PMID 7953597, PMID 8800948) show the core problem is a shortened stride length, and that
  a higher step rate shows up as a compensation for those short strides. So a person in the Parkinson's
  folder can raise their step rate well above the reduced stroke cadence.

So two folders can sit at very different real-time step rates. Both signatures live in wall-clock
quantities: cadence in steps per minute, stride time in seconds. And here is the catch: the moment
you resize every clip to the same 64 frames, you throw away how long the clip actually lasted, so
those very different real-time rates land on the same 64-frame ruler. Whatever real-time gap separated
them is gone.

Why bother measuring this? Because if the resize collapses that difference, then any later claim that
"the model tells conditions apart by their walking rhythm" is at risk of being an artifact of what
survived preprocessing, not of what the model learned. This study turns a hidden preprocessing choice
into an explicit, quantified warning label. It fits the concern of Kapoor and Narayanan (2022,
arXiv:2207.07048) that a quiet preprocessing choice can destroy a decisive signal without anyone
noticing.

There is a general lesson too, one that transfers beyond this dataset: duration-warping any skeleton
clip onto a fixed frame grid is a "cadence-erasing" operation, so absolute cadence, stride time, and
walking speed cannot be recovered from it by construction. That holds for any fixed-frame skeleton
pipeline, including V-JEPA-style tube-length normalization (Bardes et al., V-JEPA, arXiv:2404.08471).

See [images/fig3.svg](./images/fig3.svg) for the beginner picture of this. How to read it: the top
row is the two-songs analogy, the bottom row is the real walking version, and they line up step for
step. The point is that both rows lose the same thing, real-time rhythm, once everything is squeezed
to the same fixed length.

## 3. What data you need

**The internal work (the whole core study) uses one dataset: the gavd5-drift GAVD cohort** (Ranjan et al.,
IEEE Access 2025, DOI 10.1109/ACCESS.2025.3545787). Concretely that is 96 sequences from 18 unique
YouTube source videos. The per-condition sequence counts are normal 12, Parkinson's 9, stroke 12,
myopathic 47, cerebral palsy 16. But the counts that really matter are the source-video counts,
because the source video, not the clip, is the independent unit: normal 1, Parkinson's 2, stroke 3,
myopathic 10, cerebral palsy 2. Those small source counts, not the bigger clip counts, set how many
independent dots we can put on a plot.

**What shape does the data take?** Each clip is a skeleton: 33 body joints found per frame by a pose
detector called MediaPipe BlazePose (Grishchenko et al. 2022, arXiv:2206.11678), each joint with an
x, a y, and a relative z coordinate. So a clip is a stack of numbers shaped like (frames, 33 joints,
3 coordinates). Source video resolution is 1280x720; measured fps clusters around 30, with real
variation (some clips are 23.976 fps, some are 29.97 fps).

**How a real team obtains and cleans it.** The GAVD CSVs give you the skeleton coordinates per clip,
but they do NOT store fps. This is the single most important data fact for this study, so read it
twice: without fps you cannot turn a frame number into real seconds, and without real seconds you
cannot measure cadence. So a real team must re-extract the coordinate cache from the actual MP4 source
videos and record, for each clip, the measured fps and the real duration in seconds. When the pose
detector fails on a frame, keep that frame on the timeline with zero visibility rather than deleting
it, because deleting frames would itself distort the timing.

**External data (reach tier, non-clinical, optional).** The core study needs NO external cohort. If a
team later wants a cross-check that the erased biomarker is real in another recording, the only
verified non-clinical cohorts available in [`../_shared_facts.md`](../_shared_facts.md) are CASIA-B,
OU-MVLP-Pose, GREW, Gait3D, and Human3.6M. Say plainly that these are non-clinical and reach-tier.
They do not turn this into a clinical claim. (The README also notes a PhysioNet Parkinson's force-and-
sensor cohort as a label-level cross-modal check only, never as skeleton clinical transfer.)

## 4. Step by step, how to do it

Good news up front: this study is **zero encoder training**. No optimizer ever runs. Everything happens
on coordinates, mostly on a normal laptop CPU. The frozen checkpoint (fingerprint `d0acc262`) is only
touched as optional context in the last week, and even then it is never fed native-rate input. The
reason we do not build a "native-rate encoder arm" is that the encoder is hard-wired to 64 frames, so
feeding it clips of other lengths would confuse "the resize effect" with "the encoder acting weird on
unfamiliar input". We avoid that trap by measuring information loss on the coordinates directly.

1. **Regenerate the coordinate cache WITH real fps. (Needs re-extraction; still zero-retrain.)** The
   GAVD CSVs do not store fps, and the existing notebook-02 extraction path probes fps from the MP4
   with `cv2.CAP_PROP_FPS` and a hard-coded 29.97 fallback (a default used when the true fps cannot be
   read). So the very first task is to re-extract the per-clip skeleton cache from the source videos and
   record, per clip, the measured fps and the real duration in seconds. Reuse the existing notebook-02
   extraction harness and proposal 01's coordinate loaders rather than rewriting them. Keep failed-pose
   rows on the timeline with zero visibility.

2. **Check the fps is real, not the fallback. (This is the Day-5 kill gate.)** Plot the fps and real-
   duration spread across the 18 source videos. If every clip got pinned to 29.97, the fallback
   contaminated the sample and the whole study is dead in the water, because it depends on real
   durations genuinely varying. If that happens, stop and re-extract. Only continue if fps is
   non-degenerate (varies, and includes the known 23.976 and 29.97 values) and real durations vary
   across sources.

3. **Build two views of the SAME clips. (Zero-retrain.)** From identical raw poses, hold two coordinate
   views per clip:
   - the **native-rate view**, which keeps the real time axis (`frame_index * (1 / fps)` gives seconds);
   - the **resized view**, the project's mandatory 64-frame temporal_resize, which maps the clip onto a
     fixed 0-to-1 grid.
   Because both views come from the exact same raw poses, any difference in recoverable timing is due to
   the resize alone.

   Reading the math: `t_seconds = frame_index * (1 / fps)`. Here `1 / fps` is the length of one frame in
   seconds (about 0.0333 s at 30 fps), and `*` is multiplication. Delete `fps`, as the resize does, and
   you can no longer form `1 / fps`, so real seconds are unrecoverable. That deletion is the whole point.

4. **Compute the three handcrafted timing targets on the native-rate view. (Zero-retrain.)** Using only
   raw coordinates and validity masks, compute per clip:
   1. absolute cadence in steps per minute, from the dominant frequency of the ankle vertical-separation
      wiggle converted to real time with fps;
   2. a walking-speed proxy in body-heights per second, from horizontal hip movement per real second;
   3. double-support fraction and its real-time length, from the moments both feet are near the ground.
   Reading the math: `cadence = peak_frequency_hz * 60`, where `peak_frequency_hz` is how many times per
   second the feet cycle and `60` turns per-second into per-minute. To get `peak_frequency_hz` you must
   know real time, which needs fps. Freeze these three definitions before any comparison. They are gait
   scalars, not diagnoses.

5. **Try to recover the same three targets from the resized view. (Zero-retrain.)** From the 64-frame
   resized coordinates alone (no fps, by construction), attempt to recover each target. On the resized
   grid you can only get "cycles per normalized clip", never per real second. The gap between the
   native-rate answer and the resized answer is the resize tax.

6. **Turn the gap into a number. (Zero-retrain.)** For each target, compute `tax = 1 - R^2`, where `R^2`
   is the fraction of the native-rate target's spread that the resized-recovered value can explain,
   measured across source-disjoint clips. A tax near 1 means the timing information is gone.

7. **Run the sanity controls (Section 6). (Zero-retrain.)** The duration-stretch identity check, the
   shuffled-time null, and the fps-side-channel check.

8. **(Optional, last week.) Overlay the frozen model as labeled context only.** If you want, you can
   show how the pooled model behaves on resized input, bound to checkpoint `d0acc262`, clearly marked as
   non-primary. Never mix it with the locally observed `dba24a` lineage.

## 5. The decision rule, decided in advance

We write the pass bar down BEFORE looking at any result, so we cannot move the goalposts later.

**Primary endpoint.** The mean resize tax across the three targets (cadence, speed, double-support),
macro-averaged over source videos. Macro-averaged means: compute the tax per source video, then average
those per-video numbers so each video counts equally no matter how many clips it gave. This stops one
video with many clips from dominating.

**Pre-registered margins.**
- cadence tax at least 0.70 (at least 70 percent of native-rate cadence spread lost),
- speed tax at least 0.70,
- double-support real-time-duration tax at least 0.50.

Reading the margins: `>=` means "at least this large". Cadence and speed get the higher bar (0.70)
because the proof says they are strictly invariant under the warp, so almost nothing should survive.
Double support gets the lower bar (0.50) because its real-time length is only partly a wall-clock
quantity. The 0.70 floor is set deliberately below the near-total loss the proof predicts, so it is a
conservative pass bar, not a knife-edge. If measured tax comes in below these values, we must explain
it by broken fps (the Day-5 gate), not claim the information survived.

**A worked example (illustrative numbers only, not measured facts).** Suppose 5 source videos have
true native-rate cadences of 55, 60, 95, 130, 150 steps per minute. Suppose the resized view, having
lost fps, can only report a near-constant readout that maps back to predictions of 92, 94, 96, 95, 93.

- Mean of the true values: `(55 + 60 + 95 + 130 + 150) / 5 = 490 / 5 = 98`.
- Total spread (sum of squared gaps from the mean): `1849 + 1444 + 9 + 1024 + 2704 = 7030`.
- Leftover error (sum of squared gaps between true and predicted): `1369 + 1156 + 1 + 1225 + 3249 = 7000`.
- `R^2 = 1 - (7000 / 7030) = 1 - 0.9957 = 0.0043` (about 0.00).
- `tax = 1 - R^2 = 0.9957` (about 1.00).

How to read it: a tax of about 1.00 clears the 0.70 cadence margin easily, matching the proof that
absolute cadence cannot survive the resize. These five numbers are made up to show the arithmetic; the
real values come from Week 2. (Illustrative numbers only, not measured facts.)

## 6. Controls that keep us honest

- **The algebraic identity is the anchor baseline.** The whole study is already non-neural, so the
  baseline is the math itself. On a duration-warped 64-frame grid, two clips of different real duration
  map to identical token time-grids, so any per-normalized-clip cadence readout carries zero real-time-
  rate information. We verify this numerically: take one clip, synthetically stretch its real duration
  by a factor `k`, resize both, and confirm the resized coordinates come out (up to interpolation
  rounding) identical while the native-rate cadence changes by exactly `k`. In symbols:
  `resize(clip) = resize(stretch(clip, k))` while `native_cadence(stretch(clip, k)) = native_cadence(clip) / k`.
- **Shuffled-time null (must fail).** Randomly scramble the real-time frame order inside each clip, then
  compute the native-rate targets. Cadence and speed must become unrecoverable in BOTH views. If a
  "cadence" number survived after we scrambled time, our estimator was reading something other than real
  rhythm, and we could not trust it.
- **fps side-channel check.** If any arm sneaks an fps or dt scalar in next to the coordinates, predict
  the target from that scalar alone. This makes sure a "win" comes from genuine kinematics, not from fps
  leaking in a side door.
- **Source-video-disjoint splits.** Never fit a threshold or scale on a clip whose source video also
  supplies an evaluated clip. Because several conditions have very few sources (normal 1, Parkinson's 2,
  cerebral palsy 2, stroke 3), we do not report per-class held-out R-squared on single-source classes;
  we pool across conditions and show every source video as one dot.
- **One-fingerprint binding.** Any optional model overlay is bound to checkpoint `d0acc262` and never
  mixed with the locally observed `dba24a` lineage.
- **Encoder-exposure labeling and no peeking.** We label numbers with encoder-exposure status even
  though the encoder is not the object, and every threshold is fit only on held-out-source clips, so no
  comparison here is transductive (peeking at the exact data used to judge it).

## 7. What could happen, and what each outcome would mean

- **A large measured tax (cadence and speed each at least 0.70), matching the proof.** This is the
  expected result. It licenses the claim that any timing-based separation this model appears to achieve
  cannot be coming from absolute cadence or speed, because that information was algebraically removed
  before the model saw a single token. In plain words: "the model distinguishes fast from slow gait" is
  not a defensible claim under fixed-length resizing.
- **A small measured tax (below the margins), with fps confirmed healthy.** This would be surprising and
  equally informative. It would mean the resize is not actually warping durations in this cohort, most
  likely because the clips already have near-identical real durations, so resizing barely changes
  anything. That is a hard, falsifiable prediction failing, and it is a useful fact about the dataset.
- **A small measured tax caused by broken fps.** If the Day-5 gate was passed wrongly and fps was
  degenerate, a small tax means nothing about the resize; it means the time base was corrupt. We then
  re-extract, not publish a "survival" claim.
- **The shuffled-time null does NOT fail (targets survive scrambling).** That would mean our estimators
  are reading a static artifact, not real rhythm, and we would fix the estimators before trusting any
  tax number.

See [images/fig1.svg](./images/fig1.svg) and [images/fig2.svg](./images/fig2.svg). How to read fig1:
follow one slow clip and one fast clip from the left; they start at different real lengths, but both get
pulled onto the same 64-frame ruler on the right, and the cadence readout printed under each ends up the
same. That "same readout for different real speeds" is the resize tax in one image. How to read fig2:
each group of bars is one timing target; the native-rate bar (real time kept) should be low error, the
resized bar (real time thrown away) should be high error, and the gap is the tax. Each dot is one source
video, so you can see the effect is not driven by a single video.

## 8. What this cannot tell us

- **Transductive framing.** Every gavd5-drift readout is transductive: the encoder saw every evaluation row
  during the curriculum, so nothing here is an out-of-sample performance estimate. This study sidesteps
  that for its own numbers by never running the encoder on native-rate input, but the honesty label
  still stands for any model overlay.
- **Tiny, unequal sample.** With as few as one source video per class (normal 1), we can only pool across
  conditions and plot each source as a dot. Per-class timing numbers would be a single point dressed up
  as a distribution. Error bars are large at these sample sizes (Varoquaux, NeuroImage 2018).
- **Provenance confound.** Most normal rows use the augmented extraction path while every abnormal row
  uses the canonical path, so any label-linked contrast could reflect an acquisition difference. This
  study measures timing loss, not condition separation, which limits exposure, but the confound is worth
  stating.
- **Monocular capture.** gavd5-drift is single-camera, so depth and true 3-D motion are estimated, not
  measured.
- **Skeleton limits.** Skeletons cannot recover kinetics or propulsion, EMG or spasticity, transverse-
  plane rotation, or an etiologic muscle diagnosis. The cadence, speed, and double-support quantities are
  handcrafted gait descriptors used to measure preprocessing loss; they are not validated clinical
  biomarkers, and the folder labels remain dataset annotations, not diagnoses.

## 9. How to make it reproducible

- **One checkpoint.** Bind any optional model overlay to the single fingerprint `d0acc262`, never mixing
  it with the `dba24a` lineage. The core study touches no checkpoint at all.
- **Seeds.** There are no training seeds here (no optimizer runs), so we do not attach seed-based error
  bars to a study that has none. The variation that matters is across source videos, which we show as
  dots. Fix seeds only for the shuffled-time scramble so the null is repeatable.
- **Save the split manifest.** Write out the list of source videos and exactly which clips fell in the
  fit set versus the evaluation set for every source-disjoint comparison, so anyone can rerun the exact
  same split.
- **Save the fps manifest.** Record the measured fps and real duration for every clip from Step 1. This
  is the artifact the whole study rests on, so it must be saved and checkable.
- **Freeze the target definitions.** Save the three handcrafted estimator definitions (cadence, speed,
  double-support) exactly as used, computed once and frozen before any comparison.
- **Save the results and the extraction script.** Package the extraction script, the fps manifest, the
  frozen target definitions, and the per-source resize-tax results together, so the figures can be
  regenerated end to end.

## References

- Ranjan et al., GAVD, IEEE Access 2025, DOI 10.1109/ACCESS.2025.3545787.
- Grishchenko et al., BlazePose GHUM, 2022, arXiv:2206.11678.
- Bardes et al., V-JEPA, 2024, arXiv:2404.08471.
- Kapoor and Narayanan, "Leakage and the Reproducibility Crisis in ML-based Science", 2022,
  arXiv:2207.07048.
- Varoquaux, "Cross-validation failure: small sample sizes lead to large error bars", NeuroImage 2018.
- Natali and Javed, Corticospinal tract anatomy, StatPearls, PMID 30571044.
- Patterson et al., Gait asymmetry in community-ambulating stroke survivors, 2008, PMID 18226655.
- Redgrave et al., basal ganglia control in Parkinson's disease, 2010, PMID 20944662.
- Wu, Hallett, Chan, Motor automaticity in Parkinson's disease, 2015, PMID 26102020.
- Morris et al., walking cadence in Parkinson's disease, Brain 1994, PMID 7953597.
- Morris et al., stride length regulation in Parkinson's disease, Brain 1996, PMID 8800948.
- Stenum et al., Two-dimensional video-based gait analysis using pose estimation, 2021, PMID 33891585.
