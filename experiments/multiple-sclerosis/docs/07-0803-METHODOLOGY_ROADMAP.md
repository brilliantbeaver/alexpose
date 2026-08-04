# Why the S-JEPA score is low, and how we could make it better

_Written 2026-08-03. This document is a plain-language companion to
[`06-0803-FINAL_REPORT.md`](06-0803-FINAL_REPORT.md) (the results) and
[`04-0803-FIXES.md`](04-0803-FIXES.md) (the repairs we already made)._

Everything below is a development estimate on a tiny collection of videos that we have already
looked at closely. Nothing here is a medical finding, and nothing here should be read as a claim
that this system can diagnose a real person. It is a study of a method, not a clinical tool.

---

## Part 1. What actually happened

### 1.1 The one number people ask about

We trained the S-JEPA model and scored it. The headline score is a "macro-F1" of **0.438**.

A quick word on what that number means, because the rest of the document depends on it. We have
three groups of walking videos: healthy people we call **normal**, people with **multiple
sclerosis (MS)**, and people with **Parkinson's disease (PD)**. For each group we can ask two
questions. First, when the model says "this person is PD," how often is it right? That is called
**precision**. Second, of all the people who really are PD, how many did the model catch? That is
called **recall**. F1 is a single number that blends precision and recall for one group. Macro-F1
is just the plain average of the three groups' F1 scores, giving each group equal weight no matter
how many videos it has.

So macro-F1 of 0.438 is the average of how well the model handles normal, MS, and PD, treating the
three as equally important. For comparison, blind guessing on three groups would score about 0.333.
So the model is doing something, but not much.

### 1.2 The scores it is being compared against

Here are the numbers on the exact same videos and the exact same splits.

| System | macro-F1 |
|---|---:|
| A control that only looks at where the joints sit and how much they wiggle | 0.694 |
| Random Forest (the classic hand-built baseline) | 0.667 |
| A control that only looks at the pose detector's confidence numbers | 0.611 |
| The old, buggy version of S-JEPA | 0.570 |
| **S-JEPA (the corrected version, this study)** | **0.438** |
| Blind guessing | 0.333 |

The uncomfortable fact is that the corrected S-JEPA sits at the bottom of the list, below even the
simple "control" systems that were built to be dumb on purpose. The rest of Part 1 explains, step
by step, exactly why.

### 1.3 First clue: the model is not broken, it just learned the wrong thing

Before blaming the model, we checked whether it had "collapsed." Collapse is a common failure where
a model gives up and returns nearly the same answer for every input. To measure this we used a number
called **effective rank**. Because that number appears throughout our results, it is worth
understanding clearly, so the next section explains it from scratch.

### 1.3a What "effective rank" means, in plain language

Every time the model looks at a walk, it produces a list of 96 numbers that together describe that
walk. Think of those 96 numbers as 96 dials. In principle the model has 96 independent dials it can
turn to describe things.

The question effective rank answers is simple: **how many of those dials is the model actually
using?**

Here is the intuition. Imagine you have 96 dials on a dashboard, but you notice that whenever you turn
dial number 5, dial number 12 always turns with it by the same amount. Then those two dials are not
really independent. They are carrying the same information twice, so together they only do the work of
one dial. If many dials move in lockstep like this, then even though there are 96 of them, the number
of *genuinely independent* things the dashboard can express is much smaller.

Effective rank is a way of counting the genuinely independent dials. If the model spread its
information evenly across all 96 dials, the effective rank would be close to 96. If the model
funneled everything into just a few dials and left the rest either dead or moving in lockstep, the
effective rank would be a small number like 1 or 2.

The precise recipe is not important for following the argument, but for completeness: we take all the
model's descriptions, find the underlying directions of variation and how strong each one is, and then
compute a single number that says how many directions carry meaningful variation. A common way to state
it is that effective rank is high when the strength is shared across many directions and low when a few
directions dominate.

**Why effective rank matters.** It is our early-warning light for collapse. A model that has collapsed
produces almost the same description for every walk, which means nearly all its dials go dead and the
effective rank crashes toward 1. When that happens, any score the model gets is meaningless, because it
is not really telling the walks apart at all. So before we trust any result, good or bad, we check that
the effective rank is healthy. A healthy effective rank does not prove the model learned something
useful, but a crashed effective rank would prove it did not.

One extra note, because it caused a real bug earlier in this project. On Apple's hardware the function
that computes this number was silently returning zero, which would have hidden a collapse if one had
occurred. We fixed that by computing the number on the regular processor, so the effective rank values
we report are real measurements.

### 1.3b What effective rank tells us about our results

We measured the effective rank on each of the five folds. The values were about **8.8, 9.5, 8.2, 8.8,
and 7.7**. They sit comfortably in the range of 8 to 9 out of a possible 96, and they are steady across
every fold with no fold crashing toward 1.

This is genuinely good news, and it changes how we read the low score. It tells us three things.

First, **the model did not collapse.** It is producing varied, non-degenerate descriptions of the
walks. The low 0.438 score is therefore not the "everything looks the same" failure. It is a real,
working model that simply learned the wrong description for this task.

Second, **making the model bigger is probably not the answer.** If the model were straining against its
own size, we would expect it to be cramming information into all 96 dials, pushing the effective rank up
near its ceiling. Instead it is using only 8 or 9. It has plenty of unused capacity. So the bottleneck
is not "too few dials." This is one of the main reasons the roadmap in Part 2 holds back the
"make it bigger" ideas until a diagnostic test says they would help.

Third, **the problem lives in what the dials describe, not in how many are used.** Combine the healthy
effective rank with the confident-but-wrong behavior from section 1.6, and a clear picture emerges. The
model has built a rich description of each walk, it is confident about it, and it is wrong about
Parkinson's. That is the signature of a representation that is healthy but pointed at the wrong signal,
which is exactly the diagnosis the rest of this document is built on.

So the takeaway is short. Effective rank confirms the model is awake and using a fair spread of its
capacity. It is producing rich, varied descriptions of each walk. It is simply describing the wrong
things for this task.

### 1.4 Second clue: it fails almost entirely on Parkinson's

Let us look at where the mistakes land. Here is what the model predicted for each true group. Read
each row as "of the videos that truly belong to this group, here is how the model labeled them."

| Truly... | called normal | called MS | called PD |
|---|---:|---:|---:|
| normal (19 videos) | 11 | 3 | 5 |
| MS (11 videos) | 0 | 6 | 5 |
| PD (17 videos) | 5 | 8 | **4** |

Look at the bottom row. Of the 17 Parkinson's videos, the model got only **4 right**. It called 8
of them MS and 5 of them normal. Its recall on Parkinson's is about 0.235, meaning it catches
fewer than one in four. The Random Forest, by contrast, catches 10 of the 17.

Because macro-F1 weights all three groups equally, this Parkinson's collapse alone drags the average
down. The model's per-group F1 scores are about 0.63 for normal, 0.43 for MS, and only **0.26 for
PD**. A decent normal score cannot rescue an average that includes a PD score that low.

There is a vivid single example. One Parkinson's source video was cut into five clips. The model
labeled them MS, normal, normal, normal, normal. It got none of the five right. That is the
Parkinson's problem in miniature.

### 1.5 Third clue: it confuses Parkinson's with MS, and there is a reason

The single most common mistake is calling a Parkinson's walk an MS walk. This is not random. Both
Parkinson's and MS produce **hypokinetic** gait, which is a technical word that simply means
"reduced, slowed-down movement." People in both groups tend to take shorter steps and move less.

So on raw skeleton motion, "this person moves less than a healthy person" is easy for the model to
see. But "this person moves less in the Parkinsonian way versus the MS way" is a subtle difference
in the fine details of the movement. That subtle difference is exactly what the small model, trained
from scratch on so few videos, never learned to represent. The Random Forest does better here because
it is fed hand-engineered gait measurements, such as step-timing irregularity and left-right
asymmetry, that were designed by people who know what separates these conditions.

### 1.6 Fourth clue: it is confidently wrong, not merely unsure

You might hope the model is at least hesitant when it errs, hovering near a coin-flip. It is not.

We measured how confident the model is in its top choice. When it is right, its average confidence is
about 0.81. When it is wrong, its average confidence is about 0.76. Those two numbers are almost the
same. When the model calls a Parkinson's walk "MS," it often does so at 85 or 89 percent confidence.

This matters because it tells us the problem is not indecision that a little more tuning would fix.
The model has genuinely placed these Parkinson's walks near the MS group in its internal map of the
world. The features themselves are the ceiling. You cannot fix a confidently-wrong feature by
adjusting the final classifier on top of it.

### 1.7 Fifth clue: why the "dumb" controls beat it

This is the part that surprises people most, so we go slowly.

The controls are deliberately simple systems. One of them ignores movement almost entirely and just
looks at where the joints sit and how much they wiggle. Another looks only at the confidence scores
the pose detector reports. These score 0.694 and 0.611, both above S-JEPA's 0.438.

How can a dumb system beat a learned one? The answer is a mismatch between what the controls are
allowed to exploit and what S-JEPA is built to ignore.

When we first looked at the data, we found that the way each video was filmed is tangled up with its
group label. Every single MS video was filmed at 60 frames per second and in a square 1080-by-1080
frame. No normal or PD video was. On top of that, the pose detector reports slightly higher
confidence on the MS videos than the others. These are facts about the camera and the recording, not
about how anyone walks. A simple control can quietly lean on those facts and score well without
understanding gait at all.

Now here is the key. S-JEPA is trained with a "self-supervised" method whose whole purpose is to
learn a description of movement that ignores nuisances like camera settings and detector quirks. In
other words, S-JEPA is built on purpose to *not* use the very shortcuts the controls are cashing in.
So on this particular collection, where the shortcuts happen to line up with the answers, S-JEPA is
penalized for refusing to cheat. That is not a bug. It is the honest behavior we wanted, showing up as
a lower score because the test set rewards cheating.

The same logic explains why the old, buggy S-JEPA scored higher at 0.570. That older version had a
label leak and a fixed masking pattern that let it lean partly on those shortcuts. Removing the leak
and fixing the masking took the crutches away, and the score dropped. That drop is progress, even
though the number went down.

### 1.8 Sixth clue: is the gap even real, or just bad luck?

With only 47 videos, a fair question is whether the 0.438-versus-0.667 gap is real or just the luck
of a small sample. We checked this properly. We resampled the source videos many thousands of times
and recomputed the gap each time. The gap held up. The 95 percent confidence range for how far
S-JEPA trails the Random Forest runs from about 0.07 to 0.39, and it never touches zero. So the gap
is real, not an artifact of the small sample. That is an important thing to know before spending
effort trying to close it.

### 1.9 The one-paragraph summary of Part 1

S-JEPA scores 0.438 because it cannot tell Parkinson's from MS. It catches only 4 of 17 Parkinson's
walks and confidently mislabels most of the rest as MS, because both conditions look like "reduced
movement" and the fine distinction was never learned from so little data. It falls below the controls
and the Random Forest because those systems lean on filming-related shortcuts and hand-built gait
measurements, while S-JEPA is deliberately built to ignore shortcuts and had to learn everything from
47 videos on its own. The model is healthy and confident. It simply learned a description of walking
that does not carry the Parkinson's-versus-MS signal. This is a clean negative result, and it is worth
building on rather than hiding.

---

## Part 2. How we could make it better

The improvements below were developed by proposing many ideas and then trying hard to break each one
before keeping it. Several tempting ideas were rejected because they attacked a problem the evidence
says we do not have. We list the survivors first, then the rejected ideas, because knowing what *not*
to do is just as useful.

Two rules shaped every choice. First, we do not make the model bigger or fancier until we have shown
that the thing we are adding attacks the real cause of the failure. Making a model that learned the
wrong thing larger just learns the wrong thing more expensively. Second, no idea is allowed to break
the guardrails: the self-supervised training never sees the diagnosis label, masking is never biased
toward high-motion moments, and every result stays a development estimate with no clinical claim.

The evidence points at two root causes, so the roadmap is organized around fixing them in order.

- **Root cause one, the data.** The saved skeleton data was normalized in a way that erases walking
  speed. Speed and stride are among the strongest signals that separate these groups, and they are
  being destroyed before the model ever sees them.
- **Root cause two, the shortcut mismatch.** The filming shortcuts described in Part 1 mean the test
  set rewards behavior that S-JEPA is built to avoid. We need to measure and control that, honestly.

### Stage R2: honest measurement and fixing the data, with no new data and no bigger model

The goal of this stage is to make the R1 result trustworthy and to repair the input data. Every idea
here is cheap and safe. We do the fast measurements first, then the one real rebuild of the data.

#### R2.1 Put an honest error bar on every comparison (priority: high)

**What we do.** Take the predictions we already saved and resample the source videos many thousands of
times, keeping all the clips of one source together each time. Recompute the scores on each resample.
This gives an honest range, a confidence interval, around every number and around every gap between
two systems.

**Why it helps.** With 47 videos, a single score can mislead. A range tells you whether a gap is solid
or shaky. We already ran this once and found the S-JEPA-versus-Random-Forest gap is solid. Reporting
the range on every future comparison keeps us honest and stops us from chasing noise.

**How we would know it worked.** We can state a clear confidence range for each score and each gap,
computed only from predictions we have already saved. No retraining is needed.

#### R2.2 Measure how much of the shortcut is actually visible in the saved data (priority: medium)

**What we do.** Build a tiny model that tries to guess the group label using only recording-related
clues that we can compute from the saved data, such as clip length, coordinate scale, and detector
confidence. See how well it does. Then subtract those clues from the control features and re-score, to
see how much of the controls' lead was really shortcut.

**Why it helps.** When we checked the saved data carefully, we found something important. The strongest
shortcut, the true frame rate, is not usable from the saved data. Every saved clip was already brought
down to a common 15 frames per second when it was first processed, and the frame-rate value stored in
each saved file is simply that constant 15 for all 47 clips, the same for every group. So the "60 frames
per second means MS" trick is not available to any model reading the saved cache. The frame size is a
weaker story: the saved files do not store the recording resolution as its own field, but the raw
coordinates are kept in pixels, so a rough sense of frame scale can still leak through the coordinate
range. That is exactly why our shortcut-only model above uses coordinate scale as one of its clues. Even
so, when we tested acquisition-only clues earlier they scored near blind guessing. Taken together, this
means the controls' good scores are probably reading *real* movement and body-shape signal, not a
filming shortcut hidden in the saved data. That flips the story in a useful way. It tells us the job is
not mainly to remove a shortcut from the saved data, but to teach S-JEPA to read the real signal the
controls are already reading.

**How we would know it worked.** The shortcut-only model should score near blind guessing, and
removing those clues from the controls should barely change their scores. Both outcomes would confirm
that the saved data is cleaner than we first feared.

#### R2.3 Rebuild the skeleton data so it keeps walking speed (priority: high)

This is the single most important next step.

**What we do.** Create a brand-new copy of the processed data. We leave the old copy untouched. In the
new copy we change how the skeletons are normalized.

The old method re-centered every single frame on the person's hips. Picture filming someone walking
across a room, then shifting every frame so the hips are always dead center. The person now appears to
walk in place. All forward progress, and therefore all information about walking speed, is gone. We
measured this directly: a real forward drift of nearly 700 pixels collapses to essentially zero after
the old normalization.

The new method fixes the person's position using only the *first* frame, so that movement across the
following frames survives. We scale everyone by a single body-size measurement for the whole clip, so
that a tall person and a short person are comparable but stride length is preserved. We also recover and
store the true frame rate of each source video. The current pipeline actually reads the true rate at the
start, uses it once to decide how many frames to skip, and then stores only the common value of 15,
discarding the real rate. We want the real rate kept, because it is needed to measure speed in real
seconds. Finally we record a simple flag marking which frames are real versus filled-in or padded.

**Why it helps.** Walking speed and stride length are among the clearest differences between healthy
and slowed gait. Right now they are erased before the model sees them. Restoring them gives the model a
real chance to separate normal walks from slowed walks.

**One honest caution.** This will most likely help separate healthy walking from slowed walking. It
will *not* automatically fix the Parkinson's-versus-MS confusion, because both of those are slowed
walks. We should expect the normal group to improve and should not promise that the Parkinson's
mistakes vanish.

**A second honest caution.** Because all MS videos were filmed at 60 frames per second, if we measured
speed in "pixels per frame" we would accidentally smuggle the frame-rate shortcut back in. So speed
must be measured in real seconds, using the true frame rate we recovered. Any improvement must be
checked to make sure it survives a frame-rate control test.

**How we would know it worked.** Retrain S-JEPA with the exact same settings on the new data and
compare. The improvement is real only if it clears the old buggy score of 0.570 and survives the
frame-rate and body-scale control checks. If it does not, we learn that speed was not the missing
piece, which is also useful.

#### R2.4 Run three cheap tests to decide what kind of problem we have (priority: medium)

**What we do.** Without changing the model at all, run three quick checks on inner folds. First, sweep
the strength of the final classifier to see if a stronger or weaker classifier changes the Parkinson's
score; if it stays flat, the classifier is not the limit. Second, watch the training health numbers at
300, 1000, and 3000 training steps to see whether the model is still improving or has plateaued. Third,
scramble the time order of a walk and re-score; if scrambling barely hurts, the model is treating a
walk as a bag of poses rather than a motion, which would be a red flag.

**Why it helps.** These three checks together tell us *which* kind of problem we are facing: is the
model too small, is it starved for data, or is it a shortcut-and-objective mismatch? We should decide
this with evidence before spending effort on expensive fixes. Given everything in Part 1, we expect the
answer to be "shortcut and objective mismatch, not too small," which is exactly why the expensive
"make it bigger" ideas are held back until this test says otherwise.

**How we would know it worked.** We end with a labeled verdict, either "capacity," "data-starved," or
"shortcut and objective mismatch," and that verdict decides what we do in Stage R3.

#### R2.5 Report the headline score as a band across several random seeds (priority: medium)

**What we do.** Re-run the same frozen setup with a few different random seeds and report the average
with a spread, rather than the single point 0.438.

**Why it helps.** A single run can land a little high or low by chance. A small band is a more honest
headline. This is quick and reuses the existing pipeline. It does not change the diagnosis; it just
states the result more carefully.

**How we would know it worked.** We can report something like "about 0.44, give or take a small band,"
and confirm the band does not overlap the Random Forest's 0.667.

### Stage R3: the expensive levers, used only after Stage R2 says they will help

Stage R3 spends real effort. We only reach for each lever if the Stage R2 verdict says the model can
actually benefit from it, and every item carries a pre-agreed rule for when to declare it a failure and
stop.

#### R3.1 Ask the model to predict motion, not just position (priority: medium)

**What we do.** Today the model learns by predicting the position of hidden joints. There is good
published evidence that predicting *motion*, meaning the frame-to-frame change in position, teaches a
better description of movement. We would add a second prediction target based on motion, computed from
the new speed-preserving data.

**Why it helps.** For telling apart kinds of gait, how the body *moves* is more informative than where
it happens to be in a single frame. A published skeleton method found a large gain from switching to
motion targets.

**Why it is gated.** This idea is *impossible* on the old data, because the old normalization erased
motion. It only makes sense after the Stage R2 data rebuild. We would also first run a quick check to
see whether the model already captures motion implicitly through its grouped-frame tokens; if so, we
drop the idea rather than add complexity for nothing.

**How we would know it worked.** The Parkinson's confidence should rise measurably, the collapse
warning signs should stay clear, and, importantly, the improvement must not widen the gap versus the
shortcut controls. If motion mostly amplifies detector jitter or re-encodes the frame-rate shortcut, we
reject it even if the raw score goes up.

#### R3.2 Warm up the model on outside walking data before training on ours (priority: medium)

**What we do.** Train the model first on a large public collection of walking data, with no labels, so
it learns a general sense of human gait. Then continue training on our small collection. The outside
data uses a different skeleton format, so we would carefully map its joints onto ours, keep only the
two-dimensional information, and take care not to import new shortcuts from the outside data.

**Why it helps.** This is the strongest evidence-backed lever we have. A published Parkinson's study
found that pre-training on a large outside movement dataset lifted a small clinical score substantially.
Our model is starved for data, and a good warm start is the classic remedy for that.

**Why it is gated.** We only do this if the Stage R2 verdict is "data-starved." We also first run the
cheapest possible version, using the outside model with no extra training on our data, to check a
falsifier: if training on our small collection makes things *worse* than the outside model alone, then
our data is actively harmful and we stop before building anything elaborate.

**How we would know it worked.** The Parkinson's score should rise versus the frozen R1 result, and the
gap versus the strongest control should shrink or reverse. The predicted size of the gain is a
hypothesis, not a promise, because the outside data is about severity rather than diagnosis and comes
from a different kind of camera.

#### R3.3 Test the real promise of S-JEPA: doing more with fewer labels (priority: medium)

This item deserves emphasis, because it tests what S-JEPA was actually supposed to be good at.

**What we do.** Freeze the trained encoder. Then fit the final classifier using only 25 percent, then
50 percent, then 100 percent of the training labels, and draw a curve of score versus label budget. Do
the same for the Random Forest and the strongest control on the identical label subsets, and compare
the three curves.

**Why it helps.** S-JEPA's whole scientific bet was never "beat the Random Forest when everyone has all
the labels." Its bet is "learn a useful description from unlabeled video, so that when labels are scarce
it needs fewer of them." We have never actually tested that bet on this data. This experiment is the
test. If S-JEPA's curve rises above the Random Forest and the control in the low-label region, that is a
genuine win even if it loses when all labels are present.

**Why it is honest either way.** With no outside warm start, we do not expect S-JEPA to win in the
low-label region on this tiny set, and we say so up front. The value of running it now is that it
forecloses the objection "you only measured the setting that was unfavorable to S-JEPA." The real
version of this test is run *after* the outside pre-training in R3.2, where a low-label win would be the
affirmative result.

**How we would know it worked.** We report the full curves with several seeds and their spread. A win is
S-JEPA crossing above both the Random Forest and the control at the 50 percent label point, with an
error bar that clears zero. Parallel, non-crossing curves would honestly localize the failure to
feature quality rather than label budget.

#### R3.4 Build a proper per-person split, as a careful annotation project (priority: low)

**What we do.** Right now our splits keep videos from the same *source* apart, but a source is a
YouTube video, not a verified person. One source can contain several patients, and one patient may
appear across several sources. To split truly by person, someone has to manually adjudicate the handful
of ambiguous cases and publish a documented registry, erring toward merging when unsure. Only then can
we re-run the comparison on a person-disjoint split.

**Why it helps.** Splitting by person is the honest standard for this kind of study, because it prevents
the same individual from appearing on both sides of a test. The clinical literature shows that loose
splits inflate scores by a meaningful margin.

**Why it is low priority and not a quick job.** The information needed to link two videos to the same
person is not sitting in our files today, so this is real annotation work, not a one-line change. It is
also a matter of honesty rather than a fix for the Parkinson's problem, and with so few videos a
person-disjoint split may force wider error bars. We should do it, but we should not pretend it is fast
or that it will raise the score.

### The rejected ideas, and why we rejected them

These ideas sound reasonable but were dropped after we tried to break them. Each was rejected for a
concrete reason, not a vague one.

- **Re-process to "fix a 4x frame-rate distortion."** There is no 4x distortion. The pipeline already
  brings every clip to a common 15 frames per second when it first reads the video. The only real
  leftover is a small timing error on 24 and 25 frame-per-second clips, which we fold into the R2.3
  rebuild rather than treat as a big separate fix.

- **Add a "this frame is padding" channel and mask it out during training.** We counted the padded
  windows. Only 3 out of 481 windows contain any padding, and none of them are MS videos. Since the
  problem is Parkinson's-versus-MS and MS has no padding, this channel cannot move the Parkinson's
  score. We still record the padding flag as useful bookkeeping, but not as a training mechanism.

- **Mask one side of the body and ask the model to reconstruct the other, to teach left-right
  asymmetry.** This is defeated by how we read out the model. The readout averages the left and right
  joints together, so any left-versus-right contrast the model learns is averaged away before the
  classifier ever sees it. The idea cannot work until the readout is redesigned to keep a left-minus-right
  signal.

- **Various clever masking and pooling changes.** Several proposals reshaped which joints or time spans
  are hidden. They all attack a "the task is too easy" or "we pool the wrong joints" story that the
  evidence contradicts. The model is healthy and already pools a mostly-clinical set of joints, and no
  masking geometry can invent Parkinson's signal that the data erased.

- **Retune the internal training temperatures to fix the confident-wrong problem.** This is a mix-up of
  two different things. The internal temperature shapes a math step deep inside the self-supervised
  training. The "confidently wrong" behavior lives in the final three-way classifier on top. Adjusting
  the former cannot change the latter.

- **Add a skeleton-shape bias to the attention, or predict several steps into the future.** The
  attention bias is switched off at the start and is nudged only by a training signal that already
  rewards the shortcut, so it is most likely inert. The multi-step future target is largely redundant
  with what the predictor already does and is corrupted by the frame-rate issue in the old data. Neither
  attacks the diagnosed cause.

- **Add hand-built gait features into the classifier, or match the folds on filming settings.** The
  hand-built features are essentially the Random Forest baseline again, so this quietly turns the
  "learned" system back into the classic one. Matching folds on filming settings fails because MS is so
  tightly tied to its filming settings that the matched comparison would leave MS with almost nothing to
  score.

- **Split spatial and temporal attention and add multi-scale time tokens.** This is a make-it-bigger
  change whose justification depends on a "the model is too small" verdict that the R2 tests will
  probably reject. It also opens several tuning knobs that, on so few videos, invite accidental
  cheating through the back door.

---

## Part 3. The bottom line

**The single most valuable next step** is to rebuild the skeleton data so it keeps walking speed,
stores the true frame rate, and marks padded frames, and then retrain the exact same model on it. This
removes a verified defect in the input, unlocks every motion-based idea in Stage R3, and can be checked
immediately by retraining. We should pair it with the cheap confidence-interval measurement so that both
the current gap and any new improvement come with honest error bars. We should say plainly, before we
run it, that its likely win is separating healthy from slowed walking, not fixing the
Parkinson's-versus-MS confusion, and that any gain must survive the frame-rate control.

**The rule for deciding whether S-JEPA earns its place on this dataset.** S-JEPA earns its place if
either of two things happens, judged on inner folds and confirmed by touching each outer test fold once.
Either, after the data rebuild and the motion-prediction change, its score clears the old buggy 0.570,
its Parkinson's confidence rises meaningfully, and the gain survives the shortcut controls. Or, after
outside pre-training, the fewer-labels experiment shows its curve crossing above both the Random Forest
and the control when labels are scarce, with an error bar that clears zero. That second outcome would be
S-JEPA delivering its real promise, better use of scarce labels, even if it never beats the Random
Forest when all labels are present.

S-JEPA should be set aside for this dataset if, after the data is confirmed clean and outside
pre-training is added and the diagnostic tests have ruled out a simple size fix, it still trails the
Random Forest and the controls at every label budget with flat, non-crossing curves and no
shortcut-proof improvement in Parkinson's confidence. That outcome would tell us the limit is the
representation itself on so few source videos, and the honest, publishable result is the well-measured
negative with its error bars.

In every case the guardrails hold: the diagnosis label never enters the self-supervised training, the
masking is never biased toward high-motion moments, and every number stays a provisional,
source-grouped development estimate rather than a clinical claim.
