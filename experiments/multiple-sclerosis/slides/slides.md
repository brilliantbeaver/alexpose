---
marp: true
theme: default
paginate: true
size: 16:9
title: S-JEPA for gait classification
---

<!--
This deck uses Marp (https://marp.app). To render:
  npx @marp-team/marp-cli@latest slides.md -o slides.html
  npx @marp-team/marp-cli@latest slides.md --pdf
Or install the "Marp for VS Code" extension and preview it directly.
The images are the same SVGs the notebooks display, in ../images.
-->

# S-JEPA for gait

### Telling normal, MS, and Parkinson's apart from a walking video

A hands-on tutorial series that learns motion without labels, then classifies three
health conditions and compares against a classical baseline.

---

## The question

Given a short video of someone walking, can we tell whether their gait is:

- **normal**,
- affected by **multiple sclerosis (MS)**, or
- affected by **Parkinson's disease (PD)**?

We try two approaches and compare them fairly:

1. a **self-supervised** model, S-JEPA, that learns motion features with no labels, then
2. a **classical Random Forest** on hand-made gait features (the exp5 recipe).

---

## The data, counted honestly

| Condition | Clips | Independent sources |
|---|---|---|
| normal | 19 | 16 |
| ms | 11 | 11 |
| pd | 17 | 8 |
| **total** | **47 usable** | **~35** |

Some clips are pieces of one longer video. That matters: two clips of one walk are not
independent, so we must split by **source**, never by clip.

---

## The pipeline

![w:1000](../images/pipeline_flowchart.svg)

Both approaches share one pose front-end, then split into two branches, then meet again
for a fair comparison on the same test videos.

---

## Step 1: video to skeleton

MediaPipe BlazePose gives **33 landmarks** per frame. Each video becomes an array of
shape `(T, 33, 3)`: frames, joints, and `[x, y, visibility]`.

We normalize each frame by centering on the pelvis and scaling by the torso length, so
the model sees the **shape of the motion**, not where the person stood or how close the
camera was.

No bounding boxes and no annotation files are needed, unlike the GAVD pipeline. We just
run pose on the whole frame.

---

## Step 2: tokens

![w:900](../images/tokenization.svg)

Group `l = 4` adjacent frames of one joint into a token. With a 32-frame window and 33
joints that is `8 x 33 = 264` tokens. Each token also gets a spatial and a temporal
position embedding.

---

## How S-JEPA learns

![w:1000](../images/sjepa_two_lane.svg)

Hide some joints. Predict their **features** (not their coordinates) from the rest. The
answers come from a slow **teacher** encoder, an exponential moving average of the
student. The teacher is what stops the model from collapsing to a constant.

---

## The masking choice

The paper hides the joints that move the **most**. For clinical gait that is backwards,
because the sign is often **less** motion: short steps, stiff knees, reduced arm swing.

So we hide a **fixed set of twelve neurologically relevant joints**: both shoulders and
both complete legs. No other joints. Ever.

![w:420](../images/anatomical_mask.svg)

---

## The masked joints

| Idx | Joint | Idx | Joint |
|---|---|---|---|
| 11 | LEFT_SHOULDER | 12 | RIGHT_SHOULDER |
| 23 | LEFT_HIP | 24 | RIGHT_HIP |
| 25 | LEFT_KNEE | 26 | RIGHT_KNEE |
| 27 | LEFT_ANKLE | 28 | RIGHT_ANKLE |
| 29 | LEFT_HEEL | 30 | RIGHT_HEEL |
| 31 | LEFT_FOOT_INDEX | 32 | RIGHT_FOOT_INDEX |

Twelve joints, chosen once, used every step. This replaces motion-aware masking entirely.

---

## Keeping the features from collapsing

The paper uses three tools: a slow EMA teacher, centering, and sharpening. Remove the
teacher and the model collapses to near chance.

We add a fourth, **VICReg**, clearly labeled as an extension. It has three terms:

- **variance**: keep each feature dimension spread out,
- **invariance**: two views of one window should match,
- **covariance**: decorrelate the dimensions.

We use it to pull the three condition clusters apart.

---

## VICReg, visually

![w:900](../images/vicreg_clusters.svg)

---

## Progressive training

![w:1000](../images/progressive_timeline.svg)

Learn ordinary walking first, then broaden the model's world with MS and PD, then add
VICReg to separate the classes.

---

## Fair evaluation: split by source

![w:950](../images/grouped_split.svg)

All clips of one source stay together. We report **grouped k-fold** with mean and
standard deviation, because a single split of a few dozen videos is too noisy to trust.

---

## Results (grouped k-fold, laptop profile)

![w:640](../images/results_bars.svg)

| Metric | Random Forest | S-JEPA probe |
|---|---|---|
| accuracy | ~0.66 +/- 0.09 | ~0.57 +/- 0.10 |
| macro F1 | ~0.67 +/- 0.10 | ~0.57 +/- 0.11 |

On a few dozen fully labeled videos the classical baseline is strong, exactly as
expected. Numbers move with the profile and the random seed.

---

## What this does and does not show

**Fair.** Both models saw identical, leakage-safe folds and identical metrics.

**Honest.** With ~35 independent sources, a few videos moving between folds swings the
score several points. This is a methodology demonstration, not a clinical result.

**Where S-JEPA earns its keep.** Its features come from unlabeled motion, so its edge
shows up when labels are scarce or when it is pretrained on far more walking than we
have here. The label-efficiency sweep hints at this.

---

## Try it yourself

- Local: `cd experiments/multiple-sclerosis && uv sync && uv run jupyter lab`
- Colab: click the badge at the top of any notebook
- Switch model size with `SJEPA_PROFILE=laptop` or `gpu` in `.env`
- Fast check: `SJEPA_SMOKE=1 uv run python sjepa/tests/test_smoke.py`

Seven notebooks take you from raw video to a fair, reproducible comparison.

**Thank you.**
