# Proposal 2: Future-Innovation Distillation

## The idea in one sentence

Measure which parts of a video model's prediction of the future come specifically from body motion, then teach those parts—and only those parts—to a smaller skeleton model.

## The basic idea

Suppose we want a skeleton model to learn from a large video model.

The video model can see almost everything in a clip: the person's motion, clothing, nearby objects, the background, and camera movement. The skeleton model sees only tracked body points. It would therefore be unfair to ask the skeleton model to copy the video model's entire internal representation. Much of that representation may describe information that is absent from a skeleton.

Instead, this proposal asks a narrower question:

> After accounting for what the current video frame and the recording conditions already tell us, how much additional information about the future is present in the person's recent skeleton motion?

The experiment has three stages:

1. Use current video information and simple recording clues to predict the video model's representation of a future moment.
2. Measure what that prediction misses. This leftover is the **future innovation**.
3. Test whether skeleton history can predict some of that leftover. If it can, train a skeleton-only student to reproduce the predictable part.

This is both a measurement and a distillation experiment. The measurement tells us whether useful motion information exists. Distillation is attempted only if the measurement succeeds.

![Distill only the future innovation](images/02-distillation-mechanism.svg)

## The main concepts, from first principles

### A model representation is a compressed description

A neural network does not need to store a video as raw pixels at every layer. It converts the pixels into arrays of numbers called **representations**, **features**, or **latents**. These numbers summarize patterns that help the model understand or predict the video.

For example, some latent features might respond to body position, walking direction, camera motion, or the layout of the scene. We usually cannot assign a simple name to every number, but the full latent vector acts as a compact description of the clip.

### The teacher and student see different information

The **teacher** is V-JEPA 2.1, a large pretrained model that processes RGB video. RGB means the ordinary red, green, and blue pixel channels in a video. V-JEPA stands for **Video Joint-Embedding Predictive Architecture**. Instead of trying to reproduce every future pixel, it predicts latent features that summarize hidden or future parts of a video.

The **student** processes a skeleton: a time series of tracked body landmarks such as the shoulders, hips, knees, ankles, heels, and toes. A skeleton keeps motion and rough body geometry while discarding most appearance information.

**Knowledge distillation** normally trains a small student to copy a larger teacher. Direct copying is inappropriate here because the teacher and student do not receive the same evidence. A skeleton cannot recover the color of a shirt or the shape of a background object, no matter how good the model is.

### A baseline removes information that is already easy to obtain

Walking is repetitive, and consecutive video frames usually look similar. A model may predict a future video representation fairly well simply by using the current frame, the person's location in the image, or camera motion.

We therefore begin with a strong **baseline predictor**. It receives the current video representation and nuisance variables—recording details that can help prediction but are not the body-motion signal we want to study.

Examples include crop size, camera movement, frame rate, pose confidence, and where the person appears in the image.

The baseline answers:

> How well can we predict the future without using skeleton history?

### Future innovation is the part the baseline misses

The word **innovation** does not mean a new invention here. In time-series analysis, it means new information that was not predicted from the information already supplied to a baseline.

We subtract the baseline's predicted future representation from the teacher's actual future representation. The difference is the future innovation.

- If the baseline prediction is already exact, the innovation is zero.
- If the future contains an unexpected change, the innovation is larger.
- The innovation can still contain several kinds of information, including motion, appearance changes, and noise.

The skeleton model receives credit only for the part of this leftover that skeleton history can predict on previously unseen source videos.

### Why the order matters

The logic is deliberately conditional:

```text
future video representation
        minus
prediction from current video + recording clues
        equals
future innovation

future innovation
        predicted from skeleton history
        equals
skeleton-explainable future information
```

This order prevents an easy but misleading result. Without the baseline, skeletons might appear useful merely because they reveal the person's location, scale, walking phase, or the source video's identity. The experiment asks whether skeletons add something after those easier explanations are already available.

## Research question

The formal two-week question is:

> On GAVD source videos excluded from training, does whole-body skeleton history increase the prediction \(R^2\) for V-JEPA 2.1 future representations by at least 0.05 beyond current RGB features and the full nuisance baseline? Is the real-skeleton gain at least twice the gain from time-shuffled skeletons, while skeletons from mismatched clips provide approximately no gain?

In plainer language, the experiment must satisfy all three conditions:

1. Real skeleton history must make a meaningful improvement.
2. Correct timing must matter.
3. The skeleton must belong to the correct video example.

The primary dataset is GAVD because it contains in-the-wild RGB videos with matching 2D skeleton tracks. GAVD does not provide reliable participant identities, so entire source videos—not assumed individuals—are held out for testing. Presentation labels are never used to train this experiment or compute its headline result.

## Mathematical definition

The notation below describes the same logic precisely.

Let:

- \(v_t\) be the frozen V-JEPA representation of the observed RGB context up to time \(t\);
- \(v_{t+h}\) be the teacher's representation of a future video block \(h\) frames later;
- \(n_t\) be nuisance variables available at time \(t\);
- \(s_{\leq t}\) be all skeleton observations up to time \(t\).

The horizon \(h\) says how far into the future we are trying to predict. The experiment uses horizons of 8, 16, and 32 frames.

### Step 1: predict the future without skeleton history

Fit a baseline function \(g\):

$$
\hat{v}_{t+h}^{\mathrm{base}} = g(v_t, n_t)
$$

The hat means “predicted.” Thus, \(\hat{v}_{t+h}^{\mathrm{base}}\) is the baseline's best prediction of the future teacher representation.

### Step 2: calculate what the baseline missed

$$
e_{t,h} = v_{t+h} - \hat{v}_{t+h}^{\mathrm{base}}
$$

The residual \(e_{t,h}\) is the future innovation. It is a vector because the teacher representation contains many features.

### Step 3: test what skeleton history adds

Fit a small predictor \(q\) for the remaining error:

$$
\hat{e}_{t,h} = q(s_{\leq t}, v_t, n_t)
$$

Add that predicted correction to the baseline:

$$
\hat{v}_{t+h}^{\mathrm{full}}
= \hat{v}_{t+h}^{\mathrm{base}} + \hat{e}_{t,h}
$$

The full system differs from the baseline by its residual-prediction head and skeleton history. The same head is therefore tested with correctly paired, shuffled, and mismatched skeletons. The extra gain from the correctly paired skeleton—beyond these matched controls—is the evidence for skeleton-specific information.

Let \(R^2_{\mathrm{base}}\) be the held-out \(R^2\) of the baseline and \(R^2_{\mathrm{full}}\) be the held-out \(R^2\) after skeleton history is added. Here, \(R^2\) measures how much variation in the true future representation a predictor explains. Higher is better; zero means no better than always predicting the training mean, and a negative value means worse than that simple mean prediction. Compute it for each of the 256 projected features, use a prespecified aggregate for the headline score, and also show the full feature-level distribution.

The raw skeleton gain is:

$$
\Delta R^2_h = R^2_{\mathrm{full}} - R^2_{\mathrm{base}}
$$

The headline threshold is \(\Delta R^2_h \geq 0.05\) at one or more horizons.

We also report the **skeleton-explainable fraction**:

$$
F_h = \frac{R^2_{\mathrm{full}} - R^2_{\mathrm{base}}}
{1 - R^2_{\mathrm{base}}}
$$

The denominator is the fraction of future variation that the baseline did not explain. Therefore, \(F_h\) asks what fraction of the baseline's remaining error is recovered after skeleton history is added.

For example, suppose the baseline has \(R^2=0.60\) and the full predictor has \(R^2=0.66\). Then:

$$
\Delta R^2 = 0.66 - 0.60 = 0.06
$$

and

$$
F = \frac{0.06}{1-0.60} = 0.15
$$

Skeleton history added 0.06 \(R^2\) and explained 15% of what the baseline had missed.

- \(F_h=0\): skeleton history adds nothing.
- \(F_h=1\): skeleton history explains everything the baseline missed.
- \(F_h<0\): adding skeleton history makes prediction worse.

Negative values are reported rather than clipped to zero because they reveal failed or harmful distillation.

## Experimental method

### 1. Lock one public video teacher

Use the official [V-JEPA 2.1 ViT-B checkpoint](https://github.com/facebookresearch/vjepa2) with its documented 384-pixel, 64-frame preprocessing. ViT-B is the base-size version of a Vision Transformer, a neural network that processes a video as a collection of space-time patches. Freeze its weights, meaning that the teacher itself is never updated during this experiment. Record the exact code commit, checkpoint hash, frame-sampling rule, preprocessing steps, and output layer so the experiment can be reproduced.

ViT-B is chosen because it is fast enough for the two-week study. A larger teacher is tested only later as a sensitivity analysis if the main mechanism works.

The prediction must be **causal**: information after time \(t\) cannot enter any input used to predict time \(t+h\). A causal mask hides future video blocks from the context pathway while retaining them as targets. Synthetic tests will alter future pixels and verify that the context representation does not change. If the official predictor cannot provide a clean past-to-future setup, frozen future encoder features will remain the targets and a separate equal-capacity predictor will be trained. That limitation will be reported explicitly.

### 2. Construct training and test clips without source leakage

Use only GAVD videos already cached on HAIC. Assign each entire source video to one data split before extracting 64-frame windows. Windows from one source must never appear across both training and test sets. This matters because nearby windows from the same uploaded video can share a person, background, camera, compression pattern, and pose-tracking errors.

Use all valid gait, exercise, and style clips because future prediction does not require presentation labels. Sample source videos with equal weight so a source containing many windows cannot dominate the result.

The main skeleton input uses every reliably observed MediaPipe body landmark. Each landmark includes an explicit validity or missingness indicator. A missing joint is not silently treated as a real coordinate of zero.

Also test **Core11**, the repository's smaller skeleton made of the pelvis plus left and right hip, knee, ankle, heel, and forefoot points. Comparing whole-body input with Core11 tests whether arms and trunk provide useful predictive information.

### 3. Build a deliberately strong baseline

The baseline \(g\) receives:

- current V-JEPA context features;
- clip duration and current temporal position;
- summaries of the first and last visible frames;
- the person's bounding-box position, size, scale, and area;
- centroid velocity and estimated camera motion;
- pose confidence and missingness;
- source resolution, frame rate, and estimated camera view;
- a static-background embedding computed outside the person's box.

These inputs cover easy explanations such as appearance, timing, framing, and tracking quality. The baseline is intentionally difficult to beat. The proposal is interesting only if skeleton dynamics add information after these shortcuts are controlled.

### 4. Train small, comparable prediction heads

Freeze both V-JEPA 2.1 and the local S-JEPA encoder. S-JEPA applies the same general JEPA idea to skeleton motion: it converts a sequence of body landmarks into latent features and learns relationships among hidden joint-time regions. Train only small “heads” that map frozen inputs to the future-video target:

1. a linear baseline using current RGB features and nuisance variables;
2. a two-layer temporal head using raw skeleton history in addition to the baseline inputs;
3. the same temporal head using frozen S-JEPA features instead of raw skeleton coordinates;
4. optionally, a rank-8 adapter on the S-JEPA predictor, but only if frozen S-JEPA passes the 48-hour gate.

A head is a small prediction module added after a frozen model. Using the same head size for raw skeletons and S-JEPA features makes their comparison fair. A **rank-8 adapter** is a small, low-cost update with a restricted number of trainable directions; it allows limited adaptation without retraining the full encoder.

V-JEPA produces many person-region features. Before fitting the heads, project those features into 256 dimensions with one fixed random orthogonal projection. This makes training cheaper while approximately preserving distances and relationships between examples. Generate the projection once before data splitting and never tune it on test results.

Run every model at 8-, 16-, and 32-frame horizons. A method should produce a sensible curve across time rather than succeed only at a conveniently selected horizon.

### 5. Train a skeleton-only student only after the measurement passes

The conditional experiment above may use RGB and nuisance controls because its purpose is to measure the unique value of skeleton history. Deployment is a separate stage.

If \(F_h\) is reliably positive, freeze the measured innovation targets \(e_{t,h}\) and train a new student that receives skeleton history alone. This student attempts to predict only the skeleton-explainable future-innovation code; it does not copy the teacher's complete video representation.

Test the student with **future retrieval**. Give it many possible future teacher codes from a held-out batch and ask whether its predicted code selects the correct future. Include difficult distractors matched by walking phase and, where the data permit, by source or person. Retrieval checks whether the predicted representation identifies the correct future, rather than merely achieving a small average improvement across latent dimensions.

## The 48-hour decision gate

For the complete setup, data contracts, code scaffolds, controls, metrics, artifact layout, and stop/advance logic, follow the [Experiment 0 guide](../../future-innovation-distillation/experiment-0-guide.md).

Begin with 50 clips from source-separated GAVD development folds. Cache the frozen teacher features once. At the 8-frame horizon, train the nuisance baseline and the raw-skeleton head.

Continue to the full experiment only if all of the following hold:

- real skeleton history adds at least 0.05 held-source \(R^2\) beyond current RGB and nuisance inputs;
- its gain is at least twice the gain from time-shuffled skeleton history;
- a skeleton taken from the wrong example gives approximately no positive gain;
- removing the person's image-region tokens from the teacher sharply reduces the effect;
- changing future motion changes the target, while replacing only the static background does not materially change it.

Stop before training adapters if teacher inference is unstable, future information leaks into the context, or these checks fail. This early gate prevents spending most of the compute on a signal that is absent or caused by a shortcut.

## Full-experiment success criteria

![The sufficiency curve must beat its placebos](images/02-distillation-gates.svg)

| Question | How it is tested | Result required to pass |
| --- | --- | --- |
| Does skeleton history add information? | Held-source \(\Delta R^2\) beyond current RGB and nuisances | At least 0.05 at one horizon and positive at two horizons. |
| Does correct motion matter? | Compare real skeletons with time-shuffled and clip-mismatched skeletons | Real gain is at least twice the shuffled gain; mismatched gain is near zero. |
| Did S-JEPA learn something beyond raw coordinates? | Same-size head on S-JEPA features versus raw skeleton history | The source-bootstrap interval for S-JEPA's \(\Delta R^2\) over raw skeletons is entirely positive. |
| Is the distilled code useful? | Retrieve the correct future among phase-matched distractors | At least 10 percentage points better than the raw-skeleton student. |
| Does the result survive difficult videos? | Repeat on unseen sources and within pose-quality groups | Positive gain in at least three of four pose-quality quartiles. |
| Does the upper body help? | Whole body versus Core11, followed by an upper-body time shuffle | Whole body improves prediction, and the improvement disappears when upper-body timing is destroyed. |

The primary figure is the complete \(F_h\) curve across all horizons, not only the best point. Also report results for individual projected teacher features so the average cannot hide a large effect in only a few unusually variable dimensions.

For uncertainty estimates, resample whole source videos rather than individual windows. This **source bootstrap** respects the fact that windows from the same source are related.

## Controls and what each one rules out

A control is an alternative explanation that the proposed method must beat.

### Can appearance or recording details explain the result?

- current RGB features alone;
- nuisance variables alone and combined with RGB;
- duration-, centroid-, crop-, foreground-, background-, and camera-only models.

If these models match the skeleton model, the result does not require body dynamics.

### Can ordinary coordinates explain the result without S-JEPA?

- raw 2D coordinates, velocity, acceleration, and confidence;
- Core11, whole-body, and validity-only inputs;
- an equal-compute, equal-parameter head on raw skeleton history;
- a randomly initialized S-JEPA encoder with the same head.

If raw coordinates match trained S-JEPA, skeleton motion may still carry real information, but the learned S-JEPA representation has added nothing.

### Is the model exploiting timing, identity, or source shortcuts?

- reverse, shuffle, or phase-shuffle skeleton time;
- pair the video with a skeleton from a different clip or source;
- use a different skeleton matched on source, walking speed, and phase;
- change the teacher layer or use a different fixed random teacher projection.

GAVD does not provide participant IDs, so the main mismatch control is defined at the clip or source level. Call it a person mismatch only in subsets where different identities can be verified. If shuffled or mismatched skeletons work, the experiment has probably found source or phase leakage rather than useful motion prediction.

### Can a simple motion rule predict the future just as well?

- future mean;
- persistence, which copies the current state forward;
- a periodic walking template;
- linear autoregression, which extrapolates from recent values.

These baselines test whether the learned model beats straightforward consequences of smooth, repetitive walking.

All learned comparisons use matched parameter counts and compute budgets. This prevents a larger model from winning simply because it has more capacity.

## Two-week schedule and compute limit

- Days 1–2: load and verify V-JEPA 2.1, cache features for the 50-clip gate, and run leakage tests.
- Days 3–5: cache the full source-held feature set; fit the baseline, raw-skeleton head, and S-JEPA head.
- Days 6–8: measure horizon curves, body-region effects, and shuffle controls.
- Days 9–10: train the optional rank-8 adapter only if frozen S-JEPA passes.
- Days 11–12: test future retrieval and stress-test low-quality pose tracks.
- Days 13–14: run three random seeds, source-bootstrap uncertainty, null-result analysis, and final figures.

Teacher inference is expected to be the largest GPU cost, so each teacher feature is computed once and cached. Limit teacher inference to 2,000 clips and all trainable work to 20 H100-hours. One H100-hour means using one NVIDIA H100 GPU for one hour.

## What is new—and what is not

[V-JEPA 2](https://arxiv.org/abs/2506.09985) already performs action anticipation. [Human-JEPA](https://arxiv.org/abs/2608.21160) already studies human-centered forecasting. Knowledge distillation between different data types is also an established idea. This proposal does not claim any of those broad ideas as new.

The proposed contribution is a specific measurement: the **conditional skeleton-explainable fraction of future video innovation**. It asks how much skeleton history adds after current RGB information and recording shortcuts have been removed. Matched time and clip shuffles test whether that added information is genuine. A skeleton-only student is trained only when the measured fraction is nonzero.

This also differs from two earlier repository proposals:

- **Past-Only Predictive Surplus** asks whether skeleton history predicts future skeleton features better than simple periodic-motion models.
- **Cycle-to-Cycle Innovation Map** locates changes between visible gait cycles.
- **Future-Innovation Distillation** instead predicts a large frozen video model's future features and removes what current RGB information already explains.

## How to interpret each possible outcome

### If skeleton history helps and S-JEPA beats raw coordinates

The result would show that a learned skeleton representation captures part of a video model's future that is not explained by current appearance or obvious recording clues. Training a compact skeleton-only student would then be justified.

### If skeleton history helps but S-JEPA does not beat raw coordinates

The cross-modal scientific result would still be real: body kinematics carry some future-video information. However, the local S-JEPA representation would not deserve credit because ordinary coordinates work just as well.

### If shuffled or mismatched skeletons also help

The apparent improvement is probably caused by timing, source, phase, or identity leakage. The main claim fails even if the raw \(R^2\) is high.

### If skeleton history does not help

This is a useful negative result. It would place an empirical ceiling on skeleton-only imitation of this teacher target. The chosen future-video representation may be dominated by appearance, objects, camera changes, noise, or other information unavailable from 2D body kinematics. Direct teacher-to-skeleton distillation would then be poorly posed.

No outcome is a diagnosis, a clinical forecast, or evidence that skeleton motion causes changes in the video representation. The study measures held-source future-latent predictability only.
