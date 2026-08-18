# GaitParity glossary

Every technical word used anywhere in this folder, explained once, in plain language, with a small example. If a sentence in another document loses you, the word is probably here.

Terms are grouped by where they come from. Inside each group they run roughly from most basic to most specialized, not alphabetically, so you can read a group top to bottom as a short lesson.

---

## Group 1: walking and the human body

**Gait**
The pattern of how a person walks. Not just speed. It includes how long each foot stays on the ground, how far each step reaches, how the hips rotate, and how hard each leg pushes.

**Gait cycle**
One complete repetition of walking, measured from one event on one leg to the next occurrence of the same event on the same leg. The usual choice is right heel strike to the next right heel strike. Everything that happens in between (the right foot rolls forward and pushes off, the right leg swings through the air, the left leg does its own version half a beat later) is one cycle. A person walking for 30 seconds gives you roughly 25 to 30 cycles.

**Stride and step**
A **stride** is one full gait cycle: right heel strike to right heel strike. A **step** is half of that: right heel strike to left heel strike. So one stride contains two steps, one from each leg.

**Stance phase**
The part of the cycle when a given foot is touching the ground. In typical adult walking this is about 60 percent of the cycle for each leg.

**Swing phase**
The part of the cycle when that same foot is in the air, about 40 percent. Because stance is longer than swing, there are two short windows in every cycle when *both* feet are on the ground at once. Those windows are called double support, and they get longer when someone walks slowly or cautiously.

**Foot contact events**
The two moments that bookend stance. **Heel strike** (also called initial contact) is when the foot lands. **Toe off** is when the foot leaves the ground. Detecting these two moments accurately is a prerequisite for almost every measurement in this project, because they define where one cycle ends and the next begins.

**Ground reaction force**
When you push on the ground, the ground pushes back with equal size and opposite direction. That push-back is the ground reaction force. It has three components: vertical (holding you up), fore-aft or anterior-posterior (slowing you down early in stance, driving you forward late in stance), and side-to-side (small).

**Force plate**
A rigid platform, usually built flush into a laboratory walkway, with sensors underneath that measure the ground reaction force hundreds or thousands of times per second. When someone steps cleanly onto one plate with one foot, you get that foot's force over the whole stance phase.

**Impulse**
Force accumulated over time. Mathematically it is the area under a force-versus-time curve. If a leg pushes with 60 newtons for 0.2 seconds, the impulse is about 12 newton-seconds. Impulse matters more than peak force for how much the body actually speeds up, because a small force applied for a long time can change your motion as much as a big force applied briefly.

**Propulsion / propulsive impulse**
The forward-driving part specifically. Late in stance, the fore-aft ground reaction force turns positive (pointing forward), and integrating only that positive portion gives the propulsive impulse for that leg. This is the number that tells you how much a given leg is actually contributing to moving the person forward. It is the central physical quantity in this project.

**Anterior, posterior, mediolateral, vertical**
Direction words used because "front" and "left" get ambiguous once you rotate a coordinate system. **Anterior** means toward the front of the body, **posterior** toward the back, **vertical** up and down, and **mediolateral** side to side (from the midline outward toward either side). "Mediolateral coordinate" just means "the left-right axis."

**Midline**
The imaginary vertical plane splitting the body into left and right halves. Joints that sit on it (pelvis center, spine, head) are midline joints. They do not have a left twin and a right twin.

**Bilateral**
Having a left version and a right version. Knees are bilateral. The pelvis center is not.

**Asymmetry**
Any measurable difference between what the left side does and what the right side does. Everyone has a little. The question is how much, in which direction, and whether it means something.

**Laterality**
Side-ness. Which of the two body sides something belongs to, or favours. "Signed laterality" is therefore just a right-minus-left number with its direction preserved.

**Parity**
Which of the two mirror categories a quantity belongs to: even or odd. The project is named GaitParity because sorting gait quantities into those two categories, and keeping them sorted, is what it is about. "Parity-aware" means the system knows and respects that distinction.

**Kinematics versus kinetics**
**Kinematics** describes motion: positions, angles, speeds, accelerations. **Kinetics** describes the forces that caused the motion. This project feeds kinematics in (a skeleton) and predicts kinetics out (a force ratio). Mixing the two up is exactly how force information leaks into the input, which would invalidate the whole result.

**Contact**
One footfall landing cleanly on a single force plate with a single foot. Partial contacts, and trials where two feet share a plate, are excluded because you cannot attribute the measured force to one leg.

**EMG (electromyography)**
Electrical activity recorded from muscles. Present in some datasets. Never fed to the encoder here, for the same reason force is not.

**Stroke**
A brain injury caused by interrupted blood supply. Because most motor pathways cross over, damage on one side of the brain usually weakens the opposite side of the body. That is why stroke is the central clinical condition here: it produces gait asymmetry with a known, documented side.

**Paretic side**
The weaker side after a stroke. "Paresis" means partial weakness. A clinician documents which side is paretic during examination. Important subtlety: the paretic side and the side that shows a smaller measured push are *usually* the same but not *always*, because people compensate. A person may push harder with the weak leg in a way that looks stronger on one measure while being clearly impaired on others.

**Compensation**
Any change in movement that a person adopts to work around an impairment. Someone with a weak right leg might swing the whole hip to advance it, lean toward the left, or spend more time on the left foot. Compensation is why a single number can point the wrong way even when the underlying impairment is real.

**Parkinson's disease, cerebral palsy, myopathy**
Three other movement conditions that appear in the datasets. Parkinson's disease affects movement initiation and scaling and often produces shuffling and freezing. Cerebral palsy is a developmental motor condition from early brain injury. Myopathy is muscle disease. They are listed together only because the datasets label them, not because they form one severity scale.

**Motion capture**
Recording 3D movement precisely, usually with reflective markers on the body tracked by many synchronized infrared cameras. Accurate but requires a laboratory.

**Pose estimation**
Getting joint positions out of ordinary video using a computer vision model, with no markers and no lab. Far more convenient, considerably noisier. This project uses BlazePose, which returns 33 body landmarks per frame plus a visibility score for each.

---

## Group 2: geometry, mirrors, and symmetry

**Operator**
A function that takes a whole object and returns another object of the same kind. "Reflect this skeleton sequence" is an operator: sequence in, sequence out. Written as $M$ in these documents, so $Mx$ means "the reflected version of $x$."

**Reflection / mirror**
Flipping something across a plane. In this project the relevant mirror is *anatomical*: it exchanges left and right body parts. Doing it correctly means three things at once, and skipping any one of them breaks the whole study. See the reflection operator section in [METHODOLOGY.md](./METHODOLOGY.md).

**Involution**
An operation that undoes itself when applied twice. Reflection is one: $M(Mx) = x$. Mirror a mirror and you are back where you started. This is the single easiest property to unit-test, which is why it is the first check in the pipeline.

**Even quantity (under reflection)**
Something that does not change when you mirror the body. Walking speed is even: mirror the person and they are still walking at 1.2 metres per second. Formally, $f(Mx) = f(x)$.

**Odd quantity (under reflection)**
Something that keeps the same size but flips sign when you mirror the body. Right-minus-left propulsion is odd: if it was $+3$ before mirroring it is $-3$ after. Formally, $f(Mx) = -f(x)$.

Note that "even" and "odd" here have nothing to do with even and odd integers. The words are borrowed from function terminology, where $x^2$ is an even function ($f(-x) = f(x)$) and $x^3$ is an odd function ($f(-x) = -f(x)$). Same idea, applied to mirroring a body instead of negating a number.

**Group (in the math sense)**
A set of operations that can be composed, that includes a "do nothing" operation, and where every operation can be undone. The group here is tiny: exactly two elements, "leave the body alone" and "mirror the body." It is written $C_2$. You never need general group theory to follow this project; you need the fact that this particular group has two elements and mirroring twice returns you to the start.

**Invariance**
A quantity is invariant under an operation if the operation does not change it. Walking speed is invariant under reflection. In the language above, invariant and even mean the same thing.

**Equivariance**
Weaker than invariance and more useful. A quantity is equivariant if the operation changes it *in a known, predictable way* rather than not at all. Odd quantities are equivariant: mirroring flips the sign, reliably, every time.

An analogy. Suppose you photograph a room and then rotate the camera 90 degrees. An *invariant* measurement would be "how many chairs are in the room," which does not care about the rotation. An *equivariant* measurement would be "which direction is the door," which does change, but changes by exactly 90 degrees, so you can always predict the new answer from the old one. Equivariance means the structure is preserved, not erased.

**Commutation**
Two operations commute when doing them in either order gives the same result. Putting on socks and putting on a hat commute. Putting on socks and putting on shoes do not. Testing equivariance layer by layer is exactly testing whether "mirror, then process" gives the same answer as "process, then mirror."

**Yaw rotation**
Spinning around the vertical axis, like a figure on a turntable. A yaw rotation changes which way the person faces in the coordinate system. It does *not* change which leg is anatomically the left leg. This distinction is easy to state and easy to violate in code, which is why the documents keep separating "reflection" from "rotation." (Yaw is aviation vocabulary; the other two axes are pitch, nodding forward and back, and roll, tipping side to side. Only yaw matters here.)

**Orbit**
Everything you can reach by applying the available operations to a starting point. Since the only operations here are "do nothing" and "mirror," the orbit of a walk $x$ is exactly the pair $\{x, Mx\}$. Two items, no more, ever. That finiteness is what makes the whole construction tractable.

**Swap operator $S$**
The operation that exchanges the two internal branches (as-recorded and mirrored). Distinct from $M$: $M$ mirrors an actual skeleton, while $S$ leaves the numbers alone and merely trades which slot they sit in. Keeping these two straight is essential when reading the equivariance condition.

**Body frame / canonical frame**
A coordinate system defined relative to the person rather than the room: origin at the pelvis, one axis along the direction of walking, one axis vertical, one axis side to side. Converting every recording into a body frame is what lets you compare a person filmed from the left with a person filmed from behind. Without it, "the x coordinate" means something different in every recording.

**Right-handed axes**
A convention for which way the third axis points once you have fixed two. Point your right index finger along the first axis and your middle finger along the second; your thumb points along the third. Getting handedness wrong silently mirrors your whole dataset, which is exactly the failure mode this project is most exposed to.

---

## Group 3: machine learning

**Model**
Any function with adjustable numbers inside it that gets tuned to fit data.

**Parameters and weights**
The adjustable numbers. Training means searching for values of them that make predictions better.

**Layer**
A neural network is a stack of processing stages, each transforming the output of the one before. Each stage is a layer. A typical encoder here has around 12 of them. This matters enormously in the full study: "equivariant at every layer" means the mirror rule is checked at each of those 12 stages, not just at the exit, and a single failing layer breaks the claim.

**Width and depth**
**Width** is how many numbers are in the representation (say 256). **Depth** is how many layers (say 12). "Effective depth" comes up because the equivariant model carries two branches through the same 12 layers; that counts as 12, not 24, and saying so explicitly matters because it affects how models get matched for fairness.

**World model**
A model that learns how bodies move in general, before anyone asks it a specific question. The term appears in the full study's title. Everything it means for this project is covered by "a JEPA pretrained on lots of unlabelled movement."

**Representation (also: features, embedding, latent)**
A compact list of numbers that summarizes an input. Instead of 64 frames times 33 joints times 3 coordinates (over 6,000 raw numbers), a representation might be 256 numbers that capture the useful structure. A good representation makes downstream questions easy to answer with a simple model.

**Encoder**
The part of the system that turns a raw input into a representation. Written $E$, so $E(x)$ is the representation of input $x$. In this project the encoder eats a skeleton sequence and produces a feature vector.

**Readout (also: head, probe)**
A small model that turns a representation into an answer. Written $q$, so $q(E(x))$ is a prediction. Readouts are deliberately kept simple, often a single linear layer, because the point is to test what the *encoder* already knows, not to see whether a large enough add-on model can eventually dig anything out.

**Frozen**
Fixed and not being trained. "Frozen encoder" means the encoder's parameters are locked while only the readout is fitted. This makes comparisons clean: if two readouts differ in performance, that difference comes from the readouts, because the representation feeding both was identical.

**Supervised learning**
Training with labelled examples. You show the model an input and the correct answer, repeatedly.

**Self-supervised learning**
Training without human labels, by hiding part of the input and asking the model to recover it from the rest. No one has to annotate anything; the data supervises itself. This matters enormously in clinical work, where unlabelled movement recordings are plentiful and expert-labelled ones are scarce.

**JEPA (Joint Embedding Predictive Architecture)**
A specific style of self-supervised learning. Hide part of the input, then predict the hidden part's *representation* rather than its raw values. The distinction is the whole point: predicting raw pixels or raw coordinates forces the model to waste capacity on unpredictable detail (exact noise, exact texture), whereas predicting a representation lets it focus on structure that is actually predictable. A skeleton JEPA hides some joints at some time steps and predicts what the hidden joints' description should look like.

**Latent**
Anything internal to the model rather than observed in the world. "Predicting in latent space" means the prediction target is the model's own internal description.

**Target encoder / EMA**
Many JEPAs run two copies of the encoder. One (the online or view encoder) is trained normally. The other (the target encoder) supplies the prediction targets and is updated only as a slow running average of the first. EMA stands for exponential moving average: the target's weights drift toward the online weights a small fraction at a time. The slow copy stops the system from cheating its way to a trivial solution.

**Collapse**
The failure mode where a self-supervised model discovers it can satisfy its training objective by outputting nearly the same representation for every input. Loss goes down, the representation becomes useless. Guarding against collapse is why extra terms like VICReg get added to the loss.

**Masking**
Deliberately hiding part of the input. The mask records which parts were hidden.

**Mask (as bookkeeping)**
Separately, a mask can record which values were genuinely *missing* in the data, for example a joint the pose estimator could not see. Both meanings appear in these documents. Context distinguishes them: a training mask is chosen, a missingness mask is observed.

**Token**
One unit the model processes. Here, one joint at one small window of time. With 33 joints and 16 time positions there are 528 possible tokens per sequence.

**Inductive bias**
Any assumption built into a model's structure rather than learned from data. Forcing a prediction to flip sign under mirroring is an inductive bias. Good inductive biases help most when data is scarce, because they supply for free what the model would otherwise have to infer from examples. Bad ones cost you, because they forbid solutions that were actually correct. Which kind this one is happens to be the research question.

**Augmentation**
Creating extra training examples by transforming existing ones. Feeding a model both a walk and its mirror image, with the label sign flipped, is sign augmentation. It *encourages* the right behaviour without *guaranteeing* it.

**Baseline**
A simpler alternative you must beat before claiming your method works.

**Control**
A deliberately handicapped variant used to isolate a cause. If you want to know whether a second look at the data helped or whether the *rule* helped, you build a control that gets the second look but no rule.

**Ablation**
Removing one component to see how much it mattered.

**Shortcut learning**
When a model gets good scores by exploiting something irrelevant that happens to correlate with the answer. A classic example: an image model that "detects" pneumonia by recognizing which hospital's scanner took the X-ray, because sicker patients were scanned at a particular hospital. Shortcut learning is not rare and not obvious; it is the default outcome unless you actively test for it.

**Nuisance variable**
Something that varies across your data and could power a shortcut, but is not what you meant to study: recording device, camera angle, frame rate, how much data was missing, which folder a file came from.

**Hyperparameter**
A setting chosen before training rather than learned during it: learning rate, model width, regularization strength. Tuning these on data you later test on is one of the most common ways to fool yourself.

**Regularization**
Any penalty that discourages a model from fitting its training data too closely, trading a little training accuracy for better performance on new data.

**Overfitting**
Learning the specific training examples, including their noise, instead of the general pattern. Detectable only by testing on data the model has never touched.

**Generalization**
Whether performance holds up on genuinely new data. The only performance that matters.

**Seed**
The number that initializes the random choices in a training run. Different seeds give different results from identical setups. Running five seeds tells you how much your result wobbles due to randomness. It does *not* give you five independent experiments, and treating seeds as sample size is a serious error.

**Checkpoint**
A saved snapshot of a model's parameters at some point in training.

**Mixed precision**
Doing some arithmetic in lower-precision number formats to run faster. **float32** is the standard 32-bit format, accurate to about 7 significant digits; training often uses a 16-bit format instead, accurate to about 3. An equality that holds cleanly at 7 digits can fail at 3, so equivariance tests have to be run in the precision actually used for training.

**Stochastic depth and dropout**
Two training tricks that deliberately introduce randomness: dropout zeroes out random features, stochastic depth skips whole layers at random. Both make models generalize better. Both are a hazard here, because if the as-recorded branch and the mirrored branch roll different random numbers, they stop matching and the equivariance guarantee quietly breaks.

**Window**
A fixed-length chunk cut out of a longer recording. Here, 64 frames, about two seconds. The order of operations matters: split participants into train and test *first*, then cut windows. Cutting first puts two windows from one person on opposite sides of the split.

**Effective rank**
Roughly, how many genuinely independent directions a set of features actually uses. If all 256 features move in lockstep, the effective rank is about 1 and you really have one feature wearing 256 hats. This is the primary number for detecting collapse.

**Energy**
Average squared size. An odd channel with near-zero energy is the all-zeros failure case: perfectly odd, perfectly useless.

**Decodability**
Whether a simple readout can pull a given quantity out of a representation at all. A yes-or-no about whether the information is present and accessible. It says nothing about whether the geometry is clean, and those two properties genuinely come apart.

**Imputation**
Filling in values that were missing, for example estimating an unseen heel position. The filling-in rule must be fitted on training data only. Fit it across everything and you have used test data to build your inputs.

**Phase bin**
One slice of the gait cycle. Splitting the cycle into eight bins (0 to 12.5 percent, 12.5 to 25 percent, and so on) and summarizing within each preserves *when* in the cycle something happened. A single average over the whole cycle throws that away, which often discards the informative part.

---

## Group 4: evaluation, splits, and statistics

**Training set, validation set, test set**
Data used to fit the model, data used to choose settings, and data used exactly once to report performance. Keeping the third genuinely untouched is the entire basis for believing a reported number.

**Held-out**
Not used for anything except final measurement.

**Leakage**
Any path by which information about the test data reaches the model before final evaluation. It can be blatant (test rows in the training set) or subtle (computing a normalization constant over all data before splitting; choosing a preprocessing step because it improved test scores). Leakage reliably makes results look better than they are, and it is the single most common reason machine-learning-in-science findings fail to replicate.

**Participant-level split (one person, one vote)**
Splitting data so that everything from one person lands entirely on one side of the split. If someone contributes 30 gait cycles and you split cycles at random, roughly 24 of their cycles train the model and 6 test it, so the model has effectively already met that person. Your test score then measures "can it recognize this person again," not "does it work on someone new."

**Group / grouping variable**
The thing you must not split across. Here it is the participant. In the GAVD dataset, participant identities are not available, so the source video is used instead: multiple clips cut from one YouTube video are not independent observations.

**Cohort**
A specific named group of people recorded under one protocol. For example, the 50 stroke survivors from one published study. "A second cohort" means a different group of people, recorded elsewhere, under a different protocol, which is what makes replication meaningful.

**Cross-validation and folds**
Split into $k$ groups, train on $k-1$ and test on the held-out one, rotate, repeat. Everyone gets one turn as test data. With 40 participants and 5 folds, each fold holds out 8 people.

**Out-of-fold prediction**
The prediction for a data point made by the fold in which that point was held out, so it was never used to fit the model producing it.

**Nested cross-validation (inner and outer folds)**
Two levels. The outer loop reserves test participants. Inside the remaining training participants, an inner loop chooses hyperparameters. This keeps setting selection from ever seeing the outer test data. Without nesting, hyperparameter tuning quietly leaks test information.

**Transductive**
A setting where the model has already seen the evaluation inputs (without their labels) during training. Results from a transductive setup can be useful for checking that code works, but they cannot tell you how the system behaves on genuinely new people. Several GAVD results in this project are transductive, and every document says so explicitly.

**MAE (mean absolute error)**
Average size of the mistakes, ignoring their direction. Predicting 0.4, 0.9, and 0.2 when the truth is 0.5, 0.5, and 0.5 gives errors of 0.1, 0.4, and 0.3, so MAE is 0.267. Reported in the target's own units, which makes it easy to interpret.

**$R^2$ (coefficient of determination)**
How much of the spread in the true values your model explains, compared to just predicting the average every time. $R^2 = 1$ is perfect, $R^2 = 0$ means you did no better than guessing the mean. **Untruncated** $R^2$ means negative values are reported honestly rather than clipped to zero. Negative $R^2$ happens on held-out data and means the model is worse than the constant guess, which is genuinely useful information.

**Calibration slope and bias**
Plot predictions on the horizontal axis and truths on the vertical, then fit a line. A slope of 1 and an intercept of 0 mean predictions are on the right scale. A slope of 0.5 means the model systematically under-reacts, moving half as much as it should. Bias is a constant offset in one direction. Two models with the same MAE can differ sharply in calibration, which is why both get reported.

**Label budget**
How many labelled examples the model is allowed. Here it is counted in *people*, not recordings.

**Learning curve**
Held-out error plotted against label budget. It answers "how much does performance improve as I label more people," which is often more decision-relevant than a single accuracy number.

**AULC (area under the learning curve)**
One number summarizing a whole learning curve: the area underneath it. Lower is better, since the curve plots error. A model that is already decent with 4 labelled people and stays decent has a small area. A model that only works once you have 40 people has a large one.

**Bootstrap**
A way to estimate uncertainty by resampling your data with replacement, many times, and watching how much the answer moves. If you have 40 participants, draw 40 at random with replacement (some appear twice, some not at all), recompute, and repeat a few thousand times. The spread of those answers estimates how much your result depends on which particular people you happened to recruit. Crucially, you resample **participants**, not cycles: resampling cycles would pretend you had far more independent information than you do.

**Confidence interval**
A range of values consistent with your data. Wide interval means you cannot tell. A wide interval that happens to be centred on a favourable number is not a positive result.

**Effect size**
How big the difference is, not just whether it is detectable. A tiny difference measured very precisely can be statistically significant and practically meaningless.

**Permutation test**
Deliberately scrambling the labels so that any real relationship is destroyed, then rerunning the whole pipeline. Performance should collapse to chance. If it does not, there is leakage or a bug. This is a sanity check on your evaluation, not a test of your model.

**Preregistration / freezing**
Writing down the analysis plan before looking at results, so that you cannot quietly adjust the question until the answer is nice. In this project the frozen items include folds, targets, model names, primary comparison, and the table mapping possible outcomes to allowed conclusions.

**Confirmatory versus exploratory**
**Confirmatory** means you committed to the exact test in advance, so the result counts as evidence. **Exploratory** means you were looking around, so the result generates hypotheses and does not count as evidence. The same number means different things depending on which one it was, which is why the distinction gets stated explicitly rather than assumed.

**Statistical power**
The chance a study would detect a real effect of a given size, if one exists. A study can fail to find an effect either because there is none or because it was too small to detect one; power is what distinguishes those.

**Sensitivity analysis**
Redoing the analysis a different reasonable way and checking whether the conclusion survives. If the answer changes when you make an arbitrary choice differently, the answer was about the choice.

**Multiple comparisons**
If you run 20 independent tests at the 5 percent level, you expect one to come out "significant" by pure luck even when nothing at all is happening. Correcting for the number of tests raises the bar so that does not occur. The list of tests has to be fixed in advance, or the count is meaningless.

**Null and ceiling**
Research shorthand worth unpacking. A **null** is a comparison you expect to fail, there to prove your method is doing something. A **ceiling** is the best score anyone could possibly achieve. The direct-coordinate model in this project is neither: it is a genuine competitor that might simply win.

**Estimand**
The quantity you are actually trying to estimate. It comes up because two ways of matching models fairly (equal data exposure versus equal compute) answer subtly different questions, so they estimate different things and both answers get reported.

**Checksum, manifest, model card**
Three record-keeping artifacts. A **checksum** is a short fingerprint of a file that changes if the file changes, proving later that you analysed what you said you did. A **manifest** is the explicit list of exactly which recordings were used. A **model card** is a short standard document stating what a model is, what it was built from, and what it must not be used for.

**Carbon accounting**
An estimate of the emissions produced by the GPU hours a study consumed, reported alongside compute.

**Replication vs transfer**
**Replication** means running the same frozen procedure in a new cohort, including refitting the readout there. **Transfer** means taking the already-fitted model and applying it directly to the new cohort with no refitting. Transfer is much harder and much stronger evidence. Calling one the other overstates results, so both documents insist on the distinction.

**Equivalence testing**
Showing two things are *similar* requires its own test with an explicit margin of what counts as similar. A non-significant difference is not evidence of equivalence; it is often just evidence of insufficient data.

**Smallest effect of interest**
The smallest improvement you would care about, decided in advance. Without it, any positive number can be spun as success.

---

## Group 5: datasets used in this project

**GAVD**
Gait Abnormality in Video Dataset. Gait clips harvested from public online video with condition annotations. Used here strictly as a code and pipeline audit, because participant identities are unavailable, the same video contributes many clips, and the historical model in this project already saw the evaluation recordings.

**Public stroke gait cohort (Van Criekinge et al.)**
Laboratory recordings of 138 able-bodied adults and 50 stroke survivors, with full-body motion capture and force plates. The decisive clinical dataset here, because it provides an independently measured force target alongside kinematics.

**AMASS**
A large unified collection of motion capture datasets covering a wide range of human movement. Used for general-purpose pretraining. Broad and clean, but non-clinical, so it can support representation learning and cannot support any clinical claim.

**MoVi**
A dataset with synchronized motion capture and calibrated video from multiple fixed cameras. Useful because it lets you test whether a prediction stays stable across *real* camera views, which involves genuine changes in visibility and projection error, rather than just mathematically rotating a coordinate frame.

**GaitRec**
A large collection of ground reaction force recordings across several impairment classes. Used to check whether the force target behaves sensibly at larger scale.

**Parkinson's disease cohort (Shida et al.)**
A second clinical dataset, deliberately left sealed until the stroke analysis is finished, so it can serve as an honest replication test.

---

## Group 6: model names used throughout this project

These names mean the same thing in every document in this folder.

| Name | What it is | What it is there to answer |
|---|---|---|
| `standard_one_view` | Ordinary encoder, ordinary readout, one look at the input | What does the plain approach get you? |
| `sign_augmented` | Same, but trained on mirrored copies with flipped labels | Is showing mirrored examples enough, without enforcing anything? |
| `two_view_free` | Readout sees both the original and the mirrored input, with no constraint on how it combines them | Was the benefit just from looking twice? |
| `odd_output` | Final answer computed as $[q(E(x)) - q(E(Mx))]/2$, guaranteed to flip sign | Does the *rule itself* help, beyond the second look? |
| `paired_unconstrained_encoder` | The same paired branches, cross-branch fusion, and exact odd output wrapper as the equivariant encoder, but without the swap-preserving weight ties | Did two-branch fusion, rather than equivariance, explain a gain? |
| `equivariant_encoder` | Mirror structure preserved through every layer, not just the final number | Does organizing the whole representation beat patching the output? |
| `raw_kinematics` | Hand-computed motion features, no learned representation | Did learning buy anything over straightforward measurement? |
| `random_encoder` | Untrained encoder with random weights, matched readout | How much of the score comes from pretraining versus from the readout? |
| `side_agnostic` | Left and right features explicitly averaged before prediction | Can a side-blind feature still correlate with the label because the cohort is imbalanced or confounded? |
| `nuisance_only` | Predicts from recording metadata alone, no gait content | Is a shortcut sufficient to explain the result? |

The last three are meant to fail on the **paired** task. If `side_agnostic` predicts an original recording and its mirror with the same signed answer, it fails the required sign reversal. If it appears useful on originals alone, first suspect an uneven cohort or another shortcut and investigate it.

---

## Where to go next

- [README.md](./README.md): what the whole program is asking and why.
- [METHODOLOGY.md](./METHODOLOGY.md): the rules both studies share.
- [README_SHORT_TERM.md](./README_SHORT_TERM.md) and [METHODOLOGY_SHORT_TERM.md](./METHODOLOGY_SHORT_TERM.md): the fast prototype.
- [README_LONG_TERM.md](./README_LONG_TERM.md) and [METHODOLOGY_LONG_TERM.md](./METHODOLOGY_LONG_TERM.md): the full study.
