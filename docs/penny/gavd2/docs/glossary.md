# Glossary: the ubiquitous language of Gait-JEPA (iteration 2)

A shared vocabulary for the iteration-2 series. Terms are defined precisely so the
notebooks, slides, and paper all mean the same thing by them. The numbers below are the
real iteration-2 run results (68 of 68 coverage), not placeholders.

## Method

**Gait-JEPA.** A Joint-Embedding Predictive Architecture applied to pose sequences
rather than pixels. It learns to predict the latent representation of hidden parts of
a walking skeleton from the visible parts, so it can be trained on unlabelled walking
video. The flavour here is a skeleton-JEPA over the 33 BlazePose joints across time.

**Context encoder.** The online network. It reads the visible joints and frames of a
masked clip and produces a context embedding. A small transformer (2 layers) plus
learned time and joint positional embeddings so the pooled clip vector carries frame
order and left-right joint identity rather than being a bag of coordinates.

**Target encoder.** An exponential-moving-average (EMA) copy of the context encoder,
updated as `target = m*target + (1-m)*context` with `m` near 0.996 and stop-gradient.
It reads the full sequence and provides the prediction targets (the answer key).

**Predictor.** A shallow MLP that maps the context embedding at the hidden positions to
a prediction of the target embedding there.

**VICReg.** Variance-Invariance-Covariance Regularization. Here the invariance term is
an L2 between the prediction and the LayerNorm-normalized EMA target, and light
variance and covariance guards are applied to the ONLINE context embedding only (never
the stop-gradient target). This keeps the representation from collapsing to a constant.

**Block masking.** Spatiotemporal masking that hides a whole limb over a window of
frames (Style A) or all joints over a short window (Style B), so the model must predict
coordinated motion rather than interpolate a single missing joint.

**Frozen probe.** After pretraining, the encoder weights are frozen (eval mode, no
gradients) and a tiny classifier is trained on top of the frozen embeddings. This tests
whether the features themselves carry class structure, and it is where the scarce
labels are spent.

**Label efficiency.** How gracefully probe accuracy holds up as the number of labelled
training examples shrinks. A flatter curve is the promise of pretraining.

## Data and the controlled comparison

**GAVD.** The Gait Abnormality in Video Dataset: 374 annotated sequences across 11 gait
conditions, each sequence a set of annotated frames from one YouTube video.

**The exp5 68 (curated subset).** The exact 68 sequences the prior Random Forest study
trained on, drawn from five classes: normal 12, parkinsons 9, stroke 12, cerebralpalsy
15, myopathic 20. Their sequence ids are the CSV stems of the curated exp4/data tree and
equal the `sample_id`s in the exp5 82-feature pickle. Iteration 2 locks its labelled
probe set to exactly these 68 so the comparison is controlled.

**Controlled comparison.** A comparison in which everything is held constant except the
one variable under study. Here task, the five classes, the exact 68-sequence set, the
pose-extraction contract, the label taxonomy, and the classification unit and split are
all held constant, so the only intended difference from exp5 is the representation: a
learned JEPA embedding versus 82 hand-crafted joint-angle and kinematic features.

**Window versus sequence unit.** A sequence is one continuous walk. A window is one
fixed-length slice of it (T=32 frames, stride 16), and because the stride is smaller
than the window, neighboring windows overlap heavily and look almost like copies of
each other. Each walk yields about 7 windows on average. The window is the pretraining
unit. The sequence is the only honest evaluation unit for the controlled comparison, for
two reasons: exp5 classifies one feature vector per sequence, and per-window scoring lets
near-duplicate windows from the same walk leak across the train and test split (see
Window leakage). In iteration 2 the probe mean-pools a sequence's window embeddings into
one per-sequence vector, then splits by sequence so every window of a walk stays on one
side.

**Window leakage.** The inflation that occurs when windows from the same sequence land
in both the train and test folds of a per-window split, letting a classifier match a
test window to a near-duplicate training window. On iteration 2, scoring the same
frozen encoder per clip (the leaky way) reads about 0.88, while scoring it the honest
per-sequence way reads about 0.49 to 0.63 (linear 0.486, matched Random Forest 0.579,
MLP 0.626). That gap of roughly 30 to 40 accuracy points is the leak, not learning.
Iteration 2 headlines the per-sequence number and keeps the per-clip number only as a
labelled leaky diagnostic.

**Co-occurring video.** A single YouTube video can back more than one sequence. If a
video backs a held-out labelled sequence, its other (unlabelled) sequences would leak
that video's motion into pretraining. Iteration 2 excludes from the pretraining bank
every window whose source video also backs a held-out labelled sequence, making the
train/probe separation video-level rather than only sequence-level.

**Canonical id hash.** A short deterministic fingerprint of the sorted 68 locked
sequence ids, stamped onto every cache artifact (manifest, skeletons, corpus, holdout,
encoder). A downstream notebook warns if the hash it reads does not match, which catches
a stale-artifact mix before it can masquerade as a controlled comparison.

**exp5 exact split.** The prior study's single seed-42 train/test partition, produced by
`np.random.seed(42); np.random.permutation(68)` over the pickle's native feature-list
ORDER, taking the first 47 as train and the last 21 as test. Reproducing it requires
that exact ordering, not a sorted id set, so iteration 2 persists it (`exp5_split.csv`)
and reports a like-for-like accuracy point on it beside the 0.76 baseline.

## Related pages

- [learning/learning-journey.md](learning/learning-journey.md) - the plain-language story of the whole gavd to gavd2 journey.
- [[pipeline]] - one page per notebook.
- ADRs under `adr/` - why iteration 2 exists and the key decisions.
