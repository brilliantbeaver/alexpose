# Latent Laterality: predictive motion representations when left and right cannot be trusted

**Study 02 proposal.** Companion documents: [theory.md](./theory.md) for the
mathematics, [experiments.md](./experiments.md) for the experimental program,
and [swap-probe.md](../../docs/studies/latent-laterality/swap-probe.md) for the
first executable gate.

---

## Abstract

A skeleton sequence is a table of numbers whose rows carry anatomical names.
Row `left_ankle` is supposed to contain the left ankle, frame after frame. When
a computer vision system produces that table from video, the numbers can stay
physically plausible while the *names* silently attach to the wrong limbs for
part of a sequence. Nothing looks broken. Every downstream quantity that
depends on which side is which becomes unreliable, and no error message is
raised.

This proposal argues that this failure deserves to be treated as a latent
variable to be inferred, rather than as noise to be smoothed away or a
preprocessing bug to be patched. We formalize it as an unknown, time varying
binary relabeling of bilateral joint slots, and we ask whether a predictive
representation learner should carry a calibrated distribution over that
relabeling inside its objective. We propose Semantic-Gauge JEPA (SG-JEPA), a
joint-embedding predictive architecture that (i) splits its representation into
a part that is unchanged by the relabeling and a part that flips sign under it,
(ii) infers a temporal posterior over relative relabeling from visible context
alone, and (iii) marginalizes that posterior when predicting masked latent
targets instead of committing to one naming convention.

The central scientific claim we can actually establish in two weeks is narrow
and falsifiable. On a controlled AMASS benchmark with exactly specified
corruption, we test whether posterior-aware predictive learning beats the
strongest transparent alternative, which is to detect and correct the swap
first and then learn representations normally. We also prove, and then respect,
a limit: under a gauge-symmetric motion prior and corruption law, with
absolute-side cues removed or balanced, the *global* left/right convention is
not recoverable from the data alone. A model that reports a confident signed
answer in that regime is not accurate but overconfident. If
the simple corrector wins, that is the result, and we will report it.

### The pitch in one paragraph

A **representation** is the compact internal vector a model uses instead of
the raw input. **Latent laterality** means that the model does not observe which
of two left/right naming conventions generated each part of a skeleton
sequence. **Calibrated uncertainty** means that when the model assigns a 70%
chance to a convention change, comparable changes occur about 70% of the time.
The proposal is to put that uncertainty inside masked motion prediction, while
keeping side-insensitive and side-sensitive information in separate channels.
The empirical question is not whether this design is elegant. It is whether it
improves a common downstream target over a transparent detect-then-correct
pipeline under an honest, leakage-free benchmark.

### A concrete eight-block example

Imagine a 32-frame clip divided into eight four-frame blocks. The pose tracker
uses the expected joint names in blocks 1 through 3, exchanges all five
left/right joint pairs in blocks 4 and 5, then returns to the expected names in
blocks 6 through 8. The global label of the first block is unknowable if all
independent anatomical cues have been removed. The two boundaries around the
swapped segment may still leave evidence under a smooth-motion model. Latent
Laterality is the problem of learning those relative boundaries, preserving the
information needed for an eventual signed readout, and declining to name the
absolute side until an independent anchor permits it.

---

## 1. Start from the data: what a skeleton sequence is, and what it is not

Human motion analysis rarely operates on raw pixels. A pose estimator converts
each video frame into a small set of **keypoints**: two- or three-dimensional
coordinates for anatomical landmarks such as the pelvis, the knees, and the
ankles. Stacking those keypoints over time gives a tensor with three axes:
time, joint, and coordinate. This study uses an eleven-joint skeleton called
**Core11**, and it groups frames into short **blocks** of four frames each, so
a 64-frame window contains sixteen blocks.

Two properties of this representation matter for everything that follows.

**The joint axis is an index, not a measurement.** Position two in the tensor
means "left knee" only because a convention says so. The number stored there is
a coordinate; the anatomical meaning lives in a lookup table outside the
tensor. If the lookup table is right, the tensor is interpretable. If the
lookup table is wrong for a stretch of frames, the tensor remains numerically
well formed and its interior frames can remain physically plausible. The
boundaries of that stretch may become discontinuous, which is the relative
evidence used by this study.

**Bilateral joints come in exchangeable pairs.** Core11 has five such pairs
(hip, knee, ankle, heel, and forefoot on each side) plus a midline pelvis. Because
human bodies are approximately mirror symmetric, exchanging the contents of the
five pairs can produce plausible individual frames. Only the transition into or
out of a swapped segment may reveal the error, and crossings or occlusion can
hide even that signal. Real ankle-only errors are partial swaps outside the
coherent five-pair model and are evaluated separately as misspecification.

We write the exchange operator as $P$. It is a **permutation** of the joint
axis: it moves the contents of slot `left_knee` into slot `right_knee` and vice
versa, and it leaves the pelvis alone. Applying it twice returns the original
tensor, written $P^2 = I$, which makes $P$ an **involution**. The set
$\{I, P\}$ with composition is the smallest nontrivial group, $\mathbb{Z}_2$.
That two-element group is the entire symmetry structure of this study, which is
what makes the problem tractable enough to analyze exactly.

## 2. The failure: coordinates stay plausible while names drift

Consider a monocular video of someone walking toward the camera. As the legs
cross, the two ankles occupy nearly the same image region for several frames.
Occlusion, motion blur, or a momentary tracking failure can cause the estimator
to resume with the two ankle tracks exchanged. The output remains a smooth,
plausible gait. It has simply started calling the left ankle "right" and the
right ankle "left." Some frames later, another crossing may swap them back.

![Three transformations that produce similar looking tensors but demand
different treatment](./images/proposal-01-three-transformations.svg)

The figure above makes a distinction that this literature routinely collapses.
Three different operations can produce a tensor that resembles a left/right
change, and they are not the same object.

| Operation | Physically, what happened | What changes in the tensor | Status in this study |
| --- | --- | --- | --- |
| **Sensor action $C_R$** | The camera or coordinate frame moved | Coordinates are transformed by $R$; joint names are untouched | The action and sampled value are declared by the experiment |
| **Anatomical reflection $M$** | A counterfactual mirror-image person | Bilateral slots exchange *and* the sideways coordinate negates | Constructed deliberately when used |
| **Semantic permutation $P^{g_k}$** | Nothing physical; the naming changed | Bilateral slots exchange when $g_k=1$; no coordinate is negated | The action $P$ is known, but the time-local bit $g_k$ is latent |

The experiment declares all three transformation families. What is hidden is
the realized semantic path $g_{1:K}$. A sensor action can also vary with time,
but it has a different declared effect on coordinates and labels. Prior work on
mirror robustness, including this project's earlier efforts, studied the known
action $M$. These are different problems with different correct answers.

## 3. Why ordinary invariance is the wrong tool

The standard machine learning response to a nuisance transformation is data
augmentation: apply the transformation to inputs, ask the model for the same
output, and teach it to ignore the transformed distinction. We can apply $P$
during training because its action is known. The problem is that the realized
deployment path $g_{1:K}$ is hidden, temporally structured, and scientifically
relevant. It may apply to blocks 5 through 8 and not to the rest. Blind
invariance removes the very signal needed to say that those blocks disagree
with their neighbours.

**The destroyed information is the information we want.** Many clinically and
biomechanically interesting quantities are explicitly signed differences
between sides: right-minus-left step length, right-minus-left joint excursion,
asymmetry in contact timing. Call a quantity $y$ **odd** if the relabeling
flips its sign, $y(Pz) = -y(z)$, and **even** if the relabeling leaves it
alone, $y(Pz) = y(z)$. Walking speed is even. Right-minus-left step length is
odd. A representation invariant to $P$ cannot recover the signed odd quantity
on its own. It may retain an even function such as $|y|$, so unsigned asymmetry
magnitude can survive while anatomical direction does not.

The alternative failure is equally bad. A model that silently picks one naming
convention and commits to it will confidently report an asymmetry with the
wrong sign whenever it picks wrong, and it will give no warning.

The position this proposal defends is that neither extreme is correct. The
representation should keep three things separate: content that is genuinely
unaffected by the relabeling, content that flips with it and can therefore be
transported once the relabeling is known, and an explicit distribution over
what the relabeling actually was.

## 4. The research question

> **Under intermittent latent bilateral token relabeling, can a predictive
> motion representation recover the identifiable relative correspondence,
> preserve parity-sensitive information for an independently anchored readout,
> and outperform correction-first baselines while remaining calibrated about an
> unanchored global sign?**

Unpacking that sentence in order:

- *Intermittent latent bilateral token relabeling* is the failure of Section 2:
  an unknown, time-local exchange of paired joint slots.
- *Recover the potentially identifiable relative correspondence* means
  answering "do these two blocks use the same naming convention?" for every
  adjacent pair of blocks. Section 5 states the motion and corruption
  assumptions under which the observations contain evidence for that answer.
- *Preserve parity-sensitive information* means keeping the odd content intact
  rather than averaging it away, so a signed quantity is still computable later.
- *Outperform correction-first baselines* is the honest competitive test. The
  obvious engineering answer is to detect swaps, fix them, and then train a
  normal model. If that works as well, it should win.
- *Calibrated about an unanchored global sign* means the model must not invent
  the one thing it provably cannot know.

## 5. What is knowable, and what is not

This is the conceptual core of the study, and it is worth stating carefully
because it determines what a legitimate output even looks like.

![Relative correspondence is identifiable from the data; the global convention
is not](./images/proposal-02-identifiability.svg)

Write $g_k \in \{0, 1\}$ for the naming convention in effect during block $k$:
zero means "slots mean what they say," one means "the bilateral slots are
exchanged." The observation model is that we see $P^{g_k} z_k$, where $z_k$ is
the true, correctly named motion.

**Relative convention can be statistically recoverable.** Whether two adjacent
blocks agree is written $r_{k} = g_k \oplus g_{k+1}$, where $\oplus$ is
exclusive-or. A local naming flip often creates a discontinuity that is less
compatible with a smooth-motion model than either consistent assignment. That
contrast is evidence, but it is not guaranteed. A symmetric pose, a limb
crossing, severe occlusion, or a genuinely abrupt motion can make the two
assignments equally plausible. Relative recovery therefore requires a declared
temporal motion model, corruption assumptions, and nondegenerate observations.
The experiments measure how often those conditions supply useful evidence
rather than treating recoverability as an algebraic fact.

**Absolute convention is not identifiable.** Now suppose an adversary flips
*every* block at once: it replaces $z$ by $Pz$ and simultaneously flips every
$g_k$. Because $P^2 = I$, the two changes cancel exactly, and the observed data
is bit-for-bit identical. Two different worlds, one requiring the answer $+y$
and the other requiring $-y$, produce exactly the same observations. No amount
of data, model capacity, or training time separates them. This is not a
weakness of any particular architecture. It is a property of the problem, and
it is proved formally in [theory.md](./theory.md#5-what-is-and-is-not-identifiable).

The practical consequence is a hard constraint on outputs. When the two worlds
are equally likely, every deterministic sign decision has error probability at
least one half. A model reporting a confident sign in that regime is not
performing well; it is exploiting some cue that the experiment was supposed to
remove, or it is fabricating.

**Anchors escape the trap.** An **anchor** is independent anatomical evidence
about the convention for one block: a visible wearable marker, an audited
foot-to-force-plate assignment, or a verified first frame. One exact anchor
plus exact relative edges on a connected chain determines the whole path. With
a noisy anchor or uncertain edges, the same information updates a posterior but
does not determine one path with certainty.

**Without an anchor, the honest output is a set or a symmetric distribution.**
Reporting the orbit $\{y, -y\}$, or a distribution symmetric about zero, or the
even functional $|y|$, states what is known without claiming what is not. This
is an unusual output contract, and it is one of the things this study is
designed to test rather than assume.

## 6. The proposed method, in words before symbols

![SG-JEPA: paired encoding, parity split, temporal gauge posterior, and
marginalized prediction](./images/proposal-03-sg-jepa.svg)

SG-JEPA is built on a **joint-embedding predictive architecture** (JEPA). The
idea behind JEPA is to learn representations by prediction in embedding space:
hide part of the input, encode the visible part with a student network, encode
the hidden part with a slowly updated teacher network, and train the student to
predict the teacher's embedding of the hidden part. Nothing is reconstructed
pixel by pixel or coordinate by coordinate, so the model is free to discard
unpredictable detail while keeping predictable structure. The teacher is an
exponential moving average of the student and receives no gradient from the
prediction loss. Those choices stabilize the target but do not, by themselves,
exclude a constant representation. The proposed objective therefore includes
VICReg anti-collapse terms, and every run checks feature variance and effective
rank.

SG-JEPA adds four components.

**Paired encoding.** Every block is encoded twice, once as observed and once
after applying $P$. The two branches share weights, so this costs computation
but not parameters.

**Exact parity split.** From the pair of encodings, form the sum and the
difference, each halved:
$h^+ = \tfrac{1}{2}(F(x) + F(Px))$ and $h^- = \tfrac{1}{2}(F(x) - F(Px))$.
By construction and with no training required, $h^+$ is unchanged when the
input is relabeled and $h^-$ negates. The representation now has a
side-agnostic channel and a side-sensitive channel, and their sum recovers the
original encoder output. Thus no information already present in $F(x)$ is lost
by the algebraic split, although evaluating both branches doubles encoder
compute. This construction is an instance
of the classical Reynolds averaging operator, described in
[theory.md](./theory.md#7-exact-even-and-odd-representation-channels).

**A temporal correspondence posterior.** A small head scores each adjacent
block pair for agreement. Those scores combine with a prespecified segment-
duration prior in a finite-duration chain **conditional random field**. Exact
forward-backward inference runs on an augmented state containing the current
convention and capped run length. This retains duration structure without
enumerating every binary path. Crucially, the head reads only visible context,
so the hidden prediction target cannot leak into the correspondence estimate.

**Marginalization instead of commitment.** When predicting a masked teacher
embedding, the odd part of the target must be transported into the reference
block's convention, and the correct transport depends on the unknown path. SG-JEPA
sums over paths weighted by the posterior rather than picking the single most
likely one. When the posterior is confident, this behaves like correction. For
block $k$ transported to reference block $r$, the posterior-mean odd target is
multiplied by $1 - 2\Pr(g_k\oplus g_r=1\mid x)$. It shrinks toward zero when
that relative orientation is unresolved. That is the mechanism this study
exists to test.

## 7. Hypotheses

The following are preregistration candidates. The margins of 5 percent, 2
percent, and 0.01 may be revised using validation-only pilot evidence before
the test set is opened, after which they are frozen with a written rationale.
H1, H2, and H4 are tested in that fixed order at one-sided $\alpha = 0.05$, and
testing stops at the first failed gate. H3 is an integrity requirement rather
than a scored hypothesis. H5 licenses only the ecological extension. Secondary
severity contrasts use Holm correction.

| ID | Claim in words | Formal statement |
| --- | --- | --- |
| **H1** | Temporal evidence contains useful information about whether neighbouring blocks agree, even when the global convention is hidden. | The one-sided 95% lower confidence bound for the identity-mean improvement in equivalence-class path negative log-likelihood over the fixed input-free duration prior is above zero. |
| **H2** | Carrying the posterior inside the predictive objective beats correcting first. | In one prespecified local-swap-plus-occlusion regime, the one-sided 95% lower bound for $(E_{\text{base}} - E_{\text{SG}})/E_{\text{base}}$ exceeds 0.05 against the strongest non-oracle corrector or synchronizer, after equal identity weighting. |
| **H3** | The model does not invent the sign it cannot know. | Global randomization prevents above-chance absolute-gauge leakage; relative-edge probabilities remain calibrated; an independent anchor improves signed kinematic recovery. The 50:50 global root is imposed by the unanchored model, so this is an integrity assertion, not evidence of learned calibration. |
| **H4** | Robustness is not bought with a clean-data tax. | Against the strongest baseline, the one-sided 95% upper bound for $E_{\text{SG}}/E_{\text{base}} - 1$ is below 0.02, and the upper bound for the absolute edge Brier-score difference is below 0.01. |
| **H5** | The phenomenon occurs in real video, not only in our generator. | In the probability-sampled, retrievable GAVD source-video frame, coherent local convention events have a one-sided 95% lower prevalence bound above 1%, with at least 20 confirmed events across 10 videos and two view strata. |

H5 is explicitly conditional. If it fails, the paper narrows to a controlled
robustness result. It does not invalidate H1 through H4.

## 8. Why this matters beyond gait

### 8.1 For representation learning

The usual framing of symmetry in deep learning starts by declaring a group
action. Data augmentation samples and applies it. Equivariant architectures
build its transformation law into the weights. Canonicalization selects or
estimates a representative of each orbit. These tools can handle unknown
realized transformations, but they do not usually carry a calibrated posterior
over a time-local semantic path inside a masked predictive objective.

This study takes a step that the framing does not cover: the group element is a
latent variable with temporal structure, inferred from data, carried as a
distribution, and marginalized inside a predictive objective. The question is
not "should the model be invariant to this transformation" but "what should the
model do when it is uncertain *what its input tokens mean*."

That question is not specific to gait. Sensor arrays can be miswired or
relabeled. Multi-object trackers swap identities. Keypoint schemas differ
between toolkits. Entity slots in structured prediction can be permuted. In all
of these, semantic correspondence is a latent variable and the transferable
recipe is the same: attach a small structured correspondence posterior to the
representation learner, and type the downstream outputs by what is actually
identifiable.

### 8.2 For biomechanics

Signed between-side differences are the quantities that carry clinical meaning
in gait analysis. Right-minus-left step length, joint excursion asymmetry, and
contact-timing differences are interpretable only if the side assignment is
trustworthy. This study tests two things about that information: whether it
survives the corruption at all, and whether the model knows when it is entitled
to attach an anatomical sign to it.

We are explicit about what this does not establish. AMASS contains motion
capture kinematics, not ground reaction force. Neither dataset in this study
licenses claims about force, balance, fall risk, diagnosis, prognosis, or
treatment response. Those require measured outcomes and a participant-safe,
independently side-anchored cohort, which is future validation and not a
two-week dependency.

## 9. Relation to the previous Ideas 05 and 09 program

Two earlier efforts in this repository approached laterality from the known-
transformation side.

[Idea 05](../archive/portfolio-ideas/ideas/05-signed-laterality-decodability/)
was a measurement instrument. Freeze a trained S-JEPA, decode a constructed
signed left-minus-right scalar from its features, compare against raw
coordinates, and check whether a known anatomical mirror negates the decoded
value.

[Idea 09](../archive/portfolio-ideas/ideas/09-reflection-equivariant-symmetry-axis/)
was an architectural inductive bias. Supply both $x$ and its known anatomical
reflection $Mx$, enforce exact layerwise equivariance, and expose even and odd
channels directly.

Both assume the action is known, global, and deliberately applied. Both ask
"does the model obey this mirror?" Latent Laterality asks a different question:
"which local token convention generated this observation, what can be inferred
only relatively, and what should the model output when the global answer is
absent?" The differences are concrete:

- a latent temporal correspondence process instead of a supplied mirror;
- posterior-aware masked prediction instead of fixed equivariance alone;
- anchored and unanchored training and evaluation as separate regimes; and
- a decision contract that returns a signed value only under independent
  anatomical evidence.

The older models are retained, but their role has changed. They are capacity
controls and negative controls. Nothing about commuting with a physical
reflection implies that a model can resolve a semantic permutation, and testing
that implication is itself informative.

## 10. Related work, and the gap that remains

No single ingredient below is new. The contribution has to be framed as a
tested integration, and any claim of being "first" should be avoided unless a
submission-time literature search supports it.

| Area | What prior work already provides | What this study tests beyond it |
| --- | --- | --- |
| Predictive skeleton learning | [S-JEPA](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/4755_ECCV_2024_paper.php) predicts masked skeleton embeddings; [seq-JEPA](https://arxiv.org/abs/2505.03176) learns invariant and equivariant content from supplied actions | The action is a latent semantic correspondence with a posterior and an unanchored global offset |
| Known reflection | [Chirality Nets](https://proceedings.neurips.cc/paper_files/paper/2019/file/1f88c7c5d7d94ae08bd752aa3d82108b-Paper.pdf) builds coordinate reflection plus side exchange into pose networks | Distinguish a known physical reflection from an unknown token-name permutation that can switch locally |
| Learned symmetry and canonicalization | [Learned canonicalization](https://proceedings.mlr.press/v202/kaba23a.html), [probabilistic symmetrization](https://proceedings.neurips.cc/paper_files/paper/2023/hash/3b5c7c9c5c7bd77eb73d0baec7a07165-Abstract-Conference.html), and [SymPE](https://proceedings.iclr.cc/paper_files/paper/2025/hash/c7138635035501eb71b0adf6ddc319d6-Abstract-Conference.html) learn or randomize symmetry choices | Couple a time-local bilateral correspondence distribution to masked motion prediction and to parity-typed decisions |
| Learned actions | [Winter et al.](https://proceedings.neurips.cc/paper_files/paper/2022/hash/cf3d7d8e79703fe947deffb587a83639-Abstract-Conference.html) separate invariant content from a learned action component, including permutations; [SEN](https://proceedings.mlr.press/v162/park22a.html) maps unknown input actions to known feature actions | Infer a time-indexed correspondence path, test its calibration, and compare joint predictive marginalization against correction-first use |
| Gauge methods | [Gauge Equivariant Transformer](https://proceedings.neurips.cc/paper/2021/hash/e57c6b956a6521b28495f2886ca0977a-Abstract.html) handles arbitrary local coordinate frames on manifolds | Use a finite gauge analogy for anatomical token semantics, not mesh coordinate orientation |
| Synchronization | [Permutation synchronization](https://proceedings.neurips.cc/paper_files/paper/2013/hash/3df1d4b96d8976ff5986393e8767f5b2-Abstract.html) aggregates noisy pairwise correspondences | Learn relative evidence from predictive motion, and compare synchronization-first against joint learning |
| Pose-swap correction | [PoseFixeR](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1009989) detects and corrects left/right leg reversals using gait structure; [SwapPF](https://www.scitepress.org/Papers/2026/144493/144493.pdf) uses gait periodicity and particle filtering | A direct threat and a required baseline: does posterior-aware representation learning add anything beyond correction? |
| Ambiguous pose outputs | [DiffPose](https://openaccess.thecvf.com/content/ICCV2023/papers/Holmquist_DiffPose_Multi-hypothesis_Human_Pose_Estimation_using_Diffusion_Models_ICCV_2023_paper.pdf) predicts multiple plausible poses under monocular ambiguity | Restrict the hypothesis set to a declared semantic orbit, and test anchored against quotient-valued decisions |
| Geometry-aware correspondence | [Telling Left from Right](https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_Telling_Left_from_Right_Identifying_Geometry-Aware_Semantic_Correspondence_CVPR_2024_paper.html) shows that visual geometry and orientation cues can resolve correspondence | State the impossibility result only when such cues and other anchors are excluded or balanced |

The defensible prospective contribution is therefore:

> a predictive representation study in which coherent bilateral token identity
> is a local latent variable; relative correspondence is inferred and
> calibrated over time; prediction marginalizes that uncertainty; and the
> output is signed only when an independent anchor fixes the component's global
> convention.

## 11. Where the project actually stands

Honesty about the current evidence is part of the proposal. The directory
`outputs/repaired-jepa-seed7-v2` contains four seed-7 histories and
checkpoints, each trained for 100 epochs on AMASS Core11 with 64-dimensional
embeddings and close parameter matching.

| Variant | Parameters | Selected epoch | Validation KL | Feature health at selection | Programmed masked commutation |
| --- | ---: | ---: | ---: | --- | ---: |
| `standard_sjepa` | 821,866 | 95 | 0.2555 | variance 0.0695; effective rank 16.21 | Not applicable |
| `paired_shared_no_cross` | 821,866 | 98 | 0.2653 | even 0.0873 / 17.16; odd 0.0775 / 16.01 | 0 |
| `reflection_equivariant` | 821,860 | 98 | 0.2791 | even 0.0896 / 16.56; odd 0.0785 / 15.27 | 0 |
| `paired_unconstrained` | 822,214 | 98 | 0.2304 | even 0.0830 / 15.93; odd 0.0844 / 15.64 | 2.7574 |

Feature-health entries are variance and effective rank. The standard arm is the
only one whose surviving summary records runtime: 7,326 seconds for 100 epochs,
0.405 GiB peak allocated CUDA memory, and 236,170 updates at the selected epoch
95, with 248,600 updates executed across the full history. The observed
full-suite time of roughly 12 hours is the planning budget.

**These KL values must not be used to rank architectures.** Each encoder
predicts its own exponential moving average teacher, so each is scored against
a different target distribution with different teacher entropy. A lower self-KL
is not a common-target representation result.

Six limitations block scientific interpretation of these artifacts:

1. The checkpoints omit `metadata.paired_mask_contract` and predate the
   branch-specific $P$-closed mask implementation. In variants with cross-branch
   interaction, the legacy same-index mask can leave a physical counterpart
   visible in the other branch and permit copying. A zero commutation residual
   does not detect that leakage.
2. `run_config.json` and `summary.csv` were overwritten by the last
   independently launched standard-only arm. Runtime and memory for the other
   arms are lost, so the directory is not an immutable four-arm run.
3. `evaluate_test=false`; there is no test result. Validation used one
   deterministic corruption and mask draw and one training seed.
4. Histories report 79,552 training-window exposures per epoch because 79,535
   windows are padded to 2,486 complete batches of 32, so the final batch
   repeats 17 examples. This is expected. However, the run records absolute
   cluster paths rather than manifest and code hashes, so the exact input cannot
   be authenticated from the copied artifacts alone.
5. The top-level configuration records a nominal 96-dimensional template while
   capacity resolution produces the effective 64-dimensional checkpoints. Future
   summaries must record both, per arm.
6. No checkpoint was trained or evaluated on semantic $P$-swaps, explicit
   missingness inputs, a gauge posterior, calibration, an anchor, an
   orbit-valued output, or a common downstream target.

What the run does establish is real but modest: the training and loading
scaffold executes, close parameter matching is achievable, selected features
are not degenerate, and the tied layers obey their programmed known-reflection
algebra under the old mask. It establishes neither a repaired fixed-reflection
comparison nor anything about latent laterality.

## 12. Data

### 12.1 AMASS-Gauge: the controlled corruption benchmark

[AMASS](https://amass.is.tue.mpg.de/) unifies many motion-capture corpora into
one body parameterization. The local Core11 manifest holds 8,854 sequences with
151 training, 19 validation, and 19 test identities, yielding 79,535, 5,936,
and 8,220 overlapping 64-frame windows under the repository's end-window rule.

The benchmark samples a piecewise-constant convention path $g_t$ **for each full
sequence**, not independently per window, applies the coherent five-pair Core11
permutation $P^{g_t}$, and records every switch, occlusion, noise draw, sensor
reflection, and anchor. Overlapping windows inherit their parent sequence's
path, which is what prevents the same source frames from carrying contradictory
labels. The design cells are:

1. clean;
2. global swap;
3. one local segment of 1, 2, 4, or 8 blocks;
4. repeated Markov switches;
5. a local segment plus boundary-centered occlusion and noise;
6. sparse trusted anchors; and
7. partial joint or limb swaps, as out-of-model stress tests.

The provisional primary regime is one 4-block (16-frame) local swap with
moderate bilateral distal-joint occlusion and noise. Exact magnitudes are fixed
from validation evidence, or from the GAVD audit if it is available, before any
test access. Exactly ten deterministic test corruption draws are shared by every
model.

**A leakage problem must be fixed first.** The current conversion is not gauge
neutral: 64.60% of sequences use a named-left/right `hip_facing_fallback` to
determine body orientation, and a further 5.95% use a related fallback after
anatomy and trajectory disagree. Applying $P$ after such a frame has been
established leaks the absolute convention into the coordinates. Before any
training, we build `gauge-neutral-v1` with no side-labelled fallback, balance a
hidden sensor reflection independently of the semantic bit, and require a strong
raw-feature absolute-gauge probe to sit near AUROC 0.5 with an upper 95% bound
below 0.55.

Only 2,607 of the 8,854 current sequences use the side-neutral travel-based
facing method. Naively excluding every fallback would leave roughly 10,921,
942, and 1,551 train, validation, and test windows and 107, 14, and 15
represented split identities, dropping four current test identities. The neutral
conversion therefore has to publish an attrition table before training and
either retain stationary motion under a genuinely orientation-free or random-yaw
contract, or revise its sample-size, runtime, and inference claims to the
retained cohort.

AMASS supplies canonical joint identity and kinematics, not ground reaction
force. The controlled odd targets are right-minus-left ankle-speed energy, joint
excursion, or contact-timing proxies, all defined before fitting. Clean speed
and total motion energy are the even controls.

### 12.2 GAVD: an optional ecological audit

The [GAVD paper](https://arxiv.org/abs/2407.04190) reports 1,874 sequences and
annotations for more than 450 public source videos; the
[official release](https://github.com/Rahmyyy/GAVD) distributes metadata, and
researchers retrieve videos independently under platform, ethics, and copyright
requirements. It carries camera-view and gait-pattern annotations but no
verified anatomical-side labels and no kinetics.

The local cache holds only 96 sequences from 18 source videos through a single
MediaPipe pipeline, which is inadequate for any prevalence or transfer claim.
The audit therefore runs in two clearly separated lanes: a probability-stratified
source-video sample used for weighted prevalence, and a candidate-enriched
sample used only for error taxonomy and ranking stress tests.

Two blinded human raters label each candidate as `correct`, `coherent swap`,
`partial or joint error`, `person-track error`, or `indeterminate`, followed by
adjudication. Pose-extractor disagreement generates candidates, not truth. A
broad ecological claim requires the probability sample's inclusion-weighted
estimate to have a one-sided 95% lower bound above 1%, plus at least 20 confirmed
local coherent events across at least 10 source videos and two view strata,
consequential change in an odd feature, and failures that survive transparent
correction. The inference frame is retrievable source videos, retrieval response
is reported by stratum, and no wider claim is made when nonresponse cannot be
adjusted. Otherwise GAVD is an audit and a null result, not support for SG-JEPA
in the wild.

## 13. Experimental design at a glance

The full program, including setup order and commands, is in
[experiments.md](./experiments.md). The comparison logic is summarized here
because it is what makes the study decisive rather than merely demonstrative.

| Arm | The question it isolates |
| --- | --- |
| Raw continuity plus Viterbi or HMM | Can kinematic continuity alone repair the sequence? |
| Corruption-trained standard S-JEPA plus gauge head | Does ordinary predictive learning plus an uncertainty head suffice? |
| Repaired fixed-reflection JEPA plus the same head | Does known anatomical-reflection structure transfer to unknown semantic relabeling? |
| Compute- and capacity-matched unconstrained paired transformer | Are gains just a second branch or more computation? |
| Relative synchronization, corrected input, then ordinary S-JEPA | Is correction before representation learning enough? |
| SG-JEPA | Does joint posterior-aware predictive learning add value? |
| Oracle path; uniform posterior | Upper mechanism bound and non-informative lower baseline |

Hard-MAP transport, replacement of the duration model by independent edge
factors, removal of the parity split, and removal of the validity embedding are
seed-7 ablations. Output
symmetrization and mirror augmentation are inexpensive secondary controls. The
*strongest* validation baseline, not the most convenient one, is carried into
the three-seed test.

The primary endpoint hierarchy is, first, the length-normalized equivalence-class
path negative log-likelihood
$-(K-1)^{-1}\log\left[q(g^\star \mid x) + q(g^\star \oplus 1 \mid x)\right]$,
which scores full-path posterior mass per identifiable relative decision up to
the unanchored global flip; and second, a shared-target parity-sensitive
prediction error in the primary corruption regime. Secondary endpoints cover
edge Brier and log loss, switch AUPRC, event F1 at a fixed one-block tolerance,
segment intersection-over-union, path Hamming error minimized over the global
flip, anchored odd normalized MAE and sign accuracy, unanchored orbit MAE and
symmetric-mixture NLL, even-task preservation, risk-coverage curves, feature
variance and effective rank, and measured FLOPs and GPU-hours.

Every corruption draw is identical across arms. Results stitch overlapping
evidence onto unique sequence blocks, aggregate draws to sequence, then
sequences to equal-weight identity. The independent unit is the test identity as
retained by the frozen neutral manifest; the current split has 19, and neutral
conversion attrition may reduce that. Seed IDs 7, 19, and 31 are paired between
SG-JEPA and the selected baseline; the confirmatory effect averages the three
seed-specific identity effects, and its identity bootstrap resamples identities
jointly and is conditional on that fixed seed set. We report every seed, the
descriptive seed mean and range, a 10,000-resample identity-cluster bootstrap
interval, and a paired identity-level randomization test. Nonlinear event and
curve metrics are recomputed inside each bootstrap replicate. Test is evaluated
once, after architecture, corruption, temperature, anchors, and margins are all
frozen.

## 14. Budget and stopping rules

The hard budget is eight current-suite equivalents, roughly 96 reference-GPU
hours if each 12-hour suite occupies one accelerator.

| Work | Budget | Decision value |
| --- | ---: | --- |
| Integrity, profiler, leakage and oracle sanity checks | 6 GPU-h | Prove the benchmark is nontrivial and gauge neutral |
| Repaired four-arm Gate-0 baseline | 12 GPU-h | Replace the invalid old-mask artifacts |
| Seed-7 learned-arm screen, then seeds 19 and 31 for SG-JEPA and the selected baseline | up to 54 GPU-h | A ceiling and reserve, not a requirement; nonfinal arms stay at seed 7 |
| Four seed-7 ablations | 8 GPU-h | Attribute a gain to marginalization, temporal evidence, parity, or validity |
| Decisive-ablation and job-failure contingency | 4 GPU-h | Spend only after a validation gate passes |
| Sealed AMASS test inference | 4 GPU-h | Ten shared draws, run once |
| Optional paired GAVD extraction | 8 GPU-h | Hard stop; retrieval and annotation remain CPU work |

Days 1 and 2 freeze claims, hashes, splits, common metrics, and the neutral
corruption generator, and implement the raw, Viterbi, and oracle baselines plus
repaired mask tests. GAVD retrieval and annotation preparation begin in
parallel. Day 3 runs short integration tests only. Days 3 through 8 run the
validation-only learned matrix and the ablations. Day 8 freezes the winning
contrast and the calibration. Day 9 opens the AMASS test once. Days 10 through
12 run clustered statistics and figures. Days 13 and 14 write and package the
controlled claim.

Full training stops if the gauge-neutrality or oracle-consequence gates fail.
Architecture expansion stops if the correction-first baseline lands within 5
percent and is equally calibrated. The GAVD workflow never blocks completion of
the controlled AMASS paper.

## 15. What each possible outcome licenses us to say

![Outcomes and the claims they license](./images/proposal-04-outcome-ladder.svg)

| Outcome | Licensed conclusion |
| --- | --- |
| SG-JEPA wins the paired common-target and calibration comparison in the frozen primary corruption cell | Posterior-aware latent correspondence improved controlled predictive motion representation under the coherent-swap model |
| SG-JEPA improves path detection but not common-target prediction | Better correspondence inference, not a representation-learning gain |
| The corrector or synchronizer matches SG-JEPA | Use the simpler correction pipeline; the architecture claim is rejected |
| Gains appear only under severe synthetic errors | A synthetic robustness result with limited practical scope |
| Unanchored outputs choose a confident global sign | The identifiability contract failed, even if point accuracy looks high |
| The natural GAVD gate passes | Add carefully bounded evidence that coherent local convention errors occur in uncontrolled video |
| The natural GAVD gate fails | Report the audit; make no in-the-wild necessity claim |

## 16. Limitations and responsible interpretation

- The coherent Core11 swap is narrower than the full space of real
  pose-estimation error. Partial and joint-specific errors are treated as
  prespecified misspecification stress tests, not as in-model examples.
- A clean AMASS teacher is itself an anchor. Unanchored batches must explicitly
  hide and quotient its global convention, or the experiment is not unanchored.
- Synthetic corruption labels make this controlled corruption-supervised
  representation learning, which is more specific than purely self-supervised
  learning and should be described that way.
- Population asymmetry or a handed coordinate frame can leak absolute side, so
  the gauge-neutrality probe is mandatory rather than optional.
- Three seeds, and only the test identities retained by neutral conversion (at
  most the current 19, and about 15 under naive fallback exclusion), give
  limited resolution on training and corpus-transfer uncertainty.
- GAVD source videos may contain repeated people, and its labels are dataset
  annotations rather than diagnoses produced here.
- Neither dataset licenses claims about force, balance, prognosis, treatment, or
  clinical deployment.

## 17. Closing

The most valuable outcome of this study is not necessarily a new architecture.
It is a clear empirical boundary between two kinds of failure that are
currently treated as one. Some left/right errors are bugs, and the right
response is a transparent corrector applied before anything else happens. Other
left/right ambiguities are genuine missing information, and the right response
is a representation that carries uncertainty and a decision interface that
refuses to invent a sign. Knowing which regime you are in is worth more than
either method alone, and this study is designed so that the answer, in either
direction, is scientifically informative and reportable.
