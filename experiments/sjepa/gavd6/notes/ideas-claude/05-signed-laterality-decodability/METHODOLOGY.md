# Idea 5 methodology: signed-laterality decodability and learned reflection-equivariance

This document is the scientific specification for the reification of Idea 5. It states the question,
the mechanism that motivates it, the datasets, the exact measurement instrument, the pre-registered
decision rule, every control, the possible futures and what each licenses, and the threats to validity.
It is grounded line for line in [`../_shared_facts.md`](../_shared_facts.md) (numbers) and
[`../_neuro_facts.md`](../_neuro_facts.md) (mechanism). The proposal narrative lives in
[`README.md`](./README.md); the runnable experiment lives in the two notebooks named below. No number
here may contradict `_shared_facts.md`. Folder labels (stroke, parkinsons) are dataset annotations, not
diagnoses, and every gavd5 readout is transductive.

## 1. The question, in one sentence

On source-video-disjoint folds, is a signed left-minus-right laterality axis linearly decodable from the
frozen S-JEPA token tensor above a raw-coordinate null by a pre-registered margin, and does the
anatomical mirror that swaps left and right landmarks negate the decoded scalar?

Two endpoints are separated deliberately. The primary endpoint is decodability (is the signed axis
linearly present in the frozen tokens, competitively with raw coordinates). The secondary, mechanistic
endpoint is equivariance (does the encoding respect the reflection symmetry by flipping sign under the
anatomical mirror). Decodability can pass while equivariance fails; that is a real and informative
outcome, not a contradiction.

## 2. Why this axis, and why signed

The clinical literature separates the conditions in this dataset along a symmetry axis. Lateralized
deficits raise a signed left-minus-right asymmetry: stroke through the crossed corticospinal tract
(one brain side drives contralateral hemiparesis), early Parkinson's through contralateral
nigrostriatal onset, and hemiplegic cerebral palsy through a unilateral lesion. Diffuse deficits do
not: myopathy is a systemic proximal weakness with near-symmetric spatiotemporal gait. The validated
biomarker is the gait symmetry ratio (Patterson et al., Gait and Posture 2010, PMID 19932621); the
myopathy negative class is grounded in a Duchenne cohort that shows no significant spatiotemporal
asymmetry (Xiong et al. 2023, PMID 37525241). The parkinsonian rhythm biomarker, stride-time
coefficient of variation (Hausdorff et al. 1998, PMID 9613733; Schaafsma et al. 2003, fallers 8.8
percent vs non-fallers 4.2 percent, PMID 12809998), is explicitly out of scope for this notebook
because a 64-frame duration-warped window erases absolute cadence; that axis is Idea 4's territory.

The previously quoted scalar R-squared values (0.154 asymmetry; 0.719 step amplitude) are
unreproducible legacy claims, not established current facts. Notebook 05 now supplies a versioned
current scalar audit with target eligibility, nested source-video-grouped ridge selection, OOF
predictions, and checkpoint/provenance hashes. Until it is run, this proposal treats the older
numbers only as a hypothesis-generating motivation. Mean/std pooling is permutation-invariant, but
that property alone does not establish the cause of a weak readout.

The built-in negative control falls straight out of the mechanism. Myopathy should sit near zero on the
signed axis. If a signed axis is decodable, myopathy sequences should decode near zero while stroke,
early PD, and hemiplegic CP decode away from zero; the myopathy near-zero prediction is a mechanism
test the dataset itself supplies.

## 3. Datasets

### 3.1 Primary arm: gavd5 canonical cohort (internal validity only)

The primary arm runs on the canonical GAVD cohort (Ranjan et al., IEEE Access 2025,
DOI 10.1109/ACCESS.2025.3545787): 96 sequences from 18 unique YouTube source videos. Per-condition
source-video counts are tiny and unequal: normal 1, Parkinson's 2, stroke 3, myopathic 10, cerebral
palsy 2, and all 12 normal sequences come from a single video (`3KnFt8bH3tE`). Two consequences bind
the design. First, condition label is nearly collinear with source-video identity, especially for
normal, so the source video, not the clip, is the independent unit. Second, because per-class source
counts are as low as one, per-class leave-one-source-out R-squared on n=1 held-out sources is not a
meaningful endpoint and is not reported; the signed axis is pooled across conditions with every source
video as its own point.

A provenance confound sits underneath: most normal rows use the augmented extraction path while every
abnormal row uses the canonical path (`_shared_facts.md`), so a naive contrast could learn an
acquisition difference rather than gait. The primary comparison therefore runs on a provenance-matched
(canonical-path) subset. Provenance is not a stored column in the `.npz` cache; it is tagged
structurally from the loader path.

### 3.2 External reach arm: public multi-view pose cohorts (non-clinical, reach-tier)

A simultaneously clinical, skeleton-based, and participant-disjoint public cohort does not exist, so
there is no honest skeleton-level clinical transfer test. What exists is a NON-CLINICAL reach test for
the reflection property itself, on public multi-view pose cohorts: CASIA-B (Yu, Tan, Tan 2006; 124
subjects across 11 camera views) and OU-MVLP-Pose (Takemura et al. 2018; about 10,000 subjects, multiple
views, released keypoints). The reach questions are symmetry-specific and view-specific, never clinical:
does the signed axis decoded from one view stay stable at a nearby view (a property of the gait, not the
camera), and does a genuine left-versus-right camera swap flip it (a real physical reflection rather than
a synthetic x-negation, using CASIA-B's symmetric view angles around 90 degrees). This arm is scaffolded
and exercised on a synthetic multi-view fixture in notebook 05b; wiring the real, large, licensed
downloads is a marked TODO and is not done in this pass.

## 4. The measurement instrument

Everything in the primary arm is a test-time-only linear read of already-frozen features. There is no
encoder retraining.

### 4.1 The frozen encoder

Bind every number to ONE checkpoint: the curriculum-final target encoder with fingerprint prefix
`d0acc262` (the augmented lineage; a canonical lineage prefix `dba24a` has also been observed locally,
and mixing them would be a confound). The architecture family is a small Transformer (embed_dim on the
order of the project default of 64, depth 2, 4 heads, GELU, pre-norm; `_shared_facts.md`), but the
notebook never hardcodes the width or depth: it always constructs the model with
`SJEPAGait(**checkpoint["config"])`, reading whatever dimensions the checkpoint actually stored, so the
`state_dict` matches key for key regardless of the exact width. The token geometry is fixed by the
project: 64 frames in 4-frame patches give 16 time positions, times 33 joints, equals 528 joint-time
tokens, and the target (EMA) encoder always sees all 528 tokens. The notebook loads the checkpoint with
the same guard set as notebook 05 (mode, 12-point mask whitelist, curriculum completion, condition
order) so it can never bind to the wrong lineage.

### 4.2 The frozen signed target (defined before any fit)

From the raw cached coordinates (not the model), the target `y` is a signed per-side excursion
difference over the anatomical pairs. This is frozen before any result is computed:

```python
LEFT_RIGHT_PAIRS = [(11, 12), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32)]

def signed_left_minus_right(coords):          # coords: (frames, joints, 3)
    total = 0.0
    for left_idx, right_idx in LEFT_RIGHT_PAIRS:
        left_excursion = coords[:, left_idx, :].std(axis=0).sum()
        right_excursion = coords[:, right_idx, :].std(axis=0).sum()
        total += left_excursion - right_excursion   # signed: left minus right
    return total                                     # positive leans left, negative leans right
```

Because it uses per-joint standard deviation, the target is translation-invariant, so it is well
defined on either raw or pelvis-centered coordinates. It is antisymmetric on raw coordinates by
construction (mirroring negates it exactly); the notebook asserts this as a self-check before any
modelling.

### 4.3 The anatomical mirror (defined before any fit)

The mirror negates x and swaps each left landmark with its right partner. For the encoder pass it uses
the full 16-pair whole-body mirror (face, arm, and lower body) so the mirrored input is a valid
reflection of a real preprocessed sequence, applied on the raw coordinate column and then run through
the identical preprocessing (short-gap interpolation, pelvis-centering and body-scale normalization,
temporal resize to 64 frames). Laterality flip is OFF by default in training (flip_probability 0.0,
`_shared_facts.md`), which is exactly why testing the mirror is a real, non-trivial probe of whether the
encoder learned reflection-equivariance rather than having it handed over by augmentation.

### 4.4 The four lanes

| Lane | Feature source | Retrain | Role | Pre-registered expectation |
|---|---|---|---|---|
| A learned probe | Frozen `d0acc262` per-token features, side-structured | No | Primary | Beat floor by at least 0.05 R-squared and reach at least 80 percent of null |
| B raw-coordinate null | Handcrafted signed left-minus-right coordinate features, no network | No | Non-neural ceiling | Reference target |
| C untrained-encoder floor | Same features from a random-init encoder of identical architecture | No | Floor | Near chance |
| D mean/std-pooled control | Permutation-invariant pooled tokens | No | Nuisance | Must NOT recover a signed axis |

Lane A gives the linear probe explicit access to a per-side contrast: for each anatomical pair it uses
the time-mean left token minus the time-mean right token (the signed contrast) alongside their sum (the
symmetric context). Lane D is the crucial negative control: a mean and a standard deviation are
permutation-invariant and discard token order and side identity, so a genuinely signed quantity cannot
be recovered from them. If Lane D nonetheless "recovers" the axis, the signed claim is an artifact
(this is future F4 in Section 6).

### 4.5 The source-disjoint probe

The split is stated before any fit. Folds are source-video-disjoint via GroupKFold on `video_id`; the
signed axis is pooled across all conditions; every source video is one point. Ridge is the probe, and
its penalty is chosen by an inner GroupKFold on the training sources of each outer fold only, so
held-out sources never influence the penalty. The reported statistics are held-out-source R-squared and
mean absolute error, pooled across conditions. The independent-unit and leakage framing follows Kapoor
and Narayanan (arXiv:2207.07048) and the small-sample error-bar warning of Varoquaux (NeuroImage 2018).

## 5. The pre-registered decision rule

The primary verdict is a positive ("signed axis present above raw coordinates") only if all three hold
at once:

1. Lane A beats Lane C (untrained floor) by at least 0.05 R-squared.
2. Lane A reaches at least 80 percent of Lane B (raw-coordinate null) R-squared.
3. The decoded sign is correct on at least 75 percent of held-out sources.

Missing any one is scored as an informative null. The secondary mirror verdict is separate: the slope of
decoded-mirrored versus decoded-original must be negative and inside the band from minus 1.25 to minus
0.8 (a band around the ideal minus 1) to count as a flip; a near-zero or positive slope licenses only
the weaker statement that the encoding is non-antisymmetric. Reaching 80 percent of the null is a
competitiveness bar, not a superiority bar: Lane A can pass while still sitting below the raw-coordinate
ceiling (Lane B). The point is whether the learned features are competitive with the non-neural ceiling,
not whether they beat it.

The two decisive panels are mocked as `images/fig1.svg` (decodability against the raw ceiling and
untrained floor) and `images/fig2.svg` (mirror-equivariance against the y equals minus x line), and are
reproduced from real or smoke data by notebook 05a.

## 6. Possible futures and what each licenses

Notebook 05b simulates the four canonical futures against the exact margins above and draws their
expected shapes in `images/idea5_possible_futures.png` (top row decodability, bottom row mirror). The
decision rule is total: every outcome maps to an unambiguous, pre-registered claim.

| Future | Shape | Primary verdict | Mirror verdict | Licensed claim |
|---|---|---|---|---|
| F1 clean-flip positive | Lane A near the ceiling, mirror on the y equals minus x line | Signed axis present above raw | Flips | Signed axis is carried competitively AND the encoding is antisymmetric under the mirror (confirms the reflection-equivariant reading, Idea 9 substrate) |
| F2 decodable but non-flipping | Lane A near the ceiling, mirror shallow | Signed axis present above raw | Does not flip | Decodability is licensed; reflection-equivariance is WITHHELD (the encoding decodes side without respecting the mirror) |
| F3 informative null | Lane A a cloud far from the ceiling | Informative null | Does not flip | Negative result: the frozen tokens do NOT add the signed axis above raw coordinates; overturns the belief that the checkpoint organized a laterality axis |
| F4 artifact | Lane A strong BUT Lane D also fires | (withdrawn) | (moot) | The signed claim is WITHDRAWN; a side-agnostic pooled control cannot carry a signed quantity, so Lane A reflects a magnitude or acquisition artifact |

Given the project's prior that asymmetry is the weakest-decoded scalar (R-squared about 0.154), F2 or F3
are the a priori more likely futures, and both are publishable: F2 sharpens the pooled-readout result by
showing side is present but the symmetry is not respected, and F3 is a clean negative for a
representation audit under the ICLR/ICML framing that values informative nulls (`_shared_facts.md`,
reviewer framing). F1 is the strong positive that would promote the reflection-equivariant symmetry-axis
idea; F4 is the trap the nuisance control exists to catch.

## 7. Controls and incorporated repairs

- Bind to ONE fingerprint (`d0acc262`) before any comparison, avoiding the `dba24a`-versus-`d0acc262`
  lineage confound.
- Run the primary comparison on the provenance-matched canonical-path subset so a decoded axis cannot be
  an augmented-versus-canonical acquisition artifact.
- Include the mean/std-pooled negative control (Lane D) that must not recover a signed axis.
- No per-class leave-one-source-out R-squared on n=1 held-out sources; the axis is pooled across
  conditions, every source is a point, and source-level permutation is used only where the number of
  held-out sources makes it meaningful.
- Treat every readout as transductive and say so next to every number; a held-out probe split is still
  transductive because the frozen encoder saw every row during the curriculum.
- Drop the rotation-invariance arm as a finding: the small y-axis training rotation (max 8 degrees) makes
  rotation invariance expected by construction, so it can appear only as a manipulation check, never as a
  falsifiable result.
- Report the mirror honestly: a decodable-but-non-flipping outcome (F2) licenses only "the encoding is
  non-antisymmetric", not automatically a camera or source artifact.

## 8. Threats to validity

- Transductivity. The encoder saw every evaluation row, so no number here is an out-of-sample
  performance estimate; they are representation diagnostics on a frozen encoder. A truly held-out
  estimate would require retraining the whole curriculum inside each outer source split, which this study
  does not do.
- Tiny, unequal source counts. With sources as few as one per class, the pooled endpoint and
  source-as-point plotting are the only defensible readouts; any per-class asymmetry number would be a
  single point dressed as a distribution.
- Provenance and label collinearity. Normal is one video on a mostly-augmented path; the canonical-path
  subset mitigates but cannot fully remove the confound at this sample size.
- Duration warping erases absolute cadence. The 64-frame resize means this notebook can speak to signed
  spatial asymmetry but not to the parkinsonian rhythm biomarker; that boundary is owned by Idea 4.
- Monocular capture. gavd5 is single-view, so the view-stability and genuine-mirror questions can only be
  answered on the external, non-clinical multi-view cohorts, and even there the claim is about the
  reflection property, not diagnosis.
- Skeleton limits. Skeletons cannot recover kinetics, EMG or spasticity, transverse-plane rotation, or an
  etiologic muscle diagnosis, so no clinical-accuracy claim is made on gavd5 at any outcome.

## 9. Reproducibility

- [`../../../nb_05a_signed_laterality_probe.ipynb`](../../../nb_05a_signed_laterality_probe.ipynb): the
  decisive probe. Copies the S-JEPA model classes verbatim so `load_state_dict` matches key for key,
  loads the `d0acc262` checkpoint under the notebook-05 guards, caches the frozen target-encoder
  features, fits Lanes A through D with source-video-disjoint ridge probes, runs the mirror-equivariance
  pass, applies the pre-registered margins, and writes `idea5_signed_laterality_result.json`. It runs in
  `GAVD_MODE=real` (reads the checkpoint and pose cache exactly as notebook 05) and degrades gracefully
  to `GAVD_MODE=smoke`, which reuses the project's synthetic fixtures plus one clearly-labelled signed
  lean overlay so the plumbing runs end to end; smoke numbers are illustrative only.
- [`../../../nb_05b_reflection_reach_and_futures.ipynb`](../../../nb_05b_reflection_reach_and_futures.ipynb):
  the possible-futures simulator (writes `idea5_futures_bundle.json` and
  `images/idea5_possible_futures.png`) and the honestly-stubbed external multi-view reach scaffold.
- Determinism. Seeds are fixed; the smoke lean overlay is deterministic. A future real run diffs its
  `idea5_signed_laterality_result.json` against the four canonical futures in `idea5_futures_bundle.json`.

## 10. References

- Patterson et al., Gait and Posture 2010, PMID 19932621 (gait symmetry ratio biomarker).
- Xiong et al. 2023, PMID 37525241 (Duchenne muscular dystrophy: no significant spatiotemporal asymmetry;
  the myopathy negative class).
- Hausdorff et al. 1998, PMID 9613733; Schaafsma et al. 2003, PMID 12809998 (stride-time variability;
  out of scope here, owned by Idea 4).
- Ranjan et al., GAVD, IEEE Access 2025, DOI 10.1109/ACCESS.2025.3545787.
- Kapoor and Narayanan, arXiv:2207.07048 (leakage taxonomy; source video as the independent unit).
- Varoquaux, NeuroImage 2018 (small-sample error bars).
- Abdelfattah and Alahi, S-JEPA, ECCV 2024, DOI 10.1007/978-3-031-73411-3_21; Assran et al., I-JEPA,
  arXiv:2301.08243; Bardes et al., V-JEPA, arXiv:2404.08471; Assran et al., V-JEPA 2, arXiv:2506.09985.
- Yu, Tan, Tan, CASIA-B, 2006; Takemura et al., OU-MVLP-Pose, 2018 (non-clinical multi-view pose cohorts).
