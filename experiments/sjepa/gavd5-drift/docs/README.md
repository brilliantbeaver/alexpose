# Documentation and paper package

This folder documents the current **GAVD-only, non-augmented-normal** run. The optional project-labeled normal-video cohort is disabled and is not used in any current result.

## Current evidence boundary

The raw GAVD inventory contains 666 sequences from 103 source videos. A video availability check retained 642 sequences from 94 videos. Pose-quality checks then removed 16 low-coverage sequences, so the trained and analyzed cohort is **626 sequences from 93 source videos**.

|Condition|Sequences|Source videos|
|---|---:|---:|
|Normal|270|29|
|Parkinson's|41|9|
|Stroke|74|18|
|Myopathic|183|28|
|Cerebral palsy|58|9|
|**Total**|**626**|**93**|

The current final checkpoint is `sjepa_curriculum_final.pt`.

- Experiment fingerprint: `7d13841aceac9eda843d43ca8434193e294d2fa10a48b6c6d21f6413a6e457e2`
- Checkpoint SHA-256: `64008d77689cefa4beb51a0dcf5ed6cae743454134c163e9087f66510af4e7ad`
- Training seed: 42
- Optional augmented-normal cohort: off
- Total training: 600 curriculum epochs and 40,800 optimizer updates

## Results that are verified for the current run

Across the 270 matched normal sequences, the mean cosine between each current representation and its own Stage-0 representation is **0.7002, 0.5021, 0.3962, and 0.2966** after the four later stages. This is not a cosine between cohort centroids. Reloading the saved checkpoints reproduces these values within `4.51e-7`.

This is verified **coordinate drift**, not yet verified forgetting. A cosine between raw latent coordinates can fall because the whole representation basis rotated. The current run does not include an alignment-invariant comparison or a held-out normal-function test.

The anchor is sequence-weighted. The two largest normal videos supply 105 of 270 anchor rows, and the three largest supply 137. Report equal-video-weighted and per-video results before making a population-level drift claim.

Across the 642 availability-filtered pose caches, 546 carry extraction label `gavd5`, 95 carry `gavd3`, and one carries `gavd4`. After coverage filtering, the 626 modeled rows contain 530, 95, and 1, respectively. Their recorded pose-model hash agrees, but extraction provenance remains a potential shortcut until it is controlled.

The final in-corpus, label-informed geometry has cosine silhouette 0.3617, minimum between-centroid distance 0.0863, and mean within-condition distance 0.0783. These are descriptive training-corpus measurements.

The sequence-split Random Forest reaches 0.920 accuracy and 0.899 macro-F1 on all 626 rows. That score is not a generalization estimate: all 188 classifier test rows were used to train the encoder, and 64 source videos occur on both sides of the classifier split. The missingness-only control reaches 0.441 accuracy and 0.355 macro-F1 on the same split.

## Results that must not be reported as current

Several files are historical or mixed-lineage artifacts:

- `*_augmented.*` belongs to the discontinued 159-sequence augmented-normal run.
- `lane_c_video_disjoint_metrics.csv` still belongs to that augmented run.
- `anchor_guard_results.json`, `anchor_drift_margin_ablation.csv`, and `sjepa_anchor_guard.pt` do not carry a valid current-run lineage. Notebook 08 still loads an augmented Stage-0 checkpoint in its ablation path and contains stale hard-coded fingerprints.
- `predictive_surprise_results.json` was produced with the old augmented checkpoint while evaluating the current canonical rows. Notebook 09 still hard-codes that old checkpoint and fingerprint.

The paper and maintained documents exclude all claims based on those files. They may be used only after the notebooks are fixed and rerun.

## Document map

- [bbfm2026_paper_draft.md](../neurips-brain-body/docs/bbfm2026_paper_draft.md): workshop paper draft; verified evidence only.
- [bbfm2026_paper_draft.tex](../neurips-brain-body/docs/bbfm2026_paper_draft.tex): anonymous workshop-style LaTeX build of the paper.
- `neurips_2026.sty`: official modified style downloaded from the workshop's [format link](https://brainbodyfm-workshop.github.io/assets/styles/brainbodyfm-neurips-2026-style.zip).
- [neurips-brain-body.md](../neurips-brain-body/docs/neurips-brain-body.md): plain-language research and submission-readiness guide.
- [progressive_training.md](progressive_training.md): exact current training contract and stage diagnostics.
- [staged_details.md](staged_details.md): step-by-step method and evaluation tutorial.
- [staged_evolution.md](staged_evolution.md): history of the project, including why the augmented-normal branch was retired.
- [staged_sjepa_gait.md](staged_sjepa_gait.md): current technical-report version of the older staged paper.
- [downstream_probe_reproduction.md](downstream_probe_reproduction.md): exact probe rerun and exposure audit.
- [tutorials/sjepa_model_internals.md](tutorials/sjepa_model_internals.md): model, masking, loss, and tensor-flow reference.

## Rebuild figures

From the `experiments/sjepa/gavd5-drift` experiment root:

```sh
MPLCONFIGDIR=cache/matplotlib .venv/bin/python docs/make_figures.py
MPLCONFIGDIR=cache/matplotlib .venv/bin/python docs/make_brainbody_figures.py
MPLCONFIGDIR=cache/matplotlib .venv/bin/python docs/make_downstream_probe_figure.py
```

Figure generators must reject a mismatched checkpoint, fingerprint, cohort, or stale secondary artifact. A failed generator is a data-lineage failure, not a cosmetic problem.

Only the four figure families listed in [figures/README.md](figures/README.md) are maintained for the current report. Other top-level figure exports are historical and must not be copied into the paper.

Build the workshop-format draft from `neurips-brain-body/docs/`:

```sh
cd neurips-brain-body/docs
tectonic bbfm2026_paper_draft.tex
```

The current anonymous build has four main-text pages. Appendices and references begin on page 5, and the PDF has six pages total.

## Workshop status

BrainBodyFM 2026 accepts at most five pages, excluding references and appendices, in its modified NeurIPS 2026 style. The deadline is September 5, 2026 AoE. Submissions are double-blind and non-archival. See the [official call for papers](https://brainbodyfm-workshop.github.io/call-for-papers).

The research direction fits the workshop, especially its topics on behavioral signals, continual learning, interpretability, evaluation, and reproducibility. The paper now builds in the official anonymous style and fits the page limit, but it is **not submission-ready as a strong empirical paper**. It still needs, at minimum, a valid multi-seed and alignment-aware drift study. A carefully labeled work-in-progress submission is possible after the ethics, anonymization, citation, and artifact checks listed in [neurips-brain-body.md](../neurips-brain-body/docs/neurips-brain-body.md).

## Author checks before submission

1. Rebuild the included official-style LaTeX source and confirm that the main text remains within five pages.
2. Do not modify `neurips_2026.sty`; it is the workshop-provided file.
3. Remove author names, affiliations, repository-identifying links, and PDF metadata for double-blind review.
4. Confirm data-use, consent, licensing, funding, and acknowledgment language.
5. Do not describe the current classifier scores as unseen-video, unseen-person, clinical, or diagnostic performance.
6. Do not describe raw anchor cosine decline as catastrophic forgetting until the alignment and functional-retention controls are complete.
