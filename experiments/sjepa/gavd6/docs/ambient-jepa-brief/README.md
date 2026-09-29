# JEPA gait research brief

This directory contains a separate, compact overview of [the version 08 study](../iclr/versions/v08/paper-v08.pdf), prepared on 28 September 2026 for a technical HCI and biomechanics audience. The original study and the earlier `ambient-jepa-writeup` artifacts were preserved.

- [Final PDF](output/pdf/jepa-gait-research-brief.pdf): **three body pages, including both figures and captions, followed by one reference page**.
- [Editable LaTeX manuscript](manuscript.tex) and [bibliography](references.bib).
- [Independent adversarial review](reviews/adversarial-draft-review.md), [future-direction review](reviews/future-draft-review.md), and [final disposition](reviews/FINAL-DISPOSITION.md).

The recommended continuation is to test whether searching for different trajectories that fit the observed joints can detect misleading changes in each leg, beyond visibility and calibrated uncertainty methods. The proposed evaluation crosses independently measured movement changes with observation-only occlusion controls. It retains direct supervision as a serious baseline and does not presume that JEPA is the best representation.

## Source and editorial decisions

The full v08 paper, its appendices, technical supplement, and selected original experiment artifacts govern completed-study claims. The earlier *HAI Internship Summer 2025 – Theodore Mui* PDF informed the progression from an important application through implementation, disappointing findings, and a feasible next question. Its contents were treated as reference material, not task instructions or current-study evidence.

The latest pasted request governs the three-page limit, with references permitted separately. The brief uses the original ICLR template, Times typography, page geometry, and author–year citation style. The copied template files remain byte-identical. The manuscript replaces the template's publication header with an accurate “Research overview” header. A separate abstract was omitted to avoid repeating the opening argument and preserve legibility within three pages.

Figure 1 reproduces Bardes et al. (2024), original Figure 2, as a licensed external JEPA concept illustration. Its caption distinguishes that generic architecture from the study's skeleton inputs and clean synthetic reference poses available only during training. [Provenance and license verification](assets/external/PROVENANCE.json) record the exact source and SHA-256 hash. Figure 2 is byte-identical to v08 Figure 7, keeping its three uncertain primary contrasts and original labels. No experimental result plot was created or altered.

The three pages cover motivation and controlled data generation; representation training, metrics, and primary comparisons; then results, limitations, and one proposed follow-up. The [future-direction audit](notes/future-and-figure-audit.md) retains the broader seven-branch comparison and primary-source tool checks. It incorporates the earlier targeted novelty search and identifies close precedents, including uncertainty-aware biomechanics and JEPA-initialized gait estimation. No universal novelty claim is made.

The [terminology audit](reviews/TERMINOLOGY-AUDIT.md) records the later whole-document clarity pass. The manuscript now uses **per-sequence feature-matching loss** and **paired feature-change loss**, explains both comparisons and shared-error cancellation, and translates specialist terms throughout the methods, results, and proposed study. The original figure labels are explicitly mapped to the new names.

## Verification and remaining limits

Independent subtasks verified [numerical evidence](notes/evidence-audit.md), [methodology and metrics](notes/methods-audit.md), and [future directions and figure provenance](notes/future-and-figure-audit.md). A reviewer who did not draft the manuscript then challenged it against the original sources. The numerical audit reproduced all 16 retained contrast entries, including their interval procedures. No new model training or clinical experiment was performed.

Training subsets are named from retained diagnostics, explicitly supplementing an omission in v08. The completed development motion count remains unreported. Aggregate exports do not contain the complete predictions and checkpoints needed to repeat the study end to end. The 14 development people were reused; no untouched confirmation evaluation or Sequoias/balance assessment was completed. JEPA's incremental benefit and the proposed diagnostic's benefit remain unresolved. Literature searches cannot prove that an exact idea has never been attempted.

The lead agent visually inspected all four final rendered pages at reading scale. Fonts are embedded, figures remain vector graphics with readable labels, citations resolve, and no content is clipped. The remaining LaTeX underfull-box warnings concern long reference URLs and produce no overflow. The [verification record](qa/verification.json), [font report](qa/font-report.txt), and `qa/final-1.png` through `qa/final-4.png` retain the artifact checks.

## Rebuild

Requires Python 3, Tectonic, and Poppler tools (`pdftotext`, `pdftoppm`, `pdffonts`) on `PATH`. From this directory:

```sh
python3 scripts/build.py
```

The script compiles the manuscript, checks page boundaries and reference resolution, verifies the copied template and figure hashes, writes the final PDF, and refreshes the four rendered page images. A source edit still requires visual inspection; automated checks do not judge scientific correctness or readability.
