# Implementation and verification of the revised manuscript

The complete manuscript implements the preceding review of the original wacv-v1.pdf. The source is fully expanded for direct editing in Overleaf. The main paper and supplement use the corrected October execution, with the earlier RF result clearly separated. No model or classifier was retrained.

## How the revision guide was applied

| Reviewed passages | Complete manuscript |
|---|---|
| R0–R2: title, abstract and contribution | Applied in the title, abstract and Introduction. The clinical background was shortened around verified references, and the final contribution paragraph was tightened to avoid carrying a single line onto the next page. |
| R3a–R3c: related work | Applied in Section 2. OpenPose validation is attributed precisely; MaskFeat has its own citation; Kaur and GaitForeMer provide the closest neurological-gait comparisons. All 16 main-paper citations resolve. |
| R4–R6: masking, training and readout | Applied in Section 4, with architecture introduced before masking so that tokens and feature vectors are defined first. Essential target-input, initialization and gradient details were restored during assembly. |
| R7–R9: baselines and evaluation | Applied in Section 5. The original source-partition counts remain in Table 2. Validation adds 8–9 sources beyond the 24 used for fitting. |
| R10–R12: results, discussion and conclusion | Applied in Sections 6–8, with the corrected results table and historical RF disclosure. Mean pose and the limits of causal or clinical interpretation remain explicit. |
| R13: corrected performance figure | Its verified numbers were rendered as a fold-level dot plot and three confusion heatmaps with a common scale, rather than the interim numerical figure in the guide. |
| R14: supporting analyses | Incorporated into a separate supplement, expanded with the original DINO-style loss, exact masking procedure, source split seeds, RF inputs and fixed readout details. |
| R15–R16: preprocessing and captions | Integrated into Section 3 and the redrawn pipeline and landmark schematic. The dataset table and spatial-normalization equation are retained. |

The original manuscript's reference inventory was reduced together with the corresponding background claims. No numerical citation labels were copied between different manuscript versions. Bibliographic keys and the compiled references determine all citation numbers.

## Independent checks and resolved findings

Parallel agents checked the assembled methods, numerical results and editorial/citation consistency. Their reviews found three small defects, all corrected before the final CLI review target was frozen:

- A mean-pose confusion cell displayed 43.7% because its serialized count was infinitesimally below seven. The exact fraction is 7/16 = 43.75%, displayed as 43.8%. The figure generator now removes floating-point representation noise before formatting labels.
- The mask description incorrectly suggested that overlapping regions themselves increase coverage. The final wording attributes overshoot above 60% to adding a whole final region.
- The supplement called an unperformed augmentation comparison “corrected,” implying that an error had been established. It now says “controlled.”

The methods check confirmed the full, untransformed target input; fresh initialization per fold; the separation between label-free feature training and supervised classification; the RF angle triplets and visibility threshold; seeds; and mask weights. The numerical check confirmed all plotted fold scores and confusion cells, current and historical scores, source/clip counts, training diagnostics and resampling intervals. The editorial check found no unsupported clinical, temporal or causal pretraining claim.

## Compilation and packaging

The main paper has eight pages including references, with the main text ending on page seven. The supplement has three pages. Both use the unchanged official WACV 2027 review style. The main paper contains three figures and three tables; the supplement contains two tables and the detailed equations.

The host inspected every final rendered page. No clipping, overlapping content, unreadable figure cells, unresolved citations or overfull/underfull boxes were found. All PDF fonts are embedded. An encoding warning in the installed lineno.sty comment produces no observed rendering defect.

The Overleaf package was extracted into an isolated directory and both sources compiled there. Extracted PDF text matches the delivered versions exactly. The [delivery record](../qa/delivery-verification.json) contains checksums and page counts; the [package check](../qa/package-check.json) records the isolated builds. The original PDF checksum is unchanged.

The submission identifier remains DRAFT because no assigned identifier was supplied. Ethics-review status and the basis for using the source videos remain author-supplied facts. No unsupported approval, consent or permission statement has been inserted.

## Codex adversarial review

The read-only CLI review examined an immutable snapshot of the assembled sources against the original manuscript and retained evidence, using gpt-6-astra at high reasoning effort. The [manifest](codex-assembly/manifest.json) identifies the exact target and hashes, and the [prompt](codex-assembly/prompt.md) states the review scope. The installed skill helper scripts were unavailable, so the host used an isolated Git snapshot and file hashes, as in the preceding review.

The [unaltered report](codex-assembly/review.md) ends with **Verdict: approve** and identifies no material defect. It confirms the bounded interpretation and agreement between the figure inputs, current results and historical comparison. The [post-run check](codex-assembly/post-run-verification.json) confirms a clean, unchanged target and all 42 recorded hashes. The six assembled source files still match the approved snapshot.

The reviewer could not independently inspect the delegated pose extractor because those additional implementation files were outside its snapshot. The host checked the local call chain: the default SequenceKeypointExtractor requests the full model, its singleton calls the factory without parameter overrides, and the factory specifies image mode, one pose and 0.5 detection/presence thresholds. The [recorded check](pose-front-end-check.json) includes source hashes and line anchors. This establishes the implemented defaults; it does not supply a historical checksum of the downloaded model asset.

Compilation and visual rendering were also outside the CLI review's scope. The completed full-manuscript and isolated-package checks above provide that evidence; they do not rely on the earlier, incomplete fragment preview. Likewise, counting 15 varying RF columns alone would not detect the old duplicate, but the separate feature-column comparison and historical disclosure address that limitation.

The parallel reviewers and CLI reviewer belong to the available Codex model family. Separate contexts add scrutiny but do not provide review by a different model family or independent scientific replication. The approval concerns the assembled sources within the stated evidence, not acceptance by the workshop or clinical validation.
