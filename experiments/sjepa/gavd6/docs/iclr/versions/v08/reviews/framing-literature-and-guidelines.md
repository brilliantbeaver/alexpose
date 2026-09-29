# v08 framing: primary literature and current ICLR requirements

Checked 26 September 2026. This review proposes context and bounded interpretation; it does not modify the manuscript, references, experiment, or title. The title remains **Evaluating Feature Prediction for 2D Pose Trajectory Restoration with Paired Synthetic Supervision**. Four new entries are supplied separately in `framing-additions.bib`; existing I-JEPA, V-JEPA, and S-JEPA entries should be reused.

## Central framing recommendation

Connect three levels without identifying them as the same contribution: (1) self-supervised feature prediction and the JEPA family; (2) the broader research program of predictive world models; (3) this study's specific test of whether a decoded pose representation preserves a signed movement response, trajectories, and anatomical side. The implemented system restores one complete observed window using bidirectional context. It does not forecast a future trajectory or execute a learned action-conditioned rollout. Its paired reference poses are privileged synthetic targets, so the completed experiment is not purely self-supervised learning from unlabelled observations.

The clinical motivation concerns reliable measurement of side-specific movement. It is not evidence that the proposed response identifies stroke, Parkinson's disease, osteoarthritis, treatment efficacy, or disease progression. Preserve the distinction between a physical 3D mirror, an input-joint naming error, an algebraic exchange of left/right summaries, and actual clinical laterality. A left/right-invariant representation may discard information needed by the signed readout; this is a task-design concern, not a proven explanation for the fitted results.

## Focused primary-source evidence

| Source and key | Verified evidence and useful claim | Boundary |
|---|---|---|
| LeCun, 2022, *A Path Towards Autonomous Machine Intelligence* (`lecun2022path`) | A position paper proposing self-supervised hierarchical joint-embedding architectures in a broader predictive-world-model program. The author's 2023 seminar explicitly connects that model to predicting action consequences and planning. This supports the intellectual connection between SSL, JEPA, and world models. [Working paper](https://openreview.net/pdf?id=BZ5a1r-kVsf); [author's seminar abstract](https://kempnerinstitute.harvard.edu/events/yann-lecun/). | A proposal, not experimental evidence that this gait system learns a world model. Direct OpenReview PDF requests encountered a browser check; title, date/version, author, and abstract were available through the primary indexed record, with the author seminar independently corroborating the broad claim. Do not cite unverified page-specific details. |
| Assran et al., CVPR 2023, I-JEPA (`assran2023ijepa`, already present) | Predicts target-region representations from a context region in the same image, demonstrating non-generative self-supervised representation learning. [Official proceedings and BibTeX](https://openaccess.thecvf.com/content/CVPR2023/html/Assran_Self-Supervised_Learning_From_Images_With_a_Joint-Embedding_Predictive_Architecture_CVPR_2023_paper.html); [author preprint](https://arxiv.org/abs/2301.08243). | Use as feature-prediction precedent. Same-image masked prediction does not itself establish future dynamics or biomechanical fidelity. The official proceedings confirms CVPR; the arXiv comments field currently inconsistently names ICCV, so retain the verified CVPR metadata. |
| Padmanabhan et al., 2020 (`padmanabhan2020symmetry`) | Nine people after stroke were studied during preferred and visually guided symmetric stepping. Improved step-length symmetry coexisted with substantial kinematic and kinetic asymmetry. This directly motivates checking movement mechanics beyond one symmetry summary. [Full primary article](https://link.springer.com/article/10.1186/s12984-020-00732-z); [metadata](https://pubmed.ncbi.nlm.nih.gov/32746886/). | Small, selected chronic-stroke sample and an acute treadmill manipulation. It neither evaluates pose restoration nor establishes that every symmetry intervention fails. Do not extrapolate its step-length measure to the paper's knee-excursion response. |
| Seuthe et al., 2024 (`seuthe2024laterality`) | In 97 people with Parkinson's disease and 36 controls, gait asymmetry and symptom laterality did not reliably coincide across the measured domains. Only 53.7% had shorter steps on the more affected side. [Full primary article](https://link.springer.com/article/10.1007/s00415-024-12379-0); [metadata](https://pubmed.ncbi.nlm.nih.gov/38652262/). | Gait, turning, and clinical motor ratings are different measures. This supports caution about inferring the affected side, not a diagnostic rule for PD or a mechanistic explanation of naming errors in this model. |
| Creaby, Bennell, and Hunt, 2012 (`creaby2012gait`) | Laboratory comparison of 91 participants with knee OA and 31 controls. Some between-knee biomechanical asymmetries were associated with unilateral pain, whereas the bilateral-pain/structural-OA group had symmetric knee biomechanics in the measured outcomes. [Primary article abstract and exact metadata](https://pubmed.ncbi.nlm.nih.gov/22385873/); [DOI](https://doi.org/10.1016/j.apmr.2011.11.029). | Evidence was verified from the primary abstract, not inaccessible full text. Symmetry can coexist with pathology; interpretation depends on the measured variable and disease/pain distribution. Do not assert a universal OA asymmetry signature. |

These three clinical sources are enough. Do not add a broad disease catalogue or imply all affected people walk asymmetrically. The literature motivates preserving geometry and side assignment so that differences remain interpretable; it does not justify enforcing symmetry as a restoration prior.

## Suggested language and placement

The following is proposed prose, not a claim that the sources tested this model. Integrate it by replacing existing framing rather than appending several disconnected mini-introductions.

**Abstract opening, adaptable to the final word budget:**

> Self-supervised feature prediction with joint-embedding predictive architectures (JEPAs) is a proposed building block for world models. For gait measurement, the learned representation must also support reliable geometry and anatomical side after decoding. We examine this requirement through 2D pose restoration with paired synthetic motion and privileged projected-reference supervision.

Then retain the development sample, the zero-response finding and paired uncertainty, unresolved delta–endpoint comparison, original-package/repair distinction, and explicitly post hoc assignment result. Do not spend the abstract's result budget naming three diseases. A possible closing implication is that these results identify a measurement requirement for predictive movement representations while leaving the benefit of the tested feature-prediction procedures unresolved. Avoid calling the experiment a clinical world-model benchmark.

**Introduction clinical paragraph:**

> Gait analysis requires more than a single symmetry score. After stroke, improved step-length symmetry can coexist with asymmetric joint and whole-body mechanics [Padmanabhan et al.]. In Parkinson's disease, gait asymmetry need not identify the clinically more affected side [Seuthe et al.], while knee-osteoarthritis studies show that the relation between pain and biomechanical asymmetry depends on whether symptoms are unilateral or bilateral [Creaby et al.]. These distinctions motivate preserving trajectories and anatomical correspondence when restoring a pose sequence: a smaller scalar error should not conceal changed movement geometry or an exchange of left and right.

The last sentence is the manuscript's inference from those studies. It must not be presented as a tested clinical outcome. Follow it promptly with the synthetic, projected, nonclinical evaluation scope.

**Introduction representation paragraph:**

> Self-supervised learning offers a way to learn structure from observations through prediction. JEPA predicts in a learned feature space rather than reconstructing every input detail [Assran et al.], and hierarchical JEPA has been proposed as part of a predictive world-model architecture [LeCun]. For quantitative movement analysis, however, abstraction is useful only if the decoded representation retains the geometry needed by the measurement. We study that requirement in an S-JEPA-inspired restoration procedure with privileged synthetic references, rather than assuming that feature predictability establishes movement fidelity.

Keep the established related-work distinctions: S-JEPA uses a target encoder downstream, whereas this adaptation deploys the student; the targets, prediction losses, and task differ from I-JEPA. The clinical paragraph and this paragraph can together replace the current first two introduction paragraphs without changing the empirical question.

**Discussion bridge:**

> The connection to world models is a question about what predictive representations preserve. In the tested restoration setting, response accuracy, trajectory fidelity, and anatomical correspondence do not induce the same ordering. This argues for measuring each when evaluating representations intended to support movement reasoning. It does not establish a learned model of disease or future dynamics: training uses privileged paired synthetic references, and the deployed network restores a complete observed window. Clinical validation would require natural observations, independently measured anatomy, and prospectively specified outcomes in the populations of interest.

The practical inference is bounded by the same 14 development people, three seeds, unequal initial auxiliary influence, clipping, and uncertain primary effects. Do not replace those consequential limitations with generic world-model aspirations. Predictive world models can be learned from passive observations; absence of actions alone is not a universal definition excluding a world model. Here it is the actual task and missing dynamics evaluation that make the broader claim unsupported.

## Current ICLR 2027 requirements and the 9/10-page ambiguity

The current **Paper formatting** paragraph explicitly gives an initial main-text limit of **9 pages** and allows **10** during discussion/rebuttal and camera-ready. Later FAQ and camera-ready wording calls 10 pages identical to submission, an internal inconsistency. The freshly downloaded official example independently confirms 9 initial / 10 later. Keep the initial-submission artifact within nine pages; do not infer a new ten-page initial limit from the inconsistent FAQ. References are excluded, appendices follow references, and reviewers need not read them. Main and supplementary material must be anonymous; self-citations use third person. The requested exact title remains unchanged. [Current author guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines).

AI disclosure remains mandatory in both the paper and submission form; its paper section is excluded from the main limit. Authors remain responsible for AI-assisted claims and artifacts. Preserve the accurate disclosure and do not claim human verification that has not occurred. [Current AI policy](https://iclr.cc/Conferences/2027/AIPolicyForAuthors).

The official example places the AI statement after the body and before references and limits it to one page. Its format remains a 5.5-by-9-inch text region, 10-point body type, and 11-point vertical spacing, with Times New Roman preferred. Figures must be legible, use sufficiently dark lines, keep captions below and with the artwork, and remain interpretable in black and white. Do not shrink or change the official text geometry to absorb additional framing. [Official template ZIP](https://media.iclr.cc/Conferences/ICLR2027/iclr-2027-style-files.zip).

Fresh download: `/tmp/iclr2027-framing-style.zip`. SHA-256 receipts:

| Artifact | SHA-256 |
|---|---|
| Official ZIP | `0d940dfa9398ae99a18f24a85a8a683f367204b6af6d17d2899e60a67102529e` |
| `iclr2027_conference.sty` | `797deef41724e93761426ac0cbcca46279a91cc650dd1f0ce76a4f08d2098ea6` |
| `iclr2027_conference.bst` | `2d67552db7ed38ccfccb5957b52f95656e25c249724761d3cf5f7922ad1844c5` |
| Official example `.tex` | `03e556d6e5593e498fd39f262ec2d184ebfe8693cbf47fb7cea3402d8d5166ac` |

The ZIP and manuscript-used style files match the prior verified official copies. No style-file replacement is needed. After reframing, recheck actual main-page boundaries, references, title and anonymous metadata, figure placement, all newly added bibliography entries, and the source/Overleaf packages against the revised manuscript. No submission or submission eligibility is asserted by this review.
