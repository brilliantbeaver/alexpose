# RISEx one-page draft package

Start with revision 08: [paper](08_risex_bilateral_motion_evaluation.md), [one-page PDF](08_risex_bilateral_motion_evaluation.pdf), [LaTeX](08_risex_bilateral_motion_evaluation.tex), and [Overleaf package](08_risex_bilateral_motion_evaluation_overleaf.zip). It defines the movement target, describes the implemented training objective and source-separated evaluation, and presents the matched negative result in one table. The abstract and submission metadata below are separate from the one-page body.

Versions 01–07 are retained without overwriting. [00_risex_submission_assessment.md](00_risex_submission_assessment.md) and [00_revision_log_and_adversarial_review.md](00_revision_log_and_adversarial_review.md) record the earlier assessment and drafting history. Their restrictive interpretation of venue fit is superseded by the revision 08 assessment below. [00_title_options.md](00_title_options.md) records the historical title discussion.

`RISEx-2026-1-page-template.docx` is the unmodified official template. Revision 08's LaTeX renderer follows its stated letter-paper dimensions, 20 mm top margin, 15 mm side/bottom margins, 11-point Times New Roman body, 9-point table caption, full-width title/author block and two-column body. It also reproduces the conference header text, which was missing from the earlier LaTeX renderer. It is an author-supplied adaptation, not an official LaTeX template. The generated PDF was checked for a single page, embedded fonts, complete content and visible layout; final comparison with the official Word template remains necessary. Both authors, their shared affiliation and Alexander's corresponding email are included.

Run `bash build_risex_v08.sh` from this folder to rebuild revision 08 only. The build rejects extra pages, overflowing boxes and missing glyphs. [Overleaf instructions](08_overleaf_instructions.md) explain the XeLaTeX setup and the explicit proofing-font fallback when Times New Roman is unavailable. Earlier drafts and their build script are unchanged. The old figure is deliberately omitted from revision 08: the equation and table convey the target and comparison more clearly within one page.

`07_risex_submission_candidate_reference_style_export.docx` is a convenience export made with Pandoc using the template as a style reference. It retains the template headers and embeds the figure, but Pandoc did not reproduce the template's internal two-column section break. It is therefore **not a submission-ready document** and should be used only as a text-transfer aid.

The public RISEx call states an OpenReview submission route and a deadline of 10 September 2026 at 11:59 PM UTC; it requires the template's header and structure to be retained. The downloaded template instead mentions a Google Form. Follow the current call and clarify the archival overlap policy before uploading overlapping work elsewhere. No submission has been made. [Official call](https://conference.albertarobotics.ca/call-for-papers/)

## Revision scores

These scores grade each draft as a RISEx one-page submission, on a 100-point editorial rubric. They are not calibrated acceptance probabilities. The weighting reflects the conference's archival one-page format: scientific focus and evidence (30 points), methodological rigor (25), AI/robotics relevance (20), one-page clarity and visual economy (15), and submission readiness (10).

| Revision | Focus & evidence / 30 | Rigor / 25 | RISEx relevance / 20 | Clarity / 15 | Readiness / 10 | Total / 100 | Assessment |
|:--|--:|--:|--:|--:|--:|--:|:--|
| 01 | 15 | 17 | 7 | 8 | 5 | **52** | Faithful condensation, but its robotics connection and causal scope are weak. |
| 02 | 17 | 18 | 9 | 9 | 5 | **58** | Better human-aware-perception framing; it still relies on broad relevance language. |
| 03 | 20 | 19 | 9 | 10 | 5 | **63** | The matched negative result becomes legible, though key safeguards remain compressed. |
| 04 | 21 | 22 | 9 | 11 | 5 | **68** | Stronger account of preserved token layout, source holdout, and conditional inference. |
| 05 | 23 | 22 | 9 | 13 | 5 | **72** | A focused one-question/table version, with useful economy but less methodological context. |
| 06 | 25 | 23 | 9 | 13 | 5 | **75** | Best treatment of alternative explanations and the bounded pipeline claim. |
| 07 | 26 | 23 | 10 | 14 | 5 | **78** | Previous candidate; its historical score understated venue fit and overlooked imprecise protocol wording. |
| 08 | 26 | 22 | 17 | 14 | 8 | **87** | Recommended current draft: direct human-movement relevance, an explicit target and implemented objective, matched controls, and appropriately bounded inference. |

Scores 01–07 are preserved as historical editorial judgments, not recomputed measurements. Revision 08's higher total reflects clearer reporting, a corrected reading of the venue's scope and a better-checked rendering; no new experiment has increased the evidence. Its rigor score is slightly lower than the earlier rating because the present assessment gives more weight to unresolved preprocessing, readout selection and development-set reuse. The table should not be interpreted as a measured improvement in the underlying science or a probability of acceptance.

The shared title—*Neurologically Motivated Self-Supervised Learning for Bilateral Gait Geometry*—is accurate because neurological conditions motivate the geometric test. It must not be read as claiming that neurological labels or measures enter training.

## Revision 08 assessment

I recommend this version as an ongoing-research submission in human movement understanding. The conference explicitly includes that topic under human-centered robotics, alongside rehabilitation robotics in its health cluster. A robot experiment would strengthen an application claim, but its absence alone does not place this representation study outside the published scope. This corrects the earlier assessment's overly narrow treatment of fit. [RISEx scope](https://conference.albertarobotics.ca/)

The contribution is a controlled evaluation of a training-plus-readout pipeline. The strongest finding is that clip-specific hidden-feature matching can improve while a fixed-form bilateral movement readout becomes less accurate. It is relevant to researchers selecting self-supervised features for monitoring human movement, because success on the training task does not establish accuracy on their intended measurement. The draft should not claim a new JEPA architecture, demonstrated loss of information from the representation, clinical diagnosis, or a validated predictive world model.

The 26/30 evidence score reflects the matched initialization control, all five training conditions and a useful counterexample to relying on the feature-prediction task alone. It stops short of full marks because the cause of the deficit remains unresolved and the result has no independent-cohort confirmation. The 22/25 rigor score credits source-held-out pretraining and readout fitting, nested readout tuning, paired training controls and source-level uncertainty, while retaining the limits below. The 17/20 relevance score reflects the published human-movement scope, rather than an untested robotics deployment. Clarity earns 14/15 because the equation and table make the comparison self-contained, although the methods remain dense. Readiness earns 8/10 with the checked rendering and supplied author details; final official-template compliance and any archival-overlap decision remain outstanding.

### Changes that matter scientifically

Revision 08 replaces the vague speed description with the five-pair equation, observed-transition requirement and original timestamp path. The reflection rule explains why geometry is useful for this test without claiming that the learned features satisfy it. The preprocessing paragraph distinguishes preserving all 33 joint identities and token positions from preserving the original timing, which resampling changes.

The training paragraph now identifies the S-JEPA-inspired implementation, four-layer encoder, two-layer predictor, 96-channel features, teacher averaging, masked cross-entropy and the full-view VICReg regularizer over 12 gait joints. That second objective matters: these experiments evaluate the combined recipe, so a result cannot be attributed to masked prediction alone. The paper states the actual workload of 125 training runs, replacing revision 07's misleading reference to five trained encoders.

The readout paragraph makes the train/test boundary explicit and confirms that trained and initial encoders use the same 2,890-feature summary. The dimensionality increase from 960 to 2,890 applies to a separate mean-only versus motion-summary comparison, not to the trained-versus-initial contrast. Inner folds tune the readout; they do not independently retrain the encoder. The outer held-out videos remain excluded from both stages.

The results retain the full trained range and give an illustrative paired contrast for the motion-family random control: $\Delta R^2=-0.109$, 95% interval [−0.170, −0.041]. This is a reference condition, not a best-performing selected mask. The interval is exploratory and conditional on saved fits. The paper removes the unsupported word “prespecified,” distinguishes five conditions from 125 models and labels the 375 correspondence checks as repeated evaluations.

### Remaining reviewer objections and the next useful tests

The principal objection is that the experiment identifies a deficit in the current pipeline without identifying its mechanism. Input preparation changes the measured target, and 49 of 125 teacher readouts select the maximum tested penalty of 10,000. A poor ridge score cannot establish that the encoder has discarded all movement information. The most informative next steps are to compare timestamp-preserving preparation with the current input path and widen the penalty search using training sources only. Motion-only and missingness-only summary ablations would then help separate useful motion information from readout dimensionality or observation support. These are proposed experiments, not results of revision 08.

The second objection concerns generalization. Ninety-three source videos are the clustered sampling units; repeated seeds, mask draws and fitted models do not create new participants. The same cohort informed successive research decisions, and people have not been verified as distinct across videos. Confirmation should use new, participant-identified sources with a fixed evaluation plan. Clinical claims additionally need affected-side information and an independent movement reference. Forecasting claims need held-out future coordinates and simple last-pose and constant-velocity baselines; masked completion alone does not supply that evidence.

The one-page selection therefore omits synthetic reflection demonstrations, earlier grids with different implementations, uncompleted real-data forecasting and disease-specific claims. It also omits a redundant plot. These choices preserve room for the target, controls, effect estimate and limitations that make the central finding interpretable.

### Evidence and review record

Revision 08 uses the latest completed motion/region comparison summarized through Notebooks 15–18, with preprocessing and readout details checked against the implementation. The exact target definition is inherited from the verified study protocol, rather than redefined to improve the reported result.

| Claim | Retained source or implementation |
|:--|:--|
| Absolute initial/teacher scores and per-seed values | [Figure provenance](../figures/physworld_figure_provenance.json), `numbers.readout_rows` |
| Paired intervals, 125 runs and correspondence counts | [Saved numerical verification report](../physworld_evidence_recomputed.json) |
| Target, preprocessing agreement and maximum-penalty counts | [Full paper](../physworld_revisions/paper_v8.md), methods, results and Appendix A; [tutorial summary](../TUTORIAL.md) |
| Shape-preserving token allocation and teacher construction | [Model](../../laterality/model.py), `SkeletonPatchEncoder` and `JEPA` |
| Actual combined training objective | [Training implementation](../../laterality_extensions/motion_structured_training.py) |
| Matched 2,890-feature summaries and ridge grid | [Readout implementation](../../laterality_extensions/motion_readout.py), `bilateral_summaries` and `RIDGE_ALPHAS` |
| Training-only fitting and inner source folds | [Evaluation implementation](../../laterality_extensions/comparative_evaluation.py), `fit_source_readout` |

The saved verification report records a previous recomputation from 125,000 prediction rows and 125 histories. The underlying motion-structured prediction grid is absent from this checkout. This revision cross-checks the retained numerical records and inspected code; it does not claim a fresh prediction-level recomputation, new bootstrap run or retraining. Restore the recorded grid and run [the verification script](../verify_physworld_evidence.py) before an independent numerical reproduction. The earlier `codex:adversarial-review` skill is unavailable in this session; the review above is a local skeptical reading, not an invocation of that skill or an independent external review.

Primary references checked for this revision: [stroke-related gait asymmetry](https://pubmed.ncbi.nlm.nih.gov/18226655/), [Parkinson's arm-swing asymmetry](https://pubmed.ncbi.nlm.nih.gov/19945285/), [GAVD](https://arxiv.org/abs/2407.04190), [S-JEPA](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04755.pdf) and [VICReg](https://arxiv.org/abs/2105.04906). The clinical papers supply motivation, not validation of the present pose-derived score; the S-JEPA citation replaces the less specific I-JEPA reference in revision 07.

## OpenReview submission text

These fields accompany revision 08 and are not additional sections of its one-page PDF.

### Authors

1. Alexander Mui — Computer Science & Engineering, ASDRP, Fremont, CA, USA. Corresponding author: alexander.mui@students.asdrp.org.
2. Penelope Inouye — Computer Science & Engineering, ASDRP, Fremont, CA, USA.

### Keywords

self-supervised learning; joint-embedding predictive architectures; human movement understanding; gait analysis; bilateral symmetry; skeleton-based representation learning

### TL;DR

Across source-held-out gait videos, masked-feature training produced clip-specific representations but reduced a matched readout's accuracy on a signed left–right movement measure.

### Abstract

Neurological conditions can affect movement differently on the two sides of the body, making bilateral geometry relevant to human-movement understanding and rehabilitation research. We ask whether self-supervised learning makes a simple measure of this difference easier to recover from skeleton features. Using 625 gait clips from 93 source videos, we train a joint-embedding predictive architecture (JEPA) to predict hidden skeleton features. We then fit a linear readout to estimate a signed contrast between left- and right-side movement speeds, comparing trained features with those from the same encoder at initialization. Every test video's clips are excluded from both pretraining and readout fitting. Training produces more clip-specific feature predictions, yet mean source-balanced $R^2$ falls from 0.223 at initialization to 0.101–0.114 across five training conditions. Motion-based and connected-region masks show no clear improvement over matched random controls. Input preparation changes the movement measure, and readout regularization remains a possible contributor to the deficit. These exploratory findings support evaluating self-supervised human-movement features against a fixed geometric measurement alongside their training objective, while leaving clinical usefulness and the cause of the poorer readout unresolved.
