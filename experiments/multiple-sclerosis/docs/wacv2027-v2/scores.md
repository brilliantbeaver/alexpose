# Scores across the seven revised drafts and the requested v9 follow-up

These are editorial judgments on the evidence and presentation, not official WACV scores or acceptance probabilities. The rubric follows the scope of the [Vision4Neuro call](https://cvpath-vision4neuro.github.io/wacv2027/vision4neuro.html) and priorities in the [WACV reviewer guide](https://wacv.thecvf.com/Conferences/2027/ReviewerGuidelines). The [venue record](evidence/venue-and-literature.md) documents the derivation. Each draft was reviewed before the next revision was produced.

## Rubric

| Category | Weight | What is assessed |
|:--|--:|:--|
| Relevance | 15 | Useful contribution to behavioral or biomedical vision within the demonstrated scope. |
| Correctness | 25 | Faithful numbers, methods, comparisons and uncertainty. |
| Evaluation | 20 | Appropriate separation of sources and model selection, with defensible generalization claims. |
| Reproducibility | 15 | Sufficient procedural detail and traceable artifacts. |
| Clinical interpretation | 10 | Accurate label/provenance statements and responsible limits on clinical inference. |
| Literature and presentation | 15 | Verified attribution, clear writing and readable WACV layout. |

Ratings range from 0 to 5. A 0 denotes absent or materially false evidence; 1 a severe deficiency; 2 substantial work needed; 3 a defensible account with important limits; 4 a strong, clearly supported account; and 5 unusually complete support for the stated scope. Decimal ratings permit finer editorial distinctions. Total = sum of each weight multiplied by its rating and divided by five. A high total does not override a material unsupported claim.

## Category ratings

| Draft | Relevance | Correctness | Evaluation | Reproducibility | Clinical | Literature / presentation | Total /100 |
|:--|--:|--:|--:|--:|--:|--:|--:|
| [v2](wacv2007-v2.pdf) | 3.8 | 4.2 | 2.9 | 3.6 | 4.0 | 3.8 | **74.2** |
| [v3](wacv2007-v3.pdf) | 4.0 | 4.6 | 3.8 | 3.9 | 4.0 | 3.7 | **81.0** |
| [v4](wacv2007-v4.pdf) | 4.0 | 4.6 | 3.8 | 4.3 | 4.0 | 4.0 | **83.1** |
| [v5](wacv2007-v5.pdf) | 4.0 | 4.7 | 3.8 | 4.5 | 4.0 | 4.5 | **85.7** |
| [v6](wacv2007-v6.pdf) | 4.0 | 4.7 | 3.8 | 4.5 | 4.0 | 4.5 | **85.7** |
| [v7](wacv2007-v7.pdf) | 4.0 | 4.7 | 3.8 | 4.5 | 4.0 | 4.5 | **85.7** |
| [v8](wacv2007-v8.pdf) | 4.0 | 4.7 | 3.8 | 4.5 | 4.0 | 4.5 | **85.7** |
| [v9](wacv2007-v9.pdf) | 4.0 | 4.7 | 3.8 | 4.5 | 4.0 | 4.6 | **86.0** |

The original v1 received [46.0/100](evidence/original-draft-review.md) and is additional context, not one of the seven new drafts. The category names were standardized as reviews progressed; the six priorities and weights remained the same. Scores plateau from v5 through v8. The requested v9 gains 0.3 points solely in presentation for a clearer, constructive account; no empirical or clinical category increases. Its separate presentation reviewer retained 85.7, illustrating the judgment involved in these small editorial differences. The table records the final scientific/numerical review's score. Unrun controls, uncertain identities and clinical-label limitations continue to restrict the evidence.

## Critiques and subsequent changes

| Reviewed draft | Main findings and suggestions | Response in the next draft |
|:--|:--|:--|
| [v2 review](reviews/v2-review.md), [Codex challenge](reviews/codex-v2/review.md) | Main figures corrected, but paired uncertainty, fold/class errors and exact artifact pointers were incomplete. Quantify the separate corrected RF check. | v3 adds the completed paired bootstrap, fold and error displays, corrected RF figures and an explicit primary-versus-verification distinction. |
| [v3 review](reviews/v3-review.md) | Explain how hidden observations are withheld, complete mask diagnostics and RF inputs, identify the splitter and evidence record, and repair pipeline text collisions. | v4 expands those methods, clarifies pixel versus body-relative coordinates, explains the reflection correspondence issue and redraws the pipeline with aligned boxes and clear spacing. |
| [v4 review](reviews/v4-review.md) | Fold labels were inconsistent; interpolation burden was implied without data; terminology and development-history language impeded general readers. | v5 uses fold indices 0–4, states the missing interpolation statistics, explains elapsed-time variation and the classifier/score in ordinary language, and completes the reproducibility record. |
| [v5 review](reviews/v5-review.md), [Codex challenge](reviews/codex-v5/review.md) | Numerical and scientific claims hold. Finish terminology cleanup, define visibility and give a concrete source-weighting example. | v6 explains visibility, adds the 13-clips-at-1/13 weighting example, removes remaining unexplained classifier shorthand, and preserves exact equations in a methods companion. |
| [v6 review](reviews/v6-review.md) | Replace “tracked” with “estimated” to match frame-independent pose estimation; explain masking and retrospective resampling more directly. | v7 uses “estimated,” defines a mask before its rule, explains the 36-draw limit and fixed-prediction resampling, and removes local machine paths from the companion record. |
| [v7 review](reviews/v7-review.md), [Codex challenge](reviews/codex-v7/review.md) | Scientific claims remain supported. Explain the untested confidence/accuracy relationship and simplify the conclusion. | v8 replaces unexplained calibration terminology with its practical meaning and gives a clearer, bounded conclusion without new empirical claims. |
| [v8 final review](reviews/v8-review.md) | No remaining numerical, scientific-framing or layout blocker found. All seven pages and retained source hashes were checked. | Preferred final draft. Further empirical improvement requires the specified new experiments or provenance verification; another wording pass cannot supply them. |
| [v9 review](reviews/v9-review.md), [presentation review](reviews/v9-presentation-review.md), [Codex challenge](reviews/codex-v9/agent-review.md) | User requested a positive RF/learned-feature emphasis and less statistical jargon. Reviews accepted the bounded interpretation and requested clear RF-defect disclosure and a narrower reproducibility statement. | Final v9 incorporates those refinements and omits an unnecessary training-error interpretation. It retains all corrected results and counter-comparisons, includes an exact methods companion, and has a locally tested Overleaf ZIP. |

The [adversarial disposition](reviews/adversarial-disposition.md) maps Codex findings to fixes and reports what each review did not assess. Separate agents reviewed notebooks, numerical evidence, sources and rendered pages. These reviews were performed by AI agents; they are not conference decisions or completed human peer review.
