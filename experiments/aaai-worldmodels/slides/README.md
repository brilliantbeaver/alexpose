# Reference-aware gait prediction and anomaly detection

Open the [current 40-slide illustrated tutorial](reference_v2.html) or its [vector PDF](reference_v2.pdf). [tutorial.html](tutorial.html) opens the current deck. The slides explain the completed `medwm-reference-masking-v2` study in full sentences, with editable diagrams, real observation masks, complete comparison arms, and evidence links beneath each page.

Current-posture prediction reduces held-out normal-source error by 40.8% relative to persistence. This useful forecasting capability is distinct from anomaly ranking: the locked conditional two-sided detector has within-fold source AUROC 0.462, compared with 0.811 for low motion. Frozen external transfer has coded-participant AUROC 0.591 versus 0.762 for the same comparator. The deck retains the paired uncertainty, stronger current-posture forecast, masking controls, and acquisition explanations that determine those findings' meaning.

Two added mean-prior controls have their own post hoc analysis identity. Current-posture prediction outperforms both a constant normal future pose and the latest anchor plus normal mean displacement on the same held-out normal sources. Slide 19 marks those outcome-exposed diagnostic comparisons with orange diamonds and preserves their distinction from the original core questions.

## Follow the argument

| Slides | Purpose |
| --- | --- |
| 1–5 | State the contribution, physiological motivation, independent units, and full-inventory attrition. |
| 6–10 | Explain Ambient ingestion, native frame numbering, the 16-past/8-future contract, body reference, missingness, and hierarchical error. |
| 11–13 | Define strong forecasting alternatives, the compact posture/velocity/dispersion/support state, and grouped evaluation. |
| 14–17 | Show the seven masks, real observation support, protected anchors, geometry controls, and shared neural training. |
| 18–23 | Report forecasting, paired effects, every mask, repeated partitions, and the movement-amount diagnostic. |
| 24–28 | Explain disjoint calibration and ranking, then display the primary endpoint and every fold. |
| 29–31 | Report shared observation stresses, view/matching, and feature diagnostics. |
| 32–37 | Explain external preparation and participant aggregation, complete transfer controls, domain shift, and the pre-score identity sensitivity. |
| 38–40 | Connect notebooks 20–26, gated extensions, and immutable evidence identities. |

The internal cohort contains 2,815 windows from 456 sequences and 91 source videos, including 29 normal sources. External transfer contains 773 windows from 168 recordings and 90 dataset-coded participants, including 27 normal, 49 osteoarthritis, and 14 Parkinson participants. These are different populations and grouping units; the deck never adds their counts into one sample size.

GAVD supports a heterogeneous public-video benchmark, but GAVD alone does not establish patient-disjoint, site-disjoint, or prospective clinical generalization. Source IDs are not verified patient IDs, annotation folders are not independently confirmed diagnoses, and all local normal sources already lie in the five-label scope. External coded grouping adds a different dataset, while age, native frame rate, framing, observation attrition, uncertain identity continuity, and possible undetected overlap still limit clinical claims.

## Reproduce the deck

From the repository root, use the experiment's Python 3.12 environment:

```bash
experiments/aaai-worldmodels/.venv/bin/python experiments/aaai-worldmodels/scripts/build_reference_v2_slides.py
```

The generator refuses missing, incomplete, synthetic-only, or hash-mismatched inputs. It requires these completed local runs:

- `reference-v2-core-v1` supplies forecasting, all mask/seed candidates, separate calibration families, and source-level ranking.
- `reference-v2-transfer-v1` supplies frozen external scoring without target-domain adaptation.
- `reference-v2-robustness-v1` supplies shared corruption, view/matching, and feature diagnostics.
- `reference-v2-identity-sensitivity-v1` supplies the separately declared participant exclusion analysis.
- `reference-v2-mean-prior-posthoc-v1` supplies the added outcome-exposed constant-pose and mean-displacement controls.

The audited cohort and prepared pose caches must also be available for the real-mask example. The example uses the first source, sequence, and window in lexical identity order with mask seed 27. The calibration diagram uses the first component in the frozen calibration index. Neither illustration is selected for model performance. The generator fits no model, downloads no media, and never changes run artifacts.

The [source generator](../scripts/build_reference_v2_slides.py) produces [editable SVG, PDF, and PNG pages](reference_v2_images/), the [complete slide source](reference_v2_source.json), the HTML navigator, and the PDF. [The manifest](reference_v2_manifest.json) pins generator, input, and output hashes. [Numeric checks](reference_v2_numeric_checks.json) independently replay the four principal ranking points with scikit-learn and every normal eight-offset forecast mean from source rows. [Layout checks](reference_v2_layout_checks.json) test rendered text against its declared containers. [The visual review](reference_v2_visual_review.md) records actual page inspection separately from programmatic checks.

The 95% intervals resample source components or coded participants and condition on saved fitted models. They do not include full model-development uncertainty, complete retraining variation, unknown participant overlap, or clinical measurement error. Seven or eight normal calibration sources give coarse empirical ranks, without automatic conformal coverage or a supported clinical threshold.

## Preserved decks

The [iteration02 HTML](iteration02.html), [iteration02 PDF](iteration02.pdf), [restart HTML](restart.html), [restart PDF](restart.pdf), and [pre-restart entry point](tutorial_before_restart.html) retain their original study identities and evidence boundaries. The iteration02 masking diagrams were proposals; the new reference-v2 deck reports a completed comparison under its own identity. Historical results and proposed extensions are not relabeled as new executed evidence.
