# Review record for the neurological research extension

> **Stage note.** This document records the proposal and its review before the expanded real-data study. The [current executive summary](EXECUTIVE_SUMMARY.md) and [draft01](../draft01/README.md) describe the completed experiments and revised conclusions.

Reviewed November 14, 2026. This is a documentation and research-design update. It does not report new video processing, training, or performance.

## Evidence and consultations

The clinical review checked 14 primary studies through PubMed or equivalent primary records, with full discussions consulted for selected studies. Official NIH material supports the broad disease definitions. The study ledger distinguishes observational associations, simulation consequences, and clinical measurements. This targeted review is not a systematic review or formal assessment of clinical evidence quality.

The methods review checked primary descriptions of OpenSim Moco, MyoSuite, Generative GaitNet, Bidirectional GaitNet, and recent physiological simulation papers. These sources establish substantial prior art. The proposed ambiguity benchmark and measurement-selection idea remain candidate contributions, not verified priority claims.

A final search checked prior probabilistic musculoskeletal analysis, measurement/model uncertainty, a gait-parameter identifiability case study, and Bayesian gait-state estimation across sensor configurations. The novelty comparison was narrowed accordingly. This prevents treating uncertainty or measurement selection alone as the new contribution.

A third real Claude CLI consultation completed successfully and reported `global.anthropic.claude-opus-4-8`. It had no tools. GPT-6 Astra and GPT-6 Sol reviewed methods and feasibility; a GPT-6.1 Sol advisor reviewed clinical evidence. Their raw advice and the [accepted synthesis](advisory_summary.md) are preserved separately.

## Corrections made during independent review

- Replaced a causal account of post-stroke compensation with the observational association actually supported by the study.
- Distinguished arm angular-velocity amplitude from a coordination statistic and from a claim of no medication effect.
- Described the CP muscle-volume study as an age-related observational comparison.
- Kept myopathic gait separate from a single neurological or Duchenne diagnosis.
- Added the two-sided score prospectively while preserving the earlier one-sided comparator.
- Matched the main raw-error baseline to the profile's joints, weights, support, and aggregation.
- Added normal-reference sensitivity, minimum support, and fixed-observation-budget checks.
- Distinguished a four-frame context control from a purely static predictor.
- Restricted the linear four-loss decomposition to signed contrasts unless the nonlinear score is explicitly recomputed.
- Prevented the future measurement selector from reading hidden generating parameters or unrevealed measurement outcomes.

## Figures and document checks

Eight new original figures were generated as SVG, vector PDF, and PNG, with a local HTML gallery. Their contact sheet and revised individual figures were visually inspected. Independent review caught two legend overlaps and an ambiguous validation arrow. The diagrams were regenerated after those corrections. All toy curves, shaded reference bands, and candidate mechanism pairs are identified as illustrations.

Local Markdown paths and heading anchors were checked. All nine SVG files in the folder parsed successfully. Existing evidence records retained their verified artifact hashes, and the original readiness score remains 44.5 before rounding. The new consultation and figure provenance is recorded in [evidence.json](evidence.json).

The resulting protocol remains exploratory. Scarce normal sources, uncertainty in reference estimation, unknown repeated participants, source/view differences, and the lack of physiological measurements limit the conclusions. No test suite was rerun because the model implementation and notebooks were not changed in this extension.
