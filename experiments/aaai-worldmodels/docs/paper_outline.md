> Historical planning outline. The completed, independently reviewed results draft is [manuscript.md](manuscript.md).

# A paper built from the evidence

Working title: **Learning Normal Articulation: A Source-Grouped Protocol for Predictive Gait Representations**.

This title describes an implemented direction without asserting clinical accuracy. If completed repeated-seed comparisons show a reliable benefit, revise the title and claims around that finding. If the pilot remains weak or inconclusive, a position paper can still explain why data grouping, causal prediction boundaries and uncertainty evaluation matter for medical world models.

## Opening argument

Video-derived skeletons are compact observations of movement. A normal-motion model could provide a reference for measuring departures and support inspection of where predictions fail. However, web-video datasets offer annotation labels rather than verified diagnoses and contain many correlated clips. A method can appear accurate through source recognition, camera effects or pose-detection failure. Evaluation must distinguish those explanations from learned movement structure.

## Contributions that are already concrete

1. An auditable fresh inventory and source-grouped benchmark specification for five requested GAVD labels, including media exclusions and source counts.
2. A laptop implementation of uniform-mask skeleton JEPA, normal-only pretraining, a preserved anomaly reference and separately adapted representations, with explicit deviations from published methods.
3. A matched RF/JEPA evaluation pipeline, future-only leakage tests, persistence and untrained-encoder controls, and a documented inability to obtain a useful 5% normal-only calibration threshold from a tiny source cohort.
4. Independently runnable teaching artifacts and a provenance-recording execution workflow.

These contributions establish a protocol and implementation. They do not yet establish superior classification or useful uncertainty.

## Recommended structure

For a four-page position paper, use roughly one page for the clinical-observation problem and data limitations, one for the predictive representation and causal boundaries, one for the evaluation protocol and bounded pilot, and one for failure analysis and research agenda. Use the official workshop requirements to confirm page limits and formatting.

For a full empirical paper, add completed repeated-seed matched comparisons, ablations, uncertainty evaluation and failure examples before asserting a method contribution. Report the source-level support next to every result table. Explain the one-source cerebral palsy test limitation directly.

## Main table design

Rows: RF, untrained transformer probe, normal-pretrained probe, progressively adapted probe, mixed-adapted probe. Columns: test sources, binary balanced accuracy, AUROC, macro F1, paired binary-accuracy difference against RF and training time. A second table reports future-coordinate MSE for the trained forecast and persistence, on identical valid target joints and equally weighted sources.

Never place synthetic learning curves in a table labeled clinical performance. Never combine scores from exp5 or an older ASTAD study with current runs as if they shared the same cohort. If a result is unavailable, write unavailable and explain the prerequisite.

## Figures with evidence roles

Use `images/source_split.svg` to explain how related sequences stay together. Use `images/jepa.svg` for the architecture, while retaining the normal-only versus adapted checkpoint distinction. A real result figure should include per-source paired predictions or residuals tied to saved IDs. A forecast figure should mark the observed prefix and withheld future, with prediction and persistence plotted on the same coordinates.

## Falsifiable hypotheses

H1: Pretraining improves held-out source classification over the same untrained architecture. H2: The normal forecast improves future-articulation error over persistence. H3: Conditional latent sampling improves a declared uncertainty metric beyond deterministic errors and simple confidence features. H4: Any apparent improvement persists after camera-view and pose-quality sensitivity checks.

Negative results reject or qualify these hypotheses. Test outcomes must not select the architecture, threshold or source split. A changed method requires a new declared run identity and a clearly separated development cycle.
