# Adversarial review of paper_v3.md

Reviewed 9 September 2026. The entire v3 manuscript was read, including its new methods table and exploratory contrast appendix. This review concentrates on inferential correctness and the connection to Physical World AI. The newer masking estimates are accepted subject to the independent extension audit; this reviewer independently checked the core cohort, split, original summary, and input-reconstruction diagnostic.

V3 resolves most of the serious objections to v1. The original-run provenance gap is prominent, its R² estimand is correctly defined, the original and prepared measurement lanes are separate, the initial-encoder control now retains its trained pose detector and fitted ridge readout, and new bootstrap analyses are explicitly exploratory. The added contrast intervals strengthen the report of poorer performance after pretraining under the evaluated readout, without establishing a general information loss. No remaining fatal statistical overclaim was found in the central tables. The following changes would improve precision and the final argument.

## 1. “Preservation of a physical measurement” still overreaches in the abstract

The abstract ends by calling the endpoint a “task-relevant physical measurement.” This quantity combines image-relative x/y with inferred depth and was not independently validated against movement measurements. The methods appropriately call it non-metric, but readers should not need to reach the limitations to qualify the abstract.

Use “coordinate-derived movement measurement” or “an anatomically defined movement contrast.” The main abstract result can remain the discrepancy between clip-specific latent prediction and downstream access to that contrast. That is a credible evaluation finding without implying calibrated physical state reconstruction.

## 2. A small q can arise from a nonzero insensitive representation

The paragraph restricting q to identity-channel transformation is good. It still presents initial q = 0.083 without mentioning that a constant nonzero encoding can be perfectly consistent under this operation while predicting nothing. The zero-energy check in the original code excludes only all-zero degeneracy.

Add a brief warning that q is interpreted jointly with feature variation and predictive utility. The relevant finding is an increase in the chosen discrepancy, not a claim that initialization understands symmetry or that training destroys geometry generally. If showing a threshold, include the learned q interval [0.095, 0.138]; it straddles 0.1 even though its upper-bound acceptance criterion fails.

## 3. Define the feature summaries and capacity difference

V3 says the motion-sensitive summary includes temporal variation, feature changes, and observation support, but the actual feature dimensions and bilateral aggregation remain absent. This matters because increasing feature dimension changes the supervised problem and its ridge penalty. The initial encoder's strong improvement could reflect several parts of that changed readout.

Give a concise formula or appendix table for the mean and motion-sensitive summaries, including their dimensions. Keep the support-feature confound and add capacity/regularization to that same sentence. A future comparison should remove support features, compare matched dimensions or controlled projections, and select every regularization choice on training sources only. The paper can acknowledge this without running further experiments during revision.

## 4. Explain the coordinate convention and time convention together

The preprocessing table is a useful addition. A reader still cannot reproduce the target's norm from the phrase “estimated coordinates.” The archive extraction normalizes x by image width, y by image height, and inferred z by width after crop correction. Elapsed time comes from frame-number increments divided by frame rate. Uniform relative-index resizing in the model lane does not preserve that timing.

An appendix should record these conventions, epsilon, the bilateral body-scale rule, and the target/input distinction when pelvis landmarks are missing. This is consequential because image aspect ratio, detector error, and missing pelvis observations can affect the endpoint and eligibility even when the mirror algebra is exact. Do not call the normalization a recovery of calibrated physical geometry.

## 5. The validity-mask table has one implementation-dependent phrase to refine

The last row of the shape table says “Gathering and padding retain each clip's own targets and common mask count.” That is not a general description of the original full-token-grid implementation, whose targets have a shared selected count and are gathered only after token processing. The word “padding” can suggest arbitrary padded target values enter the loss.

Use a direct description such as “The full joint/time grid remains allocated; target selection and validity identify the values used by the loss.” If a later trainer does pad variable target sets, describe that case separately and state how padded entries are excluded. The four-frame patch row should say “four three-coordinate observations,” rather than “four coordinates,” to make its final dimension 12 immediately clear.

## 6. Condition motivation should distinguish asymmetry constructs

The planned condition-specific motivation is worthwhile if it remains a literature-based explanation. The endpoint averages signed speed contrasts over pairs and over each clip, so opposite affected sides or opposite pair-level deviations can cancel. A near-zero y can accompany substantial temporal, spatial, phase, or absolute asymmetry. Shoulder movement is also only an indirect link to literature on arm-swing amplitude.

Make these distinctions in the motivation and target limitations. Avoid claiming that the project has discovered asymmetries associated with any listed condition. The clinical literature should justify studying bilateral structure, while this experiment tests one measurement and a representation pipeline.

## 7. Clarify the contribution through a falsifiable next comparison

The revised manuscript is honest about having no demonstrated gait forecasting, planning, or multimodal fusion. Its workshop relevance still needs a more direct articulation than an assertion that the findings could matter to future dynamics models.

State the reusable contribution as a linked evaluation: a known observation transformation, a common target that transforms predictably, a paired initialization comparison, and a distinction between feature-prediction diagnostics and measurable output utility. Then propose one decisive prospective follow-up: retain a fixed input/target path and readout evaluation, introduce an explicit reflection objective or architecture, and test whether it improves both useful prediction and transformation behavior on held-out sources. A positive q-only result would not refute the observed utility gap; improved held-out target recovery with preserved variation would be more informative. This gives the research trajectory a clear intellectual consequence without asserting that an untested repair will work.

## Smaller editorial and evidentiary refinements

- Number the new measurement-path subsection 4.1 and renumber the rest; 4.0 makes the paper look like an append-only working log.
- The abstract's “information already present at initialization” is acceptable only as access under the evaluated feature construction. “Improves the initial-encoder readout” is narrower and less likely to be interpreted as an information-theoretic finding.
- Define “enrichment” in the mask audit: it is a difference in the implemented token-motion statistic, not a percentage increase or a biological motion measure. State whether negative values are possible and keep the source-weighting rule in the appendix.
- Explain that a zero seed SD for deterministic baseline rows means the predictions were reused and did not depend on optimization seed; it is not zero sampling uncertainty. V3 already notes reuse, so a short table-caption clarification is sufficient.
- The causal language “predictor training can succeed” should stay tied to the own-source versus other-source target diagnostic. This diagnostic can be driven by static posture, acquisition, or support. V3 handles this well; later revisions should resist tightening it into evidence of learned dynamics.
- The reproducibility sentence “Original notebooks and experiment artifacts are preserved” remains ambiguous after the explicit statement that the historical report/checkpoint chain is absent. A manuscript can instead state which artifacts are available; the revision README can say existing files were left unchanged.
- Figures should use actual saved aggregates, with the initial/trained and alternative/reference pairings clear. A constructed toy trajectory needs a visible explanatory label; historical per-clip examples cannot be reconstructed without the absent prediction chain.

The planned v4 additions on condition-specific motivation, endpoint cancellation, authoritative related work, and a falsifiable next experiment are appropriate. They should supplement the current numerical evidence rather than expand the empirical claim from a small pose-representation audit into validated clinical gait modeling or a general finding about world models.
