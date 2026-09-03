# Prospective reviewer scorecard

This is a planning tool, not an acceptance forecast. Scores use the 1 to 4 dimensions in current ICML and NeurIPS reviewer forms, plus reproducibility and feasibility. A 4 means the plan can plausibly meet that dimension if executed as written. It does not mean the result is already known.

## Scorecard

| Proposal | Soundness | Significance | Originality | Clarity | Reproducibility | Three-week feasibility | Main condition for success |
|---|---:|---:|---:|---:|---:|---:|---|
| 05 Temporal readout | 4 | 3 | 3 | 4 | 4 | 4 | Same tokens, matched capacity, timing-first endpoint |
| 04 Motion target | 3 | 3 | 3 | 4 | 4 | 3 | Scale-matched targets and strict inductive retraining |
| 01 Evaluation repair | 4 | 4 | 2 | 4 | 4 | 3 | Reusable split plus a result that changes a prior conclusion |
| 03 Rank audit | 4 | 3 | 3 | 4 | 4 | 4 | Links an axis mismatch to downstream behavior, not only a metric |
| 06 Missingness audit | 4 | 2 | 2 | 4 | 4 | 4 | Includes provenance and source prediction, plus an intervention if live |
| 07 Selective invariance | 3 | 3 | 3 | 3 | 4 | 3 | Separates camera-like nuisance transforms from anatomical changes |
| 02 Construct audit | 2 | 2 | 2 | 4 | 4 | 4 | Remains descriptive and source-level; avoids inferential clinical claims |

## Fatal concern and required repair

### Proposal 05

**Fatal concern:** a temporal head wins because it has more parameters.  
**Repair:** match trainable parameter count and regularization, tune inside training sources only, and include a hand-crafted kinematic baseline plus a temporal-permutation sanity check.

### Proposal 04

**Fatal concern:** motion targets differ in scale, entropy, or noise, so the ablation changes more than semantics.  
**Repair:** standardize targets on training sources, add position-plus-motion and shuffled-motion controls, and keep masks, compute, parameters, optimizer, and steps fixed.

### Proposal 01

**Fatal concern:** the normal-only detector is presented as generalizing to normal people despite one canonical normal source.  
**Repair:** treat the output as a corpus audit, harmonize extraction provenance, use added-normal sources only with explicit caveats, and avoid a clinical anomaly claim.

### Proposal 03

**Fatal concern:** the result is a tensor-axis bug report with no broader lesson.  
**Repair:** show why token-level variance and sequence-level rank disagree, connect the disagreement to readout performance, and test the smallest aligned remedy.

### Proposal 06

**Fatal concern:** a near-chance missingness classifier produces an uninformative null.  
**Repair:** predict source and extraction pathway as well as condition, use source-level permutation, and measure nuisance information inside the learned embedding.

### Proposal 07

**Fatal concern:** synthetic rotation is called cross-view generalization.  
**Repair:** call it a transformation stress test, separate index-preserving camera transforms from anatomical left-right edits, and do not claim real multi-view performance.

### Proposal 02

**Fatal concern:** sequence-level tests treat clips as independent clinical observations.  
**Repair:** show every source mean, mark encoder exposure, treat literature mapping as measurement documentation, and make no population-level significance claim.

## Hard gates shared by all proposals

- Reject a claimed inductive result if the encoder saw the held-out source.
- Reject a confidence interval that uses only seed variation.
- Reject an ablation that changes the named factor plus capacity, scale, compute, or tuning budget without a matched control.
- Reject clinical, causal, invariant, disentangled, calibrated, or world-model language without an operational test.
- Reject a plan with no simple baseline, no Day 5 gate, or no useful null result.

## Official criteria

- [ICLR 2026 Reviewer Guide](https://iclr.cc/Conferences/2026/ReviewerGuide)
- [ICML 2026 Reviewer Instructions](https://icml.cc/Conferences/2026/ReviewerInstructions)
- [NeurIPS 2026 Reviewer Guidelines](https://neurips.cc/Conferences/2026/ReviewerGuidelines)
