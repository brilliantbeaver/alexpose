# GaitParity tutorials

This short series documents the fixed-reflection baseline workflow and its
evidence. You do not need a background in artificial intelligence, biomechanics,
or statistics to follow it. The proposed semantic-gauge paper has a separate
[tutorial space](./semantic-gauge/) and must pass its decision gates before a
training runbook is added.

- [From video to a movement model](./01-foundations-notebooks-00-06.md) explains how the project turns ordinary walking video into a model that can notice patterns of movement.
- [What the current experiments tell us](./02-current-direction-and-results.md) explains the first test of left-right reasoning, including an important result that did *not* support the hoped-for claim.
- [What the planned fixed-reflection study will test](./03-planned-research-direction.md) explains the retained force-validation route.
- [Reading the notebook graphs](./04-notebook-graph-guide.md) explains the saved graphs from the research notebooks in everyday language.
- [Two-stage AMASS training](./05-two-stage-amass-training-overview.md) explains why the proposed full study first learns from broad motion and then continues on walking-focused motion.
- [Stage 1: broad AMASS pretraining](./06-stage-1-broad-amass-pretraining.md) gives a preparation and HAIC runbook for general-motion pretraining.
- [Stage 2: BABEL-guided walking continuation](./07-stage-2-walking-focused-continuation.md) explains how to build the walking corpus, continue matched checkpoints, and run the second stage on HAIC.
- [Downstream evaluation and baseline roadmap](./08-downstream-evaluation.md) audits the seed-7 pilot and gives the gated path from repaired pretraining to video-disjoint GAVD transfer.

Every graph in this guide is a saved output from an executed notebook. We do not replace those results with cleaned-up vector redraws. Captions identify what each graph can show and, just as importantly, what it cannot show.

## Scope

These tutorials ask the fixed-reflection question: does building a *known*
left-right mirror rule into a model help it learn something useful about
walking? They are Study 01 material. The main proposed contribution instead
asks when anatomical laterality is uncertain; see [Study 02](../studies/02-semantic-gauge-predictive-representations/).

## A note about the evidence

The work so far shows that the research tools can run and that the left-right rule can be checked. It does not show that the model can diagnose a health condition, estimate forces for a new person, or help make clinical decisions. Those are separate questions for the planned study.
