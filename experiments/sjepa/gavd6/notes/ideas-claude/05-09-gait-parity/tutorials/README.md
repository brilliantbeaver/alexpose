# GaitParity: a plain-language guide

This short series explains a research project about how people walk and how a computer model can learn from movement. You do not need a background in artificial intelligence, biomechanics, or statistics to follow it.

- [From video to a movement model](./01-foundations-notebooks-00-06.md) explains how the project turns ordinary walking video into a model that can notice patterns of movement.
- [What the current experiments tell us](./02-current-direction-and-results.md) explains the first test of left-right reasoning, including an important result that did *not* support the hoped-for claim.
- [What the planned study will test](./03-planned-research-direction.md) explains how the research will test the idea fairly with independent movement and force measurements.
- [Reading the notebook graphs](./04-notebook-graph-guide.md) explains the saved graphs from the research notebooks in everyday language.

Every graph in this guide is a saved output from an executed notebook. We do not replace those results with cleaned-up vector redraws. Captions identify what each graph can show and, just as importantly, what it cannot show.

## The central question

When a person is viewed as a moving skeleton, can a model keep track of which side of the body is left and which is right? And does building that left-right rule into the model help it learn something useful about walking?

## A note about the evidence

The work so far shows that the research tools can run and that the left-right rule can be checked. It does not show that the model can diagnose a health condition, estimate forces for a new person, or help make clinical decisions. Those are separate questions for the planned study.
