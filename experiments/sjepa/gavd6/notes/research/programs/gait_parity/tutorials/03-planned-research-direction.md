# The planned fixed-reflection baseline study

> **Study 01/03 material.** This document describes the retained
> fixed-reflection force-validation route. The program's proposed
> representation-learning contribution is [Study 02](../studies/02-semantic-gauge-predictive-representations/).

## The decision the study is designed to make

The planned study asks a focused question: does teaching a movement model the left-right mirror rule throughout its inner workings help it estimate a meaningful left-right difference in walking?

The independent measurement comes from force plates in the floor. They measure how each foot pushes against the ground during a step. The study compares the push from one leg with the push from the other. A stronger push on one side creates a signed difference; making an anatomical mirror image must reverse that difference.

This is an association study, not a force simulator or a diagnostic device. Video-derived skeleton motion cannot fully reveal force, because people can use different muscle strategies, body weights, and contact patterns while looking similar on camera.

## Why a simple comparison matters

A conventional prediction can be made to obey the mirror rule at its final output by comparing a walk with its mirrored version. That is a powerful baseline because it is simple and it guarantees the right sign change at the end.

The proposed mirror-aware model goes further. It carries the original walk and its mirror through the whole model together. If it helps, it must help more than the simple final-output correction and more than an ordinary model that merely sees both views. Otherwise, the extra structure is unnecessary.

## One person, one vote

Many steps from the same person are related observations, not many independent people. The study will keep all of one person's walks together whenever it trains, chooses settings, or checks results. That makes the final test closer to the real question: what happens with a person the model has not encountered before?

The main outcome will be how close the estimates are to the force-plate measurement for each person. The study will also report whether the model gets the left-right direction right and whether its confidence matches its errors. The rules for deciding what counts as a worthwhile improvement will be set before the researchers inspect the final results.

## The evidence path

The program moves through a sequence of gates. Each gate supports a different claim.

### Check the research instrument

The completed local work checks the mirror operation, model geometry, saved model loading, comparison conditions, and warning signs of empty internal features. This work shows that the instrument can be inspected. It does not make a clinical claim.

### Test the simple mirror rule on independent people

The first force-plate study will compare an ordinary prediction, an ordinary prediction with the final-output mirror correction, and models that see both the original and mirrored walk. If the force measurements themselves are not dependable enough, the project will say so rather than turning a weaker proxy into the main claim.

### Test the full mirror-aware model

The study will then compare the model with built-in left-right structure against both strong alternatives. Before reading the force results, the researchers will check that the mirror rule holds throughout the model and that the part meant to carry left-right information is genuinely active.

They will also test whether the idea remains useful when body landmarks are missing or noisy, and whether it remains stable across real camera views. A mirror should reverse a left-right answer; a camera change should not.

### Lock the method and repeat it elsewhere

After the analysis choices are fixed, the team will repeat the study with a separate group. Repeating the full recipe in a new group is different from applying an already trained model without changing it, so those outcomes will be named separately rather than blended together.

## How outcomes change the program

| What happens | What the team should conclude |
|---|---|
| The mirror-aware model beats both strong alternatives | Internal left-right structure appears useful for this task |
| The simple final-output correction works just as well, or simple movement features win | Prefer the simpler approach |
| Seeing both views helps but the built-in rule adds nothing | Credit paired views, not the mirror-aware design |
| The model breaks the rule or its changing channel is empty | Repair, rename, or stop before interpreting predictions |
| The evidence is too uncertain | Defer the conclusion and gather stronger evidence |

Where permissions allow, the project will share the mirror operation, landmark mappings, person-safe evaluation rules, test conditions, compute records, and person-level predictions. That makes it possible for others to check the work rather than rely on a headline.

## Bottom line

The planned study separates a model following a rule from a model being useful. It also separates a promising result in one group from a result that holds up elsewhere. Those distinctions are what make a left-right gait claim worth trusting.
