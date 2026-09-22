# Methods and experimental protocols

[Research proposal](../README.md) · [Data specification](../data/README.md) · [Interactive paper](../proposal.html)

Read the main proposal for the scientific argument, then use these protocols to make each comparison reproducible. They describe proposed experiments, with completed implementation audits identified separately.

| Protocol | Question it resolves | Read before |
| --- | --- | --- |
| [Evaluation and measurement](evaluation.md) | What is the signed measurement, how is change distortion scored, and which references/populations support a claim? | Freezing data eligibility, losses, primary comparisons or statistical analysis |
| [Anatomical masking](masking.md) | How are graph/time masks sampled, matched and audited without hiding the same evidence repeatedly? | Implementing the body12 adapter or launching pretraining |
| [Data and references](../data/README.md) | Which observed tracks, reference labels and identities belong in each experiment? | Selecting or rendering source windows |
| [Method and novelty audit](../literature/novelty.md) | Which controls isolate feature prediction, paired supervision and anatomical assumptions? | Promoting a development gain into a method claim |

The first endpoint is the right-minus-left difference in projected knee excursion, with excursion defined as the 95th minus 5th percentile of the image-plane hip–knee–ankle angle. It remains an engineering quantity until an independent clinical measurement pathway is established. The [measurement contract](evaluation.md#primary-engineering-endpoint) specifies geometry, paired time support and missingness requirements.

Keep two comparisons distinct when introducing the change constraint. Direct end-to-end training with and without the constraint measures practical benefit. Coordinate-pretrained and JEPA encoders with the same frozen-encoder readout stage, each with and without the constraint, isolate representation choice more closely. The initialized/shuffled feature controls receive the same selected readout objective. A contrast between differently trainable recipes alone cannot attribute an interaction specifically to feature prediction.

Proceed from data/reference checks to mask and gradient smoke checks, then the staged development comparisons in the [proposal](../README.md#55-attribute-each-improvement). Fresh confirmation starts only after the method, endpoint, margins and sample/resource bound are frozen.
