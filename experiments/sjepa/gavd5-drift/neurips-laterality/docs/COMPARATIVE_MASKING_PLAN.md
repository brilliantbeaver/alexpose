# Comparing the information hidden during JEPA learning

This extension asks which prediction tasks preserve useful movement differences on videos excluded from training. Notebooks 11–14 implement the comparisons below. Their default executions use generated movement for software checks. The completed real-data evidence remains the gait-target versus all-landmark comparison described in [TUTORIAL.md](TUTORIAL.md).

## Experimental questions

| Question | Conditions | Controls |
|:--|:--|:--|
| Does restricting the target pool help? | Gait landmarks, all landmarks, three fixed random twelve-landmark sets, and a soft gait preference | Scattered masks, shared realized hidden counts, the same input and objective |
| Does the arrangement or movement of hidden observations matter? | Motion-weighted sampling, whole trajectories, and connected regions, each with a scattered reference | All landmarks eligible; each reference hides the same count as its paired condition |
| What changes when an interval is missing? | Interior temporal gaps with a count-matched scattered reference | Observations remain available on both sides; this measures completion |
| Do predicted future features express future movement? | Matched and mismatched future training, observed-future decoder diagnostic, and past-only forecasting references | The same future endpoints and times; all input preparation uses the observed prefix |

The twelve-landmark regularizer pooling and five-pair laterality summary remain part of the first comparison. Broadening those summaries, changing input landmarks, adding a different loss, and increasing predictor capacity are separate experiments.

## Training and evaluation

The starting real-data recipe comes from the saved Notebook 08 manifests: 1,200 updates, batches of 20, embedding dimension 96, four encoder layers, two predictor layers, four attention heads, and four samples per token. AdamW uses a constant learning rate of 0.001, weight decay 0.05, and betas (0.9, 0.95). Teacher momentum is constant at 0.999. The feature-variation regularizer weight is 0.05. Target centering, temperatures, clipping, and geometric views retain the existing implementation's settings. These settings are an extension reference, not a reproduction of the original S-JEPA paper.

Every comparison pairs model initialization, source-video sampling, geometric views, and optimizer updates. Separate random streams generate masks and fixed landmark subsets. Structured masks can have different numbers of valid targets in different clips; their losses first average targets within a clip, then average clips. A comparison cannot silently break a trajectory or temporal interval to satisfy an incompatible token count.

Movement evaluation fits imputation, scaling, and ridge penalties using training videos, with inner source-separated validation. Within each seed, predictions from all five outer folds are pooled before calculating source-balanced R² and mean absolute error. Online and teacher features have separate readouts. Initial encoders, direct-pose summaries, and a training-source mean provide references. The pretraining predictor's feature error is accompanied by variation and rank diagnostics because each trained teacher defines its own feature space.

Missing-observation tests distinguish removal before input preparation from masking already prepared coordinates. The unaltered recording supplies the reference movement score. Future prediction uses a decoder fitted on observed future-teacher features from training videos; applying that frozen decoder to observed and predicted future features separates an inadequate representation or decoder from inaccurate future-feature prediction.

## Execution stages

1. Execute the four synthetic tutorials and focused tests, including an informative synthetic readout and checks for unintended access to withheld observations.
2. Display and validate the complete real-data workload without training. Inspect the retained reference settings and source partitions.
3. Use training sources to check feasibility and, when necessary, select settings. Choosing a pretraining recipe requires keeping its inner validation sources out of encoder training as well as supervised fitting. Readout-only penalty selection has a separate boundary.
4. Explicitly enable a small declared comparison. The full design uses five outer folds and seeds 42–46. Report development evidence and every incomplete comparison as such.
5. Examine performance on a shared movement endpoint before adding more masks or objectives. Report an unfavorable result with the same controls and coverage as a favorable one.

The implementation saves new artifacts under a separate comparison directory and checks compatibility before reuse. Interrupted runs remain incomplete; automatic resume is outside this first implementation. This avoids implying reproducible resumption without restoring and verifying every optimizer, teacher, buffer, and random state.

## Method sources

The motion-weighted sampler is an adaptation informed by [MAMP](https://arxiv.org/abs/2308.07092) and [S-JEPA](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04755.pdf). Whole trajectories are a skeleton analogue of [VideoMAE's temporal tubes](https://arxiv.org/abs/2203.12602). Connected targets also have precedent in [I-JEPA](https://arxiv.org/abs/2301.08243) and the skeleton-specific [SLiM preprint](https://arxiv.org/html/2603.10648v3). The supplied samplers keep our current objective fixed and do not reproduce these papers' full methods. Their action-recognition results do not establish an expected gain on gait laterality.
