# FMTS V3 adversarial review

| Perspective | Objection and severity | Evidence | Correction in V4 | Residual limitation |
|---|---|---|---|---|
| Temporal researcher | Major: a speed-derived target may still be insensitive to temporal direction. | The median norm of a transition divided by its interval is preserved under trajectory/time-interval reversal. | Link this property directly to the limit on claiming learned dynamics. | No phase, timing, future behavior or persistent-state test. |
| Evidence auditor | Moderate: “matching improves” may invite comparison of MSE from differently scaled teachers. | Each training arm has its own teacher; diagnostic contrasts pair correct and mismatched targets within a model. | Describe increased consistency of correct-clip preference, not a common cross-model loss ranking. | No causal mechanism identified. |
| Statistical reviewer | Major: expanded summaries change capacity and nuisance information jointly. | 960 to 2,890 inputs; SD, absolute increments and support added together. | Explain order and direction limitations at the reported gain. | Requires matched-dimension, support-only and regularization ablations. |
| Editor | Moderate: Related Work is a short list of papers rather than a scientific synthesis. | Recognition, masking, prediction and transformation papers address distinct outcomes. | Organize around what variation prediction retains and how downstream behavior evaluates it. | Literature comparisons remain conceptual, not empirical baselines. |

V4 sharpens the temporal contribution by making its weakest interpretation explicit. Removing generic workshop language does more for relevance than adding unsupported topics.
