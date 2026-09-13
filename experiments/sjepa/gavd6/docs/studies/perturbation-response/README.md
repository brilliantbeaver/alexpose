# Cross-protocol perturbation-response prediction

**Deferred prospective study; no completed results.** Predict recovery after a known walking perturbation from pre-onset motion, then test whether the information transfers between independent experimental protocols.

| Chunk | Purpose |
| --- | --- |
| [Protocol](protocol/) | Research question, calibrated recovery outcomes, baselines and participant-held evaluation. |
| [Data harmonization](protocol/data-harmonization.md) | Availability, timing, units and harmonization checks required before fitting. |
| [Shared data utilities](../../../src/gavd6_sjepa/data_foundations/) | Existing infrastructure to inspect before implementing a protocol-specific adapter. |
| [Current research agenda](../../../notes/research-agenda/) | Why this direction is deferred relative to movement preservation. |

The proposed comparison separates intervention-only prediction, raw pre-state information, representation benefit and cross-protocol transfer. These are ordered research gates, not achieved findings. No dedicated perturbation-study implementation, notebook sequence or Slurm launcher is established here; existing skeleton code is not evidence that this experiment has run.

Keep this prospective recovery-forecasting question separate from the [current offline tracking-repair study](../motion-preservation/). Dataset access, synchronized intervention timing and harmonized outcomes must be demonstrated before adopting the proposed performance thresholds as an executable experiment.

The [development record](development/) preserves the original setup request.
