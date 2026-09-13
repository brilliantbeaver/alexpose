# Future-feature prediction and student accessibility

**Historical feasibility studies and research proposals.** These experiments ask different questions about future video-teacher features. Their protocols, reference models and conclusions remain separate.

| Chunk | Question and status |
| --- | --- |
| [Prediction gate](gate/) | Does skeleton history improve future-teacher prediction beyond RGB? The calibrated direct-v3 development result is STOP. |
| [Source scaling](scaling/) | Does increasing the number of recordings change that incremental benefit? The expanded run retains a development STOP. |
| [Student accessibility](accessibility/) | Which future targets can skeleton history predict beyond its declared current-state/validity reference? The cached comparison does not establish a positive temporal gain. |
| [Manuscript and research strategy](manuscript/) | Evidence, derivations, figures, drafts and proposed selective-distillation experiments. Actual student-transfer benefit remains untested. |

The [gate notebooks](../../../notebooks/future_innovation/) and [execution guide](../../../slurm/future-prediction/README.md) accompany the [prediction implementation](../../../src/gavd6_sjepa/research_directions/future_prediction/). [Source-scaling jobs](../../../slurm/source-scaling/) use a [separate implementation](../../../src/gavd6_sjepa/research_directions/source_scaling/). [Accessibility notebooks](../../../notebooks/target_accessibility/README.md) accompany the [cached-panel implementation](../../../src/gavd6_sjepa/research_directions/target_accessibility/). Their existing identifiers and notebook numbering are historical execution interfaces. [Builders](../../../scripts/research_directions/) remain the source of generated notebooks and publication assets.

A small skeleton increment conditional on RGB does not settle skeleton-only accessibility, and predictable targets do not establish useful student distillation. Preserve the original negative gates, source reservations and frozen protocols. These findings motivate the [current movement-preservation study](../motion-preservation/), whose restoration task and evaluation are different.

The result registry records historical run identities; the September 2026 assessment found its registered bundles absent in this checkout. A manuscript's recorded result and a locally installed executable evidence bundle are separate states.
