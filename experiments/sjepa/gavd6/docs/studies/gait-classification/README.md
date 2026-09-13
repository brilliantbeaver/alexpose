# Historical gait representation and classification study

**Historical teaching and evaluation study.** The original augmented-normal curriculum learned skeleton features from a known gait-video corpus, then inspected representation geometry and condition readouts. It does not measure the current objective of preserving real movement during tracking repair.

| Chunk | Contents |
| --- | --- |
| [Study record](tutorial.md) | Original curriculum narrative, dataset roles, evaluation caveats and development history. |
| [Tutorials](../../tutorials/) | S-JEPA model explanation and retained technical teaching material. |
| [Foundation notebooks](../../../notebooks/gait_classification/) | Video manifests, pose extraction, masking, curriculum training and classifier diagnostics. |
| [Historical manuscript](../../../docs/history/urtc-2026/) | Drafts, figures, references and the original result ledger. |

The [data-foundation package](../../../src/gavd6_sjepa/data_foundations/) provides reusable acquisition and conversion utilities. The [archived augmentation implementation](../../../src/gavd6_sjepa/archive/gavd96_augmentation/) and [launchers](../../../scripts/archive/gavd96_augmentation_launchers/) belong to the earlier added-normal cohort. These utilities retain their original paths; shared raw-data functionality remains relevant to newer studies.

The five-stage curriculum used condition labels after its initial stage, and several original classifier evaluations reused encoder-exposed sources. Read each report's evidence boundary before interpreting its metrics. A within-corpus classification score is not an unseen-person result or evidence that true movement survives a repair model.

Earlier design requests are indexed in [development](development/).
