# Manuscripts and evidence notes

The current title is *Auditing Reflection Symmetry in Self-Supervised Skeleton Representations*. The long paper and extended abstract present the completed laterality v2.1 experiment. They are internal anonymous drafts; the project governance record still blocks submission and artifact release.

## Reading and building

- [paper.md](paper.md) and [paper.pdf](paper.pdf): the full scientific account.
- [extended_abstract.md](extended_abstract.md) and [extended_abstract.pdf](extended_abstract.pdf): the shorter workshop version.
- [TUTORIAL.md](TUTORIAL.md) and [TUTORIAL.pdf](TUTORIAL.pdf): a concise explanation of the research trajectory and new notebook suite for workshop discussion.
- [00_critique_and_tutorial.md](00_critique_and_tutorial.md): the detailed guide to the target, split, probes, and interpretation.
- [TRAIN_TEST_SPLIT.md](TRAIN_TEST_SPLIT.md): the canonical numeric GAVD cohort,
  outer train/test folds, inner validation ranges, and leakage guarantees.
- [figures/README.md](figures/README.md): figure provenance and regeneration instructions.

The two Markdown manuscripts are now the editable sources. Their YAML headers supply the title and abstract, and citations resolve against the existing shared bibliography at `../../docs/references.bib`. Generate the corresponding TeX and PDFs with:

```bash
bash neurips-laterality/docs/build_manuscripts.sh
```

The build uses Pandoc, Tectonic, the shared [manuscript template](manuscript_template.tex), and the existing NeurIPS 2026 style. It preserves the style's margins and type sizes. The checked PDFs have eight and four main-text pages, respectively, followed by a separate references page. Edit Markdown and rebuild to keep the text, equations, and captions synchronized.

## Workshop positioning

The [PhysWorldAI call for papers](https://physworld-org.github.io/physworld.github.io/cfp/), checked on September 5, 2026, lists a maximum of eight pages for long papers and four for extended abstracts, excluding references and appendices, with double-blind review. Its themes guide the emphasis as follows.

| Workshop area | Connection supported by this experiment |
|---|---|
| Physical Geometry | A known reflection action on articulated pose representations; direct token alignment and a signed motion target. This is the primary fit. |
| Cross-cutting evaluation | Source-disjoint representation training, paired initializations, matched parity controls, and separate tests of prediction and geometric consistency. |
| Physical Sensors | Validity-aware analysis of video-derived landmarks provides a limited sensing connection. No additional sensor or fusion system was evaluated. |
| Physical Characteristics | No measured material property, contact model, or physical dynamics result. These are possible future applications, not current contributions. |

The paper treats the S-JEPA adaptation as a representation component relevant to world models. Masked latent prediction alone does not establish future rollout or action-conditioned physical reasoning. The algebraic odd projection is attributed to established equivariance and group-averaging methods; its use as an experimental control is the contribution here.

## Results that determine the argument

All empirical figures and the manuscript results use the completed report under `../artifacts/paper/protocol_6f7baefbda07/report/`. The canonical notebooks are output-free; their explanatory cells and saved real-data artifacts were read together with the implementation. Synthetic smoke outputs and older transductive estimates are excluded.

| Finding | Estimate and 95% source-bootstrap interval | Report row |
|---|---|---|
| Vanilla learned minus initial token error | +0.03055 [0.01576, 0.04763] | `strict_representation_equivariance_source_bootstrap.csv`: `learned_minus_initial_strict_equivariance`, vanilla |
| Augmented minus vanilla token error | −0.00843 [−0.01020, −0.00687] | Same file: `reflection_minus_vanilla_strict_equivariance` |
| Native learned predictive utility | 0.05979 [−0.02527, 0.12571] | `checkpoint_source_bootstrap.csv`: `absolute_primary` |
| Native learned minus initial prediction | −0.01798 [−0.03851, 0.00248] | Same file: `primary_training_content` |
| Augmented minus vanilla prediction | +0.00408 [−0.00556, 0.01277] | Same file: `reflection_minus_vanilla_primary` |
| Constructed learned predictive utility | 0.04302 [−0.04356, 0.11283] | Same file: `absolute_constructed` |
| Constructed learned minus initial prediction | −0.05874 [−0.09549, −0.01740] | Same file: `constructed_training_content` |
| Native output antisymmetry error | 0.21548 [0.19352, 0.23635] | `native_output_symmetry_source_bootstrap.csv`: `absolute_native_learned_symmetry`, vanilla |

The paired augmentation effect corrects a material error in the earlier drafts: its improvement in token error is distinguishable from zero. Both trained variants nevertheless remain worse than initialization under the specified token action and fail the absolute acceptance criterion. Their absolute token-error intervals cross 0.10, which is why three-decimal precision matters.

The primary predictive estimand averages five seed-specific scores, each computed from pooled out-of-fold predictions. It does not average 50 separately calculated checkpoint scores or score a seed-mean prediction ensemble. The high-coverage sensitivity results are explicitly secondary ensemble estimates. Token and native-output errors have different normalizations, so the revised absolute figure gives them separate panels.

## Interpretation decisions

The revised argument distinguishes the observed training effect from the broader question of whether an encoder represents physical geometry. The strict token action fixes latent channels, and uncentered error can be low for shared or input-insensitive features. A ridge-probe failure also cannot locate a loss of information among input resizing, representation learning, pooling, and linear decoding. The target-component oracle verifies its own arithmetic and does not resolve that attribution.

Exact sign reversal follows from the representation's transformation law, including its validity masks. It does not require a physiologically symmetric person or symmetric gait. Initial features already receive the same anatomical schema and probe construction as trained features, so this is not a test of discovering geometry without supplied structure.

The protocol was frozen internally following prior development. No external preregistration is claimed. The bootstrap intervals are pointwise and conditional on the fitted pipeline. The 0.10 margins are operational choices, with no application-level calibration, and inconclusive predictive contrasts do not establish equivalence.

Read-only verification with the suite's own cohort, split, and evaluation loaders passed for all 100,000 prediction rows, 50 jobs, and 16 lanes. This checked artifact lineage, checkpoint and CSV fingerprints, source coverage and weights, and the saved token-error algebra. The experiment, protocol, trained checkpoints, and governance statuses were not changed during manuscript revision.
