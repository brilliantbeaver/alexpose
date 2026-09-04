# Repository map

## Scope and repository state

This map describes the working tree on 2026-09-03. It does not audit the HAIC datasets. The branch is `main`, six commits behind `origin/main`, with extensive user edits in progress. The active package has moved from deleted flat modules into responsibility-based directories. This report treats the working tree as authoritative and preserves those edits.

A world model predicts how a system evolves. This repository currently implements small skeleton motion predictors, not a general video world model. A joint-embedding predictive architecture (JEPA) learns by predicting hidden internal vectors, called latent representations, from visible motion tokens.

## Inventory

| Area | Authoritative paths | Current role |
| --- | --- | --- |
| Command router | `src/gavd6_sjepa/command_line_interface.py`, `scripts/navigation_guide.md` | One `gavd6` command exposes AMASS, GAVD, laterality, swap-probe, and notebook-validation jobs. |
| Data foundations | `src/gavd6_sjepa/data_foundations/` | Inventories AMASS, resolves subject identities, converts AMASS, downloads GAVD videos, and converts pre-extracted GAVD poses. |
| Models and probes | `src/gavd6_sjepa/research_directions/reflection_equivariance/` | Defines the JEPA, trains matched variants, adapts GAVD poses, runs frozen probes, and evaluates bilateral swaps. |
| Laterality extension | `src/gavd6_sjepa/research_directions/latent_laterality/` | Creates persistent left/right corruptions, enforces benchmark gates, trains three comparison arms, and evaluates held-out identities. |
| Shared contracts | `src/gavd6_sjepa/shared_infrastructure/artifact_io_operations.py` | Provides atomic saves and SHA-256 fingerprints. |
| Notebooks | `notebooks/foundations/00` through `06`, `notebooks/amass/`, `notebooks/experiments/`, `notebooks/evaluate_strokepig_frozen_jepa.ipynb` | Foundations reproduce the historical GAVD96 curriculum. Experiment notebooks cover signed laterality and reflection equivariance. Executed copies live under `artifacts/notebook_runs/`. |
| Configuration | `pyproject.toml`, `uv.lock`, `slurm/`, saved `run_config.json`, `effective_config.json`, and `evaluation_contract.json` files | Dependencies are pinned. Scientific settings live in command flags, environment variables, Slurm scripts, and immutable run records. There is no separate config directory. |
| Checkpoints | `outputs/repaired-jepa-seed7-v2/`, `outputs/latent-laterality/decisive/amass-v2-chart-paired/`, `cache/artifacts/smoke/`, `work/archive/gait-parity-pre-orbit-mask-2026-08-19/` | The first two contain the current AMASS baseline and decisive seed-7 laterality weights. Smoke and archived weights are execution or provenance assets, not headline evidence. |
| Results | `outputs/latent-laterality/`, `work/artifacts/gavd_core11_frozen_probe/`, `work/artifacts/strokepig_frozen_jepa_probe/`, `docs/studies/` | Tables contain the current quantitative evidence. Study documents state the permitted interpretation. |

## Pipeline as it runs

1. `uv sync` installs the package defined by `pyproject.toml` and `uv.lock`.
2. AMASS archives pass through `gavd6 amass inventory`, subject-registry finalization, and `gavd6 amass convert`. The converter uses SMPL+H, a parametric 3D body-and-hands representation, to create 30 Hz Core11 tensors. Core11 means the pelvis plus five bilateral lower-body joint pairs.
3. `gavd6 amass train` fits a standard skeleton JEPA (S-JEPA) and matched paired variants. It saves best checkpoints, histories, capacity tables, and a run contract. The exponential-moving-average, or EMA, target encoder supplies frozen latent targets.
4. The laterality path builds a sequence corruption manifest, runs a shortcut eligibility gate, trains correction-first S-JEPA, Semantic-Gauge JEPA (SG-JEPA), and a 50/50 uncertainty control, then fits train-only linear readouts and evaluates validation or explicitly unsealed test identities.
5. The GAVD path downloads source videos and validates them. A separate pose manifest must already contain MediaPipe 33-landmark arrays. `gavd6 gavd convert-core11` maps them into an unanchored, scale-normalized Core11 frame. `gavd6 gavd evaluate-core11` then compares frozen AMASS encoders with raw, validity-only, and random controls.

The original foundation notebooks follow a different historical path: GAVD manifest, video cache, MediaPipe extraction, masking, a five-stage label-aware curriculum, latent inspection, then descriptive classifiers. The README correctly warns that this is not the active research contract.

## Three strongest verified empirical results

1. **The probability-aware mechanism failed its decisive control.** On 15 unseen validation identities and 1,146 windows, SG-JEPA reached odd/even normalized mean absolute errors of **0.8671/0.6696**, versus **0.9300/0.7472** for correction-first S-JEPA. An odd target changes sign under a left/right exchange. An even target stays fixed. Lower error is better. The 50/50 control was slightly better at **0.8668/0.6690**. The proposed probabilities therefore cannot explain the gain. Source: `outputs/latent-laterality/amass-gauge-v2-seed7-validation/gauge_readout_summary.csv`. The sealed test was not evaluated, as recorded in `evaluation_contract.json` beside it.
2. **Frozen AMASS representations did not improve five-class GAVD recognition.** On the strict 90-frame, no-padding cohort, mean macro-F1, the class-balanced mean of per-class F1 scores, was **0.4231** for raw Core11, **0.2339** for the EMA paired shared/no-cross encoder, and **0.2451** for the EMA reflection-equivariant encoder. Random controls ranged from **0.3336** to **0.5366**. The split shared a mean of 11 source videos and cannot measure unseen-video generalization. Source: `work/artifacts/gavd_core11_frozen_probe/strict90_no_short_clip_padding_nested_probe_summary.csv`.
3. **Current frozen representations did not predict StrokePIG force targets.** All 24-participant held-out coefficients of determination, or R-squared values, were negative. The least negative was **-0.05147** for a random paired-unconstrained encoder. The EMA reflection-equivariant result was **-0.05804**. Source: `work/artifacts/strokepig_frozen_jepa_probe/nested_probe_summary.csv`.

## Top five known weaknesses

1. No large pretrained video or motion foundation model is integrated. The decisive laterality runs start without an initial checkpoint.
2. The active full-GAVD route has no CLI pose-extraction stage between video download and the pose-manifest-dependent Core11 converter.
3. Available GAVD probe evidence covers the 96-row legacy cohort. Source-video grouping is blocked by class support, so the result is source-confounded.
4. The main laterality comparison uses one seed and validation only. Its uniform control reproduces the apparent SG-JEPA gain.
5. The shared AMASS-to-GAVD bridge compresses clean metric 3D motion and noisy monocular pseudo-3D poses into 11 joints. It discards appearance, assistive objects, scene contact, and metric depth. The laterality targets are synthetic coordinate summaries, not clinical outcomes.

## Reusable on day one

- Use the full-GAVD manifests, downloader, validation reports, and source-video split checks as acquisition scaffolding.
- Reuse both Core11 converters as an explicit AMASS-to-GAVD bridge and as a baseline that new richer bridges must beat.
- Reuse the checkpoint loader, frozen EMA encoder, matched-capacity controls, raw controls, and random-encoder controls.
- Reuse the identity-separated splits, shortcut gates, paired-view construction, sealed-test switch, and identity-macro aggregation.
- Reuse Slurm launchers, atomic artifact writes, fingerprints, saved run contracts, focused tests, and notebook builders.

## Open questions

- Which maintained component will extract MediaPipe or stronger poses for the full downloaded GAVD set?
- Four baseline checkpoints exist in `outputs/repaired-jepa-seed7-v2/`, but its `summary.csv` contains only `standard_sjepa`. Which per-arm histories define the accepted comparison record?
- `notebooks/experiments/idea09_reflection_equivariance/09_gavd_frozen_probe.ipynb` exists but is absent from `notebooks/README.md`. Is it generated, hand-maintained, or retired?
- The historical checkpoint named in `docs/history/amass-core11-historical-pilot-probe.md` is not present at its documented `outputs/archive/run1/` path in this checkout.
