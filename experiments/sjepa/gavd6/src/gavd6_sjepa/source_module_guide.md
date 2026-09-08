# Source ownership guide

The package is organized by scientific ownership. Filenames remain globally
unique across `scripts` and `src`, so every search result identifies one file.

| Directory | Status | Responsibility |
|---|---|---|
| `data_foundations/` | Shared and active | AMASS inventory/identity/Core11 conversion and full-GAVD acquisition/Core11 conversion |
| `shared_infrastructure/` | Shared and active | Atomic artifact writes and fingerprints |
| `research_directions/reflection_equivariance/` | Active research | Matched JEPA architecture, AMASS training, frozen GAVD probes, and bilateral swap probe |
| `research_directions/latent_laterality/` | Active research | Sequence corruption, eligibility benchmark, Semantic-Gauge JEPA, and source-transfer evaluation |
| `research_directions/future_innovation/` | Active research | Frozen V-JEPA 2.1 features, the 50-window innovation feasibility gate, nested source-fold controls, and decision reports |
| `workspace_validation/` | Active support | Generated-notebook startup validation |
| `archive/historical_checkpoints/` | Archived | Frozen historical checkpoint compatibility verification |
| `archive/gavd96_augmentation/` | Archived | Superseded local-normal augmentation for the earlier GAVD96 workflow |
| `archive/legacy_imports/` | Archived | Import adapters retained for inspecting completed experiments |

`command_line_interface.py` remains the only user-facing command router and
imports handlers lazily. Archive commands remain reachable for reproducibility,
but active scientific modules must never import from `archive/`.

Future Innovation reuses shared atomic artifact utilities, the full-GAVD video
cache resolver, and `data_foundations/gavd_pose_extraction_primitives.py`. The
historical augmentation extractor now imports those same pose primitives.
See [the HAIC run guide](../../slurm/future-innovation/README.md).

The dependency direction is:

```text
shared infrastructure + data foundations
                    ↓
reflection-equivariance foundation
                    ↓
latent-laterality extension
                    ↓
CLI and workspace validation
```

The future Wrench-JEPA, counterfactual-mechanics, and force-calibration agenda
does not yet have implementation code here. When added, each contribution
should receive its own directory under `research_directions/` and reuse only
the shared/data layers it actually needs.
