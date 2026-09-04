# Command and research-script guide

Use the installed `gavd6` command for data, training, and evaluation workflows.
The `scripts` tree now contains only research notebook builders and explicitly
archived compatibility tools:

```bash
uv run gavd6 --help
uv run gavd6 amass --help
uv run gavd6 gavd --help
uv run gavd6 laterality --help
```

## Common workflows

| Goal | Command |
|---|---|
| Inventory AMASS archives | `uv run gavd6 amass inventory` |
| Convert AMASS to Core11 | `uv run gavd6 amass convert --help` |
| Train the AMASS baselines | `uv run gavd6 amass train` |
| Download full-GAVD videos | `uv run gavd6 gavd download --help` |
| Convert GAVD poses to Core11 | `uv run gavd6 gavd convert-core11 --help` |
| Build a laterality manifest | `uv run gavd6 laterality build-manifest --help` |
| Run the benchmark gate | `uv run gavd6 laterality benchmark --help` |
| Train SG-JEPA | `uv run gavd6 laterality train --help` |
| Evaluate SG-JEPA | `uv run gavd6 laterality evaluate --help` |
| Run the swap probe | `uv run gavd6 swap-probe run --help` |
| Validate generated notebooks | `uv run gavd6 notebooks validate` |

## Directory ownership

| Directory | Status | Purpose |
|---|---|---|
| `research_directions/reflection_equivariance/` | Active | Builders for the reflection-equivariant JEPA, GAVD, AMASS, and replication notebooks |
| `research_directions/signed_laterality/` | Scoped research | Builders for the signed-laterality probe and reach notebooks |
| `archive/legacy_command_launchers/` | Archived | Thin launchers superseded by `gavd6`; retained only to reproduce older commands |
| `archive/gavd96_augmentation_launchers/` | Archived | Utilities for the superseded local GAVD96 augmentation workflow |

New scientific implementations belong under the matching
`src/gavd6_sjepa/research_directions/` directory. New dataset plumbing belongs
under `src/gavd6_sjepa/data_foundations/`. Do not add new top-level Python
scripts or depend on anything under `archive/` from active code.
