# Research notebook builders

Generated notebooks are organized by study. These builders own their cell sources; existing script paths remain stable execution interfaces.

| Study and notebook destination | Builders |
| --- | --- |
| [Motion preservation](../../notebooks/motion_preservation/README.md) | [build_notebooks.py](motion_preservation/build_notebooks.py) |
| [Future-feature prediction gate](../../notebooks/future_innovation) | [build_notebooks.py](future_prediction/build_notebooks.py) |
| [Source scaling](../../notebooks/source_scaling) | [build_notebook.py](source_scaling/build_notebook.py) |
| [Target accessibility](../../notebooks/target_accessibility/README.md) | [build_notebooks.py](target_accessibility/build_notebooks.py) |
| [Reflection equivariance](../../notebooks/reflection_equivariance) | [Encoder contract](reflection_equivariance/build_encoder_notebook.py), [futures](reflection_equivariance/build_extension_notebook.py), [GAVD](reflection_equivariance/build_gavd_notebooks.py), [replication](reflection_equivariance/build_replication_notebooks.py), [AMASS training](reflection_equivariance/build_amass_notebook.py) |
| [Signed-laterality probes](../../notebooks/signed_laterality) | [Probe](signed_laterality/build_probe_notebook.py), [futures](signed_laterality/build_extension_notebook.py) |

Run the selected script from the project root, for example:

```bash
uv run python scripts/research_directions/motion_preservation/build_notebooks.py
```

Builders reset cell outputs. Preserve scientifically relevant executed notebooks in their run bundles before regeneration. Notebook aliases under earlier directory names identify the same canonical files.

The [prediction-gate maintenance guide](future_prediction/README.md) covers selective generation and fresh-kernel verification. Motion-preservation and scaling notebook tests check their builders against the saved sources. After rebuilding reflection-equivariance notebooks, run:

```bash
uv run gavd6 notebooks validate
```

That validator checks cell syntax, cell IDs, root discovery from multiple working directories, and builder references. It does not execute a research experiment or establish a model result.
