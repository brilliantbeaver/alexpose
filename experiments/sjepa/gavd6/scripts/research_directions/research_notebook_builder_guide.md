# Research notebook builders

The [Future Innovation maintenance guide](future_innovation/future_innovation_tutorial_guide.md)
covers its five connected tutorials, selective generation, source checks and
fresh-kernel verification. Its builder is
`future_innovation/build_future_innovation_notebooks.py`.

These scripts are the authoritative cell-source representation for generated
research notebooks. Run them from the project root with `uv`:

```bash
uv run python scripts/research_directions/signed_laterality/build_signed_laterality_probe_notebook.py
uv run python scripts/research_directions/signed_laterality/build_signed_laterality_futures_notebook.py
uv run python scripts/research_directions/reflection_equivariance/build_reflection_encoder_contract_notebook.py
uv run python scripts/research_directions/reflection_equivariance/build_reflection_futures_notebook.py
uv run python scripts/research_directions/reflection_equivariance/build_reflection_gavd_notebooks.py
uv run python scripts/research_directions/reflection_equivariance/build_reflection_replication_notebooks.py
uv run python scripts/research_directions/reflection_equivariance/build_reflection_amass_training_notebook.py
```

Builders reset cell outputs. Preserve an executed notebook as a run artifact
before regenerating it if its outputs matter. After rebuilding reflection-
equivariance notebooks, run:

```bash
uv run gavd6 notebooks validate
```

The validator checks cell syntax, cell IDs, root discovery from multiple
working directories, and builder references.
