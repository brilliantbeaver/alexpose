# Research notebook builders

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
