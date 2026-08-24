# Notebook builders

These scripts are the authoritative cell-source representation for generated
research notebooks. Run them from the project root with `uv`:

```bash
uv run python scripts/notebook_builders/idea05/build_probe.py
uv run python scripts/notebook_builders/idea05/build_futures_and_reach.py
uv run python scripts/notebook_builders/idea09/build_encoder_contract.py
uv run python scripts/notebook_builders/idea09/build_futures_and_reach.py
uv run python scripts/notebook_builders/idea09/build_gavd_sequence.py
uv run python scripts/notebook_builders/idea09/build_replications.py
uv run python scripts/notebook_builders/idea09/build_amass_training.py
```

Builders reset cell outputs. Preserve an executed notebook as a run artifact
before regenerating it if its outputs matter. After rebuilding Idea 09, run:

```bash
uv run python scripts/validate_notebook_startup.py
```

The validator checks cell syntax, cell IDs, root discovery from multiple
working directories, and builder references.
