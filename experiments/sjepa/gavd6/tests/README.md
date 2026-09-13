# Tests by study and responsibility

Run all tests from the repository root:

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -t . -v
```

| Package | What it checks |
| --- | --- |
| [motion_preservation](motion_preservation/) | Controlled motion/error pairs, pretrained-model adapters, repair models, calibration boundaries, GAVD stress inspection and notebooks. |
| [future_feature_prediction/gate](future_feature_prediction/gate/) | Input alignment, teacher features, validity and decision gates, versioned predictors, diagnostics, recovery and scheduling. |
| [future_feature_prediction/scaling](future_feature_prediction/scaling/) | Source reservations, cohort expansion, learning curves, launch inputs and inspection notebooks. |
| [future_feature_prediction/accessibility](future_feature_prediction/accessibility/) | Cached-panel measurement, symmetry controls, integrity-only inspection and numerical verification. |
| [reflection_equivariance](reflection_equivariance/) | Reflection contracts, representation training, frozen GAVD probes and the bilateral-correction control. |
| [latent_laterality](latent_laterality/) | Sequence corruption, unknown correspondence and benchmark gates. |
| [data](data/) | AMASS conversion and GAVD acquisition. |
| [infrastructure](infrastructure/) | Command routing, archived adapters, repository ownership and result organization. |

Run one package, for example:

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests/motion_preservation -t . -v
```

`support.py` holds ordinary shared fixtures and the repository-root locator. Package `__init__.py` files enable recursive `unittest` discovery. Test modules do not import other test classes as fixtures, and old test paths have no aliases that could double collection.

Tests requiring absent historical results or external author code retain explicit skips. Kernel execution tests require local Jupyter sockets. Smoke fixtures and software tests are not scientific evidence of a pretrained-model benefit. See the [code review](../docs/repository/code-organization.md) for migration and replay checks.
