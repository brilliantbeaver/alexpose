"""Leakage-safe laterality experiment components.

The package is intentionally local to ``neurips-laterality``.  It does not import
the historical experiment notebooks or scripts.
"""

from .config import ExperimentContext, load_context, load_protocol

__all__ = ["ExperimentContext", "load_context", "load_protocol"]
