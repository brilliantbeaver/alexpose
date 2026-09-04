"""Contracts for durable notebooks whose evidence is visible without re-execution."""

from __future__ import annotations

from typing import Any


_RESULT_MIME_TYPES = {
    "application/json",
    "text/html",
    "text/markdown",
    "text/plain",
}


def audit_inline_outputs(notebook: Any, *, name: str) -> dict[str, int]:
    """Reject an executed notebook unless figures and results are embedded inline.

    A Matplotlib display also carries a ``text/plain`` representation, so it does
    not count as a separate result. This prevents a figure-only notebook from
    satisfying the audit accidentally.
    """

    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    unexecuted = [
        index
        for index, cell in enumerate(code_cells)
        if cell.get("execution_count") is None
    ]
    if unexecuted:
        raise AssertionError(f"Unexecuted code cells in {name}: {unexecuted}")

    outputs = [output for cell in code_cells for output in cell.get("outputs", [])]
    errors = [output for output in outputs if output.get("output_type") == "error"]
    if errors:
        raise AssertionError(f"Stored error output in executed notebook: {name}")

    figures = 0
    results = 0
    for output in outputs:
        data = output.get("data", {})
        if "image/png" in data:
            payload = data["image/png"]
            if not isinstance(payload, str) or len(payload) < 100:
                raise AssertionError(f"Empty or malformed inline PNG in {name}")
            figures += 1
        elif output.get("output_type") in {"display_data", "execute_result"}:
            if _RESULT_MIME_TYPES.intersection(data):
                results += 1

    if figures < 1:
        raise AssertionError(f"No embedded image/png figure in {name}")
    if results < 1:
        raise AssertionError(f"No separate inline result payload in {name}")

    return {
        "code_cells": len(code_cells),
        "outputs": len(outputs),
        "inline_png_figures": figures,
        "inline_result_payloads": results,
        "error_outputs": len(errors),
    }


__all__ = ["audit_inline_outputs"]
