"""Small scientific figures generated only from saved observable scores."""
from __future__ import annotations

from pathlib import Path
import io

from .contracts import atomic_bytes


def plot_scores(rows, output_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid = [r for r in rows if r.get("score") is not None]
    if not valid:
        return None
    path = Path(output_path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite figure: {path}")
    if len({r.get("horizon_seconds") for r in valid}) != 1:
        raise ValueError("A score figure must use one declared horizon")
    path.parent.mkdir(parents=True, exist_ok=True)
    names = sorted({r["method"] for r in valid})
    figure, ax = plt.subplots(figsize=(8, max(3, len(names) * .3)))
    for index, name in enumerate(names):
        values = [r["score"] for r in valid if r["method"] == name]
        ax.scatter(values, [index] * len(values), s=18, alpha=.75)
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("Source-video 2D endpoint error / prefix projected body length (lower is better)")
    ax.set_title("Source videos sharing a connected group are not independent", fontsize=9)
    ax.grid(axis="x", alpha=.2)
    figure.tight_layout()
    buffer = io.BytesIO()
    figure.savefig(buffer, format=path.suffix.lstrip(".") or "png")
    atomic_bytes(path, buffer.getvalue())
    plt.close(figure)
    return str(path)
