"""Extract all notebook source and saved text without embedded media; never execute cells."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent / "notebook_text"
OUT.mkdir(parents=True, exist_ok=True)
inventory = []
for path in sorted((ROOT / "neurips-brain-body").glob("*.ipynb")):
    blob = path.read_bytes()
    nb = json.loads(blob)
    lines = []
    errors = []
    media = 0
    for i, cell in enumerate(nb["cells"]):
        lines.append(f"\n=== CELL {i} [{cell['cell_type']}] execution={cell.get('execution_count')} ===\n")
        lines.append("".join(cell.get("source", [])))
        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                error = f"{output.get('ename')}: {output.get('evalue')}"
                errors.append({"cell": i, "error": error})
                lines.append("\nERROR: " + error)
            if "text" in output:
                lines.append("\nOUTPUT:\n" + "".join(output["text"]))
            data = output.get("data", {})
            if "text/plain" in data:
                lines.append("\nOUTPUT:\n" + "".join(data["text/plain"]))
            elif "text/html" in data and not any(k.startswith("image/") for k in data):
                # Retain tables; omit embedded players and their binary payloads.
                html = "".join(data["text/html"])
                if "<table" in html and "base64" not in html:
                    lines.append("\nHTML TABLE:\n" + html)
            media += sum(k.startswith("image/") for k in data)
    target = OUT / (path.stem + ".txt")
    target.write_text("\n".join(lines), encoding="utf-8")
    inventory.append({"notebook": str(path.relative_to(ROOT)).replace("\\", "/"),
                      "sha256": hashlib.sha256(blob).hexdigest(), "cells": len(nb["cells"]),
                      "code_cells": sum(c["cell_type"] == "code" for c in nb["cells"]),
                      "executed_cells": sum(c.get("execution_count") is not None for c in nb["cells"]),
                      "saved_media_outputs": media, "errors": errors,
                      "extracted_lines": len(target.read_text(encoding="utf-8").splitlines())})
(OUT.parent / "notebook_inventory.json").write_text(json.dumps(inventory, indent=2), encoding="utf-8")
for row in inventory:
    print(row["notebook"], row["cells"], "cells;", row["extracted_lines"], "lines; errors", row["errors"])
