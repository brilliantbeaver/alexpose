"""Read-only notebook review, markdown patch preparation, and preservation checks.

This utility never writes notebooks. The ``patch`` command emits a patch for
apply_patch; only existing markdown cells may be replaced or new markdown cells
inserted. Snapshot/report files are generated review artifacts.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def source(cell):
    text = cell.get("source", [])
    return text if isinstance(text, str) else "".join(text)


def inventory(path):
    nb = json.loads(path.read_text(encoding="utf-8"))
    code = [cell for cell in nb["cells"] if cell["cell_type"] == "code"]
    return {
        "name": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "code_cells": len(code),
        "code_and_outputs_sha256": digest(code),
        "notebook_metadata_sha256": digest(nb.get("metadata", {})),
        "markdown_cells": sum(cell["cell_type"] == "markdown" for cell in nb["cells"]),
        "cells": len(nb["cells"]),
    }


def snapshot(folder):
    folder.mkdir(parents=True, exist_ok=True)
    records = []
    for path in sorted(ROOT.glob("*.ipynb")):
        nb = json.loads(path.read_text(encoding="utf-8"))
        records.append(inventory(path))
        lines = []
        for index, cell in enumerate(nb["cells"]):
            lines.append(f"\n=== CELL {index} {cell['cell_type']} id={cell.get('id', '')} ===\n")
            lines.append(source(cell))
            for output in cell.get("outputs", []):
                if "text" in output:
                    lines.append("\nSAVED OUTPUT:\n" + "".join(output["text"]))
                if output.get("output_type") == "error":
                    lines.append(f"\nSAVED ERROR: {output.get('ename')}: {output.get('evalue')}")
                data = output.get("data", {})
                if "text/plain" in data:
                    lines.append("\nSAVED DISPLAY:\n" + "".join(data["text/plain"]))
                if "image/svg+xml" in data:
                    import re
                    svg = "".join(data["image/svg+xml"])
                    lines.append("\nSVG LABELS:\n" + " | ".join(re.findall(r"<text[^>]*>(.*?)</text>", svg, flags=re.S)))
        (folder / (path.stem + ".txt")).write_text("\n".join(lines), encoding="utf-8")
    (folder / "before.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(records, indent=2))


def patch(path, edits_path):
    before = path.read_text(encoding="utf-8")
    nb = json.loads(before)
    ending = "\n" if before.endswith("\n") else ""
    # Match the existing serialization so embedded SVG/HTML stays untouched.
    ascii_mode = next((mode for mode in (False, True)
                       if json.dumps(nb, indent=1, ensure_ascii=mode) + ending == before), None)
    if ascii_mode is None:
        raise ValueError("Notebook formatting changed; inspect before preparing a patch")
    code_before = digest([c for c in nb["cells"] if c["cell_type"] == "code"])
    edits = json.loads(edits_path.read_text(encoding="utf-8"))
    replacements = {int(k): v for k, v in edits.get("replace", {}).items()}
    insertions = {int(k): v for k, v in edits.get("insert_before", {}).items()}
    cells = []

    def add(text, suffix):
        cells.append({"cell_type": "markdown", "id": "tutorial-" + hashlib.sha256((path.name + suffix).encode()).hexdigest()[:16],
                      "metadata": {}, "source": text.splitlines(keepends=True)})

    for i, cell in enumerate(nb["cells"]):
        if i in insertions:
            add(insertions[i], str(i))
        if i in replacements:
            if cell["cell_type"] != "markdown":
                raise ValueError(f"Cannot replace non-markdown cell {i}")
            cell["source"] = replacements[i].splitlines(keepends=True)
        cells.append(cell)
    if "append" in edits:
        add(edits["append"], "conclusion")
    nb["cells"] = cells
    assert digest([c for c in cells if c["cell_type"] == "code"]) == code_before
    # Preserve the repository's one-space JSON indentation and escaping mode.
    after = json.dumps(nb, indent=1, ensure_ascii=ascii_mode) + ending
    lines = list(difflib.unified_diff(before.splitlines(keepends=True), after.splitlines(keepends=True), n=2))
    if not lines:
        raise ValueError("No changes")
    relative = path.resolve().relative_to(ROOT.parent).as_posix()
    output = ["*** Begin Patch\n", f"*** Update File: {relative}\n"]
    for line in lines[2:]:
        output.append("@@\n" if line.startswith("@@") else line)
    if not output[-1].endswith("\n"):
        output[-1] += "\n"
    output.append("*** End Patch\n")
    print("".join(output), end="")


def check(folder, report_path):
    before = {item["name"]: item for item in json.loads((folder / "before.json").read_text(encoding="utf-8"))}
    reports = []
    for path in sorted(ROOT.glob("*.ipynb")):
        current = inventory(path)
        nb = json.loads(path.read_text(encoding="utf-8"))
        first = source(nb["cells"][0]).lower()
        last = source(nb["cells"][-1]).lower()
        report = {**current,
                  "code_and_outputs_preserved": current["code_and_outputs_sha256"] == before[path.name]["code_and_outputs_sha256"],
                  "notebook_metadata_preserved": current["notebook_metadata_sha256"] == before[path.name]["notebook_metadata_sha256"],
                  "opens_with_motivation": nb["cells"][0]["cell_type"] == "markdown" and "motivation" in first,
                  "closes_with_interpretations_and_takeaways": nb["cells"][-1]["cell_type"] == "markdown" and "interpret" in last and "takeaway" in last,
                  "unique_cell_ids": len({c["id"] for c in nb["cells"] if "id" in c}) == sum("id" in c for c in nb["cells"])}
        reports.append(report)
    result = {"scope": "Markdown-only tutorial revisions; no notebook cells executed.", "notebooks": reports}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    assert all(all(r[k] for k in ("code_and_outputs_preserved", "notebook_metadata_preserved", "opens_with_motivation", "closes_with_interpretations_and_takeaways", "unique_cell_ids")) for r in reports)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    snap = commands.add_parser("snapshot")
    snap.add_argument("folder", type=Path)
    prepare = commands.add_parser("patch")
    prepare.add_argument("notebook", type=Path)
    prepare.add_argument("edits", type=Path)
    verify = commands.add_parser("check")
    verify.add_argument("folder", type=Path)
    verify.add_argument("report", type=Path)
    args = parser.parse_args()
    if args.command == "snapshot":
        snapshot(args.folder)
    elif args.command == "patch":
        patch(args.notebook, args.edits)
    else:
        check(args.folder, args.report)


if __name__ == "__main__":
    main()
