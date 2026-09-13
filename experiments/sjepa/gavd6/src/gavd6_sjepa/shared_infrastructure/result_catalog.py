"""Location-independent navigation of experiment bundles; never edits run files.

Kept outside fingerprinted fitting packages. The tracked registry supplies human
interpretation; the local catalog derives availability and execution evidence.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

PROJECT = Path(__file__).resolve().parents[3]
REGISTRY = PROJECT / "docs/studies/experiment-registry.json"
SCHEMA = "experiment-catalog-v1"


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def component(value):
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", value) or value in {".", ".."}:
        raise ValueError(f"Invalid study/run identifier: {value!r}")
    return value


def inside(root, relative):
    """Reject traversal and symlink escapes in bundle-local artifact references."""
    root = Path(root).resolve()
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Expected a bundle-relative path: {relative}")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Artifact escapes bundle: {relative}")
    return path


def snapshot(root):
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Nested symlink requires an explicit migration policy: {path}")
        if path.is_file():
            stat = path.stat()
            result[str(path.relative_to(root))] = dict(sha256=digest(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns)
    return result


def registry(path=REGISTRY):
    data = read_json(path)
    if data.get("schema") != SCHEMA:
        raise ValueError("Unsupported experiment registry schema")
    ids = [r["id"] for r in data["entries"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate experiment identifier")
    for entry in data["entries"]:
        component(entry["id"])
        component(entry["study"])
        for value in [entry["legacy_path"], entry["canonical_path"], *entry.get("alternate_paths", []), *([entry["identity_file"]] if entry.get("identity_file") else [])]:
            p = Path(value)
            if p.is_absolute() or ".." in p.parts:
                raise ValueError(f"Invalid registered path: {value}")
        if not set(entry.get("parents", [])) <= set(ids):
            raise ValueError(f"Unknown parent for {entry['id']}")
    return data


def locate(project, entry):
    """Prefer an organized bundle. Conflicting locations require migration review."""
    project = Path(project)
    candidates = [project / entry[k] for k in ("canonical_path", "legacy_path")]
    candidates += [project / p for p in entry.get("alternate_paths", [])]
    marker = entry.get("identity_file")
    for path in candidates:
        if path.is_dir() and (not marker or (path / marker).is_file()):
            return path
    return candidates[0] if candidates[0].exists() else candidates[1]


def entries_for(project, definitions):
    entries = list(definitions["entries"])
    known = {locate(project, e).resolve() for e in entries}
    ids = {e["id"] for e in entries}
    for path in sorted((Path(project) / "outputs/studies").glob("*/*")):
        if path.parent.is_symlink() or not path.is_dir() or path.resolve() in known:
            continue
        if path.name in ids:
            raise ValueError(f"Duplicate run identifier at {path}; register an explicit identity")
        component(path.name)
        component(path.parent.name)
        relative = str(path.relative_to(project))
        entries.append(dict(id=path.name, study=path.parent.name, canonical_path=relative,
                            legacy_path=relative, kind="unclassified_run", description="New bundle; interpretation not yet registered",
                            parents=[], limitations=["Register its scientific scope before citing results."]))
        ids.add(path.name)
    return entries


def resolve(project, run_id, definitions=None):
    definitions = definitions or registry()
    matches = [e for e in entries_for(Path(project), definitions) if e["id"] == run_id]
    if len(matches) != 1:
        raise ValueError(f"Unknown or ambiguous experiment ID: {run_id}")
    root = locate(project, matches[0])
    entry = matches[0]
    candidates = [Path(project) / entry[k] for k in ("canonical_path", "legacy_path")]
    candidates += [Path(project) / p for p in entry.get("alternate_paths", [])]
    installed = {p.resolve() for p in candidates if p.is_dir() and
                 (not entry.get("identity_file") or (p / entry["identity_file"]).is_file())}
    if len(installed) > 1:
        raise ValueError(f"Multiple installed locations for {run_id}; verify and adopt a migration before selecting it")
    if not root.is_dir() or (matches[0].get("identity_file") and not (root / matches[0]["identity_file"]).is_file()):
        raise FileNotFoundError(f"Registered experiment is not installed locally: {run_id} ({root})")
    return root.resolve()


def new_path(project, study, run_id):
    path = Path(project).resolve() / "outputs/studies" / component(study) / component(run_id)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Run already exists; resolve it to resume: {path}")
    if not path.resolve().is_relative_to(Path(project).resolve() / "outputs/studies"):
        raise ValueError("New run path escapes outputs/studies")
    return path


def select(project, run_id, definitions, *, supplied_root=None, new_study=None):
    component(run_id)
    known = {e["id"] for e in entries_for(Path(project), definitions)}
    if run_id in known:
        selected = resolve(project, run_id, definitions)
    elif new_study:
        selected = new_path(project, new_study, run_id)
    else:
        raise ValueError(f"Unknown experiment ID: {run_id}")
    if supplied_root and (Path(project) / supplied_root).resolve() != selected.resolve():
        raise ValueError(f"Run ID {run_id} resolves to {selected}, conflicting with the supplied root {supplied_root}")
    return selected


def first_json(root, names):
    for name in names:
        if (root / name).is_file():
            return name, read_json(root / name)
    return None, {}


def describe(project, entry):
    root = locate(project, entry)
    row = {**entry, "location": str(root.absolute()), "available": root.is_dir(),
           "execution_state": "unavailable", "scientific_decision": "not_evaluated"}
    if not row["available"]:
        return row
    files = snapshot(root)
    row.update(file_count=len(files), bytes=sum(f["size"] for f in files.values()),
               snapshot_sha256=identity(files), execution_state="artifacts_present")
    name, config = first_json(root, ("config/study.json", "config/run-contract.json", "effective_config.json", "run_config.json"))
    row.update(protocol=config.get("protocol", config.get("version", entry.get("protocol"))),
               declared_run_id=config.get("run_id"),
               synthetic=config.get("synthetic", config.get("synthetic_smoke")),
               contract={"path": name, "sha256": digest(root / name)} if name else None)
    if entry["kind"] == "checkpoint_collection":
        row["checkpoints"] = []
        for relative, metadata in files.items():
            match = re.fullmatch(r"seed-(\d+)_(.+)_(best|latest)\.pt", relative)
            if not match:
                continue
            seed, variant, selection = match.groups()
            history = f"seed-{seed}_{variant}_history.csv"
            row["checkpoints"].append(dict(path=relative, sha256=metadata["sha256"], bytes=metadata["size"],
                seed_from_filename=int(seed), variant_from_filename=variant, selection_from_filename=selection,
                history=history if history in files else None,
                covered_by_run_config=variant in config.get("variants", []),
                provenance_basis="Saved filename, history presence and run configuration; no checkpoint deserialization."))
    name, report = first_json(root, ("reports/gate-decision.json", "reports/panel-report.json", "reports/learning-curve.json"))
    if name:
        row.update(primary_report=name, execution_state="measurement_complete" if report.get("measurement_complete") is True else "measurement_incomplete",
                   scientific_decision=report.get("decision", report.get("status", "not_evaluated")))
    elif (root / "config/study.json").exists():
        stages = [("reports/complete.json", "report_present"), ("manifests/plan-complete.json", "planned"),
                  ("data/audit-complete.json", "teacher_audited"), ("data/cache-complete.json", "cached"),
                  ("data/cohort-complete.json", "prepared")]
        row["execution_state"] = next((label for path, label in stages if (root / path).exists()), "frozen_awaiting_processing")
    elif (root / "COMPLETE.json").exists():
        row["execution_state"] = read_json(root / "COMPLETE.json").get("status", "completion_record_present")
    elif (root / "run_result.json").exists():
        row["execution_state"] = "training_recorded"
    elif (root / "gate_decision.json").exists():
        decision = read_json(root / "gate_decision.json")
        row["execution_state"] = "benchmark_evaluated"
        row["scientific_decision"] = "ready_for_training" if decision.get("ready_for_sg_jepa") is True else "not_ready_for_training"
    elif (root / "evaluation_contract.json").exists():
        evaluation = read_json(root / "evaluation_contract.json")
        row["execution_state"] = "evaluation_recorded"
        row["evaluation_split"] = evaluation.get("evaluation_split")
        row["test_split_evaluated"] = evaluation.get("test_split_evaluated")
    elif entry["kind"] == "checkpoint_collection":
        row["execution_state"] = "historical_checkpoints"
    row["counts"] = {}
    for path in ("reports/cohort-audit.json", "data/cohort-complete.json", "reports/gate-decision.json", "reports/panel-report.json"):
        if (root / path).exists():
            obj = read_json(root / path)
            row["counts"][path] = {k: obj[k] for k in ("annotated_sequences", "recordings", "verified_eligible_sequences",
                "verified_eligible_recordings", "eligible_windows", "eligible_sources", "cohort_sources", "cohort_windows", "clips", "sources") if k in obj}
    return row


def inspections(project, rows):
    paths = {}
    for row in rows:
        if row["available"]:
            for path in (Path(row["location"]) / "notebook_runs").rglob("*.ipynb"):
                paths[path.resolve()] = row["id"]
    for base in (project / "notebook_runs", project / "outputs/inspections"):
        for path in base.rglob("*.ipynb"):
            paths.setdefault(path.resolve(), None)
    records = []
    for path, run_id in sorted(paths.items()):
        data = read_json(path)
        record = data.get("metadata", {}).get("fi_execution", {})
        records.append(dict(path=str(path), sha256=digest(path), associated_run_id=run_id,
            recorded_run_root=record.get("run_root"), execution_status=record.get("status", "not_recorded"),
            numerical_verification_performed=record.get("numerical_verification_performed", False),
            artifact_integrity_verified=record.get("artifact_integrity_verified", False),
            scope="Notebook execution metadata; does not establish experiment completion."))
    return records


def refresh(project=PROJECT, definitions=None):
    project = Path(project).resolve()
    definitions = definitions or registry()
    rows = [describe(project, e) for e in entries_for(project, definitions)]
    verification = project / "outputs/.organization/verification-latest.json"
    saved = read_json(verification) if verification.exists() else {}
    for row in rows:
        check = saved.get("entries", {}).get(row["id"])
        row["verification"] = (check if check and check.get("snapshot_sha256") == row.get("snapshot_sha256") else
                               {"status": "not_checked" if not check else "stale", "numerical_reconstruction": False})
    known = {Path(project / e[k]).absolute() for e in definitions["entries"] for k in ("legacy_path", "canonical_path")}
    unregistered = [str(p.relative_to(project)) for p in sorted((project / "outputs").iterdir())
                    if p.is_dir() and p.name not in {"studies", "inspections", ".organization"} and p.absolute() not in known]
    value = dict(schema=SCHEMA, generated_utc=now(), project=str(project), entries=rows,
                 inspections=inspections(project, rows),
                 unregistered_roots=unregistered, interpretation="Availability and saved receipts are separate from numerical verification and scientific decisions.")
    atomic_json(project / "outputs/catalog.json", value)
    lines = ["# Local experiment results", "", f"Generated {value['generated_utc']}.", "",
             "[Organization policy](../docs/studies/output-organization.md) · [Machine-readable catalog](catalog.json)", "",
             "Scientific decisions are preserved independently of execution state. Compatibility links point to organized bundles; backups are retained under `.organization/backups/`.", "",
             "| Study / ID | Kind | Execution | Scientific decision | Verification | Report |", "|---|---|---|---|---|---|"]
    for row in rows:
        location = os.path.relpath(row["location"], project / "outputs")
        report = f"[read]({location}/{row['primary_report']})" if row.get("primary_report") and row["available"] else "—"
        verified = row['verification']['status'] + (' / numerical reconstruction' if row['verification'].get('numerical_reconstruction') else '')
        lines.append(f"| {row['study']} / [{row['id']}]({location}/) | {row['kind']} | {row['execution_state']} | {row['scientific_decision']} | {verified} | {report} |")
    if unregistered:
        lines += ["", "Unregistered roots: " + ", ".join(unregistered)]
    lines += ["", "## Dependencies and evidence boundaries", ""]
    for row in rows:
        text = f"- **{row['id']}**: {row['description']}"
        if row.get('parents'):
            text += ' Parents: ' + ', '.join(row['parents']) + '.'
        if row.get('limitations'):
            text += ' ' + ' '.join(row['limitations'])
        unavailable = row['verification'].get('unavailable_source_evidence', [])
        if unavailable:
            text += f' {len(unavailable)} original raw-processing receipt references are unavailable locally; see the catalog.'
        lines.append(text)
    lines += ["", "## Notebook inspections", "", "Execution status below describes the notebook, separately from the experiment results above.", "",
              "| Notebook | Associated run | Execution | Numerical verification performed |", "|---|---|---|---|"]
    for record in value["inspections"]:
        path = Path(record["path"])
        relative = os.path.relpath(path, project / "outputs")
        lines.append(f"| [{path.parent.name}/{path.name}]({relative}) | {record['associated_run_id'] or 'Read recorded origin in catalog'} | {record['execution_status']} | {record['numerical_verification_performed']} |")
    (project / "outputs/README.md").write_text("\n".join(lines) + "\n")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=PROJECT)
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    subs = parser.add_subparsers(dest="command", required=True)
    subs.add_parser("refresh")
    p = subs.add_parser("resolve"); p.add_argument("run_id")
    p = subs.add_parser("new-path"); p.add_argument("study"); p.add_argument("run_id")
    p = subs.add_parser("select"); p.add_argument("run_id"); p.add_argument("--root"); p.add_argument("--new-study")
    p = subs.add_parser("verify"); p.add_argument("--readers", action="store_true")
    p = subs.add_parser("migrate"); p.add_argument("--apply", action="store_true")
    p = subs.add_parser("rollback"); p.add_argument("receipt", type=Path)
    args = parser.parse_args()
    definitions = registry(args.registry)
    if args.command == "resolve":
        print(resolve(args.project, args.run_id, definitions)); return
    if args.command == "new-path":
        if args.run_id in {e['id'] for e in entries_for(args.project, definitions)}:
            raise ValueError('Identifier already exists; resolve it to resume or choose a new ID')
        print(new_path(args.project, args.study, args.run_id)); return
    if args.command == "select":
        print(select(args.project, args.run_id, definitions, supplied_root=args.root, new_study=args.new_study)); return
    if args.command == "refresh":
        result = refresh(args.project, definitions)
        print(json.dumps({"catalog": str(args.project / "outputs/catalog.json"), "entries": len(result["entries"]), "unregistered_roots": result["unregistered_roots"]}, indent=2)); return
    from .result_migration import migrate, rollback, verify
    if args.command == "verify": result = verify(args.project, definitions, readers=args.readers)
    elif args.command == "migrate": result = migrate(args.project, definitions, apply=args.apply)
    else: result = rollback(args.project, args.receipt, definitions)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
