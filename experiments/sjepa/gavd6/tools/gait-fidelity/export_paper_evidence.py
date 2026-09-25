#!/usr/bin/env python3
"""Package only compact, completed gait-fidelity paper evidence; stdlib only.

Run on HAIC, or send this script over SSH to the HAIC study interpreter.
Source experiments are read-only. No inference, fitting, or recursive copy runs.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
from datetime import datetime, timezone


COMMON = "config.json plan.json frozen.json ledger.json".split()
FILES = {
    "walking-core-01": COMMON + """
        report.md evaluation/summary.json evaluation/comparisons.json
        evaluation/complete.json evaluation/per-person.csv
        evaluation/by-condition-person.csv evaluation/coverage.csv
        evaluation/coverage-per-person.csv evaluation/calibration.json
        data/admission.json data/mask-audit.json
    """.split(),
    "jepa-response-02": COMMON + """
        report.md evaluation/summary.json evaluation/comparisons.json
        evaluation/complete.json evaluation/per-person.csv evaluation/coverage.csv
        evaluation/coverage-per-person.csv evaluation/readout-control-comparisons.json
        evaluation/response-by-condition-person.csv evaluation/response-curves-person.csv
    """.split(),
    "readout-repair-03": COMMON + """
        development-complete.json exposure-audit.json
        development/evaluation/report.md development/evaluation/summary.json
        development/evaluation/comparisons.json development/evaluation/complete.json
        development/evaluation/per-person.csv
        development/evaluation/per-person-by-extractor.csv
        development/evaluation/coverage.csv development/evaluation/coverage-per-person.csv
    """.split(),
}
OPTIONAL = {
    "walking-core-01": ["cohort/summary.json", "evaluation/coverage-by-source-motion.csv"],
    "readout-repair-03": ["development/evaluation/power-sensitivity.json", "cohort/plan.json"],
}


def collect(root, max_file, max_total):
    base = root / "outputs/gait-fidelity"
    selected, missing, problems = {}, [], []
    total = 0

    def add(path, archive_name, required=True):
        nonlocal total
        if archive_name in selected:
            return selected[archive_name][1]
        if not path.is_file():
            (problems if required else missing).append(f"Missing: {path}")
            return None
        size = path.stat().st_size
        if size > max_file:
            problems.append(f"Too large: {path} ({size:,} bytes; per-file limit {max_file:,})")
            return None
        if total + size > max_total:
            problems.append(f"Total byte limit would be exceeded by: {path} ({size:,} bytes)")
            return None
        with path.open("rb") as stream:
            data = stream.read(min(max_file, max_total - total) + 1)
        if len(data) != size:
            problems.append(f"File changed during collection: {path}; retry after writers finish")
            return None
        selected[archive_name] = (path, data)
        total += len(data)
        return data

    def document(run, relative):
        data = add(base / run / relative, f"{run}/{relative}")
        return json.loads(data) if data is not None else {}

    def bound_file(run, value):
        if not isinstance(value, str) or not value:
            problems.append(f"Missing successful evidence path in {run} ledger")
            return
        path = Path(value)
        if not path.is_absolute():
            problems.append(f"Expected absolute completed-evidence path: {value}")
            return
        try:
            relative = path.resolve().relative_to((base / run).resolve())
        except ValueError:
            problems.append(f"Completed evidence escapes {run}: {path}")
            return
        if path.name not in {"calibration.json", "diagnostics.json", "diagnostics-summary.json"}:
            problems.append(f"Unexpected diagnostic filename: {path}")
            return
        add(path, f"{run}/{relative.as_posix()}")

    for run, names in FILES.items():
        for relative in names:
            add(base / run / relative, f"{run}/{relative}")
        cfg = document(run, "config.json")
        if cfg and cfg.get("fixture") is not False:
            problems.append(f"{run} must explicitly declare fixture=false; fixture evidence is excluded")
        ledger = document(run, "ledger.json")
        completed = ledger.get("completed", {})
        if run != "readout-repair-03" and "evaluation" not in completed:
            problems.append(f"No completed evaluation recorded for {run}")
        if any(a.get("status") in {"reserved", "submitted", "running", "accounting_pending"}
               for a in ledger.get("attempts", [])):
            problems.append(f"{run} still has unresolved worker attempts; finish/account for them first")
        for relative in OPTIONAL.get(run, []):
            add(base / run / relative, f"{run}/{relative}", required=False)
        code_root = cfg.get("code_root")
        if code_root:
            add(Path(code_root) / "gait-fidelity-release.json",
                f"{run}/provenance/gait-fidelity-release.json", required=False)
        if run == "jepa-response-02":
            profile = completed.get("followup-profile", {}).get("result", {})
            bound_file(run, profile.get("calibration_receipt"))
            diagnostics = completed.get("followup-diagnostics", {})
            result = diagnostics.get("result", {})
            if result.get("status") != "TRAINING_ONLY_DIAGNOSTICS_COMPLETE" or not result.get("exports"):
                problems.append("Response representation diagnostics are not recorded as complete")
            receipt = diagnostics.get("receipt")
            if receipt:
                bound_file(run, str(Path(receipt).parent / "diagnostics/diagnostics-summary.json"))
            else:
                problems.append("Missing response diagnostics completion receipt path")
            for entry in result.get("exports", []):
                bound_file(run, entry.get("diagnostics"))

    inventory = dict(
        schema="gait-fidelity-paper-transfer-v1",
        created_utc=datetime.now(timezone.utc).isoformat(), source_root=str(root),
        status="blocked" if problems else "ready", total_source_bytes=total,
        max_file_bytes=max_file, max_total_source_bytes=max_total,
        optional_missing=missing, problems=problems,
        scope="Published development evidence only; no new derived summaries or protected confirmation exports.",
        artifacts=[dict(path=name, source=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
                   for name, (path, data) in sorted(selected.items())],
    )
    return selected, inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--max-file-mib", type=int, default=10)
    parser.add_argument("--max-total-mib", type=int, default=50)
    args = parser.parse_args()
    if min(args.max_file_mib, args.max_total_mib) < 1:
        parser.error("Size limits must be positive")
    root, archive = args.root.expanduser().resolve(), args.archive.expanduser().resolve()
    if archive.exists():
        parser.error(f"Refusing to overwrite existing archive: {archive}")
    for run in FILES:
        if archive.is_relative_to(root / "outputs/gait-fidelity" / run):
            parser.error("Write the archive outside the source experiment directories")
    selected, inventory = collect(root, args.max_file_mib * 1024**2, args.max_total_mib * 1024**2)
    if inventory["problems"]:
        print(json.dumps(inventory, indent=2))
        return 2
    archive.parent.mkdir(parents=True, exist_ok=True)
    created = False
    try:
        with archive.open("xb") as target:
            created = True
            with tarfile.open(fileobj=target, mode="w:gz", compresslevel=6) as bundle:
                for name, (_, data) in sorted(selected.items()):
                    info = tarfile.TarInfo(name); info.size = len(data); info.mode = 0o644
                    bundle.addfile(info, io.BytesIO(data))
                data = (json.dumps(inventory, indent=2) + "\n").encode()
                info = tarfile.TarInfo("transfer-inventory.json"); info.size = len(data); info.mode = 0o644
                bundle.addfile(info, io.BytesIO(data))
                sums = [f"{entry['sha256']}  {entry['path']}" for entry in inventory["artifacts"]]
                sums.append(f"{hashlib.sha256(data).hexdigest()}  transfer-inventory.json")
                data = ("\n".join(sums) + "\n").encode()
                info = tarfile.TarInfo("SHA256SUMS"); info.size = len(data); info.mode = 0o644
                bundle.addfile(info, io.BytesIO(data))
    except Exception:
        if created:
            archive.unlink(missing_ok=True)
        raise
    print(json.dumps(dict(status="PACKAGED", archive=str(archive), files=len(selected),
        source_mib=round(inventory["total_source_bytes"] / 1024**2, 3),
        archive_mib=round(archive.stat().st_size / 1024**2, 3),
        optional_missing=inventory["optional_missing"]), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
