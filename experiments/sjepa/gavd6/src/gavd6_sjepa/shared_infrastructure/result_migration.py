"""Verified bundle relocation with retained originals and compatibility paths.

No contract rewriting, model conversion, artifact deletion or remote operations.
All mutation receipts and verification reports live outside experiment bundles.
"""
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import re
import shutil
from uuid import uuid4

from .result_catalog import (
    atomic_json,
    digest,
    entries_for,
    identity,
    inside,
    locate,
    now,
    read_json,
    refresh,
    snapshot,
)


@contextmanager
def organization_lock(project):
    path = Path(project) / "outputs/.organization/operation.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another results organization operation is running") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def check_hash(root, relative, expected):
    path = inside(root, relative)
    if not path.is_file() or digest(path) != expected:
        raise ValueError(f"Artifact checksum mismatch: {path}")


def integrity(root):
    """Check declared local config/seal bindings and exact dependency snapshots."""
    root = Path(root)
    checked = 0
    dependencies = []
    unavailable_source_evidence = []
    for path in sorted(root.rglob("*.json")):
        obj = read_json(path)
        if not isinstance(obj, dict):
            continue
        config = obj.get("config_sha256", {})
        if isinstance(config, dict):
            for name, checksum in config.items():
                check_hash(root, "config/" + name, checksum)
                checked += 1
        for key in ("artifacts", "artifact_sha256"):
            bindings = obj.get(key)
            if not isinstance(bindings, dict) or not bindings:
                continue
            if not all(isinstance(v, str) and re.fullmatch(r"[0-9a-f]{64}", v) for v in bindings.values()):
                continue
            for relative, checksum in bindings.items():
                # Fold receipts use basenames; aggregate receipts use run-relative paths.
                base = path.parent if "/" not in relative and path.parent.parent.name in {"models", "support", "posture"} else root
                if path.name == "fitting-code-contract.json":
                    base = path.parent / "fitting-code"
                # The original copied gate intentionally lacks raw processing
                # artifacts. Preserve and disclose that pre-existing boundary;
                # absent models, caches, config or score receipts still fail.
                raw_receipt = path.relative_to(root).as_posix() in {"config/candidates-contract.json", "config/cohort-contract.json"}
                raw_artifact = relative.startswith(("boxes/", "poses/", "frames/", "qc/alignment-overlays/"))
                if raw_receipt and raw_artifact and not inside(root, relative).exists():
                    unavailable_source_evidence.append({"receipt": str(path.relative_to(root)), "artifact": relative})
                    continue
                if not inside(base, relative).is_file() and base == root:
                    base = path.parent
                if not base.resolve().is_relative_to(root.resolve()):
                    raise ValueError("Receipt base escapes the run")
                check_hash(base, relative, checksum)
                checked += 1
    lineage = root / "config/parent-lineage.json"
    study = root / "config/study.json"
    contract = root / "config/run-contract.json"
    sources = []
    if lineage.exists():
        obj = read_json(lineage)
        sources.append((obj["parent_root"], obj["parent_snapshot"]))
        for relative, checksum in obj["identities"].items():
            check_hash(obj["parent_root"], relative, checksum)
            checked += 1
    if study.exists():
        sources.append((read_json(study)["parent_root"], read_json(root / "config/parent-snapshot.json")))
    if contract.exists():
        obj = read_json(contract)
        if "source_run" in obj:
            sources += [(obj["source_run"], read_json(root / "config/source-snapshot.json")),
                        (obj["parent_root"], read_json(root / "config/parent-snapshot.json"))]
    for original, saved in sources:
        if snapshot(original) != saved:
            raise ValueError(f"Dependency contents, modification times or inventory changed: {original}")
        dependencies.append(dict(declared_location=original, runtime_location=str(Path(original).resolve()), files=len(saved)))
    return dict(status="passed" if checked or dependencies else "inventory_recorded",
                method="declared_config_and_artifact_hashes_and_dependency_snapshots" if checked or dependencies else "inventory_only_no_declared_hash_bindings",
                checked_bindings=checked, dependencies=dependencies, numerical_reconstruction=False,
                unavailable_source_evidence=unavailable_source_evidence,
                scope="Retained artifacts and dependencies; unavailable original raw-processing evidence is listed separately.")


def legacy_readers(project, definitions):
    """Use unchanged production readers; reconstruct the two repaired comparisons."""
    roots = {e["id"]: locate(project, e) for e in definitions["entries"] if locate(project, e).is_dir()}
    results = {}
    from threadpoolctl import threadpool_limits
    from ..research_directions.future_prediction.cache_reuse import read_parent
    from ..research_directions.future_prediction.joint_report import verify_numerics
    from ..research_directions.source_scaling.cohort import read_study
    from ..research_directions.target_accessibility.verification import verify_panel_supplement
    with threadpool_limits(limits=1):
        if "gate-v2" in roots:
            _, cohort, _, _ = read_parent(roots["gate-v2"])
            results["gate-v2"] = dict(method="unchanged_parent_cache_and_readiness_reader", windows=len(cohort), numerical_reconstruction=False)
        key = "future-innovation-direct-v3-dev-20260911"
        if key in roots:
            results[key] = dict(method="unchanged_direct_v3_numerical_verifier", result=verify_numerics(roots[key]), numerical_reconstruction=True)
        key = "iclr-bridge-cached-20260911"
        if key in roots:
            results[key] = dict(method="unchanged_accessibility_supplement", result=verify_panel_supplement(roots[key]), numerical_reconstruction=True)
        for key, root in roots.items():
            if (root / "config/study.json").exists():
                study, _ = read_study(root, require_software=False)
                results[key] = dict(method="unchanged_study_reader", protocol=study["protocol"], numerical_reconstruction=False,
                                    scope="Frozen-study reading only; this does not authorize resuming with changed software.")
        key = "swap-probe-seed7"
        if key in roots and "repaired-jepa-seed7-v2" in roots:
            provenance = read_json(roots[key] / "effective_config.json")["provenance"]
            checkpoint = roots["repaired-jepa-seed7-v2"] / Path(provenance["checkpoint"]).name
            if digest(checkpoint) != provenance["checkpoint_sha256"]:
                raise ValueError("Swap-probe encoder dependency changed")
            results[key] = dict(method="recorded_encoder_checkpoint_hash", numerical_reconstruction=False)
    return results


def verify(project, definitions, *, readers=False, publish=True):
    project = Path(project).resolve()
    entries = entries_for(project, definitions)
    before = {e["id"]: snapshot(locate(project, e)) for e in entries if locate(project, e).is_dir()}
    results = {e["id"]: integrity(locate(project, e)) for e in entries if e["id"] in before}
    if readers:
        for key, value in legacy_readers(project, definitions).items():
            results[key].update(value)
    for entry in entries:
        key = entry["id"]
        if key not in before:
            continue
        if snapshot(locate(project, entry)) != before[key]:
            raise ValueError(f"Verifier changed run artifacts: {key}")
        results[key].update(snapshot_sha256=identity(before[key]), verified_utc=now(), read_only=True)
    result = dict(status="passed", entries=results, missing=[e["id"] for e in entries if e["id"] not in before],
                  verified_utc=now(), readers_executed=readers)
    if publish:
        directory = project / "outputs/.organization"
        atomic_json(directory / ("verification-" + uuid4().hex + ".json"), result)
        atomic_json(directory / "verification-latest.json", result)
        refresh(project, definitions)
    return result


def migration_plan(project, definitions):
    project = Path(project).resolve()
    operations = []
    for entry in definitions["entries"]:
        if not entry.get("migrate"):
            continue
        old, new = (project / entry[key] for key in ("legacy_path", "canonical_path"))
        if old.is_symlink():
            if not new.is_dir() or old.resolve() != new.resolve():
                raise ValueError(f"Unexpected compatibility link: {old}")
            continue
        if not old.is_dir():
            continue
        if locate(project, entry).resolve() != old.resolve() and not new.exists():
            # Alternate machine layouts are readable, but are not the registered
            # local migration unit (a container must not become a run).
            continue
        if entry.get("identity_file") and not (old / entry["identity_file"]).is_file():
            raise ValueError(f"Registered migration source is a container or incomplete copy: {old}")
        if new.exists():
            raise FileExistsError(f"Destination exists without a completed compatibility link: {new}")
        if old.resolve().is_relative_to(new.resolve()) or new.resolve().is_relative_to(old.resolve()):
            raise ValueError("Migration roots must be separate and nonnested")
        if not new.resolve().is_relative_to(project / "outputs/studies"):
            raise ValueError("Migration destination escapes outputs/studies")
        operations.append(dict(id=entry["id"], source=str(old), destination=str(new)))
    return operations


def navigation_links(project):
    """Preserve archived notebooks' relative documentation links outside run trees."""
    project = Path(project).resolve()
    installed = []
    for name in ("docs", "slurm"):
        target = project / name
        if not target.is_dir():
            continue
        link = project / "outputs/studies" / name
        if link.is_symlink() and link.resolve() == target:
            installed.append(str(link)); continue
        if link.exists() or link.is_symlink():
            raise FileExistsError(f"Archived notebook navigation path is occupied: {link}")
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(os.path.relpath(target, link.parent), target_is_directory=True)
        installed.append(str(link))
    return installed


def _rollback(receipt_path, receipt):
    for operation in reversed(receipt["operations"]):
        old, new, backup = (Path(operation[k]) for k in ("source", "destination", "backup"))
        saved = read_json(Path(operation["snapshot"]))
        if backup.exists():
            if snapshot(backup) != saved:
                raise ValueError(f"Backup changed; cannot restore automatically: {backup}")
            if old.is_symlink() and old.resolve() == new.resolve():
                old.unlink()
            elif old.exists() or old.is_symlink():
                raise ValueError(f"Original path is occupied: {old}")
            backup.rename(old)
        if new.exists() and operation.get('destination_owned', True):
            rejected = Path(receipt_path).parent / "retained-copies" / operation["id"]
            rejected.parent.mkdir(parents=True, exist_ok=True)
            if rejected.exists():
                raise FileExistsError(rejected)
            new.rename(rejected)
            operation["retained_copy"] = str(rejected)
        operation["state"] = "restored"
        atomic_json(receipt_path, receipt)
    receipt["status"] = "rolled_back"
    atomic_json(receipt_path, receipt)
    return receipt


def rollback(project, receipt_path, definitions):
    project = Path(project).resolve()
    receipt_path = Path(receipt_path).resolve()
    if not receipt_path.is_relative_to(project / "outputs/.organization/relocations"):
        raise ValueError("Use a relocation receipt from this project's organization directory")
    receipt = read_json(receipt_path)
    if receipt.get("project") != str(project):
        raise ValueError("Relocation receipt belongs to another project")
    expected = {e["id"]: e for e in definitions["entries"] if e.get("migrate")}
    ids = [o["id"] for o in receipt["operations"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate relocation operation")
    # Validate every mutation target before making changes.
    for operation in receipt["operations"]:
        entry = expected.get(operation["id"])
        if entry is None or any(Path(operation[k]) != project / entry[v] for k, v in
                                (("source", "legacy_path"), ("destination", "canonical_path"))):
            raise ValueError("Relocation operation differs from the registered bundle paths")
        for key in ("source", "destination", "backup", "snapshot"):
            target = Path(operation[key]).absolute()
            if not target.is_relative_to(project / "outputs") or ".." in target.parts:
                raise ValueError("Relocation receipt references a path outside project outputs")
        if not Path(operation["backup"]).resolve().is_relative_to(project / "outputs/.organization/backups"):
            raise ValueError("Backup escapes organization storage")
        if not Path(operation["snapshot"]).resolve().is_relative_to(receipt_path.parent):
            raise ValueError("Snapshot escapes relocation receipt directory")
    with organization_lock(project):
        if receipt.get('status') == 'rolled_back':
            return receipt
        result = _rollback(receipt_path, receipt)
        refresh(project, definitions)
        return result


def migrate(project, definitions, *, apply=False):
    project = Path(project).resolve()
    if not apply:
        plan = migration_plan(project, definitions)
        return dict(status="plan_only", operations=plan, policy="copy2, verify, retain originals, install compatibility links, verify readers; rollback on failure")
    with organization_lock(project):
        plan = migration_plan(project, definitions)
        if not plan:
            navigation_links(project)
            return dict(status="already_organized", verification=verify(project, definitions, readers=True))
        verify(project, definitions, readers=True, publish=False)
        label = now().replace(":", "").replace("+", "_") + "-" + uuid4().hex[:8]
        directory = project / "outputs/.organization/relocations" / label
        directory.mkdir(parents=True)
        receipt_path = directory / "relocation.json"
        receipt = dict(status="copying", project=str(project), created_utc=now(), operations=[])
        try:
            for operation in plan:
                old, new = Path(operation["source"]), Path(operation["destination"])
                saved = snapshot(old)
                snap = directory / (operation["id"] + "-snapshot.json")
                atomic_json(snap, saved)
                operation.update(backup=str(project / "outputs/.organization/backups" / label / old.name),
                                 snapshot=str(snap), state="copying", destination_owned=False)
                receipt["operations"].append(operation)
                atomic_json(receipt_path, receipt)
                new.parent.mkdir(parents=True, exist_ok=True)
                new.mkdir(exist_ok=False)
                operation['destination_owned'] = True
                atomic_json(receipt_path, receipt)
                shutil.copytree(old, new, copy_function=shutil.copy2, dirs_exist_ok=True)
                if snapshot(new) != saved or snapshot(old) != saved:
                    raise ValueError(f"Copied bundle or source changed: {old}")
                integrity(new)
                operation["state"] = "copy_verified"
                atomic_json(receipt_path, receipt)
            for operation in receipt["operations"]:
                old, new, backup = (Path(operation[k]) for k in ("source", "destination", "backup"))
                saved = read_json(operation["snapshot"])
                if snapshot(old) != saved or snapshot(new) != saved:
                    raise ValueError("Artifacts changed before compatibility switch")
                backup.parent.mkdir(parents=True, exist_ok=True)
                old.rename(backup)
                old.symlink_to(os.path.relpath(new, old.parent), target_is_directory=True)
                operation["state"] = "linked"
                atomic_json(receipt_path, receipt)
            receipt['navigation_links'] = navigation_links(project)
            result = verify(project, definitions, readers=True)
            for operation in receipt["operations"]:
                if snapshot(operation["backup"]) != read_json(operation["snapshot"]):
                    raise ValueError("Retained original differs from its snapshot")
            receipt.update(status="complete", finished_utc=now(), verification=result)
            atomic_json(receipt_path, receipt)
            return dict(status="complete", bundles=len(plan), receipt=str(receipt_path), backups_retained=True)
        except BaseException as exc:
            receipt["error"] = repr(exc)
            atomic_json(receipt_path, receipt)
            _rollback(receipt_path, receipt)
            refresh(project, definitions)
            raise
