"""Deterministic AMASS cohort plans, with identity and test boundaries intact.

Motion names select candidates, not diagnoses or validated walking intervals.
Every eligible inventory row receives a decision, including protected test rows.
No raw motion is opened while constructing a plan.
"""
from __future__ import annotations

import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .common import atomic_json, read_json, sha256
from ..synthetic_training_v2.contracts import digest

PRESETS = ("named_walking", "treadmill_walking", "all_eligible", "reviewed")
MANIFESTS = ("amass_raw_inventory_eligible.csv", "amass_subject_registry.csv",
             "amass_subject_splits.csv")


def _boolean(value, *, unknown=False):
    word = str(value).strip().lower()
    if word in {"true", "1", "yes"}: return True
    if word in {"false", "0", "no"}: return False
    if unknown and word == "unknown": return "unknown"
    raise ValueError(f"Explicit true/false{'/unknown' if unknown else ''} required; got {value!r}")


def _read(path, required):
    table = pd.read_csv(path, keep_default_na=False)
    if missing := set(required) - set(table):
        raise ValueError(f"{path} lacks columns {sorted(missing)}")
    return table


def _root(path):
    folder = Path(path).expanduser().resolve()
    return folder / "amass" if (folder / "amass").is_dir() else folder


def motion_label(name):
    """A conservative filename taxonomy, always marked as metadata-derived."""
    name = str(name).lower()
    if "treadmill" in name and re.search(r"(?:^|[_\W])(?:jog|run)", name):
        return "treadmill_jog_or_run"
    if "treadmill" in name:
        for token, label in (("slow", "treadmill_slow"), ("fast", "treadmill_fast"), ("norm", "treadmill_normal")):
            if token in name: return label
        return "treadmill_unspecified"
    if "walk" in name: return "named_walking"
    if re.search(r"damage.*(?:leg|foot)", name): return "acted_limb_condition_candidate"
    return "other_or_unresolved"


def _reservations(path, people):
    """Retain unknown histories for exploration; never infer test innocence."""
    result = {str(p): dict(reserved="unknown", exposure="unknown") for p in people.identity}
    if not path: return result
    table = _read(path, {"person_id", "canonical_person_id", "original_split", "reserved", "exposure"})
    if table.person_id.duplicated().any(): raise ValueError("Reservation ledger repeats a person_id")
    authorities = dict(zip(people.identity.astype(str), people.split.astype(str)))
    for person, rows in table.groupby("canonical_person_id", sort=True):
        if person not in authorities:
            raise ValueError(f"Reservation identity is absent from the approved registry: {person}")
        if set(rows.original_split.astype(str)) != {authorities[person]}:
            raise ValueError(f"Reservation aliases disagree with the original split: {person}")
        if not rows.person_id.astype(str).eq(str(person)).all():
            raise ValueError('Reservation person_id must use the audited canonical identity; resolve folder aliases through the registry first')
        flags = [_boolean(v, unknown=True) for v in rows.reserved]
        exposure = sorted(set(rows.exposure.astype(str).str.strip()))
        if not all(exposure): raise ValueError("Reservation exposure may be unknown but not blank")
        # Conflicting aliases cannot create an unexposed confirmation person.
        result[person] = dict(reserved=True if True in flags else "unknown" if "unknown" in flags else False,
                              exposure=exposure[0] if len(exposure) == 1 else "conflicting_alias_exposure")
    return result


def _reviews(path, known_paths):
    if not path: return {}
    table = _read(path, {"relative_path", "start_s", "end_s", "decision", "motion_label", "reviewer", "evidence"})
    if not set(table.relative_path).issubset(known_paths):
        raise ValueError("Motion review names a path outside the eligible inventory")
    if table[["relative_path", "start_s", "end_s"]].duplicated().any():
        raise ValueError("Duplicate reviewed intervals")
    output = {}
    for path, group in table.groupby("relative_path", sort=True):
        rows = []
        previous_end = -np.inf
        for row in group.sort_values("start_s").to_dict("records"):
            start, end = float(row["start_s"]), float(row["end_s"])
            if not np.isfinite([start, end]).all() or start < 0 or end <= start or start < previous_end - 1e-9:
                raise ValueError(f"Reviewed intervals must be finite, ordered and nonoverlapping: {path}")
            previous_end = end
            if row["decision"] not in {"include", "exclude"}:
                raise ValueError("Review decision must be include or exclude")
            if any(not str(row[k]).strip() for k in ("motion_label", "reviewer", "evidence")):
                raise ValueError("Reviewed intervals require a label, reviewer and evidence")
            rows.append(dict(row, start_s=start, end_s=end))
        output[path] = rows
    return output


def plan_cohort(config, output=None):
    """Plan every permitted nonoverlap interval without reading motion arrays.

    An explicit review file takes precedence for every motion it mentions;
    unreviewed portions of those motions are not silently admitted. Original
    test windows are planned but stay locked to ordinary preparation.
    """
    options = config.get("cohort", {})
    preset = options.get("preset", "named_walking")
    if preset not in PRESETS: raise ValueError(f"Unknown AMASS cohort preset: {preset}")
    data = config["data"]
    samples, hz = int(data["samples"]), float(data["hz"])
    if samples < 8 or not np.isfinite(hz) or hz <= 0:
        raise ValueError("Cohort windows need >=8 samples and a positive finite rate")
    span, stride = (samples - 1) / hz, samples / hz
    folder = _root(config["preparation"]["manifest_dir"])
    files = {str((folder / name).resolve()): sha256(folder / name) for name in MANIFESTS}
    inventory = _read(folder / MANIFESTS[0], {"relative_path", "source_dataset", "subject_id_candidate", "motion_id", "sha256", "num_frames", "mocap_framerate", "status"})
    registry = _read(folder / MANIFESTS[1], {"subject_id_candidate", "identity", "identity_audit_status", "excluded"})
    splits = _read(folder / MANIFESTS[2], {"identity", "split"})
    if inventory.relative_path.duplicated().any() or registry.subject_id_candidate.duplicated().any():
        raise ValueError("Inventory paths and registry subject aliases must be unique")
    people = splits[["identity", "split"]].drop_duplicates()
    if people.identity.duplicated().any() or not set(people.split).issubset({"train", "validation", "test"}):
        raise ValueError("An audited person must have exactly one recognized original split")
    approved = registry.loc[registry.identity_audit_status.eq("approved") & ~registry.excluded.map(_boolean)].copy()
    if not set(people.identity).issubset(set(approved.identity)):
        raise ValueError("Split manifest contains an identity without approved registry evidence")
    for key in ("reservation_csv", "motion_review_csv"):
        if options.get(key): files[str(Path(options[key]).expanduser().resolve())] = sha256(options[key])
    reservations = _reservations(options.get("reservation_csv"), people)
    reviews = _reviews(options.get("motion_review_csv"), set(inventory.relative_path))
    if preset == "reviewed" and not reviews:
        raise ValueError("The reviewed preset requires an explicit interval review CSV")
    columns = ["subject_id_candidate", "identity"]
    if "known_downstream_overlap" in approved: columns.append("known_downstream_overlap")
    joined = inventory.merge(approved[columns], on="subject_id_candidate", how="left", validate="many_to_one")
    joined = joined.merge(people, on="identity", how="left", validate="many_to_one")
    # Exact duplicate content must never be multiplied or put across people.
    duplicate_paths = set()
    for source_hash, rows in joined.groupby("sha256", sort=True):
        if len(str(source_hash)) != 64 or re.fullmatch(r"[0-9a-fA-F]{64}", str(source_hash)) is None:
            raise ValueError("Every eligible inventory motion requires a SHA-256 hash")
        if len(rows) > 1:
            if rows[["identity", "split"]].fillna("missing").drop_duplicates().shape[0] != 1:
                raise ValueError("Identical motion content crosses an identity or original split")
            duplicate_paths.update(sorted(rows.relative_path)[1:])
    raw_root = Path(config["preparation"]["amass_root"]).expanduser().resolve()
    records, decisions = [], []
    for row in joined.sort_values(["relative_path"]).to_dict("records"):
        path = str(row["relative_path"])
        relative = Path(path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Manifest paths must stay inside the AMASS root")
        raw_path = raw_root / relative
        candidate_label = motion_label(row["motion_id"])
        person = str(row["identity"]) if pd.notna(row["identity"]) else ""
        original_split = str(row["split"]) if pd.notna(row["split"]) else ""
        authority = reservations.get(person, dict(reserved="unknown", exposure="unknown"))
        duration = (float(row["num_frames"]) - 1) / float(row["mocap_framerate"]) if float(row["mocap_framerate"]) > 0 else np.nan
        reason = ""
        if not person or not original_split: reason = "unapproved_or_unsplit_identity"
        elif row["status"] != "ok": reason = "inventory_status_not_ok"
        elif path in duplicate_paths: reason = "duplicate_source_content"
        elif authority["reserved"] is True: reason = "reserved_identity"
        elif "known_downstream_overlap" in row and _boolean(row["known_downstream_overlap"]): reason = "known_downstream_identity_overlap"
        elif not np.isfinite(duration) or duration < span - 1e-9: reason = "short_or_invalid_duration"
        named = candidate_label in {"named_walking", "treadmill_normal", "treadmill_slow", "treadmill_fast", "treadmill_unspecified"}
        selected_by_name = preset == "all_eligible" or (preset == "named_walking" and named) or (preset == "treadmill_walking" and candidate_label in {"treadmill_normal", "treadmill_slow", "treadmill_fast"})
        intervals = reviews.get(path, [])
        if not reason and path not in reviews and not selected_by_name: reason = "outside_declared_motion_preset"
        if not reason and not intervals:
            intervals = [dict(start_s=0., end_s=duration, decision="include", motion_label=candidate_label,
                              reviewer="metadata_only", evidence="motion_name; walking interval not independently reviewed")]
        count = 0
        next_available_s = 0.
        if not reason:
            for interval in intervals:
                if interval["end_s"] > duration + 1e-8:
                    raise ValueError(f"Reviewed interval exceeds the source duration: {path}")
                if interval["decision"] == "exclude": continue
                # Review endpoints describe inclusive physical support. Two
                # adjacent reviewed intervals may share a boundary, but two
                # planned windows must not reuse that sample. Decide the exact
                # start here and freeze it; preparation must never move it.
                interval_start = max(float(interval['start_s']), next_available_s)
                length = interval["end_s"] - interval_start
                number = max(0, int(np.floor((length - span + 1e-9) / stride)) + 1)
                for index in range(number):
                    start = interval_start + index * stride
                    record = dict(relative_path=path, raw_path=str(raw_path), source_dataset=row["source_dataset"],
                        subject_id_candidate=row["subject_id_candidate"], person_id=person, canonical_person_id=person,
                        original_split=original_split, split={"train":"train", "validation":"development", "test":"confirmation"}[original_split],
                        role={"train":"train", "validation":"development", "test":"final"}[original_split],
                        motion_id=row["motion_id"], motion_hash=row["sha256"], historical_motion_sha256=row["sha256"],
                        duration_s=duration, start_s=float(start), end_s=float(start + span), parent_audited_start_s=float(start),
                        motion_label=interval["motion_label"], label_source="reviewed_interval" if path in reviews else "filename_candidate",
                        locomotion_status="reviewed_locomotion" if path in reviews else "metadata_candidate",
                        audit_reviewer=interval["reviewer"], audit_evidence=interval["evidence"], audit_date="recorded_in_cohort_plan",
                        review_mode="technical_geometry", reserved=authority["reserved"], exposure=authority["exposure"],
                        available=raw_path.is_file(), locked=original_split == "test")
                    record["source_family_id"] = digest([record["motion_hash"], record["start_s"], samples, hz])[:24]
                    records.append(record); count += 1
                    next_available_s = start + stride
            if not count: reason = "review_excluded_or_no_complete_interval"
        decisions.append(dict(relative_path=path, canonical_person_id=person, original_split=original_split,
            source_dataset=row["source_dataset"], motion_label=candidate_label, duration_s=duration if np.isfinite(duration) else None,
            decision=reason or ("locked_test_candidate" if original_split == "test" else "selected_candidate"),
            planned_windows=count, available=raw_path.is_file(), reserved=authority["reserved"], exposure=authority["exposure"]))
    if not records: raise ValueError("No full-length candidate intervals; inspect preset, reviews and duration requirements")
    table = pd.DataFrame(records)
    # Shuffled-reference controls need distinct windows, not augmented copies.
    inadequate = set(table.loc[table.split.eq("train")].groupby("canonical_person_id").size().loc[lambda x:x < 2].index)
    if inadequate:
        records = [r for r in records if r["canonical_person_id"] not in inadequate]
        for row in decisions:
            if row["canonical_person_id"] in inadequate and row["planned_windows"]:
                row.update(decision="fewer_than_two_training_windows", planned_windows=0)
    table = pd.DataFrame(records)
    roles = {}
    for role, group in table.groupby("split", sort=True):
        roles[role] = dict(people=int(group.canonical_person_id.nunique()), motions=int(group.relative_path.nunique()), windows=len(group))
    rules = dict(preset=preset, samples=samples, hz=hz, stride_s=stride, raw_root=str(raw_root),
        selection="metadata_candidates_or_reviewed_intervals", technical_qc="finite_nondegenerate_geometry; no normality threshold",
        unknown_exposure="exploratory_train_development_only", original_test="locked_separate_confirmation")
    identity = digest(dict(files=files, rules=rules, records=[{k:v for k,v in r.items() if k != "available"} for r in records]))
    # Rendering multiplies a physical source interval; none of these derived
    # rows is an additional person or independent recording.
    specs = config['preparation'].get('estimators', [])
    held_family = config.get('held_extractor', 'vitpose')
    levels = len(data.get('movement_levels_deg', [0, 5, 10, 15]))
    cameras = len(data.get('cameras', [{}, {}]))
    multiplier = 2 * cameras * 2  # physical original/mirror and clear/occluded
    estimates = {}
    for role, population in roles.items():
        count = population['windows']
        physical_levels = levels - (role == 'train')
        extractors = sum(s['family'] != held_family for s in specs) if role == 'train' else len(specs)
        renders = count * physical_levels * multiplier
        track_rows = count * (physical_levels + 1) * multiplier * extractors * 3  # explicit no-change + naming conditions
        estimates[role] = dict(rendered_clips=renders, rendered_frames=renders*samples,
            derived_track_rows=track_rows, array_bytes_one_copy=track_rows*samples*288,
            array_bytes_three_retained_copies=track_rows*samples*288*3,
            uncompressed_rgb_bytes_generated_not_retained=renders*samples*640*480*3)
    result = dict(schema="gait-fidelity-cohort-v1", identity=identity, files=files, rules=rules, records=records,
        inventory=decisions, exclusions=[r for r in decisions if not r["planned_windows"]],
        summary=dict(eligible_inventory_motions=len(inventory), planned_motions=int(table.relative_path.nunique()),
            planned_windows=len(records), roles=roles, candidates_missing_on_disk=int((~table.available).sum()),
            motion_labels=table.groupby("motion_label").relative_path.nunique().to_dict(),
            preparation_estimates=estimates,
            independent_confirmation=False, limitation="Filename labels are candidates; geometry checks do not validate gait or clinical status"))
    if output is not None:
        output = Path(output); output.mkdir(parents=True, exist_ok=True)
        target = output / "manifest.json"
        if target.exists():
            previous = read_json(target)
            if previous["identity"] != identity: raise FileExistsError("Existing cohort differs; create a new run directory")
        else:
            atomic_json(target, result)
            pd.DataFrame(decisions).to_csv(output / "inventory.csv", index=False)
            table.to_csv(output / "windows.csv", index=False)
            atomic_json(output / "summary.json", result["summary"])
    return result


def load_cohort(config, partition="development", require_available=True):
    """Read the frozen plan and verify its authorities before opening motions."""
    if partition not in {"development", "confirmation"}: raise ValueError("Unknown cohort partition")
    path = Path(config["cohort"]["plan_path"])
    plan = read_json(path)
    if plan.get("schema") != "gait-fidelity-cohort-v1": raise ValueError("Unknown cohort schema")
    for source, expected in plan["files"].items():
        if sha256(source) != expected: raise ValueError(f"Cohort authority changed: {source}")
    fresh = plan_cohort(config)
    if fresh["identity"] != plan["identity"]:
        raise ValueError("Cohort configuration or planned population changed after initialization")
    frozen_records = [{k:v for k,v in r.items() if k != 'available'} for r in plan['records']]
    fresh_records = [{k:v for k,v in r.items() if k != 'available'} for r in fresh['records']]
    if frozen_records != fresh_records or plan['rules'] != fresh['rules']:
        raise ValueError('Saved cohort records were changed after planning')
    rows = [r for r in plan["records"] if (r["split"] == "confirmation") == (partition == "confirmation")]
    if partition == "confirmation":
        lock_path = config.get("cohort", {}).get("confirmation_lock")
        if not lock_path: raise PermissionError("Original AMASS test identities require a frozen confirmation declaration")
        lock = read_json(lock_path)
        if lock.get("schema") != "gf-confirmation-lock-v1" or lock.get("cohort_identity") != plan["identity"]:
            raise PermissionError("Confirmation declaration does not bind this cohort")
        if not lock.get("reviewed_by") or not lock.get("evidence") or not lock.get("checkpoint_hashes"):
            raise PermissionError("Confirmation requires documented review and frozen checkpoint hashes")
        expected_people = sorted({r["canonical_person_id"] for r in rows})
        if sorted(lock.get("person_ids", [])) != expected_people:
            raise PermissionError("Confirmation declaration must bind the entire predeclared test candidate population")
        for checkpoint, expected in lock["checkpoint_hashes"].items():
            if sha256(checkpoint) != expected: raise PermissionError("A frozen confirmation checkpoint changed")
        if lock.get('exposure_ledger'):
            ledger = lock['exposure_ledger']
            if sha256(ledger) != lock.get('exposure_ledger_sha256'):
                raise PermissionError('The separately reviewed confirmation exposure ledger changed')
            authorities = pd.DataFrame([dict(identity=p, split='test') for p in expected_people])
            table = _read(ledger, {'person_id', 'canonical_person_id', 'original_split', 'reserved', 'exposure'})
            if set(table.canonical_person_id) != set(expected_people):
                raise PermissionError('Confirmation exposure review must cover exactly the predeclared test people')
            reviewed = _reservations(ledger, authorities)
            for row in rows:
                if row['exposure'] not in {'unknown', 'unexposed_verified'}:
                    raise PermissionError('Previously recorded exposure cannot be reclassified as an untouched test person')
                row.update(reviewed[row['canonical_person_id']])
        if any(r["exposure"] != "unexposed_verified" or r["reserved"] is not False for r in rows):
            raise PermissionError("Every confirmation identity requires verified unexposed history and explicit nonreservation")
    if not rows: raise ValueError(f"No windows in partition {partition}")
    if require_available:
        missing = sorted({r["raw_path"] for r in rows if not Path(r["raw_path"]).is_file()})
        if missing: raise FileNotFoundError(f"{len(missing)} planned AMASS files unavailable; first: {missing[0]}")
    return pd.DataFrame(rows), plan["exclusions"], plan["identity"]
