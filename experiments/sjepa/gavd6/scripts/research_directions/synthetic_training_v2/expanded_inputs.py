#!/usr/bin/env python3
"""Select a larger, explicitly machine-screened AMASS development panel.

The kinematic rule, physical clock and evidence schema are inherited from the
original automated screen. This command changes the requested panel size, not
the meaning of an audit. It never invents human review or reservation clearance.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import automated_inputs as legacy


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True, help="New output directory only")
    result.add_argument("--train-people", type=int, default=24)
    result.add_argument("--development-people", type=int, default=8)
    result.add_argument("--windows-per-person", type=int, default=4)
    result.add_argument("--max-candidates", type=int, default=512,
                        help="Maximum source recordings reconstructed, at most 4096")
    result.add_argument("--exclude-bundle", type=Path, action="append", default=[],
                        help="Exclude every person in an earlier bundle; repeatable")
    result.add_argument("--reservation-csv", type=Path, action="append", default=[],
                        help="Additional authoritative identity/reservation records; repeatable")
    return result


def window_starts(count):
    """Four defaults are 5, 8, 11 and 14 seconds, with no overlapping samples."""
    if type(count) is not int or not 1 <= count <= 8:
        raise ValueError("windows-per-person must be an integer between 1 and 8")
    return [5.0 + 3.0 * index for index in range(count)]


def excluded_people(bundle_paths, people):
    """Read identity metadata only; no previous coordinates or labels are opened.

    Both historical names and their current canonical aliases are excluded. The
    caller verifies its old run separately; these hashes record which identity
    manifests this panel actually used rather than certifying old predictions.
    """
    from review_motion import _sha256

    excluded, sources = set(), []
    for folder in dict.fromkeys(Path(path).expanduser().resolve(strict=True) for path in bundle_paths):
        manifest = folder / "manifest.json"
        metadata = json.loads(manifest.read_text())
        records = metadata.get("records")
        if (metadata.get("schema") != "coco-body12-xy-v1"
                or not isinstance(records, list) or not records):
            raise ValueError(f"Exclusion bundle needs a nonempty body-12 identity manifest: {manifest}")
        names = set()
        for record in records:
            for name in ("person_id", "canonical_person_id"):
                value = record.get(name)
                if not isinstance(value, str) or not value.strip() or value != value.strip():
                    raise ValueError(f"Exclusion bundle has an incomplete {name}: {manifest}")
                names.add(value)
        excluded.update(names)
        excluded.update(people[name]["canonical_person_id"] for name in names if name in people)
        sources.append(dict(path=str(manifest), sha256=_sha256(manifest), people=sorted(names)))
    return excluded, sources


def candidate_rows(table, people, protected, excluded, starts):
    """Order candidates by metadata before observing any reconstructed motion."""
    minimum_duration = starts[-1] + 63 / 25
    candidates = {}
    for row in table.to_dict("records"):
        duration = float(row["duration_s"])
        if (row["original_split"] not in {"train", "validation"} or not row["available"]
                or not math.isfinite(duration) or duration < minimum_duration):
            continue
        person = people[str(row["person_id"])]
        canonical = person["canonical_person_id"]
        if canonical in protected or canonical in excluded or person["person_id"] in excluded:
            continue
        name = Path(str(row["relative_path"])).name.lower()
        rank = 0 if "treadmill_norm" in name else 1 if "walk" in name else None
        if rank is not None:
            candidates.setdefault(canonical, []).append((rank, str(row["relative_path"]), row))
    # A person with many failing recordings must not consume the entire bound
    # before other people receive their first attempt. The ordering is fixed
    # from metadata and does not respond to downstream estimator/model errors.
    ordered = []
    for canonical, recordings in candidates.items():
        for index, (rank, relative, row) in enumerate(sorted(recordings, key=lambda value: value[:2])):
            ordered.append((index, rank, canonical, relative, row))
    return [value[4] for value in sorted(ordered, key=lambda value: value[:4])]


def _write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _validate_request(args):
    starts = window_starts(args.windows_per_person)
    requested = {"train": args.train_people, "validation": args.development_people}
    if any(type(value) is not int or not 1 <= value <= 256 for value in requested.values()):
        raise ValueError("Requested people per split must be integers between 1 and 256")
    if not sum(requested.values()) <= args.max_candidates <= 4096:
        raise ValueError("max-candidates must cover requested people and be at most 4096")
    return requested, starts


def run(args):
    requested, starts = _validate_request(args)
    if str(ROOT / "src") not in sys.path:
        sys.path.insert(0, str(ROOT / "src"))
    import numpy as np
    from gavd6_sjepa.research_directions.motion_preservation.body_geometry import SMPLHBody, PARENTS, JOINT_NAMES
    from gavd6_sjepa.research_directions.motion_preservation.motion_data import load_amass_manifest, load_motion
    from haic_inputs import AUDIT_COLUMNS, RESERVATION_COLUMNS
    from review_motion import _sha256, _fixed_views, _write_html

    config_path = args.config.expanduser().resolve(strict=True)
    config = json.loads(config_path.read_text())
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists; choose a new directory: {output}")
    protected_directories = [Path(config[name]).expanduser().resolve() for name in
                             ("amass_root", "body_model_root", "dmpl_root", "manifest_dir") if config.get(name)]
    protected_directories.extend(Path(path).expanduser().resolve() for path in args.exclude_bundle)
    if any(output.is_relative_to(path) for path in protected_directories):
        raise ValueError("Output must be outside source assets and excluded bundles")

    table = load_amass_manifest(config["manifest_dir"], config["amass_root"])
    work = config_path.parent.parent
    paths = list(args.reservation_csv)
    for path in paths:
        if not path.expanduser().is_file():
            raise FileNotFoundError(f"Explicit reservation source is missing: {path}")
    paths.extend(work / "inputs/review-drafts" / name for name in
                 ("person-reservations.csv", "person-reservations.draft.csv"))
    if config.get("reservation_csv"):
        paths.append(Path(config["reservation_csv"]))
    people, protected, used = legacy._known_people(table, paths)
    excluded, exclusion_sources = excluded_people(args.exclude_bundle, people)
    candidates = candidate_rows(table, people, protected, excluded, starts)
    available_people = {split: len({people[str(row["person_id"])]["canonical_person_id"]
                                   for row in candidates if row["original_split"] == split})
                        for split in requested}
    if any(available_people[split] < requested[split] for split in requested):
        output.mkdir(parents=True, exist_ok=False)
        _write_json(output / "screen.json", dict(
            schema="stv2-expanded-inputs-v1", status="insufficient_metadata_candidates",
            requested_people=requested, available_people=available_people,
            counts={"train": 0, "validation": 0}, attempts=[],
            metadata_eligible_recordings=len(candidates), starts_s=starts,
            minimum_duration_s=starts[-1] + 63 / 25, windows_per_person=args.windows_per_person,
            config=str(config_path), config_sha256=_sha256(config_path),
            exclusion_sources=exclusion_sources, excluded_previous_people=sorted(excluded),
            known_protected_canonical_people=sorted(protected),
            reason="Unique eligible people are fewer than requested before any body-model loading or screening.",
            evidence_status="automated-source-screen", review_mode="automated_development",
            created_utc=datetime.now(timezone.utc).isoformat(),
        ))
        raise ValueError(f"Requested {requested}, metadata permits at most {available_people}; "
                         f"no smaller panel was accepted. See {output / 'screen.json'}")
    body = SMPLHBody(config["body_model_root"], config.get("dmpl_root"), device="cpu", batch_size=16)
    if getattr(body, "dmpl_root", None) and output.is_relative_to(Path(body.dmpl_root).resolve()):
        raise ValueError("Output must be outside inferred DMPL source assets")

    output.mkdir(parents=True, exist_ok=False)
    selected, attempts, chosen = [], [], set()
    counts, hashes = {"train": 0, "validation": 0}, {}
    root = Path(config["amass_root"]).expanduser().resolve(strict=True)
    helper_hashes = dict(
        helper_sha256=_sha256(Path(__file__)),
        screening_helper_sha256=_sha256(Path(legacy.__file__)),
        geometry_helper_sha256=_sha256(SCRIPT_DIR / "review_motion.py"),
        body_geometry_sha256=_sha256(ROOT / "src/gavd6_sjepa/research_directions/motion_preservation/body_geometry.py"),
        motion_loader_sha256=_sha256(ROOT / "src/gavd6_sjepa/research_directions/motion_preservation/motion_data.py"),
    )
    report = dict(
        schema="stv2-expanded-inputs-v1", algorithm=legacy.VERSION,
        review_mode="automated_development", evidence_status="automated-source-screen",
        thresholds=legacy.THRESHOLDS,
        policy="Unvalidated kinematic screen; no human review or external reservation clearance",
        selection="One recording per canonical person; all requested windows must pass; no silent downsizing",
        candidate_order="Round-robin canonical people; treadmill_norm before walk within each person; relative-path tie break",
        created_utc=datetime.now(timezone.utc).isoformat(), config=str(config_path),
        config_sha256=_sha256(config_path), **helper_hashes,
        requested_people=requested, windows_per_person=args.windows_per_person,
        starts_s=starts, minimum_duration_s=starts[-1] + 63 / 25, frames=64, fps=25,
        max_candidates=args.max_candidates, metadata_eligible_recordings=len(candidates),
        available_people=available_people,
        reservation_sources=[dict(path=path, sha256=_sha256(path)) for path in used],
        exclusion_sources=exclusion_sources, excluded_previous_people=sorted(excluded),
        known_protected_canonical_people=sorted(protected), attempts=attempts,
    )
    for row in candidates:
        person = people[str(row["person_id"])]
        canonical, split = person["canonical_person_id"], row["original_split"]
        if canonical in chosen or counts[split] >= requested[split]:
            continue
        if len(attempts) >= args.max_candidates:
            break
        attempt = dict(relative_path=str(row["relative_path"]), person_id=str(row["person_id"]),
                       canonical_person_id=canonical, original_split=split, windows=[])
        attempts.append(attempt)
        print(f"Screen {len(attempts)}/{args.max_candidates}: {row['relative_path']}", flush=True)
        try:
            raw = Path(row["raw_path"]).resolve(strict=True)
            if not raw.is_relative_to(root) or not raw.is_file():
                raise ValueError("Source must be a file within configured AMASS root")
            source_hash = _sha256(raw)
            windows = []
            for start in starts:
                motion = load_motion(row, start_s=start, duration_s=64 / 25, fps=25)
                sequence = body.forward(motion)
                if sequence.coordinate_system != "y_up":
                    raise ValueError("SMPL-H returned unexpected coordinates")
                expected = start + np.arange(64) / 25
                if motion.timestamps.shape != (64,) or not np.allclose(motion.timestamps, expected, rtol=0, atol=1e-9):
                    raise ValueError("Unexpected motion sample times")
                screening = legacy.screen_joints(sequence.joints)
                attempt["windows"].append(dict(start_s=start, **screening))
                windows.append((start, sequence.joints.copy(), motion, screening))
            if _sha256(raw) != source_hash:
                raise ValueError("Source changed during screening")
            attempt["passed"] = all(window[3]["passed"] for window in windows)
            if attempt["passed"]:
                selected.extend((row, person, raw, source_hash, window) for window in windows)
                chosen.add(canonical)
                counts[split] += 1
                hashes[str(raw)] = source_hash
        except Exception as exc:
            attempt.update(passed=False, error=f"{type(exc).__name__}: {exc}")
        if counts == requested:
            break

    report["counts"] = counts
    if counts != requested:
        report["status"] = "insufficient_candidates"
        _write_json(output / "screen.json", report)
        raise ValueError(f"Requested {requested}, screened {counts} within {len(attempts)} attempts; "
                         f"no smaller panel was accepted. See {output / 'screen.json'}")
    for raw, source_hash in hashes.items():
        if _sha256(raw) != source_hash:
            raise ValueError(f"Selected source changed during screening: {raw}")

    playback, evidence_dir = output / "playback", output / "evidence"
    playback.mkdir()
    evidence_dir.mkdir()
    browser_windows, audits, traces, times = [], [], [], []
    date = datetime.now(timezone.utc).date().isoformat()
    for index, (row, person, raw, source_hash, (start, joints, motion, screening)) in enumerate(selected, 1):
        front, side, view = _fixed_views(joints)
        evidence_path = evidence_dir / f"window-{index:04d}.json"
        _write_json(evidence_path, dict(
            status="pass", relative_path=str(row["relative_path"]), start_s=start,
            screen_version=legacy.VERSION, reviewed_by="algorithm", review_mode="automated_development",
            **helper_hashes, source_sha256=source_hash, canonical_person_id=person["canonical_person_id"],
            metrics=screening["metrics"], thresholds=legacy.THRESHOLDS,
            playback_reference=f"{playback / 'review.html'}#window={index}",
        ))
        audits.append(dict(relative_path=str(row["relative_path"]), start_s=start,
                           locomotion_status="algorithm_screened_locomotion", audit_reviewer=legacy.VERSION,
                           audit_evidence=str(evidence_path), audit_date=date, exposure=person["exposure"],
                           canonical_person_id=person["canonical_person_id"]))
        browser_windows.append(dict(
            window=index, relative_path=str(row["relative_path"]), raw_path=str(raw),
            person_id=str(row["person_id"]), original_split=str(row["original_split"]),
            gender=motion.gender, start_s=start, end_s=float(motion.timestamps[-1]),
            frames=64, fps=25, source_fps=motion.metadata["source_fps"],
            source_sha256=source_hash, timestamps=motion.timestamps.tolist(),
            evidence_reference=str(evidence_path), view_coordinates=view,
            front=front.tolist(), side=side.tolist(), screen=screening,
        ))
        traces.append(joints)
        times.append(motion.timestamps)
    _write_html(playback / "review.html", dict(windows=browser_windows, parents=PARENTS.tolist(),
                                              joint_names=list(JOINT_NAMES), review_mode="automated_development"))
    with (playback / "joints.npz").open("xb") as stream:
        np.savez_compressed(stream, joints=np.stack(traces), timestamps=np.stack(times),
                            parents=PARENTS, joint_names=np.asarray(JOINT_NAMES))
    included_people = [record for record in people.values() if record["canonical_person_id"] in chosen]
    legacy._write_csv(output / "locomotion-audit.csv", AUDIT_COLUMNS, audits)
    legacy._write_csv(output / "person-reservations.csv", RESERVATION_COLUMNS, included_people)
    report.update(
        status="inputs_created", selected_canonical_people=sorted(chosen),
        selected_recordings=len(hashes), selected_windows=len(audits),
        windows=[{key: value for key, value in window.items() if key not in {"front", "side"}}
                 for window in browser_windows],
        artifacts={name: _sha256(output / name) for name in
                   ("locomotion-audit.csv", "person-reservations.csv", "playback/review.html", "playback/joints.npz")},
    )
    report["artifacts"].update({str(path.relative_to(output)): _sha256(path)
                                for path in sorted(evidence_dir.glob("*.json"))})
    _write_json(output / "screen.json", report)
    _write_json(playback / "metadata.json", report)
    print(f"Expanded automated development inputs: {output}", flush=True)
    print(f"{counts['train']} training + {counts['validation']} development people; "
          f"{len(audits)} machine-screened windows; no human review or confirmation claim.", flush=True)
    return report


def main(argv=None):
    try:
        run(parser().parse_args(argv))
    except Exception as exc:
        print(f"STV2 expanded inputs: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
