#!/usr/bin/env python3
"""Bounded, explicitly heuristic AMASS input screening for automated development.

No human review or unknown external reservation is invented. Outputs are new
files, accepted only by the experiment's automated_development input mode.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
VERSION = "stv2-kinematic-screen-v1"
THRESHOLDS = dict(hip_width_m=[0.08, 0.60], leg_segment_m=[0.15, 0.85],
                  pelvis_head_height_m=[0.25, 1.50], ankle_excursion_m_min=0.08,
                  ankle_correlation_max=-0.15, alternating_crossings=[2, 16],
                  ankle_vertical_excursion_m_max=0.65)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True, help="New directory only")
    p.add_argument("--reservation-csv", type=Path, action="append",
                   help="Additional authoritative identity/reservation records; repeatable")
    p.add_argument("--max-candidates", type=int, default=24,
                   help="Maximum source recordings reconstructed (default 24; at most 64)")
    return p


def screen_joints(joints):
    """Unvalidated kinematic plausibility rules, not a scientific gait classifier."""
    import numpy as np
    from review_motion import _fixed_views

    x = np.asarray(joints)
    if x.shape != (64, 22, 3) or not np.isfinite(x).all():
        return dict(passed=False, failures=["nonfinite_or_wrong_shape"], metrics={})
    try:
        _, side, _ = _fixed_views(x)
    except ValueError:
        return dict(passed=False, failures=["degenerate_initial_hips"], metrics={})
    span = lambda value: float(np.percentile(value, 95) - np.percentile(value, 5))
    hip = np.linalg.norm(x[:, 1] - x[:, 2], axis=1)
    lengths = np.stack([np.linalg.norm(x[:, a] - x[:, b], axis=1)
                        for a, b in ((1, 4), (4, 7), (2, 5), (5, 8))], axis=1)
    ankles = side[:, [7, 8], 0] - side[:, [0], 0]
    excursions = [span(ankles[:, index]) for index in range(2)]
    correlation = float(np.corrcoef(ankles.T)[0, 1]) if min(excursions) > 1e-8 else 1.0
    difference = ankles[:, 0] - ankles[:, 1]
    # Ignore tiny sign changes around zero rather than counting numerical noise.
    signs = np.sign(difference[np.abs(difference) > 0.02])
    crossings = int(np.sum(signs[1:] != signs[:-1]))
    metrics = dict(hip_width_min_m=float(hip.min()), hip_width_max_m=float(hip.max()),
                   leg_segment_min_m=float(lengths.min()), leg_segment_max_m=float(lengths.max()),
                   pelvis_head_height_m=float(np.median(x[:, 15, 1] - x[:, 0, 1])),
                   ankle_excursion_m=excursions, ankle_correlation=correlation,
                   alternating_crossings=crossings,
                   ankle_vertical_excursion_m=[span(x[:, index, 1]) for index in (7, 8)])
    metric_values = [number for value in metrics.values()
                     for number in (value if isinstance(value, list) else [value])]
    if not np.isfinite(metric_values).all():
        return dict(passed=False, failures=["nonfinite_derived_metrics"], metrics={})
    failures = []
    for name, low, high, bounds in (
        ("hip_width", metrics["hip_width_min_m"], metrics["hip_width_max_m"], THRESHOLDS["hip_width_m"]),
        ("leg_segment", metrics["leg_segment_min_m"], metrics["leg_segment_max_m"], THRESHOLDS["leg_segment_m"]),
        ("pelvis_head_height", metrics["pelvis_head_height_m"], metrics["pelvis_head_height_m"], THRESHOLDS["pelvis_head_height_m"]),
    ):
        if low < bounds[0] or high > bounds[1]:
            failures.append(name)
    if min(excursions) < THRESHOLDS["ankle_excursion_m_min"]:
        failures.append("ankle_excursion")
    if not np.isfinite(correlation) or correlation > THRESHOLDS["ankle_correlation_max"]:
        failures.append("ankle_correlation")
    if not THRESHOLDS["alternating_crossings"][0] <= crossings <= THRESHOLDS["alternating_crossings"][1]:
        failures.append("alternating_crossings")
    if max(metrics["ankle_vertical_excursion_m"]) > THRESHOLDS["ankle_vertical_excursion_m_max"]:
        failures.append("ankle_vertical_excursion")
    return dict(passed=not failures, failures=failures, metrics=metrics)


def _known_people(table, paths):
    """Carry existing facts; blanks stay unknown and protected aliases stay excluded."""
    people = {}
    for person, split in table[["person_id", "original_split"]].drop_duplicates().itertuples(index=False, name=None):
        people[str(person)] = dict(person_id=str(person), canonical_person_id=str(person),
                                  original_split=str(split), reserved="unknown", exposure="unknown")
    seen_canonical, seen_exposure = {}, {}
    used = []
    for path in dict.fromkeys(Path(p).expanduser().resolve() for p in paths):
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            if not {"person_id", "canonical_person_id", "original_split", "reserved", "exposure"} <= set(reader.fieldnames or []):
                raise ValueError(f"Reservation CSV has an unexpected schema: {path}")
            for row in reader:
                person = str(row.get("person_id") or "").strip()
                split = str(row.get("original_split") or "").strip()
                if not person or split not in {"train", "validation", "test"}:
                    raise ValueError(f"Incomplete person identity or original split in {path}")
                record = people.setdefault(person, dict(person_id=person, canonical_person_id=person,
                                                        original_split=split, reserved="unknown", exposure="unknown"))
                if record["original_split"] != split:
                    raise ValueError(f"Conflicting original splits for {person} in {path}")
                canonical = str(row.get("canonical_person_id") or "").strip()
                if canonical:
                    if person in seen_canonical and seen_canonical[person] != canonical:
                        raise ValueError(f"Conflicting canonical identity for {person}")
                    record["canonical_person_id"] = seen_canonical[person] = canonical
                reserved = str(row.get("reserved") or "unknown").strip().lower() or "unknown"
                if reserved not in {"true", "false", "unknown", "1", "0"}:
                    raise ValueError(f"Unsupported reservation value for {person}: {reserved}")
                reserved = {"1": "true", "0": "false"}.get(reserved, reserved)
                # Any known reservation dominates a false/unknown entry in another worksheet.
                if reserved == "true" or (record["reserved"] != "true" and reserved != "unknown"):
                    record["reserved"] = reserved
                exposure = str(row.get("exposure") or "").strip()
                if exposure and exposure != "unknown":
                    if person in seen_exposure and seen_exposure[person] != exposure:
                        raise ValueError(f"Conflicting exposure records for {person}")
                    record["exposure"] = seen_exposure[person] = exposure
        used.append(str(path))
    # Follow explicit aliases, including aliases whose current source files are unavailable.
    for person, record in people.items():
        canonical, seen = record["canonical_person_id"], {person}
        while canonical in people and people[canonical]["canonical_person_id"] != canonical:
            if canonical in seen:
                raise ValueError(f"Cyclic canonical identity records involving {person}")
            seen.add(canonical)
            canonical = people[canonical]["canonical_person_id"]
        record["canonical_person_id"] = canonical
    protected = set()
    for canonical in {row["canonical_person_id"] for row in people.values()}:
        aliases = [row for row in people.values() if row["canonical_person_id"] == canonical]
        if any(row["reserved"] == "true" or row["original_split"] == "test" for row in aliases) or len(
                {row["original_split"] for row in aliases}) > 1:
            protected.add(canonical)
    return people, protected, used


def _write_csv(path, columns, rows):
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    if not 4 <= args.max_candidates <= 64:
        raise ValueError("--max-candidates must be between 4 and 64")
    if str(ROOT / "src") not in sys.path:
        sys.path.insert(0, str(ROOT / "src"))
    if str(Path(__file__).parent) not in sys.path:
        sys.path.insert(0, str(Path(__file__).parent))
    import numpy as np
    from gavd6_sjepa.research_directions.motion_preservation.body_geometry import SMPLHBody, PARENTS, JOINT_NAMES
    from gavd6_sjepa.research_directions.motion_preservation.motion_data import load_amass_manifest, load_motion
    from haic_inputs import AUDIT_COLUMNS, RESERVATION_COLUMNS
    from review_motion import _sha256, _fixed_views, _write_html

    config_path = args.config.expanduser().resolve(strict=True)
    config = json.loads(config_path.read_text())
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists; choose a new directory: {output}")
    for name in ("amass_root", "body_model_root", "dmpl_root", "manifest_dir"):
        if config.get(name) and output.is_relative_to(Path(config[name]).expanduser().resolve()):
            raise ValueError(f"Output must be outside source assets: {name}")
    table = load_amass_manifest(config["manifest_dir"], config["amass_root"])
    work = config_path.parent.parent
    paths = list(args.reservation_csv or [])
    paths += [work / "inputs/review-drafts" / name for name in
              ("person-reservations.csv", "person-reservations.draft.csv")]
    if config.get("reservation_csv"):
        paths.append(Path(config["reservation_csv"]))
    for path in args.reservation_csv or []:
        if not path.expanduser().is_file():
            raise FileNotFoundError(f"Explicit reservation source is missing: {path}")
    people, protected, used = _known_people(table, paths)
    candidates = []
    for row in table.to_dict("records"):
        if row["original_split"] not in {"train", "validation"} or not row["available"] or float(row["duration_s"]) < 10.52:
            continue
        record = people[str(row["person_id"])]
        if record["canonical_person_id"] in protected:
            continue
        name = Path(str(row["relative_path"])).name.lower()
        rank = 0 if "treadmill_norm" in name else 1 if "walk" in name else None
        if rank is not None:
            candidates.append((rank, str(row["person_id"]), str(row["relative_path"]), row))
    candidates.sort(key=lambda value: value[:3])
    body = SMPLHBody(config["body_model_root"], config.get("dmpl_root"), device="cpu", batch_size=16)
    if hasattr(body, "dmpl_root") and output.is_relative_to(body.dmpl_root.resolve()):
        raise ValueError("Output must be outside source assets: inferred dmpl_root")
    selected, attempts, chosen, counts, hashes = [], [], set(), {"train": 0, "validation": 0}, {}
    root = Path(config["amass_root"]).expanduser().resolve(strict=True)
    output.mkdir(parents=True, exist_ok=False)
    report = dict(schema="stv2-automated-inputs-v1", algorithm=VERSION, review_mode="automated_development",
                   evidence_status="automated-source-screen", thresholds=THRESHOLDS,
                   policy="Unvalidated kinematic screening; no human review or unknown external reservation verification",
                   created_utc=datetime.now(timezone.utc).isoformat(), config=str(config_path),
                   config_sha256=_sha256(config_path), helper_sha256=_sha256(Path(__file__)),
                   reservation_sources=[dict(path=p, sha256=_sha256(p)) for p in used],
                   known_protected_canonical_people=sorted(protected), max_candidates=args.max_candidates,
                   starts_s=[5.0, 8.0], frames=64, fps=25, attempts=attempts)
    for _, _, _, row in candidates:
        person = people[str(row["person_id"])]
        canonical, split = person["canonical_person_id"], row["original_split"]
        if canonical in chosen or counts[split] >= 2:
            continue
        if len(attempts) >= args.max_candidates:
            break
        attempt = dict(relative_path=str(row["relative_path"]), person_id=str(row["person_id"]),
                       canonical_person_id=canonical, original_split=split, windows=[])
        attempts.append(attempt)
        print(f"Screen {len(attempts)}/{args.max_candidates}: {row['relative_path']}", flush=True)
        windows = []
        try:
            raw = Path(row["raw_path"]).resolve(strict=True)
            if not raw.is_relative_to(root) or not raw.is_file():
                raise ValueError("Source must be a file within configured AMASS root")
            digest = _sha256(raw)
            for start in (5.0, 8.0):
                motion = load_motion(row, start_s=start, duration_s=64 / 25, fps=25)
                sequence = body.forward(motion)
                if sequence.coordinate_system != "y_up":
                    raise ValueError("SMPL-H returned unexpected coordinates")
                expected = start + np.arange(64) / 25
                if not np.allclose(motion.timestamps, expected, rtol=0, atol=1e-9):
                    raise ValueError("Unexpected motion sample times")
                screen = screen_joints(sequence.joints)
                attempt["windows"].append(dict(start_s=start, **screen))
                windows.append((start, sequence.joints, motion, screen))
            if _sha256(raw) != digest:
                raise ValueError("Source changed during screening")
            attempt["passed"] = all(window[3]["passed"] for window in windows)
            if attempt["passed"]:
                selected.extend((row, person, raw, digest, window) for window in windows)
                chosen.add(canonical)
                counts[split] += 1
                hashes[str(raw)] = digest
        except Exception as exc:
            attempt.update(passed=False, error=f"{type(exc).__name__}: {exc}")
        if min(counts.values()) == 2:
            break
    report["counts"] = counts
    if min(counts.values()) < 2:
        report["status"] = "insufficient_candidates"
        (output / "screen.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        raise ValueError(f"Could not screen two people per split within {args.max_candidates} candidates; diagnostics: {output / 'screen.json'}")
    for raw, digest in hashes.items():
        if _sha256(raw) != digest:
            raise ValueError(f"Selected source changed during screening: {raw}")
    playback = output / "playback"
    playback.mkdir()
    evidence_dir = output / "evidence"
    evidence_dir.mkdir()
    browser_windows, audits, traces, times = [], [], [], []
    date = datetime.now(timezone.utc).date().isoformat()
    for index, (row, person, raw, digest, (start, joints, motion, screen)) in enumerate(selected, 1):
        front, side, view = _fixed_views(joints)
        evidence_path = evidence_dir / f"window-{index:03d}.json"
        evidence = str(evidence_path)
        evidence_path.write_text(json.dumps(dict(
            status="pass", relative_path=str(row["relative_path"]), start_s=start,
            screen_version=VERSION, reviewed_by="algorithm", review_mode="automated_development",
            helper_sha256=_sha256(Path(__file__)),
            geometry_helper_sha256=_sha256(Path(__file__).with_name("review_motion.py")),
            body_geometry_sha256=_sha256(ROOT / "src/gavd6_sjepa/research_directions/motion_preservation/body_geometry.py"),
            motion_loader_sha256=_sha256(ROOT / "src/gavd6_sjepa/research_directions/motion_preservation/motion_data.py"),
            source_sha256=digest, canonical_person_id=person["canonical_person_id"],
            metrics=screen["metrics"], thresholds=THRESHOLDS,
            playback_reference=f"{playback / 'review.html'}#window={index}",
        ), indent=2, allow_nan=False) + "\n")
        audits.append(dict(relative_path=str(row["relative_path"]), start_s=start,
                           locomotion_status="algorithm_screened_locomotion", audit_reviewer=VERSION,
                           audit_evidence=evidence, audit_date=date, exposure=person["exposure"],
                           canonical_person_id=person["canonical_person_id"]))
        browser_windows.append(dict(window=index, relative_path=str(row["relative_path"]), raw_path=str(raw),
                                    person_id=str(row["person_id"]), original_split=str(row["original_split"]),
                                    gender=motion.gender, start_s=start, end_s=float(motion.timestamps[-1]),
                                    frames=64, fps=25, source_fps=motion.metadata["source_fps"],
                                    source_sha256=digest, timestamps=motion.timestamps.tolist(),
                                    evidence_reference=evidence, view_coordinates=view,
                                    front=front.tolist(), side=side.tolist(), screen=screen))
        traces.append(joints)
        times.append(motion.timestamps)
    _write_html(playback / "review.html", dict(windows=browser_windows, parents=PARENTS.tolist(),
                                              joint_names=list(JOINT_NAMES), review_mode="automated_development"))
    with (playback / "joints.npz").open("xb") as stream:
        np.savez_compressed(stream, joints=np.stack(traces), timestamps=np.stack(times),
                            parents=PARENTS, joint_names=np.asarray(JOINT_NAMES))
    included_people = [record for record in people.values() if record["canonical_person_id"] in chosen]
    _write_csv(output / "locomotion-audit.csv", AUDIT_COLUMNS, audits)
    _write_csv(output / "person-reservations.csv", RESERVATION_COLUMNS, included_people)
    report.update(status="inputs_created", windows=[{key: value for key, value in w.items()
                                                     if key not in {"front", "side"}} for w in browser_windows],
                  artifacts={name: _sha256(output / name) for name in (
                      "locomotion-audit.csv", "person-reservations.csv", "playback/review.html", "playback/joints.npz")})
    report["artifacts"].update({str(path.relative_to(output)): _sha256(path)
                                for path in sorted(evidence_dir.glob("*.json"))})
    (output / "screen.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (playback / "metadata.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Automated development inputs: {output}", flush=True)
    print("8 algorithm-screened windows; no human review or external reservation clearance is claimed.", flush=True)
    return report


def main(argv=None):
    try:
        run(parser().parse_args(argv))
    except Exception as exc:
        print(f"STV2 automated inputs: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
