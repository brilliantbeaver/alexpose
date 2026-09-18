"""Reconstruct retained pilot aggregates without fitting or opening raw data."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean


def _read(path):
    with Path(path).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or any(any(value in (None, "") for value in row.values()) for row in rows):
        raise ValueError(f"Empty or missing CSV observations: {path}")
    return rows


def _unique(rows, fields, name):
    keys = [tuple(row[field] for field in fields) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError(f"Duplicate {name} key: {fields}")
    return keys


def reconstruct_pilot(repo_root, output_dir=None):
    """Reconstruct frozen means/oracles, optionally exporting to a fresh folder.

    Saved outcomes are already aggregated. This is not a replay of inference or
    the unavailable person-level bootstrap. Source-backed comparisons avoid
    apparent wins due only to cross-file CSV serialization at about 1e-16.
    """
    repo = Path(repo_root).resolve()
    root = repo / "notebook_runs/synthetic-training"
    source_paths = sorted((root / "run-02-v1/source").glob("*/outcomes.csv"))
    if not source_paths:
        raise FileNotFoundError(root / "run-02-v1/source")
    paths = source_paths + [root / "run-03-v1/selectors" / name for name in
                            ("validation_predictions.csv", "validation_summary.csv", "selection.json")]
    source = [row for path in source_paths for row in _read(path)]
    decisions, summary = _read(paths[-3]), _read(paths[-2])
    selection = json.loads(paths[-1].read_text())
    source_fields = ("student_id", "domain_id", "budget", "action")
    config_fields = ("view", "kind", "parameter", "budget")
    source_keys = _unique(source, source_fields, "source measurement")
    _unique(decisions, (*config_fields, "student_id", "domain_id"), "decision")
    _unique(summary, config_fields, "configuration summary")
    for row in source + decisions + summary:
        if not math.isfinite(float(row["error"])):
            raise ValueError("Nonfinite saved error")
    source_panels = defaultdict(list)
    for row in source:
        source_panels[row["student_id"]].append(row)
    positive_budgets = {r["budget"] for r in source if int(r["budget"]) > 0}
    source_actions = {"replay", "pooled", "full_replay", *selection["lessons"]}
    all_domains = {row["domain_id"] for row in source}
    for student, panel in source_panels.items():
        if len({(r["family"], r["split"]) for r in panel}) != 1:
            raise ValueError("One source student has inconsistent family/split identity")
        expected = {(d, "0", a) for d in all_domains for a in ("original", "probe")}
        expected |= {(d, b, a) for d in all_domains for b in positive_budgets for a in source_actions}
        if {(r["domain_id"], r["budget"], r["action"]) for r in panel} != expected:
            raise ValueError(f"Incomplete source outcome panel: {student}")
    lookup = dict(zip(source_keys, (float(row["error"]) for row in source)))
    groups = defaultdict(list)
    join_delta = 0.0
    for row in decisions:
        key = tuple(row[field] for field in source_fields)
        if key not in lookup:
            raise ValueError(f"Decision has no source outcome: {key}")
        join_delta = max(join_delta, abs(float(row["error"]) - lookup[key]))
        groups[tuple(row[field] for field in config_fields)].append(row)
    if join_delta > 1e-12:
        raise ValueError("Decision/source errors disagree")
    if set(groups) != {tuple(row[f] for f in config_fields) for row in summary}:
        raise ValueError("Summary/configuration coverage differs")
    budget = str(selection["budget"])
    validation = [row for row in source if row["split"] == "validation" and row["budget"] == budget]
    students = sorted({row["student_id"] for row in validation})
    domains = sorted({row["domain_id"] for row in validation})
    expected_pairs = {(student, domain) for student in students for domain in domains}
    summary_delta = 0.0
    for row in summary:
        chosen = groups[tuple(row[field] for field in config_fields)]
        if {(r["student_id"], r["domain_id"]) for r in chosen} != expected_pairs:
            raise ValueError("Incomplete validation configuration panel")
        summary_delta = max(summary_delta, abs(float(row["error"]) - mean(float(r["error"]) for r in chosen)))
    if summary_delta > 1e-12:
        raise ValueError("Saved summary arithmetic disagrees")
    actions = ["replay", *selection["lessons"]]
    for student, domain in expected_pairs:
        if any((student, domain, budget, action) not in lookup for action in actions):
            raise ValueError("Incomplete retrospective oracle action panel")
    regret_delta = 0.0
    for row in decisions:
        best = min(lookup[(row["student_id"], row["domain_id"], row["budget"], action)] for action in actions)
        regret = float(row["regret"])
        if not math.isfinite(regret):
            raise ValueError("Nonfinite saved regret")
        regret_delta = max(regret_delta, abs(regret - (float(row["error"]) - best)))
    if regret_delta > 1e-12:
        raise ValueError("Saved regret arithmetic disagrees")
    if any(selection["methods"]["source_progress_matched"][key] != selection["methods"]["full"][key]
           for key in ("kind", "parameter")):
        raise ValueError("Source-progress control is not hyperparameter matched")

    def selected(name):
        view = "source_progress" if name == "source_progress_matched" else name
        spec = selection["methods"][name]
        found = [row for row in decisions if row["view"] == view and row["budget"] == budget
                 and row["kind"] == spec["kind"] and float(row["parameter"]) == float(spec["parameter"])]
        if {(row["student_id"], row["domain_id"]) for row in found} != expected_pairs:
            raise ValueError(f"Frozen method has incomplete observations: {name}")
        result = {(r["student_id"], r["domain_id"]): r for r in found}
        error = mean(lookup[(s, d, budget, r["action"])] for (s, d), r in result.items())
        if abs(error - float(spec["validation_error"])) > 1e-12:
            raise ValueError(f"Frozen selection error differs: {name}")
        return result, error

    table = []
    for action in ("replay", "full_replay", "pooled", selection["best_fixed"]):
        table.append({"method": action, "error": mean(lookup[(s, d, budget, action)] for s, d in expected_pairs),
                      "status": "achieved_validation_policy"})
    maps = {}
    for name in selection["methods"]:
        maps[name], error = selected(name)
        table.append({"method": name, "error": error, "status": "achieved_validation_policy"})
    shared = mean(min(mean(lookup[(student, domain, budget, action)] for student in students)
                      for action in actions) for domain in domains)
    individual = mean(min(lookup[(student, domain, budget, action)] for action in actions)
                      for student, domain in expected_pairs)
    table += [{"method": "shared_scene_oracle", "error": shared, "status": "retrospective_diagnostic"},
              {"method": "student_specific_oracle", "error": individual, "status": "retrospective_diagnostic"}]
    full, matched = maps["full"], maps["source_progress_matched"]
    changes = []
    for (s, d), row in full.items():
        control = matched[(s, d)]
        if row["action"] != control["action"]:
            changes.append(dict(student_id=s, domain_id=d, full=row["action"], matched=control["action"],
                                full_minus_matched=(lookup[(s, d, budget, row["action"])]
                                                    - lookup[(s, d, budget, control["action"])])))
    gains = [lookup[(s, d, budget, "replay")] - lookup[(s, d, budget, r["action"])] for (s, d), r in full.items()]
    fixed = next(row["error"] for row in table if row["method"] == selection["best_fixed"])
    result = {
        "evidence_status": "retained_source_aggregate_recomputed", "budget": int(budget),
        "counts": {"source_rows": len(source), "source_duplicate_keys": 0,
                   "selector_decisions": len(decisions), "selector_configurations": len(summary),
                   "validation_students": len(students), "scene_conditions": len(domains),
                   "decisions_per_configuration": len(expected_pairs)},
        "checks": {"max_source_decision_difference": join_delta, "max_summary_difference": summary_delta,
                   "max_regret_difference": regret_delta,
                   "comparison_tolerance": 1e-12},
        "table": table, "selection": selection,
        "oracle": {"actions": actions, "shared_scene_error": shared, "student_specific_error": individual,
                   "additional_absolute_reduction": shared - individual,
                   "additional_relative_reduction_percent": 100 * (shared - individual) / shared,
                   "shared_fraction_of_opportunity_beyond_fixed": (fixed - shared) / (fixed - individual)},
        "full_vs_matched_changes": changes,
        "full_vs_replay": {"improved": sum(g > 0 for g in gains), "harmed": sum(g < 0 for g in gains),
                           "tied": sum(g == 0 for g in gains), "replay_selected": sum(r["action"] == "replay" for r in full.values())},
        "units": "visible body-12 Euclidean distance / reference-box diagonal; equal frames within person, equal people within source outcome; equal student-scene outcomes",
        "limitations": ["Outcome CSVs are aggregated; per-frame metric reconstruction and person bootstrap require absent predictions and manifests.",
                        "Two validation students and repeated scene renders are not 48 independent experiments.",
                        "The retrospective oracle uses unavailable deployment outcomes; its gap is not a population bound.",
                        "Validation inspected during redesign is development evidence.",
                        "Features, trained heads and per-frame predictions are absent from the retained source bundle; gradient, feature-scaling and representation diagnoses are unavailable."],
        "source_sha256": {str(path.relative_to(repo)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        "retained_binary_inventory": {
            "npz_files": len(list((root / "run-02-v1").rglob("*.npz"))),
            "pt_files": len(list((root / "run-02-v1").rglob("*.pt"))),
        },
    }
    if output_dir is not None:
        output = Path(output_dir).resolve()
        if output == root or root in output.parents:
            raise ValueError("Audit exports must not modify historical notebook bundles")
        output.mkdir(parents=True, exist_ok=True)
        if (output / "pilot-audit.json").exists() or (output / "pilot-table.csv").exists():
            raise FileExistsError("Use a fresh audit output directory")
        (output / "pilot-audit.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        with (output / "pilot-table.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["method", "error", "status"])
            writer.writeheader()
            writer.writerows(table)
    return result
