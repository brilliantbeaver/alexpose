#!/usr/bin/env python3
"""Check the saved team without launching agents, downloads or training.

Python 3.11+ is required for the standard-library TOML parser.
Use --prompt ROLE to print a portable role prompt for a tool-driven host.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import tomllib

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]


def validate_graph(team):
    roles = [role["name"] for role in team["roles"]]
    if len(roles) != len(set(roles)):
        raise ValueError("Duplicate role names")
    tasks = {task["id"]: task for task in team["tasks"]}
    if len(tasks) != len(team["tasks"]):
        raise ValueError("Duplicate task IDs")
    for task in tasks.values():
        if task["role"] not in roles:
            raise ValueError(f"Unknown owner for {task['id']}")
        if task["reviewer"] not in roles or task["reviewer"] == task["role"]:
            raise ValueError(f"Missing independent review owner for {task['id']}")
        if not task["objective"] or not task["acceptance"] or not task["requirements"]:
            raise ValueError(f"Incomplete acceptance contract: {task['id']}")
        if task["status"] != "pending":
            raise ValueError("Definition task graph must not imply executed tutorials; use a run ledger")
        if any(dep not in tasks for dep in task["depends_on"]):
            raise ValueError(f"Unknown dependency for {task['id']}")
    visiting, done = set(), set()

    def visit(task_id):
        if task_id in visiting:
            raise ValueError("Dependency cycle")
        if task_id in done:
            return
        visiting.add(task_id)
        for dep in tasks[task_id]["depends_on"]:
            visit(dep)
        visiting.remove(task_id)
        done.add(task_id)

    for task_id in tasks:
        visit(task_id)
    runtime = team["runtime"]
    if not 1 <= runtime["max_workers"] < runtime["max_total_agents"] <= 4:
        raise ValueError("Concurrency exceeds this team's declared four-agent limit")
    if runtime["heavy_compute_jobs"] != 1:
        raise ValueError("Heavy computation must be serialized")


def validate_files(team):
    for filename in ("README.md", "protocol.md", "requirements.md", "research.md",
                     "templates/task.md", "templates/handoff.md", "templates/review.md", "templates/run.json"):
        if not (BASE / filename).is_file() or not (BASE / filename).stat().st_size:
            raise ValueError(f"Missing team artifact: {filename}")
    brief = ROOT / team["brief"]["path"]
    if hashlib.sha256(brief.read_bytes()).hexdigest() != team["brief"]["sha256"]:
        raise ValueError("The confirmed brief changed; review requirements and refresh its identity")
    if team["brief"]["binding"] != "confirmed_by_user":
        raise ValueError("The source brief is not confirmed")
    names = set()
    for role in team["roles"]:
        path = (ROOT / role["definition"]).resolve()
        if path.parent != (ROOT / ".codex/agents").resolve():
            raise ValueError(f"Role file is outside the local agent directory: {path}")
        profile = tomllib.loads(path.read_text())
        if set(profile) != {"name", "description", "developer_instructions"}:
            raise ValueError(f"Unexpected custom-agent fields: {path.name}")
        if profile["name"] != role["name"] or path.stem != role["name"]:
            raise ValueError(f"Role identity mismatch: {path.name}")
        if profile["description"] != role["description"]:
            raise ValueError(f"Role description drift: {path.name}")
        if not all(isinstance(value, str) and value.strip() for value in profile.values()):
            raise ValueError(f"Missing role instructions: {path.name}")
        if "experiments/agents/sjepa-tutorials/protocol.md" not in profile["developer_instructions"]:
            raise ValueError(f"Shared protocol missing: {path.name}")
        names.add(profile["name"])
    actual = {path.stem for path in (ROOT / ".codex/agents").glob("sjepa_*.toml")}
    if actual != names:
        raise ValueError("Team registry and saved profiles disagree")
    template = json.loads((BASE / "templates/run.json").read_text())
    if template["active_leases"] or template["accepted_handoffs"] or template["output_root"]:
        raise ValueError("Run template must not contain invented execution or output decisions")


def self_test(team):
    mutations = [
        lambda value: value["roles"].append(copy.deepcopy(value["roles"][0])),
        lambda value: value["tasks"][0].update(role="missing_role"),
        lambda value: value["tasks"][0].update(depends_on=[value["tasks"][-1]["id"]]),
        lambda value: value["tasks"][0].update(depends_on=["missing_task"]),
        lambda value: value["runtime"].update(max_total_agents=20),
        lambda value: value["tasks"][0].update(status="accepted"),
        lambda value: value["tasks"][0].update(reviewer=value["tasks"][0]["role"]),
    ]
    for mutate in mutations:
        altered = copy.deepcopy(team)
        mutate(altered)
        try:
            validate_graph(altered)
        except ValueError:
            continue
        raise AssertionError("Validator accepted a deliberately invalid team")
    print(f"Passed {len(mutations)} deliberate-invalid-input checks.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--prompt", help="Print a role's instructions for a host without named-agent selection")
    args = parser.parse_args()
    try:
        team = json.loads((BASE / "team.json").read_text())
        validate_graph(team)
        validate_files(team)
        if args.prompt:
            role = next((role for role in team["roles"] if role["name"] == args.prompt), None)
            if role is None:
                raise ValueError(f"Unknown role: {args.prompt}")
            profile = tomllib.loads((ROOT / role["definition"]).read_text())
            print(profile["developer_instructions"])
            return 0
        if args.self_test:
            self_test(team)
        print(f"Validated {len(team['roles'])} role definitions and {len(team['tasks'])} dependent tasks.")
        print("Tutorial execution is not started. Resolve its output root and scope at T00.")
        return 0
    except (ValueError, KeyError, OSError) as error:
        print(f"Team validation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
