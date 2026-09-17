"""Build a source-data-free Slurm plan and submit its complete dependency graph."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

from run_stage import PHASES, STAGES, configuration, freeze_task_grid

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = {"inventory": "inventory.sbatch", "prepare": "prepare-bouts.sbatch",
           "audit": "audit-baselines.sbatch", "masked": "train-masked.sbatch",
           "future": "train-future.sbatch", "cache-video": "cache-video-teacher.sbatch",
           "extensions": "train-extensions.sbatch", "evaluate": "evaluate-development.sbatch",
           "calibrate": "calibrate.sbatch", "test": "evaluate-test.sbatch",
           "aggregate": "aggregate.sbatch"}


@dataclass(frozen=True)
class Node:
    stage: str
    parents: tuple[str, ...] = ()
    task_ids: tuple[int, ...] = ()


def nodes_for(command, tasks):
    ids = {"masked": tuple(int(t["task_id"]) for t in tasks
                           if t["arm"] in ("masked", "masked_index")),
           "future": tuple(int(t["task_id"]) for t in tasks
                           if t["arm"] in ("future", "future_wrong_source", "future_wrong_time"))}
    if command in PHASES:
        prefix = [Node("inventory"), Node("prepare", ("inventory",)),
                  Node("audit", ("prepare",))] if command == "pilot" else []
        training = [Node(stage, ("audit",) if prefix else (), selected)
                    for stage, selected in ids.items() if selected]
        if not training:
            raise ValueError("This phase contains no implemented training tasks")
        return prefix + training + [Node("evaluate", tuple(n.stage for n in training))]
    if command not in STAGES:
        raise ValueError(f"Unknown phase/stage: {command}")
    if command in ids and not ids[command]:
        raise ValueError(f"No {command} tasks in the selected phase")
    return [Node(command, task_ids=ids.get(command, ()))]


def dependency(value):
    if not value:
        return ()
    if not re.fullmatch(r"(?:afterok:)?[0-9]+(?::[0-9]+)*", value):
        raise ValueError("TG_DEPENDENCY must be a job ID or afterok:jobid[:jobid]")
    return tuple(value.removeprefix("afterok:").split(":"))


def require_expansion(cfg, command):
    if command not in ("develop", "confirm"):
        return
    path = cfg.root / "decisions" / "development.json"
    if not path.is_file():
        raise RuntimeError(f"A passing saved development decision is required: {path}")
    decision = json.loads(path.read_text())
    if decision.get("ready_for_expansion") is not True:
        raise RuntimeError("Development did not authorize expansion; inspect the saved decision")


def submit(command, *, config=None, phase=None, task_id=None, dry_run=False):
    cfg = configuration(config)
    phase = command if command in PHASES else phase or os.environ.get("TG_PHASE", "pilot")
    if phase not in PHASES:
        raise ValueError("TG_PHASE must be pilot, develop or confirm")
    if task_id is not None and command not in ("masked", "future"):
        raise ValueError("--task-id is only valid for a single masked or future stage")
    resume = getattr(cfg, "resume_from", None)
    if resume and (command not in ("masked", "future") or task_id is None):
        raise ValueError("Checkpoint resume requires one masked/future stage and an explicit --task-id")
    if command in ("extensions", "cache-video"):
        print(json.dumps({"status": "gated_not_implemented", "stage": command,
                          "jobs_submitted": False,
                          "reason": "E3 and frozen-video execution are not implemented in this milestone."}, indent=2))
        return []
    from gavd6_sjepa.research_directions.temporal_gait.workflow import plan_tasks
    tasks = plan_tasks(cfg, phase)
    if task_id is not None:
        tasks = [task for task in tasks if task["task_id"] == task_id]
        if not tasks:
            raise ValueError(f"Task {task_id} is not in the {phase} task plan")
    nodes = nodes_for(command, tasks)
    if os.environ.get("TG_NOTEBOOK_OUTPUT_DIR") and any(len(n.task_ids) > 1 for n in nodes):
        raise ValueError("One TG_NOTEBOOK_OUTPUT_DIR cannot be shared by a multi-task notebook array; unset it")
    initial_dependencies = dependency(os.environ.get("TG_DEPENDENCY", ""))
    concurrency = int(os.environ.get("TG_ARRAY_CONCURRENCY", "2"))
    if concurrency < 1:
        raise ValueError("TG_ARRAY_CONCURRENCY must be positive")
    if not dry_run:
        require_expansion(cfg, command)
        freeze_task_grid(cfg)
        (cfg.root / "logs").mkdir(parents=True, exist_ok=True)
    environment = {**os.environ, "GAVD6_ROOT": str(ROOT), "TG_RUN_ROOT": str(cfg.root),
                   "TG_PHASE": phase, "TG_EXECUTE": "1"}
    if resume:
        environment["TG_RESUME_FROM"] = str(resume)
    if config is not None:
        environment["TG_CONFIG"] = str(Path(config).expanduser().resolve())
    jobs, records = {}, []
    for node in nodes:
        parents = [*initial_dependencies, *(jobs[name] for name in node.parents)]
        args = ["sbatch", "--parsable", "--export=ALL", "--kill-on-invalid-dep=yes",
                f"--chdir={ROOT}", f"--account={os.environ.get('TG_ACCOUNT', 'mind')}",
                f"--partition={os.environ.get('TG_PARTITION', 'hai')}",
                f"--output={cfg.root}/logs/{phase}-{node.stage}-%A_%a.out",
                f"--error={cfg.root}/logs/{phase}-{node.stage}-%A_%a.err"]
        if parents:
            args.append("--dependency=afterok:" + ":".join(dict.fromkeys(parents)))
        if node.task_ids:
            args.append("--array=" + ",".join(map(str, node.task_ids)) + f"%{concurrency}")
        args.append(str(ROOT / "slurm/temporal-gait" / SCRIPTS[node.stage]))
        print(shlex.join(args), flush=True)
        if dry_run:
            job = "DRY_" + node.stage.replace("-", "_")
        else:
            completed = subprocess.run(args, env=environment, text=True,
                                       capture_output=True, check=True)
            job = completed.stdout.strip().split(";", 1)[0]
            if not re.fullmatch(r"[0-9]+", job):
                raise RuntimeError(f"Unexpected sbatch response: {completed.stdout!r}")
            with (cfg.root / "logs/submissions.tsv").open("a") as stream:
                stream.write("\t".join((phase, node.stage, job, ":".join(parents),
                                        ",".join(map(str, node.task_ids)), shlex.join(args), str(resume or ""))) + "\n")
        jobs[node.stage] = job
        records.append({"phase": phase, "stage": node.stage, "job_id": job,
                        "dependencies": parents, "task_ids": list(node.task_ids),
                        "resume_from": str(resume) if resume else None})
    print(json.dumps({"status": "dry_run" if dry_run else "submitted",
                      "mode": cfg.mode, "phase": phase, "jobs": records}, indent=2))
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=(*PHASES, *STAGES))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--phase", choices=PHASES)
    parser.add_argument("--task-id", type=int,
                        help="Select one global task ID for a masked/future job; required with TG_RESUME_FROM")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    return submit(args.command, config=args.config, phase=args.phase,
                  task_id=args.task_id, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
