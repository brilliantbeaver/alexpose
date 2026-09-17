"""Run or inspect one explicitly configured temporal-gait workflow stage."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
STAGES = ("inventory", "prepare", "audit", "masked", "future", "extensions",
          "cache-video", "evaluate", "calibrate", "test", "aggregate")
PHASES = ("pilot", "develop", "confirm")


def configuration(path=None):
    from gavd6_sjepa.research_directions.temporal_gait.config import RunConfig
    cfg = RunConfig.from_env(config_path=path)
    cfg.validate(check_input_paths=False)
    return cfg


def planned(cfg, phase="pilot", stage=None, task_id=None):
    from gavd6_sjepa.research_directions.temporal_gait.workflow import plan_tasks
    tasks = plan_tasks(cfg, phase)
    if task_id is not None and task_id not in {int(t["task_id"]) for t in tasks}:
        raise ValueError(f"Task {task_id} is not in the {phase} task plan")
    return {"status": "plan_only", "mode": cfg.mode, "phase": phase,
            "stage": stage, "task_id": task_id, "run_root": str(cfg.root),
            "tasks": tasks, "resolved_config": cfg.to_dict(),
            "media_opened": False, "jobs_submitted": False}


def freeze_task_grid(cfg):
    """Freeze metadata before submission; do not open manifests or source media."""
    from gavd6_sjepa.research_directions.temporal_gait.contracts import write_once_json
    from gavd6_sjepa.research_directions.temporal_gait.workflow import plan_tasks
    destination = cfg.root / "manifests" / "task-grid.json"
    payload = {"schema_version": 1, "resolved_config": cfg.scientific_dict(),
               "tasks": plan_tasks(cfg, "confirm")}
    if destination.is_symlink():
        raise ValueError("Task grid must not be a symlink")
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_once_json(destination, payload)
    return destination


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", type=Path)
    result.add_argument("--stage", choices=STAGES)
    result.add_argument("--task-id", type=int)
    result.add_argument("--phase", choices=PHASES, default="pilot")
    result.add_argument("--role", choices=("development", "test"), default="development")
    result.add_argument("--dry-run", action="store_true")
    result.add_argument("--write-task-grid", action="store_true",
                        help="Write only the immutable complete task mapping before submission.")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if args.dry_run and args.write_task_grid:
        raise ValueError("A dry run cannot write a task grid")
    cfg = configuration(args.config)
    if args.dry_run:
        result = planned(cfg, args.phase, args.stage, args.task_id)
    elif args.write_task_grid:
        result = {"status": "task_grid_frozen", "path": str(freeze_task_grid(cfg))}
    else:
        if args.stage is None:
            raise ValueError("Choose --stage, --dry-run or --write-task-grid")
        from gavd6_sjepa.research_directions.temporal_gait.workflow import run_stage
        result = run_stage(cfg, args.stage, task_id=args.task_id,
                           role=args.role, phase=args.phase)
    print(json.dumps(result, indent=2, default=str))
    if result.get("status") in {"incomplete", "failed", "error"}:
        raise RuntimeError(f"Stage did not complete: {result['status']}; inspect retained artifacts")
    return result


if __name__ == "__main__":
    main()
