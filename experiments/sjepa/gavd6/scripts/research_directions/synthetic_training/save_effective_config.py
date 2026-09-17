#!/usr/bin/env python3
"""Back up and save the effective five-student pilot configuration before trials."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys


def main():
    root = Path(os.environ["GAVD6_ROOT"]).resolve()
    sys.path.insert(0, str(root / "src"))
    from gavd6_sjepa.research_directions.synthetic_training.config import RunConfig

    cfg = RunConfig.from_env()
    destination = Path(os.environ["ST_CONFIG"]).resolve()
    if destination != cfg.root / "config.json":
        raise ValueError("This recipe expects ST_CONFIG=$ST_RUN_ROOT/config.json.")
    if ((cfg.root / "selectors/frozen.joblib").exists()
            or any((cfg.root / "source").glob("*/outcomes.csv"))
            or any((cfg.root / "deployment").glob("*/choices.csv"))
            or (cfg.root / "evaluation").exists()):
        raise RuntimeError("Scientific outputs exist: use a new run instead of changing this configuration.")
    cfg.context_checkpoint_key = "ema_encoder"
    cfg.context_image_size = 384
    cfg.context_frames = 64
    template = json.loads((root / "slurm/synthetic-training/pilot.example.json").read_text())
    expected = {item["student_id"]: item for item in template["students"]}
    if sorted(item["student_id"] for item in cfg.students) != sorted(expected):
        raise ValueError("Expected the five pilot students; preserve custom configurations separately.")
    for student in cfg.students:
        for field in ("config", "checkpoint"):
            relative = Path(expected[student["student_id"]][field]).relative_to("models")
            path = Path(os.environ["ST_MODEL_ROOT"]) / relative
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(f"Missing or empty student file: {path}")
            student[field] = str(path)
    cfg.validate()
    if destination.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        backup = destination.with_name(f"config.json.{stamp}.bak")
        shutil.copy2(destination, backup)
        print("Backup:", backup)
    cfg.save()
    print("SAVED EFFECTIVE CONFIGURATION:", destination)
    print(destination.read_text())


if __name__ == "__main__":
    main()
