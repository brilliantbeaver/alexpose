# Archived GAVD96 augmentation utilities

- `annotate_normal_clips.py` creates the normal-clip annotation contract.
- `extract_augmented_poses.py` extracts the selected augmented pose cohort.
- `migrate_augmented_pose_artifacts.py` validates and migrates a legacy artifact layout.

These commands remain discoverable for reproducibility, but are not part of the
active full-GAVD or AMASS research path:

```bash
uv run gavd6 gavd annotate-normal --help
uv run gavd6 gavd extract-augmented --help
uv run gavd6 gavd migrate-augmented
```

The Python files in this directory are compatibility launchers for older
commands. Their implementation lives in descriptively named source modules.
