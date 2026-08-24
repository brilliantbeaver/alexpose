# Data-preparation utilities

- `annotate_normal_clips.py` creates the normal-clip annotation contract.
- `extract_augmented_poses.py` extracts the selected augmented pose cohort.
- `migrate_augmented_pose_artifacts.py` validates and migrates a legacy artifact layout.

Run utilities from the project root so relative configuration and `.env`
discovery remain predictable, for example:

```bash
uv run python scripts/data_preparation/migrate_augmented_pose_artifacts.py
```
