"""Regenerate or verify all mask-demo outputs without relying on a notebook buffer.

Run from the experiment directory:
    python scripts/scripts_mask_demo.py
    python scripts/scripts_mask_demo.py --check
"""

import argparse
from pathlib import Path
import sys

EXP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP_DIR))

from sjepa.config import get_config
from sjepa.data import sliding_windows
from sjepa.mask_demo import verify_mask_demo, write_mask_demo
from sjepa.splits import load_full_registry, partition_records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--samples", type=int, default=8)
    args = parser.parse_args()
    artifacts = EXP_DIR / "artifacts"
    if not args.check:
        # Match the notebook bootstrap without overriding explicit shell settings.
        try:
            from dotenv import load_dotenv
        except ImportError:
            pass
        else:
            load_dotenv(EXP_DIR.parents[1] / ".env")
        cfg = get_config()
        records, registry = load_full_registry(EXP_DIR)
        train_recs, _, _ = partition_records(records, registry, 0)
        seq = sliding_windows(train_recs[0].load_norm(), cfg.window_frames, cfg.window_stride)[0]
        result = write_mask_demo(seq, artifacts, frame_group=cfg.frame_group,
                                 fps=cfg.target_fps, seed=args.seed, n_samples=args.samples)
        print("Animation:", result.animation)
        print("Timeline:", result.timeline)
    details = verify_mask_demo(artifacts)
    print("All artifact hashes match:", details["total_frames"], "frames;",
          details["num_samples"], "mask samples.")
    counts = details["visible_frames_per_joint"]
    print("Visible hip frames: left", counts[23], "right", counts[24])
    print("Visible hip blocks per sample:", details["hip_visible_blocks_per_sample"])


if __name__ == "__main__":
    main()
