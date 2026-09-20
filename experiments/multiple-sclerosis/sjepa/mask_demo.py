"""One entry point for notebook and command-line mask demonstrations."""

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import shutil

import numpy as np

from . import masking_v2, viz


@dataclass
class MaskDemo:
    animation: Path
    timeline: Path
    manifest: Path
    masks: np.ndarray


def write_mask_demo(seq, out_dir, *, frame_group=4, fps=15, seed=0, n_samples=8):
    """Sample once, then derive every demo artifact from exactly the same bits.

    Replaces the two historical GIF paths as aliases of the current animation,
    since notebook buffers can still refer to those filenames. This does not
    cause an old notebook cell to change its code; a manifest makes stale output
    verifiable and the timeline works even in viewers that freeze GIFs.
    """
    seq = np.asarray(seq)
    if seq.ndim != 3 or seq.shape[1] != 33 or seq.shape[0] == 0 or seq.shape[2] < 2:
        raise ValueError("seq must be one nonempty (frames, 33, channels) model window")
    if not isinstance(frame_group, (int, np.integer)) or frame_group < 1 or len(seq) % frame_group:
        raise ValueError("frame_group must divide the model window length")
    if not isinstance(n_samples, (int, np.integer)) or n_samples < 2:
        raise ValueError("A stochastic demo requires at least two mask samples")
    blocks = len(seq) // frame_group
    masks = masking_v2.sample_mask_batch(n_samples, 33, blocks, np.random.default_rng(seed))
    masks = masks.reshape(n_samples, blocks, 33)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    animation = viz.skeleton_animation(
        seq, out_dir / "mask_demo_samples.gif", target_mask=masks,
        frame_group=frame_group, fps=fps,
        title="Stochastic masks: same motion, successive draws",
    )
    timeline = viz.mask_timeline(masks, out_dir / "mask_demo_timeline.png", frame_group)
    aliases = [out_dir / "mask_demo.gif", out_dir / "mask_demo_temporal.gif"]
    for alias in aliases:
        shutil.copyfile(animation, alias)
    digest = hashlib.sha256(animation.read_bytes()).hexdigest()
    details = {
        "schema_version": 1,
        "seed": seed,
        "num_samples": n_samples,
        "frames_per_sample": len(seq),
        "total_frames": n_samples * len(seq),
        "frame_group": frame_group,
        "layout": "sample, time_block, joint; True = masked target",
        "hip_visible_blocks_per_sample": (~masks[:, :, [23, 24]]).sum(1).tolist(),
        "visible_frames_per_joint": ((~masks).sum((0, 1)) * frame_group).tolist(),
        "all_joints_have_context": bool((~masks).any((0, 1)).all()),
        "all_joints_have_targets": bool(masks.any((0, 1)).all()),
        "artifacts": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [animation, *aliases, timeline]
        },
        "target_masks": masks.astype(int).tolist(),
    }
    manifest = out_dir / "mask_demo_manifest.json"
    manifest.write_text(json.dumps(details, indent=2) + "\n")
    assert all(hashlib.sha256(alias.read_bytes()).hexdigest() == digest for alias in aliases)
    return MaskDemo(animation, timeline, manifest, masks)


def verify_mask_demo(out_dir):
    """Detect a legacy cell overwriting any GIF since the canonical generation."""
    out_dir = Path(out_dir)
    details = json.loads((out_dir / "mask_demo_manifest.json").read_text())
    mismatched = []
    for filename, digest in details["artifacts"].items():
        path = out_dir / filename
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            mismatched.append(filename)
    if mismatched:
        raise ValueError(
            f"Stale mask-demo artifact(s): {', '.join(mismatched)}. "
            "An older notebook cell may have overwritten them. Run "
            "python scripts/scripts_mask_demo.py to regenerate the matching set."
        )
    return details
