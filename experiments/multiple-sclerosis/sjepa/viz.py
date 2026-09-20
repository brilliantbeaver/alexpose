"""Visualisation helpers so the notebooks can show what the model sees.

Two kinds of viewing:

* :func:`show_video` embeds a browser-compatible MP4 into the notebook as an
  HTML5 player. It works the same locally and in Colab because the bytes travel
  inside the notebook, no web server needed.
* :func:`skeleton_animation` draws the BlazePose stick figure over time and saves
  an animated gif, optionally colouring the masked joints differently so a learner
  can literally see the anatomical mask.

Colours follow one hue per class for the scatter plots elsewhere: normal is blue,
ms is orange, pd is green.
"""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Dict, List, Optional, Sequence

import numpy as np

CLASS_COLORS: Dict[str, str] = {"normal": "#2b6cb0", "ms": "#dd6b20", "pd": "#38a169"}

# BlazePose-33 skeleton edges (from ambient MEDIAPIPE_33_CONNECTIONS).
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6), (0, 9), (0, 10), (9, 10),
    (11, 12), (11, 23), (12, 24), (23, 24),
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
]


def _temporal_target_mask(
    num_joints: int,
    target_mask: Optional[np.ndarray],
    masked_joints: Optional[Sequence[int]],
) -> np.ndarray:
    """Normalize visualization masks to ``(time_blocks, joints)``.

    ``True`` means hidden target and ``False`` means visible context. The static
    ``masked_joints`` argument is retained for legacy diagrams, but an all-joint
    union is rejected because it discards the temporal information and leaves no
    visible context.
    """
    if target_mask is not None and masked_joints is not None:
        raise ValueError("Pass either target_mask or masked_joints, not both")

    if target_mask is not None:
        temporal_mask = np.asarray(target_mask, dtype=bool)
        if temporal_mask.ndim == 1:
            if temporal_mask.size == 0 or temporal_mask.size % num_joints:
                raise ValueError(
                    "target_mask must contain a non-zero whole number of joint blocks"
                )
            temporal_mask = temporal_mask.reshape(-1, num_joints)
        elif (
            temporal_mask.ndim != 2
            or temporal_mask.shape[0] == 0
            or temporal_mask.shape[1] != num_joints
        ):
            raise ValueError(
                f"target_mask must have shape (time_blocks, {num_joints})"
            )
    else:
        temporal_mask = np.zeros((1, num_joints), dtype=bool)
        if masked_joints is not None:
            indices = np.asarray(list(masked_joints), dtype=int)
            if ((indices < 0) | (indices >= num_joints)).any():
                raise ValueError(f"masked_joints must be between 0 and {num_joints - 1}")
            temporal_mask[0, indices] = True

    if temporal_mask.all():
        raise ValueError(
            "Mask leaves no visible context anywhere. Pass the full temporal target_mask; "
            "do not collapse it to the union of masked joints."
        )
    return temporal_mask


def _target_mask_by_frame(temporal_mask: np.ndarray, num_frames: int) -> np.ndarray:
    """Expand token-time masks to frames without stretching or interpolation."""
    num_blocks = temporal_mask.shape[0]
    if num_blocks == 1:
        return np.repeat(temporal_mask, num_frames, axis=0)
    if num_frames % num_blocks:
        raise ValueError(
            f"Sequence has {num_frames} frames but mask has {num_blocks} time blocks; "
            "animate one model window so each block maps to a whole number of frames."
        )
    return np.repeat(temporal_mask, num_frames // num_blocks, axis=0)


def _animation_mask_schedule(
    target_mask: Optional[np.ndarray],
    masked_joints: Optional[Sequence[int]],
    num_frames: int,
    num_joints: int,
    frame_group: Optional[int],
):
    """Preserve sample, time-block and joint axes before expanding to frames."""
    supplied = np.asarray(target_mask) if target_mask is not None else None
    if supplied is not None and supplied.ndim == 3:
        if supplied.shape[0] == 0:
            raise ValueError("target_mask must contain at least one sample")
        samples = list(supplied)
    else:
        samples = [supplied]
    masks = np.stack([
        _temporal_target_mask(num_joints, sample, masked_joints) for sample in samples
    ])
    num_blocks = masks.shape[1]
    if frame_group is not None:
        if not isinstance(frame_group, (int, np.integer)) or frame_group < 1:
            raise ValueError("frame_group must be a positive integer")
        if target_mask is None or num_frames != num_blocks * frame_group:
            raise ValueError(
                "Pass a temporal target_mask for exactly one model window: "
                f"{num_blocks} blocks x frame_group={frame_group} != {num_frames} frames"
            )
    frame_masks = np.concatenate([
        _target_mask_by_frame(sample, num_frames) for sample in masks
    ])
    return masks, frame_masks


def show_video(path: str | Path, width: int = 420, max_bytes: int = 4_000_000):
    """Return an IPython Video object containing a browser-compatible MP4.

    Videos are embedded so playback works locally and in Colab without relying
    on a notebook file server. Clips larger than ``max_bytes`` are transcoded to
    a short H.264 preview. H.264 plus yuv420p is deliberate: OpenCV's ``mp4v``
    writer and source clips encoded as HEVC produce MP4 files that Chrome-based
    notebook renderers commonly cannot decode.
    """
    from IPython.display import Video

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Video does not exist: {path}")

    data = path.read_bytes()
    if len(data) > max_bytes or not _is_browser_compatible(path):
        preview = _make_preview(path, max_bytes)
        if preview is None:
            raise RuntimeError(
                "Could not create a browser-compatible video preview. "
                "Install ffmpeg and make sure its H.264 encoder is available."
            )
        data = preview

    return Video(
        data=data,
        embed=True,
        mimetype="video/mp4",
        width=width,
        html_attributes='controls playsinline preload="metadata"',
    )


def _is_browser_compatible(path: Path) -> bool:
    """Check for the MP4 video format supported by notebook webviews."""
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        # If the source cannot be inspected, transcode it to a known-good
        # format. The project ships a portable ffmpeg fallback for this path.
        return False

    command = [
        ffprobe,
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,pix_fmt",
        "-of", "default=noprint_wrappers=1",
        str(path),
    ]
    try:
        result = subprocess.run(
            command, check=True, capture_output=True, text=True, timeout=15
        )
    except (OSError, subprocess.SubprocessError):
        return False

    properties = dict(
        line.split("=", 1) for line in result.stdout.splitlines() if "=" in line
    )
    return (
        properties.get("codec_name") == "h264"
        and properties.get("pix_fmt") in {"yuv420p", "yuvj420p"}
    )


def _ffmpeg_executable() -> str | None:
    """Locate system ffmpeg, then the project-installed portable binary."""
    executable = shutil.which("ffmpeg")
    if executable is not None:
        return executable
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError):
        return None


def _make_preview(path: Path, max_bytes: int):
    """Transcode a short H.264/yuv420p preview and return its bytes, or ``None``."""
    ffmpeg = _ffmpeg_executable()
    if ffmpeg is None:
        return None

    with tempfile.TemporaryDirectory() as tmp_dir:
        output = Path(tmp_dir) / "preview.mp4"
        command = [
            ffmpeg,
            "-v", "error",
            "-y",
            "-i", str(path),
            "-t", "6",
            "-vf", "scale=640:-2:force_original_aspect_ratio=decrease",
            "-an",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "28",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(output),
        ]
        try:
            subprocess.run(command, check=True, capture_output=True, timeout=90)
        except (OSError, subprocess.SubprocessError):
            return None

        if not output.is_file() or output.stat().st_size == 0:
            return None
        data = output.read_bytes()
        return data if len(data) <= max_bytes * 2 else None


def skeleton_animation(
    seq: np.ndarray,
    out_path: str | Path,
    masked_joints: Optional[Sequence[int]] = None,
    fps: int = 15,
    invert_y: bool = True,
    title: str = "",
    target_mask: Optional[np.ndarray] = None,
    *,
    frame_group: Optional[int] = None,
):
    """Save a GIF of a BlazePose-33 skeleton and its mask over time.

    ``seq`` has shape ``(frames, 33, channels)``. A flat ``target_mask`` uses
    token order ``time_block * joints + joint``; a 2-D mask has shape
    ``(time_blocks, 33)``. A 3-D mask ``(samples, time_blocks, 33)`` replays this
    same window once per supplied sample, showing stochastic variation separately
    from motion within a window. Samples are supplied by the caller; GIF loops
    always replay the same saved draws. Pass ``frame_group=cfg.frame_group`` to
    enforce exact alignment with model tokens. Red X markers are hidden targets;
    blue circles are visible context. Individual blocks may have no context.
    ``masked_joints`` is for static masks. ``invert_y`` flips image coordinates.
    """
    import matplotlib.pyplot as plt
    from matplotlib import animation

    seq = np.asarray(seq)
    if seq.ndim != 3 or seq.shape[0] == 0 or seq.shape[1] != 33 or seq.shape[2] < 2:
        raise ValueError("seq must have shape (non-empty frames, 33, at least 2 channels)")
    T, num_joints = seq.shape[:2]
    temporal_masks, frame_masks = _animation_mask_schedule(
        target_mask, masked_joints, T, num_joints, frame_group
    )
    num_samples, num_blocks = temporal_masks.shape[:2]

    xy = seq[:, :, :2]
    xmin, xmax = np.nanmin(xy[:, :, 0]), np.nanmax(xy[:, :, 0])
    ymin, ymax = np.nanmin(xy[:, :, 1]), np.nanmax(xy[:, :, 1])
    pad_x = 0.1 * (xmax - xmin + 1e-6)
    pad_y = 0.1 * (ymax - ymin + 1e-6)

    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    fig.subplots_adjust(left=0.06, right=0.62, bottom=0.22, top=0.86)
    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(ymin - pad_y, ymax + pad_y)
    if invert_y:
        ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=12, y=0.96)

    lines = [ax.plot([], [], "-", color="#9aa5b1", lw=2)[0] for _ in CONNECTIONS]
    ctx_scatter = ax.scatter(
        [], [], s=28, marker="o",
        color="#2563eb", edgecolors="white", linewidths=0.6,
        label="visible context", zorder=3,
    )
    msk_scatter = ax.scatter(
        [], [], s=56, marker="X",
        color="#dc2626", edgecolors="white", linewidths=0.8,
        label="masked target", zorder=4,
    )
    ax.legend(
        loc="upper left", bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0, frameon=False, fontsize=8,
    )
    # Explicit status stays readable even when a hand overlaps a hip in 2-D.
    sample_label = fig.text(0.65, 0.68, "", fontsize=11, weight="bold")
    hip_labels = [
        fig.text(0.65, 0.60 - i * 0.07, "", fontsize=10)
        for i in range(2)
    ]
    if num_samples > 1:
        replay_note = ("Same motion window;\nnew mask at each sample.\n\n"
                       "GIF repeats these saved\nsamples, then loops.")
    elif target_mask is not None or masked_joints is not None:
        replay_note = "One saved mask.\nGIF loops do not\nresample it."
    else:
        replay_note = "Unmasked sequence.\nAll joints are context."
    fig.text(0.65, 0.36, replay_note, fontsize=9, color="#4a5568", va="top")
    block_label = fig.text(
        0.06, 0.13, "", fontsize=10, color="#4a5568"
    )
    context_note = fig.text(0.06, 0.08, "", fontsize=9, color="#4a5568")

    def update(t):
        sample, local_frame = divmod(t, T)
        pts = xy[local_frame]

        masked_now = frame_masks[t]
        ctx_idx = np.flatnonzero(~masked_now)
        msk_idx = np.flatnonzero(masked_now)
        block = local_frame * num_blocks // T
        sample_label.set_text(f"Mask sample {sample + 1}/{num_samples}")
        for label, j, side in zip(hip_labels, (23, 24), ("L", "R")):
            label.set_text(f"{side} hip ({j}): {'MASKED' if masked_now[j] else 'VISIBLE'}")
            label.set_color("#dc2626" if masked_now[j] else "#2563eb")
        block_label.set_text(
            f"Time block {block + 1}/{num_blocks} | frame {local_frame + 1}/{T} | "
            f"{ctx_idx.size} visible, {msk_idx.size} masked"
        )
        context_note.set_text(
            "No context in this block; context is available at other times."
            if ctx_idx.size == 0 else "Blue circle = visible context; red X = masked target."
        )

        for line, (a, b) in zip(lines, CONNECTIONS):
            line.set_data([pts[a, 0], pts[b, 0]], [pts[a, 1], pts[b, 1]])
        ctx_scatter.set_offsets(
            pts[ctx_idx] if ctx_idx.size else np.empty((0, 2))
        )
        msk_scatter.set_offsets(
            pts[msk_idx] if msk_idx.size else np.empty((0, 2))
        )
        return lines + [ctx_scatter, msk_scatter, block_label, sample_label,
                        context_note, *hip_labels]

    anim = animation.FuncAnimation(
        # Figure-level status labels are omitted from GIF export if marked as
        # animated by blitting. Redraw the full figure to preserve those labels.
        fig, update, frames=T * num_samples, interval=1000 / fps, blit=False
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        anim.save(out_path, writer="pillow", fps=fps)
    finally:
        plt.close(fig)
    return out_path


def mask_timeline(target_masks: np.ndarray, out_path: str | Path, frame_group: int = 4):
    """Show every joint/time mask cell at once, independently of GIF playback.

    Accepts ``(samples, time_blocks, 33)``. Each cell is the actual mask bit, not
    an average or the union over time. Right-side counts expose joints that have
    little context even if they pass an "ever visible" check.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    masks = np.asarray(target_masks, dtype=bool)
    if masks.ndim != 3 or masks.shape[0] == 0 or masks.shape[1] == 0 or masks.shape[2] != 33:
        raise ValueError("target_masks must have shape (samples, time_blocks, 33)")
    n_samples, n_blocks, n_joints = masks.shape
    grid = masks.reshape(n_samples * n_blocks, n_joints).T
    visible_counts = (~grid).sum(1)
    labels = [str(j) for j in range(n_joints)]
    labels[23], labels[24] = "23  LEFT HIP", "24  RIGHT HIP"
    fig, ax = plt.subplots(figsize=(12, 8.5))
    fig.subplots_adjust(left=0.16, right=0.88, bottom=0.12, top=0.84)
    ax.imshow(grid, cmap=ListedColormap(["#2563eb", "#dc2626"]),
              vmin=0, vmax=1, aspect="auto", interpolation="nearest")
    ax.set_yticks(range(n_joints), labels, fontsize=9)
    for j in [23, 24]:
        ax.get_yticklabels()[j].set_weight("bold")
    ax.set_xticks(np.arange(n_samples) * n_blocks + (n_blocks - 1) / 2,
                  [f"Sample {i + 1}" for i in range(n_samples)])
    for boundary in range(n_blocks, n_samples * n_blocks, n_blocks):
        ax.axvline(boundary - 0.5, color="white", lw=2)
    for j in [22.5, 24.5]:
        ax.axhline(j, color="#111827", lw=1.5)
    for j, count in enumerate(visible_counts):
        ax.text(n_samples * n_blocks + 0.4, j, f"{count}/{n_samples * n_blocks}",
                va="center", fontsize=9, clip_on=False,
                weight="bold" if j in [23, 24] else "normal")
    ax.text(1.02, 1.02, "Visible blocks", transform=ax.transAxes, fontsize=9)
    ax.set_xlabel(f"Each colored cell is one {frame_group}-frame time block; sample boundaries are white.")
    ax.set_ylabel("BlazePose joint index")
    fig.suptitle("All sampled masks at once: no playback or overlap ambiguity", fontsize=14, y=0.97)
    fig.legend(handles=[Patch(color="#2563eb", label="Visible context"),
                        Patch(color="#dc2626", label="Masked target")],
               loc="upper center", bbox_to_anchor=(0.5, 0.935), ncol=2, frameon=False)
    fig.text(0.16, 0.035,
             f"Left hip visible in {visible_counts[23]} blocks; right hip in {visible_counts[24]}. "
             "GIF loops replay these same bits.", fontsize=10)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(out_path, dpi=130)
    finally:
        plt.close(fig)
    return out_path


def scatter_2d(coords: np.ndarray, labels: Sequence[str], ax=None, title: str = ""):
    """A 2D scatter (for t-SNE or UMAP output) coloured one hue per class."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    for cls in ["normal", "ms", "pd"]:
        m = np.asarray(labels) == cls
        if m.any():
            ax.scatter(coords[m, 0], coords[m, 1], s=40, alpha=0.8,
                       label=cls, color=CLASS_COLORS[cls], edgecolors="white", linewidths=0.5)
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.set_xticks([])
    ax.set_yticks([])
    return ax
