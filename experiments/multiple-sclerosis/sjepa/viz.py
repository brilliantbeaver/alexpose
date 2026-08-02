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
):
    """Save a gif of the skeleton over time and return the path.

    ``seq`` is (T, 33, 3). If ``masked_joints`` is given, those joints are drawn in
    a highlight colour so the anatomical mask is visible. ``invert_y`` flips the y
    axis so the figure looks upright (image coordinates grow downward).
    """
    import matplotlib.pyplot as plt
    from matplotlib import animation

    seq = np.asarray(seq)
    T = seq.shape[0]
    masked = set(masked_joints or [])

    xy = seq[:, :, :2]
    xmin, xmax = np.nanmin(xy[:, :, 0]), np.nanmax(xy[:, :, 0])
    ymin, ymax = np.nanmin(xy[:, :, 1]), np.nanmax(xy[:, :, 1])
    pad_x = 0.1 * (xmax - xmin + 1e-6)
    pad_y = 0.1 * (ymax - ymin + 1e-6)

    fig, ax = plt.subplots(figsize=(3.5, 4.5))
    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(ymin - pad_y, ymax + pad_y)
    if invert_y:
        ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=11)

    lines = [ax.plot([], [], "-", color="#9aa5b1", lw=2)[0] for _ in CONNECTIONS]
    ctx_scatter = ax.scatter([], [], s=18, color="#4a5568", zorder=3)
    msk_scatter = ax.scatter([], [], s=34, color="#e53e3e", zorder=4)

    ctx_idx = [j for j in range(33) if j not in masked]
    msk_idx = [j for j in range(33) if j in masked]

    def update(t):
        pts = xy[t]
        for line, (a, b) in zip(lines, CONNECTIONS):
            line.set_data([pts[a, 0], pts[b, 0]], [pts[a, 1], pts[b, 1]])
        ctx_scatter.set_offsets(pts[ctx_idx] if ctx_idx else np.empty((0, 2)))
        msk_scatter.set_offsets(pts[msk_idx] if msk_idx else np.empty((0, 2)))
        return lines + [ctx_scatter, msk_scatter]

    anim = animation.FuncAnimation(fig, update, frames=T, interval=1000 / fps, blit=True)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out_path, writer="pillow", fps=fps)
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
