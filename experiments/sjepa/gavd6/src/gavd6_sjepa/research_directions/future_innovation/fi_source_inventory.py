"""Resolve full source videos without assuming one download directory layout."""

from collections import defaultdict
from pathlib import Path

import pandas as pd

from gavd6_sjepa.data_foundations.gavd_video_download_pipeline import VIDEO_SUFFIXES


def discover_sources(videos, manifest_path, youtube_dir, video_roots=()):
    """Prefer declared full-source paths; otherwise index exact IDs in declared roots.

    No fuzzy matches, downloads, clipped-sequence substitution or selection by
    diagnosis. Ambiguous exports require an explicit manifest path. Symlinks to
    the same file are one source. The later exact decoder verifies frame access.
    """
    roots = sorted({Path(p).expanduser().resolve() for p in (youtube_dir, *video_roots)})
    for root in roots:
        if not root.is_dir():
            raise FileNotFoundError(f"Configured video directory is unavailable: {root}")
    extensions = set(VIDEO_SUFFIXES) | {".avi"}
    identifiers = set(videos.video_id.astype(str))
    indexed = defaultdict(set)
    for root in roots:
        # Traverse the directory once, not once per annotation sequence.
        for path in root.rglob("*"):
            if path.stem in identifiers and path.suffix.lower() in extensions and path.is_file():
                indexed[path.stem].add(path.resolve())
    result = {}
    path_columns = ("video_path", "local_path", "cached_path", "source_path")
    for row in videos.to_dict("records"):
        video_id = str(row["video_id"])
        explicit = set()
        for column in path_columns:
            value = row.get(column)
            if value is None or pd.isna(value) or not str(value).strip():
                continue
            path = Path(str(value)).expanduser()
            options = [path] if path.is_absolute() else [Path(manifest_path).parent / path, *(r / path for r in roots)]
            explicit.update(p.resolve() for p in options if p.suffix.lower() in extensions and p.is_file())
        matches = explicit or indexed[video_id]
        # Category caches may hard-link the same source into multiple folders.
        by_file = {}
        for path in sorted(matches):
            stat = path.stat()
            by_file.setdefault((stat.st_dev, stat.st_ino), path)
        matches = set(by_file.values())
        if len(matches) == 1:
            path = next(iter(matches))
            result[video_id] = {"path": path, "error": "", "method": "manifest" if explicit else "exact source ID"}
        elif len(matches) > 1:
            result[video_id] = {"path": None, "error": f"Ambiguous source files for {video_id}; declare one video_path in the video manifest", "method": "ambiguous"}
        else:
            result[video_id] = {"path": None, "error": f"Source video not found in declared storage: {video_id}", "method": "missing"}
    return result
