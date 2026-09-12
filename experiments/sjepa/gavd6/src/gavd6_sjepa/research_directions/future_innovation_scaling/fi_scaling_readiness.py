"""Read-only input checks for expanded processing; never infer pose eligibility."""
from pathlib import Path

import pandas as pd

from ..future_innovation.fi_contracts import read_json, sha256_file
from ..future_innovation.fi_source_inventory import discover_sources


def development_media(roster, video_manifest, video_root, *, include_confirmation=False, resolution_manifest=None):
    """Resolve every development recording before candidate exclusions can hide it.

    Reservation uses metadata only and is independent of media availability.
    Confirmation files need not be present and are never decoded by this check.
    Discovery proves file presence/identity, not readable frames or pose quality.
    """
    videos = pd.read_csv(video_manifest, dtype={'video_id': str})
    if (not videos.video_id.is_unique or videos.video_id.isna().any()
            or not roster.video_id.is_unique or roster.video_id.isna().any()
            or not set(roster.role) <= {'development', 'confirmation'}
            or not set(roster.video_id) <= set(videos.video_id)):
        raise ValueError('Invalid recording inventory or source reservation')
    wanted = roster if include_confirmation else roster.loc[roster.role == 'development']
    selected = videos[videos.video_id.isin(wanted.video_id)]
    if selected.empty:
        raise ValueError('No development recordings in source reservation')
    found = discover_sources(selected, resolution_manifest or video_manifest, video_root)
    rows = []
    for source, item in sorted(found.items()):
        path = item['path']
        error = item['error']
        if path is not None and path.stat().st_size == 0:
            error = f'Empty source video: {path}'
        rows.append(dict(video_id=source, available=path is not None and not error,
                         video_path=str(path or ''), method=item['method'], reason=error))
    return pd.DataFrame(rows)


def require_development_media(inventory):
    failed = inventory.loc[~inventory.available]
    if len(failed):
        examples = '\n'.join(failed.reason.head(10))
        raise FileNotFoundError(
            f'{len(failed)} of {len(inventory)} development recordings are unavailable or ambiguous. '
            'Restore the full-source videos or set FI_VIDEO_ROOT to their directory; '
            'the source reservation will not be reduced to match available files.\n' + examples)


def original_inputs(parent, annotations, pose_model, checkpoint):
    """Check relocated inputs against the parent bindings, without loading models."""
    parent = Path(parent)
    original = read_json(parent / 'config/run-contract.json')
    by_name = {Path(p).name: h for p, h in original['inputs_sha256'].items()}
    required = {Path(p).name for p in original['input_paths']['annotations']}
    if len(annotations) != len(required) or {Path(p).name for p in annotations} != required:
        raise ValueError('Supply the same five original annotation partitions')
    for path in [*annotations, pose_model]:
        if sha256_file(path) != by_name.get(Path(path).name):
            raise ValueError(f'Original input/model checksum mismatch: {path}')
    teacher = read_json(parent / 'config/teacher-contract.json')
    if sha256_file(checkpoint) != teacher['checkpoint_sha256']:
        raise ValueError('Teacher checkpoint changed')
    return teacher
