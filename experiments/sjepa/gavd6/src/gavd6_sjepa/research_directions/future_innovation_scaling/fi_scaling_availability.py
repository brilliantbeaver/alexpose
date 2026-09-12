"""Freeze available manifest recordings without changing reservation assignments."""
from pathlib import Path

import pandas as pd

from ..future_innovation.fi_contracts import read_json, write_json, sha256_file
from .fi_scaling_readiness import development_media

COHORT_POLICY = 'available-development-v1'
PATH_COLUMNS = ('video_path', 'local_path', 'cached_path', 'source_path')
FILES = ('media-availability.csv', 'processing-sequences.csv', 'processing-videos.csv')


def discover_available(roster, video_manifest, video_root, *, resolution_manifest=None):
    """Presence and readable-file checks only; never decode or inspect targets."""
    media = development_media(roster, video_manifest, video_root,
                              include_confirmation=True, resolution_manifest=resolution_manifest)
    inventory = roster.merge(media, on='video_id', validate='one_to_one')
    inventory['file_size'] = 0
    inventory['mtime_ns'] = 0
    for i, row in inventory.iterrows():
        if not row.available:
            inventory.at[i, 'video_path'] = ''
            continue
        path = Path(row.video_path)
        try:
            stat = path.stat()
            with path.open('rb') as stream:
                if not stream.read(1):
                    raise ValueError('Empty source video')
            inventory.at[i, 'file_size'] = stat.st_size
            inventory.at[i, 'mtime_ns'] = stat.st_mtime_ns
        except (OSError, ValueError) as error:
            inventory.at[i, 'available'] = False
            inventory.at[i, 'video_path'] = ''
            inventory.at[i, 'reason'] = f'Unreadable source video: {error}'
    inventory['included'] = inventory.available & inventory.role.eq('development')
    inventory['exclusion_reason'] = inventory.reason.fillna('')
    inventory.loc[inventory.role == 'confirmation', 'exclusion_reason'] = 'reserved confirmation recording'
    if not inventory.included.any():
        raise ValueError('No available development recordings. Check FI_VIDEO_ROOT and the manifest IDs.')
    return inventory.sort_values('video_id').reset_index(drop=True)


def availability_summary(inventory):
    development = inventory.role.eq('development')
    return dict(cohort_policy=COHORT_POLICY, manifest_recordings=len(inventory),
                available_recordings=int(inventory.available.sum()),
                reserved_confirmation_recordings=int((~development).sum()),
                planned_development_recordings=int(development.sum()),
                included_development_recordings=int(inventory.included.sum()),
                unavailable_development_recordings=int((development & ~inventory.available).sum()),
                included_annotated_sequences=int(inventory.loc[inventory.included, 'annotated_sequences'].sum()))


def processing_tables(sequences, videos, inventory):
    selected = inventory.loc[inventory.included].set_index('video_id')
    subset = sequences.loc[sequences.video_id.isin(selected.index)].copy()
    # An absolute frozen choice prevents new duplicate exports or relocated
    # relative-manifest paths from changing discovery on a later attempt.
    sources = videos.loc[videos.video_id.isin(selected.index)].drop(columns=list(PATH_COLUMNS), errors='ignore').copy()
    sources['video_path'] = sources.video_id.map(selected.video_path)
    return subset, sources


def freeze_available(root, video_root, *, resolution_manifest=None):
    root = Path(root)
    config = root / 'config'
    sequences = pd.read_csv(config/'full-sequences.csv', dtype={'video_id': str, 'sequence_id': str})
    videos = pd.read_csv(config/'full-videos.csv', dtype={'video_id': str})
    roster = pd.read_csv(config/'source-reservation.csv', dtype={'video_id': str})
    inventory = discover_available(roster, config/'full-videos.csv', video_root,
                                   resolution_manifest=resolution_manifest)
    selected, sources = processing_tables(sequences, videos, inventory)
    for name, table in zip(FILES, (inventory, selected, sources)):
        table.to_csv(config/name, index=False)
    contract = dict(version=COHORT_POLICY, video_root=str(Path(video_root).expanduser().resolve()),
                    resolution_manifest=str(resolution_manifest or config/'full-videos.csv'),
                    selected_file_identity='absolute path, byte count and mtime_ns; decoded frames are hashed later',
                    artifacts={name: sha256_file(config/name) for name in FILES},
                    summary=availability_summary(inventory))
    write_json(config/'availability-contract.json', contract)
    verify_available(root, check_files=True)
    return contract['summary']


def verify_available(root, *, check_files=False):
    """Validate selection semantics; file checks are needed only before decoding."""
    config = Path(root)/'config'
    contract = read_json(config/'availability-contract.json')
    if contract.get('version') != COHORT_POLICY or set(contract.get('artifacts', {})) != set(FILES):
        raise ValueError('Unsupported availability contract')
    for name, expected in contract['artifacts'].items():
        if sha256_file(config/name) != expected:
            raise ValueError(f'Frozen availability artifact changed: {name}')
    inventory = pd.read_csv(config/'media-availability.csv', keep_default_na=False, dtype={'video_id': str})
    roster = pd.read_csv(config/'source-reservation.csv', dtype={'video_id': str})
    sequences = pd.read_csv(config/'full-sequences.csv', dtype={'video_id': str, 'sequence_id': str})
    videos = pd.read_csv(config/'full-videos.csv', dtype={'video_id': str})
    if (not inventory.video_id.is_unique or set(inventory.video_id) != set(roster.video_id)
            or set(inventory.video_id) != set(videos.video_id)
            or inventory.available.dtype != bool or inventory.included.dtype != bool):
        raise ValueError('Invalid availability identities or masks')
    for column in roster.columns:
        if column == 'video_id':
            continue
        if column not in inventory or not inventory.set_index('video_id')[column].sort_index().equals(
                roster.set_index('video_id')[column].sort_index()):
            raise ValueError(f'Availability changed the source reservation: {column}')
    expected = inventory.available & inventory.role.eq('development')
    if not inventory.included.equals(expected) or not expected.any():
        raise ValueError('Availability selection must contain exactly the available development sources')
    if contract['summary'] != availability_summary(inventory):
        raise ValueError('Availability summary differs from frozen selection')
    selected, sources = processing_tables(sequences, videos, inventory)
    for name, expected_table in zip(FILES[1:], (selected, sources)):
        actual = pd.read_csv(config/name, dtype={'video_id': str, 'sequence_id': str})
        try:
            pd.testing.assert_frame_equal(actual.reset_index(drop=True), expected_table.reset_index(drop=True), check_dtype=False)
        except AssertionError as error:
            raise ValueError(f'Processing manifest differs from available development selection: {name}') from error
    selected_rows = inventory.loc[inventory.included]
    if any(not Path(p).is_absolute() for p in selected_rows.video_path):
        raise ValueError('Selected videos require frozen absolute paths')
    if check_files:
        for row in selected_rows.itertuples(index=False):
            check_selected_file(row)
    return inventory


def check_selected_file(row):
    path = Path(row.video_path)
    if not path.is_file():
        raise FileNotFoundError(f'Selected video disappeared after availability freeze: {path}. Restore it or use a new run.')
    stat = path.stat()
    if stat.st_size != row.file_size or stat.st_mtime_ns != row.mtime_ns:
        raise ValueError(f'Selected video changed after availability freeze: {path}')
    with path.open('rb') as stream:
        if not stream.read(1):
            raise ValueError(f'Selected video became empty: {path}')


def require_included_windows(root, cohort):
    """Keep omitted/confirmation sources out of cache and fitting even after rehashing."""
    inventory = verify_available(root)
    sequences = pd.read_csv(Path(root)/'config/processing-sequences.csv', dtype={'video_id': str, 'sequence_id': str})
    if not set(cohort.video_id) <= set(inventory.loc[inventory.included, 'video_id']):
        raise ValueError('Cohort contains a source excluded by frozen availability')
    mapping = sequences.set_index('sequence_id').video_id
    if not set(cohort.sequence_id) <= set(mapping.index) or not cohort.video_id.reset_index(drop=True).equals(
            mapping.loc[cohort.sequence_id].reset_index(drop=True)):
        raise ValueError('Cohort sequence/source identities differ from processing manifest')
