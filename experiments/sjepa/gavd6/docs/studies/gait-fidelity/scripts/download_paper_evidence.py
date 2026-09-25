"""Preview/download a bounded HAIC evidence packet into local outputs/iclr.

Uses one SSH connection, the existing allowlists, and Python's standard library.
Run in a local interactive terminal for HAIC authentication. No remote writes,
job submissions, large arrays, or recursive directory downloads are performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[4]
RUNS = {
    "walking-core": ("walking-core-01", "walking-core"),
    "jepa-response": ("jepa-response-02", "jepa-response"),
    "readout-repair": ("readout-repair-03", "readout-repair"),
}

REMOTE_CODE = r'''
import hashlib, io, json, sys, tarfile
from datetime import datetime, timezone
from pathlib import Path

p = json.loads(sys.argv[1])
base = Path(p['asset_root']).resolve()
runs = base / 'outputs/gait-fidelity'
entries, selected = [], []

def add(source, destination, allowed_root, required=True):
    source, allowed_root = Path(source), Path(allowed_root).resolve()
    item = dict(source=str(source), destination=destination, required=required)
    entries.append(item)
    try:
        resolved = source.resolve()
        resolved.relative_to(allowed_root)
        if source.is_symlink():
            raise ValueError('symbolic links are not included')
        if not resolved.is_file():
            item['status'] = 'missing'
            return
        size = resolved.stat().st_size
        item['bytes'] = size
        if size > p['max_file_bytes']:
            item['status'] = 'oversized'
            return
        content = resolved.read_bytes()
        if len(content) != size:
            raise ValueError('file changed while reading')
        item.update(status='selected', sha256=hashlib.sha256(content).hexdigest())
        selected.append((resolved, item))
    except (OSError, ValueError) as error:
        item.update(status='rejected', reason=str(error))

def read_small(path):
    path = Path(path)
    if not path.is_file() or path.is_symlink() or path.stat().st_size > p['max_file_bytes']:
        return {}
    return json.loads(path.read_text())

for local_name, spec in p['runs'].items():
    work = runs / spec['source_run']
    for name in spec['files']:
        add(work/name, local_name+'/'+name, work)
    if local_name == 'walking-core':
        name = 'evaluation/coverage-by-source-motion.csv'
        add(work/name, local_name+'/'+name, work, False)
    if local_name == 'readout-repair':
        name = 'development/evaluation/power-sensitivity.json'
        add(work/name, local_name+'/'+name, work, False)
    cfg = read_small(work/'config.json')
    if cfg.get('code_root'):
        add(Path(cfg['code_root'])/'gait-fidelity-release.json',
            local_name+'/provenance/gait-fidelity-release.json', base, False)
    if local_name != 'jepa-response':
        continue
    completed = read_small(work/'ledger.json').get('completed', {})
    profile = completed.get('followup-profile', {}).get('result', {})
    if profile.get('calibration_receipt'):
        add(profile['calibration_receipt'], local_name+'/diagnostics/loss-calibration.json', work)
    diagnostics = completed.get('followup-diagnostics', {})
    if diagnostics.get('receipt'):
        folder = Path(diagnostics['receipt']).parent/'diagnostics'
        add(folder/'diagnostics-summary.json', local_name+'/diagnostics/diagnostics-summary.json', work)
    for export in diagnostics.get('result', {}).get('exports', []):
        source = Path(export['diagnostics'])
        add(source, local_name+'/diagnostics/'+source.parent.name+'/diagnostics.json', work)

destinations = [entry['destination'] for entry in entries]
if len(destinations) != len(set(destinations)):
    raise SystemExit('Duplicate evidence destination; nothing downloaded.')
total = sum(item['bytes'] for _, item in selected)
blocked = total > p['max_total_bytes'] or any(
    item['required'] and item['status'] in ('oversized', 'rejected') for item in entries)
inventory = dict(schema='gait-fidelity-paper-transfer-v1',
    created_utc=datetime.now(timezone.utc).isoformat(), source_asset_root=str(base),
    selected_files=len(selected), selected_bytes=total,
    max_file_bytes=p['max_file_bytes'], max_total_bytes=p['max_total_bytes'],
    missing_required=[e['destination'] for e in entries if e['required'] and e['status']=='missing'],
    transfer_blocked=blocked, files=entries,
    scope='Development results and compact provenance; not raw data or a full rerun package.')
if not p['apply']:
    print(json.dumps(inventory, indent=2))
    raise SystemExit(0)
if blocked or not selected:
    print(json.dumps(inventory, indent=2), file=sys.stderr)
    raise SystemExit('Evidence exceeds limits or has rejected paths; preview and resolve before transfer.')

# Stream only explicit regular files. Nothing is staged or modified on HAIC.
with tarfile.open(fileobj=sys.stdout.buffer, mode='w|gz') as archive:
    for source, item in selected:
        content = source.read_bytes()
        if len(content) != item['bytes'] or hashlib.sha256(content).hexdigest() != item['sha256']:
            raise RuntimeError('Source changed since inventory: '+str(source))
        info = tarfile.TarInfo(item['destination']); info.size=len(content); info.mode=0o600
        archive.addfile(info, io.BytesIO(content))
    content = json.dumps(inventory, indent=2).encode()
    info = tarfile.TarInfo('_inventory.json'); info.size=len(content); info.mode=0o600
    archive.addfile(info, io.BytesIO(content))
'''


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def unpack_verified(archive, staging, max_file_bytes, max_total_bytes):
    """Extract regular bounded members only, then check the full inventory."""
    seen, total = set(), 0
    with tarfile.open(archive, 'r:gz') as source:
        for member in source:
            name = PurePosixPath(member.name)
            if (not member.isfile() or name.is_absolute() or '..' in name.parts
                    or member.name in seen or not name.parts
                    or (name.parts[0] not in RUNS and member.name != '_inventory.json')):
                raise ValueError(f'Unsafe or duplicate archive member: {member.name}')
            if member.size < 0 or member.size > max_file_bytes:
                raise ValueError(f'Oversized archive member: {member.name}')
            total += member.size
            if total > max_total_bytes + max_file_bytes or len(seen) >= 512:
                raise ValueError('Archive exceeds evidence limits')
            seen.add(member.name)
            target = staging.joinpath(*name.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.extractfile(member) as stream, target.open('xb') as output:
                shutil.copyfileobj(stream, output)
    inventory = json.loads((staging/'_inventory.json').read_text())
    expected = {e['destination']: e for e in inventory['files'] if e['status']=='selected'}
    if seen != set(expected) | {'_inventory.json'}:
        raise ValueError('Archive and evidence inventory disagree')
    if sum(e['bytes'] for e in expected.values()) > max_total_bytes:
        raise ValueError('Selected evidence exceeds the total limit')
    for name, entry in expected.items():
        target = staging/name
        if target.stat().st_size != entry['bytes'] or digest(target) != entry['sha256']:
            raise ValueError(f'Evidence checksum mismatch: {name}')
    return inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='tedmui@haic.stanford.edu')
    parser.add_argument('--asset-root', default='/hai/scratch/tedmui/alexpose/experiments/sjepa/gavd6')
    parser.add_argument('--remote-python', default='/hai/scratch/tedmui/envs/synthetic-training-cu124/bin/python')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/iclr')
    parser.add_argument('--max-file-mib', type=int, default=10)
    parser.add_argument('--max-total-mib', type=int, default=50)
    parser.add_argument('--apply', action='store_true', help='Download; default only inventories remote files.')
    args = parser.parse_args()
    if min(args.max_file_mib, args.max_total_mib) <= 0 or args.host.startswith('-'):
        parser.error('Use positive limits and a normal SSH destination')
    lists = ROOT/'docs/studies/gait-fidelity/manuscript'
    runs = {}
    for name, (source, key) in RUNS.items():
        files = (lists/f'paper-evidence-{key}-files.txt').read_text().splitlines()
        if any(not f or PurePosixPath(f).is_absolute() or '..' in PurePosixPath(f).parts for f in files):
            raise ValueError('Evidence allowlist contains an invalid path')
        runs[name] = dict(source_run=source, files=files)
    payload = dict(asset_root=args.asset_root, runs=runs, apply=args.apply,
                   max_file_bytes=args.max_file_mib*1024**2, max_total_bytes=args.max_total_mib*1024**2)
    command = ['ssh', '-o', 'ConnectTimeout=15', args.host,
               shlex.join([args.remote_python, '-', json.dumps(payload)])]
    with tempfile.TemporaryDirectory(prefix='gait-paper-download-') as temporary:
        temporary = Path(temporary); received = temporary/'received'
        with received.open('wb') as output:
            subprocess.run(command, input=REMOTE_CODE.encode(), stdout=output, check=True)
        if not args.apply:
            inventory = json.loads(received.read_text())
            for item in inventory['files']:
                print(f"{item['status']:10} {item.get('bytes',0)/1024:10.1f} KiB  {item['destination']}")
            print(f"\nSelected: {inventory['selected_files']} files, {inventory['selected_bytes']/1024**2:.2f} MiB")
            print(f"Missing required: {len(inventory['missing_required'])}; transfer blocked: {inventory['transfer_blocked']}")
            return 2 if inventory['transfer_blocked'] else 0
        staging = temporary/'verified'; staging.mkdir()
        inventory = unpack_verified(received, staging, payload['max_file_bytes'], payload['max_total_bytes'])
        destination = args.output.expanduser().resolve()
        selected = [e for e in inventory['files'] if e['status']=='selected']
        for entry in selected:
            target = destination/entry['destination']
            if not target.resolve().is_relative_to(destination):
                raise ValueError(f'Destination escapes through a symlink: {target}')
            if target.exists() and (not target.is_file() or digest(target) != entry['sha256']):
                raise FileExistsError(f'Existing file differs; preserve it and choose a fresh --output: {target}')
        for entry in selected:
            target = destination/entry['destination']
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(staging/entry['destination'], target)
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        record = destination/f'transfer-inventory-{stamp}.json'
        record.write_text(json.dumps(inventory, indent=2)+'\n')
        print(f"Downloaded and SHA-256 verified {len(selected)} files ({inventory['selected_bytes']/1024**2:.2f} MiB) into {destination}")
        print(f'Inventory: {record}')
        if inventory['missing_required']:
            print('INCOMPLETE EVIDENCE: these required source files were missing:')
            for name in inventory['missing_required']: print('  '+name)
            return 3
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, subprocess.CalledProcessError, tarfile.TarError) as error:
        print(f'Evidence download failed: {error}', file=sys.stderr)
        raise SystemExit(1)
