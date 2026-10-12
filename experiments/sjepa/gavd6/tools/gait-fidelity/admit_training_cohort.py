#!/usr/bin/env python3
"""Record and apply a common post-screening training-eligibility amendment.

Standalone recovery for a source run stopped before the combined bundle was
published. Source shards, frozen code/configuration, and evaluation membership
stay unchanged. All training methods use the same admitted population.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def summarize(records):
    people = defaultdict(set)
    families = defaultdict(set)
    for row in records:
        people[row['split']].add(row['canonical_person_id'])
        families[row['split']].add(row['source_family_id'])
    return {split: dict(people=len(people[split]), windows=len(families[split]))
            for split in sorted(people)}


def inspect(work):
    from gavd6_sjepa.research_directions.gait_fidelity.common import read_json, sha256
    from gavd6_sjepa.research_directions.gait_fidelity.config import load_config
    from gavd6_sjepa.research_directions.gait_fidelity.scheduler import _verify_frozen, preparation_phase_ids

    work = Path(work).resolve()
    cfg = load_config(work)
    if cfg['fixture'] or cfg['data'].get('source_selection') != 'full_manifest':
        raise ValueError('Recovery requires a full-manifest source run')
    _verify_frozen(cfg)
    ledger = read_json(work / 'ledger.json')
    if any(not a['phase_id'].startswith('prepare-shard-') for a in ledger['attempts']):
        raise ValueError('Admission must be settled before profiling or model training')
    if any(a['status'] not in {'complete', 'failed'} for a in ledger['attempts']):
        raise ValueError('An unresolved worker remains; do not amend an active run')
    if 'prepare' in ledger['completed']:
        raise ValueError('Combined preparation is already admitted; no changes permitted')
    parts, records = [], []
    for phase in preparation_phase_ids(cfg):
        entry = ledger['completed'][phase]
        receipt = Path(entry['receipt'])
        if sha256(receipt) != entry['sha256']:
            raise ValueError(f'Completion receipt changed: {phase}')
        parent = Path(entry['result']['bundle']).resolve()
        manifest = parent / 'manifest.json'
        receipt_data = read_json(receipt)
        if receipt_data['phase_id'] != phase or receipt_data['config_sha256'] != sha256(work/'config.json'):
            raise ValueError(f'Completion identity differs from the saved run: {phase}')
        if not parent.is_relative_to(work / 'attempts'):
            raise ValueError('Source bundle must belong to this run')
        if receipt_data['artifacts'].get(str(manifest)) != sha256(manifest):
            raise ValueError(f'Source manifest differs from verified completion: {phase}')
        meta = read_json(manifest)
        if any(r['split'] not in {'train', 'development'} for r in meta['records']):
            raise ValueError('This admission cannot open confirmation references')
        parts.append(dict(phase=phase, path=parent, meta=meta, entry=entry))
        records.extend(meta['records'])
    by_person = defaultdict(set)
    for row in records:
        if row['split'] == 'train':
            by_person[row['canonical_person_id']].add(row['source_family_id'])
    excluded = {person: sorted(values) for person, values in sorted(by_person.items()) if len(values) < 2}
    if not excluded:
        raise ValueError('No singleton training person was found; this recovery does not apply')
    kept = [r for r in records if not (r['split'] == 'train' and r['canonical_person_id'] in excluded)]
    before, after = summarize(records), summarize(kept)
    if not after.get('train', {}).get('windows') or after.get('development') != before.get('development'):
        raise ValueError('Recovery must retain training and leave development unchanged')
    roots = {p['meta']['provenance']['identity'] for p in parts}
    if len(roots) != 1:
        raise ValueError('Preparation shards have different identities')
    proposal = dict(schema='gf-training-admission-v1',
        rule='Exclude every training person with fewer than two accepted source windows from every method',
        rationale='Same-person shuffled-reference controls require a different source window',
        decision_basis='Reference geometry admission and source identities, before any model fitting',
        excluded_training_people=excluded, before=before, after=after,
        removed_records=len(records)-len(kept),
        seeds=cfg['seeds'],
        unchanged=['development records and arrays', 'source shards', 'config.json', 'plan.json', 'frozen.json'],
        config_sha256=sha256(work/'config.json'), plan_sha256=sha256(work/'plan.json'),
        frozen_sha256=sha256(work/'frozen.json'),
        tool_sha256=sha256(Path(__file__)),
        source_shards=[dict(path=str(p['path']), shard_index=p['meta']['provenance']['shard_index'],
                           manifest_sha256=sha256(p['path']/'manifest.json')) for p in parts])
    return cfg, parts, proposal


def require_finished_coordinator(work):
    receipt = json.loads((work/'control/coordinator.json').read_text())
    if receipt.get('state') == 'submitting' or not str(receipt.get('job_id', '')).isdigit():
        raise ValueError('Coordinator submission is unresolved')
    job = str(receipt['job_id'])
    queue = subprocess.run(['squeue', '-h', '-j', job, '-o', '%i'], capture_output=True, text=True)
    if queue.returncode == 0 and queue.stdout.strip():
        raise ValueError('Coordinator is queued or running; wait before admission')
    account = subprocess.run(['sacct', '-X', '-n', '-P', '-j', job, '--format=JobIDRaw,State'],
                             capture_output=True, text=True, check=True)
    states = [line.split('|')[1].split()[0].rstrip('+') for line in account.stdout.splitlines()
              if line.split('|')[0] == job and len(line.split('|')) > 1]
    from gavd6_sjepa.research_directions.gait_fidelity.scheduler import TERMINAL
    if len(states) != 1 or states[0] not in TERMINAL:
        raise ValueError('Cannot confirm that the coordinator has finished')


def apply_admission(work, parts, proposal):
    """Validate frozen parents and publish an explicitly derived combined bundle."""
    from gavd6_sjepa.research_directions.gait_fidelity.common import atomic_json, digest, read_json, sha256
    from gavd6_sjepa.research_directions.gait_fidelity.data import load_dataset, select_rows, save_dataset, merge_disk_datasets
    from gavd6_sjepa.research_directions.gait_fidelity.scheduler import verify_completed
    from gavd6_sjepa.research_directions.gait_fidelity.training import shuffled_reference_indices

    folder = work / 'admissions' / 'training-min-two-windows'
    output = work / 'data/bundle'
    identity = digest(proposal)
    if output.exists():
        meta = read_json(output/'manifest.json')
        if meta['provenance'].get('training_admission', {}).get('identity') != identity:
            raise ValueError('A different combined bundle already exists; nothing overwritten')
        load_dataset(output)
        return dict(status='ADMITTED_BUNDLE_ALREADY_PRESENT', bundle=str(output), identity=identity)
    plan = folder / 'decision.json'
    if plan.exists() and read_json(plan) != proposal:
        raise ValueError('Existing admission decision differs; nothing overwritten')
    for part in parts:
        verify_completed(part['entry'])
    folder.mkdir(parents=True, exist_ok=True)
    if not plan.exists():
        atomic_json(plan, proposal)
    retained_tool = folder/'admit_training_cohort.py'
    if retained_tool.exists():
        if sha256(retained_tool) != proposal['tool_sha256']:
            raise ValueError('Retained recovery tool changed')
    else:
        shutil.copy2(__file__, retained_tool)
    excluded = set(proposal['excluded_training_people'])
    paths, records, derived_receipts, assets = [], [], [], {}
    for part in parts:
        bundle = load_dataset(part['path'])
        indices = [i for i, r in enumerate(bundle.records)
                   if not (r['split'] == 'train' and r['canonical_person_id'] in excluded)]
        if not indices:
            raise ValueError('This recovery does not support empty derived shards')
        parent_receipt = next(r for r in proposal['source_shards'] if r['path'] == str(part['path']))
        path = part['path']
        if len(indices) != len(bundle.records):
            selected = select_rows(bundle, indices)
            selected.provenance = copy.deepcopy(bundle.provenance)
            selected.provenance.update(training_admission_identity=identity,
                source_families=sorted({r['source_family_id'] for r in selected.records}),
                parent_shard=parent_receipt)
            path = folder / 'derived-shards' / part['phase'] / 'bundle'
            if path.exists():
                existing = load_dataset(path)
                if existing.records != selected.records or existing.provenance != selected.provenance:
                    raise ValueError('Existing derived shard differs from admission decision')
                for group in ('inputs', 'targets'):
                    import numpy as np
                    for key, values in getattr(selected, group).items():
                        for start in range(0, len(values), 256):
                            if not np.array_equal(values[start:start+256], getattr(existing, group)[key][start:start+256], equal_nan=True):
                                raise ValueError('Existing derived arrays differ from parent selection')
            else:
                save_dataset(selected, path, storage='npy')
            bundle = selected
        paths.append(path)
        records.extend(bundle.records)
        for name, value in bundle.provenance.get('assets', {}).items():
            if assets.setdefault(name, value) != value:
                raise ValueError('Source asset identities differ across shards')
        derived_receipts.append(dict(path=str(path), manifest_sha256=sha256(path/'manifest.json')))
    if summarize(records) != proposal['after']:
        raise ValueError('Admitted population differs from the recorded decision')
    # Use the actual control construction, which requires distinct windows
    # within every person/condition stratum, not just a person-level count.
    training_records = [r for r in records if r['split'] == 'train']
    for seed in proposal['seeds']:
        shuffled_reference_indices(training_records, seed)
    provenance = copy.deepcopy(parts[0]['meta']['provenance'])
    original_people = set(provenance['frozen_people'])
    admitted_people = {r['canonical_person_id'] for r in records}
    provenance.update(shard_index='merged', assets=assets,
        source_families=sorted({r['source_family_id'] for r in records}),
        # Original immutable shards remain the parent lineage expected by the
        # frozen coordinator. Direct merge inputs are recorded separately.
        shard_receipts=proposal['source_shards'], admitted_shard_receipts=derived_receipts,
        coverage=dict(planned_people=sorted(original_people), admitted_people=sorted(admitted_people),
            people_without_admitted_families=sorted(original_people-admitted_people),
            training_admission_exclusions=proposal['excluded_training_people'],
            exclusions=[r for p in parts for r in p['meta']['provenance'].get('exclusions', [])]),
        training_admission=dict(proposal, identity=identity, decision=str(plan), decision_sha256=sha256(plan),
            tool=str(retained_tool)))
    result = merge_disk_datasets(paths, output, provenance=provenance)
    receipt = dict(status='ADMITTED_BUNDLE_READY', bundle=str(result), identity=identity,
                   manifest_sha256=sha256(result/'manifest.json'), before=proposal['before'], after=proposal['after'])
    atomic_json(folder/'complete.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', required=True, type=Path)
    parser.add_argument('--expected-singleton', action='append', default=[])
    parser.add_argument('--apply', action='store_true', help='Publish the common admitted dataset on CPU')
    args = parser.parse_args()
    work = args.work.expanduser().resolve()
    cfg = json.loads((work/'config.json').read_text())
    sys.path.insert(0, str(Path(cfg['code_root'])/'src'))
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ.setdefault(name, '4')
    from gavd6_sjepa.research_directions.gait_fidelity.common import locked
    if not args.apply:
        _, _, proposal = inspect(work)
        print(json.dumps(dict(status='ADMISSION_PREVIEW', **proposal), indent=2))
        return
    with locked(work/'control/submit.lock', nonblocking=True), locked(work/'locks/controller.lock', nonblocking=True):
        require_finished_coordinator(work)
        _, parts, proposal = inspect(work)
        if set(args.expected_singleton) != set(proposal['excluded_training_people']):
            raise ValueError('Pass the exact previewed people with --expected-singleton; no dataset changed')
        print(json.dumps(proposal, indent=2), flush=True)
        print(json.dumps(apply_admission(work, parts, proposal), indent=2), flush=True)


if __name__ == '__main__':
    main()
