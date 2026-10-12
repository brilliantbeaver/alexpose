"""Run a fresh CPU parent/child fixture and independently reconstruct its tables.

This performs software validation only. It never reads source-study arrays or
submits jobs. Invoke with the repository .venv interpreter and one CPU thread.
"""
from pathlib import Path
import argparse
import json
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'src'))

from gavd6_sjepa.research_directions.gait_fidelity.common import atomic_json, read_json, sha256, utc_now
from gavd6_sjepa.research_directions.gait_fidelity.config import initialize, load_config
from gavd6_sjepa.research_directions.gait_fidelity.followup import initialize_followup, verify_followup
from gavd6_sjepa.research_directions.gait_fidelity.scheduler import run, status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(1)
    root = Path(tempfile.mkdtemp(prefix='gait-readout-control-verified-'))
    parent, child = root/'parent', root/'child'
    started = time.monotonic()
    print(f'CPU fixture: {root}', flush=True)
    parent_cfg = initialize(parent, fixture=True, experiment_set='core')
    run(parent_cfg, local=True, max_jobs=8)
    before = {str(p):sha256(p) for p in parent.rglob('*') if p.is_file()}
    print('Parent fixture complete', flush=True)
    cfg = initialize_followup(child, parent_work=parent, include_base_readouts=True,
                              deadline_utc='2026-09-25T15:00:00Z')
    assert cfg == load_config(child)
    run(cfg, local=True, max_jobs=8)
    print('All 27 child phases complete; reconstructing evaluation', flush=True)
    verification = verify_followup(cfg)
    assert before == {str(p):sha256(p) for p in parent.rglob('*') if p.is_file()}
    plan, ledger = read_json(child/'plan.json'), read_json(child/'ledger.json')
    assert plan['counts']['optimization_phases'] == 27
    pairs = []
    for phase in plan['phases']:
        if phase['phase'] != 'readout' or phase['recipe']['readout_or_training_objective'] != 'base':
            continue
        base = ledger['completed'][phase['phase_id']]['result']
        paired_id = phase['phase_id'].replace('-base-seed-', '-paired_change-seed-')
        paired = ledger['completed'][paired_id]['result']
        assert base['frozen_encoder_sha256'] == paired['frozen_encoder_sha256']
        assert base['frozen_encoder_unchanged'] and paired['frozen_encoder_unchanged']
        pairs.append(dict(base_phase=phase['phase_id'], paired_change_phase=paired_id,
                          shared_encoder_sha256=base['frozen_encoder_sha256']))
    summary = read_json(child/'evaluation/response-summary.json')
    assert summary['trained_models'] == 18 and len(pairs) == 9
    assert len(cfg['followup']['parent_binding']['baseline_phases']) == 30
    result = dict(schema='gait-fidelity-readout-control-software-validation-v1', created_utc=utc_now(),
                  evidence_status='software-validation-only', source_results=False, cuda_verified=False,
                  parent=str(parent), child=str(child), elapsed_seconds=time.monotonic()-started,
                  parent_unchanged=True, parent_files=len(before), shared_encoder_readout_pairs=pairs,
                  status=status(cfg), verification=verification, plan_identity=plan['identity'],
                  primary_candidate=summary['primary_candidate'], primary_comparator=summary['primary_comparator'])
    atomic_json(root/'validation.json', result)
    atomic_json(args.output, result)
    print(json.dumps(dict(validation=str(args.output), root=str(root), elapsed_seconds=result['elapsed_seconds'],
                          completed_models=18, parent_unchanged=True,
                          metrics=verification['metric_reconstruction']['status']), indent=2))


if __name__ == '__main__':
    main()
