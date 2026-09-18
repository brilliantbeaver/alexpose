#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'src'))
from gavd6_sjepa.research_directions.synthetic_training_v2.preparation import prepare_source
from gavd6_sjepa.research_directions.synthetic_training_v2.contracts import atomic_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True);p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    config=json.loads(Path(a.config).read_text())
    if a.dry_run:print(json.dumps(dict(config=config,output=a.output,requires='allocated Torch2.6.0+cu124, audited assets and explicit budget'),indent=2));return
    # A unique attempt cost file includes loading, rendering, extraction and failures.
    path=Path(a.output).with_suffix('.attempt-cost.json')
    if path.exists():raise FileExistsError('A new attempt output ID is required, including retries')
    start=time.perf_counter();status='failed'
    try:
        from gavd6_sjepa.research_directions.synthetic_training_v2.config import RunConfig
        from gavd6_sjepa.research_directions.synthetic_training_v2.runtime import budgeted_gpu_stage
        scope=RunConfig.load(config['scope_config'])
        with budgeted_gpu_stage(scope,'prepare'):
            result=prepare_source(config,a.output,ROOT);status='completed';print(result)
    finally:atomic_json(path,dict(stage='prepare_including_initialization',gpu_seconds=time.perf_counter()-start,status=status,allocated_job=__import__('os').environ.get('SLURM_JOB_ID')))
if __name__=='__main__':main()
