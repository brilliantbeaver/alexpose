#!/usr/bin/env python3
"""Explicit study stages; no scheduler side effects."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'src'))
from gavd6_sjepa.research_directions.synthetic_training_v2.config import RunConfig,STAGES
from gavd6_sjepa.research_directions.synthetic_training_v2.workflow import run_stage

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--stage',choices=(*STAGES,'all'),required=True)
    p.add_argument('--dry-run',action='store_true');p.add_argument('--resume',action='store_true');args=p.parse_args();cfg=RunConfig.load(args.config)
    stages=STAGES if args.stage=='all' else (args.stage,)
    if args.dry_run:
        print(json.dumps(dict(configuration=cfg.as_dict(),stages=list(stages),submits_jobs=False),indent=2));return
    for stage in stages:
        result=run_stage(cfg,stage,ROOT,resume=args.resume)
        print(json.dumps(dict(stage=stage,evidence_status=result.get('evidence_status'),status=result.get('status'),output=str(cfg.root)),sort_keys=True),flush=True)
if __name__=='__main__':main()
