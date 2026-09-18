#!/usr/bin/env python3
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'src'))
from gavd6_sjepa.research_directions.synthetic_training_v2.runtime import require_haic_runtime
from gavd6_sjepa.research_directions.synthetic_training_v2.contracts import atomic_json
from gavd6_sjepa.research_directions.synthetic_training_v2.config import RunConfig
from gavd6_sjepa.research_directions.synthetic_training_v2.runtime import budgeted_gpu_stage
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--config',required=True);a=p.parse_args()
if Path(a.output).exists():raise FileExistsError('Choose a unique preflight artifact')
with budgeted_gpu_stage(RunConfig.load(a.config),'preflight'):
    report=require_haic_runtime();atomic_json(a.output,report);print(json.dumps(report,indent=2))
