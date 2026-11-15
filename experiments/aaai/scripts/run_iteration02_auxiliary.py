"""Run fixed-budget published-pose and supervised representation comparators."""
from pathlib import Path
import argparse,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'experiments/aaai'))
from aaai_gait.iteration_data import load_iteration_data
from aaai_gait.iteration_pose_baseline import run_pose_baseline
from aaai_gait.iteration_jepa_probe import run_probe_fold
from aaai_gait.study_runner import dump_json
from aaai_gait.data import file_sha256

parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
parser.add_argument('--cohort',type=Path,default=ROOT/'experiments/aaai/artifacts/iteration02/preparation/cohort_v2')
args=parser.parse_args();data=load_iteration_data(args.cohort)
out=args.run/'auxiliary';out.mkdir(parents=True,exist_ok=True)
manifest=dict(core_run=args.run.name,cohort_sha256=data.provenance['cohort_sha256'],
    steps=dict(stgnf=128,jepa=512),device='cpu',seed=42,
    purpose='required external-method adaptation and separate supervised probes',
    encoder_pretraining_mask_ablation=False,
    code={name:file_sha256(ROOT/'experiments/aaai/aaai_gait'/name) for name in ['iteration_pose_baseline.py','iteration_jepa_probe.py']})
dump_json(out/'protocol.json',manifest)
records=[]
for fold in range(4):
    for name,fn in [('stgnf_adapted',run_pose_baseline),('jepa_probes',run_probe_fold)]:
        folder=out/f'fold_{fold}'/name;folder.mkdir(parents=True,exist_ok=True)
        started=time.perf_counter()
        try:
            _,status=fn(data,fold,folder)
            record=dict(fold=fold,method=name,status=status['status'],wall_seconds=time.perf_counter()-started)
        except Exception as error:
            record=dict(fold=fold,method=name,status='failed_execution',error=repr(error),traceback=traceback.format_exc(),wall_seconds=time.perf_counter()-started)
            dump_json(folder/'failure.json',record)
        records.append(record);dump_json(out/'execution.json',records);print(record,flush=True)
