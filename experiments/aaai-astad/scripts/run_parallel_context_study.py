"""Fit distinct outer folds in separate CPU processes, then summarize once."""
import argparse,json,subprocess,sys,os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from aaai_gait.settings import EXPERIMENT
from aaai_gait.study_runner import summarize_study


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--demo',action='store_true')
    parser.add_argument('--pretrain-updates',type=int,default=512)
    parser.add_argument('--predictor-updates',type=int,default=512)
    parser.add_argument('--ridge-draws',type=int,default=256)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--workers',type=int,default=4,choices=[1,2,3,4])
    args=parser.parse_args()
    logs=EXPERIMENT/'artifacts/context_study/logs';logs.mkdir(parents=True,exist_ok=True)
    stamp=__import__('datetime').datetime.now().strftime('%Y%m%dT%H%M%S')
    def one(fold):
        path=logs/f'{stamp}_fold_{fold}.log'
        cmd=[sys.executable,str(EXPERIMENT/'scripts/run_context_study.py'),'--pretrain-updates',str(args.pretrain_updates),
             '--predictor-updates',str(args.predictor_updates),'--ridge-draws',str(args.ridge_draws),
             '--seed',str(args.seed),'--folds',str(fold),'--defer-summary']
        if args.demo:cmd.append('--demo')
        env=dict(os.environ,VECLIB_MAXIMUM_THREADS='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2')
        with path.open('w') as stream:result=subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT,env=env)
        if result.returncode:raise RuntimeError(f'Fold{fold} failed. Inspect {path}')
        lines=path.read_text().splitlines();dirs=[line.split('RUN_DIRECTORY ',1)[1] for line in lines if line.startswith('RUN_DIRECTORY ')]
        if len(dirs)!=1:raise RuntimeError(f'Expected one run directory in {path}')
        print('COMPLETED_FOLD',fold,dirs[0],flush=True)
        return Path(dirs[0])
    with ThreadPoolExecutor(max_workers=args.workers) as executor:directories=list(executor.map(one,range(4)))
    if len(set(directories))!=1:raise RuntimeError('Corpus/code changed during parallel fitting. Distinct runs cannot be pooled.')
    summary=summarize_study(directories[0],synthetic=args.demo)
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
