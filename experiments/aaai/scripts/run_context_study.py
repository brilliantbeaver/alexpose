"""Run the new source-held-out study independently of tutorial notebook order."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from aaai_gait.context_prediction import ContextConfig
from aaai_gait.study_runner import run_study

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--demo',action='store_true',help='Artificial software check, never research evidence')
    parser.add_argument('--pretrain-updates',type=int,default=512)
    parser.add_argument('--predictor-updates',type=int,default=512)
    parser.add_argument('--masks',type=int,default=16)
    parser.add_argument('--ridge-draws',type=int,default=256)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--folds',type=int,nargs='+',default=[0,1,2,3])
    parser.add_argument('--without-untrained',action='store_true',help='Incomplete control set; development runs only')
    parser.add_argument('--defer-summary',action='store_true',help='Fit assigned folds only; summarize once after all parallel fold processes finish')
    parser.add_argument('--device',default='cpu',choices=['cpu','mps','cuda'])
    args=parser.parse_args()
    config=ContextConfig(pretrain_updates=args.pretrain_updates,predictor_updates=args.predictor_updates,device=args.device)
    out=run_study(demo=args.demo,seed=args.seed,config=config,mask_repeats=args.masks,
        ridge_draws=args.ridge_draws,include_untrained=not args.without_untrained,folds=tuple(args.folds),
        finalize=not args.defer_summary)
    print(out)

if __name__=='__main__':main()
