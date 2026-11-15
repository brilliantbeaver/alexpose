"""Run supplementary controls against a completed real, frozen source study."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from aaai_gait.source_controls import run_source_controls

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    args=parser.parse_args()
    print(run_source_controls(args.run_dir))
