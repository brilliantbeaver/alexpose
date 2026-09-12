"""CLI for the separately versioned repaired Experiment 0 source learning curve."""
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'src'))
from gavd6_sjepa.research_directions.future_innovation_scaling.fi_scaling_cli import main

if __name__=='__main__':
    main()
