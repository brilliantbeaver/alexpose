#!/usr/bin/env python3
"""Fresh kernels, explicit interpreter, executed copies under unique run identity."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from datetime import datetime,timezone
import nbformat
from nbclient import NotebookClient
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'src'))
from gavd6_sjepa.research_directions.synthetic_training_v2.config import RunConfig

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--python',required=True);a=p.parse_args()
    cfg=RunConfig.load(a.config)
    if cfg.mode!='fixture':raise ValueError('Notebook executor is fixture-only; submit explicit source stages through Slurm')
    # Preserve the venv symlink path: resolving it selects the base interpreter
    # and silently discards that environment's Torch/dependency selection.
    executable=Path(a.python).absolute()
    if not executable.is_file():raise FileNotFoundError(executable)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');out=cfg.root/'notebook_runs'/stamp;out.mkdir(parents=True)
    from jupyter_client.kernelspec import KernelSpecManager
    with tempfile.TemporaryDirectory(prefix='stv2-kernel-') as tmp:
        kernel=Path(tmp)/'stv2';kernel.mkdir()
        (kernel/'kernel.json').write_text(json.dumps(dict(argv=[str(executable),'-m','ipykernel_launcher','-f','{connection_file}'],display_name='STV2 explicit Python',language='python',env={'STV2_CONFIG':str(Path(a.config).resolve()),'PYTHONPATH':str(ROOT/'src'),'MPLCONFIGDIR':str(cfg.root/'mpl-cache')})))
        manager=KernelSpecManager(kernel_dirs=[tmp],ensure_native_kernel=False)
        results=[]
        for path in sorted((ROOT/'notebooks/synthetic_training_v2').glob('*.ipynb')):
            notebook=nbformat.read(path,as_version=4)
            client=NotebookClient(notebook,timeout=600,kernel_name='stv2',resources={'metadata':{'path':str(ROOT)}},kernel_manager_class=__import__('jupyter_client').KernelManager)
            # Passing a dedicated manager binds each fresh kernel to the exact interpreter.
            from jupyter_client import KernelManager
            client.km=KernelManager(kernel_name='stv2',kernel_spec_manager=manager)
            status='failed'
            try:
                client.execute();status='completed'
            finally:
                nbformat.write(notebook,out/path.name)
                results.append({'notebook':path.name,'status':status,'interpreter':str(executable)})
                (out/'execution.json').write_text(json.dumps(results,indent=2)+'\n')
            print(path.name,status,flush=True)
    print(out)
if __name__=='__main__':main()
