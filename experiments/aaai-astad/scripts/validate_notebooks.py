"""Validate and independently execute every notebook in the uv environment.

Write executed copies only under artifacts/notebook_checks, leaving authored
notebooks clean. Force synthetic mode for bounded software checks.
"""
from pathlib import Path
import os
import sys
import json
import time
import nbformat
from nbclient import NotebookClient
BASE=Path(__file__).resolve().parents[1]
import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--real',action='store_true')
parser.add_argument('--only',default=None,help='Notebook filename prefix, such as 05')
args=parser.parse_args()
real=args.real
os.environ['AAAI_DEMO']='0' if real else '1'
os.environ['AAAI_EPOCHS']='3' if real else '1'
os.environ['AAAI_RUN_SPE_ABLATION']='0'
# The temporary kernelspec ensures each fresh kernel uses this interpreter.
import tempfile
from jupyter_client.kernelspec import KernelSpecManager
with tempfile.TemporaryDirectory(prefix='aaai-nb-',dir='/tmp') as temporary:
    spec=Path(temporary)/'aaai-validation';spec.mkdir()
    (spec/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],'display_name':'AAAI validation','language':'python'}))
    manager=KernelSpecManager(kernel_dirs=[str(Path(temporary))])
    out=BASE/'artifacts'/('real_notebook_checks' if real else 'notebook_checks');out.mkdir(parents=True,exist_ok=True)
    report=[]
    for path in sorted(BASE.glob('*.ipynb')):
        if args.only and not path.name.startswith(args.only): continue
        start=time.monotonic();book=nbformat.read(path,as_version=4);nbformat.validate(book)
        client=NotebookClient(book,timeout=180,kernel_name='aaai-validation',resources={'metadata':{'path':str(BASE)}})
        # Explicit manager prevents a system Python kernel from shadowing uv.
        from jupyter_client import AsyncKernelManager
        client.km=AsyncKernelManager(kernel_name='aaai-validation',kernel_spec_manager=manager,
                                    transport='ipc',ip=str(Path(temporary)/'ipc'))
        try:
            client.execute()
        finally:
            if client.km is not None and client.km.has_kernel:
                import asyncio
                asyncio.run(client.km.shutdown_kernel(now=True))
        nbformat.write(book,out/path.name)
        row={'notebook':path.name,'cells':len(book.cells),'seconds':round(time.monotonic()-start,2),'status':'passed','mode':'GAVD' if real else 'synthetic'}
        report.append(row);print(json.dumps(row),flush=True)
    (out/(f'execution_report_{args.only}.json' if args.only else 'execution_report.json')).write_text(json.dumps(report,indent=2))
