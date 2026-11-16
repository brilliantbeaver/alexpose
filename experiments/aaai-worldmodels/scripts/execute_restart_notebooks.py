"""Execute each restart notebook in an independent project Python kernel."""
from pathlib import Path
import argparse,hashlib,json,time,sys
import nbformat
from nbclient import NotebookClient
from jupyter_client.kernelspec import KernelSpecManager
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--pattern',default='1[0-5]_restart_*.ipynb');a=p.parse_args()
    kernel=ROOT/'artifacts/restart_kernel';kernel.mkdir(exist_ok=True)
    (kernel/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],'display_name':'MedWM restart verified interpreter','language':'python'}))
    manager=KernelSpecManager();manager.install_kernel_spec(str(kernel),kernel_name='medwm-restart-verified',user=True,replace=True)
    output=ROOT/'artifacts/executed-restart';output.mkdir(exist_ok=True)
    report=dict(status='running',interpreter=sys.executable,inputs={},notebooks=[])
    for base in ['restart-full10hz-v1','restart-dynamics-v1']:
        for name in (['windows.npz','windows.csv','preparation.json'] if 'full' in base else ['results.json','identity.json']):
            path=ROOT/'artifacts/runs'/base/name;report['inputs'][str(path.relative_to(ROOT))]=sha(path)
    record=ROOT/'docs/restart/notebook_execution.json'
    for path in sorted(ROOT.glob(a.pattern)):
        nb=nbformat.read(path,as_version=4);start=time.monotonic()
        entry=dict(name=path.name,source_sha256=sha(path),status='running')
        try:
            NotebookClient(nb,timeout=900,kernel_name='medwm-restart-verified',resources={'metadata':{'path':str(ROOT)}}).execute()
            entry.update(status='complete',code_cells=sum(c.cell_type=='code' for c in nb.cells))
        except Exception as exc:
            entry.update(status='failed',error_type=type(exc).__name__,error=str(exc)[-3000:])
        nbformat.write(nb,output/path.name)
        entry.update(seconds=time.monotonic()-start,output_sha256=sha(output/path.name))
        report['notebooks'].append(entry);record.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(entry),flush=True)
        if entry['status']=='failed':raise RuntimeError('Notebook execution failed; see recorded output')
    report['status']='complete';report['total_code_cells']=sum(n['code_cells'] for n in report['notebooks']);record.write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
