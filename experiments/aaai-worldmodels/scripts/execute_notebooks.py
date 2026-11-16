"""Execute all tutorials, or named subsets, in the declared project kernel.

Examples, from experiments/aaai-worldmodels:
    uv run python scripts/execute_notebooks.py
    uv run python scripts/execute_notebooks.py 06_capstone_rf_vs_jepa.ipynb 07_ablation_and_paper_evidence.ipynb

The notebooks themselves gate video embedding, pose extraction, and input-video
inference behind explicit WM_* settings. This runner does not enable them.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time

import nbformat
from nbclient import NotebookClient
from jupyter_client.kernelspec import KernelSpecManager

PROJECT = Path(__file__).resolve().parents[1]


def execute(names=None, kernel_name='medwm-gait', timeout=240, data_mode=None):
    mode=data_mode or os.getenv('WM_DATA_MODE','synthetic')
    if mode not in {'synthetic','real'}: raise ValueError('Unknown data mode')
    os.environ['WM_DATA_MODE']=mode
    run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    os.environ['WM_NOTEBOOK_RUN_ID']=run_id
    kernel = KernelSpecManager().get_kernel_spec(kernel_name)
    kernel_executable = str(Path(kernel.argv[0]).absolute())
    current_executable = str(Path(sys.executable).absolute())
    # Do not resolve venv symlinks: different environments can share one base
    # interpreter while exposing different packages.
    if kernel_executable != current_executable:
        raise RuntimeError(
            f'Kernel executable differs from this environment: {kernel_executable}. '
            f'Register it with {sys.executable} -m ipykernel install --user '
            f'--name {kernel_name} --display-name "Med-WM gait (Python 3.12)"')
    selected = [PROJECT/Path(name).name for name in names] if names else sorted(PROJECT.glob('*.ipynb'))
    if not selected or any(not p.is_file() or p.suffix!='.ipynb' for p in selected):
        raise FileNotFoundError('Select existing tutorial .ipynb files from the project root')
    output = PROJECT/('artifacts/executed-real' if mode=='real' else 'artifacts/executed')
    output.mkdir(parents=True, exist_ok=True)
    evidence_path = PROJECT/('docs/notebook_execution_real.json' if mode=='real' else 'docs/notebook_execution.json')
    previous = json.loads(evidence_path.read_text()) if evidence_path.exists() else {}
    old = previous.get('notebooks', [])
    records = {row['notebook']:row for row in old}
    timestamp = datetime.now(timezone.utc).isoformat()
    environment = dict(python=sys.version, platform=platform.platform(),
                       runner_executable=current_executable, kernel_name=kernel_name,
                       kernel_executable=kernel_executable,
                       versions={name:importlib.metadata.version(name) for name in ('nbclient','nbformat','ipykernel','torch','numpy','pandas','scikit-learn')})
    report = dict(schema_version=2, mode=mode, notebook_run_id=run_id, test_results_released=os.getenv('WM_RELEASE_TEST_RESULTS','0')=='1', last_execution_started_at=timestamp, environment=environment,
                  source_generator_sha256=hashlib.sha256((PROJECT/'scripts/build_notebooks.py').read_bytes()).hexdigest(),
                  execution_runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  selected=[p.name for p in selected],
                  scope='Each notebook executes independently in a fresh kernel. Real mode loads immutable GAVD poses, embeds a bounded real excerpt, and executes actual MediaPipe/model calls. Synthetic mode remains separate teaching evidence.')
    if mode=='real':
        from worldmodels.notebook_support import prepared_path, comparison_path
        report['real_inputs']={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in (prepared_path()/'windows.npz',prepared_path()/'windows.csv',comparison_path()/'results.json')}
    failures = []
    for path in selected:
        print(f'Executing {path.name}', flush=True)
        started = time.monotonic()
        book = nbformat.read(path, as_version=4)
        source_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        record = dict(notebook=path.name, source_sha256=source_sha256, started_at=datetime.now(timezone.utc).isoformat(),
                      kernel_executable=kernel_executable, evidence_type='real GAVD notebook execution' if mode=='real' else 'synthetic teaching notebook execution', mode=mode, notebook_run_id=run_id)
        try:
            NotebookClient(book, timeout=timeout, kernel_name=kernel_name,
                           resources={'metadata':{'path':str(PROJECT)}}).execute()
            record['status']='passed'
        except Exception as exc:
            record.update(status='failed', error_type=type(exc).__name__, error=str(exc))
            failures.append(path.name)
        finally:
            record['seconds']=time.monotonic()-started
            executed_path=output/path.name
            nbformat.write(book, executed_path)
            record['executed_notebook']=str(executed_path.relative_to(PROJECT))
            record['executed_sha256']=hashlib.sha256(executed_path.read_bytes()).hexdigest()
            record['code_cells']=sum(cell.cell_type=='code' for cell in book.cells)
            record['executed_code_cells']=sum(cell.cell_type=='code' and cell.get('execution_count') is not None for cell in book.cells)
            if mode=='real':
                lesson_evidence=PROJECT/'artifacts/runs/notebook-real'/run_id/path.stem/'evidence.json'
                record['lesson_evidence']=str(lesson_evidence.relative_to(PROJECT))
                if record['status']=='passed' and not lesson_evidence.exists():
                    record['status']='failed_missing_real_evidence'
                    failures.append(path.name)
            records[path.name]=record
            # Stale rows stay visible with explicit status instead of being
            # silently counted as successful checks of changed notebook sources.
            for old_name,old_record in records.items():
                source=PROJECT/old_name
                if not source.exists() or hashlib.sha256(source.read_bytes()).hexdigest()!=old_record.get('source_sha256'):
                    old_record['status']='stale_source_changed'
            report['notebooks']=[records[name] for name in sorted(records)]
            report['all_current_notebooks_passed']=(len(records)==len(list(PROJECT.glob('*.ipynb'))) and all(r['status']=='passed' for r in records.values()))
            evidence=json.dumps(report,indent=2)+'\n'
            (output/'execution.json').write_text(evidence)
            evidence_path.write_text(evidence)
        print(f'{record["status"]}: {path.name} ({record["seconds"]:.1f}s)',flush=True)
        if record['status']!='passed':
            break  # A failed or interrupted lesson must not trigger later work.
    if failures:
        raise RuntimeError('Notebook execution failed: '+', '.join(failures))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('notebooks',nargs='*',help='Optional notebook basenames; default runs all eight')
    parser.add_argument('--kernel',default='medwm-gait')
    parser.add_argument('--timeout',type=int,default=240)
    parser.add_argument('--data-mode',choices=['synthetic','real'],default=None)
    args=parser.parse_args()
    execute(args.notebooks or None,args.kernel,args.timeout,args.data_mode)


if __name__=='__main__':
    main()
