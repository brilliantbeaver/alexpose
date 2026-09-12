"""Execute the source-curve inspection notebook in a fresh, explicitly bound kernel.

Notebook success means its inspection cells executed. It is never a numerical
verification or a scientific completion verdict. Only notebook copies and logs
are written; production artifacts and canonical notebooks remain unchanged.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import sys
import tempfile
import time
from uuid import uuid4

from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
from nbclient import NotebookClient
import nbformat

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
from gavd6_sjepa.research_directions.future_innovation.fi_contracts import stage_lock

NOTEBOOKS = {'23': '23_source_learning_curves.ipynb'}
BUILDER = ROOT / 'scripts/research_directions/future_innovation/build_source_learning_curve_notebook.py'
INPUTS = (
    'config/study.json', 'reports/cohort-audit.json', 'manifests/learning-plan.json',
    'data/cohort-complete.json', 'data/cache-complete.json', 'data/audit-complete.json',
    'manifests/plan-complete.json', 'reports/complete.json',
    'reports/learning-curve.json', 'reports/learning-curve.svg',
    'data/manifests/development-windows.csv', 'data/logs/development-media.csv',
    'config/availability-contract.json', 'config/media-availability.csv',
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_text(path, text):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(text)
    temporary.replace(path)


def input_snapshot(root):
    """Hash exactly the artifacts used for display; do not load models or cache arrays."""
    result = {}
    dynamic = [str(p.relative_to(root)) for pattern in ('logs/stages/*.json', 'models/*/complete.json')
               for p in sorted(root.glob(pattern))]
    for relative in (*INPUTS, *dynamic):
        path = root / relative
        result[relative] = (dict(sha256=digest(path), size=path.stat().st_size,
                                 mtime_ns=path.stat().st_mtime_ns) if path.is_file() else None)
    return result


def load_source(number):
    spec = importlib.util.spec_from_file_location('source_curve_notebook_builder', BUILDER)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    path = builder.DESTINATION
    if path.name != NOTEBOOKS[number]:
        raise ValueError('Registered notebook does not match the canonical builder destination')
    source = nbformat.read(path, as_version=4)
    if [(c.cell_type, c.source) for c in source.cells] != [(c.cell_type, c.source) for c in builder.render().cells]:
        raise ValueError(f'Regenerate {path.name} with {BUILDER.relative_to(ROOT)} before execution')
    for cell in source.cells:
        if cell.cell_type == 'code':
            cell.outputs = []
            cell.execution_count = None
            cell.metadata.pop('execution', None)
    return path, source


def check_inputs(run_root):
    root = (ROOT / Path(run_root).expanduser()).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f'Study directory does not exist: {root}')
    config = root / 'config/study.json'
    if not config.exists():
        config = root / 'config/run-contract.json'
    if config.exists() and json.loads(config.read_text()).get('protocol') != 'source-learning-curve-v1':
        raise ValueError('FI_RUN_ROOT must be a source learning-curve study, not a historical gate')
    for number in NOTEBOOKS:
        load_source(number)
    return root


def relocate_links(notebook, output, source_directory):
    for cell in notebook.cells:
        if cell.cell_type != 'markdown':
            continue
        def replace(match):
            label, target = match.groups()
            if re.match(r'^[A-Za-z][A-Za-z0-9+.-]*:', target) or target.startswith('#'):
                return match.group(0)
            path, separator, anchor = target.partition('#')
            relative = Path(os.path.relpath((source_directory / path).resolve(), output)).as_posix()
            return f'[{label}]({relative}{separator}{anchor})'
        cell.source = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', replace, cell.source)


def execute_notebooks(run_root, *, timeout=600, output_dir=None):
    if timeout <= 0:
        raise ValueError('--timeout must be positive (seconds per code cell)')
    root = check_inputs(run_root)
    base = (root / 'notebook_runs').resolve()
    if not base.is_relative_to(root) or not (root / 'logs').resolve().is_relative_to(root):
        raise ValueError('Notebook output and log directories must stay inside FI_RUN_ROOT')
    label = ('haic-' + os.environ['SLURM_JOB_ID'] if os.environ.get('SLURM_JOB_ID') else 'manual')
    label += '-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '-' + uuid4().hex[:8]
    output = Path(output_dir).expanduser().resolve() if output_dir else base / label
    if output == base or not output.is_relative_to(base):
        raise ValueError('Notebook output must be a new directory beneath FI_RUN_ROOT/notebook_runs')
    lock_id = hashlib.sha256(str(output).encode()).hexdigest()[:24]
    with stage_lock(root, f'source-notebooks-{lock_id}'):
        # Even a finished batch is immutable: rerunning creates a new batch.
        output.mkdir(parents=True, exist_ok=False)
        logs = root / 'logs/notebooks' / output.name
        logs.mkdir(parents=True, exist_ok=False)
        for number, name in NOTEBOOKS.items():
            source_path, notebook = load_source(number)
            relocate_links(notebook, output, source_path.parent)
            destination = output / name
            before = input_snapshot(root)
            record = dict(
                notebook=name, mode='inspect', run_root=str(root), output=str(destination),
                source=str(source_path.relative_to(ROOT)),
                source_sha256=digest(source_path), builder_sha256=digest(BUILDER),
                executor_sha256=digest(Path(__file__)), python=sys.version, executable=sys.executable,
                platform=platform.platform(), kernel_executable=sys.executable,
                started_utc=datetime.now(timezone.utc).isoformat(), status='running', execution_completed=False,
                numerical_verification_performed=False, artifact_integrity_verified=False,
                evidence_scope='Saved values and file presence only; file hashes bind this inspection, not scientific validity.',
                input_snapshot_before=before,
                slurm={k: os.environ[k] for k in ('SLURM_JOB_ID','SLURM_ARRAY_JOB_ID','SLURM_ARRAY_TASK_ID','SLURM_RESTART_COUNT') if k in os.environ},
            )
            def save():
                notebook.metadata['fi_execution'] = dict(record)
                atomic_text(destination, nbformat.writes(notebook))
                atomic_text(logs / f'{number}-execution.json', json.dumps(record, indent=2) + '\n')

            def cell_start(cell, cell_index, **kwargs):
                record['active_cell'] = cell_index
                save()
                if cell.cell_type == 'code':
                    print(f'{name}: cell {cell_index + 1}/{len(notebook.cells)}', flush=True)

            def cell_finished(cell, cell_index, **kwargs):
                notebook.cells[cell_index] = cell
                record['last_finished_cell'] = cell_index
                save()

            print(f'Executed notebook: {destination}', flush=True)
            save()
            started = time.perf_counter()
            try:
                # Match the existing notebook runner: explicit interpreter,
                # temporary kernel/font caches, and atomic per-cell checkpoints.
                with tempfile.TemporaryDirectory(prefix='source-curve-kernel-') as temporary:
                    kernels = Path(temporary)
                    spec = kernels / 'source-curve'
                    spec.mkdir()
                    (spec / 'kernel.json').write_text(json.dumps(dict(
                        argv=[sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
                        display_name='Source learning curve', language='python')))
                    environment = {**os.environ, 'GAVD6_ROOT': str(ROOT), 'FI_RUN_ROOT': str(root),
                                   'FI_TUTORIAL_MODE': 'inspect', 'MPLCONFIGDIR': str(kernels / 'matplotlib'),
                                   'IPYTHONDIR': str(kernels / 'ipython'), 'PYTHONUNBUFFERED': '1'}
                    for key in ('FI_NOTEBOOK_FOLD', 'FI_NOTEBOOK_ATTEMPT', 'FI_NOTEBOOK_ARRAY'):
                        environment.pop(key, None)
                    manager = KernelManager(kernel_name='source-curve',
                                            kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernels)]))
                    kernel_client = None
                    try:
                        # Start and handshake before nbclient registers its exit
                        # handlers. A denied port or broken kernel must not leave
                        # an exit hook waiting forever on a kernel that never ran.
                        manager.start_kernel(env=environment, cwd=str(ROOT))
                        kernel_client = manager.client()
                        kernel_client.start_channels()
                        kernel_client.wait_for_ready(timeout=120)
                        client = NotebookClient(notebook, km=manager, timeout=timeout, startup_timeout=120,
                                                allow_errors=False, resources={'metadata': {'path': str(ROOT)}},
                                                on_cell_start=cell_start, on_cell_executed=cell_finished,
                                                on_cell_error=cell_finished)
                        client.kc = kernel_client
                        client.execute(env=environment, cleanup_kc=True)
                    finally:
                        try:
                            if manager.has_kernel:
                                manager.shutdown_kernel(now=True)
                        finally:
                            manager.cleanup_resources()
                            if kernel_client is not None:
                                kernel_client.stop_channels()
                record['execution_completed'] = True
                counts = [cell.execution_count for cell in notebook.cells if cell.cell_type == 'code']
                if any(count is None for count in counts):
                    record['execution_completed'] = False
                    raise RuntimeError('Kernel returned without executing every code cell')
                record['executed_code_cells'] = len(counts)
                record['input_snapshot_after'] = input_snapshot(root)
                record['input_snapshot_unchanged'] = before == record['input_snapshot_after']
                if not record['input_snapshot_unchanged']:
                    raise ValueError('Displayed inputs changed during notebook execution; rerun after the pipeline settles')
                if digest(source_path) != record['source_sha256'] or digest(BUILDER) != record['builder_sha256']:
                    raise ValueError('Notebook source or generator changed during execution; rerun with stable code')
                record['status'] = 'passed'
            except BaseException as error:
                record.update(status='failed', error=f'{type(error).__name__}: {str(error).splitlines()[-1:]}')
                raise
            finally:
                record.update(finished_utc=datetime.now(timezone.utc).isoformat(),
                              wall_seconds_including_kernel=round(time.perf_counter() - started, 3))
                save()
                print(f'{record["status"].upper()}: {destination}', flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, default=os.environ.get('FI_RUN_ROOT'))
    parser.add_argument('--timeout', type=int, default=600, help='Seconds per code cell (default 600)')
    parser.add_argument('--output-dir', type=Path, help='New batch directory beneath the study notebook_runs directory')
    parser.add_argument('--check', action='store_true', help='Validate paths and canonical sources without executing or writing')
    args = parser.parse_args()
    if args.run_root is None:
        parser.error('Set FI_RUN_ROOT or pass --run-root')
    if args.check:
        root = check_inputs(args.run_root)
        print(f'Notebook inspection: {root}; sources: {", ".join(NOTEBOOKS.values())}')
    else:
        execute_notebooks(args.run_root, timeout=args.timeout, output_dir=args.output_dir)


if __name__ == '__main__':
    main()
