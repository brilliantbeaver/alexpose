#!/usr/bin/env python3
"""Execute the tutorial notebooks against a fresh CPU software fixture."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--experiment-set', choices=('core', 'full'),
                        help='Optional fixture matrix; core mirrors the latest source sampling. Default retains full.')
    parser.add_argument('--html', action='store_true', help='Also retain readable HTML copies.')
    parser.add_argument('--through', type=int, choices=range(7), default=6,
                        help='Run numbered notebooks through this stage; default also runs A–G. F/G use mathematical examples without a child run. Evidence notebook 07 uses execute_walkthrough.py.')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    folder = root / 'notebooks/gait_fidelity'
    work, output = args.work.resolve(), args.output.resolve()
    if output.is_relative_to(folder):
        parser.error('Choose an output folder outside the source notebooks.')
    if work.is_relative_to(folder):
        parser.error('Choose a fixture work folder outside the source notebooks.')
    if output.is_relative_to(work):
        parser.error('Choose an output folder outside the fixture work folder; init requires an empty new work directory.')
    if (work / 'config.json').exists():
        config = json.loads((work / 'config.json').read_text())
        if not config.get('fixture'):
            parser.error('This execution checker accepts software fixtures only.')
    env = dict(os.environ, GF_ROOT=str(root), GF_WORK=str(work), GF_PYTHON=sys.executable,
               PYTHONPATH=str(root / 'src'), PYTHONNOUSERSITE='1', MPLBACKEND='Agg',
               MPLCONFIGDIR=str(output / 'cache/matplotlib'), XDG_CACHE_HOME=str(output / 'cache'),
               IPYTHONDIR=str(output / 'cache/ipython'), JUPYTER_RUNTIME_DIR=str(output / 'cache/jupyter'),
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               PATH=str(Path(sys.executable).parent) + os.pathsep + os.environ.get('PATH', ''))
    command = [sys.executable, '-m', 'gavd6_sjepa.research_directions.gait_fidelity',
               'init', '--fixture', '--work', str(work), '--root', str(root)]
    if args.experiment_set:
        command += ['--experiment-set', args.experiment_set]
    output.mkdir(parents=True, exist_ok=True)
    records = []
    started_utc = datetime.now(timezone.utc).isoformat()
    experiment_set = config.get('experiment_set', 'full') if (work/'config.json').exists() else (args.experiment_set or 'full')
    stage, current_notebook, source_sha256 = 'initializing_fixture', None, None

    def receipt(status, **details):
        report = dict(status=status, scope='CPU software fixture; no HAIC validation',
                      started_utc=started_utc, work=str(work), python=sys.executable,
                      experiment_set=experiment_set, notebooks=records, **details)
        # Replace any earlier success before new execution begins. A disk error
        # may leave an incomplete receipt, but cannot retain a stale valid pass.
        (output / 'execution.json').write_text(json.dumps(report, indent=2) + '\n')
        return report

    receipt('TUTORIAL_EXECUTION_RUNNING', stage=stage, current_notebook=None)
    try:
        (output / 'cache/matplotlib').mkdir(parents=True, exist_ok=True)
        subprocess.run(command, cwd=root, env=env, check=True, stdout=subprocess.DEVNULL)
        experiment_set = json.loads((work/'config.json').read_text())['experiment_set']
        import nbformat
        from nbclient import NotebookClient
        notebooks = [p for p in sorted(folder.glob('*.ipynb')) if int(p.name[:2]) <= args.through]
        if args.through == 6:
            notebooks += sorted((folder / 'experiments').glob('*.ipynb'))
        stage = 'starting_kernel_environment'
        # Run the invoking interpreter, not a possibly unrelated registered python3 kernel.
        with tempfile.TemporaryDirectory(prefix='gf-tutorial-kernel-') as temporary:
            kernel = Path(temporary) / 'kernels/gf-tutorial'
            kernel.mkdir(parents=True)
            (kernel / 'kernel.json').write_text(json.dumps({
                'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
                'display_name': 'Gait Fidelity tutorial check', 'language': 'python',
            }))
            previous = os.environ.get('JUPYTER_PATH')
            os.environ['JUPYTER_PATH'] = temporary + (os.pathsep + previous if previous else '')
            try:
                for path in notebooks:
                    current_notebook = str(path.relative_to(folder))
                    stage, source_sha256 = 'reading_notebook', None
                    print(f'Executing {current_notebook}', flush=True)
                    started = time.monotonic()
                    source_bytes = path.read_bytes()
                    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
                    notebook = nbformat.reads(source_bytes.decode('utf-8'), as_version=4)
                    stage = 'executing_notebook'
                    receipt('TUTORIAL_EXECUTION_RUNNING', stage=stage,
                            current_notebook=current_notebook, source_sha256=source_sha256)
                    NotebookClient(notebook, timeout=900, kernel_name='gf-tutorial').execute(cwd=str(root), env=env)
                    stage = 'writing_outputs'
                    target = output / path.relative_to(folder)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    # Populated copies live elsewhere; preserve links to source lessons/docs.
                    def relocate_link(match):
                        label, link = match.groups()
                        if ':' in link or link.startswith('#'):
                            return match.group(0)
                        destination = (path.parent / link).resolve()
                        return f'[{label}]({os.path.relpath(destination, target.parent)})'

                    for cell in notebook.cells:
                        if cell.cell_type == 'markdown':
                            cell.source = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', relocate_link, cell.source)
                    nbformat.write(notebook, target)
                    if args.html:
                        from nbconvert import HTMLExporter
                        html, _ = HTMLExporter().from_notebook_node(notebook)
                        target.with_suffix('.html').write_text(html)
                    code_cells = [cell for cell in notebook.cells if cell.cell_type == 'code']
                    outputs = [item for cell in code_cells for item in cell.outputs]
                    records.append(dict(notebook=current_notebook, status='passed',
                                        seconds=round(time.monotonic() - started, 3),
                                        source_sha256=source_sha256, code_cells=len(code_cells),
                                        image_outputs=sum('image/png' in item.get('data', {}) for item in outputs),
                                        error_outputs=sum(item.output_type == 'error' for item in outputs)))
                    current_notebook, source_sha256 = None, None
                    receipt('TUTORIAL_EXECUTION_RUNNING', stage='between_notebooks', current_notebook=None)
            finally:
                if previous is None:
                    os.environ.pop('JUPYTER_PATH', None)
                else:
                    os.environ['JUPYTER_PATH'] = previous
        stage = 'writing_success_receipt'
        report = receipt('TUTORIAL_EXECUTION_PASSED', completed_utc=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        status = 'TUTORIAL_EXECUTION_INTERRUPTED' if isinstance(error, KeyboardInterrupt) else 'TUTORIAL_EXECUTION_FAILED'
        try:
            receipt(status, failed_utc=datetime.now(timezone.utc).isoformat(),
                    failed_stage=stage, failing_notebook=current_notebook, source_sha256=source_sha256,
                    error=dict(type=type(error).__name__, message=str(error)))
        except OSError as receipt_error:
            print(f'Could not save failure receipt: {receipt_error}', file=sys.stderr)
        raise
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
