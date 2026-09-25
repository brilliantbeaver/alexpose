#!/usr/bin/env python3
"""Execute the tutorial notebooks against a fresh CPU software fixture."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--through', type=int, choices=range(7), default=6,
                        help='Run the numbered notebooks up to this stage; default also runs all six experiment tutorials (F uses mathematical examples without a child run).')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    work, output = args.work.resolve(), args.output.resolve()
    if (work / 'config.json').exists():
        config = json.loads((work / 'config.json').read_text())
        if not config.get('fixture'):
            parser.error('This execution checker accepts software fixtures only.')
    env = dict(os.environ, GF_ROOT=str(root), GF_WORK=str(work), GF_PYTHON=sys.executable,
               PYTHONPATH=str(root / 'src'), PYTHONNOUSERSITE='1', MPLBACKEND='Agg',
               MPLCONFIGDIR=str(output / 'cache/matplotlib'), XDG_CACHE_HOME=str(output / 'cache'),
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               PATH=str(Path(sys.executable).parent) + os.pathsep + os.environ.get('PATH', ''))
    (output / 'cache/matplotlib').mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, '-m', 'gavd6_sjepa.research_directions.gait_fidelity',
                    'init', '--fixture', '--work', str(work), '--root', str(root)],
                   cwd=root, env=env, check=True, stdout=subprocess.DEVNULL)
    import nbformat
    from nbclient import NotebookClient
    folder = root / 'notebooks/gait_fidelity'
    notebooks = [p for p in sorted(folder.glob('*.ipynb')) if int(p.name[:2]) <= args.through]
    if args.through == 6:
        notebooks += sorted((folder / 'experiments').glob('*.ipynb'))
    records = []
    for path in notebooks:
        print(f'Executing {path.relative_to(folder)}', flush=True)
        started = time.monotonic()
        notebook = nbformat.read(path, as_version=4)
        NotebookClient(notebook, timeout=900, kernel_name='python3').execute(cwd=str(root), env=env)
        target = output / path.relative_to(folder)
        target.parent.mkdir(parents=True, exist_ok=True)
        nbformat.write(notebook, target)
        code_cells = [cell for cell in notebook.cells if cell.cell_type == 'code']
        outputs = [item for cell in code_cells for item in cell.outputs]
        records.append(dict(notebook=str(path.relative_to(folder)), status='passed',
                            seconds=round(time.monotonic() - started, 3),
                            source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            code_cells=len(code_cells),
                            image_outputs=sum('image/png' in item.get('data', {}) for item in outputs),
                            error_outputs=sum(item.output_type == 'error' for item in outputs)))
    report = dict(status='TUTORIAL_EXECUTION_PASSED', scope='CPU software fixture; no HAIC validation',
                  completed_utc=datetime.now(timezone.utc).isoformat(), work=str(work), notebooks=records)
    (output / 'execution.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
