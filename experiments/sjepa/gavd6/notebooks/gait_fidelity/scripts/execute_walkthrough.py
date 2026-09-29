#!/usr/bin/env python3
"""Execute notebook 07 against compact downloaded evidence, with no training.

The notebook verifies its transfer inventory before computing results. Populated
notebook, HTML, figures and a small receipt stay outside the tutorial sources.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time

import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
NAME = '07_completed_study_walkthrough.ipynb'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, default=ROOT / 'outputs/iclr')
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'outputs/gait-fidelity/notebook-walkthrough-20260925')
    parser.add_argument('--export-figures', action='store_true')
    args = parser.parse_args()
    source = HERE / NAME
    output, evidence = args.output.resolve(), args.evidence.resolve()
    if output.is_relative_to(HERE) or output.is_relative_to(evidence):
        parser.error('Choose an output folder outside the source notebooks and evidence packet.')
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    stage, source_sha256 = 'reading_source', None

    def write_receipt(status, **details):
        receipt = dict(status=status, mode='downloaded-development-evidence',
                       source_sha256=source_sha256, python=sys.executable, evidence=str(evidence),
                       elapsed_seconds=round(time.monotonic() - started, 2),
                       notebook=str(output / NAME), html=str(output / 'index.html'),
                       training_or_cluster_work=False, **details)
        (output / 'execution.json').write_text(json.dumps(receipt, indent=2) + '\n')
        return receipt

    # Invalidate any earlier pass before a new attempt reads or executes cells.
    write_receipt('RUNNING', stage=stage)
    try:
        source_bytes = source.read_bytes()
        source_sha256 = hashlib.sha256(source_bytes).hexdigest()
        notebook = nbformat.reads(source_bytes.decode('utf-8'), as_version=4)
        # Retain working links after moving the populated copy away from its source.
        def relocate_link(match):
            label, target = match.groups()
            if ':' in target or target.startswith('#'):
                return match.group(0)
            path = (HERE / target).resolve()
            return f'[{label}]({os.path.relpath(path, output)})'

        for cell in notebook.cells:
            if cell.cell_type == 'markdown':
                cell.source = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', relocate_link, cell.source)
        if args.export_figures:
            notebook.cells[-1].source = notebook.cells[-1].source.replace(
                'EXPORT_FIGURES = False', 'EXPORT_FIGURES = True').replace(
                "ROOT / 'outputs/gait-fidelity/notebook-walkthrough-20260925/figures'",
                f'Path({str(output / "figures")!r})')
        env = dict(os.environ, GF_EVIDENCE_ROOT=str(evidence), MPLBACKEND='Agg',
                   MPLCONFIGDIR=str(Path(tempfile.gettempdir()) / 'gf-walkthrough-mpl'),
                   IPYTHONDIR=str(Path(tempfile.gettempdir()) / 'gf-walkthrough-ipython'),
                   JUPYTER_RUNTIME_DIR=str(Path(tempfile.gettempdir()) / 'gf-walkthrough-runtime'))
        stage = 'starting_kernel_environment'
        # Use the invoking interpreter, irrespective of the user's registered python3 kernel.
        with tempfile.TemporaryDirectory(prefix='gf-walkthrough-kernel-') as temporary:
            kernel = Path(temporary) / 'kernels/gf-walkthrough'
            kernel.mkdir(parents=True)
            (kernel / 'kernel.json').write_text(json.dumps({
                'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
                'display_name': 'Gait Fidelity evidence walkthrough', 'language': 'python',
            }))
            # Kernel discovery occurs in this process, while env also goes to the child.
            original_jupyter_path = os.environ.get('JUPYTER_PATH')
            os.environ['JUPYTER_PATH'] = temporary + (os.pathsep + original_jupyter_path if original_jupyter_path else '')
            try:
                stage = 'executing_notebook'
                write_receipt('RUNNING', stage=stage)
                NotebookClient(notebook, timeout=180, kernel_name='gf-walkthrough').execute(cwd=str(ROOT), env=env)
            finally:
                if original_jupyter_path is None:
                    os.environ.pop('JUPYTER_PATH', None)
                else:
                    os.environ['JUPYTER_PATH'] = original_jupyter_path
        stage = 'writing_outputs'
        nbformat.write(notebook, output / NAME)
        html, _ = HTMLExporter().from_notebook_node(notebook)
        (output / 'index.html').write_text(html)
        stage = 'validating_outputs'
        errors = [o for c in notebook.cells if c.cell_type == 'code'
                  for o in c.outputs if o.output_type == 'error']
        figures = sum('image/png' in o.get('data', {}) for c in notebook.cells if c.cell_type == 'code'
                      for o in c.outputs)
        counts = dict(code_cells=sum(c.cell_type == 'code' for c in notebook.cells),
                      figure_outputs=figures, error_outputs=len(errors))
        assert not errors and figures == 11, counts
        stage = 'writing_success_receipt'
        receipt = write_receipt('PASS', **counts)
    except BaseException as error:
        status = 'INTERRUPTED' if isinstance(error, KeyboardInterrupt) else 'FAILED'
        try:
            write_receipt(status, failed_stage=stage,
                          error=dict(type=type(error).__name__, message=str(error)))
        except OSError as receipt_error:
            print(f'Could not save failure receipt: {receipt_error}', file=sys.stderr)
        raise
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
