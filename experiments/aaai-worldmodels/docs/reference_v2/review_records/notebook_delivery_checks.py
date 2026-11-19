"""Independently inspect final notebook execution and publication reconstruction."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import nbformat
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
PROJECT = ROOT / 'experiments/aaai-worldmodels'
OUT = PROJECT / 'artifacts/reference-v2-independent-review'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    report_path = PROJECT / 'docs/reference_v2/notebook_execution.json'
    report = json.loads(report_path.read_text())
    assert report['status'] == 'complete' and report['full_sequence'] is True
    assert len(report['notebooks']) == 7 and report['total_code_cells'] == 43
    generator = PROJECT / 'scripts/build_reference_v2_notebooks.py'
    assert sha(generator) == report['generator_sha256']
    identities = {}; rows = []; errors = []; rendered = {}
    unique_inputs = {}
    for entry in report['notebooks']:
        source = PROJECT / entry['name']; output = PROJECT / entry['output']
        assert sha(source) == entry['source_sha256']
        assert sha(output) == entry['output_sha256']
        assert entry['generator_sha256'] == sha(generator)
        a = nbformat.read(source, as_version=4); b = nbformat.read(output, as_version=4)
        nbformat.validate(a); nbformat.validate(b)
        assert len(a.cells) == len(b.cells)
        count = 0; text = ''
        for original, executed in zip(a.cells, b.cells):
            assert original.cell_type == executed.cell_type
            assert original.source == executed.source
            if original.cell_type == 'code':
                count += 1
                ast.parse(original.source)
                assert original.execution_count is None and original.outputs == []
                assert executed.execution_count == count
                for value in executed.outputs:
                    if value.output_type == 'error':
                        errors.append((entry['name'], count, value.get('ename')))
                    text += value.get('text', '')
                    text += '\n'.join(str(v) for v in value.get('data', {}).values())
            else:
                assert '\u2014' not in original.source
        assert count == entry['code_cells'] == (7 if entry['name'].startswith('26') else 6)
        trace = PROJECT / 'artifacts/executed-reference-v2/inputs' / (source.stem + '.json')
        assert sha(trace) == entry['input_record_sha256']
        assert json.loads(trace.read_text()) == entry['inputs']
        for name, expected in entry['inputs']['inputs'].items():
            if name not in unique_inputs:
                actual = Path(name) if Path(name).is_absolute() else ROOT / name
                assert sha(actual) == expected, name
                unique_inputs[name] = expected
            assert unique_inputs[name] == expected
        assert 'MISSING from completed evaluation' not in text
        assert 'INCOMPLETE; inspect run ledger' not in text
        rendered[entry['name'][:2]] = text
        identities[entry['name']] = dict(source_sha256=sha(source), executed_sha256=sha(output), input_record_sha256=sha(trace))
        rows.append(dict(name=entry['name'], ordered_successful_cells=count, source_output_cells_identical=True, input_files=len(entry['inputs']['inputs'])))
    assert errors == []
    # Every saved forecast arm and both calibration families remain visible.
    core = PROJECT / 'artifacts/runs/reference-v2-core-v1'
    arms = set(pd.read_csv(core / 'refs_source_scores.csv').arm) | set(pd.read_csv(core / 'masks_source_scores.csv').arm)
    text = rendered['26']
    for arm in arms | {'mean_pose', 'mean_residual', 'calibration_models.json', 'posture_calibration_models.json'}:
        assert arm in text, arm
    for gate in ('latent_objective', 'learned_masks', 'phase_conditioning', 'stochastic_future'):
        assert gate in text, gate
    assert 'reference-v2-floor-posthoc-v1' in text and 'reference-v2-floor-posthoc-v2' in text
    # Inspect actual reconstructed files, not only the author's boolean report.
    rebuild_record = PROJECT / 'docs/reference_v2/notebook_26_publication_rebuild.json'
    rebuilt = json.loads(rebuild_record.read_text())
    assert len(rebuilt['tables']) == 6 and len(rebuilt['figures']) == 5
    assert sha(PROJECT / 'scripts/build_next_draft03.py') == rebuilt['renderer_sha256']
    for row in rebuilt['tables']:
        name = row['table'] + '.csv'
        published = pd.read_csv(PROJECT / 'docs/draft03' / name)
        reconstructed = pd.read_csv(PROJECT / 'artifacts/executed-reference-v2/manuscript_rebuild' / name)
        pd.testing.assert_frame_equal(published, reconstructed, check_exact=True)
        assert len(published) == row['rows'] and row['all_cells_match']
    for row in rebuilt['figures']:
        assert row['exact_svg_match']
        assert sha(PROJECT / 'docs/draft03/figures' / row['figure']) == row['sha256']
        assert sha(PROJECT / 'artifacts/executed-reference-v2/manuscript_rebuild/figures' / row['figure']) == row['sha256']
    result = dict(identity='reference-v2-independent-notebook-delivery-v1', status='passed', reviewed_at_utc=datetime.now(timezone.utc).isoformat(), script_sha256=sha(__file__), execution_report_sha256=sha(report_path), generator_sha256=sha(generator), notebooks=rows, notebook_identities=identities, successful_code_cells=sum(r['ordered_successful_cells'] for r in rows), error_outputs=errors, unique_input_files_verified=len(unique_inputs), complete_forecast_arm_registry=len(arms), calibration_families=2, retained_posthoc_controls=2, displayed_extension_gates=4, exact_rebuilt_tables=6, exact_rebuilt_svg_figures=5, rebuild_record_sha256=sha(rebuild_record), publication_pdf_sha256=sha(PROJECT / 'docs/draft03/manuscript.pdf'), limitations=['Independent inspection of exact saved executions, not a second seven-kernel run.', 'The final report explicitly retains six unchanged notebook executions and reruns notebook26 after publication repair.', 'Human visual inspection of publication pages is separately recorded.'])
    (OUT / 'notebook_delivery_checks.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k:result[k] for k in ('status','successful_code_cells','unique_input_files_verified','complete_forecast_arm_registry','exact_rebuilt_tables','exact_rebuilt_svg_figures')}, indent=2))


if __name__ == '__main__':
    main()
