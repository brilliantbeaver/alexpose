#!/usr/bin/env python3
"""Build the ordered Gait Fidelity tutorials from explicit prose and code cells."""
from pathlib import Path
import json
import textwrap

HERE = Path(__file__).resolve().parent


def md(text):
    return dict(cell_type='markdown', metadata={}, source=textwrap.dedent(text).strip() + '\n')


def code(text):
    return dict(cell_type='code', metadata={}, source=textwrap.dedent(text).strip() + '\n', execution_count=None, outputs=[])


SETUP = code('''
from pathlib import Path
import json, os, sys

# Find the checkout/release from the notebook's working directory.
ROOT = Path(os.environ.get('GF_ROOT', Path.cwd())).resolve()
while not (ROOT / 'src/gavd6_sjepa').is_dir() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
assert (ROOT / 'src/gavd6_sjepa').is_dir(), 'Open this notebook from the GAVD6 checkout or release.'
sys.path.insert(0, str(ROOT / 'notebooks/gait_fidelity'))
sys.path.insert(0, str(ROOT / 'src'))
from tutorial_helpers import configure, preview_images
study = configure(ROOT)
''')

from lesson_start import build_start_cells
from lesson_data import build_data_cells, build_mask_cells
from lesson_model import build_architecture_cells, build_training_cells
from lesson_evaluation import build_evaluation_cells, build_verification_cells
from lesson_experiments import build_experiment_cells
from lesson_response import build_response_cells

LESSONS = [
    ('00_start_here.ipynb', 'Start the Gait Fidelity study', build_start_cells),
    ('01_data_and_references.ipynb', 'Trace paired data and input-only preprocessing', build_data_cells),
    ('02_masking_and_controls.ipynb', 'Construct tokens and stochastic masks', build_mask_cells),
    ('03_experiment_matrix.ipynb', 'Build the architecture and inspect the experiment matrix', build_architecture_cells),
    ('04_run_and_monitor.ipynb', 'Derive losses, perform an update and run the study', build_training_cells),
    ('05_evaluate_and_visualize.ipynb', 'Compute gait fidelity and inspect restored trajectories', build_evaluation_cells),
    ('06_verify_and_write.ipynb', 'Reconstruct results and estimate uncertainty', build_verification_cells),
]

NOTEBOOKS = {
    filename: [md(f'# {index:02d} · {title}'), SETUP, *builder(md, code)]
    for index, (filename, title, builder) in enumerate(LESSONS)
}


GROUPS = [
    ('A_masking_and_change.ipynb', 'M', 'Masking and movement-change supervision', 12,
     '''This experiment crosses the representation objective, mask policy and output loss. It asks whether predicting reference features helps preserve movement, and whether that answer changes when the encoder sees structured gaps in the observed sequence. Coordinate-pretrained and paired-JEPA encoders receive the same data, mask budgets and update counts. JEPA also changes the loss and adds its feature predictor, projector, regularizer and moving-average teacher, so this comparison evaluates the full declared pretraining recipe.''',
     '''Compare coordinate versus JEPA within the same mask and output loss. Compare base versus paired-change supervision within the same encoder and mask. A comparison that changes both the encoder and the loss cannot isolate either factor. The three masks are whole-body time blocks, uniform joint-time tokens and connected anatomical regions over time.''',
     '''Keep both endpoints in every training batch with equal coordinate exposure. Read the change term together with per-example and valid re-pairing controls from experiment E. Improvement limited to the optimized knee measurement needs the prespecified untrained measurement or held conditions before a broader preservation claim.'''),
    ('B_mask_structure_controls.ipynb', 'T', 'Anatomical topology and mask-duration controls', 8,
     '''This experiment tests why a graph-time mask might help. Connected body regions change both which joints are missing and the duration of the missing interval, so the graph arm needs controls that separate those properties.''',
     '''Compare the shuffled-topology and random-joint-interval recipes with the graph-time recipes in experiment A under the same encoder, output objective and seed. Shuffled topology changes the connections defining a region; random-joint intervals use the same nominal duration distribution while removing anatomical connectivity; realized exposure and run lengths still require auditing.''',
     '''Inspect actual per-joint hide frequencies, temporal run lengths and retained visible support. Equal average mask fraction alone does not establish a matched masking task. A topology effect that disappears after matching visibility should be described at that narrower level.'''),
    ('C_practical_benchmarks.ipynb', 'P', 'Direct, static and temporal-refinement benchmarks', 4,
     '''This experiment establishes whether the proposed restoration improves on practical alternatives. Direct models learn coordinate correction end to end. The static model removes cross-frame coordinates while retaining full-window confidence, availability and time. The temporal refiner adapts a learned trajectory-refinement architecture to the same body-12 inputs.''',
     '''The direct base and direct paired-change recipes isolate the added movement objective within one trainable architecture. Static and temporal-refiner base recipes provide practical benchmarks but do not have every loss-matched variant. Also read unchanged tracks, training-only offsets and affine calibration, and fixed temporal filters from the common evaluation.''',
     '''State the temporal refiner's adaptation explicitly rather than treating it as a reproduction with the original authors' weights. Lower coordinate error and lower movement error can favor different methods; use the full paired population, support and failures before choosing a scientific conclusion.'''),
    ('D_pretraining_information.ipynb', 'I', 'Initialization and input-reference alignment controls', 4,
     '''This experiment asks whether pretraining transfers useful information. An initialized encoder tests what the architecture and output training can achieve without encoder optimization. A shuffled-reference JEPA encoder tests whether correctly aligned reference features matter.''',
     '''Compare each control with the prespecified graph-time paired-JEPA recipe under the same base or paired-change output objective. Initialized encoders have no pretraining mask or loss. All output networks receive the same unmasked deployment-style observed inputs and independently initialized output parameters.''',
     '''Check that the shuffled-reference source window differs from the input family and does not cross split boundaries. These controls attribute findings to the graph-time comparison only; they do not establish that every mask benefits equally from alignment or pretraining.'''),
    ('E_pairing_and_label_controls.ipynb', 'L', 'Per-example labels and valid movement re-pairing', 6,
     '''This experiment asks whether meaningful movement pairing contributes beyond additional supervision. It crosses coordinate-pretrained, paired-JEPA and direct encoders with per-example measurement labels or valid re-paired change labels.''',
     '''Compare the per-example control and valid re-pairing control with the corresponding paired-change arm from experiment A or C. Re-pairing rearranges training source families while preserving endpoint roles and applicable condition strata, then recomputes the true difference between the new reference endpoints.''',
     '''Read each pairing-audit.json before interpreting the comparison. The sampler packs complete two- or three-family permutation cycles into each batch, so all objectives receive the same endpoint multiset at every update. A nominal 16-endpoint batch can contain 18 or 20 endpoints when a three-cycle crosses its boundary; receipts retain the actual counts. Endpoint frequency and reference-change distribution must meet the frozen matching rule. If the declared tolerance cannot be met, report that mismatch and narrow the pairing-specific claim. Reusing the original target differences after re-pairing would create false labels.'''),
]


def group_notebook(filename, group, title, count, purpose, comparisons, cautions):
    return [md(f'# {filename[0]} · {title}\n\n{purpose}\n\nRead after notebooks 00–06. This notebook uses the prepared bundle and saved evaluation from that same run; the internal recipe group is `{group}`.'), SETUP,
            md('## Inspect the exact recipe cells\n\n' + comparisons),
            code(f'''
            import pandas as pd
            plan = study.artifact('plan.json')
            group_recipes = [r for r in plan['recipes'] if r['group'] == {group!r}]
            expected_count = {count} if plan.get('experiment_set', 'full') == 'full' else {{'M': 4, 'T': 0, 'P': 2, 'I': 4, 'L': 0}}[{group!r}]
            assert len(group_recipes) == expected_count, 'Saved plan differs from the selected experiment set.'
            recipe_ids = {{r['recipe_id'] for r in group_recipes}}
            display(pd.DataFrame(group_recipes))
            print('Final models:', len(group_recipes) * len(plan['seeds']), 'Seeds:', plan['seeds'])
            if not group_recipes:
                print('This group is outside the saved core protocol. Its worked examples are educational; no results are implied.')
            '''),
            *build_experiment_cells(group, md, code),
            md('''
            ## Follow the shared dependencies

            This tutorial inspects the existing central queue. It does not launch a separate copy of its group: that would duplicate pretraining and break the global budget. Notebook 04 launches all groups, and this table identifies the phases that belong to the present comparison.
            '''),
            code('''
            final_phases = [p for p in plan['phases'] if p['phase'] != 'pretrain' and p['recipe']['recipe_id'] in recipe_ids]
            parent_ids = {parent for p in final_phases for parent in p['depends_on'] if parent != 'prepare'}
            selected = [p for p in plan['phases'] if p in final_phases or p['phase_id'] in parent_ids]
            display(pd.DataFrame([{'phase_id': p['phase_id'], 'phase': p['phase'], 'seed': p['seed'],
                                  'depends_on': ', '.join(p['depends_on'])} for p in selected]))
            '''),
            md('## Inspect completed checkpoints and learning histories\n\nEach completed phase links its retained checkpoint, history and predictions to its source identity. Missing phases are reported as pending; a checkpoint from another study is not substituted.'),
            code('''
            ledger_path = study.work / 'ledger.json'
            completed = json.loads(ledger_path.read_text()).get('completed', {}) if ledger_path.exists() else {}
            rows = []
            for phase in selected:
                saved = completed.get(phase['phase_id'])
                result = saved.get('result', {}) if saved else {}
                rows.append({'phase_id': phase['phase_id'], 'status': 'complete' if saved else 'pending',
                             'checkpoint': result.get('checkpoint'), 'predictions': result.get('predictions')})
            display(pd.DataFrame(rows))
            history_candidates = []
            for row in rows:
                if row['checkpoint']:
                    history_path = Path(row['checkpoint']).parent / 'history.json'
                    if history_path.exists(): history_candidates.append(history_path)
            if history_candidates:
                history_path = history_candidates[0]
                print('First declared completed history:', history_path)
                display(pd.DataFrame(json.loads(history_path.read_text())))
            else:
                print('No completed histories yet. Run or resume the central queue from notebook 04.')
            '''),
            md('## Read group results on the common population\n\n' + cautions),
            code('''
            person_path = study.work / 'evaluation/per-person.csv'
            if person_path.exists():
                people = pd.read_csv(person_path)
                display(people.loc[people['method'].isin(recipe_ids)])
                coverage_path = study.work / 'evaluation/coverage.csv'
                if coverage_path.exists():
                    coverage = pd.read_csv(coverage_path)
                    display(coverage.loc[coverage['method'].isin(recipe_ids)])
            else:
                print('Evaluation is pending. These recipe cards do not fabricate or extrapolate results.')
            '''),
            md('Read the matching controls from the other experiment tutorials before attribution. Full evaluation and numerical reconstruction are covered by notebooks 05 and 06.')]


def main():
    notebooks = dict(NOTEBOOKS)
    for group in GROUPS:
        notebooks['experiments/' + group[0]] = group_notebook(*group)
    notebooks['experiments/F_jepa_response.ipynb'] = build_response_cells(md, code)
    for name, cells in notebooks.items():
        for index, cell in enumerate(cells):
            cell['id'] = f'cell-{index:03d}'
        notebook = dict(cells=cells, metadata={
            'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
            'language_info': {'name': 'python', 'version': '3.11'},
            'gait_fidelity': {'default_mode': 'software-fixture', 'scientific_evidence': False},
        }, nbformat=4, nbformat_minor=5)
        target = HERE / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(notebook, indent=1) + '\n')
    print(f'Wrote {len(notebooks)} tutorial notebooks.')


if __name__ == '__main__':
    main()
