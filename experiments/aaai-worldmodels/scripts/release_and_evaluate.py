"""Seal overlapping decisions, then evaluate each fixed experiment once."""
from pathlib import Path
import subprocess,sys
from datetime import datetime,timezone
from worldmodels.data import sha256_file
from worldmodels.workflow import json_write
ROOT=Path(__file__).resolve().parents[1];RUNS=ROOT/'artifacts/runs';release=ROOT/'docs/test_release.json'
if release.exists():raise FileExistsError('Joint release is immutable.')
names=['docs/revision_protocol.md','docs/revision_amendments.md','docs/source_exclusions.json','docs/forecast_reuse_certificate.json','docs/training_gradient_audit.json','worldmodels/binary_clean.py','worldmodels/clean_data.py','worldmodels/acquisition_control.py','scripts/analyze_research.py','scripts/release_and_evaluate.py']
names += ['worldmodels/'+n+'.py' for n in ['revision','objective_audit','evaluation','model','training','data','pose','workflow','masking','features']]
names += ['artifacts/runs/pilot-comparison-clean-v1/results.json']
for run,files in {'binary-development-clean-v1':['seal.json','selection.json'],'quality-development-clean-v1':['seal.json','selection.json'],'revision-development-v1':['freeze.json','development.json'],'objective-audit-development-v1':['freeze.json','development.json'],'pilot-input-clean-v1':['preparation.json','windows.csv','windows.npz','pose_lineage.json'],'extended-input-clean-v1':['preparation.json','windows.csv','windows.npz','row_map.csv'],'revision-validation-clean-v1':['lineage.json']}.items():names.extend('artifacts/runs/'+run+'/'+name for name in files)
for run in ['revision-development-v1','objective-audit-development-v1']:
 if (RUNS/run/'test_access.json').exists():raise ValueError('Prior final test access: '+run)
json_write(release,dict(status='frozen_before_joint_test_release',utc=datetime.now(timezone.utc).isoformat(),frozen_files={name:sha256_file(ROOT/name) for name in names},primary_binary='jepa_mixed versus RF, fixed before clean refitting',primary_forecast='weight1 residual_det versus persistence; all seeds/arms retained despite failed validation criterion',independent_review='Person correction and checkpoint inheritance independently verified; see revision_review.md.',caveat='Earlier development comparison code computed test files and a pre-gate notebook persisted tables. Authors did not inspect metric values for decisions. This is an internal holdout, not an independently preregistered untouched clinical test set.'))
def execute(module,command,**paths):
 args=[sys.executable,'-m',module,command]
 for key,value in paths.items():args.extend(['--'+key.replace('_','-'),str(RUNS/value)])
 with (ROOT/'artifacts/final_evaluation.log').open('a') as log:subprocess.run(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
 print('Completed',module,flush=True)
execute('worldmodels.binary_clean','evaluate',development='binary-development-clean-v1',output='binary-test-clean-v1')
execute('worldmodels.acquisition_control','evaluate',development='quality-development-clean-v1',output='quality-test-clean-v1')
execute('worldmodels.revision','evaluate',development='revision-development-v1',output='revision-test-v1')
consumed=[RUNS/'revision-test-v1/results.json',RUNS/'revision-test-v1/persistence_sources.csv']
consumed.extend(RUNS/f'revision-test-v1/seed_{seed}/residual_det_actual_sources.csv' for seed in [27,41,73])
pins={str(p.relative_to(ROOT)):sha256_file(p) for p in consumed}
json_write(ROOT/'docs/objective_test_inputs.json',dict(primary_inputs=pins,joint_release_sha256=sha256_file(release)))
execute('worldmodels.objective_audit','evaluate',development='objective-audit-development-v1',primary_test='revision-test-v1',output='objective-audit-test-v1')
assert all(sha256_file(ROOT/p)==h for p,h in pins.items())
json_write(ROOT/'docs/test_execution.json',dict(status='all_fixed_evaluations_complete',utc=datetime.now(timezone.utc).isoformat(),joint_release_sha256=sha256_file(release),objective_consumed_inputs_unchanged=True,source_csv_sha256={str(p.relative_to(ROOT)):sha256_file(p) for name in ['binary-test-clean-v1','quality-test-clean-v1','revision-test-v1','objective-audit-test-v1'] for p in (RUNS/name).rglob('*.csv')},results={r:sha256_file(RUNS/r/'results.json') for r in ['binary-test-clean-v1','quality-test-clean-v1','revision-test-v1','objective-audit-test-v1']}))
