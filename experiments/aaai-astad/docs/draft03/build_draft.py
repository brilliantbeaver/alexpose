"""Build a standalone gait-anomaly manuscript from its local template and saved evidence."""
from pathlib import Path
import hashlib,json,re,shutil
import numpy as np,pandas as pd,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
B=Path(__file__).resolve().parent;BASE=B.parents[1]
e=json.loads((B/'study_evidence.json').read_text());audit=B/'advisory/completion_audit_v1'
summary=pd.read_csv(audit/'completion_summary.csv',dtype={'outer_fold':str})
ex=summary.query("dataset=='external_angles' and outer_fold=='all'").iloc[0]
ga=summary.query("dataset=='GAVD' and bank=='E_leg' and outer_fold=='all'").iloc[0]
gu=summary.query("dataset=='GAVD' and bank=='U' and outer_fold=='all'").iloc[0]
normal=pd.DataFrame(e['normal_test_error']);means=normal.groupby(['policy','bank'])[['loss_base','loss_own','loss_both']].mean()
ub=next(v for v in e['normal_test_benefit'] if v['policy']=='U' and v['bank']=='E_leg')
u=means.loc['U','E_leg'];relative=(u.loss_own-u.loss_both)/u.loss_own
def pct(x):return f'{100*x:.1f}'
def pct_ci(row):return f'[{pct(row.relative_ci_low)}, {pct(row.relative_ci_high)}]'
def hashfile(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
# Figure generation has its own local source and numerical manifest.
import importlib.util
spec=importlib.util.spec_from_file_location('draft03_figures',B/'build_figures.py')
figures=importlib.util.module_from_spec(spec);spec.loader.exec_module(figures)
figures.main()
figdir=B/'figures'
extpath=BASE/'artifacts/iteration02'/e['external_protocol_hash']/'derivatives/interpolation_audit_v1/participant_comparison.csv'


fields={
    'GAVD_COMPLETION_PERCENT':pct(relative),
    'LAB_COMPLETION_PERCENT':pct(ex.relative_mse_reduction),
    'MASK_COMPLETION_PERCENT':pct(ga.relative_mse_reduction),
    'LAB_IMPROVED':str(int(ex.units_with_lower_candidate_mse)),
    'LAB_NORMALS':str(int(ex['count'])),
}
text=(B/'manuscript_template.md').read_text()
for key,value in fields.items():text=text.replace('@@'+key+'@@',value)
if re.search(r'@@[A-Z_]+@@',text):raise ValueError('Unresolved numerical field')
for phrase in ['original analysis','original primary','earlier completed','this draft','draft02','positive emphasis']:
    if phrase in text.lower():raise ValueError('Development narrative in standalone manuscript: '+phrase)
(B/'manuscript.md').write_text(text.replace('−','-'))
inputs=[B/'study_evidence.json',B/'manuscript_template.md',B/'technical_details.md',audit/'completion_summary.csv',audit/'audit_manifest.json',extpath,B/'build_figures.py',figdir/'manifest.json']
for path,expected in e['numerical_sources'].items():
    if hashfile(BASE/path)!=expected:raise ValueError('Changed evidence source: '+path)
evidence=dict(study=e,completion_summary=summary.to_dict('records'),completion_audit=json.loads((audit/'audit_manifest.json').read_text()),
    analysis_status='exploratory controlled gait-anomaly study; descriptive completion intervals',
    numerical_fields=fields,sources={str(p.relative_to(BASE)):hashfile(p) for p in inputs},generator_sha256=hashfile(__file__))
(B/'evidence.json').write_text(json.dumps(evidence,indent=2)+'\n')

print(B/'manuscript.md')
