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
figdir=B/'figures';figdir.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
extpath=BASE/'artifacts/iteration02'/e['external_protocol_hash']/'derivatives/interpolation_audit_v1/participant_comparison.csv'
ext=pd.read_csv(extpath)
cols=['loss_interpolation','loss_base','loss_own','loss_both']
if not set(cols).issubset(ext):
    cols=['interpolation','base','own','both']
    if not set(cols).issubset(ext):raise ValueError('Inspect saved interpolation schema')
fig,axes=plt.subplots(1,2,figsize=(7,2.45),gridspec_kw={'width_ratios':[1.1,1]})
for fold,frame in ext.groupby('outer_fold'):
    axes[0].plot(range(4),frame[cols].mean(),marker='o',lw=1,alpha=.65,label=f'Fold {fold+1}')
axes[0].set_xticks(range(4),['Interpolation','Retained f','Own f+h','Both f+g'],rotation=15,fontsize=7)
axes[0].set_ylabel('Healthy participant mean MSE (deg²)');axes[0].set_title('Independent laboratory completion');axes[0].grid(axis='y',alpha=.15)
axes[0].legend(fontsize=6.5,ncol=2,frameon=False)
for y,row in enumerate([gu,ga]):
    point=100*row.relative_mse_reduction;lo=100*row.relative_ci_low;hi=100*row.relative_ci_high
    axes[1].errorbar(point,y,xerr=[[point-lo],[hi-point]],fmt='o',capsize=3,color='#17858a')
axes[1].axvline(0,color='#777',lw=.8);axes[1].set_yticks([0,1],['Uniform U bank','Distal E_leg bank']);axes[1].set_xlabel('MSE reduction: A versus U training (%)');axes[1].set_title('Mask benefit depends on the task');axes[1].grid(axis='x',alpha=.15)
fig.tight_layout()
for suffix in ['svg','pdf','png']:fig.savefig(figdir/f'positive_completion.{suffix}',bbox_inches='tight',dpi=180)
plt.close(fig)


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
inputs=[B/'study_evidence.json',B/'manuscript_template.md',audit/'completion_summary.csv',audit/'audit_manifest.json',extpath]
for path,expected in e['numerical_sources'].items():
    if hashfile(BASE/path)!=expected:raise ValueError('Changed evidence source: '+path)
evidence=dict(study=e,completion_summary=summary.to_dict('records'),completion_audit=json.loads((audit/'audit_manifest.json').read_text()),
    analysis_status='exploratory controlled gait-anomaly study; descriptive completion intervals',
    numerical_fields=fields,sources={str(p.relative_to(BASE)):hashfile(p) for p in inputs},generator_sha256=hashfile(__file__))
(B/'evidence.json').write_text(json.dumps(evidence,indent=2)+'\n')
(figdir/'positive_completion_manifest.json').write_text(json.dumps({'inputs':evidence['sources'],'caption':'Measured completion comparisons, fixed targets and conditional paired intervals.','generated_from':str(Path(__file__).relative_to(BASE))},indent=2)+'\n')
print(B/'manuscript.md')
