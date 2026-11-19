"""Reframe verified results without replacing the frozen protocol or scores."""
from pathlib import Path
import hashlib,json,re,shutil
import numpy as np,pandas as pd,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
B=Path(__file__).resolve().parent;D2=B.parent/'draft02';BASE=B.parents[1]
e=json.loads((D2/'evidence.json').read_text());audit=B/'advisory/completion_audit_v1'
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
for p in (D2/'figures').glob('*'):
    if p.is_file():shutil.copy2(p,figdir/p.name)
for name in ['template.tex','build_manuscript.py','references.bib']:
    shutil.copy2(D2/name,B/name)
if not (B/'author_kit').exists():shutil.copytree(D2/'author_kit',B/'author_kit')

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
extpath=BASE/'artifacts/iteration02'/e['external_protocol_hash']/'derivatives/interpolation_audit_v1/participant_comparison.csv'
ext=pd.read_csv(extpath)
print('External interpolation columns:',ext.columns.tolist())
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

abstract=f'''Observations of the other leg can help complete missing gait motion, but successful completion must be separated from successful anomaly detection. We compare a retained own-leg predictor, an additional own-leg control with matched input width, and a both-leg correction against identical observable targets. Three motion-independent training masks share fixed evaluation banks. In an exploratory GAVD study, the uniform-trained bilateral predictor reduces normal completion MSE by {pct(relative)}% relative to the own-leg augmentation on a declared distal-leg challenge. A post-outcome descriptive analysis of the independent laboratory study finds a {pct(ex.relative_mse_reduction)}% reduction for the bilateral predictor relative to the own-leg augmentation, with conditional 95% interval {pct_ci(ex)}%, and gives lower prediction error for {int(ex.units_with_lower_candidate_mse)} of {int(ex['count'])} participants. Anatomical rather than uniform training improves the fixed challenge task by {pct(ga.relative_mse_reduction)}%, but not conclusively on the unchanged uniform task. These are improvements under the fitted predictor families, whose numerical feature ranks differ. The original primary anomaly analysis remains negative: the frozen relational fusion lowers AUROC in both datasets. A sparse timing diagnostic also fails prediction adequacy. The contribution is a controlled measurement of when opposite-leg context helps, with explicit limits on its use as abnormality or neurological evidence.'''
intro='''# Introduction

When a short part of the left ankle's movement is hidden, the visible left knee, heel and foot may already reveal the answer. The right leg may become useful when those local clues are insufficient. This suggests a practical question: **when does seeing the other leg help reconstruct missing movement, and what must be checked before using that help to detect unusual gait?**

A normal-trained bilateral predictor improves completion on a declared distal-leg challenge and on laboratory angles from independent participants. It exceeds the declared own-leg augmentation and, in the laboratory audit, direct interpolation. Different numerical feature ranks and shared body references prevent interpreting this as an isolated neural measurement.

Our method retains the local prediction, fixes the answer across branches, and restricts access before constructing features. It separates training for missing input, available evaluation information, and conversion of prediction behavior into anomaly evidence. Success at completion need not transfer to detection.

The positive finding concerns a movement-modeling component. The fixed equal-weight anomaly fusion still loses to local movement. Timing edits can also produce nonzero responses when paired inputs are too sparse for adequate prediction. These findings bound the completion result's interpretation.

The contribution is an executed comparison with observable targets, a retained local solution, a nominal-width own-leg control, matched masking budgets and independent participant replication. Tube masking, residual models and conditional surprise already exist. The new value lies in demonstrating useful completion under declared observation tasks while checking the alternative explanations that limit its meaning. This draft gives greater emphasis to a predeclared secondary completion diagnostic after outcomes were inspected. The original primary detection analysis remains unchanged and visible.
'''
base=(D2/'manuscript.md').read_text()
common=base.split('# Related work and physiological motivation',1)[1].split('# Results',1)[0]
introfig=base.split('\\begin{figure*}[t]',1)[1].split('\\end{figure*}',1)[0]
intro+='\n\\begin{figure*}[t]'+introfig+'\\end{figure*}\n\n'
results=base.split('# Results\n',1)[1]
first=results.split('## Timing adequacy fails despite recording coverage',1)[0]
table=first.split('\\begin{table}[t]',1)[1]
detection='\\begin{table}[t]'+table
normalfig=first.split('\\begin{figure*}[t]',1)[1].split('\\end{figure*}',1)[0]
timing=results.split('## Timing adequacy fails despite recording coverage',1)[1].split('## Independent participant replication',1)[0]
external=results.split('## Independent participant replication\n',1)[1].split('# Observation failures and limitations',1)[0]
external_prediction=external.split('Despite useful completion,',1)[0]
external_detection='Despite useful completion,'+external.split('Despite useful completion,',1)[1]
limits=base.split('# Observation failures and limitations\n',1)[1].split('# Conclusion',1)[0]
positive=f'''# Results: useful completion and its boundaries

## A declared distal gap reveals useful completion

We lead with the U-trained reference, rather than selecting the largest gain among the three policies. On the fixed E_leg challenge, its mean held-out normal MSE falls from {u.loss_own:.6f} for the own-leg augmentation to {u.loss_both:.6f} for the bilateral correction. This is a {pct(relative)}% reduction as a ratio of source means. The absolute benefit is {ub['mean_benefit']:.6f}, with conditional 95% interval [{ub['conditional_95_interval'][0]:.6f}, {ub['conditional_95_interval'][1]:.6f}], across 28 normal recordings. These intervals use 5,000 within-fold source resamples of saved predictions.

All three policies have positive normal benefit intervals on E_leg. On the ordinary U bank, U and A remain uncertain. The focused task hides nearby distal joints together, making local geometry less available. Its stronger benefit supports task-specific usefulness, not a claim that one task's larger loss or gain is intrinsically better. Targets remain pose estimates; artificially hiding an observed coordinate does not validate reconstruction of genuinely unobserved anatomy.

\\begin{{figure*}}[t]{normalfig}\\end{{figure*}}

## Independent healthy participants also benefit

{external_prediction}
The aggregate own-to-both reduction is {ex.absolute_mse_reduction:.3f} squared degrees, conditional interval [{ex.absolute_ci_low:.3f}, {ex.absolute_ci_high:.3f}], or {pct(ex.relative_mse_reduction)}%, {pct_ci(ex)}%. The bilateral model gives lower prediction error for {int(ex.units_with_lower_candidate_mse)} of {int(ex['count'])} held-out healthy participants. Every fold has a positive mean reduction. These additional completion intervals are post-outcome descriptive summaries of frozen predictions, using 1,000 paired participant resamples within original folds. They omit training uncertainty and do not retrospectively become a primary endpoint.

## Anatomical masking helps the focused task

An additional paired audit compares A with U while retaining exactly the same E_leg target keys, mask draws, support and U-trained baseline. Normal bilateral MSE falls from {ga.comparator_mean_mse:.6f} to {ga.candidate_mean_mse:.6f}, a {pct(ga.relative_mse_reduction)}% reduction, conditional interval {pct_ci(ga)}%. All four fold means improve and {int(ga.units_with_lower_candidate_mse)} of 28 sources improve. E_leg resembles part of A's training design, so this is evidence of task alignment rather than universal mask superiority.

The unchanged U bank provides the necessary boundary: A's reduction is only {pct(gu.relative_mse_reduction)}%, interval {pct_ci(gu)}%, with improvement in {int(gu.units_with_lower_candidate_mse)} of 28 sources and a worse mean in one fold. We therefore do not call A generally superior. Both banks were declared before model fitting; these paired raw-error uncertainty summaries and the positive emphasis are post-outcome analyses. No masks, sources, hyperparameters or predictions changed.

\\begin{{figure*}}[t]
\\centering
\\includegraphics[width=\\textwidth]{{figures/positive_completion.pdf}}
\\caption{{Left: the laboratory interpolation, retained baseline, own augmentation and bilateral predictor have the same ordering in every held-out healthy fold. Right: A-versus-U training improves the focused E\\_leg task, with an inconclusive change on U. These are different comparisons and units, not a pooled effect. Interpolation and paired mask intervals are post-outcome audits of fixed predictions.}}
\\label{{fig:positive}}
\\end{{figure*}}

## Useful completion does not establish added detection

{detection}

{external_detection}

## Sparse paired inputs limit timing interpretation

{timing}
'''
conclusion='''# Conclusion

The declared bilateral predictors improve completion on a focused video task and independent laboratory angles. Anatomical training further improves the focused task, with inconclusive benefit on ordinary uniform masking.

Equal input width does not establish equal effective capacity. The anomaly fusion remains worse than local movement, and timing adequacy fails on sparse paired inputs. The next study should strengthen the own-leg feature control and test normal-only reliability calibration on new data. Useful prediction is a component to build on, not a diagnosis.
'''
manuscript='---\ntitle: "When the Other Leg Helps: Controlled Gait Completion for Anomaly Analysis"\nabstract: |\n  '+abstract+'\n---\n\n'+intro+'# Related work and physiological motivation'+common+positive+'# Observation failures and limitations\n'+limits+conclusion
manuscript=manuscript.replace('−','-')
(B/'manuscript.md').write_text(manuscript)
evidence=dict(original_evidence=e,completion_summary=summary.to_dict('records'),completion_audit=json.loads((audit/'audit_manifest.json').read_text()),editorial_status='post-outcome positive interpretation; original primary results unchanged',sources={str(p.relative_to(BASE)):hashfile(p) for p in [D2/'evidence.json',D2/'manuscript.md',audit/'completion_summary.csv',audit/'audit_manifest.json',extpath]},generator_sha256=hashfile(__file__))
(B/'evidence.json').write_text(json.dumps(evidence,indent=2)+'\n')
(figdir/'positive_completion_manifest.json').write_text(json.dumps({'inputs':evidence['sources'],'caption':'Actual frozen completion predictions and post-outcome paired uncertainty; no new fitted model or clinical claim.','generated_from':str(Path(__file__).relative_to(BASE))},indent=2)+'\n')
print(B/'manuscript.md')
