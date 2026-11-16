"""Build draft02 and a post hoc score audit from immutable restart outputs.
No fitting occurs. The frozen builder supplies validation and rendering helpers.
"""
from __future__ import annotations
import argparse, importlib.util, json, re, shutil
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
PROJECT=Path(__file__).resolve().parents[1]
OUT=PROJECT/'docs/draft02'
RUN=PROJECT/'artifacts/runs/restart-dynamics-v1'
spec=importlib.util.spec_from_file_location('frozen_paper_helpers',PROJECT/'scripts/build_restart_manuscript.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base);base.OUT=OUT


def score_audit():
    frame=pd.read_csv(RUN/'source_anomaly_scores.csv')
    folds=pd.read_csv(RUN/'cohort.csv')[['source_id','outer_fold']].drop_duplicates()
    frame=frame.merge(folds,on='source_id',validate='one_to_one')
    frame['low_motion_diagnostic']=-frame.persistence
    y=frame.label.ne('normal').to_numpy(int)
    values=frame[['excess_error','low_motion_diagnostic']].to_numpy()
    ix=[np.flatnonzero(y==k) for k in [0,1]]
    rng=np.random.default_rng(2703)
    estimates=np.array([roc_auc_score(y,values[:,j]) for j in range(2)])
    draws=[]
    for _ in range(5000):
        ii=np.concatenate([rng.choice(a,len(a),replace=True) for a in ix])
        draws.append([roc_auc_score(y[ii],values[ii,j]) for j in range(2)])
    draws=np.asarray(draws)
    pooled=[dict(score=key,auroc=float(estimates[j]),ci95=np.quantile(draws[:,j],[.025,.975]).tolist(),average_precision=float(average_precision_score(y,values[:,j]))) for j,key in enumerate(['excess_error','low_motion_diagnostic'])]
    paired=dict(estimate=float(estimates[0]-estimates[1]),ci95=np.quantile(draws[:,0]-draws[:,1],[.025,.975]).tolist())
    within=[]
    for fold,g in frame.groupby('outer_fold'):
        target=g.label.ne('normal')
        within.append(dict(fold=int(fold),sources=len(g),normal=int((~target).sum()),abnormal=int(target.sum()),excess_auroc=float(roc_auc_score(target,g.excess_error)),low_motion_auroc=float(roc_auc_score(target,g.low_motion_diagnostic))))
    conditions=[]
    for label in ['parkinsons','stroke','cerebral palsy','myopathic']:
        g=frame[frame.label.isin(['normal',label])]
        conditions.append(dict(label=label,normal=29,abnormal=int(g.label.eq(label).sum()),excess_auroc=float(roc_auc_score(g.label.ne('normal'),g.excess_error)),low_motion_auroc=float(roc_auc_score(g.label.ne('normal'),g.low_motion_diagnostic))))
    weights=np.array([x['normal']*x['abnormal'] for x in within])
    within_average={name:float(np.average([x[name] for x in within],weights=weights)) for name in ['excess_auroc','low_motion_auroc']}
    # Pair-weighted within-fold AUROC; resample each fold/class cell and keep its count fixed.
    cells=[(np.flatnonzero((frame.outer_fold.to_numpy()==f)&(y==0)),np.flatnonzero((frame.outer_fold.to_numpy()==f)&(y==1))) for f in sorted(frame.outer_fold.unique())]
    rng=np.random.default_rng(2704);within_draws=[]
    for _ in range(5000):
        aucs=[]
        for a,b in cells:
            ii=np.concatenate([rng.choice(a,len(a),replace=True),rng.choice(b,len(b),replace=True)])
            aucs.append([roc_auc_score(y[ii],values[ii,j]) for j in range(2)])
        within_draws.append(np.average(aucs,axis=0,weights=weights))
    within_draws=np.array(within_draws)
    within_pair=dict(estimate=within_average['excess_auroc']-within_average['low_motion_auroc'],ci95=np.quantile(within_draws[:,0]-within_draws[:,1],[.025,.975]).tolist(),seed=2704,resamples=5000)
    report=dict(identity='draft03-frozen-score-audit-v1',status='completed post hoc diagnostic; no new trained model or confirmatory evaluation',inputs={str(p.relative_to(PROJECT)):base.digest(p) for p in [RUN/'source_anomaly_scores.csv',RUN/'cohort.csv']},seed=2703,resamples=5000,resampling='paired stratified source bootstrap conditional on fixed overlapping fits',score_direction='Frozen scores are unchanged. Negative persistence is an explicitly post hoc diagnostic.',pooled=pooled,excess_minus_low_motion=paired,within_fold=within,within_fold_pair_weighted_auroc=within_average,within_fold_excess_minus_low_motion=within_pair,condition_slices=conditions,limitations=['Low-motion sign chosen after outcomes; not a validated deployable score.','No threshold or model selected using these results.','Within-fold rankings exclude cross-fold pairs but training sets overlap.','Both scores use observed future motion and are retrospective.'])
    base.write_json(OUT/'posthoc_score_audit.json',report)
    frame.to_csv(OUT/'audit_source_scores.csv',index=False)
    pd.DataFrame(within).to_csv(OUT/'within_fold_scores.csv',index=False)
    pd.DataFrame(conditions).to_csv(OUT/'condition_scores.csv',index=False)
    return frame,report


def figures(frame,result,audit):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'svg.fonttype':'none','svg.hashsalt':'medwm-draft03-v1','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    (OUT/'figures').mkdir(exist_ok=True)
    fig,ax=plt.subplots(figsize=(6.6,2.15));ax.axis('off');ax.set(xlim=(0,10),ylim=(0,3.3))
    nodes=[(.05,1.35,1.9,1.05,'Public gait video\nExtract 2D pose'),(2.6,1.35,1.9,1.05,'Normal sources\nFit a forecast'),(5.15,1.35,1.9,1.05,'Held-out video\nPredict 0.8 s'),(7.7,1.35,2.2,1.05,'Observed future\nScore both models')]
    for x,y,w,h,txt in nodes:
        ax.add_patch(Rectangle((x,y),w,h,facecolor='#edf4f5',edgecolor='#285565',lw=1));ax.text(x+w/2,y+h/2,txt,ha='center',va='center',fontsize=8)
    for left,right in zip(nodes[:-1],nodes[1:]):ax.add_patch(FancyArrowPatch((left[0]+left[2],1.88),(right[0],1.88),arrowstyle='-|>',mutation_scale=11,color='#285565'))
    ax.text(5,2.9,'Forecasting and anomaly ranking answer different questions.',ha='center',fontsize=10,weight='bold')
    ax.text(5,.85,'Forecast error measures disagreement with the learned prediction.',ha='center',fontsize=8.8)
    ax.text(5,.45,'Persistence error measures movement away from the last observed pose.',ha='center',fontsize=8.8)
    ax.text(5,.05,'Excess error = forecast error minus persistence error. Larger values mean less forecast benefit.',ha='center',fontsize=8.8)
    base.save_figure(fig,'method')
    fig,ax=plt.subplots(figsize=(3.35,2.65))
    for i,key in enumerate(['excess_error','bilateral','persistence']):
        item=result['anomaly'][key];val=item['auroc'];lo,hi=item['auroc_ci95'];ax.errorbar(val,i,xerr=[[val-lo],[hi-val]],fmt='o',color='#186b60',capsize=3)
    item=audit['pooled'][1];val=item['auroc'];lo,hi=item['ci95'];ax.errorbar(val,3,xerr=[[val-lo],[hi-val]],fmt='s',color='#a76b26',capsize=3)
    ax.axvline(.5,color='#7d858a',lw=.8,linestyle='--');ax.set(yticks=range(4),yticklabels=['Excess error','Forecast error','Persistence error','Low motion*'],xlim=(0,1),xlabel='Source AUROC (higher is better)');ax.invert_yaxis();ax.grid(axis='x',alpha=.18);fig.tight_layout();base.save_figure(fig,'anomaly_scores')
    fig,ax=plt.subplots(figsize=(3.35,2.65))
    for name,color,marker in [('normal','#176d62','o'),('abnormal','#98552d','^')]:
        g=frame[frame.label.eq('normal') if name=='normal' else frame.label.ne('normal')];ax.scatter(g.persistence*1000,g.bilateral*1000,c=color,marker=marker,s=18,alpha=.8,label=name.capitalize())
    lim=max(frame.persistence.max(),frame.bilateral.max())*1000*1.05;ax.plot([0,lim],[0,lim],color='#7d858a',lw=.8,linestyle='--');ax.set(xlim=(0,lim),ylim=(0,lim),xlabel='Persistence error (MSE × 1,000)',ylabel='Forecast error (MSE × 1,000)');ax.legend(frameon=False,fontsize=8,loc='upper left');fig.tight_layout();base.save_figure(fig,'source_errors')


def prose(text):
    s=''.join(part if part.startswith('$') else base.esc(part.replace('–','--').replace('×',' x ')) for part in re.split(r'(\$[^$]*\$)',text))
    return re.sub(r'\[([a-z][a-z0-9]*(?:; [a-z][a-z0-9]*)*)\]',lambda m:r'\citep{'+m.group(1).replace('; ', ',')+'}',s)


def paper(evidence,frame,audit,args):
    result,prep,source,cohort,selection,data_audit,forecasts,contrasts,checks=evidence
    content=(OUT/'manuscript_source.md').read_text()
    def interval(x):return f"{x['estimate']:.3f} [{x['ci95'][0]:.3f}, {x['ci95'][1]:.3f}]"
    mapping={'paired_difference':interval(audit['excess_minus_low_motion']),'within_excess':f"{audit['within_fold_pair_weighted_auroc']['excess_auroc']:.3f}",'within_motion':f"{audit['within_fold_pair_weighted_auroc']['low_motion_auroc']:.3f}",'within_difference':interval(audit['within_fold_excess_minus_low_motion'])}
    for k,v in mapping.items():content=content.replace('{{'+k+'}}',v)
    assert '{{' not in content
    blocks=re.split(r'^## ',content,flags=re.M);title=blocks[0].strip().lstrip('# ')
    sections=[(b.split('\n',1)[0],b.split('\n',1)[1].strip()) for b in blocks[1:]]
    abstract=sections.pop(0)[1]
    tables={
      'cohort':(['Annotation','Sources','Sequences','Windows'],[[x['label'].capitalize(),x['sources'],x['sequences'],x['windows']] for x in result['cohort_by_label']]),
      'anomaly':(['Score','AUROC [95% interval]','AP'],[[base.SCORES[k],f"{v['auroc']:.3f} [{v['auroc_ci95'][0]:.3f}, {v['auroc_ci95'][1]:.3f}]",f"{v['average_precision']:.3f}"] for k,v in result['anomaly'].items()]),
      'forecast':(['Forecast','MSE × 1,000'],[[base.NAMES[k],f"{forecasts[k,8]['estimate']*1000:.3f}"] for k in ['persistence','damped_velocity','own','bilateral','shifted','static','bilateral_shrunk']])}
    captions={
      'method':'The workflow separates model fitting from held-out scoring. Both forecasts are compared with exactly the same observed future. The anomaly score becomes available only after that future is observed.',
      'anomaly_scores':'Ranking across 91 source videos (29 normal and 62 abnormal) depends on the reference. Bars show conditional 95% source intervals. The asterisk marks a post hoc diagnostic that assigns higher scores to less movement. It is not a validated clinical score.',
      'source_errors':'The 91 source videos comprise 29 normal and 62 abnormal sources. Axes show squared body-relative coordinates, normalized by prefix torso scale and multiplied by 1,000. Points below the equal-error line benefit from forecasting. Normal sources often have larger errors on both axes, explaining the direction of raw-error ranking.'}
    md=['# '+title,args.author,args.affiliation,'## Abstract',abstract]
    tex=[r'\documentclass[letterpaper]{article}',r'\usepackage[preprint]{aaai2027}',r'\usepackage[hyphens]{url}',r'\usepackage{graphicx}',r'\usepackage{natbib}',r'\usepackage{caption}',r'\usepackage{booktabs}',r'\usepackage{amsmath}',r'\frenchspacing',r'\urlstyle{rm}',r'\def\UrlFont{\rm}',r'\setcounter{secnumdepth}{0}','\\title{'+base.esc(title)+'}','\\author{'+base.esc(args.author)+'}','\\affiliations{'+base.esc(args.affiliation)+'}',r'\begin{document}',r'\maketitle',r'\begin{abstract}',prose(abstract),r'\end{abstract}']
    for heading,paragraphs in sections:
        md.extend(['## '+heading,paragraphs]);tex.extend(['\\section{'+heading+'}',prose(paragraphs)])
        if heading=='Introduction':
            md.extend(['![Forecast and anomaly workflow](figures/method.svg)',captions['method']]);tex.extend([r'\begin{figure*}[t]',r'\centering',r'\includegraphics[width=\textwidth]{figures/method.pdf}','\\caption{'+prose(captions['method'])+'}',r'\end{figure*}'])
        if heading=='Data and observation contract':
            md.append(base.md_table(*tables['cohort']));tex.append(base.table(*tables['cohort'],caption='The evaluated cohort groups observations by source video. Counts do not establish independent participants.'))
        if heading=='Results':
            md.append(base.md_table(*tables['forecast']))
            for name in ['anomaly_scores']:
                md.extend(['!['+name.replace('_',' ')+'](figures/'+name+'.svg)',captions[name]]);tex.extend([r'\begin{figure}[t]',r'\centering','\\includegraphics[width=\\columnwidth]{figures/'+name+'.pdf}','\\caption{'+prose(captions[name])+'}',r'\end{figure}'])
    tex.extend([r'\clearpage',r'\bibliography{references}',r'\clearpage',r'\appendix',r'\section{Complete exploratory results}','All original forecast arms and prespecified score directions are retained. Added diagnostics are explicitly post hoc.'])
    md.extend(['## Reproducibility appendix','All original forecast arms and prespecified score directions are retained. Added diagnostics are explicitly post hoc.'])
    md.extend(['![Errors by source](figures/source_errors.svg)',captions['source_errors']])
    tex.extend([r'\begin{center}',r'\begin{minipage}{\columnwidth}',r'\centering',r'\includegraphics[width=\columnwidth]{figures/source_errors.pdf}','\\captionof{figure}{'+prose(captions['source_errors'])+'}',r'\end{minipage}',r'\end{center}'])
    def appendix_table(*args,**kwargs):
        return base.table(*args,**kwargs).replace(r'\begin{table}[t]',r'\begin{center}\begin{minipage}{\columnwidth}').replace(r'\caption{',r'\captionof{table}{').replace(r'\end{table}',r'\end{minipage}\end{center}')
    for name in ['anomaly','forecast']:
        md.extend(['### '+name.capitalize(),base.md_table(*tables[name])]);tex.append(appendix_table(*tables[name],caption='All prespecified anomaly scores.' if name=='anomaly' else 'Normal-source forecast error over all eight future samples.'))
    for horizon in [4,8]:
        rows=[[base.NAMES[x['arm']],base.ci(x,1000,3)] for x in result['normal_forecast'] if x['horizon']==horizon]
        md.extend([f'### All forecasts through {horizon/10:.1f} seconds',base.md_table(['Forecast','MSE × 1,000 [95% interval]'],rows)]);tex.append(appendix_table(['Forecast','MSE [95% interval]'],rows,caption=f'All forecasts through {horizon/10:.1f} seconds. MSE is multiplied by 1,000.'))
        rows=[[base.SHORT[x['reference']]+' / '+base.SHORT[x['method']],base.ci(x,1000,3)] for x in result['paired_contrasts'] if x['horizon']==horizon]
        md.extend([f'### Paired contrasts through {horizon/10:.1f} seconds',base.md_table(['Reference / forecast','Gain × 1,000 [95% interval]'],rows)]);tex.append(appendix_table(['Reference / forecast','Gain [95% interval]'],rows,caption=f'Paired effects through {horizon/10:.1f} seconds. Positive values favor the forecast; values are multiplied by 1,000. S denotes shrinkage.'))
    fold_rows=[[str(x['fold']),str(x['normal'])+'/'+str(x['abnormal']),f"{x['excess_auroc']:.3f}",f"{x['low_motion_auroc']:.3f}"] for x in audit['within_fold']]
    cond_rows=[[x['label'].capitalize(),x['abnormal'],f"{x['excess_auroc']:.3f}",f"{x['low_motion_auroc']:.3f}"] for x in audit['condition_slices']]
    for title2,heads,rows,caption in [('Within-fold diagnostics',['Fold','N/A','Excess','Low motion'],fold_rows,'Post hoc within-fold AUROCs. N/A gives normal/abnormal source counts. Folds share training sources.'),('Condition diagnostics',['Annotation','Abnormal','Excess','Low motion'],cond_rows,'Post hoc AUROC by annotation versus the same 29 normal sources. Small groups and repeated normal comparisons limit interpretation.')]:
        md.extend(['### '+title2,base.md_table(heads,rows),caption]);tex.append(appendix_table(heads,rows,caption=caption))
    extra='Targets are BlazePose joints 25, 27, 29, and 31 on the left, and 26, 28, 30, and 32 on the right. Each leg requires at least four observed joint-time points through four future samples and eight through all eight samples. Each target joint also requires at least one prefix observation. This common support gate removes three of the 2,818 prepared windows for every model. The completed run contains 80 saved fold, leg, and variant checkpoints, some sharing fitted coefficients. There are 29 normal source groups, not 80 independent replications. Evidence tables, hashes, notebook records, and source generators accompany this manuscript. Figures and diagnostics are regenerated from frozen artifacts; the manuscript builder never refits a model.'
    md.extend(['### Artifact identity',extra,'### References','Citation keys resolve to references.bib. The PDF contains the formatted bibliography.']);tex.extend([r'\section{Reproduction}',prose(extra),r'\end{document}'])
    for txt,path in [('\n\n'.join(md)+'\n',OUT/'manuscript.md'),('\n\n'.join(tex)+'\n',OUT/'manuscript.tex')]:
        assert '\u2014' not in txt;path.write_text(txt)
    (OUT/'references.bib').write_text((PROJECT/'docs/draft01/references.bib').read_text()+'\n'+(OUT/'references_extra.bib').read_text())
    for name in ['all_forecasts.csv','paired_contrasts.csv','selection.csv','condition_results.csv']:shutil.copy2(PROJECT/'docs/draft01'/name,OUT/name)
    for name,(heads,rows) in tables.items():pd.DataFrame(rows,columns=heads).to_csv(OUT/(name+'.csv'),index=False)
    claims=[
    dict(claim='Bilateral forecasts reduce normal-source error by 33.5%.',status='supported exploratory finding',evidence='results.json paired_contrasts and source_scores.csv',boundary='Normal-source forecasting only; conditional uncertainty.'),
    dict(claim='Excess error AUROC is 0.793.',status='prespecified secondary association',evidence='results.json anomaly.excess_error',boundary='No clinical diagnosis or proof of value beyond movement alone.'),
    dict(claim='Low-movement diagnostic AUROC is 0.806.',status='completed post hoc diagnostic',evidence='posthoc_score_audit.json',boundary='Direction chosen after outcome inspection; not a changed frozen result.'),
    dict(claim='Learned anomaly scores improve on movement-only scores.',status='not established',evidence='posthoc_score_audit.json',boundary='No superiority or state-of-the-art claim.'),
    dict(claim='Bilateral temporal context is necessary.',status='not established',evidence='current-posture MSE 0.07164; own-leg difference interval includes zero',boundary='Timing control alone is insufficient.'),
    dict(claim='GAVD establishes clinical generalizability.',status='not established',evidence='data_audit.json; cohort.csv',boundary='Source IDs are not verified people; no external clinical cohort.')]
    base.write_json(OUT/'claim_ledger.json',claims)
    base.write_json(OUT/'evidence.json',dict(results=result,posthoc_audit=audit,checks=checks,inputs={str(p.relative_to(PROJECT)):base.digest(p) for p in [RUN/'results.json',RUN/'artifact_hashes.json',RUN/'source_scores.csv',RUN/'source_anomaly_scores.csv',PROJECT/'docs/restart/protocol.md',PROJECT/'scripts/build_restart_manuscript.py',OUT/'manuscript_source.md',OUT/'references_extra.bib']},generator_sha256=base.digest(__file__),scope='Standalone draft based on frozen exploratory evidence; no new trained experiment.'))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--compile',action='store_true');parser.add_argument('--render',action='store_true');parser.add_argument('--pdftex',default='/tmp/medwm-tools/TinyTeX/bin/universal-darwin/pdflatex');parser.add_argument('--render-python',default='/tmp/medwm-docs-env/bin/python');parser.add_argument('--author',default='Author details pending confirmation');parser.add_argument('--affiliation',default='Affiliation pending confirmation')
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True);evidence=base.load_evidence()
    for name in ['aaai2027.sty','aaai2027.bst','author_kit_provenance.json']:shutil.copy2(PROJECT/'docs/draft01'/name,OUT/name)
    frame,audit=score_audit();figures(frame,evidence[0],audit);paper(evidence,frame,audit,args);record=base.compile_and_render(args)
    record['draft03_generator_sha256']=base.digest(__file__);record['helper_generator_sha256']=base.digest(PROJECT/'scripts/build_restart_manuscript.py');record['prose_source_sha256']=base.digest(OUT/'manuscript_source.md');base.write_json(OUT/'build_record.json',record)
    print(json.dumps({'output':str(OUT),'audit':audit['excess_minus_low_motion'],'within':audit['within_fold_excess_minus_low_motion'],'pdf_checks':record.get('pdf_checks')}))

if __name__=='__main__':main()
