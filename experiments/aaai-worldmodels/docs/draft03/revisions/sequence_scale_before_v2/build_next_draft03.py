"""Publish the next standalone paper from immutable reference-v2 experiments.

The editable prose is docs/draft03/manuscript_source.md. This command never
calls a fitter and never writes draft01, draft02, or historical slide outputs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import joblib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parents[1]
sys.path.insert(0, str(PROJECT))
from worldmodels import v2_evaluation as ev
from worldmodels.v2_data import load_cohort
from worldmodels.v2_external import load_external
import build_restart_manuscript as publishing

OUT = PROJECT/'docs/draft03'
RUNS = PROJECT/'artifacts/runs'
PATHS = {'core': RUNS/'reference-v2-core-v1',
         'external': RUNS/'reference-v2-transfer-v1',
         'sensitivity': RUNS/'reference-v2-identity-sensitivity-v1',
         'robustness': RUNS/'reference-v2-robustness-v1',
         'mean_prior': RUNS/'reference-v2-mean-prior-posthoc-v1',
         'floor_sensitivity': RUNS/'reference-v2-floor-posthoc-v2'}
STAMP = datetime(2026, 11, 16, tzinfo=timezone.utc)
NAMES = {'persistence':'Persistence', 'damped_velocity':'Damped velocity',
         'own':'Own-leg history', 'bilateral':'Bilateral history',
         'shifted':'Shifted opposite history', 'static':'Current-posture transition (legacy)',
         'ridge_posture':'Current-posture transition',
         'ridge_posture_velocity':'Posture plus velocity', 'untrained_probe':'Random-feature ridge',
         'mlp_none':'No prefix masking', 'mlp_uniform':'Uniform masking',
         'mlp_time_blocks':'Time blocks', 'mlp_anatomical_tubes':'Anatomical tubes',
         'mlp_motion_balanced':'Motion-balanced masking', 'mlp_random_groups':'Random joint groups',
         'mlp_motion_focused':'Motion-focused masking', 'mlp_pattern_only':'Mask-pattern diagnostic'}
NAMES.update(mean_pose='Normal mean pose (post hoc)',mean_residual='Mean displacement (post hoc)')
NAMES.update(two_sided='Conditional two-sided',posture_two_sided='Posture two-sided',
             low_motion='Low motion',conditional_one_sided='Conditional one-sided',
             marginal_two_sided='Marginal two-sided',mse='Forecast MSE',
             excess_error='Forecast minus persistence',low_prefix_motion='Low prefix motion',
             prefix_missingness='Prefix missingness')
SCORES = {'calibrated_two_sided':'Conditional two-sided', 'posture_two_sided':'Posture two-sided',
          'low_motion':'Low motion', 'one_sided':'Conditional one-sided',
          'marginal_two_sided':'Marginal two-sided', 'posture_distance':'Static-posture distance',
          'motion_quality_distance':'Motion/quality distance',
          'negative_prefix_motion':'Low prefix motion', 'high_prefix_missingness':'Prefix missingness',
          'excess_bilateral':'Bilateral excess error',
          'excess_ridge_posture_velocity':'Posture/velocity excess error'}


def digest(path):
    return publishing.digest(path)


def write_json(path, item):
    def serial(x):
        if isinstance(x, np.generic): return x.item()
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, Path): return str(x)
        raise TypeError(type(x).__name__)
    Path(path).write_text(json.dumps(item, indent=2, allow_nan=False, default=serial)+'\n')


def verify_run(path):
    result = json.loads((path/'results.json').read_text())
    if result.get('status') != 'complete':
        raise ValueError('A completed experiment is required: '+str(path))
    hashes = json.loads((path/'artifact_hashes.json').read_text())
    for name, expected in hashes.items():
        if digest(path/name) != expected:
            raise ValueError('Immutable input changed: '+str(path/name))
    return result


def name(arm):
    for seed in (27,41,73):
        if arm.endswith('_seed'+str(seed)):
            return NAMES[arm.rsplit('_seed',1)[0]]+' (seed '+str(seed)+')'
    return NAMES.get(arm, arm.replace('_',' '))


def fmt(value, digits=3):
    return f'{value:.{digits}f}'


def interval(value, digits=3):
    return '['+', '.join(fmt(v,digits) for v in value)+']'


def save_figure(fig, filename):
    for suffix in ('pdf','svg','png'):
        metadata = ({'CreationDate':STAMP,'ModDate':STAMP} if suffix=='pdf'
                    else {'Date':STAMP.isoformat()} if suffix=='svg' else None)
        fig.savefig(OUT/'figures'/f'{filename}.{suffix}', bbox_inches='tight',
                    dpi=180, facecolor='white', metadata=metadata)
    plt.close(fig)


def figures(core, external, forecast, ranks, cohort):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'svg.fonttype':'none',
                         'svg.hashsalt':'medwm-reference-v2-draft03', 'pdf.fonttype':42,
                         'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(6.8,2.4));ax.set(xlim=(0,10),ylim=(0,3.3));ax.axis('off')
    boxes=[]
    def box(x,y,w,h,text,color):
        patch=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.06',
                            facecolor=color,edgecolor='#506475',linewidth=.7)
        ax.add_patch(patch)
        label=ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=8)
        boxes.append((patch,label))
    def arrow(a,b):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=10,color='#506475'))
    box(.05,1.4,1.8,1.2,'Video pose\n16 joints in 2D\nActual times','#e8f0f6')
    box(2.3,1.4,2.5,1.2,'16 past samples\nPosture or history\nTraining masks','#e6f2ec')
    box(5.25,1.4,2.1,1.2,'8 future samples\n0.1 through 0.8 s\nFit normal sources','#eae8f7')
    box(7.8,1.4,2.1,1.2,'Common targets\nReference errors\nScores by group','#fbefdb')
    for a,b in [((1.91,2),(2.24,2)),((4.86,2),(5.19,2)),((7.41,2),(7.74,2))]:arrow(a,b)
    for i in range(24):
        xx=.5+i*.38
        ax.scatter(xx,.7,c='#246b57' if i<16 else '#c47c26',s=22)
    ax.plot([6.4,6.4],[.42,.99],color='#52616b',lw=.8)
    ax.text(3.3,.16,'Visible prefix: 1.5 s between endpoints',ha='center',fontsize=8)
    ax.text(8.15,.16,'Withheld future queries',ha='center',fontsize=8)
    ax.text(5,3.03,'Forecasting skill and abnormality ranking are tested separately.',ha='center',weight='bold',fontsize=10)
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer();layout=[]
    for patch,label in boxes:
        outer=patch.get_window_extent(renderer);inner=label.get_window_extent(renderer)
        fits=inner.x0>=outer.x0+2 and inner.x1<=outer.x1-2 and inner.y0>=outer.y0+2 and inner.y1<=outer.y1-2
        layout.append({'label':label.get_text(),'fits_with_two_pixel_padding':bool(fits)})
    write_json(OUT/'figures/diagram_layout.json',{'status':'passed' if all(r['fits_with_two_pixel_padding'] for r in layout) else 'failed','boxes':layout})
    if not all(r['fits_with_two_pixel_padding'] for r in layout):raise ValueError('Workflow diagram label exceeds its container')
    save_figure(fig,'observation_task')

    x,valid,meta,times=load_cohort()
    # Choose by identity and observation completeness, never by model accuracy.
    target=[8,9,10,11,12,13,14,15]
    eligible=meta.label.eq('normal').to_numpy() & valid[:,16:,target].all((1,2))
    ids=meta.loc[eligible].sort_values('window_id')
    if not len(ids):raise ValueError('No complete normal illustration available')
    index=ids.index[0]
    with np.load(PATHS['core']/'refs_predictions.npz') as saved:
        prediction=saved['ridge_posture'][index]
        persistence=saved['persistence'][index]
    # BlazePose left ankle 27 is input index10 and target index1.
    tt=times[index]-times[index,15]
    fig,ax=plt.subplots(figsize=(3.35,2.3))
    observed=valid[index,:,10]
    ax.plot(tt[observed],x[index,observed,10,0],'-o',c='#263b4a',ms=2.4,lw=1,label='Observed ankle')
    ax.plot(tt[16:],persistence[:,1,0],'--',c='#a96b22',lw=1.5,label='Persistence')
    ax.plot(tt[16:],prediction[:,1,0],'-',c='#22775d',lw=1.5,label='Posture transition')
    ax.axvline(0,c='#80909b',lw=.8);ax.axvspan(0,tt[-1],color='#e9f1f6',alpha=.6)
    ax.set(xlabel='Time from latest prefix sample (s)',ylabel='Left ankle x (torso units)')
    ax.legend(frameon=False,fontsize=6.5,loc='best');fig.tight_layout();save_figure(fig,'forecast_example')
    write_json(OUT/'figures/forecast_example_selection.json',{
        'rule':'Lexicographically first normal window ID with all eight distal targets observed at every future query; accuracy was not used.',
        'window_id':meta.iloc[index].window_id,'source_id':meta.iloc[index].source_id,
        'sequence_id':meta.iloc[index].sequence_id,'input_index':int(index),'joint':'BlazePose27 left ankle, image x',
        'predictions':'reference-v2-core-v1/refs_predictions.npz:ridge_posture and persistence'})
    pd.DataFrame({'time_s':tt[16:],'observed_x':x[index,16:,10,0],
                  'posture_x':prediction[:,1,0],'persistence_x':persistence[:,1,0]}).to_csv(OUT/'figures/forecast_example_data.csv',index=False)

    arms=['mlp_none','mlp_uniform','mlp_time_blocks','mlp_anatomical_tubes',
          'mlp_random_groups','mlp_motion_balanced','mlp_motion_focused']
    fig,ax=plt.subplots(figsize=(3.35,2.55))
    for i,arm in enumerate(arms):
        item=forecast[arm];lo,hi=np.array(item['ci95'])*1000;mean=item['mse']*1000
        ax.errorbar(mean,i,xerr=[[mean-lo],[hi-mean]],fmt='o',ms=3.5,c='#306b86',capsize=2)
    ax.axvline(forecast['ridge_posture']['mse']*1000,c='#28745a',ls='--',lw=1,label='Posture transition')
    ax.set(yticks=range(len(arms)),yticklabels=[name(a) for a in arms],xlabel='Normal-source MSE x 1,000 (lower is better)')
    ax.invert_yaxis();ax.tick_params(axis='y',labelsize=7)
    ax.grid(axis='x',alpha=.18);fig.tight_layout();save_figure(fig,'masking_results')

    fig,axes=plt.subplots(1,2,figsize=(6.8,2.6),sharex=True)
    internal=core['primary_anomaly'];transfer=external['primary_anomaly']
    for ax,item,title in zip(axes,[internal,transfer],['GAVD: 91 source videos','External: 90 coded participants']):
        for i,(key,col) in enumerate([('candidate','#326c88'),('reference','#a56e2b')]):
            val=item[key+'_auroc'];low,high=item[key+'_ci95']
            ax.errorbar(val,i,xerr=[[val-low],[high-val]],fmt='o',color=col,capsize=3,ms=5)
        ax.axvline(.5,c='#89959d',ls=':',lw=1)
        ax.set(yticks=[0,1],yticklabels=['Conditional two-sided','Low motion'],xlim=(.2,1.0),
               xlabel='AUROC (higher is better)',title=title)
        ax.invert_yaxis();ax.grid(axis='x',alpha=.17);ax.tick_params(labelsize=8)
        ax.text(.5,-.35,'Paired difference '+fmt(item['estimate'])+' '+interval(item['ci95']),
                ha='center',transform=ax.transAxes,fontsize=7.3)
    fig.tight_layout();save_figure(fig,'ranking_comparison')

    scores=pd.read_csv(PATHS['core']/'source_scores.csv')
    fig,ax=plt.subplots(figsize=(3.35,2.5))
    for label,group in scores.groupby('label'):
        ax.scatter(group.persistence,group.bilateral,label=label,s=19,alpha=.75)
    limit=max(scores.persistence.max(),scores.bilateral.max())*1.05
    ax.plot([1e-4,limit],[1e-4,limit],c='#80919d',ls='--',lw=.8)
    ax.set(xscale='log',yscale='log',xlabel='Persistence MSE (log scale)',ylabel='Bilateral forecast MSE (log scale)')
    ax.legend(frameon=False,fontsize=6.3);fig.tight_layout();save_figure(fig,'motion_and_error')


def sampling_summary():
    rows={}
    for name,loader in [('gavd',load_cohort),('external',load_external)]:
        _,_,_,times=loader()
        error=times[:,16:]-(times[:,15,None]+np.arange(1,9)[None]*.1)
        duration=times[:,23]-times[:,15]
        rows[name]={'minimum_query_gap_s':float(np.diff(times,axis=1).min()),
                    'maximum_query_gap_s':float(np.diff(times,axis=1).max()),
                    'minimum_final_offset_s':float(duration.min()),
                    'maximum_final_offset_s':float(duration.max()),
                    'maximum_absolute_query_grid_deviation_s':float(abs(error).max()),
                    'median_absolute_query_grid_deviation_s':float(np.median(abs(error)))}
    write_json(OUT/'sampling_summary.json',{'identity':'reference-v2-native-timing-description-v1','observations':rows})
    return rows


def table_md(headers, rows):
    return publishing.md_table(headers, rows)


def table_tex(headers,rows,caption,label,wide=False):
    text=publishing.table(headers,rows,caption=prose(caption),label='tab:'+label)
    if not wide:
        old='l'+'r'*(len(headers)-1)
        new=r'>{\raggedright\arraybackslash}p{0.43\columnwidth}'+'r'*(len(headers)-1)
        text=text.replace(r'\begin{tabular}{'+old+'}',r'\setlength{\tabcolsep}{3pt}'+'\n'+r'\begin{tabular}{'+new+'}')
    text=text.replace(r'\small',r'\footnotesize')
    return text.replace('table}', 'table*}') if wide else text


def prose(text):
    text=text.replace('–','--').replace('×','x').replace('≥','at least ').replace('²',' squared')
    # Protect citations and inline code before ordinary TeX escaping.
    chunks=re.split(r'(\[@[^\]]+\]|`[^`]+`)',text)
    out=[]
    for chunk in chunks:
        if chunk.startswith('[@'):
            keys=[k.strip().lstrip('@') for k in chunk[1:-1].split(';')]
            out.append(r'\citep{'+','.join(keys)+'}')
        elif chunk.startswith('`'):
            out.append(r'\texttt{'+publishing.esc(chunk[1:-1])+'}')
        else:
            escaped=publishing.esc(chunk)
            escaped=re.sub(r'\*\*(.+?)\*\*',r'\\textbf{\1}',escaped)
            out.append(escaped)
    return ''.join(out)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compile',action='store_true');parser.add_argument('--render',action='store_true')
    parser.add_argument('--pdftex',default=shutil.which('pdflatex') or '/tmp/medwm-tools/TinyTeX/bin/universal-darwin/pdflatex')
    parser.add_argument('--render-python',default='/tmp/medwm-docs-env/bin/python')
    parser.add_argument('--author',default='AlexPose Research Team')
    parser.add_argument('--affiliation',default='Author names and affiliations to be confirmed before submission')
    args=parser.parse_args()
    OUT.mkdir(exist_ok=True);(OUT/'figures').mkdir(exist_ok=True)
    results={key:verify_run(path) for key,path in PATHS.items()}
    core,external=results['core'],results['external']
    if core.get('test_only') or core['sources']!=91 or external['participants']!=90:
        raise ValueError('This manuscript identity requires the declared complete real populations')
    forecasts={r['arm']:r for r in core['normal_forecast'] if r['horizon']==8}
    for row in results['mean_prior']['comparisons']:
        if row['reference']=='persistence':
            forecasts[row['method']]={'mse':row['method_mse'],'ci95':row['method_ci95'],'normal_sources':29}
    contrasts={(r['reference'],r['method']):r for r in core['forecast_contrasts']}
    ranks={r['score']:r for r in core['rankings']}
    cohort=pd.read_csv(PATHS['core']/'cohort.csv')
    source=pd.read_csv(PATHS['core']/'source_scores.csv')
    # Independently reconstruct every displayed normal MSE from the source CSV.
    for arm,row in forecasts.items():
        if arm in source:
            np.testing.assert_allclose(source.loc[source.label.eq('normal'),arm].mean(),row['mse'],atol=1e-12)
    np.testing.assert_allclose(roc_auc_score(source.label.ne('normal'),source.low_motion),ranks['low_motion']['pooled']['auroc'],atol=1e-12)
    figures(core,external,forecasts,ranks,cohort)
    timing=sampling_summary()
    gain=contrasts['persistence','ridge_posture'];pv=contrasts['ridge_posture','ridge_posture_velocity']
    primary,transfer=core['primary_anomaly'],external['primary_anomaly']
    sensitivity=results['sensitivity']['primary_anomaly']
    prior=next(r for r in results['mean_prior']['comparisons']
               if r['method']=='mean_pose' and r['reference']=='ridge_posture')
    vals={
        'posture_mse':fmt(forecasts['ridge_posture']['mse'],5),
        'persistence_mse':fmt(forecasts['persistence']['mse'],5),
        'posture_gain':fmt(gain['estimate'],5),'posture_gain_ci':interval(gain['ci95'],5),
        'posture_relative':fmt(gain['relative_reduction']*100,1),
        'pv_mse':fmt(forecasts['ridge_posture_velocity']['mse'],5),
        'pv_posture_gain':fmt(pv['estimate'],5),'pv_posture_ci':interval(pv['ci95'],5),
        'internal_auroc':fmt(primary['candidate_auroc']),'internal_low':fmt(primary['reference_auroc']),
        'internal_difference':fmt(primary['estimate']),'internal_ci':interval(primary['ci95']),
        'external_auroc':fmt(transfer['candidate_auroc']),'external_low':fmt(transfer['reference_auroc']),
        'external_difference':fmt(transfer['estimate']),'external_ci':interval(transfer['ci95']),
        'sensitivity_auroc':fmt(sensitivity['candidate_auroc']),'sensitivity_low':fmt(sensitivity['reference_auroc']),
        'sensitivity_difference':fmt(sensitivity['estimate']),'sensitivity_ci':interval(sensitivity['ci95']),
        'uniform_mse':fmt(forecasts['mlp_uniform']['mse'],5),
        'tubes_mse':fmt(forecasts['mlp_anatomical_tubes']['mse'],5),
        'balanced_mse':fmt(forecasts['mlp_motion_balanced']['mse'],5),
        'tubes_difference':fmt(contrasts['mlp_uniform','mlp_anatomical_tubes']['estimate'],5),
        'tubes_ci':interval(contrasts['mlp_uniform','mlp_anatomical_tubes']['ci95'],5),
        'balanced_difference':fmt(contrasts['mlp_uniform','mlp_motion_balanced']['estimate'],5),
        'balanced_ci':interval(contrasts['mlp_uniform','mlp_motion_balanced']['ci95'],5),
        'random_tube_difference':fmt(contrasts['mlp_random_groups','mlp_anatomical_tubes']['estimate'],5),
        'random_tube_ci':interval(contrasts['mlp_random_groups','mlp_anatomical_tubes']['ci95'],5),
        'mean_pose_mse':fmt(forecasts['mean_pose']['mse'],5),
        'mean_residual_mse':fmt(forecasts['mean_residual']['mse'],5),
        'posture_mean_gain':fmt(-prior['estimate'],5),
        'posture_mean_ci':interval([-prior['ci95'][1],-prior['ci95'][0]],5),
        'gavd_offset_min':fmt(timing['gavd']['minimum_final_offset_s'],3),
        'gavd_offset_max':fmt(timing['gavd']['maximum_final_offset_s'],3),
        'gavd_query_deviation':fmt(timing['gavd']['maximum_absolute_query_grid_deviation_s'],4),
        'floor_unchanged_sources':str(results['floor_sensitivity']['sources']),
        'core_seconds':fmt(core['seconds_this_invocation'],1),'fits':str(core['fits'])}
    source_text=(OUT/'manuscript_source.md').read_text()
    for key,value in vals.items():source_text=source_text.replace('{{'+key+'}}',value)
    captions={
        'observation_task':'The benchmark asks two separate questions: can a model predict future articulation, and does its error identify departures beyond simple references? Each GAVD window contains sixteen past and eight future observations at nominal 10 Hz. Methods use the same evaluation windows and future target support, with their declared past inputs. Future availability never conditions a prediction.',
        'forecast_example':'A real normal-window example compares the learned current-posture transition and persistence on the same left-ankle x coordinate. Time zero is the latest prefix sample; the shaded area contains future queries. Coordinates use prefix-torso units. The example is the first eligible window in identifier order, selected without using forecast accuracy; it illustrates the task rather than population performance.',
        'masking_results':'Forecasting with six prefix-corruption policies and an unmasked prefix baseline on the same 29 held-out normal source videos. Points are source-average MSE multiplied by 1,000; lines are conditional 95% source-bootstrap intervals. Predictions are averaged over three fixed seeds before errors are computed. The dashed line is the compact current-posture transition. Primary evaluation retains the full observed prefix. Structured masks do not improve on uniform masking under this contract.',
        'ranking_comparison':'The locked posture-plus-velocity forecaster supplies the conditional two-sided anomaly score. Its smaller normal forecast-development fits and disjoint calibration groups differ from the full-pool forecasts in Table 2. The comparator is negative persistence error, which assigns higher abnormality to smaller realized displacement over the same observed future segment. The left panel shows pair-count-weighted within-fold AUROC on 91 GAVD sources. The right panel uses all 90 external coded participants without target-domain fitting. Lines are conditional 95% paired group-bootstrap intervals; differences compare the same resamples.',
        'motion_and_error':'Source-average persistence and bilateral-forecast MSE for all 91 GAVD source videos. Each point is one source; colors identify dataset annotations. The dashed line marks equal error. Both axes use logarithmic scales in squared prefix-torso units. The scatter motivates explicit movement controls and does not identify a physiological mechanism.'}
    cohort_rows=[]
    for label,g in cohort.groupby('label'):
        cohort_rows.append([label,str(g.source_id.nunique()),str(g.sequence_id.nunique()),str(len(g))])
    external_rows=[['normal','27','49','84'],['osteoarthritis','49','92','483'],['parkinsons','14','27','206']]
    forecast_arms=['persistence','damped_velocity','own','bilateral','static','ridge_posture','ridge_posture_velocity','untrained_probe','mean_pose','mean_residual']
    forecast_rows=[[name(a),fmt(forecasts[a]['mse']*1000,2),interval(np.array(forecasts[a]['ci95'])*1000,2)] for a in forecast_arms]
    anomaly_arms=['calibrated_two_sided','posture_two_sided','low_motion','one_sided','marginal_two_sided',
                  'posture_distance','motion_quality_distance','negative_prefix_motion','high_prefix_missingness',
                  'excess_bilateral','excess_ridge_posture_velocity']
    anomaly_rows=[[SCORES[a],fmt(ranks[a]['within_fold']['pair_weighted_auroc']),
                   fmt(ranks[a]['pooled']['auroc']),fmt(ranks[a]['pooled']['average_precision'])] for a in anomaly_arms]
    external_rows_scores=[[name(a),fmt(r['auroc']),interval(r['auroc_ci95']),fmt(r['average_precision'])]
                          for a,r in external['anomaly_scores'].items()]
    complete_forecast=[[name(a),fmt(row['mse']*1000,2),interval(np.array(row['ci95'])*1000,2)] for a,row in forecasts.items()]
    tables={
        'cohort':(['GAVD annotation','Sources','Sequences','Windows'],cohort_rows,'The internal model cohort after common observation gates. Source counts are video identifiers, not independently verified patients.',False),
        'external_cohort':(['External annotation','People','Recordings','Windows'],external_rows,'The external model cohort uses dataset-coded participants. Repeated recordings are averaged within participant before calibration.',False),
        'forecasts':(['Forecast','MSE x1,000','95% interval'],forecast_rows,'Normal-source forecasting over the eight nominal future offsets, 0.1 through 0.8 seconds. Lower MSE is better. Units are squared prefix-torso units; values and conditional source intervals are multiplied by 1,000. The separately implemented legacy and current posture transitions are both retained.',False),
        'rankings':(['Anomaly score','Within-fold','Pooled','Pooled AP'],anomaly_rows,'Internal rankings on 29 normal and 62 abnormal GAVD sources. Higher values are better. Within-fold AUROC weights each fold by its number of normal-abnormal pairs; pooled AUROC and average precision (AP) combine all sources. Abnormal prevalence is 0.681. Scores use different reference mechanisms as explained in the text.',False),
        'external_rankings':(['External score','AUROC','95% interval','AP'],external_rows_scores,'Every declared external score on 27 normal and 63 abnormal coded participants. Higher AUROC and AP are better. These marginal intervals use 2,000 participant bootstrap draws within label strata and condition on fixed GAVD models. The principal paired contrast uses 5,000 draws. Abnormal prevalence is 0.700.',True),
        'complete_forecasts':(['Every declared forecast','MSE x1,000','95% interval'],complete_forecast,'Complete candidate registry at the primary eight-query horizon, including all optimization seeds and the mask-pattern diagnostic. Lower values are better. Seeds do not multiply the 29 normal source groups.',True)}
    equations={
        'excess':(r'\begin{equation}E_f-E_p=\lVert d\rVert^2-2\langle d,y-a\rangle.\end{equation}',
                  'E_f - E_p = ||d||² - 2 <d, y - a>.'),
        'calibration':(r'\begin{equation}r_s=\log(E_{f,s}+\epsilon)-g(z_s),\quad A_s=|2F_{cal}(r_s)-1|.\end{equation}',
                       'r_s = log(E_f,s + epsilon) - g(z_s); A_s = abs(2 F_cal(r_s) - 1).')}
    # Build paired formats from the same prose and evidence tokens.
    tex=[r'\documentclass[letterpaper]{article}',r'\usepackage[preprint]{aaai2027}',
         r'\usepackage[hyphens]{url}',r'\usepackage{graphicx}',r'\usepackage{natbib}',
         r'\usepackage{caption}',r'\usepackage{booktabs}',r'\usepackage{amsmath}',r'\usepackage{array}',
         r'\frenchspacing',r'\urlstyle{rm}',r'\def\UrlFont{\rm}',r'\setcounter{secnumdepth}{0}']
    blocks=source_text.strip().split('\n\n');md=[];begun=False;abstract=False
    for block in blocks:
        if block.startswith('# '):
            tex += [r'\title{'+prose(block[2:])+'}',r'\author{'+prose(args.author)+'}',
                    r'\affiliations{'+prose(args.affiliation)+'}',r'\begin{document}',r'\maketitle'];begun=True;md.append(block)
        elif block=='## Abstract':tex.append(r'\begin{abstract}');abstract=True;md.append(block)
        elif block=='<!-- APPENDIX -->':
            tex += [r'\bibliography{references}',r'\clearpage',r'\appendix'];md += ['## References','See references.bib and the formatted bibliography in the PDF.','## Supplementary material']
        elif block.startswith('## '):
            if abstract:tex.append(r'\end{abstract}');abstract=False
            tex.append(r'\section{'+prose(block[3:])+'}');md.append(block)
        elif block.startswith('### '):tex.append(r'\subsection{'+prose(block[4:])+'}');md.append(block)
        elif re.fullmatch(r'\{\{figure:[a-z_]+\}\}',block):
            key=block[9:-2];wide=key in ['observation_task','ranking_comparison']
            env='figure*' if wide else 'figure'
            width=r'\textwidth' if wide else r'\columnwidth'
            tex += [r'\begin{'+env+'}[t!]',r'\centering',r'\includegraphics[width='+width+']{figures/'+key+'.pdf}',r'\caption{'+prose(captions[key])+'}',r'\label{fig:'+key+'}',r'\end{'+env+'}']
            md.append('!['+captions[key]+'](figures/'+key+'.svg)\n\n'+captions[key])
        elif re.fullmatch(r'\{\{table:[a-z_]+\}\}',block):
            key=block[8:-2];headers,rows,caption,wide=tables[key]
            tex.append(table_tex(headers,rows,caption,key,wide));md.append(table_md(headers,rows)+'\n\n'+caption)
        elif re.fullmatch(r'\{\{equation:[a-z_]+\}\}',block):
            key=block[11:-2];tex.append(equations[key][0]);md.append(equations[key][1])
        else:tex.append(prose(block.replace('\n',' ')));md.append(block)
    if not begun or abstract:raise ValueError('Invalid manuscript section structure')
    tex.append(r'\end{document}')
    text='\n\n'.join(tex)+'\n';markdown='\n\n'.join(md)+'\n'
    if '{{' in text or '{{' in markdown or '\u2014' in text or '\u2014' in markdown:
        raise ValueError('Unresolved source token or em dash')
    (OUT/'manuscript.tex').write_text(text);(OUT/'manuscript.md').write_text(markdown)
    for key,(headers,rows,_,_) in tables.items():pd.DataFrame(rows,columns=headers).to_csv(OUT/(key+'.csv'),index=False)
    for directory,names in [(PATHS['core'],['source_scores.csv','source_anomaly_scores.csv','condition_scores.csv','retraining_source_scores.csv','retraining_summary.json','refs_source_scores.csv','masks_source_scores.csv','calibrated_source_scores.csv']),
                            (PATHS['external'],['participant_scores.csv','participant_component_scores.csv','covariate_shift.json']),
                            (PATHS['robustness'],['source_stress_scores.csv','nuisance_matched_pairs.csv','matched_source_scores.csv'])]:
        for file in names:
            if (directory/file).exists():shutil.copy2(directory/file,OUT/file)
    shutil.copy2(PATHS['mean_prior']/'source_scores.csv',OUT/'mean_prior_source_scores.csv')
    shutil.copy2(PATHS['floor_sensitivity']/'results.json',OUT/'floor_sensitivity_results.json')
    for file in ['source_components.csv','fold_counts.csv','relation_graph.json']:
        shutil.copy2(RUNS/'reference-v2-cohort-v1'/file,OUT/file)
    exposure=[];health=[]
    for item in json.loads((PATHS['core']/'model_fit_ledger.json').read_text()).values():
        roles=[r for r in item['roles'] if re.fullmatch(r'outer\d+:mask_[a-z_]+:seed\d+',r)]
        if not roles:continue
        checkpoint=joblib.load(PATHS['core']/item['checkpoint'])
        config=checkpoint['config']
        for role in roles:
            health.append(dict(role=role,policy=config['mask_policy'],seed=config['seed'],
                training_sources=len(checkpoint['training_source_ids']),training_windows=checkpoint['training_windows'],
                steps=config['steps'],parameters=checkpoint['parameter_count'],seconds=checkpoint['seconds'],
                checkpoint_sha256=item['checkpoint_sha256'],**checkpoint['diagnostics']))
            for bank in checkpoint['mask_audit']:
                exposure.append(dict(role=role,policy=config['mask_policy'],seed=config['seed'],
                    realized_rate=bank['hidden_total']/bank['eligible_total'] if bank['eligible_total'] else 0.,**bank))
    pd.DataFrame(exposure).to_csv(OUT/'mask_exposure.csv',index=False)
    pd.DataFrame(health).to_csv(OUT/'model_health.csv',index=False)
    claims=[
        {'id':'C1','claim':'A compact current-posture transition reduces normal-source forecast error relative to persistence.',
         'artifact':'reference-v2-core-v1/results.json:forecast_contrasts[persistence,ridge_posture]',
         'estimate':gain['estimate'],'ci95':gain['ci95'],'relative_reduction':gain['relative_reduction'],
         'scope':'29 normal GAVD sources; fixed source-grouped fits; relative projected articulation.',
         'post_hoc_prior_control':'reference-v2-mean-prior-posthoc-v1/results.json:comparisons; normal mean pose and mean displacement are both retained.'},
        {'id':'C2','claim':'The tested structured masks do not improve on uniform masking under the fixed coordinate objective.',
         'artifact':'reference-v2-core-v1/results.json:forecast_contrasts',
         'contrasts':[contrasts['mlp_uniform','mlp_anatomical_tubes'],contrasts['mlp_uniform','mlp_motion_balanced']],
         'scope':'One compact model,40% eligible-token masking,three seeds; exploratory comparisons.'},
        {'id':'C3','claim':'The locked conditional two-sided score does not improve on the low-motion baseline internally or in frozen external transfer.',
         'artifact':'reference-v2-core-v1/results.json:primary_anomaly; reference-v2-transfer-v1/results.json:primary_anomaly',
         'internal':primary,'external':transfer,
         'scope':'GAVD source groups and separate external dataset-coded participants; no clinical diagnosis or causal claim.'},
        {'id':'C4','claim':'The benchmark makes forecast utility, anomaly ranking, and recording-domain limitations separately auditable.',
         'artifact':'Protocol freeze,complete candidate registry,clean-kernel notebooks,independent reconstruction and vector figures.',
         'scope':'Reproducibility and evaluation contribution; no claim that masking or normal-only forecasting was invented here.'}]
    write_json(OUT/'claim_ledger.json',{'identity':'medwm-reference-v2-draft03-evidence-v1','claims':claims})
    inputs={str((path/file).relative_to(PROJECT)):digest(path/file)
            for path in PATHS.values() for file in ['results.json','artifact_hashes.json']}
    for path in [PROJECT/'docs/reference_v2/protocol_freeze.json',PROJECT/'docs/reference_v2/protocol.md',
                 PROJECT/'docs/reference_v2/data_audit.json',PROJECT/'docs/reference_v2/data_external_preparation_audit.json',
                 PROJECT/'docs/reference_v2/external_identity_sensitivity_protocol.json',OUT/'references.bib',OUT/'manuscript_source.md']:
        inputs[str(path.relative_to(PROJECT))]=digest(path)
    write_json(OUT/'evidence.json',{
        'identity':'medwm-reference-v2-draft03-evidence-v1','status':'completed_measured_inputs_only',
        'method_identity':'medwm-reference-masking-v2','run_identities':{k:str(v.relative_to(PROJECT)) for k,v in PATHS.items()},
        'code_revision':json.loads((PATHS['core']/'request.json').read_text())['code_revision'],
        'generator_sha256':digest(__file__),'inputs':inputs,'displayed_values':vals,
        'counts':{'gavd_windows':2815,'gavd_sequences':456,'gavd_sources':91,'gavd_normal_sources':29,
                  'external_windows':773,'external_recordings':168,'external_coded_participants':90},
        'checks':['All completed input-run manifests match actual artifact bytes.',
                  'Displayed normal MSEs independently recomputed from source rows.',
                  'Pooled low-motion ranking independently recomputed.',
                  'Every declared candidate and seed retained in the exported registry.'],
        'uncertainty':'Paired independent-group bootstrap,conditional on saved fits. Unknown GAVD participant overlap and full-pipeline retraining variation are excluded.',
        'boundary':'All reused GAVD outcomes remain exploratory. External coded identity and acquisition differences limit clinical generalization.'})
    for file in ['aaai2027.sty','aaai2027.bst']:
        shutil.copy2(PROJECT/'docs/draft02'/file,OUT/file)
    kit=PROJECT/'docs/draft02/author_kit_provenance.json'
    if kit.exists():shutil.copy2(kit,OUT/kit.name)
    publishing.OUT=OUT
    record=publishing.compile_and_render(args)
    record['generator_sha256']=digest(__file__)
    record['publishing_helper_sha256']=digest(publishing.__file__)
    record['editable_source_sha256']=digest(OUT/'manuscript_source.md')
    if 'compiler_distribution' in record:
        record['compiler_distribution']['additional_packages'].append('courier')
    if args.render:
        record['pdf_checks']['appendix_start_pages']=[int(p.stem.split('-')[1])
            for p in (OUT/'rendered').glob('page-*.txt')
            if 'Complete candidates and implementation' in p.read_text()]
        write_json(OUT/'rendered/checks.json',record['pdf_checks'])
    write_json(OUT/'build_record.json',record)
    write_json(OUT/'publication_manifest.json',{
        'identity':'medwm-reference-v2-draft03-publication-v1',
        'source':'manuscript_source.md','build_record_sha256':digest(OUT/'build_record.json'),
        'artifacts':{str(p.relative_to(OUT)):digest(p) for p in OUT.rglob('*')
                     if p.is_file() and p.suffix in ['.md','.tex','.bib','.pdf','.svg','.csv','.json']
                     and p.name not in ['publication_manifest.json','final_acceptance.json']
                     and 'rendered' not in p.parts},
        'review_status':'Requires independent acceptance of these exact final hashes.'})
    print(json.dumps({'output':str(OUT.relative_to(PROJECT)), 'compiled':record['compiled'],
                      'rendered':record['rendered'],'claim_count':len(claims)},indent=2))


if __name__=='__main__':main()
