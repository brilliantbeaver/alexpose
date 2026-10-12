"""Generate all displayed numerical inventories directly from audited exports."""
from pathlib import Path
import pandas as pd
HERE=Path(__file__).resolve().parents[1]
E=HERE/'evidence';OUT=HERE/'tables';OUT.mkdir(exist_ok=True)
FAMILIES=[('P-direct-none','Direct'),('I-initialized-none','Initialized'),('I-shuffled_jepa-graph_time','Shuffled JEPA'),('M-coordinate-graph_time','Coordinate'),('M-paired_jepa-graph_time','Core JEPA'),('F-response-coordinate_delta_v1-graph_time','Coordinate delta'),('F-response-jepa_endpoint_v1-graph_time','Endpoint JEPA'),('F-response-jepa_delta_v1-graph_time','Delta JEPA')]
ORDER=[(f+'-'+o,n+' / '+s) for f,n in FAMILIES for o,s in [('base','coordinate'),('paired_change','change')]]
def write(name,cols,head,rows):
    s='\\begin{tabular}{'+cols+'}\\toprule\n'+head+'\\\\\\midrule\n'
    s+='\n'.join(' & '.join(r)+r' \\' for r in rows)
    s+='\n\\bottomrule\\end{tabular}\n';(OUT/name).write_text(s)
def f(x,n=2): return f'{x:.{n}f}'
def ci(row):return f'{row.estimate:.2f} [{row.ci_low:.2f}, {row.ci_high:.2f}]'
m=pd.read_csv(E/'method_summary.csv').set_index('method')
write('inventory.tex','lrrrrr','Method / objective & Response & Waveform & NLE & Fail (\\%) & Assign (\\%)',[[n,f(m.loc[k,'response_error']),f(m.loc[k,'waveform_error']),f(m.loc[k,'synthetic_all_nle'],4),f(m.loc[k,'failure_percent'],3),f(100*m.loc[k,'assignment_failure_rate'])] for k,n in ORDER])
write('failure-inventory.tex','lrrrrr','Method / objective & Fail (\\%) & Success & Cost & Total & Conditional',[[n,f(m.loc[k,'failure_percent'],3),f(m.loc[k,'response_success_contribution'],3),f(m.loc[k,'response_failure_contribution'],3),f(m.loc[k,'response_error'],3),f(m.loc[k,'conditional_error_under_original_weights'],3)] for k,n in ORDER])
c=pd.read_csv(E/'condition_summary.csv');e=pd.read_csv(E/'condition_effects.csv')
keys=['P-direct-none-base','F-response-jepa_endpoint_v1-graph_time-paired_change','F-response-jepa_delta_v1-graph_time-paired_change']
def value(st,group,k,metric='response_error'):
    return c[c.stratum_type.eq(st)&c.stratum.eq(group)&c.method.eq(k)&c.metric.eq(metric)].iloc[0].estimate
rows=[]
groups=[('clear | False','Clear, 5/10$^\\circ$'),('clear | True','Clear, 15$^\\circ$'),('occluded | False','Occluded, 5/10$^\\circ$'),('occluded | True','Occluded, 15$^\\circ$')]
for group,label in groups:
    r=e[e.stratum_type.eq('observation_by_intervention')&e.stratum.eq(group)&e.candidate.eq(keys[0])&e.comparator.eq('zero_response')&e.metric.eq('response_error')].iloc[0]
    rows.append([label,f(value('observation_by_intervention',group,keys[0],'zero_response_error'))]+[f(value('observation_by_intervention',group,k)) for k in keys]+[ci(r)])
write('strata-main.tex','lrrrrl','Observation, edit & Zero & Direct & Endpt. & Delta & Zero $-$ direct [95\\% CI]',rows)
for st,groups2,filename in [
 ('observation_by_intervention',[g for g,l in groups],'strata-levels.tex'),
 ('observation_by_extractor',['clear | rtmpose_m','clear | hrnet_w32','clear | vitpose_base','occluded | rtmpose_m','occluded | hrnet_w32','occluded | vitpose_base'],'strata-extractors.tex')]:
    headers=['Clear 5/10','Clear 15','Occl. 5/10','Occl. 15'] if 'intervention' in st else ['Clear R','Clear H','Clear V','Occl. R','Occl. H','Occl. V']
    rows=[['Zero response']+[f(value(st,g,keys[0],'zero_response_error')) for g in groups2]]
    rows += [[n]+[f(value(st,g,k)) for g in groups2] for k,n in ORDER]
    write(filename,'l'+'r'*len(groups2),'Method / objective & '+' & '.join(headers),rows)
p=pd.read_csv(E/'penalty_sensitivity.csv')
rows=[['Zero response']+[f(m.zero_response_error.iloc[0],3)]*4]
for k,n in ORDER:
 rows.append([n]+[f(p[p.method.eq(k)&p.failure_penalty_deg.eq(cost)].iloc[0].estimate,3) for cost in [0,180,360,720]])
write('penalties.tex','lrrrr','Method / objective & Cost 0 & Cost 180 & Cost 360 & Cost 720',rows)
rep=pd.read_csv(E/'repair_means.csv');rows=[]
for k in rep.method.unique():
 if k.startswith('P-'): label='Direct / coordinate (joint)'
 else:
  label=('Delta' if 'jepa_delta' in k else 'Endpoint')+' / '+('coordinate' if k.endswith('-base') else 'original change' if k.endswith('-paired_change') else 'low scalar' if k.endswith('-scalar_low') else 'dense')
 vals=[]
 for metric in ['waveform_error','response_error','synthetic_all_nle','response_failure_rate']:
  r=rep[rep.method.eq(k)&rep.metric.eq(metric)].iloc[0]
  vals.append(f(r.estimate*100 if metric.endswith('rate') else r.estimate,4 if metric=='synthetic_all_nle' else 3 if metric.endswith('rate') else 2))
 rows.append([label]+vals)
write('repair-inventory.tex','lrrrr','Method / supervision & Waveform & Response & NLE & Response fail (\\%)',rows)
print('Wrote',len(list(OUT.glob('*.tex'))),'data-derived tables.')
