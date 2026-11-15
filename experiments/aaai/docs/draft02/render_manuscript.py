"""Resolve manuscript numerical fields exclusively from saved evidence."""
from pathlib import Path
import json,re
import pandas as pd
B=Path(__file__).resolve().parent;A=B.parents[1]
e=json.loads((B/'evidence.json').read_text())
r=A/'artifacts/iteration02'/e['protocol_hash']
def num(v):return f'{v:.3f}'
def ci(v):return '['+', '.join(num(x) for x in v)+']'
g=e['primary_difference'];b=e['broad_difference'];ex=e['external'];eg=ex['primary_difference'];match=ex['age_sex_sensitivity']
tokens={
'G_M':num(e['main']['focal']['M']['weighted_auroc']),'G_S':num(e['main']['focal']['S']['weighted_auroc']),
'G_DIFF':num(g['observed']),'G_CI':ci(g['conditional_95_interval']),
'B_DIFF':num(b['observed']),'B_CI':ci(b['conditional_95_interval']),
'E_M':num(ex['metrics']['M']['weighted_auroc']),'E_S':num(ex['metrics']['S']['weighted_auroc']),
'E_DIFF':num(eg['observed']),'E_CI':ci(eg['conditional_95_interval']),'E_R':num(ex['metrics']['R']['weighted_auroc']),
'MATCH_M':num(match['metrics']['M']['weighted_auroc']),'MATCH_S':num(match['metrics']['S']['weighted_auroc']),
'MATCH_DIFF':num(match['S_minus_M']),'MATCH_CI':ci(match['conditional_95_interval']),
'AP_S':num(e['main']['focal']['S']['equal_fold_average_precision']),'AP_M':num(e['main']['focal']['M']['equal_fold_average_precision']),
'PREVALENCE':', '.join(num(f['prevalence']) for f in e['main']['focal']['S']['folds']),
'ELEG_VALUES':', '.join(num(e['all_mask_metrics'][p+'/E_leg/focal']['S']['weighted_auroc']) for p in ['U','J','A'])}
reg=pd.read_csv(r/'source_registry.csv');meta=pd.read_csv(A/'artifacts/iteration02/preparation/cohort_v2/window_index.csv')
tokens['FOCAL_WINDOWS']=f"{int(meta.source_id.isin(reg.loc[reg.is_focal,'source_id']).sum()):,}"
for key,method in [('RF_F1','random_forest'),('JEPA_F1','learned_jepa_probe'),('RANDOM_F1','random_encoder_probe')]:
    tokens[key]=num(pd.DataFrame(e['supervised']).query('method==@method').macro_f1.mean())
names=[('M','Local movement M'),('R','Relational departure R'),('S','Fixed S, U training'),('motion36','Richer motion, 36 features'),('S_sham','M + own-capacity sham'),('S_constituent','M + constituent error'),('S_base_error','M + baseline error'),('timing8','Timing/waveform distance'),('isolation_forest','Isolation Forest'),('stgnf_adapted','STG-NF, adapted'),('nuisance_distance','Availability and viewpoint'),('nuisance_supervised','Supervised nuisance')]
rows=[f"{label} & {num(e['main']['focal'][k]['weighted_auroc'])} & {num(e['main']['broad'][k]['weighted_auroc'])} \\\\" for k,label in names]
for policy in ['J','A']:
    rows.append(f"Fixed S, {policy} training & {num(e['all_mask_metrics'][policy+'/U/focal']['S']['weighted_auroc'])} & {num(e['all_mask_metrics'][policy+'/U/broad']['S']['weighted_auroc'])} \\\\")
tokens['DETECTION_ROWS']='\n'.join(rows)
text=(B/'manuscript_template.md').read_text()
for name,value in tokens.items():text=text.replace('@@'+name+'@@',value)
assert not re.search(r'@@[A-Z_]+@@',text)
text=text.replace('−', '-')
(B/'manuscript.md').write_text(text)
(B/'numerical_fields.json').write_text(json.dumps(tokens,indent=2)+'\n')
print(B/'manuscript.md')
