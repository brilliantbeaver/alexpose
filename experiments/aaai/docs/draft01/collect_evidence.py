"""Derive manuscript tables from frozen real runs without fitting a model."""
from pathlib import Path
import json,hashlib,sys
import numpy as np
import pandas as pd
BASE=Path(__file__).resolve().parent;EXP=BASE.parents[1]
sys.path.insert(0,str(EXP))
from aaai_gait.predictive_profile import fold_ranking_metrics,paired_fold_bootstrap
from aaai_gait.study_runner import dump_json
PRIMARY=EXP/'artifacts/context_study/gavd/3044566091261819'
LONGER=EXP/'artifacts/context_study/gavd/eba2bd5a6e05efa1'
CONTROL=PRIMARY/'source_controls/7407af320461a092'
files=[]
def read(path):
    files.append(path);return json.loads(path.read_text()) if path.suffix=='.json' else pd.read_csv(path)
def mdtable(frame):
    values=frame.map(lambda v:f'{v:.3f}' if isinstance(v,(float,np.floating)) else str(v))
    return '\n'.join(['| '+' | '.join(frame.columns)+' |','| '+' | '.join(['---']*len(frame.columns))+' |']+['| '+' | '.join(row)+' |' for row in values.itertuples(index=False,name=None)])

primary=read(PRIMARY/'summary.json');longer=read(LONGER/'summary.json')
for run in (PRIMARY,LONGER):
    verification=read(run/'verification.json')
    if not verification['passed']:raise RuntimeError('A real run failed verification')
    if read(run/'protocol.json')['synthetic']:raise RuntimeError('Synthetic evidence cannot enter manuscript')
main=read(PRIMARY/'ranking_metrics.json')['methods'];long=read(LONGER/'ranking_metrics.json')['methods']
keys=['trained_latent_all_profile_score','trained_latent_all_raw_full','trained_latent_all_full_distance','trained_latent_all_own_distance','untrained_latent_all_profile_score','trained_coordinate_all_profile_score','untrained_coordinate_all_profile_score','ridge_coordinate_all_profile_score','trained_latent_budget4_profile_score','trained_latent_all_anomaly_contrast']
names=['Latent benefit profile','Raw full latent error','Full-error reference distance','Own-error reference distance','Random encoder, latent heads','Coordinate benefit profile','Random encoder, coordinate heads','Ridge coordinate profile','Latent profile, four patches','Reference-free contrast']
budget=pd.DataFrame([{'Method':name,'Primary512':main[key]['primary_auroc'],'Sensitivity4096':long[key]['primary_auroc']} for name,key in zip(names,keys)])
budget.to_csv(BASE/'budget_comparison.csv',index=False)
wide=read(LONGER/'paired_source_scores.csv');intervals={}
for key in ('trained_latent_all_full_distance','trained_latent_all_own_distance','untrained_latent_all_profile_score'):
    intervals[key]=paired_fold_bootstrap(wide,'trained_latent_all_profile_score',key,repeats=1000,seed=42)
dump_json(LONGER/'additional_matched_intervals.json',intervals);files.append(LONGER/'additional_matched_intervals.json')
# This secondary restriction was determined by the four condition families.
focal={ 'parkinsons','stroke','cerebral palsy','myopathic' }
primary_wide=read(PRIMARY/'paired_source_scores.csv')
focus=primary_wide[primary_wide.binary_label.eq(0)|primary_wide.categories.map(lambda value:bool(set(value.split('|'))&focal))]
focal_metrics=fold_ranking_metrics(focus,keys);dump_json(PRIMARY/'focal_family_descriptive_metrics.json',focal_metrics);files.append(PRIMARY/'focal_family_descriptive_metrics.json')
profiles=read(PRIMARY/'fold_0/development_controls/paired_profiles.csv')
changes=[]
for kind,group in profiles.groupby('target_type'):
    pivot=group.pivot(index='source_id',columns='intervention',values='profile_score').dropna()
    for name in ('relative_limb_shift','whole_body_shift','coordinate_noise','observation_dropout'):
        delta=pivot[name]-pivot.original
        changes.append(dict(target_type=kind,intervention=name,n_sources=len(pivot),median_paired_change=float(delta.median()),minimum=float(delta.min()),maximum=float(delta.max())))
changes=pd.DataFrame(changes);changes.to_csv(BASE/'development_control_changes.csv',index=False)
teacher=read(PRIMARY/'fold_0/development_controls/teacher_decomposition/source_joint_components.csv')
source_abs=teacher.groupby('source_id')[['predictor_own','teacher_own','interaction_own']].agg(lambda values:np.abs(values).mean())
teacher_summary=dict(sources=len(source_abs),source_joint_rows=len(teacher),median_source_mean_abs_teacher_own=float(source_abs.teacher_own.median()),min_source_mean_abs_teacher_own=float(source_abs.teacher_own.min()),max_source_mean_abs_teacher_own=float(source_abs.teacher_own.max()),maximum_abs_predictor_own=float(teacher.predictor_own.abs().max()))
training=[]
for budget_name,run in [('512',PRIMARY),('4096',LONGER)]:
    for fold in range(4):
        h=read(run/f'fold_{fold}/trained_history.json')
        last=h['predictors'][-1];pre=h['pretrain'][-1]
        training.append(dict(head_updates=budget_name,fold=fold,own_coordinate=last['validation_coordinate_own'],full_coordinate=last['validation_coordinate_full'],own_latent=last['validation_latent_own'],full_latent=last['validation_latent_full'],effective_rank=pre['validation_effective_rank'],embedding_std=pre['validation_embedding_std']))
training=pd.DataFrame(training);training.to_csv(BASE/'normal_validation_diagnostics.csv',index=False)
evidence=dict(primary_run=PRIMARY.name,sensitivity_run=LONGER.name,cohort=primary['cohort'],budget_comparison=budget.to_dict('records'),primary_difference=read(PRIMARY/'primary_paired_bootstrap.json'),longer_difference=read(LONGER/'primary_paired_bootstrap.json'),matched_primary=read(PRIMARY/'matched_comparator_intervals.csv').to_dict('records'),reference_loo_ranges=read(PRIMARY/'reference_loo_auroc_ranges.csv').to_dict('records'),source_controls=read(CONTROL/'ranking_metrics.json'),source_control_intervals=read(CONTROL/'paired_bootstrap.json'),focal_family_metrics=focal_metrics,teacher_diagnostic=teacher_summary,development_controls=changes.to_dict('records'),normal_validation=training.to_dict('records'))
evidence['files']=[dict(path=str(path.relative_to(EXP)),sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in sorted(set(files))]
dump_json(BASE/'evidence.json',evidence)
report=f'''# Additional real-data evidence

The 512-head-update experiment remains primary. The 4,096-update experiment was motivated by normal-validation head adequacy before the new outer-test scores were inspected. It uses the same frozen source roles, masks and 512-update pretrained backbones. Neither budget establishes a learned-encoder advantage.

{mdtable(budget)}

The primary latent profile minus raw error difference is {evidence['primary_difference']['point_difference']:.3f}, with a conditional 95% interval from {evidence['primary_difference']['interval_95'][0]:.3f} to {evidence['primary_difference']['interval_95'][1]:.3f}. The corresponding longer-budget difference is {evidence['longer_difference']['point_difference']:.3f}, with a conditional 95% interval from {evidence['longer_difference']['interval_95'][0]:.3f} to {evidence['longer_difference']['interval_95'][1]:.3f}. Both intervals include zero. The random encoder's prediction heads are fitted in both budgets.

## Normal validation

{mdtable(training)}

Coordinate full-context MSE is above own-context MSE in all four primary folds and three of four longer-budget folds. Fold 2 reverses that ordering after longer training. This remaining learner-adequacy limitation is not evidence that additional body information is intrinsically harmful.

## Development probes

{mdtable(changes)}

Twelve fitting-normal sources supplied long windows; five remain scorable after common support across every edit, including dropout. These are signed paired score changes on that small development set. They do not validate a neurological mechanism or held-out generalization. Geometry reports separately measure damage at limb-to-torso connections.

For the contextual-teacher diagnostic, {teacher_summary['sources']} fitting-normal sources contribute {teacher_summary['source_joint_rows']} source-joint summaries. The own-predictor-only change is exactly zero when its local history is unchanged. The median across sources of the mean absolute teacher-only own-KL change is {teacher_summary['median_source_mean_abs_teacher_own']:.4f}. This verifies that a latent target can change because other joints changed, even when the local joint and its own-context prediction are fixed. It is an expected property of contextual targets, not a neural mechanism.

## Traceability

The [manuscript evidence file](draft01/evidence.json) records source artifact hashes. The primary results remain in [context_study_results.md](context_study_results.md). The longer-budget and nuisance comparisons are supporting analyses, not replacements selected for a favorable outcome.
'''
(EXP/'docs/context_study_sensitivity.md').write_text(report)
print(BASE/'evidence.json')
