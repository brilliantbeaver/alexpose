"""Create traceable real-data tables, vector plots, and a plain results report."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from aaai_gait.settings import EXPERIMENT,ROOT
from aaai_gait.study_runner import summarize_study,dump_json
from aaai_gait.predictive_profile import paired_fold_bootstrap,fold_ranking_metrics
from sklearn.metrics import roc_auc_score

NAMES={
 'trained_latent_all_profile_score':'JEPA latent profile',
 'trained_latent_all_raw_full':'JEPA full-context error',
 'trained_coordinate_all_profile_score':'JEPA coordinate profile',
 'ridge_coordinate_all_profile_score':'Ridge coordinate profile',
 'untrained_latent_all_profile_score':'Random encoder + fitted latent heads',
 'trained_latent_all_own_distance':'Own-error profile distance',
 'trained_latent_all_full_distance':'Full-error profile distance',
 'trained_latent_all_patch_distance':'Patch-error profile distance',
 'trained_latent_budget4_profile_score':'JEPA profile, four patches',
 'trained_latent_all_anomaly_contrast':'Reference-free full-minus-own contrast',
}

def table(frame):
    def val(v):
        if isinstance(v,(float,np.floating)):return f'{v:.3f}' if np.isfinite(v) else 'Unavailable'
        return str(v)
    rows=['| '+' | '.join(map(str,frame.columns))+' |','| '+' | '.join(['---']*len(frame.columns))+' |']
    rows+=['| '+' | '.join(val(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None)]
    return '\n'.join(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-dir',type=Path,required=True)
    args=parser.parse_args();run=args.run_dir.resolve()
    protocol=json.loads((run/'protocol.json').read_text())
    if protocol['synthetic']:raise ValueError('This report is reserved for real GAVD evidence')
    summary=summarize_study(run,synthetic=False)
    metrics=json.loads((run/'ranking_metrics.json').read_text());paired=json.loads((run/'primary_paired_bootstrap.json').read_text())
    before=pd.read_csv(run/'window_index_before_profile_qc.csv');meta=pd.read_csv(run/'window_index.csv')
    manifest=pd.read_csv(run/'fresh_manifest.csv');elig=pd.read_csv(run/'profile_eligibility.csv')
    media=pd.read_csv(EXPERIMENT/'artifacts/context_study/ingestion/media_ledger.csv')
    census=manifest.groupby('label').agg(annotated_sequences=('sequence_id','size'),annotated_sources=('source_id','nunique'))
    census=census.join(before.groupby('label').agg(pose_sequences=('sequence_id','nunique'),pose_sources=('source_id','nunique'),pose_windows=('source_id','size')))
    census=census.join(meta.groupby('label').agg(profile_sources=('source_id','nunique'),profile_windows=('source_id','size'))).fillna(0).astype(int)
    census.to_csv(run/'attrition_by_label.csv')
    prior=meta.drop_duplicates('source_id').groupby(['binary_label','prior_use']).size().rename('sources').reset_index()
    prior.to_csv(run/'prior_exploration_sources.csv',index=False)
    selected=pd.DataFrame([dict(Method=display,AUROC=metrics['methods'][name]['primary_auroc'],AP=metrics['methods'][name]['normal_weighted_ap']) for name,display in NAMES.items() if name in metrics['methods']])
    selected.to_csv(run/'manuscript_main_table.csv',index=False)
    wide=pd.read_csv(run/'paired_source_scores.csv');contrasts=[]
    for baseline in ('trained_latent_all_full_distance','trained_latent_all_own_distance','untrained_latent_all_profile_score','ridge_coordinate_all_profile_score'):
        item=paired_fold_bootstrap(wide,'trained_latent_all_profile_score',baseline,repeats=1000,seed=42)
        contrasts.append(dict(Comparator=NAMES[baseline],Difference=item['point_difference'],CI_low=item['interval_95'][0],CI_high=item['interval_95'][1]))
    contrasts=pd.DataFrame(contrasts);contrasts.to_csv(run/'matched_comparator_intervals.csv',index=False)
    source_meta=meta.drop_duplicates('source_id')[['source_id','prior_use']]
    novelty=wide.merge(source_meta,on='source_id',validate='one_to_one')
    novelty=novelty[novelty.binary_label.eq(0)|novelty.prior_use.eq('not_in_previous_study')]
    new_abnormal_metrics=fold_ranking_metrics(novelty,list(NAMES))
    dump_json(run/'new_abnormal_descriptive_metrics.json',new_abnormal_metrics)
    loo_records=[]
    for folder in sorted(run.glob('fold_*')):
        loo=pd.read_csv(folder/'trained_reference_sensitivity.csv')
        labels=wide[['source_id','binary_label']]
        loo=loo.merge(labels,on='source_id',validate='many_to_one')
        for (fold,kind,omitted),part in loo.groupby(['outer_fold','target_type','left_out_source']):
            loo_records.append(dict(fold=fold,target_type=kind,omitted=omitted,auroc=roc_auc_score(part.binary_label,part.score),n_normal=int(part.binary_label.eq(0).sum())))
    loo_table=pd.DataFrame(loo_records);loo_table.to_csv(run/'reference_loo_auroc.csv',index=False)
    ranges=[]
    for kind,rows in loo_table.groupby('target_type'):
        g=rows.groupby('fold').agg(low=('auroc','min'),high=('auroc','max'),weight=('n_normal','first'))
        ranges.append(dict(target_type=kind,minimum=(g.low*g.weight).sum()/g.weight.sum(),maximum=(g.high*g.weight).sum()/g.weight.sum()))
    reference_ranges=pd.DataFrame(ranges);reference_ranges.to_csv(run/'reference_loo_auroc_ranges.csv',index=False)


    categories=metrics['methods']['trained_latent_all_profile_score']['categories'];cat=[]
    for label,value in categories.items():
        cat.append(dict(Category=label,Sources=value['n_category_sources'],Folds=value['supported_folds'],
            AUROC=value['normal_weighted_auroc'],AP=value['normal_weighted_ap']))
    category_table=pd.DataFrame(cat);category_table.to_csv(run/'manuscript_category_table.csv',index=False)
    condition=[];learning=[]
    for folder in sorted(run.glob('fold_*')):
        if (folder/'condition_metrics.json').exists():
            value=json.loads((folder/'condition_metrics.json').read_text())
            for name,result in value['metrics'].items():condition.append(dict(Fold=folder.name,Method=name,Macro_F1=result['macro_f1'],Balanced_accuracy=result['balanced_accuracy'],Sources=result['n_test_sources']))
        history=json.loads((folder/'trained_history.json').read_text())
        for stage in ('pretrain','predictors'):
            for row in history[stage]:learning.append(dict(fold=folder.name,stage=stage,**row))
    condition_table=pd.DataFrame(condition);condition_table.to_csv(run/'manuscript_condition_table.csv',index=False)
    curves=pd.DataFrame(learning);curves.to_csv(run/'training_curves.csv',index=False)
    images=EXPERIMENT/'images';images.mkdir(exist_ok=True)
    plt.rcParams.update({'svg.fonttype':'none','pdf.fonttype':42,'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    core=selected.head(5)
    fig,ax=plt.subplots(figsize=(9,4),layout='constrained')
    colors=['#1c7882','#526378','#b67b30','#6e5a9d','#7d8c93']
    bars=ax.barh(core.Method,core.AUROC,color=colors)
    ax.axvline(.5,color='#999999',ls=':',label='Chance ranking');ax.set_xlim(0,1);ax.invert_yaxis()
    ax.set_xlabel('Normal-count-weighted within-fold AUROC');ax.set_title('Real GAVD source-held-out comparison')
    for bar,value in zip(bars,core.AUROC):ax.text(value+.015,bar.get_y()+bar.get_height()/2,f'{value:.3f}',va='center')
    for ext in ('svg','pdf','png'):fig.savefig(images/f'context_real_comparison.{ext}',dpi=160)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,3.8),layout='constrained')
    for folder in sorted(run.glob('fold_*')):
        h=json.loads((folder/'trained_history.json').read_text())
        pre=pd.DataFrame(h['pretrain']);heads=pd.DataFrame(h['predictors'])
        axes[0].plot(pre['update'],pre.validation_prediction,label=folder.name.replace('_',' '))
        axes[1].plot(heads['update'],heads.validation_latent_full,label=folder.name.replace('_',' '))
    axes[0].set(title='Normal validation: JEPA pretraining',xlabel='Optimizer updates',ylabel='Latent cross-entropy')
    axes[1].set(title='Normal validation: full-context head',xlabel='Optimizer updates',ylabel='Latent KL')
    axes[1].legend(fontsize=9)
    for ext in ('svg','pdf','png'):fig.savefig(images/f'context_training_curves.{ext}',dpi=160)
    plt.close(fig)
    lo,hi=paired['interval_95'];delta=paired['point_difference']
    interpretation=('The conditional interval is above zero for the prespecified comparison. This supports an advantage over raw error within this fitted cohort, while the other controls and external validity remain separate questions.' if lo>0 else
                    'The conditional interval is below zero: the proposed profile performs worse than raw full-context error under this protocol.' if hi<0 else
                    'The conditional interval includes zero. The experiment does not establish an advantage of the proposed profile over raw full-context error.')
    text=f'''# Real GAVD results for the revised context study

This report was generated from a completed real-data artifact. It is separate from the earlier five-condition exploratory results in [results.md](results.md). No synthetic score enters the tables below.

## Cohort and acquisition

The current annotations contain {len(manifest):,} sequences and {manifest.source_id.nunique()} source IDs. The acquisition ledger records {media.status.isin(['downloaded','verified_present']).sum()} locally decodable videos. Pose-window quality checks retain {len(before):,} windows from {before.sequence_id.nunique():,} sequences and {before.source_id.nunique()} sources. The fixed twelve-joint profile-support check leaves **{len(meta):,} windows, {meta.sequence_id.nunique():,} sequences, and {meta.source_id.nunique()} sources**, including **{meta.loc[meta.binary_label.eq(0),'source_id'].nunique()} normal sources**.

The profile check is based on observation validity and the fixed uniform masks before any fit or test score. Every retained source supplies at least four distinct physical patches for every readout joint. Category source memberships may overlap. These counts are source videos, not verified independent people.

{table(census.reset_index())}

## Previous use of these recordings

{table(prior)}

The new protocol was specified before its new score evaluation, but the dataset was previously used for exploratory work. All retained normal sources appeared in that earlier study. The outer folds provide source separation for each fitted model; they are not a newly sequestered clinical test cohort.

## Prespecified anomaly comparison

All normal-only models use the same outer source folds. Five normal reference sources per fold are excluded from encoder and predictor fitting. The primary statistic averages within-fold AUROC using normal-source counts. AP is averaged with the same weights and must be interpreted with each fold's abnormal prevalence, available in the machine-readable metrics.

{table(selected)}

The primary paired difference, latent profile minus full-context error, is **{delta:.3f} AUROC**, with a conditional 95% bootstrap interval **[{lo:.3f}, {hi:.3f}]**. The bootstrap resamples sources within fold and binary label, carrying both methods together. It conditions on fitted models, chosen references, and folds. It does not include retraining uncertainty, reference-selection uncertainty, or unknown repeated participants.

{interpretation}

![Real source-held-out ranking](../images/context_real_comparison.svg)

*Bars are point estimates on the common cohort. The interval above refers to the paired primary difference, not to each bar separately.*

## Matched alternative explanations

{table(contrasts)}

These additional paired intervals use the same fixed-fit bootstrap. The own- and full-error distances receive the same two-sided reference calibration as the proposed profile. An advantage over raw error alone cannot establish that subtracting errors adds information. These are exploratory control comparisons without a multiple-comparison correction.

## Reference sensitivity and newly used abnormal sources

{table(reference_ranges)}

These are the smallest and largest aggregate AUROCs attainable by omitting one inner reference source in each fold while holding the models fixed. They are perturbation ranges, not confidence intervals. The reference-free contrast above remains a secondary comparison and is not promoted to a new primary endpoint.

A descriptive restriction retains all normal test sources and only the 170 abnormal sources absent from the previous study. Its full metrics are saved in `new_abnormal_descriptive_metrics.json`. Because the normal sources remain previously explored, this restriction is not external validation.

## Category descriptions

{table(category_table)}

Category summaries reuse normal sources and may reuse mixed-category abnormal sources. They are dependent. A category can lack abnormal support in a fold; its descriptive average uses supported folds only. Two- and three-source categories cannot support stable population estimates. Folder labels are not verified diagnoses.

## Separate supervised condition recognition

Both classifiers receive focal-condition training labels and exclude outer test sources and the inner normal reference sources. The JEPA encoder remains frozen after normal-only pretraining. Predictions average windows within sequences and sequences within source-label pairs; a mixed-label source has total metric weight one. These supervised macro-F1 scores are not comparable numerically with normal-only AUROC.

{table(condition_table)}

## Training and robustness

The run uses {protocol['config']['pretrain_updates']} JEPA pretraining updates and {protocol['config']['predictor_updates']} updates for each independent context head. Every window uses the same sixteen-mask scoring bank, independent of motion intensity and pose validity. Ridge uses {protocol['ridge_draws']} source-balanced training draws. These are fixed budgets and do not establish convergence by themselves.

![Actual normal-validation curves](../images/context_training_curves.svg)

*Validation sources influence neither weights nor target centers. The curves describe this budget, not an optimized stopping rule.*

The run also saves the four-patch observation-budget sensitivity and leave-one-normal-reference-out scores. No score direction, fold assignment, or category selection is changed after test inspection. The development timing/observation controls are reported separately when completed. Their edits are not disease simulations. A published SeeKer or STG-NF comparison remains unvalidated for this partially observed sixteen-joint corpus; see the [feasibility report](published_baseline_feasibility.md).

## Reproducibility and claim boundary

The run directory is `artifacts/context_study/gavd/{run.name}/`. Its protocol, source roles, fresh manifest, acquisition and eligibility ledgers, checkpoints, cache hashes, frozen references, source scores, and training histories support the reported tables. `scripts/summarize_context_study.py` regenerates this report and the vector plots. The [implemented protocol](context_study_protocol.md) explains the estimand and controls.

A result on this cohort cannot identify neural causes or establish clinical screening performance. The normal sample is small, diagnoses and participant identities are unverified, and pose exclusions select for visible joints. GAVD supports a bounded observational comparison. The [draft01 manuscript and readiness assessment](draft01/README.md) state which conclusions survive those limitations.
'''
    (EXPERIMENT/'docs/context_study_results.md').write_text(text)
    summary['metrics']=[dict(method=NAMES.get(v['method'],v['method']),value=v['value']) for v in summary['metrics']]
    summary['notes']+=[interpretation,f'Primary paired AUROC difference {delta:.3f}; conditional95%CI [{lo:.3f},{hi:.3f}].']
    dump_json(run/'slides_summary.json',summary)
    print(EXPERIMENT/'docs/context_study_results.md')

if __name__=='__main__':main()
