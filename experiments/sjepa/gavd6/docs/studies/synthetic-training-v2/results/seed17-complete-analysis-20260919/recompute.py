"""Independent reaggregation of the complete seed-17 diagnostic download.

Requires numpy, pandas and matplotlib. Source outputs are read only. Raw neural
predictions/targets are absent, so this verifies saved tables, not fresh inference.
Bootstrap intervals are exploratory, pointwise and conditional on the fitted seed.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

ROOT = Path(__file__).resolve().parents[5]
SOURCE = ROOT / "outputs/full-updates-2000-seed-17/updates-2000-seed-17"
OUT = Path(__file__).resolve().parent
STRATA = ["method", "split", "seed", "extractor", "evidence_status"]
UNIT = ["person_id", "motion_id", "window_id", "variant"]
METRICS = ["visible_nle", "lower_limb_nle", "missing_rate", "displacement_nle",
           "ankle_separation_mae", "amplitude_error", "amplitude_ratio", "event_timing_mae_s"]
DRAW_SEED, DRAWS = 20260919, 50000


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(name):
    return pd.read_csv(SOURCE / name, low_memory=False)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def complete_average(frame, keys, measures):
    group = frame.groupby(keys, sort=True, dropna=False)[measures]
    means = group.mean()
    return means.where(group.count().eq(group.size(), axis=0)).reset_index()


def balanced(frame, measures=METRICS):
    window = complete_average(frame, STRATA + UNIT[:3], measures)
    motion = complete_average(window, STRATA + UNIT[:2], measures)
    person = complete_average(motion, STRATA + UNIT[:1], measures)
    summary = complete_average(person, STRATA, measures)
    return person, summary.melt(id_vars=STRATA, var_name="metric", value_name="value")


def compare_values(actual, saved, keys, values, label):
    require(not actual.duplicated(keys).any() and not saved.duplicated(keys).any(), label + ': duplicate keys')
    a, b = actual.set_index(keys).sort_index(), saved.set_index(keys).sort_index()
    require(a.index.equals(b.index), label + ': different row identities')
    for column in values:
        np.testing.assert_allclose(a[column], b[column], equal_nan=True, rtol=1e-11, atol=1e-13,
                                   err_msg=f'{label}: {column}')
    delta = (a[values] - b[values]).abs()
    return dict(table=label, rows=len(a), numeric_cells=len(a)*len(values),
                unsupported_cells=int(a[values].isna().sum().sum()),
                max_absolute_difference=float(delta.max().max()) if delta.notna().any().any() else 0.)


def audit_aggregates(frame, saved, extra=(), measures=METRICS, condition_column='condition'):
    keys = ['endpoint', 'scale_policy', *extra]
    chunks, persons = [], []
    for labels, block in frame.groupby(keys, sort=True):
        tags = dict(zip(keys, labels))
        for condition, selected in [('all_conditions', block), *list(block.groupby('variant'))]:
            people, summary = balanced(selected, measures)
            chunks.append(summary.assign(**tags, **{condition_column: condition}))
            persons.append(people.assign(**tags, condition=condition))
    result = pd.concat(chunks, ignore_index=True)
    checked = compare_values(result, saved, [*STRATA, 'metric', *keys, condition_column], ['value'],
                             'aggregates:' + ','.join(extra or ['coordinates']))
    return checked, pd.concat(persons, ignore_index=True), result


def timing_group(g):
    eligible = g[g.reference_eligible]
    matches = int(eligible.matched_timing_denominator.sum())
    reference = int(eligible.reference_event_denominator.sum())
    predicted = int(eligible.prediction_event_denominator.sum())
    complete = int(eligible.complete_prediction_window_denominator.sum())
    timed = eligible[eligible.matched_timing_denominator.gt(0)]
    return dict(records=len(g), reference_eligible=len(eligible), reference_ineligible=len(g)-len(eligible),
                original_supported=int(g.original_support.sum()), complete_predictions=complete,
                incomplete_predictions=len(eligible)-complete,
                count_mismatch_records=int(eligible.prediction_status.eq('event_count_mismatch').sum()),
                reference_events=reference, matched_events=matches,
                missed_events=reference-matches if len(eligible) else np.nan,
                known_extra_events=float(eligible.extra_event_count.sum()) if complete else np.nan,
                extra_count_unknown_records=len(eligible)-complete,
                event_recall=matches/reference if reference else np.nan,
                known_event_precision=matches/predicted if predicted else np.nan,
                conditional_timing_mae_s=float((timed.matched_timing_mae_s*timed.matched_timing_denominator).sum()/matches)
                    if matches else np.nan)


def contrasts(people):
    selected = people[people.condition.eq('all_conditions') & people.endpoint.eq('visible')
                      & people.scale_policy.eq('frame_reference')]
    pairs = [('direct', m) for m in ('unchanged', 'joint_affine', 'static', 'filter1', 'filter2', 'paired_jepa')]
    pairs += [('paired_jepa', m) for m in ('initialized', 'coordinate', 'ordinary_jepa', 'shuffled_jepa')]
    rows, person_rows = [], []
    rng = np.random.default_rng(DRAW_SEED)
    indices = rng.integers(8, size=(DRAWS, 8))
    for metric in ('visible_nle', 'displacement_nle'):
        for extractor, group in selected.groupby('extractor'):
            wide = group.pivot(index='person_id', columns='method', values=metric).sort_index()
            require(len(wide) == 8 and np.isfinite(wide.to_numpy()).all(), 'Incomplete primary person panel')
            for candidate, comparator in pairs:
                x, y = wide[candidate].to_numpy(), wide[comparator].to_numpy()
                difference = y-x
                xa, ya = x[indices].mean(1), y[indices].mean(1)
                relative = 100*(ya-xa)/ya
                lo, hi = np.quantile(relative, [.025, .975])
                absolute_ci = np.quantile(ya-xa, [.025, .975])
                loo = [100*np.delete(difference,i).mean()/np.delete(y,i).mean() for i in range(8)]
                rows.append(dict(metric=metric, extractor=extractor, candidate=candidate, comparator=comparator,
                    candidate_mean=float(x.mean()), comparator_mean=float(y.mean()), people=8,
                    improvement=float(difference.mean()), improvement_percent=float(100*difference.mean()/y.mean()),
                    ci95_low_percent=float(lo), ci95_high_percent=float(hi),
                    ci95_low=float(absolute_ci[0]), ci95_high=float(absolute_ci[1]),
                    people_improved=int((difference>0).sum()), people_worsened=int((difference<0).sum()),
                    leave_one_person_out_min_percent=min(loo), leave_one_person_out_max_percent=max(loo),
                    bootstrap_draws=DRAWS, bootstrap_seed=DRAW_SEED,
                    scope='Exploratory pointwise interval; conditional on seed 17; no multiplicity adjustment'))
                for p,a,b in zip(wide.index,x,y):
                    person_rows.append(dict(person_id=p, metric=metric, extractor=extractor,
                        candidate=candidate, comparator=comparator, candidate_value=a, comparator_value=b,
                        improvement=b-a, improvement_percent=100*(b-a)/b))
    return pd.DataFrame(rows), pd.DataFrame(person_rows)


def plots(comparisons, primary, exploratory):
    extractors = ['hrnet_w32','rtmpose_m','vitpose_base']
    labels = ['HRNet-W32','RTMPose-M','ViTPose-Base']
    fig, axes = plt.subplots(1, 2, figsize=(12,5.6))
    options = [('direct','static','Direct vs static'),('direct','joint_affine','Direct vs affine'),
               ('direct','filter2','Direct vs 5-frame filter'),('paired_jepa','initialized','Paired vs initialized'),
               ('paired_jepa','shuffled_jepa','Paired vs shuffled')]
    for ax,metric,title in zip(axes,['visible_nle','displacement_nle'],['Visible position error','0.20-second displacement error']):
        for j,(e,label,color) in enumerate(zip(extractors,labels,['#2266aa','#bd652d','#258a70'])):
            for i,(a,b,_) in enumerate(options):
                r=comparisons[(comparisons.metric==metric)&(comparisons.extractor==e)&
                    (comparisons.candidate==a)&(comparisons.comparator==b)].iloc[0]
                ax.errorbar(r.improvement_percent,i+(j-1)*.17,
                    xerr=[[r.improvement_percent-r.ci95_low_percent],[r.ci95_high_percent-r.improvement_percent]],
                    fmt='o',color=color,capsize=2,label=label if i==0 else None,markersize=4)
        ax.set_yticks(range(len(options)),[z[2] for z in options])
        ax.invert_yaxis(); ax.axvline(0,color='#444444',lw=.8)
        ax.set_title(title); ax.set_xlabel('Error reduction relative to comparator (%)')
        ax.grid(axis='x',alpha=.15); ax.spines[['top','right']].set_visible(False)
    axes[1].legend(fontsize=9,loc='lower left')
    fig.suptitle('Eight-person paired comparisons, seed 17')
    fig.text(.5,.025,'50,000 person-cluster bootstrap draws. Pointwise 95% intervals; training-seed uncertainty excluded.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.055,1,.94))
    for ext in ('png','svg'):fig.savefig(OUT/f'paired-effects.{ext}',dpi=180)
    plt.close(fig)

    allvalid=exploratory[exploratory.endpoint.eq('all_valid_synthetic')&
        exploratory.scale_policy.eq('window_median_reference')&exploratory.condition.eq('all_conditions')]
    methods=['unchanged','joint_affine','static','direct','paired_jepa','filter2']
    fig,axes=plt.subplots(1,3,figsize=(13,4.8))
    for ax,e,label in zip(axes,extractors,labels):
        s=allvalid[allvalid.extractor.eq(e)&allvalid.metric.eq('amplitude_ratio')].set_index('method')
        vals=[s.loc[m,'value'] for m in methods]
        ax.bar(range(len(methods)),vals,color=['#999999','#a87945','#78855f','#c64f47','#3b77b4','#6d5284'])
        for i,v in enumerate(vals):
            if not np.isfinite(v):ax.text(i,.08,'unsupported',rotation=90,ha='center',fontsize=8)
        ax.axhline(1,color='#444444',lw=1,ls='--');ax.set_ylim(0,1.9);ax.set_xlim(-.6,len(methods)-.4)
        ax.set_xticks(range(len(methods)),['Unchanged','Affine','Static','Direct','Paired JEPA','Filter 2'],rotation=45,ha='right')
        ax.set_title(label);ax.set_ylabel('Predicted / reference RMS amplitude')
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Exploratory ankle-separation amplitude: all valid synthetic joints, fixed window scale')
    fig.text(.5,.025,'Means of per-window ratios; no waveform/timing guarantee. RTMPose unsupported rows retain a missing-input failure.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.055,1,.94))
    for ext in ('png','svg'):fig.savefig(OUT/f'amplitude-diagnostics.{ext}',dpi=180)
    plt.close(fig)


def main():
    before={str(p.relative_to(SOURCE)):sha(p) for p in SOURCE.rglob('*') if p.is_file()}
    meta=json.loads((SOURCE/'analysis.json').read_text())
    expansion=json.loads((SOURCE/'expansion-analysis.json').read_text())
    manifest={n:before.get(n)==h for n,h in expansion['files'].items()}
    code={n:sha(ROOT/n)==h for n,h in expansion['analysis_code'].items()}
    require(all(manifest.values()) and all(code.values()),'Downloaded evidence or diagnostic code hash mismatch')
    f,saved,by_condition=read('per-window.csv'),read('per-person-balanced-summary.csv'),read('by-condition.csv')
    x,xs=read('exploratory-per-window.csv'),read('exploratory-balanced.csv')
    lr,lrs=read('exploratory-left-right.csv'),read('exploratory-left-right-balanced.csv')
    t,ts=read('exploratory-timing.csv'),read('exploratory-timing-summary.csv')
    require(len(f)==2688 and len(x)==21504 and len(lr)==64512 and len(t)==23040,'Unexpected downloaded row roster')
    require(not x.duplicated(STRATA+UNIT+['endpoint','scale_policy']).any(),'Duplicate experimental observation')
    require(x.groupby(['method','endpoint','scale_policy']).size().eq(384).all(),'Unequal method rosters')
    require(set(x.seed)=={17} and x.person_id.nunique()==8 and x.window_id.nunique()==32,'Unexpected development panel')
    primary=x[x.endpoint.eq('visible')&x.scale_policy.eq('frame_reference')].copy()
    roster=primary[primary.method.eq('unchanged')].set_index(UNIT+['extractor']).sort_index()
    require(roster.reset_index().groupby('person_id').window_id.nunique().eq(4).all(),'Expected four windows per person')
    for method,g in primary.groupby('method'):
        candidate=g.set_index(UNIT+['extractor']).sort_index()
        require(candidate.index.equals(roster.index),'Different per-method evaluation identities: '+method)
        require(candidate[['visible_count','lower_limb_count','synthetic_all_count']].equals(
            roster[['visible_count','lower_limb_count','synthetic_all_count']]),'Changed per-method reference support')
    orig=primary[primary.method.isin(f.method.unique())]
    checks=[compare_values(orig,f,STRATA+UNIT,METRICS,'standard-versus-exploratory-primary')]
    _,recomputed=balanced(f)
    checks.append(compare_values(recomputed,saved,STRATA+['metric'],['value'],'standard-aggregates'))
    condition_values=[]
    for condition,g in f.groupby('variant'):
        _,agg=balanced(g);condition_values.append(agg.assign(variant=condition))
    checks.append(compare_values(pd.concat(condition_values),by_condition,STRATA+['metric','variant'],['value'],'standard-conditions'))
    check,people,summary=audit_aggregates(x,xs);checks.append(check)
    check,lr_people,lr_summary=audit_aggregates(lr,lrs,extra=['pair'],measures=['left_right_nle']);checks.append(check)
    people.to_csv(OUT/'per-person-metrics.csv',index=False)
    lr_people.to_csv(OUT/'per-person-left-right.csv',index=False)
    timing_rows=[]
    tkeys=['endpoint','scale_policy','method','extractor','seed']
    for labels,g in t.groupby(tkeys,sort=True):
        for condition,b in [('all_conditions',g),*list(g.groupby('variant'))]:
            timing_rows.append(dict(zip(tkeys,labels))|dict(condition=condition)|timing_group(b))
    tr=pd.DataFrame(timing_rows)
    tvalues=[c for c in tr if c not in tkeys+['condition']]
    checks.append(compare_values(tr,ts,tkeys+['condition'],tvalues,'exploratory-timing-summary'))
    standard_timing=read('timing-summary.csv')
    st=tr[tr.endpoint.eq('visible')&tr.scale_policy.eq('frame_reference')&tr.condition.eq('all_conditions')&
          tr.method.isin(standard_timing.method.unique())]
    checks.append(compare_values(st,standard_timing,['method','extractor','seed'],tvalues,'standard-timing-summary'))
    pd.DataFrame(checks).to_csv(OUT/'table-verification.csv',index=False)
    effects,person_effects=contrasts(people)
    effects.to_csv(OUT/'paired-comparisons.csv',index=False)
    person_effects.to_csv(OUT/'person-effects.csv',index=False)
    # Rank all displayed methods, preserving each endpoint/support/scale/condition.
    ranking=summary.copy()
    ranking['rank']=ranking.groupby(['endpoint','scale_policy','condition','extractor','metric'])['value'].rank(method='min')
    ranking.loc[ranking.metric.eq('amplitude_ratio'),'rank']=np.nan
    ranking.to_csv(OUT/'all-method-results.csv',index=False)
    # Hidden-joint coordinate errors remain explicitly separate from visible scores.
    occluded=primary.copy()
    hidden_tables=[]
    for condition,g in [('all_conditions',occluded),*list(occluded.groupby('variant'))]:
        _,oa=balanced(g,['synthetic_occluded_nle','synthetic_all_nle'])
        hidden_tables.append(oa.assign(condition=condition))
    pd.concat(hidden_tables,ignore_index=True).to_csv(OUT/'synthetic-hidden-coordinate-results.csv',index=False)
    # Reference records repeat across extractors. Keep one extractor per definition.
    oracle=t[t.method.eq('reference_oracle')&t.extractor.eq('hrnet_w32')].copy()
    ref_check=t[t.method.eq('reference_oracle') & t.endpoint.eq('all_valid_synthetic')
                & t.scale_policy.eq('window_median_reference')]
    for _,g in ref_check.groupby(UNIT[:3]):
        require(len(g)==12 and g[['reference_event_count','reference_amplitude_normalized']].nunique().eq(1).all(),
                'Synthetic reference differs across rendering/extractor copies')
    refcols=['endpoint','scale_policy','person_id','motion_id','window_id','variant','reference_eligible',
             'reference_reasons','reference_visible_frames','reference_valid_frames','reference_finite_frames',
             'reference_raw_event_count','reference_event_count','reference_amplitude_raw_px',
             'reference_amplitude_normalized','bbox_scale_cv']
    oracle[refcols].to_csv(OUT/'reference-coverage.csv',index=False)
    coverage=oracle.groupby(['endpoint','scale_policy','variant']).agg(records=('reference_eligible','size'),
        eligible=('reference_eligible','sum'),peaks=('reference_event_count','sum'))
    coverage.to_csv(OUT/'reference-coverage-summary.csv')
    tr.to_csv(OUT/'timing-results.csv',index=False)
    # Fit provenance and update history, without loading model weights.
    models=json.loads((SOURCE/'calibration-models.json').read_text())
    train_people=set(models['joint_offset']['provenance']['canonical_people'])
    require(len(train_people)==24 and train_people.isdisjoint(primary.person_id),'Invalid calibration person separation')
    for model in models.values():
        p=model['provenance']
        require(p['held_extractor']=='vitpose' and p['pooled_extractors']==['hrnet_w32','rtmpose_m']
                and not p['evaluation_labels_used'],'Calibration provenance mismatch')
    arrays={}
    for path in (SOURCE/'predictions').glob('*.npz'):
        with np.load(path,allow_pickle=False) as z:
            a=z['prediction']
            require(a.shape==(384,64,12,2),'Unexpected calibration prediction shape')
            arrays[path.name]=dict(shape=list(a.shape),nonfinite_coordinates=int((~np.isfinite(a)).sum()),
                records_with_nonfinite=int((~np.isfinite(a)).any(axis=(1,2,3)).sum()))
    history,training=read('training-history.csv'),read('training-summary.csv')
    require(len(history)==30000 and training.status.eq('complete').all(),'Incomplete training history')
    trends=[]
    for (arm,phase),g in history.groupby(['arm','phase']):
        g=g.sort_values('phase_update');s=training[training.arm.eq(arm)&training.phase.eq(phase)].iloc[0]
        require(list(g.phase_update)==list(range(1,int(s.planned_phase_updates)+1)),'Missing phase updates')
        require(np.isfinite(g.loss).all() and np.isfinite(g.gradient_norm).all(),'Nonfinite optimizer history')
        require(s.optimizer_updates==s.planned_updates,'Optimizer count differs from plan')
        np.testing.assert_allclose([g.loss.iloc[0],g.loss.iloc[-1]],[s.first_loss,s.last_loss])
        row=dict(arm=arm,phase=phase,updates=len(g),first_loss=g.loss.iloc[0],last_loss=g.loss.iloc[-1],
            last100_mean=g.loss.tail(100).mean(),previous100_mean=g.loss.iloc[-200:-100].mean(),
            final_learning_rate=g.learning_rate.iloc[-1],preclip_gradient_max=g.gradient_norm.max(),
            min_supported_examples=g.supported_examples.min(),min_targets=g.target_count.min())
        row['last100_improvement_percent']=100*(row['previous100_mean']-row['last100_mean'])/row['previous100_mean']
        trends.append(row)
        if arm in {'ordinary_jepa','paired_jepa','shuffled_jepa'} and phase=='pretrain':
            np.testing.assert_allclose(g.ema.prod(),g.teacher_initialization_weight.iloc[-1],rtol=1e-10)
    pd.DataFrame(trends).to_csv(OUT/'training-verification.csv',index=False)
    # These per-joint pixel summaries are conditional on finite predictions.
    residuals=read('per-joint-residuals.csv')
    residuals.groupby(['method','extractor','joint'])[['bias_x_px','bias_y_px','mae_x_px','mae_y_px',
        'scored_frames','missing_predictions']].mean().to_csv(OUT/'conditional-joint-residuals.csv')
    plots(effects,primary,summary)
    after={str(p.relative_to(SOURCE)):sha(p) for p in SOURCE.rglob('*') if p.is_file()}
    require(before==after,'Downloaded evidence changed during analysis')
    result=dict(source=str(SOURCE.relative_to(ROOT)),input_sha256=before,manifest_files_verified=len(manifest),
        local_diagnostic_code_verified=code,source_identity=meta['source_identity'],remote_verification=meta['verification'],
        tables_verified=checks,calibration_arrays=arrays,training_updates=len(history),training_fits=int(training.arm.nunique()),
        training_phases=len(trends),development_people=sorted(primary.person_id.unique()),
        independent_people=8,physical_windows=32,methods=sorted(primary.method.unique()),
        bootstrap=dict(draws=DRAWS,seed=DRAW_SEED,unit='person; all windows/variants retained together',
                       interval='Pointwise percentile; post-hoc; no multiple-comparison or training-seed uncertainty correction'),
        source_unchanged=True,
        limitations=['Raw neural prediction arrays and original targets/bundle are not in this download.',
                     'Remote prediction reconstruction is recorded evidence; local checks independently reaggregate its tables.',
                     'No second training seed, short-budget results, or final suite scheduler accounting in this download.',
                     'Synthetic references remain projected anatomical proxies with machine-screened visibility.'])
    (OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['manifest_files_verified','training_updates','training_fits',
                                           'independent_people','physical_windows','methods','source_unchanged']},indent=2))
    print(pd.DataFrame(checks).to_string(index=False))


if __name__=='__main__':
    main()
