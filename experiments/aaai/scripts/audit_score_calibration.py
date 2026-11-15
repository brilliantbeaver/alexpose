"""Post-outcome descriptive score-scale audit, without refitting or selection."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
import numpy as np
import pandas as pd
BASE=Path(__file__).resolve().parents[1]
GAVD=BASE/'artifacts/iteration02/d66e1b126c9d9a9adf1b'
EXTERNAL=BASE/'artifacts/iteration02/external_fb9d0d23b56ce29fd846'
OUT=BASE/'docs/draft02/reviews/calibration_audit'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'audit_manifest.json').exists():raise ValueError('Preserve the completed descriptive audit')
    inputs={};quantiles=[];references=[];mse=[]
    def read(path):inputs[str(path.relative_to(BASE))]=sha(path);return pd.read_csv(path)
    for dataset,run in [('GAVD_U_U',GAVD),('external_angles',EXTERNAL)]:
        for fold in range(4):
            path=run/f'fold_{fold}'/('U_U/source_scores.csv' if dataset=='GAVD_U_U' else 'scores.csv');scores=read(path)
            if scores.source_id.duplicated().any() or not scores.outer_fold.eq(fold).all():raise ValueError('Invalid frozen test identities')
            normal=scores[scores.binary_label.eq(0)]
            if len(normal)!=(7 if dataset=='GAVD_U_U' else 32):raise ValueError('Unexpected normal test support')
            for score in ['M','R','R_raw']:
                a=normal[score].to_numpy(float)
                if not np.isfinite(a).all():raise ValueError('Missing normal score')
                q=np.quantile(a,[.5,.9,.99],method='linear')
                quantiles.append(dict(dataset=dataset,outer_fold=fold,normal_test_count=len(a),score=score,median=float(q[0]),p90=float(q[1]),p99=float(q[2]),maximum=float(a.max()),fallback_count=int(normal.fallback.sum()) if 'fallback' in normal else int((~normal.relational_supported).sum())))
            if dataset!='external_angles':continue
            reference_path=run/f'fold_{fold}/reference_audit.json';inputs[str(reference_path.relative_to(BASE))]=sha(reference_path);ref=json.loads(reference_path.read_text())['benefit'];ids=ref['sources']
            profiles=read(run/f'fold_{fold}/predictions/participant_angle_profiles.csv');wide=profiles.pivot(index='source_id',columns='angle',values='benefit');bilateral=pd.DataFrame(dict(left=wide[[0,2,4]].mean(1),right=wide[[1,3,5]].mean(1))).loc[ids]
            if len(ids)!=20 or len(set(ids))!=20 or not np.isfinite(bilateral.to_numpy()).all():raise ValueError('Incomplete frozen reference profiles')
            references.append(dict(outer_fold=fold,reference_normals=20,left_right_pearson=float(bilateral.left.corr(bilateral.right)),left_right_spearman=float(bilateral.left.corr(bilateral.right,method='spearman')),left_mean=float(bilateral.left.mean()),right_mean=float(bilateral.right.mean()),left_sample_variance=float(bilateral.left.var(ddof=1)),right_sample_variance=float(bilateral.right.var(ddof=1)),sample_cross_covariance=float(bilateral.cov().loc['left','right']),fitted_left_variance=float(ref['variance'][0]),fitted_right_variance=float(ref['variance'][1]),off_diagonal_used_in_R=False))
            for label in [0,1]:
                names=scores.loc[scores.binary_label.eq(label),'source_id'];part=profiles[profiles.source_id.isin(names)].groupby('source_id')[['loss_base','loss_own','loss_both','benefit']].mean()
                if len(part)!=len(names):raise ValueError('Missing test participant profiles')
                for method in ['base','own','both']:
                    values=part['loss_'+method];mse.append(dict(outer_fold=fold,group='healthy' if label==0 else 'stroke',participants=len(part),method=method,mean_participant_mse_deg2=float(values.mean()),median_participant_mse_deg2=float(values.median()),mean_own_minus_both_benefit_deg2=float(part.benefit.mean())))
    q=pd.DataFrame(quantiles);r=pd.DataFrame(references);m=pd.DataFrame(mse)
    q.to_csv(OUT/'normal_score_quantiles.csv',index=False);r.to_csv(OUT/'external_reference_correlations.csv',index=False);m.to_csv(OUT/'external_heldout_mse.csv',index=False)
    record=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),status='complete_post_outcome_descriptive_only',producer_sha256=sha(__file__),inputs=inputs,all_planned_folds=[0,1,2,3],quantile_method='numpy linear empirical interpolation; p99 is not a well-estimated population tail with7or32 test normals',normal_counts={'GAVD_U_U':7,'external_angles':32},score_definitions='M and R are frozen dimension-normalized distances from different reference constructions. This audit does not recalibrate them or convert them to probabilities.',causal_boundary='Observed score-scale/tail differences and reference component correlation do not establish the cause of reduced fusion AUROC. No counterfactual calibration or refitted detector was evaluated.',outputs={n:sha(OUT/n) for n in ['normal_score_quantiles.csv','external_reference_correlations.csv','external_heldout_mse.csv']})
    (OUT/'audit_manifest.json').write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
    print(q[q.score.isin(['M','R'])].to_string(index=False));print(r.to_string(index=False));print(m.pivot(index=['outer_fold','group','participants'],columns='method',values='mean_participant_mse_deg2').to_string())
if __name__=='__main__':main()
