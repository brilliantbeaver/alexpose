"""Immutable pre-test identity correction from provenance-verified real poses."""
from pathlib import Path
import hashlib,json,time
import numpy as np
import pandas as pd
from .settings import configure
from .data import build_inventory,assign_splits,save_inventory,sha256_file
from .pose import make_windows,_ADAPTER_SHA256
from .extended_data import gate_person
from .workflow import load_prepared,json_write
EXCLUDED={'CMls0LwK5OU':'validation scene cuts and person identity switches','eBvzFkcvScg':'training person identity switches'}
def digest_array(a):
    a=np.ascontiguousarray(a)
    return hashlib.sha256(str((a.shape,a.dtype)).encode()+a.tobytes()).hexdigest()
def main():
    project,repo=configure();runs=project/'artifacts/runs';started=time.monotonic()
    ledger=dict(identity='manual-source-exclusion-before-joint-test-v1',date='2026-11-16',evidence_sha256=sha256_file(project/'artifacts/research_analysis/person_review.png'),excluded_sources=EXCLUDED,reviewed_frames={'CMls0LwK5OU':[1,33,63,493,525,555,758,790,820,1667,1699,1729],'eBvzFkcvScg':[27,59,89,116,148,178,478,510,540]},reviewers=['coordinator','independent literature agent'],reason='Visible discontinuities and identity changes invalidate person-specific motion; no diagnosis or role inferred from appearance.',scope='All sequences and windows of both sources, all study cohorts; before final test metric access.')
    json_write(project/'docs/source_exclusions.json',ledger)
    old=runs/'extended-input-v1';new=runs/'extended-input-clean-v1';new.mkdir()
    x,v,m=load_prepared(old);keep=~m.source_id.isin(EXCLUDED);ids=np.flatnonzero(keep);clean=m.loc[keep].reset_index(drop=True)
    np.savez_compressed(new/'windows.npz',x=x[keep],valid=v[keep]);clean.to_csv(new/'windows.csv',index=False)
    pd.DataFrame({'new_row':range(len(ids)),'parent_row':ids}).to_csv(new/'row_map.csv',index=False)
    cert=dict(identity='inherited-normal-forecast-checkpoints-v1',parent=str(old),derivative=str(new),exclusions_sha256=sha256_file(project/'docs/source_exclusions.json'),checks={},lineage='No forecast retraining claimed. Original checkpoints and stochastic validation predictions inherited. Only excluded abnormal train/validation rows are removed.',parent_preparation_sha256=sha256_file(old/'preparation.json'))
    for name,split in [('normal_train','train'),('normal_validation','validation'),('all_test','test')]:
        mask=m.split==split;cmask=clean.split==split
        if name!='all_test':mask&=m.label=='normal';cmask&=clean.label=='normal'
        assert np.array_equal(x[mask],x[keep][cmask]) and np.array_equal(v[mask],v[keep][cmask])
        pd.testing.assert_frame_equal(m.loc[mask].reset_index(drop=True),clean.loc[cmask].reset_index(drop=True),check_exact=True)
        cert['checks'][name]=dict(equal=True,rows=int(mask.sum()),sources=int(m.loc[mask].source_id.nunique()),x_sha256=digest_array(x[mask]),valid_sha256=digest_array(v[mask]),metadata_sha256=hashlib.sha256(m.loc[mask].reset_index(drop=True).to_csv(index=False).encode()).hexdigest())
    cert['retained_row_map_sha256']=sha256_file(new/'row_map.csv');cert['all_retained_x_sha256']=digest_array(x[ids]);cert['all_retained_valid_sha256']=digest_array(v[ids])
    json_write(new/'preparation.json',dict(mode='immutable-person-reviewed-derivative',frames=32,context_frames=16,n_windows=len(clean),n_sources=int(clean.source_id.nunique()),data_sha256=sha256_file(new/'windows.npz'),metadata_sha256=sha256_file(new/'windows.csv'),parent_preparation_sha256=cert['parent_preparation_sha256'],source_exclusions_sha256=cert['exclusions_sha256'],adapter_sha256=sha256_file(__file__)))
    filtered=runs/'revision-validation-clean-v1';filtered.mkdir();lineage=[]
    for parent in ['revision-development-v1','objective-audit-development-v1']:
        for path in sorted((runs/parent).rglob('*.csv')):
            t=pd.read_csv(path)
            if not {'source_id','split'}.issubset(t.columns) or not (t.split=='validation').all():continue
            target=filtered/parent/path.relative_to(runs/parent);target.parent.mkdir(parents=True,exist_ok=True);t.loc[~t.source_id.isin(EXCLUDED)].to_csv(target,index=False)
            lineage.append(dict(parent=str(path.relative_to(project)),parent_sha256=sha256_file(path),derivative=str(target.relative_to(project)),derivative_sha256=sha256_file(target),removed_rows=int(t.source_id.isin(EXCLUDED).sum())))
    json_write(filtered/'lineage.json',lineage);cert['validation_lineage_sha256']=sha256_file(filtered/'lineage.json')
    for parent in ['revision-development-v1','objective-audit-development-v1']:cert[parent+'_freeze_sha256']=sha256_file(runs/parent/'freeze.json')
    json_write(project/'docs/forecast_reuse_certificate.json',cert)
    table,audit=build_inventory(repo);table=assign_splits(table,seed=27);output=runs/'pilot-input-clean-v1';output.mkdir();save_inventory(table,audit,output)
    import ambient.pose.keypoint_extractor as extractor
    mh=sha256_file(repo/'data/models/pose_landmarker_lite.task');eh=sha256_file(extractor.__file__)
    video_hashes={};all_x=[];all_v=[];all_m=[];attrition=[];lineage=[];gates=[]
    for row in table.loc[table.eligible].itertuples():
        if row.source_id in EXCLUDED:
            attrition.append(dict(sequence_id=row.sequence_id,source_id=row.source_id,label=row.label,split=row.split,windows=0,exclusion_reason='manual_source_identity_failure'));continue
        video=Path(row.video_path)
        if not video.is_absolute():video=repo/video
        if row.source_id not in video_hashes:video_hashes[row.source_id]=sha256_file(video)
        expected=dict(csv_sha256=row.csv_sha256,video_sha256=video_hashes[row.source_id],model_sha256=mh,extractor_sha256=eh,adapter_sha256=_ADAPTER_SHA256,frame_step=2,max_frames=16)
        pose=None
        for cache in sorted((project/'artifacts/poses').glob(row.sequence_id+'-*.npz')):
            with np.load(cache,allow_pickle=False) as z:
                identity=json.loads(str(z['provenance_json']))
                if any(identity.get(k)!=value for k,value in expected.items()) or identity.get('schema')!=1:continue
                pose={k:z[k].copy() for k in z.files};parent_hash=sha256_file(cache);break
        if pose is None:raise ValueError('No verified raw pose cache for '+row.sequence_id)
        rows=pd.read_csv(repo/row.csv_path,keep_default_na=False).sort_values('frame_num').iloc[::2].iloc[:16]
        assert np.array_equal(rows.frame_num.to_numpy(int),pose['frame_numbers'])
        width,height=pose['image_dimensions'];reasons=[]
        for i,record in enumerate(rows.itertuples()):
            pose['confidence'][i],reason=gate_person(pose['xy'][i],pose['confidence'][i],record.bbox,record.vid_info,int(width),int(height));reasons.append(reason)
            gates.append(dict(source_id=row.source_id,sequence_id=row.sequence_id,frame_number=int(record.frame_num),reason=reason))
        identity.update(schema='person-reviewed-short-v1',parent_pose_sha256=parent_hash,clean_adapter_sha256=sha256_file(__file__),gate_adapter_sha256=sha256_file(Path(__file__).with_name('extended_data.py')),source_exclusions_sha256=cert['exclusions_sha256'])
        fp=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest();pose.update(fingerprint=np.array(fp),provenance_json=np.array(json.dumps(identity,sort_keys=True)),gate_reason=np.asarray(reasons))
        target=project/'artifacts/poses'/f'{row.sequence_id}-{fp[:20]}.npz';np.savez_compressed(target,**pose)
        xx,vv,mm=make_windows(pose,frames=16,stride=8,context_frames=8)
        attrition.append(dict(sequence_id=row.sequence_id,source_id=row.source_id,label=row.label,split=row.split,windows=len(xx),exclusion_reason='' if len(xx) else 'pose_quality_or_person_gate'))
        lineage.append(dict(sequence_id=row.sequence_id,parent_pose_sha256=parent_hash,clean_pose_sha256=sha256_file(target),pose_fingerprint=fp))
        if len(xx):mm['split']=row.split;all_x.append(xx);all_v.append(vv);all_m.append(mm)
    xx=np.concatenate(all_x);vv=np.concatenate(all_v);mm=pd.concat(all_m,ignore_index=True);assert not mm.source_id.isin(EXCLUDED).any()
    np.savez_compressed(output/'windows.npz',x=xx,valid=vv);mm.to_csv(output/'windows.csv',index=False)
    pd.DataFrame(attrition).to_csv(output/'attrition.csv',index=False);pd.DataFrame(gates).to_csv(output/'person_gate_frames.csv',index=False)
    json_write(output/'pose_lineage.json',lineage);json_write(output/'source_exclusions.json',ledger)
    json_write(output/'preparation.json',dict(mode='all-sequences-person-reviewed-short',frames=16,context_frames=8,max_frames=16,frame_step=2,n_windows=len(xx),n_sources=int(mm.source_id.nunique()),seconds=time.monotonic()-started,data_sha256=sha256_file(output/'windows.npz'),metadata_sha256=sha256_file(output/'windows.csv'),source_exclusions_sha256=cert['exclusions_sha256'],pose_lineage_sha256=sha256_file(output/'pose_lineage.json'),adapter_sha256=sha256_file(__file__),pose_model_sha256=mh,note='Real raw MediaPipe calls reused after full provenance verification; fresh CSV person gate applied before causal 8-frame normalization.'))
    print(json.dumps({'short_windows':len(xx),'short_sources':int(mm.source_id.nunique()),'long_windows':len(clean),'long_sources':int(clean.source_id.nunique()),'reuse_checks':cert['checks']},indent=2))
if __name__=='__main__':main()
