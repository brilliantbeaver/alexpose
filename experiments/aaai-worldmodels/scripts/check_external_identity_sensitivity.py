"""Apply a pre-score visual-QA exclusion to fixed external participant scores."""
from pathlib import Path
import json,sys,hashlib
import pandas as pd

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from worldmodels import v2_evaluation as ev

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def run():
    protocol=PROJECT/'docs/reference_v2/external_identity_sensitivity_protocol.json'
    rule=json.loads(protocol.read_text())
    if rule['status']!='declared_before_external_efficacy_scoring':raise ValueError('Missing pre-score declaration')
    source=PROJECT/'artifacts/runs/reference-v2-transfer-v1'
    scores=pd.read_csv(source/'participant_scores.csv');excluded=rule['excluded_participant_ids']
    if not set(excluded)<=set(scores.source_id):raise ValueError('Flagged participant absent from primary support')
    kept=scores[~scores.source_id.isin(excluded)].copy()
    output=PROJECT/'artifacts/runs/reference-v2-identity-sensitivity-v1'
    if output.exists():raise FileExistsError('Use a new analysis identity')
    output.mkdir(parents=True);kept.to_csv(output/'participant_scores.csv',index=False)
    compare=lambda frame,ref:ev.paired_within_fold_bootstrap(frame,'two_sided',ref,reps=5000,seed=5703)
    conditions=[]
    for label in sorted(set(kept.label)-{'normal'}):
        part=kept[kept.label.isin(['normal',label])]
        conditions.append({'label':label,'normal':int(part.label.eq('normal').sum()),
            'abnormal':int(part.label.ne('normal').sum()),'paired_comparison':compare(part,'low_motion')})
    result={'identity':'reference-v2-external-identity-sensitivity-v1','status':'complete',
        'primary_population':'Preserved in reference-v2-transfer-v1; this is an additional sensitivity.',
        'excluded_participant_ids':excluded,'participants':len(kept),'participants_by_label':kept.label.value_counts().to_dict(),
        'primary_anomaly':compare(kept,'low_motion'),'posture_calibration_contrast':compare(kept,'posture_two_sided'),
        'condition_results':conditions,'inputs':{'primary_results_sha256':sha(source/'results.json'),
            'source_scores_sha256':sha(source/'participant_scores.csv'),'declaration_sha256':sha(protocol),'code_sha256':sha(__file__)},
        'limitation':'Exclusion changes severe-Parkinson coverage. The identity-uncertainty flag preceded all external efficacy scores and is not clinical reassessment.'}
    (output/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    (output/'artifact_hashes.json').write_text(json.dumps({p.name:sha(p) for p in output.iterdir() if p.is_file() and p.name!='artifact_hashes.json'},indent=2)+'\n')
    print(json.dumps({'status':'complete','participants':len(kept),'excluded':excluded}))

if __name__=='__main__':run()
