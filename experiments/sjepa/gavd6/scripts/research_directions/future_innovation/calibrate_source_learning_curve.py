"""Run fixed synthetic source learning curves through production nested selection.

Seeds and tolerances are fixed here before execution. Synthetic positive controls
never authorize a scientific advance. No real-cohort target is opened.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'src'))
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from gavd6_sjepa.research_directions.future_innovation.fi_contracts import write_json, DIRECT_ARMS, sha256_file
from gavd6_sjepa.research_directions.future_innovation.fi_joint_calibration import fixture
from gavd6_sjepa.research_directions.future_innovation.fi_joint_models import JointModelContract
from gavd6_sjepa.research_directions.future_innovation_scaling.fi_scaling_nested import nested_partition
from gavd6_sjepa.research_directions.future_innovation_scaling.fi_scaling_cohort import POLICY, reserve_sources, make_plan, fingerprint
from gavd6_sjepa.research_directions.future_innovation_scaling.fi_scaling_training import indices, prediction_payload, compute_curve


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',required=True,type=Path)
    args=parser.parse_args(); root=args.output_root
    if (root/'calibration.json').exists(): raise ValueError('Calibration already exists; preserve it and choose another root')
    root.mkdir(parents=True,exist_ok=True)
    design={'fixtures':[['rgb',261401],['rgb',261402],['rgb',261403],['temporal',261404]],
            'irrelevant_mean_increment_max':.03,'planted_matched_increment_min':.10,'planted_real_minus_shuffle_min':.10,
            'policy':POLICY,'software':fingerprint()}
    write_json(root/'design.json',design)
    results=[]
    with threadpool_limits(1):
        for kind,seed in design['fixtures']:
            cohort,arrays,schema=fixture(seed,kind)
            roster=reserve_sources(cohort.assign(sequence_id=cohort.window_id),set(cohort.video_id))
            cohort=cohort.drop(columns='outer_fold').merge(roster[['video_id','source_group','outer_fold']],on='video_id')
            plan=make_plan(cohort,roster)
            directory=root/f'{kind}-{seed}'; directory.mkdir()
            write_json(directory/'plan.json',plan)
            payloads={}; selection=[]
            for identity,spec in plan['fits'].items():
                train,test=indices(cohort,spec['train_windows']),indices(cohort,spec['test_windows'])
                result=nested_partition(cohort,arrays,train,test,schema,JointModelContract(),penalty_reference_windows=40,group_column='source_group')
                if not all(x['complete'] for x in result['selection'].values()): raise ValueError('Calibration candidate failed')
                path=directory/f'{identity}.joblib'
                joblib.dump({'baseline':result['baseline'],'models':result['models']},path)
                payloads[identity]=prediction_payload(joblib.load(path),cohort,arrays,train,test)
                for arm in DIRECT_ARMS: np.testing.assert_array_equal(payloads[identity][arm],result['predictions'][arm])
                selection.append({'fit_id':identity,'candidates':result['selection']})
            report,metrics,bootstrap,raw=compute_curve(plan,payloads,synthetic=True)
            write_json(directory/'report.json',report)
            write_json(directory/'selection.json',selection)
            metrics.to_csv(directory/'per-subset.csv',index=False)
            bootstrap.to_parquet(directory/'bootstrap.parquet',index=False)
            results.append({'kind':kind,'seed':seed,'contrasts':report['contrasts'],'unique_fits':len(payloads)})
            print(f'Calibrated {kind}/{seed}: {len(payloads)} distinct source-subset fits',flush=True)
    def value(result,size,contrast):
        return next(c['estimate'] for c in result['contrasts'] if c['size']==size and c['contrast']==contrast)
    null=[r for r in results if r['kind']=='rgb']; positive=results[-1]
    checks={}
    for size in ('40','all'):
        checks[f'irrelevant_mean_increment_{size}']=float(np.mean([value(r,size,'matched_increment') for r in null]))<=.03
        checks[f'planted_increment_{size}']=value(positive,size,'matched_increment')>.10
        checks[f'planted_shuffle_reduction_{size}']=value(positive,size,'real_minus_shuffle')>.10
    if design['software']!=fingerprint():
        raise ValueError('Implementation changed during calibration; preserve this attempt and rerun')
    output=dict(status='synthetic_calibration_only',passed=all(checks.values()),checks=checks,results=results,
                software=design['software'],design_sha256=sha256_file(root/'design.json'),
                limitations='Fixed synthetic curves test software; no real-data scaling or student improvement is measured.')
    write_json(root/'calibration.json',output)
    print(json.dumps({'passed':output['passed'],'checks':checks},indent=2))
    return 0 if output['passed'] else 2

if __name__=='__main__': raise SystemExit(main())
