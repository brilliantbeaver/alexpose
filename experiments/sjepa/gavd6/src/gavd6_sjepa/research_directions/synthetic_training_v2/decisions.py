"""Conservative source adjudication, enabled only by a pre-fit calibration record."""
import json
from pathlib import Path
import numpy as np
from .evaluation import aggregate_metrics,paired_cluster_bootstrap,scientific_gate
from .contracts import sha256_file,SCHEMA


def load_decision_spec(path):
    path=Path(path)
    spec=json.loads(path.read_text())
    required={'calibration_artifact','calibration_artifact_sha256','minimum_people','minimum_seeds','amplitude_error_max','event_timing_max_s','coordinate_margin','displacement_margin','clean_degradation_max'}
    if required-set(spec):raise ValueError('Decision specification lacks actual calibration artifact/margins')
    artifact=Path(spec['calibration_artifact'])
    if not artifact.is_absolute():artifact=path.parent/artifact
    if not artifact.is_file() or sha256_file(artifact)!=spec['calibration_artifact_sha256']:
        raise ValueError('Calibration artifact missing or SHA256 mismatch')
    calibration=json.loads(artifact.read_text())
    if (calibration.get('schema')!=SCHEMA or calibration.get('samples')!=64 or calibration.get('hz')!=25
        or calibration.get('evidence_status') not in {'source-run','real-development'}
        or calibration.get('candidate_outputs_used') is not False
        or not calibration.get('reference_manifest_sha256')
        or not calibration.get('reviewer')):
        raise ValueError('Calibration artifact lacks compatible independently reviewed reference provenance')
    numeric=required-{'calibration_artifact','calibration_artifact_sha256'}
    for name in numeric:
        if not np.isfinite(spec[name]) or spec[name]<0:raise ValueError('Finite nonnegative calibration values required')
        if calibration.get('decision_values',{}).get(name)!=spec[name]:raise ValueError('Decision margin differs from retained calibration')
    if spec['minimum_people']<2 or spec['minimum_seeds']<3:raise ValueError('Decisive gate requires at least2 independent groups and3 training seeds')
    return spec


def adjudicate_gate_b(metrics,cfg):
    status=str(metrics.evidence_status.iloc[0])
    pending=dict(status='insufficient_evidence',evidence_status=status)
    if status=='fixture-tested':return {**pending,'reason':'Fixtures cannot authorize a scientific advance'}
    if not cfg.decision_spec:return {**pending,'reason':'Preservation margins/support minima need independent development calibration before adjudication'}
    spec=load_decision_spec(cfg.decision_spec)
    if metrics.seed.nunique()<spec['minimum_seeds']:return {**pending,'reason':'Decisive training-seed minimum unmet'}
    comparators=('coordinate','direct','smoothnet','initialized')
    results=[]
    for (extractor,seed),group in metrics.groupby(['extractor','seed']):
        if group.person_id.nunique()<spec['minimum_people'] or not {'paired_jepa',*comparators}<=set(group.method):
            return {**pending,'reason':'Independent group count or required comparators missing'}
        for comparator in comparators:
            coord=paired_cluster_bootstrap(group,'paired_jepa',comparator,draws=cfg.bootstrap_draws,seed=cfg.seed)
            displacement=paired_cluster_bootstrap(group,'paired_jepa',comparator,'displacement_nle',draws=cfg.bootstrap_draws,seed=cfg.seed)
            amp=aggregate_metrics(group[group.method.eq('paired_jepa')],'amplitude_error').value.iloc[0]
            timing=aggregate_metrics(group[group.method.eq('paired_jepa')],'event_timing_mae_s').value.iloc[0]
            preservation=None
            if displacement.get('relative_improvement') is not None and np.isfinite([amp,timing]).all():
                preservation=bool(displacement['relative_improvement']>=spec['displacement_margin'] and amp<=spec['amplitude_error_max'] and timing<=spec['event_timing_max_s'])
            clean=group[group.variant.eq('clean')];retention=None
            if {'paired_jepa','unchanged'}<=set(clean.method):
                values=aggregate_metrics(clean).set_index('method').value
                baseline=values['unchanged'];candidate=values['paired_jepa']
                if np.isfinite([baseline,candidate]).all() and baseline>1e-8:
                    retention=bool(candidate/baseline-1<=spec['clean_degradation_max'])
            gate=scientific_gate(evidence_status=status,relative_improvement=coord.get('relative_improvement'),ci95=coord.get('relative_ci95'),
                meaningful_margin=spec['coordinate_margin'],preservation_ok=preservation,clean_retention_ok=retention)
            results.append(dict(extractor=extractor,seed=int(seed),comparator=comparator,**gate))
    verdict='fail' if any(r['status']=='fail' for r in results) else 'insufficient_evidence' if any(r['status']!='pass' for r in results) else 'pass'
    return dict(status=verdict,evidence_status=status,reason='Conjunction across prespecified comparators, seeds and extractors; synthetic proxy only',comparisons=results,
                real_transfer_authorized=False,calibration=spec)
