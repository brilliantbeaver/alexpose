"""Attach verified full-duration execution evidence to the data audit."""
from pathlib import Path
from collections import Counter
import json
import numpy as np
import pandas as pd

PROJECT=Path(__file__).resolve().parents[2]
run=PROJECT/'artifacts/runs/restart-full10hz-v1'
report=json.loads((run/'preparation.json').read_text())
verification=json.loads((PROJECT/'artifacts/restart_audit/full_cohort_verification.json').read_text())
meta=pd.read_csv(run/'windows.csv')
attrition=pd.read_csv(run/'attrition.csv')
rejections=Counter()
for value in attrition.rejection_counts:rejections.update(json.loads(value))
window_rejections={k:v for k,v in rejections.items() if k!='fewer_than_24_sampled_frames_in_contiguous_span'}
assert int(attrition.candidate_windows.sum())==len(meta)+sum(window_rejections.values())
with np.load(run/'windows.npz',allow_pickle=False) as data:
    quantiles={str(q):float(np.quantile(np.abs(data['x'])[data['valid']],q)) for q in [0,.5,.9,.99,1]}
counts=pd.read_csv(run/'person_gate_frames.csv').reason.value_counts()
summary=dict(preparation=report,verification=verification,
    candidate_windows=int(attrition.candidate_windows.sum()),window_rejection_counts=window_rejections,
    sequences_without_duration_support=int(rejections['fewer_than_24_sampled_frames_in_contiguous_span']),
    person_gate_counts={str(k):int(v) for k,v in counts.items()},
    absolute_normalized_coordinate_quantiles=quantiles,
    final_horizon_seconds={str(q):float(meta.final_horizon_seconds.quantile(q)) for q in [0,.5,1]},
    maximum_grid_error_seconds=float(meta.maximum_grid_error_seconds.max()))
path=PROJECT/'docs/restart/data_audit.json';audit=json.loads(path.read_text());audit['full10hz_execution']=summary
path.write_text(json.dumps(audit,indent=2)+'\n')
table='\n'.join(f'| {r["label"]} | {r["windows"]} | {r["sequences"]} | {r["sources"]} |' for r in report['by_label'])
text=f'''\n## Completed full-duration extraction

The frozen `restart-full10hz-v1` run made **{report['actual_model_calls_this_invocation']:,} actual MediaPipe model calls** across all {report['selected_sources']} candidate sources in {report['seconds']:.2f} seconds. It retained **{report['n_windows']:,} windows from {report['n_sequences']} sequences and {report['n_sources']} source IDs**. No new source-video download was needed, and no earlier pose cache was reused in this run.

| Label | Retained windows | Retained sequences | Retained source IDs |
|---|---:|---:|---:|
{table}

There were {int(attrition.candidate_windows.sum()):,} duration-eligible candidate windows before quality filtering and deduplication. {summary['sequences_without_duration_support']} sequences had no complete 24-observation window. The run records every rejection and gate outcome in its attrition and person-gate tables. The maximum retained-window deviation from the nominal 10 Hz grid was {summary['maximum_grid_error_seconds']:.4f} seconds, and the final forecast horizon ranged from {meta.final_horizon_seconds.min():.4f} to {meta.final_horizon_seconds.max():.4f} seconds.

The complete replay reconstructed all {report['n_windows']:,} retained windows from their saved native-frame pose records with **zero array difference**. It checked every target-support mask, source partition, frame tuple, timestamp, window ID, source-pose payload hash, artifact hash and attrition count. Five focused contract tests passed. The verifier reuses the shared normalization API and therefore establishes artifact consistency, not independent clinical validation. The exact running extraction source is preserved under the run's `source_snapshot/` directory.

The prepared arrays and metadata are ready for a separately identified source-grouped model study. This expanded observation cohort retains the generalizability limits described above. More dependent windows do not establish more independent patients, and source groups do not establish verified subject identity.
'''
md=PROJECT/'docs/restart/data_audit.md'
existing=md.read_text().split('\n## Completed full-duration extraction')[0]
md.write_text(existing+text)
(PROJECT/'artifacts/restart_audit/full10hz_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k not in ['preparation','verification']},indent=2))
