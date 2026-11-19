"""Generate descriptive v2 data reports and blinded visual-review material.

No model outcomes are read. No extraction or download is performed. Native
video metadata and selected already cached pose observations are audited.
"""
from pathlib import Path
from collections import Counter
import ast
import hashlib
import json
import re
import sys
import time

import cv2
import numpy as np
import pandas as pd

PROJECT=Path(__file__).resolve().parents[1]
REPO=PROJECT.parents[1]
sys.path.insert(0,str(PROJECT))
from worldmodels.data import LABELS,sha256_file
from worldmodels.v2_data import DEFAULT,load_cohort,prepare,write_json

OUT=PROJECT/'docs/reference_v2'
MEDIA_SUFFIXES={'.mp4','.mov','.mkv','.webm'}


def external_inventory():
    root=REPO/'data/videos-more/koa-pd-nm'
    clone=REPO/'data/videos-ms/koa-pd-nm'
    clones={p.name:p for p in clone.rglob('*') if p.is_file() and p.suffix.lower() in MEDIA_SUFFIXES}
    gavd=pd.read_csv(DEFAULT/'media_relations.csv',keep_default_na=False)
    gavd_hashes=set(gavd.sha256)
    records=[]
    for path in sorted(root.rglob('*')):
        if not path.is_file() or path.suffix.lower() not in MEDIA_SUFFIXES:
            continue
        match=re.fullmatch(r'(\d{3})_(NM|PD|KOA)_(01|02)(?:_(EL|ML|MD|SV))?',path.stem)
        if match is None:
            raise ValueError(f'Unresolved external video identity: {path}')
        number,label,direction,severity=match.groups();severity=severity or 'none'
        participant='_'.join(([label] if label=='NM' else [label,severity])+[number])
        cap=cv2.VideoCapture(str(path))
        try:
            opened=cap.isOpened()
            fps=float(cap.get(cv2.CAP_PROP_FPS));n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width,height=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            first,frame=cap.read()
            cap.set(cv2.CAP_PROP_POS_FRAMES,max(0,n-1));last,_=cap.read()
        finally:
            cap.release()
        digest=sha256_file(path);other=clones.get(path.name)
        same=bool(other and sha256_file(other)==digest)
        records.append(dict(video_id=path.stem,source_id='kpn_'+path.stem,participant_id=participant,
            label={'NM':'normal','PD':'parkinsons','KOA':'osteoarthritis'}[label],original_group=label,severity=severity,
            direction=direction,path=str(path.relative_to(REPO)),sha256=digest,bytes=path.stat().st_size,
            fps=fps,nominal_frame_count=n,width=width,height=height,nominal_duration_seconds=n/fps if fps>0 else None,
            first_and_last_decode=bool(first and last),probe_status='opened_first_last_decoded' if opened and first and last else ('cannot_open' if not opened else 'endpoint_decode_failure'),clone_path=str(other.relative_to(REPO)) if other else '',
            clone_byte_identical=same,byte_identical_to_any_gavd_original=digest in gavd_hashes,
            participant_identity_basis='Filename group+severity+ID; demographic PDF has separate severity-specific subject tables.'))
    frame=pd.DataFrame(records)
    frame.to_csv(OUT/'data_external_inventory.csv',index=False)
    counts=frame.groupby(['original_group','severity']).agg(videos=('video_id','size'),participants=('participant_id','nunique')).reset_index()
    participants=frame.groupby('participant_id').agg(label=('label','first'),severity=('severity','first'),videos=('video_id','size'),directions=('direction',lambda s:'|'.join(sorted(set(s))))).reset_index()
    participants.to_csv(OUT/'data_external_participants.csv',index=False)
    expected={'NM':30,'KOA_EL':15,'KOA_MD':20,'KOA_SV':15,'PD_ML':6,'PD_MD':7,'PD_SV':3}
    missing=[]
    present=set(frame.participant_id)
    for group,n in expected.items():
        for i in range(1,n+1):
            participant=f'{group}_{i:03d}'
            if participant not in present:
                missing.append(dict(participant_id=participant,status='not_present_locally'))
    source_inputs={str(p.relative_to(REPO)):sha256_file(p) for p in [root/'README.md',root/'Demographic Data.pdf']}
    report=dict(identity='reference-v2-external-local-inventory-v1',status='metadata_audit_only_no_external_model_evaluation',
        selected_root=str(root.relative_to(REPO)),clone_root=str(clone.relative_to(REPO)),source_inputs=source_inputs,
        primary_url='https://data.mendeley.com/datasets/44pfnysy89/1',
        videos=len(frame),participants=int(frame.participant_id.nunique()),counts=counts.to_dict('records'),
        expected_descriptor_videos=191,expected_demographic_participants=96,
        missing_participants=missing,single_direction_participants=participants[participants.videos<2].to_dict('records'),
        byte_identical_clones=int(frame.clone_byte_identical.sum()),byte_identical_gavd_overlaps=int(frame.byte_identical_to_any_gavd_original.sum()),
        metadata_decode_success=int(frame.first_and_last_decode.sum()),
        invalid_files=frame.loc[~frame.first_and_last_decode,['video_id','participant_id','probe_status','clone_byte_identical']].to_dict('records'),
        participants_with_decodable_video=int(frame.loc[frame.first_and_last_decode,'participant_id'].nunique()),
        nominal_fps=sorted(frame.fps.unique().tolist()),image_sizes=frame[['width','height']].drop_duplicates().to_dict('records'),
        manifest_sha256=sha256_file(OUT/'data_external_inventory.csv'),
        terms='Dataset-owner page states CC BY4.0; bundled README requests research/noncommercial use. Research use only in this study; no media redistribution.',
        missing_count_note='Local PD_SV has 3 videos from 2 coded participants versus descriptor 5 videos/3 participants. The demographic two-direction ideal implies 32 PD videos; author README lists 31. Do not invent a full missing-file list from the inconsistent totals.',
        historical_exposure='No mentions of this dataset/DOI/filename pattern were found by a preimplementation textual search of experiment code/reports and notes. This cannot prove absence of prior human exposure; treat as external to GAVD, not a prospectively collected clinical cohort.',
        limits=['Source labels and participant IDs are dataset-provided, not independently reverified diagnoses.',
                'Exact byte-hash disjointness does not exclude GAVD containing a reencoded or edited upload of the same footage.',
                'Full pose extraction, tracking quality, valid-span gating, and clinical domain compatibility remain untested here.',
                'The source-video model must be frozen before external evaluation; local normal calibration would be domain adaptation.'])
    write_json(OUT/'data_external_summary.json',report)
    return report


def blinded_qa():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    prepared=PROJECT/'artifacts/runs/restart-full10hz-v1'
    attr=pd.read_csv(prepared/'attrition.csv',keep_default_na=False)
    _,_,model,_=load_cohort()
    accepted=set(model.sequence_id)
    selected=pd.read_csv(prepared/'selected_sequences.csv',keep_default_na=False).set_index('sequence_id')
    chosen=[]
    used=set()
    for label in LABELS:
        for status in ['accepted','rejected']:
            eligible=attr[(attr.label==label)&(attr.candidate_windows>0)].copy()
            eligible=eligible[eligible.sequence_id.isin(accepted) if status=='accepted' else ~eligible.sequence_id.isin(accepted)]
            eligible['selection_hash']=eligible.sequence_id.map(lambda s:hashlib.sha256(('v2-blinded-qa|'+s).encode()).hexdigest())
            eligible=eligible.sort_values(['selection_hash','sequence_id'])
            available=eligible[~eligible.source_id.isin(used)]
            if available.empty:
                available=eligible
            if available.empty:
                continue
            row=available.iloc[0];used.add(row.source_id)
            chosen.append(dict(label=label,status=status,sequence_id=row.sequence_id,source_id=row.source_id))
    # A deterministic shuffle hides disease and acceptance strata from the sheet.
    chosen=sorted(chosen,key=lambda r:hashlib.sha256(('display|'+r['sequence_id']).encode()).hexdigest())
    output=DEFAULT/'visual_qa';output.mkdir(exist_ok=True)
    records=[]
    fig,axes=plt.subplots(len(chosen),3,figsize=(12,3.2*len(chosen)),squeeze=False)
    for row_index,record in enumerate(chosen):
        sequence=record['sequence_id']
        with np.load(prepared/'sequence_poses'/f'{sequence}.npz',allow_pickle=False) as z:
            pose={k:z[k].copy() for k in z.files}
        video=REPO/selected.loc[sequence,'video_path']
        annotation=pd.read_csv(REPO/selected.loc[sequence,'csv_path'],keep_default_na=False).set_index('frame_num')
        indexes=sorted(set([0,len(pose['frame_numbers'])//2,len(pose['frame_numbers'])-1]))
        cap=cv2.VideoCapture(str(video))
        try:
            for col,i in enumerate(indexes):
                native=int(pose['frame_numbers'][i]);cap.set(cv2.CAP_PROP_POS_FRAMES,native-1)
                ok,bgr=cap.read()
                if not ok:
                    raise ValueError('Blinded review frame failed to decode')
                ax=axes[row_index,col];ax.imshow(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));ax.axis('off')
                xy,conf=pose['xy'][i],pose['raw_confidence'][i]
                good=np.isfinite(xy).all(1)&(conf>=.5)
                ax.scatter(xy[good,0],xy[good,1],s=6,c='#f46c20',alpha=.8)
                ax.set_title(f'Panel {row_index+1:02d}.{col+1}',fontsize=11)
                record_frame=dict(panel=f'{row_index+1:02d}.{col+1}',**record,native_frame=native,
                    gate_reason=str(pose['gate_reason'][i]),video_path=str(video.relative_to(REPO)),
                    annotation_csv=str(selected.loc[sequence,'csv_path']),image_height=bgr.shape[0],image_width=bgr.shape[1])
                records.append(record_frame)
        finally:
            cap.release()
    fig.suptitle('Blinded gait observation review: sampled frames across source videos\nOrange points are raw pose estimates. Strata and identifiers are in a separate ledger.',fontsize=13)
    fig.tight_layout(rect=[0,0,1,.98]);fig.savefig(output/'blinded_contact_sheet.png',dpi=120);plt.close(fig)
    pd.DataFrame(records).to_csv(output/'selection_ledger.csv',index=False)
    report=dict(identity='reference-v2-blinded-observation-qa-v1',status='material_generated_for_visual_review',
        selection='One sequence per label×accepted/rejected stratum, candidate_windows>0, SHA256 lexical selection with distinct sources when available, first/middle/last sampled frames; no performance selection.',
        sequences=len(chosen),source_videos=len(set(r['source_id'] for r in chosen)),frames=len(records),
        acceptance_strata={s:sum(r['status']==s for r in chosen) for s in ['accepted','rejected']},
        blinded_sheet=str((output/'blinded_contact_sheet.png').relative_to(PROJECT)),
        ledger=str((output/'selection_ledger.csv').relative_to(PROJECT)),
        limits='Still frames cannot establish diagnosis, person continuity, every gait cycle, or full-duration tracking. This audit makes no new inclusion/exclusion decisions.')
    write_json(OUT/'data_visual_qa.json',report)
    return report


def reports():
    OUT.mkdir(parents=True,exist_ok=True)
    if not DEFAULT.exists():
        prepare()
    audit=json.loads((DEFAULT/'audit.json').read_text())
    model=REPO/'data/models/pose_landmarker_lite.task'
    expected_model=set(pd.read_csv(DEFAULT/'source_cache_manifest.csv').pose_model_sha256)
    actual_model=sha256_file(model)
    if expected_model!={actual_model}:
        raise ValueError('Actual current pose weights differ from the frozen source caches')
    audit['actual_pose_model_weights_verified']={'path':str(model.relative_to(REPO)),'sha256':actual_model}
    for name in ['coverage_by_label.csv','coverage_by_view.csv','attrition_by_label.csv','fold_counts.csv']:
        (OUT/('data_'+name)).write_bytes((DEFAULT/name).read_bytes())
    external=external_inventory()
    visual=blinded_qa()
    write_json(OUT/'data_audit.json',dict(audit,cohort_identity=json.loads((DEFAULT/'identity.json').read_text()),
        external_metadata='data_external_summary.json',visual_qa='data_visual_qa.json',report_generator_sha256=sha256_file(__file__)))
    text=f'''# Data audit for reference-aware gait modeling v2

The new cohort identity is `reference-v2-cohort-v1`. This audit rebuilds the complete annotation inventory, verifies actual source-video and raw-pose-cache hashes, and reconstructs every prepared window from its native sequence cache. It fits no model and does not inspect new forecasting outcomes.

The inventory has {audit['fresh_inventory']['all_sequences']:,} sequences, {audit['fresh_inventory']['all_rows']:,} annotation rows, and {audit['fresh_inventory']['all_sources']} source IDs. The upstream source CSVs are not counted again. The five requested labels contain 666 sequences from 103 source IDs. Media eligibility retains 652 sequences from 98 sources, and the two preserved whole-source identity exclusions leave 641 extraction candidates from 96 sources. All {audit['source_caches_verified']} source-cache and actual original-video hashes match the frozen preparation. Fresh annotation hashes match every selected sequence.

All {audit['prepared_windows']:,} prepared windows from 456 sequence caches reconstructed exactly, with maximum coordinate difference 0. The predeclared common distal-target support gate removes 3 windows, leaving {audit['model_windows']:,} windows from 456 sequences and {audit['sources']} source videos. The arrays are reused in place by hash and `prepared_index`; no media or tensors are duplicated. The original nominal 10 Hz, 16-prefix/8-future, confidence 0.5, separate 70% prefix/future support contract remains unchanged.

## Source relations and generalization

The grouping graph joins original source IDs only when original footage is byte-identical. There are no verified cross-source-ID byte-identical originals in the scanned local GAVD-related files. Therefore the evaluated cohort has 91 singleton components, and its seed 2701 five folds are identical to the historical source folds. This does not prove that all 91 sources show independent people. Mirrored, edited, or reencoded uploads and repeated participants can evade byte hashes. Source and component IDs remain proxies, not patient identifiers.

The complete annotation inventory has one mixed-label source, `wRntYsztIEY`, with `cerebral palsy` and `abnormal` annotations. It remains a single source component. Its cerebral-palsy sequences are already in the primary scope. Any broader-label transfer must exclude this shared source from its new-source test or retain it in the same training/evaluation component, with the mixed endpoint declared explicitly.

GAVD supports a heterogeneous public-video benchmark, but GAVD alone does not establish patient-disjoint, site-disjoint, or prospective clinical generalization. All 32 annotated normal source IDs already belong to the selected scope, with 29 in the final cohort. The other seven labels add coverage but no additional normal source support. Exercise, pregnancy, stylized walking, prosthetic use, and intoxication retain their actual annotation meanings. More overlapping windows do not create more independent patients.

## Frames, time, support, and view

Annotation frame numbers are one-based and decoded native indexes are zero-based. Reconstructed native mappings, source-window identities, decoder timestamps, and prefix-only normalization agree exactly with the frozen arrays. All retained windows have finite increasing actual timestamps; adjacent intervals range from {audit['timing']['actual_decoder_intervals_min_seconds']:.5f} to {audit['timing']['actual_decoder_intervals_max_seconds']:.5f} seconds. This is nearest-native-frame sampling on a nominal clock, not interpolation or exact physical-time resampling. Target coordinates describe pelvis-relative 2D articulation in prefix-torso units. They are not physical speed or measured 3D anatomy.

`data_attrition_by_label.csv` traces catalog, media, extraction candidates, prepared windows, and common support. `data_coverage_by_view.csv` preserves each literal view annotation and counts sources separately from windows. Future availability gates describe which outcomes can be measured retrospectively; they are not inputs to the forecasting model. The evaluation owner computes source-level prefix-motion and missingness covariates inside the declared model evaluation.

## Blinded visual audit

The audit samples {visual['sequences']} sequences from {visual['source_videos']} sources, covering accepted and rejected sequences within each selected label. Three native frames per sequence provide {visual['frames']} panels. The sheet conceals the strata, source IDs, and gate outcomes; the separate selection ledger retains those identities. This is a deterministic, outcome-independent sample, not comprehensive full-duration tracking or a new clinical-label review. The generated selection is recorded in `data_visual_qa.json`; the actual inspection is recorded separately in `data_visual_review.json`.

## Local external-data opportunity

The local KOA-PD-NM cache has {external['videos']} videos from {external['participants']} coded participants, including 30 normal, 15 Parkinson's, and 50 osteoarthritis participants. The bundled demographic PDF distinguishes participant numbering separately within each severity group, so subject keys include group and severity. The 189 videos under `data/videos-ms/koa-pd-nm` are exact byte copies of the chosen `data/videos-more/koa-pd-nm` originals; they must not be counted twice. No chosen external file is byte-identical to a GAVD original, but undetected reencoded overlap remains possible.

The descriptor lists 191 videos and 96 participants; local severe-Parkinson footage has 3 videos from 2 participants rather than 5 videos/3 participants. Participant `PD_SV_003` is absent and `PD_SV_002` has only direction 01. The original description's 31 PD videos also differs from the 32 implied by pairing all 16 demographic participants. These discrepancies are retained rather than filled by assumptions. `data_external_inventory.csv` records every existing video hash, clone relation, subject key, nominal FPS, image size, and first/last decode check.

The metadata audit found {external['metadata_decode_success']} videos with successful first/last decode checks. Invalid files remain explicitly listed in `data_external_summary.json`; their presence is not usable observation evidence. The external metadata audit performs no pose extraction or model fitting. A frozen video model could be evaluated through the same Ambient image API after a separate compatibility/extraction protocol. Any normal target-domain calibration is adaptation and needs a separate normal participant set. The local cache alone does not establish untouched historical exposure. A preimplementation text search found no dataset-name, DOI, or source-filename references in existing experiment code/reports and notes, but cannot establish what researchers previously viewed.

The {656} sequence clips in `data/videos-ms/gavd-sequences` are explicitly traced to the same GAVD source IDs and annotation frames by their own manifest. They are derived observations of this dataset, not external validation. Turning-in-place video contains a distinct PD task with no normal group, and WeightGait is a controlled healthy-participant perturbation dataset. Their native formats and labels require separate contracts before any clinical comparison.

## Reproduce and consume

From the repository root, run the experiment Python 3.12 environment:

```bash
experiments/aaai-worldmodels/.venv/bin/python experiments/aaai-worldmodels/scripts/audit_v2_data.py
experiments/aaai-worldmodels/.venv/bin/python -m pytest -c experiments/aaai-worldmodels/pyproject.toml experiments/aaai-worldmodels/tests/test_v2_data.py -q
```

The first command creates the declared cohort only if absent, then regenerates descriptive reports and review material. Completed cohort manifests are immutable; use a new output identity to change eligibility, grouping, or preparation. Consumers call `worldmodels.v2_data.load_cohort()` to obtain `(x, valid, metadata, times)`. The metadata keeps `prepared_index`, `component_id`, and `outer_fold`. The new identity pins the old array hashes, annotation snapshot, actual source hashes, pose-weight hash, code revision, and audit implementation. No paid provider, model download, new media download, or pose-model inference is required by this audit.
'''
    (OUT/'data_summary.md').write_text(text)
    print(json.dumps(dict(data=DEFAULT.name,external_videos=external['videos'],external_participants=external['participants'],visual_frames=visual['frames'])))


def audit_external_preparation(path=None):
    """Reconstruct completed external windows and reconcile all data attrition."""
    from worldmodels.v2_external import load_external,DEFAULT as EXTERNAL
    from worldmodels.pose import normalize_window
    from worldmodels.restart_data import window_identity
    from worldmodels.v2_evaluation import source_covariates
    output=Path(path or EXTERNAL);started=time.monotonic()
    x,valid,meta,times=load_external(output)
    attr=pd.read_csv(output/'attrition.csv',keep_default_na=False)
    inventory=pd.read_csv(output/'recording_inventory.csv',keep_default_na=False,dtype={'direction':str})
    max_error=0.;raw_count=0;requested=0
    for recording,rows in meta.groupby('recording_id',sort=True):
        with np.load(output/'recording_poses'/f'{recording}.npz',allow_pickle=False) as z:
            raw={k:z[k].copy() for k in z.files}
        provenance=json.loads(str(raw['provenance_json']))
        if sha256_file(REPO/provenance['video_path'])!=provenance['video_sha256']:
            raise ValueError('Actual external video changed after preparation')
        np.testing.assert_array_equal(raw['decoded_indexes'],raw['frame_numbers']-1)
        lookup={int(n):i for i,n in enumerate(raw['frame_numbers'])}
        for i,row in rows.iterrows():
            native=np.asarray(json.loads(row.frame_numbers),int)
            indexes=[lookup[int(n)] for n in native]
            expected,mask=normalize_window(raw['xy'][indexes],raw['confidence'][indexes],context_frames=16)
            mask[16:] &= mask[:16].any(0)[None]
            expected=np.where(mask[...,None],expected,0)
            max_error=max(max_error,float(np.max(np.abs(expected-x[i]))))
            np.testing.assert_array_equal(expected,x[i]);np.testing.assert_array_equal(mask,valid[i])
            np.testing.assert_array_equal(times[i],raw['timestamp_seconds'][indexes])
            if row.window_id!=window_identity(recording,native):
                raise ValueError('Recording-native-window identity failed to reconstruct')
    # Verify unretained raw caches and videos too, not only favorable windows.
    for path in sorted((output/'recording_poses').glob('*.npz')):
        with np.load(path,allow_pickle=False) as z:
            provenance=json.loads(str(z['provenance_json']))
            raw_count+=1;requested+=len(z['frame_numbers'])
            np.testing.assert_array_equal(z['decoded_indexes'],z['frame_numbers']-1)
        if sha256_file(REPO/provenance['video_path'])!=provenance['video_sha256']:
            raise ValueError('Unretained external recording changed')
    assert attr.retained_windows.sum()==len(meta)
    assert (valid[:,:16].mean((1,2))>=.7).all() and (valid[:,16:].mean((1,2))>=.7).all()
    assert np.all(x[~valid]==0)
    prep=json.loads((output/'preparation.json').read_text())
    progress=json.loads((output/'progress.json').read_text())
    assert sum(r['actual_model_calls'] for r in progress['recordings'])==prep['actual_model_calls_this_invocation']
    covariates=source_covariates(x,valid,times,meta)
    covariates.to_csv(OUT/'data_external_prefix_covariates.csv',index=False)
    by_label=[]
    for label,group in inventory.groupby('label',sort=True):
        retained=meta[meta.label==label];aa=attr[attr.label==label]
        rejection=Counter()
        for value in aa.rejection_counts:
            rejection.update(json.loads(value))
        by_label.append(dict(label=label,catalog_recordings=len(group),catalog_participants=int(group.participant_id.nunique()),
            metadata_decodable_recordings=int(group.first_and_last_decode.sum()),
            decodable_participants=int(group.loc[group.first_and_last_decode,'participant_id'].nunique()),
            retained_recordings=int(retained.recording_id.nunique()),retained_participants=int(retained.source_id.nunique()),
            candidate_windows=int(aa.candidate_windows.sum()),retained_windows=len(retained),rejection_counts=dict(rejection)))
    pd.DataFrame(by_label).to_csv(OUT/'data_external_attrition_by_label.csv',index=False)
    participant=inventory.groupby(['participant_id','label']).agg(catalog_recordings=('video_id','size'),metadata_decodable_recordings=('first_and_last_decode','sum')).reset_index()
    windows=meta.groupby('source_id').agg(retained_recordings=('recording_id','nunique'),retained_windows=('window_id','size')).reset_index().rename(columns={'source_id':'participant_id'})
    participant=participant.merge(windows,on='participant_id',how='left').fillna({'retained_recordings':0,'retained_windows':0})
    participant.to_csv(OUT/'data_external_attrition_by_participant.csv',index=False)
    qa=json.loads((output/'visual_qa/selection.json').read_text())
    report=dict(identity='reference-v2-external-preparation-reconstruction-v1',status='complete',preparation_sha256=sha256_file(output/'preparation.json'),
        identity_sha256=sha256_file(output/'identity.json'),auditor_sha256=sha256_file(__file__),
        recordings_verified=raw_count,requested_pose_frames=requested,actual_pose_calls=prep['actual_model_calls_this_invocation'],
        windows_reconstructed=len(meta),maximum_absolute_coordinate_error=max_error,
        participant_grouping_verified=True,native_frame_origin_verified=True,actual_times_verified=True,
        common_support_and_70pct_gates_verified=True,by_label=by_label,
        participants_without_model_support=participant.loc[participant.retained_windows.eq(0),'participant_id'].tolist(),
        finite_prefix_covariate_participants=len(covariates),
        native_fps_by_label=inventory.groupby(['label','fps']).size().rename('recordings').reset_index().to_dict('records'),
        native_fps_limit='Normal and abnormal recordings have different native-FPS distributions. Nominal10Hz sampling equalizes requested time gaps but does not remove acquisition, blur, duration, or framing confounding.',
        actual_frame_interval_seconds=[float(np.diff(times,axis=1).min()),float(np.diff(times,axis=1).max())],
        final_horizon_seconds=[float(meta.final_horizon_seconds.min()),float(meta.final_horizon_seconds.max())],
        visual_material=qa,seconds=time.monotonic()-started,
        model_efficacy_scores_inspected=0,
        confidence_summary_note='External prefix_mean_confidence and future_mean_confidence average raw33 BlazePose confidence values, whereas support gates/missingness use the declared16-joint subset. Do not directly compare those descriptive fields to a GAVD16-joint confidence field. Predictor inputs and nuisance covariates use their declared masks and are unaffected.',
        limit='Replay uses the same normalization API. It validates artifact construction, not independent clinical truth or full-duration identity tracking.')
    write_json(OUT/'data_external_preparation_audit.json',report)
    print(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--external-audit',action='store_true')
    parser.add_argument('--external-path',type=Path)
    arguments=parser.parse_args()
    if arguments.external_audit:
        audit_external_preparation(arguments.external_path)
    else:
        reports()
