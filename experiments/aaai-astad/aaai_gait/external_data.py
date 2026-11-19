"""Bounded public C3D download and parsing audit, with no fitted outcomes.

The marker and model-angle data are not interchangeable with BlazePose XY.
This module prepares a separate participant dataset for possible method
replication. It never pools it with GAVD and never transfers fitted weights.
Optional parser: uv pip install --python .venv/bin/python c3d==0.6.0.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path, PurePosixPath
import re
import time
import urllib.request
import zipfile

FILES=[dict(group='healthy',article_id=24192480,file_id=42450060,name='138_HealthyPiG.zip',size=463159769,md5='85235c73455d4df8434810d61517efbb'),
       dict(group='stroke',article_id=24192483,file_id=42450069,name='50_StrokePiG.zip',size=316448944,md5='71925450f3c058b145791de7951c8dc9')]


def _hash(path,algorithm='sha256'):
    h=hashlib.new(algorithm)
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def download_archives(output_dir):
    """Download only the two known C3D archives and verify published MD5/size."""
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True);records=[]
    for spec in FILES:
        path=out/spec['name'];url=f"https://ndownloader.figshare.com/files/{spec['file_id']}"
        request=urllib.request.Request(f"https://api.figshare.com/v2/articles/{spec['article_id']}",headers={'User-Agent':'alexpose-research/iteration02'})
        with urllib.request.urlopen(request,timeout=45) as response:metadata=json.load(response)
        current=next(f for f in metadata['files'] if f['id']==spec['file_id'])
        if current['size']!=spec['size'] or current['computed_md5']!=spec['md5']:raise ValueError('Published file identity changed')
        (out/f"article_{spec['article_id']}.json").write_text(json.dumps(metadata,indent=2)+'\n')
        if not path.exists():
            print(f"Downloading {spec['group']} C3D archive ({spec['size']} bytes)",flush=True)
            partial=path.with_suffix('.zip.part');count=partial.stat().st_size if partial.exists() else 0
            if count>spec['size']:raise ValueError('Partial file exceeds expected archive size')
            requests=0
            while count<spec['size']:
                requests+=1
                if requests>32:raise ValueError('Bounded download retry budget exhausted')
                start=count;end=min(start+64*1024*1024-1,spec['size']-1)
                request=urllib.request.Request(url,headers={'User-Agent':'alexpose-research/iteration02','Range':f'bytes={start}-{end}'})
                with urllib.request.urlopen(request,timeout=60) as response,partial.open('ab') as stream:
                    expected=f"bytes {start}-{end}/{spec['size']}"
                    if response.status!=206 or response.headers.get('Content-Range')!=expected:
                        raise ValueError('Server did not honor the exact bounded byte range')
                    for block in iter(lambda:response.read(4*1024*1024),b''):
                        count+=len(block)
                        if count>end+1:raise ValueError('Download exceeded requested byte range')
                        stream.write(block)
                event=dict(group=spec['group'],start=start,end=end,received=count-start,complete_range=count==end+1)
                with (out/'download_events.jsonl').open('a') as log:log.write(json.dumps(event)+'\n')
                if count==start:raise ValueError('Download made no progress')
                print(f"{spec['group']} archive: {count}/{spec['size']} bytes",flush=True)
            if count!=spec['size'] or _hash(partial,'md5')!=spec['md5']:raise ValueError('Incomplete archive or checksum mismatch')
            partial.replace(path)
        if path.stat().st_size!=spec['size'] or _hash(path,'md5')!=spec['md5']:raise ValueError('Cached archive fails checksum')
        record={**spec,'path':str(path.resolve()),'url':url,'sha256':_hash(path),'license':metadata.get('license'),'verified_at_utc':datetime.now(timezone.utc).isoformat()};records.append(record)
        print(f"Verified {spec['group']} archive",flush=True)
    (out/'archive_provenance.json').write_text(json.dumps(records,indent=2)+'\n')
    return records


def _archive_rows(spec,archive):
    rows=[]
    for item in archive.infolist():
        if item.is_dir() or not item.filename.lower().endswith('.c3d'):continue
        parts=PurePosixPath(item.filename).parts
        if any(part=='..' for part in parts) or PurePosixPath(item.filename).is_absolute():raise ValueError('Unsafe archive member')
        subject=next((p for p in parts[:-1] if re.fullmatch(r'(SUBJ|TVC)\d+',p,re.IGNORECASE)),None)
        if subject is None:subject='unresolved'
        stem=PurePosixPath(item.filename).stem
        static=bool(re.search(r'(static|cal|\(0\))',stem,re.IGNORECASE))
        rows.append(dict(group=spec['group'],archive=spec['name'],member=item.filename,participant_id=f"{spec['group']}:{subject}",
            participant_directory=subject,trial=stem,static=static,bytes=item.file_size,compressed_bytes=item.compress_size,crc32=f'{item.CRC:08x}'))
    return rows


def archive_inventory(output_dir):
    """Enumerate participant/trial names before selecting deterministic samples."""
    rows=[]
    for spec in FILES:
        with zipfile.ZipFile(Path(output_dir)/spec['name']) as archive:
            rows.extend(_archive_rows(spec,archive))
    return rows


class _RemoteZip(io.RawIOBase):
    """Seekable public byte ranges, capped at 32 MB per archive for this pilot."""
    def __init__(self,spec,cache):
        self.spec=spec;self.pos=0;self.received=0;self.events=[];self.cache=Path(cache)/str(spec['file_id']);self.cache.mkdir(parents=True,exist_ok=True)
    def seekable(self):return True
    def readable(self):return True
    def tell(self):return self.pos
    def seek(self,offset,whence=0):
        self.pos=offset if whence==0 else self.pos+offset if whence==1 else self.spec['size']+offset
        if self.pos<0:raise ValueError('Negative archive position')
        return self.pos
    def read(self,size=-1):
        size=self.spec['size']-self.pos if size<0 else min(size,self.spec['size']-self.pos)
        if size<=0:return b''
        if size>8*1024*1024 or self.received+size>32*1024*1024:raise ValueError('Remote pilot byte budget exceeded')
        start=self.pos;end=start+size-1;path=self.cache/f'{start}-{end}.bin'
        if path.exists():data=path.read_bytes()
        else:
            for attempt in range(3):
                try:
                    url=f"https://ndownloader.figshare.com/files/{self.spec['file_id']}?pilot={time.time_ns()}"
                    request=urllib.request.Request(url,headers={'Range':f'bytes={start}-{end}','User-Agent':'alexpose-research/iteration02'})
                    with urllib.request.urlopen(request,timeout=30) as response:
                        if response.status!=206 or response.headers.get('Content-Range')!=f"bytes {start}-{end}/{self.spec['size']}":raise ValueError('Remote range identity mismatch')
                        data=response.read();self.received+=len(data)
                    if len(data)!=size:raise ValueError('Truncated byte range')
                    path.write_bytes(data);break
                except Exception as exc:
                    self.events.append(dict(start=start,end=end,attempt=attempt,error=f'{type(exc).__name__}: {str(exc).split("https://")[0]}'))
                    if attempt==2:raise
        if len(data)!=size:raise ValueError('Cached range length mismatch')
        self.events.append(dict(start=start,end=end,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
        self.pos+=len(data);return data


def parse_c3d(path):
    """Read synchronized raw-time point samples and audit marker/angle fields."""
    import c3d
    import numpy as np
    with Path(path).open('rb') as stream:
        reader=c3d.Reader(stream)
        labels=[str(s).strip() for s in reader.point_labels]
        point_rate=float(reader.point_rate);analog_rate=float(reader.analog_rate)
        frames=[];numbers=[];analog_shapes=[]
        for number,points,analog in reader.read_frames(copy=True):
            numbers.append(int(number));frames.append(points);analog_shapes.append(list(analog.shape))
        data=np.stack(frames) if frames else np.empty((0,len(labels),5))
        names={name.split(':')[-1].strip():i for i,name in enumerate(labels)}
        expected=['LHipAngles','RHipAngles','LKneeAngles','RKneeAngles','LAnkleAngles','RAnkleAngles']
        expected_markers=['LSHO','RSHO','LELB','RELB','LWRA','RWRA','LWRB','RWRB','LASI','RASI','LPSI','RPSI','LKNE','RKNE','LANK','RANK','LHEE','RHEE','LTOE','RTOE']
        angles={name:dict(label=labels[names[name]],index=names[name],finite_fraction=float(np.isfinite(data[:,names[name],:3]).all(axis=1).mean()),valid_residual_fraction=float((data[:,names[name],3]>=0).mean())) for name in expected if name in names}
        events={}
        for field in ['EVENT:TIMES','EVENT:LABELS','EVENT:CONTEXTS','EVENT:USED','POINT:UNITS','SUBJECTS:NAMES']:
            parameter=reader.get(field)
            if parameter is not None:
                try:value=parameter.string_array.tolist() if field.endswith(('LABELS','CONTEXTS','NAMES')) else parameter.float_array.tolist() if field.endswith('TIMES') else parameter.string_value if field.endswith('UNITS') else int(parameter.uint16_value)
                except Exception:value='present_unparsed'
                if field=='SUBJECTS:NAMES':
                    value=dict(present=True,sha256=hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest(),use='Not used. Participant IDs come from the archive directory.')
                events[field]=value
    return dict(path=str(Path(path).resolve()),sha256=_hash(path),parser_version=importlib.metadata.version('c3d'),point_rate_hz=point_rate,analog_rate_hz=analog_rate,
                frames=len(numbers),first_frame=numbers[0] if numbers else None,last_frame=numbers[-1] if numbers else None,
                monotone_adjacent_frames=bool(len(numbers)>1 and np.all(np.diff(numbers)==1)),sample_span_seconds=(numbers[-1]-numbers[0])/point_rate if numbers else None,
                n_point_channels=len(labels),point_labels=labels,angle_fields=angles,marker_fields={m:labels[names[m]] for m in expected_markers if m in names},
                analog_shapes=sorted({tuple(s) for s in analog_shapes}),events=events,
                bilateral_clock='Both sides share the same raw C3D frame numbers. No independent stride normalization was applied.',
                mapping_warning='PiG marker surfaces and model-derived angles are not equivalent to BlazePose keypoint XY. Angles need plane/axis convention confirmation.')


def run_pilot(output_dir,report_dir,max_participants_per_group=1):
    out=Path(output_dir);reports=Path(report_dir);reports.mkdir(parents=True,exist_ok=True)
    provenance=download_archives(out);inventory=archive_inventory(out)
    (reports/'c3d_inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
    selected=[];parses=[];failures=[]
    for spec in FILES:
        group=[r for r in inventory if r['group']==spec['group'] and not r['static'] and r['participant_directory']!='unresolved']
        for subject in sorted({r['participant_id'] for r in group})[:max_participants_per_group]:
            row=sorted((r for r in group if r['participant_id']==subject),key=lambda r:r['member'])[0];selected.append(row)
            path=out/'pilot_c3d'/spec['group']/Path(row['member']).name;path.parent.mkdir(parents=True,exist_ok=True)
            with zipfile.ZipFile(out/spec['name']) as archive,archive.open(row['member']) as src,path.open('wb') as dst:
                dst.write(src.read())
            try:parses.append({**row,**parse_c3d(path)})
            except Exception as exc:failures.append(dict(member=row['member'],error=f'{type(exc).__name__}: {exc}'))
    summary=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),status='parsing_feasibility_only_no_model_outcomes',archives=provenance,
        archive_counts=[dict(group=g,participants=len({r['participant_id'] for r in inventory if r['group']==g and r['participant_directory']!='unresolved'}),c3d_files=sum(r['group']==g for r in inventory),unresolved_files=sum(r['group']==g and r['participant_directory']=='unresolved' for r in inventory)) for g in ['healthy','stroke']],
        selected_trials=selected,parses=parses,failures=failures,protocol_gate='No fitting or scoring until representation, clock, participant split and age/speed sensitivities are frozen.')
    (reports/'parsing_pilot.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:summary[k] for k in ['status','archive_counts','failures']},indent=2),flush=True)
    return summary


def run_remote_pilot(output_dir,report_dir):
    """Fetch archive directory plus one deterministic dynamic trial per group.

    ZIP entry CRC and extracted SHA are checked. Full archive MD5 cannot be
    verified from partial ranges and is explicitly not claimed.
    """
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True);reports=Path(report_dir);reports.mkdir(parents=True,exist_ok=True)
    inventory=[];parses=[];failures=[];sources=[];events=[]
    for spec in FILES:
        request=urllib.request.Request(f"https://api.figshare.com/v2/articles/{spec['article_id']}",headers={'User-Agent':'alexpose-research/iteration02'})
        with urllib.request.urlopen(request,timeout=30) as response:metadata=json.load(response)
        current=next(f for f in metadata['files'] if f['id']==spec['file_id'])
        if current['size']!=spec['size'] or current['computed_md5']!=spec['md5']:raise ValueError('Published file identity changed')
        (out/f"article_{spec['article_id']}.json").write_text(json.dumps(metadata,indent=2)+'\n')
        remote=_RemoteZip(spec,out/'range_cache')
        try:
            with zipfile.ZipFile(remote) as archive:
                rows=_archive_rows(spec,archive);inventory.extend(rows)
                candidates=sorted((r for r in rows if not r['static'] and r['participant_directory']!='unresolved'),key=lambda r:(r['participant_id'],r['member']))
                if not candidates:raise ValueError('No dynamic trial with resolved participant directory')
                row=candidates[0];payload=archive.read(row['member'])
                path=out/'pilot_c3d'/spec['group']/Path(row['member']).name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(payload)
                parsed=parse_c3d(path);parses.append({**row,**parsed,'zip_crc_verified':True})
                print(f"Parsed {spec['group']}: {row['participant_id']}, {parsed['frames']} frames, {parsed['point_rate_hz']} Hz",flush=True)
        except Exception as exc:failures.append(dict(group=spec['group'],error=f'{type(exc).__name__}: {str(exc).split("https://")[0]}'))
        sources.append({**spec,'license':metadata.get('license'),'archive_checksum_status':'Published MD5 recorded; full archive not downloaded or verified','download_mode':'bounded HTTP byte ranges','bytes_downloaded_this_run':remote.received,'validated_range_cache_bytes':sum(p.stat().st_size for p in remote.cache.glob('*.bin'))})
        events.append(dict(group=spec['group'],events=remote.events))
    summary=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),status='parsing_feasibility_only_no_model_outcomes',archives=sources,
        archive_counts=[dict(group=g,participants=len({r['participant_id'] for r in inventory if r['group']==g and r['participant_directory']!='unresolved'}),c3d_files=sum(r['group']==g for r in inventory),unresolved_files=sum(r['group']==g and r['participant_directory']=='unresolved' for r in inventory)) for g in ['healthy','stroke']],
        parses=parses,failures=failures,protocol_gate='No fitting or scoring until representation, clock, participant split and age/speed sensitivities are frozen.')
    (reports/'c3d_inventory.json').write_text(json.dumps(inventory,indent=2)+'\n');(reports/'range_requests.json').write_text(json.dumps(events,indent=2)+'\n')
    (reports/'parsing_pilot.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:summary[k] for k in ['status','archive_counts','failures']},indent=2),flush=True)
    return summary


def main():
    parser=argparse.ArgumentParser();root=Path(__file__).resolve().parents[3]
    parser.add_argument('--data',type=Path,default=root/'data/external_gait/van_criekinge_2023_figshare_v1')
    parser.add_argument('--report',type=Path,default=root/'experiments/aaai/artifacts/iteration02/preparation/external_pilot')
    parser.add_argument('--remote-only',action='store_true',help='Fetch only ZIP metadata and one C3D per cohort, capped at 32 MB each')
    args=parser.parse_args();(run_remote_pilot if args.remote_only else run_pilot)(args.data,args.report)

if __name__=='__main__':main()
