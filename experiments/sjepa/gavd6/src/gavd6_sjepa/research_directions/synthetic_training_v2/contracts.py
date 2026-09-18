"""Strict data, provenance and exposure boundaries for offline body-12 restoration."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import numpy as np
from ..temporal_gait.contracts import atomic_json, digest, sha256_file

JOINTS = ('left_shoulder','right_shoulder','left_elbow','right_elbow',
          'left_wrist','right_wrist','left_hip','right_hip','left_knee',
          'right_knee','left_ankle','right_ankle')
INPUT_KEYS = frozenset(('xy','confidence','observed','timestamps'))
TARGET_KEYS = frozenset(('xy','valid','visible','eval_scale'))
SCHEMA = 'coco-body12-xy-v1'
PREPROCESSING = 'observed-clip-median-quantile-diagonal-v1'


def schema_permutation(names):
    names = list(names)
    if len(names) != 12 or len(set(names)) != 12 or set(names) != set(JOINTS):
        raise ValueError('Schema requires all twelve named anatomical joints exactly once')
    return np.array([names.index(j) for j in JOINTS])


def validate_inputs(inputs, *, samples=64, hz=25.):
    if set(inputs) != INPUT_KEYS:
        raise ValueError(f'Inference allow-list violation: {set(inputs) ^ INPUT_KEYS}')
    x, c, o, t = (np.asarray(inputs[k]) for k in ('xy','confidence','observed','timestamps'))
    if x.ndim != 4 or x.shape[1:] != (samples,12,2) or c.shape != x.shape[:-1] or o.shape != c.shape:
        raise ValueError('Expected xy[N,T,12,2], confidence/observed[N,T,12]')
    if o.dtype != np.bool_ or t.shape != x.shape[:2]:
        raise ValueError('Explicit boolean observation mask and timestamps[N,T] required')
    if not np.isfinite(t).all() or not np.allclose(np.diff(t,axis=1),1/hz,atol=1e-6,rtol=0):
        raise ValueError('Physical clock must be an increasing fixed-rate grid; no duration stretching')
    if np.any(o & ~np.isfinite(x).all(-1)):
        raise ValueError('Observed coordinates must be finite')
    if np.any(o & (~np.isfinite(c) | (c < 0) | (c > 1))):
        raise ValueError('Observed confidence must be an actual score in [0,1]')
    return len(x)


def validate_targets(targets, inputs):
    if set(targets) != TARGET_KEYS:
        raise ValueError('Targets require xy, valid, visible and independent eval_scale')
    xy, valid, visible = (np.asarray(targets[k]) for k in ('xy','valid','visible'))
    if xy.shape != inputs['xy'].shape or valid.shape != xy.shape[:-1] or visible.shape != valid.shape:
        raise ValueError('Target shape mismatch')
    if valid.dtype != np.bool_ or visible.dtype != np.bool_ or np.any(visible & ~valid):
        raise ValueError('Visible targets must be target-valid; masks must be boolean')
    if np.any(valid & ~np.isfinite(xy).all(-1)):
        raise ValueError('Valid target coordinates must be finite')
    scale = np.asarray(targets['eval_scale'])
    if scale.shape != xy.shape[:2] or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError('Independent per-frame evaluation scale must be positive finite pixels')


def validate_records(records, *, evidence_status, held_extractor=None):
    required = {'person_id','canonical_person_id','motion_id','motion_hash','window_id',
                'split','variant','extractor','box_source','exposure','original_split',
                'locomotion_status','audit_reviewer','audit_evidence','audit_date'}
    seen, people, motions, raw_people = set(), {}, {}, {}
    for r in records:
        if required - r.keys() or any(str(r[k]).strip()=='' for k in required):
            raise ValueError(f'Missing provenance: {sorted(required-r.keys())}')
        key = (r['window_id'],r['variant'],r['extractor'])
        if key in seen: raise ValueError('Duplicate window/variant/extractor')
        seen.add(key)
        known=raw_people.setdefault(r['person_id'],(r['canonical_person_id'],r['original_split']))
        if known!=(r['canonical_person_id'],r['original_split']):raise ValueError('Known person changed canonical identity or original split')
        if r['split'] not in {'train','development','confirmation'}: raise ValueError('Unknown split')
        if r['locomotion_status'] not in {'audited_locomotion','fixture'}: raise ValueError('Unaudited locomotion')
        if evidence_status != 'fixture-tested' and r['locomotion_status'] == 'fixture': raise ValueError('Fixture is not source evidence')
        if r['original_split'] == 'test' and r['split'] != 'confirmation': raise ValueError('Protected test identity reassigned')
        if r['original_split'] != 'train' and r['split'] == 'train': raise ValueError('Historical nontraining identity entered fit')
        if r['split']=='confirmation' and r['exposure']!='unexposed_verified': raise ValueError('Exposed/unknown confirmation')
        if r.get('reserved',False) and r['split']!='confirmation': raise ValueError('Reserved identity reassigned')
        if held_extractor and held_extractor in {r['extractor'],r.get('extractor_family','')} and r['split']=='train': raise ValueError('Held extractor entered fit')
        for mapping,k in ((people,r['canonical_person_id']),(motions,r['motion_hash'])):
            if k in mapping and mapping[k] != r['split']: raise ValueError('Person/alias or motion leaks across splits')
            mapping[k] = r['split']
    return {'windows':len({r['window_id'] for r in records}), 'rows':len(records),
            'people':len(people),'motions':len(motions), 'splits':{s:sum(r['split']==s for r in records) for s in sorted(set(people.values()))}}


def array_digest(arrays):
    h=hashlib.sha256()
    for k in sorted(arrays):
        a=np.ascontiguousarray(arrays[k]); h.update(k.encode());h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()


def cache_identity(*, source_hash, checkpoint_hash, config, code_hash, preprocessing=PREPROCESSING):
    if not source_hash or not checkpoint_hash or not code_hash: raise ValueError('Cache needs source/checkpoint/code identities')
    return digest(dict(source=source_hash,checkpoint=checkpoint_hash,schema=SCHEMA,
                       configuration=config,preprocessing=preprocessing,code=code_hash))


def code_identity(repo):
    root=Path(repo)
    paths=sorted((root/'src/gavd6_sjepa/research_directions/synthetic_training_v2').glob('*.py'))
    paths += [root/'src/gavd6_sjepa'/name for name in (
        'research_directions/temporal_gait/objectives.py',
        'research_directions/temporal_gait/contracts.py',
        'research_directions/synthetic_training/rendering.py',
        'research_directions/synthetic_training/estimators.py',
        'research_directions/synthetic_training/measurements.py',
        'research_directions/motion_preservation/motion_data.py',
        'research_directions/motion_preservation/body_geometry.py',
        'data_foundations/amass_conversion.py',
    )]
    return digest({str(p.relative_to(root)):sha256_file(p) for p in paths})


@dataclass
class TrackBundle:
    inputs: dict
    targets: dict
    records: list
    evidence_status: str
    provenance: dict

    def validate(self,held_extractor=None):
        n=validate_inputs(self.inputs);validate_targets(self.targets,self.inputs)
        if len(self.records)!=n: raise ValueError('Records and arrays differ')
        return validate_records(self.records,evidence_status=self.evidence_status,held_extractor=held_extractor)

    def subset(self, split):
        if split=='confirmation': raise PermissionError('Use explicit confirmation label access')
        ix=np.array([i for i,r in enumerate(self.records) if r['split']==split],dtype=int)
        return TrackBundle({k:v[ix] for k,v in self.inputs.items()}, {k:v[ix] for k,v in self.targets.items()},
                           [self.records[i] for i in ix],self.evidence_status,self.provenance)

    def save(self, folder):
        self.validate();folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
        np.savez_compressed(folder/'inputs.npz',**self.inputs)
        np.savez_compressed(folder/'targets.npz',**self.targets)
        atomic_json(folder/'manifest.json',dict(schema=SCHEMA,evidence_status=self.evidence_status,
            records=self.records,provenance=self.provenance,
            inputs_sha256=sha256_file(folder/'inputs.npz'),targets_sha256=sha256_file(folder/'targets.npz')))

    @classmethod
    def load(cls,folder,*,expected_identity=None):
        folder=Path(folder);meta=json.loads((folder/'manifest.json').read_text())
        # Inspect exposure records BEFORE touching any target file.
        if any(r['split']=='confirmation' for r in meta['records']):
            raise PermissionError('Development loader forbids confirmation targets')
        if meta['schema']!=SCHEMA: raise ValueError('Incompatible schema cache')
        if expected_identity is not None and meta['provenance'].get('identity')!=expected_identity:
            raise ValueError('Stale cache identity')
        arrays=[]
        for kind in ('inputs','targets'):
            path=folder/f'{kind}.npz'
            if sha256_file(path)!=meta[f'{kind}_sha256']: raise ValueError('Cache content hash mismatch')
            with np.load(path,allow_pickle=False) as f: arrays.append({k:f[k].copy() for k in f.files})
        result=cls(*arrays,meta['records'],meta['evidence_status'],meta['provenance']);result.validate()
        return result


def verify_preservation(repo):
    root=Path(repo);p=root/'docs/studies/synthetic-training-v2/preservation-manifest.json'
    saved=json.loads(p.read_text())['files']
    changed=[name for name,h in saved.items() if not (root/name).is_file() or sha256_file(root/name)!=h]
    if changed: raise RuntimeError(f'Pre-existing artifacts changed: {changed}')
    return {'status':'pass','files_checked':len(saved)}
