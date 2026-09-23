"""Generate one saved configuration from the user's working HAIC installation."""
from __future__ import annotations

import importlib
import math
import os
from pathlib import Path
import shlex
import shutil
import sys

from .common import atomic_json, code_identity, read_json, sha256, utc_now
from .spec import build_plan


def repository_root():
    return Path(__file__).resolve().parents[4]


def _previous_preparation(root, explicit=None, *, require_legacy_audit=True):
    if explicit:
        paths = [Path(explicit).expanduser().resolve()]
    else:
        base = root / 'outputs/synthetic-training-v2'
        paths = [base / 'full-01/paired/preparation-provenance.json',
                 base / 'full-01/config/preparation.json']
        paths += sorted((base / 'source-smoke-01').glob('paired-*/preparation-provenance.json'), reverse=True)
        paths += [base / 'source-smoke-01/config/preparation.json']
    for path in paths:
        if path.is_file():
            raw = read_json(path)
            value = raw.get('configuration', raw)
            required = ('manifest_dir', 'amass_root',
                        'body_model_root', 'dmpl_root', 'uv_path', 'texture_dir', 'background_dir', 'estimators')
            if require_legacy_audit:
                required += ('locomotion_audit', 'reservation_csv')
            if not all(k in value for k in required):
                if explicit:
                    raise ValueError(f'Incomplete preparation configuration: {path}')
                continue
            return value, path
    raise FileNotFoundError('No completed synthetic-training-v2 preparation configuration found. '
                            'Pass --source-config /absolute/path/to/preparation.json.')


def _environment_preparation(root):
    """Reuse the documented HAIC assets without inventing a historical roster."""
    user = os.environ.get('USER', '')
    models = Path(os.environ.get('ST_MODEL_ROOT', f'/hai/scratch/{user}/models'))
    body = Path(os.environ.get('ST_BODY_MODEL_ROOT', f'/hai/scratch/{user}/body_models'))
    mmpose = models / 'mmpose/configs/body_2d_keypoint'
    estimators = [
        dict(student_id='rtmpose_m', family='rtmpose',
             config=str(mmpose/'rtmpose/coco/rtmpose-m_8xb256-420e_coco-256x192.py'),
             checkpoint=str(models/'pose/rtmpose-m.pth')),
        dict(student_id='hrnet_w32', family='hrnet',
             config=str(mmpose/'topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py'),
             checkpoint=str(models/'pose/hrnet-w32.pth')),
        dict(student_id='vitpose_base', family='vitpose',
             config=str(mmpose/'topdown_heatmap/coco/td-hm_ViTPose-base_8xb64-210e_coco-256x192.py'),
             checkpoint=str(models/'pose/vitpose-base.pth'))]
    return dict(manifest_dir=str(root/'manifests/amass'),
        amass_root=os.environ.get('ST_AMASS_ROOT', str(Path(os.environ.get('AMASS_ROOT', str(root/'data/amass')))/'extracted')),
        body_model_root=str(body), dmpl_root=os.environ.get('ST_DMPL_ROOT', str(body/'dmpls')),
        uv_path=os.environ.get('ST_UV_PATH', str(models/'synthetic-rendering/smplitex/smpl_uv.obj')),
        texture_dir=os.environ.get('ST_TEXTURE_DIR', str(models/'synthetic-rendering/smplitex/textures')),
        background_dir=os.environ.get('ST_BACKGROUND_DIR', str(models/'synthetic-rendering/coco-backgrounds')),
        estimators=estimators)


def initialize(work, *, root=None, fixture=False, source_config=None, source_bundle=None,
               source_selection='full_manifest', cohort_preset=None, reservation_csv=None,
               motion_review_csv=None, num_shards=None, experiment_set=None):
    work = Path(work).expanduser().resolve()
    asset_root = Path(root or os.environ.get('GAVD6_ROOT') or repository_root()).expanduser().resolve()
    code_root = repository_root()
    if (work / 'config.json').exists():
        saved = load_config(work)
        if bool(saved['fixture']) != bool(fixture):
            raise ValueError('Existing run has a different mode; choose another work directory')
        for requested,key in ((root,'asset_root'),(source_config,'inherited_configuration'),(source_bundle,'source_bundle')):
            if requested is not None and str(Path(requested).expanduser().resolve())!=saved[key]:
                raise ValueError(f'Existing run has a different {key}; use a new run directory')
        if not fixture and saved['data'].get('source_selection', 'legacy_roster') != source_selection:
            raise ValueError('Existing run uses a different source selection; use a new run directory')
        if experiment_set is not None and saved.get('experiment_set','full') != experiment_set:
            raise ValueError('Existing run uses a different experiment set; use a new run directory')
        if cohort_preset is not None and saved.get('cohort',{}).get('preset') != cohort_preset:
            raise ValueError('Existing run uses a different cohort preset; use a new run directory')
        if num_shards is not None and saved['data'].get('num_shards',8) != num_shards:
            raise ValueError('Existing run uses a different shard count; use a new run directory')
        for requested, key in ((reservation_csv, 'reservation_csv'), (motion_review_csv, 'motion_review_csv')):
            if requested is not None and str(Path(requested).expanduser().resolve()) != saved.get('cohort', {}).get(key):
                raise ValueError(f'Existing cohort has a different {key}; use a new run directory')
        return saved
    if work.exists() and any(work.iterdir()):
        raise FileExistsError('A new study needs an empty run directory')
    if source_selection not in {'full_manifest', 'legacy_roster'}:
        raise ValueError('source_selection must be full_manifest or legacy_roster')
    cohort_preset = cohort_preset or 'named_walking'
    num_shards = 8 if num_shards is None else num_shards
    if type(num_shards) is not int or num_shards < 1:
        raise ValueError('num_shards must be a positive integer')
    if fixture:
        preparation, inherited = {}, None
    else:
        try:
            preparation, inherited = _previous_preparation(asset_root, source_config,
                require_legacy_audit=source_selection == 'legacy_roster')
        except FileNotFoundError:
            if source_config or source_selection == 'legacy_roster': raise
            preparation, inherited = _environment_preparation(asset_root), None
    if fixture or source_selection == 'full_manifest':
        bundle = None
        if source_bundle and not fixture:
            raise ValueError('source_bundle applies only to explicit legacy_roster selection')
    else:
        candidates = [Path(source_bundle)] if source_bundle else [
            asset_root / 'outputs/synthetic-training-v2/full-01/paired/bundle',
            inherited.parent / 'bundle']
        bundle = next((p.expanduser().resolve() for p in candidates if (p / 'manifest.json').is_file()), None)
        if bundle is None:
            raise FileNotFoundError('Retained roster bundle missing; pass --source-bundle PATH/bundle.')
        records=read_json(bundle/'manifest.json')['records']
        counts={s:len({r['canonical_person_id'] for r in records if r['split']==s}) for s in ('train','development')}
        if not all(counts.values()):
            raise ValueError(f'Legacy roster requires both training and development identities; found {counts}')
    experiment_set = experiment_set or ('full' if fixture else 'core')
    cfg = dict(schema='gait-fidelity-config-v1', created_utc=utc_now(), fixture=fixture,mode='fixture' if fixture else 'source',
               work=str(work), code_root=str(code_root), asset_root=str(asset_root),
               python=sys.executable, device='cpu' if fixture else 'cuda',
               source_bundle=str(bundle) if bundle else None, preparation=preparation,
               inherited_configuration=str(inherited) if inherited else None,
               experiment_set=experiment_set,
               seeds=[17,29,43], evidence_scope='software-fixture' if fixture else 'synthetic-development',
               data=dict(samples=32 if fixture else 128, hz=25., movement_levels_deg=[0,5,10,15],
                         source_selection='fixture' if fixture else source_selection,
                         partition='development', num_shards=1 if fixture else num_shards,
                         held_level_deg=15, physical_states=['original','mirrored'],
                         naming=['correct','global_swap','temporary_swap'], observations=['clear','occluded'],
                         cameras=[dict(id='oblique',azimuth_deg=45),dict(id='side',azimuth_deg=90)],
                         occlusion_fraction=.15, keep_videos=True, device='cpu' if fixture else 'cuda',
                         storage='npy', review_videos_per_person=2),
               model=dict(width=16 if fixture else 96, encoder_layers=1 if fixture else 4,
                          predictor_layers=1 if fixture else 2, heads=2 if fixture else 4,
                          patch_size=4, window_size=32 if fixture else 128),
               training=dict(pretraining_updates=1 if fixture else 2000,
                             readout_updates=1 if fixture else 2000,
                             end_to_end_updates=2 if fixture else 4000,
                             batch_size=4 if fixture else 16, learning_rate=.0003,
                             sampling='matched_cycles' if experiment_set == 'full' else 'person_motion',
                             mask_fraction=.5, change_weight=1., measurement_weight=1.,
                             repaired_tolerance=.1),
               measurement=dict(min_segment_px=2., min_frames=16, min_coverage=.8,
                                sign_tolerance_deg=1., missing_angle_penalty_deg=360.,
                                assignment_separation_px=4.,assignment_margin_px=2.),
               evaluation=dict(bootstrap_draws=100 if fixture else 2000, bootstrap_seed=731,
                               primary_candidate='M-paired_jepa-graph_time-paired_change',
                               primary_comparator='P-direct-none-paired_change',
                               confirmation_admitted=False, meaningful_margin_deg=None),
               resources=dict(account=os.environ.get('ST_ACCOUNT','mind'),
                              partition=os.environ.get('ST_PARTITION','hai'), max_jobs=8,
                              gpu='h100:1', gpu_hours=360., cpu_workers=4,
                              prepare_wall_minutes=720, phase_wall_minutes=120,
                              memory='64G', controller_memory='64G', poll_seconds=20, max_attempts=2))
    if fixture:
        cfg['resources']['gpu_hours'] = 0.
    if not fixture and source_selection == 'full_manifest':
        inherited_reservation = preparation.get('reservation_csv')
        if inherited_reservation and reservation_csv is None and not Path(inherited_reservation).is_file():
            raise FileNotFoundError(f'Declared historical AMASS reservation is missing: {inherited_reservation}. '
                                    'Restore that file or supply its reviewed replacement with --reservation-csv.')
        cfg['cohort'] = dict(preset=cohort_preset, plan_path=str(work/'cohort/manifest.json'),
            reservation_csv=str(Path(reservation_csv or inherited_reservation).expanduser().resolve()) if reservation_csv or inherited_reservation else None,
            motion_review_csv=str(Path(motion_review_csv).expanduser().resolve()) if motion_review_csv else None)
        from .cohort import plan_cohort
        cohort = plan_cohort(cfg)
        if not {'train', 'development'}.issubset(cohort['summary']['roles']):
            raise ValueError('The selected full-manifest cohort needs both training and development identities')
    work.mkdir(parents=True, exist_ok=True)
    if not fixture and source_selection == 'full_manifest':
        plan_cohort(cfg, work/'cohort')
    atomic_json(work / 'config.json', cfg)
    atomic_json(work / 'plan.json', build_plan(cfg['seeds'], experiment_set=experiment_set))
    env = {'GF_ROOT':str(code_root),'GF_WORK':str(work),'GF_PYTHON':sys.executable}
    (work / 'session.env').write_text(''.join(f'export {k}={shlex.quote(v)}\n' for k,v in env.items()))
    return cfg


def load_config(work):
    cfg = read_json(Path(work) / 'config.json')
    if cfg.get('schema') != 'gait-fidelity-config-v1':
        raise ValueError('Unknown study configuration')
    if Path(cfg['work']).resolve() != Path(work).resolve():
        raise ValueError('Run moved; preserve absolute configuration provenance')
    if cfg['mode']!=('fixture' if cfg['fixture'] else 'source') or cfg['device']!=('cpu' if cfg['fixture'] else 'cuda'):
        raise ValueError('Fixture/source mode and CPU/CUDA device must agree')
    for name in ('pretraining_updates','readout_updates','end_to_end_updates','batch_size'):
        if type(cfg['training'][name]) is not int or cfg['training'][name] <= 0:
            raise ValueError(f'Positive integer required: {name}')
    if cfg['fixture'] and (cfg['device'] != 'cpu' or max(cfg['training'][k] for k in ('pretraining_updates','readout_updates','end_to_end_updates')) > 10):
        raise ValueError('Fixtures are CPU software checks with at most ten updates')
    r = cfg['resources']
    if not 1 <= r['max_jobs'] <= 8 or not math.isfinite(r['gpu_hours']) or r['gpu_hours'] < 0:
        raise ValueError('Invalid combined concurrency or study allowance')
    if r['gpu']!='h100:1':
        raise ValueError('This scheduler reserves and requests exactly one H100 per worker')
    if any(type(r[k]) is not int or r[k]<1 for k in ('prepare_wall_minutes','phase_wall_minutes','cpu_workers','max_attempts')):
        raise ValueError('Positive allocation limits and worker/retry counts required')
    if not 0<cfg['measurement']['min_coverage']<=1 or cfg['measurement']['min_segment_px']<=0:
        raise ValueError('Invalid reference geometry/coverage admission thresholds')
    if cfg['model']['window_size'] != cfg['data']['samples']:
        raise ValueError('Model and data window sizes differ')
    if type(cfg['data'].get('num_shards', 8)) is not int or cfg['data'].get('num_shards', 8) < 1:
        raise ValueError('Preparation shard count must be a positive integer')
    if cfg['data'].get('partition', 'development') != 'development':
        raise ValueError('Ordinary runs cannot open confirmation; use the separate locked evaluation workflow')
    if read_json(Path(work)/'plan.json')!=build_plan(cfg['seeds'], experiment_set=cfg.get('experiment_set', 'full')):
        raise ValueError('Saved phase plan differs from configured seed set and registered recipes')
    return cfg


def preflight(cfg, *, gpu=False):
    """Imports and asset reads only on login; real GPU kernels on a worker."""
    import torch
    packages = {}
    names = ['numpy','pandas','scipy','matplotlib','torch','nbformat']
    if not cfg['fixture']:
        if shutil.which('ffmpeg') is None:
            raise RuntimeError('ffmpeg is required for review videos; make the existing HAIC FFmpeg executable available before launch')
        names += ['torchvision','mmcv','mmpose','mmengine','pyrender','trimesh','cv2','human_body_prior']
    for name in names:
        module = importlib.import_module(name)
        packages[name] = getattr(module,'__version__','imported')
    if not cfg['fixture']:
        if packages['torch'] != '2.6.0+cu124' or packages['torchvision'] != '0.21.0+cu124':
            raise RuntimeError('Use the working synthetic-training-cu124 interpreter (Torch2.6/Torchvision0.21).')
        p = cfg['preparation']
        asset_keys = ('manifest_dir','amass_root','body_model_root','dmpl_root','uv_path','texture_dir','background_dir')
        if cfg['data'].get('source_selection', 'legacy_roster') == 'legacy_roster':
            asset_keys += ('locomotion_audit','reservation_csv')
        for key in asset_keys:
            if not Path(p[key]).exists():
                raise FileNotFoundError(f'{key}: {p[key]}')
        if cfg['data'].get('source_selection') == 'full_manifest':
            from .cohort import load_cohort
            load_cohort(cfg, partition=cfg['data'].get('partition', 'development'))
        for estimator in p['estimators']:
            for key in ('config','checkpoint'):
                if not Path(estimator[key]).is_file():
                    raise FileNotFoundError(f"{estimator['student_id']} {key}: {estimator[key]}")
        if gpu:
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA is unavailable in the allocated GPU worker')
            x = torch.ones(16,16,device='cuda',requires_grad=True)
            (x @ x).sum().backward()
            torch.cuda.synchronize()
            from mmcv.ops import nms
            nms(torch.tensor([[0.,0.,10.,10.]],device='cuda'),torch.ones(1,device='cuda'),.5)
    return dict(status='GPU_PREFLIGHT_PASSED' if gpu else 'CPU_PREFLIGHT_PASSED',
                packages=packages, cuda_tested=gpu and not cfg['fixture'],
                device_name=torch.cuda.get_device_name(0) if gpu and not cfg['fixture'] else None)


def freeze(cfg):
    work = Path(cfg['work'])
    frozen = work / 'frozen.json'
    current = dict(config_sha256=sha256(work/'config.json'),plan_sha256=sha256(work/'plan.json'),
                   code=code_identity(cfg['code_root']))
    if frozen.exists():
        if read_json(frozen) != current:
            raise RuntimeError('Configuration, plan or code changed after launch. Use a new run directory.')
    else:
        atomic_json(frozen,current)
    return current
