"""Unlabeled inference contracts and pixel/frame boundaries, using fixtures."""
import importlib.metadata
import json
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pytest
import torch
from sklearn.dummy import DummyClassifier

from worldmodels.data import LABELS, sha256_file
from worldmodels.demo import make_demo
from worldmodels.features import build_features
from worldmodels.inference import classify_pose, extract_video_range, _check_run
from worldmodels.model import ModelConfig, SkeletonJEPA
from worldmodels.training import save_checkpoint


@pytest.fixture
def run_and_pose(tmp_path):
    import worldmodels.inference as module
    import ambient.pose.joint_angles as shared_angles
    import ambient.pose.keypoint_extractor as shared_extractor
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    run = tmp_path / 'trusted-run'
    run.mkdir()
    demo = make_demo(n=5, frames=16, conditions=True)
    features, names = build_features(demo['x'], demo['valid'], fps=15)
    model = SkeletonJEPA(ModelConfig(frames=16, dim=16, heads=2, encoder_layers=1))
    for filename in ['normal_completion.pt', 'adaptation_4.pt', 'mixed_adaptation.pt', 'normal_forecast.pt']:
        save_checkpoint(model, run / filename)
    for name in ['rf', 'jepa_normal', 'jepa_adapted', 'jepa_mixed']:
        estimator = DummyClassifier(strategy='uniform').fit(features, demo['labels'])
        joblib.dump(estimator, run / f'{name}_classifier.joblib')
    summary = dict(identity='medwm-gait-v1', status='synthetic_fixture', features=names, pose_model_sha256='0'*64,
                   anomaly=dict(threshold=None, threshold_status='insufficient_normal_sources_for_finite_5pct_threshold'),
                   versions={name: importlib.metadata.version(name) for name in ('torch','numpy','scikit-learn','mediapipe')},
                   code_sha256={name: sha256_file(Path(module.__file__).with_name(name))
                                for name in ('data.py','pose.py','features.py','model.py','masking.py','training.py','evaluation.py','inference.py')},
                   ambient_source_sha256={'joint_angles':sha256_file(shared_angles.__file__),'keypoint_extractor':sha256_file(shared_extractor.__file__)},
                   artifact_sha256={p.name: sha256_file(p) for p in run.iterdir()})
    (run / 'results.json').write_text(json.dumps(summary))
    pose = dict(xy=demo['x'][0] * 100 + [300, 200], confidence=np.ones((16,16)),
                frame_numbers=np.arange(1,33,2), fps=np.array(30.), sequence_id='unlabeled-input',
                source_id='fixture-source', label='', fingerprint='fixture-only',
                provenance={'pose_model_sha256':'0'*64})
    yield run, pose
    torch.set_num_threads(old)


def test_unlabeled_sequence_probabilities_and_missing_threshold(run_and_pose):
    run, pose = run_and_pose
    report = classify_pose(pose, run)
    assert report['n_windows'] == 1
    assert report['person_review_status'] == 'required'
    assert report['quality']['sampled_fps'] == [15.]
    assert len(report['classifiers']) == 4
    assert report['forecast']['exceeds_threshold'] is None
    assert 'untrained_probe' in report['unavailable']
    for item in report['classifiers'].values():
        assert list(item['probabilities']) == list(LABELS)
        assert sum(item['probabilities'].values()) == pytest.approx(1)
    json.dumps(report, allow_nan=False)


def test_low_quality_returns_no_prediction(run_and_pose):
    run, pose = run_and_pose
    pose['confidence'][:] = 0
    report = classify_pose(pose, run)
    assert report['status'] == 'quality_failure'
    assert report['classifiers'] == {}
    assert report['forecast'] is None


def test_changed_classifier_refused_before_deserialization(run_and_pose, monkeypatch):
    run, _ = run_and_pose
    (run / 'rf_classifier.joblib').write_bytes(b'changed artifact')
    monkeypatch.setattr(joblib, 'load', lambda *args: pytest.fail('Must not unpickle changed files'))
    with pytest.raises(ValueError, match='artifact identity'):
        _check_run(run)


def test_one_based_decoding_rgb_and_real_sample_rate(tmp_path, monkeypatch):
    import cv2
    import ambient.pose.keypoint_extractor as module
    images = [np.broadcast_to(np.array([i+10,0,i], dtype=np.uint8),(4,6,3)).copy() for i in range(7)]
    seen = []
    class Capture:
        def __init__(self, *_): self.position = 0
        def isOpened(self): return True
        def get(self, key):
            return {cv2.CAP_PROP_FPS:30., cv2.CAP_PROP_FRAME_WIDTH:6,
                    cv2.CAP_PROP_FRAME_HEIGHT:4, cv2.CAP_PROP_POS_MSEC:self.position/30*1000}.get(key,0)
        def read(self):
            if self.position == len(images): return False, None
            frame = images[self.position]
            self.position += 1
            return True, frame
        def release(self): pass
    class Extractor:
        def extract_from_image(self, image, model):
            seen.append(tuple(map(int,image[0,0])))
            return SimpleNamespace(keypoints=[SimpleNamespace(id=11,x=3,y=2,confidence=.9)])
        def cleanup(self): pass
    monkeypatch.setattr(cv2, 'VideoCapture', Capture)
    monkeypatch.setattr(module, 'SequenceKeypointExtractor', Extractor)
    video, model = tmp_path/'video.mp4', tmp_path/'pose.task'
    video.write_bytes(b'fixture video'); model.write_bytes(b'fixture model')
    pose = extract_video_range(video, model, start_frame=2, end_frame=6)
    np.testing.assert_array_equal(pose['frame_numbers'], [2,4,6])
    np.testing.assert_array_equal(pose['decoded_indexes'], [1,3,5])
    assert seen == [(1,0,11),(3,0,13),(5,0,15)]
    assert pose['provenance']['sample_fps'] == 15
    np.testing.assert_array_equal(pose['xy'][:,11], [[3,2]]*3)


def test_pose_weight_mismatch_is_rejected(run_and_pose):
    run, pose = run_and_pose
    pose['provenance']['pose_model_sha256'] = '1'*64
    with pytest.raises(ValueError, match='pose-model weights'):
        classify_pose(pose, run)
