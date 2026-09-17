"""Checks that the measured mechanism cannot be explained by label leakage."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from gavd6_sjepa.research_directions.synthetic_training.config import RunConfig
from gavd6_sjepa.research_directions.synthetic_training.measurements import (
    landmark_errors, paired_recording_interval, prediction_summary, response_summary,
)
from gavd6_sjepa.research_directions.synthetic_training.selectors import (
    FEATURE_VIEWS, LessonSelector, feature_vector, selection_regret,
)
from gavd6_sjepa.research_directions.synthetic_training.trials import crop_context_images


def fixture_features(response=0.0):
    return {name: np.full(3 if name != 'budget' else 1, response if name == 'target_delta' else 0.0,
                          dtype=np.float32)
            for keys in FEATURE_VIEWS.values() for name in keys}


class SelectionTests(unittest.TestCase):
    def test_source_progress_cannot_read_or_reconstruct_target_change(self):
        first = fixture_features(1)
        second = {k: v.copy() for k, v in first.items()}
        second['target_delta'][:] = -50
        second['target_pre'][:] = 19
        second['magnitude'][:] = 3
        for view in ('after', 'source_progress', 'weakness', 'domain'):
            np.testing.assert_array_equal(feature_vector(first, view), feature_vector(second, view))
        self.assertFalse(np.array_equal(feature_vector(first, 'full'), feature_vector(second, 'full')))

    def test_response_lookup_can_recover_opposite_rankings_at_identical_snapshots(self):
        train = [fixture_features(-1), fixture_features(1)]
        gains = np.array([[1.3, .3], [.4, 1.4]])
        full = LessonSelector('full', 'nearest', 1, ['overlap', 'resolution']).fit(train, gains)
        source = LessonSelector('source_progress', 'nearest', 1, ['overlap', 'resolution']).fit(train, gains)
        self.assertEqual(full.select(train), ['overlap', 'resolution'])
        self.assertEqual(len(set(source.select(train))), 1)
        np.testing.assert_array_equal(selection_regret(full.select(train),
            np.array([[9.8, 8.5, 9.5], [9.8, 9.4, 8.4]]), full.lessons), [0, 0])

    def test_negative_utility_selects_replay_and_counts_it_in_regret(self):
        features = [fixture_features()]
        selector = LessonSelector('full', 'ridge', 1, ['a', 'b']).fit(features, np.array([[-2., -1.]]))
        self.assertEqual(selector.select(features), ['replay'])
        np.testing.assert_equal(selection_regret(['replay'], [[1, 3, 2]], ['a', 'b']), [0])

    def test_fit_scaling_does_not_use_deployment_features(self):
        source = [fixture_features(-1), fixture_features(1)]
        selector = LessonSelector('full', 'ridge', 1, ['a']).fit(source, [[-1], [1]])
        scaler = selector.model.steps[0][1]
        mean = scaler.mean_.copy()
        selector.select([fixture_features(1000)])
        np.testing.assert_equal(mean, scaler.mean_)


class MeasurementTests(unittest.TestCase):
    def test_missing_predictions_get_penalty_without_dropping_reference(self):
        ref = np.zeros((2, 12, 2))
        pred = ref.copy()
        pred[0, 0] = np.nan
        visible = np.ones((2, 12), bool)
        visible[1, 1] = False
        boxes = np.tile([0, 0, 3, 4], (2, 1))
        nle, px, failed = landmark_errors(pred, ref, visible, boxes)
        self.assertEqual(nle[0, 0], 1)
        self.assertEqual(px[0, 0], 5)
        self.assertTrue(failed[0, 0])
        self.assertTrue(np.isnan(nle[1, 1]))

    def test_pair_bootstrap_weights_recordings_not_repeated_frames(self):
        records = []
        for recording, improvement in [('a', .1), ('b', .3), ('c', -.1)]:
            for method, error in [('full', 1 - improvement), ('base', 1.)]:
                records.append(dict(recording_id=recording, domain_id='side_low', method=method, nle=error))
        original = pd.DataFrame(records)
        repeated = pd.concat([original, *([original.loc[original.recording_id.eq('a')]] * 30)])
        first = paired_recording_interval(original, 'full', 'base', samples=100)
        second = paired_recording_interval(repeated, 'full', 'base', samples=100)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first['improvement'], .1)

    def test_response_is_scale_normalized_and_handles_all_missing_joint(self):
        before = np.ones((4, 12, 2))
        after = before + 2
        before[:, 0] = np.nan
        boxes = np.tile([0, 0, 20, 10], (4, 1))
        first, magnitude = response_summary(before, after, boxes)
        second, magnitude2 = response_summary(before * 4, after * 4, boxes * 4)
        np.testing.assert_allclose(first, second)
        np.testing.assert_allclose(magnitude, magnitude2)
        self.assertTrue(np.isfinite(first).all())
        self.assertTrue(np.isfinite(prediction_summary(before, boxes)).all())

    def test_context_crop_follows_off_center_person_not_image_center(self):
        image = np.zeros((100, 200, 3), np.uint8)
        image[20:80, 10:40] = 255
        clip = crop_context_images([image, image], np.tile([10, 20, 40, 80], (2, 1)))
        self.assertGreater(clip.mean(), 100)
        self.assertLess(clip.shape[2], 50)


class ConfigurationTests(unittest.TestCase):
    def test_environment_override_precedence_and_absent_legacy_fallback(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            (root / 'run').mkdir()
            (root / 'run/config.json').write_text('{"amass_root":"json-amass","device":"cpu"}')
            env = dict(GAVD6_ROOT=str(root), ST_RUN_ROOT='run', MP_AMASS_ROOT='legacy-amass')
            with patch.dict('os.environ', env, clear=True):
                cfg = RunConfig.from_env()
                self.assertEqual(cfg.amass_root, str(root / 'json-amass'))
                self.assertEqual(cfg.root, root / 'run')
            with patch.dict('os.environ', {**env, 'ST_AMASS_ROOT': 'explicit-amass'}, clear=True):
                self.assertEqual(RunConfig.from_env().amass_root, str(root / 'explicit-amass'))

    def test_held_family_cannot_be_used_as_source_under_different_student_id(self):
        students = [dict(student_id='a', family='vitpose', role='train', config='x', checkpoint='y'),
                    dict(student_id='b', family='vitpose', role='held', config='x', checkpoint='z')]
        with self.assertRaisesRegex(ValueError, 'Held architecture'):
            RunConfig(students=students).validate()


if __name__ == '__main__':
    unittest.main()
