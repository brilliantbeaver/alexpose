"""Source-style nested metadata must survive saved-metric verification."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from gavd6_sjepa.research_directions.synthetic_training_v2.data import fixture_bundle
from gavd6_sjepa.research_directions.synthetic_training_v2.workflow import _write_prediction, reconstruct_metrics

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('stv2_result_checker',
    ROOT / 'scripts/research_directions/synthetic_training_v2/check_results.py')
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)
STATUS = dict(frames=64, missing_detections=0, native_score_max=.9552077054977417,
              native_score_min=.6736583709716797, nonfinite_joints=0,
              nonfinite_score_joints=0, nonpositive_score_joints=0,
              score_semantics='native_mmpose_keypoint_scores_not_probabilities',
              score_threshold=0., unsupported_scores=0)


class SavedMetricVerificationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        bundle = fixture_bundle().subset('development')
        for record in bundle.records:
            record['extraction_status'] = copy.deepcopy(STATUS)
        _write_prediction(self.root / 'predictions', 'unchanged', 17, bundle.inputs['xy'], bundle)
        self.metrics = reconstruct_metrics(self.root)
        self.path = self.root / 'per-window.csv'
        self.metrics.to_csv(self.path, index=False)

    def test_nested_source_metadata_roundtrip_matches_without_rewriting_evidence(self):
        loaded = pd.read_csv(self.path)
        self.assertIsInstance(loaded.extraction_status.iloc[0], str)
        self.assertIsInstance(self.metrics.extraction_status.iloc[0], dict)
        with self.assertRaises(AssertionError):
            pd.testing.assert_frame_equal(loaded, self.metrics, check_dtype=False, check_exact=False)
        before = self.path.read_bytes()
        CHECKER.assert_saved_metrics_match(self.path, self.metrics)
        self.assertEqual(self.path.read_bytes(), before)

    def test_changed_nested_value_is_still_rejected(self):
        saved = self.metrics.copy(deep=True)
        changed = dict(STATUS, missing_detections=1)
        saved.at[0, 'extraction_status'] = changed
        saved.to_csv(self.path, index=False)
        with self.assertRaisesRegex(AssertionError, 'extraction_status'):
            CHECKER.assert_saved_metrics_match(self.path, self.metrics)

    def test_changed_coordinate_metric_is_still_rejected(self):
        saved = self.metrics.copy()
        saved.loc[0, 'visible_nle'] += .01
        saved.to_csv(self.path, index=False)
        with self.assertRaisesRegex(AssertionError, 'visible_nle'):
            CHECKER.assert_saved_metrics_match(self.path, self.metrics)

    def test_key_order_and_row_order_do_not_change_identity(self):
        saved = self.metrics.copy()
        saved['extraction_status'] = saved.extraction_status.map(lambda x: dict(reversed(list(x.items()))))
        saved.iloc[::-1].to_csv(self.path, index=False)
        CHECKER.assert_saved_metrics_match(self.path, self.metrics)

    def test_absent_score_range_uses_none_without_becoming_a_string(self):
        metrics = self.metrics.copy()
        metrics['extraction_status'] = [dict(STATUS, native_score_min=None, native_score_max=None)
                                        for _ in range(len(metrics))]
        metrics.to_csv(self.path, index=False)
        CHECKER.assert_saved_metrics_match(self.path, metrics)

    def test_expression_or_malformed_metadata_is_not_evaluated_or_ignored(self):
        for value in ('dict(frames=64)', '{broken', ''):
            with self.subTest(value=value):
                saved = self.metrics.copy()
                saved.at[0, 'extraction_status'] = value
                saved.to_csv(self.path, index=False)
                with self.assertRaises((ValueError, SyntaxError)):
                    CHECKER.assert_saved_metrics_match(self.path, self.metrics)

    def test_fixture_without_nested_metadata_remains_supported(self):
        metrics = self.metrics.drop(columns=['extraction_status'])
        metrics.to_csv(self.path, index=False)
        CHECKER.assert_saved_metrics_match(self.path, metrics)

    def test_missing_column_is_rejected(self):
        self.metrics.drop(columns=['extraction_status']).to_csv(self.path, index=False)
        with self.assertRaisesRegex(ValueError, 'columns differ'):
            CHECKER.assert_saved_metrics_match(self.path, self.metrics)


if __name__ == '__main__':
    unittest.main()
