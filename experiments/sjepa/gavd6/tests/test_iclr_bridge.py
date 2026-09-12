"""Exact invariants and fixed synthetic calibration for the cached bridge panel."""
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from gavd6_sjepa.research_directions.future_innovation.fi_contracts import equal_source_weights, read_json, write_json, sha256_file
from gavd6_sjepa.research_directions.future_innovation.fi_joint_calibration import fixture
from gavd6_sjepa.research_directions.future_innovation.fi_joint_models import SupportedInput
from gavd6_sjepa.research_directions.future_innovation.fi_joint_training import nested_partition, predict_selected
from gavd6_sjepa.research_directions.future_innovation.fi_joint_models import JointModelContract
from gavd6_sjepa.research_directions.iclr_bridge.cached_panel import (
    accessibility_features, fit_panel_fold, evaluate_bundle, _reload_fold, _validate_roots,
)


def calibration_fixture():
    """Fixed seed48021, 80sources×2clips; future label from earliest motion bins.

    Last-frame posture is independent of earlier motion. The strong positive
    fixture validates sensitivity only, not power on subtle GAVD motion.
    """
    cohort, arrays, _ = fixture(48021, "temporal")
    cohort["decoded_fps"] = 25.
    early = arrays["skeleton"][:, :8, 0, 0].mean(axis=1)
    noise = np.random.default_rng(48022).normal(scale=.1, size=(len(cohort), 4))
    arrays["person"] = 2 * early[:, None] + noise
    return cohort, arrays


class BridgeTests(unittest.TestCase):
    def test_feature_schema_and_missing_endpoint(self):
        cohort, arrays = calibration_fixture()
        arrays["skeleton"][0,31,2,3] = 0
        arrays["skeleton"][0,31,2,:2] = 999
        support, schema = accessibility_features(cohort, arrays["skeleton"], "support")
        posture, pose_schema = accessibility_features(cohort, arrays["skeleton"], "posture")
        self.assertEqual(support.shape, (160,331)); self.assertEqual(posture.shape, (160,397))
        self.assertEqual(len(set(pose_schema[0])),397)
        np.testing.assert_array_equal(support,posture[:,:331])
        self.assertTrue(np.isnan(posture[0,335:337]).all())
        self.assertEqual(schema[0][-1],"decoded_fps")
        self.assertEqual(pose_schema[1][-1],"coordinate")

    def test_heldout_never_changes_fitted_statistics(self):
        cohort, arrays = calibration_fixture()
        x, schema = accessibility_features(cohort, arrays["skeleton"], "posture")
        train = np.arange(120); held = x[120:].copy()
        scaler = SupportedInput.fit(x[train], equal_source_weights(cohort.iloc[train].video_id),
                                    cohort.iloc[train].window_id, cohort.iloc[train].video_id, *schema)
        before = scaler.record(); held[:,0] = 1e8
        scaler.transform(held)
        self.assertEqual(before,scaler.record())
        self.assertTrue(np.equal(scaler.transform(held)[:,~scaler.mask],0).all())

    def test_source_held_temporal_signal_full_grid_and_reload(self):
        cohort, arrays = calibration_fixture()
        with tempfile.TemporaryDirectory() as temp, threadpool_limits(limits=1):
            result = fit_panel_fold(temp,"posture",0,cohort,arrays)
            test = result["test"]
            true = result["baseline"].y_scaler.transform(arrays["person"][test])
            real = np.mean((true-result["predictions"]["real-skeleton"])**2)
            null = np.mean((true-result["predictions"]["no-skeleton"])**2)
            shuffle = np.mean((true-result["predictions"]["time-shuffle"])**2)
            self.assertLess(real, .25*null)
            self.assertLess(real, .5*shuffle)
            self.assertTrue(result["complete"])
            self.assertEqual(len(result["selection"]["real-skeleton"]["candidates"]),37)
            repeat = _reload_fold(temp,"posture",0,cohort,arrays)
            np.testing.assert_array_equal(result["predictions"]["real-skeleton"],repeat["predictions"]["real-skeleton"])
            # The typed exact fallback has no arbitrary correction after reload.
            saved = result["models"]["no-skeleton"]
            self.assertEqual(saved["checkpoint_type"],"baseline_only")
            x, _ = accessibility_features(cohort,arrays["skeleton"],"posture")
            np.testing.assert_array_equal(result["predictions"]["no-skeleton"],result["baseline"].predict(x[test]))
            # Keep37candidates but alter one penalty, updating its receipt hash.
            directory = Path(temp)/"models/posture/fold-0"
            path = directory/"selection.json"; original = path.read_bytes()
            ledger = read_json(path); ledger["real-skeleton"]["candidates"][1]["lambda_x"] = 999.
            write_json(path,ledger)
            receipt_path=directory/"complete.json";receipt=read_json(receipt_path)
            receipt["artifacts"][path.name]=sha256_file(path);write_json(receipt_path,receipt)
            with self.assertRaisesRegex(ValueError,"penalty grid"):_reload_fold(temp,"posture",0,cohort,arrays)
            path.write_bytes(original);receipt["artifacts"][path.name]=sha256_file(path);write_json(receipt_path,receipt)
            # Even updated file hashes cannot excuse an altered selected coefficient.
            path = Path(temp)/"models/posture/fold-0/real-skeleton.joblib"
            model = joblib.load(path);model["model"].intercept[0] += .01;joblib.dump(model,path)
            receipt_path=path.parent/"complete.json";receipt=read_json(receipt_path)
            receipt["artifacts"][path.name]=sha256_file(path);write_json(receipt_path,receipt)
            with self.assertRaises(AssertionError):_reload_fold(temp,"posture",0,cohort,arrays)

    def test_source_bootstrap_and_exact_zero_increment(self):
        true=np.array([[1.,2.],[2.,3.],[-1.,-2.]])
        bundle={"y_true":true,"baseline":true*.7,"valid_features":np.ones(2,bool),"video_id":np.array(["a","a","b"])}
        for arm in ("real-skeleton","no-skeleton","time-shuffle","clip-mismatch"):bundle[arm]=bundle["baseline"].copy()
        scores,contrasts,bootstrap=evaluate_bundle(bundle)
        self.assertEqual(len(bootstrap),8000)
        self.assertEqual(contrasts["real_minus_no-skeleton"]["estimate"],0)
        self.assertEqual(contrasts["real_minus_no-skeleton"]["ci025"],0)
        self.assertEqual(contrasts["real_minus_no-skeleton"]["positive_fraction"],0)
        self.assertAlmostEqual(scores["real-skeleton"]["r2_full"],.91)

    def test_nested_output_roots_rejected(self):
        with self.assertRaises(ValueError):_validate_roots(Path('/a/source'),Path('/a/source/child'),Path('/b/parent'))
        _validate_roots(Path('/a/source'),Path('/a/child'),Path('/b/parent'))


if __name__ == '__main__':
    unittest.main()
