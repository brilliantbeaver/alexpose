"""Independent review probes; expected behavior describes the scientific boundary."""
import copy
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from gavd6_sjepa.research_directions.synthetic_training_v2.config import RunConfig
from gavd6_sjepa.research_directions.synthetic_training_v2.contracts import TrackBundle, validate_records
from gavd6_sjepa.research_directions.synthetic_training_v2.data import fixture_bundle, normalize_inputs
from gavd6_sjepa.research_directions.synthetic_training_v2.preparation import audited_motion_windows
from gavd6_sjepa.research_directions.synthetic_training_v2.training import _input_tensors


class IndependentContractReview(unittest.TestCase):
    def test_missing_confidence_adapter_is_input_only(self):
        bundle = fixture_bundle()
        before = {k: v.copy() for k, v in bundle.inputs.items()}
        normalized, norm = normalize_inputs(bundle.inputs)
        self.assertTrue(np.isfinite(normalized["confidence"]).all())
        self.assertTrue(np.array_equal(normalized["observed"], before["observed"]))
        for key in before:
            np.testing.assert_equal(bundle.inputs[key], before[key])
        _input_tensors(normalized, "cpu")
        changed = copy.deepcopy(bundle)
        changed.targets["xy"] *= 1000
        changed.targets["valid"][:] = False
        other, other_norm = normalize_inputs(changed.inputs)
        for key in normalized:
            np.testing.assert_equal(normalized[key], other[key])
        np.testing.assert_equal(norm.origin, other_norm.origin)
        np.testing.assert_equal(norm.scale, other_norm.scale)

    def test_raw_person_cannot_change_canonical_identity_to_cross_split(self):
        one = fixture_bundle().records[0]
        other = dict(one, canonical_person_id="invented-new-person",
                     window_id="other-window", motion_hash="other-motion", split="development")
        with self.assertRaises(ValueError):
            validate_records([one, other], evidence_status="fixture-tested")

    def test_gpu_budget_nan_negative_and_infinity_rejected(self):
        for field, value in (("projected_gpu_hours", np.nan),
                             ("measured_gpu_hours", -1.0),
                             ("measured_gpu_hours", np.nan),
                             ("projected_gpu_hours", np.inf)):
            with self.subTest(field=field, value=value), self.assertRaises((ValueError, PermissionError)):
                args = dict(run_id="review-budget", mode="source", bundle="explicit-bundle",
                            device="cuda", authorized_gpu_hours=1., projected_gpu_hours=.5,
                            cost_ledger="explicit-ledger")
                args[field] = value
                RunConfig(**args).require_gpu_scope()

    def test_reserved_validation_motion_excluded_before_loading(self):
        base = pd.DataFrame([dict(relative_path="motion.npz", original_split="validation",
                                  available=True, duration_s=10., person_id="known-person")])
        audit = dict(relative_path="motion.npz", start_s=0., locomotion_status="audited_locomotion",
                     audit_reviewer="reviewer", audit_evidence="independent audit",
                     audit_date="2026-09-18", exposure="unexposed_verified",
                     canonical_person_id="known-person", reserved=True)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "audit.csv"
            pd.DataFrame([audit]).to_csv(path, index=False)
            with patch("gavd6_sjepa.research_directions.motion_preservation.motion_data.load_amass_manifest",
                       return_value=base):
                allowed, excluded = audited_motion_windows("explicit-registry", "explicit-root", path)
            self.assertEqual(len(allowed), 0)
            self.assertEqual(len(excluded), 1)

    def test_confirmation_block_happens_before_array_open(self):
        bundle = fixture_bundle()
        bundle.records[0]["split"] = "confirmation"
        bundle.records[0]["exposure"] = "unexposed_verified"
        # Write only manifest: if loader opened any arrays before the protection
        # check, this deliberately incomplete bundle would fail differently.
        import json
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "manifest.json").write_text(json.dumps(dict(records=bundle.records)))
            with patch("numpy.load", side_effect=AssertionError("opened protected targets")):
                with self.assertRaises(PermissionError):
                    TrackBundle.load(folder)


if __name__ == "__main__":
    unittest.main()
