"""Cached flow diagnostics: event support, exact pairing and person weights."""
import unittest

import numpy as np
import pandas as pd

from gavd6_sjepa.research_directions.motion_preservation.flow_diagnostics import (
    analyze_flow_case, paired_flow_cases, paired_flow_summary,
)
from gavd6_sjepa.research_directions.motion_preservation.rendering import Camera


def fixture():
    clean = np.zeros((4, 2, 3), np.float32)
    clean[..., 2] = 2
    event = clean.copy()
    event[1, 0, 0] = .2
    # One-frame event supports transitions 0->1 and 1->2 only.
    raw_transport = np.array([[2, 90], [3, 90], [90, 90], [999, 999]], np.float32)
    prior_transport = np.array([[5, 0], [7, 0], [0, 0], [-999, -999]], np.float32)
    return dict(raw=event.copy(), prior=clean.copy(), clean=clean, event_reference=event,
                observed=np.ones((4, 2), bool), confidence=np.full((4, 2), .8),
                raw_transport=raw_transport, prior_transport=prior_transport,
                flow_valid=np.ones((4, 2), bool), frame_indices=np.arange(4),
                timestamps=np.arange(4)/20, flow_reference_epe=np.array(.3))


def metadata(real=True, person="p", motion="m"):
    suffix = "real" if real else "tracking_failure"
    return dict(case_id=f"{motion}_matched_{suffix}", person_id=person, role="development",
                fixture="matched", event_present=real, prior_id="momask", flow_backend="sea_raft")


def pair_rows(person="p", motion="m", real_gap=2, failure_gap=-1):
    case = fixture()
    case["raw_transport"][:] = 10
    case["prior_transport"][:] = 10 + real_gap
    a = analyze_flow_case(case, metadata(True, person, motion))
    case["prior_transport"][:] = 10 + failure_gap
    b = analyze_flow_case(case, metadata(False, person, motion))
    return [a, b]


class CachedFlowTests(unittest.TestCase):
    def test_event_transition_union_ignores_padding_and_other_joints(self):
        case = fixture()
        camera = Camera(np.zeros(3), np.eye(3), width=32, height=32, focal_px=20)
        result = analyze_flow_case(case, metadata(), camera)
        self.assertEqual(result["event_frame_joint_count"], 1)
        self.assertEqual(result["event_transition_count"], 2)
        self.assertEqual(result["event_flow_valid_count"], 2)
        self.assertAlmostEqual(result["event_flow_coverage"], 1)
        self.assertAlmostEqual(result["event_raw_transport_mean_px"], 2.5)
        self.assertAlmostEqual(result["event_transport_gap_mean_px"], 3.5)
        self.assertAlmostEqual(result["event_displacement_mean_px"], 2)
        self.assertAlmostEqual(result["flow_reference_epe"], .3)
        # A cache without padding must produce the same measured contrast.
        for key in ("raw_transport", "prior_transport", "flow_valid"):
            case[key] = case[key][:-1]
        self.assertAlmostEqual(analyze_flow_case(case, metadata())["event_transport_gap_mean_px"], 3.5)

    def test_unavailable_measurements_are_not_zero_error_or_zero_displacement(self):
        case = fixture()
        case["flow_valid"][0, 0] = False
        case["prior_transport"][1, 0] = np.nan
        result = analyze_flow_case(case, metadata())
        self.assertEqual(result["event_flow_valid_count"], 0)
        self.assertEqual(result["event_flow_coverage"], 0)
        self.assertTrue(np.isnan(result["event_transport_gap_mean_px"]))
        self.assertTrue(np.isnan(result["event_displacement_max_px"]))
        self.assertTrue(np.isnan(result["event_projected_in_frame_count"]))
        case = fixture()
        case["observed"][1, 0] = False
        self.assertEqual(analyze_flow_case(case, metadata())["event_flow_valid_count"], 0)

    def test_reference_projection_requires_front_facing_in_frame_points(self):
        case = fixture()
        case["clean"][..., 2] = -2
        case["event_reference"][..., 2] = -2
        camera = Camera(np.zeros(3), np.eye(3), width=32, height=32, focal_px=20)
        result = analyze_flow_case(case, metadata(), camera)
        self.assertEqual(result["event_projected_in_frame_count"], 0)
        self.assertTrue(np.isnan(result["event_displacement_max_px"]))

    def test_exact_pair_contrast_and_input_mismatch(self):
        rows = pair_rows()
        pairs = paired_flow_cases(pd.DataFrame(rows))
        self.assertEqual(pairs.iloc[0].status, "valid_available_support_pair")
        self.assertAlmostEqual(pairs.iloc[0].paired_gap_difference_px, 3)
        self.assertEqual(pairs.iloc[0].both_explanations_favor_expected_path, 1)
        case = fixture()
        case["confidence"][0, 0] = .1
        rows[1] = analyze_flow_case(case, metadata(False))
        pairs = paired_flow_cases(pd.DataFrame(rows))
        self.assertEqual(pairs.iloc[0].status, "unmatched_skeleton_inputs")
        self.assertTrue(np.isnan(pairs.iloc[0].paired_gap_difference_px))

    def test_summary_weights_people_not_repeated_motions(self):
        rows = []
        for i in range(3):
            rows.extend(pair_rows("p1", f"m{i}", 3, 1))  # contrast 2 for one person
        rows.extend(pair_rows("p2", "m3", 9, 1))  # contrast 8 for another
        summary = paired_flow_summary(pd.DataFrame(rows)).iloc[0]
        self.assertEqual(summary.n_pairs, 4)
        self.assertEqual(summary.n_valid_people, 2)
        self.assertAlmostEqual(summary.paired_gap_difference_px, 5)
        self.assertEqual(summary.both_explanations_favor_expected_path, 0)

    def test_unresolved_and_incomplete_pairs_do_not_become_successes(self):
        case = fixture()
        case["flow_valid"][:] = False
        rows = [analyze_flow_case(case, metadata(flag)) for flag in (True, False)]
        pairs = paired_flow_cases(pd.DataFrame(rows))
        self.assertEqual(pairs.iloc[0].status, "unresolved_flow_support")
        summary = paired_flow_summary(pd.DataFrame(rows)).iloc[0]
        self.assertEqual(summary.n_valid_pairs, 0)
        self.assertEqual(summary.real_event_flow_coverage, 0)
        self.assertTrue(np.isnan(summary.paired_gap_difference_px))
        self.assertTrue(np.isnan(summary.both_explanations_favor_expected_path))
        self.assertEqual(paired_flow_cases(pd.DataFrame(rows[:1])).iloc[0].status, "incomplete_pair")
        duplicated = paired_flow_cases(pd.DataFrame(rows + rows[:1]))
        self.assertEqual(duplicated.iloc[0].status, "duplicate_explanation")
        ambiguous = pd.DataFrame(rows).assign(fixture="ambiguous")
        self.assertTrue(paired_flow_summary(ambiguous).empty)


if __name__ == "__main__":
    unittest.main()
