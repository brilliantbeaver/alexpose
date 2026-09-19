"""Post-run plots retain physical time, masks and metadata-only selection."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("stv2_postrun_plots", ROOT /
    "scripts/research_directions/synthetic_training_v2/diagnostics/plots.py")
PLOTS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PLOTS)


class PostrunPlotsTests(unittest.TestCase):
    def data(self):
        time = 17. + np.arange(12) / 25.
        xy = np.full((2, 12, 12, 2), 100.)
        xy[:, :, :, 0] += 6 * np.sin(np.arange(12)[:, None] / 3)
        rows = [dict(canonical_person_id="person/a", window_id="window:0", split="development",
                     extractor="fixture-extractor", variant=variant) for variant in ("clean", "blur")]
        inputs = dict(xy=xy.copy(), timestamps=np.tile(time, (2, 1)), observed=np.ones((2, 12, 12), bool))
        inputs["observed"][1, 3:5, 10] = False
        inputs["xy"][~inputs["observed"]] = np.nan
        targets = dict(xy=xy.copy(), valid=np.ones((2, 12, 12), bool), visible=np.ones((2, 12, 12), bool))
        targets["visible"][1, 3:5, 10] = False
        predictions = {name: xy + i * .2 for i, name in enumerate(("joint_offset", "joint_affine", "initialized",
                                                                 "coordinate", "direct", "paired_jepa"))}
        return inputs, targets, rows, predictions

    def test_selection_is_invariant_to_row_order_models_errors_and_variants(self):
        rows = [dict(canonical_person_id=p, window_id=w, split="development", variant=v, error=e)
                for p in ("person-b", "person-a") for w in ("window-1", "window-0")
                for v, e in (("blur", -100), ("clean", 100))]
        rows.append(dict(canonical_person_id="aaa", window_id="aaa", split="train", error=-1000))
        expected = [dict(canonical_person_id=p, window_id=w) for p, w in
                    (("person-a", "window-0"), ("person-b", "window-0"), ("person-a", "window-1"))]
        self.assertEqual(PLOTS.select_windows(rows, 3), expected)
        self.assertEqual(PLOTS.select_windows(list(reversed(rows)), 3), expected)
        self.assertEqual(PLOTS.select_windows([{**r, "error": 9999, "model": "other"} for r in rows], 3), expected)
        self.assertEqual(PLOTS.select_windows(rows, 0), [])
        with self.assertRaises(ValueError):
            PLOTS.select_windows(rows, -1)
        with self.assertRaises(ValueError):
            PLOTS.select_windows([dict(window_id="a")])

    def test_figures_have_svg_and_png_for_each_variant(self):
        inputs, targets, rows, predictions = self.data()
        with tempfile.TemporaryDirectory() as temp:
            files = PLOTS.plot_trajectories(temp, inputs, targets, rows, predictions, PLOTS.select_windows(rows))
            self.assertEqual(len(files), 4)
            self.assertEqual({Path(f).suffix for f in files}, {".svg", ".png"})
            self.assertTrue(all(Path(f).stat().st_size > 1000 for f in files))
            for file in files:
                if file.endswith(".svg"):
                    tree = ET.parse(file)
                    text = " ".join(tree.getroot().itertext())
                    for title in ("Left ankle x", "Left ankle y", "Right ankle x", "Right ankle y", "Paired JEPA"):
                        self.assertIn(title, text)
                    self.assertIn("Reference (occluded proxies included)", text)
                    if "-blur-" in file:
                        self.assertIn("Bilateral visible reference: 10/12 frames", text)
                    else:
                        self.assertIn("Bilateral visible reference: 12/12 frames", text)
                    self.assertNotIn("nan", " ".join(element.attrib.get("d", "") for element in tree.iter()).lower())
            self.assertEqual(len(set(files)), 4)

    def test_actual_trace_time_and_missing_input_are_preserved(self):
        inputs, targets, rows, predictions = self.data()
        captured = []
        def capture(figure, folder, stem):
            captured.append(figure)
            return []
        with tempfile.TemporaryDirectory() as temp, patch.object(PLOTS, "_save_figure", side_effect=capture):
            PLOTS.plot_trajectories(temp, inputs, targets, rows, predictions, PLOTS.select_windows(rows))
        blur = captured[0]  # Metadata sorts blur before clean; no error-based selection.
        first = blur.axes[0]
        curves = {line.get_label(): line for line in first.lines}
        np.testing.assert_allclose(curves["Unchanged input"].get_xdata(), np.arange(12) / 25.)
        self.assertTrue(np.isnan(curves["Unchanged input"].get_ydata()[3:5]).all())
        self.assertTrue(np.isfinite(curves["Joint offset"].get_ydata()[3:5]).all())
        self.assertEqual(len(blur.axes), 8)
        self.assertLessEqual(max(len(axis.lines) for axis in blur.axes), 5)
        for figure in captured:
            figure.clear()

    def test_multi_seed_ambiguous_records_and_invalid_times_fail(self):
        inputs, targets, rows, predictions = self.data()
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "select one declared seed"):
                PLOTS.plot_trajectories(temp, inputs, targets, rows, {**predictions, "paired_jepa_seed18": predictions["paired_jepa"]},
                                        PLOTS.select_windows(rows))
            with self.assertRaisesRegex(ValueError, "Repeated window"):
                PLOTS.plot_trajectories(temp, inputs, targets, [rows[0], rows[0]], predictions, PLOTS.select_windows(rows))
            inputs["timestamps"][0, 3] = inputs["timestamps"][0, 2]
            with self.assertRaisesRegex(ValueError, "strictly increasing"):
                PLOTS.plot_trajectories(temp, inputs, targets, rows, predictions, PLOTS.select_windows(rows))

    def test_training_exports_varied_histories_without_convergence_claim(self):
        with tempfile.TemporaryDirectory() as temp:
            source, output = Path(temp) / "source", Path(temp) / "diagnostics"
            reports = {
                "paired-17": dict(arm="paired_jepa", seed=17, status="complete", optimizer_updates=4, planned_updates=4,
                    elapsed_seconds=11., teacher_initialization_weight=.98, objective="latent then readout",
                    phases=[dict(name="pretrain", updates=2), dict(name="readout", updates=2)],
                    history=[dict(phase="pretrain", phase_update=1, update=1, loss=3., ema=.99),
                             dict(phase="pretrain", phase_update=2, update=2, loss=2., ema=.999,
                                  online_features=dict(count=12, mean_std=.3, effective_rank=4.)),
                             dict(phase="readout", phase_update=1, update=3, loss=.5),
                             dict(phase="readout", phase_update=2, update=4, loss=.3)]),
                "direct-18": dict(arm="direct", seed=18, status="interrupted", phases=[dict(name="end_to_end", updates=4)],
                    history={"end_to_end": [dict(phase_update=1, loss=2), dict(phase_update=2, loss=None),
                                                dict(phase_update=3, loss=1)]}),
                "empty-17": dict(arm="initialized", seed=17, history=[]),
            }
            for fit, report in reports.items():
                path = source / "fits" / fit / "training.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps(report))
            before = {str(p): p.read_bytes() for p in source.rglob("*") if p.is_file()}
            frame = PLOTS.inspect_training(source, output)
            self.assertEqual(len(frame), 4)
            row = frame[(frame.fit == "paired-17") & (frame.phase == "pretrain")].iloc[0]
            self.assertAlmostEqual(row.loss_slope, -1.)
            self.assertEqual(row.recorded_phase_updates, 2)
            self.assertEqual(row.last_online_features_effective_rank, 4.)
            self.assertAlmostEqual(row.phase_last_ema, .999)
            self.assertEqual(frame[frame.fit == "direct-18"].iloc[0].nonfinite_loss_entries, 1)
            self.assertEqual(frame[frame.fit == "empty-17"].iloc[0].history_entries, 0)
            self.assertEqual(len(frame.attrs["figure_paths"]), 6)
            self.assertTrue(all(Path(path).parent == output / "images" for path in frame.attrs["figure_paths"]))
            self.assertEqual(Path(frame.attrs["summary_path"]).parent, output)
            history = pd.read_csv(frame.attrs["history_path"])
            self.assertEqual(len(history), 7)
            self.assertIn("online_features_effective_rank", history.columns)
            self.assertEqual(before, {str(p): p.read_bytes() for p in source.rglob("*") if p.is_file()})
            for path in (output / "images").glob("training-*.svg"):
                self.assertIn("does not establish convergence", " ".join(ET.parse(path).getroot().itertext()))

    def test_no_training_reports_produces_readable_empty_tables(self):
        with tempfile.TemporaryDirectory() as temp:
            frame = PLOTS.inspect_training(Path(temp) / "absent", Path(temp) / "diagnostics")
            self.assertTrue(frame.empty)
            self.assertEqual(frame.attrs["figure_paths"], [])
            self.assertTrue(pd.read_csv(frame.attrs["summary_path"]).empty)
            self.assertTrue(pd.read_csv(frame.attrs["history_path"]).empty)


if __name__ == "__main__":
    unittest.main()
