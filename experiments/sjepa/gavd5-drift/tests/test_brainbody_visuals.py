import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "neurips-brain-body"
PAPER_FIGURE_DIR = ROOT / "docs" / "figures"


class ScientificVisualContractTests(unittest.TestCase):
    def test_numbered_notebooks_retain_vector_visual_summaries(self):
        for path in sorted(NOTEBOOK_DIR.glob("[0-9][0-9]_*.ipynb")):
            notebook = json.loads(path.read_text(encoding="utf-8"))
            cells = [
                cell
                for cell in notebook["cells"]
                if cell.get("cell_type") == "code"
                and "visual_summary = render_notebook_summary(" in "".join(cell.get("source", []))
            ]
            self.assertEqual(len(cells), 1, path.name)
            self.assertIn("paper-figure", cells[0].get("metadata", {}).get("tags", []), path.name)
            self.assertIsNotNone(cells[0].get("execution_count"), path.name)
            mime_types = {
                mime
                for output in cells[0].get("outputs", [])
                for mime in output.get("data", {})
            }
            self.assertIn("image/svg+xml", mime_types, path.name)

    def test_paper_figures_exist_in_vector_and_raster_formats(self):
        stems = [
            "bbfm_data_funnel",
            "bbfm_protocol_execution",
            "bbfm_training_dynamics",
            "bbfm_readout_evaluation",
            "bbfm_temporal_readout_v2",
            "bbfm_anchor_drift_v2",
        ]
        for stem in stems:
            for suffix in ("svg", "png", "pdf"):
                path = PAPER_FIGURE_DIR / f"{stem}.{suffix}"
                self.assertTrue(path.is_file(), path)
                self.assertGreater(path.stat().st_size, 1_000, path)

    def test_paper_labels_single_fold_evidence(self):
        source = (NOTEBOOK_DIR / "docs" / "bbfm2026_paper_draft.tex").read_text(
            encoding="utf-8"
        )
        self.assertIn("not a cross-fold estimate", source)
        self.assertIn("bbfm_data_funnel.pdf", source)
        self.assertIn("bbfm_protocol_execution.pdf", source)


if __name__ == "__main__":
    unittest.main()
