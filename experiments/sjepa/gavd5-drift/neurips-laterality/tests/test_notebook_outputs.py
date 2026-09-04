from __future__ import annotations

import unittest

from nbformat.v4 import new_code_cell, new_notebook, new_output

from laterality.notebook_outputs import audit_inline_outputs


def _executed_notebook(*outputs):
    return new_notebook(
        cells=[new_code_cell("pass", execution_count=1, outputs=list(outputs))]
    )


class InlineOutputContractTests(unittest.TestCase):
    def test_accepts_embedded_figure_and_separate_result(self):
        notebook = _executed_notebook(
            new_output(
                "display_data",
                data={"image/png": "a" * 1000, "text/plain": "<Figure>"},
            ),
            new_output("execute_result", data={"text/plain": "{'passed': True}"}),
        )
        audit = audit_inline_outputs(notebook, name="valid.ipynb")
        self.assertEqual(audit["inline_png_figures"], 1)
        self.assertEqual(audit["inline_result_payloads"], 1)

    def test_rejects_figure_without_separate_result(self):
        notebook = _executed_notebook(
            new_output(
                "display_data",
                data={"image/png": "a" * 1000, "text/plain": "<Figure>"},
            )
        )
        with self.assertRaisesRegex(AssertionError, "No separate inline result"):
            audit_inline_outputs(notebook, name="figure-only.ipynb")

    def test_rejects_results_without_figure(self):
        notebook = _executed_notebook(
            new_output("execute_result", data={"text/plain": "42"})
        )
        with self.assertRaisesRegex(AssertionError, "No embedded image/png"):
            audit_inline_outputs(notebook, name="result-only.ipynb")

    def test_rejects_unexecuted_or_error_cells(self):
        unexecuted = new_notebook(cells=[new_code_cell("pass")])
        with self.assertRaisesRegex(AssertionError, "Unexecuted code cells"):
            audit_inline_outputs(unexecuted, name="unexecuted.ipynb")

        errored = _executed_notebook(
            new_output("error", ename="ValueError", evalue="bad", traceback=[]),
            new_output("display_data", data={"image/png": "a" * 1000}),
            new_output("execute_result", data={"text/plain": "42"}),
        )
        with self.assertRaisesRegex(AssertionError, "Stored error output"):
            audit_inline_outputs(errored, name="errored.ipynb")


if __name__ == "__main__":
    unittest.main()
