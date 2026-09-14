"""Documentation checks distinguish parsed math, rendered links and literal code."""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import shutil
import tempfile
import unittest

from scripts.workspace_management.validate_documentation import (
    HTMLReferences, discover, file_reference, looks_like_math_code,
    source_table_rows, unsafe_table_math, validate,
)


class DocumentationSyntaxTests(unittest.TestCase):
    def test_table_math_checks_only_table_rows_outside_fences(self):
        source = """Prose $|x|$ is not a table.

| quantity | expression |
| --- | --- |
| norm | $|x|$ |
| safe | $\\lvert x\\rvert$ |

```markdown
| example | math |
| --- | --- |
| value | $|x|$ |
```
"""
        self.assertEqual(list(unsafe_table_math(source)), [(5, "$|x|$")])
        self.assertEqual(len(list(source_table_rows(source))), 3)

    def test_html_references_include_images_and_anchors(self):
        parser = HTMLReferences()
        parser.feed('<a id="here" href="doc.md#section">doc</a><img src="a.svg" srcset="a.svg 1x, b.svg 2x">')
        self.assertEqual(parser.anchors, {"here"})
        self.assertEqual(set(parser.links), {"doc.md#section", "a.svg", "b.svg"})

    def test_code_math_requires_complete_delimiters(self):
        self.assertTrue(looks_like_math_code(r"$x^2$"))
        self.assertTrue(looks_like_math_code(r"\[x+y\]"))
        self.assertFalse(looks_like_math_code("cost is $10"))
        self.assertFalse(looks_like_math_code(r"\operatorname"))

    def test_file_reference_does_not_interpret_commands_or_patterns(self):
        self.assertTrue(file_reference("docs/study.md"))
        self.assertFalse(file_reference("python scripts/run.py"))
        self.assertFalse(file_reference("outputs/*.json"))

    def test_discovery_excludes_runtime_and_replay_but_includes_archived_prose(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            names = ["docs/README.md", "notes/archive/old.md", "work/report.md",
                     "scripts/archive/code_layout_20260913/README.md", ".venv/note.md"]
            for name in names:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("text")
            (root / "alias.md").symlink_to(root / "docs/README.md")
            self.assertEqual({str(p.relative_to(root)) for p in discover(root, [])},
                             {"alias.md", "docs/README.md", "notes/archive/old.md"})


@unittest.skipUnless(shutil.which("pandoc"), "local Pandoc is required for real math-renderer validation")
class DocumentationRenderingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        (cls.root / "docs").mkdir()
        (cls.root / "images").mkdir()
        (cls.root / "README.md").write_text(r"""# Heading *One*
# Heading *One*

[good](docs/target.md#target-heading) [duplicate](#heading-one-1)
[raw](docs/target.md#explicit) [bad anchor](docs/target.md#absent)
[broken](docs/missing.md) [evidence](outputs/absent.json)

![image](images/valid.svg)
<img src="images/valid.svg">
[reference][target]
[target]: docs/target.md#target-heading

Code `[fake](missing.md)` is not a link. `$z^2$` is displayed literally.
Prose \operatorname is not authored math. Price \$10.
`docs/not-yet-written.md` is a reported code reference.

Actual $x^2+y^2$ and $\operatorname{rank}(X)$ and $\unknowncommand{x}$.

Legacy \(x+1\), and
\[
y^2
\]

| expression | meaning |
| --- | --- |
| $|x|$ | absolute value |
""")
        (cls.root / "docs/target.md").write_text('# Target heading\n<a id="explicit"></a>\n')
        (cls.root / "images/valid.svg").write_text('<svg xmlns="http://www.w3.org/2000/svg"><text>x²</text></svg>')
        (cls.root / "images/raw.svg").write_text(r'<svg xmlns="http://www.w3.org/2000/svg"><text>$\alpha$</text></svg>')
        notebook = {"cells": [
            {"cell_type": "markdown", "source": ["Math $a+b$. ![ok](attachment:ok.png) ![bad](attachment:bad.png)"],
             "attachments": {"ok.png": {"image/png": "ignored"}}},
            {"cell_type": "code", "source": "ignored", "outputs": [{"text": "$\\operatorname{ignored}$"}]},
        ]}
        (cls.root / "example.ipynb").write_text(json.dumps(notebook))
        cls.report = validate(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def issues(self, kind):
        return [issue for issue in self.report["issues"] if issue["kind"] == kind]

    def test_math_is_parsed_and_unknown_commands_really_fail_rendering(self):
        self.assertEqual(len(self.issues("forbidden_math_macro")), 1)
        self.assertEqual(len(self.issues("math_render")), 1)
        self.assertIn(r"\unknowncommand", self.issues("math_render")[0]["detail"])
        self.assertEqual(len(self.issues("legacy_math_delimiters")), 2)
        self.assertEqual(len(self.issues("table_math_pipe")), 1)
        self.assertEqual(len(self.issues("math_in_code")), 1)
        self.assertEqual(self.report["summary"]["rendered_expressions"], self.report["summary"]["unique_expressions"] - 1)

    def test_resolves_real_links_and_distinguishes_evidence_and_code(self):
        self.assertEqual([i["target"] for i in self.issues("broken_local_link")], ["docs/missing.md", "outputs/absent.json"])
        self.assertEqual([i["target"] for i in self.issues("broken_local_anchor")], ["docs/target.md#absent"])
        self.assertEqual([i["target"] for i in self.issues("unavailable_code_reference")], ["docs/not-yet-written.md"])

    def test_svg_math_and_notebook_attachments_are_not_ignored(self):
        self.assertEqual(len(self.issues("svg_literal_math")), 1)
        self.assertEqual([i["target"] for i in self.issues("missing_attachment")], ["attachment:bad.png"])
        self.assertEqual(self.issues("missing_attachment")[0]["cell"], 1)
        self.assertFalse(any("ignored" in e["tex"] for e in self.report["expressions"]))

    def test_selected_doc_resolves_anchors_in_unselected_targets(self):
        selected = validate(self.root, [Path("README.md")])
        self.assertEqual([i["target"] for i in selected["issues"] if i["kind"] == "broken_local_anchor"],
                         ["docs/target.md#absent"])

    def test_latex_float_labels_and_math_are_retained(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "paper.tex").write_text(r"""\documentclass{article}
\begin{document}
See Table~\ref{tab:values}.
\begin{table*}
\caption{Values}\label{tab:values}
\begin{tabular}{ll}
Quantity & Value \\
Ratio & $\frac{x}{y}$ \\
\end{tabular}
\end{table*}
\end{document}
""")
            report = validate(root)
            self.assertEqual(report["summary"]["errors"], 0, report["issues"])
            self.assertTrue(any(e["tex"] == r"\frac{x}{y}" for e in report["expressions"]))

    def navigation_fixture(self, root):
        (root / "docs/repository").mkdir(parents=True)
        notebook = root / "saved.ipynb"
        notebook.write_text(json.dumps({"cells": [{"cell_type": "markdown",
            "source": "[old](old.md) and $x^2$."}]}))
        (root / "current.md").write_text("# Current\n")
        (root / "companion.md").write_text("# Navigation\n")
        registry = root / "docs/repository/notebook-navigation.json"
        registry.write_text(json.dumps({"schema_version": 1, "companion": "companion.md#navigation",
            "notebooks": [{"path": "saved.ipynb", "sha256": hashlib.sha256(notebook.read_bytes()).hexdigest(),
                "links": [{"historical_target": "old.md", "current_target": "current.md#current"}]}]}))
        return notebook, registry

    def test_preserved_notebook_navigation_is_exact_and_strict_mode_reports_original(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            notebook, _ = self.navigation_fixture(root)
            # Identical bytes at an unregistered path do not inherit an exception.
            (root / "unregistered.ipynb").write_bytes(notebook.read_bytes())
            report = validate(root)
            preserved = [i for i in report["issues"] if i["kind"] == "preserved_notebook_navigation"]
            self.assertEqual(len(preserved), 1)
            self.assertIn("current.md#current", preserved[0]["detail"])
            broken = [i for i in report["issues"] if i["kind"] == "broken_local_link"]
            self.assertEqual([i["file"] for i in broken], ["unregistered.ipynb"])
            self.assertEqual(report["summary"]["math_occurrences"], 2)
            strict = validate(root, strict_archive_links=True)
            self.assertEqual(len([i for i in strict["issues"] if i["kind"] == "broken_local_link"]), 2)

    def test_changed_historical_bytes_invalidate_navigation_and_math_is_still_audited(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            notebook, _ = self.navigation_fixture(root)
            data = json.loads(notebook.read_text())
            data["cells"][0]["source"] += r" $\operatorname{rank}(X)$"
            notebook.write_text(json.dumps(data))
            kinds = [i["kind"] for i in validate(root)["issues"]]
            self.assertIn("notebook_navigation_hash", kinds)
            self.assertIn("broken_local_link", kinds)
            self.assertIn("forbidden_math_macro", kinds)
            self.assertNotIn("preserved_notebook_navigation", kinds)

    def test_navigation_requires_both_replacement_and_companion_anchors(self):
        for filename in ("current.md", "companion.md"):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                self.navigation_fixture(root)
                (root / filename).write_text("# Wrong anchor\n")
                kinds = [i["kind"] for i in validate(root)["issues"]]
                self.assertIn("notebook_navigation_target", kinds)
                self.assertIn("broken_local_link", kinds)
                self.assertNotIn("preserved_notebook_navigation", kinds)

    def test_markdown_list_and_setext_heading_cannot_silently_consume_formulas(self):
        for expression in ("x\n + y", "x\n=\ny"):
            with self.subTest(expression=expression), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                (root / "math.md").write_text("$$\n" + expression + "\n$$\n")
                report = validate(root)
                kinds = [i["kind"] for i in report["issues"]]
                self.assertIn("unparsed_math", kinds)
                self.assertIn("orphan_math_delimiter", kinds)
                # The failed GFM formula is still mathematically parsed/rendered.
                self.assertEqual(report["summary"]["math_occurrences"], 1)
                self.assertEqual(report["summary"]["rendered_expressions"], 1)

    def test_orphan_delimiters_respect_code_escaping_currency_and_valid_math(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = r"""Price $10 and $20.

Escaped \$\$. Code `$$` and `$x$`.

```markdown
$$
x
 + y
$$
```

$$
\begin{aligned}
x &= y \\
&\quad + z
\end{aligned}
$$
"""
            path = root / "math.md"
            path.write_text(source)
            self.assertEqual(validate(root)["summary"]["errors"], 0)
            # No closing delimiter: the secondary reader cannot create a Math
            # node, so source-position inspection must still catch this case.
            path.write_text(source + "\nUnfinished display:\n$$\nx + y\n")
            report = validate(root)
            self.assertEqual(len([i for i in report["issues"] if i["kind"] == "orphan_math_delimiter"]), 1)

    def test_explicit_tex_syntax_examples_differ_from_untagged_math_in_code(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "syntax.md").write_text("Use `$...$`.\n\n```latex\n$$x^2$$\n```\n\n```\n$$x^2$$\n```\n")
            report = validate(root)
            examples = [i for i in report["issues"] if i["kind"] == "math_syntax_example"]
            self.assertEqual(len(examples), 2)
            self.assertTrue(all(i["severity"] == "info" for i in examples))
            accidental = [i for i in report["issues"] if i["kind"] == "math_in_code"]
            self.assertEqual(len(accidental), 1)
            self.assertEqual(accidental[0]["severity"], "warning")

    def test_symlink_content_is_checked_from_the_directory_where_it_is_opened(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "docs/study").mkdir(parents=True)
            (root / "docs/study/figure.svg").write_text('<svg xmlns="http://www.w3.org/2000/svg"/>')
            canonical = root / "docs/study/README.md"
            canonical.write_text("![Figure](figure.svg)\n")
            (root / "README.md").symlink_to(canonical)
            # Directory aliases are deliberately not recursively traversed.
            (root / "directory_alias").symlink_to(root / "docs", target_is_directory=True)
            report = validate(root)
            broken = [i for i in report["issues"] if i["kind"] == "broken_local_link"]
            self.assertEqual([(i["file"], i["target"]) for i in broken], [("README.md", "figure.svg")])
            self.assertEqual(report["scope"]["files"], 3)
            self.assertEqual(report["scope"]["symlink_files"], 1)


if __name__ == "__main__":
    unittest.main()
