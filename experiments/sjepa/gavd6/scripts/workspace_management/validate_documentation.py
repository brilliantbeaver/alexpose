#!/usr/bin/env python3
"""Validate authored documentation without rewriting it or fetching remote resources.

Pandoc reads GFM links, headings and math. Its local TeXMath renderer verifies
actual Math AST nodes by converting them to MathML; a second Markdown reader
identifies legacy backslash math delimiters that GFM would display as text.
Missing experiment outputs and code-example paths are reported separately from
broken document links. This is a rendering/link check, not a scientific audit or
a full LaTeX publication build. Requires a local ``pandoc`` executable.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from html.parser import HTMLParser
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Iterator
from urllib.parse import unquote, urlsplit
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
SUFFIXES = {".md", ".ipynb", ".tex", ".svg"}
SKIP_DIRS = {".git", ".venv", "node_modules", "__pycache__", ".ipynb_checkpoints",
             ".pytest_cache", ".mypy_cache", ".ruff_cache"}
SKIP_ROOTS = {"work", "outputs", "data", "cache", ".cache", "tmp", ".codex", ".agents"}
IMMUTABLE_REPLAY = Path("scripts/archive/code_layout_20260913")
EVIDENCE_ROOTS = {"work", "outputs", "artifacts", "data", "cache", ".cache", "runs", "results"}
LEGACY_READER = "markdown+tex_math_single_backslash+gfm_auto_identifiers-smart-latex_macros"


def nodes(value: Any) -> Iterator[dict[str, Any]]:
    """Walk Pandoc nodes, including nodes in tables and nested link labels."""
    if isinstance(value, dict):
        if "t" in value:
            yield value
        for child in value.values():
            yield from nodes(child)
    elif isinstance(value, list):
        for child in value:
            yield from nodes(child)


@dataclass
class Issue:
    severity: str
    kind: str
    file: str
    detail: str
    cell: int | None = None
    line: int | None = None
    target: str | None = None


@dataclass
class Document:
    path: Path
    source: str
    cell: int | None = None
    ast: dict[str, Any] = field(default_factory=dict)
    legacy_ast: dict[str, Any] | None = None
    attachments: set[str] = field(default_factory=set)
    parse_error: str | None = None
    parse_warnings: str = ""

    def line_for(self, text: str) -> int | None:
        offset = self.source.find(text)
        return self.source.count("\n", 0, offset) + 1 if offset >= 0 else None


class HTMLReferences(HTMLParser):
    """Read actual HTML attributes rather than matching apparent links in prose."""
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[str] = []
        self.anchors: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        for key in ("id", "name" if tag == "a" else "id"):
            if values.get(key):
                self.anchors.add(values[key] or "")
        for key in ("href", "src", "poster", "data" if tag == "object" else "src"):
            if values.get(key) and values[key] not in self.links:
                self.links.append(values[key] or "")
        # Data URLs may contain commas; they are self-contained, not local files.
        if values.get("srcset") and "data:" not in (values["srcset"] or ""):
            self.links.extend(part.strip().split()[0] for part in
                              (values["srcset"] or "").split(",") if part.strip())

    handle_startendtag = handle_starttag


class MathMLResults(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.current: str | None = None
        self.rendered: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "div":
            self.current = dict(attrs).get("id")
        elif tag == "math" and self.current:
            self.rendered.add(self.current)

    def handle_endtag(self, tag: str) -> None:
        if tag == "div":
            self.current = None


def discover(root: Path, selected: list[Path]) -> list[Path]:
    result: set[Path] = set()
    for selection in selected or [root]:
        selection = selection if selection.is_absolute() else root / selection
        if selection.is_file():
            candidates = [selection]
        elif selection.is_dir() and not selection.is_symlink():
            candidates = []
            for directory, subdirs, filenames in os.walk(selection, followlinks=False):
                here = Path(directory)
                subdirs[:] = [name for name in subdirs if name not in SKIP_DIRS
                              and not (here == root and name in SKIP_ROOTS)
                              and not (here / name).is_symlink()
                              and not (here / name).is_relative_to(root / IMMUTABLE_REPLAY)]
                candidates.extend(here / name for name in filenames)
        else:
            raise FileNotFoundError(f"Selected documentation path does not exist: {selection}")
        for path in candidates:
            if (path.suffix.lower() in SUFFIXES
                    and not path.is_relative_to(root / IMMUTABLE_REPLAY)
                    and not path.resolve().is_relative_to(root / IMMUTABLE_REPLAY)):
                result.add(path)
    return sorted(result)


def parse_document(doc: Document, pandoc: str) -> Document:
    # Do not expand document-class/preamble macros: etoolbox and IEEE class
    # redefinitions are not executable math and may recurse in a text reader.
    reader = "latex-latex_macros" if doc.path.suffix == ".tex" else "gfm+sourcepos"
    # Pandoc drops table* blocks although their body has ordinary table/math
    # syntax. Treat their float width as a layout instruction for inspection.
    source = (doc.source.replace(r"\begin{table*}", r"\begin{table}")
              .replace(r"\end{table*}", r"\end{table}")) if doc.path.suffix == ".tex" else doc.source
    try:
        result = subprocess.run([pandoc, "--from=" + reader, "--to=json"],
                                input=source, text=True, capture_output=True,
                                cwd=doc.path.parent, timeout=60)
        if result.returncode:
            doc.parse_error = result.stderr.strip() or "Pandoc parser failed"
            return doc
        doc.ast = json.loads(result.stdout)
        doc.parse_warnings = result.stderr.strip()
        if reader.startswith("gfm") and ("$" in doc.source or r"\(" in doc.source or r"\[" in doc.source):
            legacy = subprocess.run([pandoc, "--from=" + LEGACY_READER, "--to=json"],
                                    input=doc.source, text=True, capture_output=True,
                                    cwd=doc.path.parent, timeout=60)
            if legacy.returncode:
                doc.parse_error = legacy.stderr.strip() or "Legacy math parser failed"
            else:
                doc.legacy_ast = json.loads(legacy.stdout)
    except (OSError, ValueError, subprocess.TimeoutExpired) as error:
        doc.parse_error = str(error)
    return doc


def document_parts(path: Path) -> list[Document]:
    source = path.read_text(encoding="utf-8")
    if path.suffix != ".ipynb":
        return [Document(path, source)]
    notebook = json.loads(source)
    return [Document(path, "".join(cell.get("source", "")), cell=index + 1,
                     attachments=set(cell.get("attachments", {})))
            for index, cell in enumerate(notebook.get("cells", []))
            if cell.get("cell_type") == "markdown"]


def supplementary_math(doc: Document) -> list[tuple[str, str, bool]]:
    """Math-first parsing exposes formulas that GFM consumed as lists/headings."""
    result: list[tuple[str, str, bool]] = []
    # GFM strips indentation after newlines inside a display; this whitespace
    # does not change TeX. Preserve spaces within lines (notably in \text{}).
    def key_for(kind: str, tex: str) -> tuple[str, str]:
        return kind, re.sub(r"(?<=\n)[ \t]+", "", tex)

    existing = Counter(key_for(node["c"][0]["t"], node["c"][1])
                       for node in nodes(doc.ast) if node["t"] == "Math")
    for node in nodes(doc.legacy_ast):
        if node["t"] != "Math":
            continue
        key = node["c"][0]["t"], node["c"][1]
        normalized = key_for(*key)
        if existing[normalized]:
            existing[normalized] -= 1
        else:
            # GFM's optional backticks inside inline dollars are delimiters;
            # the generic Markdown reader retains them in the TeX string.
            unquoted = key_for(key[0], key[1][1:-1]) if key[1].startswith("`") and key[1].endswith("`") else None
            if unquoted and existing[unquoted]:
                existing[unquoted] -= 1
                continue
            left, right = (r"\(", r"\)") if key[0] == "InlineMath" else (r"\[", r"\]")
            legacy = bool(re.search(re.escape(left) + r"\s*" + re.escape(key[1]) + r"\s*" + re.escape(right), doc.source))
            result.append((*key, legacy))
    return result


def math_nodes(doc: Document) -> list[tuple[str, str, bool]]:
    return [(node["c"][0]["t"], node["c"][1], False)
            for node in nodes(doc.ast) if node["t"] == "Math"] + supplementary_math(doc)


def orphan_display_delimiters(doc: Document) -> Iterator[int]:
    """Find actual unescaped dollars rendered as prose using GFM source spans.

    Code, comments and Math nodes do not expose prose Str spans. Escaped
    currency has a different raw source slice, so it is not treated as math.
    """
    offsets = [0]
    for line in doc.source.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    dollars: set[int] = set()
    for node in nodes(doc.ast):
        if node["t"] != "Span":
            continue
        attr, children = node["c"]
        if len(children) != 1 or children[0]["t"] != "Str" or "$" not in children[0]["c"]:
            continue
        position = dict(attr[2]).get("data-pos", "")
        match = re.fullmatch(r"(\d+):(\d+)-(\d+):(\d+)", position)
        if not match:
            continue
        start_line, start_col, end_line, end_col = map(int, match.groups())
        start = offsets[start_line - 1] + start_col - 1
        end = offsets[end_line - 1] + end_col - 1
        if doc.source[start:end] == children[0]["c"]:
            dollars.update(start + index for index, character in enumerate(children[0]["c"]) if character == "$")
    for offset in sorted(dollars):
        if offset + 1 in dollars and offset - 1 not in dollars:
            yield doc.source.count("\n", 0, offset) + 1


def source_table_rows(source: str) -> Iterator[tuple[int, str]]:
    """Only inspect source rows belonging to a GFM pipe table, outside code."""
    lines = source.splitlines()
    fence: str | None = None
    in_table = False
    for index, line in enumerate(lines):
        opening = re.match(r"^\s{0,3}(`{3,}|~{3,})", line)
        if opening:
            marker = opening.group(1)
            if fence is None:
                fence = marker
            elif marker[0] == fence[0] and len(marker) >= len(fence):
                fence = None
            in_table = False
            continue
        if fence:
            continue
        is_separator = bool(re.fullmatch(r"\s*\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)+\|?\s*", line))
        if is_separator and index:
            in_table = True
            yield index, lines[index - 1]
        elif in_table and line.strip() and "|" in line:
            yield index + 1, line
        else:
            in_table = False


def unsafe_table_math(source: str) -> Iterator[tuple[int, str]]:
    for line, row in source_table_rows(source):
        # GFM splits cells before parsing math. Audit only delimited fragments
        # within confirmed table rows, not arbitrary dollar signs in prose.
        for match in re.finditer(r"(?<![\\$])\$(?!\$)(.*?)(?<!\\)\$(?!\$)", row):
            if re.search(r"(?<!\\)\|", match.group(1)):
                yield line, match.group(0)


def looks_like_math_code(value: str) -> bool:
    value = value.strip()
    return bool(re.fullmatch(r"\$\$?[\s\S]+?\$\$?", value)
                or re.fullmatch(r"\\\([\s\S]+?\\\)", value)
                or re.fullmatch(r"\\\[[\s\S]+?\\\]", value))


def file_reference(value: str) -> bool:
    """Recognize whole backtick file references, without treating commands as links."""
    value = value.strip()
    return (not any(char in value for char in ("\n", "*", "<", ">", "{", "}"))
            and bool(re.match(r"(?:\.{1,2}/|/|docs/|notes/|notebooks/|images/|scripts/|src/|slurm/|tests/|outputs/|work/|artifacts/|data/)", value))
            and not value.startswith("//"))


def local_target(root: Path, source: Path, target: str) -> tuple[Path, str] | None:
    split = urlsplit(target)
    if split.scheme or split.netloc:
        return None
    decoded = unquote(split.path)
    if decoded.startswith("/"):
        # GitHub root-relative links differ from explicit local-machine paths.
        path = Path(decoded)
        if not path.exists() and not decoded.startswith(("/Users/", "/home/", "/tmp/", "/private/")):
            path = root / decoded.lstrip("/")
    else:
        path = source.parent / decoded if decoded else source
    return path.resolve(), unquote(split.fragment)


def historical_target(root: Path, path: Path) -> bool:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return True
    return bool(relative.parts and relative.parts[0] in EVIDENCE_ROOTS)


def collect_anchors(documents: list[Document]) -> dict[Path, set[str]]:
    result: dict[Path, set[str]] = {}
    for doc in documents:
        anchors = result.setdefault(doc.path.resolve(), set())
        for node in nodes(doc.ast):
            if node["t"] == "Header":
                # Pandoc's GFM reader implements GitHub-style header IDs and
                # suffixes repeated headings within each markdown document.
                anchors.add(node["c"][1][0])
            elif node["t"] in {"Div", "Span", "Table", "Figure", "Image", "Code", "CodeBlock"} and node["c"][0][0]:
                anchors.add(node["c"][0][0])
            elif node["t"] in {"RawInline", "RawBlock"} and node["c"][0] == "html":
                parser = HTMLReferences()
                parser.feed(node["c"][1])
                anchors.update(parser.anchors)
    return result


def render_math(expressions: list[tuple[str, str]], pandoc: str,
                api_version: list[int]) -> tuple[set[tuple[str, str]], str, str]:
    """Render once per distinct formula. TeX fallback spans indicate failure."""
    if not expressions:
        return set(), "", ""
    ast = {"pandoc-api-version": api_version, "meta": {}, "blocks": [
        {"t": "Div", "c": [[f"formula-{index}", [], []], [
            {"t": "Para", "c": [{"t": "Math", "c": [{"t": kind}, tex]}]}]]}
        for index, (kind, tex) in enumerate(expressions)]}
    result = subprocess.run([pandoc, "--from=json", "--to=html", "--mathml"],
                            input=json.dumps(ast), text=True, capture_output=True, timeout=120)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "MathML renderer failed")
    parser = MathMLResults()
    parser.feed(result.stdout)
    failures = {expression for index, expression in enumerate(expressions)
                if f"formula-{index}" not in parser.rendered}
    return failures, result.stdout, result.stderr.strip()


def validate(root: Path, selected: list[Path] | None = None, pandoc: str | None = None,
             preview: Path | None = None, strict_archive_links: bool = False) -> dict[str, Any]:
    root = root.resolve()
    executable = pandoc or shutil.which("pandoc")
    if not executable:
        raise RuntimeError("Documentation validation requires local pandoc; no renderer was downloaded.")
    paths = discover(root, selected or [])
    issues: list[Issue] = []
    docs: list[Document] = []
    svg_paths: list[Path] = []

    def add(doc: Document, severity: str, kind: str, detail: str,
            text: str | None = None, target: str | None = None, line: int | None = None) -> None:
        name = str(doc.path.relative_to(root)) if doc.path.is_relative_to(root) else str(doc.path)
        issues.append(Issue(severity, kind, name, detail, doc.cell,
                            line if line is not None else doc.line_for(text or ""), target))

    for path in paths:
        if path.suffix == ".svg":
            svg_paths.append(path)
            continue
        try:
            docs.extend(document_parts(path))
        except (OSError, ValueError) as error:
            add(Document(path, ""), "error", "document_parse", str(error))
    with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as pool:
        docs = list(pool.map(lambda doc: parse_document(doc, executable), docs))
    anchors = collect_anchors(docs)
    math: list[tuple[Document, str, str, bool]] = []
    links: list[tuple[Document, str]] = []
    counts: Counter[str] = Counter()
    inventory: list[dict[str, Any]] = []
    for doc in docs:
        if doc.parse_error:
            add(doc, "error", "document_parse", doc.parse_error)
            continue
        if doc.parse_warnings:
            add(doc, "warning", "document_parser_warning", doc.parse_warnings)
        for _, tex, legacy in supplementary_math(doc):
            if not legacy:
                add(doc, "error", "unparsed_math",
                    "Math-first parsing found a formula missing from GFM math; a Markdown list, heading, or other block may have interrupted its delimiters.", tex)
        for line in orphan_display_delimiters(doc):
            add(doc, "error", "orphan_math_delimiter",
                "Unescaped $$ remains literal prose in the GFM AST; the display formula is not rendering.", line=line)
        local_counts: Counter[str] = Counter()
        for kind, tex, legacy in math_nodes(doc):
            math.append((doc, kind, tex, legacy))
            local_counts[kind] += 1
            if legacy:
                local_counts["legacy_delimiters"] += 1
                add(doc, "error", "legacy_math_delimiters",
                    "GFM displays backslash-delimited math as text; use $...$ or $$...$$.", tex)
            if re.search(r"\\operatorname\b", tex):
                add(doc, "error", "forbidden_math_macro", "Authored math uses \\operatorname.", tex)
        for node in nodes(doc.ast):
            kind, value = node["t"], node.get("c")
            if kind in {"Link", "Image"}:
                links.append((doc, value[-1][0]))
                local_counts["images" if kind == "Image" else "links"] += 1
            elif kind in {"RawBlock", "RawInline"} and value[0] == "html":
                parser = HTMLReferences()
                parser.feed(value[1])
                links.extend((doc, target) for target in parser.links)
                local_counts["html_references"] += len(parser.links)
            elif kind in {"Code", "CodeBlock"}:
                code = value[-1]
                if looks_like_math_code(code):
                    syntax_example = (kind == "CodeBlock" and bool(set(value[0][1]) & {"latex", "tex"})) or code.strip() in {
                        "$...$", "$$...$$", r"\(...\)", r"\[...\]",
                    }
                    add(doc, "info" if syntax_example else "warning",
                        "math_syntax_example" if syntax_example else "math_in_code",
                        "Explicit TeX syntax example displays literally." if syntax_example
                        else "Math delimiters inside a code span/block display literally.", code)
                if kind == "Code" and file_reference(code):
                    candidate = code.strip()
                    # Root-prefixed references in prose conventionally refer to
                    # the repository; real Markdown links remain doc-relative.
                    source = root / "README.md" if re.match(r"^(docs|notes|notebooks|images|scripts|src|slurm|tests|outputs|work|artifacts|data)/", candidate) else doc.path
                    resolved = local_target(root, source, candidate)
                    local_counts["code_file_references"] += 1
                    if resolved and not resolved[0].exists():
                        add(doc, "info", "unavailable_code_reference",
                            "Backtick file reference does not exist; may describe generated data or an example.", code, candidate)
        for line, expression in unsafe_table_math(doc.source):
            add(doc, "error", "table_math_pipe",
                "Unescaped vertical bar splits this GFM table cell before math is parsed; use \\lvert/\\rvert or \\lVert/\\rVert.", expression, line=line)
        counts.update(local_counts)
        inventory.append({"file": str(doc.path.relative_to(root)), "cell": doc.cell,
                          **dict(local_counts)})

    for path in svg_paths:
        doc = Document(path, path.read_text(encoding="utf-8"))
        try:
            tree = ET.fromstring(doc.source)
        except ET.ParseError as error:
            add(doc, "error", "svg_parse", str(error))
            continue
        anchors[path.resolve()] = {node.attrib["id"] for node in tree.iter() if "id" in node.attrib}
        for node in tree.iter():
            if node.tag.rsplit("}", 1)[-1] in {"text", "tspan"}:
                label = "".join(node.itertext())
                if re.search(r"\\(?:[A-Za-z]+|[([])", label) or looks_like_math_code(label):
                    add(doc, "error", "svg_literal_math",
                        "Static SVG text contains raw LaTeX; use rendered glyphs or Unicode labels.", label)
            for attr in ("href", "{http://www.w3.org/1999/xlink}href"):
                if node.attrib.get(attr):
                    links.append((doc, node.attrib[attr]))

    def destination_error(target: str) -> str | None:
        """Navigation replacements must be real local files and real anchors."""
        resolved = local_target(root, root / "README.md", target)
        if resolved is None or not resolved[0].is_relative_to(root):
            return "Navigation destination must be a repository-local target."
        path, fragment = resolved
        if not path.exists():
            return "Navigation destination does not exist."
        if fragment:
            if path not in anchors:
                try:
                    if path.suffix == ".svg":
                        anchors[path] = {n.attrib["id"] for n in ET.parse(path).iter() if "id" in n.attrib}
                    elif path.suffix in {".md", ".ipynb", ".tex"}:
                        extra = [parse_document(part, executable) for part in document_parts(path)]
                        if any(part.parse_error for part in extra):
                            return "Navigation destination cannot be parsed."
                        anchors.update(collect_anchors(extra))
                except (OSError, ValueError, ET.ParseError) as error:
                    return str(error)
            if fragment not in anchors.get(path, set()):
                return "Navigation destination anchor is absent."
        return None

    navigation: dict[tuple[str, str], tuple[str, str]] = {}
    navigation_path = root / "docs/repository/notebook-navigation.json"
    if navigation_path.exists():
        registry_doc = Document(navigation_path, navigation_path.read_text(encoding="utf-8"))
        try:
            registry = json.loads(registry_doc.source)
            if registry["schema_version"] != 1:
                raise ValueError("Unsupported notebook-navigation schema_version")
            for entry in registry["notebooks"]:
                name = entry["path"]
                original = root / name
                if (Path(name).is_absolute() or ".." in Path(name).parts
                        or original.suffix != ".ipynb" or not original.is_file()):
                    raise ValueError(f"Invalid registered notebook path: {name}")
                if hashlib.sha256(original.read_bytes()).hexdigest() != entry["sha256"]:
                    add(registry_doc, "error", "notebook_navigation_hash",
                        "Registered historical notebook bytes changed; navigation exception is invalid.", target=name)
                    continue
                companion = entry.get("companion", registry.get("companion"))
                if not isinstance(companion, str):
                    raise ValueError(f"Missing notebook navigation companion for {name}")
                for mapping in entry["links"]:
                    old, current = mapping["historical_target"], mapping["current_target"]
                    key = name, old
                    if key in navigation:
                        raise ValueError(f"Duplicate notebook navigation entry: {key}")
                    errors = [(target, destination_error(target)) for target in (current, companion)]
                    for target, error in errors:
                        if error:
                            add(registry_doc, "error", "notebook_navigation_target", error, target=target)
                    if not any(error for _, error in errors):
                        navigation[key] = current, companion
        except (KeyError, TypeError, ValueError, OSError) as error:
            add(registry_doc, "error", "notebook_navigation_registry", str(error))

    # Link targets outside the selected subset still need their real headings.
    for doc, target in links:
        if target.startswith("attachment:"):
            if unquote(target.split(":", 1)[1]) not in doc.attachments:
                add(doc, "error", "missing_attachment", "Notebook attachment is absent.", target, target)
            continue
        resolved = local_target(root, doc.path, target)
        if resolved is None:
            counts["remote_references_unchecked"] += 1
            continue
        path, fragment = resolved
        if not path.exists():
            replacement = navigation.get((str(doc.path.relative_to(root)), target))
            if replacement:
                current, companion = replacement
                add(doc, "info", "preserved_notebook_navigation",
                    f"SHA-verified historical notebook retains unresolved original link {target}; "
                    f"validated current target: {current}; companion: {companion}.", target, target)
                counts["preserved_notebook_navigation"] += 1
                if not strict_archive_links:
                    continue
            historical = historical_target(root, path)
            add(doc, "error", "broken_local_link",
                "Historical/generated evidence or external-workspace target is unavailable locally."
                if historical else "Local Markdown/HTML/image target does not exist.", target, target)
            continue
        counts["resolved_local_references"] += 1
        if fragment and path.suffix.lower() in {".md", ".ipynb", ".tex", ".svg"}:
            if path not in anchors:
                try:
                    if path.suffix.lower() == ".svg":
                        tree = ET.parse(path)
                        anchors[path] = {n.attrib["id"] for n in tree.iter() if "id" in n.attrib}
                    else:
                        extra = [parse_document(part, executable) for part in document_parts(path)]
                        if any(part.parse_error for part in extra):
                            raise ValueError("Could not parse linked document anchors")
                        anchors.update(collect_anchors(extra))
                except (OSError, ValueError, ET.ParseError) as error:
                    add(doc, "error", "anchor_parse", str(error), target, target)
                    continue
            if fragment not in anchors.get(path, set()):
                add(doc, "error", "broken_local_anchor", "Target exists but its heading/explicit anchor is absent.", target, target)

    expressions = sorted({(kind, tex) for _, kind, tex, _ in math})
    api_version = next((doc.ast["pandoc-api-version"] for doc in docs if doc.ast), [1, 23, 1, 2])
    failures, rendered, renderer_diagnostics = render_math(expressions, executable, api_version)
    for doc, kind, tex, _ in math:
        if (kind, tex) in failures:
            add(doc, "error", "math_render", "Local TeXMath parser cannot convert this expression to MathML: " + tex, tex)
    if preview:
        preview.parent.mkdir(parents=True, exist_ok=True)
        preview.write_text("<!doctype html><html lang='en'><meta charset='utf-8'><title>Documentation math validation</title>"
                           "<style>body{max-width:1000px;margin:2rem auto;font:18px sans-serif}div{padding:1rem;border-bottom:1px solid #ccc}</style>"
                           "<h1>Parsed documentation expressions</h1><p>Local Pandoc TeXMath to native MathML. "
                           "Formula order matches the report's expressions list.</p>" + rendered + "</html>", encoding="utf-8")
    summary = Counter(issue.severity for issue in issues)
    expression_sources: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for doc, kind, tex, _ in math:
        expression_sources.setdefault((kind, tex), []).append({
            "file": str(doc.path.relative_to(root)), "cell": doc.cell,
            "line": doc.line_for(tex),
        })
    return {"schema_version": 1, "root": str(root), "renderer": "Pandoc TeXMath → MathML",
            "pandoc": subprocess.run([executable, "--version"], text=True, capture_output=True).stdout.splitlines()[0],
            "scope": {"files": len(paths), "markdown_tex_documents_and_notebook_cells": len(docs), "svg_files": len(svg_paths),
                      "excluded": sorted(SKIP_ROOTS | SKIP_DIRS) + [str(IMMUTABLE_REPLAY)],
                      "symlink_files": sum(path.is_symlink() for path in paths),
                      "symlink_content": "file aliases are parsed at their opened path, with relative links resolved from the alias directory; directory symlinks are not traversed",
                      "remote_links": "not fetched", "notebook_outputs": "not inspected; authored markdown only",
                      "strict_archive_links": strict_archive_links,
                      "latex_reader": "preamble macro expansion disabled; table* float width normalized for AST inspection; not a full publication build"},
            "summary": {"errors": summary["error"], "warnings": summary["warning"], "information": summary["info"],
                        "math_occurrences": len(math), "unique_expressions": len(expressions),
                        "rendered_expressions": len(expressions) - len(failures), **dict(counts)},
            "issues": [vars(issue) for issue in issues], "inventory": inventory,
            "expressions": [{"id": f"formula-{index}", "kind": kind, "tex": tex,
                             "rendered": (kind, tex) not in failures,
                             "sources": expression_sources[(kind, tex)]}
                            for index, (kind, tex) in enumerate(expressions)],
            "renderer_diagnostics": renderer_diagnostics}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, help="Files/directories, relative to --root (default: authored repository documents).")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--pandoc", help="Path to a locally installed Pandoc executable.")
    parser.add_argument("--report", type=Path, help="Write full JSON findings to this path.")
    parser.add_argument("--preview", type=Path, help="Write standalone MathML expression preview HTML.")
    parser.add_argument("--strict-archive-links", action="store_true",
                        help="Also fail on original unresolved links in SHA-verified historical notebooks with validated companion navigation.")
    args = parser.parse_args(argv)
    try:
        report = validate(args.root, args.paths, args.pandoc, args.preview, args.strict_archive_links)
    except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as error:
        print(json.dumps({"error": str(error)}))
        return 2
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"renderer": report["renderer"], "scope": report["scope"], "summary": report["summary"],
                      "report": str(args.report) if args.report else None}, indent=2))
    if not args.report:
        for issue in report["issues"]:
            if issue["severity"] != "info":
                location = f"{issue['file']}:{issue['line'] or 1}" + (f" (cell {issue['cell']})" if issue["cell"] else "")
                print(f"{issue['severity']}: {location}: {issue['kind']}: {issue['detail']}")
    return int(report["summary"]["errors"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
