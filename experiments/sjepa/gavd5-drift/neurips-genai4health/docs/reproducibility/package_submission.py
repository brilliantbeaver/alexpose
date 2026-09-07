"""Build separate, allowlisted local review bundles, with release holds explicit.

Packaging does not submit anything or approve dissemination. Per-source numerical
records remain in their own permission-sensitive bundle, outside manuscript
source bundles. No original video, pose trajectory, checkpoint, or notebook is
included. ZIP timestamps and entry order are fixed for reproducibility.
"""

from pathlib import Path
import hashlib
import json
import subprocess
import sys
import zipfile


DOCS = Path(__file__).resolve().parents[1]
FORBIDDEN = ("c:\\users", "c:/users", "alexm", "brilliantbeaver", "alexpose_root", "/users/pmui", "pmui@")
STATUS = "Local review artifact; not submitted and not approved for public release."
NUMERICAL = [f"numerical_supplement/{name}" for name in (
    "README.md", "verify.py", "test_source_predictions.csv", "normal_validation_weighting.csv", "provenance.json")]
BUNDLES = {
    "genai4health_position_source.zip": [
        "genai4health_paper_draft.tex", "genai4health_paper_draft.pdf", "references.bib",
        "neurips_2026.sty", "figures/weighting_comparison.pdf",
    ],
    "genai4health_companion_source.zip": [
        "genai4health_extended_abstract.tex", "genai4health_extended_abstract.pdf", "references.bib",
        "neurips_2026.sty",
    ],
    "genai4health_numerical_supplement.zip": NUMERICAL,
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_entry(archive: zipfile.ZipFile, name: str, data: bytes) -> None:
    entry = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    entry.compress_type = zipfile.ZIP_DEFLATED
    entry.create_system = 3
    entry.external_attr = 0o100644 << 16
    archive.writestr(entry, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def main() -> None:
    qa = json.loads((DOCS / "review/pdf_qa/qa_summary.json").read_text(encoding="utf-8"))
    expected = {"genai4health_paper_draft.pdf": 5, "genai4health_extended_abstract.pdf": 2}
    assert {record["pdf"] for record in qa} == set(expected)
    for record in qa:
        assert not record["errors"] and all(record["checks"].values()), record["pdf"]
        assert record["main_text_pages"] <= expected[record["pdf"]]
        pdf = DOCS / record["pdf"]
        assert digest(pdf) == record["pdf_sha256"], "PDF changed after QA."
        assert digest(pdf.with_suffix(".tex")) == record["canonical_tex_sha256"], "Source changed after QA."
    subprocess.run([sys.executable, "-B", str(DOCS / "numerical_supplement/verify.py")], check=True)

    report = []
    for name, entries in BUNDLES.items():
        sensitive = name == "genai4health_numerical_supplement.zip"
        companion = name == "genai4health_companion_source.zip"
        notice = STATUS + "\n\n"
        if sensitive:
            notice += (
                "This separate numerical bundle contains aliased per-source annotation predictions "
                "and source-level feature similarities. Aliases do not establish person anonymity. "
                "The responsible authors must resolve project-specific ethics and data-use requirements "
                "before providing these records to reviewers or releasing them publicly. No approval "
                "or exemption is claimed.\n\nRun python numerical_supplement/verify.py to check recorded "
                "arithmetic only; this does not rerun encoder training or establish clinical validity.\n"
            )
        else:
            manuscript = "genai4health_extended_abstract" if companion else "genai4health_paper_draft"
            notice += (
                "This bundle contains manuscript source and aggregate reported results, without the "
                "separate per-source numerical data bundle. Final author review, required data-use "
                "and ethics determinations, and submission decisions remain outstanding.\n\n"
                f"Build with: tectonic --keep-logs {manuscript}.tex\n"
            )
            if companion:
                notice += (
                    "\nThis is a companion summary of the position paper, not an independent second study. "
                    "The workshop does not list a separate extended-abstract track; obtain workshop guidance "
                    "before attempting an additional submission of this material.\n"
                )
        files = []
        for entry in entries:
            path = DOCS / entry
            assert path.is_file(), f"Missing allowlisted input: {entry}"
            if path.suffix in (".tex", ".bib", ".csv", ".json", ".py", ".md"):
                content = path.read_text(encoding="utf-8").lower()
                assert not any(term in content for term in FORBIDDEN), f"Potential identifying text in {entry}"
            files.append({"path": entry, "sha256": digest(path)})
        with zipfile.ZipFile(DOCS / name, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            write_entry(archive, "LOCAL_REVIEW_STATUS.txt", notice.encode("utf-8"))
            for entry in sorted(entries):
                write_entry(archive, entry, (DOCS / entry).read_bytes())
        with zipfile.ZipFile(DOCS / name) as archive:
            assert archive.testzip() is None
            assert set(archive.namelist()) == set(entries) | {"LOCAL_REVIEW_STATUS.txt"}
            chart_files = [entry for entry in archive.namelist() if entry.startswith("figures/")]
            assert chart_files == (["figures/weighting_comparison.pdf"] if "position_source" in name else [])
        report.append({
            "bundle": name, "sha256": digest(DOCS / name), "files": files,
            "generated_notice": "LOCAL_REVIEW_STATUS.txt", "status": STATUS,
            "permission_sensitive_source_records": sensitive,
            "source_record_release_approved": False,
            "companion_not_an_independent_submission": companion,
        })
    (DOCS / "review/package_manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps([{key: record[key] for key in ("bundle", "status", "permission_sensitive_source_records")}
                      for record in report], indent=2))


if __name__ == "__main__":
    main()
