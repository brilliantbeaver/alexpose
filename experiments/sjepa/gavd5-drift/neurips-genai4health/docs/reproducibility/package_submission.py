"""Package only explicitly allowed anonymous manuscript/numerical files."""
from pathlib import Path
import hashlib
import json
import zipfile

DOCS = Path(__file__).resolve().parents[1]
qa = json.loads((DOCS / "review/pdf_qa/qa_summary.json").read_text(encoding="utf-8"))
assert len(qa) == 2
for record, limit in zip(qa, (5, 2)):
    assert record["main_text_pages"] <= limit
    assert not record["metadata"]["author"]
    assert not record["anonymity_string_hits"] and not record["outside_page_text"]

numerical = [f"numerical_supplement/{name}" for name in (
    "README.md", "verify.py", "test_source_predictions.csv", "normal_validation_weighting.csv", "provenance.json")]
paper_files = ["genai4health_paper_draft.tex", "genai4health_paper_draft.pdf", "references.bib", "neurips_2026.sty",
               "figures/audited_findings.pdf", "figures/source_weighting_and_drift.pdf"]
bundles = {"genai4health_numerical_supplement.zip":numerical,
           "genai4health_position_source.zip":paper_files + numerical}
report = []
for name, entries in bundles.items():
    for entry in entries:
        path = DOCS / entry
        assert path.is_file()
        if path.suffix in (".tex", ".bib", ".csv", ".json", ".py", ".md"):
            text = path.read_text(encoding="utf-8").lower()
            for forbidden in ("c:\\users", "c:/users", "alexm", "brilliantbeaver", "alexpose_root"):
                assert forbidden not in text, (entry, forbidden)
    with zipfile.ZipFile(DOCS / name, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for entry in entries: z.write(DOCS / entry, arcname=entry)
    with zipfile.ZipFile(DOCS / name) as z: assert z.testzip() is None
    report.append({"bundle":name, "sha256":hashlib.sha256((DOCS/name).read_bytes()).hexdigest(),
                   "files":[{"path":p,"sha256":hashlib.sha256((DOCS/p).read_bytes()).hexdigest()} for p in entries]})
(DOCS / "review/package_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps([{"bundle":r["bundle"], "files":len(r["files"]), "sha256":r["sha256"]} for r in report], indent=2))
