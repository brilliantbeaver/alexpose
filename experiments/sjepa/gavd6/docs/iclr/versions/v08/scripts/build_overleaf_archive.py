"""Create the minimal Overleaf upload archive without changing paper content."""

from pathlib import Path
import hashlib
import json
import re
import zipfile
from pypdf import PdfReader


HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "paper-v08-overleaf.zip"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def build():
    sources = {
        "main.tex": HERE / "paper-v08.tex",
        "appendix.tex": HERE / "appendix.tex",
        "supplement/technical-details.tex": HERE / "supplement/technical-details.tex",
        "supplement/README.md": HERE / "supplement/README.md",
        "references.bib": HERE / "references.bib",
        "README.md": HERE / "OVERLEAF.md",
        "template/iclr2027_conference.sty": HERE / "template/iclr2027_conference.sty",
        "template/iclr2027_conference.bst": HERE / "template/iclr2027_conference.bst",
    }
    manuscript = sources["main.tex"].read_text() + sources["appendix.tex"].read_text() + sources["supplement/technical-details.tex"].read_text()
    # Resolve only the figure calls and table inputs actually used by the paper.
    for name in sorted(set(re.findall(r"\\figurefile\{([^}]+)\}", manuscript))):
        relative = f"figures/{name}.pdf"
        sources[relative] = HERE / relative
    for relative in sorted(set(re.findall(r"\\input\{(tables/[^}]+)\}", manuscript))):
        sources[relative] = HERE / relative
    for relative in sorted(set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{(figures/[^}]+)\}", manuscript))):
        if "#" not in relative:
            sources[relative] = HERE / relative
    assert len([s for s in sources if s.startswith("figures/")]) == 8
    assert len([s for s in sources if s.startswith("tables/")]) == 7

    payload = {}
    for name, source in sources.items():
        assert source.is_file(), source
        data = source.read_bytes()
        if source.suffix != ".pdf":
            content = data.decode("utf-8")
            for token in ("/Users/", "/home/", "theodoremui", "tedmui", "haic.stanford"):
                assert token not in content, (name, token)
        payload[name] = data
    assert sum(b"\\documentclass" in data for name, data in payload.items()
               if name.endswith(".tex")) == 1

    manifest = {
        "manuscript_version": "v08",
        "main_document": "main.tex",
        "recommended_overleaf_compiler": "XeLaTeX",
        "compiler_selection": "Select in Overleaf project settings after upload.",
        "reference_pdf": {
            "file": "paper-v08.pdf (delivered separately)",
            "sha256": sha((HERE / "paper-v08.pdf").read_bytes()),
            "main_pages": 9,
            "total_pages": len(PdfReader(HERE / "paper-v08.pdf").pages),
        },
        "scope": "Typesetting assets for the concise manuscript and optional full technical reference.",
        "files": {
            name: {"source": str(sources[name].relative_to(HERE)),
                   "bytes": len(data), "sha256": sha(data)}
            for name, data in sorted(payload.items())
        },
    }
    payload["bundle-manifest.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    # Stable names, ordering, timestamps, and permissions make regeneration exact.
    with zipfile.ZipFile(OUT, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in sorted(payload.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data, compresslevel=9)
    print(json.dumps({"archive": OUT.name, "entries": len(payload),
                      "bytes": OUT.stat().st_size,
                      "sha256": sha(OUT.read_bytes())}, indent=2))


if __name__ == "__main__":
    build()
