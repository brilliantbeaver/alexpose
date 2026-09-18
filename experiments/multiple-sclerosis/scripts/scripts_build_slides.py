"""Export the tutorial as portable HTML and PDF from its Marp source.

From the experiment directory:
    python scripts/scripts_build_slides.py

Requires Node.js/npm and a Chromium browser. The pinned Marp CLI is obtained by
npx if it is not already cached. SVG figures are embedded into the HTML so the
export can be shared as one file. PowerPoint has a separate exporter.
"""
from __future__ import annotations

import base64
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SLIDES = ROOT / "slides"
SOURCE = SLIDES / "slides.md"
MARP = ["npx", "--yes", "@marp-team/marp-cli@4.5.0"]


def main() -> None:
    html_path = SLIDES / "slides.html"
    subprocess.run(MARP + [str(SOURCE), "--html", "-o", str(html_path)], cwd=ROOT, check=True)
    html = html_path.read_text()

    def embed_svg(match: re.Match) -> str:
        reference = match.group(1)
        if not reference.lower().endswith(".svg") or ":" in reference:
            return match.group(0)
        svg_path = (SLIDES / reference).resolve()
        encoded = base64.b64encode(svg_path.read_bytes()).decode("ascii")
        return f'src="data:image/svg+xml;base64,{encoded}"'

    html = re.sub(r'src="([^"]+)"', embed_svg, html)
    html_path.write_text(html)
    subprocess.run(
        MARP + [str(SOURCE), "--html", "--pdf", "--pdf-outlines",
                "--allow-local-files", "-o", str(SLIDES / "slides.pdf")],
        cwd=ROOT, check=True,
    )
    print("Built slides/slides.html (embedded SVGs) and slides/slides.pdf")


if __name__ == "__main__":
    main()
