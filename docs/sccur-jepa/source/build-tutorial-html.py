"""Compile the self-contained student site from the four shared lesson sources."""
import argparse
import base64
import csv
import json
import html
import os
from pathlib import Path
import re
import subprocess

SOURCE = Path(__file__).resolve().parent
OUTPUT = SOURCE.parent
LESSONS = OUTPUT / "tutorial"
CHAPTERS = [
    ("start", "Start", "Guide to the experiments", "start.md", "data-units.svg",
     "Each recording can supply several labeled sequences. Each sequence can supply several windows."),
    ("classifier", "00–07", "Learn from hidden joints", "classifier-study.md", "masked-learner.svg",
     "The student predicts a description of the hidden observation. The teacher sees the measurement and supplies the target."),
    ("restart", "10–16", "Forecast the next step", "restart-study.md", "forecast-window.svg",
     "The forecast begins after the observed part of the walk. Both methods are scored against the same eight future samples."),
    ("forecast", "20–26", "Test an anomaly score", "forecast-study.md", "ranking-pairs.svg",
     "Three of these four pairs are in the intended order. The resulting AUROC is 0.75 in this teaching example."),
]

def lesson_markup(filename):
    markdown = (LESSONS / filename).read_text()
    def fix_link(match):
        label, link = match.groups()
        if '://' not in link and not link.startswith('#'):
            resolved = (LESSONS / link).resolve()
            if not resolved.exists():
                raise FileNotFoundError(resolved)
            link = os.path.relpath(resolved, OUTPUT)
        return f'[{label}]({link})'
    markdown = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', fix_link, markdown)
    return subprocess.run(['pandoc','-f','gfm','-t','html5','--wrap=none'],
                          input=markdown, capture_output=True, text=True, check=True).stdout

def build(preview=False):
    pages, nav = [], []
    for slug, code, name, filename, illustration, caption in CHAPTERS:
        source = lesson_markup(filename)
        start, remainder = source.split('</h1>',1)
        # Place the figure after the question that motivates it. Keep deeper
        # notebook material below the hands-on demonstration.
        lead, rest = remainder.split('</p>',1)
        svg = (LESSONS / 'figures' / illustration).read_text()
        svg = svg[svg.index('<svg'):]
        # SVG accessibility IDs remain unique across the four inline images.
        svg = svg.replace('id="title"', f'id="{slug}-title"')
        svg = svg.replace('id="desc"', f'id="{slug}-desc"')
        svg = svg.replace('aria-labelledby="title desc"', f'aria-labelledby="{slug}-title {slug}-desc"')
        figure = f'<figure>{svg}<figcaption>{html.escape(caption)}</figcaption></figure>'
        label = 'Student guide' if slug == 'start' else f'Notebooks {code}'
        if slug == 'forecast':
            study = OUTPUT.parents[1] / 'experiments/aaai-worldmodels/docs/draft03'
            with (study / 'figures/ranking_comparison_data.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            for row in rows:
                for key in ('estimate', 'lower95', 'upper95'):
                    row[key] = float(row[key])
            panel = '<div id="measured-rankings"></div><script type="application/json" id="ranking-evidence">' + json.dumps(rows) + '</script>'
            rest = re.sub(r'(?=<h2[^>]*>25 ·)', lambda _: panel, rest, count=1)
        pages.append(f'<article class="lesson" id="{slug}"><p class="chaptercode">{label}</p>'
                     f'{start}</h1>{lead}</p>{figure}<div class="jump"></div>'
                     f'<div id="lab-{slug}"></div>{rest}<div class="quiz"></div>'
                     '<div class="footnav"></div></article>')
        nav.append(f'<a href="#{slug}"><span>{code}</span>{name}</a>')
    script = '\n'.join((SOURCE / name).read_text() for name in
                       ['tutorial-labs.js','tutorial-evidence.js','tutorial-navigation.js'])
    page = (SOURCE / 'tutorial-template.html').read_text()
    page = page.replace('%%NAV%%',''.join(nav)).replace('%%PAGES%%',''.join(pages)).replace('%%SCRIPT%%',script)
    page = page.replace('<table', '<div class="tablewrap"><table').replace('</table>', '</table></div>')
    doc = OUTPUT / 'Gait-Video-Tutorial.docx'
    if not preview:
        # Keep the Word download usable even when the HTML alone is copied elsewhere.
        encoded = base64.b64encode(doc.read_bytes()).decode('ascii')
        mime = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        page = page.replace('href="Gait-Video-Tutorial.docx" download',
                            f'href="data:{mime};base64,{encoded}" download="{doc.name}"')
    (OUTPUT / 'Gait-Video-Tutorial.html').write_text(page)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--preview',action='store_true',help='Use a relative link for the Word download.')
    build(parser.parse_args().preview)
