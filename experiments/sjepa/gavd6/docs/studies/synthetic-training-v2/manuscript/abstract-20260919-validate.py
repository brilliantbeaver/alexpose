"""Validate scores for versions 01–07 and local links across all abstract drafts.

Uses only the Python standard library. Does not execute scientific experiments.
Existing output is refused; supply --output-dir for a newly named audit rerun.
"""
import argparse
import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

DIMENSIONS = [
    ('Conference relevance and contribution', 20),
    ('Claim accuracy and evidence support', 20),
    ('Evaluation and statistical rigor', 15),
    ('Scientific insight and related-work positioning', 15),
    ('Reproducibility', 10), ('Clarity and narrative', 10),
    ('Figures', 5), ('Submission fit', 5),
]


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=root)
    args = parser.parse_args()
    paths = [args.output_dir/f'abstract-20260919-{name}' for name in ('scores.json', 'scores.csv', 'validation.json')]
    if any(p.exists() for p in paths):
        raise FileExistsError('Preserve earlier audit outputs; use a fresh --output-dir.')
    versions = []
    for number in range(1, 8):
        matches = sorted(root.glob(f'abstract-v{number:02d}-*.md'))
        assert len(matches) == 1, (number, matches)
        p = matches[0]
        text = p.read_text()
        title = re.search(r'## (?:Proposed|Recommended) title\s+\*\*(.*?)\*\*', text).group(1)
        abstract = text.split('## Abstract\n\n', 1)[1].split('\n\n## ', 1)[0].strip()
        scores = []
        for name, weight in DIMENSIONS:
            found = re.search(r'^\| '+re.escape(name)+r' \| (\d+)% \| ([0-9.]+) \|', text, re.M)
            assert found, (p, name)
            assert int(found.group(1)) == weight
            score = Decimal(found.group(2))
            assert 0 <= score <= 10
            scores.append(score)
        total = sum(Decimal(weight)*score/10 for (_, weight), score in zip(DIMENSIONS, scores))
        recorded = Decimal(re.search(r'\*\*Weighted total: ([0-9.]+)', text).group(1))
        errata = []
        if recorded != total:
            assert number == 1 and recorded == 62 and total == Decimal('61.25')
            assert '**Arithmetic correction' in text and '**61.25 / 100**' in text
            errata.append('Initial total62.0 corrected to61.25 in appended note; components unchanged.')
        if '35.6–45.3%' in abstract:
            assert number <= 4 and '**Precision note from final validation:**' in text
            assert '**35.6–45.2%**' in text
            errata.append('Early rounded upper endpoint45.3 corrected to45.2 at one decimal in appended note.')
        assert scores[2] == Decimal('3.5') and scores[4] == Decimal('5.5') and scores[6] == Decimal('7.5')
        versions.append(dict(version=number, file=p.name, title=title, abstract=abstract,
                             word_count=len(abstract.split()), scores=[float(s) for s in scores],
                             weighted_total=float(total), preserved_draft_errata=errata,
                             sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    assert versions[5]['weighted_total'] == versions[6]['weighted_total'] == 70.5
    assert len({v['title'] for v in versions}) == 7

    metrics = json.loads((root/'abstract-20260919-exploratory-summary.json').read_text())
    by_name = {row['extractor']: row for row in metrics['results']}
    for name in ('hrnet_w32', 'vitpose_base'):
        assert 0 < by_name[name]['paired_reduction_vs_coordinate_percent'] < 1
    assert round(-by_name['rtmpose_m']['paired_reduction_vs_coordinate_percent'], 1) == 14.0
    raw = [row['paired_reduction_vs_unchanged_percent'] for row in metrics['results']]
    assert round(min(raw), 1) == 35.6 and round(max(raw), 1) == 45.2
    assert metrics['source_sha256'] == hashlib.sha256((root.parent/'proposal/README.md').read_bytes()).hexdigest()

    links_checked = 0
    files = sorted(root.glob('abstract-*.md'))
    for p in files:
        text = p.read_text()
        for url in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', text):
            if url.startswith(('https://', 'http://')):
                continue
            target, _, anchor = url.partition('#')
            resolved = (p.parent/target).resolve()
            # These two outputs are created only after the checks have passed.
            if resolved in [path.resolve() for path in paths]:
                continue
            assert resolved.exists(), (p.name, url)
            if anchor and resolved.is_file() and resolved.suffix == '.md':
                content = resolved.read_text()
                headings = re.findall(r'^#+ (.+)$', content, re.M)
                slugs = [re.sub(r'[^\w -]', '', h.lower()).replace(' ', '-') for h in headings]
                assert f'id="{anchor}"' in content or anchor in slugs, (p.name, url)
            links_checked += 1
    svg = root/'images/abstract-training-pipeline-20260919-v01.svg'
    ET.parse(svg)
    layout = json.loads(svg.with_suffix('.layout.json').read_text())
    assert not layout['findings'] and layout['checked_text_labels'] == 55 and layout['checked_edges'] == 14
    output_names = {p.name for p in paths}
    inputs = sorted(p for p in root.glob('abstract-*') if p.is_file() and p.name not in output_names)
    manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    report = dict(status='passed_with_preserved_early_draft_errata', revisions=7,
                  weighted_totals=[v['weighted_total'] for v in versions],
                  abstract_word_counts=[v['word_count'] for v in versions],
                  local_links_checked=links_checked,
                  numerical_scope='Rounded-summary ratios only; no prediction or confidence-interval reconstruction.',
                  figure_layout=layout, files_sha256=manifest,
                  notes=['Revision01 retains an appended total-arithmetic correction.',
                         'Revisions01–04 retain appended precision corrections. Scores cover versions01–07; later drafts are unscored. See abstract-versions.md for the current draft.',
                         'No source experiment, model, protocol, or pre-existing study record was changed by this drafting exercise.'])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with paths[0].open('x') as stream:
        json.dump(dict(rubric=[dict(dimension=n,weight_percent=w) for n,w in DIMENSIONS], versions=versions), stream, indent=2)
        stream.write('\n')
    with paths[1].open('x', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['version', 'file', 'title', 'word_count', *[n for n,_ in DIMENSIONS], 'weighted_total'])
        for version in versions:
            writer.writerow([version['version'],version['file'],version['title'],version['word_count'],*version['scores'],version['weighted_total']])
    with paths[2].open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps({key:report[key] for key in ('status','revisions','weighted_totals','abstract_word_counts','local_links_checked')}, indent=2))


if __name__ == '__main__':
    main()
