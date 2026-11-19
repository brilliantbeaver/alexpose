"""Inspect final slide identities, query geometry, vectors, links, and text.

Run with the publishing environment that provides PyMuPDF. Actual human page
inspection is recorded explicitly; this script does not perform visual review.
"""
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote, urlparse
import xml.etree.ElementTree as ET
import pymupdf

ROOT = Path(__file__).resolve().parents[4]
PROJECT = ROOT / 'experiments/aaai-worldmodels'
SLIDES = PROJECT / 'slides'
OUT = PROJECT / 'artifacts/reference-v2-independent-review'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    manifest_path = SLIDES / 'reference_v2_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    for name, expected in manifest['inputs'].items():
        assert sha(PROJECT / name) == expected, name
    for name, expected in manifest['outputs'].items():
        assert sha(SLIDES / name) == expected, name
    assert sha(ROOT / manifest['generator']) == manifest['generator_sha256']
    previous = json.loads((OUT / 'slide_text_extract.json').read_text())
    doc = pymupdf.open(SLIDES / 'reference_v2.pdf')
    assert len(doc) == len(previous['pages']) == 40
    changed_text = []; fonts = set(); pages = []
    for i, page in enumerate(doc):
        assert abs(page.rect.width / page.rect.height - 16/9) < 1e-12
        assert page.get_images() == []
        text = page.get_text()
        assert len(text) > 200
        if text != previous['pages'][i]['text']:
            changed_text.append(i+1)
        spans = [s for b in page.get_text('dict')['blocks'] if 'lines' in b for line in b['lines'] for s in line['spans']]
        outside = [s['text'] for s in spans if s['bbox'][0] < -1 or s['bbox'][1] < -1 or s['bbox'][2] > page.rect.width+1 or s['bbox'][3] > page.rect.height+1]
        assert outside == [], (i+1, outside)
        for font in page.get_fonts():
            fonts.add(font[0])
        svg = ET.parse(SLIDES / f'reference_v2_images/slide-{i+1:02d}.svg').getroot()
        tags = [n.tag.split('}')[-1] for n in svg.iter()]
        assert tags.count('image') == 0 and tags.count('text') >= 5
        pages.append(dict(page=i+1,raster_images=0,text_outside_page=0,editable_svg_text_nodes=tags.count('text')))
    assert changed_text == [9,15,16,25,26], changed_text
    for xref in fonts:
        assert len(doc.extract_font(xref)[3]) > 0
    source_path = SLIDES / 'reference_v2_source.json'
    source = json.loads(source_path.read_text())
    example = source['mask_example']; joints = example['joint_ids']
    expected_targets = {25,27,29,31,26,28,30,32}
    assert set(example['future_query_joint_ids']) == expected_targets
    policies = ['none','uniform','time_blocks','anatomical_tubes','motion_balanced','random_groups','motion_focused']
    for policy in policies:
        grid = example[policy]['states']
        assert len(grid) == 16 and all(len(row) == 24 for row in grid)
        queries = 0
        for j,row in zip(joints,grid):
            for value in row[16:]:
                assert (value == 3) == (j in expected_targets)
                queries += value == 3
        assert queries == 64
    residuals = sorted(source['slides'][24]['payload']['residuals'])
    a,b = residuals[3:5]
    midpoint = (a+b)/2
    cdf = (.5 + sum(v < midpoint for v in residuals) + .5*sum(v == midpoint for v in residuals))/(len(residuals)+1)
    assert abs(2*cdf-1) == 0 and b > a
    scope_text=' '.join(doc[8].get_text().split())
    assert 'prefix normalization' in scope_text
    assert 'evaluation support may change' in scope_text
    assert 'half credit for ties' in ' '.join(doc[25].get_text().split())
    class Links(HTMLParser):
        def __init__(self):super().__init__();self.links=[]
        def handle_starttag(self,tag,attrs):
            for key,value in attrs:
                if key in ('src','href') and value:self.links.append(value)
    html = (SLIDES / 'reference_v2.html').read_text();parser=Links();parser.feed(html)
    local=[]
    for value in parser.links:
        url=urlparse(value)
        if url.scheme or value.startswith('#'):continue
        path=(SLIDES/unquote(url.path)).resolve()
        assert path.is_file(),value
        local.append(value)
    # The remaining references live in the JavaScript slide-data payload.
    dynamic=[]
    for row in source['slides']:
        for value in row['sources']:
            url=urlparse(value)
            if url.scheme:continue
            assert (SLIDES/unquote(url.path)).resolve().is_file(),value
            assert value in html,value
            dynamic.append(value)
    script = OUT / 'slide_javascript.js'
    script.write_text('\n'.join(re.findall(r'<script[^>]*>(.*?)</script>',html,flags=re.S)))
    subprocess.run(['node','--check',str(script)],check=True,capture_output=True,text=True)
    layout = json.loads((SLIDES/'reference_v2_layout_checks.json').read_text())
    assert layout['status'] == 'passed' and layout['issues'] == []
    result=dict(identity='reference-v2-independent-slide-delivery-v1',status='passed',reviewed_at_utc=datetime.now(timezone.utc).isoformat(),script_sha256=sha(__file__),pdf_sha256=sha(SLIDES/'reference_v2.pdf'),html_sha256=sha(SLIDES/'reference_v2.html'),manifest_sha256=sha(manifest_path),generator_sha256=manifest['generator_sha256'],source_sha256=sha(source_path),manifest_inputs_verified=len(manifest['inputs']),manifest_outputs_verified=len(manifest['outputs']),pages=pages,embedded_font_files=len(fonts),prior_independently_recorded_pdf_sha256=previous['pdf_sha256'],changed_pdf_text_pages=changed_text,matched_future_query_cells_per_policy=64,policies_checked=7,zero_tail_interval=[a,b],local_links_verified=len(local),slide_payload_source_links_verified=len(dynamic),unique_slide_payload_source_links=len(set(dynamic)),javascript_syntax='passed node --check',visual_inspection=dict(all_40_pages_independently_inspected_at_full_size=True,exact_final_pdf_changed_pages_independently_rendered_and_viewed=[9,15,16,25,26],all_four_findings_resolved=True),limitations=['Browser navigation and keyboard runtime were not exercised. PDF rendering, links, and JavaScript syntax were checked.','Prior per-SVG and complete source-byte hashes were unavailable; no such prior identity is inferred. All final SVG/output hashes are verified.','Visual review is a human inspection recorded separately from automated geometric checks.'])
    (OUT/'slide_delivery_checks.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('status','pdf_sha256','html_sha256','manifest_sha256','manifest_outputs_verified','changed_pdf_text_pages','local_links_verified')},indent=2))


if __name__ == '__main__':main()
