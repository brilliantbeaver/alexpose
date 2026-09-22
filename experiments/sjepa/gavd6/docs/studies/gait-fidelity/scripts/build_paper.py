"""Build the offline interactive paper from README.md and refresh the vector gallery."""
from pathlib import Path
from html import escape, unescape
import json
import re
import mistune

ROOT = Path(__file__).resolve().parents[1]


def slug(text):
    text = unescape(re.sub('<[^>]+>', '', text)).lower()
    return re.sub(r'\s+', '-', re.sub(r'[^\w\s-]', '', text))


class Renderer(mistune.HTMLRenderer):
    def __init__(self):
        super().__init__(escape=False)
        self.headings = []

    def heading(self, text, level, **attrs):
        ident = slug(text)
        self.headings.append((level, ident, text))
        return f'<h{level} id="{ident}">{text}</h{level}>\n'


def explorer():
    return '''<section class="explorer" id="response-explorer" aria-labelledby="explorer-title">
<h3 id="explorer-title">Explore the error in a movement change</h3>
<p class="teaching">Illustration only. These controls define a hypothetical measurement rule; they are not model settings or experiment results. The camera stays fixed, so observation degradation does not change the reference.</p>
<div class="presets" aria-label="Teaching examples"><button disabled data-preset="faithful">Preserved change</button><button disabled data-preset="attenuated">Attenuated change</button><button disabled data-preset="sensitive">Observation-sensitive</button></div>
<div class="controls">
<label>True movement change <output id="true-change-value" for="true-change">+4.00°</output><input disabled id="true-change" type="range" min="-8" max="8" step="0.5" value="4"></label>
<label>Change retained: clean view <output id="clean-gain-value" for="clean-gain">100%</output><input disabled id="clean-gain" type="range" min="0" max="150" step="5" value="100"></label>
<label>Change retained: degraded view <output id="degraded-gain-value" for="degraded-gain">50%</output><input disabled id="degraded-gain" type="range" min="0" max="150" step="5" value="50"></label>
<label>Observation-only offset <output id="observation-shift-value" for="observation-shift">+1.00°</output><input disabled id="observation-shift" type="range" min="-3" max="3" step="0.25" value="1"></label>
</div>
<p class="grid-explanation">Each cell is a signed knee-excursion difference, right minus left, in image-plane degrees. Motion A starts at +2°; the first control determines the true change to motion B.</p>
<div class="table-wrap"><table><thead><tr><th>Movement</th><th>Clean observation</th><th>Degraded observation</th></tr></thead><tbody>
<tr><th>Motion A</th><td class="cell"><span class="reference">Reference <output id="reference-0">+2.00°</output></span><span class="estimate">Estimate <output id="estimate-0">+2.00°</output></span></td><td class="cell"><span class="reference">Reference <output id="reference-1">+2.00°</output></span><span class="estimate">Estimate <output id="estimate-1">+3.00°</output></span></td></tr>
<tr><th>Motion B</th><td class="cell"><span class="reference">Reference <output id="reference-2">+6.00°</output></span><span class="estimate">Estimate <output id="estimate-2">+6.00°</output></span></td><td class="cell"><span class="reference">Reference <output id="reference-3">+6.00°</output></span><span class="estimate">Estimate <output id="estimate-3">+5.00°</output></span></td></tr>
</tbody></table></div>
<div class="metric-grid"><div class="metric"><span>Response error · clean</span><output id="clean-response">0.00°</output></div><div class="metric"><span>Response error · degraded</span><output id="degraded-response">−2.00°</output></div><div class="metric"><span>Observation error change · motion A</span><output id="observation-a">+1.00°</output></div><div class="metric"><span>Observation error change · motion B</span><output id="observation-b">−1.00°</output></div><div class="metric wide"><span>Interaction: degraded response error minus clean response error</span><output id="interaction">−2.00°</output></div></div>
<p id="explorer-description" aria-live="polite">The reference changes by +4°, while the degraded estimate changes by +2°.</p>
<details><summary>Read the calculation and its limits</summary><p>The estimated change equals the true change multiplied by the selected retained fraction. Degraded estimates also receive the selected observation offset. Response error subtracts the reference change; observation error subtracts the reference difference between views. The interaction checks whether the response error depends on observation quality.</p><p>A shared constant offset could leave both response errors unchanged while making both measurements inaccurate. These contrasts therefore accompany level/position accuracy and waveform checks. The illustration has complete observations and fixed geometry; it does not model uncertainty, missing predictions or camera changes.</p></details>
<noscript><p>JavaScript is disabled. The displayed values illustrate the default case; the equations and static figures remain readable.</p></noscript></section>'''


def build():
    renderer = Renderer()
    md = mistune.create_markdown(renderer=renderer, plugins=['table'])
    content = md((ROOT/'README.md').read_text())
    content = content.replace('<!-- INTERACTIVE:response -->', explorer())
    content = re.sub(r'(<table>.*?</table>)', r'<div class="table-wrap">\1</div>', content, flags=re.S)
    # The explorer includes its own table wrapper; keep one container there.
    content = content.replace('<div class="table-wrap"><div class="table-wrap">','<div class="table-wrap">').replace('</table></div></div>','</table></div>')
    def figure(m):
        img = m.group(1)
        src = re.search(r'src="([^"]+)"', img).group(1)
        alt = re.search(r'alt="([^"]*)"', img).group(1)
        return f'<a class="figure-link" href="{src}" aria-label="Enlarge diagram: {alt}">{img}</a><p class="figure-hint">Select the diagram to enlarge it.</p>'
    content = re.sub(r'<p>(<img [^>]+/?>)</p>', figure, content)
    headings = [(ident, text) for level, ident, text in renderer.headings if level==2 and (re.match(r'^\d+\.',text) or text in ('Abstract','File guide'))]
    links = ''.join(f'<a href="#{ident}">{text}</a>' for ident,text in headings)
    css=(ROOT/'scripts/paper.css').read_text(); js=(ROOT/'scripts/paper.js').read_text()
    page='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Gait Fidelity — research proposal</title><meta name="description" content="A proposed study of movement-preserving pose restoration, with explicit data, methods and evaluation."><style>'''+css+'''</style></head><body><a class="skip" href="#paper">Skip to the proposal</a><details class="mobile-nav"><summary>Gait Fidelity · contents</summary><nav aria-label="Mobile contents">'''+links+'''</nav></details><div class="paper-shell"><aside class="sidebar"><a class="brand" href="#gait-fidelity">Gait Fidelity</a><p>A research proposal<br>New experiments pending</p><nav class="toc" aria-label="Paper contents">'''+links+'''</nav><div class="resources"><strong>Supporting material</strong><a href="data/README.md">Data specification</a><a href="data/video-gallery.html">Local video viewer</a><a href="methods/README.md">Method protocols</a><a href="images/gallery.html">All vector figures</a><a href="reviews/README.md">Independent reviews</a><a href="README.md">Markdown source</a></div></aside><main id="paper">'''+content+'''</main></div><dialog id="figure-dialog" aria-labelledby="figure-description"><button id="close-figure" type="button">Close figure</button><img id="large-figure" alt=""><p id="figure-description"></p></dialog><script>'''+js+'''</script></body></html>\n'''
    (ROOT/'proposal.html').write_text(page)
    # Show old and new diagrams together without regenerating accepted SVGs.
    figures=[]
    for path in sorted((ROOT/'images').glob('[0-9][0-9]-*.svg')):
        svg=path.read_text()
        title=unescape(re.search(r'<title[^>]*>(.*?)</title>',svg,re.S).group(1))
        desc=re.search(r'<desc[^>]*>(.*?)</desc>',svg,re.S)
        description=unescape(desc.group(1)) if desc else title
        figures.append((path.name,title,description))
    cards=''.join(f'<section><h2>{escape(title)}</h2><a href="{name}"><img src="{name}" alt="{escape(desc,quote=True)}"></a><p>{escape(desc)}</p><p><a href="{name}">Editable SVG</a> · <a href="previews/{Path(name).stem}-900.png">900-pixel preview</a></p></section>' for name,title,desc in figures)
    gallery='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Gait Fidelity — figure gallery</title><style>body{margin:30px auto;max-width:1200px;padding:0 20px;background:#f2f6f9;color:#183247;font:17px/1.6 Arial,sans-serif}section{background:white;padding:20px;margin:28px 0;border:1px solid #d5e0e7;border-radius:12px}img{width:100%;height:auto}h2{font-size:22px}a{color:#326bb2}</style></head><body><h1>Gait Fidelity: visual guide</h1><p>Conceptual diagrams and explicitly illustrative calculations. <a href="../proposal.html">Interactive proposal</a> · <a href="../README.md">Markdown proposal</a> · <a href="../reviews/README.md">Independent review</a></p>'''+cards+'</body></html>\n'
    (ROOT/'images/gallery.html').write_text(gallery)
    (ROOT/'records/paper-build.json').write_text(json.dumps({'sections':[i for i,t in headings],'figures':len(figures),'interactive_components':['response contrast explorer','section navigation','figure enlargement'],'scientific_source':'README.md','no_external_runtime_dependencies':True},indent=2)+'\n')
    print(f'Built proposal.html, {len(headings)} navigation entries and a {len(figures)}-figure gallery.')


if __name__=='__main__':
    build()
