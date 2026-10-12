"""Build the offline interactive paper from README.md and refresh the vector gallery."""
from pathlib import Path
from html import escape, unescape
import hashlib
import io
import json
import os
import re
import tempfile
import mistune

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'gait-proposal-matplotlib'))
import matplotlib
matplotlib.use('Agg')
from matplotlib.font_manager import FontProperties
from matplotlib.mathtext import math_to_image


def math_image(tex, display=False):
    """Render source LaTeX to offline SVG, with stable paths and baseline metrics."""
    if r'\operatorname' in tex:
        raise ValueError('Use conventional symbols or \\mathrm in proposal equations.')
    digest = hashlib.sha256(tex.encode()).hexdigest()[:16]
    relative = Path('images/equations') / f'{digest}.svg'
    target = ROOT / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    # MathText defaults to compact fractions; display equations need full-size text.
    rendered_tex = tex.replace(r'\frac', r'\dfrac') if display else tex
    with matplotlib.rc_context({'svg.hashsalt': digest, 'mathtext.fontset': 'stix',
                                'savefig.transparent': True, 'figure.facecolor': 'none'}):
        depth = math_to_image('$' + rendered_tex + '$', buffer, prop=FontProperties(size=18),
                              format='svg', color='#183247')
    svg = buffer.getvalue().decode()
    # Avoid timestamps making an unchanged equation appear modified on every build.
    svg = re.sub(r'<dc:date>.*?</dc:date>', '', svg)
    width, height = (float(re.search(rf'{dimension}="([\d.]+)pt"', svg).group(1))
                     for dimension in ('width', 'height'))
    target.write_text(svg)
    label = escape(tex, quote=True)
    sizing = f'width:{width / 18:.4f}em;height:{height / 18:.4f}em;vertical-align:-{depth / 18:.4f}em'
    img = f'<img class="math-image" src="{relative.as_posix()}" alt="{label}" style="{sizing}">'
    return (f'<div class="equation" tabindex="0">{img}</div>\n' if display
            else f'<span class="inline-math">{img}</span>')


def slug(text):
    text = unescape(re.sub('<[^>]+>', '', text)).lower()
    return re.sub(r'\s+', '-', re.sub(r'[^\w\s-]', '', text))


class Renderer(mistune.HTMLRenderer):
    def __init__(self):
        super().__init__(escape=False)
        self.headings = []
        self.equations = []

    def block_math(self, text):
        self.equations.append({'latex': text, 'display': True})
        return math_image(text, display=True)

    def inline_math(self, text):
        self.equations.append({'latex': text, 'display': False})
        return math_image(text)

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
<label>Change retained: clear observation <output id="clean-gain-value" for="clean-gain">100%</output><input disabled id="clean-gain" type="range" min="0" max="150" step="5" value="100"></label>
<label>Change retained: degraded observation <output id="degraded-gain-value" for="degraded-gain">50%</output><input disabled id="degraded-gain" type="range" min="0" max="150" step="5" value="50"></label>
<label>Observation-only offset <output id="observation-shift-value" for="observation-shift">+1.00°</output><input disabled id="observation-shift" type="range" min="-3" max="3" step="0.25" value="1"></label>
</div>
<p class="grid-explanation">Each cell is a signed knee-excursion difference, right minus left, in image-plane degrees. Motion A starts at +2°; the first control determines the true change to motion B.</p>
<div class="table-wrap"><table><thead><tr><th>Movement</th><th>Clean observation</th><th>Degraded observation</th></tr></thead><tbody>
<tr><th>Motion A</th><td class="cell"><span class="reference">Reference <output id="reference-0">+2.00°</output></span><span class="estimate">Estimate <output id="estimate-0">+2.00°</output></span></td><td class="cell"><span class="reference">Reference <output id="reference-1">+2.00°</output></span><span class="estimate">Estimate <output id="estimate-1">+3.00°</output></span></td></tr>
<tr><th>Motion B</th><td class="cell"><span class="reference">Reference <output id="reference-2">+6.00°</output></span><span class="estimate">Estimate <output id="estimate-2">+6.00°</output></span></td><td class="cell"><span class="reference">Reference <output id="reference-3">+6.00°</output></span><span class="estimate">Estimate <output id="estimate-3">+5.00°</output></span></td></tr>
</tbody></table></div>
<div class="metric-grid"><div class="metric"><span>Signed response bias · clean</span><output id="clean-response">0.00°</output></div><div class="metric"><span>Signed response bias · degraded</span><output id="degraded-response">−2.00°</output></div><div class="metric"><span>Observation error change · motion A</span><output id="observation-a">+1.00°</output></div><div class="metric"><span>Observation error change · motion B</span><output id="observation-b">−1.00°</output></div><div class="metric"><span>Absolute response error · clean</span><output id="clean-absolute">0.00°</output></div><div class="metric"><span>Absolute response error · degraded</span><output id="degraded-absolute">2.00°</output></div><div class="metric wide"><span>Interaction: degraded signed bias minus clean signed bias</span><output id="interaction">−2.00°</output></div></div>
<p id="explorer-description" aria-live="polite">The reference changes by +4°, while the degraded estimate changes by +2°.</p>
<details><summary>Read the calculation and its limits</summary><p>The estimated change equals the true change multiplied by the selected retained fraction. Degraded estimates also receive the selected observation offset. Signed response bias subtracts the reference change; the primary response error takes its absolute value. Observation error subtracts the reference difference between observation conditions. The interaction compares their signed response biases.</p><p>A shared constant offset could leave both response biases unchanged while making both measurements inaccurate. These contrasts therefore accompany level/position accuracy and waveform checks. The illustration has complete observations and fixed geometry; it does not model uncertainty, missing predictions or camera changes.</p></details>
<noscript><p>JavaScript is disabled. The displayed values illustrate the default case; the equations and static figures remain readable.</p></noscript></section>'''


def coupling_explorer():
    return """<section class="explorer" id="coupling-explorer" aria-labelledby="coupling-title">
<h3 id="coupling-title">Explore what error coupling changes</h3>
<p class="teaching">Illustration only: two centered feature entries at one joint–time position (D = 2). Each residual is a prediction minus its reference target. The values have no anatomical units.</p>
<div class="presets"><button disabled data-coupling-preset="shared">Shared error</button><button disabled data-coupling-preset="opposed">Opposite errors</button><button disabled data-coupling-preset="zero">Zero error</button></div>
<div class="controls"><label>State a: residual (u, −u)<output id="residual-a-value">2.00</output><input disabled id="residual-a" type="range" min="-4" max="4" step="0.25" value="2"></label><label>State b: residual (v, −v)<output id="residual-b-value">2.00</output><input disabled id="residual-b" type="range" min="-4" max="4" step="0.25" value="2"></label></div>
<p class="teaching">Each endpoint has two feature errors: the dark bar shows the first channel, and the pale bar shows the second. Bars extend left for negative errors and right for positive errors.</p>
<svg viewBox="0 0 620 155" role="img" aria-label="Two centered feature residuals at each endpoint; bars extend left for negative values and right for positive values."><line x1="300" y1="15" x2="300" y2="135" stroke="#9fb3c1"/><text x="10" y="40" fill="#183247">Endpoint a</text><text x="10" y="105" fill="#183247">Endpoint b</text><rect id="residual-a-x" x="300" y="20" width="80" height="16" fill="#265f9e"/><rect id="residual-a-y" x="220" y="40" width="80" height="16" fill="#93b5d6"/><rect id="residual-b-x" x="300" y="85" width="80" height="16" fill="#087e78"/><rect id="residual-b-y" x="220" y="105" width="80" height="16" fill="#8bd0c4"/><text x="270" y="151" fill="#526675">zero</text></svg>
<div class="metric-grid"><div class="metric"><span>Coupled loss: (v − u)² / 2</span><output id="delta-loss">0.00</output></div><div class="metric"><span>Endpoint loss: (u² + v²) / 2</span><output id="endpoint-loss">4.00</output></div><div class="metric wide"><span>Difference: coupled minus endpoint = −uv</span><output id="coupling-term">−4.00</output></div></div>
<p id="coupling-description" aria-live="polite">Both states have the same nonzero error, so the difference loss is zero while the endpoint loss is positive.</p>
<details><summary>Why the original loss and coordinate checks remain</summary><p>The coupled term penalizes a difference of errors. Shared endpoint bias cancels, even when each prediction is wrong. The real experiment retains the original anchoring losses, uses all feature channels and matched reference support, and evaluates restored movement and coordinates. These illustrative numbers are neither calibration values nor model results.</p></details>
<noscript><p>JavaScript is disabled. The default values show shared-error cancellation.</p></noscript></section>"""


def build():
    renderer = Renderer()
    md = mistune.create_markdown(renderer=renderer, plugins=['table', 'math'])
    content = md((ROOT/'README.md').read_text())
    # Keep following punctuation with a mathematical symbol at a line break.
    content = re.sub(r'(<span class="inline-math">.*?)(</span>)([.,;:])', r'\1\3\2', content)
    content = content.replace('<!-- INTERACTIVE:response -->', explorer())
    content = content.replace('<!-- INTERACTIVE:coupling -->', coupling_explorer())
    content = re.sub(r'(<table>.*?</table>)', r'<div class="table-wrap">\1</div>', content, flags=re.S)
    # The explorer includes its own table wrapper; keep one container there.
    content = content.replace('<div class="table-wrap"><div class="table-wrap">','<div class="table-wrap">').replace('</table></div></div>','</table></div>')
    def figure(m):
        img = m.group(1)
        src = re.search(r'src="([^"]+)"', img).group(1)
        alt = re.search(r'alt="([^"]*)"', img).group(1)
        # The context photographs have finite source resolution; retain their
        # original aspect ratios and avoid stretching them to diagram width.
        photos = {
            'images/context/opencap-capture.jpg': (362, 349, 362),
            'images/context/ambient-walking.jpg': (687, 389, 600),
        }
        classes, hint_classes, style = 'figure-link', 'figure-hint', ''
        if src in photos:
            width, height, maximum = photos[src]
            img = img.replace('<img ', f'<img width="{width}" height="{height}" ', 1)
            classes += ' photograph'
            hint_classes += ' photograph-hint'
            style = f' style="--photo-max-width:{maximum}px"'
        return (f'<a class="{classes}"{style} href="{src}" '
                f'aria-label="Enlarge figure: {alt}">{img}</a>'
                f'<p class="{hint_classes}"{style}>Select the figure to enlarge it.</p>')
    content = re.sub(r'<p>(<img [^>]+/?>)</p>', figure, content)
    headings = [(ident, text) for level, ident, text in renderer.headings if level==2 and (re.match(r'^\d+\.',text) or text in ('Abstract','Technical details and supporting material'))]
    links = ''.join(f'<a href="#{ident}">{text}</a>' for ident,text in headings)
    css=(ROOT/'scripts/paper.css').read_text(); js=(ROOT/'scripts/paper.js').read_text()
    page='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Gait Fidelity — preserving changes in human movement</title><meta name="description" content="A core-informed study of predictive representations, movement measurement and the effects of downstream supervision."><style>'''+css+'''</style></head><body><a class="skip" href="#paper">Skip to the proposal</a><details class="mobile-nav"><summary>Gait Fidelity · contents</summary><nav aria-label="Mobile contents">'''+links+'''</nav></details><div class="paper-shell"><aside class="sidebar"><a class="brand" href="#gait-fidelity">Gait Fidelity</a><p>Core results available<br>Amended follow-up proposed</p><nav class="toc" aria-label="Paper contents">'''+links+'''</nav><div class="resources"><strong>Supporting material</strong><a href="methods/core-to-followup-20260924.md">Amended study design</a><a href="data/README.md">Data specification</a><a href="data/video-gallery.html">Local video viewer</a><a href="methods/README.md">Method protocols</a><a href="images/gallery.html">All vector figures</a><a href="reviews/README.md">Independent reviews</a><a href="README.md">Markdown source</a></div></aside><main id="paper">'''+content+'''</main></div><dialog id="figure-dialog" aria-labelledby="figure-description"><button id="close-figure" type="button">Close figure</button><img id="large-figure" alt=""><p id="figure-description"></p></dialog><script>'''+js+'''</script></body></html>\n'''
    overview = ('<div class="resources"><strong>Two-page overview</strong>'
                '<a href="proposal-brief.html">Read the concise proposal</a>'
                '<a href="proposal-brief.pdf">Download the two-page PDF</a></div>')
    page = page.replace('<div class="resources"><strong>Supporting material</strong>',
                        overview + '<div class="resources"><strong>Supporting material</strong>', 1)
    (ROOT/'proposal.html').write_text(page)
    # Include measured core results alongside clearly identified design schematics.
    figures=[]
    for path in sorted((ROOT/'images').glob('*.svg')):
        svg=path.read_text()
        title=unescape(re.search(r'<title[^>]*>(.*?)</title>',svg,re.S).group(1))
        desc=re.search(r'<desc[^>]*>(.*?)</desc>',svg,re.S)
        description=unescape(desc.group(1)) if desc else title
        preview = f'previews/{path.stem}-900.png'
        if not (ROOT / 'images' / preview).exists():
            preview = f'previews/{path.stem}.png'
        figures.append((path.name,title,description,preview))
    cards=''.join(f'<section><h2>{escape(title)}</h2><a href="{name}"><img src="{name}" alt="{escape(desc,quote=True)}"></a><p>{escape(desc)}</p><p><a href="{name}">Editable SVG</a> · <a href="{preview}">PNG preview</a></p></section>' for name,title,desc,preview in figures)
    gallery='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Gait Fidelity — figure gallery</title><style>body{margin:30px auto;max-width:1200px;padding:0 20px;background:#f2f6f9;color:#183247;font:17px/1.6 Arial,sans-serif}section{background:white;padding:20px;margin:28px 0;border:1px solid #d5e0e7;border-radius:12px}img{width:100%;height:auto}h2{font-size:22px}a{color:#326bb2}</style></head><body><h1>Gait Fidelity: visual guide</h1><p>Conceptual diagrams and explicitly illustrative calculations. <a href="../proposal.html">Interactive proposal</a> · <a href="../README.md">Markdown proposal</a> · <a href="../reviews/README.md">Independent review</a></p><p><strong>Version note:</strong> Earlier diagrams retain the original 18-phase follow-up. The amended readout design and current proposal specify the 27-phase experiment. <strong>Scope key:</strong> FOLLOW-UP marks a proposed comparison; INHERITED CORE marks the existing parent design; DIAGNOSTIC explains supporting measurements; DEFERRED marks future work outside this run. <a href="README.md">Figure meanings and rebuild instructions</a>.</p>'''+cards+'</body></html>\n'
    gallery = gallery.replace(
        'Conceptual diagrams and explicitly illustrative calculations.',
        'The completed-core results chart shows measured development means. '
        'Other figures explain experimental designs or explicitly illustrative calculations; '
        'none reports a result from the proposed follow-up.')
    gallery = gallery.replace(
        'INHERITED CORE marks the existing parent design;',
        'INHERITED CORE and INHERITED DESIGN mark the existing parent design;')
    (ROOT/'images/gallery.html').write_text(gallery)
    (ROOT/'records/paper-build.json').write_text(json.dumps({'sections':[i for i,t in headings],'figures':len(figures),'interactive_components':['centered residual coupling explorer','signed and absolute response explorer','section navigation','figure enlargement'],'scientific_source':'README.md','math_rendering':'offline SVG from LaTeX with Matplotlib MathText','equations':renderer.equations,'no_external_runtime_dependencies':True},indent=2)+'\n')
    print(f'Built proposal.html, {len(headings)} navigation entries and a {len(figures)}-figure gallery.')


if __name__=='__main__':
    build()
