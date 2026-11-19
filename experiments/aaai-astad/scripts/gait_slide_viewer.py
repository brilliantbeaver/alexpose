"""Offline slide reader with explicit, persistent per-slide color themes."""
from html import escape
from pathlib import Path
import base64
import hashlib
import io
import json
import re

# Preserve the semantic color roles in the vector masters. This is not a
# browser inversion filter, which would also invert illustrations and legends.
DARK_PALETTE = {
    '#ffffff': '#111d2c', '#172e47': '#edf3fa', '#506478': '#b6c6d9',
    '#f2f6fa': '#1d2d42', '#d9e3ec': '#526980', '#087f82': '#64d8cf', '#08787b': '#64d8cf',
    '#315cb5': '#9cbcff', '#ad521b': '#ffc08d', '#6a51a3': '#cbb2ff',
}


def dark_svg(source):
    colors = set(re.findall(r'#[a-fA-F0-9]{6}\b', source))
    unknown = colors - DARK_PALETTE.keys()
    if unknown:
        raise ValueError(f'Assign explicit dark-theme colors before exporting: {unknown}')
    return re.sub(r'#[a-fA-F0-9]{6}\b', lambda m: DARK_PALETTE[m[0]], source)


def embed_svg_fonts(paths, text, destination):
    """Keep SVG typography identical offline, even without system DejaVu."""
    import matplotlib
    from fontTools import subset
    from fontTools.ttLib import TTFont
    source=Path(matplotlib.get_data_path())/'fonts/ttf'
    destination.mkdir(exist_ok=True)
    css=[];manifest=[]
    for weight,filename in [(400,'DejaVuSans.ttf'),(700,'DejaVuSans-Bold.ttf')]:
        font=TTFont(source/filename)
        if 'FFTM' in font:del font['FFTM']
        options=subset.Options();subsetter=subset.Subsetter(options=options)
        subsetter.populate(text=text);subsetter.subset(font)
        font.flavor='woff';buffer=io.BytesIO();font.save(buffer);data=buffer.getvalue()
        output=destination/f'gait-deck-{weight}.woff';output.write_bytes(data)
        encoded=base64.b64encode(data).decode('ascii')
        css.append(f"@font-face{{font-family:'Gait Deck Sans';font-style:normal;font-weight:{weight};src:url(data:font/woff;base64,{encoded}) format('woff')}}")
        manifest.append({'weight':weight,'file':output.name,'sha256':hashlib.sha256(data).hexdigest(),'source_file':filename,'source_sha256':hashlib.sha256((source/filename).read_bytes()).hexdigest()})
    (destination/'LICENSE_DEJAVU.txt').write_text((source/'LICENSE_DEJAVU').read_text())
    (destination/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    style='<style type="text/css">'+''.join(css)+'</style>'
    for path in paths:
        svg=path.read_text().replace("font-family: 'DejaVu Sans'", "font-family: 'Gait Deck Sans', sans-serif")
        path.write_text(svg.replace('<defs>','<defs>'+style,1))
    return manifest


CSS = '''
:root,[data-theme="light"]{color-scheme:light;--page:#eaf0f6;--surface:#fff;--raised:#f2f6fa;--ink:#172e47;--muted:#506478;--line:#6f8295;--accent:#087f82;--link:#2357a5;--selected:#dceff0;--focus:#ad521b}
[data-theme="dark"]{color-scheme:dark;--page:#0b1420;--surface:#111d2c;--raised:#1d2d42;--ink:#edf3fa;--muted:#b6c6d9;--line:#71869c;--accent:#64d8cf;--link:#9cbcff;--selected:#254349;--focus:#ffc08d}
*{box-sizing:border-box}html{scroll-padding-top:170px}body{margin:0;background:var(--page);color:var(--ink);font:17px/1.65 system-ui,sans-serif}a{color:var(--link);text-underline-offset:3px}button,select{font:inherit;color:var(--ink);background:var(--surface);border:1px solid var(--line);border-radius:8px;min-height:44px;padding:8px 12px;cursor:pointer}button:hover,select:hover{background:var(--raised)}button:disabled{opacity:.45;cursor:default}button[aria-pressed="true"]{color:var(--ink);background:var(--selected);border-color:var(--accent);box-shadow:inset 0 0 0 1px var(--accent);font-weight:650}button:focus-visible,select:focus-visible,a:focus-visible,summary:focus-visible{outline:3px solid var(--focus);outline-offset:3px}header{position:sticky;top:0;z-index:2;padding:12px 3vw;background:var(--surface);border-bottom:1px solid var(--line)}.toolbar,.reader-nav,.slide-tools,.theme-choices{display:flex;align-items:center;gap:12px;flex-wrap:wrap}.toolbar{justify-content:space-between}.toolbar h1{font-size:20px;margin:0}.deck-theme{display:flex;align-items:center;gap:10px;flex-wrap:wrap}.deck-theme label{font-size:15px}.reader-nav{margin-top:10px}.reader-nav select{flex:1;min-width:100px;max-width:700px}.reader-nav #count{margin-left:auto;font-variant-numeric:tabular-nums}.toolbar a{min-height:44px;display:inline-flex;align-items:center}.deck-links{display:flex;gap:20px}.skip{position:absolute;left:16px;top:-100px;background:var(--surface);padding:12px;z-index:4}.skip:focus{top:12px}main{max-width:1316px;margin:auto;padding:18px}section.slide{background:var(--surface);color:var(--ink);margin:0 0 32px;scroll-margin-top:170px;border:1px solid var(--line);border-radius:12px;overflow:hidden}.slide-tools{padding:12px 20px;background:var(--raised);border-bottom:1px solid var(--line);justify-content:space-between}.slide-number{font-size:15px;font-weight:650}.theme-choices{gap:6px}.theme-label,.theme-status{font-size:14px;color:var(--muted)}.theme-label{margin-right:5px}.theme-status{min-width:135px}.slide img{width:100%;height:auto;display:block;aspect-ratio:16/9}.slide details{padding:14px 28px}.slide summary{cursor:pointer;font-weight:650;min-height:44px;padding:8px 0}.slide h3{margin:24px 0 6px;font-size:19px}.slide p{max-width:1000px;margin:6px 0 16px}.slide code{font-size:.92em;background:var(--raised);padding:2px 4px;overflow-wrap:anywhere}.sources{font-size:15px;border-top:1px solid var(--line);padding-top:16px}.transcript{font-size:16px;background:var(--raised)}.glossary{border-top:1px solid var(--line)}.caption{padding:14px 28px 0;color:var(--muted);font-size:16px}.reader-help{font-size:15px;color:var(--muted);margin:0 0 18px}.sr-only{position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap;border:0}#storage-note{font-size:14px;color:var(--muted)}
@media(max-width:700px){header{position:static;padding:12px 16px}html{scroll-padding-top:10px}.toolbar{align-items:flex-start}.toolbar h1{font-size:18px}.reader-nav #toggle{order:3}.reader-nav select{max-width:calc(100% - 116px)}main{padding:10px}.slide-tools{padding:10px;gap:6px}.theme-choices{width:100%;justify-content:flex-start}.theme-label{flex-basis:100%}.theme-status{min-width:0}.slide details,.caption{padding-left:16px;padding-right:16px}section.slide{scroll-margin-top:10px}}
@media(prefers-reduced-motion:reduce){*{scroll-behavior:auto!important}}
@media print{header,.slide-tools,.reader-help,.transcript,.glossary{display:none}section.slide{break-after:page;margin:0;border:0}main{padding:0}.slide details{display:block}@page{size:landscape;margin:12mm}}
'''

JS = '''
const slides=[...document.querySelectorAll('section.slide')];
const systemTheme=matchMedia('(prefers-color-scheme: dark)');
const storageKey='gait-slide-reader-themes-v1';
let preference={deck:'system',slides:{}};
try {const saved=JSON.parse(localStorage.getItem(storageKey)||'null');if(saved){if(['light','dark','system'].includes(saved.deck))preference.deck=saved.deck;if(saved.slides&&typeof saved.slides==='object')preference.slides=saved.slides;}} catch(e){}
function save(){try{localStorage.setItem(storageKey,JSON.stringify(preference));}catch(e){document.getElementById('storage-note').textContent='Your browser cannot save these choices. They still apply while this page is open.';}}
function effectiveDeck(){return preference.deck==='system'?(systemTheme.matches?'dark':'light'):preference.deck;}
function applyThemes(){const base=effectiveDeck();document.documentElement.dataset.theme=base;document.getElementById('deck-theme').value=preference.deck;for(const slide of slides){const stored=preference.slides[slide.dataset.contentId];const own=['light','dark'].includes(stored)?stored:'inherit';const theme=own==='inherit'?base:own;slide.dataset.theme=theme;slide.dataset.themeChoice=own;const img=slide.querySelector('img');img.src=theme==='dark'?img.dataset.dark:img.dataset.light;for(const b of slide.querySelectorAll('[data-choice]'))b.setAttribute('aria-pressed',String(b.dataset.choice===own));slide.querySelector('.theme-status').textContent=own==='inherit'?'Deck theme: '+theme:'This slide: '+theme;}}
document.getElementById('deck-theme').addEventListener('change',e=>{preference.deck=e.target.value;applyThemes();save();document.getElementById('theme-announcement').textContent='The deck theme is '+effectiveDeck()+'. Individual slide choices are preserved.';});
document.getElementById('reset-themes').addEventListener('click',()=>{preference.slides={};applyThemes();save();document.getElementById('theme-announcement').textContent='Every slide now follows the deck theme.';});
for(const slide of slides)for(const button of slide.querySelectorAll('[data-choice]'))button.addEventListener('click',()=>{const key=slide.dataset.contentId;if(button.dataset.choice==='inherit')delete preference.slides[key];else preference.slides[key]=button.dataset.choice;applyThemes();save();document.getElementById('theme-announcement').textContent=slide.id.replace('-',' ')+' uses the '+slide.dataset.theme+' theme'+(button.dataset.choice==='inherit'?' from the deck setting.':'.');});
systemTheme.addEventListener('change',()=>{if(preference.deck==='system')applyThemes();});
applyThemes();
let current=0;const reduced=matchMedia('(prefers-reduced-motion: reduce)');
function mark(n){current=n;document.getElementById('count').textContent=(n+1)+' / '+slides.length;document.getElementById('jump').value=n;document.getElementById('prev').disabled=n===0;document.getElementById('next').disabled=n===slides.length-1;}
function go(n){mark(Math.max(0,Math.min(slides.length-1,n)));slides[current].scrollIntoView({behavior:reduced.matches?'instant':'smooth',block:'start'});}
document.getElementById('prev').onclick=()=>go(current-1);document.getElementById('next').onclick=()=>go(current+1);document.getElementById('jump').onchange=e=>go(+e.target.value);
document.getElementById('toggle').onclick=e=>{const open=e.target.getAttribute('aria-pressed')!=='true';document.querySelectorAll('details.notes').forEach(d=>d.open=open);e.target.textContent=open?'Hide all notes':'Show all notes';e.target.setAttribute('aria-pressed',String(open));};
document.addEventListener('keydown',e=>{if(document.activeElement.closest('select,input,textarea,button,summary,a')||e.ctrlKey||e.metaKey||e.altKey)return;if(['ArrowRight','PageDown'].includes(e.key)){e.preventDefault();go(current+1);}if(['ArrowLeft','PageUp'].includes(e.key)){e.preventDefault();go(current-1);}});
const observer=new IntersectionObserver(entries=>{const visible=entries.filter(e=>e.isIntersecting).sort((a,b)=>b.intersectionRatio-a.intersectionRatio);if(visible.length)mark(slides.indexOf(visible[0].target));},{threshold:[0,.25,.5,.75]});slides.forEach(s=>observer.observe(s));mark(0);
'''

GLOSSARY = [
    ('Movement score M', 'This score summarizes what the movement looks like. In the video study, its 16 measurements describe knee posture, how far the ankles move, changes in movement speed, trunk tilt, and differences between the two sides. The laboratory study instead uses 12 measurements of joint-angle levels and ranges. M compares the observed pattern with the normal fitting group. A larger score means a more unusual combination of these visible measurements.'),
    ('Cross-leg score R', 'First, compare two predictions of the same hidden movement. Both retain the initial own-leg estimate. One adds a correction using only that leg, while the other adds a correction that can also see the opposite leg. Subtract the both-leg squared error from the own-leg squared error to measure the added benefit. Next, compare the left-leg and right-leg benefits with separate normal references. The same reserved reference group supplies both sides. R is larger when those benefits are unusually high or unusually low.'),
    ('What normal-reference scaling means', 'For each leg, the study subtracts the normal reference mean benefit from the observed benefit. It then divides by an estimated normal standard deviation, which measures the spread of reference benefits, and squares the result. R averages these two squared departures. This is the meaning of calibration here. It measures departure from the reference; it does not produce a probability of disease. Each video fold has five separate normal reference recordings. Each laboratory fold has 20 separate healthy reference participants.'),
    ('Combined score S', 'S adds the movement score M and the cross-leg score R, then divides by two. Thus it gives the two numerical scores fixed equal weight. Equal weights do not guarantee that the two scores have equally reliable scales or equal influence on ranking. The experiment tests whether this particular average improves detection.'),
    ('Prediction benefit and detection accuracy', 'Prediction benefit means a reduction in error for a hidden movement. Detection accuracy means better separation of normal-labeled and abnormal-labeled recordings. A useful prediction benefit does not automatically improve detection. AUROC summarizes how often an abnormal recording receives a higher score than a normal recording; 0.5 is chance ranking and 1.0 is perfect ranking on the evaluated data. Here, ranking is measured within each held-out test group, then averaged with fixed weights.'),
]


def render_viewer(records, sources, note_fields):
    def inline(value):
        return re.sub(r'`([^`]+)`', r'<code>\1</code>', escape(value))
    total=len(records)
    html=[f'<!doctype html><html lang="en" data-theme="light"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="color-scheme" content="light dark"><title>Understanding gait prediction</title><style>{CSS}</style></head><body><a class="skip" href="#slides">Skip to slides</a><header><div class="toolbar"><h1>Understanding gait prediction</h1><div class="deck-links"><a href="tutorial.pdf">Slide PDF</a><a href="tutorial.md">Speaker notes</a></div><div class="deck-theme"><label for="deck-theme">Deck theme</label><select id="deck-theme"><option value="system">Follow device</option><option value="light">Light</option><option value="dark">Dark</option></select><button id="reset-themes">Apply deck theme to all</button></div></div><nav class="reader-nav" aria-label="Slide navigation"><button id="prev" aria-label="Previous slide">←</button><button id="next" aria-label="Next slide">→</button><select id="jump" aria-label="Choose a slide">']
    html += [f'<option value="{i-1}">{i:02d}. {escape(s["title"])}</option>' for i,s in enumerate(records,1)]
    html += [f'</select><button id="toggle" aria-pressed="false">Show all notes</button><span id="count">1 / {total}</span></nav><div id="storage-note" role="status"></div><span id="theme-announcement" class="sr-only" role="status"></span></header><main id="slides"><p class="reader-help">Choose a deck theme above, or choose Light or Dark for an individual slide. “Use deck” follows the deck setting. Your choices are saved in this browser. Arrow keys move between slides when a control is not focused.</p><noscript><p>Theme controls need JavaScript. The light slides, notes and PDF remain available without it.</p></noscript>']
    for i,s in enumerate(records,1):
        stem=f'images/slide-{i:02d}'
        html.append(f'<section class="slide" id="slide-{i}" data-content-id="{s["content_id"]}" data-theme="light" aria-labelledby="title-{i}"><h2 class="sr-only" id="title-{i}">{i}. {escape(s["title"])}</h2><div class="slide-tools"><span class="slide-number">Slide {i:02d} of {total}</span><div class="theme-choices" role="group" aria-label="Theme for slide {i}"><span class="theme-label">Slide theme</span>')
        for value,label in [('inherit','Use deck'),('light','Light'),('dark','Dark')]:
            html.append(f'<button type="button" data-choice="{value}" aria-pressed="{str(value=="inherit").lower()}" aria-label="{label} theme for slide {i}">{label}</button>')
        html.append(f'<span class="theme-status">Deck theme: light</span></div></div><img src="{stem}.svg" data-light="{stem}.svg" data-dark="{stem}-dark.svg" width="1280" height="720" alt="Slide {i}: {escape(s["title"],quote=True)}. The complete text is available below."><p class="caption">{escape(s["takeaway"])}</p><details class="notes"><summary>Speaker notes for slide {i}</summary>')
        for key,label in note_fields:
            html.append(f'<h3>{label}</h3><p>{inline(s["notes_sections"][key])}</p>')
        refs=' · '.join(f'<a href="{escape(sources[k][1],quote=True)}">{escape(sources[k][0])}</a>' for k in s['refs'])
        html.append(f'<p class="sources">Sources: {refs}</p></details>')
        if s.get('score_guide'):
            html.append('<details class="glossary"><summary>Plain-language guide to M, R and S</summary>')
            for title,text in GLOSSARY:html.append(f'<h3>{title}</h3><p>{escape(text)}</p>')
            html.append('</details>')
        html.append('<details class="transcript"><summary>Read the slide as text</summary>')
        html += [f'<p>{escape(t)}</p>' for t in s['visible_text']]
        html.append('</details></section>')
    html.append(f'</main><script>{JS}</script></body></html>')
    return '\n'.join(html)
