"""Exercise the real offline viewer and retain browser/render evidence.

Run with: uv run --project experiments/aaai --with playwright python
          experiments/aaai/scripts/validate_gait_slide_viewer.py
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw
from playwright.sync_api import sync_playwright

BASE=Path(__file__).resolve().parents[1]
OUT=BASE/'slides'
REVIEW=OUT/'review'


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def luminance(color):
    rgb=[int(color[i:i+2],16)/255 for i in (1,3,5)]
    rgb=[c/12.92 if c<=.04045 else ((c+.055)/1.055)**2.4 for c in rgb]
    return sum(c*w for c,w in zip(rgb,[.2126,.7152,.0722]))


def contrast(a,b):
    lo,hi=sorted([luminance(a),luminance(b)])
    return (hi+.05)/(lo+.05)


def run():
    records=json.loads((OUT/'deck_spec.json').read_text());count=len(records)
    report={'created_at_utc':datetime.now(timezone.utc).isoformat(),'html_sha256':digest(OUT/'index.html'),'pdf_sha256':digest(OUT/'tutorial.pdf'),'deck_sha256':digest(OUT/'deck_spec.json'),'slide_count':count,'checks':[]}
    def check(name,ok):
        assert ok,name
        report['checks'].append(name)
    check('Five structured note paragraphs on every slide',all(len(s['notes_sections'])==5 for s in records))
    for i in range(1,count+1):
        light=ET.parse(OUT/f'images/slide-{i:02d}.svg')
        dark=ET.parse(OUT/f'images/slide-{i:02d}-dark.svg')
        text=lambda tree:[t.text for t in tree.iter() if t.tag.endswith('text')]
        check(f'Slide {i}: light and dark vector text agree',text(light)==text(dark))
        check(f'Slide {i}: both themes embed both font weights',all((OUT/f'images/slide-{i:02d}{suffix}.svg').read_text().count('data:font/woff;base64,')==2 for suffix in ['', '-dark']))
    palettes={
        'light':{'surface':'#ffffff','raised':'#f2f6fa','ink':'#172e47','muted':'#506478','link':'#2357a5','teal':'#08787b','blue':'#315cb5','orange':'#ad521b','purple':'#6a51a3'},
        'dark':{'surface':'#111d2c','raised':'#1d2d42','ink':'#edf3fa','muted':'#b6c6d9','link':'#9cbcff','teal':'#64d8cf','blue':'#9cbcff','orange':'#ffc08d','purple':'#cbb2ff'},
    }
    ratios=[]
    for theme,p in palettes.items():
        for fg in ['ink','muted','link','teal','blue','orange','purple']:
            for bg in ['surface','raised']:
                value=contrast(p[fg],p[bg]);ratios.append({'theme':theme,'text':fg,'background':bg,'ratio':round(value,3)})
                check(f'{theme} {fg} on {bg} text contrast >= 4.5',value>=4.5)
    report['text_contrast']=ratios
    report['control_boundary_contrast']=[]
    for theme,color in [('light','#6f8295'),('dark','#71869c')]:
        for background in ['surface','raised']:
            value=contrast(color,palettes[theme][background])
            report['control_boundary_contrast'].append({'theme':theme,'background':background,'ratio':round(value,3)})
            check(f'{theme} control boundary on {background} >= 3',value>=3)
    with sync_playwright() as pw:
        # Use the installed Playwright browser when available. Older cached
        # headless versions also support this offline HTML/CSS test.
        candidates=sorted((Path.home()/'Library/Caches/ms-playwright').glob('chromium_headless_shell-*/chrome-headless-shell-mac-arm64/chrome-headless-shell'))
        browser=pw.chromium.launch(executable_path=str(candidates[-1]) if candidates else None)
        context=browser.new_context(viewport={'width':1440,'height':1100},color_scheme='light',reduced_motion='reduce')
        page=context.new_page();errors=[]
        page.on('pageerror',lambda e:errors.append(str(e)))
        page.goto((OUT/'index.html').as_uri());page.wait_for_load_state('networkidle')
        check('Every slide has three named theme controls',page.locator('.slide [data-choice]').count()==3*count)
        check('Theme feedback uses one shared live region',page.locator('.theme-status[role="status"]').count()==0 and page.locator('#theme-announcement[role="status"]').count()==1)
        check('Device theme starts light',page.locator('html').get_attribute('data-theme')=='light')
        page.select_option('#deck-theme','dark')
        check('Deck theme updates every inherited slide',page.locator('.slide[data-theme="dark"]').count()==count)
        for i in range(1,count+1):
            section=page.locator(f'#slide-{i}')
            for theme in ['light','dark']:
                section.locator(f'[data-choice="{theme}"]').click()
                check(f'Slide {i}: {theme} control updates image and pressed state',section.get_attribute('data-theme')==theme and section.locator(f'[data-choice="{theme}"]').get_attribute('aria-pressed')=='true' and section.locator('img').get_attribute('src').endswith('-dark.svg' if theme=='dark' else f'{i:02d}.svg'))
            section.locator('img').screenshot(path=str(REVIEW/f'dark-render-{i:02d}.png'))
            section.locator('[data-choice="inherit"]').click()
        page.locator('#slide-1 [data-choice="light"]').click()
        page.reload();page.wait_for_load_state('networkidle')
        check('Per-slide choice survives reload',page.locator('#slide-1').get_attribute('data-theme')=='light')
        check('Deck choice survives reload',page.locator('html').get_attribute('data-theme')=='dark')
        page.select_option('#deck-theme','light')
        page.locator('#slide-2 [data-choice="dark"]').click()
        check('One slide override does not alter its neighbors',page.locator('#slide-2').get_attribute('data-theme')=='dark' and page.locator('#slide-3').get_attribute('data-theme')=='light')
        page.locator('#reset-themes').click()
        check('Reset makes all slides inherit the deck',page.locator('.slide[data-theme-choice="inherit"]').count()==count)
        page.select_option('#deck-theme','system');page.emulate_media(color_scheme='dark')
        page.wait_for_function("document.documentElement.dataset.theme==='dark'")
        check('Device changes update inherited slides',page.locator('.slide[data-theme="dark"]').count()==count)
        # Activate one theme with the keyboard and verify its visible focus.
        button=page.locator('#slide-1 [data-choice="light"]');button.focus();page.keyboard.press('Enter')
        check('Theme controls respond to keyboard Enter',page.locator('#slide-1').get_attribute('data-theme')=='light')
        check('Focused button has a visible outline',button.evaluate("e=>getComputedStyle(e).outlineStyle")!='none')
        page.keyboard.press('Space')
        page.locator('#reset-themes').click()
        page.select_option('#jump','22')
        page.locator('#slide-23 details.glossary summary').click()
        page.locator('#slide-23 details.notes summary').click()
        check('Score guide and speaker notes open independently',page.locator('#slide-23 details.glossary').get_attribute('open') is not None and page.locator('#slide-23 details.notes').get_attribute('open') is not None)
        page.locator('#slide-23').screenshot(path=str(REVIEW/'score-guide-dark.png'))
        page.locator('#toggle').click()
        check('Open all notes works',page.locator('details.notes[open]').count()==count)
        page.locator('#toggle').click()
        check('Close all notes works',page.locator('details.notes[open]').count()==0)
        page.locator('#slide-23 details.glossary summary').click()
        page.select_option('#jump','0');page.wait_for_timeout(150)
        page.screenshot(path=str(REVIEW/'viewer-desktop-dark.png'))
        page.select_option('#deck-theme','light');page.screenshot(path=str(REVIEW/'viewer-desktop-light.png'))
        page.set_viewport_size({'width':375,'height':900})
        page.locator('#slide-1 details.notes summary').click()
        page.screenshot(path=str(REVIEW/'viewer-mobile-light.png'))
        check('Mobile layout has no horizontal page overflow',page.evaluate('document.documentElement.scrollWidth<=innerWidth'))
        sizes=page.locator('button,select,summary').evaluate_all('(es)=>es.filter(e=>e.getBoundingClientRect().height>0).map(e=>({tag:e.tagName,w:e.getBoundingClientRect().width,h:e.getBoundingClientRect().height}))')
        check('All visible interactive targets are at least 44 pixels tall',all(s['h']>=44 for s in sizes))
        page.set_viewport_size({'width':720,'height':550})
        check('Narrow layout at 200-percent desktop-equivalent width has no horizontal overflow',page.evaluate('document.documentElement.scrollWidth<=innerWidth'))
        check('No JavaScript runtime errors',not errors)
        report['browser']=browser.version;report['javascript_errors']=errors
        report['accessibility_scope']='Keyboard, semantic controls, contrast, reduced-motion handling, 375px layout and 720px reflow checked. No VoiceOver or NVDA session was performed; this is not a complete WCAG certification.'
        browser.close()
    for start in range(0,count,10):
        sheet=Image.new('RGB',(1280,1925),'#0b1420');draw=ImageDraw.Draw(sheet)
        for j,i in enumerate(range(start+1,min(start+11,count+1))):
            im=Image.open(REVIEW/f'dark-render-{i:02d}.png').convert('RGB');im.thumbnail((640,360));x=j%2*640;y=j//2*385;sheet.paste(im,(x,y));draw.text((x+12,y+362),f'Slide {i:02d}',fill='#edf3fa')
        sheet.save(REVIEW/f'dark-contact-sheet-{start//10+1:02d}.png')
    report['passed']=True;report['status']='passed'
    (REVIEW/'viewer_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'passed':True,'checks':len(report['checks']),'slides':count,'browser':report['browser']}))


if __name__=='__main__':run()
