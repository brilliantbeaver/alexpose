// Render the actual HTML drafts, including print CSS, using installed Playwright.
// Example: NODE_PATH=/path/to/node_modules node .../render-check.cjs
const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');
const { pathToFileURL } = require('url');
const root = path.resolve(__dirname, '../..');
const out = '/tmp/gait-draft-render';
fs.mkdirSync(out, {recursive:true});
(async () => {
  const browser = await chromium.launch({headless:true});
  const page = await browser.newPage({viewport:{width:1280,height:900},deviceScaleFactor:1});
  const results=[];
  for (const kind of ['scientific','overview']) {
    const stem = kind === 'scientific' ? 'full-writeup' : 'research-overview';
    const errors=[];
    page.on('pageerror',e=>errors.push(e.message));
    await page.goto(pathToFileURL(path.join(root,stem+'.html')).href);
    await page.evaluate(async()=>{await document.fonts.ready;await Promise.all([...document.images].map(i=>i.decode()));});
    const info=await page.evaluate(()=>({
      title:document.title,
      images:[...document.images].map(i=>({src:i.getAttribute('src'),loaded:i.complete&&i.naturalWidth>0})),
      headings:[...document.querySelectorAll('h2')].map(i=>i.textContent),
      overflow:document.documentElement.scrollWidth>innerWidth,
      tables:[...document.querySelectorAll('table')].map(t=>({width:t.getBoundingClientRect().width,parentWidth:t.parentElement.getBoundingClientRect().width})),
    }));
    await page.screenshot({path:path.join(out,kind+'-desktop.png')});
    for(const fig of await page.locator('figure').all()){
      await fig.screenshot({path:path.join(out,kind+'-'+await fig.getAttribute('id')+'.png')});
    }
    await page.pdf({path:path.join(root,stem+'.pdf'),format:'Letter',preferCSSPageSize:true,printBackground:true,
      displayHeaderFooter:true,headerTemplate:'<div></div>',
      footerTemplate:'<div style="font-family:Arial;font-size:8px;color:#687681;width:100%;text-align:center">Gait fidelity · scientific draft · <span class="pageNumber"></span> / <span class="totalPages"></span></div>'});
    await page.setViewportSize({width:390,height:844});
    await page.evaluate(()=>scrollTo({top:0,behavior:'instant'}));
    await page.screenshot({path:path.join(out,kind+'-mobile.png')});
    info.mobileOverflow=await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth);
    results.push({kind,...info,errors});
    await page.setViewportSize({width:1280,height:900});
  }
  await browser.close();
  fs.writeFileSync(path.join(__dirname,'browser-checks.json'),JSON.stringify(results,null,2)+'\n');
  console.log(JSON.stringify(results,null,2));
})().catch(e=>{console.error(e);process.exitCode=1;});
