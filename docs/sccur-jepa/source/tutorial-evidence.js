// The builder inserts the published estimates and paired bootstrap intervals.
(() => {
  const box = document.getElementById('measured-rankings');
  if (!box) return;
  const records = JSON.parse(document.getElementById('ranking-evidence').textContent);
  const names = ['GAVD', 'External'];
  const titles = ['91 GAVD recordings', '90 external participants'];
  const NS = 'http://www.w3.org/2000/svg';
  box.className = 'lab';
  box.innerHTML = '<h3 style="margin:0">Compare the measured scores</h3><p>Select a dataset to see how often each score ranked an abnormal example above a normal one. A value of 0.5 means chance ordering. Lines show 95% bootstrap intervals.</p><div class="choices"></div><svg class="plot" viewBox="0 0 600 181" role="img"></svg><div class="readout" role="status"></div>';
  const svg = box.querySelector('svg');
  function mark(type, attrs, text) {
    const el = document.createElementNS(NS, type);
    for (const [key, value] of Object.entries(attrs)) el.setAttribute(key,value);
    if (text) el.textContent = text;
    svg.append(el);
  }
  const left = 172, width = 360;
  const x = value => left + value * width;
  function display(i) {
    const rows = records.filter(row => row.dataset === names[i]);
    svg.replaceChildren();
    svg.setAttribute('aria-label',titles[i] + ': ' + rows.slice(0,2).map(row => row.quantity + ' ' + row.estimate.toFixed(3)).join(', '));
    const axis = '#657681';
    mark('line',{x1:x(.5),x2:x(.5),y1:16,y2:132,stroke:axis,'stroke-dasharray':'4 4'});
    rows.slice(0,2).forEach((row,index) => {
      const y = 37 + index * 56;
      const color = index === 0 ? '#006d60' : '#285d92';
      mark('text',{x:1,y:y+6,'font-size':17,fill:color},index === 0 ? 'Low movement' : 'Forecast score');
      mark('line',{x1:x(row.lower95),x2:x(row.upper95),y1:y,y2:y,stroke:color,'stroke-width':3});
      [row.lower95,row.upper95].forEach(v=>mark('line',{x1:x(v),x2:x(v),y1:y-6,y2:y+6,stroke:color,'stroke-width':2}));
      mark('circle',{cx:x(row.estimate),cy:y,r:6,fill:color});
      mark('text',{x:593,y:y+6,'font-size':17,'text-anchor':'end',fill:color},row.estimate.toFixed(3));
    });
    mark('line',{x1:x(0),x2:x(1),y1:133,y2:133,stroke:axis});
    [0,.5,1].forEach(v=>mark('text',{x:x(v),y:153,'text-anchor':'middle','font-size':15,fill:axis},String(v)));
    mark('text',{x:x(.5),y:176,'text-anchor':'middle','font-size':16,fill:axis},'AUROC');
    const diff = rows[2];
    const signed = v => (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(3);
    box.querySelector('.readout').textContent = 'Forecast minus low movement was ' + signed(diff.estimate) +
      ' (95% interval ' + signed(diff.lower95) + ' to ' + signed(diff.upper95) + '). ' +
      (i ? 'The interval includes zero, so the external data leave open which score would do better in another sample.' :
       'Both ends fall below zero. In this within-fold comparison, low movement ranked the recordings better.');
    [...box.querySelectorAll('button')].forEach((btn,n) => btn.setAttribute('aria-pressed', String(n===i)));
  }
  titles.forEach((text,i) => {
    const button = document.createElement('button');
    button.type='button';
    button.textContent=text;
    button.onclick=()=>display(i);
    box.querySelector('.choices').append(button);
  });
  display(0);
})();
