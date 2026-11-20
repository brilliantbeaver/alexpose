// Independent teaching examples; all displayed scores and positions in these labs are hypothetical.
(() => {
  function lab(id, title, prompt, controls) {
    const panel = document.getElementById('lab-' + id);
    if (!panel) return null;
    panel.classList.add('lab');
    panel.innerHTML = '<h2>' + title + '</h2><p><strong>Practice example.</strong> ' + prompt + '</p>' + controls +
      '<div class="readout" role="status" aria-live="polite" aria-atomic="true"></div>';
    return panel;
  }

  function splitLab() {
    const panel = lab('start','Split four clips','These invented clips come from videos A and B. Change the split and see which videos appear on both sides.',
      '<div class="choices"><button type="button" data-split="clip">Split clips</button><button type="button" data-split="source">Hold out a video</button></div><div data-clips></div>');
    if (!panel) return;
    function render(mode) {
      panel.querySelectorAll('[data-split]').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.split === mode)));
      const clips = mode === 'clip' ? [['A1','B1'],['A2','B2']] : [['A1','A2'],['B1','B2']];
      panel.querySelector('[data-clips]').innerHTML = clips.map((row, i) => '<div class="strip"><b>' + (i ? 'Test' : 'Train') + '</b>' + row.map(id => '<span class="chip ' + (id[0] === 'B' ? 'b' : '') + '">' + id + '</span>').join('') + '</div>').join('');
      panel.querySelector('.readout').textContent = mode === 'clip' ? 'Both test clips come from videos seen during training. A model could recognize details of those recordings.' : 'Neither test clip comes from a training video. Both test clips are still from the same recording, B.';
    }
    panel.querySelectorAll('[data-split]').forEach(button => button.addEventListener('click', () => render(button.dataset.split)));
    render('clip');
  }

  function maskLab() {
    const panel = lab('classifier','Hide a joint observation','Each cell is one joint at one moment in an invented walking clip. Select a visible cell to hide it from the student. Select it again to reveal it.',
      '<div class="maskgrid" role="group" aria-label="Joint observations by frame"></div><div class="choices"><button type="button" data-reset>Reveal all observed joints</button></div>');
    if (!panel) return;
    const grid = panel.querySelector('.maskgrid');
    grid.innerHTML = '<span></span>' + [1,2,3,4].map(i => '<span class="axis">Frame ' + i + '</span>').join('');
    const buttons = [];
    for (const joint of ['Hip','Knee','Ankle']) {
      const rowLabel = document.createElement('strong');
      rowLabel.textContent = joint;
      grid.append(rowLabel);
      for (let frame = 1; frame <= 4; frame++) {
        const b = document.createElement('button');
        b.type = 'button';
        const label = joint + ', frame ' + frame;
        b.disabled = joint === 'Ankle' && frame === 4;
        function show(hidden) {
          b.textContent = b.disabled ? 'Missing' : hidden ? 'Hidden' : 'Visible';
          b.setAttribute('aria-label',label + ': ' + b.textContent);
          if (!b.disabled) b.setAttribute('aria-pressed', String(hidden));
        }
        show(false);
        b.addEventListener('click', () => { show(b.getAttribute('aria-pressed') !== 'true'); feedback(); });
        grid.append(b);
        buttons.push({b,show});
      }
    }
    function feedback() {
      const n = buttons.filter(({b}) => b.getAttribute('aria-pressed') === 'true').length;
      let msg = n + ' hidden ' + (n === 1 ? 'observation gives the teacher one target.' : 'observations give the teacher ' + n + ' targets.');
      if (n === 0) msg = 'No observed joints are hidden yet. Hide one to give the student a prediction task.';
      if (n === 11) msg += ' The student now has no visible context; reveal some cells to make a useful task.';
      panel.querySelector('.readout').textContent = msg + ' The camera missed the last ankle position, so it cannot supply a target.';
    }
    panel.querySelector('[data-reset]').addEventListener('click',() => { buttons.forEach(({show}) => show(false)); feedback(); });
    feedback();
  }

  function slider(panel,id,label,initial) {
    const input = panel.querySelector('input[type="range"]');
    input.id = id;
    const control = panel.querySelector('label');
    control.htmlFor = id;
    control.textContent = label + ' ';
    const output = document.createElement('output');
    output.setAttribute('for',id);
    control.append(output);
    input.value = initial;
    return (render) => {
      function update() { const v=Number(input.value); output.textContent=String(v); render(v); }
      input.addEventListener('input',update);
      update();
    };
  }
  const range = '<p><label></label><br><input type="range" min="0" max="6" step="1"></p>';
  function errorLab(){
    const panel = lab('restart','Move the observed ankle','In these invented coordinates, the ankle started at 2. Copying predicts 2; a learned model predicts 4. Move the observation and compare their errors.',
      range + '<svg class="plot" viewBox="0 0 540 160" role="img"></svg>');
    if (!panel) return;
    const plot=panel.querySelector('svg');
    slider(panel,'ankle-observation','Observed ankle position:',5)(v=>{
      const at=n=>138+n*61;
      const labels=[0,1,2,3,4,5,6].map(n=>'<text x="'+at(n)+'" y="147" text-anchor="middle">'+n+'</text>').join('');
      const rows=[['Copy',2,44,'#285d92'],['Learned',4,90,'#006d60']].map(([name,pred,y,c])=>
        '<text x="5" y="'+(y+5)+'">'+name+'</text><line x1="'+at(pred)+'" x2="'+at(v)+'" y1="'+y+'" y2="'+y+'" stroke="'+c+'" stroke-width="4"/><circle cx="'+at(pred)+'" cy="'+y+'" r="6" fill="'+c+'"/><circle cx="'+at(v)+'" cy="'+y+'" r="10" fill="none" stroke="#1f3864" stroke-width="2"/>').join('');
      plot.setAttribute('aria-label','Observed ankle at '+v+'. The filled dots show forecasts at 2 and 4. Lines show the distance to the observation.');
      plot.innerHTML='<g font-family="Arial,sans-serif" font-size="18" fill="#20394d"><line x1="'+at(v)+'" x2="'+at(v)+'" y1="23" y2="122" stroke="#94a6b0" stroke-dasharray="5 4"/>'+rows+'<path d="M138 122H504" stroke="#91a4ad"/>'+labels+'</g>';
      const a=(2-v)**2,b=(4-v)**2;
      panel.querySelector('.readout').textContent='Squared errors: copying = '+a+'; learned forecast = '+b+'. '+(a===b?'The two forecasts tie.':a<b?'Copying is closer here.':'The learned forecast is closer here.')+' The open circles show the observation.';
    });
  }
  function rankLab(){
    const panel=lab('forecast','Rank four pairs','Two normal recordings have invented scores of 1 and 4. One abnormal recording scores 5. Move the other abnormal score. Higher scores are meant to indicate more unusual gait.',
      range+'<div class="tablewrap"><table><caption>For each pair, does the abnormal score rank higher?</caption><thead><tr><th scope="col">Abnormal score</th><th scope="col">Normal score 1</th><th scope="col">Normal score 4</th></tr></thead><tbody></tbody></table></div>');
    if(!panel)return;
    slider(panel,'abnormal-score','First abnormal recording’s score:',3)(v=>{
      let points=0;
      panel.querySelector('tbody').innerHTML=[v,5].map((a,i)=>'<tr><th scope="row">'+(i?'Fixed: ':'Adjustable: ')+a+'</th>'+[1,4].map(n=>{
        const p=a>n?1:a===n?.5:0; points+=p;
        return '<td>'+a+(p===1?' &gt; ':p===0?' &lt; ':' = ')+n+'<br>'+(p===1?'Correct · 1':p===0?'Reversed · 0':'Tie · ½')+'</td>';
      }).join('')+'</tr>').join('');
      panel.querySelector('.readout').textContent='These four comparisons earn '+points+' of 4 possible points. AUROC = '+points+' ÷ 4 = '+(points/4)+'. Each tie earns half a point.';
    });
  }
  function init(){splitLab();maskLab();errorLab();rankLab();}
  if(document.readyState==='loading') document.addEventListener('DOMContentLoaded',init,{once:true}); else init();
})();
