// Source: frozen draft03/refs_source_scores.csv. Each MSE summarizes all eight
// future offsets within one recording. Selection was fixed as the median gain
// among held-out normal recordings; ties sort by source ID. In draft03 this is
// rank 15/29, source MN4vnaNwIsA (116.019857 − 64.796131 = 51.223726 ×10⁻³).
const fs = require('node:fs');
const path = require('node:path');

function medianExample() {
  const file=path.join(__dirname,'../../../experiments/aaai-worldmodels/docs/draft03/refs_source_scores.csv');
  const pairs=new Map();
  for (const row of fs.readFileSync(file,'utf8').trim().split(/\r?\n/).slice(1)) {
    const [id,label,,,mse,arm,horizon]=row.split(',');
    if(label!=='normal'||horizon!=='8'||!['persistence','ridge_posture'].includes(arm)) continue;
    const pair=pairs.get(id)??{id};
    if(arm in pair || !mse.trim() || !Number.isFinite(+mse)) throw Error('Invalid paired error for '+id);
    pair[arm]=1000*Number(mse);
    pairs.set(id,pair);
  }
  const ordered=[...pairs.values()].map(p=>({...p,gain:p.persistence-p.ridge_posture}));
  if(ordered.length!==29||!ordered.every(p=>Number.isFinite(p.gain))) throw Error('Expected 29 complete normal pairs');
  ordered.sort((a,b)=>a.gain-b.gain||(a.id<b.id?-1:a.id>b.id?1:0));
  return ordered[Math.floor(ordered.length/2)];
}

module.exports=function drawWorkedExample(pptx,slide,x,y,w,h) {
  const record=medianExample();
  const scale=Math.min(w/20.1,h/2);
  const navy='183B4D',teal='00796D',orange='B64916';
  function txt(str,left,top,width,height,pt,bold=false,color=navy) {
    slide.addText(str,{x:x+w*left/20.1,y:y+h*top/2,w:w*width/20.1,h:h*height/2,
      color,fontFace:'Arial',fontSize:pt*scale,bold,margin:0,valign:'mid',align:'center'});
  }
  txt('How to read the middle bar',0,.01,6.4,.63,32,true);
  txt('Recording 15 of 29 · MSE × 1,000',0,.81,6.4,.53,24);
  txt('Repeat last pose',6.47,.04,4.03,.49,25,true);
  txt(record.persistence.toFixed(1),6.47,.68,4.03,.87,44,true);
  txt('−',10.50,.68,.7,.86,40);
  txt('Learned forecast',11.15,.04,4.34,.49,25,true);
  txt(record.ridge_posture.toFixed(1),11.15,.68,4.34,.87,44,true);
  txt('=',15.49,.68,.7,.86,40);
  const color=record.gain>=0?teal:orange;
  txt('Bar height',16.1,.04,3.98,.49,25,true,color);
  txt((record.gain>=0?'+':'')+record.gain.toFixed(1),16.1,.68,3.98,.87,44,true,color);
};
