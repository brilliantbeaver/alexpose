// Per-recording differences for the eight-offset, held-out normal comparison.
const fs = require('node:fs');
const path = require('node:path');

function getGains() {
  const file = path.join(__dirname, '../../../experiments/aaai-worldmodels/docs/draft03/refs_source_scores.csv');
  const pairs = new Map();
  for (const row of fs.readFileSync(file, 'utf8').trim().split(/\r?\n/).slice(1)) {
    const [source, label, , , mse, arm, horizon] = row.split(',');
    if (label !== 'normal' || horizon !== '8' || !['ridge_posture', 'persistence'].includes(arm)) continue;
    const pair = pairs.get(source) ?? {};
    if (arm in pair || !mse.trim() || !Number.isFinite(+mse)) throw Error('Unexpected source result: ' + source);
    pair[arm] = +mse;
    pairs.set(source, pair);
  }
  const gains = [...pairs.values()].map(pair => 1000*(pair.persistence - pair.ridge_posture));
  if (gains.length !== 29 || !gains.every(Number.isFinite)) throw Error('Missing forecast pair');
  return gains.sort((a,b) => a-b);
}

module.exports = function recordingVisual(pptx,slide,x,y,w,h) {
  const teal='00796D', orange='B64916', ink='173A48', muted='46565E', grid='CBD5D6';
  const scale=Math.min(w/20.1,h/5.4), X=a=>x+a*w/20.1, Y=b=>y+b*h/5.4;
  const txt=(str,a,b,width,height,pt,opts={})=>slide.addText(str,{
    x:X(a),y:Y(b),w:width*w/20.1,h:height*h/5.4,fontFace:'Arial',
    fontSize:pt*scale,color:ink,margin:0,valign:'mid',...opts
  });
  const hline=(left,right,top,color,pt,dashType)=>slide.addShape(pptx.ShapeType.line,{
    x:X(left),y:Y(top),w:X(right)-X(left),h:0,line:{color,width:pt*scale,dashType}});
  const gains=getGains(), neg=gains.filter(v=>v<0).length, pos=gains.filter(v=>v>0).length;
  const average=gains.reduce((sum,v)=>sum+v,0)/gains.length;
  const left=1.13,right=19.92,step=(right-left)/gains.length;
  const zero=4.00, factor=.015, ordinate=v=>zero-v*factor;
  const edge=left+neg*step;
  txt('Forecast error reduction per recording (MSE × 1,000)',0,0,19.9,.59,35,{bold:true});
  for (const v of [0,50,100,150]) {
    const yy=ordinate(v);
    hline(left,right,yy,v===0?ink:grid,v===0?1.9:.9);
    txt(String(v),0,yy-.22,.94,.44,24,{align:'right',color:muted});
  }
  for (const [i,v] of gains.entries()) {
    const color=v<0?orange:teal;
    slide.addShape(pptx.ShapeType.rect,{objectName:'Recording gain '+v.toFixed(3),
      x:X(left+(i+.12)*step),y:Y(Math.min(ordinate(v),zero)),
      w:.76*step*w/20.1,h:Math.abs(Y(zero)-Y(ordinate(v))),
      fill:{color},line:{transparency:100,color}});
  }
  // Paired mean on the chart’s absolute-error scale, distinct from the headline relative gain.
  hline(left,right,ordinate(average),ink,1.8,'dash');
  txt('Mean gain  +'+average.toFixed(1),left+.12,ordinate(average)-.43,6,.40,24,
    {color:ink,bold:true});
  const groupY=4.42;
  hline(left,edge-.04,groupY,orange,2.2);
  hline(edge+.04,right,groupY,teal,2.2);
  txt(neg+' worsened',left,groupY+.06,3.6,.42,25,{bold:true,color:orange});
  txt(pos+' improved',8.6,groupY+.06,5,.42,25,{bold:true,color:teal,align:'center'});
  txt('One bar per held-out normal recording. Above zero, the learned forecast beat repeating the last pose.',0,4.99,20,.40,24,{color:muted});
};
