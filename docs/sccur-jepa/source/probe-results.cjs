// Released direct binary classifiers. Every interval compares predictions for the same test sources.
const fs = require('node:fs');
const path = require('node:path');

module.exports = function probeResults(p, s, x, y, w, h) {
  const source = path.join(__dirname, '../../../experiments/aaai-worldmodels/docs/research_results.json');
  const study = JSON.parse(fs.readFileSync(source,'utf8')).binary;
  const navy='1F3864', teal='237C4B', orange='AC551D', neutral='536777', light='DCE4E7';
  const scale=Math.min(w/18.1,h/5.65);
  const rows=[
    ['rf','Random forest',neutral],
    ['untrained_probe','Random encoder + fitted classifier',neutral],
    ['jepa_normal','Normal-only S-JEPA',navy],
    ['jepa_mixed','Mixed S-JEPA (primary)',teal],
    ['jepa_adapted','Staged S-JEPA (secondary)',orange]
  ];
  function text(str,u,v,ww,hh,pt,opts={}) {
    s.addText(str,{x:x+u*w,y:y+v*h,w:ww*w,h:hh*h,fontFace:'Calibri',fontSize:pt*scale,
      color:navy,margin:0,valign:'mid',...opts});
  }
  function line(u,v,du,dv,color,width,dash=false) {
    s.addShape(p.ShapeType.line,{x:x+u*w,y:y+v*h,w:du*w,h:dv*h,
      line:{color,width:width*scale,...(dash?{dashType:'dash'}:{})}});
  }
  const axis = value => .59+.37*(value+40)/80;
  text('Classifier',0,0,.324,.09,29,{bold:true});
  text('Balanced accuracy',.32,0,.236,.09,26,{align:'center',bold:true});
  text('Difference vs. random forest',.55,0,.45,.09,27,{align:'center',bold:true});
  line(0,.105,1,0,light,1.5);
  line(axis(0),.123,0,.583,neutral,1.8,true);
  rows.forEach(([id,label,color],index)=>{
    const cy=.18+index*.122;
    const report=study.test_reports[id];
    if(report.n_sources!==17) throw Error('Unexpected source count for '+id);
    text(label,0,cy-.049,.331,.098,26,{bold:id==='jepa_mixed',color});
    text((100*report.binary_balanced_accuracy).toFixed(1)+'%',.34,cy-.048,.193,.096,31,{bold:true,align:'center',color});
    if(id==='rf') {
      text('Reference',.58,cy-.049,.15,.098,24,{color:neutral});
      return;
    }
    const c=study.paired_against_rf[id], [lo,hi]=c.ci95.map(v=>100*v);
    const point=axis(100*c.estimate), diameter=.19*scale;
    line(axis(lo),cy,axis(hi)-axis(lo),0,color,3.2);
    [lo,hi].forEach(endpoint=>line(axis(endpoint),cy-.021,0,.042,color,2.4));
    s.addShape(p.ShapeType.ellipse,{x:x+point*w-diameter/2,y:y+cy*h-diameter/2,w:diameter,h:diameter,fill:{color},line:{color}});
  });
  line(axis(-40),.735,axis(40)-axis(-40),0,neutral,1.5);
  for(const tick of [-40,0,40]){
    const u=axis(tick);
    text(tick>0?'+'+tick:String(tick).replace('-','−'),u-.04,.75,.08,.076,22,{align:'center'});
  }
  text('Percentage points',.565,.837,.423,.07,23,{align:'center',color:neutral});
  text('Dots show paired differences. Lines show 95% bootstrap intervals across the 17 test recordings.',0,.922,1,.075,23,{color:neutral});
};
