// Translate within-fold AUROC to a frequency scale. The bars represent expectations.
const fs = require('node:fs');
const path = require('node:path');

module.exports = function interpretationPairs(pptx, slide, x, y, w, h) {
  const file=path.join(__dirname,'../../../experiments/aaai-worldmodels/docs/draft03/figures/ranking_comparison_data.csv');
  const csv=fs.readFileSync(file,'utf8').trim().split(/\r?\n/).slice(1).map(row=>row.split(','));
  const scale=Math.min(w/11.9,h/3.6), ink='192C45', navy='1F3864';
  const barStart=.313, barWidth=.453, midpoint=barStart+barWidth/2;
  function text(val,u,v,ww,hh,size,style={}) {
    slide.addText(val,{x:x+u*w,y:y+v*h,w:ww*w,h:hh*h,fontFace:'Arial',margin:0,
      fontSize:size*scale,color:ink,valign:'mid',...style});
  }
  function bar(u,v,ww,hh,color) {
    slide.addShape(pptx.ShapeType.rect,{x:x+u*w,y:y+v*h,w:ww*w,h:hh*h,
      fill:{color},line:{transparency:100}});
  }
  text('Out of 100 normal–abnormal recording pairs',0,0,1,.165,32,{bold:true,color:navy});
  text('Expected pairs in which the abnormal recording scores higher',0,.185,1,.135,24);
  const arms = [
    ['Forecast + normal reference','Forecast score','2358AB'],
    ['Future movement only','Low movement','00796D']
  ];
  arms.forEach(([quantity,name,color],i)=>{
    const row=csv.find(r=>r[0]==='GAVD' && r[2]===quantity);
    if (!row) throw Error('Missing GAVD within-fold estimate: '+quantity);
    const estimate=Number(row[3]), top=.41+i*.245;
    text(name,0,top-.03,.303,.155,28,{bold:true,color});
    bar(barStart,top,barWidth,.105,'E1E7ED');
    bar(barStart,top,barWidth*estimate,.105,color);
    for(let step=1;step<10;step++) {
      const u=barStart+barWidth*step/10;
      slide.addShape(pptx.ShapeType.line,{x:x+u*w,y:y+(top-(step===5?.022:0))*h,
        w:0,h:(step===5?.15:.105)*h,line:{color:step===5?ink:'FFFFFF',width:step===5?2:1}});
    }
    text('≈'+Math.round(100*estimate)+' / 100',.774,top-.038,.226,.16,30,{align:'right',bold:true,color});
  });
  text('0',barStart-.035,.835,.07,.13,22,{align:'center'});
  text('50 = chance',midpoint-.13,.835,.26,.13,22,{bold:true,align:'center'});
  text('100',barStart+barWidth-.05,.835,.10,.13,22,{align:'center'});
};
