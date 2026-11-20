// Native PowerPoint marks, values and guides for the draft03 comparison figure.
const fs = require('node:fs');
const path = require('node:path');

module.exports = function interpretationChart(pptx, slide, x, y, w, h) {
  const csv = path.join(__dirname,'../../../experiments/aaai-worldmodels/docs/draft03/figures/ranking_comparison_data.csv');
  const rows = fs.readFileSync(csv,'utf8').trim().split(/\r?\n/).slice(1).map(line=>line.split(','));
  const navy='1F3864', green='1E6B3A', ink='1A1A1A', gray='4A5568', grid='CDD5DE';
  const scale=Math.min(w/11.9,h/3.9);
  function label(text,u,v,ww,hh,pt,opts={}) {
    slide.addText(text,{x:x+u*w,y:y+v*h,w:ww*w,h:hh*h,margin:0,valign:'mid',
      fontFace:'Arial',fontSize:pt*scale,color:ink,...opts});
  }
  function line(u,v,du,dv,color,width,dashed=false) {
    slide.addShape(pptx.ShapeType.line,{x:x+u*w,y:y+v*h,w:du*w,h:dv*h,
      line:{color,width:width*scale,...(dashed?{dashType:'dash'}:{})}});
  }
  function mark(u,v,shape,color) {
    const d=.21*scale;
    slide.addShape(shape,{x:x+u*w-d/2,y:y+v*h-d/2,w:d,h:d,fill:{color},line:{color,width:.8}});
  }
  const scores=[
    {key:'Forecast + normal reference',label:'Forecast score',shape:pptx.ShapeType.ellipse,color:navy},
    {key:'Future movement only',label:'Low movement',shape:pptx.ShapeType.diamond,color:green}
  ];
  scores.forEach((s,i)=>{
    const u=.04+i*.46;
    mark(u,.095,s.shape,s.color);
    label(s.label,u+.028,.015,.4,.15,27,{bold:true,color:s.color});
  });
  const axis=v=>.285+.665*v;
  line(axis(.5),.19,0,.565,gray,1.6,true);
  ['GAVD','External'].forEach((dataset,i)=>{
    const v=.37+i*.285;
    label(dataset+(i?' [2]':' [1]'),0,v-.075,.265,.15,26,{bold:true});
    line(axis(0),v,axis(1)-axis(0),0,grid,1.6);
    for (const s of scores) {
      const row=rows.find(r=>r[0]===dataset && r[2]===s.key);
      if (!row) throw new Error('Missing AUROC estimate for '+dataset+' / '+s.key);
      const val=Number(row[3]), u=axis(val);
      mark(u,v,s.shape,s.color);
      label(val.toFixed(3),u-.057,v-.164,.114,.133,23,{align:'center',bold:true,color:s.color});
    }
  });
  line(axis(0),.77,axis(1)-axis(0),0,gray,1.5);
  label('AUROC',0,.78,.22,.17,25,{bold:true,color:gray});
  for (const tick of [0,.5,1]) {
    const u=axis(tick);
    line(u,.77,0,.034,gray,1.3);
    label(tick===.5?'0.5 (chance)':String(tick),u-(tick===.5?.13:.04),.823,tick===.5?.26:.08,.15,22,{align:'center',color:gray});
  }
};
