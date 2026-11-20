// Draft03 coordinate benchmark. All plotted values come from released research artifacts.
const fs = require('node:fs');
const path = require('node:path');

module.exports = function forecastEvidence(p,s,x,y,w,h) {
  const draft=path.join(__dirname,'../../../experiments/aaai-worldmodels/docs/draft03');
  const read=name=>fs.readFileSync(path.join(draft,name),'utf8');
  const source=JSON.parse(read('sequence_weighting_results.json')).recording_weighted_primary;
  const example=read('figures/forecast_example_data.csv').trim().split(/\r?\n/).slice(1).map(r=>r.split(',').map(Number));
  const scores=read('figures/ranking_comparison_data.csv').trim().split(/\r?\n/).slice(1).map(r=>r.split(','));
  const n=read('cohort.csv').trim().split(/\r?\n/).slice(1).reduce((sum,row)=>sum+Number(row.split(',')[3]),0);
  if(example.length!==8 || n!==91 || source.normal_components!==29)throw Error('Unexpected observation population');
  const navy='193B59', ink='233746', teal='00796D', orange='AA461A', grey='536674', light='CBD5DD';
  const sx=w/12,sy=h/13.5,k=Math.min(sx,sy);
  function text(t,a,b,sw,sh,sz,style={}) {
    s.addText(t,{x:x+a*sx,y:y+b*sy,w:sw*sx,h:sh*sy,fontFace:'Calibri',fontSize:sz*k,color:ink,margin:0,valign:'mid',...style});
  }
  function line(x1,y1,x2,y2,col,pt,dash=false) {
    s.addShape(p.ShapeType.line,{x:x+Math.min(x1,x2)*sx,y:y+Math.min(y1,y2)*sy,w:Math.abs(x2-x1)*sx,h:Math.abs(y2-y1)*sy,
      flipV:(x2-x1)*(y2-y1)<0,line:{color:col,width:pt*k,...(dash?{dashType:'dash'}:{})}});
  }
  function point(a,b,r,col) {
    s.addShape(p.ShapeType.ellipse,{x:x+a*sx-r*k,y:y+b*sy-r*k,w:2*r*k,h:2*r*k,fill:{color:col},line:{color:col,width:.6*k}});
  }
  const pct=v=>(v*100).toFixed(1)+'%';
  text('Can posture predict the next step?',0,0,12,.61,37,{bold:true,color:navy});
  text('Could forecast errors help identify unusual gait? We trained a separate model on normal walks to predict eight leg joints from the current posture.',0,.68,12,.98,25);
  text('One held-out walk: left ankle position',0,1.74,12,.54,30,{bold:true,color:navy});
  text('Horizontal position relative to the hips, in torso lengths',0,2.34,12,.48,24,{color:grey});
  const px=t=>.70+9.08*t/.8, py=v=>5.73-(v+.6)*1.51;
  for(const v of [-.5,0,.5,1]) {
    line(.70,py(v),9.78,py(v),light,1);
    text(String(v).replace('-','−'),0,py(v)-.18,.54,.36,21,{align:'right',color:grey});
  }
  const anchor=example[0][3],data=[[0,anchor,anchor,anchor],...example];
  for(const [j,col,pt,dash] of [[3,orange,2.4,true],[2,teal,3.2,false],[1,navy,2.5,false]]) {
    for(let i=1;i<data.length;i++)line(px(data[i-1][0]),py(data[i-1][j]),px(data[i][0]),py(data[i][j]),col,pt,dash);
  }
  data.forEach(row=>point(px(row[0]),py(row[1]),.064,navy));
  // Stagger the labels and connect them to the actual close-spaced endpoints.
  [[2,'Forecast',teal,4.00],[3,'Copy last pose',orange,4.52],[1,'Observed',navy,5.04]].forEach(([j,label,color,labelY])=>{
    line(9.80,py(data.at(-1)[j]),10.06,labelY+.20,color,1.5,j===3);
    text(label,10.11,labelY,1.88,.40,20,{bold:true,color});
  });
  for(const [t,str] of [[0,'Now'],[.2,'0.2'],[.4,'0.4'],[.6,'0.6'],[.8,'0.8 s']]) {
    text(str,px(t)-.42,5.79,.84,.40,21,{align:'center',color:grey});
  }
  text('Time after the last observed pose',.60,6.20,9.30,.40,22,{align:'center',color:grey});
  text('The forecast missed the peak in this clip. The example was chosen by a fixed rule without using prediction accuracy.',0,6.78,12,.90,26);
  text(pct(source.relative_reduction),0,7.81,3.17,.90,54,{bold:true,color:teal});
  text('lower mean squared error than copying the last pose',3.19,7.74,8.79,.94,29,{bold:true,color:navy});
  text('Across '+source.normal_components+' normal recordings. Paired 95% interval: '+source.relative_ci95.map(pct).join('–')+'.',0,8.78,12,.50,25);
  line(0,9.45,12,9.45,light,1.4);
  text('Does forecast error flag unusual gait?',0,9.64,12,.66,35,{bold:true,color:navy});
  text('A second model used posture and velocity to flag unusually easy or difficult forecasts.',0,10.33,12,.65,25);
  const start=3.76, span=6.05;
  for(const [idx,[key,name,color]] of [['Forecast + normal reference','Forecast score',navy],['Future movement only','Low movement',teal]].entries()) {
    const row=scores.find(r=>r[0]==='GAVD'&&r[2]===key);
    if(!row)throw Error('Missing GAVD score '+key);
    const v=Number(row[3]), b=11.06+idx*.57;
    text(name,0,b,3.50,.42,24,{bold:true,color});
    line(start,b+.21,start+span,b+.21,light,5);
    line(start,b+.21,start+span*v,b+.21,color,5);
    line(start+span/2,b+.04,start+span/2,b+.39,grey,1.8);
    point(start+span*v,b+.21,.065,color);
    text(v.toFixed(3),10.07,b,1.68,.44,28,{bold:true,color,align:'right'});
  }
  text('0',start-.20,12.18,.40,.35,20,{align:'center'});
  text('0.5 = chance',start+span/2-1.0,12.18,2,.35,20,{align:'center'});
  text('1',start+span-.20,12.18,.40,.35,20,{align:'center'});
  text('AUROC measures how often an abnormal video ranks above a normal one. We paired videos within test folds ('+n+' recordings). Both scores need later frames.',0,12.58,12,.92,23,{color:grey});
};
