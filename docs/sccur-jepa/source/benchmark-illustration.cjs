// Notebook 20–26 coordinate benchmark, distinct from the JEPA classification studies.
// Reconstruct the released ankle example (draft03 Fig. 2) and original recording-weighted estimate.
const fs=require('node:fs'),path=require('node:path');
module.exports=function benchmarkIllustration(p,s,x,y,w,h){
  const draft=path.join(__dirname,'../../../experiments/aaai-worldmodels/docs/draft03');
  const read=name=>fs.readFileSync(path.join(draft,name),'utf8');
  const rows=read('figures/forecast_example_data.csv').trim().split(/\r?\n/).slice(1).map(row=>row.split(',').map(Number));
  const primary=JSON.parse(read('sequence_weighting_results.json')).recording_weighted_primary;
  if(rows.length!==8||rows.some(row=>row.length!==4||row.some(value=>!Number.isFinite(value))))throw Error('Incomplete released ankle example');
  const sx=w/11.41,sy=h/7.15,u=Math.min(sx,sy);
  const navy='1F3864',teal='007868',blue='255DA3',gray='435361',grid='CCD7DF';
  function txt(t,a,b,ww,hh,pt,o={}){
    s.addText(t,{x:x+a*sx,y:y+b*sy,w:ww*sx,h:hh*sy,fontFace:'Calibri',fontSize:pt*u,margin:0,valign:'mid',color:gray,...o});
  }
  function line(a,b,c,d,color,width,dash){
    s.addShape(p.ShapeType.line,{x:x+Math.min(a,c)*sx,y:y+Math.min(b,d)*sy,w:Math.abs(c-a)*sx,h:Math.abs(d-b)*sy,
      flipV:(c-a)*(d-b)<0,line:{color,width:width*u,...(dash?{dashType:'dash'}:{})}});
  }
  function node(a,b,r,col){
    s.addShape(p.ShapeType.ellipse,{x:x+a*sx-r*u,y:y+b*sy-r*u,w:2*r*u,h:2*r*u,fill:{color:col},line:{color:col,width:u}});
  }
  txt('Can we predict where a foot will go?',0,0,11.38,.62,34,{bold:true,color:navy});
  txt('A model learned from normal walking. It used the latest posture to predict eight leg joints over the next 0.8 seconds.',0,.69,11.35,.90,26);
  txt('One clip: horizontal ankle position in torso lengths',0,1.71,11.38,.46,25,{bold:true,color:navy});
  function key(a,str,col,dash,mark){
    line(a,2.43,a+.52,2.43,col,2.6,dash);if(mark)node(a+.26,2.43,.051,col);
    txt(str,a+.62,2.23,3.06,.40,22,{color:col,bold:true});
  }
  key(.02,'Observed',navy,false,true);key(3.38,'Learned forecast',teal,false,false);key(7.76,'Copy last pose',blue,true,false);
  const px=t=>.86+10.16*t/.8,py=v=>4.10-(v+.6)*.75;
  for(const tick of [-.5,0,.5,1]){
    line(.86,py(tick),11.02,py(tick),grid,1);
    txt(String(tick).replace('-','−'),0,py(tick)-.13,.70,.26,18,{align:'right'});
  }
  const anchor=rows[0][3],data=[[0,anchor,anchor,anchor],...rows];
  for(const [j,col,dash] of [[3,blue,true],[2,teal,false],[1,navy,false]]){
    for(let i=1;i<data.length;i++)line(px(data[i-1][0]),py(data[i-1][j]),px(data[i][0]),py(data[i][j]),col,j===1?1.6:3,dash);
  }
  data.forEach(row=>node(px(row[0]),py(row[1]),.055,navy));
  [[0,'Now'],[.4,'+0.4 s'],[.8,'+0.8 s']].forEach(([t,label])=>txt(label,Math.min(10.52,px(t)-.42),4.19,.88,.35,20,{align:'center'}));
  line(0,4.71,11.41,4.71,grid,1.4);
  txt((100*primary.relative_reduction).toFixed(1)+'%',0,4.85,2.58,.77,51,{bold:true,color:teal});
  txt('lower mean squared error than copying the last pose, across 29 held-out normal recordings.',2.83,4.81,8.49,.96,25,{bold:true,color:navy});
  line(0,5.90,11.41,5.90,grid,1.4);
  txt('Does error flag atypical gait?',0,6.02,11.38,.44,27,{bold:true,color:navy});
  txt('A separate score flagged unusually easy or hard forecasts. Low observed movement ranked abnormal videos better on GAVD.',0,6.48,11.38,.65,23);
};
