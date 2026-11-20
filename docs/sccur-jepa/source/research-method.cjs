// Both classification studies, with the FAQ's Study A patch masking as the training illustration.
// Landmark coordinates are schematic in BlazePose index order. Study B indices: worldmodels/data.py.
module.exports = function researchMethod(p,s,x,y,w,h){
  const sx=w/12.1,sy=h/13.05,u=Math.min(sx,sy);
  const navy='1F3864',teal='00796D',ink='193449',mid='486072',gray='728694',grid='D0DDE1',rust='A34819';
  function text(str,a,b,ww,hh,pt,opts={}){
    s.addText(str,{x:x+a*sx,y:y+b*sy,w:ww*sx,h:hh*sy,margin:0,fontFace:'Calibri',fontSize:pt*u,
      color:ink,valign:'mid',...opts});
  }
  function line(a,b,c,d,color,width,dash=false){
    s.addShape(p.ShapeType.line,{x:x+Math.min(a,c)*sx,y:y+Math.min(b,d)*sy,w:Math.abs(c-a)*sx,
      h:Math.abs(d-b)*sy,flipV:(c-a)*(d-b)<0,line:{color,width:width*u,...(dash?{dashType:'dash'}:{})}});
  }
  function box(a,b,ww,hh,fill,stroke=fill){
    s.addShape(p.ShapeType.rect,{x:x+a*sx,y:y+b*sy,w:ww*sx,h:hh*sy,line:{color:stroke,width:1.25*u},fill:{color:fill}});
  }
  function dot(a,b,r,fill,stroke=fill,name='Point'){
    s.addShape(p.ShapeType.ellipse,{x:x+a*sx-r*u,y:y+b*sy-r*u,w:2*r*u,h:2*r*u,objectName:name,fill:{color:fill},line:{color:stroke,width:1.2*u}});
  }
  function stage(i,title,top){
    if(i>1)line(0,top-.18,12.1,top-.18,grid,1.5);
    text(String(i).padStart(2,'0'),0,top,.88,.59,31,{color:teal,bold:true});
    text(title,1,top,11.04,.59,31,{bold:true,color:navy});
  }
  function arrow(a,b,c,color=teal){
    line(a,b,c-.05,b,color,2.6);line(c-.22,b-.12,c-.05,b,color,2.6);line(c-.22,b+.12,c-.05,b,color,2.6);
  }
  stage(1,'Turn a walking video into joint tracks',.02);
  // Film frames show simplified silhouettes; the center drawing has the full 33 landmarks.
  box(.06,1.61,2.65,1.30,'F7FAFC',navy);
  for(let j=0;j<2;j++){
    const a=.13+j*1.26,t=j*.12,b=1.70;
    const v=[[.58,.12],[.53,.33],[.49,.56],[.28,.50],[.83,.39],[.25+t,.78],[.77-t,.78],[.13+t,1.02],[.91-t,1.02]];
    [[0,1],[1,2],[1,3],[1,4],[2,5],[2,6],[5,7],[6,8]].forEach(([i,k])=>line(a+v[i][0],b+v[i][1],a+v[k][0],b+v[k][1],navy,2.5));
    dot(a+v[0][0],b+v[0][1],.095,'FFFFFF',navy);
    if(j)line(a-.01,1.67,a-.01,2.84,grid,1.4);
  }
  text('Video frames',0,3.10,2.71,.44,24,{align:'center',bold:true,color:navy});
  arrow(2.85,2.34,3.51);
  const q=[[2.8,1.02],[2.91,.84],[3.03,.80],[3.15,.84],[2.69,.84],[2.57,.80],[2.45,.84],[3.35,1],[2.25,1],[2.94,1.19],[2.66,1.19],[3.52,1.76],[2.08,1.76],[4.07,2.66],[1.56,2.68],[4.29,3.47],[1.30,3.48],[4.61,3.76],[.97,3.78],[4.43,3.85],[1.13,3.86],[4.19,3.75],[1.40,3.77],[3.30,3.72],[2.29,3.72],[3.52,5.11],[2.03,5.16],[3.62,6.31],[1.81,6.29],[3.52,6.55],[1.93,6.53],[3.97,6.65],[1.42,6.68]];
  const points=q.map(([a,b])=>[3.62+(a-.97)*.61,.88+(b-.80)*.54]);
  const edges=[[0,1],[1,2],[2,3],[3,7],[0,4],[4,5],[5,6],[6,8],[9,10],[11,12],[11,13],[13,15],[12,14],[14,16],[15,17],[15,19],[15,21],[17,19],[16,18],[16,20],[16,22],[18,20],[11,23],[12,24],[23,24],[23,25],[24,26],[25,27],[26,28],[27,29],[29,31],[27,31],[28,30],[30,32],[28,32]];
  const used=i=>(i>=11&&i<=16)||i>=23;
  s.addShape(p.ShapeType.ellipse,{x:x+4.35*sx,y:y+.68*sy,w:.70*sx,h:.68*sy,fill:{color:'FFFFFF',transparency:100},line:{color:grid,width:1.2*u}});
  line(4.60,1.32,4.54,1.41,grid,1.2);line(4.84,1.32,4.92,1.41,grid,1.2);
  edges.forEach(([a,b])=>line(...points[a],...points[b],used(a)&&used(b)?navy:gray,used(a)&&used(b)?2.6:1));
  points.forEach(([a,b],i)=>dot(a,b,used(i)?.068:.038,used(i)?teal:'FFFFFF',used(i)?teal:gray,'BlazePose landmark '+i));
  text('Study A',6.26,.82,5.69,.46,26,{bold:true,color:navy});
  text('33 landmarks tracked across 64 frames per clip.',6.26,1.36,5.71,.96,26);
  text('Study B',6.26,2.48,5.69,.46,26,{bold:true,color:teal});
  text('16 landmarks tracked in 16-frame windows.',6.26,3.00,5.72,.88,26);
  dot(3.69,4.48,.067,teal);text('Used in B',3.88,4.28,2.09,.41,22);
  dot(6.29,4.48,.057,'FFFFFF',gray);text('Other landmarks',6.50,4.28,5.24,.41,22);
  stage(2,'Learn by predicting hidden observations',4.98);
  text('Study A example: one knee across four time patches',0,5.68,12.02,.48,25,{bold:true});
  text('Student',.04,6.48,2.02,.43,26,{bold:true,color:navy});
  text('Teacher',.04,7.42,2.02,.43,26,{bold:true,color:teal});
  for(let row=0;row<2;row++){
    for(let j=0;j<4;j++){
      const a=2.03+j*.88,b=6.41+row*.92,hidden=row===0&&j===2;
      box(a,b,.69,.58,hidden?'FFFFFF':j===2?'DBF0E8':'ECF2F6',hidden?rust:j===2?teal:grid);
      if(hidden){text('?',a,b,.69,.58,27,{align:'center',bold:true,color:rust});}
      else for(let k=0;k<3;k++) line(a+.10+k*.15,b+.29+Math.sin(j+k)*.105,a+.25+k*.15,b+.29+Math.sin(j+k+1)*.105,j===2?teal:navy,2);
    }
  }
  arrow(5.63,6.70,6.30);
  arrow(5.63,7.62,6.30);
  text('Predict the hidden patch’s features.',6.46,6.28,5.54,.80,25);
  text('Read the full clip to supply its target.',6.46,7.17,5.54,.87,25);
  text('Study A used 12 normal clips from one video. The teacher supplied the targets for the hidden joints.',.04,8.29,11.98,1.01,25);
  stage(3,'Freeze the encoder and fit a classifier',9.62);
  text('The classifier learns from labeled examples. For a new clip, the fixed encoder reads the observed study joints.',.03,10.32,12.02,.92,25);
  text('A',.02,11.43,.49,.46,26,{bold:true,color:navy});
  text('384 features per clip',.62,11.38,3.86,.61,24,{bold:true});arrow(4.52,11.65,5.04);
  text('Random forest',5.15,11.38,2.76,.61,24,{bold:true});arrow(8.03,11.65,8.53);
  text('Five gait labels',8.68,11.38,3.38,.61,24,{bold:true,color:navy});
  text('B',.02,12.18,.49,.46,26,{bold:true,color:teal});
  text('Features per window',.62,12.12,3.88,.59,24,{bold:true});arrow(4.52,12.40,5.04);
  text('Linear classifier',5.15,12.12,2.81,.59,24,{bold:true});arrow(8.03,12.40,8.53);
  text('Normal or abnormal',8.68,12.12,3.40,.59,23,{bold:true,color:teal});
  text('Study B averages scores first by sequence, then by recording.',.62,12.63,11.45,.39,22,{color:mid});
};
