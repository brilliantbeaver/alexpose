// Worldmodels notebooks 00–07, corrected 16-frame binary experiment.
// Protocol: docs/revision_protocol.md; model.py, binary_clean.py, evaluation.aggregate_sources.
module.exports=function videoWorkflow(p,s,x,y,w,h){
  const sx=w/21.64,sy=h/5.15,u=Math.min(sx,sy), navy='1F3864', teal='077868', blue='2358AB',muted='40556B',lineGray='C7D5DE';
  function txt(text,a,b,ww,hh,pt,opt={}){s.addText(text,{x:x+a*sx,y:y+b*sy,w:ww*sx,h:hh*sy,margin:0,fontFace:'Calibri',fontSize:pt*u,color:muted,valign:'mid',...opt});}
  function rect(a,b,ww,hh,fill,stroke=fill){s.addShape(p.ShapeType.rect,{x:x+a*sx,y:y+b*sy,w:ww*sx,h:hh*sy,fill:{color:fill},line:{color:stroke,width:u}});}
  function edge(a,b,c,d,color,weight){
    s.addShape(p.ShapeType.line,{x:x+Math.min(a,c)*sx,y:y+Math.min(b,d)*sy,w:Math.abs(c-a)*sx,h:Math.abs(d-b)*sy,flipV:(c-a)*(d-b)<0,line:{color,width:weight*u}});
  }
  function dot(a,b,r,color,fill=color,name='Landmark'){s.addShape(p.ShapeType.ellipse,{x:x+a*sx-r*u,y:y+b*sy-r*u,w:2*r*u,h:2*r*u,objectName:name,fill:{color:fill},line:{color,width:1*u}});}
  function walker(a,b,t){
    const v=[[.49,.12],[.47,.33],[.41,.60],[.21,.44],[.72,.41],[.21+t,.76],[.68-t,.78],[.06+t,.95],[.78-t,.95]];
    [[0,1],[1,2],[1,3],[1,4],[2,5],[2,6],[5,7],[6,8]].forEach(([i,j])=>edge(a+v[i][0],b+v[i][1],a+v[j][0],b+v[j][1],navy,2.3));
    v.forEach(([cx,cy],i)=>dot(a+cx,b+cy,i===0?.088:.041,i>4?teal:navy));
  }
  // BlazePose index order. Study B uses 11–16 and 23–32, documented by JOINT_IDS.
  function fullPose(){
    const q=[[2.8,1.02],[2.91,.84],[3.03,.80],[3.15,.84],[2.69,.84],[2.57,.80],[2.45,.84],[3.35,1],[2.25,1],[2.94,1.19],[2.66,1.19],[3.52,1.76],[2.08,1.76],[4.07,2.66],[1.56,2.68],[4.29,3.47],[1.30,3.48],[4.61,3.76],[.97,3.78],[4.43,3.85],[1.13,3.86],[4.19,3.75],[1.40,3.77],[3.30,3.72],[2.29,3.72],[3.52,5.11],[2.03,5.16],[3.62,6.31],[1.81,6.29],[3.52,6.55],[1.93,6.53],[3.97,6.65],[1.42,6.68]];
    const points=q.map(([a,b])=>[3.09+(a-.97)*.35,.89+(b-.80)*.315]);
    const links=[[0,1],[1,2],[2,3],[3,7],[0,4],[4,5],[5,6],[6,8],[9,10],[11,12],[11,13],[13,15],[12,14],[14,16],[15,17],[15,19],[15,21],[17,19],[16,18],[16,20],[16,22],[18,20],[11,23],[12,24],[23,24],[23,25],[24,26],[25,27],[26,28],[27,29],[29,31],[27,31],[28,30],[30,32],[28,32]];
    const used=i=>(i>=11&&i<=16)||i>=23;
    for(const [i,j] of links)edge(...points[i],...points[j],used(i)&&used(j)?navy:lineGray,used(i)&&used(j)?1.9:1);
    points.forEach(([a,b],i)=>dot(a,b,used(i)?.047:.022,used(i)?teal:'687784',used(i)?teal:'FFFFFF','BlazePose '+i+(used(i)?' retained':' omitted')));
  }
  function arrow(a,b){txt('→',a,b,.61,.56,29,{align:'center',color:teal,bold:true});}
  txt('Training on known recordings',0,.02,9,.51,29,{color:navy,bold:true});
  rect(.02,.90,2.14,1.28,'F8FAFB',navy);
  for(let i=0;i<2;i++){
    const a=.05+i*1.07;
    walker(a+.025,1.03,i*.13);
    if(i)edge(a-.025,.95,a-.025,2.13,lineGray,1);
    for(const b of [.95,2.04])for(let j=0;j<4;j++)rect(a+.04+j*.24,b,.13,.06,navy);
  }
  txt('Walking video',.03,2.26,2.12,.42,23,{align:'center',bold:true,color:navy});
  arrow(2.27,1.38);
  fullPose();
  txt('Extract 33 landmarks',4.56,.71,3.89,.47,24,{bold:true,color:navy});
  dot(4.68,1.43,.065,teal);
  txt('16 used',4.86,1.21,3.40,.44,23,{bold:true,color:teal});
  dot(4.68,1.87,.060,'687784','FFFFFF');
  txt('17 omitted',4.86,1.65,3.40,.44,23);
  txt('Center at hips and scale by torso size.',4.56,2.11,3.86,.73,23);
  arrow(8.47,1.34);
  txt('Learn from hidden joints',9.20,.73,5.58,.45,26,{bold:true,color:navy});
  txt('The student sees masked poses and predicts features from a teacher that sees the full pose. Start with normal walks, then mix gait types.',9.20,1.29,5.49,1.49,24);
  arrow(14.86,1.34);
  txt('Fit a binary classifier',15.59,.73,6.00,.45,26,{bold:true,color:navy});
  txt('Freeze the encoder. Use labeled training windows to fit a linear classifier on its features.',15.59,1.29,5.94,1.49,24);
  edge(0,2.96,21.64,2.96,lineGray,1.7);
  txt('Testing on a new recording',0,3.12,10,.54,29,{color:teal,bold:true});
  const blocks=[
    [0,3.72,3.42,'New video','Use the 16 joints without masking.'],
    [4.24,3.72,4.18,'Same pose processing','Build 16-frame windows.'],
    [9.20,3.72,5.52,'Frozen encoder + classifier','Score each window.'],
    [15.59,3.72,6.00,'One label per recording','Average scores by sequence, then by video: normal or abnormal.']
  ];
  for(const [a,b,ww,head,body] of blocks){txt(head,a,b,ww,.45,25,{color:navy,bold:true});txt(body,a,b+.55,ww,.82,23);}
  [3.52,8.47,14.86].forEach(a=>arrow(a,3.97));
};
