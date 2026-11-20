// Study A (GAVD3). Teaching illustrations adapted from FAQ Figures 2, 6 and 7.
// The mask and feature patterns illustrate the calculation, not measured joint values.
module.exports=function pipeline(p,s,x,y,w,h){
  const xs=w/11.41,ys=h/14.25,u=Math.min(xs,ys);
  const ink='203A60',teal='077868',rust='A34A21',gray='344D60',light='D5DFE5';
  function txt(text,a,b,ww,hh,pt,more={}){
    s.addText(text,{x:x+a*xs,y:y+b*ys,w:ww*xs,h:hh*ys,margin:0,fontFace:'Arial',fontSize:pt*u,color:gray,valign:'mid',...more});
  }
  function rect(a,b,ww,hh,fill,stroke=fill){
    s.addShape(p.ShapeType.rect,{x:x+a*xs,y:y+b*ys,w:ww*xs,h:hh*ys,fill:{color:fill},line:{color:stroke,width:.9*u}});
  }
  function line(a,b,c,d,color,weight,dash){
    s.addShape(p.ShapeType.line,{x:x+Math.min(a,c)*xs,y:y+Math.min(b,d)*ys,w:Math.abs(c-a)*xs,h:Math.abs(d-b)*ys,
      flipV:(c-a)*(d-b)<0,line:{color,width:weight*u,...(dash?{dashType:'dash'}:{})}});
  }
  function dot(a,b,r,stroke,fill){
    s.addShape(p.ShapeType.ellipse,{x:x+a*xs-r*u,y:y+b*ys-r*u,w:2*r*u,h:2*r*u,line:{color:stroke,width:1.7*u},fill:{color:fill}});
  }
  function step(n,title,top){
    if(n>1)line(.02,top-.19,11.39,top-.19,light,1.4);
    rect(.02,top,.51,.51,ink);
    txt(String(n),.02,top,.51,.51,23,{bold:true,color:'FFFFFF',align:'center'});
    txt(title,.74,top-.015,10.63,.56,30,{bold:true,color:ink});
  }
  step(1,'Group the joint tracks into tokens',.03);
  txt('Each clip contains 33 landmarks across 64 frames. Four frames of one landmark form one token.',.02,.70,11.32,.93,26);
  const trace=[.13,.03,-.12,-.10,.00,.12,.19,.13,-.01,-.12,-.14,-.03];
  trace.forEach((v,i)=>{
    const xx=.23+i*.47+(Math.floor(i/4)*.14);
    if(i%4)line(xx-.47,2.07+trace[i-1],xx,2.07+v,teal,2.5);
    dot(xx,2.07+v,.071,teal,teal);
    if(i%4===0){line(xx-.10,2.45,xx+1.51,2.45,ink,1.7);line(xx-.10,2.35,xx-.10,2.45,ink,1.7);line(xx+1.51,2.35,xx+1.51,2.45,ink,1.7);}
  });
  txt('…',5.91,1.78,.6,.51,33,{bold:true,color:teal});
  txt('16 groups of frames × 33 joints\n= 528 tokens per clip',6.76,1.69,4.62,.93,25,{bold:true,color:ink});
  step(2,'Hide selected joints from the student',3.00);
  txt('Ten landmarks can be masked. We aim to hide 60% of their valid tokens. The two views show the same knee.',.02,3.69,11.32,.98,26);
  const pt=[[1.02,.21],[.69,.62],[1.31,.62],[.45,1.00],[1.58,.94],[.18,1.27],[1.89,.71],
    [.75,1.28],[1.20,1.28],[.43,1.83],[1.77,1.60],[.29,2.38],[1.43,2.11],[.05,2.42],[1.75,2.23]];
  const edges=[[0,1],[0,2],[1,2],[1,3],[2,4],[3,5],[4,6],[1,7],[2,8],[7,8],[7,9],[9,11],[11,13],[8,10],[10,12],[12,14]];
  function pose(a,b,masked){
    const color=masked?rust:teal;
    for(const [i,j] of edges){
      const focus=i===10||j===10;
      line(a+pt[i][0],b+pt[i][1],a+pt[j][0],b+pt[j][1],focus?color:ink,focus?3.9:3.1,masked&&focus);
    }
    pt.forEach(([dx,dy],i)=>{
      if(i===0)dot(a+dx,b+dy-.06,.17,ink,'FFFFFF');
      else if(i===10)dot(a+dx,b+dy,.128,color,masked?'FFFFFF':color);
      else dot(a+dx,b+dy,.050,ink,ink);
    });
  }
  pose(.34,4.98,true);pose(6.03,4.98,false);
  txt('Student',2.53,5.30,2.75,.53,28,{color:rust,bold:true});
  txt('The knee is hidden here.',2.53,5.94,2.68,1.19,25);
  line(5.65,4.91,5.65,7.47,light,1.1);
  txt('Teacher',8.23,5.30,2.98,.53,28,{color:teal,bold:true});
  txt('All observed joints are visible.',8.23,5.94,3.12,1.19,25);
  step(3,'Predict features for the hidden joints',7.86);
  txt('The student predicts a pattern of numbers for each hidden token. The teacher supplies the target.',.02,8.53,11.31,.99,26);
  const colors=[['4C7492','BCD0D7','7495A4','2D5A78','88A5B4','AAC4CB'],['52798E','B3CDD5','789BAB','35667C','93ADB8','A3BFC8']];
  ['Prediction','Target'].forEach((label,i)=>{
    txt(label,.04,9.65+.52*i,2.18,.44,23,{bold:true,color:i?teal:ink});
    colors[i].forEach((c,j)=>rect(2.17+j*.49,9.71+.52*i,.41,.30,c));
  });
  txt('Training matches these patterns only at hidden tokens.',5.39,9.56,5.92,1.04,25,{bold:true,color:teal});
  txt('The teacher updates by averaging the student encoder’s weights over time.',.02,10.85,11.35,.79,24);
  step(4,'Keep the encoder fixed and test its features',12.00);
  txt('The teacher now reads a full clip without changing its weights. We summarize its features and train a classifier to predict one of five gait labels.',.02,12.68,11.33,.92,25);
  txt('Full clip',.04,13.79,1.89,.41,23,{color:ink,bold:true});
  txt('→',1.93,13.73,.67,.50,30,{align:'center',color:teal});
  txt('384 numbers',2.65,13.79,3.17,.41,23,{color:ink,bold:true,align:'center'});
  txt('→',5.85,13.73,.67,.50,30,{align:'center',color:teal});
  txt('Gait classifier',6.73,13.79,4.54,.41,23,{color:ink,bold:true});
};
