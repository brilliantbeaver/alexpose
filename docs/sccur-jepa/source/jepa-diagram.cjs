// Adapted from gait-with-jepa.docx, Fig. 2 (two encoders) and Fig. 6 (GAVD3 token/mask policy).
// Source: experiments/penny/gavd3/03_neurologic_keypoint_masking.ipynb and 04_pretrain_sjepa_on_normal.ipynb.
function jepaDiagram(p,s,x,y,w,h){
  const fx=w/12.8,fy=h/6.5,u=Math.min(fx,fy);
  const ink='1F3864',teal='00796D',rust='A34819',gray='3D5061';
  function txt(str,a,b,width,height,size,style={}){
    s.addText(str,{x:x+a*fx,y:y+b*fy,w:width*fx,h:height*fy,fontFace:'Calibri',fontSize:size*u,
      color:ink,margin:0,valign:'mid',...style});
  }
  function line(a,b,c,d,color,weight,dashType){
    s.addShape(p.ShapeType.line,{x:x+Math.min(a,c)*fx,y:y+Math.min(b,d)*fy,w:Math.abs(c-a)*fx,
      h:Math.abs(d-b)*fy,flipV:(c-a)*(d-b)<0,line:{color,width:weight*u,...(dashType?{dashType}:{})}});
  }
  function node(a,b,r,stroke,fill){
    s.addShape(p.ShapeType.ellipse,{x:x+a*fx-r*u,y:y+b*fy-r*u,w:2*r*u,h:2*r*u,
      line:{color:stroke,width:2*u},fill:{color:fill}});
  }
  txt('Each clip supplies a prediction task',0,0,12.8,.61,34,{bold:true});
  txt('We group each landmark into four-frame pieces. We aim to hide 60% of the valid pieces at ten selected joints.',0,.72,12.8,.94,26,{color:gray});
  // A schematic frame of the same walker in both branches. The withheld knee and ankle are examples.
  const joints=[[1.21,.30],[.87,.77],[1.48,.77],[.55,1.19],[1.78,1.08],[.24,1.50],[2.12,.86],
    [.90,1.58],[1.39,1.57],[.63,2.13],[1.92,1.92],[.44,2.67],[1.51,2.50],[.12,2.72],[1.86,2.62]];
  const edges=[[0,1],[0,2],[1,2],[1,3],[2,4],[3,5],[4,6],[1,7],[2,8],[7,8],[7,9],[9,11],[11,13],[8,10],[10,12],[12,14]];
  function pose(left,top,masked){
    const selected=i=>i===10||i===12;
    for(const [a,b] of edges){
      const target=selected(a)||selected(b);
      line(left+joints[a][0],top+joints[a][1],left+joints[b][0],top+joints[b][1],
        target?(masked?rust:teal):ink,masked&&target?2.7:3.6,masked&&target?'dash':null);
    }
    joints.forEach(([dx,dy],i)=>{
      if(i===0){node(left+dx,top+dy,.20,ink,'FFFFFF');return;}
      const target=selected(i),color=target?(masked?rust:teal):ink;
      node(left+dx,top+dy,target?.111:.059,color,target&&masked?'FFFFFF':color);
    });
  }
  txt('Student · partial view',0,1.76,6.28,.48,27,{color:rust,bold:true});
  txt('Teacher · full view',6.62,1.76,6.14,.48,27,{color:teal,bold:true});
  pose(.02,2.41,true);
  pose(6.65,2.41,false);
  txt('The student sees the remaining joints and predicts features for the hidden ones.',2.46,2.62,3.72,2.16,25,{color:gray});
  txt('The teacher sees the full clip and supplies target features for those same joints.',9.08,2.62,3.65,2.16,25,{color:gray});
  line(6.40,1.85,6.40,5.12,'CDD9E2',1.2);
  line(0,5.40,12.8,5.40,'AABCC9',1.3);
  txt('How it learns',0,5.61,2.24,.47,25,{bold:true,color:teal});
  txt('Match the teacher’s features at hidden positions. The teacher slowly averages the student encoder’s weights.',2.45,5.52,10.34,.95,25,{color:gray});
}
module.exports = jepaDiagram;
