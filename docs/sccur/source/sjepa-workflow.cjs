// Conceptual drawing of Abdelfattah & Alahi, ECCV 2024 (doi:10.1007/978-3-031-73411-3_21).
// The two feature vectors are illustrative. The study's coordinate-masking model is separate.
module.exports = function sjepaWorkflow(p,s,x,y,w,h) {
  const kx=w/32.7, ky=h/3.7, u=Math.min(kx,ky);
  const navy='1F3864', teal='00796D', rust='A34819', ink='253951', subtle='C7D5DB';
  const txt=(str,a,b,ww,hh,size,style={})=>s.addText(str,{x:x+a*kx,y:y+b*ky,w:ww*kx,h:hh*ky,
    fontFace:'Arial',fontSize:size*Math.min(kx,Math.max(.88,ky)),color:ink,valign:'mid',margin:0,...style});
  function line(a,b,c,d,color,weight,dashType) {
    s.addShape(p.ShapeType.line,{x:x+Math.min(a,c)*kx,y:y+Math.min(b,d)*ky,
      w:Math.abs(c-a)*kx,h:Math.abs(d-b)*ky,flipV:(c-a)*(d-b)<0,
      line:{color,width:weight*u,...(dashType?{dashType}:{})}});
  }
  function circle(a,b,r,color,fill) {
    s.addShape(p.ShapeType.ellipse,{x:x+a*kx-r*u,y:y+b*ky-r*u,w:2*r*u,h:2*r*u,
      line:{color,width:2*u},fill:{color:fill}});
  }
  function skeleton(offset,masked) {
    const joint=[[1.15,.86],[.79,1.28],[1.51,1.28],[.55,1.58],[1.86,1.53],[.27,1.79],[2.12,1.37],
      [.86,1.98],[1.43,1.98],[.52,2.54],[1.69,2.47],[.41,3.02],[1.92,2.92],[.15,3.08],[2.19,2.97]];
    const edge=[[0,1],[0,2],[1,2],[1,3],[2,4],[3,5],[4,6],[1,7],[2,8],[7,8],[7,9],[9,11],[11,13],[8,10],[10,12],[12,14]];
    const region=new Set([10,12,14]);
    for(const [a,b] of edge){
      const [ax,ay]=joint[a], [bx,by]=joint[b], leg=region.has(b);
      line(offset+ax,ay,offset+bx,by,leg?(masked?rust:teal):navy,leg&&masked?2.1:3.4,leg&&masked?'dash':null);
    }
    joint.forEach(([ax,ay],i)=>{
      if(i===0){circle(offset+ax,ay-.13,.17,navy,'FFFFFF');return;}
      const c=region.has(i)?(masked?rust:teal):navy;
      circle(offset+ax,ay,region.has(i)?.096:.058,c,masked&&region.has(i)?'FFFFFF':c);
    });
  }
  // Each branch receives the same sequence. A single frame illustrates the masked region.
  skeleton(.08,true);
  txt('The student sees part of the pose.',2.62,.47,7.74,.60,32,{bold:true,color:navy});
  txt('It uses the visible joints to predict a pattern of numbers for the hidden leg.',2.62,1.30,7.42,1.40,28);
  txt('The dashed leg is hidden.',2.62,2.91,7.58,.44,24,{color:rust,bold:true});
  line(10.61,.35,10.61,3.40,subtle,1.3);
  skeleton(10.96,false);
  txt('The teacher sees the whole pose.',13.58,.47,7.72,.60,32,{bold:true,color:navy});
  txt('It uses the full sequence to supply the target pattern for that leg.',13.58,1.30,7.75,1.40,28);
  txt('The green leg is visible.',13.58,2.91,7.78,.44,24,{color:teal,bold:true});
  line(21.63,.35,21.63,3.40,subtle,1.3);
  txt('The student learns to match the teacher.',22.10,.47,10.50,.60,32,{bold:true,color:teal});
  const swatches=[['97B7C9','30567B','C1D3DD','50758E'],['8CB3C4','3A6088','BED1D9','4B728A']];
  ['Predicted','Target'].forEach((label,i)=>{
    txt(label,22.12,1.30+.56*i,2.0,.46,24,{bold:true});
    swatches[i].forEach((c,j)=>s.addShape(p.ShapeType.rect,{x:x+(24.18+j*.56)*kx,y:y+(1.36+.56*i)*ky,
      w:.49*kx,h:.32*ky,line:{transparency:100},fill:{color:c}}));
  });
  txt('Training reduces the gap between the two patterns.',26.75,1.20,5.77,1.30,26);
  txt('The target comes from the walking sequence itself.',22.12,2.91,10.45,.44,24,{color:teal,bold:true});
};
