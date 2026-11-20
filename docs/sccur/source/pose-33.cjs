// Schematic in BlazePose index order, viewer-facing (subject's left is on the right).
// Study subset: worldmodels/data.py JOINT_IDS; targets: v2_models.py TARGET_IDS.
module.exports = function drawPose33(pptx, slide, x, y, w, h) {
  const navy = '173B57', green = '1E6B3A', gray = '77838B', pale = 'CBD3D6';
  const u = Math.min(w / 11.65, h / 7.3);
  const X = a => x + a*w/11.65, Y = b => y + b*h/7.3;
  function edge(a,b,color,width) {
    slide.addShape(pptx.ShapeType.line, {
      x:X(Math.min(a[0],b[0])), y:Y(Math.min(a[1],b[1])),
      w:Math.abs(X(a[0])-X(b[0])), h:Math.abs(Y(a[1])-Y(b[1])),
      flipV:(a[0]-b[0])*(a[1]-b[1])<0, line:{color,width:width*u}
    });
  }
  function mark(a,r,fill,stroke,name) {
    slide.addShape(pptx.ShapeType.ellipse,{
      x:X(a[0])-r*u, y:Y(a[1])-r*u, w:r*2*u, h:r*2*u,
      objectName:name, fill:{color:fill},line:{color:stroke,width:1.4*u}
    });
  }
  function label(str,left,top,width,height,size,bold,color) {
    slide.addText(str,{x:X(left),y:Y(top),w:width*w/11.65,h:height*h/7.3,
      fontFace:'Arial',fontSize:size*u,margin:0,bold,color,align:'left',valign:'mid'});
  }
  const q = [
    [2.8,1.02], [2.91,.84],[3.03,.80],[3.15,.84],
    [2.69,.84],[2.57,.80],[2.45,.84],[3.35,1.00],[2.25,1.00],
    [2.94,1.19],[2.66,1.19], [3.52,1.76],[2.08,1.76],
    [4.07,2.66],[1.56,2.68], [4.29,3.47],[1.30,3.48],
    [4.61,3.76],[.97,3.78], [4.43,3.85],[1.13,3.86],
    [4.19,3.75],[1.40,3.77], [3.30,3.72],[2.29,3.72],
    [3.52,5.11],[2.03,5.16], [3.62,6.31],[1.81,6.29],
    [3.52,6.55],[1.93,6.53], [3.97,6.65],[1.42,6.68]
  ];
  // Head silhouette and neck in pale gray, distinct from landmark nodes.
  slide.addShape(pptx.ShapeType.ellipse,{x:X(2.24),y:Y(.42),w:1.12*w/11.65,
    h:1.07*h/7.3,fill:{color:'FFFFFF',transparency:100},line:{color:pale,width:2*u}});
  edge([2.58,1.44],[2.52,1.76],pale,2);
  edge([3.02,1.44],[3.09,1.76],pale,2);
  const edges=[[0,1],[1,2],[2,3],[3,7],[0,4],[4,5],[5,6],[6,8],[9,10],
    [11,12],[11,13],[13,15],[12,14],[14,16],[15,17],[15,19],[15,21],
    [17,19],[16,18],[16,20],[16,22],[18,20],[11,23],[12,24],[23,24],
    [23,25],[24,26],[25,27],[26,28],[27,29],[29,31],[27,31],[28,30],[30,32],[28,32]];
  for(const [a,b] of edges) {
    const hand=b>=17&&b<=22, face=b<11;
    edge(q[a],q[b],(hand||face)?pale:(a>=25?green:navy),(hand||face)?1.6:3.5);
  }
  q.forEach((a,i)=>{
    if(i<11||(i>=17&&i<=22)) mark(a,.044,'FFFFFF',gray,'Unused landmark '+i);
    else if(i<25) mark(a,.081,navy,navy,'Retained landmark '+i);
    else mark(a,.095,green,green,'Forecast target '+i);
  });
  label('33 BlazePose landmarks',5.12,.40,6.48,.70,35,true,navy);
  label('16 used for gait analysis',5.12,1.25,6.48,.58,29,true,green);
  const rows=[
    [2.29,green,green,'8 forecast targets','Knees, ankles, heels\nand forefeet'],
    [3.86,navy,navy,'8 other study landmarks','Shoulders, elbows, wrists\nand hips'],
    [5.46,'FFFFFF',gray,'17 unused landmarks','Face and fingers']
  ];
  for (const [top,fill,stroke,heading,desc] of rows) {
    mark([5.31,top+.24],.083,fill,stroke,'Legend '+heading);
    label(heading,5.65,top,5.87,.49,27,true,navy);
    label(desc,5.65,top+.56,5.87,.86,24,false,'344552');
  }
};
