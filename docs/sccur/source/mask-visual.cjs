// Equal-budget diagrams explain the policies. Only the measured MSE values are data.
const fs = require('node:fs');
const path = require('node:path');

module.exports = function maskVisual(pptx, slide, x, y, w, h) {
  const file = path.join(__dirname,'../../../experiments/aaai-worldmodels/docs/draft03/claim_ledger.json');
  const ledger = JSON.parse(fs.readFileSync(file,'utf8'));
  const comparisons = ledger.claims.find(c=>c.id==='C2').contrasts;
  const tubes = comparisons.find(c=>c.method==='mlp_anatomical_tubes');
  const balanced = comparisons.find(c=>c.method==='mlp_motion_balanced');
  if (tubes.reference!=='mlp_uniform' || balanced.reference_mse!==tubes.reference_mse) {
    throw Error('Masking scores lost their shared uniform reference');
  }
  const rows = [
    ['Uniform random','Choose positions at random.',tubes.reference_mse,[0,3,7,9]],
    ['Connected limb joints','Hide neighbors over consecutive times.',tubes.method_mse,[1,2,6,7]],
    ['Motion-balanced','Sample from slow and fast movements.',balanced.method_mse,[0,2,8,9]]
  ];
  const scale=Math.min(w/11.9,h/5.8), ink='192C45', navy='1F3864', teal='00796D', grid='CCD5DA';
  const shape=pptx.ShapeType;
  function label(text,left,top,width,height,pt,opts={}) {
    slide.addText(text,{x:x+left*w,y:y+top*h,w:width*w,h:height*h,margin:0,
      valign:'mid',fontFace:'Arial',fontSize:pt*scale,color:ink,...opts});
  }
  function box(left,top,width,height,masked) {
    slide.addShape(shape.rect,{x:x+left*w,y:y+top*h,w:width*w,h:height*h,
      fill:{color:masked?teal:'FFFFFF'},line:{color:masked?teal:'71869A',width:1.35*scale}});
  }
  function rule(top) {
    slide.addShape(shape.line,{x,y:y+top*h,w,h:0,line:{color:grid,width:1.3}});
  }
  label('Schematic masks',0,0,.32,.092,25,{bold:true,color:navy});
  label('Training rule',.32,0,.385,.092,25,{bold:true,color:navy});
  label('MSE × 1,000',.70,0,.30,.092,25,{bold:true,color:navy,align:'right'});
  rule(.105);
  rows.forEach(([name,desc,mse,hidden],i)=>{
    const top=.123+i*.222;
    for (let t=0;t<10;t++) {
      box(.016+(t%5)*.050,top+.026+Math.floor(t/5)*.065,.042,.055,hidden.includes(t));
    }
    label(name,.32,top,.48,.097,29,{bold:true});
    label(desc,.32,top+.098,.675,.095,24);
    label((mse*1000).toFixed(2),.81,top,.19,.098,32,{bold:true,align:'right',color:navy});
    rule(top+.205);
  });
  box(.014,.825,.026,.052,true);
  label('Hidden',.052,.802,.155,.096,24);
  box(.235,.825,.026,.052,false);
  label('Visible',.273,.802,.16,.096,24);
  label('Each grid hides 4 of 10 observations (40%).',.438,.802,.562,.096,23);
  label('Each row is a joint. Time increases from left to right.',0,.907,1,.085,24);
};
