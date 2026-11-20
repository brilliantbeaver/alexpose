// Each schematic has five eligible tokens and one unavailable token.
module.exports = function maskTrainTest(pptx, slide, x, y, w, h) {
  const scale = Math.min(w/11.9,h/1.6);
  const ink='192C45', navy='1F3864', teal='00796D', outline='71869A';
  const shapes=pptx.ShapeType;
  function text(value,u,v,width,height,pt,opts={}) {
    slide.addText(value,{x:x+u*w,y:y+v*h,w:width*w,h:height*h,fontFace:'Arial',
      fontSize:pt*scale,margin:0,valign:'mid',color:ink,...opts});
  }
  function icon(start,train) {
    for(let i=0;i<6;i++) {
      const u=start+(i%3)*.0257, v=.195+Math.floor(i/3)*.203;
      const hidden=train && (i===1 || i===3);
      slide.addShape(shapes.rect,{x:x+u*w,y:y+v*h,w:.023*w,h:.176*h,
        fill:{color:hidden?teal:'FFFFFF'},line:{color:hidden?teal:outline,width:1.45*scale}});
      if(i===5) text('×',u,v-.01,.023,.183,20,{bold:true,color:ink,align:'center'});
    }
  }
  icon(0,true);
  text('Train',.089,.018,.31,.35,29,{bold:true,color:navy});
  text('Hide 40% of eligible past inputs.',.089,.369,.363,.28,23);
  slide.addShape(shapes.rightArrow,{x:x+.46*w,y:y+.23*h,w:.062*w,h:.24*h,
    line:{transparency:100},fill:{color:navy}});
  icon(.55,false);
  text('Test',.64,.018,.34,.35,29,{bold:true,color:navy});
  text('Use all available past inputs.',.64,.369,.36,.28,23);
  text('× = missing in both stages. All methods score the same observed future joints.',0,.69,1,.30,22);
};
