// Editable teaching schematic. Positions illustrate the scoring rule, not observed data.
function forecastExplainer(pptx, slide, x, y, w, h) {
  const navy = '1F3864', blue = '2358AB', teal = '00796D', gray = '495D6D';
  const sx = w / 20.1, sy = h / 5.0, scale = Math.min(sx, sy);
  function label(text, left, top, width, height, size, opts = {}) {
    slide.addText(text, {x:x+left*sx, y:y+top*sy, w:width*sx, h:height*sy,
      margin:0, fontFace:'Arial', fontSize:size*scale, color:navy, valign:'mid', ...opts});
  }
  function line(ax, ay, bx, by, color, weight) {
    slide.addShape(pptx.ShapeType.line, {x:x+Math.min(ax,bx)*sx, y:y+Math.min(ay,by)*sy,
      w:Math.abs(ax-bx)*sx, h:Math.abs(ay-by)*sy, flipV:(bx-ax)*(by-ay)<0,
      line:{color,width:weight*scale}});
  }
  function dot(cx, cy, radius, fill, stroke) {
    slide.addShape(pptx.ShapeType.ellipse, {x:x+cx*sx-radius*scale, y:y+cy*sy-radius*scale,
      w:2*radius*scale, h:2*radius*scale, fill:{color:fill},line:{color:stroke,width:1.9*scale}});
  }

  function step(name, left, top, width, color) {
    slide.addShape(pptx.ShapeType.rect,{x:x+left*sx,y:y+top*sy,w:width*sx,h:.45*sy,
      line:{transparency:100},fill:{color}});
    label(name,left,top,width,.45,23,{bold:true,color:'FFFFFF',align:'center'});
  }
  const gapColor = 'A34819';
  step('Start',0,.01,1.47,navy);
  label('Both models start from the same pose.',1.71,0,12,.52,27,{bold:true});
  label('One coordinate · schematic',13.68,0,6.37,.52,23,{align:'right',color:gray});
  step('Forecast',0,.61,1.85,teal);
  label('Each line predicts eight positions.',2.08,.58,10.7,.52,26);
  dot(13.62,.825,.092,'FFFFFF',gray);
  label('Observed position',13.87,.58,6.15,.52,25,{color:gray});
  const start = 2.54;
  const observed = [2.50,2.30,1.98,1.69,1.59,1.78,2.10,2.47];
  const learned = [2.28,2.07,1.87,1.72,1.73,1.88,2.10,2.27];
  function panel(offset,title,forecast,color) {
    label(title,offset,1.30,9.65,.48,30,{bold:true,color});
    const py = value => value + .80;
    const px = i => offset + .62 + i * 1.05;
    forecast.forEach((v,i) => line(px(i), py(i===0?start:forecast[i-1]), px(i+1),py(v),color,3.1));
    forecast.forEach((v,i) => {
      const highlight = offset === 0 && i === 4;
      line(px(i+1),py(v),px(i+1),py(observed[i]),highlight?gapColor:'8C9CA4',highlight?4:1.5);
      dot(px(i+1),py(v),.055,color,color);
      dot(px(i+1),py(observed[i]),.087,'FFFFFF',gray);
    });
    dot(px(0),py(start),.11,navy,navy);
    label('Now',offset+.04,3.52,1.18,.42,24,{align:'center',color:gray});
    label('+0.1 s',px(1)-.48,3.52,1.02,.42,23,{align:'center',color:gray});
    label('+0.8 s',px(8)-.64,3.52,1.28,.42,23,{align:'center',color:gray});
  }
  panel(0,'Repeat the latest position',Array(8).fill(start),blue);
  line(9.94,1.31,9.94,3.96,'C9D4DD',1.1);
  panel(10.35,'Predict from the latest posture',learned,teal);
  step('Score',0,4.21,1.47,gapColor);
  label('Each vertical gap is a miss. We square the gaps and average them.',1.71,4.15,18.3,.59,27,{bold:true,color:gray});
}
module.exports = forecastExplainer;
