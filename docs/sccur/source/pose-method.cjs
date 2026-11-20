// Editable PowerPoint geometry. The gait pose is a schematic, not patient data.
module.exports = function drawPoseMethod(pptx, slide, x, y, w, h, pacs = false) {
  const unit = Math.min(w / 11.1, h / 6);
  const ink = pacs ? '1A1A1A' : '182F39', teal = pacs ? '1E6B3A' : '007A73', gray = pacs ? '4A5568' : '657983';
  const px = v => x + v * w / 11.1;
  const py = v => y + v * h / 6;
  function line(a, b, color, weight) {
    slide.addShape(pptx.ShapeType.line, {
      x: px(Math.min(a[0], b[0])), y: py(Math.min(a[1], b[1])),
      w: Math.abs(px(a[0]) - px(b[0])), h: Math.abs(py(a[1]) - py(b[1])),
      flipV: (a[1] - b[1]) * (a[0] - b[0]) < 0,
      line: { color, width: weight * unit, beginArrowType: undefined }
    });
  }
  function node(xx, yy, radius, fill, outline = fill) {
    slide.addShape(pptx.ShapeType.ellipse, {
      x: px(xx) - radius * unit, y: py(yy) - radius * unit,
      w: 2 * radius * unit, h: 2 * radius * unit,
      fill: { color: fill }, line: { color: outline, width: 1.9 * unit }
    });
  }
  function label(s, xx, yy, ww, hh, size, color, bold = false) {
    slide.addText(s, {
      x: px(xx), y: py(yy), w: ww * w / 11.1, h: hh * h / 6,
      fontFace: 'Arial', fontSize: size * unit, color, bold,
      margin: 0, breakLine: false, valign: 'mid', align: 'left', paraSpaceAfterPt: 0
    });
  }
  // Shoulders, elbows, wrists, hips, knees, ankles, heels and foot indices.
  const joints = [
    [2.28,.64],[2.78,.73],[1.89,1.28],[3.17,1.21],[2.17,1.77],[3.61,1.02],
    [2.10,1.92],[2.59,2.01],[1.91,2.69],[3.31,2.57],
    [1.17,3.05],[3.03,3.45],[.99,3.04],[2.86,3.46],[1.30,3.33],[3.52,3.49]
  ];
  const edges = [[0,1],[0,2],[2,4],[1,3],[3,5],[0,6],[1,7],[6,7],
                 [6,8],[8,10],[10,12],[12,14],[10,14],
                 [7,9],[9,11],[11,13],[13,15],[11,15]];
  edges.forEach(([a,b]) => line(joints[a],joints[b],a >= 8 ? teal : gray,3.4));
  joints.forEach(([xx,yy],i) => node(xx,yy,i >= 8 ? .078 : .062,i >= 8 ? teal : 'FFFFFF',i >= 8 ? teal : gray));
  label('16 joints observed',4.27,.52,6.68,.55,32,ink,true);
  label('Each frame records joint positions\nrelative to the hips.',4.27,1.12,6.65,.88,25,ink);
  label('8 joints forecast',4.27,2.14,6.65,.52,32,teal,true);
  label('Knee, ankle, heel and forefoot\non each leg (' + (pacs ? 'green' : 'teal') + ' dots).',4.27,2.74,6.68,.88,25,ink);
  line([.18,3.96],[10.89,3.96],'B9C7C9',1.2);
  label('16 past observations',.20,4.22,6.38,.50,28,ink,true);
  label('8 future positions',7.12,4.22,3.94,.50,27,teal,true);
  const positions = Array.from({length:24},(_,i) => .35 + .443 * i + (i >= 16 ? .14 : 0));
  line([positions[0],5.02],[positions[15],5.02],gray,2);
  line([positions[16],5.02],[positions[23],5.02],teal,2);
  positions.forEach((xx,i) => node(xx,5.02,.072,i < 16 ? gray : 'FFFFFF',i < 16 ? gray : teal));
  label('−1.5 s',.19,5.29,1.38,.46,23,gray);
  label('0 s',6.67,5.29,.76,.46,23,gray);
  label('+0.8 s',9.56,5.29,1.45,.46,23,teal);
};
