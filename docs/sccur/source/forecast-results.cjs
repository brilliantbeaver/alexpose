// Rebuild the point estimates and intervals directly from the published draft03 table.
const fs = require("node:fs");
const path = require("node:path");

function forecastResults(pptx, slide, x, y, w, h, pacs = false, compact = false) {
  const table = path.join(__dirname, "../../../experiments/aaai-worldmodels/docs/draft03/forecasts.csv");
  const records = new Map();
  for (const line of fs.readFileSync(table, "utf8").trim().split(/\r?\n/).slice(1)) {
    const fields = line.match(/^([^,]+),([\d.]+),"\[([\d.]+), ([\d.]+)\]"$/);
    if (!fields) throw new Error("Unexpected forecast table row: " + line);
    records.set(fields[1], fields.slice(2).map(Number));
  }
  const rows = [
    ["Persistence", "Repeat last pose"],
    ["Own-leg history", "One leg’s history"],
    ["Bilateral history", "Both legs’ history"],
    ["Current-posture transition", "Current posture"]
  ];
  const shapes = pptx.ShapeType;
  const ink = pacs ? '1A1A1A' : '182F39', teal = pacs ? '1E6B3A' : '007A73', grey = pacs ? '4A5568' : '546A71', grid = "CBD7D8";
  const fontScale = compact ? .91 : Math.min(w / 20, h / 7.5);
  const txt = (s, left, top, width, height, pt, options = {}) => {
    slide.addText(s, {x:left, y:top, w:width, h:height, margin:0, fontFace:"Arial",
      fontSize:pt*fontScale, color:ink, breakLine:false, valign:"mid", ...options});
  };
  const line = (left, top, width, height, color, weight) => {
    slide.addShape(shapes.line, {x:left,y:top,w:width,h:height,line:{color,width:weight*fontScale,beginArrowType:undefined,endArrowType:undefined}});
  };
  const leftPlot = x + w * .334, widthPlot = w * .50;
  const pos = (v) => leftPlot + widthPlot * v / 160;
  const axisY = y + .785*h;
  txt("Forecast error on held-out normal walking", x, y, w, h*.118, 42, {bold:true});
  const tickValues = [0, 40, 80, 120, 160];
  for (const tick of tickValues) {
    line(pos(tick),y+.165*h,0,axisY-(y+.165*h),grid,1);
    txt(String(tick),pos(tick)-.7*fontScale,axisY+.015*h,1.4*fontScale,h*.084,26,{align:"center",color:grey});
  }
  rows.forEach(([key, label], i) => {
    const [mean,low,high] = records.get(key);
    const cy = y + h*(.22+i*.162);
    const accent = i === rows.length - 1, color = compact && accent ? "00796D" : compact && i === 0 ? "2358AB" : accent ? teal : grey;
    txt(label,x,cy-.065*h,w*.315,h*.13,32,{bold:accent,color:accent?teal:ink});
    line(pos(low),cy,pos(high)-pos(low),0,color,3.8);
    for (const endpoint of [low,high]) line(pos(endpoint),cy-.037*h,0,.074*h,color,3);
    const d=.25*fontScale;
    slide.addShape(shapes.ellipse,{x:pos(mean)-d/2,y:cy-d/2,w:d,h:d,line:{color},fill:{color}});
    txt(mean.toFixed(2),x+w*.854,cy-.06*h,w*.14,h*.12,30,{bold:accent,color:accent?teal:ink,align:"right"});
  });
  line(pos(0),axisY,widthPlot,0,grey,1.4);
  txt("Mean squared error × 1,000  ·  Lower is better",x,y+.89*h,w,h*.10,29,{align:"center",color:grey});
}

module.exports = forecastResults;
module.exports.forecastResults = forecastResults;
