// Each bar represents one held-out normal recording, pooled as in draft03.
const fs = require("node:fs");
const path = require("node:path");

function recordingGains() {
  const csv = path.resolve(__dirname, "../../../experiments/aaai-worldmodels/docs/draft03/refs_source_scores.csv");
  const recordings = new Map();
  for (const row of fs.readFileSync(csv, "utf8").trim().split(/\r?\n/).slice(1)) {
    const [source, label, , , mse, arm, horizon] = row.split(",");
    if (label !== "normal" || horizon !== "8" || !["ridge_posture", "persistence"].includes(arm)) continue;
    const pair = recordings.get(source) || {};
    if (Object.hasOwn(pair, arm)) throw new Error("Duplicate source and arm: " + source);
    pair[arm] = Number(mse);
    recordings.set(source, pair);
  }
  const gains = Array.from(recordings.values(), pair => 1000 * (pair.persistence - pair.ridge_posture));
  if (gains.length !== 29 || !gains.every(Number.isFinite)) throw new Error("Incomplete locked recording pairs");
  return gains.sort((a, b) => a - b);
}

function pairedRecordings(pptx, slide, x, y, w, h, pacs = false) {
  const gains = recordingGains();
  const count = gains.filter(value => value > 0).length;
  const ink = pacs ? '1A1A1A' : '182F39', teal = pacs ? '1E6B3A' : '007A73', grey = pacs ? '4A5568' : '546A71', adverse = pacs ? '9B2C2C' : '94441D';
  const scale = Math.min(w / 20, h / 7.5);
  const txt = (text, left, top, width, height, size, style = {}) => slide.addText(text, {
    x: x + left*w, y: y + top*h, w: width*w, h: height*h, margin:0,
    fontFace:"Arial", fontSize:size*scale, color:ink, valign:"mid", ...style
  });
  const ordinate = value => y + h * (.77 - .51 * value / 180);
  const left = x + .071*w, right = x + .985*w;
  txt('Posture improved forecasts in ' + count + ' of ' + gains.length + ' recordings', 0, 0, 1, .12, 42, {bold:true});
  txt('Error reduction relative to repeating the last pose (MSE × 1,000)', 0, .135, 1, .095, 29, {color:grey});
  for (const tick of [0,50,100,150]) {
    const py = ordinate(tick);
    slide.addShape(pptx.ShapeType.line, {x:left, y:py, w:right-left, h:0,line:{color:tick===0?ink:"CEDBDB",width:tick===0?1.8:1}});
    txt(String(tick), 0, (py-y)/h-.04, .055, .08, 25, {align:"right",color:grey});
  }
  const step = (right-left)/gains.length;
  for (const [i,value] of gains.entries()) {
    const color = value > 0 ? teal : adverse;
    const height = Math.abs(ordinate(0)-ordinate(value));
    slide.addShape(pptx.ShapeType.rect, {x:left+(i+.14)*step,y:Math.min(ordinate(0),ordinate(value)),w:step*.72,h:height,
      line:{color,transparency:100},fill:{color}});
  }
  txt("One bar per recording, ordered by error reduction", 0, .895, 1, .105, 29, {align:"center",color:grey});
}
module.exports = pairedRecordings;
module.exports.pairedRecordings = pairedRecordings;
