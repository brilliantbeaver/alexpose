// Source of paired resampling intervals: the locked draft03 claim ledger.
const fs = require("node:fs");
const path = require("node:path");

module.exports = function maskComparison(pptx, slide, x, y, w, h) {
  const file = path.join(__dirname, "../../../experiments/aaai-worldmodels/docs/draft03/claim_ledger.json");
  const ledger = JSON.parse(fs.readFileSync(file, "utf8"));
  const contrasts = ledger.claims.find(claim => claim.id === "C2").contrasts;
  const fmt = value => (1000 * value).toFixed(2);
  const rows = [["Uniform random", fmt(contrasts[0].reference_mse), "Reference"]];
  for (const [id, name] of [["mlp_anatomical_tubes", "Anatomical tubes"], ["mlp_motion_balanced", "Motion-balanced"]]) {
    const item = contrasts.find(row => row.method === id);
    if (item.reference !== "mlp_uniform" || item.reference_mse !== contrasts[0].reference_mse) {
      throw Error("Inconsistent masking comparison");
    }
    // Report candidate minus uniform, reversing the ledger's signed benefit and interval.
    const [lo, hi] = item.ci95;
    rows.push([name, fmt(item.method_mse), "+"+fmt(-item.estimate)+" ["+fmt(-hi)+", "+fmt(-lo)+"]"]);
  }
  const scale = Math.min(w / 11.9, h / 3.7);
  function label(text, left, top, width, height, size, style={}) {
    slide.addText(text, {x:x+left*w,y:y+top*h,w:width*w,h:height*h,margin:0,
      fontFace:"Arial",fontSize:size*scale,color:"1A1A1A",valign:"mid",...style});
  }
  function rule(top) {
    slide.addShape(pptx.ShapeType.line,{x,y:y+top*h,w,h:0,line:{color:"CDD5DE",width:1.2}});
  }
  label("Training mask",0,0,.39,.18,25,{bold:true,color:"1F3864"});
  label("MSE × 1,000",.39,0,.24,.18,24,{bold:true,align:"right",color:"1F3864"});
  label("Extra error [95% CI]",.635,0,.365,.18,23,{bold:true,align:"right",color:"1F3864"});
  rule(.185);
  rows.forEach(([mask,score,delta],i)=>{
    const top=.19+i*.194;
    label(mask,0,top,.39,.19,27);
    label(score,.39,top,.24,.19,27,{align:"right",bold:true});
    label(delta,.635,top,.365,.19,25,{align:"right"});
    rule(top+.19);
  });
  label("Extra error is relative to uniform masking, in the same units.",0,.79,1,.19,22,{color:"404040"});
};
