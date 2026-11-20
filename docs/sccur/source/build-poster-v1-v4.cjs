// Rebuild with Node and pptxgenjs 4.0.1: node build-poster.cjs 4 (use 1 or 2 for earlier layouts).
const pptxgen = require('pptxgenjs');
const fs = require('node:fs'), path = require('node:path');
const version = Number(process.argv[2] || 4);
const p = new pptxgen(); p.defineLayout({name:'Poster',width:48,height:34}); p.layout='Poster';
p.author='Alexander Mui, Jason Liu, Sofia Fan, Phil Mui'; p.title='Forecasting the next steps of a walking skeleton';
p.theme={headFontFace:'Arial',bodyFontFace:'Arial',lang:'en-US'};
p.defineSlideMaster({title:'Paper',background:{color:'FFFFFF'},objects:[]});
// PACS-algebra.pptx: title and section navy, favorable green, adverse red, neutral slate.
const pacs=version>=4;
const navy='1F3864';
const s=p.addSlide('Paper'), dark=pacs?'1A1A1A':'182F39', teal=pacs?'1E6B3A':'007A73', gray=pacs?'404040':'4F656B', sep=pacs?'CDD5DE':'C6D3D4';
function band(x,y,w,h,color){s.addShape(p.ShapeType.rect,{x,y,w,h,line:{transparency:100},fill:{color}});}
const draft=path.resolve(__dirname,'../../../experiments/aaai-worldmodels/docs/draft03');
const weights=JSON.parse(fs.readFileSync(path.join(draft,'sequence_weighting_results.json')));
const primary=weights.recording_weighted_primary;
const pct=x=>(100*x).toFixed(1)+'%';
function txt(t,x,y,w,h,pt,opt={}){s.addText(t,{isTextBox:true,x,y,w,h,fontFace:'Arial',fontSize:pt,color:dark,margin:0,valign:'mid',paraSpaceAfterPt:0,...opt});}
function rule(x,y,w){s.addShape(p.ShapeType.line,{x,y,w,h:0,line:{color:sep,width:1.5}});}
function label(t,x,y,w){
  if(!pacs || y<1){txt(t,x,y,w,.55,28,{bold:true,color:pacs?navy:teal,charSpacing:2});return;}
  const headings={'INTRODUCTION':'Introduction','STUDY DESIGN':'Study design','MAIN RESULT    /    HELD-OUT NORMAL WALKING':'Results on held-out normal walking','S-JEPA    /    EARLIER EXPERIMENT':'Earlier S-JEPA experiment','INTERPRETING FORECAST ERROR':'Interpreting forecast error','REFERENCES':'References','ACKNOWLEDGMENTS':'Acknowledgments'};
  band(x,y-.06,w,.73,navy);
  txt(headings[t]||t,x,y-.045,w,.69,30,{bold:true,color:'FFFFFF',align:'center'});
}
function title(t,x,y,w){txt(t,x,y,w,1.45,48,{bold:true});}
function body(t,x,y,w,h,pt=31){txt(t,x,y,w,h,pt,{valign:'top'});}
function divide(x){s.addShape(p.ShapeType.line,{x,y:5.8,w:0,h:23.6,line:{color:sep,width:1.2}});}
const L=1, C=13.8, R=35.1, lw=11.65,cw=20.1,rw=11.9;
label('SCCUR 2026    /    SAN DIEGO STATE UNIVERSITY',1,.55,30);
txt('NOVEMBER 21, 2026',36,.55,11,.55,26,{align:'right',color:gray});
if(pacs)band(0,1.21,48,1.82,navy);
txt(version>=2?'How well does current posture predict leg movement?':'Forecasting the next steps of a walking skeleton',1,1.25,pacs?39.2:46,1.72,pacs?89:95,{bold:true,...(pacs?{fontFace:'Georgia',color:'FFFFFF'}:{})});
if(version>=3){
  // Original ASDRP logo from the approved PACS-algebra.pptx, ppt/media/image7.png.
  if(pacs)band(41.45,1.21,5.8,1.82,'FFFFFF');
  s.addImage({path:path.join(__dirname,'asdrp-logo.png'),x:pacs?42.15:41.7,y:pacs?1.31:1.78,w:pacs?4.4:5.3,h:(pacs?4.4:5.3)*594/1607,altText:'ASDRP — Aspiring Scholars Directed Research Program'});
}
txt([{text:'Alexander Mui',options:{underline:true}},{text:', Jason Liu, Sofia Fan, and Phil Mui'}],1,3.27,46,.72,40,{bold:true});
txt('Department of Computer Science and Engineering, Aspiring Scholars Directed Research Program (ASDRP), Fremont, California',1,4.15,46,.64,31,{color:gray});
rule(1,5.23,46);divide(13.12);divide(34.47);
if(version>=3){
  label('INTRODUCTION',L,5.95,lw);
  title('Reading a walking video',L,6.65,lw);
  body('As one foot swings forward, the other supports the body. Can a model use the positions of the legs to predict where the knees and feet will move next?',L,8.40,lw,2.34,31);
  body('We tracked joints in videos of normal walking. Over the next 0.8 seconds, a model trained on those poses reduced mean squared position error by '+pct(primary.relative_reduction)+' compared with repeating the last observed position.',L,10.94,lw,3.03,31);
  require('./pose-method.cjs')(p,s,L,14.16,lw,5.62,pacs);
} else {
label('RESEARCH QUESTION',L,5.95,lw);
title('What can pose tell us?',L,6.66,lw);
body('Given a short video of normal walking, how accurately can we predict where the lower legs will move next?',L,8.45,lw,2.55,34);
require('./pose-method.cjs')(p,s,L,11.45,lw,6.3);
}
rule(L,version>=3?20.08:18.4,lw);
label('STUDY DESIGN',L,version>=3?20.49:19,lw);
txt('456 sequences',L,version>=3?21.34:19.95,lw,.92,55,{bold:true});
body('The Gait Abnormality Video Dataset [1] contributed 91 recordings and 2,815 eligible forecast windows.',L,version>=3?22.45:21.25,lw,2.2);
txt('29 normal recordings',L,version>=3?24.5:23.8,lw,.92,45,{bold:true});
body('These supplied 141 sequences for the main comparison. We trained on normal walking and kept related recordings together across five validation folds. Earlier work used the same dataset, so internal results are exploratory.',L,version>=3?25.75:25.18,lw,3.7);
label('MAIN RESULT    /    HELD-OUT NORMAL WALKING',C,5.95,cw);
txt(pct(primary.relative_reduction),C,6.67,8.7,2.2,124,{bold:true,color:teal});
txt('lower forecast error',C+(version>=2?5.65:8.15),7.05,12,1.5,64,{bold:true});
body('A ridge regression used each joint’s latest observed position to predict eight future positions over 0.8 seconds. It reduced error relative to repeating the last position.',C,9.25,cw,2.05,35);
if(pacs)band(C,11.27,10.75,.9,'E2F0E6');
txt('Paired 95% interval  '+pct(primary.relative_ci95[0])+' to '+pct(primary.relative_ci95[1]),C+(pacs?.16:0),11.35,cw,.72,32,{color:teal,bold:true});
rule(C,12.49,cw);
require('./forecast-results.cjs')(p,s,C,13.05,cw,6.75,pacs);
body('Selected models on the same targets. Whiskers show 95% intervals from 5,000 recording-group resamples with predictions fixed. Error is in squared torso-length units.',C,20.0,cw,1.45,26);
rule(C,21.77,cw);
if(version>=2){require('./paired-recordings.cjs')(p,s,C,22.15,cw,6.95,pacs);}
else {title('Does the weighting matter?',C,22.5,cw);body('Giving every sequence equal weight reduced error by '+pct(weights.relative_reduction)+' in a post hoc check using the same predictions.',C,24.5,cw,2.5,40);}
label('S-JEPA    /    EARLIER EXPERIMENT',R,5.95,rw);
title('Predicting hidden joints',R,6.66,rw);
body('S-JEPA learns by predicting an encoded representation of hidden skeleton observations [2]. Our adaptation used this training before forecasting several possible futures.',R,8.48,rw,4.2);
txt('0.00727',R,12.95,rw,1.3,75,{color:teal,bold:true});
body('Mean improvement in energy score for 16 sampled futures versus one deterministic forecast. This score rewards forecasts that cover the observed pose with appropriate spread.',R,14.35,rw,3.65,29);
body('Six held-out normal recordings, three seeds. Paired 95% interval 0.00661 to 0.00789. The sampled model also had more parameters.',R,18.24,rw,2.25,26);
if(version>=2){
  const names=['Mask half the\nobservations','Predict hidden\nrepresentations','Sample future\nposes'];
  names.forEach((name,i)=>{const xx=R+4.1*i;txt(name,xx,11.52,3.63,1.09,25,{align:'center',bold:true,color:teal});if(i<2)txt('→',xx+3.52,11.62,.64,.6,30,{color:gray,align:'center'});});
}
rule(R,20.92,rw);
label('INTERPRETING FORECAST ERROR',R,21.55,rw);
txt('Can error flag atypical gait?',R,22.43,rw,1.18,44,{bold:true});
body('After observing the future, low movement ranked gait labels better on GAVD. The external difference was uncertain.',R,23.98,rw,1.7,29);
const tableRows=[['AUROC','Forecast','Low motion'],['GAVD [1]','0.462','0.811'],['External [3]','0.591','0.762']];
s.addTable(tableRows.map((r,i)=>r.map((text,j)=>({text,options:{bold:i===0,color:i===0?(pacs?'FFFFFF':teal):dark,...(pacs?{fill:{color:i===0?navy:'FFFFFF'}}:{}),align:j?'right':'left'}}))),{x:R,y:25.9,w:rw,h:1.92,rowH:.64,colW:[4.4,3.2,4.3],border:{color:sep,pt:.8,type:'solid'},fill:{color:'FFFFFF'},margin:.09,fontSize:25,fontFace:'Arial',autoPage:false});
body('AUROC measures ranking accuracy (chance = 0.50). GAVD comparisons were within folds. The external dataset had 90 coded participants.',R,28.03,rw,1.14,24);
rule(1,29.75,46);label('REFERENCES',L,30.17,pacs?32.8:29);
body('[1]  Ranjan et al. Computer Vision for Clinical Gait Analysis: A Gait Abnormality Video Dataset. IEEE Access (2025). doi:10.1109/ACCESS.2025.3545787\n[2]  Abdelfattah & Alahi. S-JEPA: A Joint Embedding Predictive Architecture for Skeletal Action Recognition. ECCV (2024).\n[3]  Kour et al. Gait Dataset for Knee Osteoarthritis and Parkinson’s Disease Analysis With Severity Levels. Mendeley Data (2020). doi:10.17632/44pfnysy89.1',L,31.04,32.3,2.36,22);
label('ACKNOWLEDGMENTS',R,30.17,rw);
body('We thank the Department of Computer Science and Engineering at ASDRP and the researchers who released the gait datasets.',R,31.05,rw,2.1,26);
s.addNotes('Primary evidence: experiments/aaai-worldmodels/docs/draft03/forecasts.csv, refs_source_scores.csv, sequence_weighting_results.json, figures/ranking_comparison_data.csv. The internal benchmark is exploratory. Earlier S-JEPA study: docs/research_results.json and docs/final_scientific_review.md in the same experiment. The earlier 16-future-frame study differs from the main eight-query benchmark.');
p.writeFile({fileName:path.join(__dirname,'..',`Gait-Forecasting-SCCUR-2026-v${version}.pptx`)});
