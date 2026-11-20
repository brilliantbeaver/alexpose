// Build the current SCCUR poster with Node and pptxgenjs 4.0.1.
const pptxgen = require('pptxgenjs');
const fs = require('node:fs');
const path = require('node:path');
const p = new pptxgen();
p.defineLayout({name:'Poster',width:48,height:34});
p.layout='Poster';
p.author='Alexander Mui, Jason Liu, Sofia Fan, Phil Mui';
p.title='How well does current posture predict leg movement?';
p.theme={headFontFace:'Arial',bodyFontFace:'Arial',lang:'en-US'};
p.defineSlideMaster({title:'Paper',background:{color:'FFFFFF'},objects:[]});
const s=p.addSlide('Paper');
const navy='1F3864', dark='192C45', green='00796D', gray='46566A', border='CBD8E3';
const draft=path.resolve(__dirname,'../../../experiments/aaai-worldmodels/docs/draft03');
const weights=JSON.parse(fs.readFileSync(path.join(draft,'sequence_weighting_results.json')));
const primary=weights.recording_weighted_primary;
const pct=v=>(100*v).toFixed(1)+'%';
const L=1,C=13.8,R=35.1,lw=11.65,cw=20.1,rw=11.9;
function txt(t,x,y,w,h,pt,opt={}){s.addText(t,{isTextBox:true,x,y,w,h,fontFace:'Arial',fontSize:pt,color:dark,margin:0,valign:'mid',paraSpaceAfterPt:0,...opt});}
function body(t,x,y,w,h,pt=31){txt(t,x,y,w,h,pt,{valign:'top'});}
function band(x,y,w,h,color){s.addShape(p.ShapeType.rect,{x,y,w,h,line:{transparency:100},fill:{color}});}
function line(x,y,w,h=0){s.addShape(p.ShapeType.line,{x,y,w,h,line:{color:border,width:1.4}});}
function section(t,x,y,w){
  txt(t,x,y,w,.7,33,{bold:true,color:navy});
  line(x,y+.83,w);
}
function result(number,t,x,y,w){
  band(x,y,.77,.77,navy);
  txt(String(number),x,y,.77,.77,35,{align:'center',bold:true,color:'FFFFFF'});
  txt(t,x+1.04,y,w-1.04,.8,34,{bold:true,color:navy});
}
function heading(t,x,y,w,h=1.1,pt=43){txt(t,x,y,w,h,pt,{bold:true});}
txt('SCCUR 2026    /    SAN DIEGO STATE UNIVERSITY',1,.46,30,.55,28,{bold:true,color:navy,charSpacing:2});
txt('NOVEMBER 21, 2026',36,.46,11,.55,26,{align:'right',color:gray});
band(0,1.13,40.95,1.82,navy);
txt(p.title,1,1.17,39.1,1.72,90,{bold:true,fontFace:'Georgia',color:'FFFFFF'});
s.addImage({path:path.join(__dirname,'asdrp-logo.png'),x:41.65,y:1.15,w:5.1,h:5.1*594/1607,altText:'ASDRP — Aspiring Scholars Directed Research Program'});
txt([{text:'Alexander Mui',options:{underline:true}},{text:', Jason Liu, Sofia Fan, and Phil Mui'}],1,3.12,46,.72,40,{bold:true});
txt('Department of Computer Science and Engineering, Aspiring Scholars Directed Research Program (ASDRP), Fremont, California',1,3.97,46,.64,31,{color:gray});
line(1,4.96,46);line(13.12,5.52,0,22.86);line(34.47,5.52,0,26.61);

section('Abstract',L,5.46,lw);
body('A walker’s posture offers a clue to where the legs will move next. We trained a model on normal walking videos to forecast knee and foot positions over 0.8 seconds.',L,6.56,lw,2.1,30);
body('On held-out normal recordings, it cut mean squared error by '+pct(primary.relative_reduction)+' compared with repeating the last pose. A separate error score was less effective at ranking abnormal gait in the original dataset than measuring how little the legs moved.',L,8.83,lw,2.83,29);
band(L,11.91,lw,2.9,'ECF3FC');
txt('Research question',L+.28,12.11,lw-.56,.64,30,{bold:true,color:navy});
body('How much can current posture tell us about the next part of a stride?',L+.28,12.97,lw-.56,1.51,36);
section('What the model sees',L,15.15,lw);
require('./pose-33.cjs')(p,s,L,16.08,lw,6.12);
section('Study design',L,22.42,lw);
const cohort=[['91','recordings'],['456','sequences'],['2,815','windows']];
cohort.forEach(([n,name],i)=>{
  const x=L+i*3.96;
  txt(n,x,23.52,3.64,.92,49,{bold:true,color:navy});
  txt(name,x,24.48,3.68,.55,25,{bold:true,color:gray});
});
body('From GAVD walking videos [1], each window pairs 16 past poses with eight future poses, sampled 0.1 seconds apart. We held out related recordings together across five folds.',L,25.37,lw,2.10,28);
body('The main test used 141 normal sequences from 29 recordings.',L,27.64,lw,.85,25);

result(1,'Build two forecasts from the same starting pose',C,5.46,cw);
require('./forecast-explainer.cjs')(p,s,C,6.42,cw,5.0);
txt(pct(primary.relative_reduction),C,11.55,4.94,1.25,91,{bold:true,color:green});
txt('lower mean squared error',C+5.10,11.61,14.8,.95,48,{bold:true});
txt('Paired 95% interval   '+primary.relative_ci95.map(pct).join(' to '),C+5.12,12.62,14.6,.56,27,{color:green,bold:true});
require('./forecast-results.cjs')(p,s,C,13.45,cw,4.54,true,true);
body('Positions are measured relative to the hips and scaled by torso length. Dots are mean errors; whiskers show 95% intervals from resampling whole recordings 5,000 times with predictions fixed.',C,18.14,cw,1.14,25);
result(2,'Check whether the gain holds for individual recordings',C,19.47,cw);
require('./paired-worked-example.cjs')(p,s,C,20.54,cw,2.0);
require('./recording-visual.cjs')(p,s,C,22.72,cw,4.45);
band(C,27.37,cw,.96,'E5F4F0');
txt('Weighting check',C+.20,27.46,3.69,.63,25,{bold:true,color:green});
txt('Giving each sequence equal weight yielded a '+pct(weights.relative_reduction)+' reduction in a post hoc check.',C+3.98,27.45,cw-4.19,.67,25);

result(3,'Change what the model sees during training',R,5.46,rw);
body('We asked whether hiding joints helps a model learn from the rest of the body. We tested three rules with a model that predicts future joint positions.',R,6.44,rw,1.60,27);
require('./mask-train-test.cjs')(p,s,R,8.09,rw,1.30);
require('./mask-visual.cjs')(p,s,R,9.46,rw,4.41);
body('Compare the last column: lower error is better. Neither structured pattern improved on random hiding.',R,13.97,rw,.94,25);
section('Interpretation',R,15.02,rw);
heading('Why might an unusual step be easy to predict?',R,15.93,rw,.94,33);
// A schematic example, with the two lower-leg poses sharing a hip anchor.
const hx=R+1.00, hy=17.06;
function segment(x1,y1,x2,y2,color,style){
  s.addShape(p.ShapeType.line,{x:Math.min(x1,x2),y:Math.min(y1,y2),w:Math.abs(x2-x1),h:Math.abs(y2-y1),flipV:(x2-x1)*(y2-y1)<0,line:{color,width:3.2,dashType:style}});
}
function joint(x,y,color,open){s.addShape(p.ShapeType.ellipse,{x:x-.075,y:y-.075,w:.15,h:.15,line:{color,width:1.6},fill:{color:open?'FFFFFF':color}});}
for(const [dx,color,style] of [[0,'728096','dash'],[.32,navy,'solid']]){
  const kx=hx+.20+dx/2, ky=hy+.83, ax=hx+.13+dx, ay=hy+1.66;
  segment(hx,hy,kx,ky,color,style);segment(kx,ky,ax,ay,color,style);segment(ax,ay,ax+.55,ay+.06,color,style);
  joint(kx,ky,color,dx===0);joint(ax,ay,color,dx===0);
}
joint(hx,hy,navy,false);
txt('Two poses',R,18.83,2.71,.42,21,{align:'center',color:gray});
body('Imagine a knee that barely changes position during a step. Repeating its last position could give a small error, even if a reviewer labels the gait abnormal.',R+2.58,17.02,rw-2.58,1.82,26);
body('Pair a normal and an abnormal recording. Which ranks the abnormal one higher: unusual forecast error from posture and velocity, or low movement?',R,19.46,rw,1.39,25);
require('./interpretation-pairs.cjs')(p,s,R,20.97,rw,2.72);
body('Across 90 participants in a separate dataset [2], the estimates were 59 versus 76 per 100 pairs. The difference was uncertain.',R,23.75,rw,.95,23);

line(1,28.51,32.92);
txt('How does S-JEPA learn when joints are hidden? [3]',L,28.67,32.74,.66,37,{bold:true,color:navy});
require('./sjepa-workflow.cjs')(p,s,L,29.39,32.86,2.57);
txt('Learning is self-supervised because the video supplies its own training target. S-JEPA predicts hidden features. Our gait models predicted joint positions.',L,32.00,32.80,.45,22,{color:navy});
section('Next Steps',R,24.96,rw);
body('In an exploratory analysis, low movement alone ranked Parkinson’s and stroke recordings above normal recordings in about 87–88% of pairs. Each condition shared the same 29 normal recordings.',R,26.02,rw,1.73,25);
function nextStudy(y,title,detail){
  band(R,y+.07,.09,.46,green);
  txt(title,R+.22,y,rw-.22,.50,25,{bold:true,color:navy});
  body(detail,R+.22,y+.58,rw-.31,.99,24);
}
nextStudy(27.61,'What does each leg add?','We propose testing step timing in Parkinson’s gait and left–right knee differences after stroke, at similar movement levels and camera views.');
nextStudy(29.33,'Does the score work across conditions?','We would test S-JEPA against low movement in new walkers and report cerebral palsy (8 recordings here) and myopathic gait (27) separately.');
nextStudy(31.08,'Can we trust the tracked joints?','We would review continuous clips for blocked joints and tracking errors before testing on participants the model has never seen.');
line(1,32.59,46);
txt('References',1,32.64,3.4,.39,22,{bold:true,color:navy});
txt('Acknowledgments',35.1,32.64,11.9,.39,22,{bold:true,color:navy});
body('[1] Ranjan et al. Computer Vision for Clinical Gait Analysis: A Gait Abnormality Video Dataset. IEEE Access (2025). doi:10.1109/ACCESS.2025.3545787\n[2] Kour et al. Gait Dataset for Knee Osteoarthritis and Parkinson’s Disease Analysis With Severity Levels. Mendeley Data (2020). doi:10.17632/44pfnysy89.1\n[3] Abdelfattah & Alahi. S-JEPA: A Joint Embedding Predictive Architecture for Skeletal Action Recognition. ECCV (2024). doi:10.1007/978-3-031-73411-3_21',1,33.03,33.1,.88,19);
body('We thank the Department of Computer Science and Engineering at ASDRP and the teams who released the gait datasets.',R,33.05,rw,.83,20);
s.addNotes('The masking proposal and condition-specific diagnostics come from experiments/aaai-worldmodels/docs/draft02/manuscript.md and condition_scores.csv. Each condition was compared with the same 29 normal source recordings. The Parkinson’s and stroke low-motion AUROCs are descriptive pooled estimates (0.868 and 0.877). These are comparisons against normal gait labels, not differential diagnoses. The proposed tests on new walkers have not been performed. The S-JEPA diagram illustrates the representation-learning objective described by Abdelfattah and Alahi (2024). The main results and the coordinate masking experiment come from docs/draft03. Draft03 forecast intervals condition on saved predictions; its primary GAVD ranking comparison is within fold. The external score difference was uncertain. Source recordings in GAVD are not verified patient identities.');
p.writeFile({fileName:path.join(__dirname,'..','Gait-Forecasting-SCCUR-2026.pptx')});
