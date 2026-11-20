// Build the single-slide landscape poster using PptxGenJS 4.0.1.
const fs = require('node:fs');
const path = require('node:path');
const PowerPoint = require('pptxgenjs');
const drawMethod = require('./research-method.cjs');
const drawResults = require('./research-results.cjs');
const drawForecast = require('./forecast-evidence.cjs');
const drawCohortTable = require('./notebook-cohort.cjs');

const pptx = new PowerPoint();
pptx.defineLayout({name:'SCCUR', width:48, height:34});
pptx.layout = 'SCCUR';
pptx.title = 'Representational Learning with Neurological Gait Videos';
pptx.author = 'Alexander Mui, Jason Liu, Sofia Fan, Phil Mui';
pptx.theme = {headFontFace:'Cambria', bodyFontFace:'Calibri', lang:'en-US'};

const slide = pptx.addSlide();
const navy = '1F3864', teal = '00796D', ink = '182F39', slate = '495F6C', grid = 'C7D4D9';
const left = {x:.95,w:12.1}, center = {x:13.85,w:20.4}, right = {x:35.05,w:12.0};

function text(value,x,y,w,h,fontSize,style={}) {
  slide.addText(value,{x,y,w,h,fontFace:'Calibri',fontSize,color:ink,margin:0,
    valign:'mid',align:'left',paraSpaceAfterPt:0,...style});
}
function fill(x,y,w,h,color) {
  slide.addShape(pptx.ShapeType.rect,{x,y,w,h,line:{transparency:100},fill:{color}});
}
function section(label,column,y) {
  slide.addShape(pptx.ShapeType.roundRect,{x:column.x,y,w:column.w,h:.78,
    rectRadius:.07,radius:.07,line:{transparency:100},fill:{color:navy}});
  text(label,column.x+.12,y,column.w-.24,.78,34,{color:'FFFFFF',bold:true,align:'center'});
}
function rule(x,y,w) {
  slide.addShape(pptx.ShapeType.line,{x,y,w,h:0,line:{color:grid,width:1.35}});
}
function logo(file,x,y,w,h,altText) {
  const data=fs.readFileSync(path.join(__dirname,file));
  const iw=data.readUInt32BE(16), ih=data.readUInt32BE(20);
  const scale=Math.min((w-.3)/iw,(h-.23)/ih);
  fill(x,y,w,h,'FFFFFF');
  slide.addImage({data:'image/png;base64,'+data.toString('base64'),
    x:x+(w-iw*scale)/2,y:y+(h-ih*scale)/2,w:iw*scale,h:ih*scale,altText});
}

fill(0,0,48,4.05,navy);
logo('asdrp.png',.91,1.1,5.05,1.83,'Aspiring Scholars Directed Research Program');
logo('mui-group.png',42.04,1.1,5.05,1.83,'Mui Group');
text('Representational Learning with\nNeurological Gait Videos',6.1,.12,35.8,2.22,72,
  {fontFace:'Cambria',bold:true,align:'center',color:'FFFFFF'});
text([{text:'Alexander Mui',options:{underline:true}},{text:', Jason Liu, Sofia Fan, and Phil Mui'}],
  5.9,2.47,36.2,.61,36,{align:'center',color:'FFFFFF'});
text('Department of Computer Science and Engineering, Aspiring Scholars Directed Research Program, Fremont, California',
  5.45,3.29,37.1,.49,26,{align:'center',color:'E5EBF4',bold:true});

// Orient a passerby before asking them to follow the three columns.
text('Motivation',left.x,4.27,16.8,.42,24,{color:teal,bold:true});
text('Camera angle and missing joints can make a walking video recognizable even when the gait pattern is unclear.',
  left.x,4.82,16.75,1.03,30);
fill(18.21,4.36,.035,1.45,grid);
text('Research question',18.74,4.27,27.8,.42,24,{color:teal,bold:true});
text('Does learning to predict hidden joints help a model recognize abnormal gait in a new recording?',
  18.74,4.76,28.16,1.06,39,{bold:true,color:navy});

section('Abstract',left,6.17);
section('Gait classification results',center,6.17);
section('Movement forecasting results',right,6.17);

const abstract=[
  'Walking videos offer a way to study changes in gait across neurological conditions. We used the Gait Abnormality Video Dataset [1] to test what a model can learn from joint positions.',
  'Skeleton Joint-Embedding Predictive Architecture (S-JEPA) [2] learned to summarize a walk in numbers by predicting features of hidden joints. A classifier used those numbers to label clips from familiar recordings. On 17 unseen recordings, classifiers using pretrained or random features both reached 81.8% balanced accuracy.',
  'In a separate experiment, current posture reduced short-term forecast error by 40.8% compared with copying the last pose. Movement was predictable, while the benefit of pretraining for recognizing abnormal gait remains uncertain.'
];
text(abstract.join('\n'),left.x,7.23,left.w,5.30,27,{valign:'top',paraSpaceAfterPt:8});

text('Which recordings answered each question?',left.x,12.63,left.w,.66,32,{color:navy,bold:true});
const studyRows=[
  ['Experiment','Walking samples','Recordings'],
  ['A · Five gait labels','96 clips','18'],
  ['B · Normal or abnormal','405 windows','89'],
  ['Motion forecasting','2,815 windows','91']
];
slide.addTable(studyRows.map((row,i)=>row.map((value,j)=>({text:value,options:{
  bold:i===0||j===0,color:i===0?'FFFFFF':ink,align:j===2?'center':'left',
  fill:{color:i===0?navy:i%2?'FFFFFF':'F0F4F6'}}}))),{
  x:left.x,y:13.53,w:left.w,h:2.36,rowH:.59,colW:[5.42,3.63,3.05],
  fontFace:'Calibri',fontSize:25,margin:[.045,.12,.045,.12],
  border:{type:'solid',pt:.7,color:grid},valign:'middle',autoPage:false});
text('A window is a short stretch of video. Gait labels describe the recordings, and separate videos can show the same person.',
  left.x,16.03,left.w,.95,25,{color:slate,valign:'top'});

section('Methods',left,17.48);
drawMethod(pptx,slide,left.x,18.53,left.w,13.05);
drawResults(pptx,slide,center.x,7.25,center.w,24.32);
drawForecast(pptx,slide,right.x,7.24,right.w,13.42);

text('Forecasting cohort by gait annotation',right.x,20.93,right.w,.60,29,{color:navy,bold:true});
drawCohortTable(pptx,slide,right.x,21.69,right.w,3.42);
section('Conclusions and Next Steps',right,25.41);
text('Current posture helped predict leg motion. Low movement was a stronger clue to abnormal gait than forecast error. The tie on unseen videos leaves the value of S-JEPA pretraining uncertain.',
  right.x,26.40,right.w,1.60,26,{valign:'top'});
text('Test transfer to new people.',right.x,28.18,right.w,.48,27,{bold:true,color:teal});
text('Hold out every video of each test participant. Compare trained and random features at similar camera angles.',
  right.x,28.76,right.w,1.04,25,{valign:'top'});
text('Test movement within each condition.',right.x,29.98,right.w,.49,27,{bold:true,color:teal});
text('Mark foot contacts in Parkinson’s gait and knee motion after stroke. Test whether timing or asymmetry adds information at similar walking speeds.',
  right.x,30.56,right.w,1.30,25,{valign:'top'});

rule(.95,32.05,46.10);
text('References',.95,32.20,4.30,.47,25,{bold:true,color:navy});
text('[1] Ranjan et al. Computer Vision for Clinical Gait Analysis: A Gait Abnormality Video Dataset. IEEE Access (2025). doi:10.1109/ACCESS.2025.3545787',
  .95,32.76,15.7,1.00,21,{valign:'top',color:slate});
text('[2] Abdelfattah & Alahi. S-JEPA: A Joint Embedding Predictive Architecture for Skeletal Action Recognition. ECCV (2024). doi:10.1007/978-3-031-73411-3_21',
  17.20,32.76,16.54,1.00,21,{valign:'top',color:slate});
text('Acknowledgments',right.x,32.20,right.w,.47,25,{bold:true,color:navy});
text('We thank ASDRP’s Department of Computer Science and Engineering and the GAVD researchers. SCCUR 2026, San Diego State University.',
  right.x,32.76,right.w,1.00,21,{color:slate,valign:'top'});

slide.addNotes([
  'Study A is the GAVD3 study described in docs/sccur/gait-with-jepa.docx. Its 96 clips came from 18 videos. The 12 clips selected for normal-only pretraining came from one recording. The teacher supplies training targets without gait labels. During classifier evaluation, all nine test videos also contributed training clips. Three of the 21 test clips appeared in self-supervised pretraining.',
  'Study A recording identity audit: over three clip splits, learned features identified the video with 49.4% mean accuracy. Missing-joint patterns alone scored 42.5%. Uniform guessing among 18 source videos gives 5.6%. The separate exploratory ankle-phase test used 12 training clips and missed by 87.7 degrees versus 78.6 degrees for repeating the last phase. Both controls are described in the illustrated FAQ.',
  'Study B is the direct-binary experiment from notebooks 00–07. The encoder used one representation seed. The mixed adaptation model is primary. A second trained classifier read features from an encoder kept at its initial random weights. The separate staged model introduced the conditions sequentially. Two thousand paired test-recording bootstrap draws produced the intervals.',
  'The coordinate-regression benchmark in notebooks 20–26 used 91 GAVD recording groups, 456 annotated sequences and 2,815 windows in five source-grouped folds. Its trajectory and forecast and ranking values come from the saved draft03 release. Gait categories are dataset annotations. GAVD source IDs do not verify independent participants.'
].join('\n\n'));
pptx.writeFile({fileName:path.join(__dirname,'..','SCCUR-JEPA.pptx')});
