// Central classifier results, sized for a 20.4 × 24.4 inch panel.
// Study A: exact exp5 47/21 split in the GAVD3 FAQ evidence.
// Study B: clean, released, direct-binary classifiers; one representation seed.
const fs = require('node:fs');
const path = require('node:path');
const probeResults = require('./probe-results.cjs');
const root = path.resolve(__dirname, '../../..');

function records(file) {
  const text = fs.readFileSync(path.join(root, file), 'utf8').trim();
  const [header, ...lines] = text.split(/\r?\n/);
  const names = header.split(',');
  return lines.map(line => {
    const fields = (line.match(/("(?:[^"]|"")*"|[^,]*)(?:,|$)/g) || [])
      .map(s => s.replace(/,$/, '').replace(/^"|"$/g, '').replace(/""/g, '"'));
    return Object.fromEntries(names.map((name, i) => [name, fields[i]]));
  });
}

function getEvidence() {
  const dir = 'experiments/penny/docs/faq/evidence/original/';
  const rows = name => records(dir + name);
  const exact = rows('leakage_audit.csv').find(r => r.split === 'exp5_exact_video_confounded');
  const n = +exact.test_sequences;
  const comparisons = rows('exp5_comparison.csv');
  const hand = comparisons.find(r => r.system.startsWith('exp5 handcrafted'));
  const learned = comparisons.find(r => r.split === exact.split);
  const missing = rows('missingness_only_classifier_metrics.csv').find(r => r.split === exact.split);
  const majority = rows('five_class_classification_report.csv').find(r => r[''] === 'myopathic');
  const counts = [learned, hand, missing].map(r => Math.round(+r.accuracy * n)).concat(+majority.support);
  const study = JSON.parse(fs.readFileSync(path.join(root,
    'experiments/aaai-worldmodels/docs/research_results.json'), 'utf8')).binary;
  if (n !== 21 || counts.join() !== '13,16,7,7' || study.test_reports.jepa_mixed.n_sources !== 17)
    throw Error('Unexpected study endpoint. Review the annotations before updating this panel.');
  return { n, counts, exact, study };
}

module.exports = function researchResults(p, s, x, y, w, h) {
  const {n, counts, exact, study} = getEvidence();
  const identity = records('experiments/penny/docs/faq/evidence/later/07_video_identity_probe.csv')
    .find(r=>r.representation.startsWith('missingness'));
  const percent = v=>(Number(v)*100).toFixed(1)+'%';
  const sx=w/20.4, sy=h/24.4, u=Math.min(sx,sy);
  const navy='1F3864', teal='00796D', slate='526575', ink='182F39', grid='CFD9DE', blank='E4EAED';
  const X=a=>x+a*sx, Y=b=>y+b*sy;
  function text(str,a,b,ww,hh,pt,opts={}) {
    s.addText(str,{x:X(a),y:Y(b),w:ww*sx,h:hh*sy,margin:0,fontFace:'Calibri',fontSize:pt*u,
      align:'left',valign:'mid',color:ink,breakLine:false,paraSpaceAfterPt:0,...opts});
  }
  function rule(a,b,ww,color=grid,pt=1.35) {
    s.addShape(p.ShapeType.line,{x:X(a),y:Y(b),w:ww*sx,h:0,line:{color,width:pt*u}});
  }
  function dot(a,b,r,fill,name){
    s.addShape(p.ShapeType.ellipse,{x:X(a)-r*u,y:Y(b)-r*u,w:2*r*u,h:2*r*u,
      objectName:name,fill:{color:fill},line:{color:fill,transparency:100}});
  }
  function heading(number,str,b) {
    text(number,0,b,.9,.82,44,{bold:true,color:teal});
    text(str,1.06,b,19.31,.82,43,{bold:true,color:navy});
  }
  function band(str,a,b,ww) {
    text(str,a,b,ww,.42,24,{bold:true,color:slate,charSpacing:1.0});
  }
  function chip(a,b,id) {
    const sourceA=id[0]==='A', color=sourceA?navy:teal;
    s.addShape(sourceA?p.ShapeType.rect:p.ShapeType.roundRect,{x:X(a),y:Y(b),w:1.02*sx,h:.47*sy,
      radius:.06*u,rectRadius:.06*u,fill:{color:sourceA?'EAF0F8':'E5F2F0'},line:{color,width:1.2*u}});
    text(id,a,b,1.02,.47,23,{bold:true,color,align:'center'});
  }

  heading('1','Can learned features identify five gait labels?',.05);
  text('Study A trained a classifier on 47 clips to distinguish five gait labels. Each row shows how often a method correctly labeled the same 21 test clips.',0,1.11,20.31,1.12,30);
  band('METHOD',0,2.56,6.20);
  band('DARK DOTS = CORRECTLY LABELED CLIPS',6.43,2.56,9.25);
  band('CORRECT',16.90,2.56,3.46);
  rule(0,3.12,20.4);
  const methods=['Frozen S-JEPA features','Handcrafted gait features','Missing-joint patterns','Most frequent training label'];
  methods.forEach((label,i)=>{
    const top=3.37+i*.82, color=i===0?teal:slate;
    text(label,0,top,6.25,.57,27,{bold:i===0,color});
    for(let j=0;j<n;j++)dot(6.60+j*.43+Math.floor(j/7)*.18,top+.29,.132,
      j<counts[i]?color:blank,label+' '+(j<counts[i]?'correct':'incorrect')+' clip');
    text(counts[i]+' / '+n+'  ('+(100*counts[i]/n).toFixed(1)+'%)',16.89,top,3.49,.57,27,{bold:true,color});
  });
  text('The handcrafted comparison used a separate feature pipeline on the same clips.',0,6.92,20.26,.67,25,{color:slate});
  rule(0,7.80,20.4);
  text('Why does the way we split the videos matter?',0,8.04,20.35,.64,34,{bold:true,color:navy});
  text('In this example, A1 and A2 are clips from video A. B1 and B2 come from video B.',0,8.85,20.3,.54,26);
  function split(a,title,train,test) {
    text(title,a,9.64,9.89,.54,29,{bold:true,color:navy});
    text('Train',a,10.37,1.18,.48,24,{bold:true,color:slate});
    train.forEach((id,i)=>chip(a+1.32+i*1.20,10.37,id));
    text('Test',a+4.36,10.37,1.08,.48,24,{bold:true,color:slate});
    test.forEach((id,i)=>chip(a+5.51+i*1.20,10.37,id));
  }
  split(0,'Study A · Split clips',['A1','B1'],['A2','B2']);
  split(10.48,'Study B · Split whole videos',['A1','A2'],['B1','B2']);
  text('All '+exact.shared_train_test_videos+' test videos also supplied classifier training clips, and '+exact.test_sequences_seen_in_ssl_pretraining+' test clips appeared in pretraining. In a separate check, missing joints alone identified the source video '+percent(identity.mean_video_id_accuracy)+' of the time. Uniform guessing would average '+percent(identity.chance_1_of_n_videos)+'.',0,11.20,20.32,1.28,25);
  rule(0,12.64,20.4,navy,2.0);

  heading('2','What transfers to an unseen recording?',13.00);
  text('Study B asked a different question: could a classifier distinguish normal from abnormal gait? It used 405 windows from 89 recordings. Whole videos stayed together: 53 for training, 19 for choosing settings and 17 for testing.',0,14.00,20.25,1.16,29);
  text('Balanced accuracy gives each class equal weight. The mixed model correctly classified 6 of 6 normal recordings and 7 of 11 abnormal recordings.',0,15.32,20.25,.89,26,{color:slate});
  text('Mixed training combined gait types. Staged training introduced them one at a time.',0,16.37,20.25,.53,25,{color:slate});
  probeResults(p,s,X(0),Y(17.05),20.4*sx,5.04*sy);
  rule(0,22.28,20.4);
  const mixed=study.test_reports.jepa_mixed;
  const pct=r=>(r.binary_balanced_accuracy*100).toFixed(1)+'%';
  const interval=study.paired_against_rf.jepa_mixed.ci95.map(v=>(100*v).toFixed(1).replace('-','−'));
  text('Interpretation',0,22.42,20.22,.48,30,{bold:true,color:navy});
  text('Trained and random features tied at '+pct(mixed)+' on new recordings.',0,23.05,20.32,.49,29,{bold:true,color:teal});
  text('Mixed S-JEPA scored 3.0 points above the random forest. The paired 95% interval ranged from '+interval[0]+' to +'+interval[1]+' points.',0,23.66,20.32,.65,25,{color:ink});
};
