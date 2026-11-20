// Notebook 20 groups windows by original annotation, sequence ID and recording.
// Cross-check the publication table against the saved sequence-level identity rows.
const fs=require('node:fs'), path=require('node:path');
module.exports=function notebookCohort(p,s,x,y,w,h) {
  const dir=path.join(__dirname,'../../../experiments/aaai-worldmodels/docs/draft03');
  const read=name=>fs.readFileSync(path.join(dir,name),'utf8').trim().split(/\r?\n/).slice(1).map(r=>r.split(','));
  const cohort=new Map(read('cohort.csv').map(([label,n,windows,rec])=>[label,[+n,+windows,+rec]]));
  const identity=read('sequence_level_forecast_errors.csv');
  const names=[['normal','Normal'],['cerebral palsy','Cerebral palsy'],['myopathic','Myopathic'],['parkinsons','Parkinson’s'],['stroke','Stroke']];
  const counts=(rows,col)=>new Set(rows.map(row=>row[col])).size;
  const rows=names.map(([key,name])=>{
    const values=cohort.get(key), members=identity.filter(row=>row[2]===key);
    if(counts(members,1)!==values[0] || counts(members,0)!==values[2] || counts(members,3)!==values[2]) throw Error('Unexpected identities in '+key);
    return [name,...values];
  });
  const total=[counts(identity,1),rows.reduce((sum,row)=>sum+row[2],0),counts(identity,3)];
  if(total.join(',')!=='456,2815,91' || counts(identity,0)!==91) throw Error('Unexpected cohort totals');
  rows.push(['Total',...total]);
  const widths=[.34,.215,.19,.255].map(q=>q*w), size=Math.min(w/11.41,h/4.8);
  const header=['Gait annotation','Sequences','Windows','Recordings'];
  const cells=[header,...rows].map((row,i)=>row.map((value,j)=>({
    text:typeof value==='number'?value.toLocaleString('en-US'):value,
    options:{align:j===0?'left':'right',bold:i===0||i===6,color:i===0?'FFFFFF':'1F3864',
      fill:{color:i===0?'1F3864':i===6?'E9F1F3':i%2===1?'FFFFFF':'F4F7F8'}}
  })));
  s.addTable(cells,{x,y,w,h,colW:widths,rowH:h/7,fontFace:'Calibri',fontSize:Math.max(21,24*size),
    margin:[.025,.10,.025,.10],valign:'middle',border:{type:'solid',pt:.6,color:'D2DDDF'},autoPage:false});
};
