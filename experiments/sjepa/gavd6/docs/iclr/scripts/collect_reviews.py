"""Collect eight fixed-rubric dimensions from independent review tables."""
from pathlib import Path
import re,json,csv
ROOT=Path(__file__).resolve().parents[1]
WEIGHTS=[20,20,15,15,10,10,5,5]
DIMS=['Relevance and contribution','Claim accuracy and evidence support','Evaluation and statistical rigor','Scientific insight and positioning','Reproducibility','Clarity and narrative','Figures','Submission fit']
records=[]
for v in range(1,8):
 panels=[]
 for role in ['world-models','evidence-statistics','editor']:
  p=ROOT/f'reviews/v{v:02d}-{role}.md'
  if not p.exists():continue
  rows=[]
  for ln in p.read_text().splitlines():
   cells=[x.strip().replace('**','') for x in ln.split('|')[1:-1]]
   if len(cells)>=4 and re.fullmatch(r'\d+%',cells[1]) and re.fullmatch(r'\d+(\.\d+)?',cells[2]):
    rows.append({'dimension':cells[0],'weight':float(cells[1][:-1]),'score':float(cells[2]),'reason':cells[3]})
  assert len(rows)==8,(p,len(rows))
  assert [r['weight'] for r in rows]==WEIGHTS,p
  score=sum(r['weight']*r['score']/10 for r in rows)
  panels.append({'role':role,'path':str(p.relative_to(ROOT)),'dimensions':rows,'weighted_total':score})
 if len(panels)==3:
  scores=[sum(p['dimensions'][i]['score'] for p in panels)/3 for i in range(8)]
  records.append({'version':v,'panels':panels,'dimension_scores':scores,'weighted_total':sum(w*x/10 for w,x in zip(WEIGHTS,scores))})
(ROOT/'reviews/scores.json').write_text(json.dumps({'aggregation':'Arithmetic mean of three independent panels per dimension; the evidence/statistics panel separately covers two requested perspectives. Scores are artifact-quality assessments, not acceptance probabilities.','weights':dict(zip(DIMS,WEIGHTS)),'versions':records},indent=2)+'\n')
with (ROOT/'reviews/scores.csv').open('w') as f:
 writer=csv.writer(f);writer.writerow(['version',*DIMS,'weighted_total_100']);writer.writerows([[r['version'],*[round(x,4) for x in r['dimension_scores']],round(r['weighted_total'],4)] for r in records])
print([(r['version'],round(r['weighted_total'],2)) for r in records])
