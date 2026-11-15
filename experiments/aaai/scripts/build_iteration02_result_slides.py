"""Build the measured iteration02 deck from an explicit verified run.

The planning deck and draft01 deck are immutable archives. This builder only
reads evidence and writes presentations; it never fits models or changes scores.
"""
from pathlib import Path
import argparse, json, hashlib, math, re, subprocess, sys
from html import escape
from urllib.parse import unquote, urlsplit
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import build_iteration02_slides as design

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from aaai_gait.iteration02_notebook_support import EvidenceRun, prediction_example, file_hash
OUT = BASE / 'slides'
IMG = OUT / 'images'
SHARED = BASE / 'images/iteration02'
COMPLETION_AUDIT = BASE / 'docs/draft03/advisory/completion_audit_v1'
Canvas, cards = design.Canvas, design.cards
INK, TEAL, BLUE, ORANGE, MUTED, PALE = design.INK, design.TEAL, design.BLUE, design.ORANGE, design.MUTED, design.PALE
SOURCES = dict(design.SOURCES)
SOURCES.update({
 'protocol':('Executed protocol','../docs/iteration02/protocol.md'),
 'current_results':('Verified study results','../docs/iteration02/results.md'),
 'limits':('Interpretation limits','../docs/iteration02/limitations.md'),
 'draft02':('Current manuscript and readiness','../docs/draft02/README.md'),
 'nb12':('Independent notebooks 12–17','../12_observation_population_and_video_context.ipynb'),
 'external_results':('Independent participant study','../docs/iteration02/external_replication_results.md'),
 'draft03':('Accessible manuscript and technical notes','../docs/draft03/README.md'),
 'completion_audit':('Post-outcome paired completion audit','../docs/draft03/advisory/completion_audit_v1/README.md'),
})


def frame(s, n, total):
 c=Canvas();c.box(0,0,18,720,TEAL,TEAL,0);c.text(60,38,s['tag'],15,TEAL,True)
 size=min(34,34*1155/max(1155,design.width(s['title'],34,True)))
 if size<29:raise ValueError('Shorten title: '+s['title'])
 c.text(60,87,s['title'],size,INK,True)
 if c.para(60,142,s['lede'],1150,24,MUTED)>181:raise ValueError('Shorten lede: '+s['lede'])
 c.box(45,638,1190,47,PALE,PALE,8)
 c.text(60,668,s['takeaway'],min(19,19*1150/max(1150,design.width(s['takeaway'],19))),INK)
 c.text(60,704,'Gait prediction | A guide to the measured study',15,MUTED)
 c.text(1220,704,f'{n:02d} / {total}',16,MUTED,align='right')
 return c


def table(c, headers, rows, widths, size=23, row_h=65):
 c.table(headers, [[str(v) for v in r] for r in rows], widths, y=214, row_h=row_h, size=size)


def forest(c, rows, label='Paired AUROC difference', limits=(-.25,.16)):
 left,right=550,1160;lo,hi=limits
 x=lambda v:left+(float(v)-lo)/(hi-lo)*(right-left)
 c.line(x(0),215,x(0),545,MUTED,1,':')
 for i,(name,point,ci) in enumerate(rows):
  y=255+i*65;c.text(60,y+6,name,23,INK)
  c.line(x(ci[0]),y,x(ci[1]),y,TEAL,3);c.ax.scatter([x(point)],[720-y],color=TEAL,s=43,zorder=4)
  c.text(1210,y+6,f'{point:+.3f}',20,TEAL,True,align='right')
 for tick in np.linspace(lo,hi,5):c.text(x(tick),580,f'{tick:+.2f}',18,MUTED,align='center')
 c.text((left+right)/2,613,label,20,MUTED,align='center')


def points(c, values, names, ylabel, limits=None):
 # Values are measured numbers; each column is a named group, each row a fold.
 arr=np.asarray(values,float);lo,hi=limits or (float(np.nanmin(arr)),float(np.nanmax(arr)))
 pad=(hi-lo)*.15 or .01;lo-=pad;hi+=pad
 xx=np.linspace(300,1110,arr.shape[1]) if arr.shape[1]>1 else np.array([700]);yy=lambda v:560-(float(v)-lo)/(hi-lo)*260
 if lo<=0<=hi:c.line(175,yy(0),1160,yy(0),MUTED,1,':')
 for i,row in enumerate(arr):
  color=[TEAL,BLUE,ORANGE,'#6a51a3'][i%4]
  for j,value in enumerate(row):c.ax.scatter([xx[j]+(i-1.5)*8],[720-yy(value)],color=color,s=45)
  c.text(235+i*235,260,f'Fold {i+1}',19,color,True)
 for j,name in enumerate(names):c.text(xx[j],600,name,23,INK,True,align='center')
 for tick in np.linspace(lo,hi,5):c.text(160,yy(tick)+6,f'{tick:.4g}',18,MUTED,align='right')
 c.text(220,208,ylabel,21,MUTED)


def prediction(c, row, arrays):
 target=arrays['Observed target'];allv=np.concatenate(list(arrays.values()))
 labels=list(arrays);colors=[INK,TEAL,BLUE,ORANGE]
 for axis in range(2):
  x0=135+axis*590;y0=540;lo=allv[:,axis].min();hi=allv[:,axis].max();pad=max((hi-lo)*.15,.01);lo-=pad;hi+=pad
  x=lambda k:x0+k*140;y=lambda v:y0-(v-lo)/(hi-lo)*205
  c.text(x0,294,['Horizontal coordinate','Vertical coordinate'][axis],24,INK,True)
  c.line(x0,y0,x0+450,y0,MUTED,1)
  for name,color in zip(labels,colors):
   vals=arrays[name][:,axis]
   for k in range(3):c.line(x(k),y(vals[k]),x(k+1),y(vals[k+1]),color,2)
   c.ax.scatter([x(k) for k in range(4)],[720-y(v) for v in vals],color=color,s=22)
  for k in range(4):c.text(x(k),569,str(k),19,MUTED,align='center')
  for tick in np.linspace(lo,hi,3):c.text(x0-18,y(tick)+5,f'{tick:.2f}',16,MUTED,align='right')
  c.text(x0+225,602,'Sample within the hidden block',20,MUTED,align='center')
 for i,(name,color) in enumerate(zip(labels,colors)):c.text(65+(i%2)*610,218+(i//2)*27,name.replace('Local baseline','Own-leg baseline'),19,color,True)


def residual(c,top=210):
 c.card(60,top+75,265,132,'Frozen f(O)','Retain its output.',TEAL,23)
 c.card(395,top+5,355,128,'Own-leg h(O)','Add capacity with own-leg access.',TEAL,22)
 c.card(395,top+182,355,128,'Both-leg g(O,C)','Use both legs.',BLUE,22)
 c.card(875,top+5,345,128,'Prediction: f + h','Evaluate the same target Y.',TEAL,23)
 c.card(875,top+182,345,128,'Prediction: f + g','Evaluate the same target Y.',BLUE,23)
 c.arrow(330,top+114,388,top+75);c.arrow(330,top+170,388,top+238,BLUE)
 c.arrow(755,top+69,868,top+69);c.arrow(755,top+246,868,top+246,BLUE)
 c.banner(top+350,'Benefit = loss(f + h) minus loss(f + g). Both corrections can be zero.',size=24)


def external_gain(c, values):
 c.text(160,225,'Error reduction: both-leg versus own-leg correction (%)',23,MUTED)
 for tick in [0,2,4,6]:
  y=555-tick/6*260;c.line(170,y,1170,y,'#d9e3ec',1);c.text(150,y+6,str(tick),20,MUTED,align='right')
 for fold,value in enumerate(values):
  x=290+250*fold;y=555-float(value)/6*260
  c.box(x-38,y,76,555-y,TEAL,TEAL,3);c.text(x,y-13,f'{value:.2f}%',23,TEAL,True,align='center');c.text(x,598,f'Fold {fold+1}',23,INK,True,align='center')


class Deck:
 def __init__(self):self.records=[]
 def add(self,title,lede,draw,notes='',refs=('protocol','current_results'),takeaway='',tag='MEASURED STUDY | DESIGN AND EVIDENCE'):
  self.records.append(dict(title=title,lede=lede,draw=draw,notes=notes,refs=list(refs),takeaway=takeaway,tag=tag))


def completion_appendix(d, run, external):
 """Append descriptive evidence without refitting or replacing primary results."""
 manifest=json.loads((COMPLETION_AUDIT/'audit_manifest.json').read_text())
 verification=json.loads((COMPLETION_AUDIT/'verification.json').read_text())
 assert manifest['status']=='complete_post_outcome_descriptive_only'
 assert manifest['original_runs']=={'GAVD':run.path.name,'external':external.name}
 assert verification['audit_manifest_sha256']==file_hash(COMPLETION_AUDIT/'audit_manifest.json')
 assert verification['all_output_hashes_reverified']
 for name,digest in manifest['outputs'].items():
  assert file_hash(COMPLETION_AUDIT/name)==digest, name
 summary=pd.read_csv(COMPLETION_AUDIT/'completion_summary.csv',dtype={'outer_fold':str})
 get=lambda dataset,bank:summary.loc[summary.dataset.eq(dataset)&summary.bank.eq(bank)&summary.outer_fold.eq('all')].iloc[0]
 lab=get('external_angles','U');mask_u=get('GAVD','U');mask_leg=get('GAVD','E_leg')
 errors=run.table('normal_error_table.csv')
 normal=errors.loc[errors.policy.eq('U')&errors.bank.eq('E_leg')]
 assert len(normal)==4
 own,both=normal.loss_own.mean(),normal.loss_both.mean();gain=(own-both)/own
 assert np.isclose(both,mask_leg.comparator_mean_mse)
 tag='POST-OUTCOME APPENDIX | FROZEN PREDICTIONS, NO NEW FITTING'
 refs=('completion_audit','draft03','protocol')
 def paired_completions(c):
  table(c,['Normal population','Own correction','Both-leg correction','MSE reduction'],[
   ('GAVD: 28 sources',f'{own:.6f}',f'{both:.6f}',f'{100*gain:.1f}%'),
   ('Laboratory: 128 people',f'{lab.comparator_mean_mse:.3f}',f'{lab.candidate_mean_mse:.3f}',f'{100*lab.relative_mse_reduction:.1f}%')
  ],[400,260,280,220],row_h=85,size=22)
  c.para(60,521,'GAVD uses normalized image coordinates; laboratory errors use square degrees. Absolute errors are not pooled.',1130,23,MUTED)
  c.text(60,594,'Equal input width does not equalize feature rank or effective capacity.',22,INK,True)
 d.add('The other leg helps on declared completion tasks.','The predeclared distal-leg diagnostic and independent laboratory study show positive prediction results.',paired_completions,
  notes='This appendix emphasizes completed secondary completion results after outcomes were inspected. It does not alter the primary detection endpoint, fitted models, targets or populations. Both rows compare loss(f+h) with loss(f+g), retaining the same local f. The GAVD row uses the uniform-trained reference on E_leg, not the largest gain selected across policies. Its absolute mean benefit is 0.015748, with the earlier 5,000-draw conditional interval [0.010721, 0.021300]. Percentages are differences of source or participant mean MSE divided by the comparator mean, not means of individual percentages. The observations are withheld measured poses or angles, not independent ground truth during real occlusion. The feature families and numerical ranks differ, so this is useful completion under fitted predictor families rather than isolated neural information.',
  refs=refs,takeaway='19.5% and 5.2% compare the both-leg correction with the own-leg correction.',tag=tag)
 labfolds=summary.loc[summary.dataset.eq('external_angles')&summary.outer_fold.ne('all')].sort_values('outer_fold')
 def laboratory_summary(c):
  c.card(60,211,360,180,f'{100*lab.relative_mse_reduction:.2f}% lower MSE','The own-correction mean falls from 201.038 to 190.584 square degrees.',TEAL,22)
  c.card(460,211,360,180,'Conditional 95% interval',f'[{100*lab.relative_ci_low:.2f}, {100*lab.relative_ci_high:.2f}]% from paired participant resampling.',BLUE,22)
  c.card(860,211,360,180,f'{int(lab.units_with_lower_candidate_mse)} of {int(lab["count"])} people','Completion error decreases. This is not a change in health.',TEAL,22)
  c.table(['Held-out fold','Fold 0','Fold 1','Fold 2','Fold 3'],[['MSE reduction',*[f'{100*z:.2f}%' for z in labfolds.relative_mse_reduction]]],[300,215,215,215,215],y=437,row_h=63,size=23)
  c.text(60,601,'Every participant contributes once to the held-out summary.',22,MUTED)
 d.add('The laboratory benefit is paired and consistent.','All four held-out fold means favor the both-leg predictor under a separately fitted angle-based measurement.',laboratory_summary,
  notes='The post-outcome audit uses the frozen predictions for 128 healthy participants, with 32 in each original outer fold. It resamples participants jointly across compared methods within folds, fixes the original normal-count fold weights, and uses 1,000 draws with seed 42. The exact relative interval is [3.5530025847, 6.6975555170] percent. The absolute reduction is 10.454 square degrees, with conditional interval [7.154, 13.387]. Error decreases for 109 participants. Shared training data across fits mean that the four fold means are not independent replications. These intervals omit fitting, seed and protocol-selection uncertainty. The separate laboratory study is method replication, not transfer of GAVD weights or validation across all disease families.',
  refs=refs+('external_results',),takeaway='The gain concerns prediction error; it is not a clinical benefit or an AUROC gain.',tag=tag)
 def mask_summary(c):
  table(c,['Fixed evaluation task','MSE reduction','Conditional 95% interval','Sources improved'],[
   ('Uniform bank (U)',f'{100*mask_u.relative_mse_reduction:.1f}%',f'[{100*mask_u.relative_ci_low:.1f}, {100*mask_u.relative_ci_high:.1f}]%',f'{int(mask_u.units_with_lower_candidate_mse)} / 28'),
   ('Distal-leg bank (E_leg)',f'{100*mask_leg.relative_mse_reduction:.1f}%',f'[{100*mask_leg.relative_ci_low:.1f}, {100*mask_leg.relative_ci_high:.1f}]%',f'{int(mask_leg.units_with_lower_candidate_mse)} / 28')
  ],[390,230,320,220],row_h=85,size=22)
  c.para(60,521,'A and U training are evaluated on identical targets within each bank. The uniform-task interval includes zero.',1130,23,MUTED)
  c.text(60,594,'The frozen anomaly fusion still loses to the preserved local channel.',22,INK,True)
 d.add('Anatomical masking helps the focused gap task.','This is A versus U training for the bilateral predictor. It is a different contrast from adding opposite-leg context.',mask_summary,
  notes='The paired audit checks identical normal sources, window metadata, physical target keys, mask-draw counts, evaluation mask hashes and distinct-patch support for A and U within each bank. On E_leg, MSE changes from 0.065020 to 0.060009; the relative interval is [6.3559980507, 9.0031877518] percent, with 27 of 28 sources improved and positive means in all four folds. On U, the relative interval is [-0.9546471486, 2.5838432525] percent; 16 sources improve and one fold mean is worse. The new intervals are post-outcome, unadjusted descriptive analyses of frozen outputs. They condition on training recipes, selected regularization, validity-dependent target exposure and the current sources. This supports task-specific completion specialization, not a new mask family, general mask superiority, a JEPA pretraining result or a neurological mechanism. Both banks and the negative original primary detection results remain visible.',
  refs=refs,takeaway='The 7.7% compares training masks on E_leg; the unchanged uniform-task advantage is uncertain.',tag=tag)


def build(run, external):
 d=Deck();m=run.table('metrics.csv');registry=run.table('source_registry.csv');paired=run.json('paired_intervals.json');adequacy=run.table('adequacy.csv');normal=run.json('normal_test_benefit.json');timing=run.json('timing_summary.json')
 ext=json.loads((external/'metrics.json').read_text());extpair=json.loads((external/'paired_effects.json').read_text());matched=json.loads((external/'age_sex_sensitivity.json').read_text());exterr=pd.read_csv(external/'heldout_prediction_error.csv');complete=json.loads((external/'completion.json').read_text())
 f=lambda method,endpoint='focal',policy='U',bank='U':float(m.loc[m.policy.eq(policy)&m.bank.eq(bank)&m.endpoint.eq(endpoint)&m.method.eq(method)&m.metric.eq('auroc'),'value'].iloc[0])
 find=lambda first,second,endpoint='focal',contrast='method':next(z for z in paired if z['bank']=='U' and z['endpoint']==endpoint and z['first']==first and z['second']==second and z['contrast']==contrast and (contrast!='method' or z.get('policy')=='U'))
 p=find('S','M');e=extpair['M'];focal=int(registry.is_focal.sum());total=len(registry);normals=int(registry.binary_label.eq(0).sum())
 d.add('Useful prediction does not guarantee better anomaly detection.','Real GAVD video and an independent laboratory dataset both favor local movement over the fixed relational fusion.',lambda c:cards(c,[('GAVD focal task',f"M = {f('M'):.3f}; S = {f('S'):.3f}. The paired difference favors M."),('Independent participants',f"M = {ext['M']['weighted_auroc']:.3f}; S = {ext['S']['weighted_auroc']:.3f}. The same limitation remains."),('Scientific lesson','Predictive dependence and anomaly separation must be tested separately.')],size=24),notes='S is the fixed average of local movement M and calibrated relational evidence R. These are within-fold pair-count-weighted AUROCs. The independent experiment refits on laboratory angles and is not a transfer of GAVD weights. The negative fusion outcome is retained.',takeaway='The evidence supports a controlled measurement study, not a state-of-the-art detector.')
 d.add('Two different datasets answer complementary questions.','The source-video study tests uncontrolled projected pose. The laboratory study tests a separately fitted participant-level measurement.',lambda c:table(c,['Evidence','Population','Measurement'],[('GAVD focal',f'{focal} sources; {normals} normal','Normal versus four requested families'),('GAVD broad',f'{total} sources; {normals} normal','All non-normal folders'),('Independent laboratory',f"{complete['participants']} participants",f"{complete['normal']} controls; {complete['stroke']} stroke")],[370,360,430],row_h=85),notes='GAVD source IDs are not verified unique people. The laboratory cohort contains distinct participant IDs. Never pool the two datasets or compare their absolute AUROCs as an effect of sensor quality.',takeaway='Repeated windows increase measurement detail, not independent sample size.')
 d.add('Start with a hidden ankle and a concrete question.','If the left ankle disappears for four frames, does the right leg help after the visible left leg already constrains the answer?',lambda c:cards(c,[('Own-joint history','Nearby ankle observations can interpolate a smooth movement.'),('Own-leg geometry','The visible knee and foot can constrain where the ankle should be.'),('Other-leg context','The opposite leg may add information beyond geometry and extra capacity.')],size=24),notes='This is an illustration of conditional prediction, not a clinical example or an empirical finding. All branches predict the identical coordinate target under the same declared body reference.',refs=('neuro','protocol'),takeaway='The meaningful comparison adds information after a competent local explanation.',tag='CONCEPTUAL ILLUSTRATION')
 d.add('Neurological reasoning motivates observations, not diagnoses.','Different impairments and adaptations can produce similar projected motion, while one diagnostic label can contain varied gait patterns.',lambda c:table(c,['Family','Plausible movement consequence'],[('Stroke','Weakness and altered selective control can produce stable compensations.'),('Parkinsonian labels','Small excursion or reduced arm swing can coexist with regular timing.'),('Cerebral palsy','Developmental motor limitations can alter posture and joint coordination.'),('Myopathic labels','Muscle-related limitations can change trunk and pelvic compensation.')],[330,830],row_h=81,size=23),notes='These are physiology-informed hypotheses supported by the linked primary studies, not diagnoses inferred from GAVD. Medication, lesion location, strength, EMG and forces are unavailable. Myopathy is not one central neurological lesion.',refs=('neuro','stroke','pd','cp','dmd'),takeaway='A stable, predictable compensation can still be abnormal.')
 d.add('Preserve amplitude, asymmetry and relative information.','Regular timing can accompany small steps. A changed relationship can accompany ordinary excursion.',lambda c:design.rationale(c,210),notes='Illustrative patterns explain why a local channel and a relational channel need separate evaluation. Independent phase normalization could erase bilateral timing differences; per-limb amplitude normalization could erase a phenotype.',refs=('neuro','protocol'),takeaway='The primary representation keeps raw time and a common body scale.',tag='CONCEPTUAL ILLUSTRATION')
 d.add('Close prior work limits the novelty claim.','Normal-only gait modeling, dependence scores and anatomical masks already exist.',lambda c:cards(c,[('Gait anomaly methods','GenGait, FSGait and SeeKer limit claims of a new normal-gait learning idea.'),('Dependence and masks','Beyond Marginals and SLiM limit claims of new dependency scoring or body tubes.'),('Our narrower contribution','A controlled test separates prediction benefit, support and detection value.')],size=24),notes='The manuscript should credit the closest primary work and avoid a novel-neural-world-model claim. The possible contribution is the experimental distinction and falsification controls, especially its independent participant replication.',refs=('gengait','fsgait','seeker','marginals','slim','neuro'),takeaway='A new name for a known ingredient is not the contribution.')
 counts=registry.groupby('five_class_label').size();countrows=[(k,int(counts.get(k,0))) for k in ['normal','parkinsons','stroke','cerebral palsy','myopathic']]
 d.add('The focal labels contain small disease-family samples.','Use every eligible requested family, while preserving ambiguous labels and the broad secondary endpoint.',lambda c:table(c,['Unambiguous supervised label','Sources'],countrows,[900,260],row_h=64),notes=f'The five-class eligible total is {sum(v for _,v in countrows)}. The focal binary total is {focal}; one focal source has ambiguous condition membership and is excluded from the five-class task. The full annotation inventory has348source IDs and347known recording clusters, but the duplicate pair fails primaryQC. Retained304IDs are304recording groups.',takeaway='Binary anomaly detection and five-class classification use different declared eligibility rules.')
 d.add('Trace annotations through local media and cached pose.','The refreshed condition CSVs determine labels. Reusable pose caches require matching media and extraction provenance.',lambda c:cards(c,[('Annotations','Read data/gavd and preserve CSV hashes and source-to-sequence mappings.'),('Video and pose','Keep downloads under data/videos-gavd. Display actual intervals and observed joints.'),('Cohort artifact','Record exclusions, missingness, frame times, body scale and source roles.')],size=24),notes='The notebooks reload current CSVs and assert agreement with the saved manifest. Local source previews remain under data/videos-gavd. Invalid coordinates stay missing; a visually plausible pose can still track the wrong person.',refs=('data','protocol','nb12'),takeaway='Availability and pose quality are part of the result population.')
 d.add('Short windows are useful but do not guarantee gait cycles.','The primary 32 samples at 15 Hz span 31 intervals, about 2.07 seconds. The timing parent uses 48 samples.',lambda c:cards(c,[('Primary observation','Require four distinct physical target patches for every lower-leg readout.'),('Longer intervals','Longer duration can improve cycle estimates while excluding short videos.'),('Selection boundary','Support filtering can alter severity, viewpoint and activity composition.')],size=24),notes='Repeated masks and overlapping windows cannot create new physical target support. The earlier read-only96-frame candidate analysis retained only12normal sources and was not a model result. No validated normal contact-event phase score is available.',refs=('protocol','data','selection'),takeaway='The actual supported population matters more than the number of extracted windows.')
 d.add('Mask 16 approved joints and read the lower eight.','The allowlist is fixed and deduplicated. The score does not quietly introduce other landmarks.',lambda c:table(c,['Anatomical role','BlazePose 33 indices'],[('Shoulders, elbows and wrists','11,12,13,14,15,16'),('Shared-reference hip context','23,24'),('Primary lower-leg readout','25,26,27,28,29,30,31,32')],[650,510],row_h=85),notes='The notebooks include every keypoint in numerical order with its name and feature role. The shared normalization reference Q already uses body measurements. Conditional information-access claims must acknowledge Q.',takeaway='Mask restrictions and score readouts are separate design decisions.')
 d.add('Source roles separate fitting, calibration and evaluation.','Each outer fold holds out seven normals. Remaining normal sources split into 16 fitting and 5 reference sources.',lambda c:cards(c,[('Fit and develop','Fit local models and corrections on normal sources with three internal folds.'),('Calibrate R','Use five separately reserved normals after model settings are fixed.'),('Evaluate','Hold out all source windows together. Keep label-based classifiers separate.')],size=24),notes='All28normal sources had earlier exploratory use. New folds do not create independent confirmation. Unknown repeated identities across different videos remain unresolved. The standalone laboratory dataset strengthens a separate measurement test.',takeaway='The new study remains exploratory despite strict within-run separation.')
 d.add('The baseline must work before added context is interpreted.','Known-answer fixtures and competent local prediction distinguish estimator failure from absence of a relationship.',lambda c:cards(c,[('Known dependence','Recover a constructed dependency with the allowed context.'),('Conditional independence','Do not invent a benefit when the extra input contains no target information.'),('Access checks','Hidden inputs stay hidden before features; zero correction retains the baseline.')],size=24),notes='Software fixtures are explicitly synthetic checks, not gait efficacy. The real experiment then requires normal-development adequacy. A failed relational gate sets R=0 and retains the fold.',takeaway='A poor predictor cannot tell us that human gait lacks coordination.')
 d.add('Retain the local prediction while adding a correction.','The primary benefit compares own-leg and both-leg corrections with matched input width and the same frozen baseline.',lambda c:residual(c,210),notes='The diagram shows f+g for simplicity. The primary contrast is loss(f+h) minus loss(f+g), where h receives own-leg context only. g receives own and opposite context. Both have a zero option. All predict identical Y under fixed Q.',takeaway='Extra model capacity must not masquerade as extra information.',tag='IMPLEMENTED ACCESS DESIGN | ILLUSTRATION')
 d.add('Three masks change what disappears together.','Every recipe hides 64 of 128 tokens without reading motion intensity, confidence, condition or loss.',lambda c:design.mask_families(c,210),notes='U is uniform. J selects one joint uniformly and removes all eight temporal patches, then fills56tokens. A selects an anatomical group by width/16, removes an aligned eight-token core and fills56tokens. These illustrative draws are not model outputs.',refs=('mask_method','mamp','sjepa','slim','protocol'),takeaway='The comparison is a declared adaptation, not a reproduction of motion-aware S-JEPA.',tag='IMPLEMENTED MASK RULES | ILLUSTRATIVE DRAWS')
 d.add('Exact budgets and token marginals are matched.','The structured core has eight tokens. Its width determines the number of consecutive time patches.',lambda c:table(c,['Group','Indices','Width','Patches'],[('Torso','11,12,23,24','4','2'),('Left / right arm','13,15 /14,16','2','4'),('Left distal leg','25,27,29,31','4','2'),('Right distal leg','26,28,30,32','4','2')],[300,420,180,260],row_h=76),notes='For every token P(core)=1/16 and P(hidden)=1/16+(15/16)*(56/120)=1/2. This matches first-order probabilities, not task difficulty or valid-target exposure. Starts are0,2,4,6forwidth4and0,4forwidth2; no wrapping.',takeaway='Equal marginal masking probability does not equal equal missing-context difficulty.')
 d.add('Evaluate all training arms on the same frozen bank.','The uniform reference and distal-core challenge are separate tasks with separate references.',lambda c:design.mask_comparison(c,215),notes='E_U has16uniform masks, seed17. E_leg balances two sides and four starts twice,seed117. Every mask hides64tokens, but E_leg scores only its8coretargets. CompareU/J/Awithinbank, never native task against native task.',takeaway='A training-mask effect requires a fixed evaluation question.',tag='IMPLEMENTED COMPARISON DESIGN')
 d.add('Only the correction masks change in the core ablation.','A common U-trained f is frozen across U, J and A corrections. Normal development chooses the small ridge and gate grid.',lambda c:cards(c,[('Matched core','Use shared targets, output dimension and baseline with own/both corrections.'),('Effective support','Draw first, filter invalid observations afterward and retain exposure records.'),('Separate JEPA factor','The encoder-mask ablation is deferred. Actual JEPA probes use a separate declared recipe.')],size=24),notes='Do not claim that the completed UJA coordinate study proves a best JEPA mask. VICReg discourages collapse, not disease overlap. Stochastic positional embeddings and stochastic latent variables remain deferred.',takeaway='The completed result isolates correction-mask structure.')
 grouped=adequacy.groupby(['outer_fold','policy'])[['own','both']].mean();vals=np.array([[grouped.loc[(i,k),'own']-grouped.loc[(i,k),'both'] for k in ['U','J','A']] for i in range(4)])
 d.add('All correction policies pass normal-development gates.','This gate is evaluated on internal fitting-normal validation, not on abnormal detection performance.',lambda c:points(c,vals,['U','J','A'],'Own-capacity MSE minus both-leg MSE'),notes='Every outerfold/policy satisfies the operational two-of-three positive internalfold rule versus both baseline and own-capacity correction. The points show mean internal validation benefit and are post-selection diagnostics, not independent test evidence.',takeaway='Development competence permits evaluation; it does not establish test benefit.')
 normalrows=[(z['policy'],z['bank'],f"{z['mean_benefit']:.5f}",f"[{z['conditional_95_interval'][0]:.5f}, {z['conditional_95_interval'][1]:.5f}]") for z in normal]
 d.add('Held-out normal benefit depends on the observation task.','The uniform-bank U and A intervals include zero. All three distal-core challenge intervals are positive.',lambda c:table(c,['Policy','Bank','Mean benefit','Conditional 95% interval'],normalrows,[140,180,300,540],row_h=55,size=21),notes='All28outer-testnormal sources enter each result. The benefit is own-capacity loss minus both-leg loss after declared hierarchical aggregation. The diagnostic bank removes local geometry more strongly. It is not evidence of a primary-bank improvement.',takeaway='A harder missing-context task can reveal useful dependence without improving detection.')
 row,arrays=prediction_example(run)
 d.add('Inspect one actual predicted normal coordinate patch.','The example is chosen deterministically by source and target order, not by the largest benefit.',lambda c:prediction(c,row,arrays),notes=f"Actual saved source{row.source_id}, sequence{row.sequence_id}, joint{int(row.joint)}, patch{int(row.patch_index)}. Notebook13shows its corresponding local video and rawpose overlay. Allfour predictions use the same observed target. This is an illustration, not an aggregate accuracy estimate.",takeaway='The real-array example helps audit coordinate order, scale and target identity.',tag='MEASURED EXAMPLE | NOT A REPRESENTATIVE ACCURACY ESTIMATE')
 d.add('Local movement and relational departure stay separate.','M uses 16 movement descriptors. R measures departure from reserved-normal benefit profiles. S = (M + R) / 2 is fixed.',lambda c:cards(c,[('M: local movement','Preserve projected angles, excursion, velocity variation and bilateral differences.'),('R: added information','Calibrate left/right matched prediction benefits using reserved normal sources.'),('S: fixed combination','Average M and R. Do not tune the mixture after seeing abnormal outcomes.')],size=24),notes='M excludes local prediction error and fits imputation/scaling/covariance on fittingnormals. R uses reservednormals after hyperparameterselection. Stable compensations can increase or decrease benefit; R is two-sided. Five-reference estimation is fragile.',takeaway='The fixed fusion can be worse even when its relational component is meaningful.')
 d.add('Aggregate repeated targets before ranking sources.','Masks, frames and windows are correlated observations, not independent test cases.',lambda c:cards(c,[('Measurement hierarchy','Average repeated targets, then windows, sequences and sources with declared joint weights.'),('Primary AUROC','Compute within-fold rankings and weight by fixed normal × abnormal pair counts.'),('Uncertainty','Resample paired recording groups within fold and label, retaining every method score.')],size=24),notes='Intervals condition on fitted models, selected protocol and originalfoldweights. They omit training-seed and unknown-person dependence. AP must be reported byfold withprevalence. A pooled score plot is descriptive, not theprimaryAUROC.',takeaway='No missing fold or unsupported source is silently removed from a required comparison.')
 d.add('The primary fusion loses to local movement.','The negative result appears in the focal endpoint and the required broad secondary endpoint.',lambda c:table(c,['Endpoint','Sources','M AUROC','S AUROC'],[('Focal primary',focal,f"{f('M'):.3f}",f"{f('S'):.3f}"),('Broad secondary',total,f"{f('M','broad'):.3f}",f"{f('S','broad'):.3f}")],[410,210,270,270],row_h=100),notes='The focal endpoint was selected for the new exploratory iteration after reviewing earlier broadresults. It must not be described as preregistered independent confirmation. Both endpoints use exactly the same frozen source scores.',takeaway='Keep this result even though it contradicts the intended improvement.')
 d.add('Paired uncertainty rules out an established fusion gain.','Conditional intervals favor M. The apparent anatomical-mask increase is not established on the primary task.',lambda c:forest(c,[("GAVD focal: S minus M",p['observed'],p['conditional_95_interval']),('GAVD broad: S minus M',find('S','M','broad')['observed'],find('S','M','broad')['conditional_95_interval']),('Focal: A minus U',find('A','U',contrast='mask_policy')['observed'],find('A','U',contrast='mask_policy')['conditional_95_interval']),('Laboratory: S minus M',e['observed'],e['conditional_95_interval'])]),notes='Intervals are1000pairedbootstrap resamples with fixed originalfoldweights. Maskpolicycontrast uses S under the same uniform evaluationbank. A confidence interval crossingzero doesnotestablish equivalence or benefit.',takeaway='A larger point estimate is not a demonstrated improvement.')
 d.add('The anatomical mask does not establish a detection advantage.','Read all three recipes within the unchanged uniform evaluation task.',lambda c:table(c,['Training mask','Focal S AUROC','Broad S AUROC'],[(k,f"{f('S',policy=k):.3f}",f"{f('S','broad',k):.3f}") for k in ['U','J','A']],[460,350,350],row_h=85),notes='AminusU focaldifference ispositive butitsconditional95%interval crosseszero. AminusJ andJminusU also crosszero. Retain allarms, ratherthanselectAasthenewprimary afterscoring.',takeaway='The completed mask study provides a controlled comparison, not a best-mask claim.')
 d.add('The distal-core challenge is a different measurement task.','All retained sources support this diagnostic, but its result cannot replace the uniform reference.',lambda c:table(c,['Training mask','Focal S AUROC','Broad S AUROC'],[(k,f"{f('S',policy=k,bank='E_leg'):.3f}",f"{f('S','broad',k,'E_leg'):.3f}") for k in ['U','J','A']],[460,350,350],row_h=85),notes='Eachbankusesitsownreservednormalreference fittedwiththesamesourceidentities. Stronger normalpredictionbenefit onE_legdoesnot implysuperiordetection. Theseareprespecifieddiagnosticquestionswithseparatetargets.',takeaway='Compare within a bank and preserve the reason the bank exists.')
 baseline_names=[('M: movement 16','M'),('Richer movement 36','motion36'),('Timing waveform 8','timing8'),('Isolation Forest','isolation_forest'),('S: fixed fusion','S')]
 d.add('Inexpensive baselines remain difficult to beat.','All listed anomaly models use normal-only fitting and the same focal test sources.',lambda c:table(c,['Method','Focal AUROC','Broad AUROC'],[(name,f"{f(k):.3f}",f"{f(k,'broad'):.3f}") for name,k in baseline_names],[680,240,240],row_h=64,size=23),notes='Timing8isawaveformdescriptor, notPCIorvalidatedphysiologicalphase. Nuisance,sham,constituentandcalibratederrorcomparatorsremaininrawmetrics. The strongestlocalbaselinemustremainvisible.',takeaway='The new measurement must demonstrate value beyond simple observable motion.')
 auxiliary=run.json('auxiliary_metrics.json')
 d.add('The published-method comparator is a bounded adaptation.','The actual STG-NF code ran with a 16-joint graph, fitting-normal median imputation and 128 updates.',lambda c:table(c,['Method','Focal AUROC','Broad AUROC'],[('Adapted STG-NF',f"{auxiliary['focal']['stgnf_adapted']['weighted_auroc']:.3f}",f"{auxiliary['broad']['stgnf_adapted']['weighted_auroc']:.3f}"),('Local movement M',f"{f('M'):.3f}",f"{f('M','broad'):.3f}"),('Fixed fusion S',f"{f('S'):.3f}",f"{f('S','broad'):.3f}")],[680,240,240],row_h=85),notes='The upstreamrevisionandadaptationarelogged. Thisisnotafaithfulpublishedbenchmarkreproduction. ItsweakboundedresultcannotjustifyaSOTAclaim. Allfourfoldscompletedandcoverageismatched.',takeaway='Be precise about what was implemented and what its result can establish.')
 classifier=pd.json_normalize(run.json('five_class_metrics.json')).groupby('method')[['macro_f1','balanced_accuracy']].mean()
 d.add('Supervised condition classification is a separate task.','Real normal-only JEPA and random encoders feed supervised probes. The Random Forest uses permitted condition labels.',lambda c:table(c,['Method','Mean macro-F1','Mean balanced accuracy'],[({'learned_jepa_probe':'Learned JEPA probe','random_encoder_probe':'Random encoder probe','random_forest':'Random Forest'}[k],f"{v.macro_f1:.3f}",f"{v.balanced_accuracy:.3f}") for k,v in classifier.iterrows()],[640,260,260],row_h=85,size=22),notes='FreshJEPAandrandombackbonesareevaluatedunderthesamesourcefolds. JEPAhas512updatesperfoldandVICRegregularization. Thesearemeansoffourheldoutfoldmetrics,notclinicalintervals. Thefiveclassevaluationcontains89unambiguoussources. Learnedfeaturesdonotimprovehere.',takeaway='A completed JEPA run is not evidence that representation learning helps this task.')
 d.add('The timing control keeps the answer fixed.','A 48-frame parent supplies earlier, aligned and later distal context. No sample wraps across the boundary.',lambda c:design.timing(c,210),notes='Fixedtarget/ownwindow8:40;donors4:36and12:44. Subtractpermitteddonorhipandattachtopermittedoriginaloppositehip. PreserveQandpairedsupport. Theedit changesdistaltimingandgeometry,notapurephysiologicalphase.',takeaway='Mask and validity restrictions apply to donor and attachment hips.',tag='IMPLEMENTED TIMING DESIGN | ILLUSTRATION')
 gates=pd.DataFrame(timing['paired_support_adequacy']);passes=int(gates.paired_support_adequacy_passed.sum())
 d.add('Sparse timing support fails the normal-competence gate.','The paired-input model is worse than the own-capacity predictor even before the context is shifted.',lambda c:cards(c,[('Available observations','The timing census supports 293 sources and 27 normals at both offsets.'),('Executed diagnostic','Frozen models run on outer-test normal pairs and internal normal support.'),('Adequacy result',f'{passes} of {len(gates)} policy × fold × offset gates pass. Timing causality is unsupported.')],size=24),notes='ThecoreU-bankadequacygateandtheexactpaired-supportgatearedifferent. Sparsepairedsupport leavesaround1.75permitteddistaltokensper target. Theoperationaltwo-of-threeinternalnormalrulefailsall24checks. Modelsandscoreswerenotretuned.',takeaway='A timing response is not convincing physiological evidence when aligned inputs are inadequate.')
 summary=[z for z in timing['summaries'] if z['scope']=='outer_test_normal' and z['variant']=='timing' and z['policy']=='U']
 d.add('A measured input sensitivity survives, with a strict limit.','Earlier context increases loss slightly. Later-context uncertainty includes zero. Both comparisons fail paired-support adequacy.',lambda c:table(c,['U-policy offset','Mean loss change','Conditional 95% interval'],[(str(z['offset']),f"{z['mean_effect']:.6f}",f"[{z['conditional_source_bootstrap_95_interval'][0]:.6f}, {z['conditional_source_bootstrap_95_interval'][1]:.6f}]") for z in summary],[320,340,500],row_h=100,size=23),notes='Theseare27outer-testnormals,oneaggregatedsourceperoffset. Zeroeditsproducezerodifference;ownpredictionandtargetstayfixed. Noiseisnotscale-matchedtotimingdisplacement. Whole-bodyshiftsareseparatecontrolsandnotfixed-targetinterventions.',takeaway='Report compound input sensitivity, not a neurological phase mechanism.')
 d.add('Independent participants test the measurement elsewhere.','The public laboratory dataset contains synchronized bilateral signals and explicit participant identities.',lambda c:cards(c,[('Selected trials','Download 188 participants; ten controls lack the required usable angle support.'),('Eligible cohort',f"Use {complete['normal']} controls and {complete['stroke']} stroke participants in four folds."),('Separate fitting','Fit six-angle models and normal references from scratch on a common clock.')],size=24),notes='DatafromVanCriekingeet al., ScientificData2023. The boundedreplicationusesoneprespecifiedtrialperparticipantandcommon-clocksixanglewindows. Itisnotsensortransfer,weighttransferorvalidationacrossallfourconditionfamilies.',refs=('external_paper','external_results','protocol'),takeaway='This is an independent method replication with different measurements.')
 ev=exterr.loc[exterr.binary_label.eq(0)].sort_values('outer_fold')
 d.add('Other-leg context helps held-out laboratory normals.','The both-leg correction improves on the own-capacity correction in every outer test fold.',lambda c:external_gain(c,ev.both_vs_own_relative_gain.to_numpy()*100),notes='Thefourheldoutnormalfoldscontain32participantseach. Thegainrangesfrom4.78%to5.52%. Thesearedegree-anglepredictionerrors,notGAVDcoordinateunits. Stablecross-leginformationexistsunderthismeasurement.',takeaway='This is the clearest evidence that the prediction task can learn useful normal dependence.')
 d.add('Laboratory detection still favors local movement.','The relational channel is above chance, but averaging it with M reduces separation.',lambda c:table(c,['Laboratory score','Weighted AUROC'],[(k,f"{ext[k]['weighted_auroc']:.3f}") for k in ['M','R','S']],[900,260],row_h=85),notes=f"SminusM={e['observed']:.4f},conditional95%interval[{e['conditional_95_interval'][0]:.4f},{e['conditional_95_interval'][1]:.4f}]. Thisnegativeresultdoesnotproverelationalinformationcanneverhelp. Ittestsonefrozenmeasurementandfusionrecipe.",takeaway='Useful normal completion and useful anomaly fusion are distinct empirical claims.')
 d.add('Age and sex balancing does not remove this limitation.','The frozen sensitivity matches same-sex participants within five years and within each outer fold.',lambda c:table(c,['Age/sex-balanced subset','Value'],[('Matched pairs',str(sum(matched['by_fold'].values()))),('Mean absolute age gap',f"{matched['mean_absolute_age_difference']:.2f} years"),('M / S AUROC',f"{matched['metrics']['M']['weighted_auroc']:.3f} / {matched['metrics']['S']['weighted_auroc']:.3f}"),('S minus M 95% interval',f"[{matched['conditional_95_interval'][0]:.3f}, {matched['conditional_95_interval'][1]:.3f}]")],[760,400],row_h=77,size=23),notes='Themaximum-cardinality/minimum-age-gapmatchingrulewasfixedbeforeoutcomes. Ituses45pairs:16femaleand29male,foldcounts11/12/11/11. No modelsarefitagain. Bootstrapwholematchedpairswithin-fold. Thisisabalancedsensitivity,notcausaladjustment.',refs=('external_results','external_paper'),takeaway='Demographic sensitivity strengthens a narrow result without establishing clinical validity.')
 d.add('The remaining confounds still matter.','Camera view, confidence filtering and unknown repeated identities limit GAVD generalization.',lambda c:cards(c,[('Acquisition','View-conditioned subsets often lack normals in all four folds; camera motion remains unresolved.'),('Source selection','Confidence filtering can preferentially remove severe or occluded movement.'),('Labels and physiology','Web labels lack standardized clinical confirmation, treatment and severity measures.')],size=24),notes='Notebook17showsdeterministiccontextandexplicitoutcome-selectedfailurecandidates. Do notinventcausalclinicalstoriesforindividualvideos. TheexternaldatasetreducessomeacquisitionuncertaintybutdoesnotvalidateGAVDdiagnoses.',refs=('limits','selection','nb12'),takeaway='A source-level demonstration is not prospective clinical diagnosis.')
 d.add('The defensible contribution is a tested distinction.','The results explain why accurate conditional prediction alone is an insufficient objective for anomaly detection.',lambda c:cards(c,[('Supported','The fitted bilateral predictors improve completion on declared tasks.'),('Unsupported','The fixed relational fusion does not improve either primary detector.'),('Not identified','Neurological causes, physiological phase effects and broad clinical generalization.')],size=24),notes='GAVDuniformheldoutnormalUbenefititselfhasanintervalcrossingzero;theclearnormaldependenceresultisexternalanddistal-bank. No positive statementshouldcollapseallbanksanddatasets. Noveltymustrestontransparentcontrolsandreplication,notcomponents.',refs=('current_results','limits','neuro','draft02'),takeaway='The negative fusion result is part of the contribution, not an inconvenience to hide.')
 d.add('Six independent notebooks expose the evidence.','Each uses the exact verified run, current dotenv setup, real video examples and source-hashed figures.',lambda c:table(c,['Notebook','Question'],[('12–13','Who is observed, and what does the other leg add?'),('14–15','What survives timing controls, and how are M/R/S constructed?'),('16–17','Which baselines win, and what can the manuscript claim?')],[270,890],row_h=90,size=24),notes='Useuvsync--projectexperiments/aaai andsetAAAI_ITERATION02_RUN_DIR. Theexecutionreportrecordsindependentfreshkernels. Historical00–11retainscientificcellsandoutputs;navigationisupdatedandnewreplayswriteisolatedartifacts. Seeexecutionreportforcompletionratherthaninferringitfromnotebookexistence.',refs=('nb12','protocol'),takeaway='The notebook execution record and exact run identity accompany every measured figure.')
 d.add('The next paper should test a measurement, not promise a cure.','A strong submission needs a narrow result, transparent failed gates and a clear account of what would change the conclusion.',lambda c:cards(c,[('Evidence to preserve','Report useful completion and the fixed negative fusion with independent participant replication.'),('Next experiment','Test prespecified reliability-aware fusion and acquisition-matched validation on new participants.'),('Future physiology','Use synchronized forces, EMG or clinical measures to distinguish compensatory mechanisms.')],size=24),notes='Reliability-awarefusionisfuturework,notapost-hocreplacementforfixedS. Encoder-maskablations,SPE/SLVandmulti-seedtrainingremainunexecuted. Submitonlyafterthemanuscript,format,linksandclaimledgeragreewiththisevidence. Workshopacceptancecannotbeestimatedfromthesescores.',refs=('draft02','workshop','limits'),takeaway='The evidence should determine the claim and the next falsifiable experiment.')

 # Complete, readable scientific notes are part of the presentation artifact.
 notes = {
 9: "Repeated masks and overlapping windows cannot create new physical target support. The earlier read-only 96-frame candidate analysis retained only 12 normal sources and was not a model result. Validated normal contact-event phase scores are unavailable.",
 11: "All 28 normal sources had earlier exploratory use. New folds do not create independent confirmation. Unknown repeated identities across different videos remain unresolved. The standalone laboratory dataset strengthens a separate measurement test.",
 13: "The diagram shows the primary own-capacity and both-leg comparison explicitly. The benefit is loss(f+h) minus loss(f+g). Both branches retain the same frozen f, predict the same target Y and can choose zero correction. The restriction is conditional on the shared body reference Q. Equal coefficient dimensions do not establish equal effective capacity, rank or inductive bias.",
 14: "U is uniform. J selects one joint uniformly and removes all eight temporal patches, then fills 56 tokens. A selects an anatomical group with probability equal to width divided by 16, removes an aligned eight-token core, then fills 56 tokens. These example draws illustrate the implemented rule and are not model outputs.",
 15: "For every token, P(core)=1/16 and P(hidden)=1/16+(15/16)*(56/120)=1/2. This matches first-order probabilities, not difficulty or valid-target exposure. Starts are 0, 2, 4 and 6 for four-joint groups, and 0 or 4 for two-joint groups. No block wraps.",
 17: "The completed coordinate comparison does not establish the best JEPA mask. VICReg discourages collapse; it does not separate disease categories by itself. Encoder-mask ablation, stochastic positional embeddings and stochastic latent variables remain deferred. Dedicated mask random streams preserve other sampling choices.",
 7: f"The supervised five-class total is {sum(v for _,v in countrows)}. The focal binary total is {focal}; one focal source has ambiguous condition membership and is excluded from five-class classification. The full inventory has 348 source IDs and 347 known recording groups. The duplicate pair fails primary quality criteria, so the retained 304 IDs represent 304 known recording groups.",
 16: "The uniform reference has 16 masks with seed 17. The distal-leg bank balances two sides and four starts twice, with seed 117. Each mask hides 64 tokens, while the diagnostic scores only its eight core targets. Compare training policies within one bank, never native task against native task.",
 18: "Every outer-fold and policy combination meets the operational two-of-three internal-fold rule against both the baseline and own-capacity correction. Points show the mean internal validation benefit after selection. These are development checks, not independent test evidence.",
 19: "All 28 outer-test normal sources enter each result. Benefit is own-capacity loss minus both-leg loss after the declared hierarchical aggregation. The diagnostic bank removes more local geometry and reveals a different prediction task. It cannot establish an improvement on the uniform reference.",
 20: f"Actual saved source {row.source_id}, sequence {row.sequence_id}, joint {int(row.joint)}, patch {int(row.patch_index)}. Notebook 13 displays the corresponding video and raw pose overlay. Every line uses the same observed target. This is an example for auditing coordinates, not a representative accuracy estimate.",
 21: "M excludes local prediction error. Its imputation, scale and covariance use fitting normals. R uses separately reserved normals after model selection. Stable compensations can increase or decrease predictive benefit, so R uses a two-sided reference. A reference of only five sources remains fragile.",
 22: "Intervals condition on the fitted models, selected protocol and original fold weights. They do not include training-seed variation or unknown repeated-person dependence. Average precision is reported per fold with prevalence. A pooled score plot is descriptive and does not replace the primary AUROC.",
 24: "Intervals use 1,000 paired bootstrap resamples with fixed original fold weights. The mask-policy contrast compares S under the same uniform evaluation bank. An interval crossing zero does not establish equivalence or improvement.",
 25: "The A-minus-U focal difference is positive, but its conditional 95% interval crosses zero. A-minus-J and J-minus-U intervals also cross zero. All arms remain in the record; A is not selected as a new primary method after evaluation.",
 26: "Each bank has its own relational reference, fitted using the same reserved source identities. Greater normal prediction benefit on the distal-leg bank does not imply better detection. The bank is a declared diagnostic with different targets.",
 27: "Timing8 is a waveform descriptor, not the phase coordination index or a validated physiological phase estimate. Nuisance, sham, constituent-error and calibrated-error comparisons remain in the complete metrics. The strongest local baseline must stay visible.",
 28: "The upstream revision and adaptation are recorded. This is not a faithful reproduction of a published benchmark. The weak result from a bounded implementation cannot justify a state-of-the-art claim. All four folds completed on matched coverage.",
 29: "Fresh JEPA and random backbones use the same source folds. JEPA has 512 updates per fold and VICReg regularization. The table contains means of four held-out fold metrics, not clinical confidence intervals. Five-class evaluation contains 89 unambiguous sources. The learned representation does not improve this bounded comparison.",
 30: "The target and own-leg window is frames 8:40, with donor intervals 4:36 and 12:44. Subtract the permitted donor hip and attach the offset to the permitted original opposite hip. Keep the common body reference and paired support fixed. This edits distal timing and geometry, not an isolated physiological phase.",
 31: "The core uniform-bank gate and the exact paired-support gate ask different questions. Sparse paired support leaves about 1.75 permitted distal tokens per target. The operational two-of-three internal-normal rule fails all 24 comparisons. Models and scores were not retuned after this result.",
 32: "There are 27 outer-test normal sources, aggregated once per offset. Zero edits produce zero differences; own-leg predictions and targets stay fixed. Noise is not matched to timing displacement. Whole-body shifts are separate controls and do not qualify as fixed-target interventions.",
 33: "Van Criekinge and colleagues released this laboratory dataset in Scientific Data in 2023. The bounded replication uses one prespecified trial per participant and six synchronized angle signals on a common clock. It is not sensor transfer, weight transfer or validation of all four condition families.",
 34: "Each held-out normal fold has 32 participants. Relative own-capacity error reduction ranges from 4.78% to 5.52%. The errors use laboratory angle units, not GAVD coordinate units. This measurement contains useful cross-leg predictive information.",
 35: f"S minus M is {e['observed']:.4f}, with conditional 95% interval [{e['conditional_95_interval'][0]:.4f}, {e['conditional_95_interval'][1]:.4f}]. The negative result tests one fixed measurement and fusion recipe. It does not prove that relational information can never help.",
 36: "The maximum-cardinality, minimum-age-gap matching rule was fixed before outcomes. It produces 45 same-sex pairs: 16 female and 29 male, with fold counts 11, 12, 11 and 11. Models are not refitted. Bootstrap resampling carries whole matched pairs within each fold. This is a balanced sensitivity analysis, not causal adjustment.",
 37: "Notebook 17 shows deterministic context examples and explicitly labels outcome-selected failure candidates. Individual videos do not support invented causal clinical explanations. The independent laboratory data reduce some acquisition uncertainty without validating GAVD diagnoses.",
 38: "The GAVD uniform-bank U normal-benefit interval itself includes zero. The clearest useful-dependence findings come from the external dataset and distal-leg diagnostic bank. Do not combine these into a stronger universal claim. The contribution rests on transparent controls and replication, not on novel model components.",
 39: "Use uv with the existing project configuration and set AAAI_ITERATION02_RUN_DIR to the exact run. The execution report records independent fresh kernels. Historical notebooks 00–11 preserve scientific cells and outputs; new replay checks write isolated artifacts. Read the execution record rather than assuming a notebook has run because its file exists.",
 40: "Reliability-aware fusion is future work, not a replacement selected after the fixed S result. Encoder-mask ablations, stochastic positional embeddings, stochastic latent variables and multiple training seeds remain unexecuted. The manuscript, format, links and claim ledger must agree with the evidence. These scores cannot determine workshop acceptance probability.",
 }
 for number, note in notes.items():
  d.records[number - 1]['notes'] = note
 assert len(d.records)==40,len(d.records)
 completion_appendix(d,run,external)
 assert len(d.records)==43,len(d.records)
 from accessible_gait_deck import revise
 return revise(d,run,external,sys.modules[__name__])


def shared_diagrams():
 SHARED.mkdir(parents=True,exist_ok=True)
 for name,(title,draw) in design.DIAGRAMS.items():
  if name == 'residual_predictor': draw = residual
  c=Canvas(1280,600);c.text(60,34,'STUDY DESIGN | ILLUSTRATION, NOT AN EMPIRICAL RESULT',16,TEAL,True);c.text(60,83,title,31,INK,True);draw(c,130)
  c.text(60,582,'The illustrated comparisons do not identify neurological causes.',17,MUTED);c.save(SHARED/name)


def export(d,run,external):
 OUT.mkdir(exist_ok=True);IMG.mkdir(exist_ok=True);review=OUT/'review';review.mkdir(exist_ok=True)
 total=len(d.records)
 assert file_hash(OUT/'archive/draft01_completed/tutorial.pdf')=='bb0502c207305ac9a5f2746c3af5dfaac7eda8c5d7805aa6007726f86fc317d8'
 assert file_hash(OUT/'archive/iteration02_plan/tutorial.pdf')=='bbf2487f18224eda8b5bd09701c96e59f237887d7c9d717d76ce149ffd1099f6'
 for path in IMG.glob('slide-*'):
  if path.suffix in {'.svg','.pdf','.png'}:path.unlink()
 for path in review.glob('*'):
  if path.is_file():path.unlink()
 from accessible_gait_deck import NOTE_FIELDS
 from gait_slide_viewer import render_viewer, dark_svg, embed_svg_fonts, GLOSSARY, DARK_PALETTE
 records=[]
 for number,s in enumerate(d.records,1):
  c=frame(s,number,total);s['draw'](c);text=[t.get_text() for t in c.texts]
  # Darken accent text slightly for AA contrast on both light backgrounds.
  # Keep the historical drawing helper and its archived outputs unchanged.
  for artist in c.texts:
   if artist.get_color()=='#087f82':artist.set_color('#08787b')
  c.save(IMG/f'slide-{number:02d}')
  light=IMG/f'slide-{number:02d}.svg'
  (IMG/f'slide-{number:02d}-dark.svg').write_text(dark_svg(light.read_text()))
  records.append({**{k:v for k,v in s.items() if k!='draw'},'visible_text':text})
 font_manifest=embed_svg_fonts(sorted(IMG.glob('slide-*.svg')),''.join(t for s in records for t in s['visible_text']),OUT/'fonts')
 subprocess.run(['pdfunite',*[str(IMG/f'slide-{i:02d}.pdf') for i in range(1,total+1)],str(OUT/'tutorial.pdf')],check=True)
 md=['# Understanding gait prediction','',f'This {total}-slide guide is written for general AI/ML researchers. It explains the question, measurements, prediction comparisons and anomaly results using the verified experiments.','', '[Slide PDF](tutorial.pdf) · [Offline viewer](index.html) · [Manuscript](../docs/draft03/README.md) · [Technical details](../docs/draft03/technical_details.md)','', 'Every slide has notes on purpose, explanation, reading the visual, evidence and limits, and the transition to the next slide.','']
 def inline(text):return re.sub(r'`([^`]+)`',r'<code>\1</code>',escape(text))
 for i,s in enumerate(records,1):
  refs=' · '.join(f'<a href="{escape(SOURCES[k][1],quote=True)}">{escape(SOURCES[k][0])}</a>' for k in s['refs'])
  md += [f'## {i:02d}. {s["title"]}','',s['lede'],'',f'![Slide {i}: {s["title"]}](images/slide-{i:02d}.svg)','',f'*Figure caption: {s["takeaway"]}*','']
  for key,label in NOTE_FIELDS:
   value=s['notes_sections'][key];md += [f'### {label}','',value,'']
  if s.get('score_guide'):md += ['[Plain-language guide to the three scores](score_guide.md)','']
  md += ['Sources: '+' · '.join(f'[{SOURCES[k][0]}]({SOURCES[k][1]})' for k in s['refs']),'']
 (OUT/'index.html').write_text(render_viewer(records,SOURCES,NOTE_FIELDS));(OUT/'tutorial.md').write_text('\n'.join(md));(OUT/'deck_spec.json').write_text(json.dumps(records,indent=2))
 (OUT/'score_guide.md').write_text('# Understanding the three anomaly scores\n\n'+'\n\n'.join('## '+title+'\n\n'+body for title,body in GLOSSARY)+'\n')
 inputs={str(run.path.relative_to(BASE))+'/'+k:v for k,v in run.loaded.items()};inputs.update({str(p.relative_to(BASE)):file_hash(p) for p in external.glob('*.json')})
 inputs[str((external/'heldout_prediction_error.csv').relative_to(BASE))]=file_hash(external/'heldout_prediction_error.csv')
 for filename in ['run_manifest.json','verification.json','protocol.json']:inputs[str((run.path/filename).relative_to(BASE))]=file_hash(run.path/filename)
 for filename in ['audit_manifest.json','verification.json','completion_summary.csv']:inputs[str((COMPLETION_AUDIT/filename).relative_to(BASE))]=file_hash(COMPLETION_AUDIT/filename)
 sources={str(p.relative_to(BASE)):file_hash(p) for p in [Path(__file__),BASE/'scripts/accessible_gait_deck.py',BASE/'scripts/gait_slide_viewer.py',BASE/'scripts/validate_gait_slide_viewer.py',*sorted((OUT/'source').glob('*.json'))]}
 (OUT/'result_input.json').write_text(json.dumps({'gavd_run':run.protocol_hash,'external_run':external.name,'input_sha256':inputs,'presentation_sources':sources,'descriptive_completion_slide_positions':[i for i,s in enumerate(records,1) if s['content_id'] in [41,42,43]]},indent=2))
 (OUT/'README.md').write_text(f'''# Understanding gait prediction

This {total}-slide deck explains the study for general AI/ML researchers. It begins with a hidden-ankle example, introduces the data and three predictors, and then examines prediction quality, anomaly scores and independent laboratory measurements. The experiments and results are unchanged.

Open the [offline viewer](index.html), [slide PDF](tutorial.pdf), or [detailed speaker notes](tutorial.md). The viewer has a slide selector, keyboard navigation, a text version of each slide and a button to open all notes. The viewer offers Light and Dark buttons on every slide, a device-aware deck default, and a control that returns all slides to that default. Theme choices persist in this browser. The PDF retains the light theme. Both SVG themes embed their fonts, so they work offline without a system-font dependency. Score-focused slides also include a [plain-language guide to M, R and S](score_guide.md). Every slide has five note sections: purpose, explanation, reading the visual, evidence and limits, and transition.

The [manuscript](../docs/draft03/README.md) uses the same terminology. Random, whole-joint and anatomical masks describe training. Scattered and grouped hiding describe evaluation. The movement score describes posture and movement size; the cross-leg score compares the benefit of seeing the other leg with separate normal reference recordings. Benefits that are unusually high or unusually low increase that score. The combined score adds these two numerical scores and divides by two. This fixed average is tested against movement alone; none of the scores is a probability of disease.

The [input record](result_input.json), [build manifest](build_manifest.json), [render checks](review/validation.json), [browser checks](review/viewer_validation.json), and [visual review](visual_review.md) document the evidence and verification. The [previous completed deck](archive/before_accessibility/tutorial.pdf), [planning deck](archive/iteration02_plan/tutorial.pdf) and [draft01 deck](archive/draft01_completed/tutorial.pdf) remain archived.

From the repository root, rebuild with:

```bash
uv run --project experiments/aaai python experiments/aaai/scripts/build_iteration02_result_slides.py --run-dir {run.path.relative_to(BASE.parents[1])} --external-dir {external.relative_to(BASE.parents[1])}
```

After rebuilding, run the browser checks with `uv run --project experiments/aaai --with playwright python experiments/aaai/scripts/validate_gait_slide_viewer.py`. The report is marked pending until these checks finish.

Visible content is maintained in `scripts/accessible_gait_deck.py`. Structured note sources live in `slides/source/notes_*.json`. The offline viewer and its explicit dark SVG palette are maintained in `scripts/gait_slide_viewer.py`. The builder exports light and dark SVG pages, light PDF and PNG pages, assembles the deck, renders every PDF page and checks links, bounds and fonts. It reads saved evidence without training models or changing scores.
''')
 subprocess.run(['pdftoppm','-r','80','-png',str(OUT/'tutorial.pdf'),str(review/'render')],check=True)
 pages=sorted(review.glob('render-*.png'));assert len(pages)==total
 for start in range(0,total,10):
  sheet=Image.new('RGB',(1280,1925),'#e4ebf1');draw=ImageDraw.Draw(sheet)
  for j,p in enumerate(pages[start:start+10]):
   im=Image.open(p).convert('RGB');im.thumbnail((640,360));x=j%2*640;y=j//2*385;sheet.paste(im,(x,y));draw.text((x+12,y+362),f'Slide {start+j+1:02d}',fill=INK)
  sheet.save(review/f'contact-sheet-{start//10+1:02d}.png')
 words=[len(s['notes'].split()) for s in records]
 (OUT/'build_manifest.json').write_text(json.dumps({'status':'Verified study explained for general AI/ML researchers','slides':total,'builder':'scripts/build_iteration02_result_slides.py','presentation_sources':sources,'gavd_run':run.protocol_hash,'external_run':external.name,'pdf_sha256':file_hash(OUT/'tutorial.pdf'),'rendered_pdf_pages':len(pages),'themes':['light','dark'],'dark_svg_pages':len(list(IMG.glob('slide-*-dark.svg'))),'dark_palette':DARK_PALETTE,'embedded_svg_fonts':font_manifest,'notes_sections_per_slide':5,'notes_word_count':sum(words),'notes_minimum_words':min(words),'prior_active_pdf_sha256':file_hash(OUT/'archive/before_accessibility/tutorial.pdf')},indent=2))
 (review/'viewer_validation.json').write_text(json.dumps({'status':'pending_browser_validation','command':'uv run --project experiments/aaai --with playwright python experiments/aaai/scripts/validate_gait_slide_viewer.py'},indent=2))
 return validate()


def validate():
 review=OUT/'review';(review/'validation.json').touch(exist_ok=True);subprocess.run(['pdftotext','-bbox',str(OUT/'tutorial.pdf'),str(review/'text-bounds.html')],check=True)
 root=ET.parse(review/'text-bounds.html').getroot();bad=[];pages=0
 for p in root.iter():
  if p.tag.endswith('page'):
   pages+=1;w,h=float(p.attrib['width']),float(p.attrib['height'])
   for t in p.iter():
    if t.tag.endswith('word'):
     z=t.attrib
     if float(z['xMin'])<0 or float(z['yMin'])<0 or float(z['xMax'])>w+.1 or float(z['yMax'])>h+.1:bad.append((pages,t.text))
 broken=[];checked=0
 for p in [OUT/'README.md',OUT/'tutorial.md',OUT/'index.html']:
  links=re.findall(r'(?:href|src)="([^"]+)"',p.read_text()) if p.suffix=='.html' else re.findall(r'\]\(([^\)]+)\)',p.read_text())
  for link in links:
   if link.startswith(('https:','http:','#','mailto:')):continue
   checked+=1
   if not (p.parent/unquote(urlsplit(link).path)).exists():broken.append((p.name,link))
 fonts=subprocess.check_output(['pdffonts',str(OUT/'tutorial.pdf')],text=True);rasters=subprocess.check_output(['pdfimages','-list',str(OUT/'tutorial.pdf')],text=True)
 result={'pages':pages,'out_of_bounds_words':bad,'local_links_checked':checked,'broken_links':broken,'type3_fonts':'Type 3' in fonts,'raster_objects':max(0,len(rasters.strip().splitlines())-2),'draft01_pdf_unchanged':file_hash(OUT/'archive/draft01_completed/tutorial.pdf')=='bb0502c207305ac9a5f2746c3af5dfaac7eda8c5d7805aa6007726f86fc317d8','planning_pdf_unchanged':file_hash(OUT/'archive/iteration02_plan/tutorial.pdf')=='bbf2487f18224eda8b5bd09701c96e59f237887d7c9d717d76ce149ffd1099f6'}
 expected=len(json.loads((OUT/'deck_spec.json').read_text()))
 (review/'validation.json').write_text(json.dumps(result,indent=2));assert pages==expected and not bad and not broken and not result['type3_fonts'] and result['raster_objects']==0 and result['draft01_pdf_unchanged'] and result['planning_pdf_unchanged'],result
 return result


if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--run-dir',required=True);parser.add_argument('--external-dir',required=True);args=parser.parse_args()
 run=EvidenceRun(args.run_dir);external=Path(args.external_dir).resolve()
 assert json.loads((external/'completion.json').read_text())['status']=='complete'
 deck=build(run,external);print(export(deck,run,external))
